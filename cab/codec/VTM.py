# cab/codec/vtm.py
import torch
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image

from cab.codec.abs import ImageCodecIface
from cab.complexity import time_ms

class VTMImageCodec(ImageCodecIface):
    """VTM (H.266/VVC) Image Codec wrapper using VTM reference software."""

    def __init__(
        self,
        qp=32,
        vtm_encoder_path=None,
        vtm_decoder_path=None,
        *args,
        **kwargs,
    ):
        """
        Args:
            qp: Quantization parameter.
            vtm_encoder_path: Path to VTM encoder executable.
            vtm_decoder_path: Path to VTM decoder executable.
        """
        super().__init__(*args, **kwargs)
        self.qp = qp
        self.vtm_encoder_path = vtm_encoder_path
        self.vtm_decoder_path = vtm_decoder_path

    def fake_input(self, image_size=256, batch_size=1, device=None):
        if device is None:
            device = torch.device("cpu")
        return torch.rand(batch_size, 3, image_size, image_size, device=device)

    def encode_params_m(self):
        return 0.0

    def decode_params_m(self):
        return 0.0

    def encode_gflops(self, x):
        return None

    def decode_gflops(self, x):
        return None

    def _write_yuv420(self, img_np, yuv_path):
        pil_img = Image.fromarray(img_np, mode="RGB")
        yuv_img = pil_img.convert("YCbCr")
        yuv_arr = np.array(yuv_img).astype(np.uint8)

        Y = yuv_arr[:, :, 0]
        Cb = yuv_arr[:, :, 1][::2, ::2]
        Cr = yuv_arr[:, :, 2][::2, ::2]

        with open(yuv_path, "wb") as f:
            Y.tofile(f)
            Cb.tofile(f)
            Cr.tofile(f)

    def _read_yuv420_rec(self, rec_path, H, W):
        buf = np.fromfile(rec_path, dtype=np.uint8)

        n_y = H * W
        n_c = (H // 2) * (W // 2)
        expected = n_y + 2 * n_c

        if buf.size < expected:
            raise RuntimeError(
                f"Invalid reconstructed YUV size: got {buf.size}, expected {expected}"
            )

        Y = buf[:n_y].reshape(H, W)
        Cb = buf[n_y:n_y + n_c].reshape(H // 2, W // 2)
        Cr = buf[n_y + n_c:n_y + 2 * n_c].reshape(H // 2, W // 2)

        Cb = np.repeat(np.repeat(Cb, 2, axis=0), 2, axis=1)
        Cr = np.repeat(np.repeat(Cr, 2, axis=0), 2, axis=1)

        rec_yuv = np.stack([Y, Cb, Cr], axis=-1)
        rec_pil = Image.fromarray(rec_yuv, mode="YCbCr").convert("RGB")
        return np.array(rec_pil, dtype=np.float32) / 255.0

    def _run_encoder(self, yuv_path, bin_path, rec_path, H, W):
        if self.vtm_encoder_path is None:
            raise RuntimeError("vtm_encoder_path is None.")

        cmd = [
            self.vtm_encoder_path,
            "-c", "config/image_codecs/encoder_intra_vtm.cfg",
            "-c", "config/image_codecs/HM.cfg",
            "-i", str(yuv_path),
            "-wdt", str(W),
            "-hgt", str(H),
            "-f", "1",
            "-fr", "1",
            "-q", str(self.qp),
            "-o", str(rec_path),
            "-b", str(bin_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"VTM encoder failed with code {result.returncode}\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

    def _run_decoder(self, bin_path, rec_path):
        if self.vtm_decoder_path is None:
            raise RuntimeError("vtm_decoder_path is None; cannot run VTM decoder.")

        cmd = [
            self.vtm_decoder_path,
            "-b", str(bin_path),
            "-o", str(rec_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"VTM decoder failed with code {result.returncode}\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

    @torch.no_grad()
    def encode_time_ms(self, x, warmup=3, repeat=10):
        x_cpu = (
            (x.detach() * 255)
            .clamp(0, 255)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        B, _, H, W = x.shape

        with tempfile.TemporaryDirectory(prefix="vtm_enc_profile_") as tmpdir:
            yuv_path = Path(tmpdir) / "input.yuv"
            bin_path = Path(tmpdir) / "output.bin"
            enc_rec_path = Path(tmpdir) / "enc_rec.yuv"

            self._write_yuv420(x_cpu[0], yuv_path)

            t = time_ms(
                lambda: self._run_encoder(
                    yuv_path,
                    bin_path,
                    enc_rec_path,
                    H,
                    W,
                ),
                torch.device("cpu"),
                warmup=warmup,
                repeat=repeat,
            )

        return t / B

    @torch.no_grad()
    def decode_time_ms(self, x, warmup=3, repeat=10):
        if self.vtm_decoder_path is None:
            return None

        x_cpu = (
            (x.detach() * 255)
            .clamp(0, 255)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        B, _, H, W = x.shape

        with tempfile.TemporaryDirectory(prefix="vtm_dec_profile_") as tmpdir:
            yuv_path = Path(tmpdir) / "input.yuv"
            bin_path = Path(tmpdir) / "output.bin"
            enc_rec_path = Path(tmpdir) / "enc_rec.yuv"
            dec_rec_path = Path(tmpdir) / "dec_rec.yuv"

            self._write_yuv420(x_cpu[0], yuv_path)

            # 先编码生成 bitstream，不计入 decode 时间
            self._run_encoder(
                yuv_path,
                bin_path,
                enc_rec_path,
                H,
                W,
            )

            t = time_ms(
                lambda: self._run_decoder(
                    bin_path,
                    dec_rec_path,
                ),
                torch.device("cpu"),
                warmup=warmup,
                repeat=repeat,
            )

        return t / B

    @torch.no_grad()
    def forward(self, x, *args, **kwargs):
        """
        Args:
            x: (B, 3, H, W), range [0, 1]

        Returns:
            xhat: reconstructed image (B, 3, H, W), range [0, 1]
            bpp: (B,)
        """
        B, C, H, W = x.shape
        device = x.device

        x_cpu = (
            (x * 255)
            .clamp(0, 255)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        recons = []
        bpps = []

        with tempfile.TemporaryDirectory(prefix="vtm_") as tmpdir:
            for i, img_np in enumerate(x_cpu):
                yuv_path = Path(tmpdir) / f"input_{i}.yuv"
                bin_path = Path(tmpdir) / f"output_{i}.bin"
                enc_rec_path = Path(tmpdir) / f"enc_rec_{i}.yuv"
                dec_rec_path = Path(tmpdir) / f"dec_rec_{i}.yuv"

                self._write_yuv420(img_np, yuv_path)

                self._run_encoder(
                    yuv_path,
                    bin_path,
                    enc_rec_path,
                    H,
                    W,
                )

                bytes_encoded = bin_path.stat().st_size
                bpp = bytes_encoded * 8.0 / (H * W)

                if self.vtm_decoder_path is not None:
                    self._run_decoder(
                        bin_path,
                        dec_rec_path,
                    )
                    rec_np = self._read_yuv420_rec(
                        dec_rec_path,
                        H,
                        W,
                    )
                else:
                    rec_np = self._read_yuv420_rec(
                        enc_rec_path,
                        H,
                        W,
                    )

                recons.append(torch.from_numpy(rec_np).permute(2, 0, 1))
                bpps.append(bpp)

        rec_tensor = torch.stack(recons, dim=0).to(
            device=device,
            dtype=x.dtype,
        )

        bpp_tensor = torch.tensor(
            bpps,
            dtype=torch.float32,
            device=device,
        )

        return rec_tensor, bpp_tensor