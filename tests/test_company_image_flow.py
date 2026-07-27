import importlib
import importlib.machinery
import inspect
import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from cab.codec.abs import (
    merge_image_patches,
    patch_bits_to_image_bpp,
    split_image_into_patches,
)
from cab.codec.jpeg import JPEGImageCodec
from cab.dataset.enterprise_camera_image import CompanyImageDataset


class CompanyImageDatasetTest(unittest.TestCase):
    def test_returns_complete_center_crop_for_multiple_sizes(self):
        cases = [
            ((512, 512), (512, 512)),
            ((700, 1200), (512, 1024)),
            ((1300, 1100), (1024, 1024)),
            ((1200, 1700), (1024, 1536)),
        ]
        for (source_h, source_w), (expected_h, expected_w) in cases:
            with self.subTest(source=(source_h, source_w)):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    source = np.zeros(
                        (source_h, source_w, 3),
                        dtype=np.uint8,
                    )
                    source[..., 0] = (
                        np.arange(source_w, dtype=np.uint32) % 256
                    )
                    source[..., 1] = (
                        np.arange(source_h, dtype=np.uint32)[:, None] % 256
                    )
                    Image.fromarray(source, mode="RGB").save(
                        root / "sample.png"
                    )

                    dataset = CompanyImageDataset(root, zero_mean=False)
                    sample = dataset[0]

                    self.assertEqual(set(sample), {"img", "path"})
                    self.assertEqual(
                        tuple(sample["img"].shape),
                        (3, expected_h, expected_w),
                    )
                    self.assertEqual(sample["img"].shape[-2] % 512, 0)
                    self.assertEqual(sample["img"].shape[-1] % 512, 0)

                    left = (source_w - expected_w) // 2
                    top = (source_h - expected_h) // 2
                    expected_top_left = torch.tensor(
                        source[top, left],
                        dtype=torch.float32,
                    ) / 255.0
                    torch.testing.assert_close(
                        sample["img"][:, 0, 0],
                        expected_top_left,
                    )

    def test_constructor_has_no_dataset_patch_controls(self):
        parameters = inspect.signature(CompanyImageDataset).parameters
        self.assertEqual(set(parameters), {"root", "zero_mean"})

    def test_rejects_image_smaller_than_one_crop_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (511, 700)).save(root / "small.png")
            dataset = CompanyImageDataset(root)

            with self.assertRaisesRegex(ValueError, "smaller than"):
                dataset[0]


class PatchRoundTripTest(unittest.TestCase):
    @staticmethod
    def patterned_image(height, width, batch_size=1):
        values = torch.arange(height * width, dtype=torch.int32)
        values = values.reshape(1, 1, height, width)
        values = torch.cat((values, values + 1, values + 2), dim=1)
        return torch.cat(
            [values + index * height * width * 4 for index in range(batch_size)],
            dim=0,
        )

    def assert_round_trip(self, height, width, patch_size, batch_size=1):
        image = self.patterned_image(height, width, batch_size)
        patches = split_image_into_patches(image, patch_size)
        reconstructed = merge_image_patches(
            patches,
            image_size=image.shape[-2:],
        )

        expected_patches = (height // patch_size) * (width // patch_size)
        self.assertEqual(
            tuple(patches.shape),
            (
                batch_size,
                expected_patches,
                3,
                patch_size,
                patch_size,
            ),
        )
        self.assertTrue(torch.equal(reconstructed, image))

    def test_256_and_512_round_trip_for_all_legal_sizes(self):
        for height, width in [
            (512, 512),
            (512, 1024),
            (1024, 1024),
            (1024, 1536),
        ]:
            for patch_size in (256, 512):
                with self.subTest(
                    image_size=(height, width),
                    patch_size=patch_size,
                ):
                    self.assert_round_trip(height, width, patch_size)

    def test_patch_helpers_support_batch_size_two(self):
        self.assert_round_trip(512, 1024, 256, batch_size=2)
        self.assert_round_trip(512, 1024, 512, batch_size=2)

    def test_patch_order_is_row_major(self):
        image = torch.arange(
            512 * 1024,
            dtype=torch.int32,
        ).reshape(1, 1, 512, 1024)
        patches = split_image_into_patches(image, 256)
        patch_origins = patches[0, :, 0, 0, 0]
        expected_origins = torch.tensor(
            [
                0,
                256,
                512,
                768,
                256 * 1024,
                256 * 1024 + 256,
                256 * 1024 + 512,
                256 * 1024 + 768,
            ],
            dtype=torch.int32,
        )
        self.assertTrue(torch.equal(patch_origins, expected_origins))

    def test_patch_bits_are_summed_per_complete_image(self):
        patch_bits = torch.tensor([8.0, 16.0, 24.0, 32.0])
        bpp = patch_bits_to_image_bpp(
            patch_bits,
            batch_size=2,
            patches_per_image=2,
            image_size=(4, 4),
        )
        torch.testing.assert_close(bpp, torch.tensor([1.5, 3.5]))
        self.assertEqual(tuple(bpp.shape), (2,))
        self.assertTrue(torch.isfinite(bpp).all())

    def test_invalid_patch_inputs_raise_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "not divisible"):
            split_image_into_patches(torch.zeros(1, 3, 512, 768), 512)
        with self.assertRaisesRegex(ValueError, "Expected 4 patches"):
            merge_image_patches(
                torch.zeros(1, 3, 3, 256, 256),
                image_size=(512, 512),
            )
        with self.assertRaisesRegex(ValueError, "Expected 4 patch bit counts"):
            patch_bits_to_image_bpp(
                torch.ones(3),
                batch_size=1,
                patches_per_image=4,
                image_size=(512, 512),
            )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            split_image_into_patches(torch.zeros(1, 3, 0, 512), 256)
        with self.assertRaisesRegex(ValueError, "must all be finite"):
            patch_bits_to_image_bpp(
                torch.tensor([float("nan")]),
                batch_size=1,
                patches_per_image=1,
                image_size=(512, 512),
            )
        with self.assertRaisesRegex(ValueError, "must not be negative"):
            patch_bits_to_image_bpp(
                torch.tensor([-1.0]),
                batch_size=1,
                patches_per_image=1,
                image_size=(512, 512),
            )


class JPEGSmokeTest(unittest.TestCase):
    def test_complete_images_shape_and_one_finite_bpp(self):
        codec = JPEGImageCodec(quality=75)
        for height, width in [
            (512, 512),
            (512, 1024),
            (1024, 1024),
            (1024, 1536),
        ]:
            with self.subTest(image_size=(height, width)):
                image = torch.rand(1, 3, height, width)
                reconstructed, bpp = codec(image)

                self.assertEqual(reconstructed.shape, image.shape)
                self.assertEqual(tuple(bpp.shape), (1,))
                self.assertTrue(torch.isfinite(reconstructed).all())
                self.assertTrue(torch.isfinite(bpp).all())
                self.assertGreater(bpp.item(), 0)


class CompanyCollateTest(unittest.TestCase):
    def test_variable_resolution_batch_has_actionable_error(self):
        omegaconf = types.ModuleType("omegaconf")
        omegaconf.__spec__ = importlib.machinery.ModuleSpec(
            "omegaconf",
            loader=None,
        )
        omegaconf.OmegaConf = object
        with patch.dict(sys.modules, {"omegaconf": omegaconf}):
            company_eval = importlib.import_module("ddp_test_for_company")

        samples = [
            {"img": torch.zeros(3, 512, 512), "path": "a.png"},
            {"img": torch.zeros(3, 512, 1024), "path": "b.png"},
        ]
        with self.assertRaisesRegex(ValueError, "--batch_size 1"):
            company_eval._collate_without_resize_or_padding(samples)

        self.assertEqual(
            company_eval.parse_args([]).image_dataset_config,
            "config/image_datasets/company_full.yaml",
        )

    def test_company_script_uses_unified_codec_interface(self):
        source = Path("ddp_test_for_company.py").read_text(encoding="utf-8")
        self.assertIn('img = batch["img"].cuda()', source)
        self.assertIn("rec, bpp = codec(img)", source)
        self.assertNotIn('batch["patches"]', source)
        self.assertNotIn('batch["mode"]', source)
        self.assertNotIn("reconstruct_from_patch", source)


class CompanyDatasetConfigTest(unittest.TestCase):
    def test_deprecated_patch_config_is_not_instantiable(self):
        config_path = (
            Path(__file__).parents[1]
            / "config"
            / "image_datasets"
            / "company_patch.yaml"
        )
        content = config_path.read_text(encoding="utf-8")

        self.assertIn(
            "config/image_datasets/company_full.yaml",
            content,
        )
        self.assertNotIn("type:", content)


class PatchCodecAdapterTest(unittest.TestCase):
    def test_bsq_adapter_for_multiple_sizes_and_batch_two(self):
        omegaconf = types.ModuleType("omegaconf")
        omegaconf.__spec__ = importlib.machinery.ModuleSpec(
            "omegaconf",
            loader=None,
        )
        omegaconf.OmegaConf = object
        with patch.dict(sys.modules, {"omegaconf": omegaconf}):
            bsq_module = importlib.import_module("cab.codec.bsq")

        class FakeBSQ(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.input_shapes = []
                self.index_shapes = []

            def forward(self, patches):
                self.input_shapes.append(tuple(patches.shape))
                indices = torch.zeros(
                    (patches.shape[0], 32 * 32),
                    dtype=torch.int64,
                )
                self.index_shapes.append(tuple(indices.shape))
                return patches.clone(), None, {"indices": indices}

        codec = bsq_module.BSQImageTokenizer.__new__(
            bsq_module.BSQImageTokenizer
        )
        torch.nn.Module.__init__(codec)
        codec.model = FakeBSQ()
        codec.bits_per_token = 18
        codec.downsample_factor = 8
        codec.bsq_config = types.SimpleNamespace(
            optimizer=types.SimpleNamespace(
                disable_amp=True,
                use_bf16=False,
            )
        )

        cases = [
            (1, 512, 512),
            (1, 512, 1024),
            (1, 1024, 1024),
            (1, 1024, 1536),
            (2, 512, 1024),
        ]
        for batch_size, height, width in cases:
            with self.subTest(
                batch_size=batch_size,
                image_size=(height, width),
            ):
                image = torch.rand(batch_size, 3, height, width)
                expected_patch_count = (
                    batch_size * (height // 256) * (width // 256)
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    reconstructed, bpp = codec(image)

                self.assertTrue(torch.equal(reconstructed, image))
                self.assertEqual(tuple(bpp.shape), (batch_size,))
                self.assertTrue(torch.isfinite(bpp).all())
                self.assertEqual(
                    codec.model.input_shapes[-1],
                    (expected_patch_count, 3, 256, 256),
                )
                self.assertEqual(
                    codec.model.index_shapes[-1],
                    (expected_patch_count, 32 * 32),
                )
                torch.testing.assert_close(
                    bpp,
                    torch.full((batch_size,), 18.0 / 64.0),
                )

    def test_perco_adapter_for_multiple_sizes_and_batch_two(self):
        transformers = types.ModuleType("transformers")
        transformers.Blip2Processor = object
        transformers.Blip2ForConditionalGeneration = object

        torchac = types.ModuleType("torchac")
        torchac.encode_float_cdf = lambda *args, **kwargs: b""

        pipeline_module = types.ModuleType(
            "cab.models.perco_src.pipeline_sd_perco"
        )
        pipeline_module.StableDiffusionPipelinePerco = object

        torchvision = types.ModuleType("torchvision")
        torchvision_transforms = types.ModuleType("torchvision.transforms")
        torchvision_transforms.ToTensor = object
        torchvision_transforms.ToPILImage = object
        torchvision.transforms = torchvision_transforms

        with patch.dict(
            sys.modules,
            {
                "transformers": transformers,
                "torchac": torchac,
                "cab.models.perco_src.pipeline_sd_perco": pipeline_module,
                "torchvision": torchvision,
                "torchvision.transforms": torchvision_transforms,
            },
        ):
            perco_module = importlib.import_module("cab.codec.perco")

        codec = perco_module.PerCoImageCodec.__new__(
            perco_module.PerCoImageCodec
        )
        torch.nn.Module.__init__(codec)
        codec.observed_patch_shapes = []

        def fake_forward_patches(self, patches):
            self.observed_patch_shapes.append(tuple(patches.shape))
            patch_bits = (
                torch.arange(
                    1,
                    patches.shape[0] + 1,
                    dtype=torch.float32,
                )
                * 8
            )
            return patches.clone(), patch_bits

        codec._forward_patches = types.MethodType(
            fake_forward_patches,
            codec,
        )

        cases = [
            (1, 512, 512),
            (1, 512, 1024),
            (1, 1024, 1024),
            (1, 1024, 1536),
            (2, 512, 1024),
        ]
        for batch_size, height, width in cases:
            with self.subTest(
                batch_size=batch_size,
                image_size=(height, width),
            ):
                image = self._small_integer_image(
                    batch_size,
                    height,
                    width,
                )
                patches_per_image = (height // 512) * (width // 512)
                total_patches = batch_size * patches_per_image
                reconstructed, bpp = codec(image)

                expected = []
                for batch_index in range(batch_size):
                    start = batch_index * patches_per_image + 1
                    stop = start + patches_per_image
                    expected.append(
                        sum(range(start, stop)) * 8 / (height * width)
                    )

                self.assertTrue(torch.equal(reconstructed, image))
                self.assertEqual(tuple(bpp.shape), (batch_size,))
                self.assertTrue(torch.isfinite(bpp).all())
                self.assertEqual(
                    codec.observed_patch_shapes[-1],
                    (total_patches, 3, 512, 512),
                )
                torch.testing.assert_close(
                    bpp,
                    torch.tensor(expected),
                )

    @staticmethod
    def _small_integer_image(batch_size, height, width):
        values = torch.arange(
            batch_size * 3 * height * width,
            dtype=torch.int32,
        )
        return values.reshape(batch_size, 3, height, width)


if __name__ == "__main__":
    unittest.main()
