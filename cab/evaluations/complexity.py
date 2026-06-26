import torch
from cab.evaluations.abs import MetricIface


class ComplexityMetric(MetricIface):
    """
    Treat complexity as a metric.

    It does not compare x_input and x_recon.
    It measures the bound codec once per codec/dataset on rank0.
    """

    is_complexity_metric = True

    def __init__(
        self,
        image_size=256,
        batch_size=1,
        warmup=3,
        repeat=10,
        **kwargs,
    ):
        super().__init__()
        self.name = "complexity"
        self.image_size = image_size
        self.batch_size = batch_size
        self.warmup = warmup
        self.repeat = repeat
        self.codec = None
        self.result = None

    def bind_codec(self, codec):
        self.codec = codec
        return self

    @torch.no_grad()
    def compute(self, device=None):
        if self.codec is None:
            raise RuntimeError("ComplexityMetric must bind a codec before compute().")

        if device is None:
            device = next(self.codec.parameters()).device
        device = torch.device(device)

        self.codec.eval()

        if hasattr(self.codec, "fake_input"):
            x = self.codec.fake_input(
                image_size=self.image_size,
                batch_size=self.batch_size,
                device=device,
            )
        else:
            x = torch.randn(
                self.batch_size, 3, self.image_size, self.image_size,
                device=device,
            )

        def safe_call(name, *args, default=None, **kwargs):
            fn = getattr(self.codec, name, None)
            if fn is None:
                return default
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                print(f"[ComplexityMetric] {name} failed: {e}")
                return default

        self.result = {
            "encode_params_m": safe_call("encode_params_m", default=None),
            "decode_params_m": safe_call("decode_params_m", default=None),
            "encode_time_ms": safe_call(
                "encode_time_ms", x,
                warmup=self.warmup,
                repeat=self.repeat,
                default=None,
            ),
            "decode_time_ms": safe_call(
                "decode_time_ms", x,
                warmup=self.warmup,
                repeat=self.repeat,
                default=None,
            ),
            "encode_gflops": safe_call("encode_gflops", x, default=None),
            "decode_gflops": safe_call("decode_gflops", x, default=None),
        }
        return self.result

    def forward(self, x_input, x_recon, zero_mean=False, **kwargs):
        # DDP evaluation loop will skip this metric in per-batch metric computation.
        return None

    def format_result(self):
        if self.result is None:
            return "complexity  : N/A"

        def fmt(v, suffix=""):
            if v is None:
                return "N/A"
            return f"{v:.3f}{suffix}"

        r = self.result
        return (
            f"{'complexity':12s}: "
            f"EncParams={fmt(r['encode_params_m'], 'M')}, "
            f"DecParams={fmt(r['decode_params_m'], 'M')}, "
            f"EncTime={fmt(r['encode_time_ms'], 'ms')}, "
            f"DecTime={fmt(r['decode_time_ms'], 'ms')}, "
            f"EncFLOPs={fmt(r['encode_gflops'], 'G')}, "
            f"DecFLOPs={fmt(r['decode_gflops'], 'G')}"
        )