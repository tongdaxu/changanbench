"""CAB metric interface for VLM output agreement."""
from types import SimpleNamespace

import torch
from cab.evaluations.abs import MetricIface
from .benchmark_tasks import TASKS, fields_for
from .metrics import evaluate_records, read_records


class VLMMetric(MetricIface):
    """Return one consistency score per sample, like other CAB metrics.

    With predictor=None, inputs are batches of VLM answers. With a predictor,
    inputs are image/video tensors and predictor(batch) must return one answer
    per sample, using the same prompt for original and reconstructed inputs.
    """

    def __init__(self, task, score="json_exact_match", predictor=None,
                 embedding_model=None, device=None, batch_size=32,
                 iou_threshold=0.5):
        super().__init__()
        if task not in TASKS:
            raise ValueError(f"Unknown VLM task: {task}")
        if not 0 < iou_threshold <= 1 or batch_size < 1:
            raise ValueError("Invalid IoU threshold or batch size")
        self.name = "vlm"
        self.score = score
        self.predictor = predictor
        exact, text, sets, detections = fields_for(task)
        self.args = SimpleNamespace(task=task, exact_fields=exact,
            text_fields=text, set_fields=sets, detection_fields=detections,
            embedding_model=embedding_model, device=device,
            batch_size=batch_size, iou_threshold=iou_threshold)
        self.encoder = None

    def evaluate(self, reference, candidate):
        """Return summary and details for ID-keyed answer dictionaries."""
        needs_text = self.args.text_fields or self.args.task == "rec_cabin"
        if needs_text and self.encoder is None:
            if not self.args.embedding_model:
                raise ValueError("This task requires embedding_model")
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.args.embedding_model,
                                                device=self.args.device)
        return evaluate_records(reference, candidate, self.args, self.encoder)

    def evaluate_files(self, reference, candidate):
        return self.evaluate(read_records(reference), read_records(candidate))

    @torch.no_grad()
    def forward(self, x_input, x_recon, zero_mean=False, **kwargs):
        device = x_input.device if isinstance(x_input, torch.Tensor) else "cpu"
        if len(x_input) != len(x_recon) or len(x_input) == 0:
            raise ValueError("Expected equally sized, nonempty batches")
        if self.predictor is not None:
            if zero_mean:
                x_input, x_recon = (x_input + 1) / 2, (x_recon + 1) / 2
            reference, candidate = self.predictor(x_input), self.predictor(x_recon)
        else:
            if isinstance(x_input, torch.Tensor) or isinstance(x_recon, torch.Tensor):
                raise ValueError("Tensor inputs require a VLM predictor")
            reference, candidate = x_input, x_recon
        if len(reference) != len(x_input) or len(candidate) != len(x_input):
            raise ValueError("Predictor must return one answer per sample")
        _, rows = self.evaluate(dict(enumerate(reference)), dict(enumerate(candidate)))
        if not any(self.score in row["metrics"] for row in rows):
            raise ValueError(f"Score unavailable for this batch: {self.score}")
        return torch.tensor([row["metrics"].get(self.score) if
            row["metrics"].get(self.score) is not None else float("nan")
            for row in rows], dtype=torch.float32, device=device)
