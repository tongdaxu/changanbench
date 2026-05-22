from .fid_score import calculate_frechet_distance
from .inception import InceptionV3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cab.evaluations.abs import MetricIface


class FIDMetric(MetricIface):
    """Frechet Inception Distance (FID)
    """

    def __init__(self, normalize_input: bool = False, zero_mean: bool = False, **kwargs):
        super().__init__()
        self.normalize_input = normalize_input
        self.name = "fid"
        
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inception = InceptionV3(
            [block_idx], 
            normalize_input=normalize_input
        )
        self.inception.eval().cuda()
        
        self.reset()

    def reset(self):
        self.all_activations_x = [] 
        self.all_activations_xr = []  

    @torch.no_grad()
    def forward(self, x_input: torch.Tensor, x_recon: torch.Tensor,
                zero_mean: bool = False, **kwargs) -> None:
        
        if not zero_mean:
            x_input = 2 * x_input - 1
            x_recon = 2 * x_recon - 1
        
        pred_x = self._extract_activation(x_input)
        pred_xr = self._extract_activation(x_recon)
        
        self.all_activations_x.append(pred_x.detach().cpu().numpy())
        self.all_activations_xr.append(pred_xr.detach().cpu().numpy())

    def _extract_activation(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.inception(x)[0]  # (B, 2048, H, W)
        
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = F.adaptive_avg_pool2d(pred, (1, 1))
        
        # (B, 2048, 1, 1) → (B, 2048)
        pred = pred.squeeze(3).squeeze(2)
        return pred

    def compute(self) -> float:
        
        if len(self.all_activations_x) == 0:
            raise RuntimeError("No activations accumulated. Call forward() first.")
        
        all_pred_x = np.vstack(self.all_activations_x)
        all_pred_xr = np.vstack(self.all_activations_xr)
        
        
        mu_x = np.mean(all_pred_x, axis=0)
        sigma_x = np.cov(all_pred_x, rowvar=False)
        
        mu_xr = np.mean(all_pred_xr, axis=0)
        sigma_xr = np.cov(all_pred_xr, rowvar=False)
        
        fid_score = calculate_frechet_distance(mu_xr, sigma_xr, mu_x, sigma_x)
        
        return float(fid_score)





_fid_metric_instance = None

def get_fid_batch(x_input, x_recon, metrics_zero_mean=None):
    
    global _fid_metric_instance
    
    if _fid_metric_instance is None:
        _fid_metric_instance = FIDMetric()
    
    _fid_metric_instance(x_input, x_recon, zero_mean=metrics_zero_mean)
    
    
    if len(_fid_metric_instance.all_activations_x) > 0:
        pred_x = _fid_metric_instance.all_activations_x[-1]
        pred_xr = _fid_metric_instance.all_activations_xr[-1]
        return torch.from_numpy(pred_x), torch.from_numpy(pred_xr)
    return None, None


def compute_fid_from_activations(all_pred_x, all_pred_xr):
    mu_x = np.mean(all_pred_x, axis=0)
    sigma_x = np.cov(all_pred_x, rowvar=False)
    
    mu_xr = np.mean(all_pred_xr, axis=0)
    sigma_xr = np.cov(all_pred_xr, rowvar=False)
    
    fid_value = calculate_frechet_distance(mu_xr, sigma_xr, mu_x, sigma_x)
    return fid_value