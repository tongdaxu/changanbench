from .fid_score import calculate_frechet_distance
from .inception import InceptionV3
import numpy as np
import torch.nn.functional as F

def ensure_image_normalized(x, metrics_zero_mean):
    if metrics_zero_mean:
        return x
    else:
        return 2 * x - 1
    
def build_inception():
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx], normalize_input=False).cuda()
    model.eval()
    return model

def get_fid_batch(x_input, x_recon, metrics_zero_mean=None):

    # Get activations
    x_input = ensure_image_normalized(x_input, metrics_zero_mean)
    x_recon = ensure_image_normalized(x_recon, metrics_zero_mean)
    
    pred_x = build_inception()(x_input)[0]
    if pred_x.size(2) != 1 or pred_x.size(3) != 1:
        pred_x = F.adaptive_avg_pool2d(pred_x, (1, 1))
    pred_x = pred_x.squeeze(3).squeeze(2)

    pred_xr = build_inception()(x_recon)[0]
    if pred_xr.size(2) != 1 or pred_xr.size(3) != 1:
        pred_xr = F.adaptive_avg_pool2d(pred_xr, (1, 1))
    pred_xr = pred_xr.squeeze(3).squeeze(2)
    return pred_x, pred_xr

def compute_fid_from_activations(all_pred_x, all_pred_xr):
    # Calculate statistics
    mu_x = np.mean(all_pred_x, axis=0)
    sigma_x = np.cov(all_pred_x, rowvar=False)
    
    mu_xr = np.mean(all_pred_xr, axis=0)
    sigma_xr = np.cov(all_pred_xr, rowvar=False)
    
    # Calculate FID
    fid_value = calculate_frechet_distance(mu_xr, sigma_xr, mu_x, sigma_x)
    
    return fid_value