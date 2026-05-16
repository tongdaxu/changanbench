from DISTS_pytorch import DISTS

def get_dists(x_input, x_recon, zero_mean=False):
    if zero_mean:
        x_input = x_input * 0.5 + 0.5
        x_recon = x_recon * 0.5 + 0.5
    dists = DISTS().to(x_input.device)
    return dists(x_input, x_recon)