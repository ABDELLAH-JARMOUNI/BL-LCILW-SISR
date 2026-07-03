"""
evaluate_models.py
========================================================================
Set5 evaluation of the interpolation-stage LCI priors: compares standard
LCI (w=1), the global learned weights (Algorithm 1), and the spatial
learned weights (patch-wise) at x4 scale, reporting MATLAB-Y PSNR.

Author: Abdellah Jarmouni
"""

import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from weighted_lci_upsampler import GlobalWeightedLCI2D, PatchWeightedLCI2D
from psnr_luminance import calculate_benchmark_psnr

def _degrade_image(hr_t: torch.Tensor, scale: int, device: torch.device) -> torch.Tensor:
    """Matches the exact 5x5 Gaussian degradation used in training."""
    coords = torch.arange(5, dtype=torch.float32, device=device) - 2
    g1d = torch.exp(-coords ** 2 / 2.0)
    g2d = (g1d[:, None] @ g1d[None, :])
    ker = (g2d / g2d.sum()).view(1, 1, 5, 5).repeat(3, 1, 1, 1)
    
    blur = F.conv2d(hr_t, ker, padding=2, groups=3)
    return F.avg_pool2d(blur, scale, scale)

@torch.no_grad()
def run_evaluation(set5_dir: str, global_path: str, spatial_path: str, scale: int = 4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running evaluation on: {device}\n")

    # 1. Initialize Models
    global_model = GlobalWeightedLCI2D(n_nodes=8, scale_factor=scale).to(device)
    spatial_model = PatchWeightedLCI2D(n_nodes=8, scale_factor=scale).to(device)

    # 2. Load Weights (Fail gracefully if not found)
    if os.path.exists(global_path):
        global_model.load_state_dict(torch.load(global_path, map_location=device))
        print(f"[+] Loaded Global Weights: {global_path}")
    else:
        print(f"[-] Missing Global Weights: {global_path}")

    if os.path.exists(spatial_path):
        spatial_model.load_state_dict(torch.load(spatial_path, map_location=device))
        print(f"[+] Loaded Spatial Weights: {spatial_path}")
    else:
        print(f"[-] Missing Spatial Weights: {spatial_path}")

    global_model.eval()
    spatial_model.eval()

    # 3. Process Dataset
    paths = sorted(glob.glob(os.path.join(set5_dir, '*.*')))
    if not paths:
        print(f"\nError: No images found in {set5_dir}")
        return

    psnr_std, psnr_global, psnr_spatial = 0.0, 0.0, 0.0

    print("\nEvaluating images...")
    for p in paths:
        # Load and prep HR image
        hr = np.array(Image.open(p).convert('RGB'), np.float32) / 255.0
        hr_t = torch.from_numpy(hr).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Crop to divisible by scale
        _, _, h0, w0 = hr_t.shape
        hr_t = hr_t[:, :, :(h0 // scale) * scale, :(w0 // scale) * scale]
        
        # Generate LR image
        lr_t = _degrade_image(hr_t, scale, device)

        # Ensure spatial dims match for PSNR calculation
        def get_h_w(out_t, ref_t):
            return min(out_t.shape[2], ref_t.shape[2]), min(out_t.shape[3], ref_t.shape[3])

        # --- A. Standard LCI (Global model with theta=0) ---
        saved_theta = global_model.theta.detach().clone()
        global_model.theta.data.zero_()
        out_std = global_model(lr_t).clamp(0, 1)
        h, w = get_h_w(out_std, hr_t)
        psnr_std += float(calculate_benchmark_psnr(out_std[:, :, :h, :w], hr_t[:, :, :h, :w], scale))

        # --- B. Global Learned LCI ---
        global_model.theta.data.copy_(saved_theta)
        out_global = global_model(lr_t).clamp(0, 1)
        h, w = get_h_w(out_global, hr_t)
        psnr_global += float(calculate_benchmark_psnr(out_global[:, :, :h, :w], hr_t[:, :, :h, :w], scale))

        # --- C. Spatial Learned LCI ---
        out_spatial = spatial_model(lr_t).clamp(0, 1)
        h, w = get_h_w(out_spatial, hr_t)
        psnr_spatial += float(calculate_benchmark_psnr(out_spatial[:, :, :h, :w], hr_t[:, :, :h, :w], scale))

    # 4. Average results
    num_imgs = len(paths)
    psnr_std /= num_imgs
    psnr_global /= num_imgs
    psnr_spatial /= num_imgs

    # 5. Print Report-Ready Table
    print("\n" + "="*60)
    print(f"{'Method':<30} | {'PSNR (dB)':<10} | {'Delta':<10}")
    print("-" * 60)
    print(f"{'Standard LCI (w=1)':<30} | {psnr_std:<10.3f} | {'Baseline':<10}")
    print(f"{'Global Weights (Algorithm 1)':<30} | {psnr_global:<10.3f} | {psnr_global - psnr_std:<+10.3f}")
    print(f"{'Spatial Weights (Patch-wise)':<30} | {psnr_spatial:<10.3f} | {psnr_spatial - psnr_std:<+10.3f}")
    print("=" * 60)
    print("\n* Evaluated on Set5 (x4 Scale)")
    print("* Stage: Interpolation Prior (Lower-Level initialization)")

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--set5_dir', default='./Set5')
    ap.add_argument('--global_weights', default='w_star_global.pth', help='Path to saved global weights')
    ap.add_argument('--spatial_weights', default='phi_xi_spatial_weights.pth', help='Path to saved spatial weights')
    ap.add_argument('--scale', type=int, default=4)
    args = ap.parse_args()

    run_evaluation(
        set5_dir=args.set5_dir,
        global_path=args.global_weights,
        spatial_path=args.spatial_weights,
        scale=args.scale
    )
