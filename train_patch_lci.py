"""
train_patch_lci.py
========================================================================
Algorithm 1 (patch-wise / spatially adaptive variant): trains the
PatchWeightNetwork that produces a dense, space-variant weight map for the
Learning-Weighted LCI upsampler U_w on BSDS500 super-resolution.

The objective combines a data loss with the Lebesgue-constant penalty
(Eq. 16) and an interpolation-preservation penalty (w -> 1).

Author: Abdellah Jarmouni
"""

import math
import os
import glob
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from weighted_lci_upsampler import PatchWeightedLCI2D, all_lagrange_bases
from psnr_luminance import calculate_benchmark_psnr

# ============================================================
# Dataset: BSDS500 Super-Resolution
# ============================================================

class BSDS500SRDataset(Dataset):
    """Loads images from a BSDS500-style directory and returns (LR, HR) patch pairs."""
    def __init__(self, root_dir: str, split: str = 'train', hr_patch: int = 128, 
                 scale: int = 4, samples_per_image: int = 30, augment: bool = True):
        self.hr_patch  = hr_patch
        self.scale     = scale
        self.spi       = samples_per_image
        self.augment   = augment

        self._blur_kernel = self._make_gaussian_kernel(sigma=1.0, kernel_size=5)

        self.paths = []
        candidates = [
            os.path.join(root_dir, 'images', split),
            os.path.join(root_dir, split),
            root_dir,
        ]
        for d in candidates:
            if os.path.isdir(d):
                for ext in ('*.jpg', '*.png', '*.jpeg'):
                    self.paths += glob.glob(os.path.join(d, '**', ext), recursive=True)
                if self.paths: break

        self.images = []
        for p in tqdm(self.paths, desc=f'Pre-loading {split}'):
            try:
                img = np.array(Image.open(p).convert('RGB'), dtype=np.float32) / 255.0
                self.images.append(img)
            except Exception: pass

    @staticmethod
    def _make_gaussian_kernel(sigma: float = 1.0, kernel_size: int = 5) -> torch.Tensor:
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        g1d    = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g2d    = g1d.outer(g1d)
        g2d    = g2d / g2d.sum()
        return g2d.view(1, 1, kernel_size, kernel_size)

    def _degrade(self, hr_t: torch.Tensor) -> torch.Tensor:
        C = hr_t.shape[0]
        kernel = self._blur_kernel.repeat(C, 1, 1, 1).to(hr_t.device)
        pad = kernel.shape[-1] // 2
        blurred = F.conv2d(hr_t.unsqueeze(0), kernel, padding=pad, groups=C).squeeze(0)
        lr = F.avg_pool2d(blurred.unsqueeze(0), self.scale, stride=self.scale).squeeze(0)
        return lr.clamp(0.0, 1.0)

    def __len__(self) -> int:
        return len(self.images) * self.spi

    def __getitem__(self, idx: int):
        img = self.images[idx % len(self.images)]
        H, W = img.shape[:2]

        if H >= self.hr_patch and W >= self.hr_patch:
            i = random.randint(0, H - self.hr_patch)
            j = random.randint(0, W - self.hr_patch)
            hr = img[i:i + self.hr_patch, j:j + self.hr_patch].copy()
        else:
            from skimage.transform import resize
            scale_up = max(self.hr_patch / H, self.hr_patch / W) + 0.01
            big = resize(img, (int(math.ceil(H * scale_up)), int(math.ceil(W * scale_up)), 3), anti_aliasing=True).astype(np.float32)
            i = random.randint(0, big.shape[0] - self.hr_patch)
            j = random.randint(0, big.shape[1] - self.hr_patch)
            hr = big[i:i + self.hr_patch, j:j + self.hr_patch].copy()

        if self.augment:
            if random.random() > 0.5: hr = hr[:, ::-1, :].copy()
            if random.random() > 0.5: hr = hr[::-1, :, :].copy()
            k = random.randint(0, 3)
            if k > 0: hr = np.rot90(hr, k=k).copy()

        hr_t = torch.from_numpy(hr).permute(2, 0, 1).float()
        lr_t = self._degrade(hr_t)
        return lr_t, hr_t

# ============================================================
# Algorithm 1: Patch-Wise Training Loop
# ============================================================

def train_patch_weights(
    dataloader,
    val_dataloader=None,
    epochs: int       = 100,
    lr: float         = 1e-3,
    lambda_leb: float = 1e-6,  # Scaled down for mean reduction
    lambda_int: float = 1e-4,  # Scaled down for mean reduction
    beta: float       = 10.0,
    device: str       = 'cuda',
    save_path: str    = 'phi_xi_spatial_weights.pth',
):
    model = PatchWeightedLCI2D(n_nodes=8, scale_factor=4).to(device)
    optimizer = optim.Adam(model.weight_net.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Precompute dense Lagrange matrix for Lebesgue penalty
    nodes = model.nodes
    t_dense = torch.linspace(-1.0, 1.0, 500, device=device)
    L_dense = all_lagrange_bases(t_dense, nodes)  # [500, n_nodes]

    best_val_psnr = -float('inf')

    # Ensure baseline standard LCI PSNR is logged correctly before training
    print("\nEvaluating initial Standard LCI baseline (w=1)...")
    if val_dataloader is not None:
        best_val_psnr = _evaluate_psnr(model, val_dataloader, device)
        print(f"Baseline Val PSNR: {best_val_psnr:.3f} dB\n")

    for epoch in range(epochs):
        model.train()
        epoch_data, epoch_leb, epoch_int, epoch_total = 0.0, 0.0, 0.0, 0.0
        num_batches = 0

        for lr_imgs, hr_imgs in dataloader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            optimizer.zero_grad()

            # Forward pass: x_hat = U_{w(y)} * y
            x_hat = model(lr_imgs)

            # Retrieve the spatial weight map: [B, n_nodes, H, W]
            w_map = model.get_w_map(lr_imgs)
            B, n, H, W = w_map.shape
            
            # Flatten spatial dimensions to compute penalties per-pixel: [B*H*W, n_nodes]
            w_flat = w_map.permute(0, 2, 3, 1).reshape(-1, n)

            # 1. Data Loss (Standard PyTorch Mean Reduction)
            L_data = F.mse_loss(x_hat, hr_imgs)

            # 2. Lebesgue Penalty (Vectorized across all spatial pixels)
            # row_sums shape: [B*H*W, 500]
            row_sums = torch.matmul(w_flat, L_dense.abs().T)
            max_val = row_sums.max(dim=1, keepdim=True)[0]
            
            leb_pixels = (1.0 / beta) * (
                max_val.squeeze(-1) + 
                torch.log(torch.mean(torch.exp(beta * (row_sums - max_val)), dim=1))
            )
            L_leb = lambda_leb * leb_pixels.mean()

            # 3. Interpolation Penalty (Enforces w_k ≈ 1 penalty per pixel)
            L_int = lambda_int * torch.mean(torch.sum((w_flat - 1.0) ** 2, dim=1))

            # Total Loss
            L_total = L_data + L_leb + L_int

            L_total.backward()
            optimizer.step()

            epoch_data  += L_data.item()
            epoch_leb   += L_leb.item()
            epoch_int   += L_int.item()
            epoch_total += L_total.item()
            num_batches += 1

        scheduler.step()

        # Validation
        val_str = ""
        if val_dataloader is not None:
            val_psnr = _evaluate_psnr(model, val_dataloader, device)
            val_str  = f" | Val PSNR: {val_psnr:.3f} dB"
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                torch.save(model.state_dict(), save_path)

        # Log typical weight spread to ensure it is learning (not pinned, not chaotic)
        w_min_val = w_flat.min().item()
        w_max_val = w_flat.max().item()

        print(
            f"Epoch [{epoch+1:03d}/{epochs}] "
            f"| Data: {epoch_data/num_batches:.6f} "
            f"| Leb: {epoch_leb/num_batches:.6f} "
            f"| Int: {epoch_int/num_batches:.6f}"
            f"{val_str}\n"
            f"  -> w_map range: [{w_min_val:.3f}, {w_max_val:.3f}]"
        )

    print(f"\nTraining complete. Best model saved to '{save_path}'.")
    return model

# ============================================================
# Validation Helper
# ============================================================

def _evaluate_psnr(model: nn.Module, dataloader, device: str) -> float:
    """Average MATLAB-Y PSNR over a dataloader."""
    model.eval()
    total_psnr = 0.0
    count      = 0
    with torch.no_grad():
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            sr = model(lr_imgs).clamp(0.0, 1.0)
            
            h, w = min(sr.shape[2], hr_imgs.shape[2]), min(sr.shape[3], hr_imgs.shape[3])
            p = calculate_benchmark_psnr(
                sr[:, :, :h, :w], hr_imgs[:, :, :h, :w], crop_border=model.scale
            )
            total_psnr += float(p)
            count += 1
    model.train()
    return total_psnr / max(count, 1)

@torch.no_grad()
def evaluate_on_set5(model, set5_dir, device, scale=4):
    """Reports U_{w(y)} PSNR on Set5 against the Standard LCI baseline."""
    paths = sorted(glob.glob(os.path.join(set5_dir, '*.*')))
    if not paths: return

    def run_eval(use_network=True):
        model.eval()
        tot = 0.0
        for p in paths:
            hr = np.array(Image.open(p).convert('RGB'), np.float32) / 255.0
            hr_t = torch.from_numpy(hr).permute(2, 0, 1).unsqueeze(0).to(device)
            _, _, h0, w0 = hr_t.shape
            hr_t = hr_t[:, :, :(h0 // scale) * scale, :(w0 // scale) * scale]
            
            # Degrade
            coords = torch.arange(5, dtype=torch.float32, device=device) - 2
            g1d = torch.exp(-coords ** 2 / 2.0)
            g2d = (g1d[:, None] @ g1d[None, :])
            ker = (g2d / g2d.sum()).view(1, 1, 5, 5).repeat(3, 1, 1, 1)
            blur = F.conv2d(hr_t, ker, padding=2, groups=3)
            lr_t = F.avg_pool2d(blur, scale, scale)
            
            # Forward Pass
            uw = model(lr_t).clamp(0, 1)
            h, w = min(uw.shape[2], hr_t.shape[2]), min(uw.shape[3], hr_t.shape[3])
            tot += float(calculate_benchmark_psnr(uw[:, :, :h, :w], hr_t[:, :, :h, :w], crop_border=scale))
        return tot / len(paths)

    # Standard LCI (zero out the final layer bias/weights temporarily)
    saved_weights = model.weight_net.mlp[-1].weight.clone()
    saved_bias = model.weight_net.mlp[-1].bias.clone()
    nn.init.zeros_(model.weight_net.mlp[-1].weight)
    nn.init.zeros_(model.weight_net.mlp[-1].bias)
    psnr_std = run_eval()
    
    # Learned Network LCI
    model.weight_net.mlp[-1].weight.copy_(saved_weights)
    model.weight_net.mlp[-1].bias.copy_(saved_bias)
    psnr_learned = run_eval()

    print("\n" + "=" * 56)
    print("Set5 Spatial U_{w(y)} PSNR (interpolation stage)")
    print("=" * 56)
    print(f"  standard LCI (w=1)   : {psnr_std:.3f} dB")
    print(f"  learned spatial map  : {psnr_learned:.3f} dB")
    print(f"  delta                : {psnr_learned - psnr_std:+.3f} dB")
    print("=" * 56)

# ============================================================
# Entry Point
# ============================================================

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='./data/BSDS500')
    ap.add_argument('--set5_dir', default='./Set5')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lambda_leb', type=float, default=1e-6)
    ap.add_argument('--lambda_int', type=float, default=1e-4)
    ap.add_argument('--beta', type=float, default=10.0)
    ap.add_argument('--hr_patch', type=int, default=128)
    ap.add_argument('--scale', type=int, default=4)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--save_path', default='phi_xi_spatial_weights.pth')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading dataset...")
    train_dataset = BSDS500SRDataset(
        root_dir=args.data_root, split='train', hr_patch=args.hr_patch,
        scale=args.scale, samples_per_image=30, augment=True)
    val_dataset = BSDS500SRDataset(
        root_dir=args.data_root, split='val', hr_patch=args.hr_patch,
        scale=args.scale, samples_per_image=5, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                            num_workers=4, pin_memory=True)

    print("\nStarting spatial weight network training...")
    trained_model = train_patch_weights(
        dataloader=train_loader, val_dataloader=val_loader,
        epochs=args.epochs, lr=args.lr, lambda_leb=args.lambda_leb,
        lambda_int=args.lambda_int, beta=args.beta, device=device,
        save_path=args.save_path)

    trained_model.load_state_dict(torch.load(args.save_path, map_location=device))
    evaluate_on_set5(trained_model, args.set5_dir, device, scale=args.scale)
