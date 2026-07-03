"""
train_rgdn_bsds500.py
========================================================================
RGDN denoiser training for BL-LCILW / PnP-ADMM.

Key properties:
  1. Default model is 64 features / 8 blocks (BatchNorm-free).
  2. Uses safer gradient-attention normalization (see rgdn_model.py).
  3. Adds a finite-difference local Jacobian penalty to discourage local
     expansiveness while preserving denoising power.
  4. Training sigma distribution matches the ADMM operating range:
     70% in [0, 0.15], 30% in [0.15, 0.50].

Recommended first run:
  python train_rgdn_bsds500.py \
    --data_root ./data/BSDS500 \
    --save_dir ./rgdn_checkpoints \
    --batch_size 32 \
    --num_workers 8 \
    --val_every 10 \
    --save_every 10 \
    --jacobian_weight 3e-4 \
    --jacobian_eps 0.003

Author: Abdellah Jarmouni
"""

import os
import argparse
import random
import glob
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import warnings
import torch.nn.functional as F

warnings.filterwarnings('ignore')

from rgdn_model import RGDN


# ============================================================
# CUDNN & HARDWARE OPTIMIZATIONS
# ============================================================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ============================================================
# DATASET (70/30 Sigma Split)
# ============================================================
class BSDS500Dataset(Dataset):
    def __init__(self, root_dir: str, split: str = 'train', patch_size: int = 64, augment: bool = True):
        self.patch_size = patch_size
        self.augment = augment
        self.image_paths = []
        candidates = [
            os.path.join(root_dir, 'images', split),
            os.path.join(root_dir, split),
            os.path.join(root_dir, 'BSDS500', 'images', split),
            root_dir,
        ]
        for d in candidates:
            if os.path.exists(d):
                for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']:
                    self.image_paths.extend(glob.glob(os.path.join(d, '**', ext), recursive=True))
                if self.image_paths:
                    break

        if not self.image_paths:
            raise FileNotFoundError(f"No images found under '{root_dir}'.")

    def __len__(self):
        return len(self.image_paths) * 50

    def __getitem__(self, idx):
        img_idx = idx % len(self.image_paths)
        try:
            img = np.array(Image.open(self.image_paths[img_idx]).convert('RGB'), dtype=np.float32) / 255.0
        except Exception:
            return self.__getitem__(random.randint(0, len(self) - 1))

        H, W = img.shape[:2]
        p = self.patch_size

        if H > p and W > p:
            i, j = random.randint(0, H - p), random.randint(0, W - p)
            clean = img[i:i + p, j:j + p].copy()
        else:
            from skimage.transform import resize
            clean = resize(img, (p, p), anti_aliasing=True).astype(np.float32)

        if self.augment:
            if random.random() > 0.5:
                clean = np.flip(clean, axis=1).copy()
            if random.random() > 0.5:
                clean = np.flip(clean, axis=0).copy()
            if random.random() > 0.5:
                clean = np.rot90(clean, k=random.randint(1, 3)).copy()

        clean = torch.from_numpy(clean).permute(2, 0, 1).float()

        # Match ADMM operating range: sigma = alpha/rho can reach ~0.50 when rho=0.05.
        if random.random() < 0.70:
            sigma_val = random.uniform(0.00, 0.15)
        else:
            sigma_val = random.uniform(0.15, 0.50)

        if random.random() > 0.5:
            sigma_map = torch.full((1, p, p), sigma_val, dtype=torch.float32)
        else:
            grid_size = max(1, p // 8)
            low_res_sigma = torch.clamp(
                torch.randn(1, 1, grid_size, grid_size) * 0.05 + sigma_val,
                0.0,
                0.50,
            )
            sigma_map = F.interpolate(low_res_sigma, size=(p, p), mode='bicubic', align_corners=False).squeeze(0)

        noisy = (clean + torch.randn_like(clean) * sigma_map).clamp(0.0, 1.0)
        return noisy, clean, sigma_map


# ============================================================
# LOSS FUNCTION
# ============================================================
class RGDNLoss(nn.Module):
    def __init__(self, gradient_weight: float = 0.1, identity_weight: float = 0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.gradient_weight = gradient_weight
        self.identity_weight = identity_weight

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def _gradients(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True) if x.shape[1] > 1 else x
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, identity_pred: torch.Tensor = None):
        loss_img = self.l1(pred, target)
        loss_grad = self.l1(self._gradients(pred), self._gradients(target))

        loss_id = torch.tensor(0.0, device=pred.device)
        if identity_pred is not None:
            loss_id = self.l1(identity_pred, target)

        total = loss_img + (self.gradient_weight * loss_grad) + (self.identity_weight * loss_id)
        return total, loss_img, loss_grad, loss_id


# ============================================================
# LOCAL JACOBIAN PENALTY — FP32, outside AMP
# ============================================================
def local_jacobian_penalty_fp32(model, noisy, sigma, eps: float, margin: float, detach_base: bool = False):
    """Finite-difference local Lipschitz penalty computed in full FP32.

    Why FP32 matters:
      The input perturbation has L2 norm eps, so its per-pixel amplitude is tiny.
      Computing this inside AMP/FP16 can make eta noisy and can train the wrong
      signal. For PnP stability, keep this estimate in FP32.

    The estimate is:
        eta = ||D(x + eps*d) - D(x)||_2 / eps,
    where ||d||_2 = 1 for each sample.

    This is a training regularizer, not a mathematical proof of non-expansiveness.
    """
    noisy32 = noisy.float()
    sigma32 = sigma.float()

    d = torch.randn_like(noisy32)
    d = d / (d.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12)
    noisy_pert = (noisy32 + eps * d).clamp(0.0, 1.0)

    # Disable autocast explicitly so both forward passes are full FP32.
    with autocast(enabled=False):
        base = model(noisy32, sigma32)
        if detach_base:
            base = base.detach()
        pert = model(noisy_pert, sigma32)
        eta = (pert - base).flatten(1).norm(dim=1) / eps
        penalty = torch.relu(eta - margin).pow(2).mean()

    return penalty, eta.detach().mean()


# ============================================================
# VALIDATION
# ============================================================
@torch.no_grad()
def validate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    sigma_levels = [15, 25, 50, 100, 127]
    results = {}

    for sigma_val in sigma_levels:
        sigma_norm = sigma_val / 255.0
        total_psnr = count = 0.0

        for _, clean, _ in val_loader:
            clean = clean.to(device)
            noisy_fixed = (clean + torch.randn_like(clean) * sigma_norm).clamp(0.0, 1.0)

            with autocast():
                denoised = model(noisy_fixed, sigma_norm)

            mse = torch.mean((denoised.float() - clean.float()) ** 2)
            psnr = 20.0 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
            total_psnr += psnr.item()
            count += 1

        avg_psnr = total_psnr / max(count, 1)
        results[f'sigma_{sigma_val}'] = avg_psnr
        print(f"    σ={sigma_val:3d} ({sigma_norm:.4f}): PSNR = {avg_psnr:.2f} dB")

    return results


# ============================================================
# TRAINING LOOP
# ============================================================
def train_rgdn(args):
    print("=" * 70)
    print("RGDN Training — BN-free + FP32 Jacobian penalty")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    model = RGDN(
        in_channels=3,
        num_features=args.num_features,
        num_blocks=args.num_blocks,
        use_attention=True,
    ).to(device)

    if args.compile and hasattr(torch, 'compile'):
        print("[+] torch.compile enabled. First epoch may trace slowly; disable if checkpoint loading becomes annoying.")
        model = torch.compile(model)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"Model: RGDN({args.num_features} features, {args.num_blocks} blocks)")
    print(f"Parameters: {total_p:,}")
    print(f"Jacobian penalty: weight={args.jacobian_weight:g}, margin={args.jacobian_margin:g}, eps={args.jacobian_eps:g}, every={args.jacobian_every}, detach_base={args.jacobian_detach_base}")

    print("\nLoading datasets...")
    train_dataset = BSDS500Dataset(root_dir=args.data_root, split=args.train_split,
                                   patch_size=args.patch_size, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)

    val_loader = None
    val_path = os.path.join(args.data_root, 'images', args.val_split)
    if args.val_split and os.path.exists(val_path):
        val_dataset = BSDS500Dataset(root_dir=args.data_root, split=args.val_split,
                                     patch_size=args.patch_size, augment=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
    else:
        print(f"[!] Validation folder not found at {val_path}; training will still run, but no best_model will be selected.")

    criterion = RGDNLoss(gradient_weight=args.gradient_weight, identity_weight=args.identity_weight).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    os.makedirs(args.save_dir, exist_ok=True)
    best_psnr = 0.0

    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = epoch_img = epoch_grad = epoch_id = epoch_jac = epoch_eta = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for noisy, clean, sigma in pbar:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            sigma = sigma.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            # Main denoising losses can use AMP for speed.
            with autocast():
                denoised = model(noisy, sigma)
                tiny_sigma = torch.full_like(sigma, 0.001)
                identity_pred = model(clean, tiny_sigma)
                loss, loss_img, loss_grad, loss_id = criterion(denoised, clean, identity_pred)

            # The local Jacobian penalty must be computed in FP32, outside AMP.
            if args.jacobian_weight > 0 and ((global_step % args.jacobian_every) == 0):
                jac_penalty, eta_mean = local_jacobian_penalty_fp32(
                    model, noisy, sigma,
                    eps=args.jacobian_eps,
                    margin=args.jacobian_margin,
                    detach_base=args.jacobian_detach_base,
                )
                loss = loss + args.jacobian_weight * jac_penalty
            else:
                jac_penalty = torch.tensor(0.0, device=device)
                eta_mean = torch.tensor(0.0, device=device)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

            epoch_loss += loss.item()
            epoch_img += loss_img.item()
            epoch_grad += loss_grad.item()
            epoch_id += loss_id.item()
            epoch_jac += jac_penalty.item()
            epoch_eta += eta_mean.item()

            pbar.set_postfix({
                'L1': f'{loss_img.item():.4f}',
                'Id': f'{loss_id.item():.4f}',
                'Jac': f'{jac_penalty.item():.3f}',
                'eta': f'{eta_mean.item():.2f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
            })

        scheduler.step()

        n = len(train_loader)
        print(
            f"\nEpoch {epoch+1}/{args.epochs}: "
            f"Loss: {epoch_loss/n:.6f} | L1: {epoch_img/n:.6f} | "
            f"Grad: {epoch_grad/n:.6f} | Id: {epoch_id/n:.6f} | "
            f"Jac: {epoch_jac/n:.6f} | eta_mean: {epoch_eta/n:.3f}"
        )

        if val_loader and (epoch + 1) % args.val_every == 0:
            val_results = validate(model, val_loader, device)
            avg_psnr = sum(val_results.values()) / len(val_results)
            print(f"  Avg PSNR across σ=[15,25,50,100,127]: {avg_psnr:.2f} dB")
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'args': args,
                    'best_avg_psnr': best_psnr,
                }, os.path.join(args.save_dir, 'best_model_rgdn.pth'))
                print(f"  ✓ New best saved (avg PSNR = {best_psnr:.2f} dB)")

        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'args': args,
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    print("\nTraining complete!")
    print(f"Best avg PSNR: {best_psnr:.2f} dB")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./BSDS500')
    parser.add_argument('--train_split', type=str, default='train')
    parser.add_argument('--val_split', type=str, default='val')
    parser.add_argument('--save_dir', type=str, default='./rgdn_checkpoints')
    parser.add_argument('--patch_size', type=int, default=64)

    # Model capacity: the recommended safe prior is 64 features / 8 blocks.
    parser.add_argument('--num_features', type=int, default=64)
    parser.add_argument('--num_blocks', type=int, default=8)

    # Training Params
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--val_every', type=int, default=10)

    # Hardware Params
    parser.add_argument('--compile', action='store_true', help='Enable torch.compile. Not recommended for first run.')

    # Loss Params
    parser.add_argument('--gradient_weight', type=float, default=0.1)
    parser.add_argument('--identity_weight', type=float, default=0.1)

    # Local Jacobian regularization.
    parser.add_argument('--jacobian_weight', type=float, default=1e-4,
                        help='Weight of finite-difference local Lipschitz penalty. Use 0 to disable.')
    parser.add_argument('--jacobian_margin', type=float, default=0.9,
                        help='Penalty activates when local finite-difference eta exceeds this value.')
    parser.add_argument('--jacobian_eps', type=float, default=3e-3,
                        help='Finite-difference epsilon for local Jacobian penalty. 3e-3 is safer with AMP main training.')
    parser.add_argument('--jacobian_every', type=int, default=1,
                        help='Compute Jacobian penalty every N optimizer steps. Use 1 for strongest regularization.')
    parser.add_argument('--jacobian_detach_base', action='store_true',
                        help='Detach D(x) baseline inside Jacobian penalty. Usually leave disabled.')

    args = parser.parse_args()
    if args.jacobian_every < 1:
        raise ValueError('--jacobian_every must be >= 1')
    os.makedirs(args.save_dir, exist_ok=True)
    train_rgdn(args)


if __name__ == '__main__':
    main()
