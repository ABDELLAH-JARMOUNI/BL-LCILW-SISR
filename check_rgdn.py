"""
check_rgdn.py
========================================================================
Three sanity tests for a trained RGDN checkpoint:

  1. Standalone denoising at the ADMM operating point (sigma ~ 15/255).
  2. Identity / texture preservation at tiny sigma (identity-loss check).
  3. Empirical contractivity / Lipschitz stability of the relaxed
     operator D_r(v) = v + r*(RGDN(v) - v) for r in {1.0, 0.7, 0.5}.

The model architecture is inferred from the checkpoint, so both 64/8 and
128/12 variants load correctly.

Author: Abdellah Jarmouni
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import argparse
from rgdn_model import RGDN, strip_compile_prefix, infer_rgdn_arch_from_state
from psnr_luminance import calculate_benchmark_psnr

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='best_model_rgdn.pth')
    parser.add_argument('--image', type=str, default='./Set5/bird.png')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model: {args.model}")

    # Load the checkpoint and infer the architecture from its state dict.
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    state = strip_compile_prefix(state)
    num_features, num_blocks = infer_rgdn_arch_from_state(state)
    model = RGDN(in_channels=3, num_features=num_features, num_blocks=num_blocks, use_attention=True).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"Model: RGDN({num_features} features, {num_blocks} blocks)")

    # Load image
    img = np.array(Image.open(args.image).convert('RGB'), dtype=np.float32) / 255.0
    clean = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    print("\n" + "="*50)
    print(f"TEST 1: Standalone Denoising (ADMM Operating Point)")
    print("="*50)
    # Test at sigma = 0.06 (approx 15/255), which is what ADMM usually sends it
    sigma_val = 15.0 / 255.0
    noisy = (clean + torch.randn_like(clean) * sigma_val).clamp(0, 1)
    denoised = model(noisy, sigma_val)

    psnr_noisy = calculate_benchmark_psnr(noisy, clean, 0)
    psnr_denoised = calculate_benchmark_psnr(denoised, clean, 0)
    print(f"Input Noisy PSNR : {psnr_noisy:.2f} dB")
    print(f"Denoised PSNR    : {psnr_denoised:.2f} dB")
    if psnr_denoised > 30.5:
        print("✅ PASS: Exceptional denoising power.")
    elif psnr_denoised > 29.75:
        print("✅ PASS: Strong denoising power.")
    else:
        print("❌ FAIL: Below the expected denoising baseline.")

    print("\n" + "="*50)
    print(f"TEST 2: Identity / Texture Preservation Test")
    print("="*50)
    # Test at tiny sigma to see if the identity loss worked
    tiny_sigma = 0.001
    identity_out = model(clean, tiny_sigma)
    psnr_id = calculate_benchmark_psnr(identity_out, clean, 0)
    print(f"Identity PSNR    : {psnr_id:.2f} dB")
    if psnr_id > 40.0:
        print("✅ PASS: Safely ignores textures when noise is low.")
    else:
        print("❌ FAIL: Still blurring textures. Identity loss may need tuning.")

    print("\n" + "="*50)
    print(f"TEST 3: Contractivity / Lipschitz Stability")
    print("="*50)
    # Generate two different noisy inputs
    v1 = clean + torch.randn_like(clean) * sigma_val
    v2 = clean + torch.randn_like(clean) * sigma_val
    dist_in = torch.norm(v1 - v2, p=2)

    for r in [1.0, 0.7, 0.5]:
        def D_r(v):
            return v + r * (model(v, sigma_val) - v)

        out1 = D_r(v1)
        out2 = D_r(v2)
        dist_out = torch.norm(out1 - out2, p=2)
        lipschitz = dist_out / (dist_in + 1e-8)

        status = "✅ SAFE" if lipschitz < 1.0 else "❌ UNSTABLE"
        print(f"Relaxation r={r:.1f} | Lipschitz Constant: {lipschitz:.4f} | {status}")

    print("\nDone.")

if __name__ == '__main__':
    main()
