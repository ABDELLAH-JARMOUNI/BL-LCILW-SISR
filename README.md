# BL-LCILW: Bilevel Learning-Weighted Lagrange–Chebyshev Interpolation for Single Image Super-Resolution with Plug-and-Play Regularized Gradient Denoising 

Reference implementation of **BL-LCILW**, a single-image super-resolution
(SISR) method that couples a *learning-weighted* Lagrange–Chebyshev
interpolation (LCI) upsampler with a Plug-and-Play (PnP) ADMM reconstruction
driven by a sigma-aware RGDN denoiser. Everything is trained end-to-end through
a **bilevel** formulation: an inner reconstruction problem produces the
high-resolution image, and an outer problem learns the interpolation and
regularization parameters that make that reconstruction match the ground truth.

Classical interpolation (bicubic, standard LCI) applies the same rule
everywhere and tends to over-smooth edges and textures. BL-LCILW keeps the
mathematical structure of Lagrange–Chebyshev interpolation but lets the
contribution of each Chebyshev node adapt to local image content, then refines
the result with a regularized inverse-problem solver.

The pipeline has three stages:

1. **Interpolation prior (Algorithm 1)** — a weighted LCI upsampler `U_w`
   whose Chebyshev weights are learned, either as a single global vector
   (`train_global_lci.py`) or as a dense per-pixel map produced by a small CNN
   (`train_patch_lci.py`).
2. **Denoiser prior** — a BatchNorm-free, sigma-conditioned RGDN denoiser
   (`train_rgdn_bsds500.py`), trained with a finite-difference local Jacobian
   penalty so it stays close to non-expansive inside the PnP loop.
3. **Bilevel refinement (Algorithm 2)** — a PnP-ADMM outer loop
   (`train_algo2_bilevel.py`) that jointly refines the interpolation weight map
   `w` and the spatial regularization map `α` via truncated-Neumann implicit
   differentiation.

---

## Repository layout

```
bl-lcilw/
├── README.md
├── LICENSE
├── .gitignore
├── configs/                     # hyperparameter snapshots (one YAML per script)
│   ├── README.md
│   ├── train_global_lci.yaml
│   ├── train_patch_lci.yaml
│   ├── train_rgdn.yaml
│   ├── train_algo2_bilevel.yaml
│   └── infer_algo2.yaml
├── docs/                        # report / presentation (add locally, see below)
│
├── weighted_lci_upsampler.py    # U_w operators (global + spatial) and AlphaNet
├── rgdn_model.py                # sigma-aware RGDN denoiser + checkpoint loaders
├── psnr_luminance.py            # MATLAB-convention Y-channel PSNR
│
├── train_global_lci.py          # Algorithm 1 (global);  also defines the SR dataset
├── train_patch_lci.py           # Algorithm 1 (spatial / patch-wise)
├── train_rgdn_bsds500.py        # RGDN denoiser training
├── train_algo2_bilevel.py       # Algorithm 2 (bilevel PnP-ADMM)
│
├── infer_algo2.py               # full Algorithm 2 inference + PSNR
├── infer_algo2_inspect.py       # inference + per-image w / α statistics
├── evaluate_models.py           # interpolation-stage comparison on Set5
├── check_rgdn.py                # RGDN denoising / identity / Lipschitz checks
├── expansive_rgdn_check.py      # empirical local-expansiveness probe
│
└── *.pth                        # released checkpoints (see below)
```

### Released checkpoints

| File | Produced by | Contents |
|------|-------------|----------|
| `w_star_global.pth` | `train_global_lci.py` | Global LCI weight vector. |
| `phi_xi_spatial_weights.pth` | `train_patch_lci.py` | Spatial LCI weight network `Φ_ξ`. |
| `best_model_rgdn.pth` | `train_rgdn_bsds500.py` | RGDN denoiser (64 features / 8 blocks). |
| `bl_lcilw_algo2_full.pth` | `train_algo2_bilevel.py` | Full bilevel checkpoint (W + Alpha + RGDN + hparams). |
| `bl_lcilw_algo2_w_final.pth` | `train_algo2_bilevel.py` | Weight network only. |
| `bl_lcilw_algo2_alpha_final.pth` | `train_algo2_bilevel.py` | Alpha network only. |

---

## Setup

Python 3.8+ with:

```bash
pip install torch numpy pillow matplotlib tqdm scikit-image
```

Expected data layout — a BSDS500-style tree for training and Set5 / Set14 for
evaluation:

```
data/
  BSDS500/
    images/
      train/
      val/
      test/
Set5/
  baby.png  bird.png  butterfly.png  head.png  woman.png
Set14/
  ...
```

---

## Reproducing the results

The commands below produced the shipped checkpoints. The same values are stored
as YAML snapshots in [`configs/`](configs/). Run the stages in order.

### 1. Interpolation priors — Algorithm 1

```bash
python train_global_lci.py \
  --data_root ./data/BSDS500 --set5_dir ./Set5 \
  --lambda_int 0.1 --lambda_leb 1e-3 --epochs 100

python train_patch_lci.py \
  --data_root ./data/BSDS500 --set5_dir ./Set5 \
  --lambda_int 0.1 --lambda_leb 1e-3 --epochs 100
```

Common settings: scale ×4, 8 Chebyshev nodes, 128×128 HR patches, Adam at
`1e-3`, batch size 16, log-sum-exp sharpness `β = 10`.
Outputs: `w_star_global.pth`, `phi_xi_spatial_weights.pth`.

### 2. RGDN denoiser

```bash
python train_rgdn_bsds500.py \
  --data_root ./data/BSDS500 \
  --save_dir ./rgdn_checkpoints \
  --batch_size 32 --num_workers 8 --epochs 300 \
  --val_every 10 --save_every 10 \
  --jacobian_weight 3e-4 --jacobian_eps 0.003
```

64 features / 8 blocks, 64×64 patches, Adam at `3e-4`, weight decay `1e-4`,
gradient clip `1.0`, L1 + gradient (`0.1`) + identity (`0.1`) losses, local
Jacobian penalty (margin `0.9`). Training noise matches the ADMM operating
range: 70 % of samples with `σ ∈ [0, 0.15]`, 30 % with `σ ∈ [0.15, 0.50]`.
Output: `rgdn_checkpoints/best_model_rgdn.pth` — copy it to the repository root
(or pass `--rgdn` / `--rgdn_init`) for the next stage.

### 3. Bilevel refinement — Algorithm 2

```bash
python train_algo2_bilevel.py \
  --data_root ./data/BSDS500 \
  --epochs 30 --batch_size 4 --hr_patch 192 --scale 4 \
  --admm_iters 12 --neumann_iters 8 \
  --mu 0.01 --rho 0.05 \
  --lr_w 3e-5 --lr_alpha 1e-4 \
  --lambda_leb 1e-3 --lambda_int 1e-3 \
  --denoiser_relax 1.0 --pad 8 --simultaneous
```

`L = 12` ADMM steps, `K = 8` Neumann terms, `μ = 0.01`, `ρ = 0.05`, HR reflect
padding of 8 px. `--simultaneous` uses one shared implicit gradient for `w` and
`α` per step; omit it for the sequential (Gauss–Seidel) update where `α` sees
the just-updated `w`.
Outputs: `bl_lcilw_algo2_full.pth`, `bl_lcilw_algo2_w_final.pth`,
`bl_lcilw_algo2_alpha_final.pth`.

### 4. Inference and evaluation

```bash
# Full Algorithm 2 pipeline + PSNR-Y on a test set
python infer_algo2.py \
  --test_dir ./Set5 --ckpt bl_lcilw_algo2_full.pth --rgdn best_model_rgdn.pth

# Same, plus per-image w / alpha statistics
python infer_algo2_inspect.py \
  --test_dir ./Set5 --ckpt bl_lcilw_algo2_full.pth --rgdn best_model_rgdn.pth

# Interpolation-stage comparison (standard vs global vs spatial LCI)
python evaluate_models.py --set5_dir ./Set5

# RGDN sanity checks
python check_rgdn.py --model best_model_rgdn.pth --image ./Set5/bird.png
python expansive_rgdn_check.py
```

---

## Results

All numbers are average **PSNR-Y** (luminance-channel PSNR, MATLAB `rgb2ycbcr`
convention, peak 255, 4-pixel border crop) at scale ×4.

### Interpolation stage (Algorithm 1) — output `U_w y`

| Method | Set5 | Set14 | Urban100 |
|--------|:----:|:-----:|:--------:|
| Bicubic | 28.173 | 25.927 | 23.011 |
| Standard LCI (`w = 1`) | 28.376 | 26.027 | 23.107 |
| Global LCILW | 28.379 | 26.030 | 23.111 |
| **Spatial LCILW** | **28.917** | **26.373** | **23.375** |

The global weights stay very close to `1`, so global LCILW behaves almost like
standard LCI. The spatial (patch-wise) model is where the interpolation-stage
gain comes from: predicting location-dependent Chebyshev weights adds roughly
**+0.54 dB** over standard LCI on Set5.

### Full system (Algorithm 2) — output after PnP-ADMM reconstruction

| Method | Set5 | Set14 | Urban100 |
|--------|:----:|:-----:|:--------:|
| Direct interpolation `U_w y` | 28.41 | 25.98 | 23.16 |
| **Full BL-LCILW** | **29.97** | **27.12** | **24.09** |

The bilevel reconstruction adds about **+1.5 dB** on Set5 over the direct
interpolation output, confirming that the PnP-ADMM loop and the RGDN prior are
not redundant with the learned interpolation.

### RGDN behaviour (diagnostics)

At the ADMM operating point (`σ ≈ 15/255`) the denoiser lifts a noisy input from
**29.73 dB → 38.05 dB**, and at a tiny `σ = 0.001` it returns the clean image at
**67.51 dB** (it does not touch textures when little denoising is requested).
Empirical Lipschitz ratios of the relaxed operator `D_r(v) = v + r·(RGDN(v) − v)`
stay below 1 (`0.24 / 0.40 / 0.56` for `r = 1.0 / 0.7 / 0.5`), and
finite-difference expansiveness `η` on real inputs stays around `0.21–0.31`
across `σ ∈ [0.05, 0.5]`. These are empirical stability checks, not a formal
non-expansiveness proof.

---


## Roadmap

This project is under active development. Planned directions include broader
benchmark coverage, alternative denoiser priors, and beating deep learning methods.

## Author

Abdellah Jarmouni

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
