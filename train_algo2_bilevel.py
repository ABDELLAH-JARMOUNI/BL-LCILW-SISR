"""
train_algo2_bilevel.py
========================================================================
Algorithm 2 (BL-LCILW) implemented line-for-line, "simultaneous single-step"
schedule:

    Initialize x^0 = U_{w^0} y,  z^0 = x^0,  u^0 = 0
    for l = 0 .. L-1:
        w^{l+1} = Proj_W ( w^l - theta_w * grad_w J_eps(x^l, w^l,     alpha^l) )   (line 6)
        a^{l+1} = Proj_+ ( a^l - theta_a * grad_a J_eps(x^l, w^{l+1}, alpha^l) )   (line 7)
        x^{l+1} = (H^T D^T D H + (mu+rho) I)^{-1}(H^T D^T y + mu U_{w^{l+1}} y
                                                  + rho (z^l - u^l))               (line 9)
        z^{l+1} = RGDN(x^{l+1} + u^l ; a^{l+1}/rho)                                (line 10)
        u^{l+1} = u^l + x^{l+1} - z^{l+1}                                          (line 11)
        # adaptive smoothing
        if ||grad J_eps|| < sigma*gamma*eps_l:  eps_{l+1} = gamma*eps_l            (lines 13-17)
        if sigma*eps_{l+1} < eps_tol:           break                             (lines 18-20)
    return x^{l+1}, w^{l+1}, a^{l+1}

Faithful-implementation notes:

* grad_w J / grad_a J use IMPLICIT differentiation with the inverse Hessian
  approximated by a TRUNCATED Neumann series (Sec. 3.4), exactly as the paper
  states. The series is  sum_{k=0..K} (J_F^T)^k g  with the UNDAMPED iteration
  p <- g + J_F^T p  (eta = 1; the truncation order K is `neumann_iters`).
  This is the single-step regime: the gradient is evaluated at the CURRENT ADMM
  iterate x^l (the paper writes grad J(x^l, .) as if x^l were the lower-level
  solution). It is therefore an APPROXIMATE bilevel gradient by construction --
  it becomes exact only as the inner/outer iterates co-converge. The *exact*
  gradient of J(x*(w,a)) would require solving the lower level to a fixed point
  and iterating the adjoint to convergence.

* Proj_W and Proj_{R^sN_+} are realised by the network parameterisations
  themselves: PatchWeightedLCI2D maps params -> w via softmax (so w in
  [w_min, .], sum = n automatically) and AlphaNet maps params -> alpha via a
  scaled sigmoid (so alpha in [0.001, 0.025] > 0). The constraint sets are thus
  always satisfied; the upper-level steps are unconstrained Adam steps on the
  network parameters (Adam = the "step-size" mechanism, as in Algorithm 1).

* lines 6 and 7 are SEQUENTIAL (alpha uses w^{l+1}); we recompute the implicit
  gradient after the w-step so alpha sees the updated w. Set sequential=False to
  use one shared gradient (cheaper; true-simultaneous instead of Gauss-Seidel).

* the folded-frequency x-update solver is verified to invert
  (H^T D^T D H + (mu+rho)I) to machine precision on the periodic operator.

* eps / J_eps: the paper does not give an explicit form for the smoothing
  J_eps(.,eps), so eps is implemented as the convergence gate of lines 13-20
  exactly as written, using the implicit-gradient norm. It does not modify the
  loss.

* RGDN is the FIXED prior (frozen, eval mode). Its Jacobian still enters
  grad_w/grad_a via the VJPs. The 3-variable ADMM map is contractive only if
  RGDN is averaged (Assumption 1(ii)); with a raw/under-trained RGDN the
  Neumann terms are unreliable -- use the trained RGDN, or --denoiser_relax r<1
  to use the averaged operator v + r*(RGDN(v)-v).

Upper objective is the paper's J = 1/2 ||x - x_gt||_2^2 (sum); with Adam the
overall gradient scale is largely irrelevant.

Checkpoint format matches infer_algo2.py.

Author: Abdellah Jarmouni
"""

import os
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.fft
from tqdm import tqdm

from weighted_lci_upsampler import PatchWeightedLCI2D, AlphaNet, all_lagrange_bases
from rgdn_model import RGDN
from train_global_lci import BSDS500SRDataset


# ============================================================
# Lower-level math helpers (verified correct)
# ============================================================

def forward_A(x, kernel_base, scale):
    C = x.shape[1]; pad = kernel_base.shape[-1] // 2
    kernel = kernel_base.repeat(C, 1, 1, 1).to(x.device)
    return F.avg_pool2d(F.conv2d(x, kernel, padding=pad, groups=C), scale, scale)


def adjoint_AT(y, kernel_base, scale):
    C = y.shape[1]; pad = kernel_base.shape[-1] // 2
    kernel = kernel_base.repeat(C, 1, 1, 1).to(y.device)
    up = F.interpolate(y, scale_factor=scale, mode='nearest') / (scale ** 2)
    return F.conv2d(up, kernel, padding=pad, groups=C)


def psf2otf(psf, shape):
    kH, kW = psf.shape[2:]
    psf_padded = F.pad(psf, (0, shape[1] - kW, 0, shape[0] - kH))
    psf_padded = torch.roll(psf_padded, shifts=(-(kH // 2), -(kW // 2)), dims=(2, 3))
    return torch.fft.fft2(psf_padded)


def get_effective_otf(kernel_base, scale, img_shape):
    B, C, H, W = img_shape
    otf_H = psf2otf(kernel_base, (H, W))
    box_psf = torch.zeros(1, 1, H, W, device=kernel_base.device, dtype=kernel_base.dtype)
    for i in range(scale):
        for j in range(scale):
            box_psf[0, 0, -i % H, -j % W] = 1.0 / (scale ** 2)
    return otf_H * torch.fft.fft2(box_psf)


def fft_solve_balanced(otf, scale, mu, rho, rhs):
    """Closed-form x-update via frequency folding. Solves
    (H^T D^T D H + (mu+rho) I) x = rhs."""
    B, C, H, W = rhs.shape
    lam = mu + rho
    V = torch.fft.fft2(rhs)
    F_bar = torch.conj(otf)
    h, w = H // scale, W // scale
    num_fold = (otf * V).reshape(B, C, scale, h, scale, w).mean(dim=(2, 4))
    den_fold = (torch.abs(otf) ** 2).reshape(1, 1, scale, h, scale, w).mean(dim=(2, 4))
    inv_fold = num_fold / (den_fold + lam)
    inv_up = (inv_fold.unsqueeze(2).unsqueeze(4)
              .expand(B, C, scale, h, scale, w).reshape(B, C, H, W))
    X = (V - F_bar * inv_up) / lam
    return torch.fft.ifft2(X).real


# ============================================================
# Implicit gradient at the CURRENT iterate (truncated Neumann, eta=1)
# ============================================================


def crop_valid_region(x, valid_pad):
    """Return the non-padded HR region. valid_pad is measured in HR pixels."""
    if valid_pad <= 0:
        return x
    return x[:, :, valid_pad:-valid_pad, valid_pad:-valid_pad]


def make_valid_upper_grad(x_l, x_gt, valid_pad):
    """Gradient of J = 1/2||x - x_gt||^2, but only on the valid center region.

    The padded reflect margin is used only as a boundary buffer. It must not
    contribute to the bilevel gradient; otherwise the optimizer learns from
    artificial boundary pixels.
    """
    if valid_pad <= 0:
        return x_l - x_gt
    g_x = torch.zeros_like(x_l)
    g_x[:, :, valid_pad:-valid_pad, valid_pad:-valid_pad] = (
        x_l[:, :, valid_pad:-valid_pad, valid_pad:-valid_pad]
        - x_gt[:, :, valid_pad:-valid_pad, valid_pad:-valid_pad]
    )
    return g_x


def valid_upper_loss_value(x_l, x_gt, valid_pad):
    """Scalar upper loss value, evaluated only on the valid center region."""
    xv = crop_valid_region(x_l, valid_pad)
    gv = crop_valid_region(x_gt, valid_pad)
    return 0.5 * ((xv - gv) ** 2).sum().item()


def implicit_grads_split(x_l, z_l, u_l, x_gt, y, net_W, net_Alpha, denoiser,
                         otf, ATy, mu, rho, scale, neumann_iters, eta=1.0,
                         valid_pad=0):
    """grad_w J and grad_a J at the current state (x^l, z^l, u^l), via implicit
    differentiation of ONE ADMM step with a truncated Neumann inverse Hessian.

    Returns (w_params, w_grads), (a_params, a_grads), grad_norm.
    Truncated series:  p = sum_{k=0..K} (J_F^T)^k g   via   p <- g + eta * J_F^T p.
    """
    x_in = x_l.detach().requires_grad_(True)
    z_in = z_l.detach().requires_grad_(True)
    u_in = u_l.detach().requires_grad_(True)

    # one differentiable ADMM step F(x_in, z_in, u_in; w, alpha)
    U_w_y = net_W(y)
    alpha_map = net_Alpha(y)
    rhs = ATy + mu * U_w_y + rho * (z_in - u_in)
    x_out = fft_solve_balanced(otf, scale, mu, rho, rhs)
    z_out = denoiser(x_out + u_in, alpha_map / rho)
    u_out = u_in + x_out - z_out
    outs, ins = (x_out, z_out, u_out), (x_in, z_in, u_in)

    # paper's upper-level gradient seed: grad_x J = (x^l - x_gt),
    # but only on the valid center region when reflect padding is enabled.
    g_x = make_valid_upper_grad(x_l, x_gt, valid_pad)
    g = (g_x, torch.zeros_like(z_in), torch.zeros_like(u_in))

    # truncated Neumann (undamped, eta=1): p <- g + eta * J_F^T p
    p = tuple(t.clone() for t in g)
    for _ in range(neumann_iters):
        Jtp = torch.autograd.grad(outs, ins, grad_outputs=p,
                                  retain_graph=True, allow_unused=True)
        Jtp = tuple(j if j is not None else torch.zeros_like(pi) for j, pi in zip(Jtp, p))
        p = tuple(gi + eta * ji for gi, ji in zip(g, Jtp))

    w_params = [pp for pp in net_W.parameters() if pp.requires_grad]
    a_params = [pp for pp in net_Alpha.parameters() if pp.requires_grad]
    allg = torch.autograd.grad(outs, w_params + a_params, grad_outputs=p,
                               retain_graph=False, allow_unused=True)
    nW = len(w_params)
    w_grads, a_grads = allg[:nW], allg[nW:]
    gnorm = math.sqrt(sum(float((gg * gg).sum()) for gg in allg if gg is not None))
    return (w_params, w_grads), (a_params, a_grads), gnorm


def _assign_grads(params, grads):
    for p, g in zip(params, grads):
        p.grad = g if g is not None else None


# ============================================================
# Admissibility penalty on w  (soft realization of Proj_W, line 6)
# ============================================================

def w_admissibility_penalty(net_W, y, L_dense, lambda_leb, lambda_int, beta=10.0):
    """Soft version of the paper's admissible-set projection Proj_W (Def. 3):

        L_leb = lambda_leb * Lebesgue-constant(w)      (log-sum-exp over sampled t)
        L_int = lambda_int * mean_k (w_k - 1)^2        (interpolation preservation,
                                                        since L_k(t_j) = delta_kj)

    Differentiable in net_W's parameters. This keeps the bilevel-refined w inside
    the admissible set, so the weak (mu-scaled) reconstruction gradient cannot
    random-walk it into the unconstrained nullspace and corrupt the standalone
    interpolation quality. NOTE: the L_int term pulls toward w = 1 (standard
    LCI). In Algorithm 2 there is no interpolation data-loss to counter it
    (unlike Algorithm 1), so keep lambda_int LIGHT -- it is a guard, not a
    driver. The Lebesgue term only bounds conditioning and does not force
    w = 1, so it can stay at the Algorithm-1 strength.
    """
    # get_w_map() is no_grad, so recompute the weight map differentiably here.
    theta = net_W.weight_net(y)                                            # [B, n, H, W]
    n = net_W.n_nodes
    w = net_W.w_min + (n - n * net_W.w_min) * F.softmax(theta, dim=1)      # [B, n, H, W]
    w_flat = w.permute(0, 2, 3, 1).reshape(-1, n)                          # [B*H*W, n]

    row = torch.matmul(w_flat, L_dense.abs().T)                            # [B*H*W, T]
    m = row.max(dim=1, keepdim=True)[0]
    leb = (1.0 / beta) * (m.squeeze(-1) +
                          torch.log(torch.mean(torch.exp(beta * (row - m)), dim=1)))
    L_leb = lambda_leb * leb.mean()
    L_int = lambda_int * torch.mean(torch.sum((w_flat - 1.0) ** 2, dim=1))
    return L_leb + L_int


# ============================================================
# Algorithm 2 (simultaneous single-step), per image
# ============================================================

def run_algorithm2_on_image(
    y, x_gt, net_W, net_Alpha, denoiser, opt_W, opt_Alpha,
    otf, ATy, mu, rho, scale, L_iterations, neumann_iters,
    L_dense, lambda_leb, lambda_int, beta=10.0,
    eps0=1.0, gamma=0.9, sigma=0.5, eps_tol=1e-4,
    grad_clip=1.0, sequential=True, valid_pad=0,
):
    # line 1-2: initialise x^0 = U_{w^0} y, z^0 = x^0, u^0 = 0
    with torch.no_grad():
        x_l = net_W(y)
    z_l = x_l.clone()
    u_l = torch.zeros_like(x_l)
    eps_l = eps0

    for l in range(L_iterations):
        # ---- line 6: w-update at (x^l, w^l, alpha^l) ----
        (wP, wG), (aP, aG), gnorm = implicit_grads_split(
            x_l, z_l, u_l, x_gt, y, net_W, net_Alpha, denoiser,
            otf, ATy, mu, rho, scale, neumann_iters, valid_pad=valid_pad)
        opt_W.zero_grad(set_to_none=True)
        # admissibility penalty (soft Proj_W) -> fills .grad of net_W params
        pen = w_admissibility_penalty(net_W, y, L_dense, lambda_leb, lambda_int, beta)
        pen.backward()
        # add the implicit (bilevel) w-gradient on top of the penalty gradient
        for p, g in zip(wP, wG):
            if g is not None:
                p.grad = g if p.grad is None else (p.grad + g)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(wP, grad_clip)   # clip w SEPARATELY from alpha
        opt_W.step()                                  # -> w^{l+1}

        # ---- line 7: alpha-update at (x^l, w^{l+1}, alpha^l) ----
        if sequential:
            (wP, wG), (aP, aG), gnorm = implicit_grads_split(
                x_l, z_l, u_l, x_gt, y, net_W, net_Alpha, denoiser,
                otf, ATy, mu, rho, scale, neumann_iters, valid_pad=valid_pad)   # recompute with w^{l+1}
        opt_Alpha.zero_grad(set_to_none=True)
        _assign_grads(aP, aG)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(aP, grad_clip)
        opt_Alpha.step()                              # -> alpha^{l+1}

        # ---- lines 9-11: one PnP-ADMM step with w^{l+1}, alpha^{l+1} ----
        with torch.no_grad():
            U_w_y = net_W(y)
            alpha_map = net_Alpha(y)
            rhs = ATy + mu * U_w_y + rho * (z_l - u_l)
            x_l = fft_solve_balanced(otf, scale, mu, rho, rhs)   # line 9
            z_l = denoiser(x_l + u_l, alpha_map / rho)           # line 10
            u_l = u_l + x_l - z_l                                # line 11

        # ---- lines 13-20: adaptive smoothing (gnorm ~ ||grad J_eps||) ----
        if gnorm < (sigma * gamma * eps_l):
            eps_l = gamma * eps_l
        if (sigma * eps_l) < eps_tol:
            break

    J = valid_upper_loss_value(x_l, x_gt, valid_pad)
    return x_l, J


# ============================================================
# Training driver (loops Algorithm 2 over the dataset)
# ============================================================

def train_bilevel_algorithm2(
    dataloader,
    device='cuda',
    epochs=10,
    L_iterations=15,
    neumann_iters=5,
    lr_w=1e-5,             # w-grad is ~mu (=100x) smaller than alpha's; step smaller
    lr_alpha=1e-4,
    mu=0.01,
    rho=0.05,
    scale=4,
    grad_clip=1.0,
    sequential=True,
    denoiser_relax=1.0,
    lambda_leb=1e-3,       # Lebesgue-constant penalty (conditioning; no w=1 pull)
    lambda_int=1e-2,       # interpolation-preservation (LIGHT; pulls toward w=1)
    beta=10.0,
    eps0=1.0, gamma=0.9, sigma=0.5, eps_tol=1e-4,
    train_pad=8,
    w_init='phi_xi_spatial_weights.pth',
    rgdn_init='best_model_rgdn.pth',
    out_prefix='bl_lcilw_algo2',
):
    print("Initializing Algorithm 2 (simultaneous single-step)...")

    train_pad = int(train_pad)
    if train_pad < 0:
        raise ValueError("train_pad must be >= 0")
    if train_pad > 0 and train_pad % scale != 0:
        raise ValueError(f"train_pad is in HR pixels and must be divisible by scale={scale}. Got {train_pad}.")

    net_W = PatchWeightedLCI2D(n_nodes=8, scale_factor=scale).to(device)
    net_Alpha = AlphaNet(in_channels=3, scale_factor=scale).to(device)
    net_RGDN = RGDN(in_channels=3, num_features=64, num_blocks=8, use_attention=True).to(device)

    try:
        net_W.load_state_dict(torch.load(w_init, map_location=device, weights_only=False))
        net_RGDN.load_state_dict(torch.load(rgdn_init, map_location=device, weights_only=False)['model_state_dict'])
        print("[+] Loaded pre-trained weights for W and RGDN.")
    except Exception as e:
        print(f"[-] Missing pre-trained weights. Starting from scratch. ({e})")

    net_RGDN.eval()                          # frozen prior
    for p in net_RGDN.parameters():
        p.requires_grad_(False)
    if denoiser_relax >= 1.0:
        denoiser = lambda v, sigma_: net_RGDN(v, sigma_)
    else:
        r = float(denoiser_relax)
        denoiser = lambda v, sigma_: v + r * (net_RGDN(v, sigma_) - v)

    opt_W = optim.Adam(net_W.parameters(), lr=lr_w)
    opt_Alpha = optim.Adam(net_Alpha.parameters(), lr=lr_alpha)

    coords = torch.arange(5, dtype=torch.float32, device=device) - 2
    g1d = torch.exp(-coords ** 2 / 2.0)
    g2d = g1d.outer(g1d)
    kernel_base = (g2d / g2d.sum()).view(1, 1, 5, 5).to(device)

    # dense Lagrange-basis matrix for the Lebesgue-constant penalty
    t_dense = torch.linspace(-1.0, 1.0, 500, device=device)
    L_dense = all_lagrange_bases(t_dense, net_W.nodes)        # [500, n_nodes]
    print(f"[cfg] lr_w={lr_w} lr_alpha={lr_alpha} | penalty lambda_leb={lambda_leb} "
          f"lambda_int={lambda_int} | denoiser_relax={denoiser_relax} | HR train_pad={train_pad}")

    for epoch in range(epochs):
        net_W.train(); net_Alpha.train()
        epoch_loss, nb = 0.0, 0
        for y_unused, x_gt in tqdm(dataloader, desc=f"Bilevel Epoch {epoch+1}"):
            # Boundary-safe training path:
            #   HR -> reflect pad -> degrade -> ADMM/PnP -> loss only on valid center.
            # The LR returned by the dataset is intentionally ignored because it was
            # generated from the unpadded HR crop.
            x_gt = x_gt.to(device)

            # Keep the HR crop divisible by the scale, then add an HR reflect margin.
            _, _, H0, W0 = x_gt.shape
            H = (H0 // scale) * scale
            W = (W0 // scale) * scale
            x_gt = x_gt[:, :, :H, :W]

            if train_pad > 0:
                x_gt_train = F.pad(x_gt, (train_pad, train_pad, train_pad, train_pad), mode='reflect')
            else:
                x_gt_train = x_gt

            # Generate the LR observation from the padded HR target, matching inference.
            y = forward_A(x_gt_train, kernel_base, scale)

            B, C, h_lr, w_lr = y.shape
            otf = get_effective_otf(kernel_base, scale, (B, C, h_lr * scale, w_lr * scale))
            ATy = adjoint_AT(y, kernel_base, scale)

            _, J = run_algorithm2_on_image(
                y, x_gt_train, net_W, net_Alpha, denoiser, opt_W, opt_Alpha,
                otf, ATy, mu, rho, scale, L_iterations, neumann_iters,
                L_dense, lambda_leb, lambda_int, beta=beta,
                eps0=eps0, gamma=gamma, sigma=sigma, eps_tol=eps_tol,
                grad_clip=grad_clip, sequential=sequential, valid_pad=train_pad)
            epoch_loss += J; nb += 1

        print(f"Epoch {epoch+1} complete. Mean upper loss J = {epoch_loss / max(nb,1):.6e}")

        torch.save({
            'net_W': net_W.state_dict(),
            'net_Alpha': net_Alpha.state_dict(),
            'net_RGDN': net_RGDN.state_dict(),
            'hparams': {'mu': mu, 'rho': rho, 'scale': scale, 'L': L_iterations,
                        'denoiser_relax': denoiser_relax, 'train_pad': train_pad},
        }, f'{out_prefix}_full.pth')
        torch.save(net_W.state_dict(), f'{out_prefix}_w_final.pth')
        torch.save(net_Alpha.state_dict(), f'{out_prefix}_alpha_final.pth')

    return net_W, net_Alpha, net_RGDN


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='./data/BSDS500')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--hr_patch', type=int, default=128)
    ap.add_argument('--scale', type=int, default=4)
    ap.add_argument('--admm_iters', type=int, default=15, help='L: ADMM/upper steps per image (Algorithm 2)')
    ap.add_argument('--neumann_iters', type=int, default=5, help='K: truncated-Neumann order for the inverse Hessian')
    ap.add_argument('--mu', type=float, default=0.01)
    ap.add_argument('--rho', type=float, default=0.05)
    ap.add_argument('--lr_w', type=float, default=1e-5,
                    help='w step; ~mu (=100x) smaller than lr_alpha since dJ/dw is mu-scaled')
    ap.add_argument('--lr_alpha', type=float, default=1e-4)
    ap.add_argument('--grad_clip', type=float, default=1.0, help='clipped per-network (w and alpha separately)')
    ap.add_argument('--lambda_leb', type=float, default=1e-3, help='Lebesgue-constant penalty on w (soft Proj_W)')
    ap.add_argument('--lambda_int', type=float, default=1e-2,
                    help='interpolation-preservation penalty on w; keep LIGHT (pulls toward w=1)')
    ap.add_argument('--beta', type=float, default=10.0, help='log-sum-exp sharpness for the Lebesgue penalty')
    ap.add_argument('--simultaneous', action='store_true',
                    help='use one shared gradient for w & alpha (cheaper) instead of the paper-exact sequential update')
    ap.add_argument('--denoiser_relax', type=float, default=1.0,
                    help='<1 uses the averaged operator v+r*(RGDN(v)-v) for guaranteed contractivity')
    ap.add_argument('--pad', type=int, default=8,
                    help='HR reflect padding before degradation during training; loss/gradient use center crop. Use 0 to disable.')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading BSDS500 dataset...")
    train_dataset = BSDS500SRDataset(
        root_dir=args.data_root, split='train', hr_patch=args.hr_patch,
        scale=args.scale, samples_per_image=10, augment=True)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)

    print("\nStarting Algorithm 2 (simultaneous single-step)...")
    train_bilevel_algorithm2(
        dataloader=train_loader, device=device, epochs=args.epochs,
        L_iterations=args.admm_iters, neumann_iters=args.neumann_iters,
        lr_w=args.lr_w, lr_alpha=args.lr_alpha, mu=args.mu, rho=args.rho, scale=args.scale,
        grad_clip=args.grad_clip, sequential=not args.simultaneous,
        denoiser_relax=args.denoiser_relax, train_pad=args.pad,
        lambda_leb=args.lambda_leb, lambda_int=args.lambda_int, beta=args.beta)
    print("\nDone. Checkpoints saved with prefix bl_lcilw_algo2.")
