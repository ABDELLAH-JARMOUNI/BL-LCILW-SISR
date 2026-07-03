"""
expansive_rgdn_check.py
========================================================================
Empirical local-expansiveness probe for the RGDN denoiser.

For a realistic ADMM-like iterate v = clean + matching noise, estimates
    eta(v) = max_d ||D(v + eps*d) - D(v)|| / eps
over random unit directions d, at several sigma levels. Values above 1
indicate local expansiveness at that operating point.

Author: Abdellah Jarmouni
"""

import torch, numpy as np
from PIL import Image
from rgdn_model import load_rgdn

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
m = load_rgdn('best_model_rgdn.pth', dev); m.eval()

@torch.no_grad()
def eta_on(v, sigma, trials=300, eps=1e-3):
    s = torch.full((1,1,v.shape[2],v.shape[3]), sigma, device=dev)
    f0 = m(v, s); worst = 0.0
    for _ in range(trials):
        d = torch.randn_like(v); d /= d.norm()
        worst = max(worst, ((m(v + eps*d, s) - f0).norm() / eps).item())
    return worst

img = np.array(Image.open('Set5/baby.png').convert('RGB'), np.float32)/255.0
hr  = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(dev)[:, :, :256, :256]
for s in [0.05, 0.1, 0.2, 0.3, 0.5]:
    v = (hr + torch.randn_like(hr)*s).clamp(0,1)     # image + matching noise ~ a real iterate
    print(f"sigma={s:.2f}  eta(real input) ~ {eta_on(v, s):.3f}")
