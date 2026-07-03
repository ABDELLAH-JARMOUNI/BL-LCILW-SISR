# Configs

One YAML per training / inference script, recording the exact hyperparameters
used to produce the shipped checkpoints. They are a **documentation snapshot**
of the settings — the scripts themselves take their values from command-line
arguments (see each file's header for the matching command), so these YAMLs do
not need to be passed to run anything.

| Config | Script | Output |
|--------|--------|--------|
| `train_global_lci.yaml` | `train_global_lci.py` | `w_star_global.pth` |
| `train_patch_lci.yaml` | `train_patch_lci.py` | `phi_xi_spatial_weights.pth` |
| `train_rgdn.yaml` | `train_rgdn_bsds500.py` | `best_model_rgdn.pth` |
| `train_algo2_bilevel.yaml` | `train_algo2_bilevel.py` | `bl_lcilw_algo2_*.pth` |
| `infer_algo2.yaml` | `infer_algo2.py` / `infer_algo2_inspect.py` / `evaluate_models.py` | — |

