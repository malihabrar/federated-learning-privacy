"""
Results summary script.

Reads all training and attack shelves and produces:
  - results/summary.csv        -- one row per (model, method, alpha, seed, image)
  - results/summary_table.txt  -- two human-readable tables (iDLG and InvGrad)

Run after fetching results from HPC / Lightning.ai:
  python results_summary.py
"""
import os
import csv
import json
import shelve
import numpy as np

MODELS     = ["lenet", "vit"]
ALPHAS     = [1, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.05, 0.0]
SEEDS      = [10, 20, 30]
N_IMAGES   = 30
N_RESTARTS = 5
IDLG_BIN   = "results/idlg_experiments/bin"
INVGRAD_BIN = "results/invgrad/bin"
TRAIN_BIN  = "results/bin"
OUT_DIR    = "results"
os.makedirs(OUT_DIR, exist_ok=True)

CONDITIONS = [("prune", a) for a in ALPHAS] + [("clip", 1)]

# Load exported training results JSON if available (for cross-platform use)
_TRAIN_JSON_PATH = "train_results_export.json"
_train_json = {}
if os.path.exists(_TRAIN_JSON_PATH):
    with open(_TRAIN_JSON_PATH) as f:
        _train_json = json.load(f)
    print(f"Loaded {len(_train_json)} training results from {_TRAIN_JSON_PATH}")

# Load exported iDLG results JSON if available (for cross-platform use)
_IDLG_JSON_PATH = "idlg_results_export.json"
_idlg_json = {}
if os.path.exists(_IDLG_JSON_PATH):
    with open(_IDLG_JSON_PATH) as f:
        _idlg_json = json.load(f)
    print(f"Loaded {len(_idlg_json)} iDLG results from {_IDLG_JSON_PATH}")

# Load iDLG MSE export (best-restart MSE per image, derived from stored PSNR)
_IDLG_MSE_JSON_PATH = "idlg_mse_export.json"
_idlg_mse_json = {}
if os.path.exists(_IDLG_MSE_JSON_PATH):
    with open(_IDLG_MSE_JSON_PATH) as f:
        _idlg_mse_json = json.load(f)
    print(f"Loaded {len(_idlg_mse_json)} iDLG MSE values from {_IDLG_MSE_JSON_PATH}")

# Load InvGrad results JSON (HPC export covers both LeNet and ViT)
_INVGRAD_JSON_PATH = "invgrad_hpc_export.json"
_invgrad_json = {}
if os.path.exists(_INVGRAD_JSON_PATH):
    with open(_INVGRAD_JSON_PATH) as f:
        _invgrad_json = json.load(f)
    print(f"Loaded {len(_invgrad_json)} InvGrad results from {_INVGRAD_JSON_PATH}")
# Merge local invgrad_export.json if present (local shelve export may have more LeNet entries)
_INVGRAD_LOCAL_JSON_PATH = "invgrad_export.json"
if os.path.exists(_INVGRAD_LOCAL_JSON_PATH):
    with open(_INVGRAD_LOCAL_JSON_PATH) as f:
        _local_invgrad = json.load(f)
    before = len(_invgrad_json)
    _invgrad_json = {**_invgrad_json, **_local_invgrad}
    print(f"Merged {len(_local_invgrad)} local InvGrad results; total now {len(_invgrad_json)}")


def open_shelve(path):
    try:
        return shelve.open(path, flag="r")
    except Exception:
        return None


def get_test_acc(model, method, alpha, seed):
    """Read test accuracy from shelve, falling back to JSON export if unavailable."""
    for fname in [
        f"{TRAIN_BIN}/pruning_exp_model_{model}_seed_{seed}_alpha_{alpha:.2f}_method_{method}",
        f"{TRAIN_BIN}/pruning_exp_seed_{seed}_alpha_{alpha:.2f}_method_{method}",
    ]:
        d = open_shelve(fname)
        if d is None:
            continue
        try:
            val = d["val_accuracy"]
            return val[-1] if val else np.nan
        finally:
            d.close()
    # Fall back to JSON
    key = f"{model}_{method}_{alpha:.2f}_{seed}"
    entry = _train_json.get(key)
    if entry and entry.get("val_accuracy"):
        return entry["val_accuracy"][-1]
    return np.nan


def cond_label(method, alpha):
    if method == "clip":
        return "Gradient Clipping"
    if alpha == 1.0:
        return "No Defense (α=1)"
    if alpha == 0.0:
        return "Strict LGP (α=0)"
    if alpha == 0.05:
        return "PLGP (α=0.05)"
    return f"PLGP (α={alpha:.2f})"


# ── collect raw rows ──────────────────────────────────────────────────────────

all_rows = []

for model in MODELS:
    for method, alpha in CONDITIONS:
        for seed in SEEDS:
            test_acc = get_test_acc(model, method, alpha, seed)

            for img in range(N_IMAGES):
                best_idlg_psnr = np.nan
                best_idlg_ssim = np.nan
                invgrad_psnr   = np.nan
                invgrad_ssim   = np.nan
                invgrad_mse    = np.nan

                idlg_mse = np.nan
                if model == "lenet":
                    # iDLG MSE (best restart, derived from stored PSNR)
                    idlg_mse_key = f"{method}_{alpha:.2f}_{seed}_{img}"
                    idlg_mse = _idlg_mse_json.get(idlg_mse_key, {}).get("mse", np.nan)

                    # iDLG: pick best restart by PSNR — try shelve first, fall back to JSON
                    for r in range(N_RESTARTS):
                        fname = f"{IDLG_BIN}/img_{img}_seed_{seed}_alpha_{alpha:.2f}_method_{method}_{r}"
                        d = open_shelve(fname)
                        if d is not None:
                            try:
                                p = d["psnr"][-1]
                                s = d["ssim"][-1]
                                if np.isnan(best_idlg_psnr) or p > best_idlg_psnr:
                                    best_idlg_psnr = p
                                    best_idlg_ssim = s
                            finally:
                                d.close()
                        else:
                            key = f"{method}_{alpha:.2f}_{seed}_{img}_{r}"
                            entry = _idlg_json.get(key)
                            if entry and entry.get("psnr"):
                                p = entry["psnr"][-1]
                                s = entry["ssim"][-1] if entry.get("ssim") else np.nan
                                if np.isnan(best_idlg_psnr) or p > best_idlg_psnr:
                                    best_idlg_psnr = p
                                    best_idlg_ssim = s

                # InvGrad — try shelve first, fall back to JSON (runs for both lenet and vit)
                inv_key = f"img_{img}_seed_{seed}_alpha_{alpha:.2f}_method_{method}_model_{model}_iters_2000_restarts_3"
                fname = f"{INVGRAD_BIN}/{inv_key}"
                d = open_shelve(fname)
                if d is not None:
                    try:
                        invgrad_psnr = d["psnr"]
                        invgrad_ssim = d["ssim"]
                        invgrad_mse  = d.get("mse", np.nan)
                    finally:
                        d.close()
                else:
                    entry = _invgrad_json.get(inv_key)
                    if entry:
                        invgrad_psnr = entry.get("psnr", np.nan)
                        invgrad_ssim = entry.get("ssim", np.nan)
                        invgrad_mse  = entry.get("mse",  np.nan)

                all_rows.append({
                    "model":        model,
                    "method":       method,
                    "alpha":        alpha,
                    "seed":         seed,
                    "image":        img,
                    "idlg_psnr":    best_idlg_psnr,
                    "idlg_ssim":    best_idlg_ssim,
                    "idlg_mse":     idlg_mse,
                    "invgrad_psnr": invgrad_psnr,
                    "invgrad_ssim": invgrad_ssim,
                    "invgrad_mse":  invgrad_mse,
                    "test_acc":     test_acc,
                })


# ── write CSV ─────────────────────────────────────────────────────────────────

csv_path = os.path.join(OUT_DIR, "summary.csv")
fieldnames = ["model", "method", "alpha", "seed", "image",
              "idlg_psnr", "idlg_ssim", "idlg_mse",
              "invgrad_psnr", "invgrad_ssim", "invgrad_mse",
              "test_acc"]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)
print(f"Wrote {len(all_rows)} rows to {csv_path}")


# ── table helpers ─────────────────────────────────────────────────────────────

nan = float("nan")

def _stat(vals):
    v = [x for x in vals if not np.isnan(x)]
    return (np.mean(v), np.std(v), len(v)) if v else (nan, nan, 0)

def _fmt(m, s, width=13, decimals=2):
    if np.isnan(m):
        return ("—").center(width)
    return f"{m:.{decimals}f} ± {s:.{decimals}f}".center(width)

def _fmt_acc(m, s, width=13):
    if np.isnan(m):
        return ("—").center(width)
    return f"{m*100:.1f} ± {s*100:.1f}%".center(width)


def print_table(title, model, rows, columns):
    """
    columns: list of (header, key, is_acc) tuples
    """
    col_w = 16
    cond_w = 22

    header_parts = [f"{'Condition':<{cond_w}}"]
    for h, _, _, _ in columns:
        header_parts.append(f"{h:^{col_w}}")
    header_parts.append(f"{'N':^6}")
    header = " | ".join(header_parts)

    sep  = "─" * len(header)
    sep2 = "═" * len(header)

    lines = [
        "",
        title,
        sep2,
        header,
        sep2,
    ]

    model_rows = [r for r in rows if r["model"] == model]
    printed_any = False

    for method, alpha in CONDITIONS:
        subset = [r for r in model_rows if r["method"] == method and r["alpha"] == alpha]
        if not subset:
            continue

        # skip rows where all attack metrics (non-accuracy) are missing;
        # for accuracy-only tables, show any row with test_acc data
        attack_keys = [k for _, k, is_acc, _ in columns if not is_acc]
        if attack_keys:
            has_data = any(not np.isnan(r[k]) for r in subset for k in attack_keys)
        else:
            has_data = any(not np.isnan(r["test_acc"]) for r in subset)
        if not has_data:
            continue

        printed_any = True
        parts = [f"{cond_label(method, alpha):<{cond_w}}"]
        n = 0
        for _, key, is_acc, decimals in columns:
            vals = [r[key] for r in subset]
            m, s, cnt = _stat(vals)
            if is_acc:
                parts.append(_fmt_acc(m, s, col_w))
            else:
                parts.append(_fmt(m, s, col_w, decimals=decimals))
            if key == "idlg_psnr" or key == "invgrad_psnr":
                n = cnt
        parts.append(f"{n:^6}")
        lines.append(" | ".join(parts))

    if not printed_any:
        lines.append("  (no data)")

    lines.append(sep)
    return "\n".join(lines)


# ── build tables ──────────────────────────────────────────────────────────────

idlg_cols = [
    ("PSNR (dB)",  "idlg_psnr", False, 2),
    ("SSIM",       "idlg_ssim", False, 2),
    ("MSE",        "idlg_mse",  False, 5),
    ("Test Acc",   "test_acc",  True,  1),
]

invgrad_cols = [
    ("PSNR (dB)",  "invgrad_psnr", False, 2),
    ("SSIM",       "invgrad_ssim", False, 2),
    ("MSE",        "invgrad_mse",  False, 4),
    ("Test Acc",   "test_acc",     True,  1),
]

output_lines = ["Results Summary — Privacy-Utility Tradeoff", "=" * 80]

for model in MODELS:
    model_rows = [r for r in all_rows if r["model"] == model]
    has_any = any(
        not np.isnan(r[k])
        for r in model_rows
        for k in ["idlg_psnr", "invgrad_psnr", "test_acc"]
    )
    if not has_any:
        continue

    output_lines.append(f"\n{'─'*80}")
    output_lines.append(f"  Model: {model.upper()}")
    output_lines.append(f"{'─'*80}")

    if model == "vit":
        vit_rows = [r for r in all_rows if r["model"] == "vit" and r["seed"] == 10]
        vit_invgrad_cols = [
            ("PSNR (dB)",  "invgrad_psnr", False, 2),
            ("SSIM",       "invgrad_ssim", False, 2),
            ("MSE",        "invgrad_mse",  False, 4),
            ("Test Acc",   "test_acc",     True,  1),
        ]
        output_lines.append(print_table(
            f"  Table: Inverting Gradients Attack  (seed=10 only)",
            model, vit_rows, vit_invgrad_cols
        ))
    else:
        output_lines.append(print_table(
            f"  Table 1: iDLG Attack  (higher PSNR/SSIM = worse privacy)",
            model, all_rows, idlg_cols
        ))
        output_lines.append(print_table(
            f"  Table 2: Inverting Gradients Attack  (higher PSNR/SSIM = worse privacy)",
            model, all_rows, invgrad_cols
        ))

output_lines += [
    "",
    "Notes:",
    "  PSNR / SSIM : higher = attacker reconstructs better = worse privacy",
    "  MSE         : lower  = better reconstruction = worse privacy",
    "  Test Acc    : higher = better model utility",
    "  N           : number of (image × seed) pairs averaged",
    "  iDLG MSE: derived from stored PSNR via MSE = data_range² / 10^(PSNR/10)",
    "  ViT: InvGrad only (iDLG not evaluated on ViT), seed=10",
]

table_str = "\n".join(output_lines)

table_path = os.path.join(OUT_DIR, "summary_table.txt")
with open(table_path, "w", encoding="utf-8") as f:
    f.write(table_str + "\n")
print(f"\nWrote table to {table_path}")
