# -*- coding: utf-8 -*-
# compare_imputation_L32.py
# Baselines: Mean imputation, k-NN imputation
# Metrics: MAE_T, Acc_phi, ImpAcc, and physics observables (E, |M|, C1, Cv) per temperature bin.
# Saves: compare_global_L32.csv, compare_per_temp_bin_L32.csv, physics_compare_L32.pdf
#
# Expected (from your proposed code run at L=32):
#   PROPOSED_OUT/metrics_global.csv
#   PROPOSED_OUT/metrics_per_temp_bin.csv

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Optional but recommended for kNN
from sklearn.impute import KNNImputer

# -----------------------
# User parameters (edit)
# -----------------------
L = 128              # set to 32, 64, or 128
Tc = 2.269
N_BINS = 16
SEED = 123

# Auto-resolve paths from L
CLEAN_CSV = f"../JOB5_Noise/J5Data/MCD{L}.csv"
NOISY_CSV = f"../JOB5_Noise/J5Data/MCDN{L}.csv"

PROPOSED_OUT = f"./Outputs_L{L}"    # output folder from proposed model
OUT_DIR = f"./Compare_L{L}"         # comparison outputs
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# Plot style (PRE/PRB)
# -----------------------
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "lines.linewidth": 2.2,
    "legend.fontsize": 14,
})

rng = np.random.default_rng(SEED)

# -----------------------
# Helpers
# -----------------------
def spin_cols(L):
    return [f"spin_{i}" for i in range(L * L)]

def phase_to_int(phase_series):
    # Map 'F'->0, 'P'->1; if numeric keep.
    if phase_series.dtype == object or str(phase_series.dtype).startswith("string"):
        out = np.zeros(len(phase_series), dtype=np.int64)
        for i, v in enumerate(phase_series):
            if isinstance(v, str):
                vv = v.strip().upper()
                if vv == "F": out[i] = 0
                elif vv == "P": out[i] = 1
                else:
                    try: out[i] = int(float(v))
                    except: out[i] = 1
            else:
                try: out[i] = int(v)
                except: out[i] = 1
        return out
    return phase_series.astype(np.int64).to_numpy()

def split_indices(N, seed=SEED):
    # Match your 80/10/10 split with fixed seed
    idx = np.arange(N)
    rng_local = np.random.default_rng(seed)
    rng_local.shuffle(idx)
    n_train = int(0.8 * N)
    n_val = int(0.1 * N)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val:]
    return train_idx, val_idx, test_idx

def energy_per_site_np(S):  # S: (N,L,L) values in {-1,+1}
    # E = - (1/(N_sites)) * sum_{<ij>} s_i s_j / 2 with PBC
    # Implement using rolls.
    right = np.roll(S, shift=-1, axis=2)
    left  = np.roll(S, shift=+1, axis=2)
    down  = np.roll(S, shift=-1, axis=1)
    up    = np.roll(S, shift=+1, axis=1)
    nbr_sum = right + left + down + up
    return - (S * nbr_sum).mean(axis=(1,2)) / 2.0

def mag_abs_np(S):
    return np.abs(S.mean(axis=(1,2)))

def c1_np(S):
    # C1 = <s_i s_j> over nn; with 4 neighbors, divide by 4
    right = np.roll(S, shift=-1, axis=2)
    left  = np.roll(S, shift=+1, axis=2)
    down  = np.roll(S, shift=-1, axis=1)
    up    = np.roll(S, shift=+1, axis=1)
    nbr_sum = right + left + down + up
    return (S * nbr_sum).mean(axis=(1,2)) / 4.0

def cv_from_derivative(T_centers, E_means):
    # Cv ~ d<E>/dT via central difference on binned means
    if len(T_centers) < 2:
        return np.zeros_like(T_centers, dtype=float)
    order = np.argsort(T_centers)
    T = np.array(T_centers)[order]
    E = np.array(E_means)[order]
    Cv = np.zeros_like(T, dtype=float)
    # boundaries
    Cv[0]  = (E[1] - E[0]) / (T[1] - T[0] + 1e-12)
    Cv[-1] = (E[-1] - E[-2]) / (T[-1] - T[-2] + 1e-12)
    # interior
    for i in range(1, len(T)-1):
        Cv[i] = (E[i+1] - E[i-1]) / (T[i+1] - T[i-1] + 1e-12)
    # clip negatives (optional)
    return np.maximum(Cv, 0.0)

def per_temp_bin_stats(T_true, S_true, S_imp, miss_mask, n_bins=N_BINS):
    # Bins across [1,4] consistent with your code
    bins = np.linspace(1.0, 4.0, n_bins + 1)
    bin_ids = np.digitize(T_true, bins) - 1
    centers = 0.5 * (bins[:-1] + bins[1:])

    rows = []
    # Precompute per-sample physics
    E_t = energy_per_site_np(S_true)
    E_p = energy_per_site_np(S_imp)
    M_t = mag_abs_np(S_true)
    M_p = mag_abs_np(S_imp)
    C_t = c1_np(S_true)
    C_p = c1_np(S_imp)

    # Per-bin means for Cv derivative
    T_cent, E_t_m, E_p_m = [], [], []
    for b in range(n_bins):
        idx = (bin_ids == b)
        if np.any(idx):
            T_cent.append(float(centers[b]))
            E_t_m.append(float(E_t[idx].mean()))
            E_p_m.append(float(E_p[idx].mean()))
    Cv_t = cv_from_derivative(T_cent, E_t_m)
    Cv_p = cv_from_derivative(T_cent, E_p_m)
    cv_lookup = {T_cent[i]: (Cv_t[i], Cv_p[i]) for i in range(len(T_cent))}

    for b in range(n_bins):
        idx = (bin_ids == b)
        if not np.any(idx):
            rows.append({
                "T_center": float(centers[b]), "count": 0,
                "E_true": np.nan, "E_pred": np.nan,
                "Mabs_true": np.nan, "Mabs_pred": np.nan,
                "C1_true": np.nan, "C1_pred": np.nan,
                "Cv_true": np.nan, "Cv_pred": np.nan,
                "ImpAcc_missing": np.nan,
            })
            continue

        # ImpAcc only on missing locations
        miss_b = miss_mask[idx]  # (nb,L,L) boolean
        correct_missing = ((S_imp[idx] == S_true[idx]) & miss_b).sum()
        total_missing = miss_b.sum()
        imp_acc = float(correct_missing / max(1, total_missing))

        tbar = float(centers[b])
        cv_t, cv_p = cv_lookup.get(tbar, (0.0, 0.0))

        rows.append({
            "T_center": tbar,
            "count": int(idx.sum()),
            "E_true": float(E_t[idx].mean()), "E_pred": float(E_p[idx].mean()),
            "Mabs_true": float(M_t[idx].mean()), "Mabs_pred": float(M_p[idx].mean()),
            "C1_true": float(C_t[idx].mean()), "C1_pred": float(C_p[idx].mean()),
            "Cv_true": float(cv_t), "Cv_pred": float(cv_p),
            "ImpAcc_missing": imp_acc,
        })

    return pd.DataFrame(rows).sort_values("T_center").reset_index(drop=True)

def phase_from_T(T, Tc=Tc):
    # Consistent with your F/P split: F (0) if T < Tc else P (1)
    return (T >= Tc).astype(np.int64)

def temperature_baseline_from_imputed(S_imp):
    # Minimal temperature proxy for baselines (since they do not predict T directly):
    # Use energy as a monotonic proxy: fit linear map T ˜ a*E + b on TRAIN set.
    # This gives a fair "MAE_T" for baselines without adding complex models.
    E = energy_per_site_np(S_imp)
    return E

def fit_linear_map(x_train, y_train):
    # y ˜ a*x + b
    x = np.asarray(x_train).reshape(-1)
    y = np.asarray(y_train).reshape(-1)
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)

# -----------------------
# Load data
# -----------------------
df_c = pd.read_csv(CLEAN_CSV)
df_n = pd.read_csv(NOISY_CSV)

cols = spin_cols(L)

T_all = df_c["Temperature"].to_numpy(dtype=np.float32)
P_all = phase_to_int(df_c["Phase"])

S_true_all = df_c[cols].to_numpy(dtype=np.int8).reshape(-1, L, L)   # {-1,+1}
Y_all = df_n[cols].to_numpy(dtype=np.int8).reshape(-1, L, L)       # {-1,+1,0}
miss_all = (Y_all == 0)                                            # missing mask
obs_all = ~miss_all

N = len(df_c)
train_idx, val_idx, test_idx = split_indices(N, SEED)

# -----------------------
# Baseline 1: Mean imputation
# -----------------------
# Mean of observed spins over TRAIN (scalar), then impute missing with sign(mean)
train_obs_vals = Y_all[train_idx][obs_all[train_idx]]
mu = float(train_obs_vals.mean()) if train_obs_vals.size else 0.0
fill_mean = 1.0 if mu >= 0 else -1.0

S_mean = Y_all.astype(np.float32).copy()
S_mean[miss_all] = fill_mean
S_mean = np.where(S_mean >= 0, 1.0, -1.0)  # force to {-1,+1}

# -----------------------
# Baseline 2: k-NN imputation (vectorized per sample)
# -----------------------
# Prepare matrix with NaN for missing (required by KNNImputer)
Y_flat = Y_all.reshape(N, -1).astype(np.float32)
Y_flat_nan = Y_flat.copy()
Y_flat_nan[Y_flat_nan == 0] = np.nan

imputer = KNNImputer(n_neighbors=5, weights="distance")
# Fit on TRAIN only (standard protocol), transform all, then evaluate on TEST
imputer.fit(Y_flat_nan[train_idx])
Y_knn = imputer.transform(Y_flat_nan)  # (N, L*L) real-valued
Y_knn = np.where(Y_knn >= 0, 1.0, -1.0).reshape(N, L, L)  # discretize to {-1,+1}

# -----------------------
# Load proposed outputs if available (for MAE_T and Acc_phi)
# -----------------------
proposed_maeT = np.nan
proposed_accphi = np.nan
proposed_impacc = np.nan
if os.path.exists(os.path.join(PROPOSED_OUT, "metrics_global.csv")):
    mg = pd.read_csv(os.path.join(PROPOSED_OUT, "metrics_global.csv"))
    # Your file uses: Temperature_MAE, Phase_Accuracy, Imputation_Accuracy_on_missing
    proposed_maeT = float(mg.loc[0, "Temperature_MAE"])
    proposed_accphi = float(mg.loc[0, "Phase_Accuracy"])
    proposed_impacc = float(mg.loc[0, "Imputation_Accuracy_on_missing"])

# -----------------------
# For baselines: derive T_pred via linear map from energy proxy on TRAIN
# -----------------------
def eval_method(name, S_imp_all):
    # Evaluate on TEST split
    S_imp = S_imp_all[test_idx]
    S_true = S_true_all[test_idx]
    miss = miss_all[test_idx]
    T_true = T_all[test_idx]

    # ImpAcc on missing
    impacc = float(((S_imp == S_true) & miss).sum() / max(1, miss.sum()))

    # Temperature baseline: energy proxy + linear calibration on TRAIN
    E_proxy_train = temperature_baseline_from_imputed(S_imp_all[train_idx])
    a, b = fit_linear_map(E_proxy_train, T_all[train_idx])
    E_proxy_test = temperature_baseline_from_imputed(S_imp_all[test_idx])
    T_pred = a * E_proxy_test + b
    maeT = float(np.mean(np.abs(T_true - T_pred)))

    # Phase baseline: from predicted temperature threshold
    P_pred = phase_from_T(T_pred, Tc=Tc)
    P_true = phase_from_T(T_true, Tc=Tc)  # ensure consistent definition
    accphi = float(np.mean(P_true == P_pred))

    # Physics per-bin
    perbin = per_temp_bin_stats(T_true, S_true, S_imp, miss, n_bins=N_BINS)
    perbin.insert(0, "Method", name)
    return maeT, accphi, impacc, perbin

mae_mean, acc_mean, imp_mean, perbin_mean = eval_method("MeanImpute", S_mean)
mae_knn,  acc_knn,  imp_knn,  perbin_knn  = eval_method("KNNImpute",  Y_knn)

# -----------------------
# Global comparison table
# -----------------------
rows = []
if not np.isnan(proposed_maeT):
    rows.append({"Method": "Proposed", "MAE_T": proposed_maeT, "Acc_phi": proposed_accphi, "ImpAcc": proposed_impacc})
rows.append({"Method": "MeanImpute", "MAE_T": mae_mean, "Acc_phi": acc_mean, "ImpAcc": imp_mean})
rows.append({"Method": "KNNImpute",  "MAE_T": mae_knn,  "Acc_phi": acc_knn,  "ImpAcc": imp_knn})

global_df = pd.DataFrame(rows)
global_csv = os.path.join(OUT_DIR, "compare_global_L{L}.csv")
global_df.to_csv(global_csv, index=False)

# -----------------------
# Per-bin combined CSV
# -----------------------
perbin_df = pd.concat([perbin_mean, perbin_knn], ignore_index=True)

# If proposed per-bin exists, append it (rename columns to match "pred" naming)
ppath = os.path.join(PROPOSED_OUT, "metrics_per_temp_bin.csv")
if os.path.exists(ppath):
    p = pd.read_csv(ppath)
    # Expected columns in your perbin: T_center,count,E_true,E_pred,Mabs_true,Mabs_pred,C1_true,C1_pred,Cv_true,Cv_pred,...
    p2 = p[["T_center","count","E_true","E_pred","Mabs_true","Mabs_pred","C1_true","C1_pred","Cv_true","Cv_pred"]].copy()
    # ImpAcc per bin (if present in your file)
    if "Accuracy_missing" in p.columns:
        p2["ImpAcc_missing"] = p["Accuracy_missing"]
    else:
        p2["ImpAcc_missing"] = np.nan
    p2.insert(0, "Method", "Proposed")
    perbin_df = pd.concat([p2, perbin_df], ignore_index=True)

perbin_csv = os.path.join(OUT_DIR, "compare_per_temp_bin_L{L}.csv")
perbin_df.to_csv(perbin_csv, index=False)

# -----------------------
# Plot PDF: physics curves (True vs each method's pred)
# -----------------------
pdf_path = os.path.join(OUT_DIR, "physics_compare_L{L}.pdf")
with PdfPages(pdf_path) as pdf:
    # For each observable: plot True + each method's pred
    def plot_panel(ax, obs_true_col, obs_pred_col, title, ylabel):
        # True curve (from any one method's rows) -> use Proposed if present else MeanImpute
        methods = perbin_df["Method"].unique().tolist()
        base_method = "Proposed" if "Proposed" in methods else "MeanImpute"

        base = perbin_df[perbin_df["Method"] == base_method]
        ax.plot(base["T_center"], base[obs_true_col], marker="o", color="black", label="True")

        for m in methods:
            d = perbin_df[perbin_df["Method"] == m]
            ax.plot(d["T_center"], d[obs_pred_col], marker="s", label=m)

        ax.axvline(Tc, linestyle="--", color="gray", alpha=0.8)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("T", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.legend()

    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
    plot_panel(axs[0,0], "Mabs_true", "Mabs_pred", r"$|M|(T)$", r"$|M|$")
    plot_panel(axs[0,1], "E_true",    "E_pred",    r"$E(T)$",    "Energy")
    plot_panel(axs[1,0], "C1_true",   "C1_pred",   r"$C1(T)$",   "C1")
    plot_panel(axs[1,1], "Cv_true",   "Cv_pred",   r"$C_v(T)$",  r"$C_v$")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print("Saved:")
print(" -", global_csv)
print(" -", perbin_csv)
print(" -", pdf_path)
