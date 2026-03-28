# -*- coding: utf-8 -*-
# ablation_physics_terms.py
#
# Baseline + ablation study for physics-guided loss terms:
#   1) Baseline: no physics-based loss terms
#   2) E only
#   3) M only
#   4) C1 only
#   5) E + M
#   6) E + C1
#   7) M + C1
#   8) E + M + C1  (full)
#
# Saves:
#   - ablation_summary_L{L}.csv
#   - ablation_predictions_L{L}.csv
#   - ablation_perbin_physics_L{L}.csv
#   - ablation_metrics_L{L}.pdf
#   - ablation_physics_L{L}.pdf
#
# Fonts:
#   - titles/ticks/general text: bold, size 16
#   - axis labels: bold, size 14

import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# =========================================================
# USER SETTINGS
# =========================================================
L = 32                  # choose 32 / 64 / 128
Tc = 2.269
SEED = 123
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLEAN_CSV = f"../JOB5_Noise/J5Data/MCD{L}.csv"
NOISY_CSV = f"../JOB5_Noise/J5Data/MCDN{L}.csv"

OUT_DIR = f"./Ablation_L{L}"
os.makedirs(OUT_DIR, exist_ok=True)

EPOCHS = 20
BATCH_SIZE = 32
PATIENCE = 4
N_BINS = 16

# Default optimized settings
LAM_REC = 1.0
LAM_T = 1.0
LAM_PHASE = 1.0
LAM_PHYS_DEFAULT = 0.1
LR = 5e-4
DEPTH = 4
CHANNELS = 48

# =========================================================
# PLOT STYLE
# =========================================================
plt.rcParams.update({
    "font.size": 16,
    "font.weight": "bold",
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "axes.labelweight": "bold",
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "lines.linewidth": 2.4
})

# =========================================================
# REPRODUCIBILITY
# =========================================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# =========================================================
# DATA LOADING
# =========================================================
def load_csvs(clean_path, noisy_path, L):
    df_clean = pd.read_csv(clean_path)
    df_noisy = pd.read_csv(noisy_path)
    assert len(df_clean) == len(df_noisy), "Clean and noisy CSV row counts must match."

    # flexible spin column handling
    spin_cols = [c for c in df_clean.columns if c.lower().startswith("spin")]
    if len(spin_cols) != L * L:
        spin_cols = [f"spin_{i}" for i in range(L * L)]

    T = df_clean["Temperature"].to_numpy(np.float32)

    phase_raw = df_clean["Phase"]
    if phase_raw.dtype == object or str(phase_raw.dtype).startswith("string"):
        phase = np.zeros(len(phase_raw), dtype=np.int64)
        for i, v in enumerate(phase_raw):
            vv = str(v).strip().upper()
            phase[i] = 0 if vv == "F" else 1
    else:
        phase = phase_raw.astype(np.int64).to_numpy()

    S = df_clean[spin_cols].to_numpy(np.int8).reshape(-1, L, L).astype(np.float32)   # clean
    Y = df_noisy[spin_cols].to_numpy(np.int8).reshape(-1, L, L).astype(np.float32)   # noisy
    M = (Y != 0).astype(np.float32)

    X = np.stack([Y, M], axis=1)  # (N,2,L,L)
    return X, S, T, phase, Y, M

class IsingDataset(Dataset):
    def __init__(self, X, S, T, P, Y, M):
        self.X = torch.from_numpy(X)                    # (N,2,L,L)
        self.S = torch.from_numpy(S)[:, None, ...]      # (N,1,L,L)
        self.T = torch.from_numpy(T)[:, None]           # (N,1)
        self.P = torch.from_numpy(P)                    # (N,)
        self.Y = torch.from_numpy(Y)[:, None, ...]      # (N,1,L,L)
        self.M = torch.from_numpy(M)[:, None, ...]      # (N,1,L,L)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {
            "X": self.X[idx],
            "S": self.S[idx],
            "T": self.T[idx],
            "P": self.P[idx],
            "Y": self.Y[idx],
            "M": self.M[idx]
        }

def make_loaders(X, S, T, P, Y, M, batch_size=BATCH_SIZE):
    ds = IsingDataset(X, S, T, P, Y, M)
    N = len(ds)
    n_train = int(0.8 * N)
    n_val = int(0.1 * N)
    n_test = N - n_train - n_val

    g = torch.Generator().manual_seed(SEED)
    ds_train, ds_val, ds_test = random_split(ds, [n_train, n_val, n_test], generator=g)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

# =========================================================
# PHYSICS HELPERS
# =========================================================
class PeriodicConv(nn.Module):
    def __init__(self, kernel_3x3):
        super().__init__()
        k = torch.tensor(kernel_3x3, dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("weight", k)

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode="circular")
        return F.conv2d(x, self.weight, bias=None, stride=1, padding=0)

NN_KERNEL = [[0, 1, 0],
             [1, 0, 1],
             [0, 1, 0]]

nn_conv = PeriodicConv(NN_KERNEL).to(DEVICE)

def energy_per_site(s):
    nbr = nn_conv(s)
    return -(s * nbr).mean(dim=(1, 2, 3)) / 2.0

def nn_corr(s):
    nbr = nn_conv(s)
    return (s * nbr).mean(dim=(1, 2, 3)) / 4.0

def mag_abs(s):
    return s.mean(dim=(2, 3)).abs().squeeze(1)

def cv_from_bin_means(T_centers, E_means):
    T_centers = np.asarray(T_centers, dtype=float)
    E_means = np.asarray(E_means, dtype=float)

    if len(T_centers) < 2:
        return np.zeros_like(T_centers)

    order = np.argsort(T_centers)
    T_centers = T_centers[order]
    E_means = E_means[order]

    Cv = np.zeros_like(T_centers)
    Cv[0] = (E_means[1] - E_means[0]) / (T_centers[1] - T_centers[0] + 1e-12)
    Cv[-1] = (E_means[-1] - E_means[-2]) / (T_centers[-1] - T_centers[-2] + 1e-12)

    for i in range(1, len(T_centers) - 1):
        Cv[i] = (E_means[i + 1] - E_means[i - 1]) / (T_centers[i + 1] - T_centers[i - 1] + 1e-12)

    return np.maximum(Cv, 0.0)

# =========================================================
# MODEL
# =========================================================
class InvNet(nn.Module):
    def __init__(self, C=CHANNELS, depth=DEPTH):
        super().__init__()

        layers = [
            nn.Conv2d(2, C, 3, padding=1, padding_mode="circular"),
            nn.ReLU(inplace=True)
        ]
        for _ in range(depth - 1):
            layers += [
                nn.Conv2d(C, C, 3, padding=1, padding_mode="circular"),
                nn.ReLU(inplace=True)
            ]
        self.enc = nn.Sequential(*layers)

        self.dec = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1, padding_mode="circular"),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Tanh()
        )

        self.head_T = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(C, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

        self.head_P = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(C, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        h = self.enc(x)
        S_hat = self.dec(h)
        T_hat = self.head_T(h)
        P_log = self.head_P(h)
        return S_hat, T_hat, P_log

# =========================================================
# METRICS
# =========================================================
def classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return precision, recall, f1

# =========================================================
# TRAIN + EVALUATE ONE CONFIG
# =========================================================
def run_one_experiment(cfg, train_loader, val_loader, test_loader, device=DEVICE):
    model = InvNet(C=cfg["C"], depth=cfg["DEPTH"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["LR"])
    ce = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    patience_count = 0

    train_losses = []
    val_losses = []

    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_train = 0.0

        for batch in train_loader:
            X = batch["X"].to(device)
            S = batch["S"].to(device)
            Tt = batch["T"].to(device)
            P = batch["P"].to(device)
            Y = batch["Y"].to(device)
            M = batch["M"].to(device)

            S_hat, T_hat, P_log = model(X)

            L_rec = (M * (S_hat - Y).abs()).sum() / (M.sum() + 1e-8)
            L_T = (T_hat - Tt).abs().mean()
            L_phase = ce(P_log, P)

            S_sign = torch.sign(S_hat)
            L_E = (energy_per_site(S_sign) - energy_per_site(S)).abs().mean()
            L_M = (mag_abs(S_sign) - mag_abs(S)).abs().mean()
            L_C1 = (nn_corr(S_sign) - nn_corr(S)).abs().mean()

            loss = (
                cfg["LAM_REC"] * L_rec +
                cfg["LAM_T"] * L_T +
                cfg["LAM_PHASE"] * L_phase +
                cfg["LAM_E"] * L_E +
                cfg["LAM_M"] * L_M +
                cfg["LAM_C1"] * L_C1
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running_train += float(loss.item())

        running_train /= max(1, len(train_loader))
        train_losses.append(running_train)

        # validation
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for batch in val_loader:
                X = batch["X"].to(device)
                S = batch["S"].to(device)
                Tt = batch["T"].to(device)
                P = batch["P"].to(device)
                Y = batch["Y"].to(device)
                M = batch["M"].to(device)

                S_hat, T_hat, P_log = model(X)

                L_rec = (M * (S_hat - Y).abs()).sum() / (M.sum() + 1e-8)
                L_T = (T_hat - Tt).abs().mean()
                L_phase = ce(P_log, P)

                S_sign = torch.sign(S_hat)
                L_E = (energy_per_site(S_sign) - energy_per_site(S)).abs().mean()
                L_M = (mag_abs(S_sign) - mag_abs(S)).abs().mean()
                L_C1 = (nn_corr(S_sign) - nn_corr(S)).abs().mean()

                loss = (
                    cfg["LAM_REC"] * L_rec +
                    cfg["LAM_T"] * L_T +
                    cfg["LAM_PHASE"] * L_phase +
                    cfg["LAM_E"] * L_E +
                    cfg["LAM_M"] * L_M +
                    cfg["LAM_C1"] * L_C1
                )
                running_val += float(loss.item())

        running_val /= max(1, len(val_loader))
        val_losses.append(running_val)

        if running_val < best_val:
            best_val = running_val
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break

    elapsed = time.time() - t0

    model.load_state_dict(best_state)
    model.eval()

    # test predictions
    all_rows_pred = []
    all_rows_bins = []

    T_true_list, T_pred_list = [], []
    P_true_list, P_pred_list = [], []
    S_true_list, S_pred_list, Mask_list = [], [], []

    E_true_list, E_pred_list = [], []
    M_true_list, M_pred_list = [], []
    C_true_list, C_pred_list = [], []

    with torch.no_grad():
        for batch in test_loader:
            X = batch["X"].to(device)
            S = batch["S"].to(device)
            Tt = batch["T"].to(device)
            P = batch["P"].to(device)
            M = batch["M"].to(device)

            S_hat, T_hat, P_log = model(X)
            S_sign = torch.sign(S_hat)
            P_hat = P_log.argmax(1)

            T_true_np = Tt.cpu().numpy().reshape(-1)
            T_pred_np = T_hat.cpu().numpy().reshape(-1)
            P_true_np = P.cpu().numpy().reshape(-1)
            P_pred_np = P_hat.cpu().numpy().reshape(-1)

            S_true_np = S.cpu().numpy()
            S_pred_np = S_sign.cpu().numpy()
            Mask_np = M.cpu().numpy()

            E_true_np = energy_per_site(S).cpu().numpy()
            E_pred_np = energy_per_site(S_sign).cpu().numpy()
            M_true_np = mag_abs(S).cpu().numpy()
            M_pred_np = mag_abs(S_sign).cpu().numpy()
            C_true_np = nn_corr(S).cpu().numpy()
            C_pred_np = nn_corr(S_sign).cpu().numpy()

            for i in range(len(T_true_np)):
                all_rows_pred.append({
                    "Config": cfg["NAME"],
                    "T_true": float(T_true_np[i]),
                    "T_pred": float(T_pred_np[i]),
                    "P_true": int(P_true_np[i]),
                    "P_pred": int(P_pred_np[i]),
                    "E_true": float(E_true_np[i]),
                    "E_pred": float(E_pred_np[i]),
                    "M_true": float(M_true_np[i]),
                    "M_pred": float(M_pred_np[i]),
                    "C1_true": float(C_true_np[i]),
                    "C1_pred": float(C_pred_np[i]),
                })

            T_true_list.append(T_true_np)
            T_pred_list.append(T_pred_np)
            P_true_list.append(P_true_np)
            P_pred_list.append(P_pred_np)
            S_true_list.append(S_true_np)
            S_pred_list.append(S_pred_np)
            Mask_list.append(Mask_np)
            E_true_list.append(E_true_np)
            E_pred_list.append(E_pred_np)
            M_true_list.append(M_true_np)
            M_pred_list.append(M_pred_np)
            C_true_list.append(C_true_np)
            C_pred_list.append(C_pred_np)

    T_true = np.concatenate(T_true_list).reshape(-1)
    T_pred = np.concatenate(T_pred_list).reshape(-1)
    P_true = np.concatenate(P_true_list).reshape(-1)
    P_pred = np.concatenate(P_pred_list).reshape(-1)

    S_true = np.concatenate(S_true_list, axis=0)
    S_pred = np.concatenate(S_pred_list, axis=0)
    Mask = np.concatenate(Mask_list, axis=0)

    E_true = np.concatenate(E_true_list).reshape(-1)
    E_pred = np.concatenate(E_pred_list).reshape(-1)
    M_true = np.concatenate(M_true_list).reshape(-1)
    M_pred = np.concatenate(M_pred_list).reshape(-1)
    C_true = np.concatenate(C_true_list).reshape(-1)
    C_pred = np.concatenate(C_pred_list).reshape(-1)

    mae_T = float(np.mean(np.abs(T_true - T_pred)))
    acc_phi = float(np.mean(P_true == P_pred))

    precision, recall, f1 = classification_metrics(P_true, P_pred)

    miss = (Mask == 0)
    imp_acc = float(((S_true == S_pred) & miss).sum() / max(1, miss.sum()))

    # per-bin
    bins = np.linspace(1.0, 4.0, N_BINS + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    bin_ids = np.digitize(T_true, bins) - 1

    T_used, E_true_mean, E_pred_mean = [], [], []
    perbin_temp = []

    for b in range(N_BINS):
        idx = (bin_ids == b)
        if not np.any(idx):
            continue

        T_bar = float(centers[b])
        e_t = float(E_true[idx].mean())
        e_p = float(E_pred[idx].mean())
        m_t = float(M_true[idx].mean())
        m_p = float(M_pred[idx].mean())
        c_t = float(C_true[idx].mean())
        c_p = float(C_pred[idx].mean())

        T_used.append(T_bar)
        E_true_mean.append(e_t)
        E_pred_mean.append(e_p)

        perbin_temp.append({
            "Config": cfg["NAME"],
            "T_bin": T_bar,
            "E_true_mean": e_t,
            "E_pred_mean": e_p,
            "M_true_mean": m_t,
            "M_pred_mean": m_p,
            "C1_true_mean": c_t,
            "C1_pred_mean": c_p
        })

    Cv_true = cv_from_bin_means(T_used, E_true_mean)
    Cv_pred = cv_from_bin_means(T_used, E_pred_mean)

    for i in range(len(perbin_temp)):
        perbin_temp[i]["Cv_true"] = float(Cv_true[i])
        perbin_temp[i]["Cv_pred"] = float(Cv_pred[i])

    all_rows_bins.extend(perbin_temp)

    # overall bin-wise errors
    if len(all_rows_bins) > 0:
        df_tmp = pd.DataFrame(all_rows_bins)
        E_err = float(np.mean(np.abs(df_tmp["E_true_mean"] - df_tmp["E_pred_mean"])))
        M_err = float(np.mean(np.abs(df_tmp["M_true_mean"] - df_tmp["M_pred_mean"])))
        C1_err = float(np.mean(np.abs(df_tmp["C1_true_mean"] - df_tmp["C1_pred_mean"])))
        Cv_err = float(np.mean(np.abs(df_tmp["Cv_true"] - df_tmp["Cv_pred"])))
    else:
        E_err = np.nan
        M_err = np.nan
        C1_err = np.nan
        Cv_err = np.nan

    result = {
        "NAME": cfg["NAME"],
        "LAM_E": cfg["LAM_E"],
        "LAM_M": cfg["LAM_M"],
        "LAM_C1": cfg["LAM_C1"],
        "MAE_T": mae_T,
        "Acc_phi": acc_phi,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ImpAcc": imp_acc,
        "E_err": E_err,
        "Mabs_err": M_err,
        "C1_err": C1_err,
        "Cv_err": Cv_err,
        "BestVal": float(best_val),
        "BestEpoch": int(best_epoch),
        "Time_s": float(elapsed),
        "Time_h": float(elapsed / 3600.0)
    }

    return result, pd.DataFrame(all_rows_pred), pd.DataFrame(all_rows_bins), train_losses, val_losses

# =========================================================
# EXPERIMENT CONFIGS
# =========================================================
BASE_COMMON = {
    "LAM_REC": LAM_REC,
    "LAM_T": LAM_T,
    "LAM_PHASE": LAM_PHASE,
    "DEPTH": DEPTH,
    "C": CHANNELS,
    "LR": LR
}

EXPERIMENTS = [
    {"NAME": "Baseline_NoPhysics", "LAM_E": 0.0,               "LAM_M": 0.0,               "LAM_C1": 0.0},
    {"NAME": "Only_E",             "LAM_E": LAM_PHYS_DEFAULT,  "LAM_M": 0.0,               "LAM_C1": 0.0},
    {"NAME": "Only_M",             "LAM_E": 0.0,               "LAM_M": LAM_PHYS_DEFAULT,  "LAM_C1": 0.0},
    {"NAME": "Only_C1",            "LAM_E": 0.0,               "LAM_M": 0.0,               "LAM_C1": LAM_PHYS_DEFAULT},
    {"NAME": "E_plus_M",           "LAM_E": LAM_PHYS_DEFAULT,  "LAM_M": LAM_PHYS_DEFAULT,  "LAM_C1": 0.0},
    {"NAME": "E_plus_C1",          "LAM_E": LAM_PHYS_DEFAULT,  "LAM_M": 0.0,               "LAM_C1": LAM_PHYS_DEFAULT},
    {"NAME": "M_plus_C1",          "LAM_E": 0.0,               "LAM_M": LAM_PHYS_DEFAULT,  "LAM_C1": LAM_PHYS_DEFAULT},
    {"NAME": "Full_E_M_C1",        "LAM_E": LAM_PHYS_DEFAULT,  "LAM_M": LAM_PHYS_DEFAULT,  "LAM_C1": LAM_PHYS_DEFAULT},
]

# =========================================================
# RUN
# =========================================================
assert os.path.exists(CLEAN_CSV), f"Missing file: {CLEAN_CSV}"
assert os.path.exists(NOISY_CSV), f"Missing file: {NOISY_CSV}"

X, S, T, P, Y, M = load_csvs(CLEAN_CSV, NOISY_CSV, L)
train_loader, val_loader, test_loader = make_loaders(X, S, T, P, Y, M, batch_size=BATCH_SIZE)

summary_rows = []
pred_rows_all = []
perbin_rows_all = []
train_history = {}
val_history = {}

for exp in EXPERIMENTS:
    cfg = dict(BASE_COMMON)
    cfg.update(exp)

    print(f"Running: {cfg['NAME']}")
    result, df_pred, df_bin, tr_hist, va_hist = run_one_experiment(cfg, train_loader, val_loader, test_loader, device=DEVICE)

    summary_rows.append(result)
    pred_rows_all.append(df_pred)
    perbin_rows_all.append(df_bin)
    train_history[cfg["NAME"]] = tr_hist
    val_history[cfg["NAME"]] = va_hist

# save CSVs
df_summary = pd.DataFrame(summary_rows).sort_values("MAE_T").reset_index(drop=True)
df_pred_all = pd.concat(pred_rows_all, axis=0, ignore_index=True)
df_bin_all = pd.concat(perbin_rows_all, axis=0, ignore_index=True)

summary_csv = os.path.join(OUT_DIR, f"ablation_summary_L{L}.csv")
pred_csv = os.path.join(OUT_DIR, f"ablation_predictions_L{L}.csv")
perbin_csv = os.path.join(OUT_DIR, f"ablation_perbin_physics_L{L}.csv")

df_summary.to_csv(summary_csv, index=False)
df_pred_all.to_csv(pred_csv, index=False)
df_bin_all.to_csv(perbin_csv, index=False)

# =========================================================
# PLOTS: PDF 1 (task metrics)
# =========================================================
pdf_metrics = os.path.join(OUT_DIR, f"ablation_metrics_L{L}.pdf")
names = df_summary["NAME"].tolist()
x = np.arange(len(names))

with PdfPages(pdf_metrics) as pdf:
    # page 1: task metrics
    fig = plt.figure(figsize=(12, 7))
    plt.plot(x, df_summary["MAE_T"], marker="o", label="MAE_T")
    plt.plot(x, df_summary["Acc_phi"], marker="s", label="Acc_phi")
    plt.plot(x, df_summary["ImpAcc"], marker="^", label="ImpAcc")
    plt.xticks(x, names, rotation=35, ha="right", fontweight="bold")
    plt.xlabel("Model configuration", fontweight="bold", fontsize=14)
    plt.ylabel("Metric value", fontweight="bold", fontsize=14)
    plt.title(f"Baseline and ablation comparison (L={L})", fontweight="bold", fontsize=16)
    plt.legend()
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # page 2: classification metrics
    fig = plt.figure(figsize=(12, 7))
    plt.plot(x, df_summary["Precision"], marker="o", label="Precision")
    plt.plot(x, df_summary["Recall"], marker="s", label="Recall")
    plt.plot(x, df_summary["F1"], marker="^", label="F1")
    plt.xticks(x, names, rotation=35, ha="right", fontweight="bold")
    plt.xlabel("Model configuration", fontweight="bold", fontsize=14)
    plt.ylabel("Score", fontweight="bold", fontsize=14)
    plt.title(f"Phase-classification ablation (L={L})", fontweight="bold", fontsize=16)
    plt.legend()
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # page 3: convergence curves
    fig = plt.figure(figsize=(12, 7))
    for name in train_history:
        plt.plot(train_history[name], label=f"{name} - train")
        plt.plot(val_history[name], linestyle="--", label=f"{name} - val")
    plt.xlabel("Epoch", fontweight="bold", fontsize=14)
    plt.ylabel("Loss", fontweight="bold", fontsize=14)
    plt.title(f"Training / validation loss curves (L={L})", fontweight="bold", fontsize=16)
    plt.legend(ncol=2)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

# =========================================================
# PLOTS: PDF 2 (physics observables)
# =========================================================
pdf_physics = os.path.join(OUT_DIR, f"ablation_physics_L{L}.pdf")

with PdfPages(pdf_physics) as pdf:
    # page 1: overall physics errors
    fig = plt.figure(figsize=(12, 7))
    plt.plot(x, df_summary["E_err"], marker="o", label="E error")
    plt.plot(x, df_summary["Mabs_err"], marker="s", label="|M| error")
    plt.plot(x, df_summary["C1_err"], marker="^", label="C1 error")
    plt.plot(x, df_summary["Cv_err"], marker="d", label="Cv error")
    plt.xticks(x, names, rotation=35, ha="right", fontweight="bold")
    plt.xlabel("Model configuration", fontweight="bold", fontsize=14)
    plt.ylabel("Absolute error", fontweight="bold", fontsize=14)
    plt.title(f"Physics-observable error comparison (L={L})", fontweight="bold", fontsize=16)
    plt.legend()
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # pages 2-5: per-bin curves
    for quantity, y_true, y_pred, title_str in [
        ("Energy", "E_true_mean", "E_pred_mean", "Energy per bin"),
        ("Magnetization", "M_true_mean", "M_pred_mean", "Absolute magnetization per bin"),
        ("Correlation", "C1_true_mean", "C1_pred_mean", "Nearest-neighbor correlation per bin"),
        ("HeatCapacity", "Cv_true", "Cv_pred", "Heat capacity per bin"),
    ]:
        fig = plt.figure(figsize=(12, 7))
        for name in df_bin_all["Config"].unique():
            sub = df_bin_all[df_bin_all["Config"] == name].sort_values("T_bin")
            plt.plot(sub["T_bin"], sub[y_pred], marker="o", label=f"{name} pred")

        # true curve once from first config
        sub0 = df_bin_all[df_bin_all["Config"] == df_bin_all["Config"].unique()[0]].sort_values("T_bin")
        plt.plot(sub0["T_bin"], sub0[y_true], color="black", linestyle="--", marker="s", label="True")

        plt.xlabel("Temperature bin", fontweight="bold", fontsize=14)
        plt.ylabel(quantity, fontweight="bold", fontsize=14)
        plt.title(f"{title_str} (L={L})", fontweight="bold", fontsize=16)
        plt.legend(ncol=2)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

print("\nSaved files:")
print(" -", summary_csv)
print(" -", pred_csv)
print(" -", perbin_csv)
print(" -", pdf_metrics)
print(" -", pdf_physics)