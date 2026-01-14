# -*- coding: utf-8 -*-
# sensitivity_analysis_min.py
# Minimal sensitivity study for:
# (1) loss weights (LAM_E, LAM_M, LAM_C1)
# (2) encoder depth and channel size
# (3) learning rate
# Saves: CSV summary + PDF plots (bold, size 16)

import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# =========================
# USER PARAMS (general)
# =========================
L = 32             # set 32 / 64 / 128
Tc = 2.269
SEED = 123
DEVICE = "cpu"

CLEAN_CSV = f"../JOB5_Noise/J5Data/MCD{L}.csv"
NOISY_CSV = f"../JOB5_Noise/J5Data/MCDN{L}.csv"

OUT_DIR = f"./Sensitivity_L{L}"
os.makedirs(OUT_DIR, exist_ok=True)

# Fast sensitivity (keep small); increase if needed
EPOCHS = 15
BATCH_SIZE = 32
PATIENCE = 3
N_BINS = 16

# Plot style: bold + 16
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "lines.linewidth": 2.2,
    "legend.fontsize": 13
})

torch.manual_seed(SEED)
np.random.seed(SEED)

# =========================
# Data loading (matches your format)
# =========================
def load_csvs(clean_path, noisy_path, L):
    df_clean = pd.read_csv(clean_path)
    df_noisy = pd.read_csv(noisy_path)
    assert len(df_clean) == len(df_noisy)

    spin_cols = [f"spin_{i}" for i in range(L*L)]
    T = df_clean["Temperature"].to_numpy(np.float32)

    # Phase: handle 'F'/'P' or numeric
    Phase_raw = df_clean["Phase"]
    if Phase_raw.dtype == object or str(Phase_raw.dtype).startswith("string"):
        Phase = np.zeros(len(Phase_raw), dtype=np.int64)
        for i, v in enumerate(Phase_raw):
            if isinstance(v, str):
                vv = v.strip().upper()
                Phase[i] = 0 if vv == "F" else 1
            else:
                Phase[i] = int(v) if pd.notna(v) else 1
    else:
        Phase = Phase_raw.astype(np.int64).to_numpy()

    Sigma = df_clean[spin_cols].to_numpy(np.int8).reshape(-1, L, L).astype(np.float32)  # true spins
    Y     = df_noisy[spin_cols].to_numpy(np.int8).reshape(-1, L, L).astype(np.float32) # noisy with 0 as missing
    Mask  = (Y != 0).astype(np.float32)

    X = np.stack([Y, Mask], axis=1)  # (N,2,L,L)
    return X, Sigma, T, Phase, Y, Mask

class IsingDS(Dataset):
    def __init__(self, X, S, T, P, Y, M):
        self.X = torch.from_numpy(X)                   # (N,2,L,L)
        self.S = torch.from_numpy(S)[:,None,...]       # (N,1,L,L)
        self.T = torch.from_numpy(T)[:,None]           # (N,1)
        self.P = torch.from_numpy(P)                   # (N,)
        self.Y = torch.from_numpy(Y)[:,None,...]       # (N,1,L,L)
        self.M = torch.from_numpy(M)[:,None,...]       # (N,1,L,L)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self,i):
        return {"X":self.X[i], "S":self.S[i], "T":self.T[i], "P":self.P[i], "Y":self.Y[i], "M":self.M[i]}

def make_loaders(X,S,T,P,Y,M):
    ds = IsingDS(X,S,T,P,Y,M)
    N = len(ds)
    n_train = int(0.8*N); n_val = int(0.1*N); n_test = N - n_train - n_val
    g = torch.Generator().manual_seed(SEED)
    tr, va, te = random_split(ds, [n_train,n_val,n_test], generator=g)
    return (
        DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0),
        DataLoader(va, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
        DataLoader(te, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
        te
    )

# =========================
# Physics helpers (diff-friendly)
# =========================
class PeriodicConv(nn.Module):
    def __init__(self, kernel_3x3):
        super().__init__()
        k = torch.tensor(kernel_3x3, dtype=torch.float32).view(1,1,3,3)
        self.register_buffer("weight", k)
    def forward(self,x):
        x = F.pad(x, (1,1,1,1), mode="circular")
        return F.conv2d(x, self.weight, bias=None, stride=1, padding=0)

_NN_K = [[0,1,0],[1,0,1],[0,1,0]]
_nn_conv = PeriodicConv(_NN_K)

def energy_per_site(s):   # (N,1,L,L)
    nbr = _nn_conv(s)
    return - (s*nbr).mean(dim=(1,2,3)) / 2.0

def nn_corr(s):
    nbr = _nn_conv(s)
    return (s*nbr).mean(dim=(1,2,3)) / 4.0

def mag_abs(s):
    return s.mean(dim=(2,3)).abs().squeeze(1)

def cv_derivative(T_centers, E_means):
    # simple central-difference Cv ~ d<E>/dT on binned means (robust, minimal deps)
    T = np.asarray(T_centers, dtype=float)
    E = np.asarray(E_means, dtype=float)
    if len(T) < 2:
        return np.zeros_like(T)
    order = np.argsort(T)
    T = T[order]; E = E[order]
    Cv = np.zeros_like(T)
    Cv[0]  = (E[1]-E[0])/(T[1]-T[0]+1e-12)
    Cv[-1] = (E[-1]-E[-2])/(T[-1]-T[-2]+1e-12)
    for i in range(1,len(T)-1):
        Cv[i] = (E[i+1]-E[i-1])/(T[i+1]-T[i-1]+1e-12)
    return np.maximum(Cv,0.0)

# =========================
# Parametric model (depth, channels)
# =========================
class InvNet(nn.Module):
    def __init__(self, C=48, depth=4):
        super().__init__()
        layers = [nn.Conv2d(2, C, 3, padding=1, padding_mode="circular"), nn.ReLU(inplace=True)]
        for _ in range(depth-1):
            layers += [nn.Conv2d(C, C, 3, padding=1, padding_mode="circular"), nn.ReLU(inplace=True)]
        self.enc = nn.Sequential(*layers)

        self.dec = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1, padding_mode="circular"),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Tanh()
        )

        self.head_T = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                    nn.Linear(C, 64), nn.ReLU(inplace=True),
                                    nn.Linear(64, 1))
        self.head_P = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                    nn.Linear(C, 64), nn.ReLU(inplace=True),
                                    nn.Linear(64, 2))
    def forward(self,x):
        h = self.enc(x)
        return self.dec(h), self.head_T(h), self.head_P(h)

# =========================
# Train + eval for one config
# =========================
def run_one(config, train_loader, val_loader, test_loader, device):
    # unpack
    LR = config["LR"]
    C  = config["C"]
    DEPTH = config["DEPTH"]
    LAM_REC = config["LAM_REC"]
    LAM_E   = config["LAM_E"]
    LAM_M   = config["LAM_M"]
    LAM_C1  = config["LAM_C1"]
    LAM_T   = config["LAM_T"]
    LAM_PH  = config["LAM_PHASE"]

    net = InvNet(C=C, depth=DEPTH).to(device)
    # move conv-kernel module to device
    global _nn_conv
    _nn_conv = _nn_conv.to(device)

    opt = torch.optim.Adam(net.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()

    best_val = float("inf")
    patience = 0

    t0 = time.time()
    for ep in range(1, EPOCHS+1):
        net.train()
        for b in train_loader:
            X = b["X"].to(device)
            S = b["S"].to(device)
            Tt= b["T"].to(device)
            P = b["P"].to(device)
            Y = b["Y"].to(device)
            M = b["M"].to(device)

            S_hat, T_hat, P_log = net(X)

            rec = (M * (S_hat - Y).abs()).sum() / (M.sum() + 1e-8)
            LT  = (T_hat - Tt).abs().mean()
            LPh = ce(P_log, P)

            # physics losses (on sign-discretized spins)
            S_sign = torch.sign(S_hat)
            LE = (energy_per_site(S_sign) - energy_per_site(S)).abs().mean()
            LM = (mag_abs(S_sign) - mag_abs(S)).abs().mean()
            LC = (nn_corr(S_sign) - nn_corr(S)).abs().mean()

            loss = (LAM_REC*rec + LAM_T*LT + LAM_PH*LPh + LAM_E*LE + LAM_M*LM + LAM_C1*LC)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # val
        net.eval()
        vloss = 0.0
        with torch.no_grad():
            for b in val_loader:
                X = b["X"].to(device)
                S = b["S"].to(device)
                Tt= b["T"].to(device)
                P = b["P"].to(device)
                Y = b["Y"].to(device)
                M = b["M"].to(device)

                S_hat, T_hat, P_log = net(X)
                rec = (M * (S_hat - Y).abs()).sum() / (M.sum() + 1e-8)
                LT  = (T_hat - Tt).abs().mean()
                LPh = ce(P_log, P)
                S_sign = torch.sign(S_hat)
                LE = (energy_per_site(S_sign) - energy_per_site(S)).abs().mean()
                LM = (mag_abs(S_sign) - mag_abs(S)).abs().mean()
                LC = (nn_corr(S_sign) - nn_corr(S)).abs().mean()
                vloss += float((LAM_REC*rec + LAM_T*LT + LAM_PH*LPh + LAM_E*LE + LAM_M*LM + LAM_C1*LC).item())

        vloss /= max(1, len(val_loader))
        if vloss < best_val:
            best_val = vloss
            best_state = {k:v.detach().cpu() for k,v in net.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    # load best
    net.load_state_dict(best_state)
    net.eval()

    # test metrics + per-bin physics
    all_Tt, all_Tp, all_Pt, all_Pp = [], [], [], []
    all_Strue, all_Spred, all_mask = [], [], []
    all_Et, all_Ep, all_Mt, all_Mp, all_Ct, all_Cp = [], [], [], [], [], []

    with torch.no_grad():
        for b in test_loader:
            X = b["X"].to(device)
            S = b["S"].to(device)
            Tt= b["T"].to(device)
            P = b["P"].to(device)
            M = b["M"].to(device)

            S_hat, T_hat, P_log = net(X)
            S_sign = torch.sign(S_hat)

            all_Tt.append(Tt.cpu().numpy()); all_Tp.append(T_hat.cpu().numpy())
            all_Pt.append(P.cpu().numpy());  all_Pp.append(P_log.argmax(1).cpu().numpy())
            all_Strue.append(S.cpu().numpy()); all_Spred.append(S_sign.cpu().numpy()); all_mask.append(M.cpu().numpy())

            all_Et.append(energy_per_site(S).cpu().numpy());     all_Ep.append(energy_per_site(S_sign).cpu().numpy())
            all_Mt.append(mag_abs(S).cpu().numpy());            all_Mp.append(mag_abs(S_sign).cpu().numpy())
            all_Ct.append(nn_corr(S).cpu().numpy());            all_Cp.append(nn_corr(S_sign).cpu().numpy())

    T_true = np.concatenate(all_Tt).reshape(-1)
    T_pred = np.concatenate(all_Tp).reshape(-1)
    P_true = np.concatenate(all_Pt).reshape(-1)
    P_pred = np.concatenate(all_Pp).reshape(-1)

    S_true = np.concatenate(all_Strue)  # (N,1,L,L)
    S_pred = np.concatenate(all_Spred)
    Mask   = np.concatenate(all_mask)
    miss = (Mask == 0)

    mae_T = float(np.mean(np.abs(T_true - T_pred)))
    acc_P = float(np.mean(P_true == P_pred))
    imp_acc = float(((S_pred == S_true) & miss).sum() / max(1, miss.sum()))

    # Per-temp-bin physics (means)
    bins = np.linspace(1.0, 4.0, N_BINS+1)
    bin_ids = np.digitize(T_true, bins) - 1
    centers = 0.5*(bins[:-1]+bins[1:])

    Et = np.concatenate(all_Et).reshape(-1)
    Ep = np.concatenate(all_Ep).reshape(-1)
    Mt = np.concatenate(all_Mt).reshape(-1)
    Mp = np.concatenate(all_Mp).reshape(-1)
    Ct = np.concatenate(all_Ct).reshape(-1)
    Cp = np.concatenate(all_Cp).reshape(-1)

    # compute Cv from derivative of binned <E>(T)
    T_cent_used, Et_m, Ep_m = [], [], []
    for b in range(N_BINS):
        idx = (bin_ids == b)
        if np.any(idx):
            T_cent_used.append(float(centers[b]))
            Et_m.append(float(Et[idx].mean()))
            Ep_m.append(float(Ep[idx].mean()))
    Cv_t = cv_derivative(T_cent_used, Et_m)
    Cv_p = cv_derivative(T_cent_used, Ep_m)
    cv_map = {T_cent_used[i]:(Cv_t[i], Cv_p[i]) for i in range(len(T_cent_used))}

    # summary physics errors (single numbers): mean absolute deviation over bins
    rows = []
    for b in range(N_BINS):
        idx = (bin_ids == b)
        if not np.any(idx):
            continue
        Tbar = float(centers[b])
        cvt, cvp = cv_map.get(Tbar,(0.0,0.0))
        rows.append([
            Tbar,
            float((Et[idx].mean() - Ep[idx].mean())),
            float((Mt[idx].mean() - Mp[idx].mean())),
            float((Ct[idx].mean() - Cp[idx].mean())),
            float((cvt - cvp))
        ])
    rows = np.array(rows) if len(rows) else np.zeros((0,5))
    if rows.shape[0] > 0:
        E_err  = float(np.mean(np.abs(rows[:,1])))
        M_err  = float(np.mean(np.abs(rows[:,2])))
        C1_err = float(np.mean(np.abs(rows[:,3])))
        Cv_err = float(np.mean(np.abs(rows[:,4])))
    else:
        E_err = M_err = C1_err = Cv_err = np.nan

    elapsed = time.time() - t0

    out = dict(config)
    out.update({
        "MAE_T": mae_T,
        "Acc_phi": acc_P,
        "ImpAcc": imp_acc,
        "E_err": E_err,
        "Mabs_err": M_err,
        "C1_err": C1_err,
        "Cv_err": Cv_err,
        "BestVal": float(best_val),
        "Time_s": float(elapsed),
    })
    return out

# =========================
# Define sensitivity sweeps (small, minimal)
# =========================
# Baseline (from your optimized defaults; physics off)
BASE = dict(LAM_REC=1.0, LAM_E=0.0, LAM_M=0.0, LAM_C1=0.0, LAM_T=1.0, LAM_PHASE=1.0,
            DEPTH=4, C=48, LR=5e-4)

# (1) Loss-weight sweep (keep small)
LOSS_SWEEP = [
    dict(LAM_E=0.0, LAM_M=0.0, LAM_C1=0.0),
    dict(LAM_E=0.1, LAM_M=0.0, LAM_C1=0.0),
    dict(LAM_E=0.0, LAM_M=0.1, LAM_C1=0.0),
    dict(LAM_E=0.0, LAM_M=0.0, LAM_C1=0.1),
    dict(LAM_E=0.1, LAM_M=0.1, LAM_C1=0.1),
]

# (2) Architecture sweep
ARCH_SWEEP = [
    dict(DEPTH=3, C=32),
    dict(DEPTH=4, C=48),
    dict(DEPTH=5, C=64),
]

# (3) Learning rate sweep
LR_SWEEP = [
    dict(LR=1e-4),
    dict(LR=5e-4),
    dict(LR=1e-3),
]

# =========================
# Run
# =========================
assert os.path.exists(CLEAN_CSV), f"Missing {CLEAN_CSV}"
assert os.path.exists(NOISY_CSV), f"Missing {NOISY_CSV}"

X,S,T,P,Y,M = load_csvs(CLEAN_CSV, NOISY_CSV, L)
train_loader, val_loader, test_loader, _ = make_loaders(X,S,T,P,Y,M)

device = torch.device(DEVICE)

results = []

def merge_cfg(base, patch, tag):
    c = dict(base)
    c.update(patch)
    c["Study"] = tag
    # ensure all keys exist
    for k in ["LAM_REC","LAM_E","LAM_M","LAM_C1","LAM_T","LAM_PHASE","DEPTH","C","LR"]:
        if k not in c: c[k] = BASE[k]
    return c

# 1) Loss weights
for patch in LOSS_SWEEP:
    cfg = merge_cfg(BASE, patch, "LossWeights")
    results.append(run_one(cfg, train_loader, val_loader, test_loader, device))

# 2) Architecture
for patch in ARCH_SWEEP:
    cfg = merge_cfg(BASE, patch, "Arch")
    results.append(run_one(cfg, train_loader, val_loader, test_loader, device))

# 3) Learning rate
for patch in LR_SWEEP:
    cfg = merge_cfg(BASE, patch, "LR")
    results.append(run_one(cfg, train_loader, val_loader, test_loader, device))

df = pd.DataFrame(results)
csv_path = os.path.join(OUT_DIR, f"sensitivity_summary_L{L}.csv")
df.to_csv(csv_path, index=False)

# =========================
# PDF plots (SEPARATE FILES)
# =========================
def plot_study(study_name, xlabels, subdf, out_pdf_path):
    x = np.arange(len(subdf))

    with PdfPages(out_pdf_path) as pdf:
        # Page 1: MAE_T, Acc_phi, ImpAcc
        fig = plt.figure(figsize=(10.5, 6.0))
        plt.plot(x, subdf["MAE_T"], marker="o", label="MAE_T")
        plt.plot(x, subdf["ImpAcc"], marker="s", label="ImpAcc")
        plt.plot(x, subdf["Acc_phi"], marker="^", label="Acc_phi")
        plt.xticks(x, xlabels, rotation=30, ha="right")
        plt.xlabel("Setting", fontweight="bold")
        plt.ylabel("Metric value", fontweight="bold")
        plt.title(f"Sensitivity: {study_name} (L={L})", fontweight="bold")
        plt.legend()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Physics error summary (E, |M|, C1, Cv)
        fig = plt.figure(figsize=(10.5, 6.0))
        plt.plot(x, subdf["E_err"], marker="o", label="E (bin-avg)")
        plt.plot(x, subdf["Mabs_err"], marker="s", label="M")
        plt.plot(x, subdf["C1_err"], marker="^", label="C1")
        plt.plot(x, subdf["Cv_err"], marker="d", label="Cv")
        plt.xticks(x, xlabels, rotation=30, ha="right")
        plt.xlabel("Setting", fontweight="bold")
        plt.ylabel("Abs. error", fontweight="bold")
        plt.title(f"Physics observable errors: {study_name} (L={L})", fontweight="bold")
        plt.legend()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

# Loss weights -> separate pdf
d1 = df[df["Study"]=="LossWeights"].reset_index(drop=True)
lab1 = [f"E{r.LAM_E}-M{r.LAM_M}-C1{r.LAM_C1}" for r in d1.itertuples()]
pdf_loss = os.path.join(OUT_DIR, f"sensitivity_LossWeights_L{L}.pdf")
plot_study("Loss weights (E, M, C1)", lab1, d1, pdf_loss)

# Architecture -> separate pdf
d2 = df[df["Study"]=="Arch"].reset_index(drop=True)
lab2 = [f"depth{r.DEPTH}-C{r.C}" for r in d2.itertuples()]
pdf_arch = os.path.join(OUT_DIR, f"sensitivity_Arch_L{L}.pdf")
plot_study("Architecture (depth, channels)", lab2, d2, pdf_arch)

# Learning rate -> separate pdf
d3 = df[df["Study"]=="LR"].reset_index(drop=True)
lab3 = [f"LR={r.LR:.0e}" for r in d3.itertuples()]
pdf_lr = os.path.join(OUT_DIR, f"sensitivity_LR_L{L}.pdf")
plot_study("Learning rate", lab3, d3, pdf_lr)

print("Saved:")
print(" -", csv_path)
print(" -", pdf_loss)
print(" -", pdf_arch)
print(" -", pdf_lr)
