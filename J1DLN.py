# -*- coding: utf-8 -*-

#Program 2 (OPTIMIZED): Physics-informed deep model for inverse inference on MCD{L}.csv (clean) + MCDN{L}.csv (noisy).
#- OPTIMIZED based on sensitivity analysis: No physics regularization, better LR, early stopping
#- Loads CSVs (Temperature, Phase, Spin1..Spin{L*L})
#- Builds tensors: X=[Y,Mask], targets=(Sigma, T, Phase)
#- Trains CNN with physics losses: Energy, |M|, C1 + masked rec + T and Phase losses
#- Saves CSVs + THREE PDFs in black & white with bold, larger fonts:
#
#  1) physics_plots.pdf
#     - Page 1: |M|(T), E(T), C1(T), C_v(T) in one 2×2 frame
#     - Page 2: Energy comparison scatter (Clean vs Recon, Clean vs Noisy)
#
#  2) training_plots.pdf
#     - Training & validation loss
#
#  3) configs_bw.pdf
#     - Single page, 4 rows (T˜1,2,Tc,4) × 3 columns (Clean | Noisy | Reconstructed)
#
#Also writes:
#    metrics_global.csv
#    metrics_per_temp_bin.csv
#    preds_test.csv
#    training_times.csv


# ==========================
# OPTIMIZED USER PARAMETERS
# ==========================
L = 128
Tc = 2.269

CLEAN_CSV = "../JOB5_Noise/J5Data/MCD128.csv"
NOISY_CSV = "../JOB5_Noise/J5Data/MCDN128.csv"

OUT_DIR   = "./Outputs_L128" # where to save pdf/csvs
EPOCHS    = 50                  # Increased for better convergence
BATCH_SIZE = 32                # Reduced for CPU memory
LR         = 5e-4               # OPTIMIZED from sensitivity analysis
DEVICE     = "cpu"              # Changed to CPU since cluster has no GPU

# OPTIMIZED Loss weights (No physics regularization - best from sensitivity analysis)
LAM_REC   = 1.0
LAM_E     = 0.0                 # No physics - best performing
LAM_M     = 0.0                 # No physics - best performing  
LAM_C1    = 0.0                 # No physics - best performing
LAM_T     = 1.0
LAM_PHASE = 1.0

# Temperature binning for evaluation/plots
N_BINS    = 16  # bins across [1,4]

# Early stopping
PATIENCE = 5

# Random seed
SEED = 123

# ==========================
# Imports
# ==========================
import os
import time
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

# Start timing
start_time = time.time()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED)

print(f"=== Physics-Informed Ising Model Reconstruction (OPTIMIZED) ===")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Output directory: {OUT_DIR}")

# ==========================
# Data loading & tensors
# ==========================
def load_csvs(clean_path, noisy_path, L):
    df_clean = pd.read_csv(clean_path)
    df_noisy = pd.read_csv(noisy_path)
    assert len(df_clean) == len(df_noisy), "Clean & noisy must have same number of rows"
    
    # Get spin column names - they are spin_0, spin_1, ..., spin_{L*L-1}
    spin_cols = [f"spin_{i}" for i in range(L*L)]
    
    print(f"Looking for spin columns: first few = {spin_cols[:5]}")
    print(f"Available columns in clean CSV: {df_clean.columns.tolist()[:10]}...")
    print(f"Available columns in noisy CSV: {df_noisy.columns.tolist()[:10]}...")
    
    # Check if all spin columns exist
    missing_in_clean = [col for col in spin_cols if col not in df_clean.columns]
    missing_in_noisy = [col for col in spin_cols if col not in df_noisy.columns]
    
    if missing_in_clean:
        print(f"WARNING: Missing columns in clean CSV: {missing_in_clean[:5]}...")
    if missing_in_noisy:
        print(f"WARNING: Missing columns in noisy CSV: {missing_in_noisy[:5]}...")
    
    # Temperature
    T = df_clean["Temperature"].to_numpy(dtype=np.float32)
    
    # Phase - handle both integer and string 'F'/'P' values
    Phase_raw = df_clean["Phase"]
    
    # Check if Phase contains string 'F'/'P' and convert to numeric
    if Phase_raw.dtype == object or Phase_raw.dtype == 'string':
        # Create a mapping: 'F' -> 0 (ferromagnetic), 'P' -> 1 (paramagnetic)
        Phase = np.zeros(len(Phase_raw), dtype=np.int64)
        for i, val in enumerate(Phase_raw):
            if isinstance(val, str):
                if val.upper() == 'F':
                    Phase[i] = 0
                elif val.upper() == 'P':
                    Phase[i] = 1
                else:
                    # Try to convert to integer
                    try:
                        Phase[i] = int(float(val)) if pd.notna(val) else 1
                    except (ValueError, TypeError):
                        Phase[i] = 1  # default to paramagnetic
            else:
                # If it's already numeric
                Phase[i] = int(val) if pd.notna(val) else 1
    else:
        Phase = Phase_raw.astype(np.int64).to_numpy()
    
    # Extract spin data - reshaping to (N, L, L)
    Sigma = df_clean[spin_cols].to_numpy(dtype=np.int8).reshape(-1, L, L)
    Y     = df_noisy[spin_cols].to_numpy(dtype=np.int8).reshape(-1, L, L)
    Mask  = (Y != 0).astype(np.float32)
    
    # Model inputs
    X = np.stack([Y.astype(np.float32), Mask], axis=1)  # (N,2,L,L)
    Sigma = Sigma.astype(np.float32)                    # (N,L,L)
    
    print(f"Phase distribution: {np.unique(Phase, return_counts=True)}")
    print(f"Spin data shapes: Sigma={Sigma.shape}, Y={Y.shape}, Mask={Mask.shape}")
    
    return X, Sigma, T, Phase, Y.astype(np.float32), Mask

data_load_start = time.time()
X, Sigma, T, Phase, Y, Mask = load_csvs(CLEAN_CSV, NOISY_CSV, L)
N = X.shape[0]
data_load_time = time.time() - data_load_start
print(f"Loaded N={N} samples. Example: X={X.shape}, Sigma={Sigma.shape}, T={T.shape}")
print(f"Data loading time: {data_load_time:.2f}s")

class IsingCSVDataset(Dataset):
    def __init__(self, X, Sigma, T, Phase, Y, Mask):
        self.X = torch.from_numpy(X)                    # (N,2,L,L)
        self.S = torch.from_numpy(Sigma)[:, None, ...]  # (N,1,L,L)
        self.T = torch.from_numpy(T)[:, None]           # (N,1)
        self.P = torch.from_numpy(Phase)                # (N,)
        self.Y = torch.from_numpy(Y)[:, None, ...]      # (N,1,L,L)
        self.M = torch.from_numpy(Mask)[:, None, ...]   # (N,1,L,L)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return {"X": self.X[i], "S": self.S[i], "T": self.T[i], "P": self.P[i], "Y": self.Y[i], "M": self.M[i]}

full_ds = IsingCSVDataset(X, Sigma, T, Phase, Y, Mask)

# Split (80/10/10)
n_train = int(0.8 * N)
n_val   = int(0.1 * N)
n_test  = N - n_train - n_val
g = torch.Generator().manual_seed(SEED)
train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test], generator=g)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ==========================
# Device selection
# ==========================
device = torch.device(DEVICE)
print("Using device:", device)

# ==========================
# Physics helpers (diff-friendly) — FIXED
# ==========================
class PeriodicConv(nn.Module):
    """3x3 conv with circular padding via F.pad; kernel is a non-trainable buffer."""
    def __init__(self, kernel_3x3):
        super().__init__()
        k = torch.tensor(kernel_3x3, dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("weight", k)  # moves with .to(device)
    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode="circular")     # (left,right,top,bottom)
        return F.conv2d(x, self.weight, bias=None, stride=1, padding=0)

_NN_K = [[0,1,0],[1,0,1],[0,1,0]]
_nn_conv = PeriodicConv(_NN_K).to(device)              # kernel on same device

def energy_per_site(s):  # s: (N,1,L,L)
    nbr = _nn_conv(s)
    return - (s * nbr).mean(dim=(1,2,3)) / 2.0

def nn_corr(s):
    nbr = _nn_conv(s)
    return (s * nbr).mean(dim=(1,2,3)) / 4.0

def mag_abs(s):
    return s.mean(dim=(2,3)).abs().squeeze(1)

# ==========================
# Model
# ==========================
class InvNet(nn.Module):
    def __init__(self, L):
        super().__init__()
        C = 48
        # Slightly deeper architecture
        self.enc = nn.Sequential(
            nn.Conv2d(2, C, 3, padding=1, padding_mode="circular"),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, padding=1, padding_mode="circular"),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, padding=1, padding_mode="circular"),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, padding=1, padding_mode="circular"),  # Additional layer
            nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1, padding_mode="circular"),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Tanh(),  # S^ in [-1,1]
        )
        self.head_T = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(C, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        self.head_P = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(C, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        h = self.enc(x)
        S_hat = self.dec(h)
        T_hat = self.head_T(h)
        P_log = self.head_P(h)
        return S_hat, T_hat, P_log

model_init_start = time.time()
net = InvNet(L).to(device)
opt = torch.optim.Adam(net.parameters(), lr=LR)
# Add learning rate scheduler (removed verbose parameter)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
ce  = nn.CrossEntropyLoss()
model_init_time = time.time() - model_init_start

print(f"Model initialized in {model_init_time:.2f}s")

# ==========================
# Training with Early Stopping
# ==========================
train_hist, val_hist = [], []
best_val_loss = float('inf')
patience_counter = 0
training_start = time.time()

print("\nStarting training...")
for epoch in range(1, EPOCHS+1):
    epoch_start = time.time()
    net.train()
    epoch_loss = 0.0
    
    for batch in train_loader:
        X = batch["X"].to(device, non_blocking=True)
        S = batch["S"].to(device, non_blocking=True)
        Tt= batch["T"].to(device, non_blocking=True)
        P = batch["P"].to(device, non_blocking=True)
        Yb= batch["Y"].to(device, non_blocking=True)
        M = batch["M"].to(device, non_blocking=True)

        S_hat, T_hat, P_log = net(X)

        rec  = (M * (S_hat - Yb).abs()).sum() / (M.sum() + 1e-8)
        L_T  = (T_hat - Tt).abs().mean()
        L_Ph = ce(P_log, P)

        # Only reconstruction, temperature and phase losses (no physics)
        loss = (LAM_REC * rec + LAM_T * L_T + LAM_PHASE * L_Ph)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        epoch_loss += float(loss.item())

    epoch_loss /= max(1, len(train_loader))
    train_hist.append(epoch_loss)

    # Validation
    net.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            X = batch["X"].to(device)
            S = batch["S"].to(device)
            Tt= batch["T"].to(device)
            P = batch["P"].to(device)
            Yb= batch["Y"].to(device)
            M = batch["M"].to(device)

            S_hat, T_hat, P_log = net(X)
            rec  = (M * (S_hat - Yb).abs()).sum() / (M.sum() + 1e-8)
            L_T  = (T_hat - Tt).abs().mean()
            L_Ph = ce(P_log, P)

            val_loss += float((LAM_REC * rec + LAM_T * L_T + LAM_PHASE * L_Ph).item())
    
    val_loss /= max(1, len(val_loader))
    val_hist.append(val_loss)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    epoch_time = time.time() - epoch_start
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'val_loss': best_val_loss,
        }, os.path.join(OUT_DIR, "best_model.pth"))
    else:
        patience_counter += 1
    
    # Get current learning rate
    current_lr = opt.param_groups[0]['lr']
    print(f"Epoch {epoch:03d} | train {epoch_loss:.4f} | val {val_loss:.4f} | time {epoch_time:.1f}s | LR {current_lr:.2e}")
    
    if patience_counter >= PATIENCE:
        print(f"Early stopping triggered at epoch {epoch}")
        break

training_time = time.time() - training_start
print(f"Training completed in {training_time:.2f}s")

# Load best model for evaluation
checkpoint = torch.load(os.path.join(OUT_DIR, "best_model.pth"), map_location=device)
net.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best model from epoch {checkpoint['epoch']} with val_loss {checkpoint['val_loss']:.4f}")

# ==========================
# Test-time evaluation & metrics
# ==========================
eval_start = time.time()
net.eval()
all_true_T, all_pred_T, all_true_P, all_pred_P = [], [], [], []
all_true_E, all_pred_E = [], []
all_true_M, all_pred_M = [], []
all_true_C1, all_pred_C1 = [], []
all_pred_S_sign, all_true_S, all_noisy_Y, all_mask = [], [], [], []

with torch.no_grad():
    for batch in test_loader:
        X = batch["X"].to(device)
        S = batch["S"].to(device)
        Tt= batch["T"].to(device)
        P = batch["P"].to(device)
        Yb= batch["Y"].to(device)
        M = batch["M"].to(device)

        S_hat, T_hat, P_log = net(X)
        S_sign = torch.sign(S_hat)  # discretize for physics metrics

        all_true_T.append(Tt.cpu().numpy())
        all_pred_T.append(T_hat.cpu().numpy())
        all_true_P.append(P.cpu().numpy())
        all_pred_P.append(P_log.argmax(1).cpu().numpy())

        all_true_E.append(energy_per_site(S).cpu().numpy())
        all_pred_E.append(energy_per_site(S_sign).cpu().numpy())

        all_true_M.append(mag_abs(S).cpu().numpy())
        all_pred_M.append(mag_abs(S_sign).cpu().numpy())

        all_true_C1.append(nn_corr(S).cpu().numpy())
        all_pred_C1.append(nn_corr(S_sign).cpu().numpy())

        all_pred_S_sign.append(S_sign.cpu().numpy())
        all_true_S.append(S.cpu().numpy())
        all_noisy_Y.append(Yb.cpu().numpy())
        all_mask.append(M.cpu().numpy())

# stack
T_true  = np.concatenate(all_true_T, axis=0).reshape(-1)
T_pred  = np.concatenate(all_pred_T, axis=0).reshape(-1)
P_true  = np.concatenate(all_true_P, axis=0).reshape(-1)
P_pred  = np.concatenate(all_pred_P, axis=0).reshape(-1)

E_true  = np.concatenate(all_true_E, axis=0).reshape(-1)
E_pred  = np.concatenate(all_pred_E, axis=0).reshape(-1)
M_true  = np.concatenate(all_true_M, axis=0).reshape(-1)
M_pred  = np.concatenate(all_pred_M, axis=0).reshape(-1)
C_true  = np.concatenate(all_true_C1, axis=0).reshape(-1)
C_pred  = np.concatenate(all_pred_C1, axis=0).reshape(-1)

S_pred  = np.concatenate(all_pred_S_sign, axis=0)  # (N,1,L,L)
S_true  = np.concatenate(all_true_S, axis=0)
Y_test  = np.concatenate(all_noisy_Y, axis=0)
M_test  = np.concatenate(all_mask, axis=0)

eval_time = time.time() - eval_start

# Global metrics
mae_T   = float(np.mean(np.abs(T_true - T_pred)))
acc_P   = float(np.mean(P_true == P_pred))
miss    = (M_test == 0)
imp_acc = float(((S_pred == S_true) & miss).sum() / max(1, miss.sum()))

pd.DataFrame([{
    "Temperature_MAE": mae_T,
    "Phase_Accuracy": acc_P,
    "Imputation_Accuracy_on_missing": imp_acc,
    "N_test": int(T_true.shape[0]),
    "Final_Val_Loss": float(best_val_loss),
}]).to_csv(os.path.join(OUT_DIR, "metrics_global.csv"), index=False)

print("\n=== Test (global) ===")
print(pd.read_csv(os.path.join(OUT_DIR, "metrics_global.csv")).to_string(index=False))

# Per-sample preds
pd.DataFrame({
    "T_true": T_true, "T_pred": T_pred,
    "Phase_true": P_true, "Phase_pred": P_pred,
    "E_true": E_true, "E_pred": E_pred,
    "|M|_true": M_true, "|M|_pred": M_pred,
    "C1_true": C_true, "C1_pred": C_pred,
}).to_csv(os.path.join(OUT_DIR, "preds_test.csv"), index=False)


# ==========================
# Per-temperature stats with proper Heat capacity (ENHANCED)
# ==========================

def calculate_heat_capacity_from_derivative(temps, energies):
    """
    Calculate heat capacity as Cv = d<E>/dT using smoothed finite differences.
    More robust than fluctuation method for reconstructed data.
    """
    if len(temps) < 2:
        return np.zeros_like(temps)
    
    # Sort by temperature
    sorted_idx = np.argsort(temps)
    temps_sorted = temps[sorted_idx]
    energies_sorted = energies[sorted_idx]
    
    # Apply smoothing to reduce noise
    from scipy.ndimage import gaussian_filter1d
    if len(temps_sorted) > 5:
        energies_smoothed = gaussian_filter1d(energies_sorted, sigma=1.0)
    else:
        energies_smoothed = energies_sorted
    
    cv = np.zeros_like(temps_sorted)
    
    # Use central difference for interior points
    for i in range(1, len(temps_sorted)-1):
        dE = energies_smoothed[i+1] - energies_smoothed[i-1]
        dT = temps_sorted[i+1] - temps_sorted[i-1]
        if abs(dT) > 1e-8:  # Avoid division by zero
            cv[i] = dE / dT
    
    # Handle boundaries with forward/backward difference
    if len(temps_sorted) >= 2:
        # First point: forward difference
        dE = energies_smoothed[1] - energies_smoothed[0]
        dT = temps_sorted[1] - temps_sorted[0]
        if abs(dT) > 1e-8:
            cv[0] = dE / dT
        
        # Last point: backward difference
        dE = energies_smoothed[-1] - energies_smoothed[-2]
        dT = temps_sorted[-1] - temps_sorted[-2]
        if abs(dT) > 1e-8:
            cv[-1] = dE / dT
    
    return cv

# Get temperature bins (from your simplified approach)
bins = np.linspace(1.0, 4.0, N_BINS+1)
bin_ids = np.digitize(T_true, bins) - 1
centers = 0.5*(bins[:-1]+bins[1:])

# First, calculate means per bin for derivative method
T_centers = []
E_true_means, E_pred_means = [], []
M_true_means, M_pred_means = [], []
C1_true_means, C1_pred_means = [], []

for b in range(N_BINS):
    idx = (bin_ids == b)
    if not np.any(idx):
        continue
    
    T_centers.append(float(centers[b]))
    E_true_means.append(float(np.mean(E_true[idx])))
    E_pred_means.append(float(np.mean(E_pred[idx])))
    M_true_means.append(float(np.mean(M_true[idx])))
    M_pred_means.append(float(np.mean(M_pred[idx])))
    C1_true_means.append(float(np.mean(C_true[idx])))
    C1_pred_means.append(float(np.mean(C_pred[idx])))

# Convert to arrays for derivative calculation
T_centers = np.array(T_centers)
E_true_means = np.array(E_true_means)
E_pred_means = np.array(E_pred_means)
M_true_means = np.array(M_true_means)
M_pred_means = np.array(M_pred_means)
C1_true_means = np.array(C1_true_means)
C1_pred_means = np.array(C1_pred_means)

# Calculate heat capacity using derivative method
Cv_true_deriv = calculate_heat_capacity_from_derivative(T_centers, E_true_means)
Cv_pred_deriv = calculate_heat_capacity_from_derivative(T_centers, E_pred_means)

# Second pass: compile all statistics with both methods
rows = []
for b in range(N_BINS):
    idx = (bin_ids == b)
    if not np.any(idx):
        rows.append({
            "T_center": centers[b], "count": 0,
            "E_true": np.nan, "E_pred": np.nan,
            "Mabs_true": np.nan, "Mabs_pred": np.nan,
            "C1_true": np.nan, "C1_pred": np.nan,
            "Cv_true": np.nan, "Cv_pred": np.nan,
            "Cv_true_fluct": np.nan, "Cv_pred_fluct": np.nan,
            "Accuracy_missing": np.nan,
            "Accuracy_observed": np.nan,
            "Accuracy_overall": np.nan,
        })
        continue
    
    Tbar = centers[b]
    count = int(idx.sum())
    
    # Get the index in T_centers array
    t_idx = np.where(np.isclose(T_centers, Tbar, rtol=1e-08))[0]
    if len(t_idx) == 0:
        cv_true_deriv = 0.0
        cv_pred_deriv = 0.0
    else:
        cv_true_deriv = Cv_true_deriv[t_idx[0]]
        cv_pred_deriv = Cv_pred_deriv[t_idx[0]]
    
    # Accuracy calculations
    miss_bin = miss[idx]
    S_pred_bin = S_pred[idx]
    S_true_bin = S_true[idx]
    
    bin_missing_acc = float(((S_pred_bin == S_true_bin) & miss_bin).sum() /
                             max(1, miss_bin.sum()))
    bin_obs_acc = float(((S_pred_bin == S_true_bin) & (~miss_bin)).sum() /
                         max(1, (~miss_bin).sum()))
    bin_overall_acc = float((S_pred_bin == S_true_bin).sum() /
                             max(1, S_pred_bin.size))
    
    # Also compute fluctuation method for comparison
    # Sample up to 500 configurations per bin for stable statistics
    n_samples = min(500, np.sum(idx))
    sample_indices = np.random.choice(np.where(idx)[0], size=n_samples, replace=False)
    
    # Compute energies for sampled configurations
    with torch.no_grad():
        # True energies
        true_spins = torch.from_numpy(S_true[sample_indices]).to(device)
        true_energies = energy_per_site(true_spins).cpu().numpy()
        
        # Predicted energies  
        pred_spins = torch.from_numpy(S_pred[sample_indices]).to(device)
        pred_energies = energy_per_site(pred_spins).cpu().numpy()
    
    # Heat capacity per site using fluctuation method
    if Tbar > 1e-8 and len(true_energies) > 1:
        cv_true_fluct = (np.mean(true_energies**2) - np.mean(true_energies)**2) / (Tbar**2 + 1e-8)
        cv_pred_fluct = (np.mean(pred_energies**2) - np.mean(pred_energies)**2) / (Tbar**2 + 1e-8)
    else:
        cv_true_fluct = 0.0
        cv_pred_fluct = 0.0
    
    # Choose which Cv method to use (derivative is more robust for reconstruction)
    use_derivative_method = True  # Set to False to use fluctuation method
    
    if use_derivative_method:
        cv_true_val = max(0.0, cv_true_deriv)  # Clip negative values
        cv_pred_val = max(0.0, cv_pred_deriv)
    else:
        cv_true_val = max(0.0, cv_true_fluct)
        cv_pred_val = max(0.0, cv_pred_fluct)
    
    rows.append({
        "T_center": float(Tbar),
        "count": count,
        "E_true": float(np.mean(E_true[idx])),
        "E_pred": float(np.mean(E_pred[idx])),
        "Mabs_true": float(np.mean(M_true[idx])),
        "Mabs_pred": float(np.mean(M_pred[idx])),
        "C1_true": float(np.mean(C_true[idx])),
        "C1_pred": float(np.mean(C_pred[idx])),
        "Cv_true": cv_true_val,
        "Cv_pred": cv_pred_val,
        "Cv_true_fluct": float(cv_true_fluct),
        "Cv_pred_fluct": float(cv_pred_fluct),
        "Accuracy_missing": bin_missing_acc,
        "Accuracy_observed": bin_obs_acc,
        "Accuracy_overall": bin_overall_acc,
    })

perbin_df = pd.DataFrame(rows).sort_values("T_center").reset_index(drop=True)
perbin_df.to_csv(os.path.join(OUT_DIR, "metrics_per_temp_bin.csv"), index=False)

# Optional: Also save a version with both Cv methods for comparison
perbin_df_comparison = perbin_df.copy()
perbin_df_comparison["Cv_method"] = "derivative" if use_derivative_method else "fluctuation"
perbin_df_comparison.to_csv(os.path.join(OUT_DIR, "metrics_per_temp_bin_with_Cv_comparison.csv"), index=False)

print(f"?? Calculated heat capacity using {'derivative' if use_derivative_method else 'fluctuation'} method")
print(f"?? Temperature bins: {len(perbin_df)}")
print(f"?? Using derivative-based Cv calculation for better reconstruction accuracy")

# =========================================================
# Plot style: black & white, bold, larger fonts
# =========================================================
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "lines.linewidth": 2.2,
})

# =========================================================
#  PDF 1: PHYSICS (dashboard + energy comparison)
# =========================================================
plot_start = time.time()
physics_pdf = os.path.join(OUT_DIR, "physics_plots.pdf")
with PdfPages(physics_pdf) as pdf:
    # Page 1: 2x2 dashboard: |M|, E, C1, Cv
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    # |M|(T)
    axs[0,0].plot(perbin_df["T_center"], perbin_df["Mabs_true"], marker="o", color="black", label="True")
    axs[0,0].plot(perbin_df["T_center"], perbin_df["Mabs_pred"], marker="s", color="dimgray", label="Recon")
    axs[0,0].axvline(Tc, linestyle="--", color="gray", alpha=0.7, label="Tc")
    axs[0,0].set_title("|M| vs T"); axs[0,0].set_xlabel("T"); axs[0,0].set_ylabel("|M|"); axs[0,0].legend()

    # E(T)
    axs[0,1].plot(perbin_df["T_center"], perbin_df["E_true"], marker="o", color="black", label="True")
    axs[0,1].plot(perbin_df["T_center"], perbin_df["E_pred"], marker="s", color="dimgray", label="Recon")
    axs[0,1].axvline(Tc, linestyle="--", color="gray", alpha=0.7)
    axs[0,1].set_title("Energy per site vs T"); axs[0,1].set_xlabel("T"); axs[0,1].set_ylabel("Energy"); axs[0,1].legend()

    # C1(T)
    axs[1,0].plot(perbin_df["T_center"], perbin_df["C1_true"], marker="o", color="black", label="True")
    axs[1,0].plot(perbin_df["T_center"], perbin_df["C1_pred"], marker="s", color="dimgray", label="Recon")
    axs[1,0].axvline(Tc, linestyle="--", color="gray", alpha=0.7)
    axs[1,0].set_title("Nearest-neighbor corr C1 vs T"); axs[1,0].set_xlabel("T"); axs[1,0].set_ylabel("C1"); axs[1,0].legend()

    # Cv(T)
    axs[1,1].plot(perbin_df["T_center"], perbin_df["Cv_true"], marker="o", color="black", label="True")
    axs[1,1].plot(perbin_df["T_center"], perbin_df["Cv_pred"], marker="s", color="dimgray", label="Recon")
    axs[1,1].axvline(Tc, linestyle="--", color="gray", alpha=0.7)
    axs[1,1].set_title("Heat capacity $C_v$ vs T"); axs[1,1].set_xlabel("T"); axs[1,1].set_ylabel("$C_v$"); axs[1,1].legend()

    plt.tight_layout(); pdf.savefig(); plt.close(fig)

    # Page 2: #8 Energy comparison scatter (Clean vs Recon, Clean vs Noisy)
    # Compute noisy energy on-device then move to CPU
    with torch.no_grad():
        E_noisy = energy_per_site(torch.from_numpy(Y_test).to(device)).cpu().numpy().reshape(-1)
    plt.figure(figsize=(7.2, 5.5))
    plt.scatter(E_true, E_pred, s=12, alpha=0.5, label="E_true vs E_pred (Recon)", color="black")
    plt.scatter(E_true, E_noisy, s=12, alpha=0.5, label="E_true vs E_noisy (Observed)", color="dimgray")
    mn = float(min(E_true.min(), E_pred.min(), E_noisy.min()))
    mx = float(max(E_true.max(), E_pred.max(), E_noisy.max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--", color="gray")
    plt.xlabel("True Energy"); plt.ylabel("Predicted / Noisy Energy"); plt.title("Energy comparison across datasets")
    plt.legend(); plt.tight_layout(); pdf.savefig(); plt.close()

# =========================================================
#  PDF 2: TRAINING (loss curves)
# =========================================================
training_pdf = os.path.join(OUT_DIR, "training_plots.pdf")
with PdfPages(training_pdf) as pdf:
    plt.figure(figsize=(7.2, 5.0))
    plt.plot(train_hist, label="Train", color="black")
    plt.plot(val_hist, label="Val", color="dimgray")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training curves"); plt.legend()
    plt.tight_layout(); pdf.savefig(); plt.close()

# =========================================================
#  PDF 3: CONFIGURATIONS (4 rows: T˜1,2,Tc,4; columns: Clean | Noisy | Recon)
# =========================================================
configs_pdf = os.path.join(OUT_DIR, "configs_bw.pdf")
with PdfPages(configs_pdf) as pdf:
    def nearest_idx(targetT):
        return int(np.argmin(np.abs(T_true - targetT)))
    targets = [1.0, 2.0, Tc, 4.0]

    fig, axes = plt.subplots(len(targets), 3, figsize=(9, 3*len(targets)))
    if len(targets) == 1:
        axes = np.array([axes])

    for r, t0 in enumerate(targets):
        idx = nearest_idx(t0)
        s_clean = S_true[idx,0]
        y_obs  = Y_test[idx,0]
        s_rec  = S_pred[idx,0]
        # Clean | Noisy | Recon (gray colormap)
        ax = axes[r, 0]; im = ax.imshow(s_clean, vmin=-1, vmax=1, cmap="gray", interpolation="nearest")
        ax.set_title(f"Clean  (T={T_true[idx]:.2f})", fontweight="bold"); ax.axis("off")
        ax = axes[r, 1]; im = ax.imshow(y_obs,  vmin=-1, vmax=1, cmap="gray", interpolation="nearest")
        ax.set_title("Noisy (0=miss)", fontweight="bold"); ax.axis("off")
        ax = axes[r, 2]; im = ax.imshow(s_rec,  vmin=-1, vmax=1, cmap="gray", interpolation="nearest")
        ax.set_title("Reconstructed", fontweight="bold"); ax.axis("off")

    plt.tight_layout(); pdf.savefig(); plt.close(fig)

plot_time = time.time() - plot_start

# ==========================
# Save timing information
# ==========================
total_time = time.time() - start_time

timing_df = pd.DataFrame([{
    "timestamp": timestamp,
    "total_time_seconds": total_time,
    "data_loading_time": data_load_time,
    "model_init_time": model_init_time,
    "training_time": training_time,
    "evaluation_time": eval_time,
    "plotting_time": plot_time,
    "epochs_completed": len(train_hist),
    "final_val_loss": best_val_loss,
    "early_stopping_triggered": patience_counter >= PATIENCE,
}])

timing_df.to_csv(os.path.join(OUT_DIR, "training_times.csv"), index=False)

print("\n" + "="*60)
print("OPTIMIZED RUN COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"\nTiming Summary:")
print(f"  Total time:          {total_time:.2f}s")
print(f"  Data loading:        {data_load_time:.2f}s")
print(f"  Model initialization: {model_init_time:.2f}s")
print(f"  Training:            {training_time:.2f}s")
print(f"  Evaluation:          {eval_time:.2f}s")
print(f"  Plotting:            {plot_time:.2f}s")
print(f"  Epochs completed:    {len(train_hist)}")

print(f"\nSaved files:")
print(f"- Global metrics:          {os.path.join(OUT_DIR, 'metrics_global.csv')}")
print(f"- Per-temp-bin metrics:    {os.path.join(OUT_DIR, 'metrics_per_temp_bin.csv')}")
print(f"- Per-sample predictions:  {os.path.join(OUT_DIR, 'preds_test.csv')}")
print(f"- Training times:          {os.path.join(OUT_DIR, 'training_times.csv')}")
print(f"- Best model:              {os.path.join(OUT_DIR, 'best_model.pth')}")
print(f"- Physics plots (PDF):     {physics_pdf}")
print(f"- Training plots (PDF):    {training_pdf}")
print(f"- Config grids (PDF):      {configs_pdf}")

print(f"\nOptimization features applied:")
print("? No physics regularization (best from sensitivity analysis)")
print("? Optimized learning rate (5e-4)")
print("? Early stopping (patience=5)")
print("? Learning rate scheduling")
print("? Slightly deeper architecture")
print("? Comprehensive timing tracking")

print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")