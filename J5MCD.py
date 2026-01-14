# -*- coding: utf-8 -*-
"""
2D Ising dataset generator using Wolff cluster updates (LxL lattice)

- EXACTLY 50,000 samples total:
    TOTAL = 50_000
    N_T   = 50 temperatures
    M_PER_T = TOTAL // N_T = 1000 samples per temperature

- Generates:
    J5Dataset/MCDL32.csv   (clean dataset)
    J5Dataset/MCDNL32.csv  (noisy dataset)

- Logs run metadata to:
    J5Dataset/J5Dataset_info.csv

Cluster dynamics (Wolff):
    * Much better decorrelation near Tc
    * Smoother heat-capacity estimates than Metropolis
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
import csv
import os 

# ======================================================
# USER PARAMETERS
# ======================================================
L = 128                # lattice size
TOTAL = 50000         # total number of rows required
N_T = 50              # number of temperatures
T_MIN, T_MAX = 1.0, 4.0
T_values = np.linspace(T_MIN, T_MAX, N_T)
M_PER_T = TOTAL // N_T    # = 1000 rows per temperature

J = 1.0               # coupling
Tc = 2.269

# Wolff parameters:
EQUIL_UPDATES = 2000      # number of Wolff cluster flips for thermalisation
GAP_UPDATES   = 200       # number of Wolff cluster flips between saved samples

# Noise params for noisy dataset
NOISE_Q   = 0.10          # flip probability
MASK_RATE = 0.30          # masking probability (set spin to 0)

SEED = 123
rng = np.random.default_rng(SEED)

# ======================================================
# OUTPUT PATHS
# ======================================================
output_dir = Path("J5Data")
output_dir.mkdir(parents=True, exist_ok=True)

clean_path = output_dir / f"MCD{L}.csv"
noisy_path = output_dir / f"MCDN{L}.csv"
meta_path  = output_dir / "J5Datat_info_128.csv"


# ======================================================
# WOLFF CLUSTER UPDATE
# ======================================================
def wolff_update(spins, beta, rng):
    """
    Perform a single Wolff cluster update on the 2D Ising lattice.
    spins : (L, L) array with values in {-1, +1}
    beta  : inverse temperature
    """
    Ls = spins.shape[0]
    # bond formation probability
    p_add = 1.0 - np.exp(-2.0 * beta * J)

    # pick random seed spin
    i0 = rng.integers(0, Ls)
    j0 = rng.integers(0, Ls)
    spin0 = spins[i0, j0]

    # BFS / DFS cluster growth
    stack = [(i0, j0)]
    cluster = [(i0, j0)]
    visited = np.zeros((Ls, Ls), dtype=bool)
    visited[i0, j0] = True

    while stack:
        i, j = stack.pop()
        # 4 nearest neighbors with periodic BC
        neighbors = [
            ((i + 1) % Ls, j),
            ((i - 1) % Ls, j),
            (i, (j + 1) % Ls),
            (i, (j - 1) % Ls),
        ]
        for ni, nj in neighbors:
            if not visited[ni, nj] and spins[ni, nj] == spin0:
                if rng.random() < p_add:
                    visited[ni, nj] = True
                    stack.append((ni, nj))
                    cluster.append((ni, nj))

    # flip entire cluster
    for i, j in cluster:
        spins[i, j] = -spins[i, j]


def wolff_sweeps(spins, beta, n_updates, rng):
#    """
#    Apply n_updates Wolff cluster flips.
#    For L=32, 100–200 updates is roughly a few effective sweeps.
#    """
    for _ in range(n_updates):
        wolff_update(spins, beta, rng)


# ======================================================
# NOISE MODEL
# ======================================================
def corrupt_lattice(spins, rng):
    """
    Create noisy version of a clean configuration:
      - flip each spin with probability NOISE_Q
      - mask each spin to 0 with probability MASK_RATE
    """
    noisy = spins.copy()
    Ls = spins.shape[0]

    # random flips
    flip_mask = rng.random((Ls, Ls)) < NOISE_Q
    noisy[flip_mask] *= -1

    # random masking
    mask = rng.random((Ls, Ls)) < MASK_RATE
    noisy[mask] = 0

    return noisy


# ======================================================
# METADATA LOGGER
# ======================================================
def write_metadata(clean_path, noisy_path, elapsed_seconds):
    timestamp_now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    df_clean = pd.read_csv(clean_path)
    df_noisy = pd.read_csv(noisy_path)

    clean_rows, clean_cols = df_clean.shape
    noisy_rows, noisy_cols = df_noisy.shape

    file_exists = os.path.isfile(meta_path)

    with open(meta_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Timestamp",
                "CleanFile", "NoisyFile",
                "L",
                "TOTAL",
                "N_T",
                "M_PER_T",
                "T_min", "T_max",
                "Algorithm",
                "EQUIL_UPDATES",
                "GAP_UPDATES",
                "NOISE_Q",
                "MASK_RATE",
                "SEED",
                "Clean_Rows", "Clean_Cols",
                "Noisy_Rows", "Noisy_Cols",
                "Time_Seconds", "Time_Minutes"
            ])

        writer.writerow([
            timestamp_now,
            clean_path.name,
            noisy_path.name,
            L,
            TOTAL,
            N_T,
            M_PER_T,
            T_MIN,
            T_MAX,
            "Wolff",
            EQUIL_UPDATES,
            GAP_UPDATES,
            NOISE_Q,
            MASK_RATE,
            SEED,
            clean_rows, clean_cols,
            noisy_rows, noisy_cols,
            round(elapsed_seconds, 2),
            round(elapsed_seconds / 60.0, 2)
        ])

    print(f"\nMetadata logged to {meta_path}")


# ======================================================
# MAIN GENERATOR
# ======================================================
def generate_clean_and_noisy():
    clean_rows = []
    noisy_rows = []

    start_time = time.time()

    for T in T_values:
        beta = 1.0 / T
        phase_label = "F" if T < Tc else "P"

        # fresh random configuration in {-1, +1}
        spins = rng.choice([-1, 1], size=(L, L))

        # Equilibration using Wolff updates
        wolff_sweeps(spins, beta, EQUIL_UPDATES, rng)

        print(f"Generating {M_PER_T} samples for T = {T:.3f} using Wolff clusters...")

        for _ in range(M_PER_T):
            # decorrelation between saved configurations
            wolff_sweeps(spins, beta, GAP_UPDATES, rng)

            flat_clean = spins.flatten().tolist()

            noisy = corrupt_lattice(spins, rng)
            flat_noisy = noisy.flatten().tolist()

            clean_rows.append([T, phase_label] + flat_clean)
            noisy_rows.append([T, phase_label] + flat_noisy)

    # build and save CSVs
    columns = ["Temperature", "Phase"] + [f"spin_{i}" for i in range(L * L)]

    pd.DataFrame(clean_rows, columns=columns).to_csv(clean_path, index=False)
    pd.DataFrame(noisy_rows, columns=columns).to_csv(noisy_path, index=False)

    elapsed = time.time() - start_time

    print(f"\nClean dataset saved  ? {clean_path}")
    print(f"Noisy dataset saved  ? {noisy_path}")
    print(f"Total time: {elapsed:.2f} s ({elapsed/60:.2f} min)")

    write_metadata(clean_path, noisy_path, elapsed)


# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":
    generate_clean_and_noisy()
