[help.md](https://github.com/user-attachments/files/24615819/help.md)
# Help Guide: Inverse Ising Generative Framework

This repository provides code, datasets, and analysis scripts for **inverse generative modeling of temperature and phase inference in the 2D Ising model under partial and noisy observations**.

---

## Repository Structure

```
.
├── J5Data/
│   ├── MCD{L}.csv        # Clean Ising datasets (L = 32, 64, 128)
│   ├── MCDN{L}.csv       # Noisy / masked datasets
│   └── J5Dataset_info.csv
│
├── J5MCD.py              # Dataset generation (Wolff cluster MC)
├── J1DLN.py              # Inverse generative neural network model
├── J1COM.py              # Physics operators (E, |M|, C1, Cv)
├── J1EVALUATION.py       # Evaluation & metric computation
├── J1SEN.py              # Sensitivity analysis (loss, arch, LR)
└── help.md               # This file
```

---

## Dataset Description

- Model: 2D Ising model with periodic boundary conditions  
- Lattice sizes: `L ∈ {32, 64, 128}`  
- Temperatures: uniformly sampled in `[1.0, 4.0]`  
- Total samples per dataset: **50,000**  
- Sampling: **Wolff cluster Monte Carlo** (efficient near criticality)

Each row in the dataset contains:
- `Temperature`
- `Phase` (`F` for ferromagnetic, `P` for paramagnetic)
- Flattened spins `spin_0 ... spin_{L*L-1}`

Noisy datasets additionally include:
- Random spin flips (probability `q = 0.10`)
- Random masking to zero (rate `m = 0.30`)

---

## Model Overview

The inverse model jointly performs:
1. **Spin reconstruction** from partial/noisy observations  
2. **Temperature regression**
3. **Phase classification**

Key components:
- Circular Conv2D encoder–decoder
- Physics-informed losses on energy, magnetization, and correlations
- Global Average Pooling heads for temperature and phase

---

## Running Experiments

### 1. Generate datasets
```bash
python J5MCD.py
```

### 2. Train and evaluate model
```bash
python J1DLN.py
```

### 3. Sensitivity analysis
```bash
python J1SEN.py
```

This produces:
- CSV summaries
- PDF plots (loss weights, architecture depth, learning rate)

---

## Evaluation Metrics

- MAE_T : Temperature prediction error
- Acc_phi : Phase classification accuracy
- ImpAcc : Imputation accuracy on masked spins
- Physics errors: Energy, |M|, C1, Cv

All physics metrics are computed after discretizing reconstructed spins using `sign(ŝ)`.

---

## Reproducibility

- Fixed random seeds
- Explicit train/val/test splits (80/10/10)
- All hyperparameters logged in CSV outputs

---

## Citation

If you use this repository, please cite the associated paper:

> *Inverse Generative Modeling for Temperature and Phase Inference in the 2D Ising Model from Partial and Noisy Observations*

---

## Contact

For questions or collaborations, please open an issue or contact the authors.
