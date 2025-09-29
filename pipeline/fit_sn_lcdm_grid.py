#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from tqdm import tqdm

# Add root path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === Load Pantheon+SH0ES data ===
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data", "Pantheon+SH0ES.dat")
df = pd.read_csv(data_path, sep=r'\s+')[['zCMB', 'MU_SH0ES', 'MU_SH0ES_ERR_DIAG']]
df.columns = ['z', 'mu', 'sigma_mu']
df = df[df['z'] > 0.023]
z, mu_obs, sigma_mu = df['z'].values, df['mu'].values, df['sigma_mu'].values
n = len(z)

# === Load M_locked ===
M_path = os.path.join(script_dir, "..", "outputs", "M_values.txt")
with open(M_path, "r") as f:
    M_locked = float(f.read().splitlines()[1].split("=")[1].strip().split()[0])
print(f"Using M_locked = {M_locked:.5f}")

# === Distance modulus model ===
def mu_LCDM(z, Om, H0, M_locked):
    c = 299792.458  # km/s
    dL = []
    for zi in z:
        if zi == 0:
            dL_i = 0.0
        else:
            num_pts = max(1000, int(zi * 2000))
            zp = np.linspace(0, zi, num_pts)
            Ez = np.sqrt(Om * (1 + zp)**3 + (1 - Om))
            integral = trapezoid(1 / Ez, zp)
            dL_i = (1 + zi) * c / H0 * integral
        dL.append(dL_i)
    dL = np.array(dL)
    mu = 5 * np.log10(dL) + 25 + M_locked
    return mu

# === Grid search ===
Om_vals = np.linspace(0.2, 0.4, 40)
H0_vals = np.linspace(66.0, 74.0, 40)
chi2_grid = np.full((len(Om_vals), len(H0_vals)), np.inf)

for i, Om in enumerate(tqdm(Om_vals, desc="Scanning Ωₘ grid")):
    for j, H0 in enumerate(H0_vals):
        mu_model = mu_LCDM(z, Om, H0, M_locked)
        if np.any(np.isnan(mu_model)) or np.any(mu_model > 200):
            continue
        chi2 = np.sum(((mu_obs - mu_model) / sigma_mu) ** 2)
        chi2_grid[i, j] = chi2

# === Best-fit location ===
imin, jmin = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
best_Om = Om_vals[imin]
best_H0 = H0_vals[jmin]
best_chi2 = chi2_grid[imin, jmin]
aic = best_chi2 + 2 * 2
bic = best_chi2 + 2 * np.log(n)

# === Estimate 1σ uncertainties from curvature ===
def estimate_uncertainty(axis_vals, chi2_slice):
    try:
        coeffs = np.polyfit(axis_vals, chi2_slice, 2)  # fit parabola
        curvature = coeffs[0]
        return np.sqrt(2 / curvature) if curvature > 0 else None
    except:
        return None

sigma_Om = estimate_uncertainty(Om_vals, chi2_grid[:, jmin])
sigma_H0 = estimate_uncertainty(H0_vals, chi2_grid[imin, :])

# === Save standardized JSON summary ===
mu_best = mu_LCDM(z, best_Om, best_H0, M_locked)
residuals = mu_obs - mu_best
rms = np.sqrt(np.mean(residuals**2))  # ✅ Fixed RMS

out_dir = os.path.join(script_dir, "..", "outputs")
summary = {
    "ΛCDM": {
        "parameters": {
            "Ωₘ": best_Om,
            "H₀": best_H0
        },
        "uncertainty": {
            "Ωₘ": sigma_Om,
            "H₀": sigma_H0
        },
        "chi2": float(best_chi2),
        "aic": float(aic),
        "bic": float(bic),
        "chi2_dof": float(best_chi2 / (n - 2)),
        "residuals": {
            "mean": float(np.mean(residuals)),
            "rms": float(rms)  # ✅ Correct RMS
        },
        "M_locked": M_locked,
        "n_params": 2
    }
}

with open(os.path.join(out_dir, "sn_lcdm_grid_fit_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("✅ Saved sn_lcdm_grid_fit_summary.json")

# === Optional: χ² heatmap plot ===
plt.figure(figsize=(8, 6))
plt.contourf(H0_vals, Om_vals, chi2_grid, levels=40, cmap="viridis")
plt.colorbar(label=r"$\chi^2$")
plt.scatter(best_H0, best_Om, color="red", label="Best-fit")
plt.xlabel(r"$H_0$")
plt.ylabel(r"$\Omega_m$")
plt.title(r"ΛCDM Grid Fit: $\chi^2$ over $(\Omega_m, H_0)$")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "sn_lcdm_grid_fit_chi2.png"))
plt.close()
