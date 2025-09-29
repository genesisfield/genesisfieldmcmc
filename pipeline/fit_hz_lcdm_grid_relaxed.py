#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from fqmtmcmc.utils import load_hz_data

# === Setup paths ===
script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, "..", "outputs")
os.makedirs(out_dir, exist_ok=True)

# === Load H(z) data ===
z, Hz, sigma = load_hz_data(include_farooq=True)
n = len(z)

# === LCDM model ===
def Hz_LCDM(z, H0, Om):
    return H0 * np.sqrt(Om * (1 + z)**3 + (1 - Om))

# === Grid setup ===
Om_vals = np.linspace(0.2, 0.4, 100)
H0_vals = np.linspace(66, 74, 100)
chi2_grid = np.full((len(Om_vals), len(H0_vals)), np.inf)

# === Compute χ² grid ===
for i, Om in enumerate(tqdm(Om_vals, desc="Ωₘ grid")):
    for j, H0 in enumerate(H0_vals):
        model = Hz_LCDM(z, H0, Om)
        chi2 = np.sum(((Hz - model) / sigma) ** 2)
        chi2_grid[i, j] = chi2

# === Find best fit ===
imin, jmin = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
best_Om = Om_vals[imin]
best_H0 = H0_vals[jmin]
best_chi2 = chi2_grid[imin, jmin]
aic = best_chi2 + 2 * 2
bic = best_chi2 + 2 * np.log(n)

# === Estimate uncertainties using local curvature
def estimate_uncertainty(axis_vals, chi2_slice):
    try:
        coeffs = np.polyfit(axis_vals, chi2_slice, 2)
        curvature = coeffs[0]
        return np.sqrt(2 / curvature) if curvature > 0 else None
    except:
        return None

sigma_Om = estimate_uncertainty(Om_vals, chi2_grid[:, jmin])
sigma_H0 = estimate_uncertainty(H0_vals, chi2_grid[imin, :])

# === Compute proper RMS of residuals
residuals = Hz - Hz_LCDM(z, best_H0, best_Om)
rms = np.sqrt(np.mean(residuals**2))  # ✅ Fixed from np.std(...) to true RMS

# === Save summary ===
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
        "residual_rms": float(rms),  # ✅ Now correctly defined
        "n_params": 2
    }
}

with open(os.path.join(out_dir, "hz_lcdm_grid_relaxed_fit_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("✅ Saved hz_lcdm_grid_relaxed_fit_summary.json")

# === Plot contour ===
plt.figure(figsize=(8, 6))
plt.contourf(H0_vals, Om_vals, chi2_grid, levels=40, cmap="viridis")
plt.colorbar(label=r"$\chi^2$")
plt.scatter(best_H0, best_Om, color="red", label="Best-fit")
plt.xlabel(r"$H_0$")
plt.ylabel(r"$\Omega_m$")
plt.title(r"H(z)-only ΛCDM Grid Fit")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "hz_lcdm_grid_relaxed_fit_chi2.png"))
plt.close()
print("✅ Saved hz_lcdm_grid_relaxed_fit_chi2.png")
