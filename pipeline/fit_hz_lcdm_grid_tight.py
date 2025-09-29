#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import json
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

# === Fixed Ωₘ from Pantheon+
fixed_Om = 0.3671  # Pantheon+-calibrated value
H0_vals = np.linspace(60, 75, 300)
chi2_vals = np.full_like(H0_vals, np.inf)

# === Compute χ² over H₀ grid
for j, H0 in enumerate(H0_vals):
    model = Hz_LCDM(z, H0, fixed_Om)
    chi2 = np.sum(((Hz - model) / sigma) ** 2)
    chi2_vals[j] = chi2

# === Find best-fit H₀
jmin = np.argmin(chi2_vals)
best_H0 = H0_vals[jmin]
best_chi2 = chi2_vals[jmin]
aic = best_chi2 + 2 * 1
bic = best_chi2 + 1 * np.log(n)

# === Estimate H₀ uncertainty from curvature
def estimate_uncertainty(x_vals, chi2_vals):
    try:
        coeffs = np.polyfit(x_vals, chi2_vals, 2)
        curvature = coeffs[0]
        return np.sqrt(2 / curvature) if curvature > 0 else None
    except:
        return None

sigma_H0 = estimate_uncertainty(H0_vals, chi2_vals)

# === Compute true residual RMS
residuals = Hz - Hz_LCDM(z, best_H0, fixed_Om)
rms = np.sqrt(np.mean(residuals**2))  # ✅ FIXED from np.std(...)

# === Save JSON summary
summary = {
    "ΛCDM": {
        "parameters": {
            "Ωₘ": fixed_Om,
            "H₀": best_H0
        },
        "uncertainty": {
            "Ωₘ": 0.0,
            "H₀": sigma_H0
        },
        "chi2": float(best_chi2),
        "aic": float(aic),
        "bic": float(bic),
        "chi2_dof": float(best_chi2 / (n - 1)),
        "residual_rms": float(rms),  # ✅ RMS now correct
        "n_params": 1
    }
}

with open(os.path.join(out_dir, "hz_lcdm_grid_tight_fit_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("✅ Saved hz_lcdm_grid_tight_fit_summary.json")

# === Plot χ² vs H₀
plt.figure(figsize=(8, 5))
plt.plot(H0_vals, chi2_vals, label="χ²(H₀)")
plt.axvline(best_H0, color="red", linestyle="--", label="Best-fit")
plt.xlabel(r"$H_0$ [km/s/Mpc]")
plt.ylabel(r"$\chi^2$")
plt.title(r"Tight $H(z)$ ΛCDM Grid Fit (Fixed $\Omega_m = 0.3671$)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "hz_lcdm_grid_tight_fit_chi2.png"))
plt.close()
print("✅ Saved hz_lcdm_grid_tight_fit_chi2.png")
