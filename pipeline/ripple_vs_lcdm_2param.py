#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# === Local import patch (no install needed) ===
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fqmtmcmc.utils import load_hz_data

# === Setup paths ===
out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")

# === Load ΛCDM H(z)-grid fit JSON ===
lcdm_path = os.path.join(out_dir, "hz_lcdm_grid_relaxed_fit_summary.json")
with open(lcdm_path, encoding="utf-8") as f:
    lcdm = json.load(f)["ΛCDM"]

H0_lcdm = lcdm["parameters"]["H₀"]
Om_lcdm = lcdm["parameters"]["Ωₘ"]
chi2_lcdm = lcdm["chi2"]
AIC_lcdm = lcdm["aic"]
BIC_lcdm = lcdm["bic"]

# === Load H(z) data ===
z, Hz, sigma = load_hz_data(include_farooq=True, deduplicate=True)

# === Genesis Field constants (locked) ===
Om = 0.36
omega_star = 0.16
phi_star = 1.18
gamma_fixed = 0.15

# === Ripple model: ε, H₀ free ===
def H_ripple(z, eps, H0):
    r = eps * np.exp(-gamma_fixed * z) * np.cos(omega_star * z + phi_star)
    r0 = eps * np.cos(phi_star)
    return H0 * (1 + r) / (1 + r0) * np.sqrt(Om * (1 + z)**3 + (1 - Om))

# === Strategic weighting ===
sigma_strat = sigma.copy()
sigma_strat[(0.15 <= z) & (z <= 0.4)] /= 2
sigma_strat[(0.6 <= z) & (z <= 1.2)] /= 2

# === Fit ripple model (ε, H₀) ===
popt, pcov = curve_fit(H_ripple, z, Hz, p0=[0.05, 68], sigma=sigma_strat, absolute_sigma=True)
eps, H0 = popt
perr = np.sqrt(np.diag(pcov))

# === Ripple model stats ===
Hz_fit = H_ripple(z, eps, H0)
chi2_ripple = np.sum(((Hz - Hz_fit) / sigma) ** 2)
AIC_ripple = chi2_ripple + 2 * 2
BIC_ripple = chi2_ripple + 2 * np.log(len(z))
resid_ripple = Hz - Hz_fit
rms_ripple = np.sqrt(np.mean(resid_ripple**2))

# === ΛCDM residual RMS
Hz_lcdm_model = H0_lcdm * np.sqrt(Om_lcdm * (1 + z)**3 + (1 - Om_lcdm))
resid_lcdm = Hz - Hz_lcdm_model
rms_lcdm = np.sqrt(np.mean(resid_lcdm**2))

# === Output summary ===
print("✅ Final Ripple Fit (Ωₘ = 0.36, φ = 1.18, ω = 0.16, γ = 0.15):")
print(f"ε = {eps:.4f} ± {perr[0]:.4f}")
print(f"H₀ = {H0:.4f} ± {perr[1]:.4f}")
print(f"χ² = {chi2_ripple:.2f}, AIC = {AIC_ripple:.2f}, BIC = {BIC_ripple:.2f}")
print(f"RMS (Ripple) = {rms_ripple:.4f} km/s/Mpc")

print("\nΛCDM Reference:")
print(f"H₀ = {H0_lcdm:.4f}, Ωₘ = {Om_lcdm:.5f}")
print(f"χ² = {chi2_lcdm:.2f}, AIC = {AIC_lcdm:.2f}, BIC = {BIC_lcdm:.2f}")
print(f"RMS (ΛCDM)   = {rms_lcdm:.4f} km/s/Mpc")

# === Save summary to JSON
summary = {
    "Ripple_Model": {
        "Ωₘ_locked": Om,
        "ω": omega_star,
        "ϕ": phi_star,
        "γ": gamma_fixed,
        "ε": float(eps),
        "ε_err": float(perr[0]),
        "H₀": float(H0),
        "H₀_err": float(perr[1]),
        "chi2": float(chi2_ripple),
        "aic": float(AIC_ripple),
        "bic": float(BIC_ripple),
        "residual_rms": float(rms_ripple),
        "n_params": 2
    },
    "ΛCDM": {
        "Ωₘ": float(Om_lcdm),
        "H₀": float(H0_lcdm),
        "chi2": float(chi2_lcdm),
        "aic": float(AIC_lcdm),
        "bic": float(BIC_lcdm),
        "residual_rms": float(rms_lcdm),
        "n_params": 2
    },
    "n_data": len(z)
}

with open(os.path.join(out_dir, "ripple_vs_lcdm_2param_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("✅ Saved ripple_vs_lcdm_2param_summary.json")

# === Plot
z_plot = np.linspace(0, 2.5, 400)

def Hz_LCDM(z):
    return H0_lcdm * np.sqrt(Om_lcdm * (1 + z)**3 + (1 - Om_lcdm))

plt.figure(figsize=(9, 5))
plt.errorbar(z, Hz, yerr=sigma, fmt='o', alpha=0.6, label='H(z) Data')
plt.plot(z_plot, Hz_LCDM(z_plot), '--', label='ΛCDM', color="orange", linewidth=2)
plt.plot(z_plot, H_ripple(z_plot, eps, H0), '-', label='Genesis Ripple Fit', color="green", linewidth=2)
plt.xlabel('z')
plt.ylabel('H(z) [km/s/Mpc]')
plt.title("H(z) Comparison (Ripple Model: ε, H₀ Free; Others Fixed)")
plt.legend()
plt.tight_layout()

# Save as PDF (vector, journal-ready)
plt.savefig(os.path.join(out_dir, "ripple_vs_lcdm_hz_2param_hero.pdf"),
            format="pdf", bbox_inches="tight")
plt.close()

print("✅ Saved ripple_vs_lcdm_hz_2param_hero.pdf")

