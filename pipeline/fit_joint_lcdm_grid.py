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
from fqmtmcmc.utils import load_hz_data

# === Load Pantheon+SH0ES SN data ===
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data", "Pantheon+SH0ES.dat")
df_sn = pd.read_csv(data_path, sep=r'\s+')[['zCMB', 'MU_SH0ES', 'MU_SH0ES_ERR_DIAG']]
df_sn.columns = ['z', 'mu', 'sigma_mu']
df_sn = df_sn[df_sn['z'] > 0.023]
z_sn, mu_obs, sigma_mu = df_sn['z'].values, df_sn['mu'].values, df_sn['sigma_mu'].values
n_sn = len(z_sn)

# === Load H(z) data ===
z_hz, Hz_obs, sigma_Hz = load_hz_data(include_farooq=True, deduplicate=True)
n_hz = len(z_hz)
n_total = n_sn + n_hz

# === Load M_locked ===
M_path = os.path.join(script_dir, "..", "outputs", "M_values.txt")
with open(M_path, "r") as f:
    M_locked = float(f.read().splitlines()[1].split("=")[1].strip().split()[0])
print(f"Using M_locked = {M_locked:.5f}")

# === ΛCDM model functions ===
def Ez(z, Om):
    return np.sqrt(Om * (1 + z)**3 + (1 - Om))

def dL_LCDM(z, Om, H0):
    c = 299792.458
    dL = []
    for zi in z:
        if zi == 0:
            dL.append(0.0)
            continue
        num_pts = max(1000, int(zi * 2000))
        zp = np.linspace(0, zi, num_pts)
        integral = trapezoid(1 / Ez(zp, Om), zp)
        dL_i = (1 + zi) * c / H0 * integral
        dL.append(dL_i)
    return np.array(dL)

def mu_LCDM(z, Om, H0, M_locked):
    dL = dL_LCDM(z, Om, H0)
    return 5 * np.log10(dL) + 25 + M_locked

def Hz_LCDM(z, Om, H0):
    return H0 * Ez(z, Om)

# === Grid search ===
Om_vals = np.linspace(0.2, 0.4, 50)
H0_vals = np.linspace(66.0, 74.0, 50)
chi2_grid = np.full((len(Om_vals), len(H0_vals)), np.inf)

for i, Om in enumerate(tqdm(Om_vals, desc="Scanning Ωₘ grid")):
    for j, H0 in enumerate(H0_vals):
        mu_model = mu_LCDM(z_sn, Om, H0, M_locked)
        Hz_model = Hz_LCDM(z_hz, Om, H0)
        chi2_sn = np.sum(((mu_obs - mu_model) / sigma_mu) ** 2)
        chi2_hz = np.sum(((Hz_obs - Hz_model) / sigma_Hz) ** 2)
        chi2_grid[i, j] = chi2_sn + chi2_hz

# === Best-fit location ===
imin, jmin = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
best_Om = Om_vals[imin]
best_H0 = H0_vals[jmin]
best_chi2 = chi2_grid[imin, jmin]
aic = best_chi2 + 2 * 2
bic = best_chi2 + 2 * np.log(n_total)

# === Uncertainty estimation ===
def estimate_uncertainty(axis_vals, chi2_slice):
    try:
        coeffs = np.polyfit(axis_vals, chi2_slice, 2)
        curvature = coeffs[0]
        return np.sqrt(2 / curvature) if curvature > 0 else None
    except:
        return None

sigma_Om = estimate_uncertainty(Om_vals, chi2_grid[:, jmin])
sigma_H0 = estimate_uncertainty(H0_vals, chi2_grid[imin, :])

# === Residuals and RMS
mu_best = mu_LCDM(z_sn, best_Om, best_H0, M_locked)
Hz_best = Hz_LCDM(z_hz, best_Om, best_H0)
resid_mu = mu_obs - mu_best
resid_Hz = Hz_obs - Hz_best
rms_mu = np.sqrt(np.mean(resid_mu**2))     # ✅ Correct RMS for SN
rms_Hz = np.sqrt(np.mean(resid_Hz**2))     # ✅ Correct RMS for Hz

# === Save JSON summary ===
out_dir = os.path.join(script_dir, "..", "outputs")
summary = {
    "ΛCDM_joint": {
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
        "chi2_dof": float(best_chi2 / (n_total - 2)),
        "residuals": {
            "mu_rms": float(rms_mu),
            "Hz_rms": float(rms_Hz)
        },
        "M_locked": M_locked,
        "n_params": 2,
        "n_data": n_total
    }
}

with open(os.path.join(out_dir, "joint_lcdm_grid_fit_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("✅ Saved joint_lcdm_grid_fit_summary.json")

# === Optional: Plot ===
plt.figure(figsize=(8, 6))
plt.contourf(H0_vals, Om_vals, chi2_grid, levels=40, cmap="viridis")
plt.colorbar(label=r"$\chi^2$")
plt.scatter(best_H0, best_Om, color="red", label="Best-fit")
plt.xlabel(r"$H_0$")
plt.ylabel(r"$\Omega_m$")
plt.title(r"Joint SN + H(z) Fit: $\Lambda$CDM Grid")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "joint_lcdm_grid_fit_chi2.png"))
plt.close()
print("✅ Saved joint_lcdm_grid_fit_chi2.png")
