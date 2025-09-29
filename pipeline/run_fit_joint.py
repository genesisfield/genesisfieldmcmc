#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
import corner
import json

# === Set random seed for reproducibility ===
np.random.seed(42)

# Add repo path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fqmtmcmc.mcmc_engine_joint import run_joint_mcmc, ripple_Hz, mu_model
from fqmtmcmc.utils import load_hz_data

# === Setup paths ===
base = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base, "..", "data")
out_dir = os.path.join(base, "..", "outputs")
os.makedirs(out_dir, exist_ok=True)

# === Load Pantheon+SH0ES SN data ===
df_sn = pd.read_csv(os.path.join(data_dir, "Pantheon+SH0ES.dat"), sep=r'\s+')[['zCMB', 'MU_SH0ES', 'MU_SH0ES_ERR_DIAG']]
df_sn.columns = ['z', 'mu', 'sigma_mu']
df_sn = df_sn[df_sn['z'] > 0.023]
z_sn = df_sn['z'].values
mu_obs = df_sn['mu'].values
sigma_mu = df_sn['sigma_mu'].values

# === Load H(z) data ===
z_hz, Hz_obs, sigma_Hz = load_hz_data(include_farooq=True, deduplicate=True)

# === Load M_locked ===
with open(os.path.join(out_dir, "M_values.txt")) as f:
    M_locked = float(f.read().splitlines()[1].split("=")[1].strip().split()[0])
print(f"Using M_locked = {M_locked:.5f}")

# === Run Joint MCMC ===
sampler = run_joint_mcmc(z_sn, mu_obs, sigma_mu, z_hz, Hz_obs, sigma_Hz, M_locked)
samples = sampler.get_chain(discard=5000, thin=10, flat=True)

# === Unicode labels
labels = ["Ωₘ", "ε", "ω", "ϕ", "γ", "H₀"]
medians = np.median(samples, axis=0)
stds = np.std(samples, axis=0)

# === Save chains and corner plot (PDF only)
np.save(os.path.join(out_dir, "joint_chain_mcmc.npy"), sampler.get_chain())
np.save(os.path.join(out_dir, "joint_log_prob_mcmc.npy"), sampler.get_log_prob())

fig = corner.corner(samples, labels=labels, truths=medians)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "joint_corner.pdf"),
            format="pdf", bbox_inches="tight")
plt.close(fig)

print("✅ Saved joint MCMC chain/log-prob and PDF corner plot")


# === Print best-fit
print("\nBest-fit Genesis Field Parameters (Joint):")
for name, val, err in zip(labels, medians, stds):
    print(f"{name:>3} = {val:.5f} ± {err:.5f}")

# === Model Predictions
Om, eps, omega, phi, gamma, H0 = medians
mu_fit = mu_model(z_sn, Om, eps, omega, phi, gamma, H0, M_locked)
Hz_fit = ripple_Hz(z_hz, Om, eps, omega, phi, gamma, H0)

# === Genesis Field Fit Statistics
chi2_sn = np.sum(((mu_obs - mu_fit) / sigma_mu) ** 2)
chi2_hz = np.sum(((Hz_obs - Hz_fit) / sigma_Hz) ** 2)
chi2_total = chi2_sn + chi2_hz
n_sn, n_hz = len(z_sn), len(z_hz)
n_total, k = n_sn + n_hz, 6
aic = chi2_total + 2 * k
bic = chi2_total + k * np.log(n_total)

# === Compute RMS of residuals
resid_sn = mu_obs - mu_fit
resid_hz = Hz_obs - Hz_fit
rms_sn = np.sqrt(np.mean(resid_sn**2))
rms_hz = np.sqrt(np.mean(resid_hz**2))

# === Print fit stats
print("\n=== Joint Fit Statistics ===")
print(f"χ²_SN       = {chi2_sn:.2f}")
print(f"χ²_Hz       = {chi2_hz:.2f}")
print(f"χ²_total    = {chi2_total:.2f}")
print(f"AIC         = {aic:.2f}")
print(f"BIC         = {bic:.2f}")
print(f"χ²/dof      = {chi2_total / (n_total - k):.3f}")
print(f"RMS_SN      = {rms_sn:.5f} mag")
print(f"RMS_Hz      = {rms_hz:.5f} km/s/Mpc")

# === Load real ΛCDM joint fit
lcdm_json_path = os.path.join(out_dir, "joint_lcdm_grid_fit_summary.json")
with open(lcdm_json_path, "r", encoding="utf-8") as f:
    lcdm_summary = json.load(f)["ΛCDM_joint"]

print("\n=== ΛCDM Joint Fit (Reference) ===")
print(f"χ²_total    = {lcdm_summary['chi2']:.2f}")
print(f"AIC         = {lcdm_summary['aic']:.2f}")
print(f"BIC         = {lcdm_summary['bic']:.2f}")
print(f"χ²/dof      = {lcdm_summary['chi2_dof']:.3f}")

# === Save combined summary
summary = {
    "M_locked": M_locked,
    "Genesis_Field": {
        "parameters": dict(zip(labels, map(float, medians))),
        "uncertainty": dict(zip(labels, map(float, stds))),
        "chi2_sn": float(chi2_sn),
        "chi2_hz": float(chi2_hz),
        "chi2_total": float(chi2_total),
        "aic": float(aic),
        "bic": float(bic),
        "chi2_dof": float(chi2_total / (n_total - k)),
        "residual_rms": {
            "mu": float(rms_sn),
            "Hz": float(rms_hz)
        },
        "n_params": k
    },
    "ΛCDM": lcdm_summary
}

with open(os.path.join(out_dir, "joint_model_comparison_summary.json"), "w", encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("✅ Saved joint_model_comparison_summary.json to /outputs/")
