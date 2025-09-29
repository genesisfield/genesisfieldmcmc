#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import json
from scipy.optimize import minimize

# === Set random seed for reproducibility ===
np.random.seed(42)

# === Local import for utility loader ===
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "..")))
from fqmtmcmc.utils import load_hz_data

# === Output directory ===
out_dir = os.path.join(script_dir, "..", "outputs")
os.makedirs(out_dir, exist_ok=True)

# === Load H(z) data ===
z, Hz_obs, sigma_Hz = load_hz_data(include_farooq=True, deduplicate=True)
n_data = len(z)

# === Genesis Field ripple model ===
def ripple_Hz(z, Om, eps, omega, phi, gamma, H0):
    r = eps * np.exp(-gamma * z) * np.cos(omega * z + phi)
    r0 = eps * np.cos(phi)
    norm = (1 + r) / (1 + r0)
    return H0 * norm * np.sqrt(Om * (1 + z)**3 + (1 - Om))

def log_likelihood(theta):
    Om, eps, omega, phi, gamma, H0 = theta
    model = ripple_Hz(z, Om, eps, omega, phi, gamma, H0)
    return -0.5 * np.sum(((Hz_obs - model) / sigma_Hz) ** 2)

def log_prior(theta):
    Om, eps, omega, phi, gamma, H0 = theta
    if (
        0.1 < Om < 0.5 and
        -0.01 < eps < 0.01 and
        0.07 < omega < 0.25 and
        -np.pi < phi < np.pi and
        0.07 < gamma < 0.25 and
        67.0 < H0 < 72.0
    ):
        return 0.0
    return -np.inf

def log_prob(theta):
    lp = log_prior(theta)
    return lp + log_likelihood(theta) if np.isfinite(lp) else -np.inf

# === Run MCMC ===
ndim, nwalkers, nsteps = 6, 50, 30000
initial = [0.33, 0.0001, 0.155, 0.0, 0.155, 70.2]
pos = initial + 1e-4 * np.random.randn(nwalkers, ndim)

print("Running Genesis Field H(z)-relaxed MCMC...")
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
sampler.run_mcmc(pos, nsteps, progress=True)

samples = sampler.get_chain(discard=5000, thin=10, flat=True)
labels = ["Ωₘ", "ε", "ω", "ϕ", "γ", "H₀"]
medians = np.median(samples, axis=0)
stds = np.std(samples, axis=0)

# === Save MCMC output (PDF only) ===
np.save(os.path.join(out_dir, "hz_chain_mcmc_relaxed.npy"), sampler.get_chain())
np.save(os.path.join(out_dir, "hz_log_prob_mcmc_relaxed.npy"), sampler.get_log_prob())

fig = corner.corner(samples, labels=labels, truths=medians)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "hz_corner_relaxed.pdf"),
            format="pdf", bbox_inches="tight")
plt.close(fig)

print("✅ Saved MCMC chain/log-prob and PDF corner plot")

# === Fit Statistics: Genesis ===
Hz_fit_genesis = ripple_Hz(z, *medians)
resid_genesis = Hz_obs - Hz_fit_genesis
chi2_genesis = np.sum((resid_genesis / sigma_Hz) ** 2)
k_genesis = ndim
aic_genesis = chi2_genesis + 2 * k_genesis
bic_genesis = chi2_genesis + k_genesis * np.log(n_data)
rms_genesis = np.sqrt(np.mean(resid_genesis**2))  # ✅ Fixed to true RMS

# === Fit Statistics: ΛCDM ===
def Hz_LCDM(z, H0, Om):
    return H0 * np.sqrt(Om * (1 + z)**3 + (1 - Om))

def chi2_LCDM(x):
    H0, Om = x
    return np.sum(((Hz_obs - Hz_LCDM(z, H0, Om)) / sigma_Hz) ** 2)

res = minimize(chi2_LCDM, x0=[70.0, 0.3], bounds=[(66, 74), (0.1, 0.5)])
H0_lcdm, Om_lcdm = res.x
Hz_fit_lcdm = Hz_LCDM(z, H0_lcdm, Om_lcdm)
resid_lcdm = Hz_obs - Hz_fit_lcdm
chi2_lcdm = np.sum((resid_lcdm / sigma_Hz) ** 2)
k_lcdm = 2
aic_lcdm = chi2_lcdm + 2 * k_lcdm
bic_lcdm = chi2_lcdm + k_lcdm * np.log(n_data)
rms_lcdm = np.sqrt(np.mean(resid_lcdm**2))  # ✅ Fixed to true RMS

# === Print Comparison ===
print("\n=== Model Comparison: Genesis Field vs ΛCDM (H(z)-Relaxed) ===")
print(f"{'Metric':<25} {'Genesis Field':>15} {'ΛCDM':>15}")
print(f"{'χ²':<25} {chi2_genesis:>15.2f} {chi2_lcdm:>15.2f}")
print(f"{'AIC':<25} {aic_genesis:>15.2f} {aic_lcdm:>15.2f}")
print(f"{'BIC':<25} {bic_genesis:>15.2f} {bic_lcdm:>15.2f}")
print(f"{'χ² / dof':<25} {chi2_genesis / (n_data - k_genesis):>15.3f} {chi2_lcdm / (n_data - k_lcdm):>15.3f}")
print(f"{'Residual RMS [km/s/Mpc]':<25} {rms_genesis:>15.4f} {rms_lcdm:>15.4f}")
print(f"{'# Free Parameters':<25} {k_genesis:>15} {k_lcdm:>15}")

# === Save JSON summary ===
summary = {
    "Genesis_Field": {
        "parameters": dict(zip(labels, map(float, medians))),
        "uncertainty": dict(zip(labels, map(float, stds))),
        "chi2": float(chi2_genesis),
        "aic": float(aic_genesis),
        "bic": float(bic_genesis),
        "chi2_dof": float(chi2_genesis / (n_data - k_genesis)),
        "residual_rms": float(rms_genesis),  # ✅ fixed
        "n_params": k_genesis
    },
    "ΛCDM": {
        "Ωₘ": float(Om_lcdm),
        "H₀": float(H0_lcdm),
        "chi2": float(chi2_lcdm),
        "aic": float(aic_lcdm),
        "bic": float(bic_lcdm),
        "chi2_dof": float(chi2_lcdm / (n_data - k_lcdm)),
        "residual_rms": float(rms_lcdm),  # ✅ fixed
        "n_params": k_lcdm
    }
}

with open(os.path.join(out_dir, "hz_model_comparison_relaxed.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("✅ Saved hz_model_comparison_relaxed.json")
