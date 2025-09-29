#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
import corner
import json
from scipy.optimize import minimize

# === Set random seed for reproducibility ===
np.random.seed(42)

# === Add fqmtmcmc path for local dev use ===
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "..")))
from fqmtmcmc.utils import load_hz_data

# === Output directory ===
out_dir = os.path.join(script_dir, "..", "outputs")
os.makedirs(out_dir, exist_ok=True)

# === Load Ωₘ from SN MCMC summary ===
sn_summary_path = os.path.join(out_dir, "sn_mcmc_pantheon_summary.json")
with open(sn_summary_path, "r", encoding='utf-8') as f:
    sn_summary = json.load(f)

if "Genesis_Field" in sn_summary:
    Om_fixed = sn_summary["Genesis_Field"]["parameters"]["Ωₘ"]
else:
    best_fit = sn_summary.get("best_fit", {})
    Om_fixed = best_fit.get("Ωₘ") or best_fit.get("Om")
    if Om_fixed is None:
        raise ValueError("Could not find Ωₘ (or Om) in SN MCMC summary.")

print(f"Loaded Ωₘ from SN MCMC: {Om_fixed:.5f}")

# === Load H(z) data ===
z, Hz_obs, sigma_Hz = load_hz_data(include_farooq=True, deduplicate=True)
n_data = len(z)

# === Genesis ripple model with Ωₘ fixed ===
def ripple_Hz(z, eps, omega, phi, gamma, H0):
    r = eps * np.exp(-gamma * z) * np.cos(omega * z + phi)
    r0 = eps * np.cos(phi)
    norm = (1 + r) / (1 + r0)
    return H0 * norm * np.sqrt(Om_fixed * (1 + z)**3 + (1 - Om_fixed))

def log_likelihood(theta):
    eps, omega, phi, gamma, H0 = theta
    model = ripple_Hz(z, eps, omega, phi, gamma, H0)
    return -0.5 * np.sum(((Hz_obs - model) / sigma_Hz)**2)

def log_prior(theta):
    eps, omega, phi, gamma, H0 = theta
    if (
        -0.01 < eps < 0.01 and
         0.07 < omega < 0.25 and
        -np.pi < phi < np.pi and
         0.07 < gamma < 0.25 and
        69.0 < H0 < 71.5
    ):
        return 0.0
    return -np.inf

def log_prob(theta):
    lp = log_prior(theta)
    return lp + log_likelihood(theta) if np.isfinite(lp) else -np.inf

# === Run MCMC ===
ndim, nwalkers, nsteps = 5, 40, 30000
initial = [0.00003, 0.1547, -0.0187, 0.1548, 70.24]
pos = initial + 1e-4 * np.random.randn(nwalkers, ndim)

print("Running Genesis Field H(z)-tight MCMC...")
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
sampler.run_mcmc(pos, nsteps, progress=True)

samples = sampler.get_chain(discard=5000, thin=10, flat=True)
labels = ["ε", "ω", "ϕ", "γ", "H₀"]
medians = np.median(samples, axis=0)
stds = np.std(samples, axis=0)

print("\nBest-fit Genesis Field Parameters (Tight Fit):")
for label, med, std in zip(labels, medians, stds):
    print(f"{label:>3} = {med: .5f} ± {std:.5f}")

# === Save MCMC outputs (PDF only) ===
np.save(os.path.join(out_dir, "hz_chain_mcmc_tight.npy"), sampler.get_chain())
np.save(os.path.join(out_dir, "hz_log_prob_mcmc_tight.npy"), sampler.get_log_prob())

fig = corner.corner(samples, labels=labels, truths=medians)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "hz_corner_tight.pdf"),
            format="pdf", bbox_inches="tight")
plt.close(fig)

print("✅ Saved MCMC chain, log-prob, and corner plot (PDF)")


# === ΛCDM fit with Ωₘ fixed ===
def Hz_LCDM(z, H0):
    return H0 * np.sqrt(Om_fixed * (1 + z)**3 + (1 - Om_fixed))

def chi2_LCDM(H0):
    return np.sum(((Hz_obs - Hz_LCDM(z, H0)) / sigma_Hz)**2)

result = minimize(chi2_LCDM, x0=[70.0])
H0_lcdm = result.x[0]
Hz_fit_lcdm = Hz_LCDM(z, H0_lcdm)
resid_lcdm = Hz_obs - Hz_fit_lcdm
chi2_lcdm = np.sum((resid_lcdm / sigma_Hz) ** 2)
k_lcdm = 1
aic_lcdm = chi2_lcdm + 2 * k_lcdm
bic_lcdm = chi2_lcdm + k_lcdm * np.log(n_data)
rms_lcdm = np.sqrt(np.mean(resid_lcdm**2))  # ✅ True RMS

# === Genesis Field stats ===
Hz_fit_genesis = ripple_Hz(z, *medians)
resid_genesis = Hz_obs - Hz_fit_genesis
chi2_genesis = np.sum((resid_genesis / sigma_Hz) ** 2)
k_genesis = ndim
aic_genesis = chi2_genesis + 2 * k_genesis
bic_genesis = chi2_genesis + k_genesis * np.log(n_data)
rms_genesis = np.sqrt(np.mean(resid_genesis**2))  # ✅ True RMS

# === Print comparison ===
print("\n=== Model Comparison: Genesis Field vs ΛCDM (H(z)-Tight) ===")
print(f"{'Metric':<25} {'Genesis Field':>15} {'ΛCDM':>15}")
print(f"{'χ²':<25} {chi2_genesis:>15.2f} {chi2_lcdm:>15.2f}")
print(f"{'AIC':<25} {aic_genesis:>15.2f} {aic_lcdm:>15.2f}")
print(f"{'BIC':<25} {bic_genesis:>15.2f} {bic_lcdm:>15.2f}")
print(f"{'χ² / dof':<25} {chi2_genesis / (n_data - k_genesis):>15.3f} {chi2_lcdm / (n_data - k_lcdm):>15.3f}")
print(f"{'Residual RMS [km/s/Mpc]':<25} {rms_genesis:>15.4f} {rms_lcdm:>15.4f}")
print(f"{'# Free Parameters':<25} {k_genesis:>15} {k_lcdm:>15}")

# === Save JSON summary ===
summary = {
    "Ωₘ_fixed": Om_fixed,
    "Genesis_Field": {
        "parameters": dict(zip(labels, map(float, medians))),
        "uncertainty": dict(zip(labels, map(float, stds))),
        "chi2": float(chi2_genesis),
        "aic": float(aic_genesis),
        "bic": float(bic_genesis),
        "chi2_dof": float(chi2_genesis / (n_data - k_genesis)),
        "residual_rms": float(rms_genesis),  # ✅ Fixed RMS
        "n_params": k_genesis
    },
    "ΛCDM": {
        "H₀": float(H0_lcdm),
        "chi2": float(chi2_lcdm),
        "aic": float(aic_lcdm),
        "bic": float(bic_lcdm),
        "chi2_dof": float(chi2_lcdm / (n_data - k_lcdm)),
        "residual_rms": float(rms_lcdm),  # ✅ Fixed RMS
        "n_params": k_lcdm
    }
}

with open(os.path.join(out_dir, "hz_model_comparison_tight.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("✅ Saved hz_model_comparison_tight.json")
