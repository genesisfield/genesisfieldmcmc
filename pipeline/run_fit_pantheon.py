#!/usr/bin/env python3
import os
import sys
import multiprocessing

# Add repo path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
import json

from fqmtmcmc.mcmc_engine_trap import run_mcmc
from fqmtmcmc.ripple_model_trap import mu_model, dL_ripple

def main():
    # === Set random seed for reproducibility ===
    np.random.seed(42)

    # === Load Data ===
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "Pantheon+SH0ES.dat")
    df = pd.read_csv(data_path, sep=r'\s+')[['zCMB', 'MU_SH0ES', 'MU_SH0ES_ERR_DIAG']]
    df.columns = ['z', 'mu', 'sigma_mu']
    df = df[df['z'] > 0.023]
    z, mu_obs, sigma_mu = df['z'].values, df['mu'].values, df['sigma_mu'].values

    # === Load M_locked ===
    M_path = os.path.join(script_dir, "..", "outputs", "M_values.txt")
    with open(M_path, "r") as f:
        lines = f.read().splitlines()
    M_locked = float(lines[1].split("=")[1].strip().split()[0])
    print(f"Using M_locked = {M_locked:.5f}")

    # === Run MCMC ===
    sampler = run_mcmc(z, mu_obs, sigma_mu, M_locked)

    # === Save chains ===
    out_dir = os.path.join(script_dir, "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "sn_chain_mcmc_pantheon.npy"), sampler.get_chain())
    np.save(os.path.join(out_dir, "sn_log_prob_mcmc_pantheon.npy"), sampler.get_log_prob())

   # === Corner plot (PDF only) ===
    samples = sampler.get_chain(discard=5000, thin=10, flat=True)
    labels_greek = ["Ωₘ", "ε", "ω", "ϕ", "γ", "H₀"]

    fig = corner.corner(samples, labels=labels_greek,
                    truths=np.median(samples, axis=0))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sn_corner_mcmc_pantheon.pdf"),
            format="pdf", bbox_inches="tight")
    plt.close(fig)

    print("✅ Saved corner plot (PDF)")


    # === Post-process stats ===
    best_fit = np.median(samples, axis=0)
    uncertainty = np.std(samples, axis=0)
    mu_pred = mu_model(z, best_fit, M_locked)
    residuals = mu_obs - mu_pred
    n, k = len(z), len(best_fit)
    chi2 = np.sum((residuals / sigma_mu) ** 2)
    aic = chi2 + 2 * k
    bic = chi2 + k * np.log(n)
    rms = np.sqrt(np.mean(residuals**2))  # ✅ Correct RMS calculation

    # === Residual plot (PDF only) ===
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.errorbar(z, residuals, yerr=sigma_mu, fmt='o', ecolor='gray', alpha=0.8)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel(r"Redshift $z$")
    ax.set_ylabel(r"Residual ($\mu_{\rm obs} - \mu_{\rm model}$)")
    ax.set_title(f"Residuals with Locked M = {M_locked}")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sn_residuals_pantheon.pdf"),
            format="pdf", bbox_inches="tight")
    plt.close(fig)

    print("✅ Saved residual plot (PDF)")


    # === Sanity check ===
    z_test = np.array([0.1])
    dL_check = dL_ripple(z_test, *best_fit)[0]
    mu_check = 5 * np.log10(dL_check) + 25 + M_locked
    print(f"✅ Sanity Check (z=0.1): μ = {mu_check:.2f} (expected ~38.3)")
    print(f"✅ Residual mean = {np.mean(residuals):.5f}, RMS = {rms:.5f}")
    print(f"χ²/dof = {chi2 / (n - k):.3f}")

    # === Save JSON summary ===
    summary = {
        "M_locked": M_locked,
        "best_fit": dict(zip(labels_greek, best_fit)),
        "uncertainty": dict(zip(labels_greek, uncertainty)),
        "chi2": chi2,
        "aic": aic,
        "bic": bic,
        "chi2_dof": chi2 / (n - k),
        "residuals": {
            "mean": float(np.mean(residuals)),
            "rms": float(rms)  # ✅ Use correct RMS
        },
        "sanity_check": {
            "mu(z=0.1)": float(mu_check)
        }
    }
    with open(os.path.join(out_dir, "sn_mcmc_pantheon_summary.json"), "w", encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("✅ Saved sn_mcmc_pantheon_summary.json")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
