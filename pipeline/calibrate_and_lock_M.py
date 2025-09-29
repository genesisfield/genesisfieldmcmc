#!/usr/bin/env python3
import sys
import os

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import multiprocessing

from mcalib_ripple.calibrate_M import compute_M_analytic, adjust_M_from_residual
from mcalib_ripple.ripple_model import mu0_model
from mcalib_ripple.mcmc_engine import log_probability, run_mcmc_theta_fit
from mcalib_ripple.utils import reshape_cov

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate and lock M with ripple cosmology"
    )
    default_procs = max(1, multiprocessing.cpu_count() // 2)
    parser.add_argument(
        "-p", "--procs", type=int, default=default_procs,
        help=f"Number of processes (default: half of {multiprocessing.cpu_count()})"
    )
    args = parser.parse_args()

    print(f"🧠 Logical CPUs available: {multiprocessing.cpu_count()}")
    print(f"⚙️  Using {args.procs} processes for MCMC\n")

    np.random.seed(42)

    # Load the Pantheon+SH0ES data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(project_root, "data")
    df = pd.read_csv(os.path.join(data_dir, "Pantheon+SH0ES.dat"), sep=r"\s+")
    z_all = df["zCMB"].astype(float).values
    mu_all = df["MU_SH0ES"].astype(float).values
    mask = z_all > 0.023
    z, mu_obs = z_all[mask], mu_all[mask]

    # Build inverse covariance
    n_all = len(z_all)
    C1 = reshape_cov(np.loadtxt(os.path.join(data_dir, "Pantheon+SH0ES_STAT+SYS.cov")), n_all)
    C2 = reshape_cov(np.loadtxt(os.path.join(data_dir, "Pantheon+SH0ES_122221_VPEC.cov")), n_all)
    C = (C1 + C2)[np.ix_(mask, mask)]
    invC = np.linalg.inv(C)

    # Initial cosmology guess
    theta0 = [0.33259, -0.00679, 0.20, 0.28128, 0.25802, 68.90867]

    # Step 1: Analytic M
    M_analytic, sigma_M = compute_M_analytic(z, mu_obs, theta0, invC)
    print(f"📏 Analytic M_locked = {M_analytic:.5f} ± {sigma_M:.5f} mag")

    # Step 2: Full MCMC over θ with fixed M
    ndim, nwalkers, nsteps = len(theta0), 32, 500
    args_mcmc = (z, mu_obs, invC, M_analytic)
    samples = run_mcmc_theta_fit(
        log_probability, ndim, nwalkers, nsteps, args_mcmc, nprocs=args.procs
    )
    theta_med = np.median(samples, axis=0)

    # Step 3: Adjust M from residuals
    M_adjusted = adjust_M_from_residual(mu_obs, z, theta_med, M_analytic)
    print(f"🔧 Adjusted M_locked   = {M_adjusted:.5f} mag")

    # ✅ Prepare top-level outputs folder
    out_dir = os.path.join(project_root, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # Write M values to text file
    txt_path = os.path.join(out_dir, "M_values.txt")
    with open(txt_path, "w") as f:
        f.write(f"Analytic M_locked = {M_analytic:.5f} ± {sigma_M:.5f} mag\n")
        f.write(f"Adjusted M_locked = {M_adjusted:.5f} mag\n")
    print(f"➡️  Written M values to {txt_path}")

    # Plot residuals
    mu_model0  = mu0_model(z, theta0)
    mu_model_m = mu0_model(z, theta_med)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, mu_model, title in zip(
        axes,
        [mu_model0 + M_analytic, mu_model_m + M_adjusted],
        ["Residuals w/ Analytic M", "Residuals after MCMC"]
    ):
        ax.scatter(z, mu_obs - mu_model, s=5, alpha=0.6)
        ax.set_xscale("log")
        ax.axhline(0, color="red", ls="--")
        ax.set_xlabel("z")
        ax.set_ylabel("Residual (mag)")
        ax.set_title(title)

    png_path = os.path.join(out_dir, "residuals.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"➡️  Saved residual plot to {png_path}")

if __name__ == "__main__":
    main()
