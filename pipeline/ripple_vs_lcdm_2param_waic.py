#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
from scipy.optimize import curve_fit

# === Local import patch (no install needed) ===
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fqmtmcmc.utils import load_hz_data

# === Setup paths ===
here = os.path.dirname(__file__)
out_dir = os.path.join(here, "..", "outputs")
os.makedirs(out_dir, exist_ok=True)

# === Load ΛCDM H(z)-grid fit JSON (reference, equal DOF) ===
lcdm_path = os.path.join(out_dir, "hz_lcdm_grid_relaxed_fit_summary.json")
with open(lcdm_path, encoding="utf-8") as f:
    lcdm = json.load(f)["ΛCDM"]

H0_lcdm = float(lcdm["parameters"]["H₀"])
Om_lcdm = float(lcdm["parameters"]["Ωₘ"])
chi2_lcdm = float(lcdm["chi2"])
AIC_lcdm  = float(lcdm["aic"])
BIC_lcdm  = float(lcdm["bic"])

# === Load H(z) data ===
z, Hz, sigma = load_hz_data(include_farooq=True, deduplicate=True)  # arrays [n]
n_data = len(z)

# === Genesis Field constants (locked) ===
Om          = 0.36
omega_star  = 0.16
phi_star    = 1.18
gamma_fixed = 0.15

# === Ripple model: ε, H₀ free ===
def H_ripple(z, eps, H0):
    r  = eps * np.exp(-gamma_fixed * z) * np.cos(omega_star * z + phi_star)
    r0 = eps * np.cos(phi_star)
    return H0 * (1.0 + r) / (1.0 + r0) * np.sqrt(Om * (1 + z)**3 + (1 - Om))

# --- Optional strategic weighting (kept from your original MLE setup) ---
sigma_strat = sigma.copy()
sigma_strat[(0.15 <= z) & (z <= 0.4)]  /= 2.0
sigma_strat[(0.6  <= z) & (z <= 1.2)]  /= 2.0

# === Fit ripple model (ε, H₀) with curve_fit (MLE) ===
popt, pcov = curve_fit(
    H_ripple, z, Hz, p0=[0.05, 68.0],
    sigma=sigma_strat, absolute_sigma=True
)
eps_hat, H0_hat = popt
eps_err, H0_err = np.sqrt(np.diag(pcov))

# === Ripple model stats (use observational sigma for chi2) ===
Hz_fit       = H_ripple(z, eps_hat, H0_hat)
chi2_ripple  = np.sum(((Hz - Hz_fit) / sigma) ** 2)
AIC_ripple   = chi2_ripple + 2 * 2     # k=2
BIC_ripple   = chi2_ripple + 2 * np.log(n_data)
rms_ripple   = np.sqrt(np.mean((Hz - Hz_fit)**2))

# === ΛCDM residual RMS (reference) ===
def Hz_LCDM(z, H0, Om):
    return H0 * np.sqrt(Om * (1 + z)**3 + (1 - Om))

Hz_lcdm_model = Hz_LCDM(z, H0_lcdm, Om_lcdm)
rms_lcdm      = np.sqrt(np.mean((Hz - Hz_lcdm_model)**2))

# === Print summary (MLE/AIC/BIC) ===
print("✅ Ripple Fit (Ωₘ=0.36, ϕ=1.18, ω=0.16, γ=0.15):")
print(f"ε = {eps_hat:.4f} ± {eps_err:.4f}")
print(f"H₀ = {H0_hat:.4f} ± {H0_err:.4f}")
print(f"χ² = {chi2_ripple:.2f}, AIC = {AIC_ripple:.2f}, BIC = {BIC_ripple:.2f}")
print(f"RMS (Ripple) = {rms_ripple:.4f} km/s/Mpc")

print("\nΛCDM Reference (from grid summary):")
print(f"H₀ = {H0_lcdm:.4f}, Ωₘ = {Om_lcdm:.5f}")
print(f"χ² = {chi2_lcdm:.2f}, AIC = {AIC_lcdm:.2f}, BIC = {BIC_lcdm:.2f}")
print(f"RMS (ΛCDM)   = {rms_lcdm:.4f} km/s/Mpc")

# =========================
# WAIC (Gaussian posterior)
# =========================

# --- Refit 2-param ΛCDM to get its covariance (equal-DOF fairness) using observational sigma ---
popt_lcdm_mle, pcov_lcdm = curve_fit(
    lambda zz, H0, Om: Hz_LCDM(zz, H0, Om),
    z, Hz, p0=[H0_lcdm, Om_lcdm], sigma=sigma, absolute_sigma=True,
    bounds=([50.0, 0.01], [90.0, 0.80])
)
H0_hat_lcdm, Om_hat_lcdm = popt_lcdm_mle

# --- Helpers: stable logsumexp & WAIC from loglik[S,n] ---
def _logsumexp(a, axis=None):
    amax = np.max(a, axis=axis, keepdims=True)
    out  = np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True)) + amax
    return np.squeeze(out, axis=axis)

def waic_from_loglik(loglik):  # loglik shape [S, n]
    lppd_i = _logsumexp(loglik, axis=0) - np.log(loglik.shape[0])  # [n]
    lppd   = float(np.sum(lppd_i))
    p_waic = float(np.sum(np.var(loglik, axis=0, ddof=1)))
    waic   = -2.0 * (lppd - p_waic)
    return waic, p_waic, lppd

# --- Build pointwise log-likelihoods using true observational sigma ---
var   = sigma**2
const = -0.5*np.log(2*np.pi*var)   # [n]

def loglik_ripple_batch(theta_block):  # theta_block: [B, 2] (eps, H0)
    eps_b = theta_block[:, 0][:, None]  # [B,1]
    H0_b  = theta_block[:, 1][:, None]  # [B,1]
    r     = eps_b * np.exp(-gamma_fixed * z[None, :]) * np.cos(omega_star * z[None, :] + phi_star)  # [B,n]
    r0    = eps_b * np.cos(phi_star)
    Hz_b  = H0_b * (1.0 + r) / (1.0 + r0) * np.sqrt(Om * (1 + z[None, :])**3 + (1 - Om))  # [B,n]
    resid = Hz[None, :] - Hz_b
    return -0.5 * ((resid**2) / var[None, :]) + const[None, :]

def loglik_lcdm_batch(theta_block):  # theta_block: [B, 2] (H0, Om)
    H0_b  = theta_block[:, 0][:, None]
    Om_b  = theta_block[:, 1][:, None]
    Hz_b  = H0_b * np.sqrt(Om_b * (1 + z[None, :])**3 + (1 - Om_b))  # [B,n]
    resid = Hz[None, :] - Hz_b
    return -0.5 * ((resid**2) / var[None, :]) + const[None, :]

# --- Draw Gaussian posterior samples from the curve_fit mean/cov ---
rng = np.random.default_rng(42)
S   = 4000  # posterior samples for WAIC; adjust if needed

def _pd_cov(C):
    C = 0.5 * (C + C.T)
    jitter = 1e-16 * np.eye(C.shape[0])
    return C + jitter

theta_ripple = rng.multivariate_normal(
    mean=np.array([eps_hat, H0_hat], dtype=float),
    cov=_pd_cov(pcov),
    size=S
)
theta_lcdm = rng.multivariate_normal(
    mean=np.array([H0_hat_lcdm, Om_hat_lcdm], dtype=float),
    cov=_pd_cov(pcov_lcdm),
    size=S
)

# --- Evaluate log-likelihoods in batches ---
B = 256
blocks_r, blocks_l = [], []
for i in range(0, S, B):
    blocks_r.append(loglik_ripple_batch(theta_ripple[i:i+B]))
    blocks_l.append(loglik_lcdm_batch(theta_lcdm[i:i+B]))

loglik_ripple = np.vstack(blocks_r)  # [S, n]
loglik_lcdm   = np.vstack(blocks_l)  # [S, n]

# --- Compute WAIC and print ΔWAIC (ripple - LCDM) ---
waic_r, p_r, lppd_r = waic_from_loglik(loglik_ripple)
waic_l, p_l, lppd_l = waic_from_loglik(loglik_lcdm)
dwaic = waic_r - waic_l  # negative favors ripple

print("\n=== WAIC (Gaussian posterior approximation) ===")
print(f"Ripple (2-param): WAIC = {waic_r:.2f}   p_WAIC = {p_r:.2f}")
print(f"ΛCDM   (2-param): WAIC = {waic_l:.2f}   p_WAIC = {p_l:.2f}")
print(f"ΔWAIC (Ripple − ΛCDM) = {dwaic:+.2f}   (negative favors Ripple)")
