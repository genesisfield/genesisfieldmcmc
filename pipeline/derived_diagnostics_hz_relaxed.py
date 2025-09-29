#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-process H(z)-relaxed MCMC to compute standard diagnostics:
q(z), w_eff(z), statefinders {r,s}, and Om(z).

- Uses hz_chain_mcmc_relaxed.npy (walkers x steps x 6) with params [Ωm, ε, ω, ϕ, γ, H0]
- No re-fitting. Pure math on existing samples.
- Also computes ΛCDM diagnostics using (H0, Ωm) from hz_model_comparison_relaxed.json if present;
  otherwise falls back to medians from the chain.

Output: ../outputs/hz_relaxed_diagnostics.json
"""
import os, json, numpy as np

# -----------------------
# Paths
# -----------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR      = os.path.join(SCRIPT_DIR, "..", "outputs")
CHAIN_FILE   = os.path.join(OUT_DIR, "hz_chain_mcmc_relaxed.npy")
RELAXED_SUMM = os.path.join(OUT_DIR, "hz_model_comparison_relaxed.json")  # optional

# -----------------------
# Helpers
# -----------------------
def try_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def E_LCDM(z, Om):
    # supports scalar or array Om (broadcast with z)
    return np.sqrt(Om*(1.0+z)**3 + (1.0-Om))

def ripple_Hz_relaxed(z, Om, eps, omega, phi, gamma, H0):
    # all params are scalars here; we loop over samples below
    r  = eps*np.exp(-gamma*z)*np.cos(omega*z + phi)
    r0 = eps*np.cos(phi)
    norm = (1.0 + r)/(1.0 + r0)
    return H0 * norm * E_LCDM(z, Om)

def diagnostics_from_H(H, z, H0):
    E = H / H0
    # second-order edges → correct r0,s0 in LCDM
    dE  = np.gradient(E,  z, edge_order=2)
    d2E = np.gradient(dE, z, edge_order=2)

    zp1 = (1.0 + z)
    q = -1.0 + zp1*(dE/E)
    w = -1.0 + (2.0/3.0)*zp1*(dE/E)
    r = 1.0 - 2.0*zp1*(dE/E) + (zp1**2)*(d2E/E + (dE/E)**2)

    s = (r - 1.0) / (3.0*(q - 0.5))
    s = np.where(np.isfinite(s), s, np.nan)

    # Safe Om(z): undefined at z=0
    den = (1.0 + z)**3 - 1.0
    Om = np.full_like(E, np.nan)
    mask = np.abs(den) > 1e-12
    Om[mask] = (E[mask]**2 - 1.0) / den[mask]
    return q, w, r, s, Om

def median_68(x, axis=0):
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    lo, med, hi = np.nanpercentile(x, [16,50,84], axis=axis)
    return float(med), float(med-lo), float(hi-med)

# -----------------------
# Load chain
# -----------------------
chain = np.load(CHAIN_FILE)  # shape: nwalkers, nsteps, 6
if chain.ndim != 3 or chain.shape[-1] != 6:
    raise ValueError(f"Unexpected chain shape {chain.shape}; expected (walkers, steps, 6).")
samples = chain.reshape(-1, chain.shape[-1])  # [N, 6] = [Ωm, ε, ω, ϕ, γ, H0]
N_total = samples.shape[0]

# Optional: subsample to speed up (same logic as tight)
N_max = 100_000
if N_total > N_max:
    idx = np.random.default_rng(123).choice(N_total, N_max, replace=False)
    samples = samples[idx]
    print(f"Subsampled {N_total} → {samples.shape[0]} for speed.")

# -----------------------
# Compute diagnostics from samples
# -----------------------
z = np.linspace(0.0, 2.0, 401)
i0  = int(np.argmin(np.abs(z-0.0)))
i05 = int(np.argmin(np.abs(z-0.5)))

q0_vals=[]; q05_vals=[]; w0_vals=[]; w05_vals=[]; r0_vals=[]; s0_vals=[]; Om05_vals=[]
for Om, eps, omg, phi, gam, H0 in samples:
    H = ripple_Hz_relaxed(z, Om, eps, omg, phi, gam, H0)
    q, w, r, s, Omz = diagnostics_from_H(H, z, H0)
    q0_vals.append(q[i0]);      q05_vals.append(q[i05])
    w0_vals.append(w[i0]);      w05_vals.append(w[i05])
    r0_vals.append(r[i0]);      s0_vals.append(s[i0])
    Om05_vals.append(Omz[i05])

q0_med = median_68(q0_vals);   q05_med = median_68(q05_vals)
w0_med = median_68(w0_vals);   w05_med = median_68(w05_vals)
r0_med = median_68(r0_vals);   s0_med = median_68(s0_vals)
Om05_med = median_68(Om05_vals)

# -----------------------
# ΛCDM diagnostics (prefer model-comparison JSON; else medians)
# -----------------------
H0_LCDM = None
Om_LCDM = None
if os.path.exists(RELAXED_SUMM):
    try:
        with open(RELAXED_SUMM, "r", encoding="utf-8") as f:
            Jt = json.load(f)
        H0_LCDM = try_get(Jt, "ΛCDM", "H₀") or try_get(Jt, "ΛCDM", "H0")
        Om_LCDM = try_get(Jt, "ΛCDM", "Ωₘ") or try_get(Jt, "ΛCDM", "Om")
    except Exception:
        H0_LCDM, Om_LCDM = None, None

if H0_LCDM is None:
    H0_LCDM = float(np.nanmedian(samples[:,5]))
if Om_LCDM is None:
    Om_LCDM = float(np.nanmedian(samples[:,0]))

H0_LCDM = float(H0_LCDM)
Om_LCDM = float(Om_LCDM)

H_LCDM = H0_LCDM * E_LCDM(z, Om_LCDM)
qL, wL, rL, sL, OmL = diagnostics_from_H(H_LCDM, z, H0_LCDM)
qL0, qL05 = float(qL[i0]), float(qL[i05])
wL0, wL05 = float(wL[i0]), float(wL[i05])
rL0, sL0  = float(rL[i0]), float(sL[i0])
OmL05     = float(OmL[i05])

# -----------------------
# Print a tiny report
# -----------------------
def fmt_triplet(name, tup):
    med, lo, hi = tup
    return f"{name} = {med:.3f} (+{hi:.3f}/-{lo:.3f})"

print("\nH(z)-Relaxed diagnostics (Genesis, Ωm free):")
print(fmt_triplet("q0", q0_med), " |  ", fmt_triplet("q(0.5)", q05_med))
print(fmt_triplet("w0", w0_med), " |  ", fmt_triplet("w(0.5)", w05_med))
print(fmt_triplet("r0", r0_med), " |  ", fmt_triplet("s0", s0_med))
print(fmt_triplet("Om(0.5)", Om05_med))

print("\nΛCDM diagnostics (H0, Ωm from relaxed summary or chain medians):")
print(f"q0 = {qL0:.3f} ; q(0.5) = {qL05:.3f}")
print(f"w0 = {wL0:.3f} ; w(0.5) = {wL05:.3f}")
print(f"r0 = {rL0:.3f} ; s0 = {sL0:.3f}")
print(f"Om(0.5) = {OmL05:.3f}")

# -----------------------
# Save JSON summary
# -----------------------
out = {
    "config": {
        "z_grid": {"min": float(z.min()), "max": float(z.max()), "N": int(z.size)},
        "z_points_reported": {"q_w": [0.0, 0.5], "Om": [0.5]},
        "H0_LCDM_used": H0_LCDM,
        "Omega_m_LCDM_used": Om_LCDM
    },
    "Genesis_Field": {
        "q0": {"median": q0_med[0], "minus": q0_med[1], "plus": q0_med[2]},
        "q_0p5": {"median": q05_med[0], "minus": q05_med[1], "plus": q05_med[2]},
        "w0": {"median": w0_med[0], "minus": w0_med[1], "plus": w0_med[2]},
        "w_0p5": {"median": w05_med[0], "minus": w05_med[1], "plus": w05_med[2]},
        "r0": {"median": r0_med[0], "minus": r0_med[1], "plus": r0_med[2]},
        "s0": {"median": s0_med[0], "minus": s0_med[1], "plus": s0_med[2]},
        "Om_0p5": {"median": Om05_med[0], "minus": Om05_med[1], "plus": Om05_med[2]}
    },
    "LCDM": {
        "q0": qL0, "q_0p5": qL05,
        "w0": wL0, "w_0p5": wL05,
        "r0": rL0, "s0": sL0,
        "Om_0p5": OmL05
    }
}
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "hz_relaxed_diagnostics.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\n✅ Saved diagnostics → {os.path.join(OUT_DIR, 'hz_relaxed_diagnostics.json')}")
