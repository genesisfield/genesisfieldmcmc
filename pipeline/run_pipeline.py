# run_pipeline.py

import subprocess

steps = [
    "pipeline/calibrate_and_lock_M.py",
    "tests/test_mu0_model.py",
    "tests/test_dl_ripple.py",
    "pipeline/fit_sn_lcdm_grid.py",
    "pipeline/fit_hz_lcdm_grid_relaxed.py",
    "pipeline/fit_hz_lcdm_grid_tight.py",
    "pipeline/fit_joint_lcdm_grid.py",
    "pipeline/run_fit_pantheon.py",
    "pipeline/run_fit_hz_tight.py",
    "pipeline/run_fit_hz_relaxed.py",
    "pipeline/run_fit_joint.py",
    "pipeline/plot_ripple_parameter_comparison.py",
    "pipeline/ripple_vs_lcdm_2param.py",
    "tests/sweep_fqmt_parameters.py",
    "pipeline/derived_diagnostics_hz_relaxed.py",
    "pipeline/derived_diagnostics_hz_tight.py",
    "pipeline/derived_diagnostics_joint.py",
    "pipeline/derived_diagnostics_sn.py",
    "pipeline/phase_portrait.py",
    "pipeline/ripple_vs_lcdm_2param_waic.py"
]

print("\n=== Running full GENESISFIELDMCMC pipeline ===\n")

for i, step in enumerate(steps, 1):
    label = f"[{i}/{len(steps)}] {step}".ljust(60, ".")
    try:
        subprocess.run(f"python {step}", shell=True, check=True, stdout=None, stderr=None)
        print(f"{label} done")
    except subprocess.CalledProcessError:
        print(f"{label} FAILED ❌")
        break

else:
    print("\n✅ Pipeline completed successfully.")
    print("📁 All results are in: outputs/")
    print("📄 Key files: joint_corner.png, ripple_vs_lcdm_2param.png, M_values.txt, posteriors/*.json\n")
