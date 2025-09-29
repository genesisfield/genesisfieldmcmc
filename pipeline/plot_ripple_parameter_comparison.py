import numpy as np
import matplotlib.pyplot as plt
import json
import os

# === Load MCMC summaries from JSON files ===
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
pantheon_path = os.path.join(output_dir, "sn_mcmc_pantheon_summary.json")
hz_path = os.path.join(output_dir, "hz_model_comparison_relaxed.json")
joint_path = os.path.join(output_dir, "joint_model_comparison_summary.json")

with open(pantheon_path, encoding='utf-8') as f:
    pantheon_summary = json.load(f)
with open(hz_path, encoding='utf-8') as f:
    hz_summary = json.load(f)
with open(joint_path, encoding='utf-8') as f:
    joint_summary = json.load(f)

# === Parameters to plot ===
labels = ["ε", "ω", "γ"]

# === Handle both new and legacy JSON formats ===
def extract_values(summary, labels):
    if "Genesis_Field" in summary:
        params = summary["Genesis_Field"]["parameters"]
        errs = summary["Genesis_Field"]["uncertainty"]
    else:
        params = summary["best_fit"]
        errs = summary["uncertainty"]
    return [params[l] for l in labels], [errs[l] for l in labels]

pantheon_vals, pantheon_errs = extract_values(pantheon_summary, labels)
hz_vals, hz_errs = extract_values(hz_summary, labels)
joint_vals, joint_errs = extract_values(joint_summary, labels)

# === Group for bar chart ===
means = {
    "Pantheon+": pantheon_vals,
    "H(z) Relaxed": hz_vals,
    "Joint Relaxed": joint_vals
}
errors = {
    "Pantheon+": pantheon_errs,
    "H(z) Relaxed": hz_errs,
    "Joint Relaxed": joint_errs
}

# === Bar Plot ===
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
colors = ['#1f77b4', '#2ca02c', '#d62728']

for i, param in enumerate(labels):
    ax = axes[i]
    y = [means[key][i] for key in means]
    yerr = [errors[key][i] for key in means]
    ax.bar(range(len(y)), y, yerr=yerr, capsize=6, color=colors)
    ax.axhline(0, linestyle='--', color='gray')
    ax.set_xticks(range(len(y)))
    ax.set_xticklabels(means.keys(), rotation=15)
    ax.set_title(f"{param} across fits")
    ax.set_ylabel(param)
    ax.grid(True)

fig.suptitle("Ripple Parameter Summary Across Fits", fontsize=14)
plt.tight_layout()

# Save as PDF (vector, journal-ready)
plt.savefig(os.path.join(output_dir, "ripple_param_comparison_barplot.pdf"),
            format="pdf", bbox_inches="tight")

plt.close()

print("✅ Saved ripple_param_comparison_barplot.pdf")

