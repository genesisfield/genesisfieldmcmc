import numpy as np
import matplotlib.pyplot as plt
import os

# --- Tunable illustrative parameters (dimensionless) ---
Gamma = 0.25   # effective damping  (maps to gamma)
Omega = 1.0    # effective frequency (maps to omega)

def f(x, y):
    """Reduced linear system in N = ln a."""
    dx = -Gamma*x + Omega*y
    dy = -Omega*x - Gamma*y
    return dx, dy

# Grid for streamplot
x = np.linspace(-1.5, 1.5, 161)
y = np.linspace(-1.5, 1.5, 161)
X, Y = np.meshgrid(x, y)
U, V = f(X, Y)

# Create figure
fig, ax = plt.subplots(figsize=(5.2, 5.2))
ax.streamplot(X, Y, U, V, density=1.4, linewidth=0.75, arrowsize=1.0)

# Optional: add sample trajectories with Euler integration
def integrate(x0, y0, nstep=600, h=0.02):
    xs = np.empty(nstep); ys = np.empty(nstep)
    x, y = x0, y0
    for k in range(nstep):
        xs[k], ys[k] = x, y
        dx, dy = f(x, y)
        x += h*dx; y += h*dy
    return xs, ys

for ic in [(1.2, 0.0), (0.8, 0.9), (-1.1, 0.8), (0.3, -1.2)]:
    xs, ys = integrate(*ic, nstep=700, h=0.02)
    ax.plot(xs, ys, lw=1.2)

# Labels and style (no title, keep caption in LaTeX)
ax.set_xlabel(r"$x \equiv \varepsilon\cos\phi$")
ax.set_ylabel(r"$y \equiv \varepsilon\sin\phi$")
ax.axhline(0, color='0.7', lw=0.5)
ax.axvline(0, color='0.7', lw=0.5)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
plt.tight_layout()

# --- Save to relative folder ---
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "fig_E1_phase_portrait.pdf"))
plt.close()
