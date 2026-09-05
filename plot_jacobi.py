"""figures/jacobi_ladder.png from results/jacobi_*.txt: separation growth and local exponent by resolution, hollow markers
where the analyticity clock says the row is past its reliable window."""
import re, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(path):
    rows = []
    for line in open(path):
        m = re.match(r"\s*(\d+\.\d+)\s+(\d+\.\d+)\s+([+-]\d+\.\d+)\s+", line)
        if m:
            rows.append((float(m.group(1)), float(m.group(2)), float(m.group(3)), "NOT RELIABLE" in line))
    return rows

fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
cols = {32: "#9aa5b1", 48: "#4a6fa5", 64: "#1b1b1b"}
for ic, ax, title in (("tg", axes[0], "Taylor-Green"), ("kp", axes[1], "Kida-Pelz")):
    for n in (32, 48, 64):
        rows = load("results/jacobi_%s_%d.txt" % (ic, n))
        if not rows:
            continue
        t = np.array([r[0] for r in rows]); g = np.array([r[1] for r in rows]); bad = np.array([r[3] for r in rows])
        ax.plot(t, g, "-", color=cols[n], lw=1.4, label="%d^3" % n)
        ax.plot(t[~bad], g[~bad], "o", color=cols[n], ms=5)
        ax.plot(t[bad], g[bad], "o", mfc="white", mec=cols[n], ms=5)
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_title("%s: Jacobi-field growth |du(t)| / |du(0)|" % title, loc="left", fontsize=9)
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([1.0, 1.2, 1.5, 2, 3, 4, 5] if ic == "kp" else [1.0, 1.1, 1.2, 1.4, 1.7])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, which="both")
axes[0].set_ylabel("separation growth (log)")
fig.suptitle("nu = 0, eps = 1e-5 random solenoidal perturbation; filled = inside the analyticity-strip window, hollow = past it", fontsize=8.5, x=0.01, ha="left", y=0.04)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig("figures/jacobi_ladder.png", dpi=130)
print("figures/jacobi_ladder.png")
