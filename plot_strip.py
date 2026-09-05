"""figures/strip_decay.png from results/strip_*.txt: the analyticity-strip width delta(t) on a log axis by resolution,
with each resolution's 2 dx reliability line. Exponential decay is a straight line; a finite-time singularity would
bend downward toward -infinity at t*."""
import re, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter

def load(path):
    t, d, dx2 = [], [], None
    for line in open(path):
        m = re.match(r"\s*(\d+\.\d+)\s+(-?\d+\.\d+)\s+", line)
        if m:
            t.append(float(m.group(1))); d.append(float(m.group(2)))
        m2 = re.search(r"2dx = (\d+\.\d+)", line)
        if m2:
            dx2 = float(m2.group(1))
    return np.array(t), np.array(d), dx2

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
cols = {48: "#9aa5b1", 64: "#4a6fa5", 96: "#1b1b1b", 128: "#b5432a"}
for ic, ns, ax, title in (("tg", (48, 64, 96), axes[0], "Taylor-Green"), ("kp", (64, 96, 128), axes[1], "Kida-Pelz")):
    for n in ns:
        t, d, dx2 = load("results/strip_%s_%d.txt" % (ic, n))
        early = t < 1.0
        i0 = int(np.argmax(np.where(early, d, -np.inf)))      # after the fill-in transient: the upper shells start at round-off
        ok = (d > dx2) & (np.arange(len(t)) >= i0)
        ax.plot(t[ok], d[ok], "-", color=cols[n], lw=1.6, label="%d^3" % n)
        ax.axhline(dx2, color=cols[n], lw=0.7, ls="--", alpha=0.6)
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1.0])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("t")
    ax.set_title("%s: analyticity-strip width delta(t)" % title, loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, title="dashed: 2 dx at that resolution", title_fontsize=7.5)
    ax.grid(alpha=0.25, which="both")
axes[0].set_ylabel("delta(t)   (log axis: exponential decay is a straight line)")
fig.suptitle("nu = 0. Each curve runs from the end of its fill-in transient to where it crosses its own 2 dx line. A singularity at t* would bend the curves down toward t*; both flatten instead.", fontsize=8.5, x=0.01, ha="left", y=0.03)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig("figures/strip_decay.png", dpi=130)
print("figures/strip_decay.png")
