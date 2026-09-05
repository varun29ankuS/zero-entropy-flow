"""figures/helicity_threshold.png from results/helicity_proj_*.txt: maximal enstrophy amplification over one time unit
at fixed Z0 versus the relative helicity held fixed by the hard constraint. Search grid (32^3 truncated system) and
the 96^3 verification, hollow where the verification is outside its reliable window."""
import re, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H, S, V, ok = [], [], [], []
for h in ("0.0", "0.25", "0.5", "0.75", "0.9"):
    txt = open("results/helicity_proj_%s.txt" % h).read()
    H.append(float(h))
    S.append(float(re.search(r"best amplification on the search grid: ([\d.]+)", txt).group(1)))
    m = re.search(r"FOUND\s+at 96\^3:\s+Z\(T\)/Z0 = ([\d.]+).*?delta\(T\) = ([\d.]+)\s+\(2dx = ([\d.]+)\)", txt)
    V.append(float(m.group(1))); ok.append(float(m.group(2)) > float(m.group(3)))
H, S, V, ok = map(np.array, (H, S, V, ok))
fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.plot(H, S, "-", color="#4a6fa5", lw=1.6, label="search grid 32^3 (truncated system, exact)")
ax.plot(H, S, "o", color="#4a6fa5", ms=6)
ax.plot(H, V, "-", color="#1b1b1b", lw=1.2, label="same fields at 96^3 (filled = resolved, hollow = past the clock)")
ax.plot(H[ok], V[ok], "o", color="#1b1b1b", ms=6)
ax.plot(H[~ok], V[~ok], "o", mfc="white", mec="#1b1b1b", ms=6)
ax.axhline(1.112, color="#b5432a", lw=0.9, ls="--")
ax.text(0.30, 1.20, "Taylor-Green at the same Z0: 1.112 (best resolved classical flow)", color="#b5432a", fontsize=8)
ax.set_xlabel("relative helicity  H / (2 sqrt(E Z))  held fixed")
ax.set_ylabel("max enstrophy amplification Z(1) / Z0")
ax.set_title("Helicity inhibits the cascade as a threshold, not a slope", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("figures/helicity_threshold.png", dpi=130)
print("figures/helicity_threshold.png")
