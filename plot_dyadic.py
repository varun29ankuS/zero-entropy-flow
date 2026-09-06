"""figures/taos_wall.png: the dyadic model (energy-conserving, proven blow-up) and Kida-Pelz Euler (open question), through
the same diagnostic. Left: delta(t) on a linear axis for the dyadic model - a straight line to zero at t*. Middle: the
local decay rate -d log(delta)/dt against the fraction of the reliable window, for both: rising without bound for the
dyadic model, falling for Kida-Pelz at 64/96/128^3. Right: enstrophy against t* - t for the dyadic model, log-log."""
import re, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("dyadic_inviscid.npz")
ts, ds, Zs = d["ts"], d["ds"], d["Zs"]
h = ts > ts[0] + 0.5 * (ts[-1] - ts[0])
pl = np.polyfit(ts[h], ds[h], 1)
tstar = -pl[1] / pl[0]


def kp(path):
    t, dd, dx2 = [], [], None
    for line in open(path):
        m = re.match(r"\s*(\d+\.\d+)\s+(-?\d+\.\d+)\s+", line)
        if m:
            t.append(float(m.group(1))); dd.append(float(m.group(2)))
        m2 = re.search(r"2dx = (\d+\.\d+)", line)
        if m2:
            dx2 = float(m2.group(1))
    t, dd = np.array(t), np.array(dd)
    i0 = int(np.argmax(np.where(t < 1.0, dd, -np.inf)))
    ok = (dd > dx2) & (np.arange(len(t)) >= i0)
    return t[ok], dd[ok]


fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
ax[0].plot(ts, ds, color="#b5432a", lw=1.8, label="dyadic model, nu = 0")
tt = np.linspace(ts[0], tstar, 50)
ax[0].plot(tt, np.polyval(pl, tt), "--", color="#1b1b1b", lw=1, label="second-half linear fit, t* = %.4f" % tstar)
ax[0].set_ylim(0, ds.max() * 1.05)
ax[0].set_xlabel("t")
ax[0].set_ylabel("delta(t)  (linear axis)")
ax[0].set_title("Energy-conserving, provably singular: delta goes to zero", loc="left", fontsize=9.5)
ax[0].legend(frameon=False, fontsize=8)

rate = -np.gradient(np.log(ds), ts)
frac = (ts - ts[0]) / (ts[-1] - ts[0])
sm = np.convolve(rate, np.ones(9) / 9, mode="same")
ax[1].plot(frac[5:-5], sm[5:-5], color="#b5432a", lw=1.8, label="dyadic model")
cols = {64: "#9aa5b1", 96: "#4a6fa5", 128: "#1b1b1b"}
for n in (64, 96, 128):
    t, dd = kp("results/strip_kp_%d.txt" % n)
    r = -np.gradient(np.log(dd), t)
    f = (t - t[0]) / (t[-1] - t[0])
    ax[1].plot(f, r, color=cols[n], lw=1.4, label="Kida-Pelz Euler %d^3" % n)
ax[1].set_yscale("log")
ax[1].set_xlabel("fraction of the reliable window")
ax[1].set_ylabel("-d log(delta)/dt   (log axis)")
ax[1].set_title("The discriminator: decay rate rising without bound vs flat", loc="left", fontsize=9.5)
ax[1].legend(frameon=False, fontsize=7.5)
ax[1].grid(alpha=0.25, which="both")

sel = h & (tstar - ts > 2e-3)      # before the front reaches the truncation
ax[2].loglog(tstar - ts[sel], Zs[sel], color="#b5432a", lw=1.8)
p = -np.polyfit(np.log(tstar - ts[sel]), np.log(Zs[sel]), 1)[0]
ax[2].set_xlabel("t* - t")
ax[2].set_ylabel("enstrophy  sum k^2 a^2")
ax[2].invert_xaxis()
ax[2].set_title("Dyadic enstrophy ~ (t* - t)^-%.2f  (energy drift 3e-8)" % p, loc="left", fontsize=9.5)
ax[2].grid(alpha=0.25, which="both")
fig.suptitle("Theorem 4 (Tao) in pictures: exact energy conservation does not prevent blow-up. The same instrument, the same clock, opposite trends.", fontsize=8.5, x=0.01, ha="left", y=0.03)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig("figures/taos_wall.png", dpi=130)
print("figures/taos_wall.png")
