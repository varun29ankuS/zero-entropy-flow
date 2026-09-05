"""Figures for the README, regenerated from the same code that produced the tables. Writes figures/*.png.
  1  burgers_blowup.png   max|u_x|(t) for the frozen scheme and upwind against the exact 1/(1-t); energy drift inset
  2  stretching_3d.png    3-D Euler: vortex stretching S(t) and enstrophy Z(t) at 64^3/96^3/128^3 (from results/)
  3  ladder_2d.png        2-D decaying turbulence: enstrophy falling while palinstrophy production stays large and positive
"""
import os, re, glob, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
INK, ACC, GREY = "#1b1b1b", "#d9731f", "#8a8a8a"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# ------------------------------------------------ 1. Burgers ------------------------------------------------
N = 512
x = np.linspace(0, 2 * np.pi, N, endpoint=False)
dx = x[1] - x[0]
k = np.fft.fftfreq(N, d=1.0 / N)
deal = np.abs(k) < N / 3


def rhs(uh):
    u = np.fft.ifft(uh).real
    ux = np.fft.ifft(1j * k * uh).real
    return -np.fft.fft((u * ux + np.fft.ifft(1j * k * np.fft.fft(u * u)).real) / 3.0) * deal


def frozen(uh, dt):
    a = rhs(uh)
    b = rhs(uh + dt / 2 * a)
    c = rhs(uh + dt / 2 * b)
    d = rhs(uh + dt * c)
    return uh + dt / 6 * (a + 2 * b + 2 * c + d)


def upwind(u, dt):
    f = 0.5 * u * u
    fl = np.where(u > 0, np.roll(f, 1), f)
    fr = np.where(u > 0, f, np.roll(f, -1))
    return u - dt * (fr - fl) / dx


def gmax_fd(u):  # finite-difference gradient, fair to both schemes
    return np.abs((np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)).max()


dt = 2e-4
uh = np.fft.fft(np.sin(x))
uu = np.sin(x)
E0 = np.mean(uu**2)
ts, g_fr, g_up, e_fr, e_up = [], [], [], [], []
t = 0.0
while t < 0.96:
    uh = frozen(uh, dt)
    uu = upwind(uu, dt)
    t += dt
    if int(round(t / dt)) % 25 == 0:
        uf = np.fft.ifft(uh).real
        ts.append(t)
        g_fr.append(gmax_fd(uf))
        g_up.append(gmax_fd(uu))
        e_fr.append(np.mean(uf**2) / E0)
        e_up.append(np.mean(uu**2) / E0)
ts = np.array(ts)
fig, ax = plt.subplots(figsize=(7.2, 4.4))
tt = np.linspace(0.01, 0.96, 300)
ax.plot(tt, 1 / (1 - tt), color=GREY, lw=6, alpha=0.35, label="exact  1/(1-t)")
ax.plot(ts, g_fr, color=INK, lw=1.8, label="frozen-rotation scheme (zero numerical entropy)")
ax.plot(ts, g_up, color=ACC, lw=1.8, ls="--", label="first-order upwind (dissipative)")
ax.set_xlabel("t   (the gradient blows up at t = 1)")
ax.set_ylabel("max |u_x|")
ax.set_yscale("log")
ax.set_title("1-D Burgers: watching a singularity form", loc="left")
ax.legend(frameon=False, loc="upper left")
ins = ax.inset_axes([0.62, 0.09, 0.35, 0.2])
ins.plot(ts, e_fr, color=INK, lw=1.5)
ins.plot(ts, e_up, color=ACC, lw=1.5, ls="--")
ins.set_title("energy / E0  (exact: 1)", fontsize=9)
ins.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig("figures/burgers_blowup.png", dpi=160)
print("figures/burgers_blowup.png")

# ------------------------------------------- 2. 3-D stretching by resolution -------------------------------------------
rows = {}
for f in sorted(glob.glob("results/budgets3d_*.txt")):
    n = int(re.search(r"_(\d+)\.txt", f).group(1))
    for line in open(f):
        m = re.match(r"\s*(\d+\.\d+)\s+([\d.]+)\s+([\d.]+)\s+([+-][\d.]+e[+-]\d+)", line)
        if m:
            rows.setdefault(n, []).append((float(m.group(1)), float(m.group(3)), float(m.group(4))))
if rows:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for n, col in zip(sorted(rows), ["#bbbbbb", "#777777", INK]):
        r = np.array(rows[n])
        a1.plot(r[:, 0], r[:, 2], "o-", color=col, lw=1.8, label="%d$^3$" % n)
        a2.plot(r[:, 0], r[:, 1], "o-", color=col, lw=1.8, label="%d$^3$" % n)
    a1.set_title("vortex stretching  S = <w.(w.grad)u>", loc="left")
    a1.set_xlabel("t")
    a1.set_ylabel("S")
    a2.set_title("enstrophy  Z", loc="left")
    a2.set_xlabel("t")
    a2.set_ylabel("Z")
    for a in (a1, a2):
        a.legend(frameon=False, title="grid")
    fig.suptitle("3-D Euler, Taylor-Green: the term that decides 3-D, converging upward with resolution   (energy = 1.000000 throughout)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig("figures/stretching_3d.png", dpi=160)
    print("figures/stretching_3d.png")

# --------------------------------------------- 3. 2-D ladder ---------------------------------------------
M = 128
km = np.fft.fftfreq(M, d=1.0 / M)
kx, ky = np.meshgrid(km, km, indexing="ij")
k2 = kx * kx + ky * ky
k2s = k2.copy()
k2s[0, 0] = 1.0
deal2 = (np.abs(kx) < M / 3) & (np.abs(ky) < M / 3)
NU = 1e-3


def vel(wh):
    psih = wh / k2s
    psih[0, 0] = 0
    return np.fft.ifft2(1j * ky * psih).real, -np.fft.ifft2(1j * kx * psih).real


def rhs2(wh):
    u, v = vel(wh)
    w = np.fft.ifft2(wh).real
    wx = np.fft.ifft2(1j * kx * wh).real
    wy = np.fft.ifft2(1j * ky * wh).real
    return -0.5 * (np.fft.fft2(u * wx + v * wy) + 1j * kx * np.fft.fft2(u * w) + 1j * ky * np.fft.fft2(v * w)) * deal2 - NU * k2 * wh


def step2(wh, dt):
    a = rhs2(wh)
    b = rhs2(wh + dt / 2 * a)
    c = rhs2(wh + dt / 2 * b)
    d = rhs2(wh + dt * c)
    return wh + dt / 6 * (a + 2 * b + 2 * c + d)


def prod2(wh):
    u, v = vel(wh)
    wx = np.fft.ifft2(1j * kx * wh).real
    wy = np.fft.ifft2(1j * ky * wh).real
    ux = np.fft.ifft2(1j * kx * np.fft.fft2(u)).real
    uy = np.fft.ifft2(1j * ky * np.fft.fft2(u)).real
    vx = np.fft.ifft2(1j * kx * np.fft.fft2(v)).real
    vy = np.fft.ifft2(1j * ky * np.fft.fft2(v)).real
    return -np.mean(wx * (ux * wx + uy * wy) + wy * (vx * wx + vy * wy))


rng = np.random.default_rng(0)
wh = np.fft.fft2(rng.standard_normal((M, M))) * ((k2 >= 9) & (k2 <= 64)) * deal2
u, v = vel(wh)
wh *= 1.0 / np.sqrt(np.mean(u * u + v * v))
dt = 2e-3
t = 0.0
T2, Z2, P2 = [], [], []
while t < 4.0:
    wh = step2(wh, dt)
    t += dt
    if int(round(t / dt)) % 25 == 0:
        T2.append(t)
        Z2.append(0.5 * np.mean(np.fft.ifft2(wh).real ** 2))
        P2.append(prod2(wh))
fig, a1 = plt.subplots(figsize=(7.2, 4.2))
a1.plot(T2, Z2, color=INK, lw=2, label="enstrophy Z  (controlled: never increases)")
a1.set_xlabel("t")
a1.set_ylabel("Z", color=INK)
a2 = a1.twinx()
a2.plot(T2, P2, color=ACC, lw=1.8, label="palinstrophy production  (unsigned, one level up)")
a2.set_ylabel("production term", color=ACC)
a2.spines["right"].set_visible(True)
a1.set_title("2-D turbulence: the dangerous term is large and positive - and regularity holds anyway", loc="left", fontsize=10.5)
h1, l1 = a1.get_legend_handles_labels()
h2, l2 = a2.get_legend_handles_labels()
a1.legend(h1 + h2, l1 + l2, frameon=False, loc="center right")
fig.tight_layout()
fig.savefig("figures/ladder_2d.png", dpi=160)
print("figures/ladder_2d.png")
