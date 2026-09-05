"""Three more exact solutions, one per dimension, compared with the scheme to machine precision where it can.

1-D  viscous Burgers, u0 = sin x, nu = 0.02: the Cole-Hopf closed form (Hopf 1950, Cole 1951), evaluated in its
     heat-kernel form  phi(x,t) = int exp(cos y / 2nu) G(x-y,t) dy,  u = -2 nu phi_x / phi  (all terms positive, so no
     cancellation near the shock; the equivalent Bessel series cancels catastrophically there).
     A viscous shock forms near x = pi around t ~ 1: the hardest closed form a 1-D scheme can be asked to track.
2-D  a RANDOM vorticity field built only from modes with |k|^2 = 25 (a single Laplacian shell): the nonlinear term
     vanishes identically (J(psi, -k^2 psi) = 0), so the exact Navier-Stokes solution is u0 * e^{-25 nu t}.
3-D  the ABC flow (Beltrami, curl u = u) with viscosity: exact solution u0 * e^{-nu t}.
Prints L2 error against the exact field at several times, and draws figures/exact_solutions.png."""
import os, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fft, ifft = np.fft.fftn, np.fft.ifftn
os.makedirs("figures", exist_ok=True)
INK, ACC, GREY = "#1b1b1b", "#d9731f", "#8a8a8a"


def solver(DIM, N, nu):
    k = np.fft.fftfreq(N, d=1.0 / N)
    K = list(np.meshgrid(*([k] * DIM), indexing="ij"))
    k2 = sum(Ki**2 for Ki in K)
    k2s = k2.copy()
    k2s[(0,) * DIM] = 1.0
    deal = np.ones_like(k2, bool)
    for Ki in K:
        deal &= np.abs(Ki) < N / 3

    def project(F):
        if DIM == 1:
            return F
        kdotF = sum(K[i] * F[i] for i in range(DIM)) / k2s
        return [F[i] - K[i] * kdotF for i in range(DIM)]

    def transport(U):
        Ud = [Ui * deal for Ui in U]
        u = [ifft(Ui).real for Ui in Ud]
        out = []
        for i in range(DIM):
            adv = sum(u[j] * ifft(1j * K[j] * Ud[i]).real for j in range(DIM))
            div = sum(ifft(1j * K[j] * fft(u[j] * u[i])).real for j in range(DIM))
            out.append(-(1.0 / 3.0 if DIM == 1 else 0.5) * fft(adv + div) * deal)
        return project(out)

    fac = lambda dt: np.exp(-nu * k2 * dt)

    def step(U, dt):
        f, f2 = fac(dt), fac(dt / 2)
        a = transport(U)
        b = transport([f2 * (U[i] + dt / 2 * a[i]) for i in range(DIM)])
        c = transport([f2 * U[i] + dt / 2 * b[i] for i in range(DIM)])
        d = transport([f * U[i] + dt * f2 * c[i] for i in range(DIM)])
        return [f * U[i] + dt / 6 * (f * a[i] + 2 * f2 * b[i] + 2 * f2 * c[i] + d[i]) for i in range(DIM)]

    return K, k2, deal, project, step


def l2(A, B):
    return np.sqrt(sum(np.mean((a - b) ** 2) for a, b in zip(A, B)))


# ------------------------------------------------ 1-D Cole-Hopf ------------------------------------------------
N, nu, dt = 512, 0.02, 1e-4
x = np.linspace(0, 2 * np.pi, N, endpoint=False)
K, k2, deal, project, step = solver(1, N, nu)
# Cole-Hopf in heat-kernel form (every term positive: no cancellation near the shock, unlike the Bessel series):
#   phi(x,t) = int phi0(y) G(x-y,t) dy,  phi0(y) = exp(cos y / (2 nu)),  G periodic heat kernel;  u = -2 nu phi_x / phi
yq = np.linspace(0, 2 * np.pi, 8192, endpoint=False)
dy = yq[1] - yq[0]
logphi0 = np.cos(yq) / (2 * nu)


def cole_hopf(t):
    u = np.empty_like(x)
    for i, xi in enumerate(x):
        d = xi - yq[None, :] + 2 * np.pi * np.arange(-3, 4)[:, None]      # periodic images
        logG = -d**2 / (4 * nu * t)
        w = logphi0[None, :] + logG
        w -= w.max()
        e = np.exp(w)
        phi = e.sum()
        phix = (e * (-d / (2 * nu * t))).sum()
        u[i] = -2 * nu * phix / phi
    return u


U = [fft(np.sin(x))]
t = 0.0
rows1 = []
print("1-D viscous Burgers vs Cole-Hopf, nu = 0.02, N = 512")
for tc in (0.5, 1.0, 1.5, 2.0):
    while t < tc - 1e-12:
        U = step(U, dt)
        t += dt
    ex = cole_hopf(t)
    err = np.sqrt(np.mean((ifft(U[0]).real - ex) ** 2))
    rows1.append((t, ifft(U[0]).real.copy(), ex, err))
    print("  t = %.1f   L2 error %.2e   max|u_x| exact %.2f" % (t, err, np.abs(np.gradient(ex, x)).max()))

# ------------------------------------------- 2-D single-shell random field -------------------------------------------
M, nu2, dt2 = 128, 0.01, 1e-3
K2, k2_2, deal2, project2, step2 = solver(2, M, nu2)
rng = np.random.default_rng(0)
shell = k2_2 == 25
wh = fft(rng.standard_normal((M, M))) * shell
psih = -wh / np.where(k2_2 == 0, 1, k2_2)
U0 = [1j * K2[1] * psih, -1j * K2[0] * psih]
U0 = [Ui / np.sqrt(sum(np.mean(ifft(V).real ** 2) for V in U0)) for Ui in U0]
U = [Ui.copy() for Ui in U0]
t = 0.0
rows2 = []
print("2-D random single-shell field |k|^2 = 25, nu = 0.01, exact decay e^{-25 nu t}")
for tc in (0.5, 1.0, 2.0, 4.0):
    while t < tc - 1e-12:
        U = step2(U, dt2)
        t += dt2
    ex = [np.exp(-25 * nu2 * t) * ifft(Ui).real for Ui in U0]
    err = l2([ifft(Ui).real for Ui in U], ex)
    rows2.append((t, err, np.exp(-50 * nu2 * t), 0.5 * sum(np.mean(ifft(Ui).real ** 2) for Ui in U) / 0.5))
    print("  t = %.1f   L2 error %.2e   E/E0 scheme %.8f  exact %.8f" % (t, err, rows2[-1][3], rows2[-1][2]))

# ----------------------------------------------- 3-D ABC with viscosity -----------------------------------------------
N3, nu3 = 32, 0.01
dt3 = 1.0 / N3
K3, k2_3, deal3, project3, step3 = solver(3, N3, nu3)
x3 = np.linspace(0, 2 * np.pi, N3, endpoint=False)
X, Y, Z = np.meshgrid(x3, x3, x3, indexing="ij")
U0 = [fft(np.sin(Z) + np.cos(Y)), fft(np.sin(X) + np.cos(Z)), fft(np.sin(Y) + np.cos(X))]
U = [Ui.copy() for Ui in U0]
t = 0.0
rows3 = []
print("3-D ABC flow with viscosity, nu = 0.01, exact decay e^{-nu t}, N = 32^3")
for tc in (1.0, 2.0, 4.0, 8.0):
    while t < tc - 1e-12:
        U = step3(U, dt3)
        t += dt3
    ex = [np.exp(-nu3 * t) * ifft(Ui).real for Ui in U0]
    err = l2([ifft(Ui).real for Ui in U], ex)
    rows3.append((t, err, np.exp(-2 * nu3 * t)))
    print("  t = %.1f   L2 error %.2e   E/E0 exact %.6f" % (t, err, np.exp(-2 * nu3 * t)))

# ----------------------------------------------------- figure -----------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
a = axes[0]
for (t, u, ex, err), col in zip(rows1, ["#cccccc", "#999999", "#555555", INK]):
    a.plot(x, ex, color=col, lw=5, alpha=0.35)
    a.plot(x, u, color=col, lw=1.2, label="t = %.1f   error %.0e" % (t, err))
a.set_title("1-D viscous Burgers vs Cole-Hopf exact (thick = exact)", loc="left", fontsize=10)
a.set_xlabel("x")
a.set_ylabel("u")
a.legend(frameon=False, fontsize=8)
a = axes[1]
a.semilogy([r[0] for r in rows2], [r[1] for r in rows2], "o-", color=INK, label="L2 error vs exact")
a.set_title("2-D random single-shell field: exact e$^{-25\\nu t}$", loc="left", fontsize=10)
a.set_xlabel("t")
a.set_ylabel("error")
a.legend(frameon=False)
a = axes[2]
a.semilogy([r[0] for r in rows3], [r[1] for r in rows3], "o-", color=INK, label="L2 error vs exact")
a.set_title("3-D ABC with viscosity: exact e$^{-\\nu t}$", loc="left", fontsize=10)
a.set_xlabel("t")
a.set_ylabel("error")
a.legend(frameon=False)
fig.suptitle("Closed-form Navier-Stokes solutions in one, two and three dimensions, and the scheme's error against each", x=0.01, ha="left", fontsize=10)
fig.tight_layout()
fig.savefig("figures/exact_solutions.png", dpi=150)
print("figures/exact_solutions.png")
