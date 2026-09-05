"""One solver for the incompressible Navier-Stokes family in d = 1, 2, 3 dimensions on the periodic box [0,2pi)^d:

    u_t = -P[(u.grad)u] + nu lap(u),   div u = 0        (P = Leray projection; in d=1 there is no projection: Burgers)

Transport in skew-symmetric form  -(1/2)[(u.grad)u_i + div(u u_i)]  (d = 2, 3; in d = 1, where div u != 0, the
energy-conserving form is -(1/3)[u u_x + (u^2)_x])  so the semi-discrete system conserves energy
exactly on the dealiased modes (2/3 rule); viscosity as an exact integrating factor per mode; RK4 in time.

Diagnostics printed at each integer time (same lines in every d):
    E/E0                        energy (conserved exactly when nu = 0)
    Z                           the gradient norm: (1/2)<u_x^2> in 1-D, enstrophy (1/2)<|w|^2> in d = 2, 3
    S                           the unsigned production term in the Z budget:
                                    d=1: -(1/2)<u_x^3>       d=2: 0 identically       d=3: <w.(w.grad)u>  (vortex stretching)
    dZ/dt vs S - nu*dissipation the budget, and its relative residual (closure = the instrument works)
usage:  DIM=1|2|3  N=grid  NU=viscosity  T=final time  IC=tg|sin|random  python ns_d.py
Reproduces: DIM=1 NU=0 IC=sin (Burgers blow-up), DIM=2 NU=0.02 IC=tg (exact decay), DIM=3 NU=0 IC=tg (Brachet)."""

import os, time, numpy as np

DIM = int(os.environ.get("DIM", 3))
N = int(os.environ.get("N", 32))
NU = float(os.environ.get("NU", 0.0))
T = float(os.environ.get("T", 4.0))
IC = os.environ.get("IC", "tg")
DT = float(os.environ.get("DT", 0)) or (2.0 / N if DIM == 3 else 0.5 / N)
k = np.fft.fftfreq(N, d=1.0 / N)
K = np.meshgrid(*([k] * DIM), indexing="ij")
K = [Ki for Ki in K]
k2 = sum(Ki**2 for Ki in K)
k2s = k2.copy()
k2s[(0,) * DIM] = 1.0
deal = np.ones_like(k2, bool)
for Ki in K:
    deal &= np.abs(Ki) < N / 3
x = np.linspace(0, 2 * np.pi, N, endpoint=False)
X = np.meshgrid(*([x] * DIM), indexing="ij")
fft, ifft = np.fft.fftn, np.fft.ifftn


def initial():
    if DIM == 1:
        return [fft(np.sin(X[0]))]
    if DIM == 2:
        if IC == "random":
            rng = np.random.default_rng(0)
            wh = fft(rng.standard_normal(X[0].shape)) * ((k2 >= 9) & (k2 <= 64)) * deal
            psih = -wh / k2s
            psih[0, 0] = 0
            return [1j * K[1] * psih, -1j * K[0] * psih]  # u = psi_y, v = -psi_x
        return [fft(np.sin(X[0]) * np.cos(X[1])), fft(-np.cos(X[0]) * np.sin(X[1]))]
    return [
        fft(np.sin(X[0]) * np.cos(X[1]) * np.cos(X[2])),
        fft(-np.cos(X[0]) * np.sin(X[1]) * np.cos(X[2])),
        fft(np.zeros_like(X[0])),
    ]


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
        out.append(
            -(1.0 / 3.0 if DIM == 1 else 0.5) * fft(adv + div) * deal
        )  # d=1: div u != 0, the skew form is (1/3)[u u_x + (u^2)_x]
    return project(out)


def step(U, dt):
    fac = np.exp(-NU * k2 * dt)
    fac2 = np.exp(-NU * k2 * dt / 2)
    a = transport(U)
    b = transport([fac2 * (U[i] + dt / 2 * a[i]) for i in range(DIM)])
    c = transport([fac2 * U[i] + dt / 2 * b[i] for i in range(DIM)])
    d = transport([fac * U[i] + dt * fac2 * c[i] for i in range(DIM)])
    return [fac * U[i] + dt / 6 * (fac * a[i] + 2 * fac2 * b[i] + 2 * fac2 * c[i] + d[i]) for i in range(DIM)]


def energy(U):
    return 0.5 * sum(np.mean(ifft(Ui).real ** 2) for Ui in U)


def grad_field(U):  # the "vorticity-like" gradient quantity per dimension
    if DIM == 1:
        return [ifft(1j * K[0] * U[0]).real]
    if DIM == 2:
        return [ifft(1j * K[0] * U[1] - 1j * K[1] * U[0]).real]
    return [
        ifft(1j * K[1] * U[2] - 1j * K[2] * U[1]).real,
        ifft(1j * K[2] * U[0] - 1j * K[0] * U[2]).real,
        ifft(1j * K[0] * U[1] - 1j * K[1] * U[0]).real,
    ]


def Zq(U):
    return 0.5 * sum(np.mean(c**2) for c in grad_field(U))


def production(U):
    w = grad_field(U)
    if DIM == 1:
        return -0.5 * np.mean(w[0] ** 3)
    if DIM == 2:
        return 0.0
    Ud = [Ui * deal for Ui in U]
    return sum(np.mean(w[i] * w[j] * ifft(1j * K[j] * Ud[i]).real) for i in range(DIM) for j in range(DIM))


def dissipation(U):  # nu * <|grad of the gradient field|^2>
    w = grad_field(U)
    return NU * sum(np.mean(ifft(1j * K[j] * fft(c)).real ** 2) for c in w for j in range(DIM))


U = initial()
E0 = energy(U)
t = 0.0
t0 = time.time()
mark = 1.0 if DIM != 1 else 0.3
print("DIM=%d  N=%d^%d  nu=%g  IC=%s  dt=%.4g" % (DIM, N, DIM, NU, IC, DT))
print("    t      E/E0        Z          S (production)    dZ/dt measured   S - nu*diss     residual")
while t < T - 1e-9:
    Up = U
    U = step(U, DT)
    t += DT
    if t >= mark - DT / 2:
        mid = step(Up, DT / 2)
        Zb, Za = Zq(Up), Zq(U)
        S = production(mid)
        rhs = S - dissipation(mid)
        dZ = (Za - Zb) / DT
        print(
            "%6.2f   %.6f   %9.5f   %+.5e      %+.5e   %+.5e   %.1e   (%.0fs)"
            % (t, energy(U) / E0, Za, S, dZ, rhs, abs(dZ - rhs) / max(abs(dZ), abs(rhs), 1e-300), time.time() - t0),
            flush=True,
        )
        mark += 1.0 if DIM != 1 else 0.3
