"""3-D Euler enstrophy budget with the vortex-stretching term measured directly, Taylor-Green initial data.
   d/dt Z = S - nu <|grad w|^2>,   Z = (1/2)<|w|^2>,   S = < w . (w . grad) u >   (unsigned: the term that decides 3-D)
With nu = 0 the budget is dZ/dt = S. The frozen scheme (skew-symmetric transport, Leray projection, RK4, 2/3 rule)
conserves energy exactly, so any residual in the enstrophy budget is time-stepping error, not hidden dissipation.
Reported at each integer time: E/E0, Z, S (measured from the field), dZ/dt (centred across one step), relative
residual, and the normalised stretching S / Z^{3/2} (the classical a-priori bound has dZ/dt <= c Z^{3/2}; a rising
ratio means the flow is amplifying faster than the bound's worst case would allow to be sustained).
usage: N=64 T=4 python budgets3d.py"""

import os, time, numpy as np

N = int(os.environ.get("N", 48))
T = float(os.environ.get("T", 4.0))
dt = 2.0 / N
k = np.fft.fftfreq(N, d=1.0 / N)
kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
K = [kx, ky, kz]
k2 = kx**2 + ky**2 + kz**2
k2[0, 0, 0] = 1.0
deal = (np.abs(kx) < N / 3) & (np.abs(ky) < N / 3) & (np.abs(kz) < N / 3)
x = np.linspace(0, 2 * np.pi, N, endpoint=False)
X, Y, Z_ = np.meshgrid(x, x, x, indexing="ij")
U = [
    np.fft.fftn(np.sin(X) * np.cos(Y) * np.cos(Z_)),
    np.fft.fftn(-np.cos(X) * np.sin(Y) * np.cos(Z_)),
    np.fft.fftn(np.zeros_like(X)),
]


def project(F):
    kdotF = sum(K[i] * F[i] for i in range(3)) / k2
    return [F[i] - K[i] * kdotF for i in range(3)]


def rhs(U):
    Ud = [Ui * deal for Ui in U]
    u = [np.fft.ifftn(Ui).real for Ui in Ud]
    out = []
    for i in range(3):
        adv = sum(u[j] * np.fft.ifftn(1j * K[j] * Ud[i]).real for j in range(3))
        div = sum(np.fft.ifftn(1j * K[j] * np.fft.fftn(u[j] * u[i])).real for j in range(3))
        out.append(-0.5 * np.fft.fftn(adv + div) * deal)
    return project(out)


def step(U, dt):
    a = rhs(U)
    b = rhs([U[i] + dt / 2 * a[i] for i in range(3)])
    c = rhs([U[i] + dt / 2 * b[i] for i in range(3)])
    d = rhs([U[i] + dt * c[i] for i in range(3)])
    return [U[i] + dt / 6 * (a[i] + 2 * b[i] + 2 * c[i] + d[i]) for i in range(3)]


def vort(U):
    return [
        np.fft.ifftn(1j * ky * U[2] - 1j * kz * U[1]).real,
        np.fft.ifftn(1j * kz * U[0] - 1j * kx * U[2]).real,
        np.fft.ifftn(1j * kx * U[1] - 1j * ky * U[0]).real,
    ]


def energy(U):
    return 0.5 * sum(np.mean(np.fft.ifftn(Ui).real ** 2) for Ui in U)


def enstrophy(U):
    w = vort(U)
    return 0.5 * sum(np.mean(c**2) for c in w)


def stretching(U):
    w = vort(U)
    Ud = [Ui * deal for Ui in U]
    S = 0.0
    for i in range(3):
        for j in range(3):
            S += np.mean(w[i] * w[j] * np.fft.ifftn(1j * K[j] * Ud[i]).real)  # w_i w_j d_j u_i
    return S


E0 = energy(U)
t = 0.0
t0 = time.time()
print("N=%d^3  dt=%.4f  Taylor-Green Euler enstrophy budget: dZ/dt = S (vortex stretching), nu = 0" % (N, dt))
print("   t     E/E0        Z        S=<w.(w.grad)u>   dZ/dt measured   residual   S/Z^1.5")
next_mark = 1.0
while t < T - 1e-9:
    Up = U
    U = step(U, dt)
    t += dt
    if t >= next_mark - dt / 2:
        mid = step(Up, dt / 2)
        Zb, Za = enstrophy(Up), enstrophy(U)
        S = stretching(mid)
        dZ = (Za - Zb) / dt
        print(
            "%5.2f   %.6f   %8.4f   %+.5e     %+.5e    %.1e    %.3f   (%.0fs)"
            % (t, energy(U) / E0, Za, S, dZ, abs(dZ - S) / max(abs(S), abs(dZ), 1e-300), S / Za**1.5, time.time() - t0),
            flush=True,
        )
        next_mark += 1.0
