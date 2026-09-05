"""Exact discrete energy conservation: the implicit midpoint rule on the skew transport.

RK4 conserves energy only to O(dt^4). The implicit midpoint rule
        u^{n+1} = u^n + dt * T( (u^n + u^{n+1}) / 2 )
conserves it EXACTLY for any skew T, because <u^{n+1} + u^n, T(mid)> = 0 gives |u^{n+1}|^2 = |u^n|^2 identically.
Solved by fixed-point iteration (a handful of sweeps at these CFL numbers). Viscosity, when present, stays an exact
integrating factor applied around the midpoint step (Strang split), so the only remaining dissipation is physical.

Compares RK4 and midpoint on: 1-D inviscid Burgers to t = 0.9; 3-D Euler Taylor-Green 32^3 to t = 4.
usage: DIM=1|3 python midpoint.py   (defaults run both)"""
import os, time, numpy as np

fft, ifft = np.fft.fftn, np.fft.ifftn


def make(DIM, N):
    k = np.fft.fftfreq(N, d=1.0 / N)
    K = list(np.meshgrid(*([k] * DIM), indexing="ij"))
    k2 = sum(Ki**2 for Ki in K)
    k2s = k2.copy()
    k2s[(0,) * DIM] = 1.0
    deal = np.ones_like(k2, bool)
    for Ki in K:
        deal &= np.abs(Ki) < N / 3
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    X = np.meshgrid(*([x] * DIM), indexing="ij")

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

    def rk4(U, dt):
        a = transport(U)
        b = transport([U[i] + dt / 2 * a[i] for i in range(DIM)])
        c = transport([U[i] + dt / 2 * b[i] for i in range(DIM)])
        d = transport([U[i] + dt * c[i] for i in range(DIM)])
        return [U[i] + dt / 6 * (a[i] + 2 * b[i] + 2 * c[i] + d[i]) for i in range(DIM)]

    def midpoint(U, dt, sweeps=6):
        Un = rk4(U, dt)  # predictor
        for _ in range(sweeps):
            Tm = transport([(U[i] + Un[i]) / 2 for i in range(DIM)])
            Un = [U[i] + dt * Tm[i] for i in range(DIM)]
        return Un

    def energy(U):
        return 0.5 * sum(np.mean(ifft(Ui).real ** 2) for Ui in U)

    if DIM == 1:
        U0 = [fft(np.sin(X[0]))]
    else:
        U0 = [fft(np.sin(X[0]) * np.cos(X[1]) * np.cos(X[2])), fft(-np.cos(X[0]) * np.sin(X[1]) * np.cos(X[2])), fft(np.zeros_like(X[0]))]
    return U0, rk4, midpoint, energy


for DIM, N, T, dt in ((1, 512, 0.9, 2e-4), (3, 32, 4.0, 2.0 / 32)):
    if os.environ.get("DIM") and int(os.environ["DIM"]) != DIM:
        continue
    U0, rk4, midpoint, energy = make(DIM, N)
    E0 = energy(U0)
    print("\nDIM=%d N=%d^%d  dt=%.4g  energy drift |E/E0 - 1| at the end" % (DIM, N, DIM, dt))
    for name, stepper in (("RK4", rk4), ("implicit midpoint", midpoint)):
        U = [u.copy() for u in U0]
        t = 0.0
        t0 = time.time()
        worst = 0.0
        while t < T - 1e-12:
            U = stepper(U, dt)
            t += dt
            worst = max(worst, abs(energy(U) / E0 - 1))
        print("  %-18s max |E/E0 - 1| over the run = %.2e   (%.0fs)" % (name, worst, time.time() - t0), flush=True)
