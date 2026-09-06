"""The price of the projection: the same nonlinearity with and without the Leray projection, from the same data.

Euler is  u_t = -P[(u.grad)u]  with P the projection onto divergence-free fields; the pressure IS the projection, and
it is the only nonlocal thing in the equation. Drop it and the equation is 3-D Burgers, u_t = -(u.grad)u, which is
purely local and blows up in finite time by a theorem. It does not conserve energy (dE/dt = 1/2 int |u|^2 div u):
Euler's energy conservation is itself a consequence of the projection, and the energy column shows it. The theorem: along characteristics the gradient A = grad u obeys
DA/Dt = -A^2, so each eigenvalue evolves as lambda0 / (1 + lambda0 t) and the first shock is at
        t* = -1 / min_x lambda_min(grad u0)              (exact)
Both systems are run with the same instrument from the same initial field, and the analyticity width delta(t) is
tracked for both. For Burgers the clock must hit its floor at the predicted t*: a sixth closed-form check of the
instrument, in 3-D, on a blow-up. For Euler it does what it does. The difference is the projection.
usage: IC=tg|kp|found N=48 T=1.5 python projection_price.py"""
import os, glob, time, numpy as np

IC = os.environ.get("IC", "tg")
N = int(os.environ.get("N", 48))
T = float(os.environ.get("T", 1.5))
fft, ifft = np.fft.fftn, np.fft.ifftn
k = np.fft.fftfreq(N, d=1.0 / N)
kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
K = [kx, ky, kz]
k2 = kx**2 + ky**2 + kz**2
k2s = k2.copy()
k2s[0, 0, 0] = 1.0
deal = (np.abs(kx) < N / 3) & (np.abs(ky) < N / 3) & (np.abs(kz) < N / 3)
x = np.linspace(0, 2 * np.pi, N, endpoint=False)
X, Y, Z_ = np.meshgrid(x, x, x, indexing="ij")
KMAG = np.sqrt(k2)
NB = int(N / 3)
shells = [(KMAG >= n - 0.5) & (KMAG < n + 0.5) for n in range(1, NB)]


def project(F):
    kd = sum(K[i] * F[i] for i in range(3)) / k2s
    return [F[i] - K[i] * kd for i in range(3)]


def transport(U, projected):
    Ud = [Ui * deal for Ui in U]
    u = [ifft(Ui).real for Ui in Ud]
    out = []
    for i in range(3):
        adv = sum(u[j] * ifft(1j * K[j] * Ud[i]).real for j in range(3))
        div = sum(ifft(1j * K[j] * fft(u[j] * u[i])).real for j in range(3))
        out.append(-0.5 * fft(adv + div) * deal if projected else -fft(adv) * deal)   # unprojected: true Burgers (advective form)
    return project(out) if projected else out


def step(U, dt, projected):
    a = transport(U, projected)
    b = transport([U[i] + dt / 2 * a[i] for i in range(3)], projected)
    c = transport([U[i] + dt / 2 * b[i] for i in range(3)], projected)
    d = transport([U[i] + dt * c[i] for i in range(3)], projected)
    return [U[i] + dt / 6 * (a[i] + 2 * b[i] + 2 * c[i] + d[i]) for i in range(3)]


def strip(U):
    e = 0.5 * sum(np.abs(Ui) ** 2 for Ui in U) / N**6
    spec = np.array([e[s].sum() for s in shells])
    ks = np.arange(1, NB)
    sel = (ks >= NB // 2) & (spec > 1e-300)
    if sel.sum() < 4:
        return np.nan
    return -np.polyfit(ks[sel], np.log(spec[sel]), 1)[0] / 2.0


def gradmax(U):
    G = np.stack([np.stack([ifft(1j * K[i] * U[j] * deal).real for j in range(3)], -1) for i in range(3)], -1)
    lam = np.linalg.eigvalsh(0.5 * (G + np.swapaxes(G, -1, -2)))       # symmetric part; for the shock time the full-A eigenvalues are used below
    lam_full = np.linalg.eigvals(G).real
    return np.abs(G).max(), lam_full.min()


if IC == "kp":
    U0 = [fft(np.sin(X) * (np.cos(3 * Y) * np.cos(Z_) - np.cos(Y) * np.cos(3 * Z_))),
          fft(np.sin(Y) * (np.cos(3 * Z_) * np.cos(X) - np.cos(Z_) * np.cos(3 * X))),
          fft(np.sin(Z_) * (np.cos(3 * X) * np.cos(Y) - np.cos(X) * np.cos(3 * Y)))]
elif IC == "found":
    path = sorted(glob.glob("results/found/*.npz"))[-1]
    u = np.load(path)["u"].astype(float)
    n0 = u.shape[1]
    U0 = []
    for c in range(3):
        uh = fft(u[c]) * (N / n0) ** 3
        big = np.zeros((N, N, N), complex)
        h = n0 // 2
        for a in (slice(0, h), slice(-h, None)):
            for b in (slice(0, h), slice(-h, None)):
                for cc in (slice(0, h), slice(-h, None)):
                    big[a, b, cc] = uh[a, b, cc]
        U0.append(big)
    U0 = project(U0)
    print("found field:", path)
else:
    U0 = [fft(np.sin(X) * np.cos(Y) * np.cos(Z_)), fft(-np.cos(X) * np.sin(Y) * np.cos(Z_)), fft(np.zeros_like(X))]
Z0 = 0.375
Zi = 0.5 * np.mean(sum(ifft(1j * K[a] * U0[b] - 1j * K[b] * U0[a]).real ** 2 for a, b in ((1, 2), (2, 0), (0, 1))))
U0 = [Ui * np.sqrt(Z0 / Zi) for Ui in U0]
E0 = 0.5 * sum(np.mean(ifft(Ui).real ** 2) for Ui in U0)
gm0, lmin0 = gradmax(U0)
tstar = -1.0 / lmin0
A0 = np.stack([np.stack([ifft(1j * K[i] * U0[j] * deal).real for j in range(3)], -1) for i in range(3)], -1)   # [N,N,N,3,3]


def burgers_exact_gradmax(t):
    """A(t) = A0 (I + A0 t)^-1 along characteristics: the set of gradient values at time t is exactly this set over x"""
    if t >= tstar:
        return float("inf")
    M = np.eye(3) + A0 * t
    return np.abs(A0 @ np.linalg.inv(M)).max()
dx2 = 2 * 2 * np.pi / N
print("IC=%s  N=%d^3  nu=0  Z0=%.3f   2dx = %.4f" % (IC, N, Z0, dx2))
print("Burgers (unprojected) exact shock time from the initial gradient: t* = -1/lambda_min(grad u0) = %.4f" % tstar)
print("   t      Euler: E/E0     delta     rate      max|grad u|   |   Burgers: E/E0     delta     rate      max|grad u|   exact A0(I+A0 t)^-1")
dprevE = dprevB = None
UE, UB = [Ui.copy() for Ui in U0], [Ui.copy() for Ui in U0]
t, mark, t0 = 0.0, 0.0, time.time()
stopB = False
while t < T - 1e-9:
    if t >= mark - 1e-9:
        dE, gE = strip(UE), gradmax(UE)[0]
        EE = 0.5 * sum(np.mean(ifft(Ui).real ** 2) for Ui in UE)
        if not stopB:
            dB, gB = strip(UB), gradmax(UB)[0]
            EB = 0.5 * sum(np.mean(ifft(Ui).real ** 2) for Ui in UB)
        pred = burgers_exact_gradmax(t)
        rE = -np.log(dE / dprevE) / 0.1 if dprevE else float("nan")
        rB = (-np.log(dB / dprevB) / 0.1 if dprevB else float("nan")) if not stopB else float("nan")
        print("%5.2f   %.8f   %.4f%s  %6.2f   %7.3f   |   %s" % (t, EE / E0, dE, "*" if dE < dx2 else " ", rE, gE,
              ("%.8f   %.4f%s  %6.2f   %7.3f   %7.3f" % (EB / E0, dB, "*" if dB < dx2 else " ", rB, gB, pred)) if not stopB else "stopped at the grid limit (shock)"), flush=True)
        dprevE = dE
        if not stopB:
            dprevB = dB
        mark += 0.1
        if not stopB and dB < dx2 and t > 0.2:
            stopB = True
            print("        Burgers strip below 2dx at t = %.2f; exact shock time %.4f (the clock stops %.0f%% of the way there)" % (t, tstar, 100 * t / tstar))
    umax = max(np.abs(ifft(Ui).real).max() for Ui in UE + (UB if not stopB else []))
    dt = min(2.0 / N, 0.5 * (2 * np.pi / N) / max(umax, 1e-9), mark - t + 1e-12)
    UE = step(UE, dt, True)
    if not stopB:
        UB = step(UB, dt, False)
    t += dt
print("(%.0fs)   * = delta below 2dx, not reliable" % (time.time() - t0))
