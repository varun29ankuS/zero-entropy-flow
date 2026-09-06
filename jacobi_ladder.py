"""Arnold's picture, measured: Euler flow is a geodesic on the group of volume-preserving maps, and the growth of a
Jacobi field (the separation of two nearby geodesics) is governed by the sectional curvature (Arnold 1966).
Negative curvature => exponential separation => the flow forgets its initial data at a rate the curvature sets.

Measured here with two copies of the numpy instrument: u (Taylor-Green or Kida-Pelz) and u + eps*v, v a
random solenoidal low-k field. Reported per half-time: the separation growth ||du(t)|| / (eps ||v||), its local
exponent lambda = d/dt log(growth) (a finite-time Lyapunov exponent in the L2 metric = Arnold's metric), the
analyticity-strip width delta(t) with the 2dx reliability flag, and E(t)/E0 for both copies (must stay 1 with nu=0).
Reversal (REVERSE=1): Euler is time-reversible, and by Liouville the Jacobian's exponents sum to zero, so the most
CONTRACTING direction of the forward flow is the most expanding direction of the reversed flow. After reaching T the
velocity is flipped and the ladder is run again for T: the reversed flow must return to the initial field (an exactness
check, printed), and its Jacobi exponent is the compression half of the forward stretching. A sheet stretches along
itself and compresses across its thickness; a collapse would be compression outrunning stretching (lambda_b > lambda_f).
usage: IC=tg|kp|found N=32|48|64 T=2 EPS=1e-5 [REVERSE=1] python jacobi_ladder.py"""
import os, glob, time, numpy as np

IC = os.environ.get("IC", "tg")
N = int(os.environ.get("N", 32))
T = float(os.environ.get("T", 2.0))
EPS = float(os.environ.get("EPS", 1e-5))
NU = float(os.environ.get("NU", 0.0))
REVERSE = int(os.environ.get("REVERSE", 0))
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


def project(F):
    kd = sum(K[i] * F[i] for i in range(3)) / k2s
    return [F[i] - K[i] * kd for i in range(3)]


def transport(U):
    Ud = [Ui * deal for Ui in U]
    u = [ifft(Ui).real for Ui in Ud]
    out = []
    for i in range(3):
        adv = sum(u[j] * ifft(1j * K[j] * Ud[i]).real for j in range(3))
        div = sum(ifft(1j * K[j] * fft(u[j] * u[i])).real for j in range(3))
        out.append(-0.5 * fft(adv + div) * deal)
    return project(out)


def step(U, dt):
    f, f2 = np.exp(-NU * k2 * dt), np.exp(-NU * k2 * dt / 2)
    a = transport(U)
    b = transport([f2 * (U[i] + dt / 2 * a[i]) for i in range(3)])
    c = transport([f2 * U[i] + dt / 2 * b[i] for i in range(3)])
    d = transport([f * U[i] + dt * f2 * c[i] for i in range(3)])
    return [f * U[i] + dt / 6 * (f * a[i] + 2 * f2 * b[i] + 2 * f2 * c[i] + d[i]) for i in range(3)]


def energy(U):
    return 0.5 * sum(np.mean(ifft(Ui).real ** 2) for Ui in U)


def norm(U):
    return np.sqrt(sum(np.mean(ifft(Ui).real ** 2) for Ui in U))


def delta_strip(U):
    e = 0.5 * sum(np.abs(Ui) ** 2 for Ui in U) / N**6
    spec = np.array([e[(KMAG >= n - 0.5) & (KMAG < n + 0.5)].sum() for n in range(1, NB)])
    ks = np.arange(1, NB)
    sel = (ks >= NB // 2) & (spec > 0)
    if sel.sum() < 4:
        return np.nan
    return -np.polyfit(ks[sel], np.log(spec[sel]), 1)[0] / 2.0


if IC == "kp":
    U = [fft(np.sin(X) * (np.cos(3 * Y) * np.cos(Z_) - np.cos(Y) * np.cos(3 * Z_))),
         fft(np.sin(Y) * (np.cos(3 * Z_) * np.cos(X) - np.cos(Z_) * np.cos(3 * X))),
         fft(np.sin(Z_) * (np.cos(3 * X) * np.cos(Y) - np.cos(X) * np.cos(3 * Y)))]
elif IC == "found":
    path = sorted(glob.glob("results/found/*.npz"))[-1]
    uf = np.load(path)["u"].astype(float)
    n0 = uf.shape[1]
    U = []
    for c in range(3):
        uh = fft(uf[c]) * (N / n0) ** 3
        big = np.zeros((N, N, N), complex)
        h = n0 // 2
        for a in (slice(0, h), slice(-h, None)):
            for b in (slice(0, h), slice(-h, None)):
                for cc in (slice(0, h), slice(-h, None)):
                    big[a, b, cc] = uh[a, b, cc]
        U.append(big)
    U = project(U)
    Zi = 0.5 * np.mean(sum(ifft(1j * K[a] * U[b] - 1j * K[b] * U[a]).real ** 2 for a, b in ((1, 2), (2, 0), (0, 1))))
    U = [Ui * np.sqrt(0.375 / Zi) for Ui in U]
else:
    U = [fft(np.sin(X) * np.cos(Y) * np.cos(Z_)), fft(-np.cos(X) * np.sin(Y) * np.cos(Z_)), fft(np.zeros_like(X))]
U_init = [Ui.copy() for Ui in U]
rng = np.random.default_rng(0)
V = project([fft(rng.standard_normal(X.shape)) * ((k2 >= 1) & (k2 <= 9)) * deal for _ in range(3)])
V = [Vi / norm(V) for Vi in V]
W = [U[i] + EPS * V[i] for i in range(3)]
E0, E0w = energy(U), energy(W)


def cfl_dt(U):
    umax = max(np.abs(ifft(Ui).real).max() for Ui in U)
    return min(2.0 / N, 0.5 * (2 * np.pi / N) / max(umax, 1e-9))


def ladder(U, W, T, label):
    E0, E0w = energy(U), energy(W)
    print("%s: Jacobi field, N=%d^3, nu=%g, eps=%g, 2dx = %.4f" % (label, N, NU, EPS, 2 * 2 * np.pi / N))
    print("   t     growth |du|/(eps|v|)   lambda (local, per unit t)   E/E0 (u)      E/E0 (u+eps v)   delta(t)")
    t, mark, t0 = 0.0, 0.5, time.time()
    g_prev, t_prev = 1.0, 0.0
    lams = []
    print("%5.2f   %10.4f            %s                   %.8f    %.8f      %.4f" % (0, 1.0, "  ---  ", 1, 1, delta_strip(U)), flush=True)
    while t < T - 1e-9:
        dt = min(cfl_dt(U), cfl_dt(W))
        if t + dt > mark:
            dt = mark - t + 1e-12
        U, W = step(U, dt), step(W, dt)
        t += dt
        if t >= mark - 1e-9:
            g = norm([W[i] - U[i] for i in range(3)]) / EPS
            lam = np.log(g / g_prev) / (t - t_prev)
            d = delta_strip(U)
            print("%5.2f   %10.4f            %+.3f                   %.8f    %.8f      %.4f%s   (%.0fs)" % (t, g, lam, energy(U) / E0, energy(W) / E0w, d, "" if d > 2 * 2 * np.pi / N else "  <-- delta < 2dx: NOT RELIABLE", time.time() - t0), flush=True)
            lams.append((t, lam, d > 2 * 2 * np.pi / N))
            g_prev, t_prev = g, t
            mark += 0.5
    return U, lams


UT, lam_f = ladder(U, W, T, "FORWARD  (IC=%s)" % IC)
if REVERSE:
    # flip the velocity at T; the same instrument now runs the flow backward in physical time
    UR = [-Ui for Ui in UT]
    VR = project([fft(rng.standard_normal(X.shape)) * ((k2 >= 1) & (k2 <= 9)) * deal for _ in range(3)])
    VR = [Vi / norm(VR) for Vi in VR]
    WR = [UR[i] + EPS * VR[i] for i in range(3)]
    UB, lam_b = ladder(UR, WR, T, "REVERSED (from t=T back to 0)")
    ret = norm([UB[i] + U_init[i] for i in range(3)]) / norm(U_init)
    print("reversibility: |u_reversed(T) - (-u0)| / |u0| = %.3e   (exact Euler would give 0; this is the instrument's round-trip error)" % ret)
    print("compression half: lambda_b is minus the most CONTRACTING forward exponent over the same window (Liouville: the full spectrum sums to zero);")
    print("the asymmetry lambda_b - lambda_f is compression in excess of stretching (0: a reversible stretch-compress pair, e.g. a sheet)")
    print("   forward t   lambda_f      reversed window   lambda_b      lambda_b - lambda_f")
    for (tf, lf, okf), (tb, lb, okb) in zip(lam_f, reversed(lam_b)):
        print("   %5.2f       %+.3f        %5.2f-%5.2f       %+.3f        %+.3f%s" % (tf, lf, T - tb, T - tb + 0.5, lb, lb - lf, "" if okf and okb else "   (past the clock)"))
