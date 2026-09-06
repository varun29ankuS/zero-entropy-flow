"""Complex-singularity tracking (Sulem, Sulem & Frisch 1983; Frisch, Matsumoto & Bec 2003): the energy spectrum of an
analytic flow decays as E(k) ~ k^-n exp(-2 delta(t) k), where delta(t) is the distance of the nearest complex-space
singularity from the real domain. A real finite-time singularity means delta(t*) = 0. The two classical hypotheses:
    exponential  delta(t) = delta0 exp(-t/tau)          no real singularity (Taylor-Green, Brachet et al. 1983, 1992)
    linear       delta(t) = c (t* - t)                  singularity at t*
Here delta(t) is sampled densely and both laws are fitted over the RELIABLE window only (delta > 2 dx); the residuals of
the two fits and the linear extrapolation t* are reported. The fit uses log E(k) = a - n log k - 2 delta k over the upper
half of the retained modes with n fixed (NFIX, default 0: the same delta as the reliability clock elsewhere in this
repository); the free-n fit is printed alongside but with 5-16 shells it is ill-conditioned and is not used.
usage: IC=tg|kp N=64 T=3 python strip_tracker.py"""
import os, glob, time, numpy as np

IC = os.environ.get("IC", "tg")
N = int(os.environ.get("N", 64))
T = float(os.environ.get("T", 3.0))
EVERY = float(os.environ.get("EVERY", 0.05))
NFIX = float(os.environ.get("NFIX", 0.0))     # prefactor exponent held fixed for the clock delta
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
    a = transport(U)
    b = transport([U[i] + dt / 2 * a[i] for i in range(3)])
    c = transport([U[i] + dt / 2 * b[i] for i in range(3)])
    d = transport([U[i] + dt * c[i] for i in range(3)])
    return [U[i] + dt / 6 * (a[i] + 2 * b[i] + 2 * c[i] + d[i]) for i in range(3)]


def strip(U):
    """delta with the prefactor exponent fixed at n = NFIX (the clock's definition, used for the fits), and the free-n
    fit (delta_free, n) for information: with few shells the free fit trades n against delta and is not trustworthy"""
    e = 0.5 * sum(np.abs(Ui) ** 2 for Ui in U) / N**6
    spec = np.array([e[s].sum() for s in shells])
    ks = np.arange(1, NB)
    sel = (ks >= NB // 2) & (spec > 1e-300)
    if sel.sum() < 4:
        return np.nan, np.nan, np.nan
    y = np.log(spec[sel]) + NFIX * np.log(ks[sel])
    d0 = -np.polyfit(ks[sel], y, 1)[0] / 2.0
    A = np.stack([np.ones(sel.sum()), -np.log(ks[sel]), -2.0 * ks[sel]], 1)
    coef = np.linalg.lstsq(A, np.log(spec[sel]), rcond=None)[0]
    return d0, coef[2], coef[1]


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
dx2 = 2 * 2 * np.pi / N


def wmax(U):
    Ud = [Ui * deal for Ui in U]
    w = [ifft(1j * K[1] * Ud[2] - 1j * K[2] * Ud[1]).real, ifft(1j * K[2] * Ud[0] - 1j * K[0] * Ud[2]).real, ifft(1j * K[0] * Ud[1] - 1j * K[1] * Ud[0]).real]
    return np.sqrt(sum(wi**2 for wi in w)).max()
print("IC=%s  N=%d^3  nu=0  2dx = %.4f   delta sampled every %.2f" % (IC, N, dx2, EVERY))
print("   t     delta (n=%g fixed)   delta_free   n_free   E/E0        max|w|    BKM int" % NFIX)
ws, bkm, wprev, tprev = [], 0.0, None, 0.0
E0 = 0.5 * sum(np.mean(ifft(Ui).real ** 2) for Ui in U)
ts, ds = [], []
t, mark, t0 = 0.0, 0.0, time.time()
while t < T + 1e-9:
    if t >= mark - 1e-9:
        d, df, n = strip(U)
        E = 0.5 * sum(np.mean(ifft(Ui).real ** 2) for Ui in U)
        wm = wmax(U)
        if wprev is not None:
            bkm += 0.5 * (wprev + wm) * (t - tprev)
        wprev, tprev = wm, t
        if d > dx2:
            ts.append(t)
            ds.append(d)
            ws.append(wm)
        print("%5.2f   %.4f              %.4f      %+.2f    %.8f   %8.3f   %7.3f%s" % (t, d, df, n, E / E0, wm, bkm, "" if d > dx2 else "   <-- below 2dx"), flush=True)
        mark += EVERY
        if d <= dx2 and len(ts) > 0:
            break
    umax = max(np.abs(ifft(Ui).real).max() for Ui in U)
    dt = min(2.0 / N, 0.5 * (2 * np.pi / N) / max(umax, 1e-9), mark - t + 1e-12)
    U = step(U, dt)
    t += dt
ts, ds, ws = np.array(ts), np.array(ds), np.array(ws)
sel = ts > 0.15
ws = ws[sel]    # skip the transient where the k^-n fit is meaningless (initial data has a single shell)
ts, ds = ts[sel], ds[sel]
print("reliable window used for the fits: t in [%.2f, %.2f] (%d samples), %.0fs" % (ts[0], ts[-1], len(ts), time.time() - t0))
if len(ts) >= 6:
    pe = np.polyfit(ts, np.log(ds), 1)
    res_e = np.sqrt(np.mean((np.log(ds) - np.polyval(pe, ts)) ** 2))
    pl = np.polyfit(ts, ds, 1)
    res_l = np.sqrt(np.mean((ds - np.polyval(pl, ts)) ** 2)) / np.mean(ds)
    # second half only: the late trend is what decides
    h = ts > 0.5 * (ts[0] + ts[-1])
    pe2 = np.polyfit(ts[h], np.log(ds[h]), 1)
    pl2 = np.polyfit(ts[h], ds[h], 1)
    print("exponential fit  delta = %.3f exp(-t/%.3f)   relative rms residual %.4f" % (np.exp(pe[1]), -1 / pe[0], res_e))
    print("linear fit       delta = %.3f (%.3f - t)     relative rms residual %.4f   -> t* = %.3f" % (-pl[0], -pl[1] / pl[0], res_l, -pl[1] / pl[0]))
    print("second half only: exponential tau = %.3f;  linear t* = %.3f" % (-1 / pe2[0], -pl2[1] / pl2[0]))
    print("local decay rate -d log(delta)/dt at start / end of window: %.3f / %.3f  (constant = exponential; rising toward 1/(t*-t) = linear)" % (-np.gradient(np.log(ds), ts)[0], -np.gradient(np.log(ds), ts)[-1]))
    # blow-up TYPE from max|w| inside the window: exponential (no singularity) vs power law (t* - t)^-gamma
    # (BKM needs int max|w| dt = inf, i.e. gamma >= 1; Type I is gamma = 1; faster is Type II)
    if len(ws) >= 6:
        pw = np.polyfit(ts, np.log(ws), 1); res_w = np.sqrt(np.mean((np.log(ws) - np.polyval(pw, ts)) ** 2))
        best = None
        for tst in np.linspace(ts[-1] + 0.02, ts[-1] + 3.0, 300):
            pp = np.polyfit(np.log(tst - ts), np.log(ws), 1); r = np.sqrt(np.mean((np.log(ws) - np.polyval(pp, np.log(tst - ts))) ** 2))
            if best is None or r < best[0]:
                best = (r, tst, -pp[0])
        rate_w = np.gradient(np.log(ws), ts)
        print("max|w|: exponential fit rate %.3f (rms %.4f); best power law (t* - t)^-gamma: t* = %.3f, gamma = %.2f (rms %.4f); local growth rate d log max|w|/dt start / end: %.3f / %.3f" % (pw[0], res_w, best[1], best[2], best[0], rate_w[0], rate_w[-1]))
        print("BLOW-UP TYPE: %s" % ("exponential growth of max|w| fits at least as well: no singularity type assignable inside the window" if res_w <= best[0] * 1.05 else ("power law with gamma = %.2f (%s)" % (best[2], "Type I-like, gamma ~ 1" if 0.7 < best[2] < 1.3 else ("Type II-like, gamma > 1" if best[2] >= 1.3 else "gamma < 1: BKM integral would stay finite, not a blow-up")))))
    print("VERDICT: %s" % ("exponential decay fits better: no finite-time singularity indicated inside the window" if res_e < res_l else "linear decay fits better: consistent with a singularity at t* = %.3f (requires the next resolution to agree)" % (-pl[1] / pl[0])))
