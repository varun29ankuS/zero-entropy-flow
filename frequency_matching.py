"""Does incommensurability protect a flow? Two-scale initial data, commensurate (k, 2k) versus incommensurate (k, ~sqrt2 k).

Quasicrystals form because atoms hold two incommensurate lengths that no periodic arrangement can satisfy at once; the
structure is aperiodic and complete. The question here is whether the same principle does anything for a fluid: does
data whose two scales cannot lock cascade more slowly, or shock later, than data whose scales are harmonic?

REGISTERED prediction (before running): NO. Conway's local-isomorphism property of aperiodic order says every local
configuration recurs within a bounded distance, so the steepest local slope of an incommensurate profile is not
avoided, only relocated. 1-D: equal shock times up to the finite-box approximation of sqrt 2. 2-D: no significant
difference in the cascade rate. A fluid transports; it does not minimise an energy over arrangements, which is what
selects the quasicrystal. If a difference appears either way, that is the result.

1-D Burgers: u0 = A [sin(k x) + sin(q x + phi)] at fixed energy, exact shock time t* = -1 / min u0', reported for the
best and worst phase phi (the quasicrystal claim is about the worst case), for q = 2k and q ~ sqrt2 k.
2-D Euler/NS: w0 = A [sin(k x) cos(k y) + sin(q x + phi) cos(q y)] at fixed energy, palinstrophy P(t)/P(0) and the
analyticity strip at t = 1, 2, 3 for both pairs, several phases; nu as given.
usage: python frequency_matching.py            (1-D exact, instant)     DIM=2 N=256 NU=1e-4 T=3 python frequency_matching.py"""
import os, time, numpy as np

DIM = int(os.environ.get("DIM", 1))
K0 = int(os.environ.get("K0", 5))
PAIRS = {"commensurate (5,10)": 2 * K0, "incommensurate (5,7 ~ sqrt2)": 7}
phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)

if DIM == 1:
    x = np.linspace(0, 2 * np.pi, 20000, endpoint=False)
    print("1-D Burgers, u0 = A[sin(k x) + sin(q x + phi)], equal energy; exact shock time t* = -1/min u0'")
    print("%-30s   t* best phase   t* worst phase   max slope ratio worst/best" % "pair")
    for name, q in PAIRS.items():
        ts = []
        for phi in phis:
            u = np.sin(K0 * x) + np.sin(q * x + phi)
            A = 1.0 / np.sqrt(np.mean(u * u))
            du = A * (K0 * np.cos(K0 * x) + q * np.cos(q * x + phi))
            ts.append(-1.0 / du.min())
        ts = np.array(ts)
        print("%-30s   %.4f          %.4f           %.3f" % (name, ts.max(), ts.min(), ts.max() / ts.min()))
    # the single-scale reference at the same energy
    u = np.sin(K0 * x); A = 1 / np.sqrt(np.mean(u * u)); print("%-30s   %.4f" % ("single scale k=5", 1.0 / (A * K0)))
    print("worst-case shock time, commensurate vs incommensurate: the registered prediction is 'equal within the box approximation'")
else:
    N = int(os.environ.get("N", 256)); NU = float(os.environ.get("NU", 1e-4)); T = float(os.environ.get("T", 3.0)); dt = 1e-3
    fft, ifft = np.fft.fftn, np.fft.ifftn
    k = np.fft.fftfreq(N, d=1.0 / N)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    k2 = kx * kx + ky * ky; k2s = k2.copy(); k2s[0, 0] = 1.0
    deal = (np.abs(kx) < N / 3) & (np.abs(ky) < N / 3)
    xs = np.linspace(0, 2 * np.pi, N, endpoint=False); X, Y = np.meshgrid(xs, xs, indexing="ij")
    KMAG = np.sqrt(k2); NB = int(N / 3)

    def vel(wh):
        psih = wh / k2s; psih[0, 0] = 0
        return ifft(1j * ky * psih).real, -ifft(1j * kx * psih).real

    def rhs(wh):
        u, v = vel(wh); w = ifft(wh).real
        return -0.5 * (fft(u * ifft(1j * kx * wh).real + v * ifft(1j * ky * wh).real) + 1j * kx * fft(u * w) + 1j * ky * fft(v * w)) * deal

    def step(wh):
        f, f2 = np.exp(-NU * k2 * dt), np.exp(-NU * k2 * dt / 2)
        a = rhs(wh); b = rhs(f2 * (wh + dt / 2 * a)); c = rhs(f2 * wh + dt / 2 * b); d = rhs(f * wh + dt * f2 * c)
        return f * wh + dt / 6 * (f * a + 2 * f2 * b + 2 * f2 * c + d)

    def strip(wh):
        e = 0.5 * np.abs(wh) ** 2 / N**4 / k2s
        spec = np.array([e[(KMAG >= n - 0.5) & (KMAG < n + 0.5)].sum() for n in range(1, NB)]); ks = np.arange(1, NB)
        sel = (ks >= NB // 2) & (spec > 1e-300)
        return -np.polyfit(ks[sel], np.log(spec[sel]), 1)[0] / 2 if sel.sum() > 4 else np.nan

    print("2-D, N=%d^2, nu=%g, w0 = A[sin(kx)cos(ky) + sin(qx+phi)cos(qy)] at equal energy; palinstrophy P/P0 and strip delta" % (N, NU))
    print("%-30s  phi     E/E0(T)    Z/Z0(1) Z/Z0(2) Z/Z0(3)    P/P0(1)  P/P0(2)  P/P0(3)    delta(1)  delta(2)  delta(3)")
    t0 = time.time()
    for name, q in PAIRS.items():
        for phi in (0.0, np.pi / 3, 2 * np.pi / 3):
            w = np.sin(K0 * X) * np.cos(K0 * Y) + np.sin(q * X + phi) * np.cos(q * Y)
            wh = fft(w) * deal
            u, v = vel(wh); wh *= 1 / np.sqrt(np.mean(u * u + v * v))
            u, v = vel(wh); E0 = 0.5 * np.mean(u * u + v * v); Z0 = 0.5 * np.mean(ifft(wh).real ** 2)
            P0 = 0.5 * np.mean(ifft(1j * kx * wh).real ** 2 + ifft(1j * ky * wh).real ** 2)
            t, Zs, Ps, Ds = 0.0, [], [], []
            while t < T - 1e-9:
                wh = step(wh); t += dt
                if abs(t - round(t)) < dt / 2 and round(t) >= 1:
                    Zs.append(0.5 * np.mean(ifft(wh).real ** 2) / Z0)
                    Ps.append(0.5 * np.mean(ifft(1j * kx * wh).real ** 2 + ifft(1j * ky * wh).real ** 2) / P0)
                    Ds.append(strip(wh))
            u, v = vel(wh); E = 0.5 * np.mean(u * u + v * v)
            print("%-30s  %.2f    %.5f    %s    %s    %s   (%.0fs)" % (name, phi, E / E0, " ".join("%.3f" % z for z in Zs), " ".join("%8.3f" % p for p in Ps), " ".join("%.3f" % d for d in Ds), time.time() - t0), flush=True)
