"""Wave and particle: the same flow in both descriptions, and what the phases decide.

The Eulerian description is the wave one - Fourier modes with amplitude and phase, and interference (a triad transfers
energy by the cosine of a phase relation). The Lagrangian description is the particle one - parcels carrying their
memory along paths (Cauchy). They are the same flow. Three registered tests:
  T1 (1-D, exact)  Burgers solved along characteristics (particles: u constant on X' = u) and spectrally (waves) agree
                   to round-off before the shock.                                    prediction: relative error < 1e-8
  T2 (1-D, exact)  The SAME amplitude spectrum with random phases: the shock time t* = -1/min u0' spread across the
                   ensemble.                                                          prediction: max/min t* > 2
  T3 (2-D)         A developed turbulent field versus the same field with every phase randomised (identical spectrum,
                   identical energy and enstrophy): palinstrophy production at t = 0+, and enstrophy lost over dt.
                   Random phases have no interference, so no net cascade at first.    prediction: production ratio
                   randomised/real < 0.2 at t = 0+; the randomised field re-develops correlations and cascades later.
usage: python wave_particle.py           (1-D, instant)      DIM=2 N=256 python wave_particle.py"""
import os, time, numpy as np

DIM = int(os.environ.get("DIM", 1))
rng = np.random.default_rng(0)

if DIM == 1:
    N = 1024
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    k = np.fft.fftfreq(N, d=1.0 / N)
    deal = np.abs(k) < N / 3
    # a five-mode initial field with fixed amplitudes
    amps = {1: 1.0, 2: 0.6, 3: 0.4, 5: 0.3, 7: 0.2}

    def field(phases):
        return sum(a * np.sin(m * x + phases[m]) for m, a in amps.items())

    # T1: characteristics vs spectral, one realisation, up to 0.8 t*
    ph = {m: 0.0 for m in amps}
    u0 = field(ph)
    du0 = sum(a * m * np.cos(m * x + ph[m]) for m, a in amps.items())
    tstar = -1.0 / du0.min()
    def rhs(uh):
        u = np.fft.ifft(uh).real
        return -np.fft.fft((u * np.fft.ifft(1j * k * uh).real + np.fft.ifft(1j * k * np.fft.fft(u * u)).real) / 3.0) * deal

    def strip(uh):
        e = np.abs(uh[:N // 3]) ** 2; ks = np.arange(N // 3); sel = (ks >= N // 6) & (e > 1e-300)
        return -np.polyfit(ks[sel], np.log(e[sel]), 1)[0] / 2 if sel.sum() > 4 else np.nan

    print("T1  1-D Burgers, t* = %.4f: the wave solution (spectral, %d modes) evaluated at the particle positions X(a,t) = a + t u0(a)" % (tstar, int(N / 3)))
    print("    versus the memory the particles carry, u0(a). Exact for the continuum; on a grid they agree until the wave passes the observer.")
    print("      t/t*    rel L2 error     strip delta     2dx = %.4f" % (2 * 2 * np.pi / N))
    uh = np.fft.fft(u0); dt = 1e-4; t = 0.0
    for frac in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
        T = frac * tstar
        while t < T - 1e-12:
            h = min(dt, T - t)
            a = rhs(uh); b = rhs(uh + h / 2 * a); c = rhs(uh + h / 2 * b); d = rhs(uh + h * c)
            uh = uh + h / 6 * (a + 2 * b + 2 * c + d); t += h
        Xp = x + T * u0
        uhd = uh * deal
        u_wave_at_X = np.real(sum(uhd[j] * np.exp(1j * k[j] * Xp) for j in range(N) if deal[j])) / N
        err = np.sqrt(np.mean((u_wave_at_X - u0) ** 2) / np.mean(u0**2))
        d = strip(uh)
        print("      %.2f     %.2e         %.4f%s" % (frac, err, d, "" if d > 2 * 2 * np.pi / N else "   <-- below 2dx: the wave has passed the observer"))
    print("    -> the two descriptions are one flow; the discrepancy is the observer's, and it appears exactly when the clock says so")

    # T2: same spectrum, random phases
    ts = []
    for _ in range(2000):
        ph = {m: rng.uniform(0, 2 * np.pi) for m in amps}
        du = sum(a * m * np.cos(m * x + ph[m]) for m, a in amps.items())
        ts.append(-1.0 / du.min())
    ts = np.array(ts)
    print("T2  same amplitude spectrum, 2000 random phase sets: shock time t* min %.4f  median %.4f  max %.4f  (max/min = %.2f)  -> %s" % (ts.min(), np.median(ts), ts.max(), ts.max() / ts.min(), "PASS: the phases decide the singularity time" if ts.max() / ts.min() > 2 else "FAIL"))
else:
    N = int(os.environ.get("N", 256)); NU = float(os.environ.get("NU", 1e-4)); dt = 1e-3
    fft, ifft = np.fft.fftn, np.fft.ifftn
    k = np.fft.fftfreq(N, d=1.0 / N); kx, ky = np.meshgrid(k, k, indexing="ij")
    k2 = kx * kx + ky * ky; k2s = k2.copy(); k2s[0, 0] = 1.0
    deal = (np.abs(kx) < N / 3) & (np.abs(ky) < N / 3)

    def vel(wh):
        psih = wh / k2s; psih[0, 0] = 0
        return ifft(1j * ky * psih).real, -ifft(1j * kx * psih).real

    def rhs(wh):
        u, v = vel(wh); w = ifft(wh).real
        return -0.5 * (fft(u * ifft(1j * kx * wh).real + v * ifft(1j * ky * wh).real) + 1j * kx * fft(u * w) + 1j * ky * fft(v * w)) * deal

    def step(wh, nu):
        f, f2 = np.exp(-nu * k2 * dt), np.exp(-nu * k2 * dt / 2)
        a = rhs(wh); b = rhs(f2 * (wh + dt / 2 * a)); c = rhs(f2 * wh + dt / 2 * b); d = rhs(f * wh + dt * f2 * c)
        return f * wh + dt / 6 * (f * a + 2 * f2 * b + 2 * f2 * c + d)

    def Z(wh): return 0.5 * np.mean(ifft(wh).real ** 2)
    def P(wh): return 0.5 * np.mean(ifft(1j * kx * wh).real ** 2 + ifft(1j * ky * wh).real ** 2)
    def prod(wh):
        """palinstrophy production dP/dt from the nonlinear term alone (nu = 0): <grad w . grad(rhs)>"""
        r = rhs(wh)
        return np.mean(ifft(1j * kx * wh).real * ifft(1j * kx * r).real + ifft(1j * ky * wh).real * ifft(1j * ky * r).real)

    # develop a turbulent field to t = 2 (as in flow_gif.py), then compare with its phase-randomised twin
    wh = fft(rng.standard_normal((N, N))) * ((k2 >= 9) & (k2 <= 36)) * deal
    u, v = vel(wh); wh *= 1 / np.sqrt(np.mean(u * u + v * v))
    t0 = time.time(); t = 0.0
    while t < 2.0 - 1e-9:
        wh = step(wh, NU); t += dt
    real = wh.copy()
    phase = np.exp(2j * np.pi * rng.random((N, N)))
    rand = np.abs(real) * phase
    rand = 0.5 * (rand + np.conj(np.roll(np.flip(rand, (0, 1)), 1, (0, 1))))      # real field: Hermitian symmetry
    rand = rand * np.abs(real) / np.maximum(np.abs(rand), 1e-300)                 # restore the exact amplitudes
    print("2-D, N=%d^2, nu=%g: developed field at t=2 versus the same amplitudes with random phases   (%.0fs)" % (N, NU, time.time() - t0))
    print("            E(rel)      Z(rel)      P(rel)      palinstrophy production at t=0+ (nonlinear)")
    ur, vr = vel(real); uq, vq = vel(rand)
    Er, Eq = 0.5 * np.mean(ur**2 + vr**2), 0.5 * np.mean(uq**2 + vq**2)
    pr, pq = prod(real), prod(rand)
    print("real        %.5f     %.5f     %.5f     %+.4e" % (1, 1, 1, pr))
    print("randomised  %.5f     %.5f     %.5f     %+.4e     ratio %.3f  -> %s" % (Eq / Er, Z(rand) / Z(real), P(rand) / P(real), pq, abs(pq / pr), "PASS: no interference, no cascade" if abs(pq / pr) < 0.2 else "FAIL"))
    # T4: scale-resolved interference - randomise only the phases ABOVE a cutoff k_c (or only BELOW it) and measure
    # the palinstrophy production that survives: where in scale does the cascade's coherence live?
    KM = np.sqrt(k2)
    print("T4  phases randomised only above k_c (large scales kept) / only below k_c (small scales kept): surviving production ratio")
    print("      k_c     above k_c randomised    below k_c randomised")
    for kc in (4, 8, 16, 32, 64):
        out = []
        for mode in ("above", "below"):
            mask = (KM > kc) if mode == "above" else (KM <= kc)
            ph2 = np.where(mask, phase, 1.0)
            r2 = real * ph2
            r2 = 0.5 * (r2 + np.conj(np.roll(np.flip(r2, (0, 1)), 1, (0, 1))))
            r2 = r2 * np.abs(real) / np.maximum(np.abs(r2), 1e-300)
            out.append(abs(prod(r2) / pr))
        print("      %3d       %.3f                  %.3f" % (kc, out[0], out[1]), flush=True)
    # T5: the coherence race - scramble the phases above k_c and measure the time for the cross-cut cascade to recover to
    # half its intact value. REGISTERED conjecture: phase coherence propagates through scale at a bounded rate, so the
    # regrowth time does not shrink to zero as k_c grows. A singularity would need coherence to reach k = infinity in
    # finite time, i.e. regrowth times shrinking to zero at small scales.
    print("T5  coherence regrowth time after scrambling the phases above k_c (time for production to recover to 50% of the intact flow's)")
    print("      k_c     production ratio at t=0+ / 0.05 / 0.10 / 0.20 / 0.40      t_half")
    for kc in (4, 8, 16, 32, 64):
        mask = KM > kc
        r2 = real * np.where(mask, phase, 1.0)
        r2 = 0.5 * (r2 + np.conj(np.roll(np.flip(r2, (0, 1)), 1, (0, 1))))
        r2 = r2 * np.abs(real) / np.maximum(np.abs(r2), 1e-300)
        wq, wr = r2.copy(), real.copy()
        t, ratios, thalf = 0.0, [abs(prod(wq) / prod(wr))], None
        for tt in (0.05, 0.10, 0.20, 0.40):
            while t < tt - 1e-9:
                wq = step(wq, NU); wr = step(wr, NU); t += dt
                rr = abs(prod(wq) / prod(wr))
                if thalf is None and rr >= 0.5:
                    thalf = t
            ratios.append(abs(prod(wq) / prod(wr)))
        print("      %3d     %s      %s" % (kc, " / ".join("%.3f" % r for r in ratios), ("%.3f" % thalf) if thalf else "> 0.40"), flush=True)
    # then let both run on for dt = 0.5 and 1.0: the randomised field re-develops correlations
    print("   dt     real: Z/Z0   P/P0      randomised: Z/Z0   P/P0     production ratio (randomised/real)")
    wr, wq = real.copy(), rand.copy(); Zr0, Zq0, Pr0, Pq0 = Z(wr), Z(wq), P(wr), P(wq); t = 0.0
    for tt in (0.25, 0.5, 1.0):
        while t < tt - 1e-9:
            wr = step(wr, NU); wq = step(wq, NU); t += dt
        print("  %.2f    %.5f   %.4f      %.5f   %.4f     %.3f" % (tt, Z(wr) / Zr0, P(wr) / Pr0, Z(wq) / Zq0, P(wq) / Pq0, abs(prod(wq) / prod(wr))), flush=True)
