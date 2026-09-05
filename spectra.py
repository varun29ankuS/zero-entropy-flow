"""Energy spectra: the two turbulence laws everyone recognises, produced by the energy-conserving scheme.
  2-D decaying turbulence, 256^2, nu = 2e-4:   direct enstrophy cascade, E(k) ~ k^-3  (Kraichnan 1967)
  3-D forced turbulence, N^3 (default 48), low-k forcing, statistically steady:  E(k) ~ k^-5/3 (Kolmogorov 1941)
Both spectra are averaged over the late part of the run. Reference slopes drawn, not fitted.
usage: N3=48 T3=20 python spectra.py -> figures/spectra.png  (3-D part takes ~40 min at 48^3 on 2 cores)"""
import os, time, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fft, ifft = np.fft.fftn, np.fft.ifftn
os.makedirs("figures", exist_ok=True)


def shell_spectrum(Uh_list, K, nbins):
    kk = np.sqrt(sum(Ki**2 for Ki in K))
    e = 0.5 * sum(np.abs(Uh)**2 for Uh in Uh_list) / Uh_list[0].size**2
    spec = np.zeros(nbins)
    for n in range(1, nbins):
        spec[n] = e[(kk >= n - 0.5) & (kk < n + 0.5)].sum()
    return spec


# ------------------------------------------ 2-D decaying, vorticity form ------------------------------------------
def run2d(M=256, nu=2e-4, T=8.0, dt=1e-3, seed=0):
    k = np.fft.fftfreq(M, d=1.0 / M)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    k2 = kx * kx + ky * ky
    k2s = k2.copy()
    k2s[0, 0] = 1.0
    deal = (np.abs(kx) < M / 3) & (np.abs(ky) < M / 3)

    def vel(wh):
        psih = wh / k2s
        psih[0, 0] = 0
        return 1j * ky * psih, -1j * kx * psih  # spectral velocity

    def rhs(wh):
        uh, vh = vel(wh)
        u, v = ifft(uh).real, ifft(vh).real
        w = ifft(wh).real
        wx, wy = ifft(1j * kx * wh).real, ifft(1j * ky * wh).real
        return -0.5 * (fft(u * wx + v * wy) + 1j * kx * fft(u * w) + 1j * ky * fft(v * w)) * deal

    rng = np.random.default_rng(seed)
    wh = fft(rng.standard_normal((M, M))) * ((k2 >= 9) & (k2 <= 64)) * deal
    uh, vh = vel(wh)
    wh *= 1.0 / np.sqrt(np.mean(ifft(uh).real ** 2 + ifft(vh).real ** 2))
    fac, fac2 = np.exp(-nu * k2 * dt), np.exp(-nu * k2 * dt / 2)
    t = 0.0
    acc = np.zeros(M // 3)
    n = 0
    while t < T:
        a = rhs(wh)
        b = rhs(fac2 * (wh + dt / 2 * a))
        c = rhs(fac2 * wh + dt / 2 * b)
        d = rhs(fac * wh + dt * fac2 * c)
        wh = fac * wh + dt / 6 * (fac * a + 2 * fac2 * b + 2 * fac2 * c + d)
        t += dt
        if t > T / 2 and int(round(t / dt)) % 200 == 0:
            acc += shell_spectrum(list(vel(wh)), [kx, ky], M // 3)
            n += 1
    return acc / max(n, 1)


# ---------------------------------------- 3-D forced, velocity form ----------------------------------------
def run3d(N=48, nu=None, T=20.0, seed=0):
    dt = 1.0 / N
    nu = nu or 8.0 / N**1.4 * 0.05  # picked so the dissipation scale sits near the grid cutoff
    k = np.fft.fftfreq(N, d=1.0 / N)
    K = list(np.meshgrid(k, k, k, indexing="ij"))
    k2 = sum(Ki**2 for Ki in K)
    k2s = k2.copy()
    k2s[0, 0, 0] = 1.0
    deal = np.ones_like(k2, bool)
    for Ki in K:
        deal &= np.abs(Ki) < N / 3
    force_band = (k2 >= 1) & (k2 <= 4)

    def project(F):
        kdotF = sum(K[i] * F[i] for i in range(3)) / k2s
        return [F[i] - K[i] * kdotF for i in range(3)]

    def transport(U):
        Ud = [Ui * deal for Ui in U]
        u = [ifft(Ui).real for Ui in Ud]
        out = []
        for i in range(3):
            adv = sum(u[j] * ifft(1j * K[j] * Ud[i]).real for j in range(3))
            div = sum(ifft(1j * K[j] * fft(u[j] * u[i])).real for j in range(3))
            out.append(-0.5 * fft(adv + div) * deal)
        return project(out)

    def forcing(U):  # constant energy-injection forcing in the low-k band (Lundgren-type): f = eps * u_band / (2 E_band)
        Eb = 0.5 * sum((np.abs(Ui) ** 2 * force_band).sum() for Ui in U) / N**6
        eps = 0.1
        return [eps * Ui * force_band / (2 * Eb + 1e-12) for Ui in U]

    rng = np.random.default_rng(seed)
    U = [fft(rng.standard_normal((N, N, N))) * ((k2 >= 1) & (k2 <= 9)) * deal for _ in range(3)]
    U = project(U)
    E = 0.5 * sum(np.mean(ifft(Ui).real ** 2) for Ui in U)
    U = [Ui / np.sqrt(2 * E) for Ui in U]
    fac, fac2 = np.exp(-nu * k2 * dt), np.exp(-nu * k2 * dt / 2)
    r = lambda V: [a + b for a, b in zip(transport(V), forcing(V))]
    t = 0.0
    acc = np.zeros(N // 3)
    n = 0
    t0 = time.time()
    while t < T:
        a = r(U)
        b = r([fac2 * (U[i] + dt / 2 * a[i]) for i in range(3)])
        c = r([fac2 * U[i] + dt / 2 * b[i] for i in range(3)])
        d = r([fac * U[i] + dt * fac2 * c[i] for i in range(3)])
        U = [fac * U[i] + dt / 6 * (fac * a[i] + 2 * fac2 * b[i] + 2 * fac2 * c[i] + d[i]) for i in range(3)]
        t += dt
        if t > T / 2 and int(round(t / dt)) % 25 == 0:
            acc += shell_spectrum(U, K, N // 3)
            n += 1
        if int(round(t / dt)) % 200 == 0:
            print("  3-D forced: t=%.1f  E=%.4f  (%.0fs)" % (t, 0.5 * sum(np.mean(ifft(Ui).real ** 2) for Ui in U), time.time() - t0), flush=True)
    return acc / max(n, 1), nu


N3 = int(os.environ.get("N3", 48))
T3 = float(os.environ.get("T3", 20.0))
print("2-D decaying 256^2 ...", flush=True)
s2 = run2d()
print("3-D forced %d^3 ..." % N3, flush=True)
s3, nu3 = run3d(N3, T=T3)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
kk = np.arange(len(s2))
m = (kk >= 3) & (s2 > 0)
a1.loglog(kk[m], s2[m], "o-", color="#1b1b1b", ms=3, lw=1.4, label="E(k), 2-D decaying, 256$^2$")
kr = np.array([8, 40])
a1.loglog(kr, s2[8] * (kr / 8.0) ** -3, "--", color="#d9731f", lw=1.6, label="k$^{-3}$  (Kraichnan enstrophy cascade)")
a1.set_xlabel("k")
a1.set_ylabel("E(k)")
a1.set_title("2-D: direct enstrophy cascade", loc="left")
a1.legend(frameon=False)
kk = np.arange(len(s3))
m = (kk >= 1) & (s3 > 0)
a2.loglog(kk[m], s3[m], "o-", color="#1b1b1b", ms=3, lw=1.4, label="E(k), 3-D forced, %d$^3$, nu=%.1e" % (N3, nu3))
kr = np.array([3, 12])
a2.loglog(kr, s3[3] * (kr / 3.0) ** (-5.0 / 3.0), "--", color="#d9731f", lw=1.6, label="k$^{-5/3}$  (Kolmogorov 1941)")
a2.set_xlabel("k")
a2.set_ylabel("E(k)")
a2.set_title("3-D: inertial range", loc="left")
a2.legend(frameon=False)
fig.suptitle("Energy spectra from the energy-conserving scheme; reference slopes drawn, not fitted", x=0.01, ha="left", fontsize=10)
fig.tight_layout()
fig.savefig("figures/spectra.png", dpi=160)
print("figures/spectra.png")
