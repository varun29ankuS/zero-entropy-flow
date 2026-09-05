"""The regularity criteria, measured along 3-D Euler flows (nu = 0 unless NU is set).

Each line below is a THEOREM giving a sufficient condition for smoothness; a blow-up must violate all of them.
  BKM   (Beale-Kato-Majda 1984)        int_0^T max|w| dt < inf  =>  smooth on [0,T].          reported: max|w|, its time integral
  ESS   (Escauriaza-Seregin-Sverak 2003) sup_t ||u||_{L3} < inf  =>  smooth (Navier-Stokes).   reported: ||u||_3 (scale-critical norm)
  CF    (Constantin-Fefferman 1993)     if the vorticity DIRECTION xi = w/|w| is Lipschitz in the high-vorticity set,
                                        stretching is depleted and the solution is smooth.
        reported: the direction-coherence  rho = <1 - (xi(x).xi(x+h))^2> over points with |w| > 0.5 max|w|, at h = one
        grid spacing (small = parallel vortex lines = CF holds with a good constant); and the local stretching
        alpha = xi.(S xi) averaged over the same set (the CF depletion factor: stretching per unit |w|^2).
Also: enstrophy Z and the stretching term S with the budget residual, as in budgets3d.py.
usage: IC=tg|abc|kp N=64 T=4 python criteria3d.py     (kp = Kida-Pelz high-symmetry flow, Kida 1985)"""
import os, time, numpy as np

IC = os.environ.get("IC", "tg")
N = int(os.environ.get("N", 48))
T = float(os.environ.get("T", 4.0))
NU = float(os.environ.get("NU", 0.0))
dt = 2.0 / N
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

if IC == "abc":
    U = [fft(np.sin(Z_) + np.cos(Y)), fft(np.sin(X) + np.cos(Z_)), fft(np.sin(Y) + np.cos(X))]
    rng = np.random.default_rng(0)
    pert = [fft(rng.standard_normal(X.shape)) * ((k2 >= 1) & (k2 <= 9)) * deal for _ in range(3)]
    kd = sum(K[i] * pert[i] for i in range(3)) / k2s
    pert = [pert[i] - K[i] * kd for i in range(3)]
    sc = 0.1 * np.sqrt(sum(np.mean(ifft(Ui).real ** 2) for Ui in U) / sum(np.mean(ifft(p).real ** 2) for p in pert))
    U = [U[i] + sc * pert[i] for i in range(3)]
elif IC == "kp":  # Kida-Pelz high-symmetry flow (Kida 1985; Boratav-Pelz 1994)
    U = [fft(np.sin(X) * (np.cos(3 * Y) * np.cos(Z_) - np.cos(Y) * np.cos(3 * Z_))),
         fft(np.sin(Y) * (np.cos(3 * Z_) * np.cos(X) - np.cos(Z_) * np.cos(3 * X))),
         fft(np.sin(Z_) * (np.cos(3 * X) * np.cos(Y) - np.cos(X) * np.cos(3 * Y)))]
else:
    U = [fft(np.sin(X) * np.cos(Y) * np.cos(Z_)), fft(-np.cos(X) * np.sin(Y) * np.cos(Z_)), fft(np.zeros_like(X))]


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


def vort(U):
    return [ifft(1j * ky * U[2] - 1j * kz * U[1]).real, ifft(1j * kz * U[0] - 1j * kx * U[2]).real, ifft(1j * kx * U[1] - 1j * ky * U[0]).real]


def diagnostics(U):
    u = [ifft(Ui).real for Ui in U]
    w = vort(U)
    wmag = np.sqrt(sum(wi**2 for wi in w))
    E = 0.5 * sum(np.mean(ui**2) for ui in u)
    Z = 0.5 * np.mean(wmag**2)
    L3 = np.mean(np.sqrt(sum(ui**2 for ui in u)) ** 3) ** (1.0 / 3.0)
    # strain and the local stretching alpha = xi . S xi
    Ud = [Ui * deal for Ui in U]
    G = [[ifft(1j * K[i] * Ud[j]).real for j in range(3)] for i in range(3)]
    Sm = [[0.5 * (G[i][j] + G[j][i]) for j in range(3)] for i in range(3)]
    S_total = sum(np.mean(w[i] * w[j] * G[j][i]) for i in range(3) for j in range(3))
    high = wmag > 0.5 * wmag.max()
    xi = [wi / (wmag + 1e-12) for wi in w]
    alpha = sum(xi[i] * Sm[i][j] * xi[j] for i in range(3) for j in range(3))
    # CF direction coherence at one grid spacing, averaged over the high-vorticity set and the three axes
    rho = 0.0
    for ax in range(3):
        dot = sum(xi[i] * np.roll(xi[i], -1, axis=ax) for i in range(3))
        rho += np.mean((1 - dot**2)[high]) / 3
    return E, Z, S_total, wmag.max(), L3, rho, np.mean(alpha[high]), high.mean()


E0 = diagnostics(U)[0]
t = 0.0
t0 = time.time()
bkm = 0.0
mark = 0.5
print("IC=%s  N=%d^3  nu=%g  dt=%.4f" % (IC, N, NU, dt))
print("   t     E/E0       Z        S        max|w|   int max|w| dt (BKM)   ||u||_L3 (ESS)   CF: rho(xi,h=dx) in |w|>0.5max   alpha=xi.S.xi there   |high set|")
E, Z, S, wm, L3, rho, al, hs = diagnostics(U)
print("%5.2f   %.6f   %7.4f   %+.4f   %7.3f   %8.3f              %.4f          %.4f                          %+.4f              %.3f" % (0, 1, Z, S, wm, 0, L3, rho, al, hs), flush=True)
wm_prev = wm
while t < T - 1e-9:
    U = step(U, dt)
    t += dt
    E, Z, S, wm, L3, rho, al, hs = diagnostics(U) if t >= mark - dt / 2 else (None,) * 8
    if E is not None:
        pass
    # integrate BKM with the trapezoid rule on every step (cheap max|w|)
    wcur = np.sqrt(sum(c**2 for c in vort(U))).max()
    bkm += 0.5 * (wm_prev + wcur) * dt
    wm_prev = wcur
    if t >= mark - dt / 2:
        print("%5.2f   %.6f   %7.4f   %+.4f   %7.3f   %8.3f              %.4f          %.4f                          %+.4f              %.3f   (%.0fs)" % (t, E / E0, Z, S, wm, bkm, L3, rho, al, hs, time.time() - t0), flush=True)
        mark += 0.5
