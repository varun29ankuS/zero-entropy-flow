"""Keep the state, track the dissipation: the Caffarelli-Kohn-Nirenberg concentration exponent.

Navier-Stokes carries the state by an exact projection-conserving transport and leaks energy only through the
dissipation nu |grad u|^2, whose time integral is bounded a priori by E(0). CKN (1982): a point (x,t) is regular if
the dissipation in the parabolic cylinder Q_r = B_r(x) x [t - r^2, t] satisfies
        (1/r) int int_{Q_r} |grad u|^2 dx ds  <  eps_0        for some small r,
so a singularity needs the local dissipation to concentrate at least like r^1. This measures how it actually
concentrates along a flow: around the point of maximum vorticity at time t,
        D(r) = int int_{Q_r} |grad u|^2 dx ds  ~  r^alpha,
fitted over a ladder of r between 2 dx and L/8. Uniform smooth dissipation: alpha = 5 (volume r^3 x time r^2);
a sheet: 4; a tube: 3; CKN-critical: 1. The local exponent between successive r is also printed; the value at the
smallest reliable r is the one that matters, and the trend in time is the question.
usage: IC=tg|kp|found N=64 NU=1e-3 T=1.0 python ckn_exponent.py"""
import os, glob, time, numpy as np

IC = os.environ.get("IC", "tg")
N = int(os.environ.get("N", 64))
NU = float(os.environ.get("NU", 1e-3))
T = float(os.environ.get("T", 1.0))
SNAP = 0.02
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
dx = 2 * np.pi / N


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


def gradsq_and_vort(U):
    Ud = [Ui * deal for Ui in U]
    g = sum(ifft(1j * K[i] * Ud[j]).real ** 2 for i in range(3) for j in range(3))
    w = [ifft(1j * ky * Ud[2] - 1j * kz * Ud[1]).real, ifft(1j * kz * Ud[0] - 1j * kx * Ud[2]).real, ifft(1j * kx * Ud[1] - 1j * ky * Ud[0]).real]
    return g.astype(np.float32), np.sqrt(sum(wi**2 for wi in w))


if IC == "kp":
    U = [fft(np.sin(X) * (np.cos(3 * Y) * np.cos(Z_) - np.cos(Y) * np.cos(3 * Z_))),
         fft(np.sin(Y) * (np.cos(3 * Z_) * np.cos(X) - np.cos(Z_) * np.cos(3 * X))),
         fft(np.sin(Z_) * (np.cos(3 * X) * np.cos(Y) - np.cos(X) * np.cos(3 * Y)))]
elif IC == "found":
    path = sorted(glob.glob("results/found/*.npz"))[-1]
    u = np.load(path)["u"].astype(float)
    n0 = u.shape[1]
    U = []
    for c in range(3):
        uh = fft(u[c]) * (N / n0) ** 3
        big = np.zeros((N, N, N), complex)
        h = n0 // 2
        for a in (slice(0, h), slice(-h, None)):
            for b in (slice(0, h), slice(-h, None)):
                for cc in (slice(0, h), slice(-h, None)):
                    big[a, b, cc] = uh[a, b, cc]
        U.append(big)
    U = project(U)
else:
    U = [fft(np.sin(X) * np.cos(Y) * np.cos(Z_)), fft(-np.cos(X) * np.sin(Y) * np.cos(Z_)), fft(np.zeros_like(X))]
Z0 = 0.375
Zi = 0.5 * np.mean(sum(ifft(1j * K[a] * U[b] - 1j * K[b] * U[a]).real ** 2 for a, b in ((1, 2), (2, 0), (0, 1))))
U = [Ui * np.sqrt(Z0 / Zi) for Ui in U]

# a ladder of radii in grid units; the cylinder integral needs snapshots back to t - r^2
radii = [int(v) for v in os.environ.get("RADII", "2,3,4,5,6").split(",")]
rmax_t = (radii[-1] * dx) ** 2
snaps = []          # (t, |grad u|^2 field)
idx = np.indices((N, N, N))


def cylinder(t_now, centre, r_cells):
    """int int_{Q_r} |grad u|^2 over the periodic ball of radius r around centre and the last r^2 of time, by trapezoid over snapshots"""
    r = r_cells * dx
    d2 = sum(((idx[a] - centre[a] + N // 2) % N - N // 2) ** 2 for a in range(3)) * dx**2
    ball = d2 <= r * r
    ts = [s[0] for s in snaps]
    vals = [s[1][ball].sum() * dx**3 for s in snaps if s[0] >= t_now - r * r - 1e-12]
    tt = [s[0] for s in snaps if s[0] >= t_now - r * r - 1e-12]
    if len(vals) < 2 or t_now < r * r - 1e-9:          # the cylinder must fit inside the elapsed time
        return np.nan
    return np.trapezoid(vals, tt)


print("IC=%s  N=%d^3  nu=%g  Z0=%.3f   radii (grid cells) %s   cylinder depth r^2" % (IC, N, NU, Z0, radii))
print("   t     Z/Z0    max|w|    D(r) for r = %s                       local exponents alpha(r_i -> r_i+1)          fit alpha" % radii)
t, mark, t0 = 0.0, 0.0, time.time()
nxt_snap = 0.0
report_every = 0.1
Zt = Z0
while t <= T + 1e-9:
    if t >= nxt_snap - 1e-9:
        g, wmag = gradsq_and_vort(U)
        snaps.append((t, g))
        snaps = [s for s in snaps if s[0] >= t - rmax_t - 0.05]
        nxt_snap += SNAP
        if t >= mark - 1e-9:
            Zt = 0.5 * np.mean(wmag**2)
            c = np.unravel_index(np.argmax(wmag), wmag.shape)
            D = np.array([cylinder(t, c, r) for r in radii])
            rr = np.array(radii, float) * dx
            ok = np.isfinite(D) & (D > 0)
            loc = [np.log(D[i + 1] / D[i]) / np.log(rr[i + 1] / rr[i]) if ok[i] and ok[i + 1] else np.nan for i in range(len(radii) - 1)]
            alpha = np.polyfit(np.log(rr[ok]), np.log(D[ok]), 1)[0] if ok.sum() >= 3 else np.nan
            print("%5.2f   %.3f   %6.2f    %s    %s    %.2f   (%.0fs)" % (t, Zt / Z0, wmag.max(), " ".join("%.2e" % v for v in D), " ".join("%5.2f" % v for v in loc), alpha, time.time() - t0), flush=True)
            mark += report_every
            if t >= T - 1e-9:
                break
    umax = max(np.abs(ifft(Ui).real).max() for Ui in U)
    dt = min(2.0 / N, 0.5 * (2 * np.pi / N) / max(umax, 1e-9), nxt_snap - t + 1e-12)
    U = step(U, dt)
    t += dt
