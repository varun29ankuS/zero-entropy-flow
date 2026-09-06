"""The constant that is really a function: the same flow under four frictions.

Navier-Stokes assumes friction -nu Laplacian u with nu a constant. Make it a function and regularity is a theorem:
  Lions (1969)          (-Laplacian)^alpha with alpha >= 5/4: global regularity (Tao 2009: 5/4 minus a log)
  Ladyzhenskaya (1967)  nu(|grad u|) = nu0 (1 + c |grad u|^p) rising with the strain rate: global regularity
The physical case (alpha = 1, constant nu) is the one open case. Here the searcher's antiparallel-sheet field (the
fastest amplifier found, results/found/*.npz) is run with the same instrument under
    A  constant nu, Laplacian                 (Navier-Stokes)
    B  hyperviscosity alpha = 1.25            (Lions), nu_alpha matched to A at k = K_MATCH
    C  hyperviscosity alpha = 1.10            (inside the open gap)
    D  Ladyzhenskaya nu0 (1 + c |grad u|^{1/2}) with c chosen so nu doubles at the initial max strain
and the enstrophy, max|w| and analyticity strip are reported: how the sheet's exponential thinning changes once the
friction becomes a function. Not new mathematics - a picture of the wall from the side named by the theorems.
usage: N=48 T=1.0 NU=2e-3 python friction_function.py       (CPU, a few minutes)"""
import os, glob, time, numpy as np

N = int(os.environ.get("N", 48))
T = float(os.environ.get("T", 1.0))
NU = float(os.environ.get("NU", 2e-3))
KM = float(os.environ.get("K_MATCH", 8.0))
IC = os.environ.get("IC", "found")
fft, ifft = np.fft.fftn, np.fft.ifftn
k = np.fft.fftfreq(N, d=1.0 / N)
kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
K = [kx, ky, kz]
k2 = kx**2 + ky**2 + kz**2
k2s = k2.copy(); k2s[0, 0, 0] = 1.0
deal = (np.abs(kx) < N / 3) & (np.abs(ky) < N / 3) & (np.abs(kz) < N / 3)
x = np.linspace(0, 2 * np.pi, N, endpoint=False)
X, Y, Z_ = np.meshgrid(x, x, x, indexing="ij")
KMAG = np.sqrt(k2); NB = int(N / 3)
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


def lady_term(U, nu0, c):
    """Ladyzhenskaya friction div( nu(x) grad u ), nu = nu0 (1 + c |grad u|^{1/2}), as a spectral tendency"""
    Ud = [Ui * deal for Ui in U]
    G = [[ifft(1j * K[i] * Ud[j]).real for j in range(3)] for i in range(3)]
    gmag = np.sqrt(sum(G[i][j] ** 2 for i in range(3) for j in range(3)))
    nu = nu0 * (1 + c * np.sqrt(gmag))
    out = [sum(1j * K[i] * fft(nu * G[i][j]) for i in range(3)) * deal for j in range(3)]
    return project(out)


def run(label, damp=None, lady=None):
    """damp: spectral damping rate array (exact integrating factor); lady: (nu0, c) explicit nonlinear friction"""
    U = [Ui.copy() for Ui in U0]
    t, t0 = 0.0, time.time()
    rows = []
    mark = 0.0
    while t <= T + 1e-9:
        if t >= mark - 1e-9:
            w = vort(U); wm = np.sqrt(sum(wi**2 for wi in w))
            Z = 0.5 * np.mean(wm**2)
            rows.append((t, Z / Z0, wm.max(), strip(U)))
            mark += 0.25
            if t >= T - 1e-9:
                break
        umax = max(np.abs(ifft(Ui).real).max() for Ui in U)
        dt = min(2.0 / N, 0.5 * (2 * np.pi / N) / max(umax, 1e-9), mark - t + 1e-12)
        if lady is not None:
            dt = min(dt, 0.2 * (2 * np.pi / N) ** 2 / (lady[0] * (1 + lady[1] * 3.0)))     # explicit friction stability
        f = np.exp(-damp * dt) if damp is not None else 1.0
        f2 = np.exp(-damp * dt / 2) if damp is not None else 1.0
        rhs = (lambda V: [a + b for a, b in zip(transport(V), lady_term(V, *lady))]) if lady is not None else transport
        a = rhs(U)
        b = rhs([f2 * (U[i] + dt / 2 * a[i]) for i in range(3)])
        c = rhs([f2 * U[i] + dt / 2 * b[i] for i in range(3)])
        d = rhs([f * U[i] + dt * f2 * c[i] for i in range(3)])
        U = [f * U[i] + dt / 6 * (f * a[i] + 2 * f2 * b[i] + 2 * f2 * c[i] + d[i]) for i in range(3)]
        t += dt
    print("%-42s " % label + "   ".join("t=%.2f Z/Z0 %.3f max|w| %6.2f delta %.3f" % r for r in rows) + "   (%.0fs)" % (time.time() - t0), flush=True)
    return rows


def vort(U):
    return [ifft(1j * ky * U[2] - 1j * kz * U[1]).real, ifft(1j * kz * U[0] - 1j * kx * U[2]).real, ifft(1j * kx * U[1] - 1j * ky * U[0]).real]


def strip(U):
    e = 0.5 * sum(np.abs(Ui) ** 2 for Ui in U) / N**6
    spec = np.array([e[s].sum() for s in shells]); ks = np.arange(1, NB)
    sel = (ks >= NB // 2) & (spec > 1e-300)
    return -np.polyfit(ks[sel], np.log(spec[sel]), 1)[0] / 2 if sel.sum() > 4 else np.nan


if IC == "found":
    path = os.environ.get("FOUND", "results/found/leashed64_dmin030.npz")
    uf = np.load(path)["u"].astype(float); n0 = uf.shape[1]
    print("found field:", path)
    U0 = []
    for c in range(3):
        uh = fft(uf[c]) * (N / n0) ** 3
        big = np.zeros((N, N, N), complex); h = n0 // 2
        for a in (slice(0, h), slice(-h, None)):
            for b in (slice(0, h), slice(-h, None)):
                for cc in (slice(0, h), slice(-h, None)):
                    big[a, b, cc] = uh[a, b, cc]
        U0.append(big)
    U0 = project(U0)
else:
    U0 = [fft(np.sin(X) * np.cos(Y) * np.cos(Z_)), fft(-np.cos(X) * np.sin(Y) * np.cos(Z_)), fft(np.zeros_like(X))]
Zi = 0.5 * np.mean(sum(vi**2 for vi in vort(U0)))
U0 = [Ui * np.sqrt(0.375 / Zi) for Ui in U0]
Z0 = 0.375
G0 = max(np.abs(ifft(1j * K[i] * U0[j] * deal).real).max() for i in range(3) for j in range(3))
print("IC=%s at Z0=%.3f, N=%d^3, T=%.1f, nu=%g; 2dx = %.4f; initial max|grad u| = %.2f" % (IC, Z0, N, T, NU, 2 * 2 * np.pi / N, G0))
run("A  Navier-Stokes, constant nu, Laplacian", damp=NU * k2)
for alpha in (1.10, 1.25):
    nu_a = NU * KM ** (2 - 2 * alpha)                     # same damping as A at k = K_MATCH
    run("%s  hyperviscosity alpha = %.2f (nu_a matched at k=%g)" % ("B" if alpha == 1.25 else "C", alpha, KM), damp=nu_a * k2 ** alpha)
cL = 1.0 / np.sqrt(G0)                                    # nu doubles at the initial max strain
run("D  Ladyzhenskaya nu0(1 + c|grad u|^1/2), c=%.3f" % cL, lady=(NU, cL))
run("E  Euler (no friction), for reference", damp=None)
