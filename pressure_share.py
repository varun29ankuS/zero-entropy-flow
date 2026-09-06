"""Does the global inform the local? The pressure Hessian P = grad grad p splits into a LOCAL part, its trace
(lap p = |w|^2/2 - |S|^2, decided at the point), and a GLOBAL part, the traceless remainder, decided by the whole
field through the Poisson equation. Restricted Euler (Vieillefosse 1982; Cantwell 1992) keeps only the local part and
blows up in finite time; the global part is what prevents it. Along each flow this reports, on the high-vorticity set:
    share   = <|P_dev|^2> / <|P|^2>             the global share of the pressure Hessian
    back    = <xi . P_dev . xi> / <xi . S^2 . xi>  the global Hessian's push along the vorticity, against the local self-stretching
    corr    = correlation of (xi.P_dev.xi) with (xi.S.xi)^2 over the high set: does the global term oppose the strongest local stretching?
for the classical flows and for the fast field found by the adversarial searcher (results/found/*.npz).
usage: N=64 T=1.0 python pressure_share.py"""
import os, glob, time, numpy as np

N = int(os.environ.get("N", 64))
T = float(os.environ.get("T", 1.0))
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


def diag(U):
    Ud = [Ui * deal for Ui in U]
    G = [[ifft(1j * K[i] * Ud[j]).real for j in range(3)] for i in range(3)]
    S = np.stack([np.stack([0.5 * (G[i][j] + G[j][i]) for j in range(3)], -1) for i in range(3)], -1)
    w = [G[1][2] - G[2][1], G[2][0] - G[0][2], G[0][1] - G[1][0]]
    wmag = np.sqrt(sum(wi**2 for wi in w)) + 1e-30
    xi = np.stack([wi / wmag for wi in w], -1)
    src = fft(sum(G[i][j] * G[j][i] for i in range(3) for j in range(3)))
    ph = src / k2s * deal
    P = np.stack([np.stack([ifft(-K[i] * K[j] * ph).real for j in range(3)], -1) for i in range(3)], -1)
    trP = np.einsum("...ii->...", P)
    Pdev = P - trP[..., None, None] / 3 * np.eye(3)
    high = wmag > 0.5 * wmag.max()
    share = (np.einsum("...ij,...ij->...", Pdev, Pdev)[high].mean()) / (np.einsum("...ij,...ij->...", P, P)[high].mean() + 1e-30)
    xiPxi = np.einsum("...i,...ij,...j->...", xi, Pdev, xi)
    S2 = np.einsum("...ij,...jk->...ik", S, S)
    xiS2xi = np.einsum("...i,...ij,...j->...", xi, S2, xi)
    alpha = np.einsum("...i,...ij,...j->...", xi, S, xi)
    back = xiPxi[high].mean() / (xiS2xi[high].mean() + 1e-30)
    corr = np.corrcoef(xiPxi[high], (alpha**2)[high])[0, 1]
    Z = 0.5 * np.mean(wmag**2)
    return share, back, corr, Z


flows = {}
flows["taylor-green"] = [fft(np.sin(X) * np.cos(Y) * np.cos(Z_)), fft(-np.cos(X) * np.sin(Y) * np.cos(Z_)), fft(np.zeros_like(X))]
flows["kida-pelz"] = [fft(np.sin(X) * (np.cos(3 * Y) * np.cos(Z_) - np.cos(Y) * np.cos(3 * Z_))),
                      fft(np.sin(Y) * (np.cos(3 * Z_) * np.cos(X) - np.cos(Z_) * np.cos(3 * X))),
                      fft(np.sin(Z_) * (np.cos(3 * X) * np.cos(Y) - np.cos(X) * np.cos(3 * Y)))]
flows["abc"] = [fft(np.sin(Z_) + np.cos(Y)), fft(np.sin(X) + np.cos(Z_)), fft(np.sin(Y) + np.cos(X))]
for path in sorted(glob.glob("results/found/*.npz")):
    d = np.load(path)
    u = d["u"].astype(float)
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
    flows["found:" + os.path.basename(path)[:-4]] = project(U)

# common initial enstrophy, as in the searches
Z0 = 0.375
print("N=%d^3, nu=0, all flows at Z0=%.3f; high-vorticity set |w| > 0.5 max" % (N, Z0))
print("%-28s %5s   %8s   %8s   %7s   %7s" % ("flow", "t", "Z(t)/Z0", "share", "back", "corr"))
for name, U in flows.items():
    Zi = 0.5 * np.mean(sum(ifft(1j * K[a] * U[b] - 1j * K[b] * U[a]).real ** 2 for a, b in ((1, 2), (2, 0), (0, 1))))
    U = [Ui * np.sqrt(Z0 / Zi) for Ui in U]
    t, mark = 0.0, 0.0
    t0 = time.time()
    while t <= T + 1e-9:
        if t >= mark - 1e-9:
            share, back, corr, Z = diag(U)
            print("%-28s %5.2f   %8.3f   %8.3f   %+7.3f   %+7.3f" % (name, t, Z / Z0, share, back, corr), flush=True)
            mark += 0.25
            if t >= T - 1e-9:
                break
        umax = max(np.abs(ifft(Ui).real).max() for Ui in U)
        dt = min(2.0 / N, 0.5 * (2 * np.pi / N) / max(umax, 1e-9), mark - t + 1e-12)
        U = step(U, dt)
        t += dt
