"""3-D Euler: the mechanism of vortex stretching, a second invariant, and the BKM quantity.

  IC=tg   Taylor-Green (Brachet 1983)        IC=abc   Arnold-Beltrami-Childress flow + 10% perturbation, helical (helicity H = <u.w> != 0, conserved)

Printed at each integer time:
  E/E0        energy (conserved)
  H           helicity <u.w>  (conserved for Euler; zero by symmetry for Taylor-Green, nonzero for ABC)
  Z           enstrophy
  max|w|      the Beale-Kato-Majda quantity: regularity on [0,T] iff  int_0^T max|w| dt < infinity
  alignment   fraction of grid points where the vorticity is most aligned with the strain tensor's
              (most stretching, intermediate, most compressing) eigenvector. The classical result (Ashurst et al. 1987,
              confirmed in every DNS since) is a preference for the INTERMEDIATE eigenvector; random would be 1/3 each.
usage: IC=tg|abc N=64 T=4 python mechanism3d.py
"""
import os, time, numpy as np

IC = os.environ.get("IC", "tg")
N = int(os.environ.get("N", 48))
T = float(os.environ.get("T", 4.0))
dt = 2.0 / N
k = np.fft.fftfreq(N, d=1.0 / N)
kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
K = [kx, ky, kz]
k2 = kx**2 + ky**2 + kz**2
k2[0, 0, 0] = 1.0
deal = (np.abs(kx) < N / 3) & (np.abs(ky) < N / 3) & (np.abs(kz) < N / 3)
x = np.linspace(0, 2 * np.pi, N, endpoint=False)
X, Y, Z_ = np.meshgrid(x, x, x, indexing="ij")
fft, ifft = np.fft.fftn, np.fft.ifftn

if IC == "abc":
    A = B = C = 1.0
    U = [fft(A * np.sin(Z_) + C * np.cos(Y)), fft(B * np.sin(X) + A * np.cos(Z_)), fft(C * np.sin(Y) + B * np.cos(X))]
    # ABC alone is Beltrami (w = u): an exact steady Euler solution. A 10% divergence-free low-k perturbation makes it move.
    rng = np.random.default_rng(0)
    pert = [fft(rng.standard_normal(X.shape)) * ((k2 >= 1) & (k2 <= 9)) * deal for _ in range(3)]
    kdot = sum(K[i] * pert[i] for i in range(3)) / k2
    pert = [pert[i] - K[i] * kdot for i in range(3)]
    scale = 0.1 * np.sqrt(sum(np.mean(ifft(Ui).real ** 2) for Ui in U) / sum(np.mean(ifft(p).real ** 2) for p in pert))
    U = [U[i] + scale * pert[i] for i in range(3)]
else:
    U = [fft(np.sin(X) * np.cos(Y) * np.cos(Z_)), fft(-np.cos(X) * np.sin(Y) * np.cos(Z_)), fft(np.zeros_like(X))]


def project(F):
    kdotF = sum(K[i] * F[i] for i in range(3)) / k2
    return [F[i] - K[i] * kdotF for i in range(3)]


def rhs(U):
    Ud = [Ui * deal for Ui in U]
    u = [ifft(Ui).real for Ui in Ud]
    out = []
    for i in range(3):
        adv = sum(u[j] * ifft(1j * K[j] * Ud[i]).real for j in range(3))
        div = sum(ifft(1j * K[j] * fft(u[j] * u[i])).real for j in range(3))
        out.append(-0.5 * fft(adv + div) * deal)
    return project(out)


def step(U, dt):
    a = rhs(U)
    b = rhs([U[i] + dt / 2 * a[i] for i in range(3)])
    c = rhs([U[i] + dt / 2 * b[i] for i in range(3)])
    d = rhs([U[i] + dt * c[i] for i in range(3)])
    return [U[i] + dt / 6 * (a[i] + 2 * b[i] + 2 * c[i] + d[i]) for i in range(3)]


def vort(U):
    return [
        ifft(1j * ky * U[2] - 1j * kz * U[1]).real,
        ifft(1j * kz * U[0] - 1j * kx * U[2]).real,
        ifft(1j * kx * U[1] - 1j * ky * U[0]).real,
    ]


def diagnostics(U):
    u = [ifft(Ui).real for Ui in U]
    w = vort(U)
    E = 0.5 * sum(np.mean(ui**2) for ui in u)
    H = sum(np.mean(u[i] * w[i]) for i in range(3))
    Z = 0.5 * sum(np.mean(wi**2) for wi in w)
    wmag = np.sqrt(sum(wi**2 for wi in w))
    # strain tensor S_ij = (d_i u_j + d_j u_i)/2, its eigenvectors, and the alignment of w with each
    G = [[ifft(1j * K[i] * U[j]).real for j in range(3)] for i in range(3)]
    S = np.stack([[0.5 * (G[i][j] + G[j][i]) for j in range(3)] for i in range(3)], axis=-1)  # [...,3,3] flattened order
    S = np.moveaxis(S.reshape(3, 3, -1), -1, 0)  # [P,3,3]
    W = np.stack([wi.reshape(-1) for wi in w], -1)  # [P,3]
    evals, evecs = np.linalg.eigh(S)  # ascending: compressing, intermediate, stretching
    cos = np.abs(np.einsum("pi,pij->pj", W, evecs)) / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    best = np.argmax(cos, axis=1)
    frac = [np.mean(best == 2), np.mean(best == 1), np.mean(best == 0)]  # stretching, intermediate, compressing
    return E, H, Z, wmag.max(), frac


E0, H0, _, _, _ = diagnostics(U)
t = 0.0
t0 = time.time()
mark = 1.0
print("IC=%s  N=%d^3  dt=%.4f   (alignment: fraction of points whose vorticity aligns best with the strain eigenvector; random = 0.33 each)" % (IC, N, dt))
print("   t     E/E0       helicity H      Z        max|w|     align: stretching  intermediate  compressing")
E, H, Z, wm, fr = diagnostics(U)
print("%5.2f   %.6f   %+.6e   %7.4f   %7.3f      %.3f         %.3f          %.3f" % (0, 1.0, H, Z, wm, *fr), flush=True)
while t < T - 1e-9:
    U = step(U, dt)
    t += dt
    if t >= mark - dt / 2:
        E, H, Z, wm, fr = diagnostics(U)
        print("%5.2f   %.6f   %+.6e   %7.4f   %7.3f      %.3f         %.3f          %.3f   (%.0fs)" % (t, E / E0, H, Z, wm, *fr, time.time() - t0), flush=True)
        mark += 1.0
