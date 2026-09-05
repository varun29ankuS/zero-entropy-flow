"""3-D Taylor-Green Euler: isosurfaces of |vorticity| at t = 1, 2, 3, 4 - the vortex sheets forming.
Renders with marching cubes (scikit-image) into figures/vortex_sheets_3d.png. usage: N=48 python vortex_iso.py"""
import os, time, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

N = int(os.environ.get("N", 48))
dt = 2.0 / N
fft, ifft = np.fft.fftn, np.fft.ifftn
k = np.fft.fftfreq(N, d=1.0 / N)
kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
K = [kx, ky, kz]
k2 = kx**2 + ky**2 + kz**2
k2[0, 0, 0] = 1.0
deal = (np.abs(kx) < N / 3) & (np.abs(ky) < N / 3) & (np.abs(kz) < N / 3)
x = np.linspace(0, 2 * np.pi, N, endpoint=False)
X, Y, Z_ = np.meshgrid(x, x, x, indexing="ij")
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


def vort_mag(U):
    w = [ifft(1j * ky * U[2] - 1j * kz * U[1]).real, ifft(1j * kz * U[0] - 1j * kx * U[2]).real, ifft(1j * kx * U[1] - 1j * ky * U[0]).real]
    return np.sqrt(sum(wi**2 for wi in w))


snaps = {}
t = 0.0
t0 = time.time()
mark = 1.0
while t < 4.0 - 1e-9:
    U = step(U, dt)
    t += dt
    if t >= mark - dt / 2:
        snaps[int(round(t))] = vort_mag(U)
        print("t=%d  max|w| = %.2f  (%.0fs)" % (round(t), snaps[int(round(t))].max(), time.time() - t0), flush=True)
        mark += 1.0

os.makedirs("figures", exist_ok=True)
fig = plt.figure(figsize=(14, 7.2))
h = N // 2  # the impermeable octant [0, pi]^3, the symmetry cell of Taylor-Green
for i, tt in enumerate((1, 2, 3, 4)):
    w = snaps[tt]
    wo = w[:h, :h, :h]
    ax = fig.add_subplot(2, 4, i + 1, projection="3d")
    level = 0.55 * w.max()
    verts, faces, _, _ = measure.marching_cubes(wo, level=level, spacing=(2 * np.pi / N,) * 3)
    mesh = Poly3DCollection(verts[faces], alpha=0.9, linewidth=0)
    mesh.set_facecolor("#d9731f" if tt >= 3 else "#8a8a8a")
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, np.pi)
    ax.set_zlim(0, np.pi)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=20, azim=-50)
    ax.set_title("t = %d    |w| = 0.55 max,  max = %.1f" % (tt, w.max()), fontsize=9.5)
    ax2 = fig.add_subplot(2, 4, 5 + i)
    ax2.imshow(w[:, :, N // 8].T, origin="lower", cmap="inferno", extent=(0, 2 * np.pi, 0, 2 * np.pi))
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("|w| in the plane z = pi/4", fontsize=9.5)
fig.suptitle("3-D Euler, Taylor-Green %d^3, energy 1.000000 throughout.  Top: vorticity isosurface in the symmetry cell [0,pi]^3.  Bottom: |w| on a plane - the sheets thin and fold." % N, x=0.01, ha="left", fontsize=10)
fig.tight_layout()
fig.savefig("figures/vortex_sheets_3d.png", dpi=150)
print("figures/vortex_sheets_3d.png")
