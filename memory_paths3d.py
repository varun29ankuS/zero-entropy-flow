"""Water's memory with stretching: 3-D Navier-Stokes from Cauchy's formula along jittered particle paths.

Cauchy (1815): w(X(a,t), t) = grad_a X . w0(a). The vorticity a parcel carries is its initial vorticity rotated and
stretched by the deformation of the path map: along the path, D w / Dt = (w . grad) u. Constantin-Iyer (2008):
Navier-Stokes is the same memory read along Brownian-jittered paths, dX = u dt + sqrt(2 nu) dW, and averaged.

Here: particles start on a fine lattice carrying w0; each step they (1) read u and grad u from the grid (trilinear),
(2) update their vorticity by the stretching D w/Dt = (w.grad)u along their own path, (3) move with u plus the
Brownian kick, (4) are deposited back to the grid; the velocity comes from the deposited vorticity by Biot-Savart
(FFT, with a Leray projection to keep the deposited field solenoidal). No viscous term is ever applied to the
vorticity. Against the spectral instrument (exact viscous integrating factor) from the same Taylor-Green data:
    nu = 0     the particles must reproduce the ENSTROPHY GROWTH by vortex stretching (Cauchy's formula, checked)
    nu > 0     and its viscous decay from the jitter alone
REGISTERED: PASS if Z(t)/Z0 of the particle method is within 5% of the spectral run at t = 1 for nu = 0 and
nu = 2e-3 at the finest particle count, and the error falls as particles per cell increase.
usage: N=32 PPC=2 NU=2e-3 T=1.0 python memory_paths3d.py"""
import os, time, numpy as np

N = int(os.environ.get("N", 32))
PPC = int(os.environ.get("PPC", 2))
NU = float(os.environ.get("NU", 2e-3))
T = float(os.environ.get("T", 1.0))
DT = float(os.environ.get("DT", 0.01))
rng = np.random.default_rng(0)
fft, ifft = np.fft.fftn, np.fft.ifftn
L = 2 * np.pi
k = np.fft.fftfreq(N, d=1.0 / N)
kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
K = [kx, ky, kz]
k2 = kx**2 + ky**2 + kz**2
k2s = k2.copy(); k2s[0, 0, 0] = 1.0
deal = (np.abs(kx) < N / 3) & (np.abs(ky) < N / 3) & (np.abs(kz) < N / 3)
x = np.linspace(0, L, N, endpoint=False)
X, Y, Z_ = np.meshgrid(x, x, x, indexing="ij")
dx = L / N


def project(F):
    kd = sum(K[i] * F[i] for i in range(3)) / k2s
    return [F[i] - K[i] * kd for i in range(3)]


def vel_from_vort(Wh):
    """u = curl^-1 w: u_hat = i k x w_hat / k^2 (solenoidal part of w only)"""
    Wh = project(Wh)
    ux = 1j * (ky * Wh[2] - kz * Wh[1]) / k2s
    uy = 1j * (kz * Wh[0] - kx * Wh[2]) / k2s
    uz = 1j * (kx * Wh[1] - ky * Wh[0]) / k2s
    for a in (ux, uy, uz):
        a[0, 0, 0] = 0
    return [ux * deal, uy * deal, uz * deal]


def transport(U):
    Ud = [Ui * deal for Ui in U]
    u = [ifft(Ui).real for Ui in Ud]
    out = []
    for i in range(3):
        adv = sum(u[j] * ifft(1j * K[j] * Ud[i]).real for j in range(3))
        div = sum(ifft(1j * K[j] * fft(u[j] * u[i])).real for j in range(3))
        out.append(-0.5 * fft(adv + div) * deal)
    return project(out)


def step_spectral(U, dt):
    f, f2 = np.exp(-NU * k2 * dt), np.exp(-NU * k2 * dt / 2)
    a = transport(U)
    b = transport([f2 * (U[i] + dt / 2 * a[i]) for i in range(3)])
    c = transport([f2 * U[i] + dt / 2 * b[i] for i in range(3)])
    d = transport([f * U[i] + dt * f2 * c[i] for i in range(3)])
    return [f * U[i] + dt / 6 * (f * a[i] + 2 * f2 * b[i] + 2 * f2 * c[i] + d[i]) for i in range(3)]


def vort(U):
    return [ifft(1j * ky * U[2] - 1j * kz * U[1]).real, ifft(1j * kz * U[0] - 1j * kx * U[2]).real, ifft(1j * kx * U[1] - 1j * ky * U[0]).real]


def cell_weights(px, py, pz):
    g = np.stack([px, py, pz]) / dx
    i0 = np.floor(g).astype(int)
    f = g - i0
    i0 %= N
    i1 = (i0 + 1) % N
    corners, wts = [], []
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                corners.append(((i1 if a else i0)[0], (i1 if b else i0)[1], (i1 if c else i0)[2]))
                wts.append((f[0] if a else 1 - f[0]) * (f[1] if b else 1 - f[1]) * (f[2] if c else 1 - f[2]))
    return corners, wts


def interp(fields, corners, wts):
    return [sum(F[ci] * w for ci, w in zip(corners, wts)) for F in fields]


def deposit(corners, wts, vals):
    num = [np.zeros((N, N, N)) for _ in vals]
    den = np.zeros((N, N, N))
    for ci, w in zip(corners, wts):
        for n, v in zip(num, vals):
            np.add.at(n, ci, w * v)
        np.add.at(den, ci, w)
    den = np.maximum(den, 1e-12)
    return [n / den for n in num]


# Taylor-Green
U0 = [fft(np.sin(X) * np.cos(Y) * np.cos(Z_)), fft(-np.cos(X) * np.sin(Y) * np.cos(Z_)), fft(np.zeros_like(X))]
w0 = vort(U0)
Z0 = 0.5 * sum(np.mean(wi**2) for wi in w0)
E0 = 0.5 * sum(np.mean(ifft(Ui).real ** 2) for Ui in U0)

s = np.linspace(0, L, N * PPC, endpoint=False) + dx / (2 * PPC)
PX, PY, PZ = [a.ravel() for a in np.meshgrid(s, s, s, indexing="ij")]
corners, wts = cell_weights(PX, PY, PZ)
PW = interp(w0, corners, wts)                 # each particle's vorticity vector: its memory, stretched along its path
NP = len(PX)
print("3-D Taylor-Green, N=%d^3, %d particles (%d per cell), nu=%g, dt=%g, T=%.1f" % (N, NP, PPC**3, NU, DT, T))
print("   t     spectral Z/Z0   E/E0      particles Z/Z0 (jitter)   Z/Z(0)_particles   E/E0      rel L2 error of w   (no viscous term on the particles)")
Zp0 = None
U = [Ui.copy() for Ui in U0]
px, py, pz = PX.copy(), PY.copy(), PZ.copy()
t, mark, t0 = 0.0, 0.0, time.time()
sig = np.sqrt(2 * NU * DT)
while t <= T + 1e-9:
    corners, wts = cell_weights(px, py, pz)
    wgrid = deposit(corners, wts, PW)
    Wh = [fft(w) * deal for w in wgrid]
    Uh = vel_from_vort(Wh)
    if t >= mark - 1e-9:
        ws = vort(U)
        wsol = vort(Uh)                                            # the solenoidal part of the deposited field
        Zs = 0.5 * sum(np.mean(wi**2) for wi in ws)
        Zp = 0.5 * sum(np.mean(wi**2) for wi in wsol)
        Es = 0.5 * sum(np.mean(ifft(Ui).real ** 2) for Ui in U)
        Ep = 0.5 * sum(np.mean(ifft(Ui).real ** 2) for Ui in Uh)
        err = np.sqrt(sum(np.mean((a - b) ** 2) for a, b in zip(wsol, ws)) / sum(np.mean(a**2) for a in ws))
        Zp0 = Zp if Zp0 is None else Zp0
        print("%5.2f   %.4f          %.5f    %.4f                    %.4f             %.5f    %.4f    (%.0fs)" % (t, Zs / Z0, Es / E0, Zp / Z0, Zp / Zp0, Ep / E0, err, time.time() - t0), flush=True)
        mark += 0.25
        if t >= T - 1e-9:
            break
    # particle update: read u and grad u at the particles, stretch the carried vorticity, move (RK2 midpoint), jitter
    ug = [ifft(Ui).real for Ui in Uh]
    G = [[ifft(1j * K[i] * Uh[j]).real for j in range(3)] for i in range(3)]      # G[i][j] = d_i u_j
    up = interp(ug, corners, wts)
    Gp = [[interp([G[i][j]], corners, wts)[0] for j in range(3)] for i in range(3)]
    # D w_j / Dt = w_i d_i u_j  (Cauchy: the memory stretched by the local deformation)
    dW = [sum(PW[i] * Gp[i][j] for i in range(3)) for j in range(3)]
    mx, my, mz = (px + 0.5 * DT * up[0]) % L, (py + 0.5 * DT * up[1]) % L, (pz + 0.5 * DT * up[2]) % L
    Wm = [PW[j] + 0.5 * DT * dW[j] for j in range(3)]
    cm, wm = cell_weights(mx, my, mz)
    um = interp(ug, cm, wm)
    Gm = [[interp([G[i][j]], cm, wm)[0] for j in range(3)] for i in range(3)]
    dWm = [sum(Wm[i] * Gm[i][j] for i in range(3)) for j in range(3)]
    PW = [PW[j] + DT * dWm[j] for j in range(3)]
    px = (px + DT * um[0] + sig * rng.standard_normal(NP)) % L
    py = (py + DT * um[1] + sig * rng.standard_normal(NP)) % L
    pz = (pz + DT * um[2] + sig * rng.standard_normal(NP)) % L
    U = step_spectral(U, DT)
    t += DT
