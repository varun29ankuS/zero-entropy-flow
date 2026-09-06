"""Water's memory as an algorithm: Navier-Stokes solved by remembering Euler along noisy particle paths (2-D).

Cauchy (1815): vorticity is the initial vorticity remembered along particle paths, w(X(a,t), t) = grad X . w0(a); in
2-D there is no stretching and w(X(a,t), t) = w0(a) exactly. Constantin and Iyer (2008): Navier-Stokes is the same
memory read along Brownian-jittered paths and averaged, dX = u dt + sqrt(2 nu) dW - viscosity is not a separate
term, it is the jitter in the memory. In 2-D this is Chorin's random vortex method (1973).

Here: N_P particles start on a fine grid carrying w0, move with the flow velocity (bilinear from the grid) plus the
Brownian kick, are deposited back to the grid each step, and the velocity comes from the deposited vorticity by
Biot-Savart (FFT). No viscous term is ever applied: the only dissipation is the noise in the paths. The same flow is
run with the spectral instrument (exact viscous integrating factor) and the two are compared:
    nu = 0     particles remember exactly, the spectral run conserves enstrophy: the memory check (deposition error only)
    nu > 0     enstrophy must DECAY at the viscous rate from noise alone; the relative L2 error of the vorticity field
               and the enstrophy ratio against the spectral run are reported; a no-noise control shows what is lost
usage: N=128 NU=2e-3 PPC=4 T=2 python memory_paths.py"""
import os, time, numpy as np

N = int(os.environ.get("N", 128))
NU = float(os.environ.get("NU", 2e-3))
PPC = int(os.environ.get("PPC", 4))          # particles per cell (per axis: PPC^2 per cell)
T = float(os.environ.get("T", 2.0))
DT = float(os.environ.get("DT", 5e-3))
rng = np.random.default_rng(0)
fft, ifft = np.fft.fft2, np.fft.ifft2
L = 2 * np.pi
k = np.fft.fftfreq(N, d=1.0 / N)
kx, ky = np.meshgrid(k, k, indexing="ij")
k2 = kx * kx + ky * ky
k2s = k2.copy()
k2s[0, 0] = 1.0
deal = (np.abs(kx) < N / 3) & (np.abs(ky) < N / 3)
x = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, x, indexing="ij")
dx = L / N


def velocity(wh):
    psih = wh / k2s
    psih[0, 0] = 0
    return ifft(1j * ky * psih).real, -ifft(1j * kx * psih).real


def rhs_spectral(wh):
    u, v = velocity(wh)
    w = ifft(wh).real
    wx, wy = ifft(1j * kx * wh).real, ifft(1j * ky * wh).real
    return -0.5 * (fft(u * wx + v * wy) + 1j * kx * fft(u * w) + 1j * ky * fft(v * w)) * deal


def step_spectral(wh, dt, nu):
    f, f2 = np.exp(-nu * k2 * dt), np.exp(-nu * k2 * dt / 2)
    a = rhs_spectral(wh)
    b = rhs_spectral(f2 * (wh + dt / 2 * a))
    c = rhs_spectral(f2 * wh + dt / 2 * b)
    d = rhs_spectral(f * wh + dt * f2 * c)
    return f * wh + dt / 6 * (f * a + 2 * f2 * b + 2 * f2 * c + d)


def interp(field, px, py):
    """bilinear, periodic"""
    gx, gy = px / dx, py / dx
    i0, j0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
    fx, fy = gx - i0, gy - j0
    i0 %= N; j0 %= N
    i1, j1 = (i0 + 1) % N, (j0 + 1) % N
    return (field[i0, j0] * (1 - fx) * (1 - fy) + field[i1, j0] * fx * (1 - fy) + field[i0, j1] * (1 - fx) * fy + field[i1, j1] * fx * fy)


def deposit(px, py, val):
    """bilinear deposition (the adjoint of interp), normalised by the deposited weight so the field is a local average"""
    gx, gy = px / dx, py / dx
    i0, j0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
    fx, fy = gx - i0, gy - j0
    i0 %= N; j0 %= N
    i1, j1 = (i0 + 1) % N, (j0 + 1) % N
    num = np.zeros((N, N)); den = np.zeros((N, N))
    for ii, jj, wgt in ((i0, j0, (1 - fx) * (1 - fy)), (i1, j0, fx * (1 - fy)), (i0, j1, (1 - fx) * fy), (i1, j1, fx * fy)):
        np.add.at(num, (ii, jj), wgt * val)
        np.add.at(den, (ii, jj), wgt)
    return num / np.maximum(den, 1e-12)


# initial vorticity: a few random large-scale modes (the same field for both instruments)
wh0 = fft(rng.standard_normal((N, N))) * ((k2 >= 1) & (k2 <= 9)) * deal
u0, v0 = velocity(wh0)
wh0 *= 1.0 / np.sqrt(np.mean(u0 * u0 + v0 * v0))
w0 = ifft(wh0).real
Z0 = 0.5 * np.mean(w0 * w0)

# particles on a fine lattice, carrying w0 (their memory, never changed)
s = np.linspace(0, L, N * PPC, endpoint=False) + dx / (2 * PPC)
PX, PY = np.meshgrid(s, s, indexing="ij")
PX, PY = PX.ravel(), PY.ravel()
PW = interp(w0, PX, PY)
print("2-D, N=%d^2, %d particles (%d per cell), nu=%g, dt=%g, T=%.1f" % (N, len(PX), PPC * PPC, NU, DT, T))
print("   t     spectral Z/Z0    particles Z/Z0 (noise)    no-noise control Z/Z0    rel L2 error w (noise)    rel L2 error (no noise)")

wh = wh0.copy()
px, py = PX.copy(), PY.copy()          # noisy paths (Navier-Stokes by memory + jitter)
qx, qy = PX.copy(), PY.copy()          # no-noise control (Euler by memory)
t, mark, t0 = 0.0, 0.0, time.time()
sig = np.sqrt(2 * NU * DT)
while t <= T + 1e-9:
    wp = deposit(px, py, PW)
    wq = deposit(qx, qy, PW)
    if t >= mark - 1e-9:
        ws = ifft(wh).real
        Zs, Zp, Zq = 0.5 * np.mean(ws**2), 0.5 * np.mean(wp**2), 0.5 * np.mean(wq**2)
        ep = np.sqrt(np.mean((wp - ws) ** 2) / np.mean(ws**2))
        eq = np.sqrt(np.mean((wq - ws) ** 2) / np.mean(ws**2))
        print("%5.2f   %.5f         %.5f                   %.5f                  %.4f                     %.4f   (%.0fs)" % (t, Zs / Z0, Zp / Z0, Zq / Z0, ep, eq, time.time() - t0), flush=True)
        mark += 0.25
        if t >= T - 1e-9:
            break
    # velocities from the deposited vorticity (each instrument uses its own field), RK2 midpoint for the paths
    for (ax, ay, wf, noise) in ((px, py, wp, True), (qx, qy, wq, False)):
        u, v = velocity(fft(wf) * deal)
        ux, uy = interp(u, ax, ay), interp(v, ax, ay)
        mx, my = (ax + 0.5 * DT * ux) % L, (ay + 0.5 * DT * uy) % L
        wm = deposit(mx, my, PW)
        u, v = velocity(fft(wm) * deal)
        ax += DT * interp(u, mx, my)
        ay += DT * interp(v, mx, my)
        if noise:
            ax += sig * rng.standard_normal(len(ax))
            ay += sig * rng.standard_normal(len(ay))
        ax %= L; ay %= L
    wh = step_spectral(wh, DT, NU)
    t += DT
print("nu = %g: the particles never see a viscous term; whatever enstrophy they lose beyond the no-noise control is the memory being read through jitter" % NU)
