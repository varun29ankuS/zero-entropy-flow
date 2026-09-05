"""Taylor-Green vortex with the frozen-rotation spectral scheme. Two parts.
 A. 2-D viscous, exact truth: u = e^{-2 nu t} sin x cos y, v = -e^{-2 nu t} cos x sin y. Energy decays as e^{-4 nu t}.
    Vorticity form: w_t + u.grad(w) = nu lap(w). Nonlinearity in conservative form, RK4, 2/3 dealiasing;
    the viscous factor applied EXACTLY (integrating factor), so the only approximation is the nonlinear term.
 B. 3-D inviscid Euler, the Brachet 1983 benchmark. Vorticity-velocity pseudo-spectral, RK4, 2/3 dealiasing.
    Energy must be conserved exactly; enstrophy grows. Run at 32^3, 48^3, 64^3 and report where they diverge.
CPU only; a few minutes."""
import numpy as np, time
def k_grid(N, dims):
    k = np.fft.fftfreq(N, d=1.0 / N); return np.meshgrid(*([k] * dims), indexing='ij')

# ---------------- A. 2-D viscous Taylor-Green ----------------
def tg2d(N=128, nu=0.02, T=2.0, dt=2e-3):
    kx, ky = k_grid(N, 2); k2 = kx * kx + ky * ky; k2[0, 0] = 1.0; deal = (np.abs(kx) < N / 3) & (np.abs(ky) < N / 3)
    x = np.linspace(0, 2 * np.pi, N, endpoint=False); X, Y = np.meshgrid(x, x, indexing='ij')
    w = -2 * np.sin(X) * np.cos(Y)                                  # vorticity of the TG field
    wh = np.fft.fft2(w)
    def vel(wh):
        psih = wh / k2; psih[0, 0] = 0
        return np.fft.ifft2(1j * ky * psih).real, -np.fft.ifft2(1j * kx * psih).real     # u = psi_y, v = -psi_x
    def nl(wh):
        u, v = vel(wh); wx = np.fft.ifft2(1j * kx * wh).real; wy = np.fft.ifft2(1j * ky * wh).real
        return -np.fft.fft2(u * wx + v * wy) * deal
    E = lambda wh: (lambda uv: np.mean(uv[0] ** 2 + uv[1] ** 2))(vel(wh))
    E0 = E(wh); t = 0.0; out = []
    fac = np.exp(-nu * k2 * dt); fac2 = np.exp(-nu * k2 * dt / 2)
    while t < T - 1e-12:
        # RK4 with exact integrating factor for the viscous part
        a = nl(wh); b = nl(fac2 * (wh + dt / 2 * a)); c = nl(fac2 * wh + dt / 2 * b); d = nl(fac * wh + dt * fac2 * c)
        wh = fac * wh + dt / 6 * (fac * a + 2 * fac2 * b + 2 * fac2 * c + d); t += dt
        if abs(t - round(t * 2) / 2) < 1e-9:
            u, v = vel(wh); ut = np.exp(-2 * nu * t) * np.sin(X) * np.cos(Y); vt = -np.exp(-2 * nu * t) * np.cos(X) * np.sin(Y)
            out.append((t, E(wh) / E0, np.exp(-4 * nu * t), np.sqrt(np.mean((u - ut) ** 2 + (v - vt) ** 2))))
    return out

print('==== A. 2-D viscous Taylor-Green (nu = 0.02), exact solution known ====')
print('   t   E/E0 (scheme)   E/E0 (exact)    L2 error vs exact')
for t, e, ex, err in tg2d(): print('%4.1f   %.8f     %.8f     %.2e' % (t, e, ex, err))

# ---------------- B. 3-D inviscid Taylor-Green (Euler) ----------------
def tg3d(N, T=4.0, dt=None):
    dt = dt or 4.0 / N * 0.5
    kx, ky, kz = k_grid(N, 3); k2 = kx ** 2 + ky ** 2 + kz ** 2; k2[0, 0, 0] = 1.0
    deal = (np.abs(kx) < N / 3) & (np.abs(ky) < N / 3) & (np.abs(kz) < N / 3)
    x = np.linspace(0, 2 * np.pi, N, endpoint=False); X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    u = np.sin(X) * np.cos(Y) * np.cos(Z); v = -np.cos(X) * np.sin(Y) * np.cos(Z); w = np.zeros_like(u)
    U = [np.fft.fftn(u), np.fft.fftn(v), np.fft.fftn(w)]
    K = [kx, ky, kz]
    def project(F):                                                  # remove the compressible part
        div = sum(1j * K[i] * F[i] for i in range(3)) / k2
        return [F[i] - 1j * K[i] * div for i in range(3)]
    def rhs(U):
        u = [np.fft.ifftn(Ui).real for Ui in U]; out = []
        for i in range(3):                                           # -(u.grad)u_i, dealiased, then projected
            adv = sum(u[j] * np.fft.ifftn(1j * K[j] * U[i]).real for j in range(3))
            out.append(-np.fft.fftn(adv) * deal)
        return project(out)
    def energy(U): return 0.5 * sum(np.mean(np.fft.ifftn(Ui).real ** 2) for Ui in U)
    def enstrophy(U):
        u = U; wx = 1j * ky * u[2] - 1j * kz * u[1]; wy = 1j * kz * u[0] - 1j * kx * u[2]; wz = 1j * kx * u[1] - 1j * ky * u[0]
        return 0.5 * sum(np.mean(np.fft.ifftn(c).real ** 2) for c in (wx, wy, wz))
    E0 = energy(U); t = 0.0; rows = []; t0 = time.time()
    while t < T - 1e-12:
        a = rhs(U); b = rhs([U[i] + dt / 2 * a[i] for i in range(3)]); c = rhs([U[i] + dt / 2 * b[i] for i in range(3)]); d = rhs([U[i] + dt * c[i] for i in range(3)])
        U = [U[i] + dt / 6 * (a[i] + 2 * b[i] + 2 * c[i] + d[i]) for i in range(3)]; t += dt
        if abs(t - round(t)) < dt / 2: rows.append((round(t), energy(U) / E0, enstrophy(U)))
    return rows, time.time() - t0

print('\n==== B. 3-D inviscid Taylor-Green (Euler), energy must be conserved; enstrophy growth by resolution ====')
res = {}
for N in (32, 48, 64):
    rows, secs = tg3d(N); res[N] = rows
    print('N=%d^3 (%.0fs):  ' % (N, secs) + '  '.join('t=%d E/E0=%.6f Z=%.3f' % r for r in rows), flush=True)
print('\nenstrophy by resolution (agreement = trustworthy; divergence = the grid, not the flow):')
for i in range(len(res[32])):
    print('  t=%d   32^3 %.3f   48^3 %.3f   64^3 %.3f' % (res[32][i][0], res[32][i][2], res[48][i][2], res[64][i][2]))
