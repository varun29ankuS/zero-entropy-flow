"""Does zero numerical entropy production let an integrator SEE a blow-up that a dissipative one hides?
Burgers u_t + u u_x = nu u_xx on [0, 2pi), u0 = sin x.  Inviscid truth: max|u_x| = 1/(1-t), blow-up at t*=1.
  upwind      first-order upwind finite differences (dissipative by construction)
  frozen      pseudo-spectral: linear transport of each Fourier mode is an exact rotation (the clock, mode by mode),
              nonlinearity in skew-symmetric (energy-conserving) form, RK4 in time, 2/3 dealiasing
  truth       characteristics: u(x,t) = sin(x - u t), solved by fixed point, valid for t < 1
Reports at t = 0.5, 0.8, 0.9, 0.95: energy E = mean(u^2) (exact: constant), max|u_x| vs 1/(1-t), L2 error vs truth,
and the numerical entropy production -d log E / dt. Then viscous nu=0.01: measured dissipation vs physical nu*mean(u_x^2)."""
import numpy as np
N = 512; x = np.linspace(0, 2 * np.pi, N, endpoint=False); dx = x[1] - x[0]; k = np.fft.fftfreq(N, d=1.0 / N)
u0 = np.sin(x); DT = 2e-4

def truth(t):
    u = np.sin(x).copy()
    for _ in range(200): u = np.sin(x - u * t)
    return u

def upwind_step(u, dt, nu):
    # conservative upwind flux for u u_x = (u^2/2)_x, plus explicit diffusion
    f = 0.5 * u * u; fm = np.roll(f, 1); fp = np.roll(f, -1)
    a = u; flux_l = np.where(a > 0, fm, f); flux_r = np.where(a > 0, f, fp)          # upwind choice
    du = -(flux_r - flux_l) / dx + nu * (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx ** 2
    return u + dt * du

dealias = np.abs(k) < N / 3
def rhs_spectral(uh, nu):
    u = np.fft.ifft(uh).real; ux = np.fft.ifft(1j * k * uh).real
    # skew-symmetric form: u u_x = (1/3)(u u_x + (u^2)_x)  -- conserves energy exactly in the semi-discrete system
    nl = (u * ux + np.fft.ifft(1j * k * np.fft.fft(u * u)).real) / 3.0
    nlh = np.fft.fft(nl) * dealias
    return -nlh - nu * k * k * uh
def frozen_step(uh, dt, nu):
    # RK4 on the skew-symmetric nonlinearity; the linear (diffusive) part could be an exact factor, kept explicit for parity
    k1 = rhs_spectral(uh, nu); k2 = rhs_spectral(uh + 0.5 * dt * k1, nu); k3 = rhs_spectral(uh + 0.5 * dt * k2, nu); k4 = rhs_spectral(uh + dt * k3, nu)
    return uh + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def grad_max(u): return np.abs(np.fft.ifft(1j * k * np.fft.fft(u)).real).max()
def energy(u): return np.mean(u * u)

for nu in (0.0, 0.01):
    print('\n==== nu = %.3f  (%s) ====' % (nu, 'inviscid: gradient blows up at t=1, energy exactly conserved' if nu == 0 else 'viscous: dissipation should equal nu*mean(u_x^2), nothing more'))
    u_up = u0.copy(); uh = np.fft.fft(u0); t = 0.0; E0 = energy(u0); marks = [0.5, 0.8, 0.9, 0.95]; mi = 0
    print('%6s | %-38s | %-38s | %s' % ('t', 'upwind: E/E0, max|u_x|, L2err', 'frozen: E/E0, max|u_x|, L2err', 'truth max|u_x|'))
    Eup_prev, Efr_prev, tprev = E0, E0, 0.0
    while mi < len(marks):
        u_up = upwind_step(u_up, DT, nu); uh = frozen_step(uh, DT, nu); t += DT
        if t >= marks[mi] - 1e-9:
            u_fr = np.fft.ifft(uh).real; tr = truth(t) if nu == 0 else None
            e_up, e_fr = energy(u_up), energy(u_fr)
            if nu == 0:
                print('%6.2f | %.5f  %7.2f  %.4f            | %.5f  %7.2f  %.4f            | %7.2f' % (t, e_up / E0, grad_max(u_up), np.sqrt(np.mean((u_up - tr) ** 2)), e_fr / E0, grad_max(u_fr), np.sqrt(np.mean((u_fr - tr) ** 2)), 1 / (1 - t)))
            else:
                phys_up = nu * np.mean(np.gradient(u_up, dx) ** 2); phys_fr = nu * np.mean(np.fft.ifft(1j * k * uh).real ** 2)
                meas_up = -(e_up - Eup_prev) / (t - tprev) / 2; meas_fr = -(e_fr - Efr_prev) / (t - tprev) / 2
                print('%6.2f | measured dissipation %.4f vs physical %.4f (x%.2f) | measured %.4f vs physical %.4f (x%.2f)' % (t, meas_up, phys_up, meas_up / phys_up, meas_fr, phys_fr, meas_fr / phys_fr))
            Eup_prev, Efr_prev, tprev = e_up, e_fr, t; mi += 1
    if nu == 0:
        print('numerical entropy production over [0,0.95]:  upwind %.3e /unit time   frozen %.3e /unit time' % (-np.log(energy(u_up) / E0) / 0.95, -np.log(energy(np.fft.ifft(uh).real) / E0) / 0.95))
