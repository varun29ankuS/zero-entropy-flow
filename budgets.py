"""Budget closure: with zero numerical entropy production, every balance law should close to machine precision,
including the UNSIGNED production terms whose 3-D cousin is the regularity problem. Tests where Hypothesis H of
THEORY.md is known to be true or false.

1-D Burgers  u_t + u u_x = nu u_xx
   energy      d/dt (1/2)<u^2>       = -nu <u_x^2>                            (transport skew: H-type identity)
   gradient    d/dt (1/2)<u_x^2>     = -(1/2)<u_x^3>  - nu <u_xx^2>           (production -(1/2)<u_x^3> has NO sign;
                                                                                inviscid: it drives the blow-up, H false)
   max         max|u(t)|             <= max|u_0|   for nu > 0                  (maximum principle: H TRUE for viscous Burgers)
2-D Navier-Stokes, decaying, vorticity w, velocity u = grad^perp psi
   energy      d/dt (1/2)<|u|^2>     = -nu <w^2>
   enstrophy   d/dt (1/2)<w^2>       = -nu <|grad w|^2>                       (no production term in 2-D: H TRUE with M = Z)
   palinstrophy d/dt (1/2)<|grad w|^2> = -<grad w . (grad u) grad w> - nu <(lap w)^2>   (production has NO sign: the 2-D
                                                                                cousin of 3-D vortex stretching)
Each budget is checked as: measured dE/dt (centred difference across one RK4 step) vs the right-hand side evaluated
from the field; residual relative to the size of the largest term. Frozen scheme vs first-order upwind (1-D).
REGISTERED: frozen residuals < 1e-6 relative for every budget; upwind residuals >> that (the residual is sigma_num)."""

import numpy as np

np.random.seed(0)

# ------------------------------------------------ 1-D Burgers ------------------------------------------------
N = 512
x = np.linspace(0, 2 * np.pi, N, endpoint=False)
dx = x[1] - x[0]
k = np.fft.fftfreq(N, d=1.0 / N)
deal = np.abs(k) < N / 3


def D(u, n=1):
    return np.fft.ifft((1j * k) ** n * np.fft.fft(u)).real


def rhs1(uh, nu):
    u = np.fft.ifft(uh).real
    ux = np.fft.ifft(1j * k * uh).real
    nl = (u * ux + np.fft.ifft(1j * k * np.fft.fft(u * u)).real) / 3.0
    return -np.fft.fft(nl) * deal - nu * k * k * uh


def frozen1(uh, dt, nu):
    a = rhs1(uh, nu)
    b = rhs1(uh + dt / 2 * a, nu)
    c = rhs1(uh + dt / 2 * b, nu)
    d = rhs1(uh + dt * c, nu)
    return uh + dt / 6 * (a + 2 * b + 2 * c + d)


def upwind1(u, dt, nu):
    f = 0.5 * u * u
    fm = np.roll(f, 1)
    fp = np.roll(f, -1)
    fl = np.where(u > 0, fm, f)
    fr = np.where(u > 0, f, fp)
    return u + dt * (-(fr - fl) / dx + nu * (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2)


def budgets1(u, nu):
    ux, uxx = D(u, 1), D(u, 2)
    return dict(energy=(-nu * np.mean(ux**2)), gradient=(-0.5 * np.mean(ux**3) - nu * np.mean(uxx**2)))


def quantities1(u):
    ux = D(u, 1)
    return dict(energy=0.5 * np.mean(u**2), gradient=0.5 * np.mean(ux**2))


def run1(nu, T, dt=1e-4, checks=(0.3, 0.6, 0.8)):
    u0 = np.sin(x)
    uh = np.fft.fft(u0)
    uu = u0.copy()
    t = 0.0
    rows = []
    maxu0 = np.abs(u0).max()
    ci = 0
    while ci < len(checks):
        # measure d/dt by centred difference across one step, evaluate rhs at the midpoint field
        uh_prev = uh.copy()
        uu_prev = uu.copy()
        uh = frozen1(uh, dt, nu)
        uu = upwind1(uu, dt, nu)
        t += dt
        if t >= checks[ci] - 1e-12:
            for name, before, after, mid in (
                (
                    "frozen",
                    np.fft.ifft(uh_prev).real,
                    np.fft.ifft(uh).real,
                    np.fft.ifft(frozen1(uh_prev, dt / 2, nu)).real,
                ),
                ("upwind", uu_prev, uu, upwind1(uu_prev, dt / 2, nu)),
            ):
                qb, qa, r = quantities1(before), quantities1(after), budgets1(mid, nu)
                for key in ("energy", "gradient"):
                    meas = (qa[key] - qb[key]) / dt
                    rhs = r[key]
                    scale = max(abs(meas), abs(rhs), 1e-300)
                    rows.append((t, name, key, meas, rhs, abs(meas - rhs) / scale))
            rows.append((t, "frozen", "max|u|/max|u0|", np.abs(np.fft.ifft(uh).real).max() / maxu0, None, None))
            ci += 1
    return rows


for nu, T in ((0.0, 0.8), (0.02, 0.8)):
    print(
        "\n==== 1-D Burgers, nu = %.2f  (%s) ===="
        % (
            nu,
            (
                "inviscid: H false, gradient production drives blow-up"
                if nu == 0
                else "viscous: H true via the maximum principle"
            ),
        )
    )
    print("  t    scheme   budget     measured d/dt    right-hand side   relative residual")
    for t, name, key, meas, rhs, res in run1(nu, T):
        if rhs is None:
            print("%4.2f   %-7s  %-10s %.6f" % (t, name, key, meas))
        else:
            print("%4.2f   %-7s  %-10s %+.6e   %+.6e   %.1e" % (t, name, key, meas, rhs, res))

# --------------------------------------------- 2-D decaying turbulence ---------------------------------------------
M = 128
km = np.fft.fftfreq(M, d=1.0 / M)
kx, ky = np.meshgrid(km, km, indexing="ij")
k2 = kx * kx + ky * ky
k2s = k2.copy()
k2s[0, 0] = 1.0
deal2 = (np.abs(kx) < M / 3) & (np.abs(ky) < M / 3)
NU2 = 1e-3


def vel2(wh):
    psih = wh / k2s
    psih[0, 0] = 0
    return np.fft.ifft2(1j * ky * psih).real, -np.fft.ifft2(1j * kx * psih).real


def rhs2(wh):
    u, v = vel2(wh)
    w = np.fft.ifft2(wh).real
    wx = np.fft.ifft2(1j * kx * wh).real
    wy = np.fft.ifft2(1j * ky * wh).real
    adv = np.fft.fft2(u * wx + v * wy)
    div = 1j * kx * np.fft.fft2(u * w) + 1j * ky * np.fft.fft2(v * w)
    return -0.5 * (adv + div) * deal2 - NU2 * k2 * wh


def step2(wh, dt):
    a = rhs2(wh)
    b = rhs2(wh + dt / 2 * a)
    c = rhs2(wh + dt / 2 * b)
    d = rhs2(wh + dt * c)
    return wh + dt / 6 * (a + 2 * b + 2 * c + d)


def q2(wh):
    u, v = vel2(wh)
    w = np.fft.ifft2(wh).real
    wx = np.fft.ifft2(1j * kx * wh).real
    wy = np.fft.ifft2(1j * ky * wh).real
    return dict(
        energy=0.5 * np.mean(u * u + v * v),
        enstrophy=0.5 * np.mean(w * w),
        palinstrophy=0.5 * np.mean(wx * wx + wy * wy),
    )


def b2(wh):
    u, v = vel2(wh)
    w = np.fft.ifft2(wh).real
    wx = np.fft.ifft2(1j * kx * wh).real
    wy = np.fft.ifft2(1j * ky * wh).real
    ux = np.fft.ifft2(1j * kx * np.fft.fft2(u)).real
    uy = np.fft.ifft2(1j * ky * np.fft.fft2(u)).real
    vx = np.fft.ifft2(1j * kx * np.fft.fft2(v)).real
    vy = np.fft.ifft2(1j * ky * np.fft.fft2(v)).real
    lapw = np.fft.ifft2(-k2 * wh).real
    prod = -np.mean(wx * (ux * wx + uy * wy) + wy * (vx * wx + vy * wy))  # -<grad w . (grad u) grad w>, no sign
    return (
        dict(
            energy=-NU2 * np.mean(w * w),
            enstrophy=-NU2 * np.mean(wx * wx + wy * wy),
            palinstrophy=prod - NU2 * np.mean(lapw**2),
        ),
        prod,
    )


wh = np.fft.fft2(np.random.standard_normal((M, M))) * ((k2 >= 9) & (k2 <= 64)) * deal2
wh *= 1.0 / np.sqrt(2 * q2(wh)["energy"])  # unit energy
dt = 2e-3
t = 0.0
print("\n==== 2-D decaying turbulence, nu = %.0e, %d^2, random initial vorticity in 3 <= |k| <= 8 ====" % (NU2, M))
print("  t    budget        measured d/dt    right-hand side   relative residual   (palinstrophy production term)")
for tc in (0.5, 1.0, 2.0, 4.0):
    while t < tc - 1e-12:
        whp = wh.copy()
        wh = step2(wh, dt)
        t += dt
    mid = step2(whp, dt / 2)
    qb, qa = q2(whp), q2(wh)
    r, prod = b2(mid)
    for key in ("energy", "enstrophy", "palinstrophy"):
        meas = (qa[key] - qb[key]) / dt
        rhs = r[key]
        scale = max(abs(meas), abs(rhs), 1e-300)
        print(
            "%4.1f   %-13s %+.6e   %+.6e   %.1e%s"
            % (
                t,
                key,
                meas,
                rhs,
                abs(meas - rhs) / scale,
                ("   production %+.4e" % prod) if key == "palinstrophy" else "",
            )
        )
    print("       enstrophy Z = %.6f (must never increase: H true in 2-D)" % qa["enstrophy"])
