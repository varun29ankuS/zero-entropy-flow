"""Animated flows from the same solver, as GIFs, with the conserved quantities printed on every frame.
  figures/turbulence_2d.gif   2-D decaying turbulence, 256^2, nu = 1e-4: vortices merge, filaments stretch, enstrophy
                              cascades to small scales; energy stays put. Vorticity, inferno colourmap.
  figures/burgers_shock.gif   1-D Burgers, u0 = sin x: inviscid (energy exactly conserved, gradient -> 1/(1-t)) and
                              viscous (Cole-Hopf exact overlaid), side by side.
usage: python flow_gif.py   (a few minutes on CPU)"""
import os, io, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

fft, ifft = np.fft.fftn, np.fft.ifftn
os.makedirs("figures", exist_ok=True)


def to_gif(frames, path, ms=60):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)
    print(path, "%d frames, %.1f MB" % (len(frames), os.path.getsize(path) / 1e6))


def fig_to_img(fig, w=None):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf)
    im = im.resize((int(im.width * 0.8), int(im.height * 0.8)), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=64)
    return im


# ------------------------------------------ 2-D decaying turbulence ------------------------------------------
M, nu, dt = 256, 1e-4, 1e-3
k = np.fft.fftfreq(M, d=1.0 / M)
kx, ky = np.meshgrid(k, k, indexing="ij")
k2 = kx * kx + ky * ky
k2s = k2.copy()
k2s[0, 0] = 1.0
deal = (np.abs(kx) < M / 3) & (np.abs(ky) < M / 3)


def vel(wh):
    psih = wh / k2s
    psih[0, 0] = 0
    return ifft(1j * ky * psih).real, -ifft(1j * kx * psih).real


def rhs(wh):
    u, v = vel(wh)
    w = ifft(wh).real
    wx, wy = ifft(1j * kx * wh).real, ifft(1j * ky * wh).real
    return -0.5 * (fft(u * wx + v * wy) + 1j * kx * fft(u * w) + 1j * ky * fft(v * w)) * deal


rng = np.random.default_rng(1)
wh = fft(rng.standard_normal((M, M))) * ((k2 >= 9) & (k2 <= 36)) * deal
u, v = vel(wh)
wh *= 1.0 / np.sqrt(np.mean(u * u + v * v))
u, v = vel(wh)
fac, fac2 = np.exp(-nu * k2 * dt), np.exp(-nu * k2 * dt / 2)
E0 = 0.5 * np.mean(u * u + v * v)
frames = []
t = 0.0
T, every = 12.0, 0.1
next_frame = 0.0
vmax = np.abs(ifft(wh).real).max()
while t < T:
    if t >= next_frame - 1e-9:
        w = ifft(wh).real
        u, v = vel(wh)
        E = 0.5 * np.mean(u * u + v * v)
        Z = 0.5 * np.mean(w * w)
        fig, ax = plt.subplots(figsize=(4.6, 4.9))
        ax.imshow(w.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="bilinear")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("2-D turbulence, 256$^2$   t = %5.2f\nenergy / E0 = %.5f      enstrophy = %.3f" % (t, E / E0, Z), fontsize=9, loc="left")
        fig.tight_layout()
        frames.append(fig_to_img(fig))
        next_frame += every
    a = rhs(wh)
    b = rhs(fac2 * (wh + dt / 2 * a))
    c = rhs(fac2 * wh + dt / 2 * b)
    d = rhs(fac * wh + dt * fac2 * c)
    wh = fac * wh + dt / 6 * (fac * a + 2 * fac2 * b + 2 * fac2 * c + d)
    t += dt
to_gif(frames, "figures/turbulence_2d.gif", ms=50)

# ------------------------------------------------ Burgers shock ------------------------------------------------
N = 512
x = np.linspace(0, 2 * np.pi, N, endpoint=False)
k1 = np.fft.fftfreq(N, d=1.0 / N)
deal1 = np.abs(k1) < N / 3


def rhs1(uh, nu):
    u = ifft(uh).real
    ux = ifft(1j * k1 * uh).real
    return -fft((u * ux + ifft(1j * k1 * fft(u * u)).real) / 3.0) * deal1 - nu * k1 * k1 * uh


def step1(uh, dt, nu):
    a = rhs1(uh, nu)
    b = rhs1(uh + dt / 2 * a, nu)
    c = rhs1(uh + dt / 2 * b, nu)
    d = rhs1(uh + dt * c, nu)
    return uh + dt / 6 * (a + 2 * b + 2 * c + d)


yq = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
NUV = 0.02
logphi0 = np.cos(yq) / (2 * NUV)


def cole_hopf(t):
    out = np.empty_like(x)
    for i, xi in enumerate(x):
        d = xi - yq[None, :] + 2 * np.pi * np.arange(-2, 3)[:, None]
        wgt = logphi0[None, :] - d**2 / (4 * NUV * t)
        wgt -= wgt.max()
        e = np.exp(wgt)
        out[i] = -2 * NUV * (e * (-d / (2 * NUV * t))).sum() / e.sum()
    return out


ui, uv = fft(np.sin(x)), fft(np.sin(x))
E0 = np.mean(np.sin(x) ** 2)
dt1 = 2e-4
frames = []
t = 0.0
next_frame = 0.0
while t < 1.6:
    if t >= next_frame - 1e-9:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
        u1 = ifft(ui).real
        u2 = ifft(uv).real
        g = np.abs(np.gradient(u1, x)).max()
        a1.plot(x, u1, color="#1b1b1b", lw=1.6)
        a1.set_ylim(-1.15, 1.15)
        ti = min(t, 0.95)
        a1.set_title("inviscid Burgers   t = %.2f%s\nenergy / E0 = %.12f   max|u_x| = %.1f  (exact 1/(1-t) = %.1f)" % (ti, "   stopped at the grid limit" if t > 0.95 else "", np.mean(u1**2) / E0, g, 1 / (1 - ti)), fontsize=8.5, loc="left")
        if t > 0.02:
            a2.plot(x, cole_hopf(t), color="#8a8a8a", lw=5, alpha=0.4, label="Cole-Hopf exact")
        a2.plot(x, u2, color="#d9731f", lw=1.6, label="scheme")
        a2.set_ylim(-1.15, 1.15)
        a2.set_title("viscous Burgers, nu = 0.02   t = %.2f\nerror vs exact = %.1e" % (t, np.sqrt(np.mean((u2 - cole_hopf(t)) ** 2)) if t > 0.02 else 0.0), fontsize=8.5, loc="left")
        a2.legend(frameon=False, fontsize=8, loc="lower left")
        for a in (a1, a2):
            a.set_xticks([])
            a.set_yticks([])
        fig.tight_layout()
        frames.append(fig_to_img(fig))
        next_frame += 0.02
    ui = step1(ui, dt1, 0.0) if t < 0.95 else ui
    uv = step1(uv, dt1, NUV)
    t += dt1
to_gif(frames, "figures/burgers_shock.gif", ms=60)
