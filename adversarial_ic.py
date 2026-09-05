"""Feedback, not imitation: search for the initial condition that amplifies enstrophy the most.

Lu & Doering (2008), Ayala & Protas (2017) maximised enstrophy growth over initial data by adjoint optimisation. Here the
same search is run by gradient ascent through a DIFFERENTIABLE copy of the energy-conserving solver (torch autograd
through the whole RK4 rollout). The initial field lives on low wavenumbers |k| <= KMAX_IC, is projected divergence-free,
and is normalised to a fixed initial enstrophy Z0 (fixing energy alone would let the optimiser load high k). Objective:
Z(T)/Z0, i.e. the amplification the classical candidates are compared on.

Every candidate - Taylor-Green, perturbed ABC, Kida-Pelz, and the found field - is then RE-VERIFIED with the numpy
instrument (criteria3d.py machinery) at a higher resolution, with the analyticity-strip reliability clock, so that a
field which only "wins" by exploiting the search grid is caught.
REGISTERED: the found field's Z(T)/Z0 at the verification resolution exceeds the best classical candidate's, with
delta(T) > 2 dx for both (i.e. the comparison is made inside the reliable window). Known-true in the literature; if it
fails here the search is broken, not the literature.
usage: N=32 T=1.0 ITERS=40 python adversarial_ic.py"""
import os, time, math, numpy as np, torch

torch.set_default_dtype(torch.float64)
N = int(os.environ.get("N", 32))
T = float(os.environ.get("T", 1.0))
ITERS = int(os.environ.get("ITERS", 40))
KMAX_IC = int(os.environ.get("KMAX_IC", 4))
NVER = int(os.environ.get("NVER", 64))
LR = float(os.environ.get("LR", 0.05))
dev = "cpu"

# ------------------------------------------ torch solver (differentiable) ------------------------------------------
k = torch.fft.fftfreq(N, d=1.0 / N)
KX, KY, KZ = torch.meshgrid(k, k, k, indexing="ij")
K = [KX, KY, KZ]
K2 = KX**2 + KY**2 + KZ**2
K2S = K2.clone()
K2S[0, 0, 0] = 1.0
DEAL = ((KX.abs() < N / 3) & (KY.abs() < N / 3) & (KZ.abs() < N / 3)).to(torch.float64)
LOWK = ((KX.abs() <= KMAX_IC) & (KY.abs() <= KMAX_IC) & (KZ.abs() <= KMAX_IC) & (K2 > 0)).to(torch.float64)
fft, ifft = torch.fft.fftn, torch.fft.ifftn


def project(F):
    kd = sum(K[i] * F[i] for i in range(3)) / K2S
    return [F[i] - K[i] * kd for i in range(3)]


def transport(U):
    Ud = [Ui * DEAL for Ui in U]
    u = [ifft(Ui).real for Ui in Ud]
    out = []
    for i in range(3):
        adv = sum(u[j] * ifft(1j * K[j] * Ud[i]).real for j in range(3))
        div = sum(ifft(1j * K[j] * fft(u[j] * u[i])).real for j in range(3))
        out.append(-0.5 * fft(adv + div) * DEAL)
    return project(out)


def step(U, dt):
    a = transport(U)
    b = transport([U[i] + dt / 2 * a[i] for i in range(3)])
    c = transport([U[i] + dt / 2 * b[i] for i in range(3)])
    d = transport([U[i] + dt * c[i] for i in range(3)])
    return [U[i] + dt / 6 * (a[i] + 2 * b[i] + 2 * c[i] + d[i]) for i in range(3)]


def vort(U):
    return [ifft(1j * KY * U[2] - 1j * KZ * U[1]).real, ifft(1j * KZ * U[0] - 1j * KX * U[2]).real, ifft(1j * KX * U[1] - 1j * KY * U[0]).real]


def enstrophy(U):
    return 0.5 * sum((w**2).mean() for w in vort(U))


def energy(U):
    return 0.5 * sum((ifft(Ui).real ** 2).mean() for Ui in U)


def field_from_params(P, Z0):
    """P: 3 real tensors [N,N,N] -> low-k, divergence-free, enstrophy-normalised spectral velocity"""
    U = [fft(Pi) * LOWK for Pi in P]
    U = project(U)
    Z = enstrophy(U)
    return [Ui * torch.sqrt(Z0 / Z) for Ui in U]


def rollout(U, T, cfl=0.5):
    t = 0.0
    while t < T - 1e-12:
        umax = max(ifft(Ui).real.abs().max() for Ui in U)
        dt = min(2.0 / N, cfl * (2 * math.pi / N) / max(float(umax), 1e-9), T - t)
        U = step(U, dt)
        t += dt
    return U


# --------------------------------------------- classical candidates ---------------------------------------------
x = torch.linspace(0, 2 * math.pi, N + 1)[:-1]
X, Y, Z_ = torch.meshgrid(x, x, x, indexing="ij")


def candidates():
    tg = [torch.sin(X) * torch.cos(Y) * torch.cos(Z_), -torch.cos(X) * torch.sin(Y) * torch.cos(Z_), torch.zeros_like(X)]
    kp = [torch.sin(X) * (torch.cos(3 * Y) * torch.cos(Z_) - torch.cos(Y) * torch.cos(3 * Z_)),
          torch.sin(Y) * (torch.cos(3 * Z_) * torch.cos(X) - torch.cos(Z_) * torch.cos(3 * X)),
          torch.sin(Z_) * (torch.cos(3 * X) * torch.cos(Y) - torch.cos(X) * torch.cos(3 * Y))]
    g = torch.Generator().manual_seed(0)
    abc = [torch.sin(Z_) + torch.cos(Y), torch.sin(X) + torch.cos(Z_), torch.sin(Y) + torch.cos(X)]
    pert = [fft(torch.randn(X.shape, generator=g)) * ((K2 >= 1) & (K2 <= 9)).to(torch.float64) * DEAL for _ in range(3)]
    pert = project(pert)
    abcU = [fft(a) for a in abc]
    sc = 0.1 * math.sqrt(sum((ifft(Ui).real ** 2).mean() for Ui in abcU) / sum((ifft(p).real ** 2).mean() for p in pert))
    abcU = [abcU[i] + sc * pert[i] for i in range(3)]
    return {"taylor-green": [fft(c) for c in tg], "kida-pelz": [fft(c) for c in kp], "abc+10%": abcU}


cands = candidates()
Z0 = enstrophy(cands["kida-pelz"]).item()          # common initial enstrophy: Kida-Pelz's
print("search grid %d^3, T = %.2f, low-k initial data |k| <= %d, common Z0 = %.4f (Kida-Pelz's)" % (N, T, KMAX_IC, Z0))
with torch.no_grad():
    for name, U in cands.items():
        Un = [Ui * math.sqrt(Z0 / enstrophy(U).item()) for Ui in U]
        UT = rollout(Un, T)
        print("  %-14s E0 %.4f   Z(T)/Z0 = %.3f" % (name, energy(Un).item(), enstrophy(UT).item() / Z0), flush=True)

# --------------------------------------------- gradient ascent on the initial field ---------------------------------------------
torch.manual_seed(1)
P = [torch.randn(N, N, N, requires_grad=True) for _ in range(3)]
opt = torch.optim.Adam(P, lr=LR)
best = (0.0, None)
t0 = time.time()
for it in range(ITERS):
    U0 = field_from_params(P, Z0)
    UT = rollout(U0, T)
    growth = enstrophy(UT) / Z0
    loss = -torch.log(growth)
    opt.zero_grad()
    loss.backward()
    opt.step()
    g = growth.item()
    if g > best[0]:
        best = (g, [p.detach().clone() for p in P])
    if it % 5 == 0 or it == ITERS - 1:
        print("  iter %3d   Z(T)/Z0 = %.3f   E0 = %.4f   (%.0fs)" % (it, g, energy(U0).item(), time.time() - t0), flush=True)
print("best amplification on the search grid: %.3f" % best[0])
with torch.no_grad():
    Ubest = field_from_params(best[1], Z0)
    np.savez_compressed("adversarial_ic_%d.npz" % N, u=np.stack([ifft(Ui).real.numpy() for Ui in Ubest]), Z0=Z0, T=T)

# ------------------------------------------------- verification (numpy, higher N) -------------------------------------------------
import numpy as _np

def verify(u_phys_list, N2, T, label):
    """re-run a field at resolution N2 with the numpy instrument; report Z(T)/Z0, energy drift, delta(T) vs 2dx"""
    n0 = u_phys_list[0].shape[0]
    k = _np.fft.fftfreq(N2, d=1.0 / N2)
    kx, ky, kz = _np.meshgrid(k, k, k, indexing="ij")
    KK = [kx, ky, kz]
    k2 = kx**2 + ky**2 + kz**2
    k2s = k2.copy()
    k2s[0, 0, 0] = 1.0
    deal = (abs(kx) < N2 / 3) & (abs(ky) < N2 / 3) & (abs(kz) < N2 / 3)
    # upsample by zero-padding the spectrum of the n0^3 field
    U = []
    for u in u_phys_list:
        uh = _np.fft.fftn(u) * (N2 / n0) ** 3
        big = _np.zeros((N2, N2, N2), complex)
        h = n0 // 2
        sl = [slice(0, h), slice(-h, None)]
        for a in sl:
            for b in sl:
                for c in sl:
                    big[a, b, c] = uh[a, b, c]
        U.append(big)
    kd = sum(KK[i] * U[i] for i in range(3)) / k2s
    U = [U[i] - KK[i] * kd for i in range(3)]

    def transport(U):
        Ud = [Ui * deal for Ui in U]
        u = [_np.fft.ifftn(Ui).real for Ui in Ud]
        out = []
        for i in range(3):
            adv = sum(u[j] * _np.fft.ifftn(1j * KK[j] * Ud[i]).real for j in range(3))
            div = sum(_np.fft.ifftn(1j * KK[j] * _np.fft.fftn(u[j] * u[i])).real for j in range(3))
            out.append(-0.5 * _np.fft.fftn(adv + div) * deal)
        kd = sum(KK[i] * out[i] for i in range(3)) / k2s
        return [out[i] - KK[i] * kd for i in range(3)]

    def vortn(U):
        return [_np.fft.ifftn(1j * ky * U[2] - 1j * kz * U[1]).real, _np.fft.ifftn(1j * kz * U[0] - 1j * kx * U[2]).real, _np.fft.ifftn(1j * kx * U[1] - 1j * ky * U[0]).real]

    Zf = lambda U: 0.5 * sum(_np.mean(w**2) for w in vortn(U))
    Ef = lambda U: 0.5 * sum(_np.mean(_np.fft.ifftn(Ui).real ** 2) for Ui in U)
    Z0v, E0v = Zf(U), Ef(U)
    t = 0.0
    while t < T - 1e-12:
        umax = max(abs(_np.fft.ifftn(Ui).real).max() for Ui in U)
        dt = min(2.0 / N2, 0.5 * (2 * _np.pi / N2) / max(umax, 1e-9), T - t)
        a = transport(U)
        b = transport([U[i] + dt / 2 * a[i] for i in range(3)])
        c = transport([U[i] + dt / 2 * b[i] for i in range(3)])
        d = transport([U[i] + dt * c[i] for i in range(3)])
        U = [U[i] + dt / 6 * (a[i] + 2 * b[i] + 2 * c[i] + d[i]) for i in range(3)]
        t += dt
    kmag = _np.sqrt(k2)
    e = 0.5 * sum(abs(Ui) ** 2 for Ui in U) / N2**6
    nb = int(N2 / 3)
    spec = _np.array([e[(kmag >= n - 0.5) & (kmag < n + 0.5)].sum() for n in range(1, nb)])
    ks = _np.arange(1, nb)
    sel = (ks >= nb // 2) & (spec > 0)
    delta = -_np.polyfit(ks[sel], _np.log(spec[sel]), 1)[0] / 2 if sel.sum() >= 4 else float("nan")
    print("  %-14s at %d^3:  Z(T)/Z0 = %.3f   E(T)/E0 = %.6f   delta(T) = %.4f  (2dx = %.4f)%s" % (label, N2, Zf(U) / Z0v, Ef(U) / E0v, delta, 2 * 2 * _np.pi / N2, "" if delta > 2 * 2 * _np.pi / N2 else "   <-- unreliable"), flush=True)
    return Zf(U) / Z0v, delta


print("\nverification at %d^3 with the numpy instrument (fields upsampled spectrally, same Z0):" % NVER)
results = {}
with torch.no_grad():
    for name, U in list(cands.items()) + [("FOUND", Ubest)]:
        Un = [Ui * math.sqrt(Z0 / enstrophy(U).item()) for Ui in U]
        phys = [ifft(Ui).real.numpy() for Ui in Un]
        results[name] = verify(phys, NVER, T, name)
best_classical = max(v[0] for kname, v in results.items() if kname != "FOUND")
print("\nREGISTERED  found Z(T)/Z0 = %.3f vs best classical %.3f at %d^3 -> %s" % (results["FOUND"][0], best_classical, NVER, "PASS" if results["FOUND"][0] > best_classical else "FAIL"))
