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
import torch.utils.checkpoint

torch.set_default_dtype(torch.float64)
N = int(os.environ.get("N", 32))
T = float(os.environ.get("T", 1.0))
ITERS = int(os.environ.get("ITERS", 40))
KMAX_IC = int(os.environ.get("KMAX_IC", 4))
NVER = int(os.environ.get("NVER", 64))
LR = float(os.environ.get("LR", 0.05))
OBJ = os.environ.get("OBJ", "enstrophy")          # enstrophy | helicity | jacobi
HEL = float(os.environ.get("HEL", 0.0))          # helicity target as a fraction of the max possible at this Z0 (OBJ=helicity)
HELW = float(os.environ.get("HELW", 50.0))       # penalty weight
HELMODE = os.environ.get("HELMODE", "penalty")   # penalty | project  (project: Newton-project P onto H/Hmax = HEL after every step, a hard constraint)
DMIN = float(os.environ.get("DMIN", 0.0))        # if > 0: penalise analyticity-strip width delta(T) below DMIN on the search grid (stay resolved)
DW = float(os.environ.get("DW", 20.0))
CKPT = int(os.environ.get("CKPT", 1))            # gradient checkpointing per step (memory ~ N^3 x steps instead of x stages x steps)
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


def helicity(U):
    u = [ifft(Ui).real for Ui in U]
    w = vort(U)
    return sum((u[i] * w[i]).mean() for i in range(3))


KMAG_T = torch.sqrt(K2)
NB_T = int(N / 3)


def delta_torch(U):
    """analyticity-strip width on the search grid: fit log E(k) = c - 2 delta k over the upper half of retained modes"""
    e = 0.5 * sum(Ui.abs() ** 2 for Ui in U) / N**6
    ks = torch.arange(NB_T // 2, NB_T, dtype=torch.float64)
    spec = torch.stack([e[(KMAG_T >= n - 0.5) & (KMAG_T < n + 0.5)].sum() for n in ks])
    y = torch.log(spec + 1e-30)
    xm, ym = ks.mean(), y.mean()
    slope = ((ks - xm) * (y - ym)).sum() / ((ks - xm) ** 2).sum()
    return -slope / 2.0


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
        if CKPT and any(Ui.requires_grad for Ui in U):
            # gradient checkpointing: keep only the state at each step, recompute the RK4 stages in the backward pass
            U = list(torch.utils.checkpoint.checkpoint(lambda *Us: tuple(step(list(Us), dt)), *U, use_reentrant=False))
        else:
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
Z0 = float(os.environ.get("Z0", 0)) or enstrophy(cands["taylor-green"]).item()   # common initial enstrophy (default: Taylor-Green's, the mild one)
print("search grid %d^3, T = %.2f, low-k initial data |k| <= %d, common Z0 = %.4f, objective %s%s" % (N, T, KMAX_IC, Z0, OBJ, (" (helicity target %.2f of max)" % HEL) if OBJ == "helicity" else ""))
with torch.no_grad():
    for name, U in cands.items():
        Un = [Ui * math.sqrt(Z0 / enstrophy(U).item()) for Ui in U]
        UT = rollout(Un, T)
        print("  %-14s E0 %.4f   H/Hmax %+.3f   Z(T)/Z0 = %.3f" % (name, energy(Un).item(), (helicity(Un) / (2 * torch.sqrt(energy(Un) * Z0))).item(), enstrophy(UT).item() / Z0), flush=True)

# --------------------------------------------- gradient ascent on the initial field ---------------------------------------------
torch.manual_seed(1)
P = [torch.randn(N, N, N, requires_grad=True) for _ in range(3)]


def hel_of(P):
    with torch.no_grad():
        Uc = field_from_params(P, Z0)
        return (helicity(Uc) / (2 * torch.sqrt(energy(Uc) * Z0))).item()


def hel_project(P, iters=40):
    """hard constraint: damped Newton steps on c(P) = H/Hmax - HEL using the exact gradient of the constraint alone
    (no rollout). Steps are capped at 0.1 in c per iteration so the projection cannot overshoot on the curved level set."""
    for _ in range(iters):
        Uc = field_from_params(P, Z0)
        c = helicity(Uc) / (2 * torch.sqrt(energy(Uc) * Z0)) - HEL
        if abs(c.item()) < 1e-6:
            return True
        gr = torch.autograd.grad(c, P)
        gg = sum((gi ** 2).sum() for gi in gr)
        step_c = max(min(c.item(), 0.1), -0.1)
        with torch.no_grad():
            saved = [Pi.clone() for Pi in P]
            for frac in (1.0, 0.5, 0.25, 0.125, 0.0625):     # backtracking: accept a step only if |c| decreases
                for Pi, Si, gi in zip(P, saved, gr):
                    Pi.copy_(Si - frac * step_c * gi / gg)
                if abs(hel_of(P) - HEL) < abs(c.item()):
                    break
            else:
                for Pi, Si in zip(P, saved):
                    Pi.copy_(Si)
                return abs(c.item()) < 1e-3
    return abs(c.item()) < 1e-3


if OBJ == "helicity" and HELMODE == "project":
    # initial point: relative helicity is continuous along the segment from the random field (H ~ 0) to the Beltrami
    # ABC field (H/Hmax = 1 exactly), so bisect on the blend to the target, then let the Newton projection polish it
    with torch.no_grad():
        abc0 = [torch.sin(Z_) + torch.cos(Y), torch.sin(X) + torch.cos(Z_), torch.sin(Y) + torch.cos(X)]
        rnd = [Pi.clone() for Pi in P]
        sc = math.sqrt(sum((a ** 2).mean() for a in abc0) / sum((r ** 2).mean() for r in rnd))
        rnd = [r * sc for r in rnd]
        lo, hi = 0.0, 1.0
        for _ in range(60):
            al = 0.5 * (lo + hi)
            for Pi, ai, ri in zip(P, abc0, rnd):
                Pi.copy_(al * ai + (1 - al) * ri)
            if hel_of(P) < HEL:
                lo = al
            else:
                hi = al
    print("bisection on the random-to-Beltrami segment reached H/Hmax = %.4f" % hel_of(P), flush=True)
    ok0 = hel_project(P, iters=200)
    print("initial projection onto H/Hmax = %.2f: %s (reached %.4f)" % (HEL, "ok" if ok0 else "NOT REACHED", hel_of(P)), flush=True)
opt = torch.optim.Adam(P, lr=LR)
best = (0.0, None)
t0 = time.time()
for it in range(ITERS):
    U0 = field_from_params(P, Z0)
    UT = rollout(U0, T)
    if OBJ == "jacobi":
        # geodesic spreading: maximise the growth of a small perturbation of Taylor-Green along the flow (Arnold's curvature)
        base = [Ui * math.sqrt(Z0 / enstrophy(cands["taylor-green"]).item()) for Ui in cands["taylor-green"]]
        eps = 1e-3
        Up = [base[i] + eps * U0[i] for i in range(3)]
        UTb = rollout(base, T)
        UTp = rollout(Up, T)
        growth = torch.sqrt(sum(((ifft(UTp[i] - UTb[i]).real) ** 2).mean() for i in range(3))) / (eps * torch.sqrt(sum((ifft(U0[i]).real ** 2).mean() for i in range(3))))
        loss = -torch.log(growth)
    else:
        growth = enstrophy(UT) / Z0
        loss = -torch.log(growth)
        if DMIN > 0:
            d = delta_torch(UT)
            loss = loss + DW * torch.relu(DMIN - d) ** 2
        if OBJ == "helicity":
            H = helicity(U0)
            Hmax = 2 * torch.sqrt(energy(U0) * Z0)      # |H| <= |u| |w| = 2 sqrt(E Z) (Cauchy-Schwarz), equality for a single-shell Beltrami field (ABC: exactly 1)
            loss = loss + HELW * (H / Hmax - HEL) ** 2
    opt.zero_grad()
    loss.backward()
    opt.step()
    if OBJ == "helicity" and HELMODE == "project":
        hel_project(P)
    g = growth.item()
    if g > best[0] and (OBJ != "helicity" or HELMODE != "project" or abs(hel_of(P) - HEL) < 1e-3):
        best = (g, [p.detach().clone() for p in P])
    if it % 5 == 0 or it == ITERS - 1:
        print("  iter %3d   objective = %.3f   E0 = %.4f   H/Hmax = %+.3f   delta(T) on search grid = %.3f   (%.0fs)" % (it, g, energy(U0).item(), (helicity(U0) / (2 * torch.sqrt(energy(U0) * Z0))).item(), delta_torch(UT).item() if OBJ != "jacobi" else float("nan"), time.time() - t0), flush=True)
if best[1] is None:
    best = (growth.item(), [p.detach().clone() for p in P])
print("best amplification on the search grid: %.3f" % best[0])
with torch.no_grad():
    Ubest = field_from_params(best[1], Z0)
    os.makedirs("results/found", exist_ok=True)
    np.savez_compressed("results/found/%s.npz" % os.environ.get("TAG", "adversarial_ic_%d" % N), u=np.stack([ifft(Ui).real.numpy() for Ui in Ubest]).astype(np.float32), Z0=Z0, T=T)

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


NVERS = [int(v) for v in os.environ.get("NVERS", str(NVER)).split(",")]     # e.g. NVERS=128,192: every candidate at the first, then only FOUND and Taylor-Green at the rest
for iv, NV in enumerate(NVERS):
    print("\nverification at %d^3 with the numpy instrument (fields upsampled spectrally, same Z0):" % NV, flush=True)
    results = {}
    with torch.no_grad():
        for name, U in list(cands.items()) + [("FOUND", Ubest)]:
            if iv > 0 and name not in ("taylor-green", "FOUND"):
                continue
            Un = [Ui * math.sqrt(Z0 / enstrophy(U).item()) for Ui in U]
            phys = [ifft(Ui).real.numpy() for Ui in Un]
            results[name] = verify(phys, NV, T, name)
    ok = {kname: v[1] > 2 * 2 * math.pi / NV for kname, v in results.items()}
    classical_ok = {kname: v[0] for kname, v in results.items() if kname != "FOUND" and ok[kname]}
    best_classical = max(classical_ok.values()) if classical_ok else float("nan")
    reliable = ok["FOUND"] and bool(classical_ok)
    print("\nREGISTERED  found Z(T)/Z0 = %.3f (delta %s 2dx) vs best RELIABLE classical %.3f at %d^3 -> %s" % (
        results["FOUND"][0], ">" if ok["FOUND"] else "<", best_classical, NV,
        "PASS" if reliable and results["FOUND"][0] > best_classical else ("FAIL: outside the reliable window" if not reliable else "FAIL")), flush=True)
