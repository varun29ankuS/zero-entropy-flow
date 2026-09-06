"""Searching for the missing functional, with an adversary in the loop.

Hypothesis H (THEORY.md): a functional M(u) with (a) M >= enstrophy Z, (b) dM/dt <= 0 along every Navier-Stokes
trajectory, (c) finite for smooth data, would bound Z for all time and settle 3-D regularity. Theorem 4 (Tao) says M
cannot be built from energy and scaling alone; it must use what the averaged equation lacks: LOCAL structure in
physical space, the geometry of stretching. So the candidate here is local by construction:

    M_theta(u) = Z exp(Phi_theta(u)),   Phi_theta = sum_x w(x) g_theta(f(x)),   w = |w|^2 / sum |w|^2,   0 <= g <= B

with f(x) dimensionless pointwise features of the vorticity and strain fields (stretching rate along the vorticity
xi.S xi / sqrt(Z), strain eigenvalues / sqrt(Z), direction roughness |grad xi| / k_rms, local enstrophy density,
local energy density, alignment of xi with the strain eigenvectors). M >= Z holds by construction, M <= exp(B) Z,
and no grid quantity enters, so the trivial truncated-system bound Z <= k_max^2 E is not available to the learner.

dM/dt is computed EXACTLY as the Lie derivative <dM/du, F(u)> by autograd through M, with F the same skew-symmetric
transport plus exact viscous term used everywhere in this repository. The loss is the relative violation
relu(dM/dt) / M on states sampled along trajectories. After each training round an ADVERSARY (the differentiable
solver, as in adversarial_ic.py) searches low-k initial data of fixed enstrophy for the trajectory with the largest
violation; its worst offenders join the training set. Registered verdict: PASS only if, after the last round, the
adversary's best violation is below TOL and held-out classical flows (Taylor-Green, Kida-Pelz, perturbed ABC) show
none. The expected outcome is FAIL; the content of the failure - which features the candidate leans on, and which
flows break it - is what is reported.
usage: N=24 NU=2e-3 T=0.6 ROUNDS=3 python lyapunov_search.py"""
import os, time, math, numpy as np, torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
N = int(os.environ.get("N", 24))
NU = float(os.environ.get("NU", 2e-3))
T = float(os.environ.get("T", 0.6))
ROUNDS = int(os.environ.get("ROUNDS", 3))
TRAIN = int(os.environ.get("TRAIN", 300))
ADV_ITERS = int(os.environ.get("ADV_ITERS", 15))
B = float(os.environ.get("B", 4.0))
TOL = float(os.environ.get("TOL", 1e-3))
KMAX_IC = 3
fft, ifft = torch.fft.fftn, torch.fft.ifftn
k1 = torch.fft.fftfreq(N, d=1.0 / N) * 1.0
KX, KY, KZ = torch.meshgrid(k1, k1, k1, indexing="ij")
K = [KX, KY, KZ]
K2 = KX**2 + KY**2 + KZ**2
K2S = K2.clone()
K2S[0, 0, 0] = 1.0
DEAL = ((KX.abs() < N / 3) & (KY.abs() < N / 3) & (KZ.abs() < N / 3)).to(torch.float64)
LOWK = ((K2 <= KMAX_IC**2) & (K2 > 0)).to(torch.float64)
x = torch.linspace(0, 2 * math.pi, N + 1)[:-1]
X, Y, Z_ = torch.meshgrid(x, x, x, indexing="ij")


def project(F):
    kd = sum(K[i] * F[i] for i in range(3)) / K2S
    return [F[i] - K[i] * kd for i in range(3)]


def rhs(U):
    """spectral tendency: skew-symmetric transport + exact viscous term (used both for the Lie derivative and the rollout)"""
    Ud = [Ui * DEAL for Ui in U]
    u = [ifft(Ui).real for Ui in Ud]
    out = []
    for i in range(3):
        adv = sum(u[j] * ifft(1j * K[j] * Ud[i]).real for j in range(3))
        div = sum(ifft(1j * K[j] * fft(u[j] * u[i])).real for j in range(3))
        out.append(-0.5 * fft(adv + div) * DEAL)
    out = project(out)
    return [out[i] - NU * K2 * Ud[i] for i in range(3)]


def step(U, dt):
    f, f2 = torch.exp(-NU * K2 * dt), torch.exp(-NU * K2 * dt / 2)
    tr = lambda V: project([(-0.5 * fft(sum(ifft(V[j] * DEAL).real * ifft(1j * K[j] * V[i] * DEAL).real for j in range(3)) + sum(ifft(1j * K[j] * fft(ifft(V[j] * DEAL).real * ifft(V[i] * DEAL).real)).real for j in range(3))) * DEAL) for i in range(3)])
    a = tr(U)
    b = tr([f2 * (U[i] + dt / 2 * a[i]) for i in range(3)])
    c = tr([f2 * U[i] + dt / 2 * b[i] for i in range(3)])
    d = tr([f * U[i] + dt * f2 * c[i] for i in range(3)])
    return [f * U[i] + dt / 6 * (f * a[i] + 2 * f2 * b[i] + 2 * f2 * c[i] + d[i]) for i in range(3)]


def rollout(U, T, every=None):
    t, out = 0.0, []
    nxt = 0.0
    while t < T - 1e-12:
        if every is not None and t >= nxt - 1e-12:
            out.append((t, [Ui.detach().clone() for Ui in U]))
            nxt += every
        umax = max(ifft(Ui).real.abs().max() for Ui in U)
        dt = min(2.0 / N, 0.5 * (2 * math.pi / N) / max(float(umax), 1e-9), T - t)
        U = step(U, dt)
        t += dt
    if every is not None:
        out.append((t, [Ui.detach().clone() for Ui in U]))
        return out
    return U


def vort(U):
    return [ifft(1j * KY * U[2] - 1j * KZ * U[1]).real, ifft(1j * KZ * U[0] - 1j * KX * U[2]).real, ifft(1j * KX * U[1] - 1j * KY * U[0]).real]


def enstrophy(U):
    return 0.5 * sum((w**2).mean() for w in vort(U))


def energy(U):
    return 0.5 * sum((ifft(Ui).real ** 2).mean() for Ui in U)


FEATURES = ["stretch xi.S.xi / sqrt Z", "|S|^2 / Z", "det S / Z^1.5", "xi.S^2.xi / |S|^2", "|grad xi| / k_rms",
            "|w|^2 / 2Z", "|u|^2 / 2E", "(xi.S.xi)^2 / |S|^2"]


def features(U):
    """pointwise dimensionless features [8, N, N, N] from the velocity field (spectral)"""
    Ud = [Ui * DEAL for Ui in U]
    u = [ifft(Ui).real for Ui in Ud]
    w = vort(Ud)
    Z = 0.5 * sum((wi**2).mean() for wi in w)
    E = 0.5 * sum((ui**2).mean() for ui in u)
    sZ = torch.sqrt(Z) + 1e-12
    G = [[ifft(1j * K[i] * Ud[j]).real for j in range(3)] for i in range(3)]
    S = torch.stack([torch.stack([0.5 * (G[i][j] + G[j][i]) for j in range(3)], -1) for i in range(3)], -1)   # [N,N,N,3,3]
    wmag = torch.sqrt(sum(wi**2 for wi in w) + 1e-30)
    xi = torch.stack([wi / (wmag + 1e-3 * sZ) for wi in w], -1)                                                # [N,N,N,3]
    alpha = torch.einsum("...i,...ij,...j->...", xi, S, xi)
    S2 = torch.einsum("...ij,...jk->...ik", S, S)
    s2 = torch.einsum("...ii->...", S2) + 1e-12 * Z                                                            # |S|^2 = tr S^2
    detS = torch.linalg.det(S)
    xiS2xi = torch.einsum("...i,...ij,...j->...", xi, S2, xi)
    # direction roughness: |grad xi| via spectral derivatives of the direction field
    gx = torch.zeros_like(wmag)
    for c in range(3):
        xh = fft(xi[..., c]) * DEAL
        for d in range(3):
            gx = gx + ifft(1j * K[d] * xh).real ** 2
    krms = torch.sqrt(Z / (E + 1e-30))
    f = torch.stack([alpha / sZ, s2 / Z, detS / (Z * sZ), xiS2xi / s2, torch.sqrt(gx + 1e-30) / krms,
                     wmag**2 / (2 * Z), sum(ui**2 for ui in u) / (2 * E), alpha**2 / s2], 0)
    return f, wmag**2 / (wmag**2).sum(), Z


class G(torch.nn.Module):
    def __init__(self, nf=8, h=32):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(nf, h), torch.nn.Tanh(), torch.nn.Linear(h, h), torch.nn.Tanh(), torch.nn.Linear(h, 1))

    def forward(self, f):
        return B * torch.sigmoid(self.net(f.permute(1, 2, 3, 0)).squeeze(-1))


g = G()
opt = torch.optim.Adam(g.parameters(), lr=3e-3)


def M_of(U):
    f, wgt, Z = features(U)
    Phi = (wgt * g(f)).sum()
    return Z * torch.exp(Phi), Phi, Z


def violation(U):
    """relative Lie derivative (dM/dt)/M at the state U, exact by autograd; positive = M increasing = violation"""
    Ur = [ifft(Ui).real.detach().clone().requires_grad_(True) for Ui in U]
    Uh = [fft(ui) for ui in Ur]
    M, Phi, Z = M_of(Uh)
    grads = torch.autograd.grad(M, Ur, create_graph=True)
    F = [ifft(Fi).real for Fi in rhs([Ui.detach() for Ui in Uh])]
    dMdt = sum((grads[i] * F[i]).sum() for i in range(3))
    return dMdt / M, M, Phi, Z


# ------------------------------------------------ flows ------------------------------------------------
def classical():
    tg = [torch.sin(X) * torch.cos(Y) * torch.cos(Z_), -torch.cos(X) * torch.sin(Y) * torch.cos(Z_), torch.zeros_like(X)]
    kp = [torch.sin(X) * (torch.cos(3 * Y) * torch.cos(Z_) - torch.cos(Y) * torch.cos(3 * Z_)),
          torch.sin(Y) * (torch.cos(3 * Z_) * torch.cos(X) - torch.cos(Z_) * torch.cos(3 * X)),
          torch.sin(Z_) * (torch.cos(3 * X) * torch.cos(Y) - torch.cos(X) * torch.cos(3 * Y))]
    gen = torch.Generator().manual_seed(0)
    abc = [torch.sin(Z_) + torch.cos(Y), torch.sin(X) + torch.cos(Z_), torch.sin(Y) + torch.cos(X)]
    pert = project([fft(torch.randn(X.shape, generator=gen)) * ((K2 >= 1) & (K2 <= 9)).to(torch.float64) * DEAL for _ in range(3)])
    abcU = [fft(a) for a in abc]
    sc = 0.1 * math.sqrt(sum((ifft(Ui).real ** 2).mean() for Ui in abcU) / sum((ifft(p).real ** 2).mean() for p in pert))
    return {"taylor-green": [fft(a) for a in tg], "kida-pelz": [fft(a) for a in kp], "abc+10%": [abcU[i] + sc * pert[i] for i in range(3)]}


Z0 = 0.375


def normalise(U):
    U = project([Ui * DEAL for Ui in U])
    return [Ui * torch.sqrt(Z0 / enstrophy(U)) for Ui in U]


def field_from_params(P):
    return normalise([fft(Pi) * LOWK for Pi in P])


def random_ic(seed):
    gen = torch.Generator().manual_seed(seed)
    return field_from_params([torch.randn(N, N, N, generator=gen) for _ in range(3)])


EVERY = T / 8
print("lyapunov search: N=%d^3  nu=%g  T=%.2f  Z0=%.3f  B=%.1f  rounds=%d  train steps/round=%d  adversary iters=%d" % (N, NU, T, Z0, B, ROUNDS, TRAIN, ADV_ITERS), flush=True)
cands = {k: normalise(v) for k, v in classical().items()}
train_ics = {"random-%d" % s: random_ic(s) for s in range(4)}
train_ics["taylor-green"] = cands["taylor-green"]
train_ics["abc+10%"] = cands["abc+10%"]
heldout = {"kida-pelz": cands["kida-pelz"], "random-9": random_ic(9)}
t0 = time.time()


def trajectories(ics):
    out = []
    with torch.no_grad():
        for name, U in ics.items():
            for (t, S) in rollout(U, T, every=EVERY):
                out.append((name, t, S))
    return out


data = trajectories(train_ics)
print("training states: %d from %d flows   (%.0fs)" % (len(data), len(train_ics), time.time() - t0), flush=True)


def report(states, label):
    worst = (-1e9, None, None)
    tot = 0.0
    for name, t, S in states:
        v, M, Phi, Z = violation(S)
        v = v.item()
        tot += max(v, 0.0)
        if v > worst[0]:
            worst = (v, name, t)
    print("  %-22s worst relative dM/dt = %+.4e  (%s at t=%.2f)   mean positive part %.2e" % (label, worst[0], worst[1], worst[2], tot / len(states)), flush=True)
    return worst[0]


for rnd in range(ROUNDS):
    print("\n== round %d ==" % rnd, flush=True)
    for it in range(TRAIN):
        batch = [data[i] for i in torch.randperm(len(data))[:6].tolist()]
        loss = 0.0
        for name, t, S in batch:
            v, M, Phi, Z = violation(S)
            loss = loss + torch.relu(v) ** 2
        loss = loss / len(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 100 == 0 or it == TRAIN - 1:
            print("  train %3d   mean relu(dM/dt / M)^2 = %.3e   (%.0fs)" % (it, loss.item(), time.time() - t0), flush=True)
    report(data, "training states")
    report(trajectories(heldout), "held-out flows")
    # ---- adversary: initial data whose trajectory maximises the worst relative violation of the CURRENT M ----
    P = [torch.randn(N, N, N, requires_grad=True) for _ in range(3)]
    aopt = torch.optim.Adam(P, lr=0.05)
    best = (-1e9, None)
    for it in range(ADV_ITERS):
        U = field_from_params(P)
        worst = None
        t = 0.0
        while t < T - 1e-12:
            umax = max(ifft(Ui).real.abs().max() for Ui in U)
            dt = min(2.0 / N, 0.5 * (2 * math.pi / N) / max(float(umax), 1e-9), T - t)
            U = step(U, dt)
            t += dt
            if int(t / EVERY) != int((t - dt) / EVERY):
                Ur = [ifft(Ui).real for Ui in U]
                Uh = [fft(ui) for ui in Ur]
                M, Phi, Z = M_of(Uh)
                grads = torch.autograd.grad(M, Ur, create_graph=True)
                F = [ifft(Fi).real for Fi in rhs(Uh)]
                v = sum((grads[i] * F[i]).sum() for i in range(3)) / M
                worst = v if worst is None else torch.maximum(worst, v)
        aopt.zero_grad()
        (-worst).backward()
        aopt.step()
        if worst.item() > best[0]:
            best = (worst.item(), [p.detach().clone() for p in P])
        if it % 5 == 0 or it == ADV_ITERS - 1:
            print("  adversary %2d   worst relative dM/dt along its trajectory = %+.4e   (%.0fs)" % (it, worst.item(), time.time() - t0), flush=True)
    adv_ic = field_from_params(best[1])
    with torch.no_grad():
        for (t, S) in rollout(adv_ic, T, every=EVERY):
            data.append(("adversary-%d" % rnd, t, S))
    print("  adversary round %d: best violation %+.4e; its %d states added to training" % (rnd, best[0], int(T / EVERY) + 1), flush=True)

print("\n== final ==")
w_train = report(data, "all training states")
w_held = report(trajectories(heldout), "held-out flows")
w_adv = best[0]
# what does the candidate lean on? sensitivity of Phi to each feature over the held-out states
sens = torch.zeros(8)
cnt = 0
for name, t, S in trajectories(heldout):
    f, wgt, Z = features(S)
    f = f.detach().requires_grad_(True)
    Phi = (wgt * g(f)).sum()
    gr = torch.autograd.grad(Phi, f)[0]
    sens += (gr * f.detach()).abs().sum(dim=(1, 2, 3))
    cnt += 1
sens = sens / cnt
print("feature sensitivity of the learned Phi (|dPhi/df . f| summed over the field, held-out states):")
for name, s in sorted(zip(FEATURES, sens.tolist()), key=lambda z: -z[1]):
    print("   %-28s %.3e" % (name, s))
print("\nREGISTERED  last adversary violation %+.3e, held-out worst %+.3e, tolerance %.0e -> %s" % (
    w_adv, w_held, TOL, "PASS: no violation found (a candidate, not a theorem)" if max(w_adv, w_held) < TOL else "FAIL: the adversary (or a held-out flow) still finds states where the candidate M increases"))
torch.save(g.state_dict(), "results/lyapunov_g.pt")
