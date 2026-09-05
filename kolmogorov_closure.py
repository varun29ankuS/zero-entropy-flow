"""Learned coarse simulators of 2-D Kolmogorov flow: does freezing the conservative transport and learning only the
dissipation give stable long rollouts where a fully learned model drifts?

Truth : 2-D Navier-Stokes in vorticity form, velocity forcing (0.08 sin 4y, 0), nu = 5e-3 (Re ~ 1250), 128^2 pseudo-spectral (skew-symmetric
        transport, exact viscous factor, RK4, 2/3 dealiasing). Long spin-up, then SNAP snapshots every 20 DNS steps,
        each sharp-filtered to 32^2. Coarse time step DT = 20 * dt_dns.
Arms  (all at 32^2, all trained on one-step + 4-step rollout loss against the filtered truth, same data, same steps):
  learned     a small CNN predicts the next coarse vorticity (residual). Nothing conserved.   [the field's default]
  frozen+gate exact conservative transport at 32^2 (same scheme as the truth, coarser) + a learned eddy viscosity
              nu_t(x) = softplus(CNN(w)) >= 0 applied as div(nu_t grad w): the closure can ONLY dissipate.
  frozen      the same transport with no closure (under-resolved DNS), for reference.
Rollout: 2000 coarse steps from held-out initial conditions. Reported: energy ratio to truth over time (drift),
horizon where correlation with truth drops below 0.9, and enstrophy-spectrum error at the end.
REGISTERED: frozen+gate horizon >= 3x learned; frozen+gate energy within [0.5, 1.5] x truth for all 2000 steps while
learned leaves the band. CPU only."""

import math, time, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(8)
NF, NC = 128, 32
NU = 5e-3
KF = 4
AMP = 0.08
DT_DNS = 1e-3
SUB = 20
DT = DT_DNS * SUB  # velocity forcing (AMP sin 4y, 0): laminar speed AMP/(nu k^2) = 1, Re ~ 1250
SPIN = int(os.environ.get("SPIN", 30000))
SNAP = int(os.environ.get("SNAP", 1200))
STEPS = int(os.environ.get("STEPS", 3000))
ROLL = int(os.environ.get("ROLL", 2000))


def grids(N):
    k = np.fft.fftfreq(N, d=1.0 / N)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    k2 = kx * kx + ky * ky
    k2[0, 0] = 1.0
    deal = (np.abs(kx) < N / 3) & (np.abs(ky) < N / 3)
    return kx, ky, k2, deal


# ---------------------------------------------- truth (numpy) ----------------------------------------------
kx, ky, k2, deal = grids(NF)
x = np.linspace(0, 2 * np.pi, NF, endpoint=False)
X, Y = np.meshgrid(x, x, indexing="ij")
Fh = np.fft.fft2(-KF * AMP * np.cos(KF * Y)) * deal  # vorticity forcing = curl of (AMP sin 4y, 0) = -4 AMP cos 4y


def vel(wh, kx, ky, k2):
    psih = wh / k2
    psih = psih.copy()
    psih[0, 0] = 0
    return np.fft.ifft2(1j * ky * psih).real, -np.fft.ifft2(1j * kx * psih).real


def nl_np(wh, kx, ky, k2, deal):
    u, v = vel(wh, kx, ky, k2)
    wx = np.fft.ifft2(1j * kx * wh).real
    wy = np.fft.ifft2(1j * ky * wh).real
    adv = np.fft.fft2(u * wx + v * wy)
    div = 1j * kx * np.fft.fft2(u * np.fft.ifft2(wh).real) + 1j * ky * np.fft.fft2(v * np.fft.ifft2(wh).real)
    return -0.5 * (adv + div) * deal  # skew-symmetric transport of vorticity


def dns_step(wh, dt):
    fac = np.exp(-NU * k2 * dt)
    fac2 = np.exp(-NU * k2 * dt / 2)
    r = lambda w: nl_np(w, kx, ky, k2, deal) + Fh
    a = r(wh)
    b = r(fac2 * (wh + dt / 2 * a))
    c = r(fac2 * wh + dt / 2 * b)
    d = r(fac * wh + dt * fac2 * c)
    return fac * wh + dt / 6 * (fac * a + 2 * fac2 * b + 2 * fac2 * c + d)


def filt(wh):  # sharp spectral filter 128 -> 32 (keep |k| < 16)
    h = np.zeros((NC, NC), complex)
    n = NC // 2
    h[:n, :n] = wh[:n, :n]
    h[:n, -n:] = wh[:n, -n:]
    h[-n:, :n] = wh[-n:, :n]
    h[-n:, -n:] = wh[-n:, -n:]
    return np.fft.ifft2(h).real * (NC * NC) / (NF * NF)


t0 = time.time()
rng = np.random.default_rng(0)
wh = np.fft.fft2(rng.standard_normal((NF, NF)) * 0.5) * (k2 < 36) * deal
for i in range(SPIN):
    wh = dns_step(wh, DT_DNS)
print(
    "spin-up done (%.0fs), energy %.4f" % (time.time() - t0, 0.5 * np.mean(sum(c**2 for c in vel(wh, kx, ky, k2)))),
    flush=True,
)
snaps = []
for i in range(SNAP * SUB):
    wh = dns_step(wh, DT_DNS)
    if i % SUB == SUB - 1:
        snaps.append(filt(wh))
W = torch.tensor(np.stack(snaps), dtype=torch.float32)  # [SNAP, 32, 32] filtered truth, spacing DT
print("snapshots %d (%.0fs)" % (len(W), time.time() - t0), flush=True)
ntr = int(SNAP * 0.8)
Wtr, Wte = W[:ntr], W[ntr:]

# ------------------------------------------ coarse machinery (torch) ------------------------------------------
ckx, cky, ck2, cdeal = [torch.tensor(a) for a in grids(NC)]
ckx, cky, ck2 = ckx.float(), cky.float(), ck2.float()
cdeal = cdeal.float()
cx = torch.linspace(0, 2 * math.pi, NC + 1)[:-1]
CX, CY = torch.meshgrid(cx, cx, indexing="ij")
cFh = torch.fft.fft2(-KF * AMP * torch.cos(KF * CY)) * cdeal


def cvel(wh):
    psih = wh / ck2
    psih = psih.clone()
    psih[..., 0, 0] = 0
    return torch.fft.ifft2(1j * cky * psih).real, -torch.fft.ifft2(1j * ckx * psih).real


def cnl(wh):
    u, v = cvel(wh)
    w = torch.fft.ifft2(wh).real
    wx = torch.fft.ifft2(1j * ckx * wh).real
    wy = torch.fft.ifft2(1j * cky * wh).real
    adv = torch.fft.fft2(u * wx + v * wy)
    div = 1j * ckx * torch.fft.fft2(u * w) + 1j * cky * torch.fft.fft2(v * w)
    return -0.5 * (adv + div) * cdeal + cFh


def frozen_step(w, nut=None):
    """one coarse step of the exact conservative transport (+ exact viscosity); optional learned eddy viscosity nut(x)>=0
    applied as div(nu_t grad w), which is dissipative by construction"""
    wh = torch.fft.fft2(w)
    fac = torch.exp(-NU * ck2 * DT)
    fac2 = torch.exp(-NU * ck2 * DT / 2)

    def r(z):
        out = cnl(z)
        if nut is not None:
            zx = torch.fft.ifft2(1j * ckx * z).real
            zy = torch.fft.ifft2(1j * cky * z).real
            out = out + (1j * ckx * torch.fft.fft2(nut * zx) + 1j * cky * torch.fft.fft2(nut * zy)) * cdeal
        return out

    a = r(wh)
    b = r(fac2 * (wh + DT / 2 * a))
    c = r(fac2 * wh + DT / 2 * b)
    d = r(fac * wh + DT * fac2 * c)
    return torch.fft.ifft2(fac * wh + DT / 6 * (fac * a + 2 * fac2 * b + 2 * fac2 * c + d)).real


class CNN(nn.Module):
    def __init__(self, cout, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1, padding_mode="circular"),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, padding_mode="circular"),
            nn.GELU(),
            nn.Conv2d(width, cout, 3, padding=1, padding_mode="circular"),
        )

    def forward(self, w):
        return self.net(w[:, None])[:, 0]


class Learned(nn.Module):  # arm (a): fully learned residual step
    def __init__(self):
        super().__init__()
        self.f = CNN(1)

    def step(self, w):
        return w + 0.1 * self.f(w)


class FrozenGate(nn.Module):  # arm (b): exact transport + learned dissipation
    def __init__(self):
        super().__init__()
        self.f = CNN(1)

    def step(self, w):
        return frozen_step(w, nut=0.02 * F.softplus(self.f(w)))


class Frozen(nn.Module):  # arm (c): exact transport only
    def step(self, w):
        return frozen_step(w)


def energy(w):  # kinetic energy from vorticity
    u, v = cvel(torch.fft.fft2(w))
    return 0.5 * (u * u + v * v).mean(dim=(-2, -1))


def train(model, name):
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    g = torch.Generator().manual_seed(1)
    t0 = time.time()
    for step in range(STEPS):
        i = torch.randint(0, ntr - 5, (16,), generator=g)
        w = Wtr[i]
        loss = 0
        for h in range(4):  # 4-step rollout loss
            w = model.step(w)
            loss = loss + F.mse_loss(w, Wtr[i + h + 1])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 499:
            print(
                "  %-12s step %4d  loss %.5f  %.0fs" % (name, step + 1, loss.item() / 4, time.time() - t0), flush=True
            )
    return model.eval()


@torch.no_grad()
def rollout(model, w0, n):
    w = w0.clone()
    out = [w.clone()]
    for _ in range(n):
        w = model.step(w)
        out.append(w.clone())
    return torch.stack(out)


def spectrum(w):
    wh = torch.fft.fft2(w)
    kk = torch.sqrt(ckx**2 + cky**2)
    s = torch.zeros(NC // 2)
    for k in range(1, NC // 2):
        s[k] = (wh.abs() ** 2)[(kk >= k - 0.5) & (kk < k + 0.5)].mean()
    return s


arms = {"learned": train(Learned(), "learned"), "frozen+gate": train(FrozenGate(), "frozen+gate"), "frozen": Frozen()}
# held-out rollouts from 3 initial conditions; truth = the continuing filtered DNS for the available window, extended by fresh DNS
print("\nrollouts of %d coarse steps from held-out initial conditions" % ROLL, flush=True)
n_avail = len(Wte) - 1
results = {}
for name, m in arms.items():
    hor, edrift, spec_err = [], [], []
    for s0 in (0, n_avail // 3, 2 * n_avail // 3):
        w0 = Wte[s0]
        nmax = min(ROLL, n_avail - s0)
        R = rollout(m, w0[None], nmax)[:, 0]
        T = Wte[s0 : s0 + nmax + 1]
        corr = torch.tensor([F.cosine_similarity(R[t].flatten(), T[t].flatten(), dim=0) for t in range(nmax + 1)])
        below = torch.where(corr < 0.9)[0]
        hor.append(int(below[0]) if len(below) else nmax)
        eR = energy(R)
        eT = energy(T)
        edrift.append((eR / eT.mean()).min().item())
        edrift.append((eR / eT.mean()).max().item())
        spec_err.append(
            (torch.log(spectrum(R[-1]) + 1e-12) - torch.log(spectrum(T[-1]) + 1e-12))[1:].abs().mean().item()
        )
    results[name] = (np.mean(hor), min(edrift), max(edrift), np.mean(spec_err))
    print(
        "%-12s  horizon(corr>0.9) %6.0f steps | energy/truth min %.2f max %.2f | log-spectrum error %.2f"
        % (name, *results[name]),
        flush=True,
    )
a, b = results["learned"], results["frozen+gate"]
print(
    "\nREGISTERED  horizon ratio frozen+gate/learned = %.1f (>=3)  -> %s"
    % (b[0] / max(a[0], 1), "PASS" if b[0] >= 3 * a[0] else "FAIL")
)
print(
    "            frozen+gate energy band [%.2f, %.2f] within [0.5,1.5] -> %s | learned band [%.2f, %.2f] leaves it -> %s"
    % (
        b[1],
        b[2],
        "PASS" if 0.5 <= b[1] and b[2] <= 1.5 else "FAIL",
        a[1],
        a[2],
        "yes" if a[1] < 0.5 or a[2] > 1.5 else "no",
    )
)
