"""Liouville's theorem for the truncated system (Lee 1952; Kraichnan 1973): the Galerkin-truncated Euler equations are a
divergence-free vector field on the finite-dimensional phase space of retained Fourier modes, so phase-space volume, and
with it the Gibbs entropy of any ensemble of solutions, is exactly conserved. This is the dynamical-system form of the
zero-entropy-production statement: the transport operator does not just conserve one energy, it conserves the
information in the initial ensemble. Viscosity contracts volume at the fixed rate -2 nu sum_k k^2 (per component).

Measured with autograd: div F = tr(dF/dU) exactly (one reverse pass per coordinate) on real-space velocity vectors, for the skew-symmetric transport used everywhere
in this repository and, for contrast, the plain advective form and the divergence form separately. Then the
one-step map: log|det J_step| = dt * div F + O(dt^2) for RK4, which is what the numerical entropy production sigma_num
means at the level of ensembles.
usage: DIM=2|3 N=16 python liouville.py"""
import os, math, time, torch

torch.set_default_dtype(torch.float64)
DIM = int(os.environ.get("DIM", 2))
N = int(os.environ.get("N", 16))
NU = float(os.environ.get("NU", 0.0))
PROBES = int(os.environ.get("PROBES", 64))
fft = torch.fft.fftn
ifft = torch.fft.ifftn
k1 = torch.fft.fftfreq(N, d=1.0 / N) * 1.0
K = list(torch.meshgrid(*([k1] * DIM), indexing="ij"))
K2 = sum(Ki**2 for Ki in K)
K2S = K2.clone()
K2S[(0,) * DIM] = 1.0
DEAL = torch.ones_like(K2, dtype=torch.bool)
for Ki in K:
    DEAL &= Ki.abs() < N / 3
DEAL = DEAL.to(torch.float64)


def project(F):
    kd = sum(K[i] * F[i] for i in range(DIM)) / K2S
    return [F[i] - K[i] * kd for i in range(DIM)]


# a member of Tao's averaged class: B~(u,u) = M B(Mu, Mu) with M a symmetric Fourier multiplier (random, radial here).
# It keeps <B~(u,u), u> = <B(Mu,Mu), Mu> = 0, the scaling, and the convolution structure; it is not Navier-Stokes.
_g = torch.Generator().manual_seed(1)
_rad = torch.sqrt(K2).round().long()
_prof = 0.3 + torch.rand(int(_rad.max().item()) + 1, generator=_g)
TAO_M = _prof[_rad]
TAO_M[(0,) * DIM] = 0.0


def transport(u, form):
    """u: DIM real-space fields -> DIM real-space tendencies (nu = 0 part)"""
    Ud = project([fft(ui) * DEAL for ui in u])      # phase space = retained solenoidal modes; F = F o P
    if form == "tao-class":
        Ud = [Ui * TAO_M for Ui in Ud]
    ur = [ifft(Ui).real for Ui in Ud]
    out = []
    for i in range(DIM):
        adv = sum(ur[j] * ifft(1j * K[j] * Ud[i]).real for j in range(DIM))
        div = sum(ifft(1j * K[j] * fft(ur[j] * ur[i])).real for j in range(DIM))
        if form in ("skew", "tao-class"):
            out.append(-0.5 * fft(adv + div) * DEAL * (TAO_M if form == "tao-class" else 1.0))
        elif form == "advective":
            out.append(-fft(adv) * DEAL)
        else:
            out.append(-fft(div) * DEAL)
    Fh = project(out)
    if NU > 0:
        Fh = [Fh[i] - NU * K2 * Ud[i] for i in range(DIM)]
    return [ifft(Fi).real for Fi in Fh]


def divergence(u, form, probes):
    """exact trace: one reverse-mode pass per coordinate of the real-space representation (DIM N^DIM of them).
    F(u) = F(P u) for the dealiasing/Leray projector P, so tr(dF/du) is the trace over the retained solenoidal
    subspace the dynamics actually lives on. Also returns the largest |J_ii| for scale."""
    uu = [ui.clone().requires_grad_(True) for ui in u]
    F = transport(uu, form)
    Fflat = torch.cat([Fi.reshape(-1) for Fi in F])
    tr, jmax = 0.0, 0.0
    n = Fflat.numel()
    for i in range(n):
        gr = torch.autograd.grad(Fflat[i], uu, retain_graph=True)
        gflat = torch.cat([gi.reshape(-1) for gi in gr])
        tr += gflat[i].item()
        jmax = max(jmax, gflat.abs().max().item())
    return tr, jmax


x = torch.linspace(0, 2 * math.pi, N + 1)[:-1]
Xs = torch.meshgrid(*([x] * DIM), indexing="ij")
if DIM == 2:
    u = [torch.sin(Xs[0]) * torch.cos(Xs[1]), -torch.cos(Xs[0]) * torch.sin(Xs[1])]
else:
    u = [torch.sin(Xs[0]) * torch.cos(Xs[1]) * torch.cos(Xs[2]), -torch.cos(Xs[0]) * torch.sin(Xs[1]) * torch.cos(Xs[2]), torch.zeros_like(Xs[0])]
g = torch.Generator().manual_seed(0)
noise = project([fft(torch.randn(Xs[0].shape, generator=g)) * ((K2 >= 1) & (K2 <= 16)).to(torch.float64) * DEAL for _ in range(DIM)])
u = [u[i] + 0.3 * ifft(noise[i]).real for i in range(DIM)]   # a generic (non-symmetric) point of phase space
scale = sum((transport(u, "skew")[i] ** 2).sum() for i in range(DIM)).sqrt().item()   # |F| for scale
ndof = int(DEAL.sum().item()) * (DIM - 1)
print("DIM=%d  N=%d  nu=%g  retained solenoidal dof ~ %d  |F| = %.3f  (exact trace over %d coordinates)" % (DIM, N, NU, ndof, scale, DIM * N**DIM))
print("expected viscous contraction  -nu (DIM-1) sum_k k^2 = %.4f" % (-NU * (DIM - 1) * (K2 * DEAL).sum().item()))
t0 = time.time()
for form in ("skew", "advective", "divergence", "tao-class"):
    m, jmax = divergence(u, form, PROBES)
    print("  %-11s  div F = tr(dF/dU) = %+.3e    (largest Jacobian entry %.2f; relative %.1e)   (%.0fs)" % (form, m, jmax, abs(m) / (jmax * ndof), time.time() - t0), flush=True)
