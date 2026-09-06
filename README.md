# zero-entropy-flow

**One pseudo-spectral solver for the incompressible Navier-Stokes family in 1-D, 2-D and 3-D, built so that its own
numerical dissipation is zero** - the transport is exactly energy-conserving on the grid, viscosity is an exact
per-mode contraction - and then used to look at the things numerical dissipation normally hides: a singularity
forming, vortex stretching, the terms of every balance law.

What is here, each with the command that reproduces it and the observation that would refute it (`CLAIMS.md`):

- **Verified against five closed-form solutions**, one per dimension and then some, to machine precision:
  inviscid Burgers approaching its blow-up exactly as 1/(1-t); the Cole-Hopf viscous shock; 2-D Taylor-Green;
  a *random* 2-D exact solution; the 3-D viscous ABC flow. Errors 1e-12 to 1e-15 except through the shock, where
  the error is the grid's and says so.
- **Every budget closes term by term** - energy, enstrophy, palinstrophy - including the unsigned production terms,
  at residuals 1e-6 to 1e-9. The unsigned term is measured driving the 1-D blow-up, vanishing identically in 2-D,
  and in 3-D **vortex stretching is measured directly at 64^3, 96^3, 128^3 and converges upward** with resolution;
  energy 1.000000 throughout. Helicity conserved to six digits under 3-D dynamics that amplify enstrophy seven-fold.
- **A learning result:** a coarse simulator with frozen exact transport plus a learned correction that may *move*
  energy between scales but cannot *create* it cuts the long-rollout spectrum error six-fold against a fully learned
  model that drifts to 1.9x the true energy (`kolmogorov_v2.py`, reproduced on two machines). A dissipative-only
  learned closure did nothing in 2-D; the missing physics there is backscatter.
- **The regularity criteria, measured with a clock that says when to stop believing them:** BKM, the L3 norm,
  the Constantin-Fefferman direction coherence at fixed physical scale, and the analyticity-strip width delta(t),
  on Taylor-Green and Kida-Pelz at 64/96/128^3. Numbers past the clock were withdrawn from this page, and it says so.
- **Feedback, not imitation:** a differentiable copy of the solver searches initial data for the fastest enstrophy
  growth (Lu-Doering; Ayala-Protas) and a verifier at higher resolution with the clock rejects what only wins on the
  search grid - it has rejected three so far. The same search with helicity held fixed orders attainable growth by
  helicity (Moffatt's direction), and along Arnold's geodesic it measures the Jacobi-field exponent by resolution.
- **Liouville, exactly:** the truncated inviscid system conserves phase volume (Lee 1952) - exact Jacobian trace
  1e-16 by autograd - and the viscous contraction is a state-independent constant, so the Gibbs entropy of any
  ensemble falls linearly. Zero entropy production per solution *and* per ensemble; what the flow does is stretch.
- **Theory** (`THEORY.md`): the elementary propositions proved; Tao's 2016 theorem that this structure alone cannot
  decide 3-D regularity; the conditional theorem for what would; the open hypothesis stated so it can be attacked.
- **Not claimed:** anything about the regularity of 3-D Navier-Stokes.

Everything runs on numpy, on a laptop or on this repository's GitHub Actions runners, which produced the results in
`results/`. Issues and Discussions are open; refutations welcome, bring the output.

## Two animations

![2-D turbulence at 256^2, vorticity, with energy and enstrophy on every frame](figures/turbulence_2d.gif)

*2-D decaying turbulence, 256^2, nu = 1e-4, vorticity. Vortices merge, filaments stretch and roll up, enstrophy
cascades to small scales; the energy on each frame decays only by the physical viscous rate. `flow_gif.py`.*

![1-D Burgers: inviscid blow-up (stopped at the grid limit) and the viscous shock on the Cole-Hopf exact solution](figures/burgers_shock.gif)

*Left: inviscid Burgers, energy 1.000000000000 on every frame, the gradient climbing toward 1/(1-t) until the grid
limit, where the animation stops and says so. Right: viscous Burgers on top of the Cole-Hopf closed form through the
shock, error printed per frame.*

## Three pictures

![1-D Burgers: the frozen-rotation scheme sits on the exact 1/(1-t) blow-up curve while upwind drifts and loses energy](figures/burgers_blowup.png)

*1-D Burgers. The scheme with zero numerical entropy production (black) lies on the exact blow-up curve (grey) until the
grid can no longer resolve the gradient; the standard dissipative scheme (orange) departs from it and, inset, loses the
energy the equation says is conserved.*

![3-D Euler: vortex stretching and enstrophy at three resolutions, converging upward](figures/stretching_3d.png)

*3-D Euler, Taylor-Green. The vortex-stretching term - the one that decides the 3-D question - measured directly with the
budget closed, at 64^3, 96^3, 128^3. Energy is 1.000000 throughout. Coarser grids under-estimate the amplification; the
curves converge upward.*

![3-D Taylor-Green: vorticity isosurfaces in the symmetry cell and planar slices at t = 1..4](figures/vortex_sheets_3d.png)

*3-D Euler, Taylor-Green at 48^3. Top: isosurface of |ω| in the symmetry cell [0,π]^3; bottom: |ω| on a plane. The
initial tori become sheets (t = 2) that thin and stretch (t = 3) and fold to the grid scale (t = 4) - the last frame
visibly under-resolved, as the resolution table says. Energy 1.000000 throughout. `vortex_iso.py`.*

![2-D turbulence: enstrophy decreasing while the unsigned palinstrophy production is large and positive](figures/ladder_2d.png)

*2-D decaying turbulence. The unsigned production term one level above enstrophy is large and positive throughout, and
enstrophy falls anyway: regularity is not the absence of the dangerous term but the presence of a controlled norm one
level below it. Figures are regenerated by `plots.py` from the same code as the tables.*

## Poke holes
Every claim, what would refute it, the command that tests it, and the list of our own weak points: [CLAIMS.md](CLAIMS.md).
Issues and Discussions are open. Refutations are welcome; bring the output.

## The equations

**Burgers (1-D):**

$$u_t + u u_x = \nu u_{xx}, \qquad x \in [0, 2\pi),\ \text{periodic},\ u(x,0)=\sin x .$$

For $\nu = 0$ the solution is given implicitly by characteristics, $u = \sin(x - u t)$, and its gradient blows up in
finite time:

$$\max_x |u_x(\cdot,t)| = \frac{1}{1-t}, \qquad t^* = 1 .$$

Energy $E(t) = \tfrac{1}{2}\langle u^2\rangle$ is exactly conserved for $\nu=0$ while the solution is smooth, and for
$\nu>0$ obeys

$$\frac{dE}{dt} = - \nu \langle u_x^2\rangle .$$

**Navier-Stokes / Euler (2-D vorticity form and 3-D velocity form):**

$$\omega_t + (\mathbf{u}\cdot\nabla) \omega = \nu \Delta\omega, \qquad \mathbf{u} = \nabla^\perp\psi,\ \ \Delta\psi = \omega \quad (\text{2-D})$$

$$\mathbf{u}_t + (\mathbf{u}\cdot\nabla) \mathbf{u} = -\nabla p + \nu \Delta\mathbf{u}, \qquad \nabla\cdot\mathbf{u}=0 \quad (\text{3-D})$$

with the Taylor-Green initial condition $\mathbf{u}_0 = (\sin x\cos y\cos z,\ -\cos x\sin y\cos z,\ 0)$. In 2-D the
viscous Taylor-Green flow is exact: $u = e^{-2\nu t}\sin x\cos y,\ v = -e^{-2\nu t}\cos x\sin y$, so $E(t)=E_0 e^{-4\nu t}$.
In 3-D with $\nu=0$ energy is conserved exactly and enstrophy $Z = \tfrac12\langle|\omega|^2\rangle$ grows by vortex
stretching; whether that growth stays finite for all time is the open question.

## The scheme, and what "numerical entropy production" means

Write the equation as **conservative transport + dissipation**:

$$\partial_t \mathbf{u} = \underbrace{\mathcal{T}(\mathbf{u})}_{\text{conserves } E}  +  \underbrace{\nu \Delta\mathbf{u}}_{\text{dissipates}} .$$

The transport term is discretised so that the semi-discrete system conserves energy **exactly**: in Fourier space each
linear mode advances by an exact rotation (a unitary factor exp(-i k c \Delta t), which is an isometry), and the
nonlinearity is written in skew-symmetric form so that $\langle \mathbf{u}, \mathcal{T}(\mathbf{u})\rangle = 0$
identically. For Burgers:

$$u u_x  =  \tfrac{1}{3}\Big(u u_x + (u^2)_x\Big) ,$$

which conserves $\langle u^2\rangle$ term by term on the dealiased grid (2/3 rule). Viscosity enters as an exact
integrating factor exp(-\nu k^2 \Delta t) per mode; time stepping of the nonlinear term is RK4.

For any integrator define the **numerical entropy production rate** as the drift of the energy the equation says must
be conserved, per unit time:

$$\sigma_{\text{num}}  =  -\frac{1}{t} \log\frac{E(t)}{E(0)}\Big|_{\nu=0} .$$

For an exact isometric scheme $\sigma_{\text{num}} = 0$ up to round-off; for a dissipative scheme (upwind
differencing) $\sigma_{\text{num}} > 0$, and that spurious dissipation acts like an extra viscosity that damps precisely
the small-scale growth a blow-up produces. With physical viscosity present, a fair test is whether the **measured**
dissipation equals the **physical** one, $-\tfrac{dE}{dt} = \nu\langle u_x^2\rangle$, with nothing added.

The reference scheme is first-order upwind on the conservative flux $f = u^2/2$:

$$u_j^{n+1} = u_j^n - \frac{\Delta t}{\Delta x}\big(F_{j+1/2} - F_{j-1/2}\big) + \nu \Delta t \frac{u_{j+1}-2u_j+u_{j-1}}{\Delta x^2},
\qquad F_{j+1/2} = \begin{cases} f_j & u_j>0 \ f_{j+1} & u_j\le 0 \end{cases}$$

## Results so far

### 1-D Burgers, inviscid, `u0 = sin x` (`burgers_entropy.py`)
The gradient provably blows up at $t^*=1$ with $\max|u_x| = 1/(1-t)$. Grid $N=512$, dt = 2e-4.

| t | frozen-rotation scheme: E/E0, max\|u_x\|, L2 error | truth max\|u_x\| | first-order upwind: E/E0, L2 error |
|---|---|---|---|
| 0.50 | 1.00000, 2.00, 0.0000 | 2.00 | 0.9973, 0.0026 |
| 0.80 | 1.00000, 5.00, 0.0000 | 5.00 | 0.9952, 0.0066 |
| 0.90 | 1.00000, 9.99, 0.0000 | 10.00 | 0.9941, 0.0097 |
| 0.95 | 1.00000, 18.72, 0.0004 | 20.00 | 0.9934, 0.0116 |

Numerical entropy production sigma_num over $[0, 0.95]$: upwind 7e-3 per unit time; frozen -3e-15 (zero to machine precision). The frozen scheme tracks the exact approach to the singularity until the 512-point grid can no longer
resolve a gradient of 20 - a visible resolution limit, not hidden dissipation.

### 2-D viscous Taylor-Green, exact solution known (`taylor_green.py`, part A)
nu = 0.02, 128^2 grid, dt = 2e-3, viscosity as an exact integrating factor per mode.

| t | E/E_0 scheme | E/E_0 exact = exp(-4nu t) | L_2 error vs the exact field |
|---|---|---|---|
| 0.5 | 0.96078944 | 0.96078944 | 5e-15 |
| 1.0 | 0.92311635 | 0.92311635 | 1e-14 |
| 1.5 | 0.88692044 | 0.88692044 | 1.5e-14 |
| 2.0 | 0.85214379 | 0.85214379 | 2e-14 |

The dissipation is exactly the physical $\nu\langle|\nabla u|^2\rangle$ and nothing more. (Taylor-Green in 2-D is a
single decaying mode, so this is a precision test, not a cascade test.)

### 3-D inviscid Taylor-Green / Euler, the Brachet (1983) benchmark (`taylor_green.py`, part B)
Skew-symmetric nonlinearity, Leray projection, RK4, 2/3 dealiasing, $\Delta t = 2/N$.

| | t = 1 | t = 2 | t = 3 | t = 4 |
|---|---|---|---|---|
| E/E_0, all three grids | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| enstrophy Z, 32^3 | 0.417 | 0.574 | 0.914 | 1.570 |
| enstrophy Z, 48^3 | 0.417 | 0.574 | 0.931 | 1.724 |
| enstrophy Z, 64^3 | 0.417 | 0.574 | 0.939 | 1.823 |

Energy is conserved to six decimals in 3-D: zero numerical entropy production on the benchmark flow. Enstrophy
agrees across resolutions to three decimals through $t = 2$ and then fans out as the cascade reaches scales the
coarser grids cannot hold; the spread shrinks with resolution (convergence). Because nothing dissipates
numerically, under-resolution appears as **disagreement between grids** rather than as a smooth, plausible and wrong
curve - which is the point of the instrument. The trustworthy window at $64^3$ is roughly $t \le 3$; the literature
uses far higher resolution beyond that.

### Three more closed-form solutions, one per dimension (`exact_solutions.py`)
| exact solution | grid | error against the closed form |
|---|---|---|
| **1-D** viscous Burgers, u0 = sin x, nu = 0.02: Cole-Hopf, evaluated in heat-kernel form | 512 | 5e-14 (t=0.5), 6e-12 (t=1), 2e-6 through the shock (t=1.5-2, gradient 22 on dx = 0.012: the resolution floor) |
| **2-D** a *random* vorticity field on the single shell k^2 = 25 (nonlinear term vanishes identically): u0 exp(-25 nu t) | 128^2 | 2e-14 to 3e-14 out to t = 4 |
| **3-D** ABC flow with viscosity (Beltrami, curl u = u): u0 exp(-nu t) | 32^3 | 1e-15 to 7e-15 out to t = 8 |

![Exact solutions in 1-D, 2-D, 3-D and the scheme's error against each](figures/exact_solutions.png)

*The 2-D case is a random exact solution, not a hand-picked mode; the 3-D case is the closed form the Beltrami
property provides. The 1-D shock comparison is the demanding one, and its error is the grid's, visibly.*

### 3-D Euler: the vortex-stretching term measured, by resolution (`budgets3d.py`, run on GitHub's runners)
$\dot Z = S$ with $S=\langle\omega\cdot(\omega\cdot\nabla)u\rangle$ measured from the field; energy $E/E_0 = 1.000000$ at
every grid and time; budget residual 1e-5-1e-6.

| t | 64^3: Z / S | 96^3: Z / S | 128^3: Z / S | S/Z^3/2 at 128^3 |
|---|---|---|---|---|
| 1 | 0.4169 / 0.0876 | 0.4169 / 0.0882 | 0.4169 / 0.0885 | 0.329 |
| 2 | 0.5743 / 0.238 | 0.5743 / 0.239 | 0.5743 / 0.239 | 0.550 |
| 3 | 0.9386 / 0.511 | 0.9430 / 0.526 | 0.9447 / 0.533 | 0.581 |
| 4 | 1.823 / 1.502 | 1.918 / 1.801 | 1.975 / 2.029 | 0.731 |

Converged to four digits through $t=2$. At $t=4$ the stretching still rises with resolution (1.50, 1.80, 2.03) with
shrinking differences: under-resolved grids **under**-estimate the amplification - they hide growth by inability, not
by damping, which is the opposite of a dissipative scheme's failure mode. The normalised rate $S/Z^{3/2}$ rises with
resolution as well (0.61, 0.68, 0.73 at $t=4$). This is a measurement of the term that decides the 3-D question, at
a resolution where its trend can be trusted, and nothing more.

### Mechanism, second invariant, and the BKM quantity (`mechanism3d.py`, $64^3$, from CI)
| flow | helicity < u.w> | enstrophy Z | max\|w\| (BKM integrand) | alignment with the intermediate strain eigenvector |
|---|---|---|---|---|
| Taylor-Green | 1e18 (zero by symmetry, preserved) | 0.375 → 1.823 | 2.0 → 13.7 | 0.356 at t=2 vs 0.317 / 0.327 (random 0.333) |
| perturbed ABC | **2.979345 → 2.979345** through t=3; 2.979187 at t=4 with energy 0.999979 | 1.58 → 11.3 | 3.1 → 47.5 | 0.339 vs 0.322 / 0.339 |

Helicity is conserved to six digits under dynamics that amplify enstrophy seven-fold; at $t=4$ the two invariants drift
together (within a factor 2.5), which is the grid being reached ($\max\lvert\omega\rvert = 47$ at $\Delta x = 0.1$).
The classical preference of vorticity for the intermediate strain eigenvector (Ashurst et al. 1987) is present but
weak here - a few points above random - because neither flow is developed turbulence by $t=4$; the forced run in
`spectra.py` is where the textbook signal belongs, and that check is pending.

### The regularity criteria along the classical candidates (`criteria3d.py`, CI)
**Reliability first.** The analyticity-strip clock (Sulem-Sulem-Frisch 1983) falls below 2 dx for full-box Kida-Pelz by
t = 1.0 at 64^3, 96^3 AND 128^3 (delta 0.124, 0.102, 0.079 against 0.196, 0.131, 0.098): the flow is trustworthy only to
t ~ 0.5-0.75 without symmetry reduction. Numbers previously shown here for Kida-Pelz at t = 3 were outside that window and
are withdrawn. Inside the window the picture converges and is physical:

| Kida-Pelz at t = 0.5 | Z | S | max\|w\| | CF coherence rho at h = 2pi/32 | Lipschitz exponent of the direction field |
|---|---|---|---|---|---|
| 64^3 | 5.4197 | 5.816 | 6.08 | 0.128 | 1.49 |
| 96^3 | 5.4197 | 5.817 | 6.08 | 0.127 | 1.51 |
| 128^3 | 5.4197 | 5.817 | 6.12 | 0.124 | 1.53 |

Four-digit agreement, and the geometric quantity is converged: the early roughening of the vorticity direction
(exponent 1.5 against 2 for a smooth field, from 2.7 at t = 0) is the flow's, not the grid's. The 64^3 run's later
"collapse" of the exponent to 0.5 was a grid artefact (96^3 and 128^3 give 1.15 at t = 1, itself past the clock). On
Taylor-Green and ABC (inside their windows) the critical L3 norm is flat to three digits, the direction field stays
coherent, and the high-vorticity set localises to under 1% of the volume. With nu = 1e-3, Kida-Pelz's enstrophy is
capped near 30 where the inviscid run reached 69, and the L3 norm falls 0.97 -> 0.83. Time integration is converged
(dt/2 changes the state by 3e-5). `THEORY.md` section 6 lists the criteria.

### Feedback, not imitation: searching initial data (`adversarial_ic.py`, CI)
Gradient ascent through the differentiable solver on the enstrophy amplification at fixed initial enstrophy, from smooth
|k| <= 4 data; every candidate re-verified at higher resolution with the analyticity clock.

| objective | search grid (32^3) | verified | classical best (reliable) | verdict |
|---|---|---|---|---|
| enstrophy, Kida-Pelz amplitude | 8.75x | 20.1x at 64^3, delta = 0 | 3.0x | outside the reliable window |
| enstrophy, Taylor-Green amplitude | 4.96x | 8.03x at 96^3, delta 0.06 < 0.13 | 1.11x | outside the reliable window |
| enstrophy with helicity = 0 (penalty) | 1.21x | 1.213x at 64^3, delta 0.28 > 0.20 | 1.112x | **pass** |
| enstrophy with helicity = 0.71 max (penalty) | 1.13x | 1.097x at 64^3 | 1.112x | fail (below classical) |
| enstrophy with helicity ~ 0.99 max (penalty) | 1.13x | 1.006x: no amplification | 1.112x | fail (below classical) |
| enstrophy, Taylor-Green amplitude, corrected verdict | 4.96x | 7.89x at 96^3 (delta 0.050), 8.70x at 128^3 (delta 0.040) | 1.112x | outside the window at both |
| enstrophy, leashed: delta(T) >= 0.30 on the 32^3 search grid | 4.44x, delta 0.27 | 8.28x at 128^3, delta 0.042 | 1.112x | outside the window |
| enstrophy, leashed: delta(T) >= 0.45 on the 32^3 search grid | 3.82x, delta 0.41 | 5.85x at 128^3, delta 0.047 | 1.112x | outside the window |
| enstrophy, leashed on a 64^3 search grid (checkpointed autograd) | 3.09x, delta 0.26 | 3.175x at 128^3 (delta 0.071 vs 0.098), 3.182x at 192^3 (delta 0.055 vs 0.065) | 1.112x | outside the window at both |
| helicity held at 0 (hard constraint, corrected bound) | 4.54x | 7.38x at 96^3, delta 0.047 | 1.112x | outside the window |
| helicity held at 0.25 max | 4.68x | 7.11x at 96^3, delta 0.065 | 1.112x | outside the window |
| helicity held at 0.50 max | 4.47x | 6.35x at 96^3, delta 0.069 | 1.112x | outside the window |
| helicity held at 0.75 max | 3.11x | 3.99x at 96^3, delta 0.076 | 1.112x | outside the window |
| helicity held at 0.90 max | 1.27x | 1.266x at 96^3, delta 0.20 > 0.13 | 1.112x | **pass** |
| Jacobi growth along Taylor-Green | 1.90x spreading | - | - | measurement only |

Pointed at enstrophy, the searcher twice found smooth low-k data that cascades to the grid cutoff within one time unit
while every classical flow stays smooth - the Lu-Doering / Ayala-Protas phenomenon - and twice the verifier rejected
it as unresolved, which is the design working. The leashed search (a penalty when delta(T) on the search grid drops
below a threshold) satisfied its leash on the 32^3 grid and still failed at 128^3, and that failure is the finding:
the coarse grid's truncation blocks the cascade the field triggers, so the field's spectrum tail *looks* resolved at
32^3 (delta 0.27-0.41) while its true continuation has delta 0.04. Resolution measured on the search grid is not
resolution. The leash has to be measured where the cascade lives. On a 64^3 search grid (gradient checkpointing, 2.2 GB)
the amplification is nearly converged between grids (3.09 at 64^3, 3.175 at 128^3) and the strip at 128^3 is 0.071
against a threshold of 0.098: a three-fold amplifier from smooth |k| <= 4 data that is almost, not yet, resolved.
At 192^3 the amplification is converged (3.182) but the strip is 0.055 against 0.065: delta keeps *falling*
with resolution (0.26 at 64^3, 0.071 at 128^3, 0.055 at 192^3) as each finer grid sees more of a flattening tail.
Registered FAIL. The separation is the finding: the enstrophy amplification of this field is a robust number, and its
small scales are not resolved at any grid tried. The field is committed (`results/found/leashed64_dmin030.npz`, 64^3
float32, 3 MB) for anyone with a bigger box. Topology, with the
helicity held fixed by a hard constraint (Newton projection onto the level set after every step, `HELMODE=project`):
on the 32^3 truncated system the maximal amplification over one time unit is *flat* up to relative helicity 0.5
(4.54, 4.68, 4.47) and then collapses (3.11 at 0.75, 1.27 at 0.9). Helicity inhibits the cascade, as Moffatt's
conjecture wants, but as a threshold rather than a slope: half the maximal helicity costs nothing, ninety percent
costs a factor of four.

![Maximal enstrophy amplification versus relative helicity held fixed](figures/helicity_threshold.png)

The four fast fields cascade to the cutoff at 96^3 and are registered as outside the window;
the 0.9 field stays resolved and beats every classical flow (1.266 vs 1.112, pass). The earlier penalty-method rows
above, which showed a gentle monotone decline, were the penalty failing to explore, not the physics. (Correction, 2026-09-06: the first version of this table labelled these
0.5 and 0.7; the code's helicity bound was 2 sqrt(2 E Z) instead of the Cauchy-Schwarz 2 sqrt(E Z), a factor
sqrt 2. Fixed in `adversarial_ic.py`; a hard-constraint ladder at 0, 0.25, 0.5, 0.75, 0.9 is the replacement.) Arnold's
geodesic spreading along mild Taylor-Green is 1.9x over one time unit; it needs a ladder before it means more.

### Learned coarse simulators of 2-D Kolmogorov flow (`kolmogorov_v2.py`, run on CI)
$32^2$ coarse models against a $128^2$ truth, 2000-step rollouts from held-out states, $Re\approx1250$.

| coarse simulator | energy / truth | spectrum error (2nd half, averaged) |
|---|---|---|
| fully learned CNN (width 64, 16-step unroll) | 0.95 - **1.91** | 0.50 |
| frozen exact transport, no closure | 0.96 - 1.07 | 0.53 |
| frozen + dissipative gate only (nu_tge0; v1) | 0.95 - 1.07 | 0.53 |
| **frozen + energy-neutral learned redistribution + gate** | 0.87 - 1.07 | **0.09** |

In 2-D the missing physics at coarse resolution is backscatter, which a dissipative closure cannot express: the gate
alone changes nothing (0.53 = no closure). A learned term projected to move energy between scales *without creating
it* cuts the long-rollout spectrum error six-fold, while the fully learned model drifts to 1.9x the true energy.
Registered: spectrum ratio 0.18 (bar 0.6), energy band within [0.8, 1.2], learned leaves it - all pass. The dip to 0.87
is NOT the first-order leak of the skew projection: v3 (`kolmogorov_v3.py`, exact energy-neutral rescaling) reproduces
v2 to the digit, so the dip comes from the gate, which trained beside the skew term learns to dissipate more than
viscosity alone - and that is what best matches the truth's spectrum. A prediction of ours refuted by its own test.

### Energy spectra (`spectra.py`, CI) and exact time integration (`midpoint.py`)
![Energy spectra: 2-D k^-3 and 3-D k^-5/3 with a pile-up at the grid cutoff](figures/spectra.png)

*2-D decaying turbulence at $256^2$ follows Kraichnan's $k^{-3}$ over a decade (a little steeper, as decaying 2-D
turbulence does). 3-D forced turbulence at $48^3$ follows $k^{-5/3}$ for a few wavenumbers and then **piles up at
the cutoff**: energy arrives faster than viscosity removes it on a grid this small. A dissipative scheme would have
damped that into a plausible slope; the conserving scheme shows the under-resolution. $96^3$ is queued.*

Implicit midpoint on the skew transport makes the discrete energy conservation exact: 3-D drift
3.5e-8 (RK4) -> 5.7e-12 (midpoint), round-off. Weak point 5 of `CLAIMS.md`, closed.

### Budget closure in 1-D and 2-D, where Hypothesis H is known (`budgets.py`)
Every balance law checked term by term along the flow (measured $d/dt$ across one step vs the right-hand side from
the field), including the **unsigned production terms** whose 3-D cousin is the regularity problem.

| case | budget | frozen scheme residual | upwind residual | what it shows |
|---|---|---|---|---|
| 1-D inviscid (H false) | gradient, -(1/2)<u_x^3> production | 1e8 | 1-5 % | production +0.13, +0.47, +1.59 at t = 0.3, 0.6, 0.8: the blow-up, measured |
| 1-D viscous (H true) | energy | 1e9 | **10 %** | upwind's numerical dissipation is a tenth of the physical viscosity |
| 1-D viscous | maximum principle max\|u\| | 0.994, 0.988, 0.984 | - | never increases: the controlled norm that gives 1-D regularity |
| 2-D decaying (H true, M=Z) | energy / enstrophy / palinstrophy | 1e8 / 1e7 / 3-9e-6 | - | palinstrophy production **+1387, +2149, +1579, +576** (unsigned, large) while enstrophy falls 12.86 -> 6.09 |

The 2-D row is Theorem 5 of `THEORY.md` with its terms filled in: the dangerous unsigned term is present and large,
and regularity holds because a norm one level below it is controlled. The registered residual bar of 1e-6 is
narrowly missed by the 2-D palinstrophy budget (terms of order $1e3$, RK4 error); halving the time step would
clear it, and the bar is left where it was.

## Liouville: the ensemble version of zero entropy production

Lee (1952) showed that the Galerkin-truncated Euler equations are a divergence-free vector field on the phase space
of retained modes: phase-space volume, and with it the Gibbs entropy of any ensemble of solutions, is exactly
conserved. That is the ensemble form of the statement this repository is built on. `liouville.py` measures it with
autograd, taking the exact trace of the Jacobian of the transport operator (one reverse pass per coordinate):

```
                          div F = tr(dF/dU)         expected
2-D  16^2  nu = 0         -2.9e-16                  0
3-D  12^3  nu = 0         +3.3e-16                  0
2-D  16^2  nu = 0.01      -24.20                    -nu (d-1) sum_k k^2 = -24.20
3-D  12^3  nu = 0.01      -82.32                    -nu (d-1) sum_k k^2 = -82.32
```

Two things worth noticing. The inviscid divergence is zero for the skew form, the advective form and the divergence
form alike: Liouville is a weaker property than energy conservation (the advective form loses energy but not phase
volume). And the viscous contraction is a constant independent of the state, so along truncated Navier-Stokes the
Gibbs entropy of any ensemble decreases linearly, S(t) = S(0) - nu (d-1) (sum_k k^2) t, however turbulent the flow.
What the flow does is not create or destroy phase volume but stretch it: the Jacobi ladder below measures the rate.

## The Riemannian view, measured: Jacobi fields along the Euler geodesic

Arnold (1966): an Euler flow is a geodesic on the group of volume-preserving maps with the L2 metric, and the
separation of two nearby geodesics (a Jacobi field) is governed by the sectional curvature; negative curvature
means exponential separation. `jacobi_ladder.py` integrates a flow and a copy perturbed by 1e-5 of a random
solenoidal field and reports the separation growth and its local exponent lambda, with the reliability clock on
every row (rows past the clock are omitted here).

```
Taylor-Green            growth |du|/(eps|v|)              local exponent lambda
    t         32^3      48^3      64^3              32^3     48^3     64^3
   0.5      1.0088    1.0128    1.0143              0.018    0.025    0.028
   1.0      1.0466    1.0553    1.0572              0.073    0.082    0.083
   1.5      1.1134    1.1301    1.1348              0.124    0.137    0.142
   2.0      1.2193    1.2488    1.2593              0.182    0.200    0.208
   2.5         -         -      1.4485                -        -      0.280

Kida-Pelz
   0.5         -      1.2301    1.2660                -      0.414    0.472
```

![Jacobi-field growth along Taylor-Green and Kida-Pelz by resolution](figures/jacobi_ladder.png)

Three readings. The growth converges with resolution inside the window (Taylor-Green at t = 2: 48^3 and 64^3
agree to 1%). The local exponent rises steadily with time, 0.03 to 0.28 for Taylor-Green: the geodesic moves into
more negatively curved parts of the group as the vortex sheets form, which is Arnold's prediction in the one
quantity a computer can measure. And Kida-Pelz is four times more unstable already at t = 0.5, before its enstrophy
has grown by more than 30%: it is not yet converged at 64^3 (3% change from 48^3), so the number is provisional.
Energy in both copies stays at 1 to 1e-8 throughout; nu = 0.

Together with Liouville above this is the whole thermodynamic picture of the truncated inviscid system in two
numbers: phase volume is conserved exactly (div F = 0), and it is stretched at rate lambda in the least stable
direction and therefore compressed elsewhere. The flow does not lose information; it moves it to where a finite
observer cannot read it. That is also, word for word, what the analyticity-strip clock measures.

## Complex singularities: the analyticity strip as a Riemann-surface question

The clock used throughout this page is itself the classical singularity detector (Sulem, Sulem and Frisch 1983;
Frisch, Matsumoto and Bec 2003): E(k) ~ exp(-2 delta(t) k) where delta(t) is the distance of the nearest
complex-space singularity of the solution from the real domain, and a real finite-time singularity is delta(t*) = 0.
The two classical hypotheses are distinguishable *inside the reliable window* by the trend of the local decay rate
-d log(delta)/dt: constant for exponential decay (no real singularity), rising as 1/(t* - t) for linear decay.
`strip_tracker.py` samples delta every 0.025-0.05 time units and fits both laws over the window only.

![Analyticity-strip width delta(t) by resolution, Taylor-Green and Kida-Pelz](figures/strip_decay.png)

```
flow            N^3   window          exponential tau   (2nd half)   linear t*   (2nd half)   decay rate start -> end   better fit
Taylor-Green     48   0.15 - 2.15       0.93            1.20          2.09        2.84           2.3 -> 0.51             exponential
Taylor-Green     64   0.15 - 2.60       1.04            1.69          2.41        3.65            -  -> 0.37             exponential
Taylor-Green     96   0.20 - 2.30       1.03            0.95          2.39        2.72            -  -> 0.77             exponential
Kida-Pelz        64   0.15 - 0.65       0.40            0.44          0.81        0.97           3.3 -> 2.0              exponential
Kida-Pelz        96   0.15 - 0.80       0.44            0.46          0.92        1.11           3.9 -> 1.7              exponential
Kida-Pelz       128   0.15 - 0.85       0.42            0.45          0.91        1.14           4.1 -> 1.9              exponential
```

Extended to 160^3 and 192^3 (`strip_hi`): tau = 0.41 and 0.375 (second half 0.50, 0.58), decay rate flat at 2.0,
linear t* retreating to 1.24 and 1.34. Five resolutions agree. The front-arrival test - the time at which the
cascade reaches the grid's reliability line 2dx = 4pi/N, which converges to t* for a real singularity (the dyadic
model below) and grows without bound otherwise:

```
   N^3     96      128     160     192       slope per ln N     prediction for 256^3
   t_arr   0.830   0.880   0.920   0.950     0.17, 0.18, 0.17   0.999
```

Logarithmic growth with a constant slope over four resolutions: no saturation. (The slope is below tau, so a single
exponential delta is too simple a model of the tail; the free-prefactor fit is unstable even with 32 shells and is
not used.)

Kida-Pelz is the case the literature argued about (Boratav and Pelz 1994 for a singularity; Hou and Li 2006, 2008
against). Here the exponential time constant is converged across 64/96/128^3 (0.40, 0.44, 0.42; second half
0.44, 0.46, 0.45), the local decay rate falls by half inside every window instead of rising, and the linear
extrapolation's t* retreats as the window lengthens - all three signatures of no real singularity in the reach of
these runs. Taylor-Green likewise (Brachet et al. 1983). The absolute delta shifts down with resolution because the
prefactor exponent n in k^-n exp(-2 delta k) is held at 0 (the clock's definition) and the fit range moves to
higher k; the decay *law* is what is compared, and it is resolution-independent. This is not a proof of anything:
it says that within t < 0.85 for Kida-Pelz and t < 2.6 for Taylor-Green, at up to 128^3, the nearest complex
singularity is moving away from the real axis at a slowing rate, exactly as the regularity side of the argument
predicts, and that a claim of a singularity from either flow would have to come from beyond where this instrument
can see.

## Searching for the missing functional, with an adversary in the loop

Hypothesis H needs a functional M(u) >= Z that never increases along Navier-Stokes trajectories. Theorem 4 says it
must be built from what Tao averaged away - local structure in physical space - so `lyapunov_search.py` looks for
one there: M = Z exp(Phi), Phi an enstrophy-weighted average of a small network over eight dimensionless pointwise
features of the vorticity and strain fields (stretching rate along the vorticity, strain magnitude and determinant,
direction roughness |grad xi|, local enstrophy and energy density, two alignment measures). No grid quantity enters,
so the trivial truncated-system bound is unavailable. dM/dt is the exact Lie derivative by autograd. After each
training round the differentiable solver searches initial data for the trajectory along which M grows fastest, and
those states join the training set. Registered verdict: PASS only if the final adversary and the held-out classical
flows show no violation above 1e-3.

```
24^3, nu = 2e-3, T = 0.6, Z0 = 0.375, three rounds (results/lyapunov_24_local.txt)
round   training worst   held-out worst (Kida-Pelz, random)   adversary's best violation (relative dM/dt)
  0        -0.020           -0.081                              +0.40
  1        -0.051           -0.283                              +2.40
  2        -0.039           -0.076                              +0.71
feature sensitivity of the learned Phi:  |w|^2/2Z 1.08,  xi.S.xi/sqrt Z 0.33,  |S|^2/Z 0.22,  |grad xi|/k_rms 0.09, ...
REGISTERED -> FAIL
```

Reading: the learner finds, every round, a local M that decreases along every classical flow and every previous
counterexample - and the adversary finds, every round, a new smooth low-k field along which it grows at 40-240% per
unit time. No trend toward closure in three rounds. The candidate leans on exactly the Constantin-Fefferman
quantities (local enstrophy density, stretching rate along the vorticity, strain magnitude), which is where a proof
would have to live; the search says that within this class - bounded, local, first-derivative features, one
time unit - the adversary always has a move. A society of six candidates with M = Z exp(min_i Phi_i) (multiple Lyapunov functions, Branicky 1998: each candidate
only has to decrease where it is the one in force, and a switch can only lower M) does no better - adversary +0.47,
+1.37, +0.66 - and is worse out of sample (Kida-Pelz now violates, +0.07). So the adversary's move is not a region no
candidate covers; from any M in this class it finds a smooth field along which M grows. The class is the problem: every
feature is local in the vorticity and strain at a point, and the quantity that decides how the stretching will
*change* is the pressure Hessian, a nonlocal singular integral of the whole field - the restricted-Euler dynamics
without it blows up in finite time (Vieillefosse 1982; Cantwell 1992), and it is precisely the specific singular
integral Tao's averaging destroys. Runs on CI at 32^3 for eight rounds then showed the one encouraging series of the week - adversary violations
falling from 1.0 to 0.03-0.09 with local features only - and the register was built for exactly this moment: the
same candidate attacked afresh by a stronger adversary (four restarts, forty iterations; `results/lyapunov_32_p0_attack.txt`)
is broken at +0.40, +0.35, +0.55. The fall was the fixed-budget searcher tiring, not the candidate hardening. With
pressure-Hessian features the eight-round series plateaued at 0.3 and the attack gives +0.74-0.79. With the signed-beta features the eight-round series reached 0.01, the best seen; attacked, it breaks at +0.42
(`results/lyapunov_32_b1_attack.txt`). A twenty-round candidate with a 25-iteration training adversary fell to 0.01-0.04 over its last five rounds
and, attacked with five restarts of sixty iterations, breaks at +1.03 (`results/lyapunov_32_r20_attack.txt`). Three
candidates, three falling series, three breaks. A candidate is only as good as the strongest adversary that has
failed against it; none has, and the class - bounded local functionals of the present vorticity and strain field, at
this resolution and window - is closed by this register: the fixed-budget searcher tires before the candidate holds. This is feedback,
not imitation, applied to the proof itself: the machine cannot produce a theorem, but it produces the counterexamples
a human would need to see before trying to.

## The price of the projection: what the global does for the local

The nonlinearity of Navier-Stokes, (u.grad)u, is purely local. Euler is that operator followed by the Leray
projection onto divergence-free fields, and the pressure *is* the projection: it is the only nonlocal thing in the
equation, decided by a Poisson equation over the whole box. Drop the projection and the equation is 3-D Burgers,
which blows up in finite time by a theorem (along characteristics the gradient obeys DA/Dt = -A^2, so
A(t) = A0 (I + A0 t)^-1 exactly and the first shock is at t* = -1/min lambda(A0)). `projection_price.py` runs both
from the same initial field with the same instrument (results at 48^3, all flows at the same initial enstrophy):

```
                            Euler (projected)                        Burgers (unprojected, local)
flow            exact t*    strip decay rate, t=0.2 -> 0.8           strip decay rate      clock stops at    max|grad u| vs exact A0(I+A0 t)^-1
Taylor-Green    1.000       2.71 -> 1.21  (falling)                  2.77 -> 3.59 (rising)  80% of t*        1.999 / 2.000, 2.490 / 2.500 until the grid fails
Kida-Pelz       1.277       1.95 -> 0.86  (falling)                  2.65 -> 1.68           71% of t*        1.001 / 1.003, 1.261 / 1.290
searcher's field 0.422      2.71 -> 1.40  (falling, Z x3)            4.34 -> 2.99           95% of t*        5.449 / 5.415
```

The unprojected run reproduces the exact characteristics solution to four digits until the grid fails, and its
clock stops at 71-95% of the exact shock time: a sixth closed-form check, in 3-D, on a blow-up. With the projection
on, from the same data, the strip's decay rate falls instead of rising in every case. Nothing else differs. Two
further readings. Energy: Euler conserves it *because* of the projection (the unprojected energy drifts by the
physical amount 1/2 int |u|^2 div u, up to 0.8% here); the conservation this repository verifies to 1e-14 is a
consequence of the nonlocal term, not independent of it. And the search: even the adversary's fastest field, whose
unprojected copy shocks at t = 0.42, is held by the projection to a three-fold enstrophy growth with a falling
decay rate.

How much of the pressure Hessian is global? Its trace is local (lap p = |w|^2/2 - |S|^2, decided at the point);
the traceless part is decided by the whole field. Restricted Euler keeps only the local part and blows up
(Vieillefosse 1982; Cantwell 1992). `pressure_share.py` measures the global share <|P_dev|^2>/<|P|^2> on the
high-vorticity set:

```
                    t = 0     0.25    0.5     0.75    1.0       Z(1)/Z0
Taylor-Green        0.39      0.41    0.49    0.58    0.67      1.11
Kida-Pelz           0.34      0.34    0.38    0.48    0.57      1.10
ABC (steady)        0.725     0.725   0.725   0.725   0.725     1.00
searcher's field    0.52      0.47    0.47    0.49    0.50      3.04
```

In the classical flows the global share of the pressure Hessian *rises* as the sheets form: the whole box
increasingly informs each point. In the adversary's field it stays flat at one half while the enstrophy triples.
That is a measurement of how the searcher wins: it finds data along which the global part of the pressure does not
grow with the stretching. Read with the Lyapunov result above, the two say the same thing from opposite sides: a
local functional cannot see the term that decides the future, and the fields that defeat it are the ones where
that term is weakest.

## The signed angle: what the squared coherence cannot see

Every alignment quantity in the literature and on this page so far is squared - the Constantin-Fefferman coherence
is 1 - (xi.xi')^2 - which folds antiparallel onto parallel: two vortex lines pointing opposite ways register as
perfectly coherent. That is right for CF's depletion argument (the Biot-Savart kernel depends on |sin|) and blind for
a diagnostic, because antiparallel neighbouring vorticity is the geometry of sheets and of the classic singularity
candidates (Kerr 1993; Hou-Li 2006). `pressure_share.py` now also reports the signed two-point angle
beta = 1 - xi(x).xi(x+h) in [0, 2] and the fraction of the strong-vorticity set whose neighbour two cells away is
antiparallel (beta > 1), at 64^3:

```
                     mean beta(h=2)  t=0 / 0.5 / 1.0     antiparallel fraction  t=0 / 0.5 / 1.0     Z(1)/Z0
Taylor-Green         0.004 / 0.006 / 0.012                0.000 / 0.000 / 0.000                       1.11
Kida-Pelz            0.009 / 0.014 / 0.039                0.000 / 0.000 / 0.000                       1.10
searcher's field     0.052 / 0.148 / 0.542                0.000 / 0.184 / 0.906                       3.12
```

By t = 1, ninety-one percent of the strongest vorticity in the searcher's field has an antiparallel neighbour: it is
built almost entirely of antiparallel pairs, while the classical flows have none at all - and the squared coherence
reported all three at the same 0.13. This is the sharpest single picture of how the adversary wins, and it names
the geometry: antiparallel sheets. Read with the concentration exponent (alpha ~ 4, flat) it says the pairs thin
without sharpening, which is what the projection does to them.

## Water's memory: reversibility and the compression half

Euler is time-reversible. `jacobi_ladder.py` with `REVERSE=1` runs a flow to T, flips the velocity, runs the same
instrument back to 0 and reports the round-trip error - the amount the flow has forgotten, which for an exact
inviscid flow is zero and for a truncated one is the information that passed below the cutoff - and the Jacobi
exponent of the reversed leg, which is minus the most CONTRACTING exponent of the forward flow: the compression half
that Liouville pairs with every stretch and that nobody measures.

```
                        round-trip error        forward stretching lambda_f (t=0.5-1.0)   reversed lambda_b (same window)   lambda_b - lambda_f
Taylor-Green   64^3     1.5e-9                  0.083                                     0.016                             -0.07
Kida-Pelz      96^3     2.0e-6                  1.27 (past the clock)                     0.57                              -0.70
searcher's     96^3     5.0e-5                  0.76 (past the clock)                     0.04                              -0.72
```

The memory is perfect until the cascade carries information below the resolvable scale: five orders of magnitude
separate the round-trip error of Taylor-Green from the searcher's field, in the order of their cascades - the
truncated analogue of what viscosity does. And in every flow the strongest compression is far weaker than the
strongest stretching: Liouville balances the books with one concentrated stretch and many diffuse squeezes. A
collapse to a point would need the compression concentrated as well (lambda_b >= lambda_f); no flow here shows it,
the adversary's field least of all. Provisional: the reversed leg starts from the least-resolved state, so most of
its rows are past the clock; the ordering is the claim, not the sizes.

## Water's memory as an algorithm: Navier-Stokes from Cauchy's memory and jitter

Cauchy (1815): vorticity is the initial vorticity remembered along particle paths. Constantin and Iyer (2008):
Navier-Stokes is the same memory read along Brownian-jittered paths and averaged - viscosity is the jitter, not a
separate term. `memory_paths.py` runs it in 2-D (Chorin's random vortex method): particles carry w0, move with the
flow plus sqrt(2 nu) dW, are deposited to the grid, and the velocity comes from the deposited vorticity by
Biot-Savart. No viscous term is ever applied. Against the spectral instrument, enstrophy at t = 2:

```
                              spectral NS     particles + jitter (2 / 4 / 8 per cell)     no-noise control
nu = 2e-3, 128^2              0.938           0.939 / 0.933 / 0.932                        0.991
nu = 2e-3, 256^2              0.956           0.956                                        0.999
nu = 1e-2, 128^2              0.766           0.770                                        0.991
nu = 0                        1.000           0.991                                        0.991
field error vs spectral, 128^2:  9.1% -> 4.5% -> 2.4% as particles per cell double (1/sqrt N, Monte Carlo)
```

The viscous decay rate is reproduced to 0.5-1% at two viscosities by jitter alone; the no-noise particles keep their
memory (the 0.99 is deposition smoothing at t = 0, identical at nu = 0). `wave_particle.py` adds the exact 1-D
version: the spectral (wave) solution evaluated at the particle positions agrees with the memory the particles carry
to 7e-15 while the wave is resolved, and the discrepancy then grows by seven orders of magnitude exactly as the
analyticity strip collapses toward the grid - the two descriptions are one flow, and the difference between them is
the observer's. Two more phase facts from the same script: the same amplitude spectrum shocks anywhere between
t = 0.16 and 0.38 depending on its phases (2000 random phase sets, factor 2.37); and in 2-D a developed turbulent field
with every phase randomised - identical spectrum, energy and enstrophy - has its palinstrophy production cut by a
factor of 250 (ratio 0.004), then regrows its phase correlations within a quarter time unit and cascades at the
original rate. The energy is in the amplitudes; the cascade is interference.

## Frequency matching: does incommensurability protect a flow?

Quasicrystals form because atoms hold two incommensurate lengths no periodic arrangement can satisfy at once. Does
data whose two scales cannot lock cascade more slowly or shock later? `frequency_matching.py`, registered prediction
"no": Conway's local-isomorphism property means every local configuration recurs, so the worst case is not avoided,
only relocated. 1-D Burgers, exact, equal energy: the harmonic pair (5, 10) has shock time 0.097 at its best phase and
0.067 at its worst (spread 1.46x); the incommensurate pair (5, 7 ~ sqrt 2) has 0.086 and 0.083 (spread 1.03x). Both
reach the fully aligned slope k + q somewhere; the incommensurate pair reaches it *whatever the phase*. Aperiodic
completeness is the reason it cannot be a shield: it guarantees the worst configuration is present. In 2-D the
question turned out not to be cleanly posable with two modes: a first design (5,10) versus (5,7) was confounded by
wavenumber content and is withdrawn; the fair design - two waves of identical |k| at 90, 53 and 37 degrees - gave
nine identical runs with no cascade at all, because w = k^2 psi makes any such field an exact steady state of 2-D
Euler (equal-magnitude waves do not interfere in 2-D). Recorded as a null with its reason.

## The blow-up type, in the frame the theorems use

The combined CKN / Seregin-Sverak / KNSS program zooms into a would-be singularity and asks for its type: max|w| ~
(t* - t)^-gamma with gamma >= 1 (the BKM gate; Type I is gamma = 1), faster (Type II), or exponential (no
singularity). `strip_tracker.py` now fits both inside the reliable window. Kida-Pelz at 96^3: a power law with
gamma = 0.21, far below the gate. The searcher's field at 96^3 and 128^3: exponential growth of max|w| at a rate of
about 1.6 per time unit fits at least as well as any power law, and the local growth rate falls toward the end of
the 128^3 window (1.74 -> 0.91). Exponential thinning is what a strained sheet does; it is not a singularity type.

## Keep the state, track the dissipation: the CKN concentration exponent

Navier-Stokes carries its state by an exact transport and leaks energy only through nu |grad u|^2, whose time
integral is bounded a priori by E(0). Caffarelli, Kohn and Nirenberg (1982) turned that into the strongest partial
result there is: a point is regular unless the dissipation in the parabolic cylinder of radius r around it is at
least eps r, so the singular set has parabolic dimension at most one. `ckn_exponent.py` measures how the dissipation
actually concentrates around the strongest vortex along a flow, D(r) ~ r^alpha over a ladder of r (uniform smooth
dissipation: alpha = 5; a sheet: 4; a tube: 3; CKN-critical: 1), reporting a radius only once the cylinder fits
inside the elapsed time. 64^3, nu = 1e-3, same initial enstrophy:

```
                     fit alpha (r = 2-6 cells)     smallest resolved pair (3 -> 4 cells)    over t = 0.4 .. 1.0
Taylor-Green         5.4                           4.4                                      flat, Z x 1.1
searcher's field     4.5 - 4.6                     3.6 - 3.8                                flat, Z x 2.8, max|w| x 3
```

The adversary's field concentrates its dissipation like sheets, a full exponent below Taylor-Green at the smallest
resolved scale, and the exponent does not fall while the enstrophy nearly triples: three and a half exponents from
the critical value and not moving toward it. This is the sixth view of the same flow (strip, direction coherence,
projection, pressure share, helicity, and now dissipation concentration), and they say one thing between them: the
searcher makes enstrophy grow by building sheets in the helicity-free regime where the global part of the pressure
stays weak, and the projection keeps the sheets from sharpening into anything a singularity needs. The exponent is
also the first search objective that points at the theorem itself: minimise alpha rather than maximise enstrophy.
Done (`OBJ=ckn`, 64^3 search, 128^3 verification, `results/sheet_conjecture_64.txt`): rewarded for concentration the
searcher builds something sharper than a sheet - spatial exponent alpha_s = 1.25 on the search grid, 1.36 at 128^3
(a tube; uniform is 3, a sheet 2) - with only 1.8x enstrophy growth, and it is unresolved at 128^3 (delta 0.078 vs
0.098), registered FAIL. So growth buys sheets and concentration buys tubes, and neither is resolved. The sheet
conjecture stands for the enstrophy objective; the stronger claim, that nothing sharper can be built, is refuted.

## Tao's wall, in pictures: an energy-conserving equation that provably blows up

Theorem 4 (Tao 2016) says that the exact structure this repository verifies - energy conservation, the scaling, the
quadratic skew nonlinearity - is not enough to rule out a singularity, because an equation with exactly that
structure blows up. The simplest such equation is the dyadic shell model of Katz and Pavlovic (2005), and its
inviscid blow-up is a theorem (Katz-Pavlovic 2005; Kiselev-Zlatos 2005). `dyadic.py` runs it through the same
instrument: RK4 with exact viscous integrating factor, energy drift reported, the analyticity width delta(t)
fitted from the shell spectrum with a reliability rule, and the same two decay laws fitted inside the window that
`strip_tracker.py` fits for Kida-Pelz.

![The dyadic model and Kida-Pelz Euler through the same diagnostic](figures/taos_wall.png)

```
                              energy drift    delta(t)                      decay rate across window     verdict
dyadic, nu = 0, 34 shells     3e-8            straight to 0 at t* = 0.5585   7.8 -> 1.4e6  (x 178000)     singularity (theorem)
dyadic, nu = 1e-6 / 1e-4      (dissipates)    front stalls at shell 12 / 7   turns negative               no singularity
Kida-Pelz Euler 64/96/128^3   1e-8            exponential, tau 0.42          4.1 -> 1.9  (falls by half)  no singularity indicated
```

Three things this shows. Energy is conserved to 3e-8 while the enstrophy goes as (t* - t)^-1.7 to the truncation:
zero numerical entropy production is a property of the *instrument*, not evidence about the *equation*. The decay
rate of delta is the discriminator that separates the two cases by five orders of magnitude, so when it falls for
Kida-Pelz that is a statement, not noise. And the dyadic model does not satisfy Liouville (div F = -sum k_{j+1}
a_{j+1}, state-dependent, printed on every row), while the true truncated Euler system does exactly. That is *not* a
property that separates the true equation from Tao's class: a multiplier-averaged nonlinearity M B(Mu, Mu), a member
of Tao's averaged family, conserves phase volume to 1e-17 as well (`liouville.py`, form `tao-class`;
`results/liouville_taoclass.txt`), because Liouville follows from the convolution structure and the dyadic model is
not a convolution. So phase-volume conservation sits on Tao's side of the wall along with energy. What the true
equation has and the averaged one lacks remains the short list Tao averaged away: locality in physical space,
particle paths, and the geometry of stretching. A proof has to be made of those.

## A boundary of the method, from the recent literature

Wang, Lai, Gomez-Serrano and Buckmaster (arXiv 2509.14185, 2025; 2511.22819) found new families of *unstable*
self-similar singularities for Boussinesq, IPM and Euler with boundary by searching profile space directly with
physics-informed networks, and state the hypothesis that unstable singularities are the route for boundary-free
Euler and Navier-Stokes. An unstable singularity lives on a measure-zero set of initial data. A gradient search from
generic smooth data - which is what `adversarial_ic.py` is - cannot find one by construction; it finds the fastest
*growth*, and what it found is sheets and tubes. Every "no singularity indicated" on this page is a statement about
the flows run and the windows resolved, never about the existence of unstable singularities, which this instrument
does not look for. Barker (arXiv 2510.20757, 2025-26) gives quantitative bounds near a potential blow-up for Hou's
approximately axisymmetric Navier-Stokes candidate that are explicitly amenable to numerical testing; that candidate
lives in a cylinder with a boundary, outside this periodic box, and is the natural next target for an instrument
built like this one. Shahmurov (arXiv 2604.09949, 2026) proves for axisymmetric model equations that the sign of the
elliptic response decides blow-up - the physical sign is global, the reversed sign blows up - which is the analytic
face of the projection experiment above.

## What this is and is not
- It is a measurement of an **instrument property**: no artefact dissipation. Numerical searches for self-similar
  blow-up (Hou; Gomez-Serrano, Buckmaster et al. 2022; and later neural-network-assisted searches) are limited by
  exactly this artefact, which damps the small-scale growth they are trying to detect.
- It is **not** a statement about 3-D Navier-Stokes regularity. Nothing here proves anything; a proof is a theorem
  or nothing.

## Further reading
[CLAIMS.md](CLAIMS.md) - every claim, its refutation condition, its command, and our weak points.

[EQUATIONS.md](EQUATIONS.md) - the one-family system in $d=1,2,3$, the skew/contractive split, the discretisation, and the budget line with its
production term per dimension: [EQUATIONS.md](EQUATIONS.md). One solver for all three: `ns_d.py`.

[THEORY.md](THEORY.md) - definitions, the elementary propositions with proofs, the classical limits (Tao 2016; Beale-Kato-Majda), and the one open
hypothesis stated attackably: [THEORY.md](THEORY.md).

## Run
```
python burgers_entropy.py      # ~1 minute, CPU
python taylor_green.py         # a few minutes, CPU
```
numpy only.
