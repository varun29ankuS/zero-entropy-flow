# zero-entropy-flow

Numerical experiments on one question: **does an integrator with zero numerical entropy production see a
fluid singularity forming, where a dissipative integrator hides it?**

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
linear mode advances by an exact rotation (a unitary factor $e^{-i k c \Delta t}$, which is an isometry), and the
nonlinearity is written in skew-symmetric form so that $\langle \mathbf{u}, \mathcal{T}(\mathbf{u})\rangle = 0$
identically. For Burgers:

$$u u_x  =  \tfrac{1}{3}\Big(u u_x + (u^2)_x\Big) ,$$

which conserves $\langle u^2\rangle$ term by term on the dealiased grid (2/3 rule). Viscosity enters as an exact
integrating factor $e^{-\nu k^2 \Delta t}$ per mode; time stepping of the nonlinear term is RK4.

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
The gradient provably blows up at $t^*=1$ with $\max|u_x| = 1/(1-t)$. Grid $N=512$, $\Delta t = 2	imes10^{-4}$.

| t | frozen-rotation scheme: E/E0, max\|u_x\|, L2 error | truth max\|u_x\| | first-order upwind: E/E0, L2 error |
|---|---|---|---|
| 0.50 | 1.00000, 2.00, 0.0000 | 2.00 | 0.9973, 0.0026 |
| 0.80 | 1.00000, 5.00, 0.0000 | 5.00 | 0.9952, 0.0066 |
| 0.90 | 1.00000, 9.99, 0.0000 | 10.00 | 0.9941, 0.0097 |
| 0.95 | 1.00000, 18.72, 0.0004 | 20.00 | 0.9934, 0.0116 |

Numerical entropy production $\sigma_{	ext{num}}$ over $[0, 0.95]$: upwind $7	imes10^{-3}$ per unit time; frozen $-3	imes10^{-15}$ (zero to machine precision). The frozen scheme tracks the exact approach to the singularity until the 512-point grid can no longer
resolve a gradient of 20 - a visible resolution limit, not hidden dissipation.

### 2-D viscous Taylor-Green, exact solution known (`taylor_green.py`, part A)
$\nu = 0.02$, $128^2$ grid, $\Delta t = 2\times10^{-3}$, viscosity as an exact integrating factor per mode.

| t | $E/E_0$ scheme | $E/E_0$ exact $= e^{-4\nu t}$ | $L_2$ error vs the exact field |
|---|---|---|---|
| 0.5 | 0.96078944 | 0.96078944 | $5\times10^{-15}$ |
| 1.0 | 0.92311635 | 0.92311635 | $1\times10^{-14}$ |
| 1.5 | 0.88692044 | 0.88692044 | $1.5\times10^{-14}$ |
| 2.0 | 0.85214379 | 0.85214379 | $2\times10^{-14}$ |

The dissipation is exactly the physical $\nu\langle|\nabla u|^2\rangle$ and nothing more. (Taylor-Green in 2-D is a
single decaying mode, so this is a precision test, not a cascade test.)

### 3-D inviscid Taylor-Green / Euler, the Brachet (1983) benchmark (`taylor_green.py`, part B)
Skew-symmetric nonlinearity, Leray projection, RK4, 2/3 dealiasing, $\Delta t = 2/N$.

| | t = 1 | t = 2 | t = 3 | t = 4 |
|---|---|---|---|---|
| $E/E_0$, all three grids | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| enstrophy $Z$, $32^3$ | 0.417 | 0.574 | 0.914 | 1.570 |
| enstrophy $Z$, $48^3$ | 0.417 | 0.574 | 0.931 | 1.724 |
| enstrophy $Z$, $64^3$ | 0.417 | 0.574 | 0.939 | 1.823 |

Energy is conserved to six decimals in 3-D: zero numerical entropy production on the benchmark flow. Enstrophy
agrees across resolutions to three decimals through $t = 2$ and then fans out as the cascade reaches scales the
coarser grids cannot hold; the spread shrinks with resolution (convergence). Because nothing dissipates
numerically, under-resolution appears as **disagreement between grids** rather than as a smooth, plausible and wrong
curve - which is the point of the instrument. The trustworthy window at $64^3$ is roughly $t \le 3$; the literature
uses far higher resolution beyond that.

### Budget closure in 1-D and 2-D, where Hypothesis H is known (`budgets.py`)
Every balance law checked term by term along the flow (measured $d/dt$ across one step vs the right-hand side from
the field), including the **unsigned production terms** whose 3-D cousin is the regularity problem.

| case | budget | frozen scheme residual | upwind residual | what it shows |
|---|---|---|---|---|
| 1-D inviscid (H false) | gradient, $-\tfrac12\langle u_x^3\rangle$ production | $10^{-8}$ | 1-5 % | production +0.13, +0.47, +1.59 at t = 0.3, 0.6, 0.8: the blow-up, measured |
| 1-D viscous (H true) | energy | $10^{-9}$ | **10 %** | upwind's numerical dissipation is a tenth of the physical viscosity |
| 1-D viscous | maximum principle $\max\|u\|$ | 0.994, 0.988, 0.984 | - | never increases: the controlled norm that gives 1-D regularity |
| 2-D decaying (H true, $M=Z$) | energy / enstrophy / palinstrophy | $10^{-8}$ / $10^{-7}$ / $3$-$9\times10^{-6}$ | - | palinstrophy production **+1387, +2149, +1579, +576** (unsigned, large) while enstrophy falls 12.86 -> 6.09 |

The 2-D row is Theorem 5 of `THEORY.md` with its terms filled in: the dangerous unsigned term is present and large,
and regularity holds because a norm one level below it is controlled. The registered residual bar of $10^{-6}$ is
narrowly missed by the 2-D palinstrophy budget (terms of order $10^3$, RK4 error); halving the time step would
clear it, and the bar is left where it was.

## What this is and is not
- It is a measurement of an **instrument property**: no artefact dissipation. Numerical searches for self-similar
  blow-up (Hou; Gomez-Serrano, Buckmaster et al. 2022; and later neural-network-assisted searches) are limited by
  exactly this artefact, which damps the small-scale growth they are trying to detect.
- It is **not** a statement about 3-D Navier-Stokes regularity. Nothing here proves anything; a proof is a theorem
  or nothing.

## The equations
The one-family system in $d=1,2,3$, the skew/contractive split, the discretisation, and the budget line with its
production term per dimension: [EQUATIONS.md](EQUATIONS.md). One solver for all three: `ns_d.py`.

## Theory
Definitions, the elementary propositions with proofs, the classical limits (Tao 2016; Beale-Kato-Majda), and the one open
hypothesis stated attackably: [THEORY.md](THEORY.md).

## Run
```
python burgers_entropy.py      # ~1 minute, CPU
python taylor_green.py         # a few minutes, CPU
```
numpy only.
