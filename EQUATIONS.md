# The equations

## The system, one family in d = 1, 2, 3

On the periodic box $\mathbb{T}^d=[0,2\pi)^d$, for a velocity field $u(x,t)\in\mathbb{R}^d$:

$$\boxed{ \partial_t u  =  - P\left[(u\cdot\nabla)u\right]  +  \nu \Delta u }, \qquad \nabla\cdot u = 0\ \ (d\ge 2),$$

with $P = I - \nabla\Delta^{-1}\nabla\cdot$ the Leray projector, $\hat P_k = I - k k^{\top}/|k|^2$ in Fourier space. It removes the
pressure gradient; pressure is the Lagrange multiplier enforcing $\nabla\cdot u=0$, not an unknown.

* $d=1$: incompressibility would force $u_x=0$, so the 1-D member is unprojected, $P=I$: **Burgers**, $u_t+u u_x=\nu u_{xx}$.
* $d=2$: with vorticity $\omega=\partial_x v-\partial_y u$, stream function $\Delta\psi=-\omega$, $u=(\partial_y\psi,-\partial_x\psi)$:
  $\omega_t+(u\cdot\nabla)\omega=\nu\Delta\omega$.
* $d=3$: as boxed; the vorticity $\omega=\nabla\times u$ obeys $\omega_t+(u\cdot\nabla)\omega=(\omega\cdot\nabla)u+\nu\Delta\omega$.
  The term $(\omega\cdot\nabla)u$ is vortex stretching; it is absent in $d=2$.

## The split

$$\partial_t u = \underbrace{T(u)}_{\langle u,T(u)\rangle=0} + \underbrace{D(u)}_{\langle u,D(u)\rangle=-\nu\|\nabla u\|^2},\qquad
T(u)=-P[(u\cdot\nabla)u],\quad D(u)=\nu\Delta u .$$

$T$ is skew because for divergence-free $u$, $\langle u,(u\cdot\nabla)u\rangle=\tfrac12\int(u\cdot\nabla)|u|^2=-\tfrac12\int|u|^2 \nabla\cdot u=0$. Hence

$$\frac{d}{dt}\tfrac12\|u\|^2=-\nu\|\nabla u\|^2 \qquad\text{(Leray, every } d).$$

## The discretisation (`ns_d.py`)

Fourier modes with $|k_i|<N/3$ (2/3 dealiasing). Transport in **skew-symmetric form**, skew on the discrete grid:

$$T_i(u)=-\tfrac12\big[(u\cdot\nabla)u_i+\nabla\cdot(u u_i)\big]\ (d=2,3),\qquad T(u)=-\tfrac13\big[u u_x+(u^2)_x\big]\ (d=1),$$

then $\hat P_k$. Viscosity as the exact integrating factor $e^{-\nu|k|^2\Delta t}$ per mode (an exact contraction). RK4 in
time for $T$. Numerical entropy production is therefore zero up to the $O(\Delta t^4)$ RK4 error: measured
$10^{-15}$ (1-D), $10^{-14}$ against the 2-D closed form, $10^{-6}$ (3-D).

## The budgets, one line in every dimension

With $Z=\tfrac12\langle u_x^2\rangle$ ($d=1$) or $Z=\tfrac12\langle|\omega|^2\rangle$ ($d=2,3$):

$$\frac{dZ}{dt}=S-\nu\langle|\nabla\omega|^2\rangle,\qquad
S=\begin{cases}-\tfrac12\langle u_x^3\rangle & d=1\ \ \text{(unsigned; drives the blow-up at }t^*=1)\ 0 & d=2\ \ \text{(identically: enstrophy is a second controlled norm, hence regularity)}\ \langle\omega\cdot(\omega\cdot\nabla)u\rangle & d=3\ \ \text{(vortex stretching; unsigned; the open question)}\end{cases}$$

Measured: $S=+0.13,+0.47,+4.92$ approaching the 1-D singularity ($t=0.3,0.6,0.9$); $S\equiv0$ in 2-D with $Z$ falling
monotonically; $S=+0.088,+0.238,+0.511,+1.502$ in 3-D at $t=1,2,3,4$ ($64^3$), budget residual $10^{-5}$, and the
normalised rate $S/Z^{3/2}$ rising $0.33\to0.61$.

## The hierarchy of budgets, and where the dimension enters

The energy identity is the same in every $d$. Climbing one derivative at a time, each norm has its own budget and
each new level brings a production term the level below did not have:

| level | quantity | budget | production term |
|---|---|---|---|
| 0 | E=(1/2)<| u|^2> | dE/dt=-nu<|grad  u|^2> | none, in any d |
| 1 | Z=(1/2)<\|w\|^2> | dZ/dt=S_1-nu<\|grad w\|^2> | S_1=-(1/2)<u_x^3> (d=1); 0 (d=2); <w.(w.grad)u> (d=3) |
| 2 | P=(1/2)<\|grad w\|^2> | dP/dt=S_2-nu<\|lap w\|^2> | unsigned in every d (measured +1387, +2149, +1579, +576 in 2-D) |
| k | (1/2)||grad ^k u||^2 | same shape | unsigned, growing with k |

Regularity is decided by where the first unsigned term lands relative to the level that controls the gradient:
$d=1$, it lands at level 1 and nothing below controls the gradient (inviscid blow-up; the viscous case is rescued by
a different controlled quantity, $\max\lvert u\rvert$). $d=2$, level 1 is clean, $S_1\equiv0$, so $Z$ is controlled
and controls $\nabla u$; the unsigned term is pushed to level 2 where its size is harmless. $d=3$, the unsigned term
sits at level 1 and level 0 does not control the gradient: open.

Forcing adds an input $\langle u\cdot f\rangle$ at level 0 and $\langle\omega\cdot\nabla\times f\rangle$ at level 1.
A learned closure projected to be skew (as in `kolmogorov_v2.py`) is energy-neutral **to first order in the step**:
its component along the energy gradient is zero to machine precision ($10^{-12}$, `verify_skew_closure.py`, random
weights), but a finite step in an orthogonal direction still changes a quadratic energy at second order. Measured:
$9	imes10^{-6}$ relative energy change per step against $3.5	imes10^{-4}$ for the unprojected correction - forty
times less, not zero. Exact neutrality needs one more line, a rescaling of the state to the energy of the frozen step
after the correction is added (v3).

## The contrast that locates the difficulty

Vacuum Maxwell, $\partial_t E=c \nabla\times B,\ \partial_t B=-c \nabla\times E$, is linear: every Fourier mode rotates at
$c|k|$, every norm is conserved, regularity is automatic. Replacing Navier-Stokes' self-transport by transport along a
fixed field $U$ gives Oseen, $\partial_t u=-P[(U\cdot\nabla)u]+\nu\Delta u$, also globally regular. The difficulty is
exactly the state-dependence of the transport: the field is both the thing carried and the carrier.
