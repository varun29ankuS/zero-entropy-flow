# Isometry plus contraction: a formal note

This note states precisely what "freeze the conservative part, let only the dissipation act" means, proves the
elementary consequences, records the classical theorems that bound what it can do, and states the one hypothesis
under which it would decide regularity. Status of each statement is marked: **[proved here]**, **[classical]**,
**[open]**.

---

## 1. Setting

Let $H$ be a real Hilbert space with inner product $\langle\cdot,\cdot\rangle$ and norm $\|\cdot\|$. Consider an
evolution

$$\dot x  =  T(x)  +  D(x), \qquad x(0)=x_0 ,$$

with the two parts characterised by their action on the norm:

* **Transport** $T$ is *skew* (norm-preserving): $\langle x, T(x)\rangle = 0$ for all $x$ in its domain.
* **Dissipation** $D$ is *contractive*: $\langle x, D(x)\rangle = - \mathcal{E}(x) \le 0$, with $\mathcal{E}\ge 0$
  called the dissipation rate.

Examples. Burgers: $T(u) = -u u_x$, $D(u)=\nu u_{xx}$ on the torus; $\mathcal{E}(u)=\nu\|u_x\|^2$. Navier-Stokes on
the torus $\mathbb{T}^d$, divergence-free velocity fields: $T(u) = -P[(u\cdot\nabla)u]$ with $P$ the Leray projector,
$D(u)=\nu\Delta u$, $\mathcal{E}(u)=\nu\|\nabla u\|^2$. A recurrent memory: $x_{t+1} = R_t x_t$ with $R_t$
orthogonal is the discrete transport; a gate $x \mapsto (1-g)\odot x$ with $g\in[0,1]$ is the discrete dissipation.

**Definition 1 (entropy production).** For a differentiable flow map $\Phi_t$ on a finite-dimensional state space,
the entropy production rate is the volume contraction rate

$$\sigma(x)  =  - \nabla\cdot F(x) \qquad\text{for } \dot x = F(x),$$

and for a discrete map $x_{t+1}=\Phi(x_t)$ it is $\sigma = -\log|\det J_\Phi(x_t)|$. (This is the phase-space
contraction rate of Ruelle; for a thermostatted system it equals the thermodynamic entropy production.)

**Definition 2 (numerical entropy production).** For a numerical scheme applied to a system whose exact dynamics
satisfies $\frac{d}{dt}\|x\|^2 = -2\mathcal{E}(x)$, the numerical entropy production over $[0,t]$ is

$$\sigma_{\text{num}}  =  -\frac{1}{t}\Big(\log\frac{\|x_t\|^2}{\|x_0\|^2}  +  \frac{2}{\|x_0\|^2}\int_0^t \mathcal{E} ds\Big),$$

the dissipation the scheme produces beyond the physical one. For $\mathcal{E}\equiv 0$ it reduces to
$-\tfrac1t\log(\|x_t\|^2/\|x_0\|^2)$, the quantity measured in `burgers_entropy.py`.

---

## 2. What isometry plus contraction gives

**Proposition 1 (energy inequality). [proved here; classical for NS, Leray 1934]**
For any solution of $\dot x = T(x)+D(x)$ on $[0,t]$ smooth enough for the identities to hold,

$$\|x(t)\|^2 + 2\int_0^t \mathcal{E}(x(s)) ds  =  \|x_0\|^2 .$$

*Proof.* $\frac{d}{dt}\|x\|^2 = 2\langle x,\dot x\rangle = 2\langle x,T(x)\rangle + 2\langle x,D(x)\rangle = 0 - 2\mathcal{E}(x)$. Integrate. $\square$

**Proposition 2 (discrete energy inequality; zero numerical entropy). [proved here]**
Let a scheme advance $x_n \mapsto x_{n+1}$ by a map that is the composition of an exact isometry $R_n$
($\|R_n x\|=\|x\|$) and a contraction $C_n$ ($\|C_n x\|\le\|x\|$), in either order. Then
$\|x_{n+1}\|\le\|x_n\|$ for every $n$; the discrete solution is bounded for all $n$ by $\|x_0\|$; and if $C_n$
is the identity the scheme has $\sigma_{\text{num}} = 0$ exactly.

*Proof.* $\|C_n R_n x\| \le \|R_n x\| = \|x\|$, and likewise in the other order. Induct. If $C_n=\mathrm{id}$, the
norm is constant and $\sigma_{\text{num}}=0$ by Definition 2. $\square$

*Remark.* The pseudo-spectral scheme in this repository realises Proposition 2 up to time-stepping error: the
skew-symmetric form makes the semi-discrete transport exactly skew ($\langle \hat u, \hat T(\hat u)\rangle = 0$ on
the dealiased modes), the viscous integrating factor exp(-\nu k^2\Delta t) is an exact contraction per mode, and
RK4 adds an $O(\Delta t^4)$ deviation. Measured: $\sigma_{\text{num}} = -3e-15 (Burgers), energy conserved
to 1e-6 (3-D Euler).

**Proposition 3 (a frozen isometric memory carries a conserved quantity exactly; a dissipative one forgets at
the entropy production rate). [proved here]**
(i) Let $x_{t+1} = R_t x_t$ with each $R_t$ orthogonal, and let $\delta_t$ be the difference between two
trajectories with the same $R_t$. Then $\|\delta_t\| = \|\delta_0\|$ for all $t$.
(ii) Let $x_{t+1} = (1-g_t)\odot R_t x_t + b_t$ with $g_t\in[0,1]^d$ (gated recurrence with input $b_t$). Then
$\|\delta_t\| \le \prod_{s<t}(1-\min_i g_{s,i}) \|\delta_0\|$, and the per-step entropy production is
$\sigma_t = -\sum_i \log(1-g_{t,i})$.
(iii) (Position clock.) For rotation planes with angular rates $w_k$ driven by increments $\Delta_t$,
$\theta_{k,t+1} = \theta_{k,t} + w_k\Delta_t$, the phases satisfy $\theta_{k,t} = \theta_{k,0} + w_k\sum_{s<t}\Delta_s$
exactly, i.e. the state is an exact function of the accumulated quantity, for every $t$, with no trained parameter.
If the increments carry i.i.d. noise of variance $\varsigma^2$, the decoded quantity has error of order
$\varsigma\sqrt{t}$ (a random walk); if the true value is re-observed every $G$ steps, the error is of order
$\varsigma\sqrt{G}$, uniformly in $t$.

*Proof.* (i) $\delta_{t+1} = R_t\delta_t$, orthogonal. (ii) $\delta_{t+1} = (1-g_t)\odot R_t\delta_t$, and
$\|(1-g)\odot y\|\le (1-\min_i g_i)\|y\|$; the Jacobian is $\mathrm{diag}(1-g_t)R_t$ with
$\log|\det| = \sum_i\log(1-g_{t,i})$. (iii) Telescoping sum; the noise statements are the variance of a sum of
$t$ (resp. at most $G$) independent increments. $\square$

*Remark.* Part (ii) is the "forgetting law" and the measured $\sim 250$ nats/step of the trained drawing hand; part
(iii) is the measured 1.8 px at 4000 moves (no noise), $\sigma\sqrt{n}$ drift under poison, and the flat 6 px with a
glance every 50 moves. Note that in the Householder update $x\mapsto x-\beta u u^{\top}x$ the map is not
orthogonal unless $\beta\in\{0,2\}$: $\det = 1-\beta = \cos\theta$, so the "transport" of that architecture is
itself a contraction along $u$ and contributes $\log|\cos\theta|$ to $\sigma$. The exact-transport claim applies to
the rotation planes, not to the Householder erasure.

---

## 3. What isometry plus contraction cannot give

**Theorem 4 (the abstraction is insufficient for regularity). [classical: Tao 2016]**
There exists an averaged Navier-Stokes system $\dot u = \tilde T(u) + \nu\Delta u$ on $\mathbb{T}^3$ with $\tilde T$
skew (so Proposition 1 holds verbatim, with the same energy identity as Navier-Stokes) and smooth divergence-free
initial data whose solution blows up in finite time.

*Consequence.* No argument that uses only the structure "skew transport + contractive dissipation" and the energy
inequality can prove global regularity for 3-D Navier-Stokes, since the same argument would apply to Tao's system.
Any proof must use properties of the specific nonlinearity $P[(u\cdot\nabla)u]$ that are not consequences of
skewness.

*Reference.* T. Tao, *Finite time blowup for an averaged three-dimensional Navier-Stokes equation*, J. Amer. Math.
Soc. 29 (2016), 601-674.

**Theorem 5 (what suffices: a second controlled norm). [classical; conditional form proved here]**
Let $u$ be a Leray-Hopf solution on $\mathbb{T}^3\times[0,T)$ that is smooth on $[0,T)$. If

$$\int_0^T \|\omega(s)\|_{L^\infty} ds < \infty, \qquad \omega=\nabla\times u,$$

then $u$ extends smoothly past $T$ (Beale-Kato-Majda 1984). Consequently: **if there is a functional $M(u)\ge 0$ with
(a) $M(u) \ge c \|\omega\|_{L^\infty}$ (or any bound implying the BKM integral is finite), and (b)
$\frac{d}{dt}M(u)\le 0$ along solutions - i.e. $M$ is a norm in which the transport is skew or contractive and the
dissipation contractive - then solutions are globally smooth.**

*Proof of the consequence.* (b) gives $M(u(t))\le M(u_0)$ on $[0,T)$; (a) gives $\|\omega\|_{L^\infty}\le M(u_0)/c$;
the BKM integral is at most $T M(u_0)/c<\infty$; apply BKM. $\square$

*Remark (why 2-D is solved and 3-D is not).* In two dimensions enstrophy $Z=\tfrac12\|\omega\|^2$ satisfies
$\frac{d}{dt}Z = -\nu\|\nabla\omega\|^2 \le 0$: the vortex-stretching term $\int \omega\cdot(\omega\cdot\nabla)u$
vanishes identically, so the transport is skew in a *second* norm that controls the gradient, and global regularity
follows (Ladyzhenskaya 1959; for the periodic setting see Temam's text). In three dimensions the same computation gives
$\frac{d}{dt}Z = \int\omega\cdot(\omega\cdot\nabla)u - \nu\|\nabla\omega\|^2$, and the first term has no sign. The
transport is an isometry of $L^2$ only. Whether any functional satisfying (a) and (b) exists is **[open]**; it is
equivalent in spirit to the Millennium problem, and Theorem 4 says it cannot be found by abstract arguments alone.

---

## 4. What this repository establishes, stated exactly

1. The scheme is an instance of Proposition 2: its transport is skew on the dealiased modes and its dissipation is
   an exact contraction, so numerical entropy production is zero up to RK4 error. **[proved here + measured]**
2. On 1-D Burgers it reproduces the exact approach to the known singularity, $\max|u_x| = 1/(1-t)$, until the grid
   limit, which appears as a visible shortfall rather than as damping. **[measured]**
3. On 2-D Navier-Stokes (Taylor-Green) it reproduces a closed-form solution to 1e-14; its dissipation is exactly
   the physical $\nu\|\nabla u\|^2$. **[measured]**
4. On 3-D Euler (Taylor-Green) energy is conserved to 1e-6 at $32^3, 48^3, 64^3$, and enstrophy agrees across
   resolutions until the cascade reaches the grid scale, after which the resolutions *disagree* rather than
   converge to a wrong value. **[measured]**
5. None of 1-4 bears on the regularity question except as an instrument free of the artefact (numerical
   dissipation) that limits numerical searches for singular solutions. **[Theorem 4]**

## 5. The one open hypothesis, stated so that it can be attacked or refuted

**Hypothesis H.** There exists a functional $M$ on divergence-free fields on $\mathbb{T}^3$, satisfying (a) and (b)
of Theorem 5 along Navier-Stokes solutions.

If H holds, 3-D regularity follows by Theorem 5. If a smooth solution blows up, H is false. Theorem 4 shows H cannot
be established from skewness and dissipation alone; a candidate $M$ must use the structure of
$P[(u\cdot\nabla)u]$. The numerical route to evidence is to compute, for candidate $M$, the sign of
$\frac{d}{dt}M$ along flows near the strongest known amplification events, with a scheme whose own dissipation is
zero so that the sign is not an artefact - which is what Proposition 2 provides.

## 6. What a proof must control: the criteria, and what is measured

Each of the following is a theorem giving a sufficient condition for smoothness; a blow-up must violate all of them.
The right-hand column is what `criteria3d.py` measures along a flow, by resolution, with the budgets closed.

| criterion | statement | measured |
|---|---|---|
| Beale-Kato-Majda (1984) | int_0^T max\|w\| dt < inf implies smooth on [0,T] | max\|w\|(t) and its time integral |
| Ladyzhenskaya-Prodi-Serrin; Escauriaza-Seregin-Sverak (2003) | sup_t \|\|u\|\|_L3 < inf implies smooth (the scale-critical norm) | \|\|u(t)\|\|_L3 |
| Constantin-Fefferman (1993) | if the vorticity direction xi = w/\|w\| is Lipschitz where \|w\| is large, stretching is depleted and the solution is smooth | direction coherence rho = <1 - (xi(x).xi(x+h))^2> over \|w\| > 0.5 max, h = dx; and the local stretching alpha = xi.S.xi there |
| Caffarelli-Kohn-Nirenberg (1982) | the singular set has one-dimensional parabolic Hausdorff measure zero | (a blow-up, if any, is at points, not sheets or lines) |
| Tao (2016) | the skew-plus-dissipation structure alone cannot decide regularity | (limits what any argument built on Sections 1-2 can prove) |

A regularity proof shows one of the first three always holds. A blow-up construction violates all of them at a point
and uses the exact nonlinearity. The measurements do neither; they report which criterion is tightest on the classical
candidate flows (Taylor-Green; the perturbed ABC flow; the Kida-Pelz high-symmetry flow, whose apparent blow-up
dissolved at high resolution, Hou and Li 2006) and whether the critical norm stays bounded on them.
