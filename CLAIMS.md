# Claims, how to refute them, and where we are weakest

Every claim below is registered with the observation that would refute it and the command that produces the number.
If you find one of these fails on your machine, open an issue with the output. If you think a claim is weaker than
stated, say so; the weak points we already know about are listed at the bottom.

## Claims

**C1. The scheme's numerical entropy production is zero to round-off.**
Refuted by: energy drift for inviscid Burgers or 3-D Euler larger than ~1e-6 relative with the default settings.
Test: `python burgers_entropy.py` (expect sigma_num ~ 1e-15) and `DIM=3 N=48 NU=0 T=4 python ns_d.py` (expect E/E0 = 1.000000).

**C2. Every balance law closes term by term, including the unsigned production terms.**
Refuted by: a relative residual above 1e-6 for any budget except the 2-D palinstrophy budget (3-9e-6, RK4 error on
terms of size 1e3; halve dt to clear it).
Test: `python budgets.py`; `DIM=1|2|3 python ns_d.py`.

**C3. The 1-D Burgers gradient follows the exact 1/(1-t) until the grid limit.**
Refuted by: max|u_x| departing from 1/(1-t) by more than the resolution shortfall before t = 0.9 at N = 512.
Test: `python burgers_entropy.py`; figure `figures/burgers_blowup.png` from `python plots.py`.

**C4. 2-D viscous Taylor-Green is reproduced to machine precision.**
Refuted by: L2 error against e^{-2 nu t} sin x cos y above 1e-12 at t = 2.
Test: `python taylor_green.py` (part A) or `DIM=2 N=128 NU=0.02 T=2 DT=2e-3 IC=tg python ns_d.py`.

**C5. In 3-D Euler (Taylor-Green) the vortex-stretching term converges upward with resolution: 1.50, 1.80, 2.03 at
t = 4 for 64^3, 96^3, 128^3, and the normalised rate S/Z^{3/2} rises with resolution.**
Refuted by: a higher-resolution run (192^3 or above) giving S(t=4) below 2.03, or a Z(t=4) inconsistent with the
published Taylor-Green curves (Brachet et al. 1983 and later).
Test: `N=128 T=4 python budgets3d.py` (~40 min on 2 cores); results in `results/budgets3d_*.txt`.

**C6. The vorticity aligns preferentially with the intermediate strain eigenvector (Ashurst et al. 1987).**
Refuted by: the intermediate-eigenvector fraction not exceeding the other two at 64^3 by t = 3.
Test: `IC=tg N=64 T=4 python mechanism3d.py`.

**C7. Helicity is conserved under 3-D Euler dynamics to the same order as energy.**
Refuted by: helicity drift exceeding energy drift by more than a factor of ~10 for the perturbed ABC flow.
Test: `IC=abc N=64 T=4 python mechanism3d.py`.

**C8. A learned coarse simulator with frozen exact transport holds energy within a few percent over 2000 steps where
a fully learned one drifts by tens of percent.**
Refuted by: a fully learned baseline of comparable size that keeps energy within the same band at the same
resolution and Reynolds number.
Test: `python kolmogorov_closure.py` (v1, ~40 min); `python kolmogorov_v2.py` (v2).

**C9 (negative). A non-negative eddy-viscosity closure learns nothing in 2-D at 32^2 and Re ~ 1250.**
Refuted by: the same closure producing a measurably different rollout from no closure.
Test: `python kolmogorov_closure.py`.

**C10. The skew-projected learned correction is energy-neutral to first order in the step (not exactly).**
Refuted by: a per-step relative energy change from the projected correction comparable to the unprojected one
(3.5e-4), or exactly zero (which would mean the second-order argument is wrong).
Test: `python verify_skew_closure.py` (expect ~1e-5 vs 3.5e-4).

**Not claimed.** Anything about the regularity of 3-D Navier-Stokes. `THEORY.md`, Theorem 4, records why the
structure used here cannot decide it.

## Where we are weakest (poke here first)

1. **One seed everywhere.** Random initial conditions (2-D decaying, ABC perturbation, Kolmogorov) use seed 0. No
   error bars.
2. **The upwind baseline is the weakest reasonable comparator.** First-order upwind is what textbooks use to show
   numerical dissipation; a high-order WENO or a standard energy-conserving finite-volume scheme would be the fair
   opponent. The point being made (dissipation hides growth) survives, but the size of the gap in the figures is
   against an easy target.
3. **The learned baseline in the Kolmogorov experiments is small** (a 3-layer CNN, 16-step unroll in v2). Stronger
   learned simulators exist (Kochkov et al. 2021, FNO variants). "Learned drifts" only counts against a strong one.
4. **Resolution.** 128^3 is small by modern standards; the t = 4 Taylor-Green numbers are a lower bound on a curve
   still rising with resolution, as stated. Nothing here approaches the resolutions of published singularity searches.
5. **RK4 is not exactly conservative.** The 10^-6 in 3-D and the 3-9e-6 palinstrophy residuals are time-stepping
   error. An exactly conservative time integrator (implicit midpoint on the skew part) would remove them.
6. **The 2/3 rule and the skew form.** Energy conservation on the dealiased modes is standard; we have not separately
   verified the aliasing error bound at the highest retained modes for the 3-D runs.
7. **Comparison with the literature is qualitative.** "In range" of Brachet's curves is not a digit-for-digit match;
   a digit-for-digit comparison at matched resolution and time step is the obvious next check.
8. **The theory note's Proposition 3 (memory) is about a different system** than the fluid experiments; it is in the
   note because the same inequality governs both, not because the fluid runs test it.
