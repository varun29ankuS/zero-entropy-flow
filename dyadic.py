"""What Tao's wall looks like: an energy-conserving equation that provably blows up, run through the same instrument.

The dyadic (shell) model of Katz and Pavlovic (2005), in the form used by Cheskidov (2008):
    da_j/dt = -nu k_j^2 a_j + k_j a_{j-1}^2 - k_{j+1} a_j a_{j+1},      k_j = 2^j,   j = 0..J-1
It has exactly the properties Theorem 4 says are insufficient: the nonlinearity is quadratic, has the scaling of
Navier-Stokes, and conserves energy E = 1/2 sum a_j^2 exactly (the two nonlinear sums telescope). And it is a theorem
that the inviscid model blows up in finite time from any nonzero data with positive components (Katz and Pavlovic
2005; Kiselev and Zlatos 2005): the enstrophy-like norm sum k_j^2 a_j^2 becomes infinite at a finite t*. For the
viscous model the answer depends on the dissipation exponent (Katz and Pavlovic 2005; Cheskidov 2008); with the full
Laplacian dissipation used here the front stalls at a finite shell and the energy dissipates, as the runs show.

What the model lacks is exactly what Tao averaged away: locality in physical space (a_j are shells, not points),
particle paths, and the geometry of stretching. It does not even satisfy Liouville: div F = -sum k_{j+1} a_{j+1},
state-dependent - phase volume is not conserved, which the true truncated Euler system does exactly (liouville.py).

Run with the same discipline as the fluid runs: RK4 with an adaptive step, energy drift reported, an analyticity
width delta(t) from the shell spectrum E_j ~ exp(-2 delta k_j) fitted over the upper active shells, and the
reliability rule delta > 4 / k_max (the analogue of delta > 2 dx). Then the two decay laws are fitted inside the
reliable window exactly as strip_tracker.py fits them for Kida-Pelz, so the two instruments can be compared: one
flow where the answer is a theorem (blow-up), one where it is the open question.
usage: J=34 NU=0 python dyadic.py     (NU=1e-4 for the viscous model)"""
import os, time, numpy as np

J = int(os.environ.get("J", 34))
NU = float(os.environ.get("NU", 0.0))
CFL = float(os.environ.get("CFL", 0.05))
k = 2.0 ** np.arange(J)


def F(a):
    nl = np.zeros_like(a)
    nl[1:] += k[1:] * a[:-1] ** 2            # k_j a_{j-1}^2
    nl[:-1] -= k[1:] * a[:-1] * a[1:]        # -k_{j+1} a_j a_{j+1}
    return nl


def step(a, dt):
    """RK4 on the nonlinearity with viscosity as an exact per-shell integrating factor, as in the fluid solver"""
    f, f2 = np.exp(-NU * k**2 * dt), np.exp(-NU * k**2 * dt / 2)
    k1 = F(a)
    k2 = F(f2 * (a + dt / 2 * k1))
    k3 = F(f2 * a + dt / 2 * k2)
    k4 = F(f * a + dt * f2 * k3)
    return f * a + dt / 6 * (f * k1 + 2 * f2 * k2 + 2 * f2 * k3 + k4)


def delta_of(a):
    E = a**2
    act = np.where(E > 1e-40 * E.max())[0]
    if len(act) < 6:
        return np.nan
    hi = act[len(act) // 2:]                 # upper half of the active shells, as in the fluid clock
    slope = np.polyfit(k[hi], np.log(E[hi] + 1e-300), 1)[0]
    return -slope / 2.0


a = np.zeros(J)
a[0] = 1.0
a[1] = 0.5                                   # the front starts at the largest scales; everything else exactly zero
E0 = 0.5 * np.sum(a**2)
dmin = 4.0 / k[-1]
print("dyadic model, J=%d shells (k_max = 2^%d), nu=%g, CFL %.3f; reliability rule delta > 4/k_max = %.1e" % (J, J - 1, NU, CFL, dmin))
print("   t        E/E0 - 1        Z = sum k^2 a^2     front shell   delta(t)     div F (Liouville check)")
t, t0 = 0.0, time.time()
ts, ds, Zs = [], [], []
next_print = 0.0
divF_stats = []
while True:
    dt = CFL / (np.max(k[1:] * np.abs(a[:-1])) + np.max(k * np.abs(a)) + 1e-300)
    a = step(a, dt)
    t += dt
    E = 0.5 * np.sum(a**2)
    Z = np.sum(k**2 * a**2)
    d = delta_of(a)
    front = int(np.argmax(k**2 * a**2))
    divF = -np.sum(k[1:] * a[1:]) - NU * np.sum(k**2)
    if t >= next_print - 1e-12:
        print("%8.5f   %+.2e     %12.4e         %2d        %.3e     %+.3e" % (t, E / E0 - 1, Z, front, d, divF), flush=True)
        next_print += 0.05
    if d > dmin and front >= 3:
        ts.append(t); ds.append(d); Zs.append(Z)
    if front >= J - 3 or not np.isfinite(Z) or t > 20:
        print("%8.5f   %+.2e     %12.4e         %2d        %.3e     %+.3e   <-- front at the truncation: stop (delta %s 4/k_max)" % (t, E / E0 - 1, Z, front, d, divF, ">" if d > dmin else "<"), flush=True)
        break
ts, ds, Zs = np.array(ts), np.array(ds), np.array(Zs)
print("energy drift over the run: %.1e   (%.1fs)%s" % (abs(E / E0 - 1), time.time() - t0, "  (nu > 0: physical dissipation, not drift)" if NU > 0 else ""))
n = len(ts)
if n >= 8:
    h = ts > ts[0] + 0.5 * (ts[-1] - ts[0])
    pe = np.polyfit(ts, np.log(ds), 1); res_e = np.sqrt(np.mean((np.log(ds) - np.polyval(pe, ts)) ** 2))
    pl = np.polyfit(ts, ds, 1); res_l = np.sqrt(np.mean((ds - np.polyval(pl, ts)) ** 2)) / np.mean(ds)
    pl2 = np.polyfit(ts[h], ds[h], 1); pe2 = np.polyfit(ts[h], np.log(ds[h]), 1)
    rate = -np.gradient(np.log(ds), ts)
    print("reliable window: t in [%.4f, %.4f] (%d samples)" % (ts[0], ts[-1], n))
    print("exponential fit  delta = %.3g exp(-t/%.4f)   relative rms residual %.4f" % (np.exp(pe[1]), -1 / pe[0], res_e))
    print("linear fit       delta = %.3g (%.4f - t)     relative rms residual %.4f   -> t* = %.4f" % (-pl[0], -pl[1] / pl[0], res_l, -pl[1] / pl[0]))
    print("second half only: exponential tau = %.4f;  linear t* = %.4f" % (-1 / pe2[0], -pl2[1] / pl2[0]))
    print("local decay rate -d log(delta)/dt at start / middle / end of window: %.3f / %.3f / %.3f  (rising = approaching a singularity)" % (rate[0], rate[n // 2], rate[-1]))
    # the blow-up scaling: Z ~ (t* - t)^-p ; fit p on the second half of the window using the linear t*
    tstar = -pl2[1] / pl2[0]
    sel = h & (tstar - ts > 2e-3)      # before the front reaches the truncation
    if sel.sum() > 4:
        p = -np.polyfit(np.log(tstar - ts[sel]), np.log(Zs[sel]), 1)[0]
        print("enstrophy growth Z ~ (t* - t)^-%.2f with t* = %.4f from the linear fit" % (p, tstar))
    if pl[0] < 0 and rate[-1] > 10 * rate[0]:
        print("VERDICT: the decay rate of delta rises %.0fx across the window (exponential decay would keep it constant): singularity at t* = %.4f (second-half linear fit)%s" % (rate[-1] / rate[0], tstar, " - a theorem for the inviscid model" if NU == 0 else ""))
    elif rate[-1] < 0:
        print("VERDICT: delta is growing again at the end of the window: the front has stalled, no singularity")
    else:
        print("VERDICT: exponential decay fits better: no singularity indicated inside the window")
np.savez("dyadic_run.npz", ts=ts, ds=ds, Zs=Zs)
