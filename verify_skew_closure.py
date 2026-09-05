"""Verification of the claim in EQUATIONS.md: a learned correction projected to be skew cannot change the energy,
whatever its weights. Uses an UNTRAINED network (random weights), because the property is the projection's, not
training's. Compares the energy injected per step by the projected and unprojected corrections."""
import math, torch, torch.nn.functional as F, importlib.util, sys, os

os.environ.update(SPIN="1", SNAP="3", STEPS="0", ROLL="1")
# load the machinery from kolmogorov_v2 without running its training/rollout (guarded by STEPS=0 and tiny data)
spec = importlib.util.spec_from_file_location("kv2", "kolmogorov_v2.py")
src = open("kolmogorov_v2.py", encoding="utf-8").read()
src = src[: src.index("arms = {")]  # everything up to the training/rollout section
kv2 = {}
exec(compile(src, "kolmogorov_v2.py", "exec"), kv2)
torch.manual_seed(0)
w = kv2["Wtr"][:2].clone()
model = kv2["FrozenSkewGate"]()
out = model.f(w)
corr_raw = out[:, 1]
corr_skew = kv2["skew_project"](out[:, 1], w)
g = kv2["energy_grad"](w)
dE_raw = (corr_raw * g).mean(dim=(-2, -1))
dE_skew = (corr_skew * g).mean(dim=(-2, -1))
print("energy change per unit correction, unprojected: %s" % [float("%.3e" % v) for v in dE_raw])
print("energy change per unit correction, skew-projected: %s" % [float("%.3e" % v) for v in dE_skew])
# and through an actual step: energy of the frozen step with and without the (scaled) correction added
E = kv2["energy"]
w_frozen = kv2["frozen_step"](w)
w_corr = w_frozen + kv2["DT"] * 0.5 * corr_skew
w_corr_raw = w_frozen + kv2["DT"] * 0.5 * corr_raw
print("relative energy difference after one step, skew correction vs none: %s" % [float("%.3e" % v) for v in ((E(w_corr) - E(w_frozen)) / E(w_frozen))])
print("relative energy difference after one step, raw  correction vs none: %s" % [float("%.3e" % v) for v in ((E(w_corr_raw) - E(w_frozen)) / E(w_frozen))])
