"""Analysis of the step-level onset experiment (results/onset/*.json, produced by src/run_onset.py).

For each (old task <- new task) pair: increase of the old task's held-out loss as a function of the
number of steps on the new task, compared with how fast the new task itself is learned, and a model
comparison (linear / power / saturating exponential / logarithmic) on the first 100 steps.
Exploratory, not pre-registered.
"""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_extra import WITHIN, fit_within

recs = [json.load(open(p)) for p in sorted(glob.glob("results/onset/onset_*.json"))]
assert recs, "no onset results"
fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
out = {}
for r in recs:
    name = f"{r['old']} <- {r['new']}"
    s = np.array([p["step"] for p in r["points"]], float)
    old = np.array([p["old"] for p in r["points"]])
    new = np.array([p["new"] for p in r["points"]])
    d_old = old - old[0]                      # forgetting: loss increase on the old task
    learned = (new[0] - new) / (new[0] - new[-1])  # share of the new task's 100-step loss drop achieved
    total = d_old[-1]
    fits = fit_within(s[1:] / s[-1], d_old[1:])
    best = min(fits, key=lambda k: fits[k]["aicc"])

    def step_reaching(frac, y):
        idx = np.where(y >= frac * y[-1])[0]
        return float(s[idx[0]]) if len(idx) and y[-1] > 0 else None

    out[name] = {
        "old_loss_zero_shot": r["zero_shot"][r["old"]], "old_loss_after_own_training": float(old[0]),
        "old_loss_after_100_new_steps": float(old[-1]), "loss_increase_100": float(total),
        "share_of_100step_increase_by_step": {str(k): float(d_old[s == k][0] / total) for k in (5, 10, 15, 20, 26, 50)},
        "steps_to_50pct_of_forgetting": step_reaching(0.5, d_old), "steps_to_90pct_of_forgetting": step_reaching(0.9, d_old),
        "steps_to_50pct_of_new_task_learning": step_reaching(0.5, new[0] - new),
        "steps_to_90pct_of_new_task_learning": step_reaching(0.9, new[0] - new),
        "best_model": best, "daicc": {k: round(v["aicc"] - fits[best]["aicc"], 1) for k, v in fits.items()},
        "power_exponent": fits["power"]["params"][1],
        "peak_then_recovery": float(d_old.max() / total) if total > 0 else None,
        # after the 10-step LR warm-up: is the remaining curve flat, linear, or still decelerating?
        "increase_steps_10_to_20": float(d_old[s == 20][0] - d_old[s == 10][0]),
        "increase_steps_50_to_100": float(d_old[-1] - d_old[s == 50][0]),
    }
    axes[0].plot(s, d_old, marker=".", ms=3, label=name)
    axes[1].plot(s[1:], d_old[1:] / total, marker=".", ms=3, label=name)
    axes[2].plot(learned, d_old / total, marker=".", ms=3, label=name)
axes[0].axvspan(0, 10, color="gray", alpha=0.15, label="LR warm-up (10 steps)")
axes[0].set_xlabel("optimizer steps on the new task"); axes[0].set_ylabel("increase in old-task held-out loss (nats/token)")
axes[0].set_title("Forgetting onset at step resolution"); axes[0].legend(fontsize=7.5)
axes[1].set_xscale("log"); axes[1].set_xlabel("steps on the new task (log scale)")
axes[1].set_ylabel("share of the 100-step loss increase"); axes[1].axhline(1, color="gray", lw=0.5)
axes[1].plot([1, 100], [0.01, 1], "k:", lw=1, label="uniform (linear) forgetting"); axes[1].legend(fontsize=7.5)
axes[1].set_title("Same, normalised (dotted = uniform degradation)")
axes[2].plot([0, 1], [0, 1], "k:", lw=1)
axes[2].set_xlabel("share of the new task's loss drop achieved"); axes[2].set_ylabel("share of old-task loss increase")
axes[2].set_title("Forgetting vs learning of the new task\n(diagonal = forgetting proportional to learning)")
fig.suptitle("Exploratory: forgetting in the first 100 steps after a task switch (Llama-3.2-1B, LoRA, seed 0)")
fig.tight_layout(); fig.savefig("figures/onset.png", dpi=140)
with open("results/onset_summary.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
