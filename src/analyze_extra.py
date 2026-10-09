"""Supplementary (exploratory, NOT pre-registered) analyses on top of results/{tag}_trajectories.csv.

Added after seeing the main results, to probe two things the planned analysis could not separate:
  (1) H1 at within-stage resolution: how does an old task's held-out loss grow during ONE interfering
      stage (10 points, every 25 steps)? Candidates: linear, power, saturating exponential, logarithmic.
  (2) Is the power-law shape of the *mean* curve also there in individual trajectories, or is it an
      averaging artefact? (2-parameter models only, so parameter count is equal.)
  (3) Replicate-block sensitivity of the H2 ratio (seed 0 block vs seed 1 block).

Usage: python src/analyze_extra.py --tag main
"""
import argparse
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from analyze import CATEGORY, COLORS, TASKS, category_ratio, fit_models

WITHIN = {  # name: (function of x = steps into the interfering stage / 250, p0, bounds)
    "linear": (lambda x, a: a * x, [1.0], ([-50], [50])),
    "power": (lambda x, a, b: a * np.power(x, b), [1.0, 0.5], ([-50, 1e-3], [50, 5])),
    "saturating exp": (lambda x, c, k: c * (1 - np.exp(-x / k)), [1.0, 0.2], ([-50, 1e-3], [50, 1e3])),
    "logarithmic": (lambda x, a, k: a * np.log1p(x / k), [1.0, 0.1], ([-50, 1e-4], [50, 1e3])),
}


def fit_within(x, y):
    out, n = {}, len(x)
    for name, (f, p0, bounds) in WITHIN.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p, _ = curve_fit(f, x, y, p0=p0, bounds=bounds, maxfev=20000)
            rss = float(np.sum((y - f(x, *p)) ** 2))
        except Exception:
            p, rss = [np.nan] * len(p0), np.inf
        k = len(p0) + 1
        out[name] = {"params": [float(v) for v in p], "rss": rss,
                     "aicc": float(n * np.log(max(rss, 1e-12) / n) + 2 * k + 2 * k * (k + 1) / max(n - k - 1, 1))}
    return out


def within_stage(df, tag, spt=250):
    """Loss increase of an old task during its FIRST interfering stage, as a function of steps into that stage."""
    d = df[(df.t <= spt) & (df.position < df.position.max() + 1)]
    rows = []
    for (run, task), sub in d.groupby(["run", "task"]):
        s = sub.set_index("t")["loss"].sort_index()
        if spt not in s.index:
            continue  # last-position task: nothing trained after it
        for t in s.index:
            rows.append({"run": run, "task": task, "x": t / spt, "dloss": s[t] - s[0],
                         "norm": (s[t] - s[0]) / (s[spt] - s[0]) if abs(s[spt] - s[0]) > 0.02 else np.nan})
    w = pd.DataFrame(rows)
    res = {"n_trajectories": int(w.groupby(["run", "task"]).ngroups)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    # mean curves (all tasks, and per task)
    for name, sub in [("all tasks", w)] + [(t, w[w.task == t]) for t in TASKS]:
        m = sub.groupby("x").dloss.mean()
        x, y = m.index.values[1:], m.values[1:]  # x=0 is 0 by construction for every model
        fits = fit_within(x, y)
        best = min(fits, key=lambda k: fits[k]["aicc"])
        res[name] = {"best": best, "daicc": {k: round(v["aicc"] - fits[best]["aicc"], 2) for k, v in fits.items()},
                     "power_exponent": fits["power"]["params"][1], "loss_increase_at_stage_end": float(y[-1])}
        if name == "all tasks":
            axes[0].plot(m.index, m.values, "ko", ms=4, label="mean over trajectories")
            xs = np.linspace(0, 1, 200)
            for k, ls in zip(WITHIN, ["--", "-", "-.", ":"]):
                axes[0].plot(xs, WITHIN[k][0](xs, *fits[k]["params"]), ls, label=f"{k} (dAICc={res[name]['daicc'][k]})")
        else:
            axes[1].plot(m.index * spt, m.values, color=COLORS[name], ls="-" if CATEGORY[name] == "reasoning" else "--", label=name)
    axes[0].set_xlabel("fraction of the first interfering stage (250 steps)"); axes[0].set_ylabel("increase in held-out loss (nats/token)")
    axes[0].set_title("Within-stage forgetting, all tasks"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("steps into the first interfering stage"); axes[1].set_ylabel("increase in held-out loss")
    axes[1].set_title("Per task (solid = reasoning, dashed = pattern)"); axes[1].legend(fontsize=8)
    fig.suptitle(f"[{tag}] Exploratory: forgetting within one interfering stage")
    fig.tight_layout(); fig.savefig(f"figures/{tag}_within_stage.png", dpi=140); plt.close(fig)
    # individual trajectories
    wins = {k: 0 for k in WITHIN}
    expo = []
    for (run, task), sub in w.groupby(["run", "task"]):
        sub = sub.sort_values("x")
        if abs(sub.dloss.iloc[-1]) <= 0.02 and sub.dloss.abs().max() <= 0.02:
            continue  # nothing happened to this task in this stage
        fits = fit_within(sub.x.values[1:], sub.dloss.values[1:])
        wins[min(fits, key=lambda k: fits[k]["aicc"])] += 1
        expo.append(fits["power"]["params"][1])
    res["per_trajectory_wins"] = wins
    res["per_trajectory_power_exponent_quartiles"] = [float(v) for v in np.percentile(expo, [25, 50, 75])]
    res["per_trajectory_share_sublinear(b<1)"] = float(np.mean(np.array(expo) < 1))
    return res


def two_param_only(df):
    """Per-trajectory power law vs exp+asymptote (equal parameter count) on R_loss over the stage-lag axis."""
    out = {"power": 0, "exp+asymptote": 0, "undecided(|dAICc|<2)": 0}
    for (run, task), sub in df[df.position <= 3].groupby(["run", "task"]):
        sub = sub.sort_values("lag")
        f = fit_models(sub.lag.values, sub.R_loss.values)
        d = f["power"][2] - f["exp+asymptote"][2]
        out["undecided(|dAICc|<2)" if abs(d) < 2 else ("power" if d < 0 else "exp+asymptote")] += 1
    return out


def block_sensitivity(df, spt=250):
    cells = df[(df.t % spt == 0) & (df.lag >= 1)]
    out = {}
    for seed, sub in cells.groupby("seed"):
        out[f"seed_block_{seed}"] = {}
        for col in ["R_loss", "R_loss_clip", "R_rel", "abs_drop"]:
            r, p = category_ratio(sub, col)
            out[f"seed_block_{seed}"][col] = {"reasoning": float(r), "pattern": float(p), "ratio": float(r / p)}
    # leave-one-task-out on the pre-registered primary measure
    out["leave_one_task_out_R_loss_ratio"] = {}
    for t in TASKS:
        r, p = category_ratio(cells[cells.task != t], "R_loss")
        out["leave_one_task_out_R_loss_ratio"][t] = float(r / p)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    df = pd.read_csv(f"results/{a.tag}_trajectories.csv")
    res = {"within_stage": within_stage(df, a.tag), "per_trajectory_2param": two_param_only(df),
           "h2_sensitivity": block_sensitivity(df)}
    with open(f"results/extra_{a.tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
