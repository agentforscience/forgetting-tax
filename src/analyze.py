"""Analysis for the Forgetting Tax study (pre-registered in planning.md).

Reads results/runs/{tag}_*.jsonl and writes tables (results/*.csv, results/summary_{tag}.json)
and figures (figures/*.png).

  H1  shape of the retention curve: linear vs exponential vs power law
  H2  reasoning vs pattern-matching forgetting rate (ratio, three normalisations)
  H3  serial-position effect beyond lag

Usage: python src/analyze.py --tag main
"""
import argparse
import glob
import itertools
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

TASKS = ["gsm8k", "arith", "logic", "mt", "summ", "emotion"]
CATEGORY = {"gsm8k": "reasoning", "arith": "reasoning", "logic": "reasoning",
            "mt": "pattern", "summ": "pattern", "emotion": "pattern"}
COLORS = {"gsm8k": "#b2182b", "arith": "#ef8a62", "logic": "#7f3b08",
          "mt": "#2166ac", "summ": "#67a9cf", "emotion": "#1b7837"}
MIN_GAIN = 0.05   # trajectories that learned less than this (metric units) are excluded from R_gain
N_BOOT = 10000
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------------- loading
def load_runs(tag, run_dir="results/runs"):
    runs = []
    for path in sorted(glob.glob(f"{run_dir}/{tag}_*.jsonl")):
        if not os.path.exists(path + ".done"):
            continue  # incomplete run
        lines = [json.loads(l) for l in open(path)]
        cfg = lines[0]["config"]
        recs = [l for l in lines if "step" in l]
        runs.append({"name": cfg["name"], "cfg": cfg, "recs": recs, "joint": cfg["joint"]})
    return runs


def trajectories(runs):
    """Long table: one row per (run, task, evaluation step at or after the task's own stage end)."""
    rows = []
    for r in runs:
        if r["joint"]:
            continue
        spt, order = r["cfg"]["steps_per_task"], r["cfg"]["order"]
        by_step = {x["step"]: x for x in r["recs"]}
        for pos, task in enumerate(order, 1):
            s_end = pos * spt
            z, star = by_step[0]["eval"][task], by_step[s_end]["eval"][task]
            for step in sorted(by_step):
                if step < s_end:
                    continue
                e = by_step[step]["eval"][task]
                row = {"run": r["name"], "seed": r["cfg"]["seed"], "task": task, "category": CATEGORY[task],
                       "position": pos, "t": step - s_end, "lag": (step - s_end) / spt,
                       "interferer": None if step == s_end else order[min((step - 1) // spt, len(order) - 1)],
                       "loss": e["loss"], "loss0": z["loss"], "loss_star": star["loss"],
                       "R_loss": (z["loss"] - e["loss"]) / (z["loss"] - star["loss"]),
                       # exploratory robustness measure (added after seeing that loss blow-ups far beyond the
                       # zero-shot level on one task dominate the unclipped mean): retention clipped to [0, 1]
                       "R_loss_clip": float(np.clip((z["loss"] - e["loss"]) / (z["loss"] - star["loss"]), 0, 1)),
                       "metric0": z["metric"], "metric_star": star["metric"], "gain": star["metric"] - z["metric"]}
                if "metric" in e:
                    row["metric"] = e["metric"]
                    row["R_rel"] = e["metric"] / star["metric"] if star["metric"] > 0 else np.nan
                    row["R_gain"] = ((e["metric"] - z["metric"]) / (star["metric"] - z["metric"])
                                     if star["metric"] - z["metric"] >= MIN_GAIN else np.nan)
                    row["abs_drop"] = star["metric"] - e["metric"]
                rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- H1: curve shape
def f_lin(x, a): return 1 - a * x
def f_exp(x, tau): return np.exp(-x / tau)
def f_pow(x, tau, b): return (1 + x / tau) ** (-b)
def f_expc(x, tau, c): return c + (1 - c) * np.exp(-x / tau)

MODELS = {  # name: (function, initial guess, bounds)
    "linear": (f_lin, [0.1], ([-10], [10])),
    "exponential": (f_exp, [5.0], ([1e-3], [1e4])),
    "power": (f_pow, [0.5, 0.5], ([1e-4, 1e-4], [1e4, 50])),
    "exp+asymptote": (f_expc, [0.5, 0.5], ([1e-3, -20], [1e4, 1.0])),
}


def fit_models(x, y):
    """Least-squares fit of each candidate; returns {model: (params, rss, AICc)}. x in units of stages."""
    out, n = {}, len(x)
    for name, (f, p0, bounds) in MODELS.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p, _ = curve_fit(f, x, y, p0=p0, bounds=bounds, maxfev=20000)
            rss = float(np.sum((y - f(x, *p)) ** 2))
        except Exception:
            p, rss = [np.nan] * len(p0), np.inf
        k = len(p0) + 1  # + residual variance
        aicc = n * np.log(max(rss, 1e-12) / n) + 2 * k + 2 * k * (k + 1) / max(n - k - 1, 1)
        out[name] = (list(map(float, p)), rss, float(aicc))
    return out


def h1_shape(df, tag, measure="R_loss"):
    res = {}
    d = df[df[measure].notna()]
    # (a) grand-mean and category-mean curves from position-1 trajectories (full 5-stage lag range,
    #     interfering tasks balanced by the Williams design)
    p1 = d[d.position == 1]
    curves = {"all tasks": p1, "reasoning": p1[p1.category == "reasoning"], "pattern": p1[p1.category == "pattern"]}
    curves.update({t: p1[p1.task == t] for t in TASKS})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    for name, sub in curves.items():
        m = sub.groupby("lag")[measure].mean()
        x, y = m.index.values, m.values
        if len(x) < 5:  # no (or too few) position-1 trajectories for this subset
            continue
        fits = fit_models(x, y)
        best = min(fits, key=lambda k: fits[k][2])
        res[name] = {"n_traj": int(sub.groupby(["run", "task"]).ngroups), "n_points": len(x), "best": best,
                     "aicc": {k: v[2] for k, v in fits.items()}, "params": {k: v[0] for k, v in fits.items()},
                     "rmse": {k: float(np.sqrt(v[1] / len(x))) for k, v in fits.items()},
                     "r2": {k: float(1 - v[1] / np.sum((y - y.mean()) ** 2)) for k, v in fits.items()}}
        if name in ("all tasks", "reasoning", "pattern"):
            ax = axes[["all tasks", "reasoning", "pattern"].index(name)]
            ax.plot(x, y, "k.", ms=5, label="mean retention (position-1 tasks)")
            xs = np.linspace(0, x.max(), 300)
            for k, ls in zip(MODELS, ["--", "-.", "-", ":"]):
                ax.plot(xs, MODELS[k][0](xs, *fits[k][0]), ls, lw=1.6,
                        label=f"{k} (dAICc={fits[k][2] - fits[best][2]:.1f})")
            ax.set_title(f"{name} (n={res[name]['n_traj']} trajectories)")
            ax.set_xlabel("subsequent tasks trained (lag, in 250-step stages)")
            ax.set_ylabel(f"retention ({measure})")
            ax.axhline(0, color="gray", lw=0.5)
            ax.legend(fontsize=7.5)
    fig.suptitle(f"[{tag}] H1: shape of the forgetting curve ({measure}; 1 = just learned, 0 = zero-shot level)")
    fig.tight_layout()
    fig.savefig(f"figures/{tag}_h1_shape_{measure}.png", dpi=140)
    plt.close(fig)

    # (b) individual trajectories with >= 3 stages of lag: which model wins?
    wins, rows = {k: 0 for k in MODELS}, []
    for (run, task), sub in d[d.position <= 3].groupby(["run", "task"]):
        sub = sub.sort_values("lag")
        if len(sub) < 5:
            continue
        fits = fit_models(sub.lag.values, sub[measure].values)
        best = min(fits, key=lambda k: fits[k][2])
        wins[best] += 1
        rows.append({"run": run, "task": task, "category": CATEGORY[task], "best": best,
                     **{f"aicc_{k}": v[2] for k, v in fits.items()}})
    tr = pd.DataFrame(rows)
    res["per_trajectory_wins"] = wins
    res["per_trajectory_wins_by_category"] = (tr.groupby(["category", "best"]).size().unstack(fill_value=0).to_dict("index")
                                              if len(tr) else {})
    res["per_trajectory_nonmonotone_share"] = None

    # (c) how "step-like" is forgetting? share of each stage's loss change that happens in the first 25 / 50 steps
    spt = int(round(d.t[d.lag == 1].iloc[0])) if (d.lag == 1).any() else 250
    onset = []
    for (run, task), sub in d.groupby(["run", "task"]):
        s = sub.set_index("t")["loss"]
        for k in range(0, int(sub.lag.max())):
            a, b = k * spt, (k + 1) * spt
            if b in s.index and a + 25 in s.index and a + 50 in s.index and abs(s[b] - s[a]) > 0.02:
                onset.append({"run": run, "task": task, "stage_lag": k + 1, "delta_stage": s[b] - s[a],
                              "share_25": (s[a + 25] - s[a]) / (s[b] - s[a]), "share_50": (s[a + 50] - s[a]) / (s[b] - s[a]),
                              "max_within": s[[i for i in s.index if a < i <= b]].max() - s[a]})
    on = pd.DataFrame(onset)
    if len(on):
        up = on[on.delta_stage > 0.02]
        res["stage_onset"] = {"n_stage_transitions_with_loss_increase": int(len(up)),
                              "median_share_of_stage_forgetting_in_first_25_steps": float(up.share_25.median()),
                              "median_share_in_first_50_steps": float(up.share_50.median()),
                              "uniform_expectation_25": 25 / spt, "uniform_expectation_50": 50 / spt}
    # monotonicity: share of stage-to-stage transitions in which the old task's loss went DOWN (recovery)
    se = d[(d.t % spt == 0)].sort_values(["run", "task", "t"])
    se = se.assign(prev=se.groupby(["run", "task"])["loss"].shift(1))
    tr2 = se[se.prev.notna()]
    res["share_of_stage_transitions_with_recovery"] = float((tr2.loss < tr2.prev - 0.01).mean()) if len(tr2) else None
    res.pop("per_trajectory_nonmonotone_share")
    return res


# ----------------------------------------------------------------------------- H2: domain ratio
def category_ratio(cells, col):
    """Reasoning/pattern ratio of forgetting (1 - retention), tasks weighted equally within category."""
    per_task = (1 - cells.groupby("task")[col].mean()) if col != "abs_drop" else cells.groupby("task")[col].mean()
    r = np.mean([per_task[t] for t in TASKS if CATEGORY[t] == "reasoning" and t in per_task.index and not np.isnan(per_task[t])])
    p = np.mean([per_task[t] for t in TASKS if CATEGORY[t] == "pattern" and t in per_task.index and not np.isnan(per_task[t])])
    return r, p


def h2_domain(df, tag):
    spt = df.t[df.lag == 1].iloc[0]
    cells = df[(df.t % spt == 0) & (df.lag >= 1)]  # stage-end evaluations after >= 1 subsequent task
    res = {"per_task": {}, "ratios": {}}
    runs = cells.run.unique()
    by_run = {r: cells[cells.run == r] for r in runs}
    for col in ["R_loss", "R_loss_clip", "R_gain", "R_rel", "abs_drop"]:
        r, p = category_ratio(cells, col)
        boots = []
        for _ in range(N_BOOT if len(runs) > 1 else 0):  # cluster bootstrap over runs
            b = pd.concat([by_run[x] for x in RNG.choice(runs, len(runs))])
            boots.append(category_ratio(b, col))
        boots = np.array(boots) if boots else np.zeros((0, 2))
        ratio_b = boots[:, 0] / boots[:, 1] if len(boots) else np.array([np.nan])
        diff_b = boots[:, 0] - boots[:, 1] if len(boots) else np.array([np.nan])
        res["ratios"][col] = {
            "forgetting_reasoning": float(r), "forgetting_pattern": float(p), "ratio": float(r / p) if p != 0 else None,
            "ratio_ci95": [float(np.nanpercentile(ratio_b, 2.5)), float(np.nanpercentile(ratio_b, 97.5))],
            "diff": float(r - p), "diff_ci95": [float(np.nanpercentile(diff_b, 2.5)), float(np.nanpercentile(diff_b, 97.5))],
            "boot_share_pattern_forgetting_le_0": float((boots[:, 1] <= 0).mean()) if len(boots) else None,
            "boot_share_ratio_in_3_5": float(((ratio_b >= 3) & (ratio_b <= 5)).mean()) if len(boots) else None,
            "boot_p_diff_le_0": float((diff_b <= 0).mean()) if len(boots) else None}
        # exact permutation test over which 3 of the 6 tasks are called "reasoning" (tasks = unit of generalisation)
        pt = (1 - cells.groupby("task")[col].mean()) if col != "abs_drop" else cells.groupby("task")[col].mean()
        pt = pt.reindex(TASKS)
        if pt.notna().all():
            obs = pt[[t for t in TASKS if CATEGORY[t] == "reasoning"]].mean() - pt[[t for t in TASKS if CATEGORY[t] == "pattern"]].mean()
            diffs = [pt[list(c)].mean() - pt[[t for t in TASKS if t not in c]].mean() for c in itertools.combinations(TASKS, 3)]
            res["ratios"][col]["task_permutation_p_two_sided"] = float(np.mean(np.abs(diffs) >= abs(obs) - 1e-12))
    g = cells.groupby("task")
    tab = pd.DataFrame({"category": pd.Series(CATEGORY), "zero_shot": g.metric0.mean(), "after_own_stage": g.metric_star.mean(),
                        "gain": g.gain.mean(), "mean_metric_later": g.metric.mean(), "abs_drop": g.abs_drop.mean(),
                        "forget_rel": 1 - g.R_rel.mean(), "forget_gain": 1 - g.R_gain.mean(),
                        "n_gain_excluded": g.R_gain.apply(lambda s: int(s.isna().sum())), "n_cells": g.size(),
                        "forget_loss": 1 - g.R_loss.mean(), "forget_loss_clip": 1 - g.R_loss_clip.mean(),
                        "share_cells_below_zero_shot_loss": g.R_loss.apply(lambda s: float((s < 0).mean())),
                        "forget_loss_lag1": 1 - cells[cells.lag == 1].groupby("task").R_loss.mean(),
                        "forget_loss_sd_over_cells": g.R_loss.std()}).reindex(TASKS)
    tab.to_csv(f"results/{tag}_per_task_forgetting.csv")
    res["per_task"] = json.loads(tab.to_json(orient="index"))

    # figures: retention vs lag per task (loss-based and metric-based)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for ax, col, ttl in zip(axes, ["R_loss", "R_rel", "R_gain"],
                            ["loss-based retention (dense, every 25 steps)", "relative metric retention m(t)/m*",
                             "gain-normalised metric retention"]):
        dd = df[df[col].notna()]
        for t in TASKS:
            sub = dd[dd.task == t]
            if not len(sub):
                continue
            m = sub.groupby("lag")[col].agg(["mean", "sem"])
            ax.plot(m.index, m["mean"], color=COLORS[t], lw=1.8, ls="-" if CATEGORY[t] == "reasoning" else "--",
                    marker=None if col == "R_loss" else "o", ms=3, label=f"{t} ({CATEGORY[t]})")
            ax.fill_between(m.index, m["mean"] - m["sem"].fillna(0), m["mean"] + m["sem"].fillna(0), color=COLORS[t], alpha=0.12)
        ax.axhline(1, color="gray", lw=0.5); ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("subsequent tasks trained (lag)"); ax.set_ylabel("retention"); ax.set_title(ttl)
        if col == "R_loss":
            lo = dd.groupby(["task", "lag"])[col].mean().min()
            ax.set_ylim(max(lo - 0.3, -6), 1.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"[{tag}] Retention by task, averaged over orders/positions (solid = reasoning, dashed = pattern; band = s.e.m.)")
    fig.tight_layout()
    fig.savefig(f"figures/{tag}_retention_by_task.png", dpi=140)
    plt.close(fig)
    return res


# ----------------------------------------------------------------------------- H3: position
def h3_position(df, tag):
    spt = df.t[df.lag == 1].iloc[0]
    cells = df[(df.t % spt == 0) & (df.lag >= 1)].copy()
    cells["loglag"] = np.log1p(cells.lag)
    res = {}
    for col in ["R_loss", "R_loss_clip", "R_rel", "R_gain"]:
        c = cells[cells[col].notna()].copy()
        c["F"] = 1 - c[col]
        out = {}
        # (a) forgetting after exactly ONE subsequent task, by position at which the task was learned
        l1 = c[c.lag == 1]
        out["lag1_forgetting_by_position"] = {int(k): float(v) for k, v in l1.groupby("position").F.mean().items()}
        rho, p = spearmanr(l1.position, l1.F)
        out["lag1_spearman_position_vs_forgetting"] = {"rho": float(rho), "p": float(p), "n": int(len(l1))}
        # task-demeaned version (removes task difficulty differences)
        dm = l1.F - l1.groupby("task").F.transform("mean")
        rho, p = spearmanr(l1.position, dm)
        out["lag1_spearman_task_demeaned"] = {"rho": float(rho), "p": float(p)}
        # (b) regression with lag and task controlled, errors clustered by run
        try:
            m = smf.ols("F ~ loglag + position + C(task)", c).fit(cov_type="cluster", cov_kwds={"groups": c.run})
            out["ols_F~loglag+position+task"] = {k: {"coef": float(m.params[k]), "se": float(m.bse[k]), "p": float(m.pvalues[k]),
                                                     "ci95": [float(x) for x in m.conf_int().loc[k]]} for k in ["loglag", "position"]}
            out["ols_n"] = int(m.nobs)
        except Exception as e:
            out["ols_error"] = repr(e)
        # (c) end-of-sequence retention by position (what a practitioner sees; confounded with lag by design)
        fin = df[(df.t + df.position * spt == df.position.max() * spt) & df[col].notna()]
        out["final_retention_by_position"] = {int(k): float(v) for k, v in fin.groupby("position")[col].mean().items()}
        res[col] = out

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    c = cells.copy(); c["F"] = 1 - c.R_loss
    for lag, sub in c.groupby("lag"):
        m = sub.groupby("position").F.agg(["mean", "sem"])
        axes[0].errorbar(m.index, m["mean"], yerr=m["sem"], marker="o", capsize=2, label=f"lag {int(lag)}")
    axes[0].set_xlabel("serial position at which the task was learned"); axes[0].set_ylabel("forgetting (1 - R_loss)")
    axes[0].set_title("Forgetting at fixed lag vs position\n(H3 predicts lines sloping down to the right)"); axes[0].legend(fontsize=8)
    for col, mk in [("R_loss", "o"), ("R_rel", "s"), ("R_gain", "^")]:
        v = res[col]["final_retention_by_position"]
        axes[1].plot(list(v), list(v.values()), marker=mk, label=col)
    axes[1].axhline(1, color="gray", lw=0.5); axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_xlabel("serial position"); axes[1].set_ylabel("retention at end of sequence")
    axes[1].set_title("End-of-sequence retention by position\n(position and lag confounded)"); axes[1].legend(fontsize=8)
    piv = c.pivot_table(index="position", columns="lag", values="F", aggfunc="mean")
    im = axes[2].imshow(piv.values, cmap="Reds", aspect="auto", origin="lower")
    axes[2].set_xticks(range(piv.shape[1])); axes[2].set_xticklabels([int(x) for x in piv.columns])
    axes[2].set_yticks(range(piv.shape[0])); axes[2].set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            if not np.isnan(piv.values[i, j]):
                axes[2].text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    axes[2].set_xlabel("lag (subsequent tasks)"); axes[2].set_ylabel("serial position"); axes[2].set_title("Mean forgetting (1 - R_loss)")
    fig.colorbar(im, ax=axes[2])
    fig.suptitle(f"[{tag}] H3: serial-position effect")
    fig.tight_layout()
    fig.savefig(f"figures/{tag}_h3_position.png", dpi=140)
    plt.close(fig)
    return res


# ----------------------------------------------------------------------------- baselines, interference, errors
def baselines(runs, df, joint_runs, tag):
    seq = [r for r in runs if not r["joint"]]
    rows = {}
    for t in TASKS:
        z = [r["recs"][0]["eval"][t]["metric"] for r in seq]
        fin = [r["recs"][-1]["eval"][t]["metric"] for r in seq]
        own = df[(df.task == t) & (df.t == 0)].metric
        rows[t] = {"zero_shot": np.mean(z), "single_task_end_of_own_stage": own.mean(), "own_stage_sd": own.std(),
                   "sequential_final": np.mean(fin), "sequential_final_sd": np.std(fin, ddof=1) if len(fin) > 1 else np.nan,
                   "joint_multitask": np.mean([r["recs"][-1]["eval"][t]["metric"] for r in joint_runs]) if joint_runs else np.nan}
    tab = pd.DataFrame(rows).T
    tab.loc["MEAN"] = tab.mean()
    tab.to_csv(f"results/{tag}_baselines.csv")
    return json.loads(tab.to_json(orient="index"))


def interference_matrix(df, tag):
    """Mean change in the old task's held-out loss caused by one stage of training on each new task."""
    spt = df.t[df.lag == 1].iloc[0]
    se = df[df.t % spt == 0].sort_values(["run", "task", "t"]).copy()
    se["dloss"] = se.groupby(["run", "task"]).loss.diff()
    se["dR"] = -se.groupby(["run", "task"]).R_loss.diff()
    se = se[se.interferer.notna()]
    piv = se.pivot_table(index="task", columns="interferer", values="dR", aggfunc="mean").reindex(index=TASKS, columns=TASKS)
    piv.to_csv(f"results/{tag}_interference_matrix.csv")
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    v = np.nanmax(np.abs(piv.values))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_xticks(range(6)); ax.set_xticklabels(TASKS, rotation=30); ax.set_yticks(range(6)); ax.set_yticklabels(TASKS)
    for i in range(6):
        for j in range(6):
            if not np.isnan(piv.values[i, j]):
                ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("task being trained (interferer)"); ax.set_ylabel("previously learned task")
    ax.set_title(f"[{tag}] Forgetting caused by one stage\n(loss-based retention lost; red = forgetting, blue = recovery)")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(f"figures/{tag}_interference_matrix.png", dpi=140); plt.close(fig)
    same = [piv.loc[a, b] for a in TASKS for b in TASKS if a != b and CATEGORY[a] == CATEGORY[b] and not np.isnan(piv.loc[a, b])]
    cross = [piv.loc[a, b] for a in TASKS for b in TASKS if CATEGORY[a] != CATEGORY[b] and not np.isnan(piv.loc[a, b])]
    return {"mean_forgetting_caused_by_interferer": {k: float(x) for k, x in piv.mean(0).items()},
            "mean_forgetting_suffered_by_task": {k: float(x) for k, x in piv.mean(1).items()},
            "same_category_pairs_mean": float(np.mean(same)), "cross_category_pairs_mean": float(np.mean(cross))}


def format_valid(task, pred, ex_prompt):
    """Does the output have the form the task asks for (regardless of correctness)?"""
    p = pred.strip()
    if task in ("gsm8k", "arith"):
        return "Answer:" in p
    if task == "logic":
        return p.lower().strip(". ") in ("true", "false", "unknown")
    if task == "emotion":
        return p.lower().strip(". ") in ("sadness", "joy", "love", "anger", "fear", "surprise")
    src = ex_prompt.split("\n\n", 1)[1]
    if task == "mt":  # invalid if (mostly) an echo of the English source or empty
        a, b = set(p.lower().split()), set(src.lower().split())
        return len(p) > 0 and len(a & b) / max(len(a), 1) < 0.6
    if task == "summ":
        return 0 < len(p) < 0.9 * len(src) and "\n" not in p
    return True


def error_analysis(runs, tag):
    """Split forgetting into 'format/interface loss' vs 'content loss' using the raw generations saved at stage ends."""
    tests = {t: [json.loads(l) for l in open(f"datasets/processed/{t}_test.jsonl")] for t in TASKS}
    rows, examples = [], []
    for r in runs:
        if r["joint"]:
            continue
        spt, order = r["cfg"]["steps_per_task"], r["cfg"]["order"]
        by_step = {x["step"]: x for x in r["recs"]}
        for pos, task in enumerate(order, 1):
            for step in range(pos * spt, len(order) * spt + 1, spt):
                e = by_step[step]["eval"][task]
                if "preds" not in e:
                    continue
                valid = np.array([format_valid(task, p, x["prompt"]) for p, x in zip(e["preds"], tests[task])])
                sc = np.array(e["scores"])
                rows.append({"run": r["name"], "task": task, "position": pos, "lag": (step - pos * spt) // spt,
                             "trained_last": order[step // spt - 1], "format_valid": valid.mean(), "metric": sc.mean(),
                             "metric_given_valid": sc[valid].mean() if valid.any() else np.nan})
                if step == len(order) * spt and pos <= 2 and len(examples) < 12:
                    examples.append({"run": r["name"], "task": task, "learned_at_position": pos, "last_task_trained": order[-1],
                                     "prompt_tail": tests[task][0]["prompt"][-160:], "reference": tests[task][0]["answer"][:160],
                                     "output_right_after_learning": by_step[pos * spt]["eval"][task]["preds"][0][:300],
                                     "output_at_end_of_sequence": e["preds"][0][:300]})
    d = pd.DataFrame(rows)
    d.to_csv(f"results/{tag}_format_analysis.csv", index=False)
    g0, g = d[d.lag == 0].groupby("task"), d[d.lag >= 1].groupby("task")
    tab = pd.DataFrame({"format_valid_just_learned": g0.format_valid.mean(), "format_valid_later": g.format_valid.mean(),
                        "metric_just_learned": g0.metric.mean(), "metric_later": g.metric.mean(),
                        "metric_given_valid_just_learned": g0.metric_given_valid.mean(),
                        "metric_given_valid_later": g.metric_given_valid.mean()}).reindex(TASKS)
    tab.to_csv(f"results/{tag}_format_vs_content.csv")
    with open(f"results/{tag}_example_outputs.json", "w") as f:
        json.dump(examples, f, indent=1, ensure_ascii=False)
    return json.loads(tab.to_json(orient="index"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--joint_tag", default="joint")
    a = ap.parse_args()
    os.makedirs("figures", exist_ok=True)
    runs = load_runs(a.tag)
    joint = load_runs(a.joint_tag)
    assert runs, f"no finished runs for tag {a.tag}"
    df = trajectories(runs)
    df.to_csv(f"results/{a.tag}_trajectories.csv", index=False)
    summary = {"tag": a.tag, "n_runs": len(runs), "runs": [r["name"] for r in runs], "n_joint_runs": len(joint),
               "model": runs[0]["cfg"]["model"],
               "run_minutes_mean": float(np.mean([json.loads(open(f'results/runs/{r["name"]}.jsonl').readlines()[-1])["train_eval_seconds"] for r in runs]) / 60),
               "baselines": baselines(runs, df, joint, a.tag),
               "H1_shape_loss": h1_shape(df, a.tag, "R_loss"),
               "H1_shape_rel_metric": h1_shape(df, a.tag, "R_rel"),
               "H2_domain": h2_domain(df, a.tag),
               "H3_position": h3_position(df, a.tag),
               "interference": interference_matrix(df, a.tag),
               "format_vs_content": error_analysis(runs, a.tag)}
    with open(f"results/summary_{a.tag}.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ["n_runs", "baselines", "H2_domain", "H3_position"]}, indent=1)[:6000])


if __name__ == "__main__":
    main()
