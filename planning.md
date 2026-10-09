# Planning: The Forgetting Tax

Written before any training run. The analysis plan in "Statistical Analysis Plan" is the
pre-registration; deviations are logged at the bottom.

## Motivation & Novelty Assessment

### Why This Research Matters
Instruction-tuned LLMs are updated sequentially in practice (new tasks, new customers, new data).
Each update silently erodes earlier skills. Practitioners budget for this with a single number
("average forgetting"), but if erosion has a characteristic *shape* over time and differs by *kind of
skill*, then replay budgets, evaluation schedules and task ordering can be planned far better.

### Gap in Existing Work
Note: the resource-finding stage was skipped for this run (`.neurico/pipeline_state.json` says
`"skipped": true`), so there is no `literature_review.md`, `papers/` or `code/`. The following is from
my own knowledge of the literature, not from a fresh review.
- Continual instruction-tuning benchmarks (TRACE, Wang et al. 2023; CITB, Zhang et al. 2023;
  Luo et al. 2023 "An Empirical Study of Catastrophic Forgetting in LLMs during Continual
  Fine-tuning"; Scialom et al. 2022) report *end-of-stream* averages (average accuracy, backward
  transfer). They rarely report the forgetting *trajectory* at sub-task resolution, and do not fit
  functional forms to it.
- Task order is usually fixed (one or two orders), so "which task type forgets more" is confounded
  with "which task came first".
- Different tasks are scored with different metrics, so cross-domain "rate" comparisons depend on an
  unstated normalisation.

### Our Novel Contribution
1. Dense forgetting trajectories (evaluation every 50 gradient steps) so decay *shape* can be tested
   by model comparison (linear vs exponential vs power law).
2. A Williams (balanced Latin-square) order design: every task appears at every position once and
   follows every other task once, which de-confounds task domain from learning position.
3. Rate comparison between reasoning and pattern-matching tasks under three explicit normalisations,
   including a metric-free one (held-out loss).

### Experiment Justification
- **Exp 1 (main, sequential LoRA, Llama-3.2-1B-Instruct, 6 orders x 2 seeds):** provides the
  trajectories all three sub-hypotheses are tested on.
- **Exp 2 (baselines: zero-shot and joint multi-task training):** zero-shot is the floor needed to
  define "learned gain"; joint training shows whether drops are forgetting or mere capacity limits.
- **Exp 3 (replication on Qwen2.5-1.5B-Instruct, 6 orders):** are conclusions model-specific?
- **Exp 4 (ablation: full fine-tuning instead of LoRA, 6 orders):** are conclusions specific to the
  low-rank adapter?

## Research Question
During sequential instruction tuning on six tasks, (H1) does retained performance on a previously
learned task decay as a power law in the amount of subsequent training, rather than linearly
("uniform degradation") or exponentially? (H2) Do reasoning tasks (math, logic) lose performance
3-5x faster than pattern-matching tasks (translation, summarisation)? (H3) Are early-learned tasks
forgotten disproportionately, i.e. more than their longer exposure to interference explains?

## Hypothesis Decomposition
- **H1 (shape).** IV: interfering gradient steps `t` since the task's own stage ended (0..1250).
  DV: retention `R(t)`. Candidates: linear `1-at` (1 param), exponential `exp(-t/tau)` (1),
  power law `(1+t/tau)^-b` (2), exponential-to-asymptote `c+(1-c)exp(-t/tau)` (2, a fair
  2-parameter competitor). Supported if the power law has the lowest AICc in a clear majority of
  task-level mean curves AND beats the 2-parameter exponential.
- **H2 (domain).** IV: task category. DV: forgetting rate. Supported if the reasoning/pattern ratio
  of mean forgetting has a bootstrap 95% CI that lies inside or overlaps [3, 5] and excludes 1,
  consistently across normalisations. A ratio >1 but well outside [3,5] is "direction right,
  magnitude wrong".
- **H3 (position).** IV: serial position 1..6. Final retention vs position is necessarily
  confounded with lag (earlier = more interference), so the "disproportionate" claim is tested by
  (a) forgetting at a *fixed* lag (after exactly one subsequent task) as a function of position, and
  (b) a regression of retention on log(1+lag), position and task with run-clustered errors.
  Supported if position has a negative effect on retention beyond lag.

## Proposed Methodology
### Approach
Real sequential fine-tuning of open instruction-tuned LLMs on local GPUs (4x RTX A6000). API models
cannot be continually fine-tuned with trajectory access, so no API models are used (and
`OPENROUTER_KEY` is not set in this environment).

### Tasks (2000 train / 200 test each; `src/prep_data.py`)
| Task | Category | Source | Metric |
|---|---|---|---|
| gsm8k | reasoning | GSM8K, worked solution + answer | exact-match accuracy |
| arith | reasoning | synthetic 3-step integer arithmetic | exact-match accuracy |
| logic | reasoning | ProofWriter OWA, depth 1-3, 3-way | accuracy |
| mt | pattern | WMT16 en->de (newstest2015 train / 2016 test) | chrF |
| summ | pattern | SAMSum | ROUGE-L F1 |
| emotion | pattern | dair-ai/emotion, 6-way | accuracy |

`emotion` is included so that the pattern category has an accuracy-scored task; otherwise category
would be perfectly confounded with metric type. The reasoning/pattern assignment is a judgment call;
results are also reported per task.

### Experimental Steps
1. Zero-shot evaluation of the base model on all six test sets.
2. Sequential training: for each order, train task 1 for 250 steps (batch 16 = 2 epochs), then task
   2, ... with a single LoRA adapter carried through (r=16, alpha=32, all attention+MLP projections,
   lr 2e-4, AdamW, **constant** LR after a 10-step warm-up per stage so that interference per step is
   not modulated by a schedule). No replay, no regularisation (naive SeqFT, the standard lower bound).
3. Every 50 steps: held-out per-token loss on all six test sets and greedy-decoded task metrics.
4. Orders: 6-order Williams design; seed 2 uses a randomly relabelled Williams square.
5. Joint baseline: 1500 steps on the mixture of all six tasks.
6. Replication on Qwen2.5-1.5B-Instruct and full-fine-tuning ablation (lr 1e-5), 6 orders each.

### Baselines
Zero-shot base model (floor), joint multi-task training (upper bound), each task's own
end-of-stage performance (per-trajectory reference).

### Evaluation Metrics
For task i in a run, with `m0` zero-shot score, `m*` the score at the end of its own stage, `m(t)`
after `t` further steps:
- **Gain-normalised retention** `R_gain = (m(t)-m0)/(m*-m0)` (fraction of learned gain kept; only
  defined where `m*-m0` is non-trivial; trajectories with gain < 0.05 are excluded and counted).
- **Relative retention** `R_rel = m(t)/m*`.
- **Loss-based retention** `R_loss = (L0-L(t))/(L0-L*)` using held-out per-token NLL: metric-free and
  comparable across tasks. This is the primary measure for H1 because it is continuous.
- Forgetting at lag k = `1 - R` after k subsequent tasks; standard final-forgetting / BWT as well.

### Statistical Analysis Plan
- H1: non-linear least squares on task-level mean curves and on individual trajectories; AICc and
  win counts; residual plots. alpha = 0.05 throughout.
- H2: ratio of category means of forgetting (averaged over lags 1-5, each task weighted equally),
  cluster bootstrap over runs (10,000 resamples), under all three normalisations; plus
  per-task table so the reader can see whether one task drives the category effect.
- H3: fixed-lag comparison with Spearman correlation and the regression above (cluster-robust by
  run); position x lag is only partially crossed, stated as a limitation.
- Multiple comparisons: Holm correction across the three normalisations within each hypothesis.
- No conclusions from individual 200-example test deltas < ~7 points (binomial SE ~3.5 points).

## Expected Outcomes
Supporting: power law wins AICc; ratio CI inside [3,5]; negative position effect beyond lag.
Refuting: exponential/linear fit as well or better; ratio near 1, reversed, or dominated by a single
task; no position effect once lag is controlled.

## Timeline and Milestones
Setup+data 30 min (done) | implementation + smoke test 40 min | main runs ~60-90 min on 4 GPUs |
replication + ablation ~60 min | analysis 45 min | report 30 min.

## Potential Challenges
- Small learned gain on tasks the base model already does well (mt, gsm8k) makes `R_gain` noisy ->
  loss-based retention is primary; exclusions are counted and reported.
- /data has ~9 GB free: no checkpoints are saved; only JSON results.
- 200 test items per task: coarse for accuracy; mitigated by 12 runs and loss-based measure.

## Success Criteria
All planned runs complete, each sub-hypothesis receives a clear supported / not supported /
inconclusive verdict with uncertainty, and negative results are reported as such.

## Deviations from plan
(appended during execution)
- (before any run, after the smoke test) Evaluation cadence changed from "loss and generation every
  50 steps" to "held-out loss every 25 steps, generation metrics every 125 steps (mid-stage and
  end-of-stage)". Reason: generation costs ~45 s per evaluation point; loss costs ~3 s. H1 (shape)
  therefore rests on the loss trajectories (51 points per lag axis) with the generation metrics as a
  coarser (11-point) check.
- (before any run) Full-fine-tuning ablation learning rate set to 2e-5 instead of 1e-5 so that 250
  steps are enough to learn each task.
- (before any run) Generation budgets for gsm8k/arith raised to 320/160 tokens: the smoke test showed
  the verbose zero-shot answers for arith were truncated at 48 tokens before the final answer, which
  would have understated the zero-shot floor.
- The GPUs are shared with other jobs that started during setup; runs are packed onto whichever
  GPUs have free memory, with automatic retry.
- (during execution, 16:25) The full-fine-tuning ablation (Exp 4) was cancelled before any of its runs
  started. Reason: the four GPUs turned out to be shared with other jobs holding ~44 GB each; LoRA runs
  that take ~10 min on a free GPU took ~100 min, and full fine-tuning needs ~3x the memory. No
  full-FT data exists, so this is a scope cut, not a selective omission.
- (during execution) Two main runs and the Qwen joint baseline died with CUDA out-of-memory when a
  neighbouring job grew; they were restarted from scratch (same seed/config). The run script was
  changed to wait for memory instead of exiting. Because micro-batching with gradient accumulation
  keeps the objective identical, runs differ only in floating-point summation order.
- (second session, 18:40) The first session was terminated by the pipeline's 4-hour limit after all 12
  main runs and one joint run had finished but before the Qwen replication, the final analysis or the
  report. A second session resumed from the saved per-run JSONL files; nothing was re-run or discarded.
  The Qwen replication (Exp 3, 6 orders, as planned) was run in the second session. The full-FT ablation
  stays cancelled (time).
- (second session, after seeing the main results; exploratory, not pre-registered) Two additions:
  (i) `src/analyze_extra.py`: within-stage curve shape, equal-parameter per-trajectory model comparison,
  seed-block and leave-one-task-out sensitivity of the H2 ratio; (ii) `src/run_onset.py` /
  `src/analyze_onset.py`: a step-resolution experiment on six (old task <- new task) pairs, because the
  25-step evaluation grid turned out to be too coarse to resolve the onset of forgetting. A first attempt
  to get this resolution by re-running two full sequences with evaluation every 5 steps was aborted after
  10 minutes (projected > 8 h) and its partial files deleted; no results from it are used.
