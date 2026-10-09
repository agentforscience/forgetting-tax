# The Forgetting Tax

Sequential (continual) LoRA instruction tuning of two small LLMs (Llama-3.2-1B-Instruct, Qwen2.5-1.5B-Instruct)
on six tasks in balanced orders, with held-out loss measured every 25 steps, to test whether forgetting
(H1) follows a power law, (H2) is 3-5x faster for reasoning than pattern-matching tasks, and (H3) hits
early-learned tasks disproportionately. Full write-up: [REPORT.md](REPORT.md).

## Key findings
- **None of the three sub-hypotheses is supported as stated.**
- **Forgetting is not uniform, but it is step-like rather than a power law.** At each task switch an old
  task's loss rises within 20-40 steps and then stays flat. A power law is the best fit only for averaged
  Llama curves; individual trajectories (0 of 36 clear wins vs a saturating exponential), averaged Qwen
  curves, and a step-resolution experiment (6 of 6 pairs) favour saturation.
- **No 3-5x reasoning/pattern gap.** Ratio 1.0-1.5 on Llama; 2.2-2.6 on Qwen for metric-based measures,
  not robust to multiple-comparison correction or normalisation. Tasks within a category differ more than
  the categories do.
- **No reliable serial-position effect** once the amount of later training is held fixed (none on Llama;
  one of several measures on Qwen).
- Forgetting is concentrated in a few (old task, new task) pairs. Sequential training ends near the
  zero-shot average (Llama 0.499 vs 0.460 zero-shot, 0.623 joint training).

Limits: two small models, LoRA only, six tasks, 200 test items per task, full-fine-tuning ablation not run.

## Reproduce
```bash
uv venv && source .venv/bin/activate && uv sync      # or: uv pip install -r requirements.txt
python src/prep_data.py                               # builds datasets/processed/*.jsonl
python src/launch.py --groups main,joint,qwen         # 12 + 3 + 6 runs; picks free GPUs, skips finished runs
python src/run_onset.py                               # step-resolution onset experiment (6 pairs)
python src/analyze.py --tag main && python src/analyze_extra.py --tag main
python src/analyze.py --tag qwen --joint_tag qwenjoint && python src/analyze_extra.py --tag qwen
python src/analyze_onset.py
```
A run used up to 28 GB of GPU memory. On the shared machine used here runs took 60-110 minutes each;
the first session estimated about 10 minutes on an idle A6000, which was not re-measured. `launch.py` sets
`HF_HUB_CACHE=/data/transformers` and offline mode; change these for another machine.

## Files
```
planning.md            pre-registered plan and logged deviations
REPORT.md              full report
src/prep_data.py       dataset construction and prompts
src/run_continual.py   one sequential (or joint) training run with dense evaluation
src/launch.py          Williams-design orders and GPU job queue
src/run_onset.py       step-resolution forgetting-onset experiment
src/analyze*.py        statistics, tables, figures
results/runs/          raw per-evaluation records (JSONL)
results/summary_*.json, extra_*.json, onset_summary.json   all reported statistics
figures/               plots
logs/                  training and analysis logs
```
