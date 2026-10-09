"""Exploratory follow-up: step-level resolution of forgetting right after a task switch.

The main runs evaluate every 25 steps and show that most of a stage's forgetting has already happened
by the first evaluation. Here, for a few (old task A -> new task B) pairs, a fresh LoRA adapter is
trained on A for 250 steps exactly as in the main runs, then on B for 100 steps while the held-out
loss of A (and of B) is measured after every step for the first 20 steps, every 2 steps up to 50 and
every 10 steps up to 100. Only losses are recorded (no generation).

Usage: python src/run_onset.py --pairs logic:arith,mt:summ --seed 0
"""
import argparse
import json
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_continual import TASKS, build_supervised, eval_loss, load_task, set_seed, train_step

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
ap.add_argument("--pairs", default="logic:arith,logic:gsm8k,mt:summ,emotion:arith,summ:mt,gsm8k:arith")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps_a", type=int, default=250)
ap.add_argument("--steps_b", type=int, default=100)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--out", default="results/onset")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
device = "cuda"
tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = "<|finetune_right_pad_id|>" if "<|finetune_right_pad_id|>" in tok.get_vocab() else tok.eos_token
test_enc = {t: build_supervised(tok, load_task(t, "test")[:200])[0] for t in TASKS}
train_enc = {t: build_supervised(tok, load_task(t, "train")[:2000])[0] for t in TASKS}
eval_at = set(range(0, 21)) | set(range(20, 51, 2)) | set(range(50, args.steps_b + 1, 10))

for pair in args.pairs.split(","):
    a, b = pair.split(":")
    path = f"{args.out}/onset_{a}-{b}_s{args.seed}.json"
    if os.path.exists(path):
        continue
    set_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    rng = random.Random(args.seed)

    def losses():
        model.eval()
        out = {a: eval_loss(model, test_enc[a], tok.pad_token_id, device), b: eval_loss(model, test_enc[b], tok.pad_token_id, device)}
        model.train()
        return out

    rec = {"old": a, "new": b, "seed": args.seed, "config": vars(args), "zero_shot": losses(), "points": []}
    t0 = time.time()
    model.train()
    for task, n_steps in ((a, args.steps_a), (b, args.steps_b)):
        pool, perm = train_enc[task], []
        if task == b:
            rec["points"].append({"step": 0, "lr": 0.0, **{("old" if k == a else "new"): v for k, v in losses().items()}})
        for s in range(n_steps):
            lr = args.lr * min(1.0, (s + 1) / args.warmup)  # same per-stage warm-up as the main runs
            for g in opt.param_groups:
                g["lr"] = lr
            if len(perm) < args.bs:
                perm = list(range(len(pool)))
                rng.shuffle(perm)
            train_step(model, opt, [pool[perm.pop()] for _ in range(args.bs)], tok.pad_token_id, device, params)
            if task == b and (s + 1) in eval_at:
                rec["points"].append({"step": s + 1, "lr": lr, **{("old" if k == a else "new"): v for k, v in losses().items()}})
    rec["seconds"] = time.time() - t0
    with open(path, "w") as f:
        json.dump(rec, f)
    p = rec["points"]
    print(f"{a}<-{b}: old-task loss {p[0]['old']:.3f} -> step10 {p[10]['old']:.3f} -> step25 "
          f"{[x for x in p if x['step'] == 26][0]['old']:.3f} -> step100 {p[-1]['old']:.3f} ({rec['seconds']:.0f}s)", flush=True)
    del model, opt, params
    torch.cuda.empty_cache()
print("ONSET DONE", flush=True)
