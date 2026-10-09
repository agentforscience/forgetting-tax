"""Sequential (continual) instruction tuning with dense forgetting evaluation.

One invocation = one run (one task order, one seed, one model). After every
`eval_every` optimizer steps all six test sets are evaluated with (a) held-out
per-token loss on the reference target and (b) greedy-decoded task metrics.
Results are appended to a JSONL file after each evaluation, so a crashed run
still leaves usable data. No model checkpoints are written (disk is scarce).

Usage:
  python src/run_continual.py --order gsm8k,arith,logic,mt,summ,emotion --seed 0 --tag main
  python src/run_continual.py --joint --seed 0 --tag joint
"""
import argparse
import gc
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from rouge_score import rouge_scorer
from sacrebleu.metrics import CHRF
from transformers import AutoModelForCausalLM, AutoTokenizer

TASKS = ["gsm8k", "arith", "logic", "mt", "summ", "emotion"]
# budgets are generous enough that verbose zero-shot answers are not cut off before the final answer
MAX_NEW = {"gsm8k": 320, "arith": 160, "logic": 4, "mt": 96, "summ": 64, "emotion": 4}
DATA = "datasets/processed"
MAX_LEN = 640  # tokens; longer training sequences are truncated (rare, see log)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_task(task, split):
    with open(f"{DATA}/{task}_{split}.jsonl") as f:
        return [json.loads(line) for line in f]


# ----------------------------------------------------------------------------- scoring
_chrf = CHRF()
_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def extract_number(text):
    """Final numeric answer: prefer the 'Answer: x' line, else the last number in the text."""
    m = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if not m:
        m = re.findall(r"-?[\d,]*\.?\d+", text)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def score(task, pred, ex):
    """Per-example score in [0, 1]."""
    if task in ("gsm8k", "arith"):
        v = extract_number(pred)
        return float(v is not None and abs(v - float(ex["answer"])) < 1e-4)
    if task in ("logic", "emotion"):
        w = re.findall(r"[a-z]+", pred.lower())
        return float(bool(w) and w[0] == ex["answer"].lower())
    if task == "mt":
        return _chrf.sentence_score(pred.strip(), [ex["answer"]]).score / 100.0
    if task == "summ":
        return _rouge.score(ex["answer"], pred.strip())["rougeL"].fmeasure
    raise ValueError(task)


# ----------------------------------------------------------------------------- encoding
def encode_prompt(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)


def build_supervised(tok, rows):
    """Token ids + labels with the prompt masked out (loss on the assistant answer only)."""
    out, n_trunc = [], 0
    for r in rows:
        p = tok(encode_prompt(tok, r["prompt"]), add_special_tokens=False)["input_ids"]
        t = tok(r["target"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids, lab = (p + t), ([-100] * len(p) + t)
        if len(ids) > MAX_LEN:
            n_trunc += 1
            ids, lab = ids[:MAX_LEN], lab[:MAX_LEN]
        out.append((ids, lab))
    return out, n_trunc


def collate(batch, pad_id, device):
    n = max(len(i) for i, _ in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for k, (i, l) in enumerate(batch):
        ids[k, :len(i)], lab[k, :len(l)], att[k, :len(i)] = torch.tensor(i), torch.tensor(l), 1
    return ids.to(device), att.to(device), lab.to(device)


def answer_nll(model, ids, att, lab):
    """Summed cross-entropy on the supervised (answer) positions only, plus their count.

    The decoder body is run once, then the LM head is applied only to hidden states
    whose next token is supervised. This is numerically the same objective as the
    built-in `labels=` loss but avoids a batch x seq x vocab logits tensor, which
    matters because the GPUs are shared and only a few GB are free.
    """
    lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    mask = lab[:, 1:] != -100
    tgt = lab[:, 1:][mask]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = lm.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1][mask]
        # LM head + CE in chunks of answer positions to bound the vocab-sized tensors
        loss = sum(torch.nn.functional.cross_entropy(lm.lm_head(hidden[c:c + 256]).float(), tgt[c:c + 256], reduction="sum")
                   for c in range(0, len(tgt), 256))
    return loss, len(tgt)


# Batch sizes are halved on a CUDA out-of-memory error and the operation is retried. Training
# uses micro-batches with gradient accumulation, so the effective batch (and the objective:
# mean NLL over all answer tokens of the 16 sequences) is unchanged when memory is tight.
BS = {"micro": 16, "loss": 16, "gen": 25}


def shrink(key):
    gc.collect()
    torch.cuda.empty_cache()
    if BS[key] == 1:
        # Already at the minimum: another job on the shared GPU has taken the memory. Wait for it
        # to be released rather than crash and lose the run (no checkpoints are kept).
        BS["waits"] = BS.get("waits", 0) + 1
        assert BS["waits"] < 240, f"out of GPU memory at batch size 1 for over an hour ({key})"
        print(f"OOM at {key} batch size 1: waiting 15s (wait #{BS['waits']})", flush=True)
        time.sleep(15)
        return
    BS[key] //= 2
    print(f"OOM: reducing {key} batch size to {BS[key]}", flush=True)


def train_step(model, opt, batch, pad_id, device, params):
    """One optimizer step on `batch`; returns the mean answer-token NLL."""
    n_tok = sum(sum(l != -100 for l in lab[1:]) for _, lab in batch)
    while True:
        oom = False
        try:
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for s in range(0, len(batch), BS["micro"]):
                ids, att, lab = collate(batch[s:s + BS["micro"]], pad_id, device)
                loss, _ = answer_nll(model, ids, att, lab)
                (loss / n_tok).backward()
                total += loss.item()
                del loss
        except torch.OutOfMemoryError:
            oom = True  # handled below: the traceback must be released before memory can be freed
        if not oom:
            break
        ids = att = lab = loss = None
        shrink("micro")
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    return total / n_tok


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def eval_loss(model, enc, pad_id, device):
    """Mean per-token NLL of the reference targets (token-weighted over the test set)."""
    tot, cnt = 0.0, 0
    order = sorted(range(len(enc)), key=lambda k: len(enc[k][0]))
    s = 0
    while s < len(order):
        oom = False
        try:
            ids, att, lab = collate([enc[k] for k in order[s:s + BS["loss"]]], pad_id, device)
            loss, n = answer_nll(model, ids, att, lab)
        except torch.OutOfMemoryError:
            oom = True
        if oom:
            shrink("loss")
            continue
        tot += loss.item()
        cnt += n
        s += len(ids)
    return tot / cnt


@torch.no_grad()
def eval_gen(model, tok, task, rows, device):
    """Greedy decoding; returns per-example scores and raw predictions."""
    prompts = [encode_prompt(tok, r["prompt"]) for r in rows]
    order = sorted(range(len(rows)), key=lambda k: len(prompts[k]))
    preds = [None] * len(rows)
    s = 0
    while s < len(order):
        idx = order[s:s + BS["gen"]]
        enc = tok([prompts[k] for k in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        oom = False
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.generate(**enc, max_new_tokens=MAX_NEW[task], do_sample=False,
                                     pad_token_id=tok.pad_token_id)
        except torch.OutOfMemoryError:
            oom = True
        if oom:
            enc = None
            shrink("gen")
            continue
        s += len(idx)
        txt = tok.batch_decode(out[:, enc["input_ids"].size(1):], skip_special_tokens=True)
        for k, t in zip(idx, txt):
            preds[k] = t
    return [score(task, p, r) for p, r in zip(preds, rows)], preds


def evaluate(model, tok, test, test_enc, device, do_gen=True):
    model.eval()
    res = {}
    for task in TASKS:
        res[task] = {"loss": eval_loss(model, test_enc[task], tok.pad_token_id, device)}
        if do_gen:
            sc, preds = eval_gen(model, tok, task, test[task], device)
            res[task].update(metric=float(np.mean(sc)), scores=sc, preds=preds)
    model.train()
    torch.cuda.empty_cache()  # return cached blocks: the GPUs are shared
    return res


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--order", default=",".join(TASKS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--steps_per_task", type=int, default=250)
    ap.add_argument("--eval_every", type=int, default=25, help="held-out loss evaluation interval (steps)")
    ap.add_argument("--gen_every", type=int, default=125, help="generation-metric evaluation interval (steps)")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--full_ft", action="store_true", help="update all weights instead of a LoRA adapter")
    ap.add_argument("--joint", action="store_true", help="multi-task baseline: train on the mixture of all tasks")
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--out_dir", default="results/runs")
    args = ap.parse_args()

    order = args.order.split(",")
    assert sorted(order) == sorted(TASKS), order
    name = f"{args.tag}_{'joint' if args.joint else '-'.join(order)}_s{args.seed}"
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = f"{args.out_dir}/{name}.jsonl"
    if os.path.exists(f"{args.out_dir}/.skip_{args.tag}"):
        print("skip marker present for tag", args.tag)  # lets a queued experiment group be cancelled
        return
    if os.path.exists(out_path + ".done"):
        print("already done:", name)
        return
    if os.path.exists(out_path) and time.time() - os.path.getmtime(out_path) < 900:
        print("in progress elsewhere (results file recently written):", name)
        return
    set_seed(args.seed)
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"  # required for batched generation; harmless for training (attention mask)
    if tok.pad_token is None:
        tok.pad_token = "<|finetune_right_pad_id|>" if "<|finetune_right_pad_id|>" in tok.get_vocab() else tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if not args.full_ft else torch.float32).to(device)
    if not args.full_ft:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    test = {t: load_task(t, "test")[:args.n_test] for t in TASKS}
    test_enc = {t: build_supervised(tok, test[t])[0] for t in TASKS}
    train_enc = {}
    for t in TASKS:
        train_enc[t], n_trunc = build_supervised(tok, load_task(t, "train")[:args.n_train])
        print(f"{t}: {len(train_enc[t])} train examples, {n_trunc} truncated to {MAX_LEN} tokens", flush=True)

    config = dict(vars(args), name=name, order=order, n_trainable=n_train_params, torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(0), started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    f = open(out_path, "w")
    f.write(json.dumps({"config": config}) + "\n")

    def log_eval(step, stage, task_in_training):
        t0 = time.time()
        do_gen = step % args.gen_every == 0
        res = evaluate(model, tok, test, test_enc, device, do_gen=do_gen)
        rec = {"step": step, "stage": stage, "training_task": task_in_training,
               "stage_end": step % args.steps_per_task == 0, "has_gen": do_gen, "eval": res}
        # raw generations are only kept at stage boundaries to limit file size
        if do_gen and not rec["stage_end"]:
            for t in TASKS:
                res[t].pop("preds")
        f.write(json.dumps(rec) + "\n")
        f.flush()
        print(f"[{name}] step {step} stage {stage} ({task_in_training}) eval {time.time() - t0:.0f}s | " +
              " ".join(f"{t}:{res[t].get('metric', float('nan')):.3f}/{res[t]['loss']:.3f}" for t in TASKS), flush=True)

    log_eval(0, 0, None)  # zero-shot baseline

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    rng = random.Random(args.seed)
    stages = [("joint", TASKS)] if args.joint else [(t, [t]) for t in order]
    steps_per_stage = args.steps_per_task * len(TASKS) if args.joint else args.steps_per_task
    step, t_start = 0, time.time()
    model.train()
    for si, (stage_name, stage_tasks) in enumerate(stages, 1):
        pool = [ex for t in stage_tasks for ex in train_enc[t]]
        perm = []
        for s in range(steps_per_stage):
            # constant LR with a short linear warm-up at the start of every stage
            for g in opt.param_groups:
                g["lr"] = args.lr * min(1.0, (s + 1) / args.warmup)
            if len(perm) < args.bs:  # new epoch
                perm = list(range(len(pool)))
                rng.shuffle(perm)
            batch = [pool[perm.pop()] for _ in range(args.bs)]
            loss = train_step(model, opt, batch, tok.pad_token_id, device, params)
            assert math.isfinite(loss), f"non-finite loss at step {step}"
            step += 1
            if step % 50 == 0:
                BS["micro"] = min(16, BS["micro"] * 2)  # memory on the shared GPU may have freed up; OOM handling re-shrinks
                print(f"[{name}] step {step} train loss {loss:.3f} elapsed {time.time() - t_start:.0f}s micro_bs {BS['micro']}", flush=True)
            if step % args.eval_every == 0:
                log_eval(step, si, stage_name)
    config["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    f.write(json.dumps({"finished": True, "peak_gpu_mem_gb": config["peak_gpu_mem_gb"], "final_batch_sizes": BS, "train_eval_seconds": time.time() - t_start}) + "\n")
    f.close()
    open(out_path + ".done", "w").close()
    print("DONE", name, f"{time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
