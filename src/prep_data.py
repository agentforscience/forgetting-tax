"""Build the six instruction-tuning tasks used in the continual-learning stream.

Every task is cast to the same (instruction prompt -> target text) format and
written to datasets/processed/{task}_{train,test}.jsonl.

Reasoning ("abstract") tasks      : gsm8k (math word problems), arith (multi-step
                                    arithmetic, synthetic), logic (ProofWriter OWA)
Pattern-matching ("surface") tasks: mt (WMT16 en->de), summ (SAMSum), emotion
                                    (dair-ai/emotion, 6-way classification)
"""
import json
import os
import random
import re

from datasets import load_dataset

SEED = 42
N_TRAIN, N_TEST = 2000, 200
OUT = "datasets/processed"
MAX_PROMPT_CHARS = 1200  # keeps sequences short enough for fast batched training

CATEGORY = {"gsm8k": "reasoning", "arith": "reasoning", "logic": "reasoning",
            "mt": "pattern", "summ": "pattern", "emotion": "pattern"}
METRIC = {"gsm8k": "accuracy", "arith": "accuracy", "logic": "accuracy",
          "mt": "chrf", "summ": "rougeL", "emotion": "accuracy"}


def sample(rows, n, rng):
    """Shuffle and take n rows. Train pools are over-sampled (n > N_TRAIN) so that
    rows duplicated in the test set can be dropped in main() without falling short."""
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:n]


def gsm8k(rng):
    ds = load_dataset("openai/gsm8k", "main")

    def conv(ex):
        sol, ans = ex["answer"].split("####")
        sol = re.sub(r"<<.*?>>", "", sol).strip()  # drop calculator annotations
        ans = ans.strip().replace(",", "")
        return {"prompt": "Solve the following math word problem. Show your work, then give the "
                          "final numeric answer on the last line as 'Answer: <number>'.\n\n" + ex["question"],
                "target": f"{sol}\nAnswer: {ans}", "answer": ans}
    return ([conv(e) for e in sample(ds["train"], N_TRAIN + 100, rng)],
            [conv(e) for e in sample(ds["test"], N_TEST, rng)])


def arith(rng):
    """Synthetic 3-operation integer arithmetic with worked steps (contamination-free)."""
    def make():
        a, b = rng.randint(11, 99), rng.randint(11, 99)
        c, d = rng.randint(2, 9), rng.randint(11, 999)
        op1, op3 = rng.choice("+-"), rng.choice("+-")
        s1 = a + b if op1 == "+" else a - b
        s2 = s1 * c
        s3 = s2 + d if op3 == "+" else s2 - d
        expr = f"({a} {op1} {b}) * {c} {op3} {d}"
        steps = f"{a} {op1} {b} = {s1}\n{s1} * {c} = {s2}\n{s2} {op3} {d} = {s3}"
        return {"prompt": "Evaluate the arithmetic expression step by step, then give the final "
                          "result on the last line as 'Answer: <number>'.\n\n" + expr,
                "target": f"{steps}\nAnswer: {s3}", "answer": str(s3), "expr": expr}
    seen, rows = set(), []
    while len(rows) < N_TRAIN + N_TEST:
        r = make()
        if r["expr"] not in seen:  # guarantees train/test disjointness
            seen.add(r["expr"])
            rows.append(r)
    return rows[:N_TRAIN], rows[N_TRAIN:]


def logic(rng):
    """ProofWriter (open-world): deductive reasoning over facts+rules, depth 1-3."""
    def take(split, n):
        ds = load_dataset("tasksource/proofwriter", split=split, streaming=True).shuffle(seed=SEED, buffer_size=20000)
        per, out = {"True": [], "False": [], "Unknown": []}, []
        for ex in ds:
            if ex["QDep"] is None or not (1 <= ex["QDep"] <= 3) or len(ex["theory"]) > MAX_PROMPT_CHARS:
                continue
            if ex["answer"] in per and len(per[ex["answer"]]) < n // 3 + 1:
                per[ex["answer"]].append(ex)
            if all(len(v) >= n // 3 + 1 for v in per.values()):
                break
        for v in per.values():
            out += v
        rng.shuffle(out)
        return [{"prompt": "Given the facts and rules below, decide whether the statement is True, False, "
                           "or Unknown (cannot be determined). Answer with a single word.\n\n"
                           f"Facts and rules: {e['theory']}\n\nStatement: {e['question']}",
                 "target": e["answer"], "answer": e["answer"]} for e in out[:n]]
    return take("train", N_TRAIN + 100), take("test", N_TEST)


def mt(rng):
    ds = load_dataset("wmt/wmt16", "de-en")

    def conv(ex):
        t = ex["translation"]
        return {"prompt": "Translate the following English text into German.\n\n" + t["en"],
                "target": t["de"], "answer": t["de"]}
    ok = lambda e: 20 <= len(e["translation"]["en"]) <= 300
    # newstest2015 (validation split) is used for training, newstest2016 (test split) for testing.
    return ([conv(e) for e in sample([e for e in ds["validation"] if ok(e)], N_TRAIN + 100, rng)],
            [conv(e) for e in sample([e for e in ds["test"] if ok(e)], N_TEST, rng)])


def summ(rng):
    ds = load_dataset("knkarthick/samsum")

    def conv(ex):
        return {"prompt": "Summarize the following dialogue in one or two sentences.\n\n" + ex["dialogue"],
                "target": ex["summary"].strip(), "answer": ex["summary"].strip()}
    ok = lambda e: e["dialogue"] and e["summary"] and 50 <= len(e["dialogue"]) <= MAX_PROMPT_CHARS
    return ([conv(e) for e in sample([e for e in ds["train"] if ok(e)], N_TRAIN + 100, rng)],
            [conv(e) for e in sample([e for e in ds["test"] if ok(e)], N_TEST, rng)])


def emotion(rng):
    ds = load_dataset("dair-ai/emotion", "split")
    names = ds["train"].features["label"].names

    def conv(ex):
        return {"prompt": "Classify the emotion expressed in the text as one of: sadness, joy, love, anger, "
                          "fear, surprise. Answer with a single word.\n\nText: " + ex["text"],
                "target": names[ex["label"]], "answer": names[ex["label"]]}
    return ([conv(e) for e in sample(ds["train"], N_TRAIN + 100, rng)],
            [conv(e) for e in sample(ds["test"], N_TEST, rng)])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [("gsm8k", gsm8k), ("arith", arith), ("logic", logic), ("mt", mt), ("summ", summ), ("emotion", emotion)]:
        rng = random.Random(SEED)
        train, test = fn(rng)
        # Leakage guard: drop duplicate prompts and any train prompt that also occurs in test.
        test_prompts, seen, clean = {r["prompt"] for r in test}, set(), []
        for r in train:
            if r["prompt"] not in test_prompts and r["prompt"] not in seen:
                seen.add(r["prompt"])
                clean.append(r)
        print(f"{name}: dropped {len(train) - len(clean)} leaked/duplicate train rows")
        train = clean[:N_TRAIN]
        assert len(train) == N_TRAIN and len(test) == N_TEST
        assert not {r["prompt"] for r in train} & {r["prompt"] for r in test}, f"train/test leak in {name}"
        for split, rows in [("train", train), ("test", test)]:
            with open(f"{OUT}/{name}_{split}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        print(f"{name}: train={len(train)} test={len(test)} | mean prompt chars="
              f"{sum(len(r['prompt']) for r in train) / len(train):.0f} target chars="
              f"{sum(len(r['target']) for r in train) / len(train):.0f}")
        print("  example:", json.dumps(test[0])[:400])
