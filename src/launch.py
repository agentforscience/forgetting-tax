"""Job launcher: runs all experiment configurations over the free GPU slots.

The machine is shared with other jobs, so each worker waits until some GPU has
enough free memory before starting a run there. Finished runs (.done marker)
are skipped, failed runs (e.g. another job took the memory) are retried.
"""
import argparse
import os
import queue
import random
import re
import subprocess
import threading
import time

TASKS = ["gsm8k", "arith", "logic", "mt", "summ", "emotion"]


def williams(n=6):
    """Williams design for even n: each item appears once per position and
    follows every other item exactly once."""
    first, lo, hi = [0], 1, n - 1
    while len(first) < n:
        first.append(lo); lo += 1
        if len(first) < n:
            first.append(hi); hi -= 1
    return [[(x + i) % n for x in first] for i in range(n)]


def orders(seed):
    """Seed 0 uses the canonical labelling, other seeds a random relabelling of tasks."""
    lab = list(TASKS)
    if seed > 0:
        random.Random(1000 + seed).shuffle(lab)
    return [",".join(lab[i] for i in row) for row in williams()]


def jobs(groups):
    out = []
    llama, qwen = "meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"
    if "main" in groups:
        out += [["--tag", "main", "--model", llama, "--order", o, "--seed", str(s)] for s in (0, 1) for o in orders(s)]
    if "joint" in groups:
        out += [["--tag", "joint", "--model", llama, "--joint", "--seed", str(s)] for s in (0, 1)]
        out += [["--tag", "qwenjoint", "--model", qwen, "--joint", "--seed", "0"]]
    if "qwen" in groups:
        out += [["--tag", "qwen", "--model", qwen, "--order", o, "--seed", "0"] for o in orders(0)]
    if "fullft" in groups:
        out += [["--tag", "fullft", "--model", llama, "--full_ft", "--lr", "2e-5", "--order", o, "--seed", "0"] for o in orders(0)]
    return out


LOCK = threading.Lock()


MINE = {}  # gpu index -> number of my runs currently on it


def free_gpu(need_mb):
    """Among GPUs with at least need_mb MiB free, the one running fewest of my jobs (ties: most free)."""
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
    cand = [(MINE.get(int(l.split(",")[0]), 0), -int(l.split(",")[1]), int(l.split(",")[0]))
            for l in out.strip().splitlines() if int(l.split(",")[1]) >= need_mb]
    return min(cand)[2] if cand else None


def worker(q, wid, need_mb):
    env = dict(os.environ, HF_HUB_CACHE="/data/transformers", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    while True:
        try:
            job, tries = q.get_nowait()
        except queue.Empty:
            return
        need = need_mb * (3 if "--full_ft" in job else 1)
        # The GPUs are shared: wait until one has room. The lock + pause staggers start-ups
        # so that two workers do not both claim the same free memory.
        with LOCK:
            while (gpu := free_gpu(need)) is None:
                time.sleep(20)
            log = open(f"logs/worker_{wid}.log", "a")
            proc = subprocess.Popen(["python", "src/run_continual.py"] + job, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)),
                                    stdout=log, stderr=subprocess.STDOUT)
            MINE[gpu] = MINE.get(gpu, 0) + 1
            time.sleep(40)
        rc = proc.wait()
        MINE[gpu] -= 1
        log.close()
        print(f"{time.strftime('%H:%M:%S')} worker {wid} gpu {gpu} rc={rc} tries={tries}: {' '.join(job)}", flush=True)
        if rc != 0 and tries < 8:
            q.put((job, tries + 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="main,joint,qwen,fullft")
    ap.add_argument("--match", default=".", help="regex; only jobs whose argument string matches are run")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--need_mb", type=int, default=6000, help="free GPU memory required to start a LoRA run")
    a = ap.parse_args()
    q = queue.Queue()
    for j in jobs(a.groups.split(",")):
        if re.search(a.match, " ".join(j)):
            q.put((j, 1))
    print("jobs:", q.qsize(), flush=True)
    ths = [threading.Thread(target=worker, args=(q, i, a.need_mb)) for i in range(a.workers)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    print("ALL DONE", flush=True)
