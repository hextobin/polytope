"""Train one arm. Writes <out>/config.json, <out>/log.jsonl, <out>/final.json (+ ckpt.pt).

Example (CPU smoke):
    python train.py --data data/arith --arm l1 --d_model 64 --n_layer 2 \
        --n_head 2 --block_size 64 --batch_size 16 --steps 200 --out runs/smoke_l1

Every --eval_every steps: val loss (fixed val batches, same for every arm), and the
stats of each projection (embeddings, then after each attention add): fraction of
tokens and of coordinates the projection changes, fraction of nonzero coordinates
(mean and median over tokens) and, for l1, the mean threshold tau.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time

import torch

from polytope.data import ByteData
from polytope.model import ARMS, GPT, GPTConfig, count_params


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--arm", default="floor", choices=ARMS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr_frac", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--max_minutes", type=float, default=0, help="stop early after this wall time (0 = off)")
    p.add_argument("--save_ckpt", action="store_true")
    for f in dataclasses.fields(GPTConfig):
        if f.name == "arm":
            continue
        p.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    return p.parse_args()


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    t = (step - a.warmup) / max(1, a.steps - a.warmup)
    return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def main():
    a = get_args()
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    cfg = GPTConfig(**{f.name: getattr(a, f.name) for f in dataclasses.fields(GPTConfig)})
    model = GPT(cfg).to(a.device)
    n_params = count_params(model)
    data = ByteData(a.data, cfg.block_size, a.device)

    # Weight decay on matrices only (not biases or LayerNorm gains).
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95))
    use_amp = a.dtype == "bf16" and a.device.startswith("cuda")
    fwd = torch.compile(model) if a.compile else model

    meta = dict(vars(a), n_params=n_params, torch=torch.__version__)
    json.dump(meta, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    print(f"arm={cfg.arm} params={n_params:,} device={a.device}")

    train_gen = torch.Generator().manual_seed(a.seed)
    log = open(os.path.join(a.out, "log.jsonl"), "w")

    @torch.no_grad()
    def evaluate():
        model.eval()
        gen = torch.Generator().manual_seed(12345)        # identical val batches for all arms
        losses = []
        for _ in range(a.eval_batches):
            x, y = data.batch("val", a.batch_size, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                _, loss = model(x, y)
            losses.append(loss.item())
        # one more pass on a single batch with projection stats on
        model.set_tracking(True)
        x, y = data.batch("val", a.batch_size, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            model(x, y)
        stats = {name: dict(m.stats) for name, m in model.proj_layers()}
        model.set_tracking(False)
        model.train()
        return sum(losses) / len(losses), stats

    t0 = time.time()
    # The first training step (torch.compile + CUDA warm-up) is excluded from every
    # elapsed_s so wall-clock comparisons are fair; it is reported as first_step_s.
    first_step_s = 0.0

    def clock():
        return time.time() - t0 - first_step_s
    tokens = 0
    step = 0
    nonfinite = 0
    for step in range(a.steps + 1):
        if step % a.eval_every == 0 or step == a.steps:
            vl, stats = evaluate()
            rec = dict(step=step, val_loss=vl, val_bpb=vl / math.log(2), tokens=tokens,
                       elapsed_s=round(clock(), 1), proj=stats)
            log.write(json.dumps(rec) + "\n"); log.flush()
            last = f"L{cfg.n_layer - 1}"
            sl = stats[last]
            extra = (f" | {last}: changed={sl['frac_tokens_changed']:.3f} "
                     f"coords={sl['frac_coords_changed']:.3f} nnz~={sl['nonzero_median']:.3f}")
            print(f"step {step:6d}  val {vl:.4f}  ({rec['elapsed_s']:.0f}s){extra}", flush=True)
        if step == a.steps:
            break
        if a.max_minutes and clock() > 60 * a.max_minutes:
            print("max_minutes reached, stopping")
            vl, stats = evaluate()
            log.write(json.dumps(dict(step=step, val_loss=vl, val_bpb=vl / math.log(2),
                                      tokens=tokens, elapsed_s=round(clock(), 1),
                                      proj=stats, stopped_early=True)) + "\n")
            break
        for gparam in opt.param_groups:
            gparam["lr"] = lr_at(step, a)
        x, y = data.batch("train", a.batch_size, train_gen)
        if step == 0:
            t_first = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            _, loss = fwd(x, y)
        if not torch.isfinite(loss):
            nonfinite += 1
            opt.zero_grad(set_to_none=True)
            if nonfinite > 20:
                print("too many non-finite losses, stopping"); break
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        if not torch.isfinite(gn):
            nonfinite += 1
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()
        tokens += x.numel()
        if step == 0:
            if a.device.startswith("cuda"):
                torch.cuda.synchronize()
            first_step_s = time.time() - t_first
        if step % 10 == 0:
            log.write(json.dumps(dict(step=step, train_loss=loss.item(), grad_norm=float(gn),
                                      lr=lr_at(step, a),
                                      elapsed_s=round(clock(), 3))) + "\n")

    elapsed = clock()
    final = dict(arm=cfg.arm, seed=a.seed, n_params=n_params, steps_done=step,
                 final_val_loss=vl, final_val_bpb=vl / math.log(2), elapsed_s=elapsed,
                 tokens_per_s=tokens / max(elapsed, 1e-9), nonfinite_steps=nonfinite,
                 first_step_s=round(first_step_s, 1))
    json.dump(final, open(os.path.join(a.out, "final.json"), "w"), indent=2)
    if a.save_ckpt:
        torch.save({"model": model.state_dict(), "cfg": dataclasses.asdict(cfg)},
                   os.path.join(a.out, "ckpt.pt"))
    print(json.dumps(final))


if __name__ == "__main__":
    main()
