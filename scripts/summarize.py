"""Summarize runs: python scripts/summarize.py runs/<sweep>  [--md out.md]

Reads every <run>/final.json and <run>/log.jsonl below the folder. Prints, per arm:
seeds, params, final val bits per byte (mean and std over seeds), gap to floor and
whether it exceeds twice the larger seed sd, throughput, non-finite steps, and the
projection stats at the last eval (mean over seeds, median over layers L0..L{n-1}):
fraction of tokens changed, fraction of coordinates changed, median nonzero fraction.
A second table gives the last-eval stats per layer for each arm (mean over seeds).
"""
import argparse
import glob
import json
import math
import os
import statistics
from collections import defaultdict


def load(root):
    rows = []
    for f in glob.glob(os.path.join(root, "**", "final.json"), recursive=True):
        d = json.load(open(f))
        logp = os.path.join(os.path.dirname(f), "log.jsonl")
        last = {}
        if os.path.exists(logp):
            for line in open(logp):
                r = json.loads(line)
                if "proj" in r:
                    last = r["proj"]
        d["proj"] = last
        rows.append(d)
    return rows


def ms(v):
    m = sum(v) / len(v)
    s = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else float("nan")
    return m, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--md")
    a = ap.parse_args()
    by = defaultdict(list)
    for r in load(a.root):
        by[r["arm"]].append(r)
    if not by:
        print("no runs found"); return

    def layer_mean(rs, layer, key):
        v = [r["proj"][layer][key] for r in rs if layer in r["proj"] and key in r["proj"][layer]]
        return sum(v) / len(v) if v else float("nan")

    def blocks(rs):
        return sorted({k for r in rs for k in r["proj"] if k.startswith("L")}, key=lambda k: int(k[1:]))

    base = ms([r["final_val_bpb"] for r in by["floor"]]) if "floor" in by else None
    lines = ["| arm | seeds | params | val bpb (mean ± sd) | Δ vs floor | > 2 sd? | tok/s | non-finite steps "
             "| tokens changed | coords changed | nonzero (median) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in sorted(by, key=lambda k: ms([r["final_val_bpb"] for r in by[k]])[0]):
        rs = by[arm]
        m, s = ms([r["final_val_bpb"] for r in rs])
        tps = sum(r["tokens_per_s"] for r in rs) / len(rs)
        d, sig = "-", "-"
        if base is not None and arm != "floor":
            d = f"{m - base[0]:+.4f}"
            sd = max(s, base[1])
            sig = "-" if math.isnan(sd) else ("yes" if abs(m - base[0]) > 2 * sd else "no")
        ls = blocks(rs)
        med = lambda key: statistics.median(layer_mean(rs, l, key) for l in ls) if ls else float("nan")
        lines.append(f"| {arm} | {len(rs)} | {rs[0]['n_params']:,} | {m:.4f} ± {s:.4f} | {d} | {sig} | "
                     f"{tps:,.0f} | {sum(r['nonfinite_steps'] for r in rs)} | "
                     f"{med('frac_tokens_changed'):.3f} | {med('frac_coords_changed'):.3f} | "
                     f"{med('nonzero_median'):.3f} |")

    lines += ["", "Per layer at the last eval (mean over seeds): tokens changed / coords changed / "
              "nonzero median" + " / tau (l1)", ""]
    arms = [k for k in ("box", "l1", "shrink") if k in by]
    if arms:
        lines += ["| layer | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
        for layer in ["emb"] + blocks(by[arms[0]]):
            cells = []
            for arm in arms:
                rs = by[arm]
                c = (f"{layer_mean(rs, layer, 'frac_tokens_changed'):.3f} / "
                     f"{layer_mean(rs, layer, 'frac_coords_changed'):.3f} / "
                     f"{layer_mean(rs, layer, 'nonzero_median'):.3f}")
                if arm == "l1":
                    c += f" / {layer_mean(rs, layer, 'tau_mean'):.3f}"
                cells.append(c)
            lines.append(f"| {layer} | " + " | ".join(cells) + " |")
    out = "\n".join(lines)
    print(out)
    if a.md:
        open(a.md, "w").write(out + "\n")


if __name__ == "__main__":
    main()
