# polytope

**Question:** in a transformer with no MLP, if the residual stream is forced
onto a polytope after every attention update, does that help the model, and
what kind of code does the stream settle into?

**Answer (at this scale):** unclear. The L1 ball shows a hint of more reliable
training, but no gain in score. It was the only constraint that trained on all 3 seeds
(1.86 ± 0.05 bits per byte), but it only tied the unconstrained attention model
on the seeds where that model trained (1.88). The box stayed stuck at about
5 bits per byte on every seed. With 3 seeds, no arm passes the 2 sd rule against
`floor`, and the stream never settled inside the L1 ball, so the projection
stayed active on every token.

## The constraint

After every attention add (and once on the embeddings), the residual stream is
normalized and then projected onto a fixed polytope P:

    x ← Proj_P(RMSNorm(x + attn(x)))

The RMSNorm comes first on purpose. Attention reads through LayerNorm, so it
can't see scale. Without the norm, the model could shrink the stream until it
sat inside P, and the projection would never do anything. With the norm, the
only way to avoid the projection is for the stream to already have P's shape.

- **L1 ball** (cross-polytope), radius r = d/4: an exact projection by sorting
  and soft-thresholding. One threshold is shared by all coordinates, so new
  attention output and old content compete for a fixed budget. The result has
  exact zeros. At initialization it keeps about 44% of coordinates for a dense
  input.
- **Box** [−1, 1]^d: clipping. It works per coordinate (it's hardtanh), and
  its vertices are dense ±1 codes.
- **Shrink** (control, not a polytope): soft-thresholding with a fixed
  threshold τ = 0.78, which keeps the same ~44% of coordinates at
  initialization as the L1 ball. It's the proximal step CRATE-style models use
  (see Prior art). Shrink vs L1 separates "sparsify" from "fixed budget": the
  same operation, but with the threshold fixed instead of set per token by the
  budget.

Learned faces are left out. With up to d orthogonal faces the projection is
exactly `x − Aᵀ relu(Ax − b)`, a ReLU MLP whose output weights are tied to its
input weights, which would bring back the MLP. With more faces it needs a
quadratic-program solver.

## Setup

- **Model and data:** polydeq's byte-level GPT (6 layers, d = 384) on enwik8,
  5000 steps at batch 64 × 256 bytes. Metric: validation bits per byte.
- **Arms**, all with the same structure (attention reads through a pre-norm,
  and the stream is renormalized after each add):
  - `floor`: attention only, RMSNorm after each add, no projection
  - `box`: floor, then clip to [−1, 1]
  - `l1`: floor, then project onto the L1 ball of radius d/4
  - `shrink`: floor, then soft-threshold at a fixed τ = 0.78
  - `ceiling`: floor plus a GELU MLP sublayer. It has more parameters and is a
    reference only. The main comparisons (`floor`, `box`, `l1`, `shrink`) are
    at equal parameter count.
- **Also logged, per layer:** how often the projection changes the vector, the
  fraction of coordinates that are nonzero (`l1`) or clipped (`box`), and
  throughput.
- 3 seeds per arm, one NVIDIA H100 (DigitalOcean `gpu-h100x1-80gb`, $4.41/hr),
  about 14 minutes of runs and 20 minutes on the box, so about $1.50, not
  counting a cancelled first box. A difference counts only if it exceeds twice
  the larger seed standard deviation.

## Results (mean ± sd over 3 seeds)

| arm | val bpb | vs `floor` | seeds at each value | seeds trained | tokens/s |
|---|---|---|---|---|---|
| `ceiling` | 1.583 ± 0.026 | -1.120 | 1.61, 1.58, 1.56 | 3/3 | 1.77M |
| `l1` | 1.865 ± 0.054 | -0.838 | 1.91, 1.80, 1.88 | 3/3 | 0.95M |
| `floor` | 2.702 ± 1.417 | - | 4.34, 1.87, 1.90 | 2/3 | 2.74M |
| `shrink` | 3.113 ± 1.186 | +0.411 | 4.15, 1.82, 3.38 | 1/3 | 2.75M |
| `box` | 5.069 ± 0.024 | +2.367 | 5.08, 5.04, 5.08 | 0/3 | 2.70M |

Bits per byte on validation; lower is better, and 8 is a uniform guess. "Trained"
means under 2 bpb at step 5000. `ceiling` has 10.8M parameters; the others have
3.75M. No difference from `floor` passes the 2 sd rule, because `floor`'s stuck
seed makes its sd 1.42. No run had a non-finite step.

## Findings

1. **Training reliability separates the arms more than final score does.** Runs
   either reached 3 to 4 bpb by step 500 or sat near 5 bpb. `box` never left
   that plateau (0/3). `floor` seed 0 and `shrink` seeds 0 and 2 left it slowly,
   and were still improving at step 5000.
2. **`l1` matches `floor`, not beats it.** Its mean gain of 0.84 bpb comes from
   `floor`'s one stuck seed. Against the two `floor` seeds that trained
   (mean 1.88), `l1` is 0.02 bpb better, well inside the seed spread. Whether
   `l1` really prevents the plateau needs more than 3 seeds.
3. **`box` is clearly worse than `l1`.** The gap is 3.20 bpb, far past 2 sd, and
   it's the only constrained comparison that clears the rule. About 47% of
   coordinates were clipped at the end. The 5 bpb level is likely what byte
   frequencies alone give, but that entropy wasn't measured here, and neither
   was why `box` stalls.
4. **A fixed threshold can do as well as the budget, but less reliably.**
   `shrink`'s one trained seed (1.82) was as good as the best `l1` seed (1.80),
   with the same operation at 3× the speed. Its other two seeds got stuck. One
   seed isn't enough to call this.
5. **The stream didn't settle into a sparse code that fits the budget.** At the
   end of `l1` the projection still changed 100% of tokens, with a median of 35%
   of coordinates nonzero (25% to 43% by layer). Deep layers got denser with
   training: the last layer went from about 12% nonzero at step 0 to about 40%.
6. **Projection cost `l1` about 3× the throughput.** It ran at 0.95M tokens/s
   against about 2.7M, from sorting on every call.

## Predictions vs outcome

| outcome | predicted odds | happened? |
|---|---|---|
| `l1` beats `floor` | ~55% | Not by the rule: 0.84 lower on the mean, tied on trained seeds |
| `box` beats `floor` | ~50% | No: 2.37 worse on the mean |
| `l1` beats `box` (the coupling helps) | ~40% | Yes: 3.20 better, past 2 sd |
| `l1` beats `shrink` (the per-token budget helps, beyond sparsifying) | ~35% | Not by the rule: 1.25 lower on the mean, but `shrink`'s trained seed matched `l1`'s best |
| `shrink` beats `floor` | ~55% | No: 0.41 worse on the mean |
| `l1` closes at least half the gap from `floor` to `ceiling` | ~10% | Not really: 75% on means, but 6% against `floor`'s trained seeds |
| every seed of `l1` and `box` trains without diverging | ~80% | Yes as worded: no non-finite steps, though `box` never learned past the plateau |
| by the end of `l1`, the stream is sparse enough that the projection changes < 20% of tokens | ~25% | No: 100% |
| by the end of `l1`, the median fraction of nonzero coordinates is < 25% | ~35% | No: 35% |

## Next, if revisited

- Run `floor`, `l1` and `shrink` on 10+ more seeds. Each run takes 30 to 90
  seconds, and this settles whether `l1` really avoids the plateau.
- Find out what the plateau is: measure enwik8's byte entropy, and check whether
  a lower learning rate frees the stuck seeds.
- Train longer, to see whether the stuck seeds catch up and whether `l1`
  keeps up with `floor`.

## Prior art

Searched 2026-11-02. No paper found that projects the residual stream onto a
polytope (or an L1 ball) as an architecture choice. The closest work:

- **CRATE** (Yu et al., 2023; CRATE-LM, arXiv 2410.16443): white-box
  transformers where the MLP is replaced by an ISTA step on the stream,
  `ReLU(z + η Dᵀ(z − Dz) − ηλ)`. That is a nonnegative soft-threshold with a
  fixed λ and a learned dictionary D, tested on language modeling. Our `shrink`
  arm is a simplified version of this step, kept as a control.
- **Attention-only white-box transformers** (Wang et al., ICML 2025, arXiv
  2506.03790; AoT-ADMM, arXiv 2608.04213): no MLP, with attention plus a
  fixed-threshold `ReLU(x − λ)` and RMS rescaling. The 2025 paper reports
  language models close to GPT-2. The 2026 one is vision only. This is the
  nearest structure to ours. The difference is that our threshold is set per
  token by a fixed L1 budget (a real projection onto a polytope), not fixed.
- **A Controlled Study of Attention-Only Transformers** (arXiv 2607.18363):
  removing the MLP costs 0.47 nats at equal depth, but only ~0.006 nats at
  equal parameters given enough tokens. So our `ceiling` gap is mostly a
  parameter gap, and the main comparisons are kept at equal parameters.
- **TopK Language Models** (arXiv 2506.21468): TopK on the MLP activations of
  ~1B Llama-style models (MLPs kept). Sparse by construction, but not on the
  stream, and with a fixed k.
- **Weight-sparse transformers** (OpenAI, arXiv 2511.13653): sparse weights
  make residual reads and writes sparse. The paper says it does not enforce
  sparsity of the stream itself.
- **Residual clipping** (arXiv 2601.22966): clipping an unnormalized residual
  stream at ≤ 100 made training diverge. That's a warning for `box`, although
  our stream is normalized before clipping.
- **Projections onto simple polytopes as layers:** sparsemax (simplex, Martins
  & Astudillo, 2016), the L1-ball projection (Duchi et al., 2008), OptNet and
  cvxpylayers (general QPs), simplicial embeddings (Lavoie et al., 2022). None
  of these is used on a transformer's residual stream.
- **nGPT** (Loshchilov et al., ICLR 2025): keeps the stream on a hypersphere.
  It's the smooth analogue of forcing the stream onto a set.

What's new here, as far as the search found: a hard per-token L1 budget on the
residual stream (the threshold adapts to each token), compared with the
fixed-threshold version in an otherwise identical attention-only model.

## Reproduce

Code: `polytope/proj.py` (the projections), `polytope/model.py` (the
attention-only GPT and arms), `train.py` (one run), `scripts/summarize.py`.
Tests (L1 projection vs a brute-force face enumeration, KKT, identity inside,
gradcheck; every arm trains a step):

    python -m pytest -q tests

One run, as on the box (enwik8 split 90/10 by `polytope.data.prepare_text`):

    python train.py --data data/enwik8 --out runs/l1_s0 --arm l1 --seed 0 \
        --steps 5000 --batch_size 64 --block_size 256 --n_layer 6 --n_head 6 \
        --d_model 384 --eval_every 500 --compile

The full sweep (5 arms × seeds 0, 1, 2) is `box/experiment.sh`, launched on a
DigitalOcean GPU with `./launch.sh`. Logs, per-seed metrics and `summary.md` are
on the `results` branch under `out/`. Early `floor_s0` commits there come from a
cancelled first box.
