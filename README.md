# polytope

**Question:** in a transformer with no MLP, if the residual stream is forced
onto a polytope after every attention update, does that help the model, and
what kind of code does the stream settle into?

**Answer (at this scale):** _pending: the run hasn't happened yet._

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
- 3 seeds per arm, one 48GB GPU, about 1.5 hours in all. A difference counts only
  if it exceeds twice the seed standard deviation.

## Predictions

Written down before any run:

| outcome | predicted odds |
|---|---|
| `l1` beats `floor` | ~55% |
| `box` beats `floor` | ~50% |
| `l1` beats `box` (the coupling helps) | ~40% |
| `l1` beats `shrink` (the per-token budget helps, beyond sparsifying) | ~35% |
| `shrink` beats `floor` | ~55% |
| `l1` closes at least half the gap from `floor` to `ceiling` | ~10% |
| every seed of `l1` and `box` trains without diverging | ~80% |
| by the end of `l1`, the stream is sparse enough that the projection changes < 20% of tokens | ~25% |
| by the end of `l1`, the median fraction of nonzero coordinates is < 25% | ~35% |

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

_To fill in after the build._
