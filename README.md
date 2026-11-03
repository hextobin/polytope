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
  its vertices are dense ±1 codes. It's the control for the L1 ball's coupling.

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
  - `ceiling`: floor plus a GELU MLP sublayer. It has more parameters and is a
    reference only.
- **Also logged, per layer:** how often the projection changes the vector, the
  fraction of coordinates that are nonzero (`l1`) or clipped (`box`), and
  throughput.
- 3 seeds per arm, one 48GB GPU, about an hour in all. A difference counts only
  if it exceeds twice the seed standard deviation.

## Predictions

Written down before any run:

| outcome | predicted odds |
|---|---|
| `l1` beats `floor` | ~55% |
| `box` beats `floor` | ~50% |
| `l1` beats `box` (the coupling helps) | ~40% |
| `l1` closes at least half the gap from `floor` to `ceiling` | ~10% |
| every seed of `l1` and `box` trains without diverging | ~80% |
| by the end of `l1`, the stream is sparse enough that the projection changes < 20% of tokens | ~25% |
| by the end of `l1`, the median fraction of nonzero coordinates is < 25% | ~35% |

## Prior art

To check before launch: OptNet (Amos & Kolter, 2017); sparsemax and entmax
(projections onto the simplex); the polytope lens (Black et al., 2022). No
search for this exact setup has been done yet.

## Reproduce

_To fill in after the build._
