"""Every arm builds, runs forward/backward, the projections bite, and the main arms share a param count."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from polytope.model import ARMS, GPT, GPTConfig, count_params  # noqa: E402


def tiny(arm):
    return GPTConfig(vocab_size=256, block_size=16, n_layer=2, n_head=2, d_model=32, arm=arm)


@pytest.mark.parametrize("arm", ARMS)
def test_arm_forward_backward(arm):
    torch.manual_seed(0)
    m = GPT(tiny(arm))
    x = torch.randint(0, 256, (3, 16))
    m.set_tracking(True)
    _, loss = m(x, x)
    assert torch.isfinite(loss)
    loss.backward()
    for n, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
    stats = {name: p.stats for name, p in m.proj_layers()}
    assert set(stats) == {"emb", "L0", "L1"}
    for st in stats.values():
        if arm in ("floor", "ceiling"):
            assert st["frac_tokens_changed"] == 0.0
        else:
            assert st["frac_tokens_changed"] > 0.5      # dense unit-RMS input: the projection bites
    if arm == "l1":
        assert stats["L0"]["nonzero_median"] < 0.6


def test_equal_params():
    n = {arm: count_params(GPT(tiny(arm))) for arm in ARMS}
    assert len({n[a] for a in ("floor", "box", "l1", "shrink")}) == 1
    assert n["ceiling"] > n["floor"]
