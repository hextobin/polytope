"""The L1-ball projection against a brute-force reference, KKT, identity inside, gradcheck."""
import itertools
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from polytope.proj import l1_tau, proj_box, proj_l1, shrink  # noqa: E402


def brute_l1(z, r):
    """Enumerate the faces of the L1 ball in z's orthant: y = z if inside, else for each
    support S, y_S = z_S - lam sign(z_S) with |y|_1 = r; keep feasible ones, take the nearest."""
    if z.abs().sum() <= r:
        return z.clone()
    d = z.numel()
    best, best_dist = None, float("inf")
    for k in range(1, d + 1):
        for S in itertools.combinations(range(d), k):
            S = list(S)
            s = torch.sign(z[S])
            lam = (z[S].abs().sum() - r) / k
            yS = z[S] - lam * s
            if (torch.sign(yS) != s).any() or (s == 0).any():
                continue
            y = torch.zeros_like(z)
            y[S] = yS
            dist = (y - z).pow(2).sum().item()
            if dist < best_dist:
                best, best_dist = y, dist
    return best


@pytest.mark.parametrize("seed", range(20))
def test_l1_matches_brute_force(seed):
    g = torch.Generator().manual_seed(seed)
    d = 6
    z = torch.randn(d, generator=g, dtype=torch.float64) * 2
    r = float(torch.rand(1, generator=g)) * 4 + 0.1
    y = proj_l1(z, r)
    assert torch.allclose(y, brute_l1(z, r), atol=1e-10)


@pytest.mark.parametrize("d", [8, 64, 384])
def test_l1_feasible_and_kkt(d):
    torch.manual_seed(d)
    z = torch.randn(256, d, dtype=torch.float64)
    r = d / 4
    y = proj_l1(z, r)
    l1 = y.abs().sum(-1)
    assert (l1 <= r + 1e-9).all()
    lam = l1_tau(z, r).squeeze(-1)
    outside = z.abs().sum(-1) > r
    assert torch.allclose(l1[outside], torch.full_like(l1[outside], r), atol=1e-9)
    # KKT: z - y = lam * g, g in the subdifferential of |.|_1 at y, lam >= 0
    res = z - y
    nz = y != 0
    lam_b = lam[:, None].expand_as(z)
    assert torch.allclose(res[nz], (lam_b * torch.sign(y))[nz], atol=1e-9)
    assert (res[~nz].abs() <= lam_b[~nz] + 1e-9).all()
    assert (lam >= 0).all()
    # at d = 384, r = d/4 and a unit-RMS Gaussian input, about 44% of coords survive
    if d == 384:
        assert 0.40 < nz.double().mean().item() < 0.48


def test_l1_identity_inside():
    torch.manual_seed(0)
    z = torch.randn(50, 16, dtype=torch.float64)
    z = z / z.abs().sum(-1, keepdim=True) * torch.rand(50, 1, dtype=torch.float64) * 4  # |z|_1 < 4
    assert torch.equal(proj_l1(z, 4.0), z)
    edge = z / z.abs().sum(-1, keepdim=True) * 4.0
    assert torch.allclose(proj_l1(edge, 4.0), edge, atol=1e-12)


def test_l1_float32_close_to_float64():
    torch.manual_seed(1)
    z = torch.randn(64, 384, dtype=torch.float64)
    assert torch.allclose(proj_l1(z.float(), 96.0).double(), proj_l1(z, 96.0), atol=1e-4)


def test_gradcheck():
    torch.manual_seed(0)
    z = (torch.randn(4, 8, dtype=torch.float64) * 2).requires_grad_()
    z_in = (torch.randn(3, 8, dtype=torch.float64) * 0.1).requires_grad_()   # inside: identity
    assert torch.autograd.gradcheck(lambda t: proj_l1(t, 2.0), (z,))
    assert torch.autograd.gradcheck(lambda t: proj_l1(t, 2.0), (z_in,))
    assert torch.autograd.gradcheck(proj_box, (z,))
    assert torch.autograd.gradcheck(lambda t: shrink(t, 0.78), (z,))


def test_shrink_and_box():
    z = torch.tensor([-2.0, -0.5, 0.0, 0.5, 0.9, 3.0])
    assert torch.allclose(shrink(z), torch.tensor([-1.22, 0.0, 0.0, 0.0, 0.12, 2.22]))
    assert torch.equal(proj_box(z), torch.tensor([-1.0, -0.5, 0.0, 0.5, 0.9, 1.0]))
