import torch

from triton_blackhole.compare import assert_close, compare
from triton_blackhole.tolerances import dtype_tolerances, suggest_tolerances


def test_compare_pass():
    a = torch.randn(8, 8)
    r = compare(a, a.clone())
    assert r.passed
    assert r.n_mismatch == 0


def test_compare_localizes_hotspot():
    ref = torch.zeros(16, 16)
    act = ref.clone()
    act[3, 7] = 5.0
    r = compare(act, ref, atol=1e-6, rtol=1e-6)
    assert not r.passed
    assert r.max_abs_index == (3, 7)
    assert r.hotspots[0].index == (3, 7)
    assert r.n_mismatch == 1


def test_bf16_default_atol_not_float32():
    tol = dtype_tolerances(torch.bfloat16)
    assert tol.atol >= 1e-3
    a = torch.randn(32, 32, dtype=torch.bfloat16)
    # tiny noise within bf16 envelope
    b = a + 1e-3
    r = compare(a, b)  # uses bf16 defaults
    # may or may not pass depending on values; just ensure no crash and atol set
    assert r.atol == tol.atol


def test_assert_close_message():
    a = torch.ones(4)
    b = torch.zeros(4)
    try:
        assert_close(a, b, atol=1e-8, rtol=1e-8)
        assert False, "expected AssertionError"
    except AssertionError as e:
        assert "triton-blackhole numerical report" in str(e)


def test_suggest_tolerances():
    ref = torch.randn(100, 100)
    act = ref + 1e-4
    sug = suggest_tolerances(act, ref)
    assert sug.atol > 0 and sug.rtol > 0
