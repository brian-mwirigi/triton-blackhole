import torch

from triton_blackhole.classify import DriftKind, classify_drift
from triton_blackhole.probe import ProbeBank


def test_classify_match():
    t = torch.randn(8, 8)
    c = classify_drift(t, t.clone())
    assert c.kind == DriftKind.MATCH


def test_classify_nonfinite():
    a = torch.tensor([1.0, float("nan")])
    b = torch.tensor([1.0, 2.0])
    c = classify_drift(a, b)
    assert c.kind == DriftKind.NONFINITE


def test_classify_localized():
    ref = torch.zeros(128, 128)
    act = ref.clone()
    act[10, 10] = 50.0
    c = classify_drift(act, ref, atol=1e-6, rtol=1e-6)
    assert c.kind == DriftKind.LOCALIZED_BUG


def test_probe_first_divergence():
    bank = ProbeBank()
    bank.capture("s0", torch.ones(4), side="tri")
    bank.capture("s0", torch.ones(4), side="ref")
    bank.capture("s1", torch.ones(4) * 2, side="tri")
    bank.capture("s1", torch.ones(4), side="ref")
    first = bank.first_divergence(atol=1e-6, rtol=1e-6)
    assert first is not None
    assert first[0] == "s1"
    report = bank.report(atol=1e-6, rtol=1e-6)
    assert "first divergence at stage: 's1'" in report
