import torch

from triton_blackhole.bisect import bisect_axes, bisect_tiles


def test_bisect_axes_finds_corner():
    ref = torch.zeros(64, 64)
    act = ref.clone()
    act[50, 51] = 10.0
    result = bisect_axes(act, ref, atol=1e-8, rtol=1e-8, min_volume=1)
    assert result.found
    assert result.minimal_slice is not None
    sl = result.minimal_slice.as_slices()
    # Region must contain the bad element.
    assert act[sl].numel() >= 1
    assert float(act[sl].abs().max()) == 10.0
    assert result.minimal_slice.volume() <= 4  # should shrink tightly


def test_bisect_no_divergence():
    t = torch.randn(16, 16)
    result = bisect_axes(t, t.clone())
    assert not result.found


def test_bisect_tiles_isolates_pid():
    ref = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    bad_pid = 2

    def launch(lo, hi):
        out = ref.clone()
        # each pid corresponds to one row
        for pid in range(lo, hi):
            if pid == bad_pid:
                out[pid] = out[pid] + 100
        return out

    result = bisect_tiles(launch, ref, num_programs=4, atol=0, rtol=0)
    assert result.found
    assert result.pid_lo <= bad_pid < result.pid_hi
    assert result.pid_hi - result.pid_lo == 1
