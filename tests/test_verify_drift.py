import ast

from triton_blackhole.gridmap import (
    index_to_program_id,
    program_id_to_tile,
    tile_to_program_id,
)
from triton_blackhole.instrument import transform_kernel_source


def test_index_to_program_id_2d():
    # shape 128x128, blocks 32x32 → 4x4 grid
    # index [50, 51] → tile (1, 1) → pid = 1*4+1 = 5
    m = index_to_program_id((50, 51), (128, 128), (32, 32))
    assert m.tile_coords == (1, 1)
    assert m.grid_shape == (4, 4)
    assert m.program_id == 5
    assert program_id_to_tile(5, (4, 4)) == (1, 1)


def test_tile_linearization_roundtrip():
    grid = (3, 5, 2)
    for pid in range(3 * 5 * 2):
        tile = program_id_to_tile(pid, grid)
        assert tile_to_program_id(tile, grid) == pid


def test_ast_injects_failing_pid_guard():
    src = '''
def kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    acc = tl.load(x_ptr + offs)
    acc = acc + 1
    tl.store(out_ptr + offs, acc)
'''
    new_src, n = transform_kernel_source(src, ["acc"])
    assert n >= 1
    assert "_BH_FAILING_PID" in new_src
    assert "_bh_dbg_ptr" in new_src
    assert "program_id" in new_src
    # parses cleanly
    ast.parse(new_src)


def test_verify_drift_decorator_raises_with_mapping():
    import torch
    from triton_blackhole import verify_drift

    def ref(x):
        return x * 2

    @verify_drift(ref, block_sizes=(8,), raise_on_fail=True)
    def tri(x):
        y = x * 2
        y = y.clone()
        y[20] = 999
        return y

    x = torch.zeros(32)
    try:
        tri(x)
        assert False, "expected AssertionError"
    except AssertionError as e:
        msg = str(e)
        assert "drift artifact" in msg
        assert "program_id" in msg
        art = tri.last_artifact
        assert art is not None and not art.passed
        assert art.grid is not None
        assert art.grid.program_id == 20 // 8


def test_run_drift_verify_pass():
    import torch
    from triton_blackhole import run_drift_verify

    t = torch.randn(8, 8)
    art = run_drift_verify(t, t.clone(), block_sizes=(4, 4))
    assert art.passed
