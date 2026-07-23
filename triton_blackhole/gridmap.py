"""Map failing output-tensor indices to Triton program_id / grid coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class GridMapping:
    """Result of mapping a tensor index onto the launch grid."""

    index: tuple[int, ...]
    block_sizes: tuple[int, ...]
    tile_coords: tuple[int, ...]  # per-axis tile index (index // block)
    grid_shape: tuple[int, ...]  # number of tiles along each mapped axis
    program_id: int  # linearized pid (row-major over tile_coords)
    axis_map: tuple[int, ...]  # which tensor axes were mapped


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def index_to_tile(
    index: Sequence[int],
    block_sizes: Sequence[int],
    *,
    axis_map: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """
    Convert a multidimensional output index into tile coordinates.

    ``block_sizes[i]`` applies to tensor axis ``axis_map[i]`` (default: leading axes).
    """
    if axis_map is None:
        axis_map = tuple(range(len(block_sizes)))
    if len(block_sizes) != len(axis_map):
        raise ValueError("block_sizes and axis_map must have the same length")
    tiles: list[int] = []
    for ax, block in zip(axis_map, block_sizes):
        if ax < 0 or ax >= len(index):
            raise IndexError(f"axis_map axis {ax} out of range for index {tuple(index)}")
        if block <= 0:
            raise ValueError(f"block size must be positive, got {block}")
        tiles.append(int(index[ax]) // int(block))
    return tuple(tiles)


def tile_to_program_id(tile_coords: Sequence[int], grid_shape: Sequence[int]) -> int:
    """Row-major linearization of tile coordinates into program_id."""
    if len(tile_coords) != len(grid_shape):
        raise ValueError("tile_coords and grid_shape rank mismatch")
    pid = 0
    for coord, extent in zip(tile_coords, grid_shape):
        if coord < 0 or coord >= extent:
            # Still report a clamped linear id for diagnostics.
            coord = min(max(coord, 0), max(extent - 1, 0))
        pid = pid * extent + coord
    return int(pid)


def program_id_to_tile(program_id: int, grid_shape: Sequence[int]) -> tuple[int, ...]:
    """Inverse of :func:`tile_to_program_id`."""
    coords = [0] * len(grid_shape)
    pid = int(program_id)
    for i in range(len(grid_shape) - 1, -1, -1):
        extent = int(grid_shape[i])
        if extent <= 0:
            raise ValueError("grid_shape extents must be positive")
        coords[i] = pid % extent
        pid //= extent
    return tuple(coords)


def index_to_program_id(
    index: Sequence[int],
    shape: Sequence[int],
    block_sizes: Sequence[int],
    *,
    axis_map: Sequence[int] | None = None,
) -> GridMapping:
    """
    Map a failing output index to the ``program_id`` that owns that tile.

    Parameters
    ----------
    index:
        Multidimensional index into the output tensor (e.g. hotspot from compare).
    shape:
        Full output tensor shape (used to compute grid extents).
    block_sizes:
        Constexpr block sizes along each mapped axis, e.g. ``(BLOCK_M, BLOCK_N)``.
    axis_map:
        Which output axes those blocks cover. Default: ``0..len(block_sizes)-1``.
    """
    if axis_map is None:
        axis_map = tuple(range(len(block_sizes)))
    axis_map = tuple(axis_map)
    block_sizes = tuple(int(b) for b in block_sizes)
    shape = tuple(int(s) for s in shape)
    index = tuple(int(i) for i in index)

    grid_shape = tuple(cdiv(shape[ax], block) for ax, block in zip(axis_map, block_sizes))
    tiles = index_to_tile(index, block_sizes, axis_map=axis_map)
    pid = tile_to_program_id(tiles, grid_shape)
    return GridMapping(
        index=index,
        block_sizes=block_sizes,
        tile_coords=tiles,
        grid_shape=grid_shape,
        program_id=pid,
        axis_map=axis_map,
    )


def format_grid_mapping(m: GridMapping) -> str:
    lines = [
        "======== output → grid mapping ========",
        f"output index   : {list(m.index)}",
        f"block sizes    : {list(m.block_sizes)} (axes {list(m.axis_map)})",
        f"tile coords    : {list(m.tile_coords)}",
        f"grid shape     : {list(m.grid_shape)}",
        f"program_id     : {m.program_id}",
    ]
    return "\n".join(lines)
