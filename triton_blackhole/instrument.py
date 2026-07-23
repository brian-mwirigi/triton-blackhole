"""
AST instrumentation for Triton kernels.

Injects failing-``program_id``-gated debug stores for named intermediates so
only one block writes to global memory (avoids OOM / device_print floods).
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Sequence


# Per-probe payload written by the injected dump (floats).
PROBE_FLOATS = 4  # [max, sum, first, numel_proxy]


@dataclass
class InstrumentResult:
    """Instrumented kernel plus metadata for the launcher."""

    kernel: Any  # triton.JITFunction
    probe_names: tuple[str, ...]
    source: str
    floats_per_probe: int = PROBE_FLOATS

    @property
    def debug_slots(self) -> int:
        return len(self.probe_names) * self.floats_per_probe


class _ProbeInjector(ast.NodeTransformer):
    """
    After ``name = <expr>`` where ``name`` is a probed identifier, inject:

        if tl.program_id(0) == _BH_FAILING_PID:
            _bh_i = <slot>
            tl.store(_bh_dbg_ptr + _bh_i + 0, tl.max(tl.ravel(name)))
            ...
    """

    def __init__(self, probes: Sequence[str]) -> None:
        self.probes = set(probes)
        self.slots = {name: i for i, name in enumerate(probes)}
        self.injected = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        # Ensure tl is available in the kernel namespace (usually already imported
        # via triton.language as tl in the module; kernels typically use tl.*).
        new_args = list(node.args.args)
        # Append instrumentation parameters if missing.
        existing = {a.arg for a in new_args}
        additions: list[ast.arg] = []
        if "_bh_dbg_ptr" not in existing:
            additions.append(ast.arg(arg="_bh_dbg_ptr", annotation=None))
        if "_BH_FAILING_PID" not in existing:
            additions.append(
                ast.arg(
                    arg="_BH_FAILING_PID",
                    annotation=ast.Attribute(
                        value=ast.Name(id="tl", ctx=ast.Load()),
                        attr="constexpr",
                        ctx=ast.Load(),
                    ),
                )
            )
        node.args.args = new_args + additions

        # Rewrite body with probe dumps after matching assigns.
        node.body = self._inject_body(node.body)
        return node

    def _inject_body(self, body: list[ast.stmt]) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for stmt in body:
            out.append(stmt)
            names = _assigned_names(stmt)
            for name in names:
                if name in self.probes:
                    out.extend(self._dump_stmts(name))
                    self.injected += 1
        return out

    def _dump_stmts(self, name: str) -> list[ast.stmt]:
        slot = self.slots[name] * PROBE_FLOATS
        # Build: if tl.program_id(0) == _BH_FAILING_PID: <stores>
        pid_call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="tl", ctx=ast.Load()),
                attr="program_id",
                ctx=ast.Load(),
            ),
            args=[ast.Constant(value=0)],
            keywords=[],
        )
        test = ast.Compare(
            left=pid_call,
            ops=[ast.Eq()],
            comparators=[ast.Name(id="_BH_FAILING_PID", ctx=ast.Load())],
        )

        def store_at(offset: int, value: ast.expr) -> ast.Expr:
            return ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="tl", ctx=ast.Load()),
                        attr="store",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.BinOp(
                            left=ast.Name(id="_bh_dbg_ptr", ctx=ast.Load()),
                            op=ast.Add(),
                            right=ast.Constant(value=slot + offset),
                        ),
                        value,
                    ],
                    keywords=[],
                )
            )

        ravel = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="tl", ctx=ast.Load()),
                attr="ravel",
                ctx=ast.Load(),
            ),
            args=[ast.Name(id=name, ctx=ast.Load())],
            keywords=[],
        )
        vmax = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="tl", ctx=ast.Load()),
                attr="max",
                ctx=ast.Load(),
            ),
            args=[ravel],
            keywords=[],
        )
        # Re-ravel for sum (AST nodes must not be reused).
        ravel2 = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="tl", ctx=ast.Load()),
                attr="ravel",
                ctx=ast.Load(),
            ),
            args=[ast.Name(id=name, ctx=ast.Load())],
            keywords=[],
        )
        vsum = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="tl", ctx=ast.Load()),
                attr="sum",
                ctx=ast.Load(),
            ),
            args=[ravel2],
            keywords=[],
        )
        ravel3 = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="tl", ctx=ast.Load()),
                attr="ravel",
                ctx=ast.Load(),
            ),
            args=[ast.Name(id=name, ctx=ast.Load())],
            keywords=[],
        )
        first = ast.Subscript(
            value=ravel3,
            slice=ast.Constant(value=0),
            ctx=ast.Load(),
        )
        # numel proxy: 1.0 marker that dump ran
        marker = ast.Constant(value=1.0)

        body = [
            store_at(0, vmax),
            store_at(1, vsum),
            store_at(2, first),
            store_at(3, marker),
        ]
        return [ast.If(test=test, body=body, orelse=[])]


def _assigned_names(stmt: ast.stmt) -> list[str]:
    names: list[str] = []
    if isinstance(stmt, ast.Assign):
        for t in stmt.targets:
            if isinstance(t, ast.Name):
                names.append(t.id)
    elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        names.append(stmt.target.id)
    return names


def kernel_source(fn: Any) -> str:
    """Return dedented source of a Python function or Triton JITFunction."""
    py_fn = getattr(fn, "fn", fn)
    return textwrap.dedent(inspect.getsource(py_fn))


def transform_kernel_source(source: str, probes: Sequence[str]) -> tuple[str, int]:
    """
    Transform kernel source; return (new_source, num_injections).

    Strips a leading ``@triton.jit`` (and similar) decorator lines from the
    parsed function so we can re-apply ``triton.jit`` after exec.
    """
    tree = ast.parse(source)
    func = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func = node
            break
    if func is None:
        raise ValueError("no function definition found in kernel source")

    # Drop decorators — caller re-applies triton.jit.
    func.decorator_list = []
    injector = _ProbeInjector(probes)
    func = injector.visit(func)
    ast.fix_missing_locations(func)
    new_mod = ast.Module(body=[func], type_ignores=[])
    return ast.unparse(new_mod), injector.injected


def instrument_kernel(
    kernel: Any,
    probes: Sequence[str],
    *,
    jit: bool = True,
) -> InstrumentResult:
    """
    Clone a Triton kernel with failing-pid probe dumps for ``probes``.

    The returned kernel expects extra args:
    ``_bh_dbg_ptr`` (pointer) and ``_BH_FAILING_PID`` (constexpr int).
    """
    if not probes:
        raise ValueError("probes must be non-empty")
    src = kernel_source(kernel)
    new_src, n = transform_kernel_source(src, probes)
    if n == 0:
        raise ValueError(
            f"no assignments to probe names {list(probes)} found in kernel source; "
            "probe identifiers must match local names assigned in the kernel body"
        )

    # Execute in a namespace that has triton.language as tl.
    import triton
    import triton.language as tl

    ns: dict[str, Any] = {
        "triton": triton,
        "tl": tl,
        "torch": __import__("torch"),
    }
    # Carry over globals from the original kernel module when possible.
    py_fn = getattr(kernel, "fn", kernel)
    mod = inspect.getmodule(py_fn)
    if mod is not None:
        ns.update({k: v for k, v in vars(mod).items() if not k.startswith("__")})

    exec(compile(new_src, filename="<triton_blackhole_instrumented>", mode="exec"), ns)
    # Find the function we just defined.
    func_name = ast.parse(new_src).body[0].name  # type: ignore[union-attr]
    new_fn = ns[func_name]
    if jit:
        new_fn = triton.jit(new_fn)
    return InstrumentResult(
        kernel=new_fn,
        probe_names=tuple(probes),
        source=new_src,
        floats_per_probe=PROBE_FLOATS,
    )


def decode_probe_buffer(
    buf: Any,
    probe_names: Sequence[str],
    *,
    floats_per_probe: int = PROBE_FLOATS,
) -> dict[str, dict[str, float]]:
    """Decode the flat float debug buffer into per-probe stats."""
    import torch

    if not isinstance(buf, torch.Tensor):
        buf = torch.as_tensor(buf)
    flat = buf.detach().float().reshape(-1).cpu()
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(probe_names):
        base = i * floats_per_probe
        chunk = flat[base : base + floats_per_probe]
        if chunk.numel() < floats_per_probe:
            break
        out[name] = {
            "max": float(chunk[0]),
            "sum": float(chunk[1]),
            "first": float(chunk[2]),
            "dumped": float(chunk[3]),
        }
    return out


def format_probe_dump(decoded: dict[str, dict[str, float]], *, failing_pid: int) -> str:
    lines = [
        "======== AST probe dump (failing block only) ========",
        f"program_id     : {failing_pid}",
    ]
    if not decoded:
        lines.append("(no probe data)")
        return "\n".join(lines)
    for name, stats in decoded.items():
        marker = "ok" if stats.get("dumped", 0) != 0 else "MISSING"
        lines.append(
            f"  {name}: max={stats['max']:.6g}  sum={stats['sum']:.6g}  "
            f"first={stats['first']:.6g}  [{marker}]"
        )
    return "\n".join(lines)
