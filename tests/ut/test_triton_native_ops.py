# SPDX-License-Identifier: Apache-2.0
"""Registration and FP8 arithmetic checks without a PPU or vLLM install."""

from __future__ import annotations

import ast
import math
import sys
import types
from pathlib import Path

import pytest

from vllm_sail.native import stubs, triton_ops


@pytest.mark.parametrize("fail_at", [None, "CUDA", "Meta"])
def test_merge_registration_is_repeatable_and_preserves_schema(monkeypatch, fail_at):
    torch = types.ModuleType("torch")
    torch.ops = types.SimpleNamespace(_C=types.SimpleNamespace())
    defined, implementations = {}, {}

    class Library:
        def __init__(self, namespace, kind):
            assert namespace == "_C" and kind == "FRAGMENT"

        def define(self, schema):
            name = schema.split("(", 1)[0]
            assert name not in defined
            defined[name] = schema
            setattr(torch.ops._C, name, types.SimpleNamespace(default=object()))

        def impl(self, name, function, key):
            nonlocal fail_at
            if key == fail_at:
                fail_at = None
                raise RuntimeError("injected registration failure")
            implementations[name, key] = function

        def _destroy(self):
            for name in defined:
                delattr(torch.ops._C, name)
            defined.clear()
            implementations.clear()

    torch.library = types.SimpleNamespace(Library=Library)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(triton_ops, "_libraries", [])
    schema = next(op for op in stubs.load_schemas() if op.name == "merge_attn_states")
    monkeypatch.setattr(stubs, "load_schemas", lambda *args: (schema,))
    if fail_at is not None:
        with pytest.raises(RuntimeError, match="injected registration failure"):
            triton_ops.install()
        assert not defined and not implementations and not triton_ops._libraries
    assert triton_ops.install() == ("_C::merge_attn_states",)
    assert defined == {schema.name: schema.schema}
    assert implementations[schema.name, "CUDA"] is triton_ops.merge_attn_states
    assert implementations[schema.name, "Meta"]() is None
    assert triton_ops.install() == ()
    assert stubs.install() == ()  # A capability stub must not replace the kernel.
    assert len(triton_ops._libraries) == 1


def test_merge_preserves_the_native_argument_order(monkeypatch):
    module = types.ModuleType("vllm_sail.attention.ops.merge_attn_states")
    calls = []
    module.merge_attn_states = lambda *args: calls.append(args)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    arguments = tuple(object() for _ in range(8))
    triton_ops.merge_attn_states(*arguments)
    assert calls == [arguments]


@pytest.fixture
def kernel_math():
    """Execute kernel math with NumPy; this does not test Triton compilation."""
    np = pytest.importorskip("numpy")

    class Array(np.ndarray):
        def to(self, dtype, bitcast=False):
            return (self.view(dtype) if bitcast else self.astype(dtype)).view(Array)

    def array(value, dtype=None):
        return np.asarray(value, dtype=dtype).view(Array)

    class Pointer:
        def __init__(self, storage, offset=0):
            self.storage, self.offset = storage, np.asarray(offset)

        def __add__(self, offset):
            return Pointer(self.storage, self.offset + offset)

    def load(pointer, mask=True, other=0):
        # Mask out-of-range lanes before accessing the host array.
        index = np.where(mask, pointer.offset, 0)
        return array(np.where(mask, pointer.storage[index], other))

    def store(pointer, value, mask=True):
        index, value = np.broadcast_arrays(pointer.offset, value)
        pointer.storage[index[mask]] = value[mask]

    tl = types.SimpleNamespace(
        constexpr=object(),
        float32=np.float32,
        uint32=np.uint32,
        int32=np.int32,
        uint8=np.uint8,
        abs=np.abs,
        minimum=np.minimum,
        maximum=np.maximum,
        exp=np.exp,
        log=np.log,
        arange=lambda start, stop: array(np.arange(start, stop)),
        load=load,
        store=store,
        where=lambda *args: array(np.where(*args)),
    )
    path = Path(__file__).parents[2] / "vllm_sail/attention/ops/merge_attn_states.py"
    functions = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_encode_e4m3fn", "_merge_kernel")
    ]
    for fn in functions:
        fn.decorator_list = []
    namespace = {"tl": tl}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return np, array, Pointer, tl, namespace


def test_software_e4m3fn_rounding_matches_the_format(kernel_math):
    np, array, _, _, namespace = kernel_math
    # Independently enumerate the positive E4M3FN representable values.
    representable = [
        math.ldexp(code & 7, -9)
        if code < 8
        else math.ldexp(1 + (code & 7) / 8, (code >> 3) - 7)
        for code in range(127)
    ]
    values, expected = [], []
    for code, value in enumerate(representable):
        for sign in (0, 128):
            values.append(-value if sign else value)
            expected.append(code | sign)
        if code == 126:
            continue
        middle = np.float32((value + representable[code + 1]) / 2)
        for sample, selected in [
            (np.nextafter(middle, np.float32(-np.inf)), code),
            (middle, code + (code & 1)),
            (np.nextafter(middle, np.float32(np.inf)), code + 1),
        ]:
            for sign in (0, 128):
                values.append(-sample if sign else sample)
                expected.append(selected | sign)
    values.extend([float("inf"), float("-inf"), float("nan"), 1e30, -1e30])
    expected.extend([126, 254, 127, 126, 254])
    with np.errstate(invalid="ignore", over="ignore"):
        actual = namespace["_encode_e4m3fn"](array(values, np.float32))
    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=np.uint8))


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("write_lse", [False, True])
def test_merge_math_masks_strides_and_empty_context(kernel_math, fp8, write_lse):
    np, _, pointer, tl, namespace = kernel_math
    # Four tokens, two heads and 13 active lanes in a padded 16-lane tile.
    pre_strides, suf_strides, out_strides = (40, 16, 1), (68, 32, 2), (36, 16, 1)
    p = np.full(160, np.nan, dtype=np.float32)
    s = np.full(272, np.nan, dtype=np.float32)
    out = np.full(144, 99, dtype=np.uint8 if fp8 else np.float32)
    lp, ls = np.empty(8, dtype=np.float32), np.empty(8, dtype=np.float32)
    ol = np.full(8, 99, dtype=np.float32)
    for token in range(4):
        for head in range(2):
            factor = head + 1
            pre = [2 * factor, np.nan, np.nan, 100][token]
            suf = [10 * factor, 7 * factor, np.nan, -5 * factor][token]
            p[token * 40 + head * 16 + np.arange(13)] = pre
            s[token * 68 + head * 32 + np.arange(13) * 2] = suf
            lp[token * 2 + head] = [math.log(3), -np.inf, np.inf, 100][token]
            ls[token * 2 + head] = [0, math.log(2), -np.inf, math.log(4)][token]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        for token in range(4):
            for head in range(2):
                tl.program_id = lambda axis, token=token, head=head: (
                    token if axis == 0 else head
                )
                namespace["_merge_kernel"](
                    pointer(out),
                    pointer(ol),
                    pointer(p),
                    pointer(lp),
                    pointer(s),
                    pointer(ls),
                    pointer(np.array([2], dtype=np.float32)),
                    out_strides,
                    pre_strides,
                    suf_strides,
                    (1, 2),
                    (1, 2),
                    (1, 2),
                    3,
                    13,
                    16,
                    write_lse,
                    fp8,
                )
    expected = np.full_like(out, 99)
    values = (
        [[0x40, 0x48], [0x46, 0x4E], [0, 0], [0xC2, 0xCA]]
        if fp8
        else [[4, 8], [7, 14], [0, 0], [-5, -10]]
    )
    for token in range(4):
        for head in range(2):
            expected[token * 36 + head * 16 + np.arange(13)] = values[token][head]
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
    expected_lse = (
        np.repeat([math.log(4), math.log(2), -np.inf, math.log(4)], 2)
        if write_lse
        else np.full(8, 99)
    )
    np.testing.assert_allclose(ol, expected_lse, rtol=1e-6, atol=1e-6)
