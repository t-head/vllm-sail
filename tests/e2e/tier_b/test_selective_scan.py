# SPDX-License-Identifier: Apache-2.0
"""Mamba1 prefill oracles, including state indirection and partial APC chunks."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.ppu


@pytest.fixture(scope="module")
def runtime():
    import torch

    from vllm_sail.native import install
    from vllm_sail.native.extensions import import_kernels

    import_kernels(strict=True)
    install()
    return torch


def reference(torch, u, delta, a, b, c, d, z, bias, initial):
    """Independent CPU recurrence; retain every state for APC boundary checks."""
    u, delta, a, b, c, d, bias = [x.cpu().float() for x in (u, delta, a, b, c, d, bias)]
    z = z.cpu().float() if z is not None else None
    delta = torch.nn.functional.softplus(delta + bias[:, None])
    b = b.repeat_interleave(u.shape[0] // b.shape[0], dim=0)
    c = c.repeat_interleave(u.shape[0] // c.shape[0], dim=0)
    state = initial.cpu().float().clone()
    outputs, states = [], []
    for token in range(u.shape[1]):
        dt = delta[:, token, None]
        state = (dt * a).exp() * state + dt * b[:, :, token] * u[:, token, None]
        out = (state * c[:, :, token]).sum(-1) + d * u[:, token]
        if z is not None:
            out *= torch.nn.functional.silu(z[:, token])
        outputs.append(out)
        states.append(state.clone())
    return torch.stack(outputs, -1), states


@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("has_z", [False, True])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("fp32_state", [False, True])
def test_scan_grouped_bc_initial_state_and_null_slot(
    runtime, dtype_name, has_z, varlen, fp32_state
):
    torch = runtime
    torch.manual_seed(11)
    dtype = getattr(torch, dtype_name)
    state_dtype = torch.float32 if fp32_state else dtype
    lengths = [129, 2051, 3] if varlen else [128, 128, 128]
    dim, groups, dstate = 4, 2, 8
    total = sum(lengths)
    shape = (dim, total) if varlen else (3, dim, lengths[0])
    bc_shape = (groups, dstate, total) if varlen else (3, groups, dstate, lengths[0])
    u = torch.randn(shape, device="cuda", dtype=dtype) * 0.2
    delta = torch.randn_like(u) * 0.1
    z = torch.randn_like(u) if has_z else None
    a = -torch.rand(dim, dstate, device="cuda") * 0.5
    b = torch.randn(bc_shape, device="cuda", dtype=dtype) * 0.2
    c = torch.randn_like(b) * 0.2
    d, bias = torch.randn(dim, device="cuda"), torch.randn(dim, device="cuda")
    state = torch.randn(6, dim, dstate, device="cuda", dtype=state_dtype)
    saved_state, saved_delta = state.clone(), delta.clone()
    saved_z = z.clone() if has_z else None
    slots = torch.tensor([2, 4, -1], device="cuda", dtype=torch.int32)
    initial = torch.tensor([True, False, False], device="cuda")
    starts = [0, lengths[0], lengths[0] + lengths[1], total]
    query = torch.tensor(starts, device="cuda", dtype=torch.int32) if varlen else None
    torch.ops._C.selective_scan_fwd(
        u,
        delta,
        a,
        b,
        c,
        d,
        z,
        bias,
        True,
        query,
        slots,
        initial,
        state,
        -1,
        2048,
        None,
        None,
        None,
        None,
        None,
    )
    torch.cuda.synchronize()
    actual = z if has_z else delta
    for batch, slot in enumerate([2, 4]):
        key = (..., slice(starts[batch], starts[batch + 1])) if varlen else batch
        ref, states = reference(
            torch,
            u[key],
            saved_delta[key],
            a,
            b[key],
            c[key],
            d,
            saved_z[key] if has_z else None,
            bias,
            saved_state[slot] if batch == 0 else torch.zeros_like(saved_state[slot]),
        )
        tolerance = 0.03 if dtype == torch.bfloat16 else 0.003
        torch.testing.assert_close(
            actual[key].cpu().float(),
            ref.to(dtype).float(),
            rtol=tolerance,
            atol=tolerance,
        )
        torch.testing.assert_close(
            state[slot].cpu().float(),
            states[-1].to(state_dtype).float(),
            rtol=tolerance,
            atol=tolerance,
        )
    skipped = (..., slice(starts[2], starts[3])) if varlen else 2
    original = saved_z if has_z else saved_delta
    torch.testing.assert_close(actual[skipped], original[skipped], rtol=0, atol=0)
    torch.testing.assert_close(
        state[[0, 1, 3, 5]], saved_state[[0, 1, 3, 5]], rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_scan_apc_preserves_partial_chunk_state_boundaries(runtime, dtype_name):
    torch = runtime
    torch.manual_seed(41)
    dtype = getattr(torch, dtype_name)
    dim, groups, dstate, total = 4, 2, 8, 1284
    u = torch.randn(dim, total, device="cuda", dtype=dtype) * 0.2
    delta = torch.randn_like(u) * 0.1
    z = torch.randn_like(u)
    b = torch.randn(groups, dstate, total, device="cuda", dtype=dtype) * 0.2
    c = torch.randn_like(b) * 0.2
    a = -torch.rand(dim, dstate, device="cuda")
    d, bias = torch.randn(dim, device="cuda"), torch.randn(dim, device="cuda")
    state = torch.randn(8, dim, dstate, device="cuda")
    saved_state, saved_delta, saved_z = state.clone(), delta.clone(), z.clone()

    def ints(values):
        return torch.tensor(values, device="cuda", dtype=torch.int32)

    # Sequence 0 fills the last three tokens of an existing 1024-token block,
    # then a whole block. Its initial state is in physical slot 5; writes go
    # to slots 1 and 3. Sequence 1 starts from zero and writes only slot 2.
    torch.ops._C.selective_scan_fwd(
        u,
        delta,
        a,
        b,
        c,
        d,
        z,
        bias,
        True,
        ints([0, 1027, total]),
        ints([[1, 3, 5], [2, 4, 6]]),
        torch.tensor([True, False], device="cuda"),
        state,
        -1,
        1024,
        ints([0, 0]),
        ints([1, 0]),
        ints([2, 2]),
        ints([0, 3, 1027, total]),
        ints([1, 2]),
    )
    torch.cuda.synchronize()
    expected_state = saved_state.cpu()
    tolerance = 0.03 if dtype == torch.bfloat16 else 0.003
    for start, end, initial_slot, writes in [
        (0, 1027, 5, [(2, 1), (1026, 3)]),
        (1027, total, None, [(256, 2)]),
    ]:
        ref, states = reference(
            torch,
            u[:, start:end],
            saved_delta[:, start:end],
            a,
            b[..., start:end],
            c[..., start:end],
            d,
            saved_z[:, start:end],
            bias,
            saved_state[initial_slot]
            if initial_slot is not None
            else torch.zeros_like(state[0]),
        )
        torch.testing.assert_close(
            z[:, start:end].cpu().float(),
            ref.to(dtype).float(),
            rtol=tolerance,
            atol=tolerance,
        )
        for token, slot in writes:
            expected_state[slot] = states[token]
    torch.testing.assert_close(
        state.cpu(), expected_state, rtol=tolerance, atol=tolerance
    )
