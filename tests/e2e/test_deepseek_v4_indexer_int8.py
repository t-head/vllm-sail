# SPDX-License-Identifier: Apache-2.0
"""Numerical PPU 1.0 indexer tests; run explicitly on an 810E host."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.ppu


@pytest.fixture(scope="module")
def runtime():
    import torch

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.platforms import current_platform

    if not (
        current_platform.is_ppu() and current_platform.is_device_capability((8, 0))
    ):
        pytest.skip("requires PPU 1.0 (810E)")
    return torch


def _rope(torch, values, positions, cache):
    result = values.float().clone()
    half = cache.shape[-1] // 2
    cs = cache.float()[positions.long()]
    while cs.ndim < values.ndim:
        cs = cs.unsqueeze(-2)
    even, odd = (
        result[..., -2 * half :: 2].clone(),
        result[..., -2 * half + 1 :: 2].clone(),
    )
    result[..., -2 * half :: 2] = even * cs[..., :half] - odd * cs[..., half:]
    result[..., -2 * half + 1 :: 2] = odd * cs[..., :half] + even * cs[..., half:]
    return result


@pytest.mark.parametrize("tokens", [1, 7, 33])
@pytest.mark.parametrize("buffered", [False, True])
@pytest.mark.parametrize("cache_dtype", ["float32", "bfloat16"])
def test_query_rope_int8_scale_folding(runtime, tokens, buffered, cache_dtype):
    torch = runtime
    # Use the by-value attention alias that appears in the reported traceback.
    from vllm.models.deepseek_v4.attention import fused_indexer_q_rope_quant

    generator = torch.Generator().manual_seed(19)
    q = (torch.randn(tokens, 64, 128, generator=generator) * 0.1).bfloat16()
    q[..., 0] = 2.125  # Stable absmax makes folded-scale assertions exact.
    q[0, 0] = 0  # Exercise the 1e-4 floor.
    positions = torch.arange(tokens) * 2 + 1
    angles = torch.randn(tokens * 2 + 1, 32, generator=generator)
    cache = torch.cat((angles.cos(), angles.sin()), -1).to(getattr(torch, cache_dtype))
    weights = torch.randn(tokens, 64, generator=generator).bfloat16()
    softmax_scale, head_scale = 128**-0.5, 64**-0.5
    rotated = _rope(torch, q, positions, cache)
    rotated[..., -64:] = rotated[..., -64:].bfloat16().float()
    scale = rotated.abs().amax(-1).clamp_min(1e-4) / 127
    expected_weights = weights.float() * scale * softmax_scale * head_scale
    buffers = (
        (
            torch.empty(tokens * 2, 64, 128, dtype=torch.int8, device="cuda")[::2],
            torch.empty(tokens * 2, 64, dtype=torch.float32, device="cuda")[::2],
        )
        if buffered
        else None
    )
    actual_q, actual_weights = fused_indexer_q_rope_quant(
        positions.cuda(),
        q.cuda(),
        cache.cuda(),
        weights.cuda(),
        softmax_scale,
        head_scale,
        output_buffers=buffers,
    )
    torch.cuda.synchronize()
    assert actual_q.dtype == torch.int8
    if buffered:
        assert actual_q is buffers[0] and actual_weights is buffers[1]
    torch.testing.assert_close(
        actual_weights.cpu(), expected_weights, rtol=2e-6, atol=1e-10
    )
    expected_q = (rotated / scale[..., None]).to(torch.int8)
    # Fused GPU arithmetic can cross a BF16 or integer truncation boundary.
    torch.testing.assert_close(actual_q.cpu(), expected_q, rtol=0, atol=1)
    assert (actual_q[0, 0] == 0).all()
    torch.testing.assert_close(
        actual_q.cpu().float() * scale[..., None],
        rotated,
        rtol=0,
        atol=float(scale.max()) * 1.05,
    )


@pytest.mark.parametrize("ratio,overlap", [(4, True), (8, False)])
def test_compressed_int8_keys_paged_layout_and_skips(runtime, ratio, overlap):
    torch = runtime
    from vllm.models.deepseek_v4.compressor import compress_norm_rope_store_triton

    generator = torch.Generator().manual_seed(43)
    head, rope, block_size, cache_block_size = 128, 64, 4, 4
    width = head * (1 + overlap)
    positions = torch.tensor(
        [ratio - 1, 2 * ratio - 1, ratio, 2 * ratio - 1, ratio - 1]
    )
    reqs = torch.tensor([0, 1, 0, 0, 1], dtype=torch.int32)
    slots = torch.tensor([0, 1, 2, -1, 4])
    kv_slots = torch.tensor([1, 6, 2, 3, -1])
    blocks_per_req = 2 * ratio // block_size
    table = torch.randperm(blocks_per_req * 2, generator=generator).reshape(2, -1).int()
    state = torch.randn(blocks_per_req * 2, block_size, width * 2, generator=generator)
    rms_w = torch.randn(head, generator=generator).bfloat16()
    angles = torch.randn(2 * ratio, rope // 2, generator=generator)
    cache = torch.cat((angles.cos(), angles.sin()), -1)
    k_cache = torch.full(
        (2, cache_block_size, head + 4), 0xA5, dtype=torch.uint8, device="cuda"
    )
    expected = k_cache.cpu().clone().reshape(2, -1)
    value_refs, scale_refs = {}, {}
    for i, position in enumerate(positions.tolist()):
        if slots[i] < 0 or kv_slots[i] < 0 or (position + 1) % ratio:
            continue
        kv_rows, scores = [], []
        start = position - (1 + overlap) * ratio + 1
        for offset in range((1 + overlap) * ratio):
            pos = start + offset
            if pos < 0:
                continue
            block = table[reqs[i], pos // block_size]
            head_offset = int(offset >= ratio) * head
            row = state[block, pos % block_size]
            kv_rows.append(row[head_offset : head_offset + head])
            scores.append(row[width + head_offset : width + head_offset + head])
        compressed = (torch.stack(kv_rows) * torch.stack(scores).softmax(0)).sum(0)
        normed = (
            compressed * torch.rsqrt(compressed.square().mean() + 1e-6) * rms_w.float()
        )
        rotated = _rope(
            torch, normed[None], torch.tensor([position // ratio * ratio]), cache
        )[0]
        rotated = rotated.bfloat16().float()
        scale = rotated.abs().max().clamp_min(1e-4) / 127
        quant = (rotated / scale).clamp(-127, 127).to(torch.int8)
        slot = int(kv_slots[i])
        page, row = divmod(slot, cache_block_size)
        expected[page, row * head : (row + 1) * head] = quant.view(torch.uint8)
        scale_offset = cache_block_size * head + row * 4
        expected[page, scale_offset : scale_offset + 4] = scale.reshape(1).view(
            torch.uint8
        )
        value_refs[slot], scale_refs[slot] = quant, scale
    compress_norm_rope_store_triton(
        state.cuda(),
        len(positions),
        reqs.cuda(),
        positions.cuda(),
        slots.cuda(),
        table.cuda(),
        block_size,
        width,
        cache.cuda(),
        k_cache,
        SimpleNamespace(slot_mapping=kv_slots.cuda()),
        {},
        head,
        rope,
        ratio,
        overlap,
        False,
        rms_w.cuda(),
        1e-6,
        head,
        head,
        4,
    )
    torch.cuda.synchronize()
    actual = k_cache.cpu().reshape(2, -1)
    for slot, quant in value_refs.items():
        page, row = divmod(slot, cache_block_size)
        values = actual[page, row * head : (row + 1) * head]
        torch.testing.assert_close(values.view(torch.int8), quant, rtol=0, atol=1)
        scale_offset = cache_block_size * head + row * 4
        scales = actual[page, scale_offset : scale_offset + 4]
        torch.testing.assert_close(
            scales.view(torch.float32)[0], scale_refs[slot], rtol=1e-2, atol=1e-8
        )
        # Exclude written bytes; all padding, other pages and skipped slots
        # must retain the sentinel exactly.
        expected[page, row * head : (row + 1) * head] = values
        expected[page, scale_offset : scale_offset + 4] = scales
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
