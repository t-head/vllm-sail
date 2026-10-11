# SPDX-License-Identifier: Apache-2.0
"""Small independent PyTorch oracles for the MiniMax MSA adapter tests."""

import torch


def make_case(device, query_lens, layout="HND"):
    torch.manual_seed(812)
    lengths = [129, 2305, 257]
    starts = [0]
    for n in query_lens:
        starts.append(starts[-1] + n)
    iq = torch.randn(starts[-1], 2, 128, device=device, dtype=torch.bfloat16)
    q = torch.randn(starts[-1], 4, 128, device=device, dtype=torch.bfloat16)
    ik = torch.randn(40, 128, 1, 128, device=device, dtype=torch.bfloat16)
    if layout == "HND":
        kv = torch.randn(40, 2, 128, 256, device=device, dtype=torch.bfloat16)
    else:
        kv = torch.randn(
            40, 128, 2, 256, device=device, dtype=torch.bfloat16
        ).transpose(1, 2)
    # Independent physical page ids for the index cache and main cache.
    index_table = torch.stack(
        [torch.randperm(40, device=device)[:20] for _ in lengths]
    ).int()
    main_table = torch.stack(
        [torch.randperm(40, device=device)[:20] for _ in lengths]
    ).int()
    return starts, lengths, iq, q, ik, kv, index_table, main_table


def score_reference(q, k_pages, physical_pages, kv_length):
    k = (
        k_pages[physical_pages.long()]
        .reshape(-1, k_pages.shape[2], 128)[:kv_length, 0]
        .float()
    )
    prefix = kv_length - q.shape[0]
    result = torch.full(
        (q.shape[1], (kv_length + 127) // 128, q.shape[0]),
        -float("inf"),
        device=q.device,
    )
    for t in range(q.shape[0]):
        visible = prefix + t + 1
        logits = q[t].float() @ k[:visible].T
        for b in range((visible + 127) // 128):
            result[:, b, t] = logits[:, b * 128 : min((b + 1) * 128, visible)].amax(-1)
    return result


def topk_reference(iq, ik, table, starts, lengths):
    out = torch.full(
        (iq.shape[0], iq.shape[1], 16), -1, dtype=torch.int32, device=iq.device
    )
    for r, length in enumerate(lengths):
        lo, hi = starts[r : r + 2]
        scores = score_reference(
            iq[lo:hi], ik, table[r, : (length + 127) // 128], length
        )
        for t in range(hi - lo):
            pages = (length - (hi - lo) + t) // 128 + 1
            scores[:, 0, t] = 1e30
            scores[:, pages - 1, t] = 1e29
            selected = scores[:, :pages, t].topk(min(16, pages), dim=-1).indices
            out[lo + t, :, : selected.shape[1]] = selected.int()
    return out


def attend_reference(q, k, v, table, starts, lengths, topk, scale):
    out = torch.empty_like(q)
    for r, length in enumerate(lengths):
        lo, hi = starts[r : r + 2]
        ids = table[r, : (length + 127) // 128].long()
        kr = k[ids].reshape(-1, k.shape[2], 128)[:length].float()
        vr = v[ids].reshape(-1, v.shape[2], 128)[:length].float()
        for t in range(lo, hi):
            visible = length - (hi - lo) + (t - lo) + 1
            tokens = torch.arange(visible, device=q.device)
            for h in range(q.shape[1]):
                kh = h // (q.shape[1] // k.shape[2])
                keep = (tokens[:, None] // 128 == topk[t, kh][None, :]).any(-1)
                logits = (kr[:visible, kh] @ q[t, h].float()) * scale
                logits.masked_fill_(~keep, -float("inf"))
                out[t, h] = logits.softmax(0) @ vr[:visible, kh]
    return out
