# SPDX-License-Identifier: Apache-2.0
"""Optional CPU tensor tests; the normal dependency-free gate skips this file.

The library double exercises planner lifetimes, causal chunk offsets, layout,
and page-table wiring. It is not evidence that SAIL kernels run on a device.
"""

import pytest

torch = pytest.importorskip("torch")

from tests.support.msa import (  # noqa: E402
    attend_reference,
    make_case,
    score_reference,
    topk_reference,
)
from vllm_sail.attention import msa  # noqa: E402


class EagerMSA:
    def __init__(self):
        self.plan_live = False

    def fmha_sm100_plan(self, qo, kv, heads, **kwargs):
        assert not self.plan_live, "a later plan overwrote an unconsumed workspace"
        self.plan_live = True
        return qo.tolist(), kv.tolist()

    def fmha_sm100(
        self,
        q,
        k,
        v,
        plan,
        *,
        kv_indices,
        sm_scale,
        output_maxscore,
        output_o=True,
        max_score=None,
        kv_block_indexes=None,
        out=None,
    ):
        qo, kv = plan
        assert self.plan_live
        self.plan_live = False
        q0, page0 = 0, 0
        for n, length in zip(qo, kv, strict=False):
            pages = (length + 127) // 128
            physical = kv_indices[page0 : page0 + pages]
            if output_maxscore:
                scores = score_reference(q[q0 : q0 + n], k, physical, length) * sm_scale
                max_score[:, : scores.shape[1], q0 : q0 + n] = scores
            else:
                out[q0 : q0 + n] = attend_reference(
                    q[q0 : q0 + n],
                    k,
                    v,
                    physical[None, :],
                    [0, n],
                    [length],
                    kv_block_indexes[q0 : q0 + n],
                    sm_scale,
                )
            q0 += n
            page0 += pages
        assert q0 == q.shape[0] and page0 == kv_indices.numel()

    def sparse_topk_select(self, scores, topk, *, num_valid_pages, output):
        assert scores.shape[1] == num_valid_pages
        output.copy_(scores.permute(2, 0, 1).topk(topk, dim=-1).indices)


@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("query_lens", [[1, 1, 1], [0, 5, 130], [1, 3, 129]])
def test_complete_adapter_with_independent_cache_tables(
    monkeypatch, layout, query_lens
):
    monkeypatch.setattr(msa, "load_msa", lambda: library)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    library = EagerMSA()
    starts, lengths, iq, q, ik, kv, it, mt = make_case("cpu", query_lens, layout)
    # Three tokens/chunk forces splits within requests and across page boundaries.
    chunks = msa.make_chunks(starts, lengths, 2, 3 * 2048)
    topk = torch.full((q.shape[0] + 4, 2, 16), -99, dtype=torch.int32)
    msa.run_chunks(
        query=iq,
        key=ik,
        value=ik,
        block_table=it,
        chunks=chunks,
        scale=128**-0.5,
        topk=topk,
        init_blocks=1,
        local_blocks=1,
    )
    expected = topk_reference(iq, ik, it, starts, lengths)
    assert torch.equal(topk[: q.shape[0]].sort(-1).values, expected.sort(-1).values)
    assert torch.all(topk[q.shape[0] :] == -99)
    k, v = msa.main_kv_views(kv)
    assert k.untyped_storage().data_ptr() == kv.untyped_storage().data_ptr()
    assert v.untyped_storage().data_ptr() == kv.untyped_storage().data_ptr()
    output = torch.empty_like(q)
    msa.run_chunks(
        query=q,
        key=k,
        value=v,
        block_table=mt,
        chunks=chunks,
        scale=128**-0.5,
        topk=topk,
        output=output,
    )
    ref = attend_reference(q, k, v, mt, starts, lengths, expected, 128**-0.5)
    torch.testing.assert_close(output, ref, atol=0, rtol=0)


def test_forced_windows_and_invalid_topk():
    scores = torch.zeros(2, 128, 3)
    pages = torch.tensor([1, 2, 4], dtype=torch.int32)
    msa.force_local_scores(scores, pages, 1, 1)
    assert scores[0, 0, 0] == pytest.approx(1e29)
    assert scores[0, 0, 1] == pytest.approx(1e30)
    assert scores[0, 1, 1] == pytest.approx(1e29)
    assert torch.isneginf(scores[:, 4:]).all()
    selected = torch.tensor([[[3, -1, 0, 8]], [[1, 0, -1, 9]]], dtype=torch.int32)
    assert msa.sorted_blocks(selected, torch.tensor([4, 2])).tolist() == [
        [[0, 3, -1, -1]],
        [[0, 1, -1, -1]],
    ]


def test_capture_is_rejected(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="enforce-eager"):
        msa.run_chunks(
            query=None,
            key=None,
            value=None,
            block_table=None,
            chunks=[],
            scale=1,
            topk=None,
        )
