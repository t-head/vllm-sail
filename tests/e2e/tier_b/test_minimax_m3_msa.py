# SPDX-License-Identifier: Apache-2.0
"""Real SAIL MSA correctness gate. A missing/incompatible library fails on PPU."""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("requires a real PPU device", allow_module_level=True)

from tests.support.msa import attend_reference, make_case, topk_reference  # noqa: E402
from vllm_sail.attention import msa  # noqa: E402

pytestmark = pytest.mark.ppu


@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("query_lens", [[1, 1, 1], [0, 5, 130], [1, 3, 129]])
@torch.inference_mode()
def test_sail_msa_index_and_attend(layout, query_lens):
    msa.load_msa()
    starts, lengths, iq, q, ik, kv, it, mt = make_case("cuda", query_lens, layout)
    chunks = msa.make_chunks(starts, lengths, 2, 64 * 2048)
    topk = torch.full((q.shape[0] + 4, 2, 16), -99, dtype=torch.int32, device="cuda")
    index_chunks = msa.prepare_chunks(chunks, it)
    main_chunks = msa.prepare_chunks(chunks, mt)
    msa.run_indexer(
        query=iq,
        key=ik,
        chunks=index_chunks,
        scale=128**-0.5,
        topk=topk,
        init_blocks=1,
        local_blocks=1,
    )
    expected_topk = topk_reference(iq, ik, it, starts, lengths)
    assert torch.equal(
        topk[: q.shape[0]].sort(-1).values, expected_topk.sort(-1).values
    )
    assert torch.all(topk[q.shape[0] :] == -99)
    k, v = msa.main_kv_views(kv)
    out = torch.empty_like(q)
    msa.run_sparse_attention(
        query=q,
        key=k,
        value=v,
        chunks=main_chunks,
        scale=128**-0.5,
        topk=topk,
        output=out,
    )
    reference = attend_reference(q, k, v, mt, starts, lengths, expected_topk, 128**-0.5)
    torch.testing.assert_close(out, reference, atol=2e-2, rtol=2e-2)
