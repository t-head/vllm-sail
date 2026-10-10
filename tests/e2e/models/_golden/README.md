# Golden baselines — offline model E2E

Per-chip reference outputs for the offline model-precision suite
(`tests/e2e/models`). Baselines are **captured on real PPU hardware**, never
hand-written; this directory ships empty (only `.gitkeep`) until the first
`--update-golden` run.

## File naming

```
<model_key>_<chip>.json      e.g. qwen3_next_810e.json, qwen3_next_890p.json
```

`chip` is derived from the detected device capability (`(8,0) -> 810e`,
`(8,9) -> 890p`). 810E (int8 / w8a8-int8) and 890P (mxfp4 / w8a8-fp8) take
different quantization paths, so goldens are archived **per chip and never
cross-compared** — `compare_golden` fails if a file's `chip` field does not
match the running device.

## Capturing / refreshing

Run the model test on the target PPU machine with `--update-golden`:

```bash
VLLM_SAIL_MODEL_ROOT=/path/to/checkpoint \
    pytest tests/e2e/models/test_model_qwen3_next.py --update-golden
```

Without `--update-golden`, a missing golden is written once and flagged
*pending-review* (a warning, not a failure); an existing golden is compared.

## Schema (`schema_version: 1`)

```jsonc
{
  "schema_version": 1,
  "model": "qwen3_next",           // <model_key>, from the file name
  "chip": "810e",                  // must equal the running device chip
  "fixed_params": {                // snapshot of the fixed knobs + tolerance
    "max_model_len": 4096, "dtype": "bfloat16",
    "gpu_memory_utilization": 0.90, "enforce_eager": true,
    "seed": 0, "trust_remote_code": true,
    "temperature": 0.0, "max_tokens": 64, "logprobs": 5, "prompt_logprobs": 0,
    "tolerance": {"atol": 1e-3, "rtol": 1e-2}
  },
  "prompts": ["...", "..."],       // the fixed prompt set (order preserved)
  "cases": [
    {
      "prompt": "...",
      "token_ids": [1, 2, 3],      // greedy decode; compared EXACTLY
      "text": "...",
      "completion_logprobs": [     // one entry per generated token
        {"<token_id>": <logprob>}  // top-k map (k = logprobs); JSON keys are strings
      ],
      "prompt_logprobs": [         // one entry per prompt token (first may be {})
        {"<token_id>": <logprob>}
      ]
    }
  ]
}
```

## Comparison rules

1. `token_ids` must match exactly (greedy decode is deterministic).
2. Logprobs are compared per token id present in the golden; a value passes
   when `|actual - expected| <= atol + rtol * |expected|`
   (`atol=1e-3`, `rtol=1e-2`).
3. On failure the assertion reports the **maximum logprob deviation** and the
   first offending case/position/token.
