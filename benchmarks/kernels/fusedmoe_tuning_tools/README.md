# Fused MoE Tuning

The tuning driver compiles candidate Triton configurations, profiles the fused
MoE kernel with the SAIL SDK's `asys` tool, and writes the selected configurations
as JSON. It requires a PPU environment with vLLM, SAIL PyTorch, Triton, pandas,
NumPy and tqdm installed.

## Run

Run from this directory so the driver can locate `run_fusedmoe_target.py`:

```bash
cd benchmarks/kernels/fusedmoe_tuning_tools
export CUDA_VISIBLE_DEVICES=0

python3 faster_autotuning_for_vllm_fusedmoe.py \
  --model MODEL_KEY \
  --tp-size 4 \
  --input-metadata /path/to/input_metadata.json \
  --batch-size 128 4096 \
  --dtype bfloat16 \
  --quant-config int8_w8a8
```

Replace `MODEL_KEY` with a key in your input metadata file. The metadata supplies
the model configuration and expert token distributions for the requested batch
sizes. Set the tensor-parallel size, precision and quantization to match the
workload. The driver accepts `int8_w8a8` and `fp8_w8a8`; omit `--quant-config`
for an unquantized run.

## Outputs

The driver writes candidate configurations, `tune_report.asysrep`, CSV profiling
data and a selected-configuration JSON file in this directory. Runs reuse these
filenames, so save results before starting another run. Review the selected
configuration and verify correctness on the target device before adding it to
the package's tuning data.
