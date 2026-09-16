// Plugin-owned op registration for the MoE extension (vllm_sail._moe_C).
//
// PPU ops register under the plugin-private namespace _ppu_moe_C.
// Keep the existing operator name and dispatch key for SAIL PyTorch callers.

#include <torch/csrc/stable/library.h>

#include <optional>

void ep_scatter_2_cuda(torch::stable::Tensor hidden_states,
                       std::optional<torch::stable::Tensor> scales_opt,
                       torch::stable::Tensor topk_ids,
                       torch::stable::Tensor expert_start_loc,
                       torch::stable::Tensor output_tensor,
                       torch::stable::Tensor output_index,
                       std::optional<torch::stable::Tensor> output_tensor_scale_opt,
                       bool with_scale);

STABLE_TORCH_LIBRARY_FRAGMENT(_ppu_moe_C, m) {
  m.def(
      "ep_scatter_2_cuda(Tensor hidden_states, "
      "                 Tensor? scales_opt, "
      "                 Tensor topk_ids, "
      "                 Tensor expert_start_loc, "
      "                 Tensor output_tensor, "
      "                 Tensor output_index, "
      "                 Tensor? output_tensor_scale_opt, "
      "                 bool with_scale) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_ppu_moe_C, CUDA, m) {
  m.impl("ep_scatter_2_cuda", TORCH_BOX(&ep_scatter_2_cuda));
}

// Stable-ABI extensions are static-initializer registered; the Python module
// only needs to exist so `import vllm_sail._moe_C` loads this library.
// (Mirrors vllm's csrc/core/registration.h REGISTER_EXTENSION macro.)
#include <Python.h>

extern "C" PyObject* PyInit__moe_C() {
  static struct PyModuleDef module = {
      PyModuleDef_HEAD_INIT, "_moe_C", nullptr, 0, nullptr};
  return PyModule_Create(&module);
}
