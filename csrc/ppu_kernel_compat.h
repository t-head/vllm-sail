// SPDX-License-Identifier: Apache-2.0
#pragma once

// Torch isolates SDK CUB in CUB_WRAPPED_NAMESPACE. Keep that isolation while
// allowing upstream and plugin HGGC kernels to spell cub:: as usual.
#include <cub/util_namespace.cuh>
#ifdef CUB_WRAPPED_NAMESPACE
namespace cub = CUB_NS_QUALIFIER;
#endif
