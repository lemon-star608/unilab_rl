#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>

namespace {

// Gather packed rows from a host-locked (cudaHostRegisterMapped) source into a
// contiguous device destination. With UVA the host pointer is directly
// dereferenceable by the kernel; bytes cross the bus once, on demand.
template <int VEC>
__global__ void gather_rows_uva_kernel(
    const char* __restrict__ src,
    char* __restrict__ dst,
    const int64_t* __restrict__ indices,
    int64_t num_indices,
    int64_t row_bytes) {
  const int64_t chunks_per_row = (row_bytes + VEC - 1) / VEC;
  const int64_t total_chunks = num_indices * chunks_per_row;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       t < total_chunks;
       t += stride) {
    const int64_t i = t / chunks_per_row;
    const int64_t c = t % chunks_per_row;
    const int64_t byte_off = c * VEC;
    const char* s = src + indices[i] * row_bytes + byte_off;
    char* d = dst + i * row_bytes + byte_off;
    const int64_t n = (row_bytes - byte_off) < static_cast<int64_t>(VEC)
                          ? (row_bytes - byte_off)
                          : static_cast<int64_t>(VEC);
    if (n == VEC) {
      if constexpr (VEC == 16) {
        *reinterpret_cast<uint4*>(d) = *reinterpret_cast<const uint4*>(s);
      } else if constexpr (VEC == 4) {
        *reinterpret_cast<uint32_t*>(d) = *reinterpret_cast<const uint32_t*>(s);
      } else {
        *d = *s;
      }
    } else {
      for (int64_t b = 0; b < n; ++b) {
        d[b] = s[b];
      }
    }
  }
}

}  // namespace

void gather_rows_uva(
    std::uintptr_t src_host_ptr,
    torch::Tensor dst,
    torch::Tensor indices,
    std::uintptr_t stream_handle) {
  if (!dst.is_cuda()) {
    throw std::runtime_error("UVA gather destination must be a CUDA tensor");
  }
  if (!dst.is_contiguous()) {
    throw std::runtime_error("UVA gather destination must be contiguous");
  }
  if (indices.scalar_type() != torch::kInt64 || !indices.is_cuda() ||
      !indices.is_contiguous()) {
    throw std::runtime_error("UVA gather indices must be contiguous int64 CUDA tensor");
  }
  const int64_t row_bytes = dst.size(1) * dst.element_size();
  if (row_bytes <= 0) {
    throw std::runtime_error("UVA gather destination rows must be non-empty");
  }
  const int64_t num_indices = indices.numel();
  if (num_indices == 0) {
    return;
  }

  auto stream = c10::cuda::CUDAStream::unpack3(
      static_cast<int64_t>(stream_handle),
      dst.device().index(),
      c10::DeviceType::CUDA);
  c10::cuda::CUDAStreamGuard guard(stream);

  const char* src = reinterpret_cast<const char*>(src_host_ptr);
  char* dst_ptr = reinterpret_cast<char*>(dst.data_ptr());
  const int64_t* index_ptr = indices.data_ptr<int64_t>();

  auto launch = [&](auto vec_tag) {
    constexpr int VEC = decltype(vec_tag)::value;
    const int64_t chunks_per_row = (row_bytes + VEC - 1) / VEC;
    const int64_t total_chunks = num_indices * chunks_per_row;
    const int block = 256;
    int64_t grid = (total_chunks + block - 1) / block;
    if (grid > 65535) {
      grid = 65535;
    }
    gather_rows_uva_kernel<VEC><<<static_cast<unsigned int>(grid), block, 0, stream>>>(
        src, dst_ptr, index_ptr, num_indices, row_bytes);
  };

  // Widest aligned vector wins; rows start at base + idx * row_bytes, so both
  // the base pointer and the row stride must preserve the alignment.
  const bool aligned16 = (src_host_ptr % 16 == 0) && (row_bytes % 16 == 0) &&
                         (reinterpret_cast<std::uintptr_t>(dst_ptr) % 16 == 0);
  const bool aligned4 = (src_host_ptr % 4 == 0) && (row_bytes % 4 == 0) &&
                        (reinterpret_cast<std::uintptr_t>(dst_ptr) % 4 == 0);
  if (aligned16) {
    launch(std::integral_constant<int, 16>{});
  } else if (aligned4) {
    launch(std::integral_constant<int, 4>{});
  } else {
    launch(std::integral_constant<int, 1>{});
  }

  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("UVA gather kernel launch failed: ") + cudaGetErrorString(err));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "gather_rows_uva",
      &gather_rows_uva,
      "Gather packed rows from host-locked memory into a CUDA tensor via UVA");
}
