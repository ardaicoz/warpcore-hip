#ifndef HELPERS_HIP_HELPERS_HPP
#define HELPERS_HIP_HELPERS_HPP

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#define HOSTQUALIFIER __host__
#define DEVICEQUALIFIER __device__
#define GLOBALQUALIFIER __global__
#define HOSTDEVICEQUALIFIER __host__ __device__
#define INLINEQUALIFIER __forceinline__

#ifndef SDIV
#define SDIV(x, y) (((x) + (y) - 1) / (y))
#endif

#ifndef MAXBLOCKSIZE
#define MAXBLOCKSIZE 256
#endif

inline constexpr hipMemcpyKind H2D = hipMemcpyHostToDevice;
inline constexpr hipMemcpyKind D2H = hipMemcpyDeviceToHost;

namespace helpers
{

inline void check_hip(
    const hipError_t error,
    const char* const expr,
    const char* const file,
    const int line) noexcept
{
    if(error == hipSuccess)
    {
        return;
    }

    std::fprintf(
        stderr,
        "HIP error at %s:%d for %s: %s (%s)\n",
        file,
        line,
        expr,
        hipGetErrorName(error),
        hipGetErrorString(error));
    std::abort();
}

inline void check_last_hip_error(
    const char* const file,
    const int line) noexcept
{
    check_hip(hipGetLastError(), "hipGetLastError()", file, line);
}

template<class Func>
GLOBALQUALIFIER
void lambda_kernel(Func func)
{
    func();
}

HOSTDEVICEQUALIFIER INLINEQUALIFIER
std::uint64_t global_thread_id() noexcept
{
    return static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

template<class T>
DEVICEQUALIFIER INLINEQUALIFIER
T atomicAggInc(T* const value) noexcept
{
    return atomicAdd(value, T{1});
}

inline std::size_t available_gpu_memory() noexcept
{
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    check_hip(hipMemGetInfo(&free_bytes, &total_bytes), "hipMemGetInfo", __FILE__, __LINE__);
    return free_bytes;
}

} // namespace helpers

#define HIP_CHECK(expr) ::helpers::check_hip((expr), #expr, __FILE__, __LINE__)
#define HIPERR ::helpers::check_last_hip_error(__FILE__, __LINE__)

#endif /* HELPERS_HIP_HELPERS_HPP */
