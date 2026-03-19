#ifndef WARPCORE_BASE_HPP
#define WARPCORE_BASE_HPP

#include <cstdint>
#include <algorithm>
#include <assert.h>
#include <limits>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <helpers/hip_helpers.hpp>
#include <helpers/packed_types.hpp>
#include <hipcub/hipcub.hpp>

#include "primes.hpp"
#include "detail/warp_utils.hpp"

namespace warpcore
{

namespace cg = cooperative_groups;

using index_t = std::uint64_t;
using status_base_t = std::uint32_t;

namespace detail
{

HOSTQUALIFIER INLINEQUALIFIER
index_t get_valid_capacity(index_t min_capacity, index_t cg_size) noexcept
{
    const auto x = SDIV(min_capacity, cg_size);
    const auto y =
        std::lower_bound(primes.begin(), primes.end(), x);
    return (y == primes.end()) ? 0 : (*y) * cg_size;
}

} // namespace detail

} // namespace warpcore

// TODO move to defaults and expose as constexpr
#if !defined(NDEBUG)
    #ifndef WARPCORE_BLOCKSIZE
    #define WARPCORE_BLOCKSIZE 128
    #endif
#else
    #ifndef WARPCORE_BLOCKSIZE
    #define WARPCORE_BLOCKSIZE MAXBLOCKSIZE // MAXBLOCKSIZE defined in hip_helpers
    #endif
#endif

#include "tags.hpp"
#include "checks.hpp"
#include "status.hpp"
#include "hashers.hpp"
#include "probing_schemes.hpp"
#include "storage.hpp"
#include "defaults.hpp"
#include "gpu_engine.hpp"

#endif /* WARPCORE_BASE_HPP */
