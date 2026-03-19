#ifndef WARPCORE_DETAIL_WARP_UTILS_HPP
#define WARPCORE_DETAIL_WARP_UTILS_HPP

#include <cstdint>

namespace warpcore
{
namespace detail
{

using lane_mask_t = std::uint64_t;

template<class Group>
DEVICEQUALIFIER INLINEQUALIFIER
lane_mask_t ballot_mask(const Group& group, const bool predicate) noexcept
{
    return static_cast<lane_mask_t>(group.ballot(predicate));
}

DEVICEQUALIFIER INLINEQUALIFIER
int first_set_bit(const lane_mask_t mask) noexcept
{
    return __ffsll(static_cast<long long>(mask)) - 1;
}

DEVICEQUALIFIER INLINEQUALIFIER
int last_set_bit(const lane_mask_t mask) noexcept
{
    return (mask == 0) ? -1 : 63 - __clzll(mask);
}

DEVICEQUALIFIER INLINEQUALIFIER
int popcount(const lane_mask_t mask) noexcept
{
    return __popcll(mask);
}

DEVICEQUALIFIER INLINEQUALIFIER
lane_mask_t single_lane_mask(const unsigned int lane) noexcept
{
    return lane_mask_t{1} << lane;
}

DEVICEQUALIFIER INLINEQUALIFIER
lane_mask_t prefix_lane_mask(const unsigned int lane) noexcept
{
    return (lane == 0) ? 0 : (single_lane_mask(lane) - 1);
}

DEVICEQUALIFIER INLINEQUALIFIER
lane_mask_t clear_lane(const lane_mask_t mask, const unsigned int lane) noexcept
{
    return mask & ~single_lane_mask(lane);
}

} // namespace detail
} // namespace warpcore

#endif /* WARPCORE_DETAIL_WARP_UTILS_HPP */
