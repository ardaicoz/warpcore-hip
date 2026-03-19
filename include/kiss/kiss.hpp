#ifndef KISS_KISS_HPP
#define KISS_KISS_HPP

#include <helpers/hip_helpers.hpp>

#include <cstdint>
#include <type_traits>

namespace kiss
{

template<class UInt>
class Kiss
{
    static_assert(
        std::is_same<UInt, std::uint32_t>::value ||
        std::is_same<UInt, std::uint64_t>::value,
        "Kiss only supports uint32_t and uint64_t");

public:
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    explicit Kiss(const std::uint32_t seed = 0) noexcept :
        state_(static_cast<std::uint64_t>(seed) + 0x9e3779b97f4a7c15ULL)
    {}

    template<class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    T next() noexcept
    {
        if constexpr(std::is_same<T, std::uint32_t>::value)
        {
            return static_cast<std::uint32_t>(next_u64());
        }
        else if constexpr(std::is_same<T, std::uint64_t>::value)
        {
            return next_u64();
        }
        else if constexpr(std::is_same<T, float>::value)
        {
            return static_cast<float>((next_u64() >> 40) * (1.0 / (1ULL << 24)));
        }
        else if constexpr(std::is_same<T, double>::value)
        {
            return static_cast<double>((next_u64() >> 11) * (1.0 / (1ULL << 53)));
        }
        else
        {
            return static_cast<T>(next_u64());
        }
    }

private:
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    std::uint64_t next_u64() noexcept
    {
        std::uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    std::uint64_t state_;
};

} // namespace kiss

#endif /* KISS_KISS_HPP */
