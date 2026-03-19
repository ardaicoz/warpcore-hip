#ifndef HELPERS_PACKED_TYPES_HPP
#define HELPERS_PACKED_TYPES_HPP

#include "hip_helpers.hpp"

#include <cstdint>
#include <type_traits>

namespace packed_types
{

namespace detail
{

HOSTDEVICEQUALIFIER INLINEQUALIFIER
constexpr std::uint64_t bit_mask(const std::uint64_t bits) noexcept
{
    return (bits >= 64) ? ~std::uint64_t{0} : ((std::uint64_t{1} << bits) - 1);
}

template<std::uint64_t Offset, std::uint64_t Bits>
HOSTDEVICEQUALIFIER INLINEQUALIFIER
constexpr std::uint64_t extract(const std::uint64_t value) noexcept
{
    return (value >> Offset) & bit_mask(Bits);
}

template<std::uint64_t Offset, std::uint64_t Bits>
HOSTDEVICEQUALIFIER INLINEQUALIFIER
constexpr std::uint64_t insert(
    const std::uint64_t storage,
    const std::uint64_t value) noexcept
{
    const auto mask = bit_mask(Bits) << Offset;
    return (storage & ~mask) | ((value & bit_mask(Bits)) << Offset);
}

} // namespace detail

template<std::uint64_t FirstBits, std::uint64_t SecondBits>
class PackedPair
{
    static_assert(FirstBits + SecondBits <= 64, "PackedPair exceeds 64 bits");

public:
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr PackedPair() noexcept : storage_(0) {}

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr PackedPair(
        const std::uint64_t first_value,
        const std::uint64_t second_value) noexcept :
        storage_(0)
    {
        first(first_value);
        second(second_value);
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr std::uint64_t first() const noexcept
    {
        return detail::extract<0, FirstBits>(storage_);
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr std::uint64_t second() const noexcept
    {
        return detail::extract<FirstBits, SecondBits>(storage_);
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void first(const std::uint64_t value) noexcept
    {
        storage_ = detail::insert<0, FirstBits>(storage_, value);
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void second(const std::uint64_t value) noexcept
    {
        storage_ = detail::insert<FirstBits, SecondBits>(storage_, value);
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool operator==(const PackedPair other) const noexcept
    {
        return storage_ == other.storage_;
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool operator!=(const PackedPair other) const noexcept
    {
        return !(*this == other);
    }

private:
    DEVICEQUALIFIER INLINEQUALIFIER
    explicit constexpr PackedPair(const std::uint64_t storage, const int) noexcept :
        storage_(storage)
    {}

    std::uint64_t storage_;

    DEVICEQUALIFIER INLINEQUALIFIER
    friend PackedPair atomicExch(
        PackedPair* const address,
        const PackedPair value) noexcept
    {
        return PackedPair(
            atomicExch(
                reinterpret_cast<unsigned long long*>(&address->storage_),
                static_cast<unsigned long long>(value.storage_)),
            0);
    }
};

template<
    std::uint64_t FirstBits,
    std::uint64_t SecondBits,
    std::uint64_t ThirdBits,
    std::uint64_t FourthBits>
class PackedQuadruple
{
    static_assert(
        FirstBits + SecondBits + ThirdBits + FourthBits <= 64,
        "PackedQuadruple exceeds 64 bits");

public:
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr PackedQuadruple() noexcept : storage_(0) {}

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr std::uint64_t first() const noexcept
    {
        return detail::extract<0, FirstBits>(storage_);
    }

    template<class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr T first_as() const noexcept
    {
        return static_cast<T>(first());
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr std::uint64_t second() const noexcept
    {
        return detail::extract<FirstBits, SecondBits>(storage_);
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr std::uint64_t third() const noexcept
    {
        return detail::extract<FirstBits + SecondBits, ThirdBits>(storage_);
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr std::uint64_t fourth() const noexcept
    {
        return detail::extract<FirstBits + SecondBits + ThirdBits, FourthBits>(storage_);
    }

    template<class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void first(const T value) noexcept
    {
        storage_ = detail::insert<0, FirstBits>(storage_, static_cast<std::uint64_t>(value));
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void second(const std::uint64_t value) noexcept
    {
        storage_ = detail::insert<FirstBits, SecondBits>(storage_, value);
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void third(const std::uint64_t value) noexcept
    {
        storage_ = detail::insert<FirstBits + SecondBits, ThirdBits>(storage_, value);
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void fourth(const std::uint64_t value) noexcept
    {
        storage_ = detail::insert<FirstBits + SecondBits + ThirdBits, FourthBits>(storage_, value);
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool operator==(const PackedQuadruple other) const noexcept
    {
        return storage_ == other.storage_;
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool operator!=(const PackedQuadruple other) const noexcept
    {
        return !(*this == other);
    }

private:
    DEVICEQUALIFIER INLINEQUALIFIER
    explicit constexpr PackedQuadruple(const std::uint64_t storage, const int) noexcept :
        storage_(storage)
    {}

    std::uint64_t storage_;

    DEVICEQUALIFIER INLINEQUALIFIER
    friend PackedQuadruple atomicCAS(
        PackedQuadruple* const address,
        const PackedQuadruple compare,
        const PackedQuadruple value) noexcept
    {
        return PackedQuadruple(
            atomicCAS(
                reinterpret_cast<unsigned long long*>(&address->storage_),
                static_cast<unsigned long long>(compare.storage_),
                static_cast<unsigned long long>(value.storage_)),
            0);
    }

    DEVICEQUALIFIER INLINEQUALIFIER
    friend PackedQuadruple atomicExch(
        PackedQuadruple* const address,
        const PackedQuadruple value) noexcept
    {
        return PackedQuadruple(
            atomicExch(
                reinterpret_cast<unsigned long long*>(&address->storage_),
                static_cast<unsigned long long>(value.storage_)),
            0);
    }
};

} // namespace packed_types

#endif /* HELPERS_PACKED_TYPES_HPP */
