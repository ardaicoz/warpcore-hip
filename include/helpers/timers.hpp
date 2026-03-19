#ifndef HELPERS_TIMERS_HPP
#define HELPERS_TIMERS_HPP

#include "hip_helpers.hpp"

#include <iostream>
#include <string>
#include <utility>

namespace helpers
{

class GpuTimer
{
public:
    explicit GpuTimer(std::string label) :
        label_(std::move(label)),
        elapsed_ms_(0.0f),
        stopped_(false)
    {
        HIP_CHECK(hipEventCreate(&start_));
        HIP_CHECK(hipEventCreate(&stop_));
        HIP_CHECK(hipEventRecord(start_, nullptr));
    }

    ~GpuTimer()
    {
        hipEventDestroy(start_);
        hipEventDestroy(stop_);
    }

    float elapsed_ms() const
    {
        stop_if_needed();
        return elapsed_ms_;
    }

    void print() const
    {
        std::cout << label_ << ": " << elapsed_ms() << " ms" << std::endl;
    }

    void print_throughput(
        const std::size_t bytes_per_item,
        const std::size_t count) const
    {
        const double seconds = static_cast<double>(elapsed_ms()) / 1000.0;
        const double total_bytes =
            static_cast<double>(bytes_per_item) * static_cast<double>(count);
        const double gib_per_second =
            (seconds > 0.0) ? total_bytes / seconds / static_cast<double>(1ULL << 30) : 0.0;

        std::cout
            << label_
            << ": "
            << elapsed_ms_
            << " ms, "
            << gib_per_second
            << " GiB/s"
            << std::endl;
    }

private:
    void stop_if_needed() const
    {
        if(stopped_)
        {
            return;
        }

        HIP_CHECK(hipEventRecord(stop_, nullptr));
        HIP_CHECK(hipEventSynchronize(stop_));
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms_, start_, stop_));
        stopped_ = true;
    }

    std::string label_;
    hipEvent_t start_;
    hipEvent_t stop_;
    mutable float elapsed_ms_;
    mutable bool stopped_;
};

} // namespace helpers

#endif /* HELPERS_TIMERS_HPP */
