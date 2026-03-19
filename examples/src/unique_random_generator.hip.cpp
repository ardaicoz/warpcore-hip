#include <iostream>
#include <numeric>
#include <warpcore/random_distributions.hpp>
#include <helpers/hip_helpers.hpp>
#include <helpers/timers.hpp>

// This example shows the easy generation of gigabytes of unique random values
// in only a few milliseconds using WarpCore's BloomFilter.

/*! \brief checks if input values are unique
* \tparam T input data type
* \param[in] in_d device pointer to the input values
* \param[in] n number of input values
* \return true iff values are unique
*/
template<class T>
HOSTQUALIFIER INLINEQUALIFIER
bool check_unique(T * in_d, std::uint64_t n) noexcept
{
    // allocate memory for the sorted values
    T * out_d = nullptr;
    hipMalloc(&out_d, sizeof(T)*n); HIPERR

    // sort the generated values using CUB
    void * tmp_d = nullptr;
    std::size_t tmp_bytes = 0;
    hipcub::DeviceRadixSort::SortKeys(tmp_d, tmp_bytes, in_d, out_d, n); HIPERR
    hipMalloc(&tmp_d, tmp_bytes); HIPERR
    hipcub::DeviceRadixSort::SortKeys(tmp_d, tmp_bytes, in_d, out_d, n); HIPERR

    // allocate memory for the result
    bool unique_h;
    bool * unique_d = nullptr;
    hipMalloc(&unique_d, sizeof(bool)); HIPERR
    hipMemset(unique_d, 1, sizeof(bool)); HIPERR

    // check if neighbouring values are equal
    helpers::lambda_kernel
    <<<SDIV(n, WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE>>>
    ([=] DEVICEQUALIFIER
    {
        // determine the global thread ID
        const std::uint64_t tid = helpers::global_thread_id();

        // if neighbouring elements are equal throw error
        if(tid < n-1 && out_d[tid] == out_d[tid+1])
        {
            *unique_d = false;
        }
    }
    ); HIPERR

    // copy the result to host
    hipMemcpy(&unique_h, unique_d, sizeof(bool), D2H); HIPERR

    // free any allocated memory
    hipFree(out_d); HIPERR
    hipFree(tmp_d); HIPERR
    hipFree(unique_d); HIPERR

    return unique_h;
}

int main()
{
    // define the data types to be generated
    using data_t = std::uint64_t;
    using index_t = std::uint64_t;
    using rng_t = kiss::Kiss<std::uint64_t>;

    // number of unique random values to generate
    static constexpr index_t n = 1UL << 28;
    // random seed
    static constexpr index_t seed = 42;

    // allocate GPU memory for the result
    data_t * data_d = nullptr;
    hipMalloc(&data_d, sizeof(data_t)*n); HIPERR

    // generate the values and measure throughput
    helpers::GpuTimer timer("generate");
    // defined in warpcore/random_distributions.hpp
    warpcore::unique_distribution<data_t, rng_t>(data_d, n, seed); HIPERR
    timer.print_throughput(sizeof(data_t), n);

    // check if the generated values are unique
    std::cout << "TEST PASSED: " << std::boolalpha << check_unique(data_d, n) << std::endl;

    // free any allocated memory
    hipFree(data_d); HIPERR
}
