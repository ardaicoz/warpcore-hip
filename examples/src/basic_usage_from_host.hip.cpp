#include <iostream>
#include <warpcore/single_value_hash_table.hpp>
#include <helpers/timers.hpp>

// This example shows the basic usage of a single value hash table using
// host-sided table operations provided by warpcore
int main()
{
    // key type of the hash table (uint32_t or uint64_t)
    using key_t   = std::uint32_t;

    // value type of the hash table
    using value_t = std::uint32_t;

    using namespace warpcore;

    // type of the hash table (with default parameters)
    using hash_table_t = SingleValueHashTable<key_t, value_t>;

    // in this example we use dummy key/value pairs which will be inserted
    // into the hast table
    const uint64_t input_size = 1UL << 27;

    std::cout << "num elements " << input_size << std::endl;

    // allocate host-sided (pinned) arrays for our input data
    key_t*   keys_h;   hipHostMalloc(&keys_h,   sizeof(key_t)*input_size);
    value_t* values_h; hipHostMalloc(&values_h, sizeof(value_t)*input_size);

    // lets generate some random data
    // (key, val)->(1, 2), (2, 3), .., (input_size, input_size+1)
    // NOTE: since we are using default parameters, key_t(0) and key_t(0)-1 are
    // invalid keys since they internally map to the empty key specifier and
    // tombstone key specifier respectively
    for(key_t i = 0; i < input_size; i++)
    {
        keys_h[i]   = i+1;
        values_h[i] = i+2;
    }

    // allocate device-sided arrays for our input data
    key_t*   keys_d;   hipMalloc(&keys_d,   sizeof(key_t)*input_size); HIPERR
    value_t* values_d; hipMalloc(&values_d, sizeof(value_t)*input_size); HIPERR

    // copy input key/value pairs from the host to the device
    hipMemcpy(keys_d,   keys_h,   sizeof(key_t)*input_size, hipMemcpyHostToDevice); HIPERR
    hipMemcpy(values_d, values_h, sizeof(value_t)*input_size, hipMemcpyHostToDevice); HIPERR

    // the target load factor of the table after inserting our dummy data
    const float load = 0.9;

    // which results in the following capacity of the hash table
    const uint64_t capacity = input_size/load;

    // INITIALIZE the hash table
    hash_table_t hash_table(capacity); HIPERR

    std::cout << "capacity " << hash_table.capacity() << std::endl;

    {
        helpers::GpuTimer timer("insert");
        // INSERT the input data into the hash_table
        hash_table.insert(keys_d, values_d, input_size);
        timer.print();
    } HIPERR

    hipDeviceSynchronize(); HIPERR
    // check if any errors occured
    std::cout  << "insertion errors: " << hash_table.pop_status().get_errors() << std::endl;

    std::cout << "load " << hash_table.load_factor() << std::endl;
    std::cout << "size " << hash_table.size() << std::endl;

    // now, we want to retrieve our dummy data from the
    // hash table again

    // first, allocate some device-sided memory to hold the result
    value_t* result_d; hipMalloc(&result_d, sizeof(value_t)*input_size);

    {
        helpers::GpuTimer timer("retrieve");
        // RETRIEVE the corresponding values of the rear half of the input keys
        // from the hash table
        hash_table.retrieve(keys_d, input_size, result_d);
        timer.print();
    } HIPERR

    // check again if any errors occured
    std::cout << "retrieval errors: " << hash_table.pop_status().get_errors() << std::endl;

    // allocate host-sided memory to copy the result back to the host
    // in order to perform a unit test
    value_t* result_h; hipHostMalloc(&result_h, sizeof(value_t)*input_size); HIPERR

    // copy the result back to the host
    hipMemcpy(result_h, result_d, sizeof(value_t)*input_size, hipMemcpyDeviceToHost); HIPERR

    // check the result
    uint64_t errors = 0;
    for(uint64_t i = 0; i < input_size; i++)
    {
        // check if output matches the input
        if(result_h[i] != values_h[i])
        {
            errors++;
        }
    }
    std::cout << "check result: " << errors << " errors occured" << std::endl;

    // free all allocated recources
    hipHostFree(keys_h);
    hipFree(keys_d);
    hipHostFree(values_h);
    hipFree(values_d);
    hipHostFree(result_h);
    hipFree(result_d);

    // check for any HIP errors
    hipDeviceSynchronize(); HIPERR
}
