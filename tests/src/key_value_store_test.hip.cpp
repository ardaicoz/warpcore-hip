#include <catch2/catch_all.hpp>
#include <warpcore/base.hpp>

template<class K, class V>
struct key_value_case
{
    using key_type = K;
    using value_type = V;
};

using key_value_case_u32_u32 = key_value_case<std::uint32_t, std::uint32_t>;
using key_value_case_u32_f32 = key_value_case<std::uint32_t, float>;

TEMPLATE_TEST_CASE(
    "SoAStore",
    "[storage][soa][template]",
    key_value_case_u32_u32,
    key_value_case_u32_f32)
{
    using namespace warpcore;
    using Key = typename TestType::key_type;
    using Value = typename TestType::value_type;

    using storage_t = storage::key_value::SoAStore<Key, Value>;

    const index_t capacity = GENERATE(as<index_t>{}, 123456, 42424242, 69696969);

    SECTION("constructor")
    {
        storage_t st = storage_t(capacity); HIPERR

        CHECK(st.status() == Status::none());

        CHECK(st.capacity() == capacity);
    }

    SECTION("set and get pairs")
    {
        storage_t st = storage_t(capacity); HIPERR

        // set pairs
        helpers::lambda_kernel
        <<<SDIV(capacity, WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE>>>
        ([=] DEVICEQUALIFIER () mutable
        {
            const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            if(tid < capacity)
            {
                auto&& p = st[tid];
                p.key = Key(tid);
                p.value = Value(tid);
            }
        });

        index_t * errors = nullptr;
        hipMallocManaged(&errors, sizeof(index_t)); HIPERR
        *errors = 0;

        // get pairs
        helpers::lambda_kernel
        <<<SDIV(capacity, WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE>>>
        ([=] DEVICEQUALIFIER () mutable
        {
            const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            if(tid < capacity)
            {
                if(st[tid].key != Key(tid) || st[tid].value != Value(tid))
                {
                    atomicAdd(errors, 1);
                }
            }
        });
        hipDeviceSynchronize(); HIPERR

        CHECK(*errors == 0);

        hipFree(errors); HIPERR
    }

    SECTION("init keys")
    {
        storage_t st = storage_t(capacity); HIPERR

        const Key key = 42;

        st.init_keys(key);

        index_t * errors = nullptr;
        hipMallocManaged(&errors, sizeof(index_t)); HIPERR
        *errors = 0;

        helpers::lambda_kernel
        <<<SDIV(capacity, WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE>>>
        ([=] DEVICEQUALIFIER () mutable
        {
            const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            if(tid < capacity)
            {
                if(st[tid].key != key)
                {
                    atomicAdd(errors, 1);
                }
            }
        });
        hipDeviceSynchronize(); HIPERR

        CHECK(*errors == 0);

        hipFree(errors); HIPERR
    }

    SECTION("init pairs")
    {
        storage_t st = storage_t(capacity); HIPERR

        const Key key = 42;
        const Value value = 1337;

        st.init_pairs(key, value);

        index_t * errors = nullptr;
        hipMallocManaged(&errors, sizeof(index_t)); HIPERR
        *errors = 0;

        helpers::lambda_kernel
        <<<SDIV(capacity, WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE>>>
        ([=] DEVICEQUALIFIER () mutable
        {
            const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            if(tid < capacity)
            {
                if(st[tid].key != key || st[tid].value != value)
                {
                    atomicAdd(errors, 1);
                }
            }
        });
        hipDeviceSynchronize(); HIPERR

        CHECK(*errors == 0);

        hipFree(errors); HIPERR
    }

    SECTION("HIP atomics")
    {
        storage_t st = storage_t(1); HIPERR

        const Key init = 0;
        const Key key = 42;
        const Value value = 1337;

        st.init_pairs(init, value);

        bool * error = nullptr;
        hipMallocManaged(&error, sizeof(bool)); HIPERR
        *error = false;

        helpers::lambda_kernel
        <<<1, 1>>>
        ([=] DEVICEQUALIFIER () mutable
        {
            atomicCAS(&st[0].key, init, key);
            *error = (st[0].key == key && st[0].value == value) ? false : true;
        });
        hipDeviceSynchronize(); HIPERR

        CHECK(*error == false);

        hipFree(error); HIPERR
    }
}

TEMPLATE_TEST_CASE(
    "AoSStore",
    "[storage][aos][template]",
    key_value_case_u32_u32,
    key_value_case_u32_f32)
{
    using namespace warpcore;
    using Key = typename TestType::key_type;
    using Value = typename TestType::value_type;

    using storage_t = storage::key_value::AoSStore<Key, Value>;

    const index_t capacity = GENERATE(as<index_t>{}, 123456, 42424242, 69696969);

    SECTION("constructor")
    {
        storage_t st = storage_t(capacity); HIPERR

        CHECK(st.status() == Status::none());

        CHECK(st.capacity() == capacity);
    }

    SECTION("set and get pairs")
    {
        storage_t st = storage_t(capacity); HIPERR

        // set pairs
        helpers::lambda_kernel
        <<<SDIV(capacity, WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE>>>
        ([=] DEVICEQUALIFIER () mutable
        {
            const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            if(tid < capacity)
            {
                auto&& p = st[tid];
                p.key = Key(tid);
                p.value = Value(tid);
            }
        });

        index_t * errors = nullptr;
        hipMallocManaged(&errors, sizeof(index_t)); HIPERR
        *errors = 0;

        // get pairs
        helpers::lambda_kernel
        <<<SDIV(capacity, WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE>>>
        ([=] DEVICEQUALIFIER () mutable
        {
            const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            if(tid < capacity)
            {
                if(st[tid].key != Key(tid) || st[tid].value != Value(tid))
                {
                    atomicAdd(errors, 1);
                }
            }
        });
        hipDeviceSynchronize(); HIPERR

        CHECK(*errors == 0);

        hipFree(errors); HIPERR
    }

    SECTION("init keys")
    {
        storage_t st = storage_t(capacity); HIPERR

        const Key key = 42;

        st.init_keys(key);

        index_t * errors = nullptr;
        hipMallocManaged(&errors, sizeof(index_t)); HIPERR
        *errors = 0;

        helpers::lambda_kernel
        <<<SDIV(capacity, WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE>>>
        ([=] DEVICEQUALIFIER () mutable
        {
            const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            if(tid < capacity)
            {
                if(st[tid].key != key)
                {
                    atomicAdd(errors, 1);
                }
            }
        });
        hipDeviceSynchronize(); HIPERR

        CHECK(*errors == 0);

        hipFree(errors); HIPERR
    }

    SECTION("init pairs")
    {
        storage_t st = storage_t(capacity); HIPERR

        const Key key = 42;
        const Value value = 1337;

        st.init_pairs(key, value);

        index_t * errors = nullptr;
        hipMallocManaged(&errors, sizeof(index_t)); HIPERR
        *errors = 0;

        helpers::lambda_kernel
        <<<SDIV(capacity, WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE>>>
        ([=] DEVICEQUALIFIER () mutable
        {
            const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            if(tid < capacity)
            {
                if(st[tid].key != key || st[tid].value != value)
                {
                    atomicAdd(errors, 1);
                }
            }
        });
        hipDeviceSynchronize(); HIPERR

        CHECK(*errors == 0);

        hipFree(errors); HIPERR
    }

    SECTION("HIP atomics")
    {
        storage_t st = storage_t(1); HIPERR

        const Key init = 0;
        const Key key = 42;
        const Value value = 1337;

        st.init_pairs(init, value);

        bool * error = nullptr;
        hipMallocManaged(&error, sizeof(bool)); HIPERR
        *error = false;

        helpers::lambda_kernel
        <<<1, 1>>>
        ([=] DEVICEQUALIFIER () mutable
        {
            atomicCAS(&st[0].key, init, key);
            *error = (st[0].key == key && st[0].value == value) ? false : true;
        });
        hipDeviceSynchronize(); HIPERR

        CHECK(*error == false);

        hipFree(error); HIPERR
    }
}
