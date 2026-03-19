# WARPCORE

**Hashing at the speed of light on modern HIP accelerators**

## Introduction
`warpcore` is a framework for creating high-throughput, purpose-built hashing data structures on HIP accelerators.

This library provides the following data structures:
- [`SingleValueHashTable`](include/warpcore/single_value_hash_table.hpp): stores a set of key-value pairs
- [`HashSet`](include/warpcore/hash_set.hpp): stores a set of keys
- [`CountingHashTable`](include/warpcore/counting_hash_table.hpp): keeps track of the number of occurrences of each inserted key
- [`BloomFilter`](include/warpcore/bloom_filter.hpp): pattern-blocked bloom filter for approximate membership queries
- [`MultiValueHashTable`](include/warpcore/multi_value_hash_table.hpp): stores a multi-set of key-value pairs
- [`BucketListHashTable`](include/warpcore/bucket_list_hash_table.hpp): alternative variant of `MultiValueHashTable`
- [`MultiBucketHashTable`](include/warpcore/multi_bucket_hash_table.hpp): alternative variant of `MultiValueHashTable`

Implementations support key types `std::uint32_t` and `std::uint64_t` together with any trivially copyable value type. In order to be adaptable to a wide range of possible usecases, we provide a multitude of combinable modules such as [hash functions](include/warpcore/hashers.hpp), [probing schemes](include/warpcore/probing_schemes.hpp), and [data layouts](include/warpcore/storage.hpp) (visit the [documentation](https://sleeepyjack.github.io/warpcore/) for further information).

`warpcore` has won the best paper award at the [IEEE HiPC 2020 conference](https://hipc.org/) ([link to manuscript](https://ieeexplore.ieee.org/document/9406635))([link to preprint](https://arxiv.org/abs/2009.07914)) and is based on our previous work on massively parallel GPU hash tables `warpdrive` which has been published in the prestigious [IEEE IPDPS conference](https://www.ipdps.org/) ([link to manuscript](https://ieeexplore.ieee.org/document/8425198)).

## Development Status

This library is still under heavy development. Users should expect breaking changes and refactoring to be common.
Developement mainly takes place on our in-house GitLab instance. However, we plan to migrate to GitHub in the near future.

## Requirements
- AMD GPU supported by ROCm
- [ROCm HIP toolkit/compiler](https://rocm.docs.amd.com/) version 6.3 or higher
- CMake 3.21 or higher
- C++17 or higher

## Dependencies
- ROCm `hip`
- ROCm `rocprim`
- ROCm `rocthrust`
- ROCm `hipcub`
- bundled helper/timer/packed-type support headers
- bundled `kiss` RNG header

These ROCm packages must be installed and discoverable by `find_package(...)` in [CMake](https://cmake.org/).

## Getting `warpcore`

`warpcore` is header only and can be incorporated manually into your project by downloading the headers and placing them into your source tree.

### Adding `warpcore` to a CMake Project

`warpcore` is designed to make it easy to include within another CMake project.
 The `CMakeLists.txt` exports a `warpcore` target that can be linked<sup>[1](#link-footnote)</sup> into a target to setup include directories, dependencies, and compile flags necessary to use `warpcore` in your project.


We recommend using [CMake Package Manager (CPM)](https://github.com/TheLartians/CPM.cmake) to fetch `warpcore` into your project.
With CPM, getting `warpcore` is easy:

```cmake
cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(my_target LANGUAGES CXX HIP)

include(path/to/CPM.cmake)

CPMAddPackage(
  NAME warpcore
  GITHUB_REPOSITORY sleeepyjack/warpcore
  GIT_TAG/VERSION XXXXX
)

target_link_libraries(my_target PRIVATE warpcore)
```

This will take care of downloading `warpcore` from GitHub and making the headers available in a location that can be found by CMake. Linking against the `warpcore` target will provide everything needed for `warpcore` to be used by `my_target`.

<a name="link-footnote">1</a>: `warpcore` is header-only and therefore there is no binary component to "link" against. The linking terminology comes from CMake's `target_link_libraries` which is still used even for header-only library targets.

## Building `warpcore`

Since `warpcore` is header-only, there is nothing to build to use it.

To build the tests, benchmarks, and examples:

```bash
cd $WARPCORE_ROOT
mkdir -p build
cd build
# If CMake does not auto-detect HIP, set HIPCXX or pass
# -DCMAKE_HIP_COMPILER=/path/to/hipcc.
cmake .. \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DWARPCORE_BUILD_TESTS=ON \
  -DWARPCORE_BUILD_BENCHMARKS=ON \
  -DWARPCORE_BUILD_EXAMPLES=ON
cmake --build .
```
Binaries will be built into:
- `build/tests/`
- `build/benchmarks/`
- `build/examples/`


## [Documentation](docs/index.html)

## Where to go from here?
Take a look at the [examples](examples/README.md), test your own system performance using the [benchmark suite](benchmarks/README.md) and be sure everything works as expected by running the [test suite](tests/README.md).

## How to cite `warpcore`?
BibTeX:
```console
@inproceedings{DBLP:conf/hipc/JungerKM0XLS20,
  author    = {Daniel J{\"{u}}nger and
               Robin Kobus and
               Andr{\'{e}} M{\"{u}}ller and
               Christian Hundt and
               Kai Xu and
               Weiguo Liu and
               Bertil Schmidt},
  title     = {WarpCore: {A} Library for fast Hash Tables on GPUs},
  booktitle = {27th {IEEE} International Conference on High Performance Computing,
               Data, and Analytics, HiPC 2020, Pune, India, December 16-19, 2020},
  pages     = {11--20},
  publisher = {{IEEE}},
  year      = {2020},
  url       = {https://doi.org/10.1109/HiPC50609.2020.00015},
  doi       = {10.1109/HiPC50609.2020.00015},
  timestamp = {Wed, 05 May 2021 09:45:30 +0200},
  biburl    = {https://dblp.org/rec/conf/hipc/JungerKM0XLS20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

@inproceedings{DBLP:conf/ipps/Junger0S18,
  author    = {Daniel J{\"{u}}nger and
               Christian Hundt and
               Bertil Schmidt},
  title     = {WarpDrive: Massively Parallel Hashing on Multi-GPU Nodes},
  booktitle = {2018 {IEEE} International Parallel and Distributed Processing Symposium,
               {IPDPS} 2018, Vancouver, BC, Canada, May 21-25, 2018},
  pages     = {441--450},
  publisher = {{IEEE} Computer Society},
  year      = {2018},
  url       = {https://doi.org/10.1109/IPDPS.2018.00054},
  doi       = {10.1109/IPDPS.2018.00054},
  timestamp = {Sat, 19 Oct 2019 20:31:38 +0200},
  biburl    = {https://dblp.org/rec/conf/ipps/Junger0S18.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
***
warpcore Copyright (C) 2018-2025 [Daniel Jünger](https://github.com/sleeepyjack)

This program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it under certain
conditions. See the file [LICENSE](LICENSE) for details.

[repository]: https://github.com/sleeepyjack/warpcore
