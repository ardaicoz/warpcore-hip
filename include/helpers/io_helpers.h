#ifndef HELPERS_IO_HELPERS_H
#define HELPERS_IO_HELPERS_H

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace helpers
{

template<class T>
std::vector<T> load_binary(
    const char* const file_name,
    const std::size_t max_elements)
{
    std::ifstream stream(file_name, std::ios::binary);
    if(!stream)
    {
        throw std::runtime_error("failed to open binary input file");
    }

    stream.seekg(0, std::ios::end);
    const auto bytes = static_cast<std::size_t>(stream.tellg());
    stream.seekg(0, std::ios::beg);

    const std::size_t available_elements = bytes / sizeof(T);
    const std::size_t elements_to_read = std::min(max_elements, available_elements);

    std::vector<T> out(elements_to_read);
    stream.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(elements_to_read * sizeof(T)));
    return out;
}

} // namespace helpers

#endif /* HELPERS_IO_HELPERS_H */
