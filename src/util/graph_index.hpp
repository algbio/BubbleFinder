#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace graph_index {

inline constexpr std::uint32_t spqr_invalid_index =
    std::numeric_limits<std::uint32_t>::max();
inline constexpr std::uint64_t spqr_count_max =
    static_cast<std::uint64_t>(spqr_invalid_index);

inline constexpr std::uint64_t packed_endpoint_id_max =
    static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max() >> 1);

template <typename T>
constexpr T invalid()
{
    static_assert(std::is_unsigned<T>::value, "graph indices must be unsigned");
    return std::numeric_limits<T>::max();
}

inline std::uint32_t require_spqr_count(std::uint64_t value,
                                        std::string_view what)
{
    if (value > spqr_count_max) {
        throw std::runtime_error(std::string(what) +
                                 " does not fit in the current 32-bit SPQR backend");
    }
    return static_cast<std::uint32_t>(value);
}

inline std::uint32_t require_spqr_count_size(std::size_t value,
                                             std::string_view what)
{
    if (value > static_cast<std::size_t>(spqr_count_max)) {
        throw std::runtime_error(std::string(what) +
                                 " does not fit in the current 32-bit SPQR backend");
    }
    return static_cast<std::uint32_t>(value);
}

inline bool fits_sp_compress_count_size(std::size_t value)
{
    return value <= static_cast<std::size_t>(spqr_count_max);
}

inline bool fits_packed_endpoint_id(std::uint64_t value)
{
    return value <= packed_endpoint_id_max;
}

inline void require_spqr_graph(std::uint64_t nodes,
                               std::uint64_t edges,
                               const char* what)
{
    if (nodes > spqr_count_max) {
        throw std::runtime_error(std::string(what) +
                                 " node count does not fit in the current 32-bit SPQR backend");
    }
    if (edges > spqr_count_max) {
        throw std::runtime_error(std::string(what) +
                                 " edge count does not fit in the current 32-bit SPQR backend");
    }
}

inline std::size_t checked_add_size(std::size_t lhs,
                                    std::size_t rhs,
                                    const char* what)
{
    if (lhs > std::numeric_limits<std::size_t>::max() - rhs) {
        throw std::runtime_error(std::string(what) + " size overflow");
    }
    return lhs + rhs;
}

inline std::size_t checked_mul_size(std::size_t lhs,
                                    std::size_t rhs,
                                    const char* what)
{
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::runtime_error(std::string(what) + " size overflow");
    }
    return lhs * rhs;
}

}
