#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace dominators {
constexpr uint32_t NONE = UINT32_MAX;

struct BiEdge
{
    uint32_t u;
    uint32_t v;
    uint8_t su; // side of u: 0 = '+', 1 = '-'
    uint8_t sv; // side of v
};

struct Graph
{
    uint32_t n = 0;
    std::vector<BiEdge> edges;
};

class InvalidInput : public std::runtime_error
{
  public:
    using std::runtime_error::runtime_error;
};
}
