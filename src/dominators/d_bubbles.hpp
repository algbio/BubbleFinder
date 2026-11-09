#pragma once

#include "dominators/dominators.hpp"

#include <functional>
#include <iosfwd>
#include <string>

namespace dominators {

struct DBubbleOptions
{
    bool bcTrees = false;
    unsigned threads = 1;
    bool ultrabubblesOnly = false;
};

struct DBubbleIndex
{
    uint32_t source = NONE;
    uint32_t components = 0;
    uint64_t reachableStates = 0;
    uint64_t corridorStates = 0;
    std::vector<uint32_t> parent;
    std::vector<uint32_t> component;
};

// Keep the graph for filtering.
struct DBubbleWorkspace
{
    std::vector<uint32_t> offsets, edges, dfsPre, dfsEnd;
};

DBubbleIndex indexDBubbles(const Graph& graph, const DBubbleOptions& options,
                           DBubbleWorkspace* workspace = nullptr);

// Report each bubble once.
uint64_t forEachDBubble(const DBubbleIndex& index,
                        const std::function<void(uint32_t, uint32_t, uint32_t)>& emit);

int runDBubbles(const Graph& graph, const std::vector<std::string>& names, const DBubbleOptions& options,
                bool includeTrivial, std::ostream& out, std::ostream& log);

}
