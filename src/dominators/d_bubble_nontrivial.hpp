#pragma once

#include "dominators/d_bubbles.hpp"

#include <stdexcept>

namespace dominators {

// Look for an internal node along the entry edges.
class DBubbleNontrivialFilter
{
    struct TreeIntervals
    {
        std::vector<uint32_t> tin, end;

        explicit TreeIntervals(const DBubbleIndex& index)
            : tin(index.parent.size(), NONE), end(index.parent.size(), NONE)
        {
            std::vector<uint32_t> first(index.parent.size(), NONE), next(index.parent.size(), NONE);
            for (uint32_t v = 0; v < index.source; ++v)
                if (index.parent[v] != NONE) {
                    next[v] = first[index.parent[v]];
                    first[index.parent[v]] = v;
                }

            uint32_t u = index.source;
            uint32_t clock = 0;
            bool entering = true;
            while (true) {
                if (entering) {
                    tin[u] = clock++;
                    if (first[u] != NONE) {
                        u = first[u];
                        continue;
                    }
                }
                end[u] = clock;
                if (u == index.source)
                    break;
                if (next[u] != NONE) {
                    u = next[u];
                    entering = true;
                } else {
                    u = index.parent[u];
                    entering = false;
                }
            }
        }

        bool ancestor(uint32_t a, uint32_t b) const
        {
            return a < tin.size() && b < tin.size() && tin[a] != NONE && tin[b] != NONE && tin[a] <= tin[b] &&
                   tin[b] < end[a];
        }
    };

  public:
    DBubbleNontrivialFilter(const Graph& graph, const DBubbleIndex& index)
        : ix(index), tree(index), offsets(static_cast<size_t>(index.source) + 1, 0)
    {
        if (graph.n != ix.component.size() || uint64_t(graph.n) * 2 != ix.source)
            throw std::logic_error("d-bubbles: graph size does not match the dominator tree");
        for (const auto& edge : graph.edges) {
            ++offsets[2u * edge.u + edge.su + 1];
            ++offsets[2u * edge.v + edge.sv + 1];
        }
        for (size_t i = 1; i < offsets.size(); ++i)
            offsets[i] += offsets[i - 1];

        adjacency.resize(offsets.back());
        auto cursor = offsets;
        for (const auto& edge : graph.edges) {
            adjacency[cursor[2u * edge.u + edge.su]++] = 2u * edge.v + (edge.sv ^ 1u);
            adjacency[cursor[2u * edge.v + edge.sv]++] = 2u * edge.u + (edge.su ^ 1u);
        }
    }

    bool containsInternalVertex(uint32_t /*component*/, uint32_t entry, uint32_t other) const
    {
        for (uint32_t i = offsets[entry]; i < offsets[entry + 1]; ++i) {
            const uint32_t next = adjacency[i];
            if (next / 2 != entry / 2 && next / 2 != other / 2 && tree.tin[next ^ 1u] != NONE &&
                !tree.ancestor(entry ^ 1u, next ^ 1u))
                return true;
        }
        return false;
    }

  private:
    const DBubbleIndex& ix;
    TreeIntervals tree;
    std::vector<uint32_t> offsets, adjacency;
};

}
