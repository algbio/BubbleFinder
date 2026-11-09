#include "dominators/d_bubble_hierarchy.hpp"
#include "dominators/d_bubbles.hpp"

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace dominators;
using paper_body_native::Bubble;

static bool ancestor(const DBubbleIndex& index, uint32_t a, uint32_t b)
{
    for (uint32_t steps = 0; b != NONE; ++steps) {
        if (b == a)
            return true;
        if (steps >= index.parent.size())
            throw std::logic_error("invalid tree");
        b = index.parent[b];
    }
    return false;
}

static std::vector<uint8_t> body(const DBubbleIndex& index, const Bubble& bubble)
{
    std::vector<uint8_t> nodes(index.component.size(), 0);
    nodes[bubble.x / 2] = nodes[bubble.y / 2] = 1;
    for (uint32_t z = 0; z < index.source; ++z)
        if (ancestor(index, bubble.x, z) && !ancestor(index, bubble.y ^ 1u, z) &&
            ancestor(index, bubble.y, z ^ 1u) && !ancestor(index, bubble.x ^ 1u, z ^ 1u))
            nodes[z / 2] = 1;
    return nodes;
}

static bool contains(const std::vector<uint8_t>& outer, const std::vector<uint8_t>& inner)
{
    for (size_t i = 0; i < outer.size(); ++i)
        if (inner[i] && !outer[i])
            return false;
    return true;
}

int main()
{
    try {
        std::mt19937 rng(20261108);
        uint32_t checked = 0, bubblesChecked = 0;
        for (uint32_t trial = 0; trial < 5000; ++trial) {
            Graph graph;
            graph.n = 2 + rng() % 8;
            if (trial % 2 == 0)
                for (uint32_t v = 1; v < graph.n; ++v)
                    graph.edges.push_back({v - 1, v, 0, 1});
            for (uint32_t e = 0, count = rng() % 15; e < count; ++e)
                graph.edges.push_back({static_cast<uint32_t>(rng() % graph.n),
                                       static_cast<uint32_t>(rng() % graph.n),
                                       static_cast<uint8_t>(rng() % 2), static_cast<uint8_t>(rng() % 2)});
            DBubbleOptions options;
            options.bcTrees = trial % 2 == 0;
            const auto index = indexDBubbles(graph, options);
            std::vector<Bubble> bubbles;
            forEachDBubble(index, [&](uint32_t c, uint32_t x, uint32_t y) { bubbles.push_back({c, x, y}); });
            if (bubbles.empty())
                continue;
            std::vector<std::vector<uint8_t>> bodies;
            for (const auto& bubble : bubbles)
                bodies.push_back(body(index, bubble));
            bool equalBodies = false;
            for (size_t i = 0; i < bodies.size(); ++i)
                for (size_t j = 0; j < i; ++j)
                    if (bodies[i] == bodies[j])
                        equalBodies = true;
            if (equalBodies)
                continue;

            const auto hierarchy = paper_body_native::build(index.source, index.reachableStates, index.parent,
                                                            index.component, bubbles);
            for (uint32_t i = 0; i < bubbles.size(); ++i) {
                uint32_t expected = paper_body_native::NONE;
                size_t best = graph.n + 1;
                for (uint32_t j = 0; j < bubbles.size(); ++j)
                    if (i != j && contains(bodies[j], bodies[i])) {
                        const auto size =
                            static_cast<size_t>(std::count(bodies[j].begin(), bodies[j].end(), 1));
                        if (size < best) {
                            best = size;
                            expected = j;
                        }
                    }
                if (hierarchy.physicalParent[i] != expected)
                    throw std::logic_error("bubble parent differs from its smallest containing region");
                ++bubblesChecked;
            }
            ++checked;
        }
        if (checked < 100 || bubblesChecked < 100)
            throw std::logic_error("not enough hierarchy cases");
        std::cout << "Checked " << bubblesChecked << " bubbles in " << checked << " graphs.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
