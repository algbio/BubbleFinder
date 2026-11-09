#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace paper_body_native {

constexpr uint32_t NONE = std::numeric_limits<uint32_t>::max();

// Each end stores a node ID and its inward sign.
struct Bubble
{
    uint32_t component = NONE;
    uint32_t x = NONE;
    uint32_t y = NONE;
};

struct Hierarchy
{
    std::vector<Bubble> bubbles;
    std::vector<uint32_t> orientedParent;
    std::vector<uint32_t> physicalParent;
    uint32_t source = NONE;
    uint64_t reachableStates = 0;
};

inline void require(bool ok, const char* why)
{
    if (!ok)
        throw std::runtime_error(std::string("d-bubbles hierarchy: ") + why);
}

inline uint32_t entry(const Bubble& bubble, unsigned reverse)
{
    return reverse ? bubble.y : bubble.x;
}

inline uint32_t exit(const Bubble& bubble, unsigned reverse)
{
    return (reverse ? bubble.x : bubble.y) ^ 1u;
}

// Skip the exit subtree when looking for a containing bubble.
inline std::vector<uint32_t> sourceCutForest(uint32_t source, uint64_t reachableNodes,
                                           const std::vector<uint32_t>& parent,
                                           const std::vector<uint32_t>& component,
                                           const std::vector<Bubble>& bubbles)
{
    require(source > 0 && source < NONE && !(source & 1u), "source ID must be positive and even");
    require(parent.size() == static_cast<size_t>(source) + 1, "parent list has the wrong size");
    require(parent[source] == NONE, "added source must have no parent");
    require(component.size() * 2 == source, "component list has the wrong size");
    require(reachableNodes <= source, "reachable node count exceeds graph size");
    require(bubbles.size() <= (NONE - 1u) / 2u, "too many bubbles for 32 bit IDs");

    const uint32_t directionCount = static_cast<uint32_t>(2 * bubbles.size());
    std::vector<uint32_t> result(directionCount, NONE);
    if (!directionCount)
        return result;
    std::vector<uint32_t> firstChild(parent.size(), NONE);
    std::vector<uint32_t> nextSibling(parent.size(), NONE);
    std::vector<uint32_t> bubbleAtEntry(parent.size(), NONE);
    uint64_t nodesWithParent = 0;
    for (uint32_t u = 0; u < source; ++u) {
        const uint32_t p = parent[u];
        if (p == NONE)
            continue;
        require(p <= source, "tree parent is outside the graph");
        require(p != u, "node cannot be its own tree parent");
        nextSibling[u] = firstChild[p];
        firstChild[p] = u;
        ++nodesWithParent;
    }
    require(nodesWithParent == reachableNodes, "tree parent count differs from the reachable node count");
    for (uint32_t i = 0; i < bubbles.size(); ++i) {
        const auto& p = bubbles[i];
        require(p.x < p.y && p.y < source && p.x / 2 != p.y / 2 && component[p.x / 2] == p.component &&
                    component[p.y / 2] == p.component && parent[p.y ^ 1u] == p.x && parent[p.x ^ 1u] == p.y,
                "bubble ends do not match their components or immediate dominators");
        require(bubbleAtEntry[p.x] == NONE && bubbleAtEntry[p.y] == NONE, "two bubbles use the same signed entry");
        bubbleAtEntry[p.x] = 2 * i;
        bubbleAtEntry[p.y] = 2 * i + 1;
    }

    const auto exitOf = [&](uint32_t o) { return exit(bubbles[o / 2], o & 1u); };
    uint32_t containingBubble = NONE;
    uint32_t u = source;
    uint64_t visited = 0;
    uint64_t visitedDirections = 0;
    bool entering = true;
    while (true) {
        if (entering) {
            require(++visited <= reachableNodes + 1, "source tree walk visited too many nodes");
            const uint32_t e = bubbleAtEntry[u];
            if (e != NONE) {
                result[e] = containingBubble;
                containingBubble = e;
                ++visitedDirections;
            }
            if (firstChild[u] != NONE) {
                const uint32_t child = firstChild[u];
                if (e != NONE && child == exitOf(e))
                    containingBubble = result[e];
                u = child;
                continue;
            }
        }
        const uint32_t e = bubbleAtEntry[u];
        if (e != NONE)
            containingBubble = result[e];
        if (u == source)
            break;
        const uint32_t p = parent[u];
        const uint32_t pe = bubbleAtEntry[p];
        if (pe != NONE && u == exitOf(pe))
            containingBubble = pe;
        if (nextSibling[u] != NONE) {
            const uint32_t child = nextSibling[u];
            if (pe != NONE && child == exitOf(pe))
                containingBubble = result[pe];
            u = child;
            entering = true;
        } else {
            u = p;
            entering = false;
        }
    }
    require(visited == reachableNodes + 1, "source tree walk missed a node");
    require(visitedDirections == directionCount, "source tree walk missed a bubble direction");
    require(containingBubble == NONE, "bubble remains active after the tree walk");
    for (uint32_t o = 0; o < directionCount; ++o) {
        const uint32_t p = result[o];
        if (p == NONE)
            continue;
        require(p < directionCount, "source parent is outside the bubble list");
        require(p != o, "bubble cannot be its own source parent");
        require(bubbles[p / 2].component == bubbles[o / 2].component,
                "source parent belongs to another component");
    }
    return result;
}

// Use both ends to find the containing bubble.
inline Hierarchy build(uint32_t source, uint64_t reachableNodes, const std::vector<uint32_t>& parent,
                       const std::vector<uint32_t>& component, std::vector<Bubble> bubbles)
{
    const auto sourceParents = sourceCutForest(source, reachableNodes, parent, component, bubbles);

    Hierarchy hierarchy;
    hierarchy.source = source;
    hierarchy.reachableStates = reachableNodes;
    hierarchy.bubbles = std::move(bubbles);
    const uint32_t directionCount = static_cast<uint32_t>(sourceParents.size());
    const uint32_t root = directionCount;
    hierarchy.orientedParent.assign(directionCount, NONE);
    hierarchy.physicalParent.assign(directionCount / 2, NONE);
    if (!directionCount)
        return hierarchy;
    std::vector<uint32_t> targetParents(directionCount, NONE);
    std::vector<uint32_t> sourceFirstChild(static_cast<size_t>(directionCount) + 1, NONE);
    std::vector<uint32_t> sourceNextSibling(directionCount, NONE);
    std::vector<uint32_t> firstChild(static_cast<size_t>(directionCount) + 1, NONE);
    std::vector<uint32_t> nextSibling(directionCount, NONE);
    for (uint32_t u = 0; u < directionCount; ++u) {
        targetParents[u] = sourceParents[u ^ 1u] == NONE ? NONE : sourceParents[u ^ 1u] ^ 1u;
        const uint32_t sourceParent = sourceParents[u] == NONE ? root : sourceParents[u];
        const uint32_t targetParent = targetParents[u] == NONE ? root : targetParents[u];
        require(sourceParent <= root && targetParent <= root, "parent is outside the bubble list");
        require(sourceParent != u && targetParent != u, "bubble cannot be its own parent");
        sourceNextSibling[u] = sourceFirstChild[sourceParent];
        sourceFirstChild[sourceParent] = u;
        nextSibling[u] = firstChild[targetParent];
        firstChild[targetParent] = u;
    }

    std::vector<uint32_t> depth(static_cast<size_t>(directionCount) + 1, 0);
    std::vector<uint32_t> size(static_cast<size_t>(directionCount) + 1, 1);
    std::vector<uint32_t> largestChild(static_cast<size_t>(directionCount) + 1, NONE);
    std::vector<uint32_t> order;
    order.reserve(static_cast<size_t>(directionCount) + 1);
    order.push_back(root);
    for (size_t i = 0; i < order.size(); ++i) {
        const uint32_t u = order[i];
        for (uint32_t v = firstChild[u]; v != NONE; v = nextSibling[v]) {
            require(order.size() <= directionCount, "target hierarchy contains a cycle");
            depth[v] = depth[u] + 1;
            order.push_back(v);
        }
    }
    require(order.size() == static_cast<size_t>(directionCount) + 1,
            "target hierarchy walk missed a bubble");
    for (size_t i = order.size(); i-- > 0;) {
        const uint32_t u = order[i];
        for (uint32_t v = firstChild[u]; v != NONE; v = nextSibling[v]) {
            size[u] += size[v];
            if (largestChild[u] == NONE || size[v] > size[largestChild[u]])
                largestChild[u] = v;
        }
    }

    std::vector<uint32_t> pathHead(static_cast<size_t>(directionCount) + 1, NONE);
    std::vector<uint32_t> position(static_cast<size_t>(directionCount) + 1, NONE);
    order.clear();
    order.push_back(root);
    uint32_t nextPosition = 0;
    while (!order.empty()) {
        const uint32_t chain = order.back();
        order.pop_back();
        for (uint32_t u = chain; u != NONE; u = largestChild[u]) {
            pathHead[u] = chain;
            position[u] = nextPosition++;
            for (uint32_t v = firstChild[u]; v != NONE; v = nextSibling[v])
                if (v != largestChild[u])
                    order.push_back(v);
        }
    }
    require(nextPosition == static_cast<uint64_t>(directionCount) + 1, "tree indexing missed a bubble");
    std::vector<uint32_t>().swap(firstChild);
    std::vector<uint32_t>().swap(nextSibling);
    std::vector<uint32_t>().swap(size);
    std::vector<uint32_t>().swap(largestChild);
    std::vector<uint32_t>().swap(order);

    size_t base = 1;
    while (base < static_cast<size_t>(directionCount) + 1)
        base *= 2;
    require(base <= std::vector<uint32_t>().max_size() / 2, "too many bubbles for the tree index");
    std::vector<uint32_t> tree(2 * base, NONE);

    const auto best = [&](uint32_t x, uint32_t y) {
        if (x == NONE)
            return y;
        if (y == NONE)
            return x;
        return depth[x] > depth[y] || (depth[x] == depth[y] && x < y) ? x : y;
    };

    const auto mark = [&](uint32_t u, bool on) {
        size_t p = base + position[u];
        tree[p] = on ? u : NONE;
        while (p > 1) {
            p /= 2;
            tree[p] = best(tree[2 * p], tree[2 * p + 1]);
        }
    };
    const auto rangeMax = [&](size_t l, size_t r) {
        uint32_t result = NONE;
        l += base;
        r += base;
        while (l < r) {
            if (l & 1u)
                result = best(result, tree[l++]);
            if (r & 1u)
                result = best(result, tree[--r]);
            l /= 2;
            r /= 2;
        }
        return result;
    };
    const auto nearest = [&](uint32_t u) {
        uint32_t result = NONE;
        while (true) {
            const uint32_t chain = pathHead[u];
            result = best(result, rangeMax(position[chain], static_cast<size_t>(position[u]) + 1));
            if (chain == root)
                break;
            u = targetParents[chain] == NONE ? root : targetParents[chain];
        }
        return result;
    };
    uint32_t u = root;
    uint64_t visited = 0;
    bool entering = true;
    while (true) {
        if (entering) {
            require(++visited <= static_cast<uint64_t>(directionCount) + 1, "source hierarchy contains a cycle");
            if (u != root) {
                hierarchy.orientedParent[u] = nearest(u);
                mark(u, true);
            }
            if (sourceFirstChild[u] != NONE) {
                u = sourceFirstChild[u];
                continue;
            }
        }
        if (u == root)
            break;
        mark(u, false);
        if (sourceNextSibling[u] != NONE) {
            u = sourceNextSibling[u];
            entering = true;
        } else {
            u = sourceParents[u] == NONE ? root : sourceParents[u];
            entering = false;
        }
    }
    require(visited == static_cast<uint64_t>(directionCount) + 1, "source hierarchy walk missed a bubble");
    require(tree[1] == NONE, "bubble remains marked after the hierarchy walk");
    for (uint32_t i = 0; i < hierarchy.bubbles.size(); ++i) {
        const uint32_t p = hierarchy.orientedParent[2 * i];
        const uint32_t q = hierarchy.orientedParent[2 * i + 1];
        require(q == (p == NONE ? NONE : p ^ 1u), "bubble parent differs between directions");
        if (p != NONE) {
            require(p < directionCount, "parent is outside the bubble list");
            require(p / 2 != i, "bubble cannot be its own parent");
            require(hierarchy.bubbles[p / 2].component == hierarchy.bubbles[i].component,
                    "parent belongs to another component");
        }
        hierarchy.physicalParent[i] = p == NONE ? NONE : p / 2;
    }
    std::vector<uint8_t> color(hierarchy.physicalParent.size(), 0);
    for (uint32_t i = 0; i < color.size(); ++i) {
        uint32_t v = i;
        while (v != NONE && !color[v]) {
            color[v] = 1;
            v = hierarchy.physicalParent[v];
        }
        require(v == NONE || color[v] == 2, "bubble hierarchy contains a cycle");
        v = i;
        while (v != NONE && color[v] == 1) {
            color[v] = 2;
            v = hierarchy.physicalParent[v];
        }
    }
    return hierarchy;
}
}
