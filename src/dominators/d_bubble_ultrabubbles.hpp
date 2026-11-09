#pragma once
#include "dominators/d_bubbles.hpp"
#include <algorithm>
#include <vector>

namespace dominators {

inline std::vector<uint8_t> classifyUltrabubbles(const DBubbleIndex& ix, const std::vector<uint32_t>& off,
                                               const std::vector<uint32_t>& adj,
                                               const std::vector<uint32_t>& dfsPre,
                                               const std::vector<uint32_t>& dfsEnd)
{
    const uint32_t n = ix.source + 1;
    std::vector<uint32_t> childOff(static_cast<size_t>(n) + 1, 0), children;
    for (uint32_t v = 0; v < n; ++v)
        if (ix.parent[v] != NONE)
            ++childOff[ix.parent[v] + 1];
    for (size_t i = 1; i < childOff.size(); ++i)
        childOff[i] += childOff[i - 1];
    children.resize(childOff.back());
    {
        auto cur = childOff;
        for (uint32_t v = 0; v < n; ++v)
            if (ix.parent[v] != NONE)
                children[cur[ix.parent[v]]++] = v;
    }
    std::vector<uint32_t> pre(n, NONE), end(n, NONE), depth(n, 0), order;
    order.reserve(static_cast<size_t>(ix.reachableStates) + 1);
    pre[ix.source] = 0;
    order.push_back(ix.source);
    uint32_t maxDepth = 0;
    {
        std::vector<std::pair<uint32_t, uint32_t>> stack{{ix.source, childOff[ix.source]}};
        while (!stack.empty()) {
            const uint32_t u = stack.back().first;
            uint32_t& it = stack.back().second;
            if (it < childOff[u + 1]) {
                const uint32_t v = children[it++];
                pre[v] = static_cast<uint32_t>(order.size());
                order.push_back(v);
                depth[v] = depth[u] + 1;
                maxDepth = std::max(maxDepth, depth[v]);
                stack.emplace_back(v, childOff[v]);
            } else {
                end[u] = static_cast<uint32_t>(order.size());
                stack.pop_back();
            }
        }
    }
    const auto reached = [&](uint32_t u) { return pre[u] != NONE; };
    const auto ancestor = [&](uint32_t a, uint32_t b) { return pre[a] <= pre[b] && pre[b] < end[a]; };
    std::vector<uint32_t> limF(n, NONE), threshold(n, NONE);
    std::vector<uint8_t> unPred(n, 0), head(n, 0), bad(n, 0);
    {
        std::vector<uint32_t> path(static_cast<size_t>(maxDepth) + 1, NONE);
        for (uint32_t u : order) {
            path[depth[u]] = u;
            for (uint32_t e = off[u]; e < off[u + 1]; ++e) {
                const uint32_t v = adj[e];
                const uint32_t bound = ancestor(v, u) ? depth[v] : depth[ix.parent[v]] + 1;
                limF[u] = std::min(limF[u], bound);
                if (v != ix.source && ix.parent[v] != u) {
                    const uint32_t c = path[depth[ix.parent[v]] + 1];
                    if (c == NONE || ix.parent[c] != ix.parent[v])
                        throw std::logic_error("ultrabubbles: edge does not match its dominator tree path");
                    bad[c] = 1;
                }
                if (dfsPre[v] <= dfsPre[u] && dfsPre[u] < dfsEnd[v])
                    head[v] = 1;
            }
        }
    }
    for (uint32_t u = 0; u < n; ++u)
        if (!reached(u))
            for (uint32_t e = off[u]; e < off[u + 1]; ++e)
                if (reached(adj[e]))
                    unPred[adj[e]] = bad[adj[e]] = 1;
    const uint32_t nr = static_cast<uint32_t>(order.size());
    std::vector<uint32_t> backPrefix(static_cast<size_t>(nr) + 1, 0);
    std::vector<uint32_t> tipPrefix(static_cast<size_t>(nr) + 1, 0);
    for (uint32_t i = 0; i < nr; ++i) {
        const uint32_t u = order[i];
        const bool tip = u < ix.source && (off[u] == off[u + 1] || off[u ^ 1u] == off[(u ^ 1u) + 1]);
        backPrefix[i + 1] = backPrefix[i] + head[u];
        tipPrefix[i + 1] = tipPrefix[i] + static_cast<uint32_t>(tip);
    }
    std::vector<uint8_t>().swap(head);
    for (uint32_t i = nr; i-- > 0;) {
        const uint32_t u = order[i];
        threshold[u] = std::min(threshold[u], std::min(limF[u], unPred[u] ? 0u : NONE));
        if (ix.parent[u] != NONE)
            threshold[ix.parent[u]] = std::min(threshold[ix.parent[u]], threshold[u]);
    }
    for (uint32_t p : order) {
        uint32_t first = NONE, second = NONE, owner = NONE;
        for (uint32_t j = childOff[p]; j < childOff[p + 1]; ++j) {
            const uint32_t c = children[j];
            if (threshold[c] < first) {
                second = first;
                first = threshold[c];
                owner = c;
            } else
                second = std::min(second, threshold[c]);
        }
        for (uint32_t j = childOff[p]; j < childOff[p + 1]; ++j) {
            const uint32_t c = children[j];
            threshold[c] = std::min(limF[p], c == owner ? second : first);
        }
    }
    const auto inInterior = [&](uint32_t u, uint32_t p, uint32_t v) {
        return reached(u) && ((pre[p] < pre[u] && pre[u] < pre[v]) || (end[v] <= pre[u] && pre[u] < end[p]));
    };
    std::vector<uint8_t> result(n, 0);
    for (uint32_t v = 0; v < ix.source; ++v) {
        const uint32_t p = ix.parent[v];
        if (p >= ix.source || p / 2 == v / 2 || ix.parent[p ^ 1u] != (v ^ 1u))
            continue;
        if (bad[v] || threshold[v] <= depth[p])
            continue;
        if (inInterior(p ^ 1u, p, v) || inInterior(v ^ 1u, p, v))
            continue;
        const uint32_t a = pre[p] + 1, b = pre[v], c = end[v], e = end[p];
        if (tipPrefix[b] - tipPrefix[a] + tipPrefix[e] - tipPrefix[c] ||
            backPrefix[b] - backPrefix[a] + backPrefix[e] - backPrefix[c])
            continue;
        const uint32_t interior = (end[p] - pre[p] - 1) - (end[v] - pre[v]);
        result[v] = interior ? 1 : 2;
    }
    return result;
}

inline uint64_t forEachUltrabubble(const DBubbleIndex& ix, const std::vector<uint8_t>& exits, bool includeTrivial,
                             const std::function<void(uint32_t, uint32_t, uint32_t)>& emit)
{
    if (exits.size() != ix.parent.size())
        throw std::logic_error("ultrabubbles: filter and tree sizes differ");
    uint64_t count = 0;
    dominators::forEachDBubble(ix, [&](uint32_t component, uint32_t x, uint32_t other) {
        const uint8_t forward = exits[other ^ 1u];
        const uint8_t reverse = exits[x ^ 1u];
        if (forward != reverse)
            throw std::logic_error("ultrabubbles: filter result differs between directions");
        if (!forward || (!includeTrivial && forward == 2))
            return;
        emit(component, x, other);
        ++count;
    });
    return count;
}

}
