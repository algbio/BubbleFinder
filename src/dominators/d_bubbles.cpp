#include "dominators/d_bubbles.hpp"
#include "dominators/bc_tree.hpp"
#include "dominators/d_bubble_nontrivial.hpp"
#include "dominators/d_bubble_ultrabubbles.hpp"
#include "dominators/directed_tree.hpp"

#include <algorithm>
#include <numeric>
#include <ostream>

namespace dominators {

namespace {

uint32_t representative(std::vector<uint32_t>& parent, uint32_t v)
{
    while (parent[v] != v) {
        parent[v] = parent[parent[v]];
        v = parent[v];
    }
    return v;
}

}

DBubbleIndex indexDBubbles(const Graph& g, const DBubbleOptions& opt, DBubbleWorkspace* workspace)
{
    if (!opt.threads)
        throw InvalidInput("d-bubbles: thread count must be positive");
    if (!g.n)
        throw InvalidInput("d-bubbles: the input graph is empty");
    if (g.n > (NONE - 2u) / 2u)
        throw std::length_error("d-bubbles: too many nodes for 32 bit IDs in the doubled graph");

    DBubbleIndex result;
    result.source = 2u * g.n;
    const uint32_t states = result.source + 1;
    result.component.resize(g.n);
    std::iota(result.component.begin(), result.component.end(), 0u);

    std::vector<uint8_t> rank(g.n, 0), sides(g.n, 0);
    for (const auto& e : g.edges) {
        if (e.u >= g.n || e.v >= g.n || e.su > 1 || e.sv > 1)
            throw InvalidInput("d-bubbles: invalid bidirected edge");
        sides[e.u] |= static_cast<uint8_t>(1u << e.su);
        sides[e.v] |= static_cast<uint8_t>(1u << e.sv);

        uint32_t a = representative(result.component, e.u);
        uint32_t b = representative(result.component, e.v);
        if (a == b)
            continue;
        if (rank[a] < rank[b])
            std::swap(a, b);
        result.component[b] = a;
        if (rank[a] == rank[b])
            ++rank[a];
    }

    // Number components in the order of their first node.
    for (uint32_t v = 0; v < g.n; ++v)
        result.component[v] = representative(result.component, v);
    std::vector<uint8_t>().swap(rank);
    {
        std::vector<uint32_t> ids(g.n, NONE);
        for (uint32_t v = 0; v < g.n; ++v) {
            const uint32_t root = result.component[v];
            if (ids[root] == NONE)
                ids[root] = result.components++;
        }
        for (auto& c : result.component)
            c = ids[c];
    }

    std::vector<uint32_t> roots;
    for (uint32_t v = 0; v < g.n; ++v) {
        if (sides[v] == 1 || sides[v] == 2)
            roots.push_back(2u * v + (sides[v] == 1 ? 0u : 1u));
        else if (sides[v] == 0) {
            // An isolated node has two source states.
            roots.push_back(2u * v);
            roots.push_back(2u * v + 1u);
        }
    }
    std::vector<uint8_t>().swap(sides);

    const uint64_t arcs = 2 * static_cast<uint64_t>(g.edges.size()) + roots.size();
    if (arcs >= NONE)
        throw std::length_error("d-bubbles: too many edges for 32 bit IDs in the doubled graph");

    std::vector<uint32_t> off(static_cast<size_t>(states) + 1, 0), adj;
    for (const auto& e : g.edges) {
        ++off[2u * e.u + e.su + 1];
        ++off[2u * e.v + e.sv + 1];
    }
    off[result.source + 1] = static_cast<uint32_t>(roots.size());
    for (size_t i = 1; i < off.size(); ++i)
        off[i] += off[i - 1];
    adj.resize(off.back());
    {
        auto cur = off;
        for (const auto& e : g.edges) {
            adj[cur[2u * e.u + e.su]++] = 2u * e.v + (e.sv ^ 1u);
            adj[cur[2u * e.v + e.sv]++] = 2u * e.u + (e.su ^ 1u);
        }
        for (uint32_t v : roots)
            adj[cur[result.source]++] = v;
    }

    std::vector<uint32_t> order;
    auto* pre = workspace ? &workspace->dfsPre : nullptr;
    auto* end = workspace ? &workspace->dfsEnd : nullptr;
    if (opt.bcTrees)
        dominatorTreeBC(states, off, adj, result.source, result.parent, order, opt.threads, pre, end);
    else
        dominatorTree(states, off, adj, result.source, result.parent, order, pre, end);

    for (uint32_t v = 0; v < g.n; ++v) {
        const bool plus = result.parent[2u * v] != NONE;
        const bool minus = result.parent[2u * v + 1u] != NONE;
        result.reachableStates += static_cast<uint32_t>(plus) + static_cast<uint32_t>(minus);

        // Both signs must be reachable from a tip.
        if (plus && minus)
            result.corridorStates += 2;
    }
    if (order.size() != result.reachableStates + 1)
        throw std::logic_error("d-bubbles: tree node count does not match the reachable node count");
    if (workspace) {
        workspace->offsets = std::move(off);
        workspace->edges = std::move(adj);
    }

    return result;
}

uint64_t forEachDBubble(const DBubbleIndex& ix, const std::function<void(uint32_t, uint32_t, uint32_t)>& emit)
{
    if (ix.source == NONE || ix.parent.size() != static_cast<size_t>(ix.source) + 1 ||
        ix.component.size() * 2 != ix.source)
        throw std::logic_error("d-bubbles: graph size does not match the dominator tree");

    uint64_t count = 0;
    for (uint32_t y = 0; y < ix.source; ++y) {
        const uint32_t x = ix.parent[y];
        if (x >= ix.source || x / 2 == y / 2)
            continue;
        // Check postdominance by reversing the signs.
        if (ix.parent[x ^ 1u] != (y ^ 1u))
            continue;
        const uint32_t other = y ^ 1u;

        // Report each bubble in one orientation only.
        if (x > other)
            continue;
        if (ix.component[x / 2] != ix.component[other / 2])
            throw std::logic_error("d-bubbles: ends belong to different components");

        emit(ix.component[x / 2], x, other);
        ++count;
    }
    return count;
}

int runDBubbles(const Graph& g, const std::vector<std::string>& names, const DBubbleOptions& opt,
                bool includeTrivial, std::ostream& out, std::ostream& log)
{
    if (names.size() != g.n)
        throw InvalidInput("d-bubbles: node name count differs from graph size");

    DBubbleWorkspace workspace;
    const auto ix = indexDBubbles(g, opt, opt.ultrabubblesOnly ? &workspace : nullptr);
    if (ix.corridorStates != ix.source)
        log << "[d-bubbles] warning: partial tip coverage\n";

    std::vector<uint8_t> exits;
    if (opt.ultrabubblesOnly) {
        exits = classifyUltrabubbles(ix, workspace.offsets, workspace.edges,
                                    workspace.dfsPre, workspace.dfsEnd);
        workspace = DBubbleWorkspace{};
    }
    const auto visit = [&](const std::function<void(uint32_t, uint32_t, uint32_t)>& emit) {
        return opt.ultrabubblesOnly ? forEachUltrabubble(ix, exits, includeTrivial, emit)
                                   : forEachDBubble(ix, emit);
    };

    uint64_t count = 0;
    std::vector<uint8_t> nontrivial;
    if (includeTrivial || opt.ultrabubblesOnly)
        count = visit([](uint32_t, uint32_t, uint32_t) {});
    else {
        const DBubbleNontrivialFilter filter(g, ix);
        nontrivial.resize((static_cast<size_t>(ix.source) + 7) / 8, 0);
        forEachDBubble(ix, [&](uint32_t component, uint32_t x, uint32_t y) {
            if (!filter.containsInternalVertex(component, x, y))
                return;
            const uint32_t exit = y ^ 1u;
            nontrivial[exit / 8] |= uint8_t(1u << (exit % 8));
            ++count;
        });
    }
    out << count << '\n';

    std::string buffer;
    buffer.reserve(1u << 20);

    uint64_t written = 0;
    visit([&](uint32_t, uint32_t x, uint32_t y) {
        const uint32_t exit = y ^ 1u;
        if (!includeTrivial && !opt.ultrabubblesOnly &&
            !(nontrivial[exit / 8] & uint8_t(1u << (exit % 8))))
            return;
        buffer += names[x / 2];
        buffer += x & 1u ? '-' : '+';
        buffer += ' ';
        buffer += names[y / 2];
        buffer += y & 1u ? '-' : '+';
        buffer += '\n';
        ++written;
        if (buffer.size() >= (1u << 20)) {
            out.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
            buffer.clear();
        }
    });

    out.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    if (!out)
        throw std::runtime_error("d-bubbles: output write failed");
    if (count != written)
        throw std::logic_error("d-bubbles: wrong bubble count in output header");

    return 0;
}

}
