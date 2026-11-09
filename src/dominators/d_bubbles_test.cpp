#include "dominators/d_bubbles.hpp"

#include <algorithm>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>

using namespace dominators;
using Bubble = std::pair<uint32_t, uint32_t>;
using Bubbles = std::set<Bubble>;
using Arcs = std::vector<std::vector<uint32_t>>;
static uint64_t cases = 0, checks = 0, refusals = 0;

static void require(bool v, const char* message)
{
    if (!v)
        throw std::runtime_error(message);
}

static std::vector<uint8_t> reach(const Arcs& a, uint32_t root, uint32_t hidden = NONE)
{
    std::vector<uint8_t> seen(a.size(), 0);
    std::vector<uint32_t> todo;
    if (root != hidden) {
        seen[root] = 1;
        todo.push_back(root);
    }
    for (size_t i = 0; i < todo.size(); ++i)
        for (auto v : a[todo[i]])
            if (v != hidden && !seen[v]) {
                seen[v] = 1;
                todo.push_back(v);
            }
    return seen;
}

static std::vector<uint32_t> literalParents(const Arcs& a, uint32_t root)
{
    const auto seen = reach(a, root);
    std::vector<std::vector<uint8_t>> dom(a.size(), std::vector<uint8_t>(a.size(), 0));
    for (uint32_t p = 0; p < a.size(); ++p) {
        auto removed = reach(a, root, p);
        for (uint32_t v = 0; v < a.size(); ++v) {
            dom[p][v] = seen[v] && !removed[v];
            ++checks;
        }
    }
    std::vector<uint32_t> result(a.size(), NONE);
    for (uint32_t v = 0; v < a.size(); ++v) {
        if (v == root || !seen[v])
            continue;
        for (uint32_t p = 0; p < a.size(); ++p) {
            if (p == v || !dom[p][v])
                continue;
            bool closest = true;
            for (uint32_t q = 0; q < a.size(); ++q)
                if (q != v && dom[q][v] && !dom[q][p])
                    closest = false;
            if (closest) {
                require(result[v] == NONE, "the direct check found more than one immediate dominator");
                result[v] = p;
            }
        }
        require(result[v] != NONE, "the direct check found no immediate dominator");
    }
    return result;
}

struct Gold
{
    bool valid = false;
    std::vector<uint32_t> source;
    std::vector<uint8_t> sourceReach, targetReach;
    Bubbles bubbles;
};

static Gold oracle(const Graph& g)
{
    Gold gold;
    if (!g.n)
        return gold;
    std::vector<uint8_t> sides(g.n, 0), visited(g.n, 0);
    Arcs physical(g.n), a(2 * g.n + 2), reverse(a.size());
    for (const auto& e : g.edges) {
        if (e.u >= g.n || e.v >= g.n || e.su > 1 || e.sv > 1)
            return gold;
        physical[e.u].push_back(e.v);
        physical[e.v].push_back(e.u);
        sides[e.u] |= 1u << e.su;
        sides[e.v] |= 1u << e.sv;
        a[2 * e.u + e.su].push_back(2 * e.v + (e.sv ^ 1u));
        a[2 * e.v + e.sv].push_back(2 * e.u + (e.su ^ 1u));
    }
    const uint32_t S = 2 * g.n, T = S + 1;
    for (uint32_t seed = 0; seed < g.n; ++seed) {
        if (visited[seed])
            continue;
        auto component = reach(physical, seed);
        for (uint32_t v = 0; v < g.n; ++v) {
            if (!component[v])
                continue;
            visited[v] = 1;
            if (sides[v] == 1 || sides[v] == 2) {
                const uint32_t exit = 2 * v + (sides[v] == 1 ? 0 : 1);
                a[S].push_back(exit);
                a[exit ^ 1u].push_back(T);
            } else if (!sides[v]) {
                a[S].push_back(2 * v);
                a[S].push_back(2 * v + 1);
                a[2 * v].push_back(T);
                a[2 * v + 1].push_back(T);
            }
        }
    }
    for (uint32_t u = 0; u < a.size(); ++u)
        for (auto v : a[u])
            reverse[v].push_back(u);
    gold.source = literalParents(a, S);
    const auto target = literalParents(reverse, T);
    gold.sourceReach = reach(a, S);
    gold.targetReach = reach(reverse, T);
    for (uint32_t x = 0; x < S; ++x)
        for (uint32_t y = 0; y < S; ++y)
            if (x / 2 != y / 2 && gold.source[y] == x && target[x] == y)
                gold.bubbles.emplace(std::min(x, y ^ 1u), std::max(x, y ^ 1u));
    gold.valid = true;
    return gold;
}

static Bubbles compare(const Graph& g)
{
    const auto gold = oracle(g);
    Bubbles final;
    for (unsigned backend = 0; backend < 3; ++backend) {
        DBubbleOptions opt;
        opt.bcTrees = backend != 0;
        opt.threads = backend == 2 ? 4 : 1;
        try {
            const auto ix = indexDBubbles(g, opt);
            require(gold.valid, "invalid input graph accepted");
            uint64_t reached = 0, corridor = 0;
            for (uint32_t v = 0; v < 2 * g.n; ++v) {
                require(ix.parent[v] == gold.source[v],
                        "tree parent disagrees with the node removal check");
                reached += gold.sourceReach[v];
                corridor += gold.sourceReach[v] && gold.targetReach[v];
            }
            require(ix.reachableStates == reached && ix.corridorStates == corridor,
                    "reachable node counts in the doubled graph disagree with the direct search");
            Bubbles actual;
            const auto count = forEachDBubble(ix, [&](uint32_t, uint32_t x, uint32_t y) {
                require(x < y && x / 2 != y / 2, "bubble ends are out of order or use the same node");
                require(gold.sourceReach[x] && gold.targetReach[x] && gold.sourceReach[y] &&
                            gold.targetReach[y],
                        "bubble has a vacuous endpoint relation");
                require(actual.emplace(x, y).second, "duplicate output bubble");
            });
            require(count == actual.size() && actual == gold.bubbles,
                    "bubble set differs from two independent S/T relations");
            final = actual;
        } catch (const InvalidInput&) {
            require(!gold.valid, "valid graph rejected");
            ++refusals;
        }
        ++cases;
    }
    return final;
}

int main()
{
    try {
        // Test every signed graph with two nodes.
        std::vector<BiEdge> possible;
        for (uint32_t u = 0; u < 2; ++u)
            for (uint32_t v = u; v < 2; ++v)
                for (uint8_t a = 0; a < 2; ++a)
                    for (uint8_t b = 0; b < 2; ++b)
                        if (u != v || a <= b)
                            possible.push_back({u, v, a, b});
        for (unsigned mask = 0; mask < (1u << possible.size()); ++mask) {
            Graph g;
            g.n = 2;
            for (unsigned i = 0; i < possible.size(); ++i)
                if (mask & (1u << i))
                    g.edges.push_back(possible[i]);
            compare(g);
        }
        std::mt19937 random(20261028);
        for (unsigned i = 0; i < 1200; ++i) {
            Graph g;
            g.n = 3 + random() % 7;
            std::vector<uint8_t> flip(g.n);
            for (auto& v : flip)
                v = random() & 1;
            for (uint32_t v = 1; v < g.n; ++v)
                g.edges.push_back({v - 1, v, flip[v - 1], static_cast<uint8_t>(flip[v] ^ 1u)});
            for (unsigned j = 0; j < i % 13; ++j)
                g.edges.push_back({static_cast<uint32_t>(1 + random() % (g.n - 2)),
                                   static_cast<uint32_t>(1 + random() % (g.n - 2)),
                                   static_cast<uint8_t>(random() & 1), static_cast<uint8_t>(random() & 1)});
            compare(g);
        }
        // Check a graph with a cycle.
        Graph atlas;
        atlas.n = 8;
        atlas.edges = {{0, 1, 0, 1}, {1, 2, 0, 1}, {1, 3, 0, 1}, {2, 4, 0, 1}, {3, 4, 0, 1},
                       {4, 5, 0, 1}, {5, 6, 0, 1}, {6, 7, 0, 1}, {1, 6, 1, 0}};
        const auto bubbles = compare(atlas);
        require(!bubbles.count({0, 15}) && bubbles.count({2, 9}),
                "atlas outer-bubble rejection or inner-bubble detection failed");
        compare(Graph{});
        Graph unreachable;
        unreachable.n = 3;
        unreachable.edges = {{0, 1, 0, 1}, {0, 2, 0, 0}, {2, 2, 0, 1}};
        compare(unreachable);
        Graph two;
        two.n = 4;
        two.edges = {{0, 1, 0, 1}, {2, 3, 1, 0}};
        compare(two);
        Graph threeTips;
        threeTips.n = 4;
        threeTips.edges = {{0, 1, 0, 1}, {1, 2, 0, 1}, {1, 3, 0, 1}};
        require(!compare(threeTips).empty(), "no bubble was found in the graph with three tips");
        Graph oneTip;
        oneTip.n = 2;
        oneTip.edges = {{0, 1, 0, 1}, {1, 1, 0, 0}};
        compare(oneTip);
        Graph isolate;
        isolate.n = 1;
        require(compare(isolate).empty(), "an isolated node produced a bubble with two different ends");
        Graph tipless;
        tipless.n = 3;
        tipless.edges = {{0, 1, 0, 1}, {1, 2, 0, 1}, {2, 0, 0, 1}};
        require(compare(tipless).empty(), "a cycle with no tips produced a bubble");
        Graph chain;
        chain.n = 200001;
        for (uint32_t v = 1; v < chain.n; ++v)
            chain.edges.push_back({v - 1, v, 0, 1});
        for (bool bc : {false, true}) {
            DBubbleOptions opt;
            opt.bcTrees = bc;
            opt.threads = 4;
            auto ix = indexDBubbles(chain, opt);
            require(forEachDBubble(ix,
                                   [](uint32_t c, uint32_t x, uint32_t y) {
                                       require(c == 0 && !(x & 1u) && y == x + 3, "long-chain bubble");
                                   }) == chain.n - 1,
                    "long-chain count");
        }
        std::cout << "{\"checks\":\"PASS\",\"new_graph_backend_cases\":" << cases
                  << ",\"literal_vertex_deletion_checks\":" << checks
                  << ",\"expected_invalid_input_refusals\":" << refusals
                  << ",\"long_chain_nodes\":200001,\"old_suite_replayed\":false}\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
