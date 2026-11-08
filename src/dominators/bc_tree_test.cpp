#include "dominators/bc_tree.hpp"

#include <iostream>
#include <random>
#include <string>

using namespace dominators;
using Edges = std::vector<std::vector<uint32_t>>;
static uint64_t checked = 0, dominanceChecks = 0;

static void verify(const Edges& edges, uint32_t root, unsigned threads = 1)
{
    const uint32_t n = static_cast<uint32_t>(edges.size());
    std::vector<uint32_t> off{0}, adj;
    for (const auto& row : edges) {
        adj.insert(adj.end(), row.begin(), row.end());
        off.push_back(static_cast<uint32_t>(adj.size()));
    }

    std::vector<uint32_t> expected, oldOrder, oldPre, oldEnd;
    dominatorTree(n, off, adj, root, expected, oldOrder, &oldPre, &oldEnd);
    std::vector<uint32_t> actual, order, pre, end;
    dominatorTreeBC(n, off, adj, root, actual, order, threads, &pre, &end);
    if (actual != expected || order != oldOrder || pre != oldPre || end != oldEnd)
        throw std::logic_error("the block tree does not match the full graph tree or DFS order");
    ++checked;

    // Remove each node and check which nodes remain reachable.
    if (n > 100)
        return;
    auto visit = [&](uint32_t hidden) {
        std::vector<uint8_t> seen(n, 0);
        std::vector<uint32_t> todo;
        if (hidden != root) {
            seen[root] = 1;
            todo.push_back(root);
        }
        while (!todo.empty()) {
            const uint32_t u = todo.back();
            todo.pop_back();
            for (uint32_t v : edges[u])
                if (v != hidden && !seen[v]) {
                    seen[v] = 1;
                    todo.push_back(v);
                }
        }
        return seen;
    };
    const auto reachable = visit(NONE);
    for (uint32_t hidden = 0; hidden < n; ++hidden) {
        if (n > 16 && hidden % 11 && hidden != root && hidden + 1 != n)
            continue;
        const auto seen = visit(hidden);
        for (uint32_t v = 0; v < n; ++v) {
            if (v == hidden || !reachable[v])
                continue;
            bool ancestor = false;
            uint32_t steps = 0;
            for (uint32_t p = actual[v]; p != NONE; p = actual[p]) {
                if (p >= n || ++steps > n)
                    throw std::logic_error("a tree parent is outside the graph or forms a cycle");
                ancestor |= p == hidden;
            }
            if (ancestor != !seen[v])
                throw std::logic_error("the dominator tree disagrees with the node removal check");
            ++dominanceChecks;
        }
    }
}

int main(int argc, char** argv)
{
    try {
        const bool quick = argc == 2 && std::string(argv[1]) == "--quick";
        if (argc != 1 && !quick)
            throw std::invalid_argument("expected no argument or --quick");
        for (uint32_t n = 1; n <= (quick ? 3u : 4u); ++n) {
            const uint64_t count = uint64_t{1} << (n * n);
            for (uint64_t mask = 0; mask < count; ++mask) {
                Edges graph(n);
                for (uint32_t u = 0; u < n; ++u)
                    for (uint32_t v = 0; v < n; ++v)
                        if (mask & (uint64_t{1} << (u * n + v)))
                            graph[u].push_back(v);
                for (uint32_t root = 0; root < n; ++root)
                    verify(graph, root);
            }
        }

        std::mt19937_64 rng(0xBC20261018ULL);
        const uint32_t randomCases = quick ? 200 : 3000;
        for (uint32_t i = 0; i < randomCases; ++i) {
            const uint32_t n = 1 + rng() % 90;
            Edges graph(n);
            for (uint32_t e = 0, m = rng() % (5 * n + 1); e < m; ++e) {
                const uint32_t u = rng() % n, v = rng() % n;
                graph[u].push_back(v);
                if (rng() % 4 == 0)
                    graph[u].push_back(v);
            }
            verify(graph, rng() % n);
        }
        for (uint32_t i = 0; i < randomCases; ++i) {
            const uint32_t segments = 1 + rng() % 35, n = 2 * segments;
            Edges graph(n + (i % 5 == 0 ? 2 : 0));
            for (uint32_t e = 0, m = rng() % (5 * segments + 1); e < m; ++e) {
                const uint32_t a = rng() % n, b = rng() % n;
                graph[a].push_back(b ^ 1u);
                graph[b].push_back(a ^ 1u);
            }
            uint32_t root = rng() % n;
            if (graph.size() != n) {
                root = n;
                for (uint32_t r = 0; r < 1 + i % 7; ++r)
                    graph[root].push_back(rng() % n);
            }
            verify(graph, root);
        }

        // Long paths exercise chain contraction.
        Edges chain(12000);
        for (uint32_t v = 1; v < chain.size(); ++v)
            chain[v - 1].push_back(v);

        Edges diamonds(12001);
        for (uint32_t b = 0; b < 4000; ++b) {
            const uint32_t x = 3 * b;
            diamonds[x].push_back(x + 1);
            diamonds[x].push_back(x + 2);
            diamonds[x + 1].push_back(x + 3);
            diamonds[x + 2].push_back(x + 3);
            if (b % 5 == 0)
                diamonds[x + 1].push_back(x + 1);
        }

        Edges hub(8001);
        for (uint32_t v = 1; v < hub.size(); ++v) {
            hub[0].push_back(v);
            hub[v].push_back(0);
            hub[v].push_back(0);
        }

        Edges large(1 + 4 * 97);
        for (uint32_t b = 0; b < 4; ++b) {
            const uint32_t first = 1 + b * 97;
            large[0].push_back(first);
            for (uint32_t v = first; v + 1 < first + 97; ++v)
                large[v].push_back(v + 1);
            large[first + 96].push_back(0);
        }

        // Back edges make the restored parent less obvious.
        Edges nestedBack(2001);
        for (uint32_t v = 1; v < nestedBack.size(); ++v)
            nestedBack[v - 1].push_back(v);
        for (uint32_t v = 2; v + 1 < nestedBack.size(); v += 2)
            nestedBack.back().push_back(v);

        for (unsigned threads : {1u, 2u, 8u}) {
            verify(chain, 0, threads);
            verify(chain, 6000, threads);
            verify(diamonds, 0, threads);
            verify(diamonds, 6001, threads);
            verify(hub, 0, threads);
            verify(hub, 500, threads);
            verify(large, 0, threads);
            verify(large, 100, threads);
            verify(nestedBack, 0, threads);
        }

        uint32_t refused = 0;
        auto refuse = [&](uint32_t n, const std::vector<uint32_t>& off, const std::vector<uint32_t>& adj,
                          uint32_t root, unsigned threads) {
            std::vector<uint32_t> parent, order;
            try {
                dominatorTreeBC(n, off, adj, root, parent, order, threads);
            } catch (const std::invalid_argument&) {
                ++refused;
            }
        };
        refuse(0, {0}, {}, 0, 1);
        refuse(1, {0, 0}, {}, 1, 1);
        refuse(1, {0, 0}, {}, 0, 0);
        refuse(2, {0, 0}, {}, 0, 1);
        refuse(2, {0, 2, 1}, {0}, 0, 1);
        refuse(2, {1, 1, 1}, {0}, 0, 1);
        refuse(2, {0, 0, 1}, {2}, 0, 1);
        refuse(2, {0, 0, 0}, {0}, 0, 1);
        if (refused != 8)
            throw std::logic_error("invalid graph arrays, root or thread count were accepted");
        std::cout << "{\"checks\":\"PASS\",\"cases\":" << checked
                  << ",\"independent_deletion_pairs\":" << dominanceChecks << ",\"intended_refusals\":" << refused
                  << ",\"quick\":" << (quick ? "true" : "false") << "}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
