#include "dominators/d_bubble_hierarchy.hpp"
#include "dominators/d_bubble_nontrivial.hpp"

#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace dominators;
static uint64_t graphChecks = 0, bodyChecks = 0, nontrivialChecks = 0, boundaryDetourChecks = 0;

static void check(bool ok, const char* message)
{
    if (!ok)
        throw std::runtime_error(message);
}

static bool ancestor(const DBubbleIndex& ix, uint32_t a, uint32_t z)
{
    for (size_t steps = 0; z != NONE; ++steps) {
        check(z < ix.parent.size() && steps <= ix.parent.size(), "the direct check found a missing tree parent or a parent cycle");
        if (a == z)
            return true;
        z = ix.parent[z];
    }
    return false;
}

static uint64_t literalSize(const DBubbleIndex& ix, const paper_body_native::Bubble& p)
{
    std::vector<uint8_t> seen(ix.component.size(), 0);
    seen[p.x / 2] = seen[p.y / 2] = 1;
    for (uint32_t z = 0; z < ix.source; ++z)
        if (ancestor(ix, p.x, z) && !ancestor(ix, p.y ^ 1u, z) && ancestor(ix, p.y, z ^ 1u) &&
            !ancestor(ix, p.x ^ 1u, z ^ 1u))
            seen[z / 2] = 1;
    uint64_t count = 0;
    for (auto v : seen)
        count += v;
    return count;
}

static bool literalInterior(const DBubbleIndex& ix, const paper_body_native::Bubble& p, uint32_t z)
{
    return ancestor(ix, p.x, z) && !ancestor(ix, p.y ^ 1u, z) && ancestor(ix, p.y, z ^ 1u) &&
           !ancestor(ix, p.x ^ 1u, z ^ 1u);
}

static bool directInternalArc(const Graph& g, const DBubbleIndex& ix, const paper_body_native::Bubble& p)
{
    const auto direct = [&](uint32_t from, uint32_t to) {
        return from == p.x && to / 2 != p.x / 2 && to / 2 != p.y / 2 && literalInterior(ix, p, to);
    };
    for (const auto& e : g.edges)
        if (direct(2u * e.u + e.su, 2u * e.v + (e.sv ^ 1u)) ||
            direct(2u * e.v + e.sv, 2u * e.u + (e.su ^ 1u)))
            return true;
    return false;
}

static std::string row(const paper_body_native::Bubble& bubble, const std::vector<std::string>& names)
{
    const auto& firstName = names[bubble.x / 2];
    const auto& secondName = names[bubble.y / 2];
    const char firstSign = (bubble.x & 1u) ? '-' : '+';
    const char secondSign = (bubble.y & 1u) ? '-' : '+';

    std::string line = firstName;
    line += firstSign;
    line += '\t';
    line += secondName;
    line += secondSign;
    return line;
}

static std::set<std::string> standardRecords(const std::string& output)
{
    std::istringstream input(output);
    std::string line;
    check(static_cast<bool>(std::getline(input, line)), "missing count line");
    size_t consumed = 0;
    const auto count = std::stoull(line, &consumed);
    check(consumed == line.size(), "invalid count line");
    std::set<std::string> result;
    while (std::getline(input, line)) {
        std::istringstream row(line);
        std::string x, y, extra;
        check(static_cast<bool>(row >> x >> y) && !(row >> extra), "invalid standard bubble line");
        check(result.insert(x + "\t" + y).second, "duplicate standard bubble");
    }
    check(result.size() == count, "standard count differs from emitted bubbles");
    return result;
}

static void exercise(const Graph& g, DBubbleOptions options)
{
    const auto ix = indexDBubbles(g, options);
    std::vector<paper_body_native::Bubble> bubbles;
    forEachDBubble(ix, [&](uint32_t c, uint32_t x, uint32_t y) { bubbles.push_back({c, x, y}); });
    std::vector<std::string> names;
    for (uint32_t i = 0; i < g.n; ++i)
        names.push_back("v" + std::to_string(i));
    std::set<std::string> allExpected, keepExpected;
    const DBubbleNontrivialFilter fast(g, ix);
    for (const auto& p : bubbles) {
        const auto size = literalSize(ix, p);
        check(!literalInterior(ix, p, p.x ^ 1u) && !literalInterior(ix, p, p.y),
              "the bubble contains the wrong sign of one of its ends");
        check(fast.containsInternalVertex(p.component, p.x, p.y) == (size > 2),
              "the filter result does not match the bubble node count");
        allExpected.insert(row(p, names));
        ++bodyChecks;
        if (size > 2) {
            keepExpected.insert(row(p, names));
            ++nontrivialChecks;
            if (!directInternalArc(g, ix, p))
                ++boundaryDetourChecks;
        }
    }
    std::ostringstream all, allLog, filtered, filteredLog;
    check(runDBubbles(g, names, options, true, all, allLog) == 0, "all bubbles failed");
    check(runDBubbles(g, names, options, false, filtered, filteredLog) == 0, "filter failed");
    check(standardRecords(all.str()) == allExpected, "all bubbles differ");
    check(standardRecords(filtered.str()) == keepExpected, "filtered bubbles differ");
    if (ix.corridorStates != ix.source)
        check(allLog.str().find("warning: partial tip coverage") != std::string::npos,
              "missing partial coverage warning");
    ++graphChecks;
}

int main()
{
    try {
        Graph diamond;
        diamond.n = 7;
        diamond.edges = {{0, 1, 0, 1}, {1, 2, 0, 1}, {1, 3, 0, 1}, {2, 4, 0, 1},
                         {3, 4, 0, 1}, {4, 5, 0, 1}, {5, 6, 0, 1}};
        for (bool bc : {false, true}) {
            DBubbleOptions o;
            o.bcTrees = bc;
            exercise(diamond, o);
        }
        std::mt19937 rng(20261102);
        for (unsigned trial = 0; trial < 5000; ++trial) {
            Graph g;
            g.n = 2 + rng() % 9;
            // Test graphs with several tips and graphs with full coverage.
            if (trial % 2 == 0)
                for (uint32_t i = 1; i < g.n; ++i)
                    g.edges.push_back({i - 1, i, 0, 1});
            for (unsigned i = 0; i < trial % 20; ++i)
                g.edges.push_back({static_cast<uint32_t>(rng() % g.n), static_cast<uint32_t>(rng() % g.n),
                                   static_cast<uint8_t>(rng() & 1u), static_cast<uint8_t>(rng() & 1u)});
            DBubbleOptions o;
            o.bcTrees = (trial & 1u);
            o.threads = 1;
            exercise(g, o);
        }
        // Add edges with different signs to a graph with two paths.
        for (unsigned trial = 0; trial < 2000; ++trial) {
            Graph g;
            g.n = 6;
            g.edges = {{0, 1, 0, 1}, {1, 2, 0, 1}, {1, 3, 0, 1}, {2, 4, 0, 1}, {3, 4, 0, 1}, {4, 5, 0, 1}};
            for (unsigned i = 0; i < trial % 6; ++i)
                g.edges.push_back({static_cast<uint32_t>(2u + rng() % 2),
                                   static_cast<uint32_t>(2u + rng() % 2), static_cast<uint8_t>(rng() & 1u),
                                   static_cast<uint8_t>(rng() & 1u)});
            DBubbleOptions o;
            o.bcTrees = (trial & 1u);
            o.threads = 1;
            exercise(g, o);
        }
        Graph disconnected;
        disconnected.n = 7;
        disconnected.edges = {{0, 1, 0, 1}, {1, 2, 0, 1}, {1, 3, 0, 1},
                              {2, 4, 0, 1}, {3, 4, 0, 1}, {5, 6, 0, 1}};
        DBubbleOptions disconnectedOptions;
        disconnectedOptions.bcTrees = true;
        exercise(disconnected, disconnectedOptions);
        check(bodyChecks > 100 && nontrivialChecks > 20 && boundaryDetourChecks == 0,
              "too few bubbles were checked or a bubble had no direct edge to its interior");
        std::cout << "{\"checks\":\"PASS\",\"graphs\":" << graphChecks
                  << ",\"literal_body_checks\":" << bodyChecks
                  << ",\"nontrivial_checks\":" << nontrivialChecks
                  << ",\"boundary_twin_detours\":" << boundaryDetourChecks << "}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
