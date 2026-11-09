#pragma once

#include "dominators/directed_tree.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <exception>
#include <mutex>
#include <thread>

namespace dominators {

inline void dominatorTreeBCBlocks(uint32_t n, const std::vector<uint32_t>& off,
                                  const std::vector<uint32_t>& adj, uint32_t root,
                                  std::vector<uint32_t>& idom, std::vector<uint32_t>& order,
                                  unsigned threads = 1, std::vector<uint32_t>* dfsPre = nullptr,
                                  std::vector<uint32_t>* dfsEnd = nullptr, bool reuseVisitOrder = false)
{
    if (!threads)
        throw std::invalid_argument("block dominators: thread count must be positive");
    if (root >= n)
        throw std::invalid_argument("block dominators: root is outside the graph");
    if (off.size() != static_cast<size_t>(n) + 1)
        throw std::invalid_argument("block dominators: edge offset count must be node count + 1");
    if (off.front() != 0)
        throw std::invalid_argument("block dominators: first edge offset must be 0");
    if (off.back() != adj.size())
        throw std::invalid_argument("block dominators: last edge offset must equal edge count");
    for (size_t i = 1; i < off.size(); ++i)
        if (off[i] < off[i - 1])
            throw std::invalid_argument("block dominators: edge offsets go backwards");
    for (uint32_t v : adj)
        if (v >= n)
            throw std::invalid_argument("block dominators: edge end is outside the graph");

    // Visit order identifies each block entry.
    std::vector<uint32_t> dfn(n, NONE);
    if (dfsEnd)
        dfsEnd->assign(n, NONE);
    order.clear();
    if (reuseVisitOrder) {
        if (root != 0)
            throw std::logic_error("block dominators: root must be node 0");
        if (dfsPre || dfsEnd)
            throw std::logic_error("block dominators: visit times require a new graph walk");
        order.resize(n);
        std::iota(order.begin(), order.end(), 0u);
        std::iota(dfn.begin(), dfn.end(), 0u);
    } else {
        dfn[root] = 0;
        order.push_back(root);
        {
            std::vector<std::pair<uint32_t, uint32_t>> stack{{root, off[root]}};
            while (!stack.empty()) {
                const uint32_t u = stack.back().first;
                uint32_t& it = stack.back().second;
                if (it < off[u + 1]) {
                    const uint32_t v = adj[it++];
                    if (dfn[v] == NONE) {
                        dfn[v] = static_cast<uint32_t>(order.size());
                        order.push_back(v);
                        stack.emplace_back(v, off[v]);
                    }
                } else {
                    if (dfsEnd)
                        (*dfsEnd)[u] = static_cast<uint32_t>(order.size());
                    stack.pop_back();
                }
            }
        }
    }
    const uint32_t nr = static_cast<uint32_t>(order.size());

    // Store the entry first. Shared nodes may appear in several blocks.
    std::vector<uint64_t> blockOff{0};
    std::vector<uint32_t> members, entry, home(nr, NONE), local(nr, NONE);
    {
        std::vector<uint32_t> inOff(static_cast<size_t>(nr) + 1, 0), incoming;
        for (uint32_t u : order)
            for (uint32_t e = off[u]; e < off[u + 1]; ++e)
                if (u != adj[e])
                    ++inOff[dfn[adj[e]] + 1];
        for (size_t i = 1; i < inOff.size(); ++i)
            inOff[i] += inOff[i - 1];
        incoming.resize(inOff.back());
        {
            auto cursor = inOff;
            for (uint32_t u = 0; u < nr; ++u)
                for (uint32_t e = off[order[u]]; e < off[order[u] + 1]; ++e)
                    if (order[u] != adj[e])
                        incoming[cursor[dfn[adj[e]]]++] = u;
        }
        std::vector<uint32_t> tin(nr, NONE), low(nr), vertexStack;

        struct Frame
        {
            uint32_t u, out, in;
        };

        std::vector<Frame> stack{{0, off[root], inOff[0]}};
        uint32_t timer = 0;
        tin[0] = low[0] = timer++;
        vertexStack.push_back(0);
        while (!stack.empty()) {
            Frame& frame = stack.back();
            const uint32_t u = frame.u;
            uint32_t v = NONE;
            if (frame.out < off[order[u] + 1])
                v = dfn[adj[frame.out++]];
            else if (frame.in < inOff[u + 1])
                v = incoming[frame.in++];
            else {
                stack.pop_back();
                if (stack.empty())
                    break;
                const uint32_t p = stack.back().u;
                low[p] = std::min(low[p], low[u]);
                if (low[u] >= tin[p]) {
                    const uint32_t b = static_cast<uint32_t>(entry.size());
                    entry.push_back(p);
                    members.push_back(p);
                    uint32_t w;
                    do {
                        if (vertexStack.empty())
                            throw std::logic_error("block dominators: node stack ended before the block");
                        w = vertexStack.back();
                        vertexStack.pop_back();
                        if (w == 0)
                            throw std::logic_error("block dominators: root was removed from the stack");
                        if (home[w] != NONE)
                            throw std::logic_error("block dominators: node already has a block assignment");
                        home[w] = b;
                        local[w] = static_cast<uint32_t>(members.size() - blockOff.back());
                        members.push_back(w);
                    } while (w != u);
                    blockOff.push_back(members.size());
                }
                continue;
            }
            if (v == u || (stack.size() > 1 && v == stack[stack.size() - 2].u))
                continue;
            // Parallel edges do not change the blocks.
            if (tin[v] == NONE) {
                tin[v] = low[v] = timer++;
                vertexStack.push_back(v);
                stack.push_back({v, off[order[v]], inOff[v]});
            } else
                low[u] = std::min(low[u], tin[v]);
        }
        if (timer != nr || vertexStack.size() != 1 || vertexStack[0] != 0)
            throw std::logic_error("block dominators: some reachable nodes have no block");
    }

    auto arcBlock = [&](uint32_t u, uint32_t v) {
        const uint32_t a = home[u], b = home[v];
        if (a != NONE && a == b)
            return a;
        if (a != NONE && entry[a] == v)
            return a;
        if (b != NONE && entry[b] == u)
            return b;
        throw std::logic_error("block dominators: edge has no matching block");
    };

    auto localId = [&](uint32_t b, uint32_t v) {
        if (entry[b] == v)
            return 0u;
        if (home[v] != b || local[v] == NONE)
            throw std::logic_error("block dominators: node is not in this block");
        return local[v];
    };

    std::vector<uint32_t> localOff(members.size() + 1, 0), localAdj;
    for (uint32_t u = 0; u < nr; ++u)
        for (uint32_t e = off[order[u]]; e < off[order[u] + 1]; ++e) {
            const uint32_t v = dfn[adj[e]];
            if (u == v)
                continue;
            const uint32_t b = arcBlock(u, v);
            if (blockOff[b + 1] - blockOff[b] > 2)
                ++localOff[blockOff[b] + localId(b, u) + 1];
        }
    for (size_t i = 1; i < localOff.size(); ++i)
        localOff[i] += localOff[i - 1];
    localAdj.resize(localOff.back());
    {
        auto cursor = localOff;
        for (uint32_t u = 0; u < nr; ++u)
            for (uint32_t e = off[order[u]]; e < off[order[u] + 1]; ++e) {
                const uint32_t v = dfn[adj[e]];
                if (u == v)
                    continue;
                const uint32_t b = arcBlock(u, v);
                if (blockOff[b + 1] - blockOff[b] > 2)
                    localAdj[cursor[blockOff[b] + localId(b, u)]++] = localId(b, v);
            }
    }

    idom.assign(n, NONE);
    const size_t blocks = entry.size();
    std::vector<uint32_t> large;
    for (uint32_t b = 0; b < blocks; ++b)
        if (blockOff[b + 1] - blockOff[b] > 64)
            large.push_back(b);
    std::sort(large.begin(), large.end(), [&](uint32_t a, uint32_t b) {
        const uint64_t ca = blockOff[a + 1] - blockOff[a] + localOff[blockOff[a + 1]] - localOff[blockOff[a]];
        const uint64_t cb = blockOff[b + 1] - blockOff[b] + localOff[blockOff[b + 1]] - localOff[blockOff[b]];
        return ca != cb ? ca > cb : a < b;
    });

    const size_t tasks = large.size() + (blocks + 63) / 64;
    unsigned workers = 1;
    if (blocks >= 128 || large.size() >= 2)
        workers = static_cast<unsigned>(std::min<size_t>(threads, tasks));

    std::atomic<size_t> next{0};
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    std::mutex errorMutex;

    auto worker = [&]() {
        try {
            std::vector<uint32_t> boff, badj, parent, traversal;
            while (!failed.load(std::memory_order_relaxed)) {
                const size_t task = next.fetch_add(1, std::memory_order_relaxed);
                if (task >= tasks)
                    break;
                const bool oneLarge = task < large.size();
                const size_t first = oneLarge ? large[task] : (task - large.size()) * 64;
                const size_t last = oneLarge ? first + 1 : std::min(blocks, first + 64);
                for (size_t b = first; b < last; ++b) {
                    const uint64_t base = blockOff[b];
                    const uint32_t k = static_cast<uint32_t>(blockOff[b + 1] - base);
                    if (!oneLarge && k > 64)
                        continue;
                    auto put = [&](uint32_t v, uint32_t p) {
                        idom[order[members[base + v]]] = order[members[base + p]];
                    };
                    if (k == 2) {
                        // The block entry dominates its only other node.
                        put(1, 0);
                        continue;
                    }
                    if (k <= 64) {
                        std::array<uint64_t, 64> predecessors{}, dom{};
                        const uint64_t all = k == 64 ? UINT64_MAX : (uint64_t{1} << k) - 1;
                        for (uint32_t u = 0; u < k; ++u)
                            for (uint32_t e = localOff[base + u]; e < localOff[base + u + 1]; ++e)
                                predecessors[localAdj[e]] |= uint64_t{1} << u;
                        dom[0] = 1;
                        for (uint32_t u = 1; u < k; ++u)
                            dom[u] = all;
                        bool changed;
                        do {
                            changed = false;
                            for (uint32_t u = 1; u < k; ++u) {
                                uint64_t pred = predecessors[u], value = all;
                                if (!pred)
                                    throw std::logic_error("block dominators: block node has no incoming edge");
                                while (pred) {
                                    const unsigned p = static_cast<unsigned>(__builtin_ctzll(pred));
                                    value &= dom[p];
                                    pred &= pred - 1;
                                }
                                value |= uint64_t{1} << u;
                                changed |= value != dom[u];
                                dom[u] = value;
                            }
                        } while (changed);
                        for (uint32_t u = 1; u < k; ++u) {
                            const uint64_t strict = dom[u] & ~(uint64_t{1} << u);
                            uint64_t remaining = strict;
                            uint32_t parentState = NONE;
                            while (remaining) {
                                const unsigned p = static_cast<unsigned>(__builtin_ctzll(remaining));
                                if (dom[p] == strict) {
                                    parentState = p;
                                    break;
                                }
                                remaining &= remaining - 1;
                            }
                            if (parentState == NONE)
                                throw std::logic_error("block dominators: node has no immediate dominator");
                            put(u, parentState);
                        }
                    } else {
                        const uint32_t firstArc = localOff[base], lastArc = localOff[base + k];
                        boff.resize(static_cast<size_t>(k) + 1);
                        for (uint32_t u = 0; u <= k; ++u)
                            boff[u] = localOff[base + u] - firstArc;
                        badj.assign(localAdj.begin() + firstArc, localAdj.begin() + lastArc);
                        dominatorTree(k, boff, badj, 0, parent, traversal);
                        if (traversal.size() != k)
                            throw std::logic_error("block dominators: block entry cannot reach every node");
                        for (uint32_t u = 1; u < k; ++u) {
                            if (parent[u] == NONE)
                                throw std::logic_error("block dominators: node has no immediate dominator");
                            put(u, parent[u]);
                        }
                    }
                }
            }
        } catch (...) {
            std::lock_guard<std::mutex> guard(errorMutex);
            if (!error)
                error = std::current_exception();
            failed.store(true, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> pool;
    try {
        for (unsigned i = 1; i < workers; ++i)
            pool.emplace_back(worker);
        worker();
    } catch (...) {
        failed.store(true, std::memory_order_relaxed);
        for (auto& thread : pool)
            thread.join();
        throw;
    }
    for (auto& thread : pool)
        thread.join();
    if (error)
        std::rethrow_exception(error);
    for (uint32_t v : order)
        if (v != root && idom[v] == NONE)
            throw std::logic_error("block dominators: node has no immediate dominator");
    if (dfsPre)
        *dfsPre = std::move(dfn);
}

// Shrink chains to reduce the work on each block.
inline void dominatorTreeBC(uint32_t n, const std::vector<uint32_t>& off, const std::vector<uint32_t>& adj,
                            uint32_t root, std::vector<uint32_t>& idom, std::vector<uint32_t>& order,
                            unsigned threads = 1, std::vector<uint32_t>* dfsPre = nullptr,
                            std::vector<uint32_t>* dfsEnd = nullptr)
{
    if (!threads)
        throw std::invalid_argument("block dominators: thread count must be positive");
    if (root >= n)
        throw std::invalid_argument("block dominators: root is outside the graph");
    if (off.size() != static_cast<size_t>(n) + 1)
        throw std::invalid_argument("block dominators: edge offset count must be node count + 1");
    if (off.front() != 0)
        throw std::invalid_argument("block dominators: first edge offset must be 0");
    if (off.back() != adj.size())
        throw std::invalid_argument("block dominators: last edge offset must equal edge count");
    for (size_t i = 1; i < off.size(); ++i)
        if (off[i] < off[i - 1])
            throw std::invalid_argument("block dominators: edge offsets go backwards");
    for (uint32_t v : adj)
        if (v >= n)
            throw std::invalid_argument("block dominators: edge end is outside the graph");

    std::vector<uint32_t> rank(n, NONE), exit;
    order.clear();
    rank[root] = 0;
    order.push_back(root);
    exit.push_back(0);
    {
        std::vector<std::pair<uint32_t, uint32_t>> stack{{root, off[root]}};
        while (!stack.empty()) {
            const uint32_t u = stack.back().first;
            uint32_t& edge = stack.back().second;
            if (edge < off[u + 1]) {
                const uint32_t v = adj[edge++];
                if (rank[v] == NONE) {
                    rank[v] = static_cast<uint32_t>(order.size());
                    order.push_back(v);
                    exit.push_back(0);
                    stack.emplace_back(v, off[v]);
                }
            } else {
                exit[rank[u]] = static_cast<uint32_t>(order.size());
                stack.pop_back();
            }
        }
    }
    const uint32_t nr = static_cast<uint32_t>(order.size());

    constexpr uint8_t IN = 1, MULTI_IN = 2, OUT = 4, MULTI_OUT = 8, REMOVED = 16, BACK_HEAD = 32;
    std::vector<uint8_t> flags(nr, 0);
    std::vector<uint32_t> predecessor(nr, NONE), successor(nr, NONE), target(nr, NONE), retained;
    for (uint32_t u = 0; u < nr; ++u)
        for (uint32_t e = off[order[u]]; e < off[order[u] + 1]; ++e) {
            const uint32_t v = rank[adj[e]];
            if (u == v)
                continue;
            if (flags[v] & IN)
                flags[v] |= MULTI_IN;
            else {
                predecessor[v] = u;
                flags[v] |= IN;
            }
            if (flags[u] & OUT)
                flags[u] |= MULTI_OUT;
            else {
                successor[u] = v;
                flags[u] |= OUT;
            }
            if (v < u && u < exit[v])
                flags[v] |= BACK_HEAD;
        }
    for (uint32_t u = 0; u < nr; ++u) {
        if (u && (flags[u] & (IN | MULTI_IN | OUT | MULTI_OUT)) == (IN | OUT))
            flags[u] |= REMOVED;
        else {
            target[u] = static_cast<uint32_t>(retained.size());
            retained.push_back(u);
        }
    }

    // Process chain ends before the nodes that lead to them.
    for (uint32_t u = nr; u-- > 0;)
        if (flags[u] & REMOVED) {
            const uint32_t v = successor[u];
            if (v >= nr || target[v] == NONE)
                throw std::logic_error("block dominators: chain end was not found");
            target[u] = target[v];
            successor[u] = flags[v] & REMOVED ? successor[v] : u;
        }

    const uint32_t k = static_cast<uint32_t>(retained.size());
    std::vector<uint8_t> needsRestore(k, 0);
    for (uint32_t v = 1; v < k; ++v)
        needsRestore[v] = (flags[retained[v]] & (MULTI_IN | BACK_HEAD)) == (MULTI_IN | BACK_HEAD);


    struct IncomingChain
    {
        uint32_t from, to, tail;
    };

    std::vector<IncomingChain> incomingChains;
    std::vector<uint32_t> reducedOff{0}, reducedAdj;
    reducedOff.reserve(static_cast<size_t>(k) + 1);
    for (uint32_t u = 0; u < k; ++u) {
        const uint32_t original = retained[u];
        for (uint32_t e = off[order[original]]; e < off[order[original] + 1]; ++e) {
            const uint32_t neighbor = rank[adj[e]], v = target[neighbor];
            reducedAdj.push_back(v);
            if (needsRestore[v])
                incomingChains.push_back({u, v, flags[neighbor] & REMOVED ? successor[neighbor] : original});
        }
        reducedOff.push_back(static_cast<uint32_t>(reducedAdj.size()));
    }

    std::vector<uint32_t> parents, localOrder;
    // The remaining nodes keep their visit order.
    dominatorTreeBCBlocks(k, reducedOff, reducedAdj, 0, parents, localOrder, threads, nullptr, nullptr, true);
    idom.assign(n, NONE);
    for (uint32_t u = 1; u < nr; ++u)
        if (flags[u] & REMOVED)
            idom[order[u]] = order[predecessor[u]];
    for (uint32_t v = 1; v < k; ++v) {
        if (parents[v] >= v)
            throw std::logic_error("block dominators: a tree parent must come before its child");
        const uint32_t original = retained[v];
        if (!(flags[original] & MULTI_IN)) {
            if (!(flags[original] & IN))
                throw std::logic_error("block dominators: remaining node has no incoming edge");
            idom[order[original]] = order[predecessor[original]];
        } else
            idom[order[original]] = order[retained[parents[v]]];
    }

    std::vector<uint32_t> domPre, domEnd;
    auto indexDominators = [&]() {
        domPre.assign(k, NONE);
        domEnd.resize(k);
        std::vector<uint32_t> childOff(static_cast<size_t>(k) + 1, 0), children(k - 1);
        for (uint32_t v = 1; v < k; ++v)
            ++childOff[parents[v] + 1];
        for (size_t v = 1; v < childOff.size(); ++v)
            childOff[v] += childOff[v - 1];
        auto cursor = childOff;
        for (uint32_t v = 1; v < k; ++v)
            children[cursor[parents[v]]++] = v;
        std::vector<std::pair<uint32_t, uint32_t>> stack{{0, childOff[0]}};
        uint32_t timer = 1;
        domPre[0] = 0;
        while (!stack.empty()) {
            const uint32_t u = stack.back().first;
            uint32_t& edge = stack.back().second;
            if (edge < childOff[u + 1]) {
                const uint32_t v = children[edge++];
                domPre[v] = timer++;
                stack.emplace_back(v, childOff[v]);
            } else {
                domEnd[u] = timer;
                stack.pop_back();
            }
        }
        if (timer != k)
            throw std::logic_error("block dominators: tree walk missed a remaining node");
    };

    uint64_t climbed = 0;
    const uint64_t climbLimit = static_cast<uint64_t>(k) + incomingChains.size();
    auto dominated = [&](uint32_t v, uint32_t u) {
        if (u == v)
            return true;
        const uint32_t a = retained[u], b = retained[v];
        if (a < b || a >= exit[b])
            return false;
        if (!domPre.empty())
            return domPre[v] <= domPre[u] && domPre[u] < domEnd[v];
        uint32_t p = u;
        while (p > v) {
            if (++climbed > climbLimit) {
                indexDominators();
                return domPre[v] <= domPre[u] && domPre[u] < domEnd[v];
            }
            p = parents[p];
        }
        return p == v;
    };

    std::vector<uint8_t> entries(k, 0);
    std::vector<uint32_t> tail(k, NONE);
    for (const auto& edge : incomingChains)
        if (entries[edge.to] < 2 && !dominated(edge.to, edge.from)) {
            if (!entries[edge.to])
                tail[edge.to] = edge.tail;
            ++entries[edge.to];
        }
    for (uint32_t v = 1; v < k; ++v)
        if (needsRestore[v]) {
            if (!entries[v])
                throw std::logic_error("block dominators: no edge enters from outside the dominator subtree");
            if (entries[v] == 1) {
                idom[order[retained[v]]] = order[tail[v]];
            }
        }
    if (dfsEnd) {
        dfsEnd->assign(n, NONE);
        for (uint32_t u = 0; u < nr; ++u)
            (*dfsEnd)[order[u]] = exit[u];
    }
    if (dfsPre)
        *dfsPre = std::move(rank);
}

}
