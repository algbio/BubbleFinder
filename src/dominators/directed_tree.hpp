#pragma once

#include "dominators/dominators.hpp"

#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dominators {

// The root and unreachable nodes keep NONE as their parent.
// order contains the reachable nodes in DFS order.
inline void dominatorTree(uint32_t n, const std::vector<uint32_t>& off, const std::vector<uint32_t>& adj,
                          uint32_t root, std::vector<uint32_t>& idom, std::vector<uint32_t>& order,
                          std::vector<uint32_t>* dfsPre = nullptr, std::vector<uint32_t>* dfsEnd = nullptr)
{
    if (root >= n)
        throw std::invalid_argument("dominator tree: root is outside the graph");
    if (off.size() != static_cast<size_t>(n) + 1)
        throw std::invalid_argument("dominator tree: edge offset count must be node count + 1");
    if (off.front() != 0)
        throw std::invalid_argument("dominator tree: first edge offset must be 0");
    if (off.back() != adj.size())
        throw std::invalid_argument("dominator tree: last edge offset must equal edge count");
    for (size_t i = 1; i < off.size(); ++i)
        if (off[i] < off[i - 1])
            throw std::invalid_argument("dominator tree: edge offsets go backwards");
    for (uint32_t v : adj)
        if (v >= n)
            throw std::invalid_argument("dominator tree: edge end is outside the graph");

    std::vector<uint32_t> dfn(n, NONE);
    if (dfsEnd)
        dfsEnd->assign(n, NONE);
    std::vector<uint32_t> parent;
    order.clear();
    std::vector<std::pair<uint32_t, uint32_t>> stack;
    dfn[root] = 0;
    order.push_back(root);
    parent.push_back(NONE);
    stack.emplace_back(root, off[root]);
    while (!stack.empty()) {
        const uint32_t x = stack.back().first;
        uint32_t& it = stack.back().second;
        if (it < off[x + 1]) {
            const uint32_t y = adj[it++];
            if (dfn[y] == NONE) {
                dfn[y] = static_cast<uint32_t>(order.size());
                order.push_back(y);
                parent.push_back(dfn[x]);
                stack.emplace_back(y, off[y]);
            }
        } else {
            if (dfsEnd)
                (*dfsEnd)[x] = static_cast<uint32_t>(order.size());
            stack.pop_back();
        }
    }

    const uint32_t N = static_cast<uint32_t>(order.size());
    std::vector<uint32_t> poff(static_cast<size_t>(N) + 1, 0);
    for (uint32_t w = 0; w < N; ++w) {
        const uint32_t x = order[w];
        for (uint32_t i = off[x]; i < off[x + 1]; ++i)
            poff[dfn[adj[i]] + 1]++;
    }
    for (size_t i = 1; i < poff.size(); ++i)
        poff[i] += poff[i - 1];
    std::vector<uint32_t> pred(poff.back());
    {
        std::vector<uint32_t> cur(poff.begin(), poff.end() - 1);
        for (uint32_t w = 0; w < N; ++w) {
            const uint32_t x = order[w];
            for (uint32_t i = off[x]; i < off[x + 1]; ++i)
                pred[cur[dfn[adj[i]]]++] = w;
        }
    }

    // Lengauer Tarjan uses the DFS visit order.
    std::vector<uint32_t> semi(N), label(N), anc(N, NONE), idomD(N, NONE);
    std::vector<uint32_t> bucketHead(N, NONE), bucketNext(N, NONE), path;
    std::iota(semi.begin(), semi.end(), 0u);
    std::iota(label.begin(), label.end(), 0u);

    auto eval = [&](uint32_t v) -> uint32_t {
        if (anc[v] == NONE)
            return v;
        path.clear();
        uint32_t x = v;
        while (anc[anc[x]] != NONE) {
            path.push_back(x);
            x = anc[x];
        }
        for (auto p = path.rbegin(); p != path.rend(); ++p) {
            const uint32_t y = *p;
            const uint32_t a = anc[y];
            if (semi[label[a]] < semi[label[y]])
                label[y] = label[a];
            anc[y] = anc[a];
        }
        return label[v];
    };

    for (uint32_t w = N - 1; w >= 1; --w) {
        for (uint32_t i = poff[w]; i < poff[w + 1]; ++i) {
            const uint32_t u = eval(pred[i]);
            if (semi[u] < semi[w])
                semi[w] = semi[u];
        }
        bucketNext[w] = bucketHead[semi[w]];
        bucketHead[semi[w]] = w;
        const uint32_t pw = parent[w];
        anc[w] = pw;
        for (uint32_t v = bucketHead[pw]; v != NONE; v = bucketNext[v]) {
            const uint32_t u = eval(v);
            idomD[v] = (semi[u] < semi[v]) ? u : pw;
        }
        bucketHead[pw] = NONE;
    }

    for (uint32_t w = 1; w < N; ++w)
        if (idomD[w] != semi[w])
            idomD[w] = idomD[idomD[w]];

    idom.assign(n, NONE);
    for (uint32_t w = 1; w < N; ++w)
        idom[order[w]] = order[idomD[w]];
    if (dfsPre)
        *dfsPre = std::move(dfn);
}

}
