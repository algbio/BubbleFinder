#include "clsd_interface.hpp"

#include <array>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>

static inline int infer_n_from_edges(const std::vector<std::pair<int,int>>& edges) {
    int mx = -1;
    for (auto [u,v] : edges) {
        if (u > mx) mx = u;
        if (v > mx) mx = v;
    }
    return mx + 1;
}

static ClsdTree clone_tree(const supertree* t) {
    ClsdTree r;
    r.entrance = std::stoi(t->entrance->getID());
    r.exit     = std::stoi(t->exit->getID());
    r.children.reserve(t->childreen.size());
    for (auto* c : t->childreen) {
        r.children.push_back(clone_tree(c));
    }
    return r;
}

std::vector<std::pair<int,int>>
compute_weak_superbubbles_from_edges(
    const std::vector<std::pair<int,int>>& edges,
    std::vector<ClsdTree>* out_trees
) {
    if (out_trees) out_trees->clear();
    if (edges.empty()) return {};
    return compute_weak_superbubbles_from_edges(infer_n_from_edges(edges), edges, out_trees);
}

std::vector<std::pair<int,int>>
compute_weak_superbubbles_from_edges(
    int n,
    const std::vector<std::pair<int,int>>& edges,
    std::vector<ClsdTree>* out_trees
) {
    if (out_trees) out_trees->clear();
    if (n <= 0 || edges.empty()) return {};

    using U = unsigned long;

    for (auto [u, v] : edges) {
        if (u < 0 || v < 0 || u >= n || v >= n) {
            throw std::runtime_error("CLSD wrapper: edge endpoint out of range");
        }
    }

    std::vector<std::string> ids((size_t)n);
    std::vector<std::array<U,3>> ele((size_t)n);

    for (int i = 0; i < n; ++i) {
        ids[(size_t)i] = std::to_string(i);
        ele[(size_t)i] = { (U)i, 0UL, 0UL };
    }

    for (auto [u, v] : edges) {
        ele[(size_t)u][1]++; // outdegree
        ele[(size_t)v][2]++; // indegree
    }

    std::vector<Vertex> vertices((size_t)n);

    std::vector<int> sources;
    sources.reserve((size_t)n);

    for (int i = 0; i < n; ++i) {
        std::pair<std::string, U*> arg(ids[(size_t)i], ele[(size_t)i].data());
        vertices[(size_t)i].init(arg);

        if (ele[(size_t)i][2] == 0UL) {
            sources.push_back(i);
        }
    }

    for (auto [u, v] : edges) {
        Vertex* from = &vertices[(size_t)u];
        Vertex* to   = &vertices[(size_t)v];
        from->addSuc(to);
        to->addPre(from);
    }

    for (Vertex& v : vertices) v.resetCounter();

    Config conf;
    conf.setVertices((U)vertices.size());
    conf.setEdges((U)edges.size());
    conf.setMultiedges(0);
    conf.trees = (out_trees != nullptr);
    conf.includeWeak = true;

    conf.startClock();

    for (int si : sources) {
        Vertex& v = vertices[(size_t)si];

        Vertex* start = nullptr;
        create_postorder(&v, &start);
        conf.addOrder(start, &v, false);
        detect(start, &v, conf);
        finish(start, &v, false);
    }

    for (Vertex& v : vertices) {
        if (v.stat == Vertex::CLEAN) {
            cycle_search(&v, conf);
        }
    }

    conf.endClock();

    std::vector<std::pair<int,int>> result;
    const auto& bubs = conf.getSuperbubbles();
    result.reserve(bubs.size());

    for (const auto& sb : bubs) {
        int ent_id = std::stoi(sb.entrance->getID());
        int ex_id  = std::stoi(sb.exit->getID());
        result.emplace_back(ent_id, ex_id);
    }

    if (out_trees != nullptr) {
        const auto& roots = conf.getTrees();
        out_trees->reserve(roots.size());
        for (auto* rt : roots) {
            out_trees->push_back(clone_tree(rt));
        }
    }

    return result;
}