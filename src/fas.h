#pragma once

#include "util/spqr_rust_all.hpp"

#include "io/graph_io.hpp"
#include "util/profiling.hpp"

#include "assert.h"

// Given a graph G,
// find all edges that if removed, would make the graph acyclic.
// O(n + m) time complexity.
class FeedbackArcSet {
private:
    spqr_compat::Graph &G;
    enum EdgeType { TREE, BACK, FORWARD, CROSS };

    void run_fas(const spqr_compat::Graph &graph, std::vector<spqr_compat::edge> &result); 
    void find_feedback_arcs(std::vector<spqr_compat::edge> &result,
                            const spqr_compat::NodeArray<bool> &toRemove);

public:
    FeedbackArcSet(spqr_compat::Graph &graph) : G(graph) {}

    std::vector<spqr_compat::edge> run();

    bool run_or_acyclic(std::vector<spqr_compat::edge> &out);
};