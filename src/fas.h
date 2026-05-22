#pragma once

#include "util/ogdf_all.hpp"

#include "io/graph_io.hpp"
#include "util/profiling.hpp"

#include "assert.h"

// Given a graph G,
// find all edges that if removed, would make the graph acyclic.
// O(n + m) time complexity.
class FeedbackArcSet {
private:
    ogdf::Graph &G;
    enum EdgeType { TREE, BACK, FORWARD, CROSS };

    void run_fas(const ogdf::Graph &graph, std::vector<ogdf::edge> &result); 
    void find_feedback_arcs(std::vector<ogdf::edge> &result,
                            const ogdf::NodeArray<bool> &toRemove);

public:
    FeedbackArcSet(ogdf::Graph &graph) : G(graph) {}

    std::vector<ogdf::edge> run();
};