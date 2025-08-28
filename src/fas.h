#pragma once

#include <ogdf/basic/graph_generators.h>
#include <ogdf/layered/DfsAcyclicSubgraph.h>
#include <ogdf/fileformats/GraphIO.h>
#include <ogdf/basic/GraphAttributes.h>
#include <ogdf/basic/simple_graph_alg.h>
#include <ogdf/planarity/PlanarizationLayout.h>
#include <ogdf/decomposition/BCTree.h>  
#include <ogdf/decomposition/DynamicSPQRForest.h>
#include <ogdf/decomposition/DynamicSPQRTree.h>
#include <ogdf/augmentation/DfsMakeBiconnected.h>
#include <ogdf/decomposition/StaticSPQRTree.h>
#include <ogdf/decomposition/SPQRTree.h>
#include <ogdf/energybased/FMMMLayout.h>
#include <ogdf/tree/TreeLayout.h>
#include <ogdf/basic/List.h>

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
    void run_fas(ogdf::Graph G, std::vector<ogdf::edge> &result); 
    void find_feedback_arcs(std::vector<ogdf::edge> &result, const ogdf::NodeArray<bool> &toRemove);
public:
    FeedbackArcSet(ogdf::Graph &graph) : G(graph) {}

    std::vector<ogdf::edge> run();
};