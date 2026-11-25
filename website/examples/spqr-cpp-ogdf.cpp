#include "spqr_compat.hpp"
#include <cassert>
#include <cstdint>
#include <iostream>

namespace api = spqr::ogdf_compat;

int main() {
    api::Graph graph;
    api::node vertices[8];
    for (auto& vertex : vertices) vertex = graph.newNode();
    const uint32_t pairs[][2] = {
        {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3},
        {0, 4}, {0, 5}, {1, 4}, {1, 5}, {4, 5},
        {0, 6}, {6, 7}, {7, 1}
    };
    for (const auto& pair : pairs)
        graph.newEdge(vertices[pair[0]], vertices[pair[1]]);

    // Create arrays after the graph is complete.
    api::NodeArray<unsigned> degree(graph, 0);
    api::EdgeArray<unsigned> seen(graph, 0);
    for (auto vertex : graph.nodes) {
        graph.forEachAdj(vertex, [&](api::node neighbour, api::edge edge) {
            assert(neighbour.index() < 8);
            ++degree[vertex];
            ++seen[edge.index()];
        });
        assert(degree[vertex] == graph.degree(vertex));
    }
    for (auto edge : graph.edges) {
        assert(edge->source().index() < 8);
        assert(edge->target().index() < 8);
        assert(seen[edge.index()] == 2);
    }

    api::StaticSPQRTree tree(graph);
    unsigned s = 0, p = 0, r = 0, original_edges = 0;
    for (auto tree_node : tree.tree().nodes) {
        using Type = api::StaticSPQRTree::NodeType;
        const auto type = tree.typeOf(tree_node);
        s += type == Type::SNode;
        p += type == Type::PNode;
        r += type == Type::RNode;
        // skeleton() returns a value. No temporary skeleton graph is needed.
        auto skeleton = tree.skeleton(tree_node);
        skeleton.forEachEdge([&](api::edge edge, api::node u, api::node v) {
            assert(skeleton.original(u).index() < 8);
            assert(skeleton.original(v).index() < 8);
            if (!skeleton.isVirtual(edge)) {
                assert(skeleton.realEdge(edge).index() < 14);
                ++original_edges;
            }
        });
    }
    assert(s == 1 && p == 1 && r == 2 && original_edges == 14);
    assert(tree.flatView().numNodes == 4);
    std::cout << "S: " << s << ", P: " << p << ", R: " << r
              << ", original edges: " << original_edges << "\n";
}
