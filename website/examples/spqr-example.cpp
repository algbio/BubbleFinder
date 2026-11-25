#include "spqr_rust_wrapper.hpp"
#include <iostream>

int main() {
    spqr_rust::RustGraph graph;
    graph.addNodes(4);
    for (const auto& edge : {std::pair{0u, 1u}, {0u, 2u}, {0u, 3u},
                            {1u, 2u}, {1u, 3u}, {2u, 3u}}) {
        graph.addEdge(edge.first, edge.second);
    }

    spqr_rust::RustSPQRResult result(graph);
    std::cout << result.treeLen() << "\n";
}
