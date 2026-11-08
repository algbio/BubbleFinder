#include "dominators/d_bubbles_cli.hpp"
#include "dominators/d_bubbles.hpp"

#include "util/context.hpp"
#include "util/timer.hpp"

#include <fstream>
#include <iostream>

namespace dominators {

int runDBubbleCommand()
{
    auto& C = ctx();
    Graph graph;
    graph.n = C.ubNumNodes;
    graph.edges.reserve(C.domLinks.size());
    for (const DomLink& edge : C.domLinks)
        graph.edges.push_back(BiEdge{edge.u, edge.v, edge.su, edge.sv});
    std::vector<DomLink>().swap(C.domLinks);
    C.ubNodeNames.resize(graph.n);

    DBubbleOptions options;
    options.bcTrees = true;
    options.ultrabubblesOnly = C.dominatorUltrabubbles;
    options.threads = C.threads;

    std::ofstream output(C.outputPath, std::ios::binary);
    if (!output) {
        std::cerr << "Error: failed to open output file '" << C.outputPath << "'\n";
        return 1;
    }
    try {
        TIME_BLOCK("Dominator bubbles");
        const int status = runDBubbles(graph, C.ubNodeNames, options, C.includeTrivial, output, std::cerr);
        output.flush();
        if (!output)
            throw std::runtime_error("d-bubbles: failed while flushing output");
        return status;
    } catch (const InvalidInput& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 2;
    }
}

}
