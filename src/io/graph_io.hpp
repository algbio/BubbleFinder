#pragma once
#include <ogdf/basic/Graph.h>
#include "util/context.hpp"

namespace GraphIO {
    void readGraph();

    void readStandard();
    void readGFA();
    void drawGraph(const ogdf::Graph& G, const std::string& file);;

    void writeSuperbubbles();
}
