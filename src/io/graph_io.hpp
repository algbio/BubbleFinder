#pragma once
#include <ogdf/basic/Graph.h>
#include "util/context.hpp"        // ctx() accessor

namespace GraphIO
{
    // top-level entry
    void readGraph();              // uses ctx() inside

    // helpers you moved out of main.cpp
    void readStandard();
    void readGFA();
    void drawGraph(ogdf::Graph& G, const std::string& file);

    void writeSuperbubbles();
} // namespace GraphIO
