#pragma once

#include "util/ogdf_all.hpp"
#include "util/context.hpp"
#include <memory>

namespace GraphIO {

    // Top-level entry point: reads graph according to ctx().inputFormat
    // and ctx().bubbleType.
    void readGraph();

    // Low-level readers (used internally by readGraph()).
    void readStandard(); // "graph" text format: n m, then m lines u v
    void readGFA();      // GFA-based formats (bidirected and snarls)

    void drawGraph(const ogdf::Graph& G, const std::string& file);

    void writeSuperbubbles();
}