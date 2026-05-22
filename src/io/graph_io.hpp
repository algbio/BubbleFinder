#pragma once

#include "util/spqr_rust_all.hpp"
#include "util/context.hpp"
#include <memory>

namespace GraphIO {

    void readGraph();

    void readStandard(); // "graph" text format: n m, then m lines u v
    void readGFA();      
    
    void drawGraph(const spqr_compat::Graph& G, const std::string& file);

    void writeSuperbubbles();
}