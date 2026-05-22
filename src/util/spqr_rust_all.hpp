#pragma once
#include "../../external/spqr-rust/include/spqr_compat.hpp"
#include <cassert>

namespace spqr_compat {
    using spqr::node;
    using spqr::edge;
    using spqr::adjEntry;
    using spqr::Graph;
    using spqr::BCTree;
    using spqr::StaticSPQRTree;
    using spqr::Skeleton;
    using spqr::TreeGraph;
    using spqr::SPQRTree;
    using spqr::tree_node;

    using spqr::INVALID_NODE;
    using spqr::INVALID_EDGE;

    using spqr::connectedComponents;
    using spqr::isAcyclic;
    using spqr::strongComponents;

    template<typename T> using NodeArray = spqr::NodeArray<T>;
    template<typename T> using EdgeArray = spqr::EdgeArray<T>;
}

using namespace spqr_compat;

#ifndef SPQR_RUST_ASSERT
#define SPQR_RUST_ASSERT(x) assert(x)
#endif
