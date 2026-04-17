#pragma once
// Compatibility layer: replaces OGDF with spqr-rust, to rename

#include "../../external/spqr-rust/include/ogdf_compat.hpp"
#include <cassert>

namespace ogdf {
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

    template<typename T> using NodeArray = spqr::ogdf_compat::NodeArray<T>;
    template<typename T> using EdgeArray = spqr::ogdf_compat::EdgeArray<T>;
}

using namespace ogdf;

#ifndef OGDF_ASSERT
#define OGDF_ASSERT(x) assert(x)
#endif
