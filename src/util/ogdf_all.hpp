#pragma once

#if defined(__clang__)
#  pragma clang system_header
#elif defined(__GNUC__)
#  pragma GCC system_header
#endif

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
#include <ogdf/tree/LCA.h>
#include <ogdf/basic/Thread.h>