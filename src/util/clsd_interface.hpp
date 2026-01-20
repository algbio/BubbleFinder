#pragma once

#include <vector>
#include <utility>

#include "../../external/clsd/graph.h"
#include "../../external/clsd/dfs.h"
#include "../../external/clsd/detect.h"
#include "../../external/clsd/cycle.h"
#include "../../external/clsd/config.h"

std::vector<std::pair<int,int>>
compute_superbubbles_from_edges(
    const std::vector<std::pair<int,int>>& edges,
    std::vector<supertree*>* out_trees = nullptr
);
