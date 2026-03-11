#pragma once

#include <vector>
#include <utility>

#include "../../external/clsd/graph.h"
#include "../../external/clsd/dfs.h"
#include "../../external/clsd/detect.h"
#include "../../external/clsd/cycle.h"
#include "../../external/clsd/config.h"

struct ClsdTree {
    int entrance = -1;
    int exit     = -1;
    std::vector<ClsdTree> children;
};

std::vector<std::pair<int,int>>
compute_weak_superbubbles_from_edges(
    int n,
    const std::vector<std::pair<int,int>>& edges,
    std::vector<ClsdTree>* out_trees = nullptr
);

std::vector<std::pair<int,int>>
compute_weak_superbubbles_from_edges(
    const std::vector<std::pair<int,int>>& edges,
    std::vector<ClsdTree>* out_trees = nullptr
);