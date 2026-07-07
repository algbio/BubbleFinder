// Read a .spqr file and keep its flat tables in RAM.

#pragma once

#include "util/graph_index.hpp"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <istream>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace spqr_index {

#include "util/spqr_index_types.inc"

class IndexedSpqrTree {
public:
    IndexedSpqrTree() = default;
    IndexedSpqrTree(const IndexedSpqrTree&) = delete;
    IndexedSpqrTree& operator=(const IndexedSpqrTree&) = delete;
    IndexedSpqrTree(IndexedSpqrTree&&) = default;
    IndexedSpqrTree& operator=(IndexedSpqrTree&&) = default;

#include "util/spqr_index_storage.inc"

#include "util/spqr_index_runtime.inc"

#include "util/spqr_index_io.inc"

private:
#include "util/spqr_index_access.inc"
#include "util/spqr_index_text.inc"
#include "util/spqr_index_records.inc"
#include "util/spqr_index_array_helpers.inc"
};

}
