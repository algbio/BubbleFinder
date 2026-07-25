// IndexedSpqrTree: load text or binary, keep data in RAM, save .spqri.
//
//   .spqr  --->  tree  <---->  .spqri
//                 |
//                 +-- SPQR and graph tables
//                 `-- haplotype paths and steps, if present

#pragma once

#include "util/graph_index.hpp"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <future>
#include <functional>
#include <ios>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sys/types.h>
#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifndef BF_HAVE_ZSTD
#define BF_HAVE_ZSTD 0
#endif

#if BF_HAVE_ZSTD
#include <zstd.h>
#endif

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
#include "util/spqr_index_format.inc"
#include "util/spqr_index_compression.inc"
#include "util/spqr_index_text.inc"
#include "util/spqr_index_container.inc"
#include "util/spqr_index_records.inc"
#include "util/spqr_index_columnar_write.inc"
#include "util/spqr_index_columnar_read.inc"
#include "util/spqr_index_array_helpers.inc"
};

}
