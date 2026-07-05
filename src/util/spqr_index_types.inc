// Records and load options used by the index.

inline constexpr std::uint32_t invalid_id = graph_index::spqr_invalid_index;
inline constexpr std::uint64_t invalid_u64 =
    std::numeric_limits<std::uint64_t>::max();
inline constexpr std::string_view graph_profile_raw = "raw";
inline constexpr std::string_view graph_profile_endpoint_parallel_subdivided =
    "endpoint-typed.parallel-subdivided";
inline constexpr std::string_view graph_profile_oriented_bidirected =
    "oriented-double.bidirected";
inline constexpr std::string_view graph_profile_oriented_directed =
    "oriented-double.directed";
inline constexpr std::uint32_t macro_cache_block_fully_sp_reducible_flag = 1u;

inline std::string canonical_graph_profile(std::string_view profile)
{
    return profile.empty() ? std::string(graph_profile_raw)
                           : std::string(profile);
}

inline bool graph_profile_is_raw(std::string_view profile)
{
    return profile.empty() || profile == graph_profile_raw;
}

inline bool graph_profile_is_parallel_subdivided(std::string_view profile)
{
    return profile == graph_profile_endpoint_parallel_subdivided;
}

inline bool graph_profile_is_oriented_bidirected(std::string_view profile)
{
    return profile == graph_profile_oriented_bidirected;
}

inline bool graph_profile_is_oriented_directed(std::string_view profile)
{
    return profile == graph_profile_oriented_directed;
}

inline bool graph_profile_is_oriented_double(std::string_view profile)
{
    return graph_profile_is_oriented_bidirected(profile) ||
           graph_profile_is_oriented_directed(profile);
}

inline bool graph_profile_is_raw_or_parallel_subdivided(std::string_view profile)
{
    return graph_profile_is_raw(profile) ||
           graph_profile_is_parallel_subdivided(profile);
}

inline bool graph_profile_matches_oriented_double(std::string_view profile,
                                                  bool directed)
{
    return directed ? graph_profile_is_oriented_directed(profile)
                    : graph_profile_is_oriented_bidirected(profile);
}

inline bool graph_profile_is_supported_spqr_tree_profile(std::string_view profile)
{
    return graph_profile_is_raw(profile) ||
           graph_profile_is_parallel_subdivided(profile) ||
           graph_profile_is_oriented_double(profile);
}

enum class ContainerKind : std::uint8_t {
    Unknown = 0,
    Component = 1,
    Block = 2,
    SpqrNode = 3
};

// Main links between the records:
//
//   block -> tree node (S/P/R)
//                    |--> real edges
//                    `--> virtual edge -> another tree node

struct ComponentRecord {
    std::uint32_t name = invalid_id;
    std::uint32_t node_begin = 0;
    std::uint32_t node_end = 0;
    std::uint32_t real_edge_begin = 0;
    std::uint32_t real_edge_end = 0;
};

struct BlockRecord {
    std::uint32_t name = invalid_id;
    std::uint32_t node_begin = 0;
    std::uint32_t node_end = 0;
    std::uint32_t spqr_begin = 0;
    std::uint32_t spqr_end = 0;
    std::uint32_t real_edge_begin = 0;
    std::uint32_t real_edge_end = 0;
    std::uint32_t tree_real_edge_count = 0;
    std::uint32_t tree_virtual_edge_count = 0;
};

struct CutRecord {
    std::uint32_t name = invalid_id;
    std::uint32_t block_begin = 0;
    std::uint32_t block_end = 0;
};

struct TreeNodeRecord {
    std::uint32_t name = invalid_id;
    std::uint32_t block = invalid_id;
    std::uint32_t block_index = invalid_id;
    std::uint8_t type = 0;
    std::uint32_t node_begin = 0;
    std::uint32_t node_end = 0;
    std::uint32_t real_edge_begin = 0;
    std::uint32_t real_edge_end = 0;
    std::uint32_t virtual_edge_begin = 0;
    std::uint32_t virtual_edge_end = 0;
};

struct VirtualEdgeRecord {
    std::uint32_t name = invalid_id;
    std::uint32_t node1 = invalid_id;
    std::uint32_t node2 = invalid_id;
    std::uint32_t node1_index = invalid_id;
    std::uint32_t node2_index = invalid_id;
    std::uint32_t pole1 = invalid_id;
    std::uint32_t pole2 = invalid_id;
};

struct RealEdgeRecord {
    std::uint32_t name = invalid_id;
    std::uint32_t container = invalid_id;
    std::uint32_t container_index = invalid_id;
    ContainerKind container_kind = ContainerKind::Unknown;
    std::uint32_t src = invalid_id;
    std::uint32_t dst = invalid_id;
};

struct GraphEdgeRecord {
    std::uint32_t src = invalid_id;
    std::uint32_t dst = invalid_id;
};

struct GraphComponentRecord {
    std::uint32_t node_begin = 0;
    std::uint32_t node_end = 0;
    std::uint32_t edge_begin = 0;
    std::uint32_t edge_end = 0;
};

// Haplotype paths are separate from the SPQR tree. Each one uses a step range:
//
//   path record -> steps[step_begin ... step_end)
//   one step    -> graph node + direction

struct HaplotypePathRecord {
    std::uint32_t name = invalid_id;
    std::uint32_t sample = invalid_id;
    std::uint32_t locus = invalid_id;
    std::uint8_t sense = 0;
    std::uint8_t flags = 0;
    std::uint16_t reserved = 0;
    std::uint64_t haplotype = invalid_u64;
    std::uint64_t phase_block = invalid_u64;
    std::uint64_t step_begin = 0;
    std::uint64_t step_end = 0;
};

struct HaplotypeStepRecord {
    std::uint32_t node = invalid_id;
    std::uint8_t is_reverse = 0;
    std::uint8_t reserved[3] = {};
};

struct BubbleEdgeTypeRecord {
    std::uint32_t edge = invalid_id;
    std::uint8_t src_type = 0;
    std::uint8_t dst_type = 0;
};

struct SlimBlockRecord {
    std::uint32_t node_begin = 0;
    std::uint32_t node_end = 0;
    std::uint32_t spqr_begin = 0;
    std::uint32_t spqr_end = 0;
    std::uint32_t real_edge_begin = 0;
    std::uint32_t real_edge_end = 0;
    std::uint32_t tree_real_edge_count = 0;
    std::uint32_t tree_virtual_edge_count = 0;
};

struct SlimVirtualEdgeRecord {
    std::uint32_t node1_index = invalid_id;
    std::uint32_t node2_index = invalid_id;
    std::uint32_t pole1 = invalid_id;
    std::uint32_t pole2 = invalid_id;
};

struct MacroCacheNodeRecord {
    std::uint8_t kind = 0;
    std::uint8_t pad[3] = {};
    std::uint32_t left = invalid_id;
    std::uint32_t right = invalid_id;
    std::uint64_t children_offset = 0;
    std::uint64_t children_count = 0;
};

struct MacroCacheCoreEdgeRecord {
    std::uint32_t u = invalid_id;
    std::uint32_t v = invalid_id;
    std::uint64_t child = 0;
};

struct MacroCacheSkeletonEdgeRecord {
    std::uint32_t src = invalid_id;
    std::uint32_t dst = invalid_id;
    std::uint32_t real_edge = invalid_id;
    std::uint32_t virtual_id = invalid_id;
    std::uint32_t twin_tree_node = invalid_id;
    std::uint32_t twin_edge_idx = invalid_id;
};

struct MacroCacheBlockRecord {
    std::uint32_t block = invalid_id;
    std::uint32_t flags = 0;
    std::uint8_t reserved[48] = {};
    std::uint64_t macro_begin = 0;
    std::uint64_t macro_end = 0;
    std::uint64_t child_begin = 0;
    std::uint64_t child_end = 0;
    std::uint64_t core_edge_begin = 0;
    std::uint64_t core_edge_end = 0;
    std::uint64_t core_node_begin = 0;
    std::uint64_t core_node_end = 0;
    std::uint64_t input_endpoint_begin = 0;
    std::uint64_t input_endpoint_end = 0;
    std::uint64_t core_node_inv_begin = 0;
    std::uint64_t core_node_inv_end = 0;
    std::uint32_t core_root = invalid_id;
    std::uint32_t core_tree_node_count = 0;
    std::uint64_t core_tree_node_type_begin = 0;
    std::uint64_t core_tree_node_type_end = 0;
    std::uint64_t core_tree_parent_begin = 0;
    std::uint64_t core_tree_parent_end = 0;
    std::uint64_t core_tree_child_offset_begin = 0;
    std::uint64_t core_tree_child_offset_end = 0;
    std::uint64_t core_tree_child_begin = 0;
    std::uint64_t core_tree_child_end = 0;
    std::uint64_t core_tree_skeleton_offset_begin = 0;
    std::uint64_t core_tree_skeleton_offset_end = 0;
    std::uint64_t core_tree_skeleton_edge_begin = 0;
    std::uint64_t core_tree_skeleton_edge_end = 0;
    std::uint64_t core_tree_mapping_offset_begin = 0;
    std::uint64_t core_tree_mapping_offset_end = 0;
    std::uint64_t core_tree_mapping_begin = 0;
    std::uint64_t core_tree_mapping_end = 0;
    std::uint64_t core_tree_skeleton_num_nodes_begin = 0;
    std::uint64_t core_tree_skeleton_num_nodes_end = 0;
};

enum class EagerLookupViewFilter : std::uint8_t {
    Any = 0,
    OrientedBidirected = 1,
    OrientedDirected = 2,
    RawOrParallelSubdivided = 3
};

struct LoadOptions {
    bool skip_graph_edge_tables = false;
    bool skip_graph_edge_tables_for_oriented_views = false;
    bool skip_graph_node_degrees = false;
    bool skip_real_edge_types = false;
    bool graph_only = false;
    bool load_graph_components = false;
    bool load_haplotypes = false;
    bool skip_macro_tree_cache = false;
    bool eager_graph_node_name_lookup = false;
    bool eager_block_hash_lookup = false;
    bool prefer_precomputed_graph_node_numeric_lookup = false;
    EagerLookupViewFilter eager_lookup_view_filter =
        EagerLookupViewFilter::Any;
};

namespace detail {

inline bool ends_with(std::string_view s, std::string_view suffix)
{
    return s.size() >= suffix.size() &&
           s.substr(s.size() - suffix.size(), suffix.size()) == suffix;
}

inline std::string shell_escape(const std::string& s)
{
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('\'');
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('\'');
    return out;
}

inline std::string decompressor_command(const std::string& path)
{
    if (ends_with(path, ".gz") || ends_with(path, ".bgz")) {
        return "gzip -dc " + shell_escape(path);
    }
    if (ends_with(path, ".bz2")) {
        return "bzip2 -dc " + shell_escape(path);
    }
    if (ends_with(path, ".xz")) {
        return "xz -dc " + shell_escape(path);
    }
    return {};
}

inline std::vector<std::string_view> split_ws(std::string_view line)
{
    std::vector<std::string_view> out;
    std::size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() &&
               (line[i] == ' ' || line[i] == '\t' || line[i] == '\r' || line[i] == '\n')) {
            ++i;
        }
        const std::size_t begin = i;
        while (i < line.size() &&
               line[i] != ' ' && line[i] != '\t' && line[i] != '\r' && line[i] != '\n') {
            ++i;
        }
        if (begin != i) {
            out.emplace_back(line.data() + begin, i - begin);
        }
    }
    return out;
}

template <typename T>
inline void write_scalar(std::ostream& out, T value)
{
    static_assert(std::is_integral<T>::value || std::is_enum<T>::value,
                  "binary SPQR index only writes scalar integral fields");
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) throw std::runtime_error("failed while writing binary SPQR index");
}

template <typename T>
inline T read_scalar(std::istream& in, const char* what)
{
    static_assert(std::is_integral<T>::value || std::is_enum<T>::value,
                  "binary SPQR index only reads scalar integral fields");
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error(std::string("truncated binary SPQR index: ") + what);
    return value;
}

inline void write_string(std::ostream& out, const std::string& s)
{
    write_scalar<std::uint64_t>(out, static_cast<std::uint64_t>(s.size()));
    out.write(s.data(), static_cast<std::streamsize>(s.size()));
    if (!out) throw std::runtime_error("failed while writing binary SPQR index string");
}

inline std::string read_string(std::istream& in, const char* what)
{
    const std::uint64_t n64 = read_scalar<std::uint64_t>(in, what);
    const std::size_t n = static_cast<std::size_t>(n64);
    if (n64 != static_cast<std::uint64_t>(n)) {
        throw std::runtime_error(std::string("binary SPQR index string too large: ") + what);
    }
    std::string s(n, '\0');
    if (n != 0) {
        in.read(&s[0], static_cast<std::streamsize>(n));
        if (!in) throw std::runtime_error(std::string("truncated binary SPQR index string: ") + what);
    }
    return s;
}

inline std::uint8_t parse_bubble_edge_type(std::string_view value)
{
    if (value == "+" || value == "PLUS" || value == "plus") return 1;
    if (value == "-" || value == "MINUS" || value == "minus") return 2;
    if (value == "." || value == "NONE" || value == "none" ||
        value == "UNKNOWN" || value == "unknown") {
        return 0;
    }
    throw std::runtime_error("invalid BubbleFinder edge endpoint type");
}

}
