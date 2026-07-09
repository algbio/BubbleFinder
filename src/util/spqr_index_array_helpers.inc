    // Compact arrays and indexes used by the current .spqri format.

    static std::uint32_t read_count(std::istream& in, const char* what)
    {
        const std::uint64_t n = detail::read_scalar<std::uint64_t>(in, what);
        return graph_index::require_spqr_count(n, what);
    }

    static void write_u32_vector(std::ostream& out, const std::vector<std::uint32_t>& v)
    {
        detail::write_scalar<std::uint64_t>(out, v.size());
        if (!v.empty()) {
            write_raw_bytes(out, v.data(), v.size() * sizeof(std::uint32_t));
        }
    }

    static void read_u32_vector(std::istream& in,
                                std::vector<std::uint32_t>& v,
                                const char* what)
    {
        const std::uint32_t n = read_count(in, what);
        v.resize(n);
        if (!v.empty()) {
            read_raw_bytes(in, v.data(), v.size() * sizeof(std::uint32_t), what);
        }
    }

    static void write_u8_vector(std::ostream& out, const std::vector<std::uint8_t>& v)
    {
        detail::write_scalar<std::uint64_t>(out, v.size());
        if (!v.empty()) {
            write_raw_bytes(out, v.data(), v.size() * sizeof(std::uint8_t));
        }
    }

    static void read_u8_vector(std::istream& in,
                               std::vector<std::uint8_t>& v,
                               const char* what)
    {
        const std::uint32_t n = read_count(in, what);
        v.resize(n);
        if (!v.empty()) {
            read_raw_bytes(in, v.data(), v.size() * sizeof(std::uint8_t), what);
        }
    }

    static void read_u16_vector(std::istream& in,
                                std::vector<std::uint16_t>& v,
                                const char* what)
    {
        const std::uint32_t n = read_count(in, what);
        v.resize(n);
        if (!v.empty()) {
            read_raw_bytes(in, v.data(), v.size() * sizeof(std::uint16_t), what);
        }
    }

    static void write_graph_node_degree_table(
        std::ostream& out,
        const std::vector<std::uint32_t>& degrees)
    {
        if (degrees.empty()) {
            detail::write_scalar<std::uint8_t>(out, 0u);
            return;
        }

        std::size_t overflow_count = 0;
        for (std::uint32_t degree : degrees) {
            if (degree > std::numeric_limits<std::uint8_t>::max()) {
                ++overflow_count;
            }
        }

        const std::size_t encoded_bytes =
            graph_index::checked_add_size(
                degrees.size(),
                graph_index::checked_mul_size(
                    overflow_count,
                    2u * sizeof(std::uint32_t),
                    "SPQR graph node degree overflow table"),
                "SPQR graph node degree table");
        const std::size_t raw_bytes =
            graph_index::checked_mul_size(
                degrees.size(),
                sizeof(std::uint32_t),
                "SPQR graph node raw degree table");
        if (encoded_bytes >= raw_bytes) {
            detail::write_scalar<std::uint8_t>(out, 2u);
            write_u32_vector(out, degrees);
            return;
        }

        detail::write_scalar<std::uint8_t>(out, 1u);
        std::vector<std::uint8_t> base;
        std::vector<std::uint32_t> overflow_nodes;
        std::vector<std::uint32_t> overflow_values;
        base.reserve(degrees.size());
        overflow_nodes.reserve(overflow_count);
        overflow_values.reserve(overflow_count);
        for (std::uint32_t i = 0; i < degrees.size(); ++i) {
            const std::uint32_t degree = degrees[i];
            if (degree > std::numeric_limits<std::uint8_t>::max()) {
                base.push_back(std::numeric_limits<std::uint8_t>::max());
                overflow_nodes.push_back(i);
                overflow_values.push_back(degree);
            } else {
                base.push_back(static_cast<std::uint8_t>(degree));
            }
        }
        write_u8_vector(out, base);
        write_u32_vector(out, overflow_nodes);
        write_u32_vector(out, overflow_values);
    }

    static void read_graph_node_degree_table(
        std::istream& in,
        std::uint32_t expected_count,
        std::vector<std::uint8_t>& base,
        std::vector<std::uint32_t>& overflow_nodes,
        std::vector<std::uint32_t>& overflow_values,
        std::vector<std::uint32_t>& raw,
        const std::string& path)
    {
        const std::uint8_t encoding =
            detail::read_scalar<std::uint8_t>(
                in, "compact graph node degree encoding");
        if (encoding == 0u) {
            return;
        }
        if (encoding == 1u) {
            read_u8_vector(in, base, "compact graph node degree base table");
            read_u32_vector(in, overflow_nodes,
                            "compact graph node degree overflow nodes");
            read_u32_vector(in, overflow_values,
                            "compact graph node degree overflow values");
            if (base.size() != expected_count ||
                overflow_nodes.size() != overflow_values.size()) {
                throw std::runtime_error(
                    "binary SPQR compact runtime has wrong graph node degree table size in " +
                    path);
            }
            std::uint32_t previous = invalid_id;
            for (std::uint32_t i = 0; i < overflow_nodes.size(); ++i) {
                const std::uint32_t node = overflow_nodes[i];
                if (node >= expected_count ||
                    (i != 0 && node <= previous) ||
                    overflow_values[i] <= std::numeric_limits<std::uint8_t>::max()) {
                    throw std::runtime_error(
                        "binary SPQR compact runtime has invalid graph node degree overflow table in " +
                        path);
                }
                previous = node;
            }
            return;
        }
        if (encoding == 2u) {
            read_u32_vector(in, raw, "compact graph node raw degree table");
            if (raw.size() != expected_count) {
                throw std::runtime_error(
                    "binary SPQR compact runtime has wrong raw graph node degree count in " +
                    path);
            }
            return;
        }
        throw std::runtime_error(
            "unsupported compact graph node degree encoding in " + path);
    }

    void read_graph_node_degree_tables(std::istream& in,
                                       std::uint32_t graph_nodes,
                                       const std::string& path)
    {
        read_graph_node_degree_table(
            in,
            graph_nodes,
            graph_node_in_degrees8,
            graph_node_in_degree_overflow_nodes,
            graph_node_in_degree_overflow_values,
            graph_node_in_degrees,
            path);
        read_graph_node_degree_table(
            in,
            graph_nodes,
            graph_node_out_degrees8,
            graph_node_out_degree_overflow_nodes,
            graph_node_out_degree_overflow_values,
            graph_node_out_degrees,
            path);
    }

    static void write_u64_vector(std::ostream& out, const std::vector<std::uint64_t>& v)
    {
        detail::write_scalar<std::uint64_t>(out, v.size());
        if (!v.empty()) {
            write_raw_bytes(out, v.data(), v.size() * sizeof(std::uint64_t));
        }
    }

    static void read_u64_vector(std::istream& in,
                                std::vector<std::uint64_t>& v,
                                const char* what)
    {
        const std::uint32_t n = read_count(in, what);
        v.resize(n);
        if (!v.empty()) {
            read_raw_bytes(in, v.data(), v.size() * sizeof(std::uint64_t), what);
        }
    }

    static bool is_identity_u32_vector(const std::vector<std::uint32_t>& v)
    {
        for (std::uint32_t i = 0; i < v.size(); ++i) {
            if (v[i] != i) return false;
        }
        return true;
    }

    static void write_identity_or_u32_vector(std::ostream& out,
                                             const std::vector<std::uint32_t>& v,
                                             std::uint32_t expected_identity_count,
                                             bool already_identity = false)
    {
        const bool identity = already_identity ||
            (v.size() == expected_identity_count && is_identity_u32_vector(v));
        detail::write_scalar<std::uint8_t>(out, identity ? 1u : 0u);
        if (identity) {
            detail::write_scalar<std::uint64_t>(out, expected_identity_count);
        } else {
            write_u32_vector(out, v);
        }
    }

    static bool read_identity_or_u32_vector(std::istream& in,
                                            std::vector<std::uint32_t>& v,
                                            const char* what,
                                            std::uint32_t expected_identity_count)
    {
        const std::uint8_t encoding =
            detail::read_scalar<std::uint8_t>(in, what);
        if (encoding == 1u) {
            const std::uint32_t count = read_count(in, what);
            if (count != expected_identity_count) {
                throw std::runtime_error(
                    std::string("binary SPQR compact identity index count mismatch: ") + what);
            }
            drop_storage(v);
            return true;
        }
        if (encoding != 0u) {
            throw std::runtime_error(
                std::string("unsupported binary SPQR compact identity index encoding: ") + what);
        }
        read_u32_vector(in, v, what);
        return false;
    }

    std::uint32_t tree_node_real_index_count() const
    {
        const std::uint32_t tree_count = tree_node_count();
        std::uint32_t count = 0;
        for (std::uint32_t i = 0; i < tree_count; ++i) {
            const auto tn = tree_node_at(i);
            count = std::max(count, tn.real_edge_end);
        }
        return count;
    }

    bool can_encode_tree_node_real_edge_ranges() const
    {
        const std::uint32_t count = tree_node_count();
        if (tree_node_real_edges_range) {
            return tree_node_real_edge_firsts.size() == count;
        }
        if (tree_node_real_edges_identity) {
            return true;
        }
        const std::uint32_t logical_count = tree_node_real_index_count();
        if (tree_node_real_edges.size() != logical_count) {
            return false;
        }
        for (std::uint32_t i = 0; i < count; ++i) {
            const auto tn = tree_node_at(i);
            if (tn.real_edge_begin == tn.real_edge_end) continue;
            if (tn.real_edge_end > tree_node_real_edges.size()) return false;
            for (std::uint32_t p = tn.real_edge_begin + 1u;
                 p < tn.real_edge_end; ++p) {
                if (tree_node_real_edges[p] != tree_node_real_edges[p - 1u] + 1u) {
                    return false;
                }
            }
        }
        return true;
    }

    std::uint32_t tree_node_real_edge_first_for_write(std::uint32_t tree_node_index) const
    {
        const auto tn = tree_node_at(tree_node_index);
        if (tn.real_edge_begin == tn.real_edge_end) {
            return invalid_id;
        }
        if (tree_node_real_edges_range) {
            if (tree_node_index >= tree_node_real_edge_firsts.size()) {
                throw std::runtime_error("SPQR compact tree-node real-edge range table is truncated");
            }
            return tree_node_real_edge_firsts[tree_node_index];
        }
        if (tree_node_real_edges_identity) {
            return tn.real_edge_begin;
        }
        if (tn.real_edge_begin >= tree_node_real_edges.size()) {
            throw std::runtime_error("SPQR compact tree-node real-edge range is out of bounds");
        }
        return tree_node_real_edges[tn.real_edge_begin];
    }

    void write_tree_node_real_edge_firsts(std::ostream& out) const
    {
        const std::uint32_t count = tree_node_count();
        detail::write_scalar<std::uint64_t>(out, count);
        constexpr std::size_t target_bytes = 4ull * 1024ull * 1024ull;
        const std::size_t values_per_chunk =
            std::max<std::size_t>(1, target_bytes / sizeof(std::uint32_t));
        std::vector<std::uint32_t> buffer;
        buffer.reserve(values_per_chunk);
        for (std::uint32_t base = 0; base < count;) {
            const std::size_t n = std::min<std::size_t>(
                values_per_chunk, count - base);
            buffer.clear();
            for (std::size_t i = 0; i < n; ++i) {
                buffer.push_back(tree_node_real_edge_first_for_write(
                    base + static_cast<std::uint32_t>(i)));
            }
            write_raw_bytes(out, buffer.data(), buffer.size() * sizeof(std::uint32_t));
            base += static_cast<std::uint32_t>(n);
        }
    }

    void write_tree_node_real_edge_index(std::ostream& out) const
    {
        const std::uint32_t logical_count = tree_node_real_index_count();
        if (can_encode_tree_node_real_edge_ranges()) {
            detail::write_scalar<std::uint8_t>(out, 2u);
            detail::write_scalar<std::uint64_t>(out, logical_count);
            write_tree_node_real_edge_firsts(out);
            return;
        }

        const bool identity = tree_node_real_edges_identity ||
            (tree_node_real_edges.size() == logical_count &&
             is_identity_u32_vector(tree_node_real_edges));
        detail::write_scalar<std::uint8_t>(out, identity ? 1u : 0u);
        if (identity) {
            detail::write_scalar<std::uint64_t>(out, logical_count);
        } else {
            write_u32_vector(out, tree_node_real_edges);
        }
    }

    void validate_tree_node_real_edge_range_count(std::uint32_t logical_count,
                                                  const std::string& path) const
    {
        const std::uint32_t count = tree_node_count();
        for (std::uint32_t i = 0; i < count; ++i) {
            const auto tn = tree_node_at(i);
            if (tn.real_edge_end < tn.real_edge_begin ||
                tn.real_edge_end > logical_count) {
                throw std::runtime_error(
                    "binary SPQR compact runtime has invalid tree-node real-edge range in " +
                    path);
            }
        }
    }

    void read_tree_node_real_edge_index(std::istream& in, const std::string& path)
    {
        const std::uint8_t encoding =
            detail::read_scalar<std::uint8_t>(in, "compact tree-node real-edge index encoding");
        if (encoding == 2u) {
            const std::uint32_t logical_count =
                read_count(in, "compact tree-node real-edge range count");
            validate_tree_node_real_edge_range_count(logical_count, path);
            read_u32_vector(in, tree_node_real_edge_firsts,
                            "compact tree-node real-edge first index");
            const std::uint32_t tree_total = tree_node_count();
            if (tree_node_real_edge_firsts.size() != tree_total) {
                throw std::runtime_error(
                    "binary SPQR compact runtime has wrong tree-node real-edge first count in " +
                    path);
            }
            const std::uint32_t real_edge_count =
                this->real_edge_count();
            for (std::uint32_t i = 0; i < tree_total; ++i) {
                const auto tn = tree_node_at(i);
                const std::uint32_t first = tree_node_real_edge_firsts[i];
                if (tn.real_edge_begin == tn.real_edge_end) continue;
                const std::uint32_t len = tn.real_edge_end - tn.real_edge_begin;
                if (first == invalid_id || len > real_edge_count ||
                    first > real_edge_count - len) {
                    throw std::runtime_error(
                        "binary SPQR compact runtime has invalid tree-node real-edge first in " +
                        path);
                }
            }
            drop_storage(tree_node_real_edges);
            tree_node_real_edges_identity = false;
            tree_node_real_edges_range = true;
            return;
        }
        if (encoding == 1u) {
            const std::uint32_t logical_count =
                read_count(in, "compact tree-node real-edge identity count");
            validate_tree_node_real_edge_range_count(logical_count, path);
            drop_storage(tree_node_real_edges);
            drop_storage(tree_node_real_edge_firsts);
            tree_node_real_edges_identity = true;
            tree_node_real_edges_range = false;
            return;
        }
        if (encoding != 0u) {
            throw std::runtime_error(
                "unsupported binary SPQR compact tree-node real-edge index encoding in " + path);
        }
        read_u32_vector(in, tree_node_real_edges, "compact tree-node real-edge index");
        drop_storage(tree_node_real_edge_firsts);
        tree_node_real_edges_identity = false;
        tree_node_real_edges_range = false;
    }

    void build_bubble_edge_type_index()
    {
        bubble_edge_type_count_hint = bubble_edge_types.size();
        real_edge_type_pairs.assign(real_edges.size(), 0);
        if (bubble_edge_types.empty()) return;

        std::unordered_map<std::uint32_t, std::uint32_t> real_edge_by_name;
        real_edge_by_name.reserve(real_edges.size());
        for (std::uint32_t i = 0; i < real_edges.size(); ++i) {
            real_edge_by_name[real_edges[i].name] = i;
        }

        for (const auto& rec : bubble_edge_types) {
            const std::uint32_t edge_index = find_or_invalid(real_edge_by_name, rec.edge);
            if (edge_index == invalid_id || edge_index >= real_edge_type_pairs.size()) continue;
            real_edge_type_pairs[edge_index] =
                static_cast<std::uint8_t>((rec.src_type << 4) | (rec.dst_type & 0x0fu));
        }
    }
