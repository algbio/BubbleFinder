    // Read and write the flat record arrays stored in .spqri sections.

    static void write_raw_bytes(std::ostream& out, const void* data, std::size_t bytes)
    {
        const char* p = static_cast<const char*>(data);
        while (bytes != 0) {
            const std::size_t chunk = std::min<std::size_t>(
                bytes, static_cast<std::size_t>(1ull << 30));
            out.write(p, static_cast<std::streamsize>(chunk));
            if (!out) throw std::runtime_error("failed while writing binary SPQR index data");
            p += chunk;
            bytes -= chunk;
        }
    }

    static void read_raw_bytes(std::istream& in, void* data, std::size_t bytes, const char* what)
    {
        char* p = static_cast<char*>(data);
        while (bytes != 0) {
            const std::size_t chunk = std::min<std::size_t>(
                bytes, static_cast<std::size_t>(1ull << 30));
            in.read(p, static_cast<std::streamsize>(chunk));
            if (!in) throw std::runtime_error(std::string("truncated binary SPQR index data: ") + what);
            p += chunk;
            bytes -= chunk;
        }
    }

    template <typename T>
    static void write_trivial_vector64(std::ostream& out, const std::vector<T>& values)
    {
        static_assert(std::is_trivially_copyable<T>::value,
                      "binary SPQR vectors require trivially-copyable records");
        detail::write_scalar<std::uint64_t>(out, values.size());
        if (!values.empty()) {
            write_raw_bytes(out, values.data(), values.size() * sizeof(T));
        }
    }

    template <typename T>
    static void read_trivial_vector64(std::istream& in,
                                      std::vector<T>& values,
                                      const char* what)
    {
        static_assert(std::is_trivially_copyable<T>::value,
                      "binary SPQR vectors require trivially-copyable records");
        const std::uint64_t count64 =
            detail::read_scalar<std::uint64_t>(in, what);
        if (count64 > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max() / sizeof(T))) {
            throw std::runtime_error(std::string("binary SPQR vector too large: ") + what);
        }
        values.resize(static_cast<std::size_t>(count64));
        if (!values.empty()) {
            read_raw_bytes(in, values.data(), values.size() * sizeof(T), what);
        }
    }

    template <typename Record, typename FillFn>
    static void write_compact_records(std::ostream& out,
                                      std::uint64_t count,
                                      FillFn fill)
    {
        detail::write_scalar<std::uint64_t>(out, count);
        constexpr std::size_t target_bytes = 4ull * 1024ull * 1024ull;
        const std::size_t records_per_chunk =
            std::max<std::size_t>(1, target_bytes / sizeof(Record));
        std::vector<Record> buffer;
        buffer.reserve(records_per_chunk);
        for (std::uint64_t base = 0; base < count;) {
            const std::size_t n = static_cast<std::size_t>(
                std::min<std::uint64_t>(records_per_chunk, count - base));
            buffer.clear();
            for (std::size_t i = 0; i < n; ++i) {
                buffer.push_back(fill(static_cast<std::uint32_t>(base + i)));
            }
            write_raw_bytes(out, buffer.data(), buffer.size() * sizeof(Record));
            base += n;
        }
    }

    struct NodeRemapTable {
        bool identity = false;
        std::uint32_t identity_size = 0;
        std::vector<std::uint32_t> table;
    };

    NodeRemapTable build_node_remap_table() const
    {
        NodeRemapTable remap;
        if (has_compact_numeric_graph_node_names()) {
            const std::uint32_t n = graph_node_count();
            if (node_names.empty()) {
                remap.identity = true;
                remap.identity_size = n;
                return remap;
            } else {
                remap.table.assign(n, invalid_id);
                for (std::uint32_t compact = 0; compact < node_names.size(); ++compact) {
                    const std::uint32_t old = node_names[compact];
                    if (old >= n) {
                        throw std::runtime_error(
                            "SPQR compact index graph node ordinal out of range");
                    }
                    if (remap.table[old] != invalid_id) {
                        throw std::runtime_error(
                            "SPQR compact index graph node table contains duplicate ordinals");
                    }
                    remap.table[old] = compact;
                }
            }
            return remap;
        }

        remap.table.assign(strings.size(), invalid_id);
        for (std::uint32_t compact = 0; compact < node_names.size(); ++compact) {
            const std::uint32_t old = node_names[compact];
            if (old >= remap.table.size()) {
                throw std::runtime_error("SPQR index graph node name id out of range");
            }
            if (remap.table[old] != invalid_id) {
                throw std::runtime_error("SPQR index graph node table contains duplicate names");
            }
            remap.table[old] = compact;
        }
        return remap;
    }

    static std::uint32_t remap_graph_node_id(const NodeRemapTable& node_remap,
                                             std::uint32_t old,
                                             const char* what)
    {
        if (node_remap.identity) {
            if (old == invalid_id || old >= node_remap.identity_size) {
                throw std::runtime_error(
                    std::string("SPQR compact index cannot remap graph node id: ") + what);
            }
            return old;
        }
        if (old == invalid_id || old >= node_remap.table.size() ||
            node_remap.table[old] == invalid_id) {
            throw std::runtime_error(
                std::string("SPQR compact index cannot remap graph node id: ") + what);
        }
        return node_remap.table[old];
    }

    std::pair<std::vector<std::uint32_t>, std::vector<std::uint32_t>>
    build_compact_graph_node_degree_vectors(const NodeRemapTable& node_remap) const
    {
        const std::uint32_t graph_nodes = graph_node_count();
        if (graph_node_in_degrees.size() == graph_nodes &&
            graph_node_out_degrees.size() == graph_nodes) {
            return {graph_node_in_degrees, graph_node_out_degrees};
        }
        if (!graph_node_in_degrees.empty() || !graph_node_out_degrees.empty()) {
            throw std::runtime_error(
                "SPQR compact index graph node degree table has wrong size");
        }

        std::vector<std::uint32_t> in(graph_nodes, 0);
        std::vector<std::uint32_t> out(graph_nodes, 0);
        auto increment = [](std::uint32_t& value, const char* what) {
            if (value == std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error(
                    std::string("SPQR graph node degree overflow while writing ") +
                    what);
            }
            ++value;
        };
        auto add_edge = [&](std::uint32_t src_old,
                            std::uint32_t dst_old,
                            const char* what) {
            const std::uint32_t src =
                remap_graph_node_id(node_remap, src_old, what);
            const std::uint32_t dst =
                remap_graph_node_id(node_remap, dst_old, what);
            increment(out[src], "out-degree");
            increment(in[dst], "in-degree");
        };

        if (!graph_edges.empty()) {
            for (const auto& edge : graph_edges) {
                add_edge(edge.src, edge.dst, "graph-edge endpoint");
            }
            return {std::move(in), std::move(out)};
        }

        const std::uint32_t real_edges_n = real_edge_count();
        if (real_edges_n == 0u && graph_edge_count_hint == 0u) {
            return {std::move(in), std::move(out)};
        }
        if (graph_profile_is_oriented_double(graph_view) &&
            graph_edge_count_hint != 0u &&
            static_cast<std::uint64_t>(real_edges_n) == graph_edge_count_hint) {
            for (std::uint32_t i = 0; i < real_edges_n; ++i) {
                const auto edge = real_edge_endpoint_at(i);
                add_edge(edge.src, edge.dst, "real-edge endpoint");
            }
            return {std::move(in), std::move(out)};
        }

        return {};
    }

    static void write_mapped_u32_vector(std::ostream& out,
                                        const std::vector<std::uint32_t>& src,
                                        const NodeRemapTable& node_remap,
                                        const char* what)
    {
        detail::write_scalar<std::uint64_t>(out, src.size());
        if (node_remap.identity) {
            if (!src.empty()) {
                write_raw_bytes(out,
                                src.data(),
                                src.size() * sizeof(std::uint32_t));
            }
            return;
        }
        constexpr std::size_t target_bytes = 4ull * 1024ull * 1024ull;
        const std::size_t values_per_chunk =
            std::max<std::size_t>(1, target_bytes / sizeof(std::uint32_t));
        std::vector<std::uint32_t> buffer;
        buffer.reserve(values_per_chunk);
        for (std::size_t base = 0; base < src.size();) {
            const std::size_t n = std::min(values_per_chunk, src.size() - base);
            buffer.clear();
            for (std::size_t i = 0; i < n; ++i) {
                buffer.push_back(remap_graph_node_id(node_remap, src[base + i], what));
            }
            write_raw_bytes(out, buffer.data(), buffer.size() * sizeof(std::uint32_t));
            base += n;
        }
    }

    static void write_remapped_edge_records(std::ostream& out,
                                            const std::vector<GraphEdgeRecord>& src,
                                            const NodeRemapTable& node_remap,
                                            const char* what)
    {
        detail::write_scalar<std::uint64_t>(out, src.size());
        if (node_remap.identity) {
            if (!src.empty()) {
                write_raw_bytes(out,
                                src.data(),
                                src.size() * sizeof(GraphEdgeRecord));
            }
            return;
        }
        constexpr std::size_t target_bytes = 4ull * 1024ull * 1024ull;
        const std::size_t records_per_chunk =
            std::max<std::size_t>(1, target_bytes / sizeof(GraphEdgeRecord));
        std::vector<GraphEdgeRecord> buffer;
        buffer.reserve(records_per_chunk);
        for (std::size_t base = 0; base < src.size();) {
            const std::size_t n = std::min(records_per_chunk, src.size() - base);
            buffer.clear();
            for (std::size_t i = 0; i < n; ++i) {
                const auto& edge = src[base + i];
                buffer.push_back(GraphEdgeRecord{
                    remap_graph_node_id(node_remap, edge.src, what),
                    remap_graph_node_id(node_remap, edge.dst, what)
                });
            }
            write_raw_bytes(out, buffer.data(), buffer.size() * sizeof(GraphEdgeRecord));
            base += n;
        }
    }

    void write_compact_block_hashes(std::ostream& out,
                                    const NodeRemapTable& node_remap) const
    {
        const std::uint32_t count = block_count();
        if (node_remap.identity && block_node_hashes.size() == count) {
            write_u64_vector(out, block_node_hashes);
            return;
        }
        detail::write_scalar<std::uint64_t>(out, count);
        constexpr std::size_t target_bytes = 4ull * 1024ull * 1024ull;
        const std::size_t values_per_chunk =
            std::max<std::size_t>(1, target_bytes / sizeof(std::uint64_t));
        std::vector<std::uint64_t> hashes;
        hashes.reserve(values_per_chunk);
        std::vector<std::uint32_t> names;
        for (std::uint32_t base = 0; base < count;) {
            const std::size_t n = std::min<std::size_t>(values_per_chunk, count - base);
            hashes.clear();
            for (std::size_t offset = 0; offset < n; ++offset) {
                const std::uint32_t bi = base + static_cast<std::uint32_t>(offset);
                const auto block = block_at(bi);
                names.clear();
                names.reserve(block.node_end - block.node_begin);
                for (std::uint32_t i = block.node_begin; i < block.node_end; ++i) {
                    if (i >= block_nodes.size()) {
                        throw std::runtime_error("SPQR compact index block node range out of bounds");
                    }
                    names.push_back(remap_graph_node_id(node_remap, block_nodes[i], "block node hash"));
                }
                std::sort(names.begin(), names.end());
                hashes.push_back(hash_sorted_name_set(names));
            }
            write_raw_bytes(out, hashes.data(), hashes.size() * sizeof(std::uint64_t));
            base += static_cast<std::uint32_t>(n);
        }
    }

    void write_graph_edge_tables_binary(
        std::ostream& out,
        const NodeRemapTable* precomputed_node_remap = nullptr) const
    {
        detail::write_scalar<std::uint32_t>(out, graph_edge_tables_magic);
        detail::write_scalar<std::uint32_t>(out, 1u);

        NodeRemapTable owned_node_remap;
        if (precomputed_node_remap == nullptr) {
            owned_node_remap = build_node_remap_table();
            precomputed_node_remap = &owned_node_remap;
        }
        const auto& node_remap = *precomputed_node_remap;
        write_remapped_edge_records(
            out, graph_edges, node_remap, "graph-edge endpoints");
        write_u8_vector(out, graph_edge_type_pairs);
        write_u8_vector(out, graph_edge_multi_flags);
    }

    void read_graph_edge_tables_binary(std::istream& in,
                                       const std::string& path,
                                       std::uint32_t expected_count)
    {
        const std::uint32_t magic =
            detail::read_scalar<std::uint32_t>(in, "graph-edge table magic");
        if (magic != graph_edge_tables_magic) {
            throw std::runtime_error("invalid binary SPQR graph-edge table section in " + path);
        }
        const std::uint32_t version =
            detail::read_scalar<std::uint32_t>(in, "graph-edge table version");
        if (version != 1u) {
            throw std::runtime_error("unsupported binary SPQR graph-edge table version in " + path);
        }

        const std::uint32_t count = read_count(in, "graph-edge table records");
        if (expected_count != 0u && count != expected_count) {
            throw std::runtime_error("binary SPQR graph-edge table count mismatch in " + path);
        }
        graph_edges.resize(count);
        if (!graph_edges.empty()) {
            read_raw_bytes(in,
                           graph_edges.data(),
                           graph_edges.size() * sizeof(GraphEdgeRecord),
                           "graph-edge table records payload");
        }
        graph_edge_count_hint = 0;
        read_u8_vector(in, graph_edge_type_pairs,
                       "graph-edge table endpoint type index");
        read_u8_vector(in, graph_edge_multi_flags,
                       "graph-edge table multi-link flags");

        if (!graph_edge_type_pairs.empty() && graph_edge_type_pairs.size() != graph_edges.size()) {
            throw std::runtime_error("binary SPQR graph-edge table has wrong endpoint type count in " + path);
        }
        if (!graph_edge_multi_flags.empty() && graph_edge_multi_flags.size() != graph_edges.size()) {
            throw std::runtime_error("binary SPQR graph-edge table has wrong multi-link flag count in " + path);
        }
    }

    void read_graph_edge_tables_section(std::istream& in,
                                        const std::string& path,
                                        const SectionEntryDisk& section,
                                        std::uint32_t expected_count)
    {
        (void)in;
        if (section.codec == static_cast<std::uint8_t>(SectionCodec::Raw)) {
            if (section.encoded_size != section.decoded_size) {
                throw std::runtime_error(
                    "raw SPQR graph-edge table section size mismatch in " + path);
            }
            RawSectionInputBuf raw_buf(
                path,
                section.offset,
                section.encoded_size,
                "raw SPQR graph-edge table section");
            std::istream section_in(&raw_buf);
            read_graph_edge_tables_binary(section_in, path, expected_count);
            if (raw_buf.consumed() != section.encoded_size) {
                throw std::runtime_error(
                    "raw SPQR graph-edge table section size mismatch in " + path);
            }
        } else if (section.codec == static_cast<std::uint8_t>(SectionCodec::Zstd)) {
#if BF_HAVE_ZSTD
            seekg_u64(in, section.offset, "SPQR graph-edge table section");
            ZstdInputBuf zstd_buf(in, section.encoded_size, section.decoded_size);
            std::istream section_in(&zstd_buf);
            read_graph_edge_tables_binary(section_in, path, expected_count);
            drain_stream(section_in, "zstd SPQR graph-edge table section", path);
            if (zstd_buf.decoded_bytes() != section.decoded_size) {
                throw std::runtime_error(
                    "zstd SPQR graph-edge table section size mismatch in " + path);
            }
#else
            throw std::runtime_error("zstd-compressed SPQR sections require zstd support: " + path);
#endif
        } else if (section.codec == static_cast<std::uint8_t>(SectionCodec::ZstdChunked)) {
#if BF_HAVE_ZSTD
            ZstdChunkedInputBuf zstd_buf(path, section, selected_zstd_chunk_prefetch());
            std::istream section_in(&zstd_buf);
            read_graph_edge_tables_binary(section_in, path, expected_count);
            drain_stream(
                section_in, "chunked zstd SPQR graph-edge table section", path);
            if (zstd_buf.decoded_bytes() != section.decoded_size) {
                throw std::runtime_error(
                    "chunked zstd SPQR graph-edge table section size mismatch in " + path);
            }
#else
            throw std::runtime_error("chunked zstd SPQR sections require zstd support: " + path);
#endif
        } else {
            throw std::runtime_error("unsupported SPQR graph-edge table section codec in " + path);
        }
    }

    void take_graph_edge_tables(IndexedSpqrTree& src)
    {
        graph_edges = std::move(src.graph_edges);
        graph_edge_type_pairs = std::move(src.graph_edge_type_pairs);
        graph_edge_multi_flags = std::move(src.graph_edge_multi_flags);
        graph_edge_count_hint = 0;
    }

    static IndexedSpqrTree load_graph_edge_tables_section_from_file(
        const std::string& path,
        const SectionEntryDisk& section)
    {
        std::ifstream in(path, std::ios::in | std::ios::binary);
        if (!in) {
            throw std::runtime_error("cannot open binary SPQR index: " + path);
        }
        IndexedSpqrTree tmp;
        tmp.read_graph_edge_tables_section(in, path, section, 0);
        return tmp;
    }

    void write_graph_components_binary(
        std::ostream& out,
        const NodeRemapTable& node_remap) const
    {
        const std::uint32_t node_count = graph_node_count();
        const std::uint32_t edge_count = graph_index::require_spqr_count_size(
            graph_edges.size(), "graph component edge count");

        std::vector<std::uint32_t> parent(node_count);
        std::vector<std::uint8_t> rank(node_count, 0u);
        std::iota(parent.begin(), parent.end(), 0u);

        auto find_root = [&](std::uint32_t v) {
            std::uint32_t root = v;
            while (parent[root] != root) root = parent[root];
            while (parent[v] != v) {
                const std::uint32_t next = parent[v];
                parent[v] = root;
                v = next;
            }
            return root;
        };
        auto unite = [&](std::uint32_t a, std::uint32_t b) {
            std::uint32_t ra = find_root(a);
            std::uint32_t rb = find_root(b);
            if (ra == rb) return;
            if (rank[ra] < rank[rb]) std::swap(ra, rb);
            parent[rb] = ra;
            if (rank[ra] == rank[rb]) ++rank[ra];
        };

        for (const auto& edge : graph_edges) {
            const std::uint32_t src =
                remap_graph_node_id(node_remap, edge.src, "graph component src");
            const std::uint32_t dst =
                remap_graph_node_id(node_remap, edge.dst, "graph component dst");
            unite(src, dst);
        }

        std::vector<std::uint32_t> root_to_component(node_count, invalid_id);
        std::vector<std::uint32_t> node_component(node_count, invalid_id);
        std::uint32_t component_count = 0;
        for (std::uint32_t node = 0; node < node_count; ++node) {
            const std::uint32_t root = find_root(node);
            std::uint32_t& comp = root_to_component[root];
            if (comp == invalid_id) comp = component_count++;
            node_component[node] = comp;
        }

        std::vector<GraphComponentRecord> records(component_count);
        for (std::uint32_t node = 0; node < node_count; ++node) {
            ++records[node_component[node]].node_end;
        }
        for (const auto& edge : graph_edges) {
            const std::uint32_t src =
                remap_graph_node_id(node_remap, edge.src, "graph component src");
            ++records[node_component[src]].edge_end;
        }

        std::uint32_t node_begin = 0;
        std::uint32_t edge_begin = 0;
        for (auto& rec : records) {
            const std::uint32_t node_count_in_component = rec.node_end;
            const std::uint32_t edge_count_in_component = rec.edge_end;
            rec.node_begin = node_begin;
            rec.node_end = node_begin + node_count_in_component;
            rec.edge_begin = edge_begin;
            rec.edge_end = edge_begin + edge_count_in_component;
            node_begin = rec.node_end;
            edge_begin = rec.edge_end;
        }

        bool nodes_identity = true;
        for (std::uint32_t node = 0; node < node_count; ++node) {
            const auto& rec = records[node_component[node]];
            if (node < rec.node_begin || node >= rec.node_end) {
                nodes_identity = false;
                break;
            }
        }

        bool edges_identity = true;
        for (std::uint32_t edge_id = 0; edge_id < edge_count; ++edge_id) {
            const auto& edge = graph_edges[edge_id];
            const std::uint32_t src =
                remap_graph_node_id(node_remap, edge.src, "graph component src");
            const auto& rec = records[node_component[src]];
            if (edge_id < rec.edge_begin || edge_id >= rec.edge_end) {
                edges_identity = false;
                break;
            }
        }

        std::vector<std::uint32_t> nodes;
        std::vector<std::uint32_t> edges;
        if (!nodes_identity || !edges_identity) {
            std::vector<std::uint32_t> node_cursor(component_count);
            std::vector<std::uint32_t> edge_cursor(component_count);
            for (std::uint32_t comp = 0; comp < component_count; ++comp) {
                node_cursor[comp] = records[comp].node_begin;
                edge_cursor[comp] = records[comp].edge_begin;
            }

            if (!nodes_identity) {
                nodes.resize(node_count);
                for (std::uint32_t node = 0; node < node_count; ++node) {
                    nodes[node_cursor[node_component[node]]++] = node;
                }
            }

            if (!edges_identity) {
                edges.resize(edge_count);
                for (std::uint32_t edge_id = 0; edge_id < edge_count; ++edge_id) {
                    const auto& edge = graph_edges[edge_id];
                    const std::uint32_t src =
                        remap_graph_node_id(node_remap, edge.src, "graph component src");
                    edges[edge_cursor[node_component[src]]++] = edge_id;
                }
            }
        }

        std::uint32_t flags = 0;
        if (nodes_identity) flags |= graph_component_nodes_identity_flag;
        if (edges_identity) flags |= graph_component_edges_identity_flag;

        detail::write_scalar<std::uint32_t>(out, graph_components_magic);
        detail::write_scalar<std::uint32_t>(out, 2u);
        detail::write_scalar<std::uint64_t>(out, node_count);
        detail::write_scalar<std::uint64_t>(out, edge_count);
        detail::write_scalar<std::uint32_t>(out, flags);
        detail::write_scalar<std::uint64_t>(
            out, static_cast<std::uint64_t>(records.size()));
        if (!records.empty()) {
            write_raw_bytes(out,
                            records.data(),
                            records.size() * sizeof(GraphComponentRecord));
        }
        if (!nodes_identity) write_u32_vector(out, nodes);
        if (!edges_identity) write_u32_vector(out, edges);
    }

    void read_graph_components_binary(std::istream& in,
                                      const std::string& path,
                                      std::uint32_t expected_nodes,
                                      std::uint32_t expected_edges)
    {
        const std::uint32_t magic =
            detail::read_scalar<std::uint32_t>(in, "graph component magic");
        if (magic != graph_components_magic) {
            throw std::runtime_error("invalid binary SPQR graph-component section in " + path);
        }
        const std::uint32_t version =
            detail::read_scalar<std::uint32_t>(in, "graph component version");
        if (version != 2u) {
            throw std::runtime_error("unsupported binary SPQR graph-component version in " + path);
        }
        const std::uint32_t node_count = graph_index::require_spqr_count(
            detail::read_scalar<std::uint64_t>(in, "graph component node count"),
            "graph component node count");
        const std::uint32_t edge_count = graph_index::require_spqr_count(
            detail::read_scalar<std::uint64_t>(in, "graph component edge count"),
            "graph component edge count");
        if (node_count != expected_nodes || edge_count != expected_edges) {
            throw std::runtime_error(
                "binary SPQR graph-component count mismatch in " + path);
        }
        const std::uint32_t flags = detail::read_scalar<std::uint32_t>(
            in, "graph component encoding flags");
        const std::uint32_t known_flags =
            graph_component_nodes_identity_flag |
            graph_component_edges_identity_flag;
        if ((flags & ~known_flags) != 0u) {
            throw std::runtime_error(
                "binary SPQR graph-component section has unknown flags in " + path);
        }
        graph_component_nodes_identity =
            (flags & graph_component_nodes_identity_flag) != 0u;
        graph_component_edges_identity =
            (flags & graph_component_edges_identity_flag) != 0u;

        graph_components.resize(read_count(in, "graph component records"));
        if (!graph_components.empty()) {
            read_raw_bytes(in,
                           graph_components.data(),
                           graph_components.size() * sizeof(GraphComponentRecord),
                           "graph component records payload");
        }
        if (!graph_component_nodes_identity) {
            read_u32_vector(in, graph_component_nodes, "graph component nodes");
        }
        if (!graph_component_edges_identity) {
            read_u32_vector(in, graph_component_edges, "graph component edges");
        }
        if ((!graph_component_nodes_identity &&
             graph_component_nodes.size() != expected_nodes) ||
            (!graph_component_edges_identity &&
             graph_component_edges.size() != expected_edges)) {
            throw std::runtime_error(
                "binary SPQR graph-component partition size mismatch in " + path);
        }

        std::uint32_t previous_node_end = 0;
        std::uint32_t previous_edge_end = 0;
        for (const auto& rec : graph_components) {
            if (rec.node_begin != previous_node_end ||
                rec.edge_begin != previous_edge_end ||
                rec.node_end < rec.node_begin ||
                rec.edge_end < rec.edge_begin ||
                rec.node_end > expected_nodes ||
                rec.edge_end > expected_edges) {
                throw std::runtime_error(
                    "binary SPQR graph-component ranges are invalid in " + path);
            }
            previous_node_end = rec.node_end;
            previous_edge_end = rec.edge_end;
        }
        if ((expected_nodes != 0 || expected_edges != 0) &&
            (graph_components.empty() ||
             previous_node_end != expected_nodes ||
             previous_edge_end != expected_edges)) {
            throw std::runtime_error(
                "binary SPQR graph-component partition is incomplete in " + path);
        }
        if (!graph_component_nodes_identity) {
            for (std::uint32_t node : graph_component_nodes) {
                if (node >= expected_nodes) {
                    throw std::runtime_error(
                        "binary SPQR graph-component node out of range in " + path);
                }
            }
        }
        if (!graph_component_edges_identity) {
            for (std::uint32_t edge : graph_component_edges) {
                if (edge >= expected_edges) {
                    throw std::runtime_error(
                        "binary SPQR graph-component edge out of range in " + path);
                }
            }
        }
    }

    static IndexedSpqrTree load_graph_components_section_from_file(
        const std::string& path,
        const SectionEntryDisk& section,
        std::uint32_t expected_nodes,
        std::uint32_t expected_edges)
    {
        IndexedSpqrTree tmp;
        read_section_payload_from_file(
            path,
            section,
            "columnar SPQR graph-component section",
            selected_zstd_chunk_prefetch(4),
            [&](std::istream& section_in) {
                tmp.read_graph_components_binary(
                    section_in, path, expected_nodes, expected_edges);
            });
        return tmp;
    }

    // Haplotype path names use a local string table. Steps stay in one flat list:
    //
    //   [small header][local strings][path records][step records]
    //   A path record keeps begin/end offsets in the step records.

    void write_haplotypes_binary(std::ostream& out,
                                 const NodeRemapTable& node_remap) const
    {
        detail::write_scalar<std::uint32_t>(out, haplotypes_magic);
        detail::write_scalar<std::uint32_t>(out, 1u);
        detail::write_scalar<std::uint64_t>(out, 0u);
        detail::write_scalar<std::uint64_t>(out, graph_node_count());
        detail::write_scalar<std::uint64_t>(
            out, static_cast<std::uint64_t>(haplotype_paths.size()));
        detail::write_scalar<std::uint64_t>(
            out, static_cast<std::uint64_t>(haplotype_steps.size()));

        std::unordered_map<std::uint32_t, std::uint32_t> local_by_global;
        local_by_global.reserve(
            graph_index::checked_mul_size(
                haplotype_paths.size(), 3u, "SPQR haplotype string map"));
        std::vector<std::uint32_t> local_global_ids;
        local_global_ids.reserve(local_by_global.bucket_count());
        auto to_local_string = [&](std::uint32_t global) -> std::uint32_t {
            if (global == invalid_id) return invalid_id;
            if (global >= strings.size()) {
                throw std::runtime_error(
                    "SPQR haplotype metadata string id out of range");
            }
            auto inserted = local_by_global.emplace(
                global,
                graph_index::require_spqr_count_size(
                    local_global_ids.size(),
                    "SPQR haplotype local string table"));
            if (inserted.second) local_global_ids.push_back(global);
            return inserted.first->second;
        };

        std::vector<HaplotypePathRecord> local_paths;
        local_paths.reserve(haplotype_paths.size());
        for (const auto& path_rec : haplotype_paths) {
            if (path_rec.step_begin > path_rec.step_end ||
                path_rec.step_end > haplotype_steps.size() ||
                path_rec.sense > 2u) {
                throw std::runtime_error("SPQR haplotype path record is invalid");
            }
            HaplotypePathRecord out_rec = path_rec;
            out_rec.name = to_local_string(path_rec.name);
            out_rec.sample = to_local_string(path_rec.sample);
            out_rec.locus = to_local_string(path_rec.locus);
            local_paths.push_back(out_rec);
        }

        detail::write_scalar<std::uint64_t>(
            out, static_cast<std::uint64_t>(local_global_ids.size()));
        for (std::uint32_t global : local_global_ids) {
            detail::write_string(out, strings[global]);
        }

        write_trivial_vector64(out, local_paths);

        detail::write_scalar<std::uint64_t>(
            out, static_cast<std::uint64_t>(haplotype_steps.size()));
        constexpr std::size_t target_bytes = 4ull * 1024ull * 1024ull;
        const std::size_t records_per_chunk =
            std::max<std::size_t>(1, target_bytes / sizeof(HaplotypeStepRecord));
        std::vector<HaplotypeStepRecord> buffer;
        buffer.reserve(records_per_chunk);
        for (std::size_t base = 0; base < haplotype_steps.size();) {
            const std::size_t n = std::min<std::size_t>(
                records_per_chunk, haplotype_steps.size() - base);
            buffer.clear();
            for (std::size_t i = 0; i < n; ++i) {
                HaplotypeStepRecord rec = haplotype_steps[base + i];
                if (rec.is_reverse > 1u) {
                    throw std::runtime_error(
                        "SPQR haplotype step orientation is invalid");
                }
                rec.node = remap_graph_node_id(
                    node_remap, rec.node, "haplotype path step node");
                buffer.push_back(rec);
            }
            write_raw_bytes(out,
                            buffer.data(),
                            buffer.size() * sizeof(HaplotypeStepRecord));
            base += n;
        }
    }

    void read_haplotypes_binary(std::istream& in,
                                const std::string& path,
                                std::uint32_t expected_nodes)
    {
        const std::uint32_t magic =
            detail::read_scalar<std::uint32_t>(in, "haplotype section magic");
        if (magic != haplotypes_magic) {
            throw std::runtime_error("invalid binary SPQR haplotype section in " + path);
        }
        const std::uint32_t version =
            detail::read_scalar<std::uint32_t>(in, "haplotype section version");
        if (version != 1u) {
            throw std::runtime_error(
                "unsupported binary SPQR haplotype section version in " + path);
        }
        (void)detail::read_scalar<std::uint64_t>(
            in, "haplotype section flags");
        const std::uint32_t node_count = graph_index::require_spqr_count(
            detail::read_scalar<std::uint64_t>(in, "haplotype graph node count"),
            "haplotype graph node count");
        if (node_count != expected_nodes) {
            throw std::runtime_error(
                "binary SPQR haplotype graph node count mismatch in " + path);
        }
        const std::uint64_t expected_paths =
            detail::read_scalar<std::uint64_t>(in, "haplotype path count");
        const std::uint64_t expected_steps =
            detail::read_scalar<std::uint64_t>(in, "haplotype step count");

        const std::uint32_t local_string_count = read_count(
            in, "haplotype local string count");
        std::vector<std::uint32_t> local_to_global;
        local_to_global.reserve(local_string_count);
        for (std::uint32_t i = 0; i < local_string_count; ++i) {
            local_to_global.push_back(
                intern(detail::read_string(in, "haplotype metadata string")));
        }

        read_trivial_vector64(in, haplotype_paths, "haplotype path records");
        if (static_cast<std::uint64_t>(haplotype_paths.size()) != expected_paths) {
            throw std::runtime_error(
                "binary SPQR haplotype path count mismatch in " + path);
        }
        auto map_local_string = [&](std::uint32_t local) -> std::uint32_t {
            if (local == invalid_id) return invalid_id;
            if (local >= local_to_global.size()) {
                throw std::runtime_error(
                    "binary SPQR haplotype string id out of range in " + path);
            }
            return local_to_global[local];
        };
        for (auto& path_rec : haplotype_paths) {
            if (path_rec.step_begin > path_rec.step_end ||
                path_rec.step_end > expected_steps ||
                path_rec.sense > 2u) {
                throw std::runtime_error(
                    "binary SPQR haplotype path range is invalid in " + path);
            }
            path_rec.name = map_local_string(path_rec.name);
            path_rec.sample = map_local_string(path_rec.sample);
            path_rec.locus = map_local_string(path_rec.locus);
        }

        read_trivial_vector64(in, haplotype_steps, "haplotype step records");
        if (static_cast<std::uint64_t>(haplotype_steps.size()) != expected_steps) {
            throw std::runtime_error(
                "binary SPQR haplotype step count mismatch in " + path);
        }
        for (const auto& step : haplotype_steps) {
            if (step.node >= expected_nodes || step.is_reverse > 1u) {
                throw std::runtime_error(
                    "binary SPQR haplotype step is invalid in " + path);
            }
        }
    }

    static IndexedSpqrTree load_haplotypes_section_from_file(
        const std::string& path,
        const SectionEntryDisk& section,
        std::uint32_t expected_nodes)
    {
        IndexedSpqrTree tmp;
        read_section_payload_from_file(
            path,
            section,
            "columnar SPQR haplotype path section",
            selected_zstd_chunk_prefetch(4),
            [&](std::istream& section_in) {
                tmp.read_haplotypes_binary(section_in, path, expected_nodes);
            });
        return tmp;
    }
