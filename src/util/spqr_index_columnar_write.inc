    // Write the current .spqri format, one array per section.

    void write_runtime_header_payload(
        std::ostream& out,
        bool external_graph_edge_tables) const
    {
        write_column_payload_header(out, section_id_runtime_header);
        const ColumnarRuntimeHeaderDisk header{
            static_cast<std::uint64_t>(strings.size()),
            cuts.empty() ? cut_count_hint : static_cast<std::uint64_t>(cuts.size()),
            component_count(),
            static_cast<std::uint64_t>(graph_node_count()),
            static_cast<std::uint64_t>(block_count()),
            static_cast<std::uint64_t>(tree_node_count()),
            static_cast<std::uint64_t>(virtual_edge_count()),
            static_cast<std::uint64_t>(real_edge_count()),
            external_graph_edge_tables
                ? static_cast<std::uint64_t>(graph_edges.size())
                : graph_edge_count_hint,
            bubble_edge_type_count_hint != 0
                ? bubble_edge_type_count_hint
                : static_cast<std::uint64_t>(bubble_edge_types.size())
        };
        detail::write_scalar<std::uint64_t>(out, header.original_string_count);
        detail::write_scalar<std::uint64_t>(out, header.cut_count);
        detail::write_scalar<std::uint64_t>(out, header.component_count);
        detail::write_scalar<std::uint64_t>(out, header.graph_node_count);
        detail::write_scalar<std::uint64_t>(out, header.block_count);
        detail::write_scalar<std::uint64_t>(out, header.tree_node_count);
        detail::write_scalar<std::uint64_t>(out, header.virtual_edge_count);
        detail::write_scalar<std::uint64_t>(out, header.real_edge_count);
        detail::write_scalar<std::uint64_t>(out, header.graph_edge_count);
        detail::write_scalar<std::uint64_t>(out, header.bubble_edge_type_count);
        detail::write_scalar<std::uint64_t>(out, 0u);
    }

    static ColumnarRuntimeHeaderDisk read_runtime_header_payload(
        std::istream& in)
    {
        ColumnarRuntimeHeaderDisk header;
        header.original_string_count =
            detail::read_scalar<std::uint64_t>(in, "columnar original string count");
        header.cut_count =
            detail::read_scalar<std::uint64_t>(in, "columnar cut count");
        header.component_count =
            detail::read_scalar<std::uint64_t>(in, "columnar component count");
        header.graph_node_count =
            detail::read_scalar<std::uint64_t>(in, "columnar graph node count");
        header.block_count =
            detail::read_scalar<std::uint64_t>(in, "columnar block count");
        header.tree_node_count =
            detail::read_scalar<std::uint64_t>(in, "columnar tree-node count");
        header.virtual_edge_count =
            detail::read_scalar<std::uint64_t>(in, "columnar virtual-edge count");
        header.real_edge_count =
            detail::read_scalar<std::uint64_t>(in, "columnar real-edge count");
        header.graph_edge_count =
            detail::read_scalar<std::uint64_t>(in, "columnar graph-edge count");
        header.bubble_edge_type_count =
            detail::read_scalar<std::uint64_t>(in, "columnar BubbleFinder edge type count");
        (void)detail::read_scalar<std::uint64_t>(in, "columnar runtime flags");
        return header;
    }

    struct NumericLookupWritePlan {
        std::uint8_t encoding = 0;
        std::uint32_t graph_node_count = 0;
        std::uint64_t base = 0;
        std::vector<std::uint32_t> dense_domain;
    };

    bool graph_node_numeric_value_for_write(std::uint32_t compact,
                                            std::uint64_t& out_numeric) const
    {
        if (has_compact_numeric_graph_node_names()) {
            if (!graph_node_numeric_name_valid_at(compact)) return false;
            out_numeric = graph_node_numeric_name_at(compact);
            return true;
        }

        if (compact >= node_names.size()) return false;
        const std::uint32_t old_name = node_names[compact];
        if (old_name >= strings.size()) {
            throw std::runtime_error("SPQR compact index graph node name out of range");
        }
        if (uses_oriented_numeric_graph_node_names()) {
            return parse_oriented_canonical_u64(strings[old_name], out_numeric);
        }
        return parse_canonical_u64(strings[old_name], out_numeric);
    }

    bool build_graph_node_numeric_lookup_write_plan(
        NumericLookupWritePlan& plan) const
    {
        plan = NumericLookupWritePlan{};
        const std::uint32_t n = graph_node_count();
        if (n == 0) return false;

        std::uint64_t min_value = std::numeric_limits<std::uint64_t>::max();
        std::uint64_t max_value = 0;
        bool affine = true;
        bool affine_base_set = false;
        std::uint64_t affine_base = 0;
        for (std::uint32_t i = 0; i < n; ++i) {
            std::uint64_t value = 0;
            if (!graph_node_numeric_value_for_write(i, value)) return false;
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
            if (affine) {
                if (value < static_cast<std::uint64_t>(i)) {
                    affine = false;
                } else if (!affine_base_set) {
                    affine_base = value - static_cast<std::uint64_t>(i);
                    affine_base_set = true;
                } else if (affine_base >
                               std::numeric_limits<std::uint64_t>::max() -
                                   static_cast<std::uint64_t>(i) ||
                           value != affine_base + static_cast<std::uint64_t>(i)) {
                    affine = false;
                }
            }
        }

        plan.graph_node_count = n;
        if (affine && affine_base_set) {
            plan.encoding = 1u;
            plan.base = affine_base;
            return true;
        }

        if (max_value < min_value) return false;
        const std::uint64_t range64 = max_value - min_value + 1u;
        if (range64 > static_cast<std::uint64_t>(
                          std::numeric_limits<std::uint32_t>::max())) {
            return false;
        }
        const std::uint64_t max_reasonable_range =
            std::max<std::uint64_t>(1024u, static_cast<std::uint64_t>(n) * 2u);
        if (range64 > max_reasonable_range) return false;

        std::vector<std::uint32_t> dense(
            static_cast<std::size_t>(range64), invalid_id);
        for (std::uint32_t i = 0; i < n; ++i) {
            std::uint64_t value = 0;
            if (!graph_node_numeric_value_for_write(i, value)) return false;
            const std::size_t offset =
                static_cast<std::size_t>(value - min_value);
            if (dense[offset] != invalid_id) return false;
            dense[offset] = i;
        }

        plan.encoding = 2u;
        plan.base = min_value;
        plan.dense_domain = std::move(dense);
        return true;
    }

    static void write_graph_node_numeric_lookup_payload(
        std::ostream& out,
        const NumericLookupWritePlan& plan)
    {
        write_column_payload_header(out, section_id_graph_node_numeric_lookup);
        detail::write_scalar<std::uint8_t>(out, plan.encoding);
        detail::write_scalar<std::uint64_t>(out, plan.graph_node_count);
        detail::write_scalar<std::uint64_t>(out, plan.base);
        if (plan.encoding == 2u) {
            write_u32_vector(out, plan.dense_domain);
        }
    }

    void read_graph_node_numeric_lookup_payload(
        std::istream& in,
        std::uint32_t expected_graph_nodes,
        const std::string& path)
    {
        const std::uint8_t encoding =
            detail::read_scalar<std::uint8_t>(
                in, "columnar graph-node numeric lookup encoding");
        const std::uint32_t graph_nodes =
            read_count(in, "columnar graph-node numeric lookup count");
        if (graph_nodes != expected_graph_nodes) {
            throw std::runtime_error(
                "columnar SPQR graph-node numeric lookup count mismatch in " + path);
        }
        graph_node_numeric_lookup_base =
            detail::read_scalar<std::uint64_t>(
                in, "columnar graph-node numeric lookup base");
        compact_graph_node_count = expected_graph_nodes;

        if (encoding == 1u) {
            if (expected_graph_nodes != 0 &&
                graph_node_numeric_lookup_base >
                    std::numeric_limits<std::uint64_t>::max() -
                        static_cast<std::uint64_t>(expected_graph_nodes - 1u)) {
                throw std::runtime_error(
                    "columnar SPQR affine graph-node numeric lookup overflows in " +
                    path);
            }
            graph_node_numeric_lookup_dense = true;
            graph_node_numeric_lookup_ready = true;
            return;
        }

        if (encoding == 2u) {
            read_u32_vector(in,
                            graph_node_by_dense_numeric_name,
                            "columnar graph-node dense numeric lookup");
            if (graph_node_by_dense_numeric_name.empty()) {
                throw std::runtime_error(
                    "columnar SPQR dense graph-node numeric lookup is empty in " +
                    path);
            }
            if (!graph_node_by_dense_numeric_name.empty() &&
                graph_node_numeric_lookup_base >
                    std::numeric_limits<std::uint64_t>::max() -
                        static_cast<std::uint64_t>(
                            graph_node_by_dense_numeric_name.size() - 1u)) {
                throw std::runtime_error(
                    "columnar SPQR dense graph-node numeric lookup overflows in " +
                    path);
            }
            for (std::uint32_t value : graph_node_by_dense_numeric_name) {
                if (value != invalid_id && value >= expected_graph_nodes) {
                    throw std::runtime_error(
                        "columnar SPQR dense graph-node numeric lookup has an out-of-range ordinal in " +
                        path);
                }
            }
            graph_node_numeric_lookup_dense_domain = true;
            graph_node_numeric_lookup_ready = true;
            return;
        }

        throw std::runtime_error(
            "unsupported columnar graph-node numeric lookup encoding in " + path);
    }

    void drop_graph_node_name_storage(std::uint32_t graph_nodes)
    {
        drop_storage(strings);
        drop_storage(node_names);
        compact_graph_node_count = graph_nodes;
        drop_storage(graph_node_numeric_names);
        drop_storage(graph_node_numeric_names32);
        graph_node_numeric_names_dense = false;
        graph_node_numeric_name_base = 0;
        drop_storage(graph_node_numeric_name_valid);
        drop_storage(graph_node_string_names);
    }

    void materialize_graph_node_names_from_numeric_lookup(std::uint32_t expected_graph_nodes,
                                                          const std::string& path)
    {
        drop_graph_node_name_storage(expected_graph_nodes);

        if (!graph_node_numeric_lookup_ready) {
            throw std::runtime_error(
                "columnar SPQR graph-node numeric lookup is not available in " + path);
        }

        if (graph_node_numeric_lookup_dense) {
            graph_node_numeric_names_dense = true;
            graph_node_numeric_name_base = graph_node_numeric_lookup_base;
            if (expected_graph_nodes != 0 &&
                graph_node_numeric_name_base >
                    std::numeric_limits<std::uint64_t>::max() -
                        static_cast<std::uint64_t>(expected_graph_nodes - 1u)) {
                throw std::runtime_error(
                    "columnar SPQR affine graph-node numeric names overflow in " + path);
            }
            return;
        }

        if (!graph_node_numeric_lookup_dense_domain) {
            throw std::runtime_error(
                "columnar SPQR graph-node numeric lookup cannot materialize names in " +
                path);
        }

        if (!graph_node_by_dense_numeric_name.empty() &&
            graph_node_numeric_lookup_base >
                std::numeric_limits<std::uint64_t>::max() -
                    static_cast<std::uint64_t>(
                        graph_node_by_dense_numeric_name.size() - 1u)) {
            throw std::runtime_error(
                "columnar SPQR dense graph-node numeric names overflow in " + path);
        }

        const std::uint64_t max_value = graph_node_by_dense_numeric_name.empty()
            ? graph_node_numeric_lookup_base
            : graph_node_numeric_lookup_base +
                  static_cast<std::uint64_t>(
                      graph_node_by_dense_numeric_name.size() - 1u);
        const bool fits_u32 =
            max_value <= static_cast<std::uint64_t>(
                std::numeric_limits<std::uint32_t>::max());
        std::vector<std::uint8_t> seen(expected_graph_nodes, 0);
        std::uint32_t seen_count = 0;

        if (fits_u32) {
            graph_node_numeric_names32.assign(expected_graph_nodes, 0u);
        } else {
            graph_node_numeric_names.assign(expected_graph_nodes, 0u);
        }

        for (std::size_t offset = 0;
             offset < graph_node_by_dense_numeric_name.size();
             ++offset) {
            const std::uint32_t ordinal = graph_node_by_dense_numeric_name[offset];
            if (ordinal == invalid_id) continue;
            if (ordinal >= expected_graph_nodes || seen[ordinal] != 0) {
                throw std::runtime_error(
                    "columnar SPQR dense graph-node numeric lookup is not a valid inverse name table in " +
                    path);
            }
            seen[ordinal] = 1u;
            ++seen_count;
            const std::uint64_t numeric =
                graph_node_numeric_lookup_base + static_cast<std::uint64_t>(offset);
            if (fits_u32) {
                graph_node_numeric_names32[ordinal] =
                    static_cast<std::uint32_t>(numeric);
            } else {
                graph_node_numeric_names[ordinal] = numeric;
            }
        }

        if (seen_count != expected_graph_nodes) {
            throw std::runtime_error(
                "columnar SPQR dense graph-node numeric lookup does not cover every graph node in " +
                path);
        }
    }

    void write_graph_node_names_payload(std::ostream& out) const
    {
        write_column_payload_header(out, section_id_graph_node_names);

        const std::uint32_t graph_nodes = graph_node_count();
        const bool compact_numeric_source = has_compact_numeric_graph_node_names();
        auto sparse_it = graph_node_string_names.begin();

        auto sparse_string_at = [&](std::uint32_t compact) -> const std::string* {
            while (sparse_it != graph_node_string_names.end() &&
                   sparse_it->first < compact) {
                ++sparse_it;
            }
            if (sparse_it != graph_node_string_names.end() &&
                sparse_it->first == compact) {
                return &sparse_it->second;
            }
            return nullptr;
        };

        auto source_numeric = [&](std::uint32_t compact,
                                  std::uint64_t& out_numeric) -> bool {
            if (compact_numeric_source) {
                if (sparse_string_at(compact) != nullptr) return false;
                out_numeric = graph_node_numeric_name_at(compact);
                return true;
            }

            const std::uint32_t old_name = node_names[compact];
            if (old_name >= strings.size()) {
                throw std::runtime_error("SPQR compact index graph node name out of range");
            }
            if (uses_oriented_numeric_graph_node_names()) {
                return parse_oriented_canonical_u64(strings[old_name], out_numeric);
            }
            return parse_canonical_u64(strings[old_name], out_numeric);
        };

        auto source_string = [&](std::uint32_t compact) -> std::string {
            if (compact_numeric_source) {
                if (const std::string* sparse = sparse_string_at(compact)) {
                    return *sparse;
                }
                if (uses_oriented_numeric_graph_node_names()) {
                    return oriented_numeric_graph_node_name(
                        graph_node_numeric_name_at(compact));
                }
                return std::to_string(graph_node_numeric_name_at(compact));
            }

            const std::uint32_t old_name = node_names[compact];
            if (old_name >= strings.size()) {
                throw std::runtime_error("SPQR compact index graph node name out of range");
            }
            return strings[old_name];
        };

        std::uint64_t numeric_name_count = 0;
        std::uint64_t string_name_count = 0;
        std::uint64_t full_string_payload_bytes = 0;
        std::uint64_t sparse_string_payload_bytes = 0;
        std::uint64_t max_numeric_name = 0;
        bool dense_numeric_possible = true;
        bool dense_numeric_base_set = false;
        std::uint64_t dense_numeric_base = 0;
        for (std::uint32_t compact = 0; compact < graph_nodes; ++compact) {
            std::uint64_t numeric = 0;
            const bool numeric_name = source_numeric(compact, numeric);
            if (numeric_name) {
                ++numeric_name_count;
                max_numeric_name = std::max(max_numeric_name, numeric);
                if (dense_numeric_possible) {
                    if (numeric < static_cast<std::uint64_t>(compact)) {
                        dense_numeric_possible = false;
                    } else if (!dense_numeric_base_set) {
                        dense_numeric_base =
                            numeric - static_cast<std::uint64_t>(compact);
                        dense_numeric_base_set = true;
                    } else if (dense_numeric_base >
                                   std::numeric_limits<std::uint64_t>::max() -
                                       static_cast<std::uint64_t>(compact) ||
                               numeric != dense_numeric_base +
                                              static_cast<std::uint64_t>(compact)) {
                        dense_numeric_possible = false;
                    }
                }
                full_string_payload_bytes += 8u + decimal_u64_digits(numeric);
            } else {
                const std::string value = source_string(compact);
                ++string_name_count;
                full_string_payload_bytes +=
                    8u + static_cast<std::uint64_t>(value.size());
                sparse_string_payload_bytes +=
                    4u + 8u + static_cast<std::uint64_t>(value.size());
            }
        }

        const bool numeric_fits_u32 =
            max_numeric_name <= static_cast<std::uint64_t>(
                std::numeric_limits<std::uint32_t>::max());
        const std::uint64_t numeric_width = numeric_fits_u32 ? 4u : 8u;
        const bool dense_numeric_source =
            dense_numeric_possible && dense_numeric_base_set;
        std::uint8_t node_name_encoding = 0u;
        if (dense_numeric_source && string_name_count == 0) {
            node_name_encoding = 5u;
        } else if (dense_numeric_source && string_name_count != 0) {
            const std::uint64_t n = static_cast<std::uint64_t>(graph_nodes);
            const std::uint64_t dense_mixed_payload_bytes =
                8u + 8u + sparse_string_payload_bytes;
            if (dense_mixed_payload_bytes <= full_string_payload_bytes ||
                numeric_name_count * 2u >= n) {
                node_name_encoding = 6u;
            } else {
                const std::uint64_t mixed_payload_bytes =
                    8u + n * numeric_width + 8u + sparse_string_payload_bytes;
                if (mixed_payload_bytes <= full_string_payload_bytes ||
                    numeric_name_count * 2u >= n) {
                    node_name_encoding = numeric_fits_u32 ? 4u : 2u;
                }
            }
        } else if (numeric_name_count != 0 && string_name_count == 0) {
            node_name_encoding = numeric_fits_u32 ? 3u : 1u;
        } else if (numeric_name_count != 0 && string_name_count != 0) {
            const std::uint64_t n = static_cast<std::uint64_t>(graph_nodes);
            const std::uint64_t mixed_payload_bytes =
                8u + n * numeric_width + 8u + sparse_string_payload_bytes;
            if (mixed_payload_bytes <= full_string_payload_bytes ||
                numeric_name_count * 2u >= n) {
                node_name_encoding = numeric_fits_u32 ? 4u : 2u;
            }
        }

        detail::write_scalar<std::uint8_t>(out, node_name_encoding);
        detail::write_scalar<std::uint64_t>(out, graph_nodes);
        if (node_name_encoding == 5u || node_name_encoding == 6u) {
            detail::write_scalar<std::uint64_t>(out, dense_numeric_base);
            if (node_name_encoding == 6u) {
                detail::write_scalar<std::uint64_t>(out, string_name_count);
                sparse_it = graph_node_string_names.begin();
                for (std::uint32_t compact = 0; compact < graph_nodes; ++compact) {
                    std::uint64_t numeric = 0;
                    if (source_numeric(compact, numeric)) continue;
                    detail::write_scalar<std::uint32_t>(out, compact);
                    const std::string value = source_string(compact);
                    detail::write_string(out, value);
                }
            }
            return;
        }
        if (node_name_encoding == 1u || node_name_encoding == 2u ||
            node_name_encoding == 3u || node_name_encoding == 4u) {
            constexpr std::size_t target_bytes = 4ull * 1024ull * 1024ull;
            if (node_name_encoding == 3u || node_name_encoding == 4u) {
                const std::size_t values_per_chunk =
                    std::max<std::size_t>(1, target_bytes / sizeof(std::uint32_t));
                std::vector<std::uint32_t> values;
                values.reserve(values_per_chunk);
                sparse_it = graph_node_string_names.begin();
                for (std::size_t base = 0; base < graph_nodes;) {
                    const std::size_t n = std::min<std::size_t>(
                        values_per_chunk,
                        static_cast<std::size_t>(graph_nodes) - base);
                    values.clear();
                    for (std::size_t i = 0; i < n; ++i) {
                        std::uint64_t numeric = 0;
                        const bool numeric_name =
                            source_numeric(static_cast<std::uint32_t>(base + i), numeric);
                        if (!numeric_name && node_name_encoding == 3u) {
                            throw std::runtime_error(
                                "SPQR compact index graph node name remap changed during write");
                        }
                        values.push_back(static_cast<std::uint32_t>(
                            numeric_name ? numeric : 0u));
                    }
                    write_raw_bytes(out, values.data(),
                                    values.size() * sizeof(std::uint32_t));
                    base += n;
                }
            } else {
                const std::size_t values_per_chunk =
                    std::max<std::size_t>(1, target_bytes / sizeof(std::uint64_t));
                std::vector<std::uint64_t> values;
                values.reserve(values_per_chunk);
                sparse_it = graph_node_string_names.begin();
                for (std::size_t base = 0; base < graph_nodes;) {
                    const std::size_t n = std::min<std::size_t>(
                        values_per_chunk,
                        static_cast<std::size_t>(graph_nodes) - base);
                    values.clear();
                    for (std::size_t i = 0; i < n; ++i) {
                        std::uint64_t numeric = 0;
                        const bool numeric_name =
                            source_numeric(static_cast<std::uint32_t>(base + i), numeric);
                        if (!numeric_name && node_name_encoding == 1u) {
                            throw std::runtime_error(
                                "SPQR compact index graph node name remap changed during write");
                        }
                        values.push_back(numeric_name ? numeric : 0u);
                    }
                    write_raw_bytes(out, values.data(),
                                    values.size() * sizeof(std::uint64_t));
                    base += n;
                }
            }

            if (node_name_encoding == 2u || node_name_encoding == 4u) {
                detail::write_scalar<std::uint64_t>(out, string_name_count);
                sparse_it = graph_node_string_names.begin();
                for (std::uint32_t compact = 0; compact < graph_nodes; ++compact) {
                    std::uint64_t numeric = 0;
                    if (source_numeric(compact, numeric)) continue;
                    detail::write_scalar<std::uint32_t>(out, compact);
                    const std::string value = source_string(compact);
                    detail::write_string(out, value);
                }
            }
        } else {
            sparse_it = graph_node_string_names.begin();
            for (std::uint32_t compact = 0; compact < graph_nodes; ++compact) {
                const std::string value = source_string(compact);
                detail::write_string(out, value);
            }
        }
    }

    void read_string_graph_node_names(std::istream& in,
                                      const char* count_what,
                                      const char* name_what)
    {
        const std::uint32_t count = read_count(in, count_what);
        strings.reserve(count);
        node_names.reserve(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            strings.push_back(detail::read_string(in, name_what));
            node_names.push_back(i);
        }
    }

    void read_graph_node_names_payload(
        std::istream& in,
        const std::string& path)
    {
        const std::uint8_t node_name_encoding =
            detail::read_scalar<std::uint8_t>(in, "columnar graph node name encoding");
        if (node_name_encoding == 5u || node_name_encoding == 6u) {
            const std::uint32_t graph_node_count =
                read_count(in, "columnar dense graph node names");
            graph_node_numeric_name_base =
                detail::read_scalar<std::uint64_t>(
                    in, "columnar dense graph node name base");
            compact_graph_node_count = graph_node_count;
            graph_node_numeric_names_dense = true;
            if (graph_node_count != 0 &&
                graph_node_numeric_name_base >
                    std::numeric_limits<std::uint64_t>::max() -
                        static_cast<std::uint64_t>(graph_node_count - 1u)) {
                throw std::runtime_error(
                    "columnar dense SPQR graph node name range overflows in " + path);
            }
            if (node_name_encoding == 6u) {
                const std::uint32_t sparse_count =
                    read_count(in, "columnar dense mixed graph node string names");
                graph_node_string_names.reserve(sparse_count);
                std::uint32_t previous = invalid_id;
                for (std::uint32_t i = 0; i < sparse_count; ++i) {
                    const std::uint32_t compact =
                        detail::read_scalar<std::uint32_t>(
                            in, "columnar dense mixed graph node string id");
                    if (compact >= graph_node_count) {
                        throw std::runtime_error(
                            "columnar dense SPQR graph node string id out of range in " +
                            path);
                    }
                    if (i != 0 && compact <= previous) {
                        throw std::runtime_error(
                            "columnar dense SPQR graph node string ids are not strictly sorted in " +
                            path);
                    }
                    previous = compact;
                    graph_node_string_names.emplace_back(
                        compact,
                        detail::read_string(
                            in, "columnar dense mixed graph node string name"));
                }
            }
        } else if (node_name_encoding == 1u) {
            read_u64_vector(in, graph_node_numeric_names,
                            "columnar numeric graph node names");
            compact_graph_node_count = graph_index::require_spqr_count_size(
                graph_node_numeric_names.size(), "columnar numeric graph node count");
        } else if (node_name_encoding == 3u) {
            read_u32_vector(in, graph_node_numeric_names32,
                            "columnar u32 graph node names");
            compact_graph_node_count = graph_index::require_spqr_count_size(
                graph_node_numeric_names32.size(), "columnar u32 graph node count");
        } else if (node_name_encoding == 2u || node_name_encoding == 4u) {
            if (node_name_encoding == 4u) {
                read_u32_vector(in, graph_node_numeric_names32,
                                "columnar mixed u32 graph node numeric names");
            } else {
                read_u64_vector(in, graph_node_numeric_names,
                                "columnar mixed graph node numeric names");
            }
            const std::uint32_t graph_node_count =
                graph_node_numeric_names32.empty()
                    ? graph_index::require_spqr_count_size(
                          graph_node_numeric_names.size(),
                          "columnar mixed graph node count")
                    : graph_index::require_spqr_count_size(
                          graph_node_numeric_names32.size(),
                          "columnar mixed graph node count");
            compact_graph_node_count = graph_node_count;
            const std::uint32_t sparse_count =
                read_count(in, "columnar mixed graph node string names");
            graph_node_string_names.reserve(sparse_count);
            std::uint32_t previous = invalid_id;
            for (std::uint32_t i = 0; i < sparse_count; ++i) {
                const std::uint32_t compact =
                    detail::read_scalar<std::uint32_t>(
                        in, "columnar mixed graph node string id");
                if (compact >= graph_node_count) {
                    throw std::runtime_error(
                        "columnar SPQR graph node string id out of range in " + path);
                }
                if (i != 0 && compact <= previous) {
                    throw std::runtime_error(
                        "columnar SPQR graph node string ids are not strictly sorted in " +
                        path);
                }
                previous = compact;
                graph_node_string_names.emplace_back(
                    compact,
                    detail::read_string(in, "columnar mixed graph node string name"));
            }
        } else if (node_name_encoding == 0u) {
            read_string_graph_node_names(
                in, "columnar graph node names", "columnar graph node name");
        } else {
            throw std::runtime_error(
                "unsupported columnar graph node name encoding in " + path);
        }
    }

    void write_graph_node_degrees_payload(
        std::ostream& out,
        const NodeRemapTable& node_remap) const
    {
        write_column_payload_header(out, section_id_graph_node_degrees);
        const std::uint32_t graph_nodes = graph_node_count();
        detail::write_scalar<std::uint64_t>(out, graph_nodes);
        const auto degrees = build_compact_graph_node_degree_vectors(node_remap);
        write_graph_node_degree_table(out, degrees.first);
        write_graph_node_degree_table(out, degrees.second);
    }

    void read_graph_node_degrees_payload(
        std::istream& in,
        std::uint32_t expected_graph_nodes,
        const std::string& path)
    {
        const std::uint32_t graph_nodes =
            read_count(in, "columnar graph node degree count");
        if (graph_nodes != expected_graph_nodes) {
            throw std::runtime_error(
                "columnar SPQR graph node degree count mismatch in " + path);
        }
        read_graph_node_degree_tables(in, graph_nodes, path);
    }

    void write_macro_tree_cache_payload(std::ostream& out) const
    {
        write_column_payload_header(out, section_id_macro_tree_cache);
        detail::write_scalar<std::uint32_t>(out, 6u);
        detail::write_scalar<std::uint32_t>(out, 0u);
        write_trivial_vector64(out, macro_cache_blocks);
        write_trivial_vector64(out, macro_cache_nodes);
        write_trivial_vector64(out, macro_cache_children);
        write_trivial_vector64(out, macro_cache_core_edges);
        write_trivial_vector64(out, macro_cache_core_nodes);
        write_trivial_vector64(out, macro_cache_input_endpoints);
        write_trivial_vector64(out, macro_cache_input_edge_ids);
        write_trivial_vector64(out, macro_cache_core_node_inv);
        write_trivial_vector64(out, macro_cache_core_tree_node_types);
        write_trivial_vector64(out, macro_cache_core_tree_parents);
        write_trivial_vector64(out, macro_cache_core_tree_child_offsets);
        write_trivial_vector64(out, macro_cache_core_tree_children);
        write_trivial_vector64(out, macro_cache_core_tree_skeleton_offsets);
        write_trivial_vector64(out, macro_cache_core_tree_skeleton_edges);
        write_trivial_vector64(out, macro_cache_core_tree_mapping_offsets);
        write_trivial_vector64(out, macro_cache_core_tree_mapping);
        write_trivial_vector64(out, macro_cache_core_tree_skeleton_num_nodes);
    }

    void read_macro_tree_cache_payload(std::istream& in,
                                       const std::string& path)
    {
        const std::uint32_t version =
            detail::read_scalar<std::uint32_t>(in, "SPQR macro-tree cache version");
        if (version != 6u) {
            throw std::runtime_error(
                "unsupported SPQR macro-tree cache version in " + path);
        }
        (void)detail::read_scalar<std::uint32_t>(
            in, "SPQR macro-tree cache flags");
        read_trivial_vector64(in, macro_cache_blocks,
                              "SPQR macro-tree cache block records");
        read_trivial_vector64(in, macro_cache_nodes,
                              "SPQR macro-tree cache macro nodes");
        read_trivial_vector64(in, macro_cache_children,
                              "SPQR macro-tree cache child refs");
        read_trivial_vector64(in, macro_cache_core_edges,
                              "SPQR macro-tree cache core edges");
        read_trivial_vector64(in, macro_cache_core_nodes,
                              "SPQR macro-tree cache core nodes");
        read_trivial_vector64(in, macro_cache_input_endpoints,
                              "SPQR macro-tree cache input endpoints");
        read_trivial_vector64(in, macro_cache_input_edge_ids,
                              "SPQR macro-tree cache input edge ids");
        read_trivial_vector64(in, macro_cache_core_node_inv,
                              "SPQR macro-tree cache core node inverse map");
        read_trivial_vector64(in, macro_cache_core_tree_node_types,
                              "SPQR macro-tree cache core tree node types");
        read_trivial_vector64(in, macro_cache_core_tree_parents,
                              "SPQR macro-tree cache core tree parents");
        read_trivial_vector64(in, macro_cache_core_tree_child_offsets,
                              "SPQR macro-tree cache core tree child offsets");
        read_trivial_vector64(in, macro_cache_core_tree_children,
                              "SPQR macro-tree cache core tree children");
        read_trivial_vector64(in, macro_cache_core_tree_skeleton_offsets,
                              "SPQR macro-tree cache core tree skeleton offsets");
        read_trivial_vector64(in, macro_cache_core_tree_skeleton_edges,
                              "SPQR macro-tree cache core tree skeleton edges");
        read_trivial_vector64(in, macro_cache_core_tree_mapping_offsets,
                              "SPQR macro-tree cache core tree mapping offsets");
        read_trivial_vector64(in, macro_cache_core_tree_mapping,
                              "SPQR macro-tree cache core tree mapping");
        read_trivial_vector64(in, macro_cache_core_tree_skeleton_num_nodes,
                              "SPQR macro-tree cache core tree skeleton node counts");

        auto require_range = [&](std::uint64_t begin,
                                 std::uint64_t end,
                                 std::uint64_t size,
                                 const char* what) {
            if (begin > end || end > size) {
                throw std::runtime_error(
                    std::string("SPQR macro-tree cache ") + what +
                    " range is out of bounds in " + path);
            }
        };
        for (const auto& rec : macro_cache_blocks) {
            require_range(rec.macro_begin, rec.macro_end,
                          macro_cache_nodes.size(), "macro-node");
            require_range(rec.child_begin, rec.child_end,
                          macro_cache_children.size(), "child-ref");
            require_range(rec.core_edge_begin, rec.core_edge_end,
                          macro_cache_core_edges.size(), "core-edge");
            require_range(rec.core_node_begin, rec.core_node_end,
                          macro_cache_core_nodes.size(), "core-node");
            require_range(rec.input_endpoint_begin, rec.input_endpoint_end,
                          macro_cache_input_endpoints.size(), "input-endpoint");
            require_range(rec.core_node_inv_begin, rec.core_node_inv_end,
                          macro_cache_core_node_inv.size(), "core-node inverse");
            require_range(rec.core_tree_node_type_begin, rec.core_tree_node_type_end,
                          macro_cache_core_tree_node_types.size(), "core tree type");
            require_range(rec.core_tree_parent_begin, rec.core_tree_parent_end,
                          macro_cache_core_tree_parents.size(), "core tree parent");
            require_range(rec.core_tree_child_offset_begin,
                          rec.core_tree_child_offset_end,
                          macro_cache_core_tree_child_offsets.size(),
                          "core tree child-offset");
            require_range(rec.core_tree_child_begin, rec.core_tree_child_end,
                          macro_cache_core_tree_children.size(),
                          "core tree child");
            require_range(rec.core_tree_skeleton_offset_begin,
                          rec.core_tree_skeleton_offset_end,
                          macro_cache_core_tree_skeleton_offsets.size(),
                          "core tree skeleton-offset");
            require_range(rec.core_tree_skeleton_edge_begin,
                          rec.core_tree_skeleton_edge_end,
                          macro_cache_core_tree_skeleton_edges.size(),
                          "core tree skeleton-edge");
            require_range(rec.core_tree_mapping_offset_begin,
                          rec.core_tree_mapping_offset_end,
                          macro_cache_core_tree_mapping_offsets.size(),
                          "core tree mapping-offset");
            require_range(rec.core_tree_mapping_begin,
                          rec.core_tree_mapping_end,
                          macro_cache_core_tree_mapping.size(),
                          "core tree mapping");
            require_range(rec.core_tree_skeleton_num_nodes_begin,
                          rec.core_tree_skeleton_num_nodes_end,
                          macro_cache_core_tree_skeleton_num_nodes.size(),
                          "core tree skeleton-node-count");
            if (((rec.input_endpoint_end - rec.input_endpoint_begin) & 1u) != 0u) {
                throw std::runtime_error(
                    "SPQR macro-tree cache input endpoint count is odd in " + path);
            }
            if (!macro_cache_input_edge_ids.empty()) {
                if ((rec.input_endpoint_begin & 1u) != 0u ||
                    (rec.input_endpoint_end & 1u) != 0u) {
                    throw std::runtime_error(
                        "SPQR macro-tree cache input endpoint offset is odd in " +
                        path);
                }
                require_range(rec.input_endpoint_begin / 2u,
                              rec.input_endpoint_end / 2u,
                              macro_cache_input_edge_ids.size(),
                              "input-edge id");
            }
            if (rec.core_tree_node_count != 0u) {
                const std::uint64_t n = rec.core_tree_node_count;
                if (rec.core_tree_node_type_end - rec.core_tree_node_type_begin != n ||
                    rec.core_tree_parent_end - rec.core_tree_parent_begin != n ||
                    rec.core_tree_child_offset_end -
                        rec.core_tree_child_offset_begin != n + 1u ||
                    rec.core_tree_skeleton_offset_end -
                        rec.core_tree_skeleton_offset_begin != n + 1u ||
                    rec.core_tree_mapping_offset_end -
                        rec.core_tree_mapping_offset_begin != n + 1u ||
                    rec.core_tree_skeleton_num_nodes_end -
                        rec.core_tree_skeleton_num_nodes_begin != n) {
                    throw std::runtime_error(
                        "SPQR macro-tree cache core tree column sizes mismatch in " +
                        path);
                }
            }
        }
    }

    void write_columnar_sectioned_runtime_binary(std::ostream& out) const
    {
        // Some commands do not need every section.
        detail::write_scalar<std::uint32_t>(out, sectioned_runtime_magic);
        detail::write_scalar<std::uint32_t>(out, 2u);
        detail::write_scalar<std::uint64_t>(out, 0u);
        const std::uint64_t table_offset_field =
            tellp_u64(out, "columnar sectioned SPQR header");
        detail::write_scalar<std::uint64_t>(out, 0u);

        const SectionCodec codec = selected_section_codec();
        std::vector<SectionEntryDisk> sections;
        std::vector<SectionChunkDirectoryRecord> chunk_directory;
        auto write_column_with_codec = [&](std::uint32_t id,
                                           const char* what,
                                           bool prefer_raw,
                                           auto&& write_payload) {
            const SectionCodec section_codec =
                selected_columnar_section_codec(codec, prefer_raw);
            sections.push_back(write_section_payload_with_codec(
                out,
                id,
                what,
                section_codec,
                std::forward<decltype(write_payload)>(write_payload),
                &chunk_directory));
        };
        auto write_column = [&](std::uint32_t id,
                                const char* what,
                                auto&& write_payload) {
            write_column_with_codec(
                id,
                what,
                false,
                std::forward<decltype(write_payload)>(write_payload));
        };

        const bool external_graph_edge_tables = !graph_edges.empty();
        NodeRemapTable shared_node_remap = build_node_remap_table();

        write_column_with_codec(
            section_id_runtime_header,
            "columnar SPQR runtime header",
            true,
            [&](std::ostream& section_out) {
                write_runtime_header_payload(
                    section_out, external_graph_edge_tables);
            });
        NumericLookupWritePlan numeric_lookup_plan;
        const bool has_numeric_lookup_plan =
            build_graph_node_numeric_lookup_write_plan(numeric_lookup_plan);
        const bool omit_graph_node_names_section = has_numeric_lookup_plan;
        if (!omit_graph_node_names_section) {
            write_column(
                section_id_graph_node_names,
                "columnar SPQR graph-node name section",
                [&](std::ostream& section_out) {
                    write_graph_node_names_payload(section_out);
                });
        }
        if (has_numeric_lookup_plan) {
            auto numeric_lookup_plan_ptr =
                std::make_shared<NumericLookupWritePlan>(
                    std::move(numeric_lookup_plan));
            write_column(
                section_id_graph_node_numeric_lookup,
                "columnar SPQR graph-node numeric lookup section",
                [numeric_lookup_plan_ptr](std::ostream& section_out) {
                    write_graph_node_numeric_lookup_payload(
                        section_out, *numeric_lookup_plan_ptr);
                });
        }
        write_column(
            section_id_graph_node_degrees,
            "columnar SPQR graph-node degree section",
            [&](std::ostream& section_out) {
                write_graph_node_degrees_payload(
                    section_out, shared_node_remap);
            });
        write_column(
            section_id_block_nodes,
            "columnar SPQR block-node section",
            [&](std::ostream& section_out) {
                write_column_payload_header(
                    section_out, section_id_block_nodes);
                write_mapped_u32_vector(
                    section_out, block_nodes, shared_node_remap, "block nodes");
            });
        write_column(
            section_id_blocks,
            "columnar SPQR block-record section",
            [&](std::ostream& section_out) {
                write_column_payload_header(section_out, section_id_blocks);
                write_compact_records<SlimBlockRecord>(
                    section_out, block_count(),
                    [&](std::uint32_t i) {
                        const auto block = block_at(i);
                        const auto tree_edge_counts = block_tree_edge_counts(i);
                        return SlimBlockRecord{
                            block.node_begin,
                            block.node_end,
                            block.spqr_begin,
                            block.spqr_end,
                            block.real_edge_begin,
                            block.real_edge_end,
                            tree_edge_counts.first,
                            tree_edge_counts.second
                        };
                    });
            });
        write_column(
            section_id_tree_node_original_nodes,
            "columnar SPQR tree-node original-node section",
            [&](std::ostream& section_out) {
                write_column_payload_header(
                    section_out, section_id_tree_node_original_nodes);
                write_mapped_u32_vector(section_out,
                                        tree_node_original_nodes,
                                        shared_node_remap,
                                        "tree-node original nodes");
            });
        write_column(
            section_id_tree_node_node_begins,
            "columnar SPQR tree-node skeleton-node begin column",
            [&](std::ostream& section_out) {
                write_column_payload_header(
                    section_out, section_id_tree_node_node_begins);
                write_tree_node_node_begins_column(section_out);
            });
        write_column(
            section_id_tree_node_real_edge_begins,
            "columnar SPQR tree-node real-edge begin column",
            [&](std::ostream& section_out) {
                write_column_payload_header(
                    section_out, section_id_tree_node_real_edge_begins);
                write_tree_node_real_edge_begins_column(section_out);
            });
        write_column(
            section_id_tree_node_virtual_edge_begins,
            "columnar SPQR tree-node virtual-edge begin column",
            [&](std::ostream& section_out) {
                write_column_payload_header(
                    section_out, section_id_tree_node_virtual_edge_begins);
                write_tree_node_virtual_edge_begins_column(section_out);
            });
        write_column(
            section_id_tree_node_node_counts,
            "columnar SPQR tree-node skeleton-node count column",
            [&](std::ostream& section_out) {
                write_column_payload_header(
                    section_out, section_id_tree_node_node_counts);
                write_tree_node_node_counts_column(section_out);
            });
        write_column(
            section_id_tree_node_real_edge_counts,
            "columnar SPQR tree-node real-edge count column",
            [&](std::ostream& section_out) {
                write_column_payload_header(
                    section_out, section_id_tree_node_real_edge_counts);
                write_tree_node_real_edge_counts_column(section_out);
            });
        write_column(
            section_id_tree_node_virtual_edge_counts,
            "columnar SPQR tree-node virtual-edge count column",
            [&](std::ostream& section_out) {
                write_column_payload_header(
                    section_out, section_id_tree_node_virtual_edge_counts);
                write_tree_node_virtual_edge_counts_column(section_out);
            });
        write_column(
            section_id_tree_node_types,
            "columnar SPQR tree-node type section",
            [&](std::ostream& section_out) {
                write_column_payload_header(
                    section_out, section_id_tree_node_types);
                const std::uint32_t count = tree_node_count();
                detail::write_scalar<std::uint64_t>(section_out, count);
                constexpr std::size_t target_bytes = 4ull * 1024ull * 1024ull;
                std::vector<std::uint8_t> buffer;
                buffer.reserve(std::min<std::size_t>(target_bytes, count));
                for (std::uint32_t base = 0; base < count;) {
                    const std::size_t n = std::min<std::size_t>(
                        target_bytes, static_cast<std::size_t>(count - base));
                    buffer.clear();
                    for (std::size_t i = 0; i < n; ++i) {
                        const std::uint32_t idx =
                            base + static_cast<std::uint32_t>(i);
                        buffer.push_back(tree_node_at(idx).type);
                    }
                    write_raw_bytes(section_out,
                                    buffer.data(),
                                    buffer.size() * sizeof(std::uint8_t));
                    base += static_cast<std::uint32_t>(n);
                }
            });
        write_column(
            section_id_virtual_edges,
            "columnar SPQR virtual-edge section",
            [&](std::ostream& section_out) {
                write_column_payload_header(
                    section_out, section_id_virtual_edges);
                write_compact_records<SlimVirtualEdgeRecord>(
                    section_out, virtual_edge_count(),
                    [&](std::uint32_t i) {
                        const auto ve = virtual_edge_at(i);
                        return SlimVirtualEdgeRecord{
                            ve.node1_index,
                            ve.node2_index,
                            remap_graph_node_id(shared_node_remap,
                                                ve.pole1,
                                                "virtual-edge pole1"),
                            remap_graph_node_id(shared_node_remap,
                                                ve.pole2,
                                                "virtual-edge pole2")
                        };
                    });
            });
        write_column(
            section_id_real_edges,
            "columnar SPQR real-edge endpoint section",
            [&](std::ostream& section_out) {
                write_column_payload_header(section_out, section_id_real_edges);
                if (!real_edge_endpoints.empty()) {
                    write_remapped_edge_records(section_out,
                                                real_edge_endpoints,
                                                shared_node_remap,
                                                "real-edge endpoints");
                } else {
                    const std::uint32_t real_edge_total = real_edge_count();
                    detail::write_scalar<std::uint64_t>(
                        section_out, real_edge_total);
                    constexpr std::size_t target_bytes = 4ull * 1024ull * 1024ull;
                    const std::size_t records_per_chunk =
                        std::max<std::size_t>(
                            1, target_bytes / sizeof(GraphEdgeRecord));
                    std::vector<GraphEdgeRecord> buffer;
                    buffer.reserve(records_per_chunk);
                    for (std::size_t base = 0; base < real_edge_total;) {
                        const std::size_t n = std::min<std::size_t>(
                            records_per_chunk,
                            static_cast<std::size_t>(real_edge_total) - base);
                        buffer.clear();
                        for (std::size_t i = 0; i < n; ++i) {
                            const auto edge = real_edge_endpoint_at(
                                static_cast<std::uint32_t>(base + i));
                            buffer.push_back(GraphEdgeRecord{
                                remap_graph_node_id(shared_node_remap,
                                                    edge.src,
                                                    "real-edge src"),
                                remap_graph_node_id(shared_node_remap,
                                                    edge.dst,
                                                    "real-edge dst")
                            });
                        }
                        write_raw_bytes(section_out,
                                        buffer.data(),
                                        buffer.size() * sizeof(GraphEdgeRecord));
                        base += n;
                    }
                }
            });
        write_column_with_codec(
            section_id_block_to_tree_nodes,
            "columnar SPQR block-to-tree-node section",
            block_to_tree_nodes_identity || block_to_tree_nodes.empty(),
            [&](std::ostream& section_out) {
                write_column_payload_header(
                    section_out, section_id_block_to_tree_nodes);
                write_identity_or_u32_vector(section_out,
                                             block_to_tree_nodes,
                                             tree_node_count(),
                                             block_to_tree_nodes_identity);
            });
        if (!block_input_edge_offsets.empty() || !block_input_edges.empty()) {
            write_column(
                section_id_block_input_edges,
                "columnar SPQR block input-edge order section",
                [&](std::ostream& section_out) {
                    write_column_payload_header(
                        section_out, section_id_block_input_edges);
                    write_u32_vector(section_out, block_input_edge_offsets);
                    write_u32_vector(section_out, block_input_edges);
                });
        }
        if (!block_input_edge_orientations.empty()) {
            write_column(
                section_id_block_input_edge_orientations,
                "columnar SPQR block input-edge orientation section",
                [&](std::ostream& section_out) {
                    write_column_payload_header(
                        section_out,
                        section_id_block_input_edge_orientations);
                    write_u8_vector(section_out, block_input_edge_orientations);
                });
        }
        if (!block_input_edge_local_endpoints.empty()) {
            if (block_input_edge_local_endpoints.size() !=
                block_input_edges.size()) {
                throw std::runtime_error(
                    "SPQR block input-edge local endpoint count mismatch");
            }
            write_column(
                section_id_block_input_edge_local_endpoints,
                "columnar SPQR block input-edge local-endpoint section",
                [&](std::ostream& section_out) {
                    write_column_payload_header(
                        section_out,
                        section_id_block_input_edge_local_endpoints);
                    write_trivial_vector64(section_out,
                                           block_input_edge_local_endpoints);
                });
        }
        write_column(
            section_id_tree_node_real_edges,
            "columnar SPQR tree-node real-edge section",
            [&](std::ostream& section_out) {
                write_column_payload_header(
                    section_out, section_id_tree_node_real_edges);
                write_tree_node_real_edge_index(section_out);
            });
        write_column(
            section_id_tree_node_virtual_edges,
            "columnar SPQR tree-node virtual-edge section",
            [&](std::ostream& section_out) {
                write_column_payload_header(
                    section_out, section_id_tree_node_virtual_edges);
                write_u32_vector(section_out, tree_node_virtual_edges);
            });
        write_column(
            section_id_real_edge_types,
            "columnar SPQR real-edge type section",
            [&](std::ostream& section_out) {
                write_column_payload_header(
                    section_out, section_id_real_edge_types);
                const std::uint64_t bf_count = bubble_edge_type_count_hint != 0
                    ? bubble_edge_type_count_hint
                    : static_cast<std::uint64_t>(bubble_edge_types.size());
                detail::write_scalar<std::uint64_t>(section_out, bf_count);
                write_u8_vector(section_out, real_edge_type_pairs);
            });
        write_column_with_codec(
            section_id_block_hashes,
            "columnar SPQR block-hash section",
            true,
            [&](std::ostream& section_out) {
                write_column_payload_header(section_out, section_id_block_hashes);
                write_compact_block_hashes(section_out, shared_node_remap);
            });
        if (!macro_cache_blocks.empty()) {
            write_column(
                section_id_macro_tree_cache,
                "columnar SPQR macro-tree/core-SPQR cache section",
                [&](std::ostream& section_out) {
                    write_macro_tree_cache_payload(section_out);
                });
        }

        if (external_graph_edge_tables) {
            sections.push_back(write_section_payload_with_codec(
                out,
                section_id_graph_edge_tables,
                "columnar SPQR graph-edge table section",
                selected_columnar_section_codec(codec, false),
                [&](std::ostream& section_out) {
                    write_graph_edge_tables_binary(
                        section_out, &shared_node_remap);
                },
                &chunk_directory));
        }

        if (external_graph_edge_tables) {
            sections.push_back(write_section_payload_with_codec(
                out,
                section_id_graph_components,
                "columnar SPQR graph-component partition section",
                selected_columnar_section_codec(codec, false),
                [&](std::ostream& section_out) {
                    write_graph_components_binary(section_out, shared_node_remap);
                },
                &chunk_directory));
        }

        // Optional haplotype paths and steps are stored in their own section.
        if (!haplotype_paths.empty() && !haplotype_steps.empty()) {
            sections.push_back(write_section_payload_with_codec(
                out,
                section_id_haplotypes,
                "columnar SPQR haplotype path section",
                selected_columnar_section_codec(codec, false),
                [&](std::ostream& section_out) {
                    write_haplotypes_binary(section_out, shared_node_remap);
                },
                &chunk_directory));
        }

        if (!chunk_directory.empty()) {
            sections.push_back(write_section_payload_with_codec(
                out,
                section_id_zstd_chunk_directory,
                "SPQRI zstd chunk directory section",
                SectionCodec::Raw,
                [&](std::ostream& section_out) {
                    write_zstd_chunk_directory_payload(
                        section_out, chunk_directory);
                }));
        }

        const std::uint64_t table_offset =
            tellp_u64(out, "columnar sectioned SPQR table");
        detail::write_scalar<std::uint32_t>(out, section_table_magic);
        detail::write_scalar<std::uint32_t>(out, 1u);
        detail::write_scalar<std::uint64_t>(out, sections.size());
        for (const auto& section : sections) write_section_entry(out, section);

        const std::uint64_t end_offset =
            tellp_u64(out, "columnar sectioned SPQR footer");
        seekp_u64(out, table_offset_field, "columnar SPQR table pointer");
        detail::write_scalar<std::uint64_t>(out, table_offset);
        seekp_u64(out, end_offset, "columnar sectioned SPQR end");
    }
