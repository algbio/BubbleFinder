    // Read the current .spqri sections. Many sections can load in parallel.
    //
    // The header is small. Once it is read, most of this can run together:
    //
    //     header
    //       |-- node names and degrees
    //       |-- SPQR arrays
    //       |-- graph edges, endpoint types and multi-edge flags
    //       |-- graph components, if requested
    //       `-- haplotype paths, if requested

    void read_columnar_runtime_binary(
        const std::string& path,
        const std::vector<SectionEntryDisk>& sections,
        const LoadOptions& options)
    {
        preload_zstd_chunk_directory(path, sections);

        ColumnarRuntimeHeaderDisk header;
        const SectionEntryDisk header_section =
            require_section(sections,
                            section_id_runtime_header,
                            path,
                            "the columnar runtime header");
        read_section_payload_from_file(
            path,
            header_section,
            "columnar SPQR runtime header",
            1,
            [&](std::istream& section_in) {
                read_column_payload_header(
                    section_in, section_id_runtime_header, path);
                header = read_runtime_header_payload(section_in);
            });

        const std::uint32_t expected_graph_nodes =
            graph_index::require_spqr_count(
                header.graph_node_count, "columnar SPQR graph node count");
        const std::uint32_t expected_components =
            graph_index::require_spqr_count(
                header.component_count, "columnar SPQR component count");
        const std::uint32_t expected_blocks =
            graph_index::require_spqr_count(
                header.block_count, "columnar SPQR block count");
        const std::uint32_t expected_tree_nodes =
            graph_index::require_spqr_count(
                header.tree_node_count, "columnar SPQR tree-node count");
        const std::uint32_t expected_virtual_edges =
            graph_index::require_spqr_count(
                header.virtual_edge_count, "columnar SPQR virtual-edge count");
        const std::uint32_t expected_real_edges =
            graph_index::require_spqr_count(
                header.real_edge_count, "columnar SPQR real-edge count");
        const std::uint32_t expected_graph_edges =
            graph_index::require_spqr_count(
                header.graph_edge_count, "columnar SPQR graph-edge count");
        const SectionLoadPlan plan = build_section_load_plan(options);
        const SectionEntryDisk* graph_node_names_section =
            find_section(sections, section_id_graph_node_names);
        const SectionEntryDisk* graph_node_numeric_lookup_section =
            find_section(sections, section_id_graph_node_numeric_lookup);
        const bool use_precomputed_graph_node_numeric_lookup =
            options.prefer_precomputed_graph_node_numeric_lookup &&
            plan.eager_graph_node_name_lookup &&
            graph_node_numeric_lookup_section != nullptr;
        const bool materialize_graph_node_names_from_numeric_lookup =
            !use_precomputed_graph_node_numeric_lookup &&
            graph_node_names_section == nullptr &&
            graph_node_numeric_lookup_section != nullptr;

        std::future<std::vector<char>> tree_node_real_edges_payload_future;
        if (plan.tree_node_real_edges) {
            const SectionEntryDisk tree_real_section =
                require_section(sections,
                                section_id_tree_node_real_edges,
                                path,
                                "the columnar tree-node real-edge section");
            tree_node_real_edges_payload_future = schedule_section_task(
                [path, tree_real_section]() {
                    return read_section_payload_bytes_from_file(
                        path,
                        tree_real_section,
                        "columnar SPQR tree-node real-edge section",
                        selected_zstd_chunk_prefetch(4));
                });
        }

        auto launch_tmp = [&](std::uint32_t id,
                              const char* what,
                              auto read_payload) {
            const SectionEntryDisk section =
                require_section(sections, id, path, what);
            return schedule_section_task(
                [path, section, id, what, read_payload]() mutable {
                    IndexedSpqrTree tmp;
                    read_section_payload_from_file(
                        path,
                        section,
                        what,
                        selected_zstd_chunk_prefetch(4),
                        [&](std::istream& section_in) {
                            read_column_payload_header(section_in, id, path);
                            read_payload(tmp, section_in);
                        });
                    return tmp;
                });
        };

        std::future<IndexedSpqrTree> node_names_future;
        if (!use_precomputed_graph_node_numeric_lookup &&
            !materialize_graph_node_names_from_numeric_lookup) {
            node_names_future = launch_tmp(
                section_id_graph_node_names,
                "the columnar graph-node name section",
                [path](IndexedSpqrTree& tmp, std::istream& section_in) {
                    tmp.read_graph_node_names_payload(section_in, path);
                });
        }
        std::future<IndexedSpqrTree> graph_node_numeric_lookup_future;
        if (use_precomputed_graph_node_numeric_lookup ||
            materialize_graph_node_names_from_numeric_lookup) {
            const SectionEntryDisk section = *graph_node_numeric_lookup_section;
            graph_node_numeric_lookup_future = schedule_section_task(
                [path, section, expected_graph_nodes]() {
                    IndexedSpqrTree tmp;
                    read_section_payload_from_file(
                        path,
                        section,
                        "the columnar graph-node numeric lookup section",
                        selected_zstd_chunk_prefetch(4),
                        [&](std::istream& section_in) {
                            read_column_payload_header(
                                section_in,
                                section_id_graph_node_numeric_lookup,
                                path);
                            tmp.read_graph_node_numeric_lookup_payload(
                                section_in, expected_graph_nodes, path);
                        });
                    return tmp;
                });
        }
        std::future<IndexedSpqrTree> degrees_future;
        if (plan.graph_node_degrees) {
            degrees_future = launch_tmp(
                section_id_graph_node_degrees,
                "the columnar graph-node degree section",
                [path, expected_graph_nodes](IndexedSpqrTree& tmp,
                                             std::istream& section_in) {
                    tmp.read_graph_node_degrees_payload(
                        section_in, expected_graph_nodes, path);
                });
        }

        const SectionEntryDisk* graph_edge_section =
            find_section(sections, section_id_graph_edge_tables);
        const bool skip_graph_edge_tables = !plan.graph_edge_tables;
        std::future<IndexedSpqrTree> graph_edge_future;
        if (!skip_graph_edge_tables &&
            expected_graph_edges != 0u &&
            graph_edge_section == nullptr) {
            throw std::runtime_error(
                "binary SPQR section table is missing the graph-edge table section in " +
                path);
        }
        if (!skip_graph_edge_tables && graph_edge_section != nullptr) {
            const SectionEntryDisk section = *graph_edge_section;
            graph_edge_future = schedule_section_task(
                [path, section]() {
                    return load_graph_edge_tables_section_from_file(path, section);
                });
        }

        std::future<IndexedSpqrTree> graph_components_future;
        const SectionEntryDisk* graph_components_section =
            find_section(sections, section_id_graph_components);
        if (plan.graph_components && graph_components_section != nullptr) {
            const SectionEntryDisk section = *graph_components_section;
            graph_components_future = schedule_section_task(
                [path, section, expected_graph_nodes, expected_graph_edges]() {
                    return load_graph_components_section_from_file(
                        path, section, expected_graph_nodes, expected_graph_edges);
                });
        }

        std::future<IndexedSpqrTree> haplotypes_future;
        const SectionEntryDisk* haplotypes_section =
            find_section(sections, section_id_haplotypes);
        if (plan.haplotypes && haplotypes_section != nullptr) {
            const SectionEntryDisk section = *haplotypes_section;
            haplotypes_future = schedule_section_task(
                [path, section, expected_graph_nodes]() {
                    return load_haplotypes_section_from_file(
                        path, section, expected_graph_nodes);
                });
        }

        auto load_requested_graph_view = [&]() {
            graph_edge_count_hint = expected_graph_edges;
            if (graph_edge_future.valid()) {
                IndexedSpqrTree graph_edge_tables = graph_edge_future.get();
                if (graph_edge_tables.graph_edges.size() != expected_graph_edges) {
                    throw std::runtime_error(
                        "columnar SPQR graph-edge count mismatch in " + path);
                }
                take_graph_edge_tables(graph_edge_tables);
            }
        };

        auto load_requested_graph_components = [&]() {
            if (graph_components_future.valid()) {
                IndexedSpqrTree graph_components_tmp =
                    graph_components_future.get();
                graph_components =
                    std::move(graph_components_tmp.graph_components);
                graph_component_nodes =
                    std::move(graph_components_tmp.graph_component_nodes);
                graph_component_edges =
                    std::move(graph_components_tmp.graph_component_edges);
                graph_component_nodes_identity =
                    graph_components_tmp.graph_component_nodes_identity;
                graph_component_edges_identity =
                    graph_components_tmp.graph_component_edges_identity;
            }
        };

        auto load_requested_haplotypes = [&]() {
            if (!haplotypes_future.valid()) return;
            IndexedSpqrTree haplotypes_tmp = haplotypes_future.get();
            std::vector<std::uint32_t> string_remap;
            string_remap.reserve(haplotypes_tmp.strings.size());
            for (const std::string& value : haplotypes_tmp.strings) {
                string_remap.push_back(intern(value));
            }
            auto remap_string = [&](std::uint32_t old) -> std::uint32_t {
                if (old == invalid_id) return invalid_id;
                if (old >= string_remap.size()) {
                    throw std::runtime_error(
                        "columnar SPQR haplotype metadata string out of range in " +
                        path);
                }
                return string_remap[old];
            };
            for (auto& rec : haplotypes_tmp.haplotype_paths) {
                rec.name = remap_string(rec.name);
                rec.sample = remap_string(rec.sample);
                rec.locus = remap_string(rec.locus);
            }
            haplotype_paths = std::move(haplotypes_tmp.haplotype_paths);
            haplotype_steps = std::move(haplotypes_tmp.haplotype_steps);
        };

        auto take_graph_node_names = [&](IndexedSpqrTree& tmp) {
            strings = std::move(tmp.strings);
            node_names = std::move(tmp.node_names);
            compact_graph_node_count = tmp.compact_graph_node_count;
            graph_node_numeric_names = std::move(tmp.graph_node_numeric_names);
            graph_node_numeric_names32 = std::move(tmp.graph_node_numeric_names32);
            graph_node_numeric_names_dense = tmp.graph_node_numeric_names_dense;
            graph_node_numeric_name_base = tmp.graph_node_numeric_name_base;
            graph_node_numeric_name_valid =
                std::move(tmp.graph_node_numeric_name_valid);
            graph_node_string_names = std::move(tmp.graph_node_string_names);
        };

        auto load_requested_graph_node_names = [&]() {
            if (node_names_future.valid()) {
                IndexedSpqrTree tmp = node_names_future.get();
                take_graph_node_names(tmp);
            } else {
                drop_graph_node_name_storage(expected_graph_nodes);
            }
            if (graph_node_count() != expected_graph_nodes) {
                throw std::runtime_error(
                    "columnar SPQR graph node count mismatch in " + path);
            }
        };

        auto load_requested_graph_degrees = [&]() {
            if (!degrees_future.valid()) return;
            IndexedSpqrTree tmp = degrees_future.get();
            graph_node_in_degrees = std::move(tmp.graph_node_in_degrees);
            graph_node_out_degrees = std::move(tmp.graph_node_out_degrees);
            graph_node_in_degrees8 = std::move(tmp.graph_node_in_degrees8);
            graph_node_out_degrees8 = std::move(tmp.graph_node_out_degrees8);
            graph_node_in_degree_overflow_nodes =
                std::move(tmp.graph_node_in_degree_overflow_nodes);
            graph_node_in_degree_overflow_values =
                std::move(tmp.graph_node_in_degree_overflow_values);
            graph_node_out_degree_overflow_nodes =
                std::move(tmp.graph_node_out_degree_overflow_nodes);
            graph_node_out_degree_overflow_values =
                std::move(tmp.graph_node_out_degree_overflow_values);
        };

        auto validate_requested_graph_view = [&]() {
            if (!graph_edge_type_pairs.empty() &&
                graph_edge_type_pairs.size() != graph_edges.size()) {
                throw std::runtime_error(
                    "columnar SPQR graph-edge type count mismatch in " + path);
            }
            if (!graph_edge_multi_flags.empty() &&
                graph_edge_multi_flags.size() != graph_edges.size()) {
                throw std::runtime_error(
                    "columnar SPQR graph-edge multi-link count mismatch in " + path);
            }
        };

        auto load_requested_graph_node_lookup = [&]() {
            if (!graph_node_numeric_lookup_future.valid()) return;
            IndexedSpqrTree tmp = graph_node_numeric_lookup_future.get();
            if (materialize_graph_node_names_from_numeric_lookup) {
                tmp.materialize_graph_node_names_from_numeric_lookup(
                    expected_graph_nodes, path);
                take_graph_node_names(tmp);
                if (graph_node_count() != expected_graph_nodes) {
                    throw std::runtime_error(
                        "columnar SPQR graph-node lookup count mismatch in " + path);
                }
            } else {
                graph_node_by_numeric_name = std::move(tmp.graph_node_by_numeric_name);
                graph_node_by_dense_numeric_name =
                    std::move(tmp.graph_node_by_dense_numeric_name);
                graph_node_numeric_lookup_dense = tmp.graph_node_numeric_lookup_dense;
                graph_node_numeric_lookup_dense_domain =
                    tmp.graph_node_numeric_lookup_dense_domain;
                graph_node_numeric_lookup_base = tmp.graph_node_numeric_lookup_base;
                graph_node_numeric_lookup_ready = tmp.graph_node_numeric_lookup_ready;
            }
        };

        if (options.graph_only) {
            load_requested_graph_node_names();
            load_requested_graph_degrees();

            component_count_hint = expected_components;
            cut_count_hint = header.cut_count;

            block_count_hint = expected_blocks;

            tree_node_count_hint = expected_tree_nodes;
            virtual_edge_count_hint = expected_virtual_edges;
            slim_record_tables = true;

            real_edge_count_hint = expected_real_edges;
            bubble_edge_type_count_hint = header.bubble_edge_type_count;

            load_requested_graph_view();
            load_requested_graph_components();
            validate_requested_graph_view();

            runtime_indices_ready = true;
            load_requested_graph_node_lookup();
            load_requested_haplotypes();
            apply_eager_load_plan_indexes(plan);
            return;
        }

        auto block_nodes_future = launch_tmp(
            section_id_block_nodes,
            "the columnar block-node section",
            [](IndexedSpqrTree& tmp, std::istream& section_in) {
                read_u32_vector(section_in, tmp.block_nodes, "columnar block nodes");
            });
        auto blocks_future = launch_tmp(
            section_id_blocks,
            "the columnar block-record section",
            [path, expected_blocks](IndexedSpqrTree& tmp,
                                    std::istream& section_in) {
                const std::uint32_t count =
                    read_count(section_in, "columnar block records");
                if (count != expected_blocks) {
                    throw std::runtime_error(
                        "columnar SPQR block count mismatch in " + path);
                }
                tmp.slim_blocks.resize(count);
                if (!tmp.slim_blocks.empty()) {
                    read_raw_bytes(section_in,
                                   tmp.slim_blocks.data(),
                                   tmp.slim_blocks.size() * sizeof(SlimBlockRecord),
                                   "columnar block records payload");
                }
                tmp.block_tree_edge_counts_ready = true;
            });
        std::future<IndexedSpqrTree> tree_node_original_nodes_future;
        if (plan.tree_node_original_nodes) {
            tree_node_original_nodes_future = launch_tmp(
                section_id_tree_node_original_nodes,
                "the columnar tree-node original-node section",
                [](IndexedSpqrTree& tmp, std::istream& section_in) {
                    read_u32_vector(section_in,
                                    tmp.tree_node_original_nodes,
                                    "columnar tree-node original nodes");
                });
        }
        std::future<IndexedSpqrTree> tree_node_node_begins_future;
        std::future<IndexedSpqrTree> tree_node_real_edge_begins_future;
        std::future<IndexedSpqrTree> tree_node_virtual_edge_begins_future;
        std::future<IndexedSpqrTree> tree_node_node_counts_future;
        std::future<IndexedSpqrTree> tree_node_real_edge_counts_future;
        std::future<IndexedSpqrTree> tree_node_virtual_edge_counts_future;
        if (plan.tree_nodes) {
            tree_node_node_begins_future = launch_tmp(
                section_id_tree_node_node_begins,
                "the columnar tree-node skeleton-node begin column",
                [path, expected_tree_nodes](IndexedSpqrTree& tmp,
                                            std::istream& section_in) {
                    read_u32_vector(section_in,
                                    tmp.compact_tree_node_node_begins,
                                    "columnar tree-node skeleton-node begins");
                    require_compact_tree_node_vector_size(
                        tmp.compact_tree_node_node_begins.size(),
                        expected_tree_nodes,
                        "skeleton-node begins",
                        path);
                });
            tree_node_real_edge_begins_future = launch_tmp(
                section_id_tree_node_real_edge_begins,
                "the columnar tree-node real-edge begin column",
                [path, expected_tree_nodes](IndexedSpqrTree& tmp,
                                            std::istream& section_in) {
                    read_u32_vector(section_in,
                                    tmp.compact_tree_node_real_edge_begins,
                                    "columnar tree-node real-edge begins");
                    require_compact_tree_node_vector_size(
                        tmp.compact_tree_node_real_edge_begins.size(),
                        expected_tree_nodes,
                        "real-edge begins",
                        path);
                });
            tree_node_virtual_edge_begins_future = launch_tmp(
                section_id_tree_node_virtual_edge_begins,
                "the columnar tree-node virtual-edge begin column",
                [path, expected_tree_nodes](IndexedSpqrTree& tmp,
                                            std::istream& section_in) {
                    read_u32_vector(section_in,
                                    tmp.compact_tree_node_virtual_edge_begins,
                                    "columnar tree-node virtual-edge begins");
                    require_compact_tree_node_vector_size(
                        tmp.compact_tree_node_virtual_edge_begins.size(),
                        expected_tree_nodes,
                        "virtual-edge begins",
                        path);
                });
            tree_node_node_counts_future = launch_tmp(
                section_id_tree_node_node_counts,
                "the columnar tree-node skeleton-node count column",
                [path, expected_tree_nodes](IndexedSpqrTree& tmp,
                                            std::istream& section_in) {
                    read_tree_node_count_column(
                        section_in,
                        expected_tree_nodes,
                        tmp.compact_tree_node_node_counts16,
                        tmp.compact_tree_node_node_count_overflow_nodes,
                        tmp.compact_tree_node_node_count_overflow_values,
                        "skeleton-node counts",
                        path);
                });
            tree_node_real_edge_counts_future = launch_tmp(
                section_id_tree_node_real_edge_counts,
                "the columnar tree-node real-edge count column",
                [path, expected_tree_nodes](IndexedSpqrTree& tmp,
                                            std::istream& section_in) {
                    read_tree_node_count_column(
                        section_in,
                        expected_tree_nodes,
                        tmp.compact_tree_node_real_edge_counts16,
                        tmp.compact_tree_node_real_edge_count_overflow_nodes,
                        tmp.compact_tree_node_real_edge_count_overflow_values,
                        "real-edge counts",
                        path);
                });
            tree_node_virtual_edge_counts_future = launch_tmp(
                section_id_tree_node_virtual_edge_counts,
                "the columnar tree-node virtual-edge count column",
                [path, expected_tree_nodes](IndexedSpqrTree& tmp,
                                            std::istream& section_in) {
                    read_tree_node_count_column(
                        section_in,
                        expected_tree_nodes,
                        tmp.compact_tree_node_virtual_edge_counts16,
                        tmp.compact_tree_node_virtual_edge_count_overflow_nodes,
                        tmp.compact_tree_node_virtual_edge_count_overflow_values,
                        "virtual-edge counts",
                        path);
                });
        }
        std::future<IndexedSpqrTree> tree_node_types_future;
        if (plan.tree_node_types) {
            tree_node_types_future = launch_tmp(
                section_id_tree_node_types,
                "the columnar tree-node type section",
                [path, expected_tree_nodes](IndexedSpqrTree& tmp,
                                            std::istream& section_in) {
                    read_u8_vector(section_in,
                                   tmp.slim_tree_node_types,
                                   "columnar tree-node type index");
                    if (tmp.slim_tree_node_types.size() != expected_tree_nodes) {
                        throw std::runtime_error(
                            "columnar SPQR tree-node type count mismatch in " + path);
                    }
                });
        }
        std::future<IndexedSpqrTree> virtual_edges_future;
        if (plan.virtual_edges) {
            virtual_edges_future = launch_tmp(
                section_id_virtual_edges,
                "the columnar virtual-edge section",
                [path, expected_virtual_edges](IndexedSpqrTree& tmp,
                                               std::istream& section_in) {
                    const std::uint32_t count =
                        read_count(section_in, "columnar virtual-edge records");
                    if (count != expected_virtual_edges) {
                        throw std::runtime_error(
                            "columnar SPQR virtual-edge count mismatch in " + path);
                    }
                    tmp.slim_virtual_edges.resize(count);
                    if (!tmp.slim_virtual_edges.empty()) {
                        read_raw_bytes(section_in,
                                       tmp.slim_virtual_edges.data(),
                                       tmp.slim_virtual_edges.size() *
                                           sizeof(SlimVirtualEdgeRecord),
                                       "columnar virtual-edge records payload");
                    }
                });
        }
        std::future<IndexedSpqrTree> real_edges_future;
        if (plan.real_edges) {
            real_edges_future = launch_tmp(
                section_id_real_edges,
                "the columnar real-edge endpoint section",
                [path, expected_real_edges](IndexedSpqrTree& tmp,
                                            std::istream& section_in) {
                    const std::uint32_t count =
                        read_count(section_in, "columnar real-edge records");
                    if (count != expected_real_edges) {
                        throw std::runtime_error(
                            "columnar SPQR real-edge count mismatch in " + path);
                    }
                    tmp.real_edge_endpoints.resize(count);
                    if (!tmp.real_edge_endpoints.empty()) {
                        read_raw_bytes(section_in,
                                       tmp.real_edge_endpoints.data(),
                                       tmp.real_edge_endpoints.size() *
                                           sizeof(GraphEdgeRecord),
                                       "columnar real-edge records payload");
                    }
                });
        }
        std::future<IndexedSpqrTree> block_to_tree_nodes_future;
        if (plan.block_to_tree_nodes) {
            block_to_tree_nodes_future = launch_tmp(
                section_id_block_to_tree_nodes,
                "the columnar block-to-tree-node section",
                [expected_tree_nodes](IndexedSpqrTree& tmp,
                                      std::istream& section_in) {
                    tmp.block_to_tree_nodes_identity =
                        read_identity_or_u32_vector(
                            section_in,
                            tmp.block_to_tree_nodes,
                            "columnar block-to-tree-node index",
                            expected_tree_nodes);
                });
        }
        std::future<IndexedSpqrTree> block_input_edges_future;
        const SectionEntryDisk* block_input_edges_section =
            find_section(sections, section_id_block_input_edges);
        if (plan.block_input_edges && block_input_edges_section != nullptr) {
            const SectionEntryDisk section = *block_input_edges_section;
            block_input_edges_future = schedule_section_task(
                [path, section, expected_blocks]() {
                    IndexedSpqrTree tmp;
                    read_section_payload_from_file(
                        path,
                        section,
                        "columnar SPQR block input-edge order section",
                        selected_zstd_chunk_prefetch(4),
                        [&](std::istream& section_in) {
                            read_column_payload_header(
                                section_in,
                                section_id_block_input_edges,
                                path);
                            read_u32_vector(
                                section_in,
                                tmp.block_input_edge_offsets,
                                "columnar block input-edge offsets");
                            read_u32_vector(
                                section_in,
                                tmp.block_input_edges,
                                "columnar block input-edge order");
                        });
                    if (!tmp.block_input_edge_offsets.empty() &&
                        tmp.block_input_edge_offsets.size() !=
                            static_cast<std::size_t>(expected_blocks) + 1u) {
                        throw std::runtime_error(
                            "columnar SPQR block input-edge offset count mismatch in " +
                            path);
                    }
                    std::uint32_t prev = 0;
                    for (std::uint32_t off : tmp.block_input_edge_offsets) {
                        if (off < prev ||
                            off > tmp.block_input_edges.size()) {
                            throw std::runtime_error(
                                "columnar SPQR block input-edge offsets are invalid in " +
                                path);
                        }
                        prev = off;
                    }
                    if (!tmp.block_input_edge_offsets.empty() &&
                        tmp.block_input_edge_offsets.back() !=
                            tmp.block_input_edges.size()) {
                        throw std::runtime_error(
                            "columnar SPQR block input-edge order size mismatch in " +
                            path);
                    }
                    return tmp;
                });
        }
        std::future<IndexedSpqrTree> block_input_orientations_future;
        const SectionEntryDisk* block_input_orientations_section =
            find_section(sections,
                         section_id_block_input_edge_orientations);
        if (plan.block_input_edges &&
            block_input_orientations_section != nullptr) {
            const SectionEntryDisk section = *block_input_orientations_section;
            block_input_orientations_future = schedule_section_task(
                [path, section]() {
                    IndexedSpqrTree tmp;
                    read_section_payload_from_file(
                        path,
                        section,
                        "columnar SPQR block input-edge orientation section",
                        selected_zstd_chunk_prefetch(4),
                        [&](std::istream& section_in) {
                            read_column_payload_header(
                                section_in,
                                section_id_block_input_edge_orientations,
                                path);
                            read_u8_vector(
                                section_in,
                                tmp.block_input_edge_orientations,
                                "columnar block input-edge orientations");
                        });
                    for (std::uint8_t orientation :
                         tmp.block_input_edge_orientations) {
                        if (orientation > 1u) {
                            throw std::runtime_error(
                                "columnar SPQR block input-edge orientation "
                                "value is invalid in " +
                                path);
                        }
                    }
                    return tmp;
                });
        }
        std::future<IndexedSpqrTree> block_input_local_endpoints_future;
        const SectionEntryDisk* block_input_local_endpoints_section =
            find_section(sections,
                         section_id_block_input_edge_local_endpoints);
        if (plan.block_input_edges &&
            block_input_local_endpoints_section != nullptr) {
            const SectionEntryDisk section = *block_input_local_endpoints_section;
            block_input_local_endpoints_future = schedule_section_task(
                [path, section]() {
                    IndexedSpqrTree tmp;
                    read_section_payload_from_file(
                        path,
                        section,
                        "columnar SPQR block input-edge local-endpoint section",
                        selected_zstd_chunk_prefetch(4),
                        [&](std::istream& section_in) {
                            read_column_payload_header(
                                section_in,
                                section_id_block_input_edge_local_endpoints,
                                path);
                            read_trivial_vector64<GraphEdgeRecord>(
                                section_in,
                                tmp.block_input_edge_local_endpoints,
                                "columnar block input-edge local endpoints");
                        });
                    return tmp;
                });
        }
        std::future<IndexedSpqrTree> tree_node_virtual_edges_future;
        if (plan.tree_node_virtual_edges) {
            tree_node_virtual_edges_future = launch_tmp(
                section_id_tree_node_virtual_edges,
                "the columnar tree-node virtual-edge section",
                [](IndexedSpqrTree& tmp, std::istream& section_in) {
                    read_u32_vector(section_in,
                                    tmp.tree_node_virtual_edges,
                                    "columnar tree-node virtual-edge index");
                });
        }
        std::future<IndexedSpqrTree> real_edge_types_future;
        if (plan.real_edge_types) {
            real_edge_types_future = launch_tmp(
                section_id_real_edge_types,
                "the columnar real-edge type section",
                [path, expected_real_edges](IndexedSpqrTree& tmp,
                                            std::istream& section_in) {
                    tmp.bubble_edge_type_count_hint =
                        detail::read_scalar<std::uint64_t>(
                            section_in, "columnar BubbleFinder edge type count hint");
                    read_u8_vector(section_in,
                                   tmp.real_edge_type_pairs,
                                   "columnar real-edge endpoint type index");
                    if (!tmp.real_edge_type_pairs.empty() &&
                        tmp.real_edge_type_pairs.size() != expected_real_edges) {
                        throw std::runtime_error(
                            "columnar SPQR real-edge type count mismatch in " + path);
                    }
                });
        }
        std::future<IndexedSpqrTree> block_hashes_future;
        if (plan.block_hashes) {
            block_hashes_future = launch_tmp(
                section_id_block_hashes,
                "the columnar block-hash section",
                [path, expected_blocks](IndexedSpqrTree& tmp,
                                        std::istream& section_in) {
                    read_u64_vector(section_in,
                                    tmp.block_node_hashes,
                                    "columnar block node hashes");
                    if (tmp.block_node_hashes.size() != expected_blocks) {
                        throw std::runtime_error(
                            "columnar SPQR block hash count mismatch in " + path);
                    }
                });
        }

        std::future<IndexedSpqrTree> macro_tree_cache_future;
        const SectionEntryDisk* macro_tree_cache_section =
            find_section(sections, section_id_macro_tree_cache);
        if (plan.macro_tree_cache && macro_tree_cache_section != nullptr) {
            const SectionEntryDisk section = *macro_tree_cache_section;
            macro_tree_cache_future = schedule_section_task(
                [path, section]() {
                    IndexedSpqrTree tmp;
                    read_section_payload_from_file(
                        path,
                        section,
                        "columnar SPQR macro-tree/core-SPQR cache section",
                        selected_zstd_chunk_prefetch(4),
                        [&](std::istream& section_in) {
                            read_column_payload_header(
                                section_in,
                                section_id_macro_tree_cache,
                                path);
                            tmp.read_macro_tree_cache_payload(section_in, path);
                        });
                    return tmp;
                });
        }

        load_requested_graph_node_names();
        load_requested_graph_degrees();

        component_count_hint = expected_components;
        cut_count_hint = header.cut_count;

        IndexedSpqrTree block_nodes_tmp = block_nodes_future.get();
        block_nodes = std::move(block_nodes_tmp.block_nodes);
        IndexedSpqrTree blocks_tmp = blocks_future.get();
        slim_blocks = std::move(blocks_tmp.slim_blocks);
        block_tree_edge_counts_ready = true;

        if (tree_node_original_nodes_future.valid()) {
            IndexedSpqrTree tree_node_original_nodes_tmp =
                tree_node_original_nodes_future.get();
            tree_node_original_nodes =
                std::move(tree_node_original_nodes_tmp.tree_node_original_nodes);
        }
        if (plan.tree_nodes) {
            IndexedSpqrTree node_begins_tmp =
                tree_node_node_begins_future.get();
            IndexedSpqrTree real_edge_begins_tmp =
                tree_node_real_edge_begins_future.get();
            IndexedSpqrTree virtual_edge_begins_tmp =
                tree_node_virtual_edge_begins_future.get();
            IndexedSpqrTree node_counts_tmp =
                tree_node_node_counts_future.get();
            IndexedSpqrTree real_edge_counts_tmp =
                tree_node_real_edge_counts_future.get();
            IndexedSpqrTree virtual_edge_counts_tmp =
                tree_node_virtual_edge_counts_future.get();
            compact_tree_node_node_begins =
                std::move(node_begins_tmp.compact_tree_node_node_begins);
            compact_tree_node_real_edge_begins =
                std::move(real_edge_begins_tmp.compact_tree_node_real_edge_begins);
            compact_tree_node_virtual_edge_begins =
                std::move(virtual_edge_begins_tmp.compact_tree_node_virtual_edge_begins);
            compact_tree_node_node_counts16 =
                std::move(node_counts_tmp.compact_tree_node_node_counts16);
            compact_tree_node_node_count_overflow_nodes =
                std::move(node_counts_tmp.compact_tree_node_node_count_overflow_nodes);
            compact_tree_node_node_count_overflow_values =
                std::move(node_counts_tmp.compact_tree_node_node_count_overflow_values);
            compact_tree_node_real_edge_counts16 =
                std::move(real_edge_counts_tmp.compact_tree_node_real_edge_counts16);
            compact_tree_node_real_edge_count_overflow_nodes =
                std::move(real_edge_counts_tmp.compact_tree_node_real_edge_count_overflow_nodes);
            compact_tree_node_real_edge_count_overflow_values =
                std::move(real_edge_counts_tmp.compact_tree_node_real_edge_count_overflow_values);
            compact_tree_node_virtual_edge_counts16 =
                std::move(virtual_edge_counts_tmp.compact_tree_node_virtual_edge_counts16);
            compact_tree_node_virtual_edge_count_overflow_nodes =
                std::move(virtual_edge_counts_tmp.compact_tree_node_virtual_edge_count_overflow_nodes);
            compact_tree_node_virtual_edge_count_overflow_values =
                std::move(virtual_edge_counts_tmp.compact_tree_node_virtual_edge_count_overflow_values);
        } else {
            tree_node_count_hint = expected_tree_nodes;
        }
        if (tree_node_types_future.valid()) {
            IndexedSpqrTree tree_node_types_tmp = tree_node_types_future.get();
            slim_tree_node_types =
                std::move(tree_node_types_tmp.slim_tree_node_types);
        }
        if (virtual_edges_future.valid()) {
            IndexedSpqrTree virtual_edges_tmp = virtual_edges_future.get();
            slim_virtual_edges = std::move(virtual_edges_tmp.slim_virtual_edges);
        } else {
            virtual_edge_count_hint = expected_virtual_edges;
        }
        slim_record_tables = true;

        if (real_edges_future.valid()) {
            IndexedSpqrTree real_edges_tmp = real_edges_future.get();
            real_edge_endpoints = std::move(real_edges_tmp.real_edge_endpoints);
        } else {
            real_edge_count_hint = expected_real_edges;
        }

        if (block_to_tree_nodes_future.valid()) {
            IndexedSpqrTree block_to_tree_nodes_tmp =
                block_to_tree_nodes_future.get();
            block_to_tree_nodes =
                std::move(block_to_tree_nodes_tmp.block_to_tree_nodes);
            block_to_tree_nodes_identity =
                block_to_tree_nodes_tmp.block_to_tree_nodes_identity;
        }
        if (block_input_edges_future.valid()) {
            IndexedSpqrTree block_input_edges_tmp =
                block_input_edges_future.get();
            block_input_edge_offsets =
                std::move(block_input_edges_tmp.block_input_edge_offsets);
            block_input_edges =
                std::move(block_input_edges_tmp.block_input_edges);
        }
        if (block_input_orientations_future.valid()) {
            IndexedSpqrTree block_input_orientations_tmp =
                block_input_orientations_future.get();
            block_input_edge_orientations =
                std::move(block_input_orientations_tmp
                              .block_input_edge_orientations);
        }
        if (block_input_local_endpoints_future.valid()) {
            IndexedSpqrTree block_input_local_endpoints_tmp =
                block_input_local_endpoints_future.get();
            block_input_edge_local_endpoints =
                std::move(block_input_local_endpoints_tmp
                              .block_input_edge_local_endpoints);
        }
        if (!block_input_edge_orientations.empty() &&
            block_input_edge_orientations.size() !=
                block_input_edges.size()) {
            throw std::runtime_error(
                "columnar SPQR block input-edge orientation size mismatch in " +
                path);
        }
        if (!block_input_edge_local_endpoints.empty()) {
            if (block_input_edge_local_endpoints.size() !=
                block_input_edges.size()) {
                throw std::runtime_error(
                    "columnar SPQR block input-edge local endpoint size mismatch in " +
                    path);
            }
            const std::uint32_t block_total = block_count();
            if (block_input_edge_offsets.size() !=
                static_cast<std::size_t>(block_total) + 1u) {
                throw std::runtime_error(
                    "columnar SPQR block input-edge local endpoints require block offsets in " +
                    path);
            }
            for (std::uint32_t bi = 0; bi < block_total; ++bi) {
                const auto block = block_at(bi);
                const std::uint32_t node_count =
                    block.node_end - block.node_begin;
                const std::uint32_t begin = block_input_edge_offsets[bi];
                const std::uint32_t end = block_input_edge_offsets[bi + 1u];
                for (std::uint32_t p = begin; p < end; ++p) {
                    const auto &ep = block_input_edge_local_endpoints[p];
                    if (ep.src >= node_count || ep.dst >= node_count) {
                        throw std::runtime_error(
                            "columnar SPQR block input-edge local endpoint out of range in " +
                            path);
                    }
                }
            }
        }

        if (tree_node_virtual_edges_future.valid()) {
            IndexedSpqrTree tree_node_virtual_edges_tmp =
                tree_node_virtual_edges_future.get();
            tree_node_virtual_edges =
                std::move(tree_node_virtual_edges_tmp.tree_node_virtual_edges);
        }

        bubble_edge_type_count_hint = header.bubble_edge_type_count;
        if (real_edge_types_future.valid()) {
            IndexedSpqrTree real_edge_types_tmp = real_edge_types_future.get();
            bubble_edge_type_count_hint =
                real_edge_types_tmp.bubble_edge_type_count_hint != 0
                    ? real_edge_types_tmp.bubble_edge_type_count_hint
                    : header.bubble_edge_type_count;
            real_edge_type_pairs =
                std::move(real_edge_types_tmp.real_edge_type_pairs);
        }

        if (block_hashes_future.valid()) {
            IndexedSpqrTree block_hashes_tmp = block_hashes_future.get();
            block_node_hashes = std::move(block_hashes_tmp.block_node_hashes);
        }
        if (macro_tree_cache_future.valid()) {
            IndexedSpqrTree macro_tree_cache_tmp = macro_tree_cache_future.get();
            macro_cache_blocks =
                std::move(macro_tree_cache_tmp.macro_cache_blocks);
            macro_cache_nodes =
                std::move(macro_tree_cache_tmp.macro_cache_nodes);
            macro_cache_children =
                std::move(macro_tree_cache_tmp.macro_cache_children);
            macro_cache_core_edges =
                std::move(macro_tree_cache_tmp.macro_cache_core_edges);
            macro_cache_core_nodes =
                std::move(macro_tree_cache_tmp.macro_cache_core_nodes);
            macro_cache_input_endpoints =
                std::move(macro_tree_cache_tmp.macro_cache_input_endpoints);
            macro_cache_input_edge_ids =
                std::move(macro_tree_cache_tmp.macro_cache_input_edge_ids);
            macro_cache_core_node_inv =
                std::move(macro_tree_cache_tmp.macro_cache_core_node_inv);
            macro_cache_core_tree_node_types =
                std::move(macro_tree_cache_tmp.macro_cache_core_tree_node_types);
            macro_cache_core_tree_parents =
                std::move(macro_tree_cache_tmp.macro_cache_core_tree_parents);
            macro_cache_core_tree_child_offsets =
                std::move(macro_tree_cache_tmp.macro_cache_core_tree_child_offsets);
            macro_cache_core_tree_children =
                std::move(macro_tree_cache_tmp.macro_cache_core_tree_children);
            macro_cache_core_tree_skeleton_offsets =
                std::move(macro_tree_cache_tmp.macro_cache_core_tree_skeleton_offsets);
            macro_cache_core_tree_skeleton_edges =
                std::move(macro_tree_cache_tmp.macro_cache_core_tree_skeleton_edges);
            macro_cache_core_tree_mapping_offsets =
                std::move(macro_tree_cache_tmp.macro_cache_core_tree_mapping_offsets);
            macro_cache_core_tree_mapping =
                std::move(macro_tree_cache_tmp.macro_cache_core_tree_mapping);
            macro_cache_core_tree_skeleton_num_nodes =
                std::move(macro_tree_cache_tmp.macro_cache_core_tree_skeleton_num_nodes);
        }

        if (tree_node_real_edges_payload_future.valid()) {
            std::vector<char> tree_node_real_edges_payload =
                tree_node_real_edges_payload_future.get();
            MemoryInputBuf tree_real_buf(tree_node_real_edges_payload.data(),
                                         tree_node_real_edges_payload.size());
            std::istream tree_real_in(&tree_real_buf);
            read_column_payload_header(
                tree_real_in, section_id_tree_node_real_edges, path);
            read_tree_node_real_edge_index(tree_real_in, path);
            drain_stream(
                tree_real_in, "columnar tree-node real-edge section", path);
        }

        load_requested_graph_view();
        load_requested_graph_components();

        if (block_count() != expected_blocks ||
            tree_node_count() != expected_tree_nodes ||
            virtual_edge_count() != expected_virtual_edges ||
            real_edge_count() != expected_real_edges) {
            throw std::runtime_error(
                "columnar SPQR runtime count mismatch in " + path);
        }
        if (!real_edge_type_pairs.empty() &&
            real_edge_type_pairs.size() != real_edge_count()) {
            throw std::runtime_error(
                "columnar SPQR real-edge type count mismatch in " + path);
        }
        validate_requested_graph_view();

        runtime_indices_ready = true;
        load_requested_graph_node_lookup();
        load_requested_haplotypes();
        apply_eager_load_plan_indexes(plan);
    }
