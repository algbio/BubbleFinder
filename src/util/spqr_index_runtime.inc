    // Build indexes in RAM and use them to find data.

    template <typename T>
    static void drop_storage(T& value)
    {
        value.clear();
        value.shrink_to_fit();
    }

    std::uint32_t intern(std::string_view value)
    {
        if (string_to_id.empty() && !strings.empty()) {
            rebuild_lookup_maps();
        }
        std::string key(value);
        auto it = string_to_id.find(key);
        if (it != string_to_id.end()) return it->second;
        invalidate_graph_node_lookup();
        const std::uint32_t id =
            graph_index::require_spqr_count_size(strings.size(), "SPQR index string table");
        strings.emplace_back(std::move(key));
        string_to_id.emplace(strings.back(), id);
        return id;
    }

    void drop_bubble_edge_type_records()
    {
        if (!bubble_edge_types.empty()) {
            bubble_edge_type_count_hint = bubble_edge_types.size();
        }
        std::vector<BubbleEdgeTypeRecord>().swap(bubble_edge_types);
    }

    void drop_graph_edge_tables()
    {
        if (!graph_edges.empty()) {
            graph_edge_count_hint = graph_edges.size();
        }
        std::vector<GraphEdgeRecord>().swap(graph_edges);
        std::vector<std::uint8_t>().swap(graph_edge_type_pairs);
        std::vector<std::uint8_t>().swap(graph_edge_multi_flags);
    }

    bool has_graph_components() const
    {
        return !graph_components.empty() &&
               (graph_component_nodes_identity || !graph_component_nodes.empty()) &&
               (graph_component_edges_identity || !graph_component_edges.empty());
    }

    const std::string& name(std::uint32_t id) const
    {
        if (id >= strings.size()) {
            throw std::out_of_range("SPQR index string id out of range");
        }
        return strings[id];
    }

    bool has_compact_numeric_graph_node_names() const
    {
        const std::uint32_t n = graph_node_count();
        return n != 0 &&
            (graph_node_numeric_names_dense ||
             graph_node_numeric_names.size() == n ||
             graph_node_numeric_names32.size() == n);
    }

    std::uint32_t graph_node_count() const
    {
        if (compact_graph_node_count != 0) return compact_graph_node_count;
        return graph_index::require_spqr_count_size(
            node_names.size(), "SPQR index graph node count");
    }

    std::uint64_t graph_edge_count() const
    {
        if (!graph_edges.empty()) return graph_edges.size();
        return graph_edge_count_hint;
    }

    std::uint64_t component_count() const
    {
        return components.empty() ? component_count_hint
                                  : static_cast<std::uint64_t>(components.size());
    }

    std::uint32_t block_count() const
    {
        if (blocks.empty() && slim_blocks.empty() && block_count_hint != 0) {
            return graph_index::require_spqr_count(
                block_count_hint, "SPQR index block count");
        }
        return graph_index::require_spqr_count_size(
            slim_record_tables ? slim_blocks.size() : blocks.size(),
            "SPQR index block count");
    }

    std::uint32_t tree_node_count() const
    {
        if (slim_record_tables) {
            if (compact_tree_node_ranges_active()) {
                return graph_index::require_spqr_count_size(
                    compact_tree_node_node_begins.size(),
                    "SPQR index compact tree-node count");
            }
            return graph_index::require_spqr_count(
                tree_node_count_hint, "SPQR index tree-node count");
        }
        return graph_index::require_spqr_count_size(
            tree_nodes.size(), "SPQR index tree-node count");
    }

    std::uint32_t virtual_edge_count() const
    {
        if (slim_virtual_edges.empty() && virtual_edges.empty() &&
            virtual_edge_count_hint != 0) {
            return graph_index::require_spqr_count(
                virtual_edge_count_hint, "SPQR index virtual-edge count");
        }
        return graph_index::require_spqr_count_size(
            slim_record_tables ? slim_virtual_edges.size() : virtual_edges.size(),
            "SPQR index virtual-edge count");
    }

    BlockRecord block_at(std::uint32_t i) const
    {
        if (slim_record_tables) {
            if (i >= slim_blocks.size()) {
                throw std::out_of_range("SPQR index slim block out of range");
            }
            const auto& r = slim_blocks[i];
            BlockRecord out{};
            out.node_begin = r.node_begin;
            out.node_end = r.node_end;
            out.spqr_begin = r.spqr_begin;
            out.spqr_end = r.spqr_end;
            out.real_edge_begin = r.real_edge_begin;
            out.real_edge_end = r.real_edge_end;
            out.tree_real_edge_count = r.tree_real_edge_count;
            out.tree_virtual_edge_count = r.tree_virtual_edge_count;
            return out;
        }
        if (i >= blocks.size()) {
            throw std::out_of_range("SPQR index block out of range");
        }
        return blocks[i];
    }

    TreeNodeRecord tree_node_at(std::uint32_t i) const
    {
        if (slim_record_tables) {
            if (!compact_tree_node_ranges_active() ||
                i >= compact_tree_node_node_begins.size()) {
                throw std::out_of_range("SPQR index compact tree node out of range");
            }
            if (i >= slim_tree_node_types.size()) {
                throw std::out_of_range("SPQR index slim tree-node type out of range");
            }
            TreeNodeRecord out{};
            out.type = slim_tree_node_types[i];
            out.node_begin = compact_tree_node_node_begins[i];
            out.real_edge_begin = compact_tree_node_real_edge_begins[i];
            out.virtual_edge_begin = compact_tree_node_virtual_edge_begins[i];
            out.node_end =
                checked_range_end(
                    out.node_begin,
                    compact_tree_node_count_at(
                        compact_tree_node_node_counts16,
                        compact_tree_node_node_count_overflow_nodes,
                        compact_tree_node_node_count_overflow_values,
                        i,
                        "tree-node skeleton-node count"),
                    "tree-node skeleton-node range");
            out.real_edge_end =
                checked_range_end(
                    out.real_edge_begin,
                    compact_tree_node_count_at(
                        compact_tree_node_real_edge_counts16,
                        compact_tree_node_real_edge_count_overflow_nodes,
                        compact_tree_node_real_edge_count_overflow_values,
                        i,
                        "tree-node real-edge count"),
                    "tree-node real-edge range");
            out.virtual_edge_end =
                checked_range_end(
                    out.virtual_edge_begin,
                    compact_tree_node_count_at(
                        compact_tree_node_virtual_edge_counts16,
                        compact_tree_node_virtual_edge_count_overflow_nodes,
                        compact_tree_node_virtual_edge_count_overflow_values,
                        i,
                        "tree-node virtual-edge count"),
                    "tree-node virtual-edge range");
            return out;
        }
        if (i >= tree_nodes.size()) {
            throw std::out_of_range("SPQR index tree node out of range");
        }
        return tree_nodes[i];
    }

    VirtualEdgeRecord virtual_edge_at(std::uint32_t i) const
    {
        if (slim_record_tables) {
            if (i >= slim_virtual_edges.size()) {
                throw std::out_of_range("SPQR index slim virtual edge out of range");
            }
            const auto& r = slim_virtual_edges[i];
            VirtualEdgeRecord out{};
            out.node1_index = r.node1_index;
            out.node2_index = r.node2_index;
            out.pole1 = r.pole1;
            out.pole2 = r.pole2;
            return out;
        }
        if (i >= virtual_edges.size()) {
            throw std::out_of_range("SPQR index virtual edge out of range");
        }
        return virtual_edges[i];
    }

    std::uint32_t real_edge_count() const
    {
        if (real_edge_endpoints.empty() && real_edges.empty() &&
            real_edge_count_hint != 0) {
            return graph_index::require_spqr_count(
                real_edge_count_hint, "SPQR index real edge count");
        }
        return graph_index::require_spqr_count_size(
            real_edge_endpoints.empty() ? real_edges.size() : real_edge_endpoints.size(),
            "SPQR index real edge count");
    }

    bool has_compact_real_edge_endpoints() const
    {
        return !real_edge_endpoints.empty();
    }

    GraphEdgeRecord real_edge_endpoint_at(std::uint32_t i) const
    {
        if (!real_edge_endpoints.empty()) {
            if (i >= real_edge_endpoints.size()) {
                throw std::out_of_range("SPQR index compact real-edge endpoint out of range");
            }
            return real_edge_endpoints[i];
        }
        if (i >= real_edges.size()) {
            throw std::out_of_range("SPQR index real-edge endpoint out of range");
        }
        return GraphEdgeRecord{real_edges[i].src, real_edges[i].dst};
    }

    std::uint64_t graph_node_numeric_name_at(std::uint32_t i) const
    {
        const std::uint32_t n = graph_node_count();
        if (graph_node_numeric_names_dense) {
            if (i >= n) {
                throw std::out_of_range("SPQR index dense numeric graph node name out of range");
            }
            if (graph_node_numeric_name_base >
                std::numeric_limits<std::uint64_t>::max() -
                    static_cast<std::uint64_t>(i)) {
                throw std::overflow_error("SPQR index dense numeric graph node name overflow");
            }
            return graph_node_numeric_name_base + static_cast<std::uint64_t>(i);
        }
        if (graph_node_numeric_names32.size() == n) {
            if (i >= graph_node_numeric_names32.size()) {
                throw std::out_of_range("SPQR index compact u32 graph node name out of range");
            }
            return graph_node_numeric_names32[i];
        }
        if (i >= graph_node_numeric_names.size()) {
            throw std::out_of_range("SPQR index compact u64 graph node name out of range");
        }
        return graph_node_numeric_names[i];
    }

    bool uses_oriented_numeric_graph_node_names() const
    {
        return graph_profile_is_oriented_double(graph_view);
    }

    static bool parse_oriented_canonical_u64(std::string_view s,
                                             std::uint64_t& out)
    {
        if (s.size() < 2) return false;
        const char orient = s.back();
        if (orient != '+' && orient != '-') return false;
        std::uint64_t base = 0;
        if (!parse_canonical_u64(s.substr(0, s.size() - 1), base)) return false;
        if (base > (std::numeric_limits<std::uint64_t>::max() >> 1)) return false;
        out = (base << 1) | (orient == '-' ? 1ull : 0ull);
        return true;
    }

    static std::string oriented_numeric_graph_node_name(std::uint64_t encoded)
    {
        std::string out = std::to_string(encoded >> 1);
        out.push_back((encoded & 1ull) ? '-' : '+');
        return out;
    }

    void copy_graph_node_numeric_names(std::vector<std::uint64_t>& out) const
    {
        const std::uint32_t n = graph_node_count();
        if (graph_node_numeric_names_dense) {
            out.resize(n);
            for (std::uint32_t i = 0; i < n; ++i) {
                out[i] = graph_node_numeric_name_at(i);
            }
            return;
        }
        if (graph_node_numeric_names.size() == n) {
            out = graph_node_numeric_names;
            return;
        }
        out.resize(graph_node_numeric_names32.size());
        for (std::uint32_t i = 0; i < graph_node_numeric_names32.size(); ++i) {
            out[i] = graph_node_numeric_names32[i];
        }
    }

    static bool parse_canonical_u64(std::string_view s, std::uint64_t& out)
    {
        if (s.empty()) return false;
        if (s.size() > 1 && s[0] == '0') return false;
        std::uint64_t value = 0;
        const char* begin = s.data();
        const char* end = s.data() + s.size();
        auto result = std::from_chars(begin, end, value);
        if (result.ec != std::errc() || result.ptr != end) return false;
        out = value;
        return true;
    }

    static std::uint64_t decimal_u64_digits(std::uint64_t value)
    {
        std::uint64_t digits = 1;
        while (value >= 10u) {
            value /= 10u;
            ++digits;
        }
        return digits;
    }

    bool has_complete_bubble_edge_types() const
    {
        const std::uint32_t count = real_edge_count();
        if (count == 0) return false;
        if (real_edge_type_pairs.size() != count) return false;
        for (std::uint8_t pair : real_edge_type_pairs) {
            const std::uint8_t src = static_cast<std::uint8_t>((pair >> 4) & 0x0fu);
            const std::uint8_t dst = static_cast<std::uint8_t>(pair & 0x0fu);
            if ((src != 1u && src != 2u) || (dst != 1u && dst != 2u)) return false;
        }
        return true;
    }

    std::pair<std::uint8_t, std::uint8_t> real_edge_endpoint_types(std::uint32_t real_edge_index) const
    {
        if (real_edge_index >= real_edge_type_pairs.size()) return {0, 0};
        const std::uint8_t pair = real_edge_type_pairs[real_edge_index];
        return {static_cast<std::uint8_t>((pair >> 4) & 0x0fu),
                static_cast<std::uint8_t>(pair & 0x0fu)};
    }

    std::uint32_t block_tree_node_at(std::uint32_t offset) const
    {
        if (block_to_tree_nodes_identity) {
            if (offset >= tree_node_count()) {
                throw std::out_of_range("SPQR index block-to-tree-node identity offset out of range");
            }
            return offset;
        }
        if (offset >= block_to_tree_nodes.size()) {
            throw std::out_of_range("SPQR index block-to-tree-node offset out of range");
        }
        return block_to_tree_nodes[offset];
    }

    std::uint32_t tree_node_real_edge_at(std::uint32_t offset) const
    {
        if (tree_node_real_edges_range) {
            throw std::out_of_range(
                "SPQR index compact tree-node real-edge range requires a tree-node index");
        }
        if (tree_node_real_edges_identity) {
            if (offset >= real_edge_count()) {
                throw std::out_of_range("SPQR index tree-node real-edge identity offset out of range");
            }
            return offset;
        }
        if (offset >= tree_node_real_edges.size()) {
            throw std::out_of_range("SPQR index tree-node real-edge offset out of range");
        }
        return tree_node_real_edges[offset];
    }

    std::uint32_t tree_node_real_edge_at(std::uint32_t tree_node_index,
                                         std::uint32_t offset) const
    {
        if (tree_node_index >= tree_node_count()) {
            throw std::out_of_range("SPQR index tree-node real-edge tree node out of range");
        }
        const auto tn = tree_node_at(tree_node_index);
        if (offset < tn.real_edge_begin || offset >= tn.real_edge_end) {
            throw std::out_of_range("SPQR index tree-node real-edge offset outside tree node range");
        }
        if (tree_node_real_edges_range) {
            if (tree_node_index >= tree_node_real_edge_firsts.size()) {
                throw std::out_of_range("SPQR index tree-node real-edge range table is truncated");
            }
            const std::uint32_t first = tree_node_real_edge_firsts[tree_node_index];
            if (first == invalid_id) {
                throw std::out_of_range("SPQR index tree-node real-edge range is empty");
            }
            const std::uint32_t delta = offset - tn.real_edge_begin;
            if (first > std::numeric_limits<std::uint32_t>::max() - delta) {
                throw std::out_of_range("SPQR index tree-node real-edge range overflows");
            }
            return first + delta;
        }
        return tree_node_real_edge_at(offset);
    }

    std::uint32_t block_tree_real_edge_count(std::uint32_t block_index) const
    {
        const auto block = block_at(block_index);
        if (block_tree_edge_counts_ready) return block.tree_real_edge_count;
        return compute_block_tree_edge_counts(block).first;
    }

    std::pair<std::uint32_t, std::uint32_t>
    block_tree_edge_counts(std::uint32_t block_index) const
    {
        const auto block = block_at(block_index);
        if (block_tree_edge_counts_ready) {
            return {block.tree_real_edge_count, block.tree_virtual_edge_count};
        }
        return compute_block_tree_edge_counts(block);
    }

    void rebuild_lookup_maps()
    {
        string_to_id.clear();
        string_to_id.reserve(strings.size());
        for (std::uint32_t i = 0; i < strings.size(); ++i) {
            string_to_id.emplace(strings[i], i);
        }
        invalidate_graph_node_lookup();
    }

    void prepare_graph_node_name_lookup()
    {
        if (has_compact_numeric_graph_node_names()) {
            if (!graph_node_numeric_lookup_ready) {
                std::uint64_t dense_base = 0;
                graph_node_by_numeric_name.clear();
                graph_node_by_numeric_name.rehash(0);
                drop_storage(graph_node_by_dense_numeric_name);
                graph_node_numeric_lookup_dense_domain = false;
                graph_node_numeric_lookup_dense =
                    detect_dense_numeric_graph_node_lookup(dense_base);
                graph_node_numeric_lookup_base = dense_base;
                if (!graph_node_numeric_lookup_dense &&
                    !build_dense_numeric_graph_node_domain_lookup()) {
                    const std::uint32_t n = graph_node_count();
                    graph_node_by_numeric_name.reserve(n);
                    for (std::uint32_t i = 0; i < n; ++i) {
                        if (!graph_node_numeric_name_valid_at(i)) continue;
                        graph_node_by_numeric_name.emplace(graph_node_numeric_name_at(i), i);
                    }
                }
                graph_node_numeric_lookup_ready = true;
            }

            if (!graph_node_string_names.empty() && !graph_node_name_lookup_ready) {
                graph_node_by_name.clear();
                graph_node_by_name.reserve(graph_node_string_names.size());
                for (const auto& item : graph_node_string_names) {
                    graph_node_by_name.emplace(std::string_view(item.second), item.first);
                }
                graph_node_name_lookup_ready = true;
            }
            return;
        }
        if (graph_node_name_lookup_ready) return;
        graph_node_by_name.clear();
        graph_node_by_name.reserve(node_names.size());
        for (std::uint32_t name_id : node_names) {
            if (name_id >= strings.size()) continue;
            graph_node_by_name.emplace(std::string_view(strings[name_id]), name_id);
        }
        graph_node_name_lookup_ready = true;
    }

    bool find_graph_node_name_id(const std::string& value, std::uint32_t& out) const
    {
        if (graph_node_numeric_lookup_ready) {
            std::uint64_t numeric = 0;
            const bool parsed = uses_oriented_numeric_graph_node_names()
                ? parse_oriented_canonical_u64(value, numeric)
                : parse_canonical_u64(value, numeric);
            if (parsed) {
                if (find_prepared_graph_node_numeric_name_id(numeric, out)) return true;
            }
        }
        if (graph_node_name_lookup_ready) {
            auto it = graph_node_by_name.find(std::string_view(value));
            if (it != graph_node_by_name.end()) {
                out = it->second;
                return true;
            }
            return false;
        }

        if (!string_to_id.empty()) {
            auto it = string_to_id.find(value);
            if (it != string_to_id.end()) {
                out = it->second;
                return true;
            }
        }
        return false;
    }

    bool find_graph_node_numeric_name_id(std::uint64_t value, std::uint32_t& out) const
    {
        if (!graph_node_numeric_lookup_ready) return false;
        return find_prepared_graph_node_numeric_name_id(value, out);
    }

    void prepare_block_hash_lookup()
    {
        if (block_node_hash_lookup_ready) return;
        const std::uint32_t count = block_count();
        if (block_node_hashes.size() != count) {
            build_block_node_signature_index();
            return;
        }
        blocks_by_node_hash.clear();
        blocks_by_node_hash.reserve(count);
        for (std::uint32_t bi = 0; bi < count; ++bi) {
            blocks_by_node_hash.push_back(bi);
        }
        sort_block_hash_index();
        block_node_hash_lookup_ready = true;
    }

    void build_indices()
    {
        runtime_indices_ready = false;
        cut_count_hint = cuts.size();
        component_count_hint = components.size();
        tree_node_count_hint = tree_nodes.size();
        virtual_edge_count_hint = virtual_edges.size();
        bubble_edge_type_count_hint = bubble_edge_types.size();
        graph_edge_count_hint = graph_edges.size();
        component_by_name.clear();
        block_by_name.clear();
        tree_node_by_name.clear();
        node_ordinal_by_name.clear();
        node_names.clear();
        if (!real_edge_endpoints.empty() && real_edges.empty()) {
            throw std::runtime_error(
                "cannot rebuild full SPQR runtime indices from compact endpoint-only real edges");
        }
        real_edge_endpoints.clear();
        block_to_tree_nodes.clear();
        tree_node_real_edges.clear();
        tree_node_real_edge_firsts.clear();
        tree_node_virtual_edges.clear();
        block_to_tree_nodes_identity = false;
        tree_node_real_edges_identity = false;
        tree_node_real_edges_range = false;
        real_edge_type_pairs.clear();
        block_node_hashes.clear();
        compact_graph_node_count = 0;
        graph_node_in_degrees.clear();
        graph_node_out_degrees.clear();
        graph_node_in_degrees8.clear();
        graph_node_out_degrees8.clear();
        graph_node_in_degree_overflow_nodes.clear();
        graph_node_in_degree_overflow_values.clear();
        graph_node_out_degree_overflow_nodes.clear();
        graph_node_out_degree_overflow_values.clear();
        graph_node_numeric_names.clear();
        graph_node_numeric_names32.clear();
        graph_node_numeric_names_dense = false;
        graph_node_numeric_name_base = 0;
        graph_node_numeric_name_valid.clear();
        graph_node_string_names.clear();
        slim_blocks.clear();
        slim_tree_node_types.clear();
        clear_compact_tree_node_ranges();
        slim_virtual_edges.clear();
        slim_record_tables = false;
        blocks_by_node_hash.clear();
        graph_node_by_name.clear();
        graph_node_by_numeric_name.clear();
        graph_node_by_dense_numeric_name.clear();
        graph_node_numeric_lookup_dense = false;
        graph_node_numeric_lookup_dense_domain = false;
        graph_node_numeric_lookup_base = 0;
        graph_components.clear();
        graph_component_nodes.clear();
        graph_component_edges.clear();
        graph_component_nodes_identity = false;
        graph_component_edges_identity = false;
        graph_node_name_lookup_ready = false;
        graph_node_numeric_lookup_ready = false;
        block_node_hash_lookup_ready = false;
        block_tree_edge_counts_ready = false;

        component_by_name.reserve(components.size());
        for (std::uint32_t i = 0; i < components.size(); ++i) {
            component_by_name[components[i].name] = i;
            components[i].real_edge_begin = components[i].real_edge_end = 0;
            for (std::uint32_t j = components[i].node_begin; j < components[i].node_end; ++j) {
                add_node_name(component_nodes[j]);
            }
        }

        block_by_name.reserve(blocks.size());
        for (std::uint32_t i = 0; i < blocks.size(); ++i) {
            block_by_name[blocks[i].name] = i;
            blocks[i].spqr_begin = blocks[i].spqr_end = 0;
            blocks[i].real_edge_begin = blocks[i].real_edge_end = 0;
            for (std::uint32_t j = blocks[i].node_begin; j < blocks[i].node_end; ++j) {
                add_node_name(block_nodes[j]);
            }
        }

        tree_node_by_name.reserve(tree_nodes.size());
        for (std::uint32_t i = 0; i < tree_nodes.size(); ++i) {
            tree_node_by_name[tree_nodes[i].name] = i;
            tree_nodes[i].block_index = find_or_invalid(block_by_name, tree_nodes[i].block);
            tree_nodes[i].real_edge_begin = tree_nodes[i].real_edge_end = 0;
            tree_nodes[i].virtual_edge_begin = tree_nodes[i].virtual_edge_end = 0;
            for (std::uint32_t j = tree_nodes[i].node_begin; j < tree_nodes[i].node_end; ++j) {
                add_node_name(tree_node_original_nodes[j]);
            }
        }

        for (auto& ve : virtual_edges) {
            ve.node1_index = find_or_invalid(tree_node_by_name, ve.node1);
            ve.node2_index = find_or_invalid(tree_node_by_name, ve.node2);
            add_node_name(ve.pole1);
            add_node_name(ve.pole2);
        }

        for (auto& re : real_edges) {
            if ((re.container_index = find_or_invalid(tree_node_by_name, re.container)) != invalid_id) {
                re.container_kind = ContainerKind::SpqrNode;
            } else if ((re.container_index = find_or_invalid(block_by_name, re.container)) != invalid_id) {
                re.container_kind = ContainerKind::Block;
            } else if ((re.container_index = find_or_invalid(component_by_name, re.container)) != invalid_id) {
                re.container_kind = ContainerKind::Component;
            } else {
                re.container_kind = ContainerKind::Unknown;
            }
            add_node_name(re.src);
            add_node_name(re.dst);
        }
        for (auto& ge : graph_edges) {
            add_node_name(ge.src);
            add_node_name(ge.dst);
        }

        build_owner_index(blocks.size(), tree_nodes.size(),
                          [&](std::uint32_t i) { return tree_nodes[i].block_index; },
                          block_to_tree_nodes,
                          [&](std::uint32_t owner, std::uint32_t begin, std::uint32_t end) {
                              blocks[owner].spqr_begin = begin;
                              blocks[owner].spqr_end = end;
                          });

        build_owner_ranges(components.size(), real_edges.size(),
                           [&](std::uint32_t i) {
                               return real_edges[i].container_kind == ContainerKind::Component
                                   ? real_edges[i].container_index
                                   : invalid_id;
                           },
                           [&](std::uint32_t owner, std::uint32_t begin, std::uint32_t end) {
                               components[owner].real_edge_begin = begin;
                               components[owner].real_edge_end = end;
                           });

        build_owner_ranges(blocks.size(), real_edges.size(),
                           [&](std::uint32_t i) {
                               return real_edges[i].container_kind == ContainerKind::Block
                                   ? real_edges[i].container_index
                                   : invalid_id;
                           },
                           [&](std::uint32_t owner, std::uint32_t begin, std::uint32_t end) {
                               blocks[owner].real_edge_begin = begin;
                               blocks[owner].real_edge_end = end;
                           });

        build_owner_index(tree_nodes.size(), real_edges.size(),
                          [&](std::uint32_t i) {
                              return real_edges[i].container_kind == ContainerKind::SpqrNode
                                  ? real_edges[i].container_index
                                  : invalid_id;
                          },
                          tree_node_real_edges,
                          [&](std::uint32_t owner, std::uint32_t begin, std::uint32_t end) {
                              tree_nodes[owner].real_edge_begin = begin;
                              tree_nodes[owner].real_edge_end = end;
                          });

        build_virtual_index();
        populate_block_tree_edge_counts();
        build_bubble_edge_type_index();
        build_block_node_signature_index();
        runtime_indices_ready = true;
    }

    static std::uint64_t hash_sorted_name_set(const std::vector<std::uint32_t>& names)
    {
        std::uint64_t h = 1469598103934665603ull;
        auto mix = [&](std::uint64_t x) {
            h ^= x + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
            h *= 1099511628211ull;
        };
        mix(static_cast<std::uint64_t>(names.size()));
        for (std::uint32_t id : names) mix(id);
        return h;
    }

    void sort_block_hash_index()
    {
        std::sort(blocks_by_node_hash.begin(),
                  blocks_by_node_hash.end(),
                  [&](std::uint32_t a, std::uint32_t b) {
                      const std::uint64_t ah = block_node_hashes[a];
                      const std::uint64_t bh = block_node_hashes[b];
                      if (ah != bh) return ah < bh;
                      return a < b;
                  });
    }

    std::uint32_t find_block_by_sorted_node_names(const std::vector<std::uint32_t>& sorted_names) const
    {
        if (!block_node_hash_lookup_ready) {
            const_cast<IndexedSpqrTree*>(this)->prepare_block_hash_lookup();
        }
        const std::uint64_t h = hash_sorted_name_set(sorted_names);
        auto it = std::lower_bound(
            blocks_by_node_hash.begin(),
            blocks_by_node_hash.end(),
            h,
            [&](std::uint32_t block, std::uint64_t value) {
                return block < block_node_hashes.size() &&
                       block_node_hashes[block] < value;
            });
        const std::uint32_t count = block_count();
        for (; it != blocks_by_node_hash.end(); ++it) {
            const std::uint32_t bi = *it;
            if (bi >= count) continue;
            if (bi >= block_node_hashes.size() || block_node_hashes[bi] != h) break;
            const auto block = block_at(bi);
            if (static_cast<std::size_t>(block.node_end - block.node_begin) != sorted_names.size())
                continue;
            std::vector<std::uint32_t> candidate(
                block_nodes.begin() + block.node_begin,
                block_nodes.begin() + block.node_end);
            std::sort(candidate.begin(), candidate.end());
            if (candidate == sorted_names) return bi;
        }
        return invalid_id;
    }
