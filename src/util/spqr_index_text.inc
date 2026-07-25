    // Read the line-based .spqr text format.

    void invalidate_graph_node_lookup()
    {
        graph_node_by_name.clear();
        graph_node_by_name.rehash(0);
        graph_node_by_numeric_name.clear();
        graph_node_by_numeric_name.rehash(0);
        drop_storage(graph_node_by_dense_numeric_name);
        graph_node_numeric_lookup_dense = false;
        graph_node_numeric_lookup_dense_domain = false;
        graph_node_numeric_lookup_base = 0;
        graph_node_name_lookup_ready = false;
        graph_node_numeric_lookup_ready = false;
    }

    static std::uint32_t find_or_invalid(const std::unordered_map<std::uint32_t, std::uint32_t>& map,
                                         std::uint32_t key)
    {
        auto it = map.find(key);
        return it == map.end() ? invalid_id : it->second;
    }

    void add_node_name(std::uint32_t name_id)
    {
        if (name_id == invalid_id) return;
        auto it = node_ordinal_by_name.find(name_id);
        if (it != node_ordinal_by_name.end()) return;
        const std::uint32_t ordinal =
            graph_index::require_spqr_count_size(node_names.size(), "SPQR index graph node table");
        node_ordinal_by_name.emplace(name_id, ordinal);
        node_names.push_back(name_id);
    }

    template <typename OwnerFn, typename AssignFn>
    static void build_owner_ranges(std::uint32_t owner_count,
                                   std::uint32_t item_count,
                                   OwnerFn owner_of,
                                   AssignFn assign_range)
    {
        std::vector<std::uint32_t> offsets(static_cast<std::size_t>(owner_count) + 1u, 0);
        for (std::uint32_t i = 0; i < item_count; ++i) {
            const std::uint32_t owner = owner_of(i);
            if (owner != invalid_id && owner < owner_count) ++offsets[owner + 1u];
        }
        for (std::uint32_t i = 0; i < owner_count; ++i) offsets[i + 1u] += offsets[i];
        for (std::uint32_t owner = 0; owner < owner_count; ++owner) {
            assign_range(owner, offsets[owner], offsets[owner + 1u]);
        }
    }

    template <typename OwnerFn, typename AssignFn>
    static void build_owner_index(std::uint32_t owner_count,
                                  std::uint32_t item_count,
                                  OwnerFn owner_of,
                                  std::vector<std::uint32_t>& out,
                                  AssignFn assign_range)
    {
        std::vector<std::uint32_t> offsets(static_cast<std::size_t>(owner_count) + 1u, 0);
        for (std::uint32_t i = 0; i < item_count; ++i) {
            const std::uint32_t owner = owner_of(i);
            if (owner != invalid_id && owner < owner_count) ++offsets[owner + 1u];
        }
        for (std::uint32_t i = 0; i < owner_count; ++i) offsets[i + 1u] += offsets[i];
        out.assign(offsets.back(), 0);
        std::vector<std::uint32_t> cursor(offsets.begin(), offsets.end());
        for (std::uint32_t i = 0; i < item_count; ++i) {
            const std::uint32_t owner = owner_of(i);
            if (owner != invalid_id && owner < owner_count) out[cursor[owner]++] = i;
        }
        for (std::uint32_t owner = 0; owner < owner_count; ++owner) {
            assign_range(owner, offsets[owner], offsets[owner + 1u]);
        }
    }

    void build_virtual_index()
    {
        for (auto& tn : tree_nodes) {
            tn.virtual_edge_begin = tn.virtual_edge_end = 0;
        }
        std::vector<std::uint32_t> offsets(tree_nodes.size() + 1u, 0);
        for (std::uint32_t i = 0; i < virtual_edges.size(); ++i) {
            const auto& ve = virtual_edges[i];
            if (ve.node1_index != invalid_id && ve.node1_index < tree_nodes.size()) {
                ++offsets[ve.node1_index + 1u];
            }
            if (ve.node2_index != invalid_id && ve.node2_index < tree_nodes.size() &&
                ve.node2_index != ve.node1_index) {
                ++offsets[ve.node2_index + 1u];
            }
        }
        for (std::uint32_t i = 0; i < tree_nodes.size(); ++i) offsets[i + 1u] += offsets[i];
        tree_node_virtual_edges.assign(offsets.back(), 0);
        std::vector<std::uint32_t> cursor(offsets.begin(), offsets.end());
        for (std::uint32_t i = 0; i < virtual_edges.size(); ++i) {
            const auto& ve = virtual_edges[i];
            if (ve.node1_index != invalid_id && ve.node1_index < tree_nodes.size()) {
                tree_node_virtual_edges[cursor[ve.node1_index]++] = i;
            }
            if (ve.node2_index != invalid_id && ve.node2_index < tree_nodes.size() &&
                ve.node2_index != ve.node1_index) {
                tree_node_virtual_edges[cursor[ve.node2_index]++] = i;
            }
        }
        for (std::uint32_t i = 0; i < tree_nodes.size(); ++i) {
            tree_nodes[i].virtual_edge_begin = offsets[i];
            tree_nodes[i].virtual_edge_end = offsets[i + 1u];
        }
    }

    void build_block_node_signature_index()
    {
        const std::uint32_t count = block_count();
        block_node_hashes.assign(count, 0);
        blocks_by_node_hash.clear();
        blocks_by_node_hash.reserve(count);

        std::vector<std::uint32_t> names;
        for (std::uint32_t bi = 0; bi < count; ++bi) {
            const auto block = block_at(bi);
            names.assign(block_nodes.begin() + block.node_begin,
                         block_nodes.begin() + block.node_end);
            std::sort(names.begin(), names.end());
            const std::uint64_t h = hash_sorted_name_set(names);
            block_node_hashes[bi] = h;
            blocks_by_node_hash.push_back(bi);
        }
        sort_block_hash_index();
        block_node_hash_lookup_ready = true;
    }

    void parse_text_stream(std::istream& in, const std::string& path)
    {
        std::string line;
        std::uint64_t line_no = 0;
        while (std::getline(in, line)) {
            ++line_no;
            parse_text_line(line, path, line_no);
        }
        if (!in.eof()) throw std::runtime_error("failed while reading SPQR text file: " + path);
    }

    void parse_text_line(std::string_view line, const std::string& path, std::uint64_t line_no)
    {
        const auto toks = detail::split_ws(line);
        if (toks.empty()) return;
        const std::string_view tag = toks[0];

        if (tag == "H") {
            if (toks.size() >= 2) format_version.assign(toks[1].data(), toks[1].size());
            if (toks.size() >= 3) format_url.assign(toks[2].data(), toks[2].size());
        } else if (tag == "G") {
            require_tokens(toks, 2, path, line_no);
            ComponentRecord rec;
            rec.name = intern(toks[1]);
            rec.node_begin = append_name_refs(component_nodes, toks, 2);
            rec.node_end = graph_index::require_spqr_count_size(
                component_nodes.size(), "SPQR component node refs");
            components.push_back(rec);
        } else if (tag == "B") {
            require_tokens(toks, 3, path, line_no);
            BlockRecord rec;
            rec.name = intern(toks[1]);
            rec.node_begin = append_name_refs(block_nodes, toks, 3);
            rec.node_end = graph_index::require_spqr_count_size(
                block_nodes.size(), "SPQR block node refs");
            blocks.push_back(rec);
        } else if (tag == "C") {
            require_tokens(toks, 2, path, line_no);
            CutRecord rec;
            rec.name = intern(toks[1]);
            rec.block_begin = append_name_refs(cut_blocks, toks, 2);
            rec.block_end = graph_index::require_spqr_count_size(
                cut_blocks.size(), "SPQR cut block refs");
            cuts.push_back(rec);
        } else if (tag == "S" || tag == "P" || tag == "R") {
            require_tokens(toks, 3, path, line_no);
            TreeNodeRecord rec;
            rec.type = static_cast<std::uint8_t>(tag[0]);
            rec.name = intern(toks[1]);
            rec.block = intern(toks[2]);
            rec.node_begin = append_name_refs(tree_node_original_nodes, toks, 3);
            rec.node_end = graph_index::require_spqr_count_size(
                tree_node_original_nodes.size(), "SPQR tree-node original node refs");
            tree_nodes.push_back(rec);
        } else if (tag == "V") {
            require_tokens(toks, 6, path, line_no);
            VirtualEdgeRecord rec;
            rec.name = intern(toks[1]);
            rec.node1 = intern(toks[2]);
            rec.node2 = intern(toks[3]);
            rec.pole1 = intern(toks[4]);
            rec.pole2 = intern(toks[5]);
            virtual_edges.push_back(rec);
        } else if (tag == "E") {
            require_tokens(toks, 5, path, line_no);
            RealEdgeRecord rec;
            rec.name = intern(toks[1]);
            rec.container = intern(toks[2]);
            rec.src = intern(toks[3]);
            rec.dst = intern(toks[4]);
            real_edges.push_back(rec);
        } else if (tag == "X" && toks.size() >= 2 && toks[1] == "BF_EDGE_TYPE") {
            require_tokens(toks, 5, path, line_no);
            BubbleEdgeTypeRecord rec;
            rec.edge = intern(toks[2]);
            try {
                rec.src_type = detail::parse_bubble_edge_type(toks[3]);
                rec.dst_type = detail::parse_bubble_edge_type(toks[4]);
            } catch (const std::exception& e) {
                std::ostringstream msg;
                msg << "invalid BF_EDGE_TYPE extension in " << path << ":" << line_no
                    << ": " << e.what();
                throw std::runtime_error(msg.str());
            }
            bubble_edge_types.push_back(rec);
        } else if (tag == "X" && toks.size() >= 2 &&
                   toks[1] == "GRAPH_PROFILE") {
            require_tokens(toks, 3, path, line_no);
            graph_view = canonical_graph_profile(toks[2]);
        }
    }

    static void require_tokens(const std::vector<std::string_view>& toks,
                               std::size_t n,
                               const std::string& path,
                               std::uint64_t line_no)
    {
        if (toks.size() < n) {
            std::ostringstream msg;
            msg << "invalid .spqr line in " << path << ":" << line_no
                << " (expected at least " << n << " tokens)";
            throw std::runtime_error(msg.str());
        }
    }

    std::uint32_t append_name_refs(std::vector<std::uint32_t>& dst,
                                   const std::vector<std::string_view>& toks,
                                   std::size_t begin)
    {
        const std::uint32_t start =
            graph_index::require_spqr_count_size(dst.size(), "SPQR name ref array");
        dst.reserve(dst.size() + (toks.size() - begin));
        for (std::size_t i = begin; i < toks.size(); ++i) {
            dst.push_back(intern(toks[i]));
        }
        return start;
    }
