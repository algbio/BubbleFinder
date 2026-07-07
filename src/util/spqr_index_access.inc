    // Lookup tables and the small functions that use them.

    bool runtime_indices_ready = false;
    bool block_node_hash_lookup_ready = false;
    bool graph_node_name_lookup_ready = false;
    bool graph_node_numeric_lookup_ready = false;
    bool graph_node_numeric_lookup_dense = false;
    bool graph_node_numeric_lookup_dense_domain = false;
    std::uint64_t graph_node_numeric_lookup_base = 0;
    static constexpr std::uint16_t compact_tree_node_count_sentinel =
        std::numeric_limits<std::uint16_t>::max();

    bool compact_tree_node_ranges_active() const
    {
        return !compact_tree_node_node_begins.empty() ||
               !compact_tree_node_real_edge_begins.empty() ||
               !compact_tree_node_virtual_edge_begins.empty();
    }

    static std::uint32_t checked_range_end(std::uint32_t begin,
                                           std::uint32_t count,
                                           const char* what)
    {
        if (begin > std::numeric_limits<std::uint32_t>::max() - count) {
            throw std::out_of_range(std::string("SPQR index ") + what + " overflows");
        }
        return begin + count;
    }

    static std::uint32_t compact_tree_node_count_at(
        const std::vector<std::uint16_t>& counts16,
        const std::vector<std::uint32_t>& overflow_nodes,
        const std::vector<std::uint32_t>& overflow_values,
        std::uint32_t i,
        const char* what)
    {
        if (i >= counts16.size()) {
            throw std::out_of_range(std::string("SPQR index compact ") + what +
                                    " table is truncated");
        }
        const std::uint16_t small = counts16[i];
        if (small != compact_tree_node_count_sentinel) {
            return static_cast<std::uint32_t>(small);
        }
        const auto it = std::lower_bound(overflow_nodes.begin(),
                                         overflow_nodes.end(),
                                         i);
        if (it == overflow_nodes.end() || *it != i) {
            throw std::out_of_range(std::string("SPQR index compact ") + what +
                                    " overflow is missing");
        }
        const std::size_t pos =
            static_cast<std::size_t>(it - overflow_nodes.begin());
        if (pos >= overflow_values.size()) {
            throw std::out_of_range(std::string("SPQR index compact ") + what +
                                    " overflow value is missing");
        }
        return overflow_values[pos];
    }

    void clear_compact_tree_node_ranges()
    {
        drop_storage(compact_tree_node_node_begins);
        drop_storage(compact_tree_node_real_edge_begins);
        drop_storage(compact_tree_node_virtual_edge_begins);
        drop_storage(compact_tree_node_node_counts16);
        drop_storage(compact_tree_node_real_edge_counts16);
        drop_storage(compact_tree_node_virtual_edge_counts16);
        drop_storage(compact_tree_node_node_count_overflow_nodes);
        drop_storage(compact_tree_node_node_count_overflow_values);
        drop_storage(compact_tree_node_real_edge_count_overflow_nodes);
        drop_storage(compact_tree_node_real_edge_count_overflow_values);
        drop_storage(compact_tree_node_virtual_edge_count_overflow_nodes);
        drop_storage(compact_tree_node_virtual_edge_count_overflow_values);
    }

    static std::uint16_t encode_compact_tree_node_count(
        std::vector<std::uint32_t>& overflow_nodes,
        std::vector<std::uint32_t>& overflow_values,
        std::uint32_t i,
        std::uint32_t count)
    {
        if (count < compact_tree_node_count_sentinel) {
            return static_cast<std::uint16_t>(count);
        }
        overflow_nodes.push_back(i);
        overflow_values.push_back(count);
        return compact_tree_node_count_sentinel;
    }

    static std::uint32_t checked_u32_range_length(std::uint32_t begin,
                                                  std::uint32_t end,
                                                  const char* what)
    {
        if (end < begin) {
            throw std::out_of_range(std::string("SPQR index invalid ") + what);
        }
        return end - begin;
    }

    bool graph_node_numeric_name_valid_at(std::uint32_t i) const
    {
        if (!graph_node_numeric_name_valid.empty()) {
            if (i >= graph_node_numeric_name_valid.size()) {
                throw std::runtime_error(
                    "SPQR index numeric graph node validity table is truncated");
            }
            return graph_node_numeric_name_valid[i] != 0;
        }
        if (graph_node_string_names.empty()) return true;
        auto it = std::lower_bound(
            graph_node_string_names.begin(),
            graph_node_string_names.end(),
            i,
            [](const std::pair<std::uint32_t, std::string>& item,
               std::uint32_t value) {
                return item.first < value;
            });
        return it == graph_node_string_names.end() || it->first != i;
    }

    bool detect_dense_numeric_graph_node_lookup(std::uint64_t& base) const
    {
        if (!has_compact_numeric_graph_node_names()) return false;
        if (graph_node_numeric_names_dense) {
            base = graph_node_numeric_name_base;
            return true;
        }
        const std::uint32_t n = graph_node_count();
        auto sparse_it = graph_node_string_names.begin();
        auto valid_at = [&](std::uint32_t i) {
            if (!graph_node_numeric_name_valid.empty()) {
                if (i >= graph_node_numeric_name_valid.size()) {
                    throw std::runtime_error(
                        "SPQR index numeric graph node validity table is truncated");
                }
                return graph_node_numeric_name_valid[i] != 0;
            }
            while (sparse_it != graph_node_string_names.end() &&
                   sparse_it->first < i) {
                ++sparse_it;
            }
            return sparse_it == graph_node_string_names.end() ||
                   sparse_it->first != i;
        };

        std::uint32_t first_valid = 0;
        while (first_valid < n && !valid_at(first_valid)) ++first_valid;
        if (first_valid == n) return false;

        const std::uint64_t first_name =
            graph_node_numeric_name_at(first_valid);
        if (first_name < first_valid) return false;
        base = first_name - first_valid;
        for (std::uint32_t i = first_valid + 1u; i < n; ++i) {
            if (!valid_at(i)) continue;
            if (base > std::numeric_limits<std::uint64_t>::max() -
                           static_cast<std::uint64_t>(i)) {
                return false;
            }
            if (graph_node_numeric_name_at(i) !=
                base + static_cast<std::uint64_t>(i)) {
                return false;
            }
        }
        return true;
    }

    bool build_dense_numeric_graph_node_domain_lookup()
    {
        if (!has_compact_numeric_graph_node_names()) return false;
        const std::uint32_t n = graph_node_count();
        bool any = false;
        std::uint64_t min_value = std::numeric_limits<std::uint64_t>::max();
        std::uint64_t max_value = 0;
        std::uint64_t valid_count = 0;

        for (std::uint32_t i = 0; i < n; ++i) {
            if (!graph_node_numeric_name_valid_at(i)) continue;
            const std::uint64_t value = graph_node_numeric_name_at(i);
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
            ++valid_count;
            any = true;
        }
        if (!any || max_value < min_value) return false;

        const std::uint64_t range64 = max_value - min_value + 1u;
        if (range64 > static_cast<std::uint64_t>(
                          std::numeric_limits<std::uint32_t>::max())) {
            return false;
        }
        const std::uint64_t max_reasonable_range =
            std::max<std::uint64_t>(1024u, valid_count * 2u);
        if (range64 > max_reasonable_range) return false;

        std::vector<std::uint32_t> dense(
            static_cast<std::size_t>(range64), invalid_id);
        for (std::uint32_t i = 0; i < n; ++i) {
            if (!graph_node_numeric_name_valid_at(i)) continue;
            const std::uint64_t value = graph_node_numeric_name_at(i);
            const std::uint64_t offset64 = value - min_value;
            const std::size_t offset = static_cast<std::size_t>(offset64);
            if (dense[offset] != invalid_id) {
                drop_storage(graph_node_by_dense_numeric_name);
                graph_node_numeric_lookup_dense_domain = false;
                return false;
            }
            dense[offset] = i;
        }

        graph_node_by_dense_numeric_name = std::move(dense);
        graph_node_numeric_lookup_base = min_value;
        graph_node_numeric_lookup_dense_domain = true;
        return true;
    }

    bool find_prepared_graph_node_numeric_name_id(std::uint64_t value,
                                                  std::uint32_t& out) const
    {
        if (graph_node_numeric_lookup_dense) {
            if (value < graph_node_numeric_lookup_base) return false;
            const std::uint64_t offset64 =
                value - graph_node_numeric_lookup_base;
            if (offset64 > std::numeric_limits<std::uint32_t>::max()) {
                return false;
            }
            const std::uint32_t offset = static_cast<std::uint32_t>(offset64);
            if (offset >= graph_node_count()) return false;
            if (graph_node_numeric_names.empty() &&
                graph_node_numeric_names32.empty() &&
                graph_node_numeric_name_valid.empty() &&
                graph_node_string_names.empty()) {
                out = offset;
                return true;
            }
            if (!graph_node_numeric_name_valid_at(offset)) return false;
            if (graph_node_numeric_name_at(offset) != value) return false;
            out = offset;
            return true;
        }
        if (graph_node_numeric_lookup_dense_domain) {
            if (value < graph_node_numeric_lookup_base) return false;
            const std::uint64_t offset64 =
                value - graph_node_numeric_lookup_base;
            if (offset64 >= graph_node_by_dense_numeric_name.size()) return false;
            const std::uint32_t candidate =
                graph_node_by_dense_numeric_name[static_cast<std::size_t>(offset64)];
            if (candidate == invalid_id) return false;
            out = candidate;
            return true;
        }
        auto it = graph_node_by_numeric_name.find(value);
        if (it == graph_node_by_numeric_name.end()) return false;
        out = it->second;
        return true;
    }

    std::pair<std::uint32_t, std::uint32_t>
    compute_block_tree_edge_counts(const BlockRecord& block) const
    {
        std::uint64_t real_count = 0;
        std::uint64_t virtual_count = 0;
        if (block.spqr_begin > block.spqr_end) {
            throw std::out_of_range("SPQR index block SPQR range is invalid");
        }
        for (std::uint32_t p = block.spqr_begin; p < block.spqr_end; ++p) {
            const std::uint32_t tree_node_index = block_tree_node_at(p);
            const auto tn = tree_node_at(tree_node_index);
            if (tn.real_edge_end < tn.real_edge_begin ||
                tn.virtual_edge_end < tn.virtual_edge_begin) {
                throw std::out_of_range("SPQR index tree-node edge range is invalid");
            }
            real_count += static_cast<std::uint64_t>(
                tn.real_edge_end - tn.real_edge_begin);
            virtual_count += static_cast<std::uint64_t>(
                tn.virtual_edge_end - tn.virtual_edge_begin);
        }
        return {
            graph_index::require_spqr_count(
                real_count, "SPQR index block tree real-edge count"),
            graph_index::require_spqr_count(
                virtual_count, "SPQR index block tree virtual-edge count")
        };
    }

    void populate_block_tree_edge_counts()
    {
        for (auto& block : blocks) {
            const auto counts = compute_block_tree_edge_counts(block);
            block.tree_real_edge_count = counts.first;
            block.tree_virtual_edge_count = counts.second;
        }
        block_tree_edge_counts_ready = true;
    }
