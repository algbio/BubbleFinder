    // Lookup tables and the small functions that use them.

    bool runtime_indices_ready = false;
    bool block_node_hash_lookup_ready = false;
    bool graph_node_name_lookup_ready = false;
    bool graph_node_numeric_lookup_ready = false;
    bool graph_node_numeric_lookup_dense = false;
    bool graph_node_numeric_lookup_dense_domain = false;
    std::uint64_t graph_node_numeric_lookup_base = 0;

    struct SectionLoadPlan {
        bool graph_node_degrees = true;
        bool tree_node_original_nodes = true;
        bool tree_nodes = true;
        bool tree_node_types = true;
        bool virtual_edges = true;
        bool real_edges = true;
        bool block_to_tree_nodes = true;
        bool tree_node_real_edges = true;
        bool block_input_edges = true;
        bool tree_node_virtual_edges = true;
        bool real_edge_types = true;
        bool block_hashes = true;
        bool macro_tree_cache = true;
        bool graph_edge_tables = true;
        bool graph_components = false;
        bool haplotypes = false;
        bool eager_graph_node_name_lookup = false;
        bool eager_block_hash_lookup = false;
    };

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

    template <typename ValueFn>
    void write_tree_node_u32_column(std::ostream& out,
                                    std::uint32_t count,
                                    ValueFn value) const
    {
        detail::write_scalar<std::uint64_t>(out, count);
        constexpr std::size_t target_bytes = 4ull * 1024ull * 1024ull;
        const std::size_t values_per_chunk =
            std::max<std::size_t>(1, target_bytes / sizeof(std::uint32_t));
        std::vector<std::uint32_t> buffer;
        buffer.reserve(values_per_chunk);
        for (std::uint32_t base = 0; base < count;) {
            const std::size_t n =
                std::min<std::size_t>(values_per_chunk, count - base);
            buffer.clear();
            for (std::size_t j = 0; j < n; ++j) {
                buffer.push_back(value(base + static_cast<std::uint32_t>(j)));
            }
            write_raw_bytes(out, buffer.data(),
                            buffer.size() * sizeof(std::uint32_t));
            base += static_cast<std::uint32_t>(n);
        }
    }

    template <typename CountFn>
    void write_tree_node_count_column(std::ostream& out,
                                      std::uint32_t count,
                                      CountFn value) const
    {
        std::uint64_t overflow_count = 0;
        for (std::uint32_t i = 0; i < count; ++i) {
            if (value(i) >= compact_tree_node_count_sentinel) {
                ++overflow_count;
            }
        }
        const std::uint64_t compact_bytes =
            static_cast<std::uint64_t>(count) * sizeof(std::uint16_t) +
            overflow_count * 2u * sizeof(std::uint32_t);
        const std::uint64_t raw_bytes =
            static_cast<std::uint64_t>(count) * sizeof(std::uint32_t);
        if (raw_bytes <= compact_bytes) {
            detail::write_scalar<std::uint8_t>(out, 2u);
            detail::write_scalar<std::uint64_t>(out, count);
            constexpr std::size_t target_bytes = 4ull * 1024ull * 1024ull;
            const std::size_t values_per_chunk =
                std::max<std::size_t>(1, target_bytes / sizeof(std::uint32_t));
            std::vector<std::uint32_t> buffer;
            buffer.reserve(values_per_chunk);
            for (std::uint32_t base = 0; base < count;) {
                const std::size_t n =
                    std::min<std::size_t>(values_per_chunk, count - base);
                buffer.clear();
                for (std::size_t j = 0; j < n; ++j) {
                    buffer.push_back(value(base + static_cast<std::uint32_t>(j)));
                }
                write_raw_bytes(out, buffer.data(),
                                buffer.size() * sizeof(std::uint32_t));
                base += static_cast<std::uint32_t>(n);
            }
            return;
        }

        detail::write_scalar<std::uint8_t>(out, 1u);
        detail::write_scalar<std::uint64_t>(out, count);
        constexpr std::size_t target_bytes = 4ull * 1024ull * 1024ull;
        const std::size_t values_per_chunk =
            std::max<std::size_t>(1, target_bytes / sizeof(std::uint16_t));
        std::vector<std::uint16_t> buffer;
        buffer.reserve(values_per_chunk);
        std::vector<std::uint32_t> overflow_nodes;
        std::vector<std::uint32_t> overflow_values;
        overflow_nodes.reserve(static_cast<std::size_t>(overflow_count));
        overflow_values.reserve(static_cast<std::size_t>(overflow_count));
        for (std::uint32_t base = 0; base < count;) {
            const std::size_t n =
                std::min<std::size_t>(values_per_chunk, count - base);
            buffer.clear();
            for (std::size_t j = 0; j < n; ++j) {
                const std::uint32_t node =
                    base + static_cast<std::uint32_t>(j);
                const std::uint32_t v = value(node);
                if (v < compact_tree_node_count_sentinel) {
                    buffer.push_back(static_cast<std::uint16_t>(v));
                } else {
                    buffer.push_back(compact_tree_node_count_sentinel);
                    overflow_nodes.push_back(node);
                    overflow_values.push_back(v);
                }
            }
            write_raw_bytes(out, buffer.data(),
                            buffer.size() * sizeof(std::uint16_t));
            base += static_cast<std::uint32_t>(n);
        }
        write_u32_vector(out, overflow_nodes);
        write_u32_vector(out, overflow_values);
    }

    void write_tree_node_node_begins_column(std::ostream& out) const
    {
        write_tree_node_u32_column(
            out,
            tree_node_count(),
            [&](std::uint32_t i) { return tree_node_at(i).node_begin; });
    }

    void write_tree_node_real_edge_begins_column(std::ostream& out) const
    {
        write_tree_node_u32_column(
            out,
            tree_node_count(),
            [&](std::uint32_t i) { return tree_node_at(i).real_edge_begin; });
    }

    void write_tree_node_virtual_edge_begins_column(std::ostream& out) const
    {
        write_tree_node_u32_column(
            out,
            tree_node_count(),
            [&](std::uint32_t i) { return tree_node_at(i).virtual_edge_begin; });
    }

    void write_tree_node_node_counts_column(std::ostream& out) const
    {
        write_tree_node_count_column(out, tree_node_count(), [&](std::uint32_t i) {
            const auto tn = tree_node_at(i);
            return checked_u32_range_length(
                tn.node_begin, tn.node_end, "tree-node skeleton-node range");
        });
    }

    void write_tree_node_real_edge_counts_column(std::ostream& out) const
    {
        write_tree_node_count_column(out, tree_node_count(), [&](std::uint32_t i) {
            const auto tn = tree_node_at(i);
            return checked_u32_range_length(
                tn.real_edge_begin, tn.real_edge_end, "tree-node real-edge range");
        });
    }

    void write_tree_node_virtual_edge_counts_column(std::ostream& out) const
    {
        write_tree_node_count_column(out, tree_node_count(), [&](std::uint32_t i) {
            const auto tn = tree_node_at(i);
            return checked_u32_range_length(
                tn.virtual_edge_begin,
                tn.virtual_edge_end,
                "tree-node virtual-edge range");
        });
    }

    static void validate_compact_tree_node_count_column(
        std::uint32_t expected_count,
        const std::vector<std::uint16_t>& counts16,
        const std::vector<std::uint32_t>& overflow_nodes,
        const std::vector<std::uint32_t>& overflow_values,
        const char* what,
        const std::string& path)
    {
        if (counts16.size() != expected_count ||
            overflow_nodes.size() != overflow_values.size()) {
            throw std::runtime_error(
                std::string("binary SPQR compact tree-node column has wrong size for ") +
                what + " in " + path);
        }
        std::uint32_t previous = invalid_id;
        for (std::uint32_t i = 0; i < overflow_nodes.size(); ++i) {
            const std::uint32_t node = overflow_nodes[i];
            if (node >= expected_count ||
                (i != 0 && node <= previous) ||
                counts16[node] != compact_tree_node_count_sentinel ||
                overflow_values[i] < compact_tree_node_count_sentinel) {
                throw std::runtime_error(
                    std::string("binary SPQR compact tree-node column has invalid overflow table for ") +
                    what + " in " + path);
            }
            previous = node;
        }
    }

    static void read_tree_node_count_column(
        std::istream& in,
        std::uint32_t expected_count,
        std::vector<std::uint16_t>& counts16,
        std::vector<std::uint32_t>& overflow_nodes,
        std::vector<std::uint32_t>& overflow_values,
        const char* what,
        const std::string& path)
    {
        const std::uint8_t encoding =
            detail::read_scalar<std::uint8_t>(
                in, "compact tree-node count-column encoding");
        if (encoding == 1u) {
            read_u16_vector(in, counts16, what);
            read_u32_vector(in, overflow_nodes, what);
            read_u32_vector(in, overflow_values, what);
            validate_compact_tree_node_count_column(
                expected_count,
                counts16,
                overflow_nodes,
                overflow_values,
                what,
                path);
            return;
        }
        if (encoding == 2u) {
            const std::uint32_t count = read_count(in, what);
            if (count != expected_count) {
                throw std::runtime_error(
                    std::string("binary SPQR compact tree-node count-column has wrong count for ") +
                    what + " in " + path);
            }
            counts16.resize(count);
            constexpr std::size_t target_bytes = 4ull * 1024ull * 1024ull;
            const std::size_t values_per_chunk =
                std::max<std::size_t>(1, target_bytes / sizeof(std::uint32_t));
            std::vector<std::uint32_t> buffer(values_per_chunk);
            for (std::uint32_t base = 0; base < count;) {
                const std::size_t n =
                    std::min<std::size_t>(values_per_chunk, count - base);
                read_raw_bytes(in, buffer.data(), n * sizeof(std::uint32_t), what);
                for (std::size_t j = 0; j < n; ++j) {
                    const std::uint32_t i =
                        base + static_cast<std::uint32_t>(j);
                    counts16[i] = encode_compact_tree_node_count(
                        overflow_nodes,
                        overflow_values,
                        i,
                        buffer[j]);
                }
                base += static_cast<std::uint32_t>(n);
            }
            validate_compact_tree_node_count_column(
                expected_count,
                counts16,
                overflow_nodes,
                overflow_values,
                what,
                path);
            return;
        }
        throw std::runtime_error(
            std::string("unsupported compact tree-node count-column encoding for ") +
            what + " in " + path);
    }

    static void require_compact_tree_node_vector_size(
        std::size_t actual,
        std::uint32_t expected,
        const char* what,
        const std::string& path)
    {
        if (actual != expected) {
            throw std::runtime_error(
                std::string("binary SPQR compact tree-node column has wrong size for ") +
                what + " in " + path);
        }
    }

    bool eager_lookup_view_matches(EagerLookupViewFilter filter) const
    {
        const std::string_view view = graph_view.empty()
                                    ? graph_profile_raw
                                    : std::string_view(graph_view);
        switch (filter) {
            case EagerLookupViewFilter::Any:
                return true;
            case EagerLookupViewFilter::OrientedBidirected:
                return graph_profile_is_oriented_bidirected(view);
            case EagerLookupViewFilter::OrientedDirected:
                return graph_profile_is_oriented_directed(view);
            case EagerLookupViewFilter::RawOrParallelSubdivided:
                return graph_profile_is_raw_or_parallel_subdivided(view);
        }
        return false;
    }

    SectionLoadPlan build_section_load_plan(const LoadOptions& options) const
    {
        SectionLoadPlan plan;
        const bool load_graph_edge_tables =
            !options.skip_graph_edge_tables &&
            !(options.skip_graph_edge_tables_for_oriented_views &&
              graph_profile_is_oriented_double(graph_view));
        plan.graph_node_degrees = !options.skip_graph_node_degrees;
        plan.real_edge_types = !options.skip_real_edge_types;
        plan.macro_tree_cache = !options.skip_macro_tree_cache;
        plan.graph_edge_tables = load_graph_edge_tables;
        plan.graph_components = options.load_graph_components;
        plan.haplotypes = options.load_haplotypes;
        const bool eager_lookup_allowed =
            eager_lookup_view_matches(options.eager_lookup_view_filter);
        plan.eager_graph_node_name_lookup =
            options.eager_graph_node_name_lookup && eager_lookup_allowed;
        plan.eager_block_hash_lookup =
            options.eager_block_hash_lookup && eager_lookup_allowed;

        if (options.graph_only) {
            plan.tree_node_original_nodes = false;
            plan.tree_nodes = false;
            plan.tree_node_types = false;
            plan.virtual_edges = false;
            plan.real_edges = false;
            plan.block_to_tree_nodes = false;
            plan.tree_node_real_edges = false;
            plan.block_input_edges = false;
            plan.tree_node_virtual_edges = false;
            plan.real_edge_types = false;
            plan.block_hashes = false;
            plan.macro_tree_cache = false;
            plan.graph_edge_tables = load_graph_edge_tables;
            plan.eager_block_hash_lookup = false;
            return plan;
        }

        return plan;
    }

    void apply_eager_load_plan_indexes(const SectionLoadPlan& plan)
    {
        if (plan.eager_graph_node_name_lookup) {
            prepare_graph_node_name_lookup();
        }
        if (plan.eager_block_hash_lookup) {
            prepare_block_hash_lookup();
        }
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
