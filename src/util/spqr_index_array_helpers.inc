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
