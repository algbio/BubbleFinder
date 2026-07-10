    // IDs and small records used in the .spqri file.
    //
    // File order:
    //
    //   [magic + version + format info]
    //   [section header + table pointer]
    //   [SPQR sections]
    //   [graph edges and components]
    //   [haplotype paths, optional]
    //   [chunk directory, optional]
    //   [section table]

    static constexpr char binary_magic[8] =
        {'B', 'F', 'S', 'Q', 'R', 'I', '\1', '\0'};

    enum Magic : std::uint32_t {
        sectioned_runtime_magic = 0x31544353u,
        section_table_magic = 0x314c4254u,
        columnar_section_magic = 0x314c4f43u,
        zstd_chunked_section_magic = 0x314b435au,
        zstd_chunk_table_magic = 0x3154435au,
        zstd_chunk_directory_magic = 0x3144435au,
        graph_edge_tables_magic = 0x31544747u,
        graph_components_magic = 0x31504347u,
        haplotypes_magic = 0x31504148u
    };

    enum SectionId : std::uint32_t {
        section_id_graph_edge_tables = 2u,
        section_id_zstd_chunk_directory = 3u,
        section_id_runtime_header = 100u,
        section_id_graph_node_names = 101u,
        section_id_graph_node_degrees = 102u,
        section_id_block_nodes = 103u,
        section_id_blocks = 104u,
        section_id_tree_node_original_nodes = 105u,
        section_id_tree_node_types = 107u,
        section_id_virtual_edges = 108u,
        section_id_real_edges = 109u,
        section_id_block_to_tree_nodes = 110u,
        section_id_tree_node_real_edges = 111u,
        section_id_tree_node_virtual_edges = 112u,
        section_id_real_edge_types = 113u,
        section_id_block_hashes = 114u,
        section_id_graph_node_numeric_lookup = 115u,
        section_id_tree_node_node_begins = 118u,
        section_id_tree_node_real_edge_begins = 119u,
        section_id_tree_node_virtual_edge_begins = 120u,
        section_id_tree_node_node_counts = 121u,
        section_id_tree_node_real_edge_counts = 122u,
        section_id_tree_node_virtual_edge_counts = 123u,
        section_id_macro_tree_cache = 125u,
        section_id_graph_components = 128u,
        section_id_block_input_edges = 129u,
        section_id_block_input_edge_orientations = 130u,
        section_id_block_input_edge_local_endpoints = 131u,
        section_id_haplotypes = 132u
    };

    enum GraphComponentFlag : std::uint32_t {
        graph_component_nodes_identity_flag = 1u << 0u,
        graph_component_edges_identity_flag = 1u << 1u
    };

    enum class SectionCodec : std::uint8_t {
        Raw = 0,
        Zstd = 1,
        ZstdChunked = 2
    };

    struct SectionEntryDisk {
        std::uint32_t id = 0;
        std::uint8_t codec = 0;
        std::uint64_t offset = 0;
        std::uint64_t encoded_size = 0;
        std::uint64_t decoded_size = 0;
    };

    struct ZstdChunkEntryDisk {
        std::uint64_t offset = 0;
        std::uint64_t encoded_size = 0;
        std::uint64_t decoded_offset = 0;
        std::uint64_t decoded_size = 0;
    };

    struct ZstdChunkTableData {
        std::uint64_t target_chunk_bytes = 0;
        std::vector<ZstdChunkEntryDisk> chunks;
    };

    struct SectionChunkDirectoryRecord {
        std::uint32_t section_id = 0;
        std::uint64_t section_offset = 0;
        std::uint64_t section_encoded_size = 0;
        std::uint64_t section_decoded_size = 0;
        std::uint64_t target_chunk_bytes = 0;
        std::vector<ZstdChunkEntryDisk> chunks;
    };

    struct ColumnarRuntimeHeaderDisk {
        std::uint64_t original_string_count = 0;
        std::uint64_t cut_count = 0;
        std::uint64_t component_count = 0;
        std::uint64_t graph_node_count = 0;
        std::uint64_t block_count = 0;
        std::uint64_t tree_node_count = 0;
        std::uint64_t virtual_edge_count = 0;
        std::uint64_t real_edge_count = 0;
        std::uint64_t graph_edge_count = 0;
        std::uint64_t bubble_edge_type_count = 0;
    };
