/**
 * @file spqr_tree.h
 * @brief C API for SPQR Tree Library
 */

#ifndef SPQR_TREE_H
#define SPQR_TREE_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPQR_INVALID UINT32_MAX
#define SPQR_INVALID_U64 UINT64_MAX

#define SPQR_NODE_TYPE_S 0
#define SPQR_NODE_TYPE_P 1
#define SPQR_NODE_TYPE_R 2

typedef struct SpqrGraphFFI SpqrGraphFFI;
typedef struct SpqrCCResult SpqrCCResult;
typedef struct SpqrBCTreeFFI SpqrBCTreeFFI;
typedef struct SpqrResult SpqrResult;
typedef struct SpqrTree SpqrTree;
typedef struct SpqrGraphFFI64 SpqrGraphFFI64;
typedef struct SpqrResult64 SpqrResult64;
typedef struct SpqrTree64 SpqrTree64;

typedef struct SkeletonEdge {
	uint32_t src;
	uint32_t dst;
	uint32_t real_edge;        // SPQR_INVALID if virtual
	uint32_t virtual_id;
	uint32_t twin_tree_node;
	uint32_t twin_edge_idx;
} SkeletonEdge;

typedef struct SkeletonEdge64 {
	uint64_t src;
	uint64_t dst;
	uint64_t real_edge;
	uint64_t virtual_id;
	uint64_t twin_tree_node;
	uint64_t twin_edge_idx;
} SkeletonEdge64;

typedef struct SkeletonEdgeInfo {
	uint32_t src;
	uint32_t dst;
	uint32_t real_edge;
	uint32_t twin_tree_node;
	bool is_virtual;
} SkeletonEdgeInfo;

typedef struct SkeletonEdgeInfo64 {
	uint64_t src;
	uint64_t dst;
	uint64_t real_edge;
	uint64_t twin_tree_node;
	bool is_virtual;
} SkeletonEdgeInfo64;


SpqrGraphFFI64* spqr_graph_new_u64(uint64_t node_capacity, uint64_t edge_capacity);
void spqr_graph_free_u64(SpqrGraphFFI64* graph);
uint64_t spqr_graph_add_nodes_u64(SpqrGraphFFI64* graph, uint64_t count);
uint64_t spqr_graph_add_edge_u64(SpqrGraphFFI64* graph, uint64_t u, uint64_t v);
SpqrGraphFFI64* spqr_graph_from_arrays_u64(uint64_t num_nodes,
                                           const uint64_t* src,
                                           const uint64_t* dst,
                                           uint64_t num_edges);
uint64_t spqr_graph_num_nodes_u64(const SpqrGraphFFI64* graph);
uint64_t spqr_graph_num_edges_u64(const SpqrGraphFFI64* graph);
uint64_t spqr_graph_edge_src_u64(const SpqrGraphFFI64* graph, uint64_t edge_id);
uint64_t spqr_graph_edge_dst_u64(const SpqrGraphFFI64* graph, uint64_t edge_id);
uint64_t spqr_graph_degree_u64(const SpqrGraphFFI64* graph, uint64_t node);
uint64_t spqr_graph_adj_cursor_u64(const SpqrGraphFFI64* graph, uint64_t node);
bool spqr_graph_adj_next_u64(const SpqrGraphFFI64* graph,
                             uint64_t cursor,
                             uint64_t* out_neighbor,
                             uint64_t* out_edge,
                             uint64_t* out_next_cursor);
typedef bool (*NeighborCallback64)(uint64_t neighbor_node, uint64_t edge_id, void* user_data);
void spqr_graph_for_each_neighbor_u64(const SpqrGraphFFI64* graph,
                                      uint64_t node,
                                      NeighborCallback64 callback,
                                      void* user_data);
uint64_t spqr_graph_neighbors_to_buffer_u64(const SpqrGraphFFI64* graph,
                                            uint64_t node,
                                            uint64_t* nodes_out,
                                            uint64_t* edges_out,
                                            uint64_t buffer_size);

SpqrResult64* spqr_build_u64(const SpqrGraphFFI64* graph);
void spqr_result_free_u64(SpqrResult64* result);
const SpqrTree64* spqr_result_tree_u64(const SpqrResult64* result);
const uint64_t* spqr_result_self_loops_u64(const SpqrResult64* result, uint64_t* out_len);

uint64_t spqr_tree_len_u64(const SpqrTree64* tree);
uint64_t spqr_tree_root_u64(const SpqrTree64* tree);
uint8_t spqr_tree_node_type_u64(const SpqrTree64* tree, uint64_t node_id);
uint64_t spqr_tree_node_parent_u64(const SpqrTree64* tree, uint64_t node_id);
uint64_t spqr_tree_node_children_copy_u64(const SpqrTree64* tree,
                                          uint64_t node_id,
                                          uint64_t* out_children,
                                          uint64_t out_capacity);
uint64_t spqr_tree_skeleton_num_edges_u64(const SpqrTree64* tree, uint64_t node_id);
uint64_t spqr_tree_skeleton_num_nodes_u64(const SpqrTree64* tree, uint64_t node_id);
void spqr_tree_skeleton_poles_u64(const SpqrTree64* tree,
                                  uint64_t node_id,
                                  uint64_t* pole1,
                                  uint64_t* pole2);
void spqr_tree_skeleton_edge_u64(const SpqrTree64* tree,
                                 uint64_t node_id,
                                 uint64_t edge_idx,
                                 SkeletonEdgeInfo64* out);
uint64_t spqr_tree_skeleton_original_node_u64(const SpqrTree64* tree,
                                              uint64_t tree_node_id,
                                              uint64_t local_node);
uint64_t spqr_tree_node_of_edge_u64(const SpqrTree64* tree, uint64_t edge_id);
uint64_t spqr_tree_edge_mapping_copy_u64(const SpqrTree64* tree,
                                         uint64_t* out_tree_nodes,
                                         uint64_t out_capacity);
const uint64_t* spqr_tree_edge_mapping_raw_u64(const SpqrTree64* tree, uint64_t* out_len);
void spqr_tree_edge_mapping_bulk_u64(const SpqrTree64* tree,
                                     uint64_t num_edges,
                                     uint64_t* out_tree_nodes);
void spqr_tree_normalize_u64(SpqrTree64* tree);
void spqr_tree_compact_u64(SpqrTree64* tree);
void spqr_tree_count_by_type_u64(const SpqrTree64* tree,
                                 uint64_t* s_count,
                                 uint64_t* p_count,
                                 uint64_t* r_count);
void spqr_tree_get_sizes_u64(const SpqrTree64* tree,
                             uint64_t* out_num_nodes,
                             uint64_t* out_total_children,
                             uint64_t* out_total_skeleton_edges);
void spqr_tree_bulk_export_u64(const SpqrTree64* tree,
                               uint8_t* node_types,
                               uint64_t* node_parents,
                               uint64_t* children_offsets,
                               uint64_t* children,
                               uint64_t* skeleton_offsets,
                               uint64_t* skeleton_src,
                               uint64_t* skeleton_dst,
                               uint64_t* skeleton_real_edge,
                               uint8_t* skeleton_is_virtual);
void spqr_tree_bulk_export_node_mapping_u64(const SpqrTree64* tree,
                                            uint64_t* node_mapping_offsets,
                                            uint64_t* node_mapping);
void spqr_tree_info_u64(const SpqrTree64* tree, uint64_t* out_num_nodes, uint64_t* out_root);
const uint8_t* spqr_tree_node_types_raw_u64(const SpqrTree64* tree);
const uint64_t* spqr_tree_node_parents_raw_u64(const SpqrTree64* tree);
const uint64_t* spqr_tree_children_offsets_raw_u64(const SpqrTree64* tree);
const uint64_t* spqr_tree_children_raw_u64(const SpqrTree64* tree, uint64_t* out_len);
const uint64_t* spqr_tree_skeleton_offsets_raw_u64(const SpqrTree64* tree);
const SkeletonEdge64* spqr_tree_skeleton_edges_raw_u64(const SpqrTree64* tree, uint64_t* out_len);
void spqr_tree_node_mapping_raw_u64(const SpqrTree64* tree,
                                    const uint64_t** out_offsets,
                                    const uint64_t** out_mapping,
                                    uint64_t* out_mapping_len);
const uint64_t* spqr_tree_skeleton_num_nodes_raw_u64(const SpqrTree64* tree);

SpqrGraphFFI* spqr_graph_new(uint32_t node_capacity, uint32_t edge_capacity);
void spqr_graph_free(SpqrGraphFFI* graph);

uint64_t spqr_get_fast_cycle_hits(void);
uint64_t spqr_get_fast_cycle_calls(void);

void spqr_set_canonicalize_root_enabled(uint8_t enabled);
uint8_t spqr_get_canonicalize_root_enabled(void);

// returns ID of first added node
uint32_t spqr_graph_add_nodes(SpqrGraphFFI* graph, uint32_t count);
uint32_t spqr_graph_add_edge(SpqrGraphFFI* graph, uint32_t u, uint32_t v);

// edges array contains pairs like: [u0, v0, u1, v1, ..]
void spqr_graph_add_edges_batch(SpqrGraphFFI* graph, const uint32_t* edges, uint32_t count);
SpqrGraphFFI* spqr_graph_from_edges(uint32_t num_nodes, const uint32_t* edges, uint32_t num_edges);

SpqrGraphFFI* spqr_graph_from_arrays(uint32_t num_nodes,
                              const uint32_t* src,
                              const uint32_t* dst,
                              uint32_t num_edges);

uint32_t spqr_graph_num_nodes(const SpqrGraphFFI* graph);
uint32_t spqr_graph_num_edges(const SpqrGraphFFI* graph);
uint32_t spqr_graph_edge_src(const SpqrGraphFFI* graph, uint32_t edge_id);
uint32_t spqr_graph_edge_dst(const SpqrGraphFFI* graph, uint32_t edge_id);
uint32_t spqr_graph_degree(const SpqrGraphFFI* graph, uint32_t node);

uint32_t spqr_graph_adj_cursor(const SpqrGraphFFI* graph, uint32_t node);
bool spqr_graph_adj_next(const SpqrGraphFFI* graph,
                         uint32_t cursor,
                         uint32_t* out_neighbor,
                         uint32_t* out_edge,
                         uint32_t* out_next_cursor);

typedef bool (*NeighborCallback)(uint32_t neighbor_node, uint32_t edge_id, void* user_data);

void spqr_graph_for_each_neighbor(const SpqrGraphFFI* graph,
                                  uint32_t node,
                                  NeighborCallback callback,
                                  void* user_data);

uint32_t spqr_graph_neighbors_to_buffer(const SpqrGraphFFI* graph,
                                        uint32_t node,
                                        uint32_t* nodes_out,
                                        uint32_t* edges_out,
                                        uint32_t buffer_size);

// CC

SpqrCCResult* spqr_connected_components(const SpqrGraphFFI* graph);
void spqr_cc_free(SpqrCCResult* cc);
uint32_t spqr_cc_count(const SpqrCCResult* cc);
uint32_t spqr_cc_component_of(const SpqrCCResult* cc, uint32_t node);

// raw pointer, valid until cc freed
const uint32_t* spqr_cc_components_raw(const SpqrCCResult* cc, uint32_t* out_len);

uint32_t spqr_cc_count_in(const SpqrCCResult* cc, uint32_t component_id);

typedef bool (*NodeCallback)(uint32_t node_id, void* user_data);

void spqr_cc_for_each_in(const SpqrCCResult* cc,
                         uint32_t component_id,
                         NodeCallback callback,
                         void* user_data);

// BC tree
SpqrBCTreeFFI* spqr_bc_tree_build(const SpqrGraphFFI* graph);
void spqr_bc_tree_free(SpqrBCTreeFFI* bc);

uint32_t spqr_bc_num_blocks(const SpqrBCTreeFFI* bc);
uint32_t spqr_bc_num_cut_vertices(const SpqrBCTreeFFI* bc);
bool spqr_bc_is_biconnected(const SpqrBCTreeFFI* bc);  // single block, no cut vertices
bool spqr_bc_is_cut_vertex(const SpqrBCTreeFFI* bc, uint32_t node);

const uint32_t* spqr_bc_block_nodes(const SpqrBCTreeFFI* bc, uint32_t block_idx, uint32_t* out_len);
const uint32_t* spqr_bc_block_edges(const SpqrBCTreeFFI* bc, uint32_t block_idx, uint32_t* out_len);
const uint32_t* spqr_bc_cut_vertices(const SpqrBCTreeFFI* bc, uint32_t* out_len);

typedef struct {
	uint32_t node_start;
	uint32_t node_count;
	uint32_t edge_start;
	uint32_t edge_count;
} BCBlock;

// zero copy access to internal arrays
const BCBlock* spqr_bc_blocks_raw(const SpqrBCTreeFFI* bc, uint32_t* out_num_blocks);
const uint32_t* spqr_bc_nodes_flat_raw(const SpqrBCTreeFFI* bc, uint32_t* out_len);
const uint32_t* spqr_bc_edges_flat_raw(const SpqrBCTreeFFI* bc, uint32_t* out_len);

void spqr_bc_get_sizes(const SpqrBCTreeFFI* bc,
                       uint32_t* out_num_blocks,
                       uint32_t* out_total_nodes,
                       uint32_t* out_total_edges);

// copies data (use raw functions above if you don't need owned copies)
void spqr_bc_bulk_export(const SpqrBCTreeFFI* bc,
                         uint32_t* block_node_offsets,
                         uint32_t* block_nodes,
                         uint32_t* block_edge_offsets,
                         uint32_t* block_edges);

// SPQR tree

SpqrResult* spqr_build(const SpqrGraphFFI* graph);  // graph should be biconnected
void spqr_result_free(SpqrResult* result);

const SpqrTree* spqr_result_tree(const SpqrResult* result);
const uint32_t* spqr_result_self_loops(const SpqrResult* result, uint32_t* out_len);

uint32_t spqr_tree_len(const SpqrTree* tree);
uint32_t spqr_tree_root(const SpqrTree* tree);

// returns SPQR_NODE_TYPE_S/P/R
uint8_t spqr_tree_node_type(const SpqrTree* tree, uint32_t node_id);

// SPQR_INVALID if root
uint32_t spqr_tree_node_parent(const SpqrTree* tree, uint32_t node_id);

const uint32_t* spqr_tree_node_children(const SpqrTree* tree,
                                        uint32_t node_id,
                                        uint32_t* out_len);

uint32_t spqr_tree_skeleton_num_edges(const SpqrTree* tree, uint32_t node_id);
uint32_t spqr_tree_skeleton_num_nodes(const SpqrTree* tree, uint32_t node_id);

// poles = separation pair (original node IDs)
void spqr_tree_skeleton_poles(const SpqrTree* tree,
                              uint32_t node_id,
                              uint32_t* pole1,
                              uint32_t* pole2);

void spqr_tree_skeleton_edge(const SpqrTree* tree,
                             uint32_t node_id,
                             uint32_t edge_idx,
                             SkeletonEdgeInfo* out);

uint32_t spqr_tree_skeleton_original_node(const SpqrTree* tree,
                                          uint32_t tree_node_id,
                                          uint32_t local_node);

uint32_t spqr_tree_node_of_edge(const SpqrTree* tree, uint32_t edge_id);

// result[edge_id] = tree_node_id, SPQR_INVALID for self-loops
const uint32_t* spqr_tree_edge_mapping_raw(const SpqrTree* tree, uint32_t* out_len);

void spqr_tree_edge_mapping_bulk(const SpqrTree* tree,
                                 uint32_t num_edges,
                                 uint32_t* out_tree_nodes);

void spqr_tree_normalize(SpqrTree* tree); // merge adjacent S-S and P-P
void spqr_tree_compact(SpqrTree* tree); // call after normalize()

void spqr_tree_count_by_type(const SpqrTree* tree,
                             uint32_t* s_count,
                             uint32_t* p_count,
                             uint32_t* r_count);

/* call get_sizes first to know how much to allocate */

void spqr_tree_get_sizes(const SpqrTree* tree,
                         uint32_t* out_num_nodes,
                         uint32_t* out_total_children,
                         uint32_t* out_total_skeleton_edges);

void spqr_tree_bulk_export(const SpqrTree* tree,
                           uint8_t* node_types,
                           uint32_t* node_parents,
                           uint32_t* children_offsets,     // CSR
                           uint32_t* children,
                           uint32_t* skeleton_offsets,     // CSR
                           uint32_t* skeleton_src,
                           uint32_t* skeleton_dst,
                           uint32_t* skeleton_real_edge,
                           uint8_t* skeleton_is_virtual);

void spqr_tree_bulk_export_node_mapping(const SpqrTree* tree,
                                        uint32_t* node_mapping_offsets,
                                        uint32_t* node_mapping);

/*
 * Zero-copy access to internal flat arrays.
 * Pointers valid until tree is free
 */

void spqr_tree_info(const SpqrTree* tree, uint32_t* out_num_nodes, uint32_t* out_root);

const uint8_t* spqr_tree_node_types_raw(const SpqrTree* tree);
const uint32_t* spqr_tree_node_parents_raw(const SpqrTree* tree);
const uint32_t* spqr_tree_children_offsets_raw(const SpqrTree* tree);
const uint32_t* spqr_tree_children_raw(const SpqrTree* tree, uint32_t* out_len);
const uint32_t* spqr_tree_skeleton_offsets_raw(const SpqrTree* tree);
const SkeletonEdge* spqr_tree_skeleton_edges_raw(const SpqrTree* tree, uint32_t* out_len);

void spqr_tree_node_mapping_raw(const SpqrTree* tree,
                                const uint32_t** out_offsets,
                                const uint32_t** out_mapping,
                                uint32_t* out_mapping_len);

const uint32_t* spqr_tree_skeleton_num_nodes_raw(const SpqrTree* tree);

char* spqr_format_to_string(const SpqrGraphFFI* graph, const SpqrResult* result);
void spqr_string_free(char* s);

#ifdef __cplusplus
}

#include <memory>
#include <stdexcept>

namespace spqr_ffi {

struct GraphDeleter { void operator()(SpqrGraphFFI* p) const { spqr_graph_free(p); } };
struct CCResultDeleter { void operator()(SpqrCCResult* p) const { spqr_cc_free(p); } };
struct BCTreeDeleter { void operator()(SpqrBCTreeFFI* p) const { spqr_bc_tree_free(p); } };
struct SpqrResultDeleter { void operator()(SpqrResult* p) const { spqr_result_free(p); } };
struct StringDeleter { void operator()(char* p) const { spqr_string_free(p); } };

using GraphPtr = std::unique_ptr<SpqrGraphFFI, GraphDeleter>;
using CCResultPtr = std::unique_ptr<SpqrCCResult, CCResultDeleter>;
using BCTreePtr = std::unique_ptr<SpqrBCTreeFFI, BCTreeDeleter>;
using SpqrResultPtr = std::unique_ptr<SpqrResult, SpqrResultDeleter>;
using StringPtr = std::unique_ptr<char, StringDeleter>;

inline GraphPtr make_graph(uint32_t node_capacity=0, uint32_t edge_capacity=0) {
    return GraphPtr(spqr_graph_new(node_capacity, edge_capacity));
}
inline SpqrResultPtr build_spqr(const SpqrGraphFFI* graph) {
    return SpqrResultPtr(spqr_build(graph));
}
inline BCTreePtr build_bc_tree(const SpqrGraphFFI* graph) {
    return BCTreePtr(spqr_bc_tree_build(graph));
}
inline CCResultPtr compute_cc(const SpqrGraphFFI* graph) {
    return CCResultPtr(spqr_connected_components(graph));
}

}

#endif
#endif
