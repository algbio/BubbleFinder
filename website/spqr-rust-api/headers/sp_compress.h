#ifndef SP_COMPRESS_H
#define SP_COMPRESS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "spqr_tree.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SpCompressHandle SpCompressHandle;

typedef struct {
    uint32_t u;
    uint32_t v;
    uint32_t original_edge_id;
} SpCompressInputEdge;

typedef struct {
    uint64_t u;
    uint64_t v;
    uint64_t original_edge_id;
} SpCompressInputEdge64;

typedef uint64_t SpCompressChildRef;
typedef uint64_t SpCompressSize;

#define SP_COMPRESS_KIND_SERIES   1
#define SP_COMPRESS_KIND_PARALLEL 2


typedef struct {
    uint8_t  kind;
    uint8_t  _pad[3];
    uint32_t left;
    uint32_t right;
    SpCompressSize children_offset;
    SpCompressSize children_count;
} SpCompressNode;

typedef struct {
    uint32_t u;
    uint32_t v;
    SpCompressChildRef child;
} SpCompressCoreEdge;

typedef struct {
    uint8_t  kind;
    uint8_t  _pad[3];
    uint64_t left;
    uint64_t right;
    uint64_t children_offset;
    uint64_t children_count;
} SpCompressNode64;

typedef struct {
    uint64_t u;
    uint64_t v;
    uint64_t child;
} SpCompressCoreEdge64;


#define SP_COMPRESS_TAG_BIT      UINT64_C(0x8000000000000000)
#define SP_COMPRESS_PAYLOAD_MASK UINT64_C(0x7FFFFFFFFFFFFFFF)

#define SP_COMPRESS_CHILD_IS_MACRO(c)  (((c) & SP_COMPRESS_TAG_BIT) != 0)
#define SP_COMPRESS_CHILD_IS_EDGE(c)   (((c) & SP_COMPRESS_TAG_BIT) == 0)
#define SP_COMPRESS_CHILD_AS_EDGE(c)   ((uint32_t)(c))
#define SP_COMPRESS_CHILD_AS_MACRO(c)  ((SpCompressSize)((c) & SP_COMPRESS_PAYLOAD_MASK))

#define SP_COMPRESS_TAG_BIT64      0x8000000000000000ull
#define SP_COMPRESS_PAYLOAD_MASK64 0x7FFFFFFFFFFFFFFFull

#define SP_COMPRESS_CHILD_IS_MACRO64(c)  (((c) & SP_COMPRESS_TAG_BIT64) != 0)
#define SP_COMPRESS_CHILD_IS_EDGE64(c)   (((c) & SP_COMPRESS_TAG_BIT64) == 0)
#define SP_COMPRESS_CHILD_AS_EDGE64(c)   ((c))
#define SP_COMPRESS_CHILD_AS_MACRO64(c)  ((c) & SP_COMPRESS_PAYLOAD_MASK64)

typedef struct {
    uint32_t input_nodes;
    uint32_t input_edges;
    uint32_t core_nodes;
    uint32_t core_edges_count;
    uint32_t macro_count;
    uint32_t macro_series;
    uint32_t macro_parallel;
    uint32_t series_reductions;
    uint32_t parallel_reductions;
    uint32_t iterations;
    uint8_t  fully_sp_reducible;
} SpCompressStats;

typedef struct {

    const SpCompressNode* macros_ptr;
    SpCompressSize macros_len;

    const SpCompressChildRef* children_ptr;
    SpCompressSize children_len;

    const SpCompressCoreEdge* core_edges_ptr;
    SpCompressSize core_edges_len;

    const uint32_t* core_nodes_ptr;
    SpCompressSize core_nodes_len;

    const uint32_t* input_endpoints_ptr;
    SpCompressSize input_endpoints_len;


    SpCompressStats stats;
} SpCompressTreeView;


typedef struct {
    uint64_t input_nodes;
    uint64_t input_edges;
    uint64_t core_nodes;
    uint64_t core_edges_count;
    uint64_t macro_count;
    uint64_t macro_series;
    uint64_t macro_parallel;
    uint64_t series_reductions;
    uint64_t parallel_reductions;
    uint64_t iterations;
    uint8_t  fully_sp_reducible;
} SpCompressStats64;

typedef struct {

    const SpCompressNode64* macros_ptr;
    uint64_t macros_len;

    const uint64_t* children_ptr;
    uint64_t children_len;

    const SpCompressCoreEdge64* core_edges_ptr;
    uint64_t core_edges_len;

    const uint64_t* core_nodes_ptr;
    uint64_t core_nodes_len;

    const uint64_t* input_endpoints_ptr;
    uint64_t input_endpoints_len;


    SpCompressStats64 stats;
} SpCompressTreeView64;


typedef struct {
    uint32_t root;
    uint32_t node_count;

    const uint8_t* node_types_ptr;
    const uint32_t* node_parents_ptr;

    const uint32_t* children_offsets_ptr;
    uint32_t children_offsets_len;
    const uint32_t* children_ptr;
    uint32_t children_len;

    const uint32_t* skeleton_offsets_ptr;
    uint32_t skeleton_offsets_len;
    const SkeletonEdge* skeleton_edges_ptr;
    uint32_t skeleton_edges_len;

    const uint32_t* node_mapping_offsets_ptr;
    uint32_t node_mapping_offsets_len;
    const uint32_t* node_mapping_ptr;
    uint32_t node_mapping_len;

    const uint32_t* skeleton_num_nodes_ptr;
    uint32_t skeleton_num_nodes_len;
} SpCompressCoreSpqrSnapshot;

typedef struct {
    const SpCompressNode* macros_ptr;
    SpCompressSize macros_len;

    const SpCompressChildRef* children_ptr;
    SpCompressSize children_len;

    const SpCompressCoreEdge* core_edges_ptr;
    SpCompressSize core_edges_len;

    const uint32_t* core_nodes_ptr;
    SpCompressSize core_nodes_len;

    const uint32_t* input_endpoints_ptr;
    SpCompressSize input_endpoints_len;

    SpCompressStats stats;

    const SpCompressCoreSpqrSnapshot* core_spqr;
    const uint32_t* core_node_inv_ptr;
    uint32_t core_node_inv_len;
} SpCompressSnapshotView;

SpCompressHandle* sp_compress_ffi(
    uint32_t n_nodes,
    const SpCompressInputEdge* edges_ptr,
    uint32_t edges_len,
    uint32_t max_original_edge_id,
    const uint8_t* contractible_ptr,
    uint32_t contractible_len,
    uint8_t build_core_spqr
);

SpCompressHandle* sp_compress_ffi64(
    uint64_t n_nodes,
    const SpCompressInputEdge64* edges_ptr,
    uint64_t edges_len,
    const uint8_t* contractible_ptr,
    uint64_t contractible_len,
    uint8_t build_core_spqr
);

SpCompressHandle* sp_compress_ffi_u64(
    uint64_t n_nodes,
    const SpCompressInputEdge64* edges_ptr,
    uint64_t edges_len,
    const uint8_t* contractible_ptr,
    uint64_t contractible_len,
    uint8_t build_core_spqr
);

SpCompressHandle* sp_compress_from_snapshot_ffi(
    const SpCompressSnapshotView* snapshot
);

void sp_compress_free(SpCompressHandle* handle);

uint8_t sp_compress_success(const SpCompressHandle* handle);

SpCompressTreeView sp_compress_get_tree(const SpCompressHandle* handle);
SpCompressTreeView64 sp_compress_get_tree_u64(const SpCompressHandle* handle);

const SpqrTree* sp_compress_get_core_spqr(const SpCompressHandle* handle);
const SpqrTree64* sp_compress_get_core_spqr_u64(const SpCompressHandle* handle);

const uint32_t* sp_compress_core_node_inv(const SpCompressHandle* handle, uint32_t* out_len);
const uint64_t* sp_compress_core_node_inv_u64(const SpCompressHandle* handle, uint64_t* out_len);

struct SpqrResult* sp_compress_reconstruct_ffi(
    uint32_t n_nodes,
    const SpCompressInputEdge* edges_ptr,
    uint32_t edges_len,
    uint32_t max_original_edge_id,
    const uint8_t* contractible_ptr,
    uint32_t contractible_len);

struct SpqrResult64* sp_compress_reconstruct_ffi64(
    uint64_t n_nodes,
    const SpCompressInputEdge64* edges_ptr,
    uint64_t edges_len,
    const uint8_t* contractible_ptr,
    uint64_t contractible_len);

struct SpqrResult64* sp_compress_reconstruct_ffi_u64(
    uint64_t n_nodes,
    const SpCompressInputEdge64* edges_ptr,
    uint64_t edges_len,
    const uint8_t* contractible_ptr,
    uint64_t contractible_len);

typedef struct {
    uint64_t t_compress_us;

    uint64_t t_build_spqr_core_us;
    uint64_t t_reconstruct_us;
    uint64_t t_normalize_us;

    uint64_t t_canonicalize_us;
    uint64_t t_canon_root_us;
    uint64_t t_canon_node_order_us;
    uint64_t t_canon_edge_orient_us;
    uint64_t t_canon_move_root_us;

    uint64_t t_reconstruct_build_builder_us;
    uint64_t t_reconstruct_normalize_in_place_us;
    uint64_t t_reconstruct_finalize_us;
    uint64_t t_reconstruct_defensive_normalize_us;

    uint64_t t_core_remap_us;
    uint64_t t_core_graph_build_us;
    uint64_t t_core_spqr_raw_us;
    uint64_t t_handle_wrap_us;

    uint64_t t_total_us;

    uint64_t t_compress_input_edges_us;
    uint64_t t_compress_init_work_us;
    uint64_t t_compress_init_dirty_us;

    uint64_t t_compress_reduce_series_us;
    uint64_t t_compress_reduce_parallel_us;

    uint64_t t_compress_materialize_us;
    uint64_t t_compress_cleanup_us;
    uint64_t t_compress_canon_series_us;
    uint64_t t_compress_sort_core_edges_us;

    uint64_t t_compress_collect_core_nodes_us;
    uint64_t t_compress_stats_shrink_us;

    uint64_t t_spqr_self_loop_scan_us;
    uint64_t t_spqr_precheck_us;
    uint64_t t_spqr_split_multi_edges_us;
    uint64_t t_spqr_work_graph_us;

    uint64_t t_spqr_triconn_us;
    uint64_t t_spqr_relabel_us;
    uint64_t t_spqr_combine_us;
    uint64_t t_spqr_merge_us;

    uint64_t t_spqr_assemble_us;
    uint64_t t_spqr_tree_total_us;

    uint64_t c_spqr_multi_components;
    uint64_t c_spqr_triconn_components;
    uint64_t c_spqr_precombine_components;
    uint64_t c_spqr_combined_components;
    uint64_t c_spqr_merged_components;
    uint64_t c_spqr_merged_real_edges;
    uint64_t c_spqr_merged_virtual_incidences;
    uint64_t c_spqr_virtual_id_span;
    uint64_t c_spqr_tree_nodes;
    uint64_t c_spqr_tree_edges;
    uint64_t c_spqr_tree_skeleton_edges;
    uint64_t c_spqr_tree_virtual_incidences;
} SpCompressTimings;

SpCompressHandle* sp_compress_timed_ffi(
    uint32_t n_nodes,
    const SpCompressInputEdge* edges_ptr,
    uint32_t edges_len,
    uint32_t max_original_edge_id,
    const uint8_t* contractible_ptr,
    uint32_t contractible_len,
    uint8_t build_core_spqr,
    SpCompressTimings* out_timings);

SpCompressHandle* sp_compress_timed_ffi64(
    uint64_t n_nodes,
    const SpCompressInputEdge64* edges_ptr,
    uint64_t edges_len,
    const uint8_t* contractible_ptr,
    uint64_t contractible_len,
    uint8_t build_core_spqr,
    SpCompressTimings* out_timings);

SpCompressHandle* sp_compress_timed_ffi_u64(
    uint64_t n_nodes,
    const SpCompressInputEdge64* edges_ptr,
    uint64_t edges_len,
    const uint8_t* contractible_ptr,
    uint64_t contractible_len,
    uint8_t build_core_spqr,
    SpCompressTimings* out_timings);

SpCompressHandle* sp_compress_indexed_ffi(
    uint32_t n_nodes,
    const uint32_t* src_ptr,
    const uint32_t* dst_ptr,
    uint32_t edges_len,
    const uint8_t* contractible_ptr,
    uint32_t contractible_len,
    uint8_t build_core_spqr);

SpCompressHandle* sp_compress_indexed_ffi_u64(
    uint64_t n_nodes,
    const uint64_t* src_ptr,
    const uint64_t* dst_ptr,
    uint64_t edges_len,
    const uint8_t* contractible_ptr,
    uint64_t contractible_len,
    uint8_t build_core_spqr);


struct SpqrResult* sp_compress_reconstruct_with_timings_ffi(
    uint32_t n_nodes,
    const SpCompressInputEdge* edges_ptr,
    uint32_t edges_len,
    uint32_t max_original_edge_id,
    const uint8_t* contractible_ptr,
    uint32_t contractible_len,
    SpCompressStats* out_stats,
    SpCompressTimings* out_timings);

struct SpqrResult64* sp_compress_reconstruct_with_timings_ffi64(
    uint64_t n_nodes,
    const SpCompressInputEdge64* edges_ptr,
    uint64_t edges_len,
    const uint8_t* contractible_ptr,
    uint64_t contractible_len,
    SpCompressStats64* out_stats,
    SpCompressTimings* out_timings);

struct SpqrResult64* sp_compress_reconstruct_with_timings_ffi_u64(
    uint64_t n_nodes,
    const SpCompressInputEdge64* edges_ptr,
    uint64_t edges_len,
    const uint8_t* contractible_ptr,
    uint64_t contractible_len,
    SpCompressStats64* out_stats,
    SpCompressTimings* out_timings);

#ifdef __cplusplus
}
#endif

#endif
