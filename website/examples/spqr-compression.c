#include "sp_compress.h"
#include <assert.h>
#include <stdio.h>

int main(void) {
    const SpCompressInputEdge edges[] = {
        {0, 1, 0}, {0, 2, 1}, {0, 3, 2}, {1, 2, 3}, {1, 3, 4},
        {2, 3, 5}, {0, 4, 6}, {0, 5, 7}, {1, 4, 8}, {1, 5, 9},
        {4, 5, 10}, {0, 6, 11}, {6, 7, 12}, {7, 1, 13}
    };
    const uint8_t contractible[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    SpCompressHandle* handle = sp_compress_ffi(
        8, edges, 14, 13, contractible, 8, 1);
    assert(handle && sp_compress_success(handle));
    SpCompressTreeView view = sp_compress_get_tree(handle);
    assert(view.core_edges_len > 0);
    printf("Core edges: %llu\n", (unsigned long long)view.core_edges_len);
    sp_compress_free(handle);

    SpqrResult* result = sp_compress_reconstruct_ffi(
        8, edges, 14, 13, contractible, 8);
    assert(result);
    const SpqrTree* tree = spqr_result_tree(result);
    uint32_t s, p, r;
    spqr_tree_count_by_type(tree, &s, &p, &r);
    assert(s == 1 && p == 1 && r == 2);
    printf("S: %u, P: %u, R: %u\n", s, p, r);
    spqr_result_free(result);
    return 0;
}
