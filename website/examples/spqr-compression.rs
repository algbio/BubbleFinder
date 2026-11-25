use spqr_rust::{EdgeId, Graph, NodeId};
use spqr_rust::sp_compress::{
    compress_and_build_spqr, CompressionInput, InputEdge,
};
use spqr_rust::sp_compress::reconstruct::reconstruct_from_compress_result;

fn main() {
    let pairs = [
        (0, 1),
        (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
        (0, 4), (0, 5), (1, 4), (1, 5), (4, 5),
        (0, 6), (6, 7), (7, 1),
    ];
    let mut graph = Graph::with_capacity(8, pairs.len());
    graph.add_nodes(8);
    let edges = pairs.iter().map(|&(u, v)| {
        let original_edge_id = graph.add_edge(NodeId(u), NodeId(v));
        InputEdge { u: NodeId(u), v: NodeId(v), original_edge_id }
    }).collect();

    let input = CompressionInput {
        n_nodes: 8,
        edges,
        contractible: vec![1; 8],
    };
    let compressed = compress_and_build_spqr(&input);
    let tree = reconstruct_from_compress_result(&compressed);
    assert!(spqr_rust::verify::verify_spqr_tree(&graph, &tree).is_ok());
    assert_eq!(tree.count_by_type(), (1, 1, 2));

    let stats = compressed.stats();
    println!("Core: {} vertices, {} edges", stats.core_nodes, stats.core_edges_count);
    let mut recovered = Vec::<EdgeId>::new();
    for edge in &compressed.macro_tree.core_edges {
        compressed.macro_tree.for_each_original_edge(
            edge.child, &mut |id| recovered.push(id));
    }
    recovered.sort_by_key(|id| id.0);
    assert_eq!(recovered, (0..pairs.len() as u32).map(EdgeId).collect::<Vec<_>>());
    println!("Recovered {} original edges", recovered.len());
}
