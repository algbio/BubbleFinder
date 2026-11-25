use std::collections::HashMap;
use spqr_rust::{build_spqr, Graph, NodeId};
use spqr_rust::biconnected::BCTree;
use spqr_rust::connected::connected_components;

fn main() {
    let mut graph = Graph::with_capacity(6, 6);
    graph.add_nodes(6);
    for (u, v) in [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)] {
        graph.add_edge(NodeId(u), NodeId(v));
    }
    let components = connected_components(&graph);
    let blocks = BCTree::build(&graph);
    assert_eq!(components.num_components, 2);
    assert_eq!(blocks.num_blocks(), 2);
    assert_eq!(blocks.num_cut_vertices(), 1);

    for block in 0..blocks.num_blocks() {
        let original_vertices = blocks.block_nodes(block);
        let map: HashMap<_, _> = original_vertices.iter().enumerate()
            .map(|(i, &v)| (v, NodeId(i as u32))).collect();
        let mut subgraph = Graph::with_capacity(
            original_vertices.len(), blocks.block_edges(block).len());
        subgraph.add_nodes(original_vertices.len());
        for &id in blocks.block_edges(block) {
            let edge = graph.edge(id);
            subgraph.add_edge(map[&edge.src], map[&edge.dst]);
        }
        let mut result = build_spqr(&subgraph);
        result.tree.normalize();
        result.tree.compact();
        assert!(spqr_rust::verify::verify_spqr_tree(&subgraph, &result.tree).is_ok());
        println!("Block {block}: {:?}", result.tree.count_by_type());
    }
}
