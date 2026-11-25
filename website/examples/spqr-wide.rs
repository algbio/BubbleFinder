use spqr_rust::wide::{build_spqr, Graph, NodeId};

fn main() {
    let pairs = [
        (0, 1),
        (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
        (0, 4), (0, 5), (1, 4), (1, 5), (4, 5),
        (0, 6), (6, 7), (7, 1),
    ];
    let mut graph = Graph::with_capacity(8, pairs.len());
    graph.add_nodes(8);
    for (u, v) in pairs {
        graph.add_edge(NodeId(u), NodeId(v));
    }
    let mut result = build_spqr(&graph);
    result.tree.normalize();
    result.tree.compact();
    assert_eq!(result.tree.count_by_type(), (1, 1, 2));
    println!("{:?}", result.tree.count_by_type());
}
