use spqr_rust::{build_spqr, Graph, NodeId};

fn main() {
    let edges = [
        (0, 1),
        (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
        (0, 4), (0, 5), (1, 4), (1, 5), (4, 5),
        (0, 6), (6, 7), (7, 1),
    ];
    let mut graph = Graph::with_capacity(8, edges.len());
    graph.add_nodes(8);
    for (u, v) in edges {
        graph.add_edge(NodeId(u), NodeId(v));
    }

    let mut result = build_spqr(&graph);
    result.tree.normalize();
    result.tree.compact();

    let report = spqr_rust::verify::verify_spqr_tree(&graph, &result.tree);
    assert!(report.is_ok());

    let (s, p, r) = result.tree.count_by_type();
    println!("S: {s}, P: {p}, R: {r}");
}
