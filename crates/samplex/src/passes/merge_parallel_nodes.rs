// This code is a Qiskit project.
//
// (C) Copyright IBM 2025, 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Merge parallel nodes: sampling graph (IR3) → sampling graph (IR3), in place.
//!
//! Fuses same-generation nodes that share a predecessor, agree on their merge key and cover
//! disjoint qubits, into one wider node.

use hashbrown::{HashMap, HashSet};
use rustworkx_core::petgraph::stable_graph::NodeIndex;
use rustworkx_core::petgraph::visit::EdgeRef;
use rustworkx_core::petgraph::Direction as PetDirection;

use qiskit_circuit::standard_gate::StandardGate;

use crate::partition::Partition;
use crate::virtual_flow_graph::{Direction, Measure, Node, NodeKind, Propagate, VirtualFlowGraph};

use super::utils::topological_generations;

#[derive(PartialEq, Eq, Hash)]
enum MergeKey {
    /// Handedness is part of the key: two conjugations of the same gate in opposite directions are
    /// different operations, so they must not be fused into one wider node.
    Propagate {
        gate: StandardGate,
        direction: Direction,
    },
    Measure,
    Reset,
}

fn merge_key(kind: &NodeKind) -> Option<MergeKey> {
    match kind {
        NodeKind::Propagate(p) => Some(MergeKey::Propagate {
            gate: p.gate,
            direction: p.direction,
        }),
        NodeKind::Measure(_) => Some(MergeKey::Measure),
        NodeKind::Reset => Some(MergeKey::Reset),
        _ => None,
    }
}

/// Merge parallel nodes throughout a sampling graph, in place.
pub fn merge_parallel_nodes(vfg: &mut VirtualFlowGraph) {
    let generations = topological_generations(&vfg.graph);

    for generation in generations {
        let mut key_groups: HashMap<MergeKey, Vec<NodeIndex>> = HashMap::new();
        for &idx in &generation {
            if let Some(key) = merge_key(&vfg.graph[idx].kind) {
                key_groups.entry(key).or_default().push(idx);
            }
        }

        for (_key, group) in key_groups {
            if group.len() < 2 {
                continue;
            }

            // Greedy clustering
            let mut clusters: Vec<Vec<NodeIndex>> = Vec::new();
            let mut cluster_elements: Vec<HashSet<usize>> = Vec::new();
            let mut cluster_predecessors: Vec<HashSet<NodeIndex>> = Vec::new();

            for &idx in &group {
                let node_elements = vfg.graph[idx].partition.all_elements().clone();
                let preds: HashSet<NodeIndex> = vfg
                    .graph
                    .neighbors_directed(idx, PetDirection::Incoming)
                    .collect();

                let is_predecessorless_reset =
                    preds.is_empty() && matches!(vfg.graph[idx].kind, NodeKind::Reset);

                let mut merged = false;
                for (ci, cluster) in clusters.iter_mut().enumerate() {
                    let disjoint = cluster_elements[ci].is_disjoint(&node_elements);
                    let shared_pred =
                        !cluster_predecessors[ci].is_disjoint(&preds) || is_predecessorless_reset;

                    if disjoint && shared_pred {
                        cluster.push(idx);
                        cluster_elements[ci].extend(&node_elements);
                        cluster_predecessors[ci].extend(&preds);
                        merged = true;
                        break;
                    }
                }

                if !merged {
                    clusters.push(vec![idx]);
                    cluster_elements.push(node_elements);
                    cluster_predecessors.push(preds);
                }
            }

            // Merge clusters with 2+ nodes
            for cluster in clusters {
                if cluster.len() < 2 {
                    continue;
                }

                let merged_node = build_merged_node(
                    &cluster.iter().map(|&idx| &vfg.graph[idx]).collect::<Vec<_>>(),
                );

                let merged_idx = vfg.graph.add_node(merged_node);

                rewire_edges(vfg, &cluster, merged_idx);

                // Remove old nodes
                for &old_idx in &cluster {
                    vfg.graph.remove_node(old_idx);
                }
            }
        }
    }
}

fn rewire_edges(
    vfg: &mut VirtualFlowGraph,
    cluster: &[NodeIndex],
    merged_idx: NodeIndex,
) {
    let mut seen_incoming: HashSet<NodeIndex> = HashSet::new();
    let mut seen_outgoing: HashSet<NodeIndex> = HashSet::new();

    for &old_idx in cluster {
        let incoming: Vec<_> = vfg
            .graph
            .edges_directed(old_idx, PetDirection::Incoming)
            .map(|e| (e.source(), *e.weight()))
            .collect();
        for (src, edge) in incoming {
            if cluster.contains(&src) {
                continue;
            }
            if seen_incoming.insert(src) {
                vfg.graph.add_edge(src, merged_idx, edge);
            }
        }

        let outgoing: Vec<_> = vfg
            .graph
            .edges_directed(old_idx, PetDirection::Outgoing)
            .map(|e| (e.target(), *e.weight()))
            .collect();
        for (tgt, edge) in outgoing {
            if cluster.contains(&tgt) {
                continue;
            }
            if seen_outgoing.insert(tgt) {
                vfg.graph.add_edge(merged_idx, tgt, edge);
            }
        }
    }
}

fn build_merged_node(nodes: &[&Node]) -> Node {
    let partitions: Vec<&Partition> = nodes.iter().map(|n| &n.partition).collect();
    let merged_partition = Partition::union(&partitions).unwrap();

    match &nodes[0].kind {
        NodeKind::Propagate(p) => Node {
            partition: merged_partition,
            kind: NodeKind::Propagate(Propagate {
                gate: p.gate,
                direction: p.direction,
            }),
        },
        NodeKind::Measure(_) => {
            let mut merged_clbits: Vec<usize> = Vec::new();
            for n in nodes {
                if let NodeKind::Measure(m) = &n.kind {
                    merged_clbits.extend(&m.clbit_indices);
                }
            }
            Node {
                partition: merged_partition,
                kind: NodeKind::Measure(Measure {
                    clbit_indices: merged_clbits,
                }),
            }
        }
        NodeKind::Reset => Node {
            partition: merged_partition,
            kind: NodeKind::Reset,
        },
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::passes::test_fixtures::*;
    use crate::virtual_flow_graph::*;

    fn propagate_node_with_gate(qubits: &[usize], gate: StandardGate) -> Node {
        propagate_node_with(qubits, gate, Direction::Right)
    }

    #[test]
    fn test_parallel_propagates_merge() {
        let mut vfg = VirtualFlowGraph::new();
        let e = vfg.graph.add_node(emit_node(&[0, 1, 2, 3]));
        let pa = vfg.graph.add_node(propagate_node(&[0, 1]));
        let pb = vfg.graph.add_node(propagate_node(&[2, 3]));
        let c = vfg.graph.add_node(collect_node(&[0, 1, 2, 3]));
        vfg.graph.add_edge(e, pa, Edge::new());
        vfg.graph.add_edge(e, pb, Edge::new());
        vfg.graph.add_edge(pa, c, Edge::new());
        vfg.graph.add_edge(pb, c, Edge::new());

        merge_parallel_nodes(&mut vfg);

        assert_eq!(vfg.graph.node_count(), 3);

        let prop_nodes: Vec<_> = vfg
            .graph
            .node_indices()
            .filter(|&idx| matches!(vfg.graph[idx].kind, NodeKind::Propagate(_)))
            .collect();
        assert_eq!(prop_nodes.len(), 1);
        assert_eq!(vfg.graph[prop_nodes[0]].partition.len(), 2);
    }

    #[test]
    fn test_different_gates_no_merge() {
        let mut vfg = VirtualFlowGraph::new();
        let e = vfg.graph.add_node(emit_node(&[0, 1, 2, 3]));
        let pa = vfg.graph.add_node(propagate_node_with_gate(&[0, 1], StandardGate::CX));
        let pb = vfg.graph.add_node(propagate_node_with_gate(&[2, 3], StandardGate::H));
        let c = vfg.graph.add_node(collect_node(&[0, 1, 2, 3]));
        vfg.graph.add_edge(e, pa, Edge::new());
        vfg.graph.add_edge(e, pb, Edge::new());
        vfg.graph.add_edge(pa, c, Edge::new());
        vfg.graph.add_edge(pb, c, Edge::new());

        merge_parallel_nodes(&mut vfg);

        assert_eq!(vfg.graph.node_count(), 4);
    }

    #[test]
    fn test_opposite_directions_no_merge() {
        // Conjugating the same gate leftward and rightward are different operations, so fusing them
        // into one wider node would make it unevaluable.
        let mut vfg = VirtualFlowGraph::new();
        let e = vfg.graph.add_node(emit_node(&[0, 1, 2, 3]));
        let pa = vfg
            .graph
            .add_node(propagate_node_with(&[0, 1], StandardGate::CX, Direction::Right));
        let pb = vfg
            .graph
            .add_node(propagate_node_with(&[2, 3], StandardGate::CX, Direction::Left));
        let c = vfg.graph.add_node(collect_node(&[0, 1, 2, 3]));
        vfg.graph.add_edge(e, pa, Edge::new());
        vfg.graph.add_edge(e, pb, Edge::new());
        vfg.graph.add_edge(pa, c, Edge::new());
        vfg.graph.add_edge(pb, c, Edge::new());

        merge_parallel_nodes(&mut vfg);

        assert_eq!(vfg.graph.node_count(), 4);
    }

    #[test]
    fn test_overlapping_partitions_no_merge() {
        let mut vfg = VirtualFlowGraph::new();
        let e = vfg.graph.add_node(emit_node(&[0, 1, 2]));
        let pa = vfg.graph.add_node(propagate_node(&[0, 1]));
        let pb = vfg.graph.add_node(propagate_node(&[1, 2]));
        let c = vfg.graph.add_node(collect_node(&[0, 1, 2]));
        vfg.graph.add_edge(e, pa, Edge::new());
        vfg.graph.add_edge(e, pb, Edge::new());
        vfg.graph.add_edge(pa, c, Edge::new());
        vfg.graph.add_edge(pb, c, Edge::new());

        merge_parallel_nodes(&mut vfg);

        assert_eq!(vfg.graph.node_count(), 4);
    }

    #[test]
    fn test_no_shared_predecessor_no_merge() {
        let mut vfg = VirtualFlowGraph::new();
        let ea = vfg.graph.add_node(emit_node(&[0, 1]));
        let eb = vfg.graph.add_node(emit_node(&[2, 3]));
        let pa = vfg.graph.add_node(propagate_node(&[0, 1]));
        let pb = vfg.graph.add_node(propagate_node(&[2, 3]));
        let c = vfg.graph.add_node(collect_node(&[0, 1, 2, 3]));
        vfg.graph.add_edge(ea, pa, Edge::new());
        vfg.graph.add_edge(eb, pb, Edge::new());
        vfg.graph.add_edge(pa, c, Edge::new());
        vfg.graph.add_edge(pb, c, Edge::new());

        merge_parallel_nodes(&mut vfg);

        assert_eq!(vfg.graph.node_count(), 5);
    }

    #[test]
    fn test_predecessorless_resets_merge() {
        let mut vfg = VirtualFlowGraph::new();
        let ra = vfg.graph.add_node(Node {
            partition: Partition::from_elements([0]),
            kind: NodeKind::Reset,
        });
        let rb = vfg.graph.add_node(Node {
            partition: Partition::from_elements([1]),
            kind: NodeKind::Reset,
        });
        let c = vfg.graph.add_node(collect_node(&[0, 1]));
        vfg.graph.add_edge(ra, c, Edge::new());
        vfg.graph.add_edge(rb, c, Edge::new());

        merge_parallel_nodes(&mut vfg);

        assert_eq!(vfg.graph.node_count(), 2);

        let reset_nodes: Vec<_> = vfg
            .graph
            .node_indices()
            .filter(|&idx| matches!(vfg.graph[idx].kind, NodeKind::Reset))
            .collect();
        assert_eq!(reset_nodes.len(), 1);
        assert_eq!(vfg.graph[reset_nodes[0]].partition.len(), 2);
    }

    #[test]
    fn test_measures_merge() {
        let mut vfg = VirtualFlowGraph::new();
        let e = vfg.graph.add_node(emit_node(&[0, 1, 2, 3]));
        let ma = vfg.graph.add_node(Node {
            partition: Partition::from_elements([0, 1]),
            kind: NodeKind::Measure(Measure {
                clbit_indices: vec![0, 1],
            }),
        });
        let mb = vfg.graph.add_node(Node {
            partition: Partition::from_elements([2, 3]),
            kind: NodeKind::Measure(Measure {
                clbit_indices: vec![2, 3],
            }),
        });
        vfg.graph.add_edge(e, ma, Edge::new());
        vfg.graph.add_edge(e, mb, Edge::new());

        merge_parallel_nodes(&mut vfg);

        assert_eq!(vfg.graph.node_count(), 2);

        let meas_nodes: Vec<_> = vfg
            .graph
            .node_indices()
            .filter(|&idx| matches!(vfg.graph[idx].kind, NodeKind::Measure(_)))
            .collect();
        assert_eq!(meas_nodes.len(), 1);
        if let NodeKind::Measure(m) = &vfg.graph[meas_nodes[0]].kind {
            assert_eq!(m.clbit_indices.len(), 4);
        } else {
            panic!("expected Measure");
        }
    }

    #[test]
    fn test_edge_deduplication() {
        let mut vfg = VirtualFlowGraph::new();
        let e = vfg.graph.add_node(emit_node(&[0, 1, 2, 3]));
        let pa = vfg.graph.add_node(propagate_node(&[0, 1]));
        let pb = vfg.graph.add_node(propagate_node(&[2, 3]));
        let c = vfg.graph.add_node(collect_node(&[0, 1, 2, 3]));
        vfg.graph.add_edge(e, pa, Edge::new());
        vfg.graph.add_edge(e, pb, Edge::new());
        vfg.graph.add_edge(pa, c, Edge::new());
        vfg.graph.add_edge(pb, c, Edge::new());

        merge_parallel_nodes(&mut vfg);

        assert_eq!(vfg.graph.edge_count(), 2);
    }
}
