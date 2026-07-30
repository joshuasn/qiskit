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

use rustworkx_core::petgraph::visit::EdgeRef;
use rustworkx_core::petgraph::Direction as PetDirection;

use crate::virtual_flow_graph::{NodeKind, VirtualFlowGraph, VirtualType};

use super::utils::topological_generations;

/// The type a source node puts onto its outgoing edges.
///
/// For an emission this is read straight off the node rather than re-derived from its distribution or
/// basis mode: IR2 resolved the type from the annotation when the emission was created, and that is the
/// authoritative value. Deriving it a second time here is how the two could disagree.
fn source_virtual_type(kind: &NodeKind) -> Option<VirtualType> {
    match kind {
        NodeKind::Emission(emission) => Some(emission.virtual_type),
        // A reset prepares a known state, so what flows out of it is a Pauli frame.
        NodeKind::Reset => Some(VirtualType::Pauli),
        NodeKind::Propagate(_) | NodeKind::Collect(_) | NodeKind::Measure(_) => None,
    }
}

/// Set the virtual type on all edges by forward-propagating from each node's output type.
/// Propagate nodes pass through the virtual type from their incoming edges unchanged.
pub fn set_virtual_types(vfg: &mut VirtualFlowGraph) {
    let generations = topological_generations(&vfg.graph);

    for generation in generations {
        for idx in generation {
            let vtype = if let Some(t) = source_virtual_type(&vfg.graph[idx].kind) {
                t
            } else if matches!(vfg.graph[idx].kind, NodeKind::Propagate(_)) {
                let incoming = vfg
                    .graph
                    .edges_directed(idx, PetDirection::Incoming)
                    .find_map(|e| e.weight().virtual_type);
                match incoming {
                    Some(t) => t,
                    None => continue,
                }
            } else {
                continue;
            };

            let edge_ids: Vec<_> = vfg
                .graph
                .edges_directed(idx, PetDirection::Outgoing)
                .map(|e| e.id())
                .collect();
            for edge_id in edge_ids {
                vfg.graph[edge_id].virtual_type = Some(vtype);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::DistEntry;
    use crate::partition::Partition;
    use crate::passes::test_fixtures::*;
    use crate::virtual_flow_graph::*;

    use rustworkx_core::petgraph::stable_graph::NodeIndex;

    fn get_outgoing_vtypes(
        vfg: &VirtualFlowGraph,
        idx: NodeIndex,
    ) -> Vec<Option<VirtualType>> {
        vfg.graph
            .edges_directed(idx, PetDirection::Outgoing)
            .map(|e| e.weight().virtual_type)
            .collect()
    }

    #[test]
    fn test_edges_start_as_none() {
        let mut vfg = VirtualFlowGraph::new();
        let e = vfg.graph.add_node(emit_node(&[0, 1]));
        let c = vfg.graph.add_node(collect_node(&[0, 1]));
        vfg.graph.add_edge(e, c, Edge::new());

        let vtypes = get_outgoing_vtypes(&vfg, e);
        assert_eq!(vtypes, vec![None]);
    }

    #[test]
    fn test_emit_pauli_type() {
        let mut vfg = VirtualFlowGraph::new();
        let e = vfg.graph.add_node(emit_node(&[0, 1]));
        let p = vfg.graph.add_node(propagate_node(&[0, 1]));
        let c = vfg.graph.add_node(collect_node(&[0, 1]));
        vfg.graph.add_edge(e, p, Edge::new());
        vfg.graph.add_edge(p, c, Edge::new());

        set_virtual_types(&mut vfg);

        assert_eq!(get_outgoing_vtypes(&vfg, e), vec![Some(VirtualType::Pauli)]);
    }

    #[test]
    fn test_propagate_pauli_past_clifford_output() {
        let mut vfg = VirtualFlowGraph::new();
        let e = vfg.graph.add_node(emit_node(&[0, 1]));
        let p = vfg.graph.add_node(propagate_node(&[0, 1]));
        let c = vfg.graph.add_node(collect_node(&[0, 1]));
        vfg.graph.add_edge(e, p, Edge::new());
        vfg.graph.add_edge(p, c, Edge::new());

        set_virtual_types(&mut vfg);

        assert_eq!(get_outgoing_vtypes(&vfg, p), vec![Some(VirtualType::Pauli)]);
    }

    #[test]
    fn test_c1_flow() {
        let mut vfg = VirtualFlowGraph::new();
        let e = vfg
            .graph
            .add_node(typed_emit_node(&[0, 1], DistributionType::UniformC1));
        let p = vfg.graph.add_node(propagate_node(&[0, 1]));
        let c = vfg.graph.add_node(collect_node(&[0, 1]));
        vfg.graph.add_edge(e, p, Edge::new());
        vfg.graph.add_edge(p, c, Edge::new());

        set_virtual_types(&mut vfg);

        assert_eq!(get_outgoing_vtypes(&vfg, e), vec![Some(VirtualType::C1)]);
        assert_eq!(get_outgoing_vtypes(&vfg, p), vec![Some(VirtualType::C1)]);
    }

    #[test]
    fn test_u2_flow() {
        let mut vfg = VirtualFlowGraph::new();
        let e = vfg
            .graph
            .add_node(typed_emit_node(&[0, 1], DistributionType::HaarU2));
        let p = vfg.graph.add_node(propagate_node(&[0, 1]));
        let c = vfg.graph.add_node(collect_node(&[0, 1]));
        vfg.graph.add_edge(e, p, Edge::new());
        vfg.graph.add_edge(p, c, Edge::new());

        set_virtual_types(&mut vfg);

        assert_eq!(get_outgoing_vtypes(&vfg, e), vec![Some(VirtualType::U2)]);
        assert_eq!(get_outgoing_vtypes(&vfg, p), vec![Some(VirtualType::U2)]);
    }

    #[test]
    fn test_basis_changes_keep_their_own_types() {
        // Two basis-change emissions into one collector, declaring different types. What the pass owes
        // is that each edge gets its own source's type — the mode-to-type mapping itself is the build
        // pass's business now that the type travels on the emission.
        let mut vfg = VirtualFlowGraph::new();
        let cb_pauli = vfg.graph.add_node(basis_node(
            &[0, 1],
            ChangeBasisMode::MeasurePauli,
            "cb.0",
            VirtualType::Pauli,
        ));
        let cb_c1 = vfg.graph.add_node(basis_node(
            &[2, 3],
            ChangeBasisMode::LocalClifford,
            "cb.1",
            VirtualType::C1,
        ));
        let c = vfg.graph.add_node(collect_node(&[0, 1, 2, 3]));
        vfg.graph.add_edge(cb_pauli, c, Edge::new());
        vfg.graph.add_edge(cb_c1, c, Edge::new());

        set_virtual_types(&mut vfg);

        assert_eq!(
            get_outgoing_vtypes(&vfg, cb_pauli),
            vec![Some(VirtualType::Pauli)]
        );
        assert_eq!(
            get_outgoing_vtypes(&vfg, cb_c1),
            vec![Some(VirtualType::C1)]
        );
    }

    fn basis_node(
        qubits: &[usize],
        mode: ChangeBasisMode,
        ref_id: &str,
        virtual_type: VirtualType,
    ) -> Node {
        Node {
            partition: Partition::from_elements(qubits.iter().copied()),
            kind: NodeKind::Emission(Emission {
                id: 0,
                entry: DistEntry::Basis {
                    mode,
                    ref_id: ref_id.to_string(),
                },
                direction: Direction::Left,
                virtual_type,
            }),
        }
    }

    #[test]
    fn test_reset_is_pauli() {
        let mut vfg = VirtualFlowGraph::new();
        let r = vfg.graph.add_node(Node {
            partition: Partition::from_elements([0]),
            kind: NodeKind::Reset,
        });
        let c = vfg.graph.add_node(collect_node(&[0]));
        vfg.graph.add_edge(r, c, Edge::new());

        set_virtual_types(&mut vfg);

        assert_eq!(
            get_outgoing_vtypes(&vfg, r),
            vec![Some(VirtualType::Pauli)]
        );
    }

    #[test]
    fn test_inject_noise_is_pauli() {
        let mut vfg = VirtualFlowGraph::new();
        let inj = vfg.graph.add_node(Node {
            partition: Partition::from_elements([0, 1]),
            kind: NodeKind::Emission(Emission {
                id: 0,
                entry: DistEntry::Noise {
                    reference: "noise.0".to_string(),
                    modifier: None,
                },
                direction: Direction::Left,
                virtual_type: VirtualType::Pauli,
            }),
        });
        let c = vfg.graph.add_node(collect_node(&[0, 1]));
        vfg.graph.add_edge(inj, c, Edge::new());

        set_virtual_types(&mut vfg);

        assert_eq!(
            get_outgoing_vtypes(&vfg, inj),
            vec![Some(VirtualType::Pauli)]
        );
    }
}
