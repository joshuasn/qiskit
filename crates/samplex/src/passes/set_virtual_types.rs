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

//! Infer virtual types: sampling graph (IR3) → sampling graph (IR3), in place.
//!
//! Forward-propagates each source's own type along its outgoing edges.

use rustworkx_core::petgraph::Direction as PetDirection;
use rustworkx_core::petgraph::visit::EdgeRef;

use crate::sampling_graph::{NodeKind, SamplingGraph, VirtualType};

/// The type a source node puts onto its outgoing edges.
///
/// For an emission this is read straight off the node rather than re-derived from its distribution
/// or basis mode: IR2 resolved the type from the annotation when the emission was created, and that
/// is the authoritative value. Deriving it a second time here is how the two could disagree.
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
pub fn set_virtual_types(sg: &mut SamplingGraph) {
    let generations = sg.topological_generations();

    for generation in generations {
        for idx in generation {
            let vtype = if let Some(t) = source_virtual_type(&sg.graph[idx].kind) {
                t
            } else if matches!(sg.graph[idx].kind, NodeKind::Propagate(_)) {
                let incoming = sg
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

            let edge_ids: Vec<_> = sg
                .graph
                .edges_directed(idx, PetDirection::Outgoing)
                .map(|e| e.id())
                .collect();
            for edge_id in edge_ids {
                sg.graph[edge_id].virtual_type = Some(vtype);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::DistKey;
    use crate::passes::test_fixtures::*;
    use crate::sampling_graph::*;

    use rustworkx_core::petgraph::stable_graph::NodeIndex;

    fn get_outgoing_vtypes(sg: &SamplingGraph, idx: NodeIndex) -> Vec<Option<VirtualType>> {
        sg.graph
            .edges_directed(idx, PetDirection::Outgoing)
            .map(|e| e.weight().virtual_type)
            .collect()
    }

    #[test]
    fn test_edges_start_as_none() {
        let mut sg = SamplingGraph::new();
        let e = sg.graph.add_node(emit_node(&[0, 1]));
        let c = sg.graph.add_node(collect_node(&[0, 1]));
        sg.graph.add_edge(e, c, Edge::new());

        let vtypes = get_outgoing_vtypes(&sg, e);
        assert_eq!(vtypes, vec![None]);
    }

    #[test]
    fn test_emit_pauli_type() {
        let mut sg = SamplingGraph::new();
        let e = sg.graph.add_node(emit_node(&[0, 1]));
        let p = sg.graph.add_node(propagate_node(&[0, 1]));
        let c = sg.graph.add_node(collect_node(&[0, 1]));
        sg.graph.add_edge(e, p, Edge::new());
        sg.graph.add_edge(p, c, Edge::new());

        set_virtual_types(&mut sg);

        assert_eq!(get_outgoing_vtypes(&sg, e), vec![Some(VirtualType::Pauli)]);
    }

    #[test]
    fn test_propagate_pauli_past_clifford_output() {
        let mut sg = SamplingGraph::new();
        let e = sg.graph.add_node(emit_node(&[0, 1]));
        let p = sg.graph.add_node(propagate_node(&[0, 1]));
        let c = sg.graph.add_node(collect_node(&[0, 1]));
        sg.graph.add_edge(e, p, Edge::new());
        sg.graph.add_edge(p, c, Edge::new());

        set_virtual_types(&mut sg);

        assert_eq!(get_outgoing_vtypes(&sg, p), vec![Some(VirtualType::Pauli)]);
    }

    #[test]
    fn test_c1_flow() {
        let mut sg = SamplingGraph::new();
        let e = sg
            .graph
            .add_node(typed_emit_node(&[0, 1], DistributionType::UniformC1));
        let p = sg.graph.add_node(propagate_node(&[0, 1]));
        let c = sg.graph.add_node(collect_node(&[0, 1]));
        sg.graph.add_edge(e, p, Edge::new());
        sg.graph.add_edge(p, c, Edge::new());

        set_virtual_types(&mut sg);

        assert_eq!(get_outgoing_vtypes(&sg, e), vec![Some(VirtualType::C1)]);
        assert_eq!(get_outgoing_vtypes(&sg, p), vec![Some(VirtualType::C1)]);
    }

    #[test]
    fn test_u2_flow() {
        let mut sg = SamplingGraph::new();
        let e = sg
            .graph
            .add_node(typed_emit_node(&[0, 1], DistributionType::HaarU2));
        let p = sg.graph.add_node(propagate_node(&[0, 1]));
        let c = sg.graph.add_node(collect_node(&[0, 1]));
        sg.graph.add_edge(e, p, Edge::new());
        sg.graph.add_edge(p, c, Edge::new());

        set_virtual_types(&mut sg);

        assert_eq!(get_outgoing_vtypes(&sg, e), vec![Some(VirtualType::U2)]);
        assert_eq!(get_outgoing_vtypes(&sg, p), vec![Some(VirtualType::U2)]);
    }

    #[test]
    fn test_basis_changes_keep_their_own_types() {
        // Two basis-change emissions into one collector, declaring different types. What the pass
        // owes is that each edge gets its own source's type — the mode-to-type mapping itself is
        // the build pass's business now that the type travels on the emission.
        let mut sg = SamplingGraph::new();
        let cb_pauli = sg.graph.add_node(basis_node(&[0, 1], VirtualType::Pauli));
        let cb_c1 = sg.graph.add_node(basis_node(&[2, 3], VirtualType::C1));
        let c = sg.graph.add_node(collect_node(&[0, 1, 2, 3]));
        sg.graph.add_edge(cb_pauli, c, Edge::new());
        sg.graph.add_edge(cb_c1, c, Edge::new());

        set_virtual_types(&mut sg);

        assert_eq!(
            get_outgoing_vtypes(&sg, cb_pauli),
            vec![Some(VirtualType::Pauli)]
        );
        assert_eq!(get_outgoing_vtypes(&sg, cb_c1), vec![Some(VirtualType::C1)]);
    }

    fn basis_node(qubits: &[usize], virtual_type: VirtualType) -> Node {
        Node::singletons(
            qubits.to_vec(),
            NodeKind::Emission(Emission {
                key: DistKey(0),
                direction: Direction::Left,
                virtual_type,
            }),
        )
    }

    #[test]
    fn test_reset_is_pauli() {
        let mut sg = SamplingGraph::new();
        let r = sg
            .graph
            .add_node(Node::singletons(vec![0], NodeKind::Reset));
        let c = sg.graph.add_node(collect_node(&[0]));
        sg.graph.add_edge(r, c, Edge::new());

        set_virtual_types(&mut sg);

        assert_eq!(get_outgoing_vtypes(&sg, r), vec![Some(VirtualType::Pauli)]);
    }

    #[test]
    fn test_inject_noise_is_pauli() {
        let mut sg = SamplingGraph::new();
        let inj = sg.graph.add_node(Node::singletons(
            vec![0, 1],
            NodeKind::Emission(Emission {
                key: DistKey(0),
                direction: Direction::Left,
                virtual_type: VirtualType::Pauli,
            }),
        ));
        let c = sg.graph.add_node(collect_node(&[0, 1]));
        sg.graph.add_edge(inj, c, Edge::new());

        set_virtual_types(&mut sg);

        assert_eq!(
            get_outgoing_vtypes(&sg, inj),
            vec![Some(VirtualType::Pauli)]
        );
    }
}
