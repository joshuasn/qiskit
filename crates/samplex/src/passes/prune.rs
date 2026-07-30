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

use hashbrown::HashSet;
use rustworkx_core::petgraph::stable_graph::{NodeIndex, StableDiGraph};
use rustworkx_core::traversal::{ancestors, descendants};

use crate::virtual_flow_graph::{Edge, Node, NodeKind, VirtualFlowGraph};

fn prune_unreachable(
    vfg: &mut VirtualFlowGraph,
    is_seed: impl Fn(&NodeKind) -> bool,
    traverse: impl Fn(&StableDiGraph<Node, Edge>, NodeIndex) -> Vec<NodeIndex>,
) {
    let seeds: Vec<NodeIndex> = vfg
        .graph
        .node_indices()
        .filter(|&idx| is_seed(&vfg.graph[idx].kind))
        .collect();

    let mut reachable: HashSet<NodeIndex> = HashSet::new();
    for seed in seeds {
        reachable.extend(traverse(&vfg.graph, seed));
    }

    let to_remove: Vec<NodeIndex> = vfg
        .graph
        .node_indices()
        .filter(|idx| !reachable.contains(idx))
        .collect();

    for idx in to_remove {
        vfg.graph.remove_node(idx);
    }
}

/// Remove nodes not reachable from any source (Emit, Reset, ChangeBasis, InjectNoise).
pub fn prune_unreachable_from_sources(vfg: &mut VirtualFlowGraph) {
    prune_unreachable(
        vfg,
        |kind| kind.is_source(),
        |g, n| descendants(g, n).collect(),
    );
}

/// Remove nodes that cannot reach any sink (Collect, Measure).
pub fn prune_unreachable_from_sinks(vfg: &mut VirtualFlowGraph) {
    prune_unreachable(
        vfg,
        |kind| kind.is_sink(),
        |g, n| ancestors(g, n).collect(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::DistEntry;
    use crate::partition::Partition;
    use crate::passes::test_fixtures::*;
    use crate::virtual_flow_graph::*;

    #[test]
    fn test_simple_chain_survives() {
        let mut vfg = VirtualFlowGraph::new();
        let e = vfg.graph.add_node(emit_node(&[0, 1]));
        let p = vfg.graph.add_node(propagate_node(&[0, 1]));
        let c = vfg.graph.add_node(collect_node(&[0, 1]));
        vfg.graph.add_edge(e, p, Edge::new());
        vfg.graph.add_edge(p, c, Edge::new());

        prune_unreachable_from_sources(&mut vfg);
        assert_eq!(vfg.graph.node_count(), 3);

        prune_unreachable_from_sinks(&mut vfg);
        assert_eq!(vfg.graph.node_count(), 3);
    }

    #[test]
    fn test_dead_branch_pruned_by_sink_pass() {
        let mut vfg = VirtualFlowGraph::new();
        let e = vfg.graph.add_node(emit_node(&[0, 1]));
        let p = vfg.graph.add_node(propagate_node(&[0, 1]));
        vfg.graph.add_edge(e, p, Edge::new());

        prune_unreachable_from_sources(&mut vfg);
        assert_eq!(vfg.graph.node_count(), 2);

        prune_unreachable_from_sinks(&mut vfg);
        assert_eq!(vfg.graph.node_count(), 0);
    }

    #[test]
    fn test_orphan_pruned_by_source_pass() {
        let mut vfg = VirtualFlowGraph::new();
        vfg.graph.add_node(propagate_node(&[0, 1]));

        prune_unreachable_from_sources(&mut vfg);
        assert_eq!(vfg.graph.node_count(), 0);
    }

    #[test]
    fn test_diamond_survives() {
        let mut vfg = VirtualFlowGraph::new();
        let e = vfg.graph.add_node(emit_node(&[0, 1, 2, 3]));
        let pa = vfg.graph.add_node(propagate_node(&[0, 1]));
        let pb = vfg.graph.add_node(propagate_node(&[2, 3]));
        let c = vfg.graph.add_node(collect_node(&[0, 1, 2, 3]));
        vfg.graph.add_edge(e, pa, Edge::new());
        vfg.graph.add_edge(e, pb, Edge::new());
        vfg.graph.add_edge(pa, c, Edge::new());
        vfg.graph.add_edge(pb, c, Edge::new());

        prune_unreachable_from_sources(&mut vfg);
        assert_eq!(vfg.graph.node_count(), 4);

        prune_unreachable_from_sinks(&mut vfg);
        assert_eq!(vfg.graph.node_count(), 4);
    }

    #[test]
    fn test_disconnected_source_and_sink() {
        let mut vfg = VirtualFlowGraph::new();
        vfg.graph.add_node(emit_node(&[0, 1]));
        vfg.graph.add_node(collect_node(&[2, 3]));

        prune_unreachable_from_sources(&mut vfg);
        assert_eq!(vfg.graph.node_count(), 1);

        prune_unreachable_from_sinks(&mut vfg);
        assert_eq!(vfg.graph.node_count(), 0);
    }

    #[test]
    fn test_multiple_source_types() {
        let mut vfg = VirtualFlowGraph::new();
        let e = vfg.graph.add_node(emit_node(&[0, 1]));
        let cb = vfg.graph.add_node(emission_node(
            &[2, 3],
            DistEntry::Basis {
                mode: ChangeBasisMode::MeasurePauli,
                ref_id: "cb.0".to_string(),
            },
        ));
        let inj = vfg.graph.add_node(emission_node(
            &[4, 5],
            DistEntry::Noise {
                reference: "noise.0".to_string(),
                modifier: None,
            },
        ));
        let c = vfg.graph.add_node(collect_node(&[0, 1, 2, 3, 4, 5]));
        vfg.graph.add_edge(e, c, Edge::new());
        vfg.graph.add_edge(cb, c, Edge::new());
        vfg.graph.add_edge(inj, c, Edge::new());

        prune_unreachable_from_sources(&mut vfg);
        assert_eq!(vfg.graph.node_count(), 4);

        prune_unreachable_from_sinks(&mut vfg);
        assert_eq!(vfg.graph.node_count(), 4);
    }

    #[test]
    fn test_measure_is_sink() {
        let mut vfg = VirtualFlowGraph::new();
        let e = vfg.graph.add_node(emit_node(&[0, 1]));
        let m = vfg.graph.add_node(measure_node(&[0, 1]));
        vfg.graph.add_edge(e, m, Edge::new());

        prune_unreachable_from_sources(&mut vfg);
        assert_eq!(vfg.graph.node_count(), 2);

        prune_unreachable_from_sinks(&mut vfg);
        assert_eq!(vfg.graph.node_count(), 2);
    }

    #[test]
    fn test_reset_is_source() {
        let mut vfg = VirtualFlowGraph::new();
        let r = vfg.graph.add_node(Node {
            partition: Partition::from_elements([0]),
            kind: NodeKind::Reset,
        });
        let c = vfg.graph.add_node(collect_node(&[0]));
        vfg.graph.add_edge(r, c, Edge::new());

        prune_unreachable_from_sources(&mut vfg);
        assert_eq!(vfg.graph.node_count(), 2);

        prune_unreachable_from_sinks(&mut vfg);
        assert_eq!(vfg.graph.node_count(), 2);
    }
}
