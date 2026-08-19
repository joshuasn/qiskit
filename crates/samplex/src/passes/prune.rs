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

//! Prune unreachable nodes: sampling graph (IR3) → sampling graph (IR3), in place.

use hashbrown::HashSet;
use rustworkx_core::petgraph::stable_graph::{NodeIndex, StableDiGraph};
use rustworkx_core::traversal::{ancestors, descendants};

use crate::sampling_graph::{Edge, Node, NodeKind, SamplingGraph};

/// Whether a node has work of its own to do, whatever else reaches it.
///
/// A collector with steps composes something: absorbed gates, local emissions, or both. It has
/// angles to synthesize even with no virtual state arriving from anywhere, so reachability says
/// nothing about whether it is needed. Only an empty one is a pure junction that pruning may drop.
fn is_self_sufficient(kind: &NodeKind) -> bool {
    match kind {
        NodeKind::Collect(collect) => !collect.steps.is_empty(),
        _ => false,
    }
}

fn prune_unreachable(
    sg: &mut SamplingGraph,
    is_seed: impl Fn(&NodeKind) -> bool,
    traverse: impl Fn(&StableDiGraph<Node, Edge>, NodeIndex) -> Vec<NodeIndex>,
) {
    let seeds: Vec<NodeIndex> = sg
        .graph
        .node_indices()
        .filter(|&idx| is_seed(&sg.graph[idx].kind))
        .collect();

    let mut reachable: HashSet<NodeIndex> = HashSet::new();
    for seed in seeds {
        reachable.extend(traverse(&sg.graph, seed));
    }

    let to_remove: Vec<NodeIndex> = sg
        .graph
        .node_indices()
        .filter(|idx| !reachable.contains(idx) && !is_self_sufficient(&sg.graph[*idx].kind))
        .collect();

    for idx in to_remove {
        sg.graph.remove_node(idx);
    }
}

/// Remove nodes not reachable from any source: an `Emission` or a `Reset`.
///
/// A collector with steps is kept whatever reaches it. Absorption leaves collectors that nothing
/// propagates into, and their steps are still angles to synthesize.
pub fn prune_unreachable_from_sources(sg: &mut SamplingGraph) {
    prune_unreachable(
        sg,
        |kind| kind.is_source(),
        |g, n| descendants(g, n).collect(),
    );
}

/// Remove nodes that cannot reach any sink: a `Collect` or a `Measure`.
///
/// A sink is its own seed, so this drops only the nodes upstream of nothing. A collector with steps
/// is kept here too, for the same reason as in the source pass.
pub fn prune_unreachable_from_sinks(sg: &mut SamplingGraph) {
    prune_unreachable(sg, |kind| kind.is_sink(), |g, n| ancestors(g, n).collect());
}

#[cfg(test)]
mod tests {
    use qiskit_circuit::standard_gate::StandardGate;

    use super::*;
    use crate::distributions::DistKey;
    use crate::passes::test_fixtures::*;
    use crate::sampling_graph::*;

    #[test]
    fn test_simple_chain_survives() {
        let mut sg = SamplingGraph::new();
        let e = sg.graph.add_node(emit_node(&[0, 1]));
        let p = sg.graph.add_node(propagate_node(&[0, 1]));
        let c = sg.graph.add_node(collect_node(&[0, 1]));
        sg.graph.add_edge(e, p, Edge::new());
        sg.graph.add_edge(p, c, Edge::new());

        prune_unreachable_from_sources(&mut sg);
        assert_eq!(sg.graph.node_count(), 3);

        prune_unreachable_from_sinks(&mut sg);
        assert_eq!(sg.graph.node_count(), 3);
    }

    #[test]
    fn test_dead_branch_pruned_by_sink_pass() {
        let mut sg = SamplingGraph::new();
        let e = sg.graph.add_node(emit_node(&[0, 1]));
        let p = sg.graph.add_node(propagate_node(&[0, 1]));
        sg.graph.add_edge(e, p, Edge::new());

        prune_unreachable_from_sources(&mut sg);
        assert_eq!(sg.graph.node_count(), 2);

        prune_unreachable_from_sinks(&mut sg);
        assert_eq!(sg.graph.node_count(), 0);
    }

    #[test]
    fn test_orphan_pruned_by_source_pass() {
        let mut sg = SamplingGraph::new();
        sg.graph.add_node(propagate_node(&[0, 1]));

        prune_unreachable_from_sources(&mut sg);
        assert_eq!(sg.graph.node_count(), 0);
    }

    #[test]
    fn test_diamond_survives() {
        let mut sg = SamplingGraph::new();
        let e = sg.graph.add_node(emit_node(&[0, 1, 2, 3]));
        let pa = sg.graph.add_node(propagate_node(&[0, 1]));
        let pb = sg.graph.add_node(propagate_node(&[2, 3]));
        let c = sg.graph.add_node(collect_node(&[0, 1, 2, 3]));
        sg.graph.add_edge(e, pa, Edge::new());
        sg.graph.add_edge(e, pb, Edge::new());
        sg.graph.add_edge(pa, c, Edge::new());
        sg.graph.add_edge(pb, c, Edge::new());

        prune_unreachable_from_sources(&mut sg);
        assert_eq!(sg.graph.node_count(), 4);

        prune_unreachable_from_sinks(&mut sg);
        assert_eq!(sg.graph.node_count(), 4);
    }

    #[test]
    fn test_collect_with_steps_survives_both_passes() {
        // Absorption leaves collectors that nothing propagates into. This one has an absorbed `h` to
        // fold into its angles, so neither pass may drop it however isolated it is.
        let mut sg = SamplingGraph::new();
        sg.graph
            .add_node(collect_node_with_gate(&[0, 1], StandardGate::H, 0));

        prune_unreachable_from_sources(&mut sg);
        assert_eq!(sg.graph.node_count(), 1);

        prune_unreachable_from_sinks(&mut sg);
        assert_eq!(sg.graph.node_count(), 1);
    }

    #[test]
    fn test_empty_collect_is_still_pruned() {
        // The exception is only for a collector with work of its own: an empty one is a junction
        // that leads nowhere, and both passes drop it.
        let mut sg = SamplingGraph::new();
        sg.graph.add_node(collect_node(&[0, 1]));

        prune_unreachable_from_sources(&mut sg);
        assert_eq!(sg.graph.node_count(), 0);
    }

    #[test]
    fn test_source_pass_keeps_only_the_collector_with_steps() {
        // Both are unreachable from any source, and they part ways on whether they have steps.
        let mut sg = SamplingGraph::new();
        let empty = sg.graph.add_node(collect_node(&[0, 1]));
        let stepped = sg
            .graph
            .add_node(collect_node_with_gate(&[2, 3], StandardGate::S, 2));

        prune_unreachable_from_sources(&mut sg);
        assert_eq!(sg.graph.node_count(), 1);
        assert!(sg.graph.node_weight(empty).is_none());
        assert!(matches!(
            sg.graph.node_weight(stepped).map(|node| &node.kind),
            Some(NodeKind::Collect(_))
        ));
    }

    #[test]
    fn test_disconnected_source_and_sink() {
        let mut sg = SamplingGraph::new();
        sg.graph.add_node(emit_node(&[0, 1]));
        sg.graph.add_node(collect_node(&[2, 3]));

        prune_unreachable_from_sources(&mut sg);
        assert_eq!(sg.graph.node_count(), 1);

        prune_unreachable_from_sinks(&mut sg);
        assert_eq!(sg.graph.node_count(), 0);
    }

    #[test]
    fn test_multiple_source_types() {
        let mut sg = SamplingGraph::new();
        let e = sg.graph.add_node(emit_node(&[0, 1]));
        let cb = sg.graph.add_node(emission_node(&[2, 3], DistKey(1)));
        let inj = sg.graph.add_node(emission_node(&[4, 5], DistKey(2)));
        let c = sg.graph.add_node(collect_node(&[0, 1, 2, 3, 4, 5]));
        sg.graph.add_edge(e, c, Edge::new());
        sg.graph.add_edge(cb, c, Edge::new());
        sg.graph.add_edge(inj, c, Edge::new());

        prune_unreachable_from_sources(&mut sg);
        assert_eq!(sg.graph.node_count(), 4);

        prune_unreachable_from_sinks(&mut sg);
        assert_eq!(sg.graph.node_count(), 4);
    }

    #[test]
    fn test_measure_is_sink() {
        let mut sg = SamplingGraph::new();
        let e = sg.graph.add_node(emit_node(&[0, 1]));
        let m = sg.graph.add_node(measure_node(&[0, 1]));
        sg.graph.add_edge(e, m, Edge::new());

        prune_unreachable_from_sources(&mut sg);
        assert_eq!(sg.graph.node_count(), 2);

        prune_unreachable_from_sinks(&mut sg);
        assert_eq!(sg.graph.node_count(), 2);
    }

    #[test]
    fn test_reset_is_source() {
        let mut sg = SamplingGraph::new();
        let r = sg
            .graph
            .add_node(Node::singletons(vec![0], NodeKind::Reset));
        let c = sg.graph.add_node(collect_node(&[0]));
        sg.graph.add_edge(r, c, Edge::new());

        prune_unreachable_from_sources(&mut sg);
        assert_eq!(sg.graph.node_count(), 2);

        prune_unreachable_from_sinks(&mut sg);
        assert_eq!(sg.graph.node_count(), 2);
    }
}
