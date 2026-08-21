// This code is a Qiskit project.
//
// (C) Copyright IBM 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! The spine: an emission circuit read as one flat run of positions.
//!
//! Propagation is never recorded in the emission circuit — it is derived from where an emission
//! stands, and this module is what that derivation reads. Flattening the box nesting away is what
//! makes it a scan rather than a tree walk: hard boxes are inlined so their gates sit in line, and
//! each collector is reduced to a single position carrying its absorbed run, so "the nearest
//! collector ahead" is just the next [`Item::Collector`] in the travel direction.
//!
//! A spine holds no `Python` token and no `DAGCircuit`. Reading one out of IR2 is the job of the
//! adapter in [`lower`](crate::passes::lower), which is the only place that touches the DAG;
//! everything here works on the flat reading alone, so the propagation rules can be exercised by
//! building a spine directly with [`Spine::new`].

use hashbrown::{HashMap, HashSet};
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use qiskit_circuit::operations::StandardGate;

use crate::annotated_circuit::SynthesizerType;
use crate::distributions::DistributionTable;
use crate::emission_circuit::Emit;
use crate::emission_circuit_navigation::Site;
use crate::error::{Result, SamplexError};
use crate::partition::Partition;
use crate::sampling_graph::{
    AbsorbedGate, CollectStep, Direction, Edge, Node, NodeKind, Propagate, SamplingGraph,
};
use crate::virtual_type::{VirtualType, propagates};

/// One collector, flattened out of the circuit.
pub struct Collector {
    /// Where this collector stood in the emission circuit, which is what identifies it across the
    /// two readings of that circuit. A `Site` is a path of node indices, so it survives the borrow
    /// that produced it and a spine carrying one still holds no `DAGCircuit`.
    pub site: Site,
    pub qubits: Vec<usize>,
    /// How those qubits group into subsystems, by index into `qubits`.
    pub partition: Partition,
    pub synthesizer: SynthesizerType,
    pub param_indices: Vec<usize>,
    /// Everything this collector composes, in the order the adapter read it out of the body.
    pub steps: Vec<CollectStep>,
}

impl Collector {
    /// The absorbed gates alone, for an enclosing emission crossing this collector: it conjugates
    /// by the gates and ignores what the collector consumes.
    pub fn gates(&self) -> impl Iterator<Item = &AbsorbedGate> {
        crate::sampling_graph::collect_step_gates(&self.steps)
    }

    /// Whether this collector can take this emission.
    ///
    /// **This is the seam where "whose emission is this" is decided, and it is deliberately incomplete.**
    /// Two conditions are in place:
    ///
    /// - It covers every qubit the emission acts on. A collector that covers only part of an emission
    ///   could not synthesize the whole of what was emitted. The emission's qubits come from the walk
    ///   rather than from the annotation: one `Emit` groups its *own qargs* by index and is shared
    ///   by every placement of it, so it cannot know which wires it landed on.
    /// - Its synthesizer accepts the emission's virtual type, so the value it would have to produce is one
    ///   it can express.
    ///
    /// Nothing here asks which annotated box the emission came from — position and these two conditions
    /// are all of it. So a collector nested inside an enclosing box will take that box's propagating
    /// emission if it happens to be the first one the walk reaches, terminating the enclosing
    /// randomization at the inner dressing with none of the enclosing content in between. That is
    /// invisible to a round-trip test, since the circuit still evaluates to the same unitary.
    ///
    /// **TO DO: make this a type question.** The intended shape is that an emission carries a type a
    /// collector either accepts or declines — a basis change becoming a distinct type rather than a Pauli
    /// that looks like any other, an inner twirl marked as unable to collect it — so that a collector that
    /// should not have it declines, the emission propagates on, and it reaches its own collector by
    /// walking rather than by consulting an id. Until then a nested twirl of the same group is collected
    /// early, and `test_sampling_graph.py::TestNestedPropagation` pins that provisional behaviour so the
    /// change of rule shows up as a test change rather than silently.
    pub fn accepts(&self, emission: &Emit, qubits: &[usize], table: &DistributionTable) -> bool {
        qubits.iter().all(|q| self.qubits.contains(q))
            && self.synthesizer.accepts(emission.virtual_type(table))
    }
}

/// What stands at one position on the spine.
pub enum Item {
    Emission(Emit, Vec<usize>),
    /// One collector, by index into [`Spine::collectors`].
    Collector(usize),
    Gate(StandardGate, Vec<usize>),
    Measure(Vec<usize>, Vec<usize>),
    Reset(Vec<usize>),
    /// A real operation with no virtual effect, kept so that a position on the spine still stands for
    /// one instruction of the circuit it was read from.
    Opaque,
}

/// What identifies one conjugation node: a gate occurrence together with the flow crossing it.
///
/// The occurrence is `(spine position, offset)`, the offset being the position within a collector's
/// absorbed run and zero for a gate that stands on its own.
pub type GateKey = (usize, usize, Direction, VirtualType);

/// The circuit as a flat sequence, which is what makes the propagation walk a simple scan.
#[derive(Default)]
pub struct Spine {
    /// The positions, in circuit order.
    pub items: Vec<Item>,
    /// The collectors, in the order the positions refer to them.
    pub collectors: Vec<Collector>,
}

impl Spine {
    /// Build a spine from a flat reading already in hand.
    ///
    /// This is the constructor that keeps the propagation rules reachable without a `DAGCircuit`.
    /// The index in every [`Item::Collector`] must be a position in `collectors`; that is the one
    /// invariant a caller upholds, and the reason [`Spine::push_collector`] exists for the
    /// incremental case.
    pub fn new(items: Vec<Item>, collectors: Vec<Collector>) -> Self {
        Self { items, collectors }
    }

    /// Append a collector together with the position that stands for it, keeping the two in step.
    pub fn push_collector(&mut self, collector: Collector) {
        self.items.push(Item::Collector(self.collectors.len()));
        self.collectors.push(collector);
    }

    /// Scan from `start` in `direction` for the first collector that can take this emission.
    ///
    /// Proximity decides, filtered by [`Collector::accepts`]. A collector that declines is simply
    /// crossed — its absorbed gates conjugate the emission on the way past — so an emission travels
    /// until it finds one that can take it, out of the box it started in and on through whatever it
    /// passes. Reaching the end of the circuit is the error case.
    pub fn resolve_collector(
        &self,
        start: usize,
        direction: Direction,
        emission: &Emit,
        qubits: &[usize],
        table: &DistributionTable,
    ) -> Option<usize> {
        let range: Box<dyn Iterator<Item = usize>> = match direction {
            Direction::Right => Box::new((start + 1)..self.items.len()),
            Direction::Left => Box::new((0..start).rev()),
        };
        for i in range {
            if let Item::Collector(index) = &self.items[i]
                && self.collectors[*index].accepts(emission, qubits, table)
            {
                return Some(*index);
            }
        }
        None
    }

    /// Wire one emission's path: every gate between it and its collector, chained per qubit.
    #[allow(clippy::too_many_arguments)]
    pub fn propagate(
        &self,
        sg: &mut SamplingGraph,
        from: usize,
        emission: &Emit,
        emission_qubits: &[usize],
        source: NodeIndex,
        target_index: usize,
        target_node: NodeIndex,
        gate_nodes: &mut HashMap<GateKey, NodeIndex>,
        table: &DistributionTable,
    ) -> Result<()> {
        let qubits: HashSet<usize> = emission_qubits.iter().copied().collect();
        let mut frontier: HashMap<usize, NodeIndex> = qubits.iter().map(|q| (*q, source)).collect();
        let direction = emission.direction.expect(
            "a local emission never surfaces as a top-level Item::Emission — it lives inside its \
             collector's body",
        );
        let virtual_type = emission.virtual_type(table);

        // Walking in the emission's own direction is what makes propagation derivable rather than
        // recorded.
        let indices: Vec<usize> = match direction {
            Direction::Right => (from + 1..self.items.len()).collect(),
            Direction::Left => (0..from).rev().collect(),
        };

        for index in indices {
            match &self.items[index] {
                Item::Collector(collector) if *collector == target_index => break,
                Item::Collector(collector) => {
                    // A foreign collector's absorbed gates are still real gates on this emission's
                    // path, so they conjugate it, even though that collector also multiplies them into
                    // its own layer.
                    let absorbed: Vec<&AbsorbedGate> =
                        self.collectors[*collector].gates().collect();
                    let order: Vec<usize> = match direction {
                        Direction::Right => (0..absorbed.len()).collect(),
                        Direction::Left => (0..absorbed.len()).rev().collect(),
                    };
                    for offset in order {
                        let gate = &absorbed[offset];
                        chain(
                            sg,
                            &mut frontier,
                            &qubits,
                            direction,
                            gate_nodes,
                            (index, offset),
                            gate.gate,
                            &gate.qubits,
                            virtual_type,
                        )?;
                    }
                }
                Item::Gate(gate, gate_qubits) => chain(
                    sg,
                    &mut frontier,
                    &qubits,
                    direction,
                    gate_nodes,
                    (index, 0),
                    *gate,
                    gate_qubits,
                    virtual_type,
                )?,
                _ => {}
            }
        }

        // Whatever each wire's virtual state ended up as is what the collector synthesizes.
        let ends: HashSet<NodeIndex> = frontier.values().copied().collect();
        for end in ends {
            sg.graph.add_edge(end, target_node, Edge::new());
        }
        Ok(())
    }
}

/// Add or reuse the node for one gate and advance the frontier over its qubits.
#[allow(clippy::too_many_arguments)]
fn chain(
    sg: &mut SamplingGraph,
    frontier: &mut HashMap<usize, NodeIndex>,
    tracked: &HashSet<usize>,
    direction: Direction,
    gate_nodes: &mut HashMap<GateKey, NodeIndex>,
    occurrence: (usize, usize),
    gate: StandardGate,
    gate_qubits: &[usize],
    virtual_type: VirtualType,
) -> Result<()> {
    if !gate_qubits.iter().any(|q| tracked.contains(q)) {
        return Ok(());
    }
    // Refuse rather than emit a node that cannot be evaluated: conjugating this virtual type by
    // this gate leaves its group, so there is no rule to apply.
    if !propagates(virtual_type, gate) {
        return Err(SamplexError::NoPropagationRule { virtual_type, gate });
    }
    let key = (occurrence.0, occurrence.1, direction, virtual_type);
    let node = *gate_nodes.entry(key).or_insert_with(|| {
        // One joint subsystem: a conjugation by a multi-qubit gate mixes its qubits, so they can
        // only be evaluated together.
        sg.graph.add_node(Node::joint(
            gate_qubits.to_vec(),
            NodeKind::Propagate(Propagate { gate, direction }),
        ))
    });
    let predecessors: HashSet<NodeIndex> = gate_qubits
        .iter()
        .filter_map(|q| frontier.get(q).copied())
        .collect();
    for predecessor in predecessors {
        sg.graph.add_edge(predecessor, node, Edge::new());
    }
    for q in gate_qubits.iter().filter(|q| tracked.contains(*q)) {
        frontier.insert(*q, node);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use qiskit_circuit::operations::Operation;
    use rustworkx_core::petgraph::Direction as PetDirection;

    use crate::annotated_circuit::DistributionType;
    use crate::distributions::{DistEntry, DistKey};
    use crate::emission_circuit::EmitPart;
    use crate::sampling_graph::{AbsorbedParam, Collect, Emission};

    // These tests are the reason the spine is its own module: none of them touches a `DAGCircuit` or
    // a `Python` token, so the propagation rules can be pinned on hand-built spines instead of on
    // circuits built through the GIL. A refusal is now a `SamplexError` variant, so a test can name
    // the failure it expects rather than only checking that there was one.

    /// A table holding one distribution, with the key to reach it.
    fn table_with(distribution: DistributionType) -> (DistributionTable, DistKey) {
        let mut table = DistributionTable::new();
        let key = table.intern(DistEntry::Distribution(distribution));
        (table, key)
    }

    /// An emission travelling `direction` over `width` wires, one part per wire.
    fn emit(direction: Direction, width: usize, key: DistKey) -> Emit {
        Emit {
            direction: Some(direction),
            partition: Partition::singletons(width),
            parts: (0..width)
                .map(|draw| EmitPart {
                    dist: key,
                    draw: draw as u32,
                    adjoint: false,
                })
                .collect(),
        }
    }

    /// A collector over `qubits` whose absorbed run is `steps`.
    ///
    /// The site is a placeholder: these tests exercise propagation, which never reads it. What does
    /// read it is the join in `lower`, tested there.
    fn collector(qubits: &[usize], steps: Vec<CollectStep>) -> Collector {
        Collector {
            site: Site {
                scope: Vec::new(),
                node: NodeIndex::new(0),
            },
            qubits: qubits.to_vec(),
            partition: Partition::singletons(qubits.len()),
            synthesizer: SynthesizerType::RzSx,
            param_indices: Vec::new(),
            steps,
        }
    }

    /// One absorbed gate on one wire, as a collector step.
    fn absorbed(gate: StandardGate, qubit: usize) -> CollectStep {
        CollectStep::Gate(AbsorbedGate {
            gate,
            qubits: vec![qubit],
            params: Vec::<AbsorbedParam>::new(),
        })
    }

    /// Wire the emission standing at `from` towards collector `target`, returning the graph it built
    /// along with the two nodes the walk runs between.
    fn wire(
        spine: &Spine,
        from: usize,
        emission: &Emit,
        qubits: &[usize],
        target: usize,
        table: &DistributionTable,
    ) -> Result<(SamplingGraph, NodeIndex, NodeIndex)> {
        let mut sg = SamplingGraph::new();
        let source = sg.graph.add_node(Node::singletons(
            qubits.to_vec(),
            NodeKind::Emission(Emission {
                key: emission.dist(),
                direction: emission.direction.unwrap(),
                virtual_type: emission.virtual_type(table),
            }),
        ));
        let target_node = sg.graph.add_node(Node::new(
            spine.collectors[target].qubits.clone(),
            spine.collectors[target].partition.clone(),
            NodeKind::Collect(Collect {
                synthesizer: SynthesizerType::RzSx,
                param_indices: Vec::new(),
                steps: Vec::new(),
            }),
        ));
        let mut gate_nodes = HashMap::new();
        spine.propagate(
            &mut sg,
            from,
            emission,
            qubits,
            source,
            target,
            target_node,
            &mut gate_nodes,
            table,
        )?;
        Ok((sg, source, target_node))
    }

    /// Every conjugation node the walk created, as `(gate, qubits)`.
    fn conjugations(sg: &SamplingGraph) -> Vec<(StandardGate, Vec<usize>)> {
        sg.graph
            .node_weights()
            .filter_map(|node| match &node.kind {
                NodeKind::Propagate(p) => Some((p.gate, node.qubits.clone())),
                _ => None,
            })
            .collect()
    }

    fn predecessors(sg: &SamplingGraph, node: NodeIndex) -> Vec<NodeIndex> {
        sg.graph
            .neighbors_directed(node, PetDirection::Incoming)
            .collect()
    }

    // --- Propagation target resolution -----------------------------------------------------------

    #[test]
    fn test_an_emission_resolves_to_the_nearest_collector_ahead() {
        let (table, key) = table_with(DistributionType::UniformPauli);
        let emission = emit(Direction::Right, 1, key);
        let spine = Spine::new(
            vec![
                Item::Emission(emission.clone(), vec![0]),
                Item::Gate(StandardGate::H, vec![0]),
                Item::Collector(0),
                Item::Collector(1),
            ],
            vec![collector(&[0], vec![]), collector(&[0], vec![])],
        );
        assert_eq!(
            spine.resolve_collector(0, Direction::Right, &emission, &[0], &table),
            Some(0)
        );
    }

    #[test]
    fn test_resolution_scans_the_way_the_emission_travels() {
        let (table, key) = table_with(DistributionType::UniformPauli);
        let spine = Spine::new(
            vec![Item::Collector(0), Item::Opaque, Item::Collector(1)],
            vec![collector(&[0], vec![]), collector(&[0], vec![])],
        );
        // The same position, read both ways: a twirl's two halves are exactly this pair.
        let far = emit(Direction::Right, 1, key);
        let near = emit(Direction::Left, 1, key);
        assert_eq!(
            spine.resolve_collector(1, Direction::Right, &far, &[0], &table),
            Some(1)
        );
        assert_eq!(
            spine.resolve_collector(1, Direction::Left, &near, &[0], &table),
            Some(0)
        );
    }

    #[test]
    fn test_a_collector_that_does_not_cover_the_emission_is_crossed() {
        let (table, key) = table_with(DistributionType::UniformPauli);
        let emission = emit(Direction::Right, 2, key);
        let spine = Spine::new(
            vec![
                Item::Emission(emission.clone(), vec![0, 1]),
                Item::Collector(0),
                Item::Collector(1),
            ],
            // The nearer collector covers only one of the emission's two wires, so it could not
            // synthesize the whole of what was emitted and the emission travels past it.
            vec![collector(&[0], vec![]), collector(&[0, 1], vec![])],
        );
        assert!(!spine.collectors[0].accepts(&emission, &[0, 1], &table));
        assert!(spine.collectors[1].accepts(&emission, &[0, 1], &table));
        assert_eq!(
            spine.resolve_collector(0, Direction::Right, &emission, &[0, 1], &table),
            Some(1)
        );
    }

    #[test]
    fn test_an_emission_with_nothing_to_collect_it_does_not_resolve() {
        let (table, key) = table_with(DistributionType::UniformPauli);
        let emission = emit(Direction::Right, 1, key);
        let spine = Spine::new(
            vec![
                Item::Emission(emission.clone(), vec![0]),
                Item::Gate(StandardGate::H, vec![0]),
            ],
            vec![],
        );
        // The caller turns this into the "randomization could not be undone" error.
        assert_eq!(
            spine.resolve_collector(0, Direction::Right, &emission, &[0], &table),
            None
        );
    }

    // --- Per-wire conjugation chaining -----------------------------------------------------------

    #[test]
    fn test_each_wire_chains_through_its_own_gates() {
        let (table, key) = table_with(DistributionType::UniformPauli);
        let emission = emit(Direction::Right, 2, key);
        let spine = Spine::new(
            vec![
                Item::Emission(emission.clone(), vec![0, 1]),
                Item::Gate(StandardGate::H, vec![0]),
                Item::Gate(StandardGate::S, vec![1]),
                Item::Collector(0),
            ],
            vec![collector(&[0, 1], vec![])],
        );
        let (sg, _source, target) = wire(&spine, 0, &emission, &[0, 1], 0, &table).unwrap();

        // One conjugation per gate, each on its own wire: the two wires never met, so nothing joins.
        let mut found = conjugations(&sg);
        found.sort_by_key(|(gate, _)| gate.name().to_string());
        assert_eq!(
            found,
            vec![(StandardGate::H, vec![0]), (StandardGate::S, vec![1]),]
        );
        // Both wires' frontiers end at the collector, so it has two distinct predecessors.
        assert_eq!(predecessors(&sg, target).len(), 2);
    }

    #[test]
    fn test_a_gate_spanning_both_wires_joins_their_chains() {
        let (table, key) = table_with(DistributionType::UniformPauli);
        let emission = emit(Direction::Right, 2, key);
        let spine = Spine::new(
            vec![
                Item::Emission(emission.clone(), vec![0, 1]),
                Item::Gate(StandardGate::CX, vec![0, 1]),
                Item::Collector(0),
            ],
            vec![collector(&[0, 1], vec![])],
        );
        let (sg, source, target) = wire(&spine, 0, &emission, &[0, 1], 0, &table).unwrap();

        // A conjugation by a two-qubit gate mixes its wires, so it is one joint node rather than two.
        assert_eq!(conjugations(&sg), vec![(StandardGate::CX, vec![0, 1])]);
        let joint = predecessors(&sg, target);
        assert_eq!(
            joint.len(),
            1,
            "both wires should have merged onto one node"
        );
        assert!(sg.graph[joint[0]].partition.len() == 1);
        assert_eq!(predecessors(&sg, joint[0]), vec![source]);
    }

    #[test]
    fn test_a_gate_off_the_emissions_wires_is_not_chained() {
        let (table, key) = table_with(DistributionType::UniformPauli);
        let emission = emit(Direction::Right, 1, key);
        let spine = Spine::new(
            vec![
                Item::Emission(emission.clone(), vec![0]),
                Item::Gate(StandardGate::H, vec![1]),
                Item::Collector(0),
            ],
            vec![collector(&[0], vec![])],
        );
        let (sg, source, target) = wire(&spine, 0, &emission, &[0], 0, &table).unwrap();

        assert!(conjugations(&sg).is_empty());
        // Nothing stood between them, so the emission feeds its collector directly.
        assert_eq!(predecessors(&sg, target), vec![source]);
    }

    #[test]
    fn test_a_crossed_collectors_absorbed_gates_still_conjugate() {
        let (table, key) = table_with(DistributionType::UniformPauli);
        let emission = emit(Direction::Right, 2, key);
        let spine = Spine::new(
            vec![
                Item::Emission(emission.clone(), vec![0, 1]),
                Item::Collector(0),
                Item::Collector(1),
            ],
            vec![
                // Declines the emission, but its absorbed gate is a real gate on the path.
                collector(&[0], vec![absorbed(StandardGate::H, 0)]),
                collector(&[0, 1], vec![]),
            ],
        );
        let target = spine
            .resolve_collector(0, Direction::Right, &emission, &[0, 1], &table)
            .unwrap();
        assert_eq!(target, 1);
        let (sg, _source, target_node) =
            wire(&spine, 0, &emission, &[0, 1], target, &table).unwrap();

        assert_eq!(conjugations(&sg), vec![(StandardGate::H, vec![0])]);
        // Wire 0 ends on the conjugation, wire 1 still on the emission itself.
        assert_eq!(predecessors(&sg, target_node).len(), 2);
    }

    // --- The `propagates` refusal ----------------------------------------------------------------

    #[test]
    fn test_propagation_refuses_a_gate_with_no_rule() {
        // A local U2 element admits single-qubit gates only, so a CX on its wire has no rule.
        let (table, key) = table_with(DistributionType::HaarU2);
        let emission = emit(Direction::Right, 1, key);
        let spine = Spine::new(
            vec![
                Item::Emission(emission.clone(), vec![0]),
                Item::Gate(StandardGate::CX, vec![0, 1]),
                Item::Collector(0),
            ],
            vec![collector(&[0], vec![])],
        );
        // Resolution still succeeds — it is the walk that discovers there is no rule.
        assert_eq!(
            spine.resolve_collector(0, Direction::Right, &emission, &[0], &table),
            Some(0)
        );
        assert!(matches!(
            wire(&spine, 0, &emission, &[0], 0, &table),
            Err(SamplexError::NoPropagationRule {
                virtual_type: VirtualType::U2,
                gate: StandardGate::CX,
            })
        ));
    }

    #[test]
    fn test_a_pauli_survives_the_same_gate_a_u2_element_does_not() {
        // The refusal is a property of the virtual type, not of the spine's shape: the same spine
        // with a Pauli emission wires cleanly.
        let (table, key) = table_with(DistributionType::UniformPauli);
        let emission = emit(Direction::Right, 1, key);
        let spine = Spine::new(
            vec![
                Item::Emission(emission.clone(), vec![0]),
                Item::Gate(StandardGate::CX, vec![0, 1]),
                Item::Collector(0),
            ],
            vec![collector(&[0], vec![])],
        );
        let (sg, _source, _target) = wire(&spine, 0, &emission, &[0], 0, &table).unwrap();
        assert_eq!(conjugations(&sg), vec![(StandardGate::CX, vec![0, 1])]);
    }
}
