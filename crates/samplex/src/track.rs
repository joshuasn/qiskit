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

//! The track: an emission circuit read as one flat run of positions.
//!
//! Propagation is never recorded in the emission circuit — it is derived from where an emission
//! stands, and this module is what that derivation reads. Flattening the box nesting away is what
//! makes it a scan rather than a tree walk: every box but a collector is dissolved so its gates sit in
//! line, and each collector is reduced to a single position carrying its absorbed run, so "the nearest
//! collector ahead" is just the next [`Item::Collector`] in the travel direction.
//!
//! A track holds no `Python` token and no `DAGCircuit`. Reading one out of IR2 is the job of the
//! adapter in [`lower`](crate::passes::lower), which is the only place that touches the DAG;
//! everything here works on the flat reading alone, so the propagation rules can be exercised by
//! building a track directly with [`Track::new`].
//!
//! [`SamplingGraphBuilder`] is the other half: the track is the reading, the builder is what writes a
//! [`SamplingGraph`] out of it. The two are here together for the same reason `DAGCircuit` and
//! `DAGCircuitBuilder` share a module — the builder's state is bookkeeping for the walk that has no
//! place on either finished thing.

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
    AbsorbedGate, Collect, CollectStep, Direction, Edge, Emission, Measure, Node, NodeKind,
    Propagate, SamplingGraph,
};
use crate::virtual_type::{VirtualType, propagates};

/// One collector, flattened out of the circuit.
pub struct Collector {
    /// Where this collector stood in the emission circuit, which is what identifies it across the
    /// two readings of that circuit. A `Site` is a path of node indices, so it survives the borrow
    /// that produced it and a track carrying one still holds no `DAGCircuit`.
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

/// Where one collector's angles live in the template's parameter vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectorParams {
    /// The collect box these angles were minted for. This is the join key: it is what lets
    /// [`Track::attach_param_indices`] find the range belonging to a given collector without relying
    /// on the two walks reporting their collectors in the same order.
    pub site: Site,
    /// The collector's qubits, ascending.
    pub qubits: Vec<usize>,
    pub synthesizer: SynthesizerType,
    /// Indices into the template's parameter vector, `qubits.len() * PARAMS_PER_QUBIT` of them,
    /// grouped per qubit in `qubits` order.
    pub param_indices: Vec<usize>,
}

/// What stands at one position on the track.
pub enum Item {
    Emission(Emit, Vec<usize>),
    /// One collector, by index into [`Track::collectors`].
    Collector(usize),
    Gate(StandardGate, Vec<usize>),
    Measure(Vec<usize>, Vec<usize>),
    Reset(Vec<usize>),
    /// A real operation with no virtual effect, kept so that a position on the track still stands for
    /// one instruction of the circuit it was read from.
    Opaque,
}

/// The circuit as a flat sequence, which is what makes the propagation walk a simple scan.
#[derive(Default)]
pub struct Track {
    /// The positions, in circuit order.
    items: Vec<Item>,
    /// The collectors, in the order the positions refer to them.
    collectors: Vec<Collector>,
}

impl Track {
    /// Build a track from a flat reading already in hand.
    ///
    /// This is the constructor that keeps the propagation rules reachable without a `DAGCircuit`.
    /// The index in every [`Item::Collector`] must be a position in `collectors`; that is the one
    /// invariant a caller upholds, and the reason [`Track::push_collector`] exists for the
    /// incremental case.
    pub fn new(items: Vec<Item>, collectors: Vec<Collector>) -> Self {
        Self { items, collectors }
    }

    /// Append one position.
    pub fn push_item(&mut self, item: Item) {
        self.items.push(item);
    }

    /// Append a collector together with the position that stands for it, keeping the two in step.
    ///
    /// The pairing is why the fields are not public: an [`Item::Collector`] index that does not name a
    /// collector is not a state a caller should be able to reach one push at a time.
    pub fn push_collector(&mut self, collector: Collector) {
        self.items.push(Item::Collector(self.collectors.len()));
        self.collectors.push(collector);
    }

    /// Give each collector the parameter range the template minted for the same collect box.
    ///
    /// The join is on [`Site`] and not on position. The two readings of the emission circuit are
    /// deliberately different walks — the template keeps a content box and recurses into it, this
    /// side dissolves it onto the track — so a shared arrival order is a convention neither walk is
    /// in a position to check, and comparing counts catches a collector that went missing but not
    /// one that moved. Keyed on identity a move is simply resolved, and the failure that remains is
    /// a collect box only one walk saw, which is reported: the alternative is a range landing on the
    /// wrong synth template, which mis-randomizes a circuit that still executes and round-trips.
    pub fn attach_param_indices(&mut self, params: &[CollectorParams]) -> Result<()> {
        let mut by_site: HashMap<&Site, &CollectorParams> = HashMap::with_capacity(params.len());
        for entry in params {
            // A site names one collect box, so two ranges under one site means a walk visited a node
            // twice — the join would have no defined answer, so it is refused rather than resolved.
            if by_site.insert(&entry.site, entry).is_some() {
                return Err(SamplexError::DuplicateCollectorParams(entry.site.clone()));
            }
        }
        for info in self.collectors.iter_mut() {
            let entry = by_site
                .remove(&info.site)
                .ok_or_else(|| SamplexError::CollectorNotInTemplate(info.site.clone()))?;
            info.param_indices = entry.param_indices.clone();
        }
        // The other direction: parameters minted for a collect box no collector on this track
        // claimed. Those angles would be in the template with nothing computing them.
        // Named in the template's own order rather than whichever key the map happens to yield, so
        // the message is identical on every run of the same failing input; determinism is a crate
        // invariant.
        if let Some(entry) = params
            .iter()
            .find(|entry| by_site.contains_key(&entry.site))
        {
            return Err(SamplexError::CollectorsNotInGraph {
                count: by_site.len(),
                site: entry.site.clone(),
            });
        }
        Ok(())
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
}

/// What identifies one conjugation node: a gate occurrence together with the flow crossing it.
///
/// The occurrence is `(track position, offset)`, the offset being the position within a collector's
/// absorbed run and zero for a gate that stands on its own.
type GateKey = (usize, usize, Direction, VirtualType);

/// Writes a [`SamplingGraph`] out of a [`Track`], holding the bookkeeping the walk needs.
///
/// The state a walk threads through — the graph, the node standing at each position, the conjugation
/// nodes shared between emissions — belongs to the construction and not to the finished graph, exactly
/// as `DAGCircuitBuilder`'s per-wire frontier does. Owning it here is what lets
/// [`propagate`](Self::propagate) take the two positions it is actually about.
///
/// Taking the track by value fixes the one order that matters. Each collector's parameter range is
/// copied into its `Collect` node, so [`Track::attach_param_indices`] has to have run before `new`
/// does — and after `new` there is no track left to join to.
pub struct SamplingGraphBuilder<'a> {
    track: Track,
    table: &'a DistributionTable,
    graph: SamplingGraph,
    /// The sink node per collector, parallel to `track.collectors`.
    collector_nodes: Vec<NodeIndex>,
    /// The node per emission, keyed by its position. Most positions are not emissions.
    emission_nodes: HashMap<usize, NodeIndex>,
    /// One node per *conjugation*, created lazily and shared by the emissions for which it is the
    /// same conjugation. Direction and virtual type are in the key because they change what the node
    /// computes, so sharing across them would fuse operations that cannot be evaluated as one.
    gate_nodes: HashMap<GateKey, NodeIndex>,
}

impl<'a> SamplingGraphBuilder<'a> {
    /// Start from a track whose collectors already carry their parameter ranges.
    ///
    /// The sinks are created here rather than during the walk, so an emission's walk always has a node
    /// to terminate at.
    pub fn new(track: Track, table: &'a DistributionTable) -> Self {
        let mut graph = SamplingGraph::new();
        let collector_nodes = track
            .collectors
            .iter()
            .map(|info| {
                graph.graph.add_node(Node::new(
                    info.qubits.clone(),
                    info.partition.clone(),
                    NodeKind::Collect(Collect {
                        synthesizer: info.synthesizer,
                        param_indices: info.param_indices.clone(),
                        steps: info.steps.clone(),
                    }),
                ))
            })
            .collect();
        Self {
            track,
            table,
            graph,
            collector_nodes,
            emission_nodes: HashMap::new(),
            gate_nodes: HashMap::new(),
        }
    }

    /// Wire every emission to its collector and hand back the finished graph.
    ///
    /// Resolution and wiring stay interleaved position by position rather than batched, because the
    /// first refusal is the one reported: an unresolvable emission late on the track must not pre-empt
    /// a gate with no propagation rule earlier on it.
    pub fn build(mut self) -> Result<SamplingGraph> {
        self.add_item_nodes()?;
        for position in 0..self.track.items.len() {
            if let Some(target) = self.target_for(position)? {
                self.propagate(position, target)?;
            }
        }
        Ok(self.graph)
    }

    /// Give every position that computes something a node, before any of them is wired.
    fn add_item_nodes(&mut self) -> Result<()> {
        let Self {
            track,
            table,
            graph,
            emission_nodes,
            ..
        } = self;
        for (position, item) in track.items.iter().enumerate() {
            match item {
                Item::Emission(emission, qubits) => {
                    let node = graph.graph.add_node(Node::new(
                        qubits.clone(),
                        emission.partition.clone(),
                        emission_kind(emission, table)?,
                    ));
                    emission_nodes.insert(position, node);
                }
                Item::Measure(qubits, clbits) => {
                    graph.graph.add_node(Node::singletons(
                        qubits.clone(),
                        NodeKind::Measure(Measure {
                            clbit_indices: clbits.clone(),
                        }),
                    ));
                }
                Item::Reset(qubits) => {
                    graph
                        .graph
                        .add_node(Node::singletons(qubits.clone(), NodeKind::Reset));
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// The collector this position's emission is headed for, or `None` if nothing is emitted here.
    ///
    /// Target resolution is purely positional: scan from the emission in its travel direction for the
    /// nearest compatible collector.
    fn target_for(&self, position: usize) -> Result<Option<usize>> {
        let Item::Emission(emission, qubits) = &self.track.items[position] else {
            return Ok(None);
        };
        // A local emission is resolved in place inside its collector's body, so it never reaches the
        // top-level track; anything here is still travelling.
        let direction = emission.direction.expect(
            "a local emission never surfaces as a top-level Item::Emission — it lives inside its \
             collector's body",
        );
        // Unreachable in well-formed IR2: build writes both of a box's collectors, so an emission
        // always has a compatible collector ahead of it. Reaching this means either the pairing was
        // broken between the two passes or a hand-built circuit has an emission nothing can collect,
        // which would otherwise show up as a randomization that is never undone — so it is reported
        // rather than skipped.
        self.track
            .resolve_collector(position, direction, emission, qubits, self.table)
            .ok_or_else(|| SamplexError::EmissionWithoutCollector {
                qubits: qubits.clone(),
                direction,
            })
            .map(Some)
    }

    /// Wire one emission's path: every gate between it and its collector, chained per qubit.
    fn propagate(&mut self, from: usize, target_index: usize) -> Result<()> {
        let Self {
            track,
            table,
            graph,
            collector_nodes,
            emission_nodes,
            gate_nodes,
        } = self;
        let Item::Emission(emission, emission_qubits) = &track.items[from] else {
            unreachable!("only a position `target_for` resolved a target for is propagated")
        };
        let source = emission_nodes[&from];
        let target_node = collector_nodes[target_index];
        let mut walk = Walk {
            frontier: emission_qubits.iter().map(|q| (*q, source)).collect(),
            tracked: emission_qubits.iter().copied().collect(),
            direction: emission.direction.expect(
                "a local emission never surfaces as a top-level Item::Emission — it lives inside its \
                 collector's body",
            ),
            virtual_type: emission.virtual_type(table),
        };

        // Walking in the emission's own direction is what makes propagation derivable rather than
        // recorded.
        let indices: Vec<usize> = match walk.direction {
            Direction::Right => (from + 1..track.items.len()).collect(),
            Direction::Left => (0..from).rev().collect(),
        };

        for index in indices {
            match &track.items[index] {
                Item::Collector(collector) if *collector == target_index => break,
                Item::Collector(collector) => {
                    // A foreign collector's absorbed gates are still real gates on this emission's
                    // path, so they conjugate it, even though that collector also multiplies them into
                    // its own layer.
                    let absorbed: Vec<&AbsorbedGate> =
                        track.collectors[*collector].gates().collect();
                    let order: Vec<usize> = match walk.direction {
                        Direction::Right => (0..absorbed.len()).collect(),
                        Direction::Left => (0..absorbed.len()).rev().collect(),
                    };
                    for offset in order {
                        let gate = &absorbed[offset];
                        chain(
                            graph,
                            gate_nodes,
                            &mut walk,
                            (index, offset),
                            gate.gate,
                            &gate.qubits,
                        )?;
                    }
                }
                Item::Gate(gate, gate_qubits) => {
                    chain(graph, gate_nodes, &mut walk, (index, 0), *gate, gate_qubits)?
                }
                _ => {}
            }
        }

        // Whatever each wire's virtual state ended up as is what the collector synthesizes.
        let ends: HashSet<NodeIndex> = walk.frontier.values().copied().collect();
        for end in ends {
            graph.graph.add_edge(end, target_node, Edge::new());
        }
        Ok(())
    }
}

/// One emission's walk towards its collector.
///
/// `frontier` moves as the walk crosses gates; the other three are fixed for the whole of it.
/// Bundling them is what keeps [`chain`] down to the gate it is chaining.
struct Walk {
    /// Where each tracked wire's virtual state currently ends.
    frontier: HashMap<usize, NodeIndex>,
    /// The emission's own wires. A gate touching none of them is not on this walk's path.
    tracked: HashSet<usize>,
    direction: Direction,
    virtual_type: VirtualType,
}

/// The graph node for one emission, resolved from the table entry its `dist` key points at.
fn emission_kind(emission: &Emit, table: &DistributionTable) -> Result<NodeKind> {
    let entry = table
        .get(emission.dist())
        .ok_or(SamplexError::MissingTableEntry {
            dist: emission.dist(),
        })?;
    Ok(NodeKind::Emission(Emission {
        key: emission.dist(),
        direction: emission.direction.expect(
            "a local emission never surfaces as a top-level Item::Emission — it lives inside its \
             collector's body",
        ),
        virtual_type: entry.virtual_type(),
    }))
}

/// Add or reuse the node for one gate and advance the walk's frontier over its qubits.
fn chain(
    graph: &mut SamplingGraph,
    gate_nodes: &mut HashMap<GateKey, NodeIndex>,
    walk: &mut Walk,
    occurrence: (usize, usize),
    gate: StandardGate,
    gate_qubits: &[usize],
) -> Result<()> {
    if !gate_qubits.iter().any(|q| walk.tracked.contains(q)) {
        return Ok(());
    }
    // Refuse rather than emit a node that cannot be evaluated: conjugating this virtual type by
    // this gate leaves its group, so there is no rule to apply.
    if !propagates(walk.virtual_type, gate) {
        return Err(SamplexError::NoPropagationRule {
            virtual_type: walk.virtual_type,
            gate,
        });
    }
    let key = (
        occurrence.0,
        occurrence.1,
        walk.direction,
        walk.virtual_type,
    );
    let node = *gate_nodes.entry(key).or_insert_with(|| {
        // One joint subsystem: a conjugation by a multi-qubit gate mixes its qubits, so they can
        // only be evaluated together.
        graph.graph.add_node(Node::joint(
            gate_qubits.to_vec(),
            NodeKind::Propagate(Propagate {
                gate,
                direction: walk.direction,
            }),
        ))
    });
    let predecessors: HashSet<NodeIndex> = gate_qubits
        .iter()
        .filter_map(|q| walk.frontier.get(q).copied())
        .collect();
    for predecessor in predecessors {
        graph.graph.add_edge(predecessor, node, Edge::new());
    }
    for q in gate_qubits.iter().filter(|q| walk.tracked.contains(*q)) {
        walk.frontier.insert(*q, node);
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
    use crate::sampling_graph::AbsorbedParam;

    // These tests are the reason the track is its own module: none of them touches a `DAGCircuit` or
    // a `Python` token, so the propagation rules can be pinned on hand-built tracks instead of on
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
    ///
    /// This drives the builder's internal seam rather than its interface, because `build` consumes and
    /// so cannot hand back the two handles the assertions are about. Everything the walk reads is
    /// still created the way `build` creates it.
    fn wire(
        track: Track,
        from: usize,
        target: usize,
        table: &DistributionTable,
    ) -> Result<(SamplingGraph, NodeIndex, NodeIndex)> {
        let mut builder = SamplingGraphBuilder::new(track, table);
        builder.add_item_nodes()?;
        let source = builder.emission_nodes[&from];
        let target_node = builder.collector_nodes[target];
        builder.propagate(from, target)?;
        Ok((builder.graph, source, target_node))
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
        let track = Track::new(
            vec![
                Item::Emission(emission.clone(), vec![0]),
                Item::Gate(StandardGate::H, vec![0]),
                Item::Collector(0),
                Item::Collector(1),
            ],
            vec![collector(&[0], vec![]), collector(&[0], vec![])],
        );
        assert_eq!(
            track.resolve_collector(0, Direction::Right, &emission, &[0], &table),
            Some(0)
        );
    }

    #[test]
    fn test_resolution_scans_the_way_the_emission_travels() {
        let (table, key) = table_with(DistributionType::UniformPauli);
        let track = Track::new(
            vec![Item::Collector(0), Item::Opaque, Item::Collector(1)],
            vec![collector(&[0], vec![]), collector(&[0], vec![])],
        );
        // The same position, read both ways: a twirl's two halves are exactly this pair.
        let far = emit(Direction::Right, 1, key);
        let near = emit(Direction::Left, 1, key);
        assert_eq!(
            track.resolve_collector(1, Direction::Right, &far, &[0], &table),
            Some(1)
        );
        assert_eq!(
            track.resolve_collector(1, Direction::Left, &near, &[0], &table),
            Some(0)
        );
    }

    #[test]
    fn test_a_collector_that_does_not_cover_the_emission_is_crossed() {
        let (table, key) = table_with(DistributionType::UniformPauli);
        let emission = emit(Direction::Right, 2, key);
        let track = Track::new(
            vec![
                Item::Emission(emission.clone(), vec![0, 1]),
                Item::Collector(0),
                Item::Collector(1),
            ],
            // The nearer collector covers only one of the emission's two wires, so it could not
            // synthesize the whole of what was emitted and the emission travels past it.
            vec![collector(&[0], vec![]), collector(&[0, 1], vec![])],
        );
        assert!(!track.collectors[0].accepts(&emission, &[0, 1], &table));
        assert!(track.collectors[1].accepts(&emission, &[0, 1], &table));
        assert_eq!(
            track.resolve_collector(0, Direction::Right, &emission, &[0, 1], &table),
            Some(1)
        );
    }

    #[test]
    fn test_an_emission_with_nothing_to_collect_it_does_not_resolve() {
        let (table, key) = table_with(DistributionType::UniformPauli);
        let emission = emit(Direction::Right, 1, key);
        let track = Track::new(
            vec![
                Item::Emission(emission.clone(), vec![0]),
                Item::Gate(StandardGate::H, vec![0]),
            ],
            vec![],
        );
        // The caller turns this into the "randomization could not be undone" error.
        assert_eq!(
            track.resolve_collector(0, Direction::Right, &emission, &[0], &table),
            None
        );
    }

    // --- Per-wire conjugation chaining -----------------------------------------------------------

    #[test]
    fn test_each_wire_chains_through_its_own_gates() {
        let (table, key) = table_with(DistributionType::UniformPauli);
        let emission = emit(Direction::Right, 2, key);
        let track = Track::new(
            vec![
                Item::Emission(emission.clone(), vec![0, 1]),
                Item::Gate(StandardGate::H, vec![0]),
                Item::Gate(StandardGate::S, vec![1]),
                Item::Collector(0),
            ],
            vec![collector(&[0, 1], vec![])],
        );
        let (sg, _source, target) = wire(track, 0, 0, &table).unwrap();

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
        let track = Track::new(
            vec![
                Item::Emission(emission.clone(), vec![0, 1]),
                Item::Gate(StandardGate::CX, vec![0, 1]),
                Item::Collector(0),
            ],
            vec![collector(&[0, 1], vec![])],
        );
        let (sg, source, target) = wire(track, 0, 0, &table).unwrap();

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
        let track = Track::new(
            vec![
                Item::Emission(emission.clone(), vec![0]),
                Item::Gate(StandardGate::H, vec![1]),
                Item::Collector(0),
            ],
            vec![collector(&[0], vec![])],
        );
        let (sg, source, target) = wire(track, 0, 0, &table).unwrap();

        assert!(conjugations(&sg).is_empty());
        // Nothing stood between them, so the emission feeds its collector directly.
        assert_eq!(predecessors(&sg, target), vec![source]);
    }

    #[test]
    fn test_a_crossed_collectors_absorbed_gates_still_conjugate() {
        let (table, key) = table_with(DistributionType::UniformPauli);
        let emission = emit(Direction::Right, 2, key);
        let track = Track::new(
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
        let target = track
            .resolve_collector(0, Direction::Right, &emission, &[0, 1], &table)
            .unwrap();
        assert_eq!(target, 1);
        let (sg, _source, target_node) = wire(track, 0, target, &table).unwrap();

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
        let track = Track::new(
            vec![
                Item::Emission(emission.clone(), vec![0]),
                Item::Gate(StandardGate::CX, vec![0, 1]),
                Item::Collector(0),
            ],
            vec![collector(&[0], vec![])],
        );
        // Resolution still succeeds — it is the walk that discovers there is no rule.
        assert_eq!(
            track.resolve_collector(0, Direction::Right, &emission, &[0], &table),
            Some(0)
        );
        assert!(matches!(
            wire(track, 0, 0, &table),
            Err(SamplexError::NoPropagationRule {
                virtual_type: VirtualType::U2,
                gate: StandardGate::CX,
            })
        ));
    }

    #[test]
    fn test_a_pauli_survives_the_same_gate_a_u2_element_does_not() {
        // The refusal is a property of the virtual type, not of the track's shape: the same track
        // with a Pauli emission wires cleanly.
        let (table, key) = table_with(DistributionType::UniformPauli);
        let emission = emit(Direction::Right, 1, key);
        let track = Track::new(
            vec![
                Item::Emission(emission.clone(), vec![0]),
                Item::Gate(StandardGate::CX, vec![0, 1]),
                Item::Collector(0),
            ],
            vec![collector(&[0], vec![])],
        );
        let (sg, _source, _target) = wire(track, 0, 0, &table).unwrap();
        assert_eq!(conjugations(&sg), vec![(StandardGate::CX, vec![0, 1])]);
    }

    // --- The `Site` join, on the two sides built by hand -----------------------------------------
    //
    // None of these needs a circuit either: a `Site` is a path of node indices and both sides of the
    // join are plain data, so what the join does with them can be pinned directly. A failed join is a
    // `SamplexError` variant, so a test names the failure it expects rather than only that there was
    // one.

    /// A site at `node`, reached through the boxes `scope` names.
    fn site(scope: &[usize], node: usize) -> Site {
        Site {
            scope: scope.iter().map(|index| NodeIndex::new(*index)).collect(),
            node: NodeIndex::new(node),
        }
    }

    /// One collector as the graph walk reports it, with no range attached yet.
    fn graph_collector(site: Site) -> Collector {
        Collector {
            site,
            qubits: vec![0],
            partition: Partition::singletons(1),
            synthesizer: SynthesizerType::RzSx,
            param_indices: Vec::new(),
            steps: Vec::new(),
        }
    }

    /// One collector as the template reports it, carrying the range it minted.
    fn template_params(site: Site, param_indices: Vec<usize>) -> CollectorParams {
        CollectorParams {
            site,
            qubits: vec![0],
            synthesizer: SynthesizerType::RzSx,
            param_indices,
        }
    }

    /// A track of nothing but collectors, pushed so their positions stay in step.
    fn track_of(collectors: Vec<Collector>) -> Track {
        let mut track = Track::default();
        for collector in collectors {
            track.push_collector(collector);
        }
        track
    }

    #[test]
    fn test_a_collector_keeps_its_own_range_when_the_walks_disagree_on_order() {
        // This is the divergence a count comparison cannot see: the same number of collectors on both
        // sides, in opposite orders. Positionally the second collector would take the first one's
        // angles, mis-randomizing a circuit that still executes. Keyed on the site there is nothing to
        // get wrong — the reorder is resolved rather than carried through.
        let template = vec![
            template_params(site(&[], 1), vec![0, 1, 2]),
            template_params(site(&[], 7), vec![3, 4, 5]),
        ];
        let mut track = track_of(vec![
            graph_collector(site(&[], 7)),
            graph_collector(site(&[], 1)),
        ]);
        track.attach_param_indices(&template).unwrap();
        assert_eq!(track.collectors[0].param_indices, vec![3, 4, 5]);
        assert_eq!(track.collectors[1].param_indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_the_same_node_index_in_two_scopes_is_two_collectors() {
        // Node indices are per-scope, so identity is the whole path and not its last step. A join keyed
        // on the node alone would fuse a nested collector with a top-level one.
        let template = vec![
            template_params(site(&[], 3), vec![0, 1, 2]),
            template_params(site(&[4], 3), vec![3, 4, 5]),
        ];
        let mut track = track_of(vec![
            graph_collector(site(&[4], 3)),
            graph_collector(site(&[], 3)),
        ]);
        track.attach_param_indices(&template).unwrap();
        assert_eq!(track.collectors[0].param_indices, vec![3, 4, 5]);
        assert_eq!(track.collectors[1].param_indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_a_collector_only_the_graph_walk_found_is_refused() {
        // Nothing minted its angles, so there is no honest range to give it. Reported, never skipped: a
        // collector left with an empty range would synthesize from parameters the template does not
        // have.
        let template = vec![template_params(site(&[], 1), vec![0, 1, 2])];
        let mut track = track_of(vec![
            graph_collector(site(&[], 1)),
            graph_collector(site(&[], 9)),
        ]);
        assert!(matches!(
            track.attach_param_indices(&template),
            Err(SamplexError::CollectorNotInTemplate(missing)) if missing == site(&[], 9)
        ));
    }

    #[test]
    fn test_a_collector_only_the_template_found_is_refused() {
        // The other direction: angles standing in the template with nothing in the graph computing
        // them.
        let template = vec![
            template_params(site(&[], 1), vec![0, 1, 2]),
            template_params(site(&[], 9), vec![3, 4, 5]),
        ];
        let mut track = track_of(vec![graph_collector(site(&[], 1))]);
        assert!(matches!(
            track.attach_param_indices(&template),
            Err(SamplexError::CollectorsNotInGraph { count: 1, site: missing })
                if missing == site(&[], 9)
        ));
    }

    #[test]
    fn test_two_ranges_for_one_site_is_refused() {
        // A site names one collect box, so the join would have no defined answer here.
        let template = vec![
            template_params(site(&[], 1), vec![0, 1, 2]),
            template_params(site(&[], 1), vec![3, 4, 5]),
        ];
        let mut track = track_of(vec![graph_collector(site(&[], 1))]);
        assert!(matches!(
            track.attach_param_indices(&template),
            Err(SamplexError::DuplicateCollectorParams(duplicated)) if duplicated == site(&[], 1)
        ));
    }
}
