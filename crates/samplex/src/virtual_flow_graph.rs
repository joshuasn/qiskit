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

use hashbrown::HashMap;
use pyo3::prelude::*;
use qiskit_circuit::operations::Operation;
use qiskit_circuit::standard_gate::StandardGate;
use rustworkx_core::petgraph::stable_graph::{NodeIndex, StableDiGraph};

use crate::distributions::{DistEntry, DistKey, DistributionTable};
use crate::emission_circuit::EmitPart;
use crate::parameters::ParamKey;
use crate::partition::Partition;
pub use crate::virtual_type::VirtualType;

// Re-export annotation enums so downstream code that imports from this module still compiles.
pub use crate::annotated_circuit::{
    ChangeBasisMode, DistributionType, Dressing, InjectionSite, Placement, SynthesizerType,
};

/// One node as seen from Python: `(kind, qubits, param_indices, steps)`.
///
/// `steps` is a [`Collect`]'s composition sequence — `("emit", qubits, [])` or `(gate name, qubits,
/// angles)` — and empty for every other kind. Only per-qubit order within it is meaningful; see
/// [`Collect::steps`].
pub type NodeSummary = (
    String,
    Vec<usize>,
    Vec<usize>,
    Vec<(String, Vec<usize>, Vec<String>)>,
);

// --- Enums ---

/// Which way virtual state flows through the circuit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    Left,
    Right,
}

impl Direction {
    /// The arrow used in rendered node labels.
    fn mark(&self) -> &'static str {
        match self {
            Self::Left => "⊲",
            Self::Right => "⊳",
        }
    }
}

// --- Edge ---

/// A flow of virtual state from one node to the next.
///
/// Carries no direction; that lives on the node the flow passes through. See
/// [`NodeKind::direction`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Edge {
    pub virtual_type: Option<VirtualType>,
}

impl Edge {
    pub fn new() -> Self {
        Edge::default()
    }
}

// --- Node structures ---

/// One node of the sampling graph: what happens, and on which circuit qubits.
///
/// A node names its qubits itself, unlike the IR2 instructions it came from — there is no qargs list
/// here to read them off — and its `partition` indexes into that list.
#[derive(Debug, Clone)]
pub struct Node {
    /// The circuit qubits this node acts on.
    pub qubits: Vec<usize>,
    /// How those qubits group into jointly-sampled subsystems, by index into `qubits`.
    pub partition: Partition,
    pub kind: NodeKind,
}

impl Node {
    /// A node with an explicit grouping of its qubits.
    ///
    /// Panics if the partition is not of `qubits.len()` — the two travel together, so a mismatch is a
    /// bug in whichever pass built them.
    pub fn new(qubits: Vec<usize>, partition: Partition, kind: NodeKind) -> Self {
        assert_eq!(
            qubits.len(),
            partition.num_qubits(),
            "a partition of {} qubits cannot describe a node on {:?}",
            partition.num_qubits(),
            qubits,
        );
        Node {
            qubits,
            partition,
            kind,
        }
    }

    /// A node sampling each of its qubits on its own: the common case.
    pub fn singletons(qubits: Vec<usize>, kind: NodeKind) -> Self {
        let partition = Partition::singletons(qubits.len());
        Node {
            qubits,
            partition,
            kind,
        }
    }

    /// A node holding all its qubits in one joint subsystem, as a multi-qubit gate's conjugation
    /// does.
    pub fn joint(qubits: Vec<usize>, kind: NodeKind) -> Self {
        let partition = Partition::whole(qubits.len());
        Node {
            qubits,
            partition,
            kind,
        }
    }

    /// This node's subsystems, as circuit qubits.
    pub fn groups(&self) -> Vec<Vec<usize>> {
        self.partition.groups(&self.qubits)
    }
}

#[derive(Debug, Clone)]
pub enum NodeKind {
    Emission(Emission),
    Collect(Collect),
    Propagate(Propagate),
    Measure(Measure),
    Reset,
}

impl NodeKind {
    pub fn is_source(&self) -> bool {
        matches!(self, Self::Emission(_) | Self::Reset)
    }

    pub fn is_sink(&self) -> bool {
        matches!(self, Self::Collect(_) | Self::Measure(_))
    }

    /// Which way virtual state flows out of this node, for the kinds that carry a flow at all.
    ///
    /// A `Propagate` node is created per handedness, so a node is never on paths of both
    /// directions.
    pub fn direction(&self) -> Option<Direction> {
        match self {
            Self::Emission(emission) => Some(emission.direction),
            Self::Propagate(propagate) => Some(propagate.direction),
            Self::Collect(_) | Self::Measure(_) | Self::Reset => None,
        }
    }
}

/// A source of virtual gates: one node per still-travelling `Emit` instruction in the emission
/// circuit.
///
/// Twirls, basis changes and noise injections share this one kind; the table entry `key` points at
/// is the source tag. Keyed rather than cloned out of the table, like [`LocalEmission`] — resolving
/// it (for drawing or [`nodes`](VirtualFlowGraph::nodes)) needs a
/// [`DistributionTable`](crate::distributions::DistributionTable) alongside.
#[derive(Debug, Clone)]
pub struct Emission {
    /// What this emission draws from; its discriminant is the source tag.
    pub key: DistKey,
    /// Which way the emitted state flows towards the collector that consumes it.
    pub direction: Direction,
    /// The algebraic type of the emitted virtual gate, as IR2 resolved it from the annotation.
    pub virtual_type: VirtualType,
}

/// One angle of an absorbed gate: ready to use, or awaiting a binding.
#[derive(Debug, Clone, PartialEq)]
pub enum AbsorbedParam {
    /// A literal angle.
    Bound(f64),
    /// A symbolic angle, keyed into the run's
    /// [`ParameterTable`](crate::parameters::ParameterTable).
    Symbolic(ParamKey),
}

/// A gate folded into a collector's synthesized layer rather than executed separately.
///
/// Not `Eq`: a bound angle is an `f64`.
#[derive(Debug, Clone, PartialEq)]
pub struct AbsorbedGate {
    pub gate: StandardGate,
    /// Circuit qubits, ascending.
    pub qubits: Vec<usize>,
    /// Angles, parallel with the gate's own parameter list. They never reach the template — the
    /// collector folds them into the angles it synthesizes.
    pub params: Vec<AbsorbedParam>,
}

/// An emission its collector owns outright, resolved in place rather than propagating.
///
/// Read straight from the distribution table at sampling time and composed at this step's position.
/// It gets no [`Emission`] node, since there is no chain to model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalEmission {
    /// The circuit qubits this emission covers.
    pub qubits: Vec<usize>,
    /// How those qubits group into subsystems, by index into `qubits`.
    pub partition: Partition,
    /// Per-part descriptors, parallel with `partition.iter()`.
    pub parts: Vec<EmitPart>,
}

impl LocalEmission {
    /// This emission's subsystems, as circuit qubits.
    pub fn groups(&self) -> Vec<Vec<usize>> {
        self.partition.groups(&self.qubits)
    }
}

/// One step in what a collector composes, in the order given by [`Collect::steps`].
///
/// Only what the collector *owns*. A still-travelling emission arrives over a graph edge instead,
/// and is not recorded here. Not `Eq`; see [`AbsorbedGate`].
#[derive(Debug, Clone, PartialEq)]
pub enum CollectStep {
    /// A local emission: read a value from the distribution table and compose it here.
    Local(LocalEmission),
    /// A gate folded into this layer rather than executed separately.
    Gate(AbsorbedGate),
}

#[derive(Debug, Clone)]
pub struct Collect {
    pub synthesizer: SynthesizerType,
    pub param_indices: Vec<usize>,
    /// Everything this collector composes, as **a linear extension of its per-qubit dependency
    /// order**. Only each qubit's own subsequence is meaningful.
    ///
    /// It is **not** circuit order and must not be read as such: steps on disjoint wires come out
    /// lowest-qubit-first however they were written. Project onto a wire before relying on relative
    /// order — an assertion on the flat list pins an arbitrary choice.
    pub steps: Vec<CollectStep>,
}

/// Filter a slice of collect steps to just the absorbed gates, keeping the sequence's order.
pub fn collect_step_gates(steps: &[CollectStep]) -> impl Iterator<Item = &AbsorbedGate> {
    steps.iter().filter_map(|step| match step {
        CollectStep::Gate(gate) => Some(gate),
        CollectStep::Local(_) => None,
    })
}

impl Collect {
    /// The absorbed gates alone, in [`steps`](Self::steps) order.
    pub fn gates(&self) -> impl Iterator<Item = &AbsorbedGate> {
        collect_step_gates(&self.steps)
    }
}

/// One conjugation of virtual state by a real gate.
#[derive(Debug, Clone)]
pub struct Propagate {
    pub gate: StandardGate,
    /// The handedness of the flow through this node, which fixes which conjugation applies.
    pub direction: Direction,
}

#[derive(Debug, Clone)]
pub struct Measure {
    pub clbit_indices: Vec<usize>,
}

// --- Graph container ---

#[pyclass(module = "qiskit._accelerate.samplex", skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct VirtualFlowGraph {
    pub graph: StableDiGraph<Node, Edge>,
}

impl VirtualFlowGraph {
    pub fn new() -> Self {
        VirtualFlowGraph {
            graph: StableDiGraph::new(),
        }
    }
}

impl Default for VirtualFlowGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl VirtualFlowGraph {
    #[getter]
    fn num_nodes(&self) -> usize {
        self.graph.node_count()
    }

    #[getter]
    fn num_edges(&self) -> usize {
        self.graph.edge_count()
    }

    /// Per-node summary for inspection and testing: `(kind, qubits, param_indices, steps)`.
    ///
    /// Nodes come in index order, and [`edges`](Self::edges) refers to positions in this list.
    /// `steps` is non-empty only for `Collect`; only its per-wire order is meaningful, per
    /// [`Collect::steps`]. A local emission's angles are empty — its payload is in the distribution
    /// table.
    ///
    /// Pass `table` to resolve an `Emission`'s kind to its full source tag (e.g.
    /// `"emit:UniformPauli"`); without it, it falls back to `"emit:#key"`.
    #[pyo3(signature = (table=None))]
    fn nodes(&self, table: Option<&DistributionTable>) -> Vec<NodeSummary> {
        self.graph
            .node_indices()
            .map(|index| {
                let node = &self.graph[index];
                // Ascending, not the node's own order: this is a flat readout for inspection, so it
                // should not vary with how a pass happened to lay the qubits out.
                let mut qubits: Vec<usize> = node.qubits.clone();
                qubits.sort_unstable();
                let (kind, params, absorbed) = match &node.kind {
                    NodeKind::Emission(emission) => {
                        (emission_label(emission.key, table), Vec::new(), Vec::new())
                    }
                    NodeKind::Collect(collect) => (
                        format!("collect:{:?}", collect.synthesizer),
                        collect.param_indices.clone(),
                        collect
                            .steps
                            .iter()
                            .map(|step| match step {
                                CollectStep::Local(local) => {
                                    let mut qs: Vec<usize> = local.qubits.clone();
                                    qs.sort_unstable();
                                    ("emit".to_string(), qs, Vec::new())
                                }
                                CollectStep::Gate(gate) => (
                                    gate.gate.name().to_string(),
                                    gate.qubits.clone(),
                                    gate.params.iter().map(param_label).collect(),
                                ),
                            })
                            .collect(),
                    ),
                    NodeKind::Propagate(propagate) => (
                        format!("propagate:{}", propagate.gate.name()),
                        Vec::new(),
                        Vec::new(),
                    ),
                    NodeKind::Measure(_) => ("measure".to_string(), Vec::new(), Vec::new()),
                    NodeKind::Reset => ("reset".to_string(), Vec::new(), Vec::new()),
                };
                (kind, qubits, params, absorbed)
            })
            .collect()
    }

    /// Edges as `(source, target, direction)`, indexing into [`nodes`](Self::nodes).
    ///
    /// The direction is the source node's, and `"none"` for a node that carries no flow of its own.
    fn edges(&self) -> Vec<(usize, usize, String)> {
        use rustworkx_core::petgraph::visit::{EdgeRef, IntoEdgeReferences};
        let order: HashMap<NodeIndex, usize> = self
            .graph
            .node_indices()
            .enumerate()
            .map(|(position, index)| (index, position))
            .collect();
        (&self.graph)
            .edge_references()
            .map(|edge| {
                let direction = match self.graph[edge.source()].kind.direction() {
                    Some(Direction::Left) => "left",
                    Some(Direction::Right) => "right",
                    None => "none",
                };
                (
                    order[&edge.source()],
                    order[&edge.target()],
                    direction.to_string(),
                )
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "VirtualFlowGraph(nodes={}, edges={})",
            self.graph.node_count(),
            self.graph.edge_count()
        )
    }

    /// Return a Graphviz DOT representation of the graph.
    ///
    /// Pass `table` to resolve each `Collect` step's local emission to its full distribution
    /// identity (e.g. `Emit(UniformPauli)`); without it, those steps render as `Emit(#key)`.
    #[pyo3(signature = (table=None))]
    fn to_dot(&self, table: Option<&DistributionTable>) -> String {
        use rustworkx_core::petgraph::visit::{EdgeRef, IntoEdgeReferences, IntoNodeReferences};
        use std::fmt::Write;

        let mut dot = String::new();
        writeln!(dot, "digraph VFG {{").unwrap();
        writeln!(dot, "    rankdir=TB;").unwrap();
        writeln!(
            dot,
            "    node [shape=box, style=filled, fontname=\"Helvetica\"];"
        )
        .unwrap();

        for (idx, node) in self.graph.node_references() {
            let (label, color) = node_label_color(&node.kind, &node.qubits, &node.partition, table);
            writeln!(
                dot,
                "    n{} [label={}, fillcolor=\"{}\"];",
                idx.index(),
                dot_escape(&label),
                color
            )
            .unwrap();
        }

        for edge in self.graph.edge_references() {
            let label = edge_label(edge.weight());
            writeln!(
                dot,
                "    n{} -> n{} [label={}];",
                edge.source().index(),
                edge.target().index(),
                dot_escape(&label)
            )
            .unwrap();
        }

        writeln!(dot, "}}").unwrap();
        dot
    }
}

fn dot_escape(s: &str) -> String {
    let escaped = s
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\l");
    format!("\"{}\"", escaped)
}

/// How a grouping of qubits reads in a rendered label: `[0, 1]` when each qubit stands alone, and
/// `[[0, 1], [2, 3]]` when some are sampled jointly.
fn format_groups(qubits: &[usize], partition: &Partition) -> String {
    let groups = partition.groups(qubits);
    if partition.is_singletons() {
        let flat: Vec<usize> = groups.iter().map(|group| group[0]).collect();
        format!("{:?}", flat)
    } else {
        format!("{:?}", groups)
    }
}

/// How one absorbed angle reads in a node summary: its value if bound, else `#key`.
fn param_label(param: &AbsorbedParam) -> String {
    match param {
        AbsorbedParam::Bound(value) => format!("{value}"),
        AbsorbedParam::Symbolic(key) => format!("#{}", key.0),
    }
}

/// The verb-form label for a distribution entry, as it reads in a rendered node's box: `Emit(...)`,
/// `ChangeBasis(...)`, or `InjectNoise(...)`.
fn dist_entry_verb_label(entry: &DistEntry) -> String {
    match entry {
        DistEntry::Distribution(distribution) => format!("Emit({distribution:?})"),
        DistEntry::Basis { mode, .. } => format!("ChangeBasis({mode:?})"),
        DistEntry::Noise { reference, .. } => format!("InjectNoise({reference})"),
    }
}

/// How one step of a [`Collect`]'s body reads in the rendered node label.
///
/// A local emission is named the same way a still-travelling one is elsewhere in the graph —
/// `Emit(UniformPauli)` and friends — when `table` resolves its key; without a table, the raw key
/// is all there is to show.
fn collect_step_label(step: &CollectStep, table: Option<&DistributionTable>) -> String {
    match step {
        CollectStep::Local(local) => {
            let label = local
                .parts
                .first()
                .map(|part| match table.and_then(|t| t.get(part.dist)) {
                    Some(entry) => dist_entry_verb_label(entry),
                    None => format!("Emit(#{})", part.dist.0),
                })
                .unwrap_or_else(|| "Emit".to_string());
            format!(
                "{} {}",
                label,
                format_groups(&local.qubits, &local.partition)
            )
        }
        CollectStep::Gate(gate) => {
            let params: Vec<String> = gate.params.iter().map(param_label).collect();
            if params.is_empty() {
                format!("{} {:?}", gate.gate.name(), gate.qubits)
            } else {
                format!(
                    "{}({}) {:?}",
                    gate.gate.name(),
                    params.join(", "),
                    gate.qubits
                )
            }
        }
    }
}

/// The `kind` string for an emission node. Callers select nodes by these prefixes.
///
/// Without a table, the key can't be resolved to its source tag, so this falls back to a generic
/// `"emit:#key"` — always that prefix, never `"change_basis:"`/`"inject_noise:"`, since there is no
/// way to tell which without resolving it. Mirrors `collect_step_label`'s `Emit(#key)` fallback.
fn emission_label(key: DistKey, table: Option<&DistributionTable>) -> String {
    match table.and_then(|t| t.get(key)) {
        Some(DistEntry::Distribution(distribution)) => format!("emit:{distribution:?}"),
        Some(DistEntry::Basis { ref_id, .. }) => format!("change_basis:{ref_id}"),
        Some(DistEntry::Noise { reference, .. }) => format!("inject_noise:{reference}"),
        None => format!("emit:#{}", key.0),
    }
}

fn node_label_color(
    kind: &NodeKind,
    node_qubits: &[usize],
    partition: &Partition,
    table: Option<&DistributionTable>,
) -> (String, &'static str) {
    let qubits = format_groups(node_qubits, partition);
    match kind {
        // One node kind, but distinct colours: which annotation produced an emission is the first
        // thing you look for in a rendered graph.
        NodeKind::Emission(e) => {
            let (label, color) = match table.and_then(|t| t.get(e.key)) {
                Some(entry) => {
                    let color = match entry {
                        DistEntry::Distribution(_) => "#a8d8ea",
                        DistEntry::Basis { .. } => "#fff2cc",
                        DistEntry::Noise { .. } => "#f8cecc",
                    };
                    (dist_entry_verb_label(entry), color)
                }
                None => (format!("Emit(#{})", e.key.0), "#d9d9d9"),
            };
            (
                format!("{} {} {}", label, e.direction.mark(), qubits),
                color,
            )
        }
        NodeKind::Collect(c) => {
            let mut label = format!("Collect({:?}) {}", c.synthesizer, qubits);
            for step in &c.steps {
                label.push_str("\n  ");
                label.push_str(&collect_step_label(step, table));
            }
            (label, "#f8c8dc")
        }
        NodeKind::Propagate(p) => (
            format!("{}{:?} {}", p.direction.mark(), p.gate, qubits),
            "#fffacd",
        ),
        NodeKind::Measure(m) => (
            format!("Measure cl{:?} {}", m.clbit_indices, qubits),
            "#d5e8d4",
        ),
        NodeKind::Reset => (format!("Reset {}", qubits), "#e1d5e7"),
    }
}

/// The edge label. Direction is on the nodes, so an edge only ever shows what type is flowing.
fn edge_label(edge: &Edge) -> String {
    match edge.virtual_type {
        Some(vt) => format!("{vt:?}"),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::DistKey;

    fn make_partition(parts: &[&[usize]]) -> Partition {
        Partition::new(parts.iter().map(|p| p.to_vec().into_boxed_slice())).unwrap()
    }

    fn emission(key: DistKey, direction: Direction) -> Emission {
        Emission {
            key,
            direction,
            virtual_type: VirtualType::Pauli,
        }
    }

    #[test]
    fn test_construct_emission_node() {
        let node = Node::singletons(
            vec![0, 1],
            NodeKind::Emission(emission(DistKey(0), Direction::Right)),
        );
        assert_eq!(node.partition.len(), 2);
        assert_eq!(node.groups(), vec![vec![0], vec![1]]);
        assert!(matches!(node.kind, NodeKind::Emission(_)));
    }

    #[test]
    fn test_a_node_resolves_its_partition_against_its_own_qubits() {
        // The partition holds indices, so the same one describes different subsystems depending on
        // which wires the node landed on.
        let kind = NodeKind::Emission(emission(DistKey(0), Direction::Right));
        let node = Node::new(vec![4, 7, 2], make_partition(&[&[0], &[2, 1]]), kind);
        assert_eq!(node.groups(), vec![vec![4], vec![2, 7]]);
        assert_eq!(node.partition.num_qubits(), 3);
    }

    #[test]
    fn test_a_node_rejects_a_partition_of_the_wrong_width() {
        let kind = NodeKind::Reset;
        let result =
            std::panic::catch_unwind(|| Node::new(vec![0, 1], Partition::singletons(3), kind));
        assert!(result.is_err());
    }

    #[test]
    fn test_emission_label_names_its_source() {
        // The three annotation kinds share one node kind, so the label is the only thing that still
        // distinguishes them — and the Python tests select nodes by these prefixes.
        let mut table = DistributionTable::new();
        let pauli = table.intern(DistEntry::Distribution(DistributionType::UniformPauli));
        let basis = table.intern(DistEntry::Basis {
            mode: ChangeBasisMode::MeasurePauli,
            ref_id: "basis_changes.b0".to_string(),
        });
        let noise = table.intern(DistEntry::Noise {
            reference: "n0".to_string(),
            modifier: None,
        });

        assert_eq!(emission_label(pauli, Some(&table)), "emit:UniformPauli");
        assert_eq!(
            emission_label(basis, Some(&table)),
            "change_basis:basis_changes.b0"
        );
        assert_eq!(emission_label(noise, Some(&table)), "inject_noise:n0");

        // Without a table (or with a key it doesn't have), it falls back to a generic tag.
        assert_eq!(emission_label(pauli, None), "emit:#0");
        assert_eq!(emission_label(DistKey(99), Some(&table)), "emit:#99");
    }

    #[test]
    fn test_construct_propagate_node() {
        let node = Node::new(
            vec![0, 1, 2, 3],
            make_partition(&[&[0, 1], &[2, 3]]),
            NodeKind::Propagate(Propagate {
                gate: StandardGate::CX,
                direction: Direction::Right,
            }),
        );
        assert_eq!(node.partition.len(), 2);
        if let NodeKind::Propagate(ref p) = node.kind {
            assert_eq!(p.gate, StandardGate::CX);
            assert_eq!(p.direction, Direction::Right);
        } else {
            panic!("expected Propagate");
        }
    }

    #[test]
    fn test_construct_collect_node() {
        let node = Node::singletons(
            vec![0, 1, 2],
            NodeKind::Collect(Collect {
                synthesizer: SynthesizerType::RzSx,
                param_indices: vec![0, 1, 2, 3, 4, 5, 6, 7, 8],
                steps: Vec::new(),
            }),
        );
        if let NodeKind::Collect(ref c) = node.kind {
            assert_eq!(c.param_indices.len(), 9);
        } else {
            panic!("expected Collect");
        }
    }

    #[test]
    fn test_collect_label_shows_its_steps() {
        // The rendered box is the only place a Collect's absorbed body is visible at a glance, so
        // the label must name each step rather than only the collector itself.
        let steps = vec![
            CollectStep::Local(LocalEmission {
                qubits: vec![0],
                partition: Partition::singletons(1),
                parts: vec![EmitPart {
                    dist: DistKey(0),
                    draw: 0,
                    adjoint: false,
                }],
            }),
            CollectStep::Gate(AbsorbedGate {
                gate: StandardGate::RZ,
                qubits: vec![1],
                params: vec![AbsorbedParam::Bound(std::f64::consts::PI)],
            }),
            CollectStep::Gate(AbsorbedGate {
                gate: StandardGate::X,
                qubits: vec![2],
                params: Vec::new(),
            }),
        ];
        let kind = NodeKind::Collect(Collect {
            synthesizer: SynthesizerType::RzSx,
            param_indices: vec![0],
            steps,
        });
        let qubits = vec![0, 1, 2];
        let partition = Partition::singletons(3);

        // Without a table, the local emission falls back to its raw key.
        let (label, _color) = node_label_color(&kind, &qubits, &partition, None);
        assert!(label.contains("\n  Emit(#0) [0]"), "label was: {label}");
        assert!(label.contains("\n  rz(3.14"), "label was: {label}");
        // A parameter-free gate gets no empty parens.
        assert!(label.contains("\n  x [2]"), "label was: {label}");
        assert!(!label.contains("x() [2]"), "label was: {label}");

        // With a table, it resolves the same way a still-travelling Emit does.
        let mut table = DistributionTable::new();
        table.intern(DistEntry::Distribution(DistributionType::UniformPauli));
        let (label, _color) = node_label_color(&kind, &qubits, &partition, Some(&table));
        assert!(
            label.contains("\n  Emit(UniformPauli) [0]"),
            "label was: {label}"
        );
    }

    #[test]
    fn test_labels_show_joint_subsystems_as_groups() {
        // A partition that is not all singletons is the whole reason a node carries one, so the label
        // has to show the grouping rather than a flat qubit list.
        let kind = NodeKind::Emission(emission(DistKey(0), Direction::Right));
        let (label, _color) =
            node_label_color(&kind, &[4, 7, 2], &make_partition(&[&[0, 1], &[2]]), None);
        assert!(label.contains("[[4, 7], [2]]"), "label was: {label}");
        let (label, _color) = node_label_color(&kind, &[4, 7], &Partition::singletons(2), None);
        assert!(label.contains("[4, 7]"), "label was: {label}");
    }

    #[test]
    fn test_construct_measure_node() {
        let node = Node::singletons(
            vec![0, 1],
            NodeKind::Measure(Measure {
                clbit_indices: vec![0, 1],
            }),
        );
        assert!(matches!(node.kind, NodeKind::Measure(_)));
    }

    #[test]
    fn test_construct_reset_node() {
        let node = Node::singletons(vec![3], NodeKind::Reset);
        assert!(matches!(node.kind, NodeKind::Reset));
    }

    #[test]
    fn test_basis_and_noise_are_emissions_too() {
        // Twirls, basis changes and noise injections all share this one node kind — the key just
        // points at whichever table entry the source annotation produced.
        let kind = NodeKind::Emission(emission(DistKey(1), Direction::Left));
        assert!(matches!(kind, NodeKind::Emission(_)));
        assert!(kind.is_source());
    }

    #[test]
    fn test_sources_and_sinks() {
        let emit = NodeKind::Emission(emission(DistKey(0), Direction::Right));
        assert!(emit.is_source() && !emit.is_sink());
        assert!(NodeKind::Reset.is_source() && !NodeKind::Reset.is_sink());
        let collect = collect_kind();
        assert!(collect.is_sink() && !collect.is_source());
        let measure = NodeKind::Measure(Measure {
            clbit_indices: vec![0],
        });
        assert!(measure.is_sink() && !measure.is_source());
    }

    #[test]
    fn test_direction_comes_from_the_node() {
        // Direction is no longer on edges, so these are the only places it can be read from.
        let emit = NodeKind::Emission(emission(DistKey(0), Direction::Left));
        assert_eq!(emit.direction(), Some(Direction::Left));
        let propagate = NodeKind::Propagate(Propagate {
            gate: StandardGate::CX,
            direction: Direction::Right,
        });
        assert_eq!(propagate.direction(), Some(Direction::Right));
        // A collector is where a flow ends, so it has no direction of its own.
        assert_eq!(collect_kind().direction(), None);
    }

    fn collect_kind() -> NodeKind {
        NodeKind::Collect(Collect {
            synthesizer: SynthesizerType::RzSx,
            param_indices: vec![0, 1, 2, 3, 4, 5],
            steps: Vec::new(),
        })
    }

    #[test]
    fn test_graph_add_nodes_and_edges() {
        let mut vfg = VirtualFlowGraph::new();

        let emit_idx = vfg.graph.add_node(Node::singletons(
            vec![0, 1],
            NodeKind::Emission(emission(DistKey(0), Direction::Right)),
        ));

        let propagate_idx = vfg.graph.add_node(Node::joint(
            vec![0, 1],
            NodeKind::Propagate(Propagate {
                gate: StandardGate::CX,
                direction: Direction::Right,
            }),
        ));

        let collect_idx = vfg
            .graph
            .add_node(Node::singletons(vec![0, 1], collect_kind()));

        vfg.graph.add_edge(emit_idx, propagate_idx, Edge::new());
        vfg.graph.add_edge(emit_idx, collect_idx, Edge::new());
        vfg.graph.add_edge(propagate_idx, collect_idx, Edge::new());

        assert_eq!(vfg.graph.node_count(), 3);
        assert_eq!(vfg.graph.edge_count(), 3);
        // The reported direction is the source node's, for every edge.
        assert_eq!(
            vfg.edges(),
            vec![
                (0, 1, "right".to_string()),
                (0, 2, "right".to_string()),
                (1, 2, "right".to_string()),
            ]
        );
    }
}
