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

use pyo3::prelude::*;
use qiskit_circuit::standard_gate::StandardGate;
use hashbrown::HashMap;
use qiskit_circuit::operations::Operation;
use rustworkx_core::petgraph::stable_graph::{NodeIndex, StableDiGraph};

use crate::distributions::DistEntry;
use crate::emission_circuit::EmitPart;
use crate::partition::Partition;
pub use crate::virtual_type::VirtualType;

// Re-export annotation enums so downstream code that imports from this module still compiles.
pub use crate::annotated_circuit::{
    ChangeBasisMode, DistributionType, Dressing, InjectionSite, Placement, SynthesizerType,
};

/// One node as seen from Python: `(kind, qubits, param_indices, steps)`.
///
/// `steps` is a `Collect`'s composition order — `("emit", [id])` or `(gate name, qubits)` — and empty
/// for every other kind.
pub type NodeSummary = (String, Vec<usize>, Vec<usize>, Vec<(String, Vec<usize>)>);

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
/// Deliberately carries no direction. Direction is fixed when an emission is created and never
/// changes along a path, so it belongs to the node the flow passes *through* rather than to each
/// edge — see [`NodeKind::direction`]. Having it in both places is what let the right-dressing
/// propagation bug hide.
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

#[derive(Debug, Clone)]
pub struct Node {
    pub partition: Partition,
    pub kind: NodeKind,
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
    /// This is where direction lives now that edges do not carry it. A `Propagate` node is created
    /// per handedness, so a node is never on paths of both directions at once.
    pub fn direction(&self) -> Option<Direction> {
        match self {
            Self::Emission(emission) => Some(emission.direction),
            Self::Propagate(propagate) => Some(propagate.direction),
            Self::Collect(_) | Self::Measure(_) | Self::Reset => None,
        }
    }
}

/// A source of virtual gates: one node per `Emit` instruction in the emission circuit.
///
/// Twirls, basis changes and noise injections are **one** node kind rather than three, mirroring IR2
/// where they are already one instruction with a source tag. The tag here is the [`DistEntry`]
/// discriminant — `Distribution` for a twirl, `Basis` for a basis change or local-Clifford
/// injection, `Noise` for an injected Pauli-Lindblad map — so nothing separate needs storing.
///
/// The entry is cloned out of the [`DistributionTable`](crate::distributions::DistributionTable)
/// rather than referenced by key, so a graph is readable without its table alongside.
#[derive(Debug, Clone)]
pub struct Emission {
    /// What this emission draws from; its discriminant is the source tag.
    pub entry: DistEntry,
    /// Which way the emitted state flows towards the collector that consumes it.
    pub direction: Direction,
    /// The algebraic type of the emitted virtual gate.
    ///
    /// Taken from the emission rather than re-derived from `entry`: IR2 already resolved it from the
    /// annotation, and that is the authoritative value.
    pub virtual_type: VirtualType,
}

/// A gate folded into a collector's synthesized layer rather than executed separately.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AbsorbedGate {
    pub gate: StandardGate,
    /// Circuit qubits, ascending.
    pub qubits: Vec<usize>,
}

/// An emission owned directly by its collector — adjacent to it, never propagating through gates.
///
/// At sampling time the collector reads the sampled value from the distribution table and composes
/// it at the position its local `Emit` instruction (`direction: None`) sits at in the collector body.
/// No VFG `Emission` node — the local emission is folded straight into the collector's own step
/// list.
///
/// Neither direction nor source is stored: position in the collector body IS the composition order,
/// and the distribution table entry (reachable through [`EmitPart::dist`]) already encodes the
/// emission kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalEmission {
    pub partition: Partition,
    /// Per-part descriptors, parallel with `partition.iter()`.
    pub parts: Vec<EmitPart>,
}

/// One step in what a collector composes, in circuit order.
///
/// Only what the collector *owns*: local emissions (table reads) and absorbed gates. Incoming
/// emissions (far twirl halves arriving via graph edges) are NOT recorded here — their composition
/// position is derived from graph topology (direction + generation distance) at evaluation time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CollectStep {
    /// A local emission: read a value from the distribution table and compose it here.
    /// No VFG Emission node — the collector owns this directly.
    Local(LocalEmission),
    /// A gate folded into this layer rather than executed separately.
    Gate(AbsorbedGate),
}

#[derive(Debug, Clone)]
pub struct Collect {
    pub synthesizer: SynthesizerType,
    pub param_indices: Vec<usize>,
    /// Everything this collector composes, in circuit order.
    ///
    /// The collector *owns* its absorbed gates rather than them being separate `Multiply` nodes on some
    /// emission's chain. That is deliberate: after merging, a collector holds absorbed gates from
    /// several boxes and there is no way to attribute each to the emission it multiplies into.
    ///
    /// It is one ordered sequence rather than a set of collected emissions plus a list of gates,
    /// because position in the layer is meaningful: a `ChangeBasis` wraps the whole box and so composes
    /// *outside* the absorbed easy gates, whereas an injection or twirl attaches to the hard content
    /// and composes *inside* them.
    ///
    /// An enclosing emission that merely *crosses* this collector still gets ordinary `Propagate` nodes
    /// for these gates, derived positionally. The two roles are independent.
    pub steps: Vec<CollectStep>,
}

/// Filter a slice of collect steps to just the absorbed gates, in order.
pub fn collect_step_gates(steps: &[CollectStep]) -> impl Iterator<Item = &AbsorbedGate> {
    steps.iter().filter_map(|step| match step {
        CollectStep::Gate(gate) => Some(gate),
        CollectStep::Local(_) => None,
    })
}

impl Collect {
    /// The absorbed gates, in circuit order, ignoring the emissions interleaved between them.
    pub fn gates(&self) -> impl Iterator<Item = &AbsorbedGate> {
        collect_step_gates(&self.steps)
    }
}

/// One conjugation of virtual state by a real gate.
///
/// There is no longer a mode: the "multiply" case existed only because the old walker made a node per
/// absorbed single-qubit gate. Absorbed gates are now data on the [`Collect`] that owns them, so every
/// `Propagate` node is a genuine conjugation.
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
    /// `steps` is only non-empty for `Collect`, and gives its composition order: `("emit", [id])` for
    /// a consumed emission, `(gate name, qubits)` for a gate folded into the layer. Both appear in one
    /// list because their relative order is meaningful.
    fn nodes(&self) -> Vec<NodeSummary> {
        self.graph
            .node_indices()
            .map(|index| {
                let node = &self.graph[index];
                let mut qubits: Vec<usize> = node.partition.all_elements().iter().copied().collect();
                qubits.sort_unstable();
                let (kind, params, absorbed) = match &node.kind {
                    NodeKind::Emission(emission) => {
                        (emission_label(&emission.entry), Vec::new(), Vec::new())
                    }
                    NodeKind::Collect(collect) => (
                        format!("collect:{:?}", collect.synthesizer),
                        collect.param_indices.clone(),
                        collect
                            .steps
                            .iter()
                            .map(|step| match step {
                                CollectStep::Local(local) => {
                                    let mut qs: Vec<usize> =
                                        local.partition.all_elements().iter().copied().collect();
                                    qs.sort_unstable();
                                    ("emit".to_string(), qs)
                                }
                                CollectStep::Gate(gate) => {
                                    (gate.gate.name().to_string(), gate.qubits.clone())
                                }
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
    /// The direction is read off the source node rather than the edge, which is where it lives now.
    /// It is `"none"` for the nodes that carry no flow of their own — today only a `Reset`, which the
    /// lowering emits as an isolated node.
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
    fn to_dot(&self) -> String {
        use rustworkx_core::petgraph::visit::{EdgeRef, IntoEdgeReferences, IntoNodeReferences};
        use std::fmt::Write;

        let mut dot = String::new();
        writeln!(dot, "digraph VFG {{").unwrap();
        writeln!(dot, "    rankdir=TB;").unwrap();
        writeln!(dot, "    node [shape=box, style=filled, fontname=\"Helvetica\"];").unwrap();

        for (idx, node) in self.graph.node_references() {
            let (label, color) = node_label_color(&node.kind, &node.partition);
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
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

fn format_partition(partition: &Partition) -> String {
    let parts: Vec<&[usize]> = partition.iter().collect();
    if parts.iter().all(|p| p.len() == 1) {
        let flat: Vec<usize> = parts.iter().map(|p| p[0]).collect();
        format!("{:?}", flat)
    } else {
        format!("{:?}", parts)
    }
}

/// The `kind` string for an emission node.
///
/// The prefix names which annotation the emission stands in for, taken from the table entry's
/// discriminant. Callers (including the Python tests) select nodes by these prefixes.
fn emission_label(entry: &DistEntry) -> String {
    match entry {
        DistEntry::Distribution(distribution) => format!("emit:{distribution:?}"),
        DistEntry::Basis { ref_id, .. } => format!("change_basis:{ref_id}"),
        DistEntry::Noise { reference, .. } => format!("inject_noise:{reference}"),
    }
}

fn node_label_color(kind: &NodeKind, partition: &Partition) -> (String, &'static str) {
    let qubits = format_partition(partition);
    match kind {
        // Emissions are one node kind but keep distinct colours, since which annotation produced one
        // is the first thing you look for in a rendered graph.
        NodeKind::Emission(e) => {
            let (label, color) = match &e.entry {
                DistEntry::Distribution(distribution) => {
                    (format!("Emit({distribution:?})"), "#a8d8ea")
                }
                DistEntry::Basis { mode, .. } => (format!("ChangeBasis({mode:?})"), "#fff2cc"),
                DistEntry::Noise { reference, .. } => {
                    (format!("InjectNoise({reference})"), "#f8cecc")
                }
            };
            (
                format!("{} {} {}", label, e.direction.mark(), qubits),
                color,
            )
        }
        NodeKind::Collect(c) => {
            (format!("Collect({:?}) {}", c.synthesizer, qubits), "#f8c8dc")
        }
        NodeKind::Propagate(p) => (
            format!("{}{:?} {}", p.direction.mark(), p.gate, qubits),
            "#fffacd",
        ),
        NodeKind::Measure(m) => {
            (format!("Measure cl{:?} {}", m.clbit_indices, qubits), "#d5e8d4")
        }
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

    fn make_partition(parts: &[&[usize]]) -> Partition {
        Partition::with_parts(parts.iter().map(|p| p.to_vec().into_boxed_slice())).unwrap()
    }

    fn emission(entry: DistEntry, direction: Direction) -> Emission {
        Emission {
            entry,
            direction,
            virtual_type: VirtualType::Pauli,
        }
    }

    #[test]
    fn test_construct_emission_node() {
        let node = Node {
            partition: Partition::from_elements([0, 1]),
            kind: NodeKind::Emission(emission(
                DistEntry::Distribution(DistributionType::UniformPauli),
                Direction::Right,
            )),
        };
        assert_eq!(node.partition.len(), 2);
        assert!(matches!(node.kind, NodeKind::Emission(_)));
    }

    #[test]
    fn test_emission_label_names_its_source() {
        // The three annotation kinds share one node kind, so the label is the only thing that still
        // distinguishes them — and the Python tests select nodes by these prefixes.
        assert_eq!(
            emission_label(&DistEntry::Distribution(DistributionType::UniformPauli)),
            "emit:UniformPauli"
        );
        assert_eq!(
            emission_label(&DistEntry::Basis {
                mode: ChangeBasisMode::MeasurePauli,
                ref_id: "basis_changes.b0".to_string(),
            }),
            "change_basis:basis_changes.b0"
        );
        assert_eq!(
            emission_label(&DistEntry::Noise {
                reference: "n0".to_string(),
                modifier: None,
            }),
            "inject_noise:n0"
        );
    }

    #[test]
    fn test_construct_propagate_node() {
        let node = Node {
            partition: make_partition(&[&[0, 1], &[2, 3]]),
            kind: NodeKind::Propagate(Propagate {
                gate: StandardGate::CX,
                direction: Direction::Right,
            }),
        };
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
        let node = Node {
            partition: Partition::from_elements([0, 1, 2]),
            kind: NodeKind::Collect(Collect {
                synthesizer: SynthesizerType::RzSx,
                param_indices: vec![0, 1, 2, 3, 4, 5, 6, 7, 8],
                steps: Vec::new(),
            }),
        };
        if let NodeKind::Collect(ref c) = node.kind {
            assert_eq!(c.param_indices.len(), 9);
        } else {
            panic!("expected Collect");
        }
    }

    #[test]
    fn test_construct_measure_node() {
        let node = Node {
            partition: Partition::from_elements([0, 1]),
            kind: NodeKind::Measure(Measure {
                clbit_indices: vec![0, 1],
            }),
        };
        assert!(matches!(node.kind, NodeKind::Measure(_)));
    }

    #[test]
    fn test_construct_reset_node() {
        let node = Node {
            partition: Partition::from_elements([3]),
            kind: NodeKind::Reset,
        };
        assert!(matches!(node.kind, NodeKind::Reset));
    }

    #[test]
    fn test_basis_and_noise_are_emissions_too() {
        // Unifying the kinds means these are no longer three separate arms to match on.
        for entry in [
            DistEntry::Basis {
                mode: ChangeBasisMode::MeasurePauli,
                ref_id: "basis_changes.0".to_string(),
            },
            DistEntry::Noise {
                reference: "noise_model.0".to_string(),
                modifier: Some("modifier.0".to_string()),
            },
        ] {
            let kind = NodeKind::Emission(emission(entry, Direction::Left));
            assert!(matches!(kind, NodeKind::Emission(_)));
            assert!(kind.is_source());
        }
    }

    #[test]
    fn test_sources_and_sinks() {
        let emit = NodeKind::Emission(emission(
            DistEntry::Distribution(DistributionType::UniformPauli),
            Direction::Right,
        ));
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
        let emit = NodeKind::Emission(emission(
            DistEntry::Distribution(DistributionType::UniformPauli),
            Direction::Left,
        ));
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

        let emit_idx = vfg.graph.add_node(Node {
            partition: Partition::from_elements([0, 1]),
            kind: NodeKind::Emission(emission(
                DistEntry::Distribution(DistributionType::UniformPauli),
                Direction::Right,
            )),
        });

        let propagate_idx = vfg.graph.add_node(Node {
            partition: make_partition(&[&[0, 1]]),
            kind: NodeKind::Propagate(Propagate {
                gate: StandardGate::CX,
                direction: Direction::Right,
            }),
        });

        let collect_idx = vfg.graph.add_node(Node {
            partition: Partition::from_elements([0, 1]),
            kind: collect_kind(),
        });

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
