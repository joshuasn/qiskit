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

//! Helpers shared by more than one pass: per-wire adjacency, annotation readback, body
//! construction.

use std::collections::VecDeque;

use std::sync::Arc;

use hashbrown::HashMap;
use rustworkx_core::petgraph::Direction as PetDirection;
use rustworkx_core::petgraph::stable_graph::NodeIndex;
use rustworkx_core::petgraph::visit::EdgeRef;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use qiskit_circuit::annotation::Annotation;
use qiskit_circuit::bit::{ShareableClbit, ShareableQubit};
use qiskit_circuit::dag_circuit::{DAGCircuit, DAGCircuitBuilder, NodeType, Wire};
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{BoxDuration, ControlFlow, ControlFlowInstruction, OperationRef};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::{Block, Clbit, Qubit};

use crate::emission_circuit::{CollectSpec, EmitSpec};
use crate::sampling_graph::{Direction, Edge, Node};

/// Extension trait that converts any `Result<T, E: Display>` into `PyResult<T>` via `PyValueError`.
pub(super) trait IntoPyResult<T> {
    fn into_py_result(self) -> PyResult<T>;
}

impl<T, E: std::fmt::Display> IntoPyResult<T> for Result<T, E> {
    fn into_py_result(self) -> PyResult<T> {
        self.map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// Compute topological generations using Kahn's algorithm.
pub(super) fn topological_generations(
    graph: &rustworkx_core::petgraph::stable_graph::StableDiGraph<Node, Edge>,
) -> Vec<Vec<NodeIndex>> {
    let mut in_degree: HashMap<NodeIndex, usize> = HashMap::new();
    for idx in graph.node_indices() {
        in_degree.insert(
            idx,
            graph
                .neighbors_directed(idx, PetDirection::Incoming)
                .count(),
        );
    }

    let mut generations = Vec::new();
    let mut queue: VecDeque<NodeIndex> = in_degree
        .iter()
        .filter(|(_, d)| **d == 0)
        .map(|(idx, _)| *idx)
        .collect();

    while !queue.is_empty() {
        let current_gen: Vec<NodeIndex> = queue.drain(..).collect();
        for &node in &current_gen {
            for succ in graph.neighbors_directed(node, PetDirection::Outgoing) {
                if let Some(d) = in_degree.get_mut(&succ) {
                    *d -= 1;
                    if *d == 0 {
                        queue.push_back(succ);
                    }
                }
            }
        }
        generations.push(current_gen);
    }

    generations
}

// --- Emission circuit (IR2) helpers -------------------------------------------------------------
//
// Reading and rewriting an IR2 circuit is common to every IR2 pass, so these live here rather than
// being duplicated per pass.

pub(super) fn params_of(inst: &PackedInstruction) -> Option<Parameters<qiskit_circuit::Block>> {
    (!inst.params_view().is_empty())
        .then(|| Parameters::Params(inst.params_view().iter().cloned().collect()))
}
/// The [`CollectSpec`] on this instruction, if it is a collector.
pub(super) fn collect_annotation(inst: &PackedInstruction) -> Option<CollectSpec> {
    box_collect_spec(inst).cloned()
}

/// Whether this instruction is a collect box.
///
/// The borrowing half of [`collect_annotation`], for the many walk sites that only need to know
/// whether to descend, not what the collector holds.
pub(super) fn is_collector(inst: &PackedInstruction) -> bool {
    box_collect_spec(inst).is_some()
}

/// Whether an instruction is a `box`.
pub(super) fn is_box(inst: &PackedInstruction) -> bool {
    matches!(inst.op.view(), OperationRef::ControlFlow(cf) if matches!(cf.control_flow, ControlFlow::Box { .. }))
}

/// Borrow the collect spec off a box's annotations.
///
/// The crate's only read of a box's annotations for its own vocabulary, and the reason none of the
/// IR2 walks need a `Python` token: a native annotation is a Rust value, so asking what a box
/// declares is a `TypeId` comparison rather than an attribute lookup.
fn box_collect_spec(inst: &PackedInstruction) -> Option<&CollectSpec> {
    let OperationRef::ControlFlow(cf) = inst.op.view() else {
        return None;
    };
    let ControlFlow::Box { annotations, .. } = &cf.control_flow else {
        return None;
    };
    annotations
        .iter()
        .find_map(|a| a.as_ref().downcast_ref::<CollectSpec>())
}

/// A collect box of the given width, carrying this spec.
///
/// Any other annotation and any duration on the box being replaced are dropped, which is sound only
/// because `build::write_collect` is the sole minter of collect boxes and gives them exactly one
/// annotation and no duration. A collect box that ever needs to carry a second annotation has to
/// revisit this.
pub(super) fn collect_op(
    spec: CollectSpec,
    num_qubits: usize,
    num_clbits: usize,
) -> PackedOperation {
    PackedOperation::from_control_flow(Box::new(ControlFlowInstruction {
        control_flow: ControlFlow::Box {
            duration: None,
            annotations: vec![Arc::new(spec)],
        },
        num_qubits: num_qubits as u32,
        num_clbits: num_clbits as u32,
    }))
}
pub(super) fn is_emission(inst: &PackedInstruction) -> bool {
    matches!(
        inst.op.view(),
        OperationRef::CustomOperation(op) if op.downcast_ref::<EmitSpec>().is_some()
    )
}
/// The single body of a box instruction.
pub(super) fn block_body<'a>(
    src: &'a DAGCircuit,
    inst: &PackedInstruction,
) -> PyResult<Option<&'a DAGCircuit>> {
    match inst.blocks_view() {
        [] => Ok(None),
        [block] => Ok(Some(&src.blocks()[*block])),
        _ => Err(PyValueError::new_err(
            "a box instruction should have exactly one body",
        )),
    }
}

/// The [`EmitSpec`] on this instruction, if it is an emission.
pub(super) fn emission_spec(inst: &PackedInstruction) -> Option<EmitSpec> {
    match inst.op.view() {
        OperationRef::CustomOperation(op) => op.downcast_ref::<EmitSpec>().cloned(),
        _ => None,
    }
}

/// The next operation node along one wire, or `None` at the end of it.
///
/// Reaching the wire's output node counts as the end.
pub(super) fn next_on_wire(
    dag: &DAGCircuit,
    from: NodeIndex,
    qubit: Qubit,
    direction: Direction,
) -> Option<NodeIndex> {
    // Per-wire, unlike `quantum_successors`, which pools every wire of a node together. "What does
    // this qubit see next" is the whole adjacency notion the IR2 passes need.
    let (search, wire) = match direction {
        Direction::Right => (PetDirection::Outgoing, Wire::Qubit(qubit)),
        Direction::Left => (PetDirection::Incoming, Wire::Qubit(qubit)),
    };
    let next = dag
        .dag()
        .edges_directed(from, search)
        .find(|edge| *edge.weight() == wire)
        .map(|edge| match direction {
            Direction::Right => edge.target(),
            Direction::Left => edge.source(),
        })?;
    matches!(dag.dag()[next], NodeType::Operation(_)).then_some(next)
}

/// The address of one instruction in a nested circuit: the box nodes descended through, outermost
/// first, then the node itself within that innermost scope.
///
/// A bare `NodeIndex` only identifies a node once you already know which `DAGCircuit` it belongs to,
/// which a walk that crosses box boundaries no longer does.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(super) struct Site {
    pub scope: Vec<NodeIndex>,
    pub node: NodeIndex,
}

/// A per-wire walk position that can descend into a box body and climb back out.
///
/// One wire, one cursor: a walk over several wires advances each independently, so a wire blocked
/// inside a box stops on its own while another goes on descending. A cursor climbs back out only of
/// boxes it descended into — it never ascends above the scope it started in, which is what keeps a
/// collector inside a box from reaching content outside it.
#[derive(Clone, Debug)]
pub(super) struct WireCursor {
    /// The scope the walk started in. Never climbed out of, which is the whole of "descends, never
    /// ascends".
    base: Vec<NodeIndex>,
    /// The boxes descended through since the start, each with this wire's index in the scope
    /// *containing* that box.
    descent: Vec<(NodeIndex, Qubit)>,
    /// The node the cursor sits on, in the scope its path names. A wire's boundary node while
    /// mid-descent, which is why this is only ever used as a starting point for the next step.
    node: NodeIndex,
    /// This wire's index in that same scope.
    qubit: Qubit,
}

impl WireCursor {
    /// A cursor sitting on `node`, walking along `qubit`, both in the scope `base` names.
    pub(super) fn new(base: Vec<NodeIndex>, node: NodeIndex, qubit: Qubit) -> Self {
        WireCursor {
            base,
            descent: Vec::new(),
            node,
            qubit,
        }
    }

    /// Where the cursor is now.
    pub(super) fn site(&self) -> Site {
        Site {
            scope: self.path(),
            node: self.node,
        }
    }

    /// The scope the cursor is currently in.
    fn path(&self) -> Vec<NodeIndex> {
        let mut path = self.base.clone();
        path.extend(self.descent.iter().map(|(node, _)| *node));
        path
    }

    /// Advance one operation along this wire, descending into any box `descend` accepts and climbing
    /// back out at the end of a body.
    ///
    /// Returns the site reached, or `None` once the wire has run out in the scope the walk started
    /// in. A box the cursor descends into and finds nothing on this wire is passed straight through:
    /// there is nothing there to reorder against, so it is not a barrier.
    pub(super) fn advance(
        &mut self,
        root: &DAGCircuit,
        direction: Direction,
        descend: &dyn Fn(&PackedInstruction) -> bool,
    ) -> PyResult<Option<Site>> {
        loop {
            let dag = scope_dag(root, &self.path())?;
            match next_on_wire(dag, self.node, self.qubit, direction) {
                Some(next) => {
                    let inst = dag.dag()[next].unwrap_operation();
                    if !descend(inst) {
                        self.node = next;
                        return Ok(Some(self.site()));
                    }
                    let body = block_body(dag, inst)?.ok_or_else(|| {
                        PyValueError::new_err("cannot descend into a box with no body")
                    })?;
                    // The walk only offers nodes adjacent along this wire, so a box reached this way
                    // covers it.
                    let local = dag
                        .qargs_interner()
                        .get(inst.qubits)
                        .iter()
                        .position(|q| *q == self.qubit)
                        .map(|index| Qubit(index as u32))
                        .expect("a box reached along a wire covers that wire");
                    // Enter at the far end of the body from the direction of travel, so the first
                    // step inside lands on the operation nearest the boundary just crossed.
                    let boundary = match direction {
                        Direction::Right => body.qubit_io_map()[local.index()][0],
                        Direction::Left => body.qubit_io_map()[local.index()][1],
                    };
                    self.descent.push((next, self.qubit));
                    self.node = boundary;
                    self.qubit = local;
                }
                // End of the wire in this scope: climb out of the box it was inside, or stop.
                None => match self.descent.pop() {
                    Some((box_node, outer)) => {
                        self.node = box_node;
                        self.qubit = outer;
                    }
                    None => return Ok(None),
                },
            }
        }
    }
}

/// The DAG of the scope a path names.
pub(super) fn scope_dag<'a>(root: &'a DAGCircuit, scope: &[NodeIndex]) -> PyResult<&'a DAGCircuit> {
    let mut dag = root;
    for node in scope {
        let inst = dag.dag()[*node].unwrap_operation();
        dag = block_body(dag, inst)?
            .ok_or_else(|| PyValueError::new_err("a scope on the path has no body"))?;
    }
    Ok(dag)
}

/// The DAG of the scope a path names, for writing.
pub(super) fn scope_dag_mut<'a>(
    root: &'a mut DAGCircuit,
    scope: &[NodeIndex],
) -> PyResult<&'a mut DAGCircuit> {
    let mut dag = root;
    for node in scope {
        let block = match dag.dag()[*node].unwrap_operation().blocks_view() {
            [block] => *block,
            _ => {
                return Err(PyValueError::new_err(
                    "a scope on the path should have exactly one body",
                ));
            }
        };
        dag = dag.view_block_mut(block);
    }
    Ok(dag)
}

/// The instruction a site names.
pub(super) fn site_instruction<'a>(
    root: &'a DAGCircuit,
    site: &Site,
) -> PyResult<&'a PackedInstruction> {
    Ok(scope_dag(root, &site.scope)?.dag()[site.node].unwrap_operation())
}

/// Map wires from the scope `path` names up into the scope the path starts in.
///
/// Absorbed content keeps its position in the collector's frame, so content taken out of a nested
/// body has to be lifted through every box between.
pub(super) fn lift_wires(
    root: &DAGCircuit,
    path: &[NodeIndex],
    wires: &[Qubit],
) -> PyResult<Vec<Qubit>> {
    // Each box's qargs, in the frame of the scope containing it.
    let mut frames: Vec<Vec<Qubit>> = Vec::with_capacity(path.len());
    let mut dag = root;
    for node in path {
        let inst = dag.dag()[*node].unwrap_operation();
        frames.push(dag.qargs_interner().get(inst.qubits).to_vec());
        dag = block_body(dag, inst)?
            .ok_or_else(|| PyValueError::new_err("a scope on the path has no body"))?;
    }
    let mut lifted = wires.to_vec();
    for frame in frames.iter().rev() {
        for wire in &mut lifted {
            *wire = *frame.get(wire.index()).ok_or_else(|| {
                PyValueError::new_err("absorbed content sits on a wire outside its box")
            })?;
        }
    }
    Ok(lifted)
}

/// Append an operation to the back of a DAG under construction.
pub(super) fn append(
    out: &mut DAGCircuitBuilder,
    op: PackedOperation,
    params: Option<Parameters<Block>>,
    qargs: &[Qubit],
    cargs: &[Clbit],
) -> PyResult<()> {
    // Exists to keep `apply_operation_back`'s `cache_pygates` argument in one place: everything
    // samplex appends is built from a `PackedOperation`, never a live Python object, so there is
    // never a cached gate to pass. `CircuitData::push_packed_operation` is the same convenience on
    // the flat side.
    out.apply_operation_back(
        op,
        qargs,
        cargs,
        params,
        None,
        #[cfg(feature = "cache_pygates")]
        None,
    )
    .into_py_result()?;
    Ok(())
}

/// Append a `box` with this body and these annotations.
///
/// The one place a box is written, so the one place its annotations become `Arc<dyn Annotation>`. Its
/// counterpart is [`box_collect_spec`]: together they are the whole of how samplex puts a declaration
/// on a box and gets it back, and neither needs a `Python` token to do it.
pub(super) fn write_box(
    out: &mut DAGCircuitBuilder,
    body: DAGCircuit,
    annotations: Vec<Arc<dyn Annotation>>,
    duration: Option<BoxDuration>,
    qargs: &[Qubit],
    cargs: &[Clbit],
) -> PyResult<()> {
    let op = PackedOperation::from_control_flow(Box::new(ControlFlowInstruction {
        control_flow: ControlFlow::Box {
            duration,
            annotations,
        },
        num_qubits: qargs.len() as u32,
        num_clbits: cargs.len() as u32,
    }));
    let block = out.add_block(body);
    append(out, op, Some(Parameters::Blocks(vec![block])), qargs, cargs)
}

/// Create an empty `DAGCircuit` body with the given dimensions and anonymous wires.
pub(super) fn new_dag_body(
    num_qubits: usize,
    num_clbits: usize,
    capacity: usize,
) -> PyResult<DAGCircuit> {
    // Anonymous because a box body's qubits are positional, addressed only through the box's qargs,
    // so there is nothing outside for them to be identified with. `with_capacity` reserves space
    // but registers no wires, hence the explicit adds.
    let mut body =
        DAGCircuit::with_capacity(num_qubits, num_clbits, None, Some(capacity), None, None);
    for _ in 0..num_qubits {
        body.add_qubit_unchecked(ShareableQubit::new_anonymous())
            .into_py_result()?;
    }
    for _ in 0..num_clbits {
        body.add_clbit_unchecked(ShareableClbit::new_anonymous())
            .into_py_result()?;
    }
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotated_circuit::{DistributionType, Dressing, SynthesizerType, TwirlSpec};
    use crate::emission_circuit::CollectPart;
    use crate::partition::Partition;
    use qiskit_circuit::annotation::Annotation;
    use qiskit_circuit::operations::StandardGate;

    /// An annotation from outside samplex's vocabulary.
    ///
    /// It never materializes into Python here, which is the point: these readers only compare
    /// `TypeId`s, so a foreign annotation costs them nothing and needs no interpreter.
    #[derive(Debug, Clone, PartialEq)]
    struct Foreign;

    impl Annotation for Foreign {
        fn namespace(&self) -> &str {
            "someone.else"
        }

        fn create_py_annotation(&self, _py: Python) -> PyResult<Py<PyAny>> {
            unimplemented!("these tests never cross into Python")
        }
    }

    /// A one-qubit `box` carrying exactly these annotations.
    ///
    /// The interner keys are the default empty-slice ones and the block list is empty: these readers
    /// only look at `inst.op`, so anything else would be fixture that no assertion depends on.
    fn box_instruction(annotations: Vec<Arc<dyn Annotation>>) -> PackedInstruction {
        PackedInstruction::from_control_flow(
            ControlFlowInstruction {
                control_flow: ControlFlow::Box {
                    duration: None,
                    annotations,
                },
                num_qubits: 1,
                num_clbits: 0,
            },
            Vec::new(),
            Default::default(),
            Default::default(),
            None,
        )
    }

    fn collect_spec() -> CollectSpec {
        CollectSpec {
            partition: Partition::singletons(1),
            parts: vec![CollectPart {
                synthesizer: SynthesizerType::RzSx,
            }],
        }
    }

    fn twirl_spec() -> TwirlSpec {
        TwirlSpec {
            distribution: DistributionType::UniformPauli,
            dressing: Dressing::Left,
            decomposition: SynthesizerType::RzSx,
        }
    }

    #[test]
    fn test_collect_annotation_reads_spec() {
        let spec = collect_spec();
        let inst = box_instruction(vec![Arc::new(spec.clone())]);
        assert_eq!(collect_annotation(&inst), Some(spec));
        assert!(is_collector(&inst));
        assert!(is_box(&inst));
    }

    #[test]
    fn test_is_collector_rejects_content_box() {
        // A content box is still a box, and it still carries a samplex annotation. What it is not is a
        // collector — the distinction every IR2 walk descends on.
        let inst = box_instruction(vec![Arc::new(twirl_spec())]);
        assert!(is_box(&inst));
        assert!(!is_collector(&inst));
        assert_eq!(collect_annotation(&inst), None);
    }

    #[test]
    fn test_collect_annotation_ignores_foreign_annotation() {
        // A stranger's annotation riding along on a collector must not hide the collector, and on its
        // own must not fabricate one.
        let spec = collect_spec();
        let shared = box_instruction(vec![Arc::new(Foreign), Arc::new(spec.clone())]);
        assert_eq!(collect_annotation(&shared), Some(spec));

        let foreign_only = box_instruction(vec![Arc::new(Foreign)]);
        assert!(is_box(&foreign_only));
        assert!(!is_collector(&foreign_only));
    }

    #[test]
    fn test_is_box_rejects_a_gate() {
        let gate = PackedInstruction::from_standard_gate(StandardGate::H, None, Default::default());
        assert!(!is_box(&gate));
        assert!(!is_collector(&gate));
        assert_eq!(collect_annotation(&gate), None);
    }

    #[test]
    fn test_collect_op_writes_one_annotation_and_no_duration() {
        // Pins what `collect_op`'s doc comment claims, and what both callers rely on when they replace
        // a collector wholesale: exactly one annotation, no duration, the width it was asked for.
        let spec = collect_spec();
        let op = collect_op(spec.clone(), 2, 1);
        let OperationRef::ControlFlow(cf) = op.view() else {
            panic!("collect_op must produce a control-flow operation");
        };
        assert_eq!(cf.num_qubits, 2);
        assert_eq!(cf.num_clbits, 1);
        let ControlFlow::Box {
            duration,
            annotations,
        } = &cf.control_flow
        else {
            panic!("collect_op must produce a box");
        };
        assert!(duration.is_none());
        assert_eq!(annotations.len(), 1);
        assert_eq!(annotations[0].downcast_ref::<CollectSpec>(), Some(&spec));
    }
}
