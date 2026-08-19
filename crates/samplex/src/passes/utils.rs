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

use hashbrown::HashMap;
use rustworkx_core::petgraph::Direction as PetDirection;
use rustworkx_core::petgraph::stable_graph::NodeIndex;
use rustworkx_core::petgraph::visit::EdgeRef;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use qiskit_circuit::bit::{ShareableClbit, ShareableQubit};
use qiskit_circuit::dag_circuit::{DAGCircuit, DAGCircuitBuilder, NodeType, Wire};
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{ControlFlow, OperationRef};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::{Block, Clbit, Qubit};

use crate::emission_circuit::{CollectSpec, EmitSpec, extract_collect};
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
/// The `Collect` annotation on this instruction, if it is a collector.
pub(super) fn collect_annotation(py: Python, inst: &PackedInstruction) -> Option<CollectSpec> {
    let OperationRef::ControlFlow(cf) = inst.op.view() else {
        return None;
    };
    let ControlFlow::Box { annotations, .. } = &cf.control_flow else {
        return None;
    };
    annotations.iter().find_map(|a| extract_collect(a.bind(py)))
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
