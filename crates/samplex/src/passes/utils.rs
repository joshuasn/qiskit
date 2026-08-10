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

use crate::emission_circuit::{CollectSpec, extract_collect, extract_emit};
use crate::virtual_flow_graph::{Direction, Edge, Node};

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
pub(super) fn is_emission(py: Python, inst: &PackedInstruction) -> bool {
    match inst.op.view() {
        OperationRef::PyCustom(py_inst) => extract_emit(py_inst.ob.bind(py)).is_some(),
        _ => false,
    }
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

/// The [`EmitSpec`](crate::emission_circuit::EmitSpec) on this instruction, if it is an emission.
pub(super) fn emission_spec(
    py: Python,
    inst: &PackedInstruction,
) -> Option<crate::emission_circuit::EmitSpec> {
    match inst.op.view() {
        OperationRef::PyCustom(py_inst) => extract_emit(py_inst.ob.bind(py)),
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
