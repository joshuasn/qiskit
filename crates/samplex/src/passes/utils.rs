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

use std::collections::VecDeque;

use hashbrown::HashMap;
use rustworkx_core::petgraph::stable_graph::NodeIndex;
use rustworkx_core::petgraph::Direction as PetDirection;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{ControlFlow, OperationRef};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::{Clbit, Qubit};

use qiskit_circuit::operations::Param;

use crate::emission_circuit::{extract_collect, extract_emit, CollectSpec};
use crate::virtual_flow_graph::{Edge, Node};

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
            graph.neighbors_directed(idx, PetDirection::Incoming).count(),
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

// --- Temporary IR2 representation bridges -------------------------------------------------------
//
// IR2 is a `DAGCircuit` at the Python boundary, but the pass bodies are being migrated one at a
// time. Until a pass reads and writes the DAG directly, its wrapper converts in and out with these.
//
// TODO: delete both once every IR2 pass body is DAG-native. Nothing outside a `py_*` wrapper should
// call them — a pass that still needs one has not been migrated yet.

/// Convert an IR2 `CircuitData` to the `DAGCircuit` the boundary expects.
///
/// Block bodies convert recursively, so a nested box body arrives as a `DAGCircuit` block.
pub(super) fn to_dag(circuit: &CircuitData) -> PyResult<DAGCircuit> {
    Ok(DAGCircuit::from_circuit_data(
        circuit, false, None, None, None, None,
    )?)
}

/// Convert an IR2 `DAGCircuit` to the flat `CircuitData` an unmigrated pass body still wants.
///
/// The instruction order is `topological_op_nodes`, which need not match the order the DAG was
/// built in — see the ordering note in `SAMPLEX_IR_DESIGN.md`.
pub(super) fn to_circuit(dag: &DAGCircuit) -> PyResult<CircuitData> {
    Ok(CircuitData::from_dag_ref(dag)?)
}

// --- Emission circuit (IR2) helpers -------------------------------------------------------------
//
// Reading and rewriting an IR2 circuit is common to every IR2 pass, so these live here rather than
// being duplicated per pass.

/// Copy an instruction into `out`, optionally substituting a rebuilt body for its block.
pub(super) fn copy_through(
    src: &CircuitData,
    inst: &PackedInstruction,
    out: &mut CircuitData,
    replacement: Option<CircuitData>,
) -> PyResult<()> {
    let qargs: Vec<Qubit> = src.qargs_interner().get(inst.qubits).to_vec();
    let cargs: Vec<Clbit> = src.cargs_interner().get(inst.clbits).to_vec();

    let params = match replacement {
        Some(body) => Some(Parameters::Blocks(vec![out.add_block(body)])),
        None => {
            let blocks = inst.blocks_view();
            if blocks.is_empty() {
                params_of(inst)
            } else {
                Some(Parameters::Blocks(
                    blocks
                        .iter()
                        .map(|b| out.add_block(src.blocks()[*b].clone()))
                        .collect(),
                ))
            }
        }
    };
    out.push_packed_operation(inst.op.clone(), params, &qargs, &cargs)
        .into_py_result()
}
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
    annotations
        .iter()
        .find_map(|a| extract_collect(a.bind(py)))
}
pub(super) fn is_emission(py: Python, inst: &PackedInstruction) -> bool {
    match inst.op.view() {
        OperationRef::PyCustom(py_inst) => extract_emit(py_inst.ob.bind(py)).is_some(),
        _ => false,
    }
}
pub(super) fn qubit_indices(src: &CircuitData, inst: &PackedInstruction) -> Vec<usize> {
    src.qargs_interner()
        .get(inst.qubits)
        .iter()
        .map(|q| q.index())
        .collect()
}
/// The single body of a box instruction.
pub(super) fn block_body<'a>(src: &'a CircuitData, inst: &PackedInstruction) -> PyResult<Option<&'a CircuitData>> {
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

/// Create an empty `CircuitData` with the given dimensions.
pub(super) fn new_circuit_body(
    num_qubits: usize,
    num_clbits: usize,
    capacity: usize,
) -> PyResult<CircuitData> {
    CircuitData::with_capacity(
        num_qubits as u32,
        num_clbits as u32,
        capacity,
        Param::Float(0.0),
    )
    .into_py_result()
}
