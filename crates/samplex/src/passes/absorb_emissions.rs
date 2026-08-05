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

//! Absorb local emissions into their collectors.
//!
//! After the build pass, every emission is a standalone `Emit` instruction and every collector
//! references it as `CollectItem::Incoming(id)`. Most emissions are *local*: adjacent to their
//! collector, never propagating through hard content. This pass identifies them and absorbs them
//! into `CollectItem::Emission(LocalEmission)`, removing the standalone instruction.
//!
//! Only the "far twirl half" — which walks through the hard box toward the far collector — remains
//! as a standalone `Emit` with a `CollectItem::Incoming(id)` reference.
//!
//! **Order-independent with `merge_collectors`.** The pass produces the same semantic result
//! whether called before or after merging. The preferred position is before merge (simpler
//! single-box structure), but correctness does not depend on it.

use hashbrown::HashSet;

use pyo3::prelude::*;
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};

use super::utils::{collect_annotation, copy_through, emission_spec, is_emission, IntoPyResult};
use crate::emission_circuit::{CollectItem, CollectSpec, EmitSpec, LocalEmission};
use crate::virtual_flow_graph::Direction;

#[pyfunction]
#[pyo3(name = "absorb_emissions")]
pub fn py_absorb_emissions(py: Python, circuit: &PyCircuitData) -> PyResult<PyCircuitData> {
    Ok(PyCircuitData {
        inner: absorb_emissions(py, &circuit.inner)?,
    })
}

/// Absorb local emissions throughout an emission circuit.
pub fn absorb_emissions(py: Python, src: &CircuitData) -> PyResult<CircuitData> {
    absorb_scope(py, src)
}

/// Which side of the content a collector sits on, and therefore which direction a local emit
/// must flow to be absorbed by it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    Left,
    Right,
}

/// A tagged instruction from the source circuit.
enum Entry {
    Collector {
        index: usize,
        spec: CollectSpec,
    },
    Emit {
        index: usize,
        spec: EmitSpec,
    },
    Other {
        index: usize,
    },
}

fn absorb_scope(py: Python, src: &CircuitData) -> PyResult<CircuitData> {
    let entries: Vec<Entry> = src
        .data()
        .iter()
        .enumerate()
        .map(|(i, inst)| {
            if let Some(spec) = collect_annotation(py, inst) {
                Entry::Collector { index: i, spec }
            } else if let Some(spec) = emission_spec(py, inst) {
                Entry::Emit { index: i, spec }
            } else {
                Entry::Other { index: i }
            }
        })
        .collect();

    // For each collector, determine which adjacent emits are local (absorbable).
    // An emit is local if:
    //   1. It is adjacent to the collector (only other emits between them, no Other instructions)
    //   2. Its direction points toward the collector
    //   3. Its ID appears in the collector's items
    //
    // We collect the set of emit IDs to absorb, keyed by which collector absorbs them.
    let absorbed_emit_ids = find_local_emissions(&entries);

    // Rebuild the circuit, converting collectors and skipping absorbed emits.
    let mut out = CircuitData::with_capacity(
        src.num_qubits() as u32,
        src.num_clbits() as u32,
        src.len(),
        qiskit_circuit::operations::Param::Float(0.0),
    )
    .into_py_result()?;

    for entry in &entries {
        match entry {
            Entry::Collector { index, spec } => {
                let inst = &src.data()[*index];
                let new_spec = rewrite_collector_spec(spec, &absorbed_emit_ids, src, py);
                write_collector_with_spec(py, src, inst, &new_spec, &mut out)?;
            }
            Entry::Emit { index, spec } => {
                if absorbed_emit_ids.contains(&spec.id) {
                    continue;
                }
                let inst = &src.data()[*index];
                copy_through(src, inst, &mut out, None)?;
            }
            Entry::Other { index } => {
                let inst = &src.data()[*index];
                // Recurse into box bodies
                if has_blocks(inst) {
                    let new_body = recurse_into_blocks(py, src, inst)?;
                    copy_through(src, inst, &mut out, new_body)?;
                } else {
                    copy_through(src, inst, &mut out, None)?;
                }
            }
        }
    }

    Ok(out)
}

/// Identify which emit IDs are local (absorbable) across the entire instruction sequence.
fn find_local_emissions(entries: &[Entry]) -> HashSet<u32> {
    let mut absorbed = HashSet::new();

    for (i, entry) in entries.iter().enumerate() {
        let Entry::Collector { spec, .. } = entry else {
            continue;
        };

        let incoming_ids: HashSet<u32> = spec
            .items
            .iter()
            .filter_map(|item| match item {
                CollectItem::Incoming(id) => Some(*id),
                _ => None,
            })
            .collect();

        // Scan leftward: emits immediately before this collector (toward lower indices)
        scan_adjacent(entries, i, Side::Left, &incoming_ids, &mut absorbed);

        // Scan rightward: emits immediately after this collector (toward higher indices)
        scan_adjacent(entries, i, Side::Right, &incoming_ids, &mut absorbed);
    }

    absorbed
}

/// Scan from a collector in one direction, absorbing emits whose direction points toward it.
fn scan_adjacent(
    entries: &[Entry],
    collector_pos: usize,
    scan_side: Side,
    incoming_ids: &HashSet<u32>,
    absorbed: &mut HashSet<u32>,
) {
    let direction_toward_collector = match scan_side {
        Side::Left => Direction::Right,
        Side::Right => Direction::Left,
    };

    let range: Box<dyn Iterator<Item = usize>> = match scan_side {
        Side::Left => Box::new((0..collector_pos).rev()),
        Side::Right => Box::new((collector_pos + 1)..entries.len()),
    };

    for j in range {
        match &entries[j] {
            Entry::Emit { spec, .. } => {
                if spec.direction == direction_toward_collector && incoming_ids.contains(&spec.id) {
                    absorbed.insert(spec.id);
                }
            }
            _ => break,
        }
    }
}

/// Build a new `CollectSpec` with local emissions absorbed.
fn rewrite_collector_spec(
    spec: &CollectSpec,
    absorbed_ids: &HashSet<u32>,
    src: &CircuitData,
    py: Python,
) -> CollectSpec {
    let items = spec
        .items
        .iter()
        .map(|item| match item {
            CollectItem::Incoming(id) if absorbed_ids.contains(id) => {
                let emit_spec = find_emit_spec_by_id(src, py, *id)
                    .expect("absorbed emit ID must exist in the circuit");
                CollectItem::Emission(LocalEmission {
                    source: emit_spec.source,
                    dist: emit_spec.dist,
                    direction: emit_spec.direction,
                    virtual_type: emit_spec.virtual_type,
                    partition: emit_spec.partition,
                })
            }
            other => other.clone(),
        })
        .collect();

    CollectSpec {
        synthesizer: spec.synthesizer,
        items,
    }
}

/// Find an `EmitSpec` in the source circuit by its ID.
fn find_emit_spec_by_id(src: &CircuitData, py: Python, id: u32) -> Option<EmitSpec> {
    src.data().iter().find_map(|inst| {
        let spec = emission_spec(py, inst)?;
        (spec.id == id).then_some(spec)
    })
}

/// Write a collector instruction to the output with a new spec.
fn write_collector_with_spec(
    py: Python,
    src: &CircuitData,
    inst: &qiskit_circuit::packed_instruction::PackedInstruction,
    spec: &CollectSpec,
    out: &mut CircuitData,
) -> PyResult<()> {
    use qiskit_circuit::annotation::PyAnnotation;
    use qiskit_circuit::instruction::Parameters;
    use qiskit_circuit::operations::{ControlFlow, ControlFlowInstruction};
    use qiskit_circuit::packed_instruction::PackedOperation;
    use qiskit_circuit::{Clbit, Qubit};

    use crate::emission_circuit::Collect;

    let qargs: Vec<Qubit> = src.qargs_interner().get(inst.qubits).to_vec();
    let cargs: Vec<Clbit> = src.cargs_interner().get(inst.clbits).to_vec();

    // Get the body from the source instruction
    let body = match inst.blocks_view() {
        [block] => src.blocks()[*block].clone(),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "collector must have exactly one body",
            ))
        }
    };

    let annotation = Py::new(py, (Collect::new_from_spec(spec.clone()), PyAnnotation))?;
    let op = PackedOperation::from_control_flow(Box::new(ControlFlowInstruction {
        control_flow: ControlFlow::Box {
            duration: None,
            annotations: vec![annotation.into_any()],
        },
        num_qubits: qargs.len() as u32,
        num_clbits: cargs.len() as u32,
    }));
    let block_index = out.add_block(body);

    out.push_packed_operation(op, Some(Parameters::Blocks(vec![block_index])), &qargs, &cargs)
        .into_py_result()
}

fn has_blocks(inst: &qiskit_circuit::packed_instruction::PackedInstruction) -> bool {
    !inst.blocks_view().is_empty()
}

/// If the instruction has block bodies, recurse into them and return the first body (for a box).
fn recurse_into_blocks(
    py: Python,
    src: &CircuitData,
    inst: &qiskit_circuit::packed_instruction::PackedInstruction,
) -> PyResult<Option<CircuitData>> {
    let blocks = inst.blocks_view();
    if blocks.len() == 1 {
        let body = &src.blocks()[blocks[0]];
        // Only recurse if the body contains emissions or collectors
        let has_ir2 = body.data().iter().any(|i| {
            collect_annotation(py, i).is_some() || is_emission(py, i)
        });
        if has_ir2 {
            return Ok(Some(absorb_scope(py, body)?));
        }
    }
    Ok(None)
}
