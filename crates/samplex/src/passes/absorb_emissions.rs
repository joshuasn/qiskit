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

//! Absorb emissions into compatible collectors.
//!
//! After the build pass, every emission is a standalone `Emit` instruction and every collector
//! carries only `CollectItem::Gates` entries. This pass scans from each emission in its travel
//! direction, crossing box boundaries recursively, and absorbs it into the first collector whose
//! synthesizer accepts the emission's virtual type.
//!
//! An emission that encounters an incompatible collector before finding a compatible one is left
//! standalone for the future `walk_emissions` pass to handle.
//!
//! **Scope-agnostic.** Emissions cross box boundaries freely — an outer emission can be absorbed
//! by an inner collector, and an inner emission can escape its box to be absorbed by an outer one.
//!
//! **Local vs propagating.** An emission directly adjacent to its target collector (no gates
//! between) is absorbed locally — it becomes a `CollectItem::Emission(LocalEmission)` on the
//! collector and is removed from the spine. An emission separated from its collector by gates
//! (typically the far twirl half separated by hard box content) stays on the spine as a standalone
//! instruction and is wired via `CollectItem::Incoming(id)`, so the lower pass can build the
//! propagation graph through the intervening gates.

use hashbrown::{HashMap, HashSet};

use pyo3::prelude::*;
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::operations::{ControlFlow, OperationRef};
use qiskit_circuit::packed_instruction::PackedInstruction;

use super::utils::{collect_annotation, copy_through, emission_spec, is_emission, IntoPyResult};
use crate::emission_circuit::{CollectItem, CollectSpec, EmitSource, EmitSpec, LocalEmission};
use crate::virtual_flow_graph::Direction;

#[pyfunction]
#[pyo3(name = "absorb_emissions")]
pub fn py_absorb_emissions(py: Python, circuit: &PyCircuitData) -> PyResult<PyCircuitData> {
    Ok(PyCircuitData {
        inner: absorb_emissions(py, &circuit.inner)?,
    })
}

/// Absorb emissions into compatible collectors throughout an emission circuit.
pub fn absorb_emissions(py: Python, src: &CircuitData) -> PyResult<CircuitData> {
    absorb_scope(py, src)
}

/// A location that uniquely identifies an instruction within a possibly-nested circuit.
/// Each element is an index into the instruction list at that nesting level.
type Path = Vec<usize>;

/// Result of scanning from an emission.
enum ScanResult {
    /// The emission is directly adjacent to a compatible collector (no gates between).
    /// Remove from spine and add as LocalEmission.
    AbsorbLocal { collector_path: Path },
    /// The emission reaches a compatible collector but has gates between them.
    /// Keep on spine and add as Incoming(id) for graph wiring.
    AbsorbPropagating { collector_path: Path },
    /// No compatible collector found; leave the emission standalone.
    Standalone,
}

/// Scan from position `start` in direction `dir` within `src`, crossing box boundaries.
/// Returns the path to the first compatible collector (local or propagating), or Standalone.
fn scan_for_collector(
    py: Python,
    src: &CircuitData,
    start: usize,
    dir: Direction,
    emit_spec: &EmitSpec,
    path_prefix: &[usize],
) -> ScanResult {
    let range: Box<dyn Iterator<Item = usize>> = match dir {
        Direction::Right => Box::new((start + 1)..src.len()),
        Direction::Left => Box::new((0..start).rev()),
    };

    let mut has_gates = false;

    for i in range {
        let inst = &src.data()[i];

        if is_emission(py, inst) {
            continue;
        }

        if let Some(spec) = collect_annotation(py, inst) {
            if spec.accepts(emit_spec.virtual_type()) {
                let mut path = path_prefix.to_vec();
                path.push(i);
                if has_gates {
                    return ScanResult::AbsorbPropagating { collector_path: path };
                } else {
                    return ScanResult::AbsorbLocal { collector_path: path };
                }
            }
            // Incompatible collector — stop.
            return ScanResult::Standalone;
        }

        // Check if this is a box we can descend into.
        if let Some(body) = box_body(src, inst) {
            if let Some(result) =
                descend_into_box(py, body, dir, emit_spec, path_prefix, i, has_gates)
            {
                return result;
            }
            // No collector found at near edge of this box — it's transparent content.
            // The emission propagates through its gates.
            has_gates = true;
            continue;
        }

        // Bare gate — the emission propagates through it.
        has_gates = true;
    }

    ScanResult::Standalone
}

/// Look inside a box body from the near edge (determined by scan direction) for an absorbable
/// collector. Returns Some(result) if the scan should terminate (found a collector or hit an
/// incompatible one), None if the box contains no collector at its near edge (transparent content).
fn descend_into_box(
    py: Python,
    body: &CircuitData,
    dir: Direction,
    emit_spec: &EmitSpec,
    parent_prefix: &[usize],
    box_index: usize,
    parent_has_gates: bool,
) -> Option<ScanResult> {
    let mut child_prefix = parent_prefix.to_vec();
    child_prefix.push(box_index);

    // Scan from the near edge of the box body.
    let range: Box<dyn Iterator<Item = usize>> = match dir {
        Direction::Right => Box::new(0..body.len()),
        Direction::Left => Box::new((0..body.len()).rev()),
    };

    for i in range {
        let inst = &body.data()[i];

        if is_emission(py, inst) {
            continue;
        }

        if let Some(spec) = collect_annotation(py, inst) {
            if spec.accepts(emit_spec.virtual_type()) {
                let mut path = child_prefix;
                path.push(i);
                if parent_has_gates {
                    return Some(ScanResult::AbsorbPropagating { collector_path: path });
                } else {
                    return Some(ScanResult::AbsorbLocal { collector_path: path });
                }
            }
            // Incompatible collector inside the box — stop.
            return Some(ScanResult::Standalone);
        }

        // Nested box — descend further.
        if let Some(inner_body) = box_body(body, inst) {
            if let Some(result) =
                descend_into_box(py, inner_body, dir, emit_spec, &child_prefix, i, parent_has_gates)
            {
                return Some(result);
            }
            // Inner box had no collector at near edge — it's transparent content, stop descent.
            return None;
        }

        // Gate at near edge — no collector here, this box is just content.
        return None;
    }

    // Empty box or box with only emissions — not a blocker but nothing to absorb.
    None
}

/// Extract the body of a box instruction, if it is one.
fn box_body<'a>(src: &'a CircuitData, inst: &PackedInstruction) -> Option<&'a CircuitData> {
    let OperationRef::ControlFlow(cf) = inst.op.view() else {
        return None;
    };
    let ControlFlow::Box { .. } = &cf.control_flow else {
        return None;
    };
    match inst.blocks_view() {
        [block] => Some(&src.blocks()[*block]),
        _ => None,
    }
}

/// Process one scope: find all absorptions, then rebuild the circuit.
fn absorb_scope(py: Python, src: &CircuitData) -> PyResult<CircuitData> {
    // Phase 1: For each emission, determine where it gets absorbed (if anywhere).
    let mut local_absorptions: HashMap<u32, Path> = HashMap::new();
    let mut propagating_absorptions: HashMap<u32, Path> = HashMap::new();
    let mut absorbed_local_ids: HashSet<u32> = HashSet::new();

    for (i, inst) in src.data().iter().enumerate() {
        let Some(spec) = emission_spec(py, inst) else {
            continue;
        };

        let result = scan_for_collector(py, src, i, spec.direction, &spec, &[]);
        match result {
            ScanResult::AbsorbLocal { collector_path } => {
                local_absorptions.insert(spec.id, collector_path);
                absorbed_local_ids.insert(spec.id);
            }
            ScanResult::AbsorbPropagating { collector_path } => {
                propagating_absorptions.insert(spec.id, collector_path);
            }
            ScanResult::Standalone => {}
        }
    }

    // Also scan for emissions inside boxes that might escape outward.
    scan_inner_emissions(
        py,
        src,
        &[],
        &mut local_absorptions,
        &mut propagating_absorptions,
        &mut absorbed_local_ids,
    );

    // Phase 2: Rebuild the circuit.
    rebuild(py, src, &local_absorptions, &propagating_absorptions, &absorbed_local_ids, &[])
}

/// Recursively scan for emissions inside boxes that might escape upward.
fn scan_inner_emissions(
    py: Python,
    src: &CircuitData,
    path_prefix: &[usize],
    local_absorptions: &mut HashMap<u32, Path>,
    propagating_absorptions: &mut HashMap<u32, Path>,
    absorbed_local_ids: &mut HashSet<u32>,
) {
    for (i, inst) in src.data().iter().enumerate() {
        let Some(body) = box_body(src, inst) else {
            continue;
        };

        // Check emissions inside this box that might escape.
        let mut child_prefix = path_prefix.to_vec();
        child_prefix.push(i);

        for (j, inner_inst) in body.data().iter().enumerate() {
            if let Some(spec) = emission_spec(py, inner_inst) {
                if absorbed_local_ids.contains(&spec.id)
                    || propagating_absorptions.contains_key(&spec.id)
                {
                    continue;
                }
                // Try to find a collector within the box first.
                let inner_result =
                    scan_for_collector(py, body, j, spec.direction, &spec, &child_prefix);
                match inner_result {
                    ScanResult::AbsorbLocal { collector_path } => {
                        local_absorptions.insert(spec.id, collector_path);
                        absorbed_local_ids.insert(spec.id);
                    }
                    ScanResult::AbsorbPropagating { collector_path } => {
                        propagating_absorptions.insert(spec.id, collector_path);
                    }
                    ScanResult::Standalone => {
                        // The emission couldn't find a collector inside its box.
                        // Try scanning outward from the box in the parent scope.
                        let parent_result =
                            scan_for_collector(py, src, i, spec.direction, &spec, path_prefix);
                        match parent_result {
                            ScanResult::AbsorbLocal { collector_path } => {
                                local_absorptions.insert(spec.id, collector_path);
                                absorbed_local_ids.insert(spec.id);
                            }
                            ScanResult::AbsorbPropagating { collector_path } => {
                                propagating_absorptions.insert(spec.id, collector_path);
                            }
                            ScanResult::Standalone => {}
                        }
                    }
                }
            }
        }

        // Recurse deeper.
        scan_inner_emissions(
            py,
            body,
            &child_prefix,
            local_absorptions,
            propagating_absorptions,
            absorbed_local_ids,
        );
    }
}

/// Rebuild the circuit, removing locally absorbed emissions and updating collector specs.
fn rebuild(
    py: Python,
    src: &CircuitData,
    local_absorptions: &HashMap<u32, Path>,
    propagating_absorptions: &HashMap<u32, Path>,
    absorbed_local_ids: &HashSet<u32>,
    current_path: &[usize],
) -> PyResult<CircuitData> {
    let mut out = CircuitData::with_capacity(
        src.num_qubits() as u32,
        src.num_clbits() as u32,
        src.len(),
        qiskit_circuit::operations::Param::Float(0.0),
    )
    .into_py_result()?;

    for (i, inst) in src.data().iter().enumerate() {
        // Skip locally absorbed emissions (they become LocalEmission on their collector).
        // Propagating emissions stay on the spine for graph wiring.
        if let Some(spec) = emission_spec(py, inst) {
            if absorbed_local_ids.contains(&spec.id) {
                continue;
            }
            copy_through(src, inst, &mut out, None)?;
            continue;
        }

        // Update collectors that absorbed something.
        if let Some(spec) = collect_annotation(py, inst) {
            let mut my_path = current_path.to_vec();
            my_path.push(i);

            let new_items = build_collector_items(
                &spec,
                &my_path,
                local_absorptions,
                propagating_absorptions,
                src,
                py,
            );
            let new_spec = CollectSpec {
                items: new_items,
                partition: spec.partition.clone(),
                parts: spec.parts.clone(),
            };
            write_collector_with_spec(py, src, inst, &new_spec, &mut out)?;
            continue;
        }

        // Recurse into boxes.
        if let Some(body) = box_body(src, inst) {
            let mut child_path = current_path.to_vec();
            child_path.push(i);
            let new_body = rebuild(
                py,
                body,
                local_absorptions,
                propagating_absorptions,
                absorbed_local_ids,
                &child_path,
            )?;
            copy_through(src, inst, &mut out, Some(new_body))?;
            continue;
        }

        copy_through(src, inst, &mut out, None)?;
    }

    Ok(out)
}

/// Build the items list for a collector, incorporating any emissions it absorbed.
fn build_collector_items(
    spec: &CollectSpec,
    collector_path: &[usize],
    local_absorptions: &HashMap<u32, Path>,
    propagating_absorptions: &HashMap<u32, Path>,
    src: &CircuitData,
    py: Python,
) -> Vec<CollectItem> {
    // Find all emissions that target this collector.
    let local_ids: Vec<u32> = local_absorptions
        .iter()
        .filter(|(_, path)| path.as_slice() == collector_path)
        .map(|(id, _)| *id)
        .collect();

    let propagating_ids: Vec<u32> = propagating_absorptions
        .iter()
        .filter(|(_, path)| path.as_slice() == collector_path)
        .map(|(id, _)| *id)
        .collect();

    if local_ids.is_empty() && propagating_ids.is_empty() {
        return spec.items.clone();
    }

    // Find the actual EmitSpecs by scanning the whole circuit tree.
    let local_specs: Vec<EmitSpec> = local_ids
        .iter()
        .filter_map(|id| find_emit_spec_in_tree(src, py, *id))
        .collect();

    let propagating_specs: Vec<EmitSpec> = propagating_ids
        .iter()
        .filter_map(|id| find_emit_spec_in_tree(src, py, *id))
        .collect();

    // Build items with correct ordering based on composition semantics:
    // - ChangeBasis → always BEFORE gates (wraps the whole box)
    // - Twirl/InjectNoise direction=Right → BEFORE gates (right-dressed near half)
    // - Twirl/InjectNoise direction=Left → AFTER gates (left-dressed near half)
    let mut before_gates: Vec<CollectItem> = Vec::new();
    let mut after_gates: Vec<CollectItem> = Vec::new();

    for emit in &local_specs {
        let item = CollectItem::Emission(LocalEmission {
            source: emit.source,
            direction: emit.direction,
            partition: emit.partition.clone(),
            parts: emit.parts.clone(),
        });
        match emit.source {
            EmitSource::ChangeBasis => before_gates.push(item),
            EmitSource::Twirl | EmitSource::InjectNoise => match emit.direction {
                Direction::Right => before_gates.push(item),
                Direction::Left => after_gates.push(item),
            },
        }
    }

    for emit in &propagating_specs {
        match emit.source {
            EmitSource::ChangeBasis => before_gates.push(CollectItem::Incoming(emit.id)),
            EmitSource::Twirl | EmitSource::InjectNoise => match emit.direction {
                Direction::Right => before_gates.push(CollectItem::Incoming(emit.id)),
                Direction::Left => after_gates.push(CollectItem::Incoming(emit.id)),
            },
        }
    }

    // Final items: [before_gates..., original items (Gates)..., after_gates...]
    let mut items = Vec::new();
    items.extend(before_gates);
    items.extend(spec.items.iter().cloned());
    items.extend(after_gates);
    items
}

/// Find an EmitSpec anywhere in the circuit tree by ID.
fn find_emit_spec_in_tree(src: &CircuitData, py: Python, id: u32) -> Option<EmitSpec> {
    for inst in src.data() {
        if let Some(spec) = emission_spec(py, inst) {
            if spec.id == id {
                return Some(spec);
            }
        }
        if let Some(body) = box_body(src, inst) {
            if let Some(spec) = find_emit_spec_in_tree(body, py, id) {
                return Some(spec);
            }
        }
    }
    None
}

/// Write a collector instruction to the output with a new spec.
fn write_collector_with_spec(
    py: Python,
    src: &CircuitData,
    inst: &PackedInstruction,
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
