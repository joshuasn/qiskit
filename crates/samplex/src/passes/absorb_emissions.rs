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

//! Absorb dressing into collectors.
//!
//! After the build pass, every emission and every easy gate sits on the spine and every collector
//! starts empty (no items, no body). This pass walks outward from each collector, absorbing
//! adjacent content in circuit order until hitting something unabsorbable:
//!
//! - A single-qubit standard gate → absorbed as `CollectItem::Gates`
//! - An emission whose direction faces this collector → absorbed as `CollectItem::Emission`
//! - Anything else (propagating emission, multi-qubit gate, box, another collector) → stop
//!
//! The walk's circuit order IS the composition order, so no `EmitSource`-based classification is
//! needed. Cross-scope absorption (emissions escaping inner boxes) is handled by descending into
//! boxes at the walk boundary.

use hashbrown::HashSet;

use pyo3::prelude::*;
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{ControlFlow, ControlFlowInstruction, OperationRef, Param};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::{Clbit, Qubit};

use super::utils::{collect_annotation, copy_through, emission_spec, IntoPyResult};
use crate::emission_circuit::{Collect, CollectItem, CollectSpec, LocalEmission};
use crate::virtual_flow_graph::Direction;

#[pyfunction]
#[pyo3(name = "absorb_emissions")]
pub fn py_absorb_emissions(py: Python, circuit: &PyCircuitData) -> PyResult<PyCircuitData> {
    Ok(PyCircuitData {
        inner: absorb_dressing(py, &circuit.inner)?,
    })
}

pub fn absorb_emissions(py: Python, src: &CircuitData) -> PyResult<CircuitData> {
    absorb_dressing(py, src)
}

/// Walk from each collector, absorbing adjacent dressing content.
fn absorb_dressing(py: Python, src: &CircuitData) -> PyResult<CircuitData> {
    absorb_scope(py, src)
}

/// Whether an instruction is a single-qubit standard gate (absorbable by a collector).
fn is_absorbable_gate(src: &CircuitData, inst: &PackedInstruction) -> bool {
    matches!(inst.op.view(), OperationRef::StandardGate(_))
        && src.qargs_interner().get(inst.qubits).len() == 1
}

/// The body of a box instruction, if it is one.
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

/// Absorb one scope: find each collector, walk outward from it, rebuild the circuit.
fn absorb_scope(py: Python, src: &CircuitData) -> PyResult<CircuitData> {
    let mut absorbed_positions: HashSet<usize> = HashSet::new();
    let mut collector_data: Vec<(usize, Vec<CollectItem>, Vec<usize>)> = Vec::new();

    for (i, inst) in src.data().iter().enumerate() {
        if collect_annotation(py, inst).is_none() {
            continue;
        }

        let mut items: Vec<CollectItem> = Vec::new();
        let mut body_positions: Vec<usize> = Vec::new();
        let mut all_absorbed: Vec<usize> = Vec::new();

        // Walk RIGHT from collector.
        walk_absorb(
            py,
            src,
            i,
            Direction::Right,
            &mut items,
            &mut body_positions,
            &mut all_absorbed,
        );

        // Walk LEFT from collector, collecting in walk order (outermost first when reversed).
        let mut left_items: Vec<CollectItem> = Vec::new();
        let mut left_body: Vec<usize> = Vec::new();
        let mut left_absorbed: Vec<usize> = Vec::new();
        walk_absorb(
            py,
            src,
            i,
            Direction::Left,
            &mut left_items,
            &mut left_body,
            &mut left_absorbed,
        );
        left_items.reverse();
        left_body.reverse();

        // Combine: left (reversed) + right.
        let mut combined_items = left_items;
        let mut combined_body = left_body;
        combined_items.extend(items);
        combined_body.extend(body_positions);

        absorbed_positions.extend(left_absorbed);
        absorbed_positions.extend(all_absorbed);

        collector_data.push((i, combined_items, combined_body));
    }

    // Phase 2: cross-scope absorption. Remaining standalone emissions that can descend into an
    // adjacent box are marked for injection. During rebuild, they are inserted into the box body
    // at the near edge so that the recursive `absorb_scope` naturally absorbs them.
    // injection_map: box_position → list of (emission_position, direction) to inject.
    let mut injection_map: Vec<(usize, Vec<(usize, Direction)>)> = Vec::new();

    for (i, inst) in src.data().iter().enumerate() {
        if absorbed_positions.contains(&i) {
            continue;
        }
        let Some(spec) = emission_spec(py, inst) else {
            continue;
        };
        // Scan in the emission's direction for a box to descend into.
        if let Some(box_pos) =
            find_descent_target(py, src, i, spec.direction, &absorbed_positions)
        {
            absorbed_positions.insert(i);
            if let Some((_, injections)) = injection_map.iter_mut().find(|(pos, _)| *pos == box_pos)
            {
                injections.push((i, spec.direction));
            } else {
                injection_map.push((box_pos, vec![(i, spec.direction)]));
            }
        }
    }

    // Phase 3: rebuild the circuit.
    rebuild(py, src, &absorbed_positions, &collector_data, &injection_map)
}

/// Walk from `start` in `direction`, absorbing single-qubit gates and facing emissions.
///
/// Stops at the first instruction that cannot be absorbed. Items are pushed in walk order.
fn walk_absorb(
    py: Python,
    src: &CircuitData,
    start: usize,
    direction: Direction,
    items: &mut Vec<CollectItem>,
    body_positions: &mut Vec<usize>,
    all_absorbed: &mut Vec<usize>,
) {
    let range: Box<dyn Iterator<Item = usize>> = match direction {
        Direction::Right => Box::new((start + 1)..src.len()),
        Direction::Left => Box::new((0..start).rev()),
    };

    // The direction an emission must have to "face" this collector.
    let facing = match direction {
        Direction::Right => Direction::Left,
        Direction::Left => Direction::Right,
    };

    let mut consecutive_gates: usize = 0;

    for i in range {
        let inst = &src.data()[i];

        // Single-qubit standard gate: absorb.
        if is_absorbable_gate(src, inst) {
            consecutive_gates += 1;
            body_positions.push(i);
            all_absorbed.push(i);
            continue;
        }

        // Emission facing this collector: absorb.
        if let Some(spec) = emission_spec(py, inst) {
            if spec.direction == facing {
                // Flush any pending gates before this emission.
                if consecutive_gates > 0 {
                    items.push(CollectItem::Gates(consecutive_gates));
                    consecutive_gates = 0;
                }
                items.push(CollectItem::Emission(LocalEmission {
                    partition: spec.partition.clone(),
                    parts: spec.parts.clone(),
                }));
                all_absorbed.push(i);
                continue;
            }
            // Propagating emission (facing away): stop.
            break;
        }

        // Anything else (box, multi-qubit gate, another collector): stop.
        break;
    }

    // Flush trailing gates.
    if consecutive_gates > 0 {
        items.push(CollectItem::Gates(consecutive_gates));
    }
}

/// Scan from an emission in its direction, looking for a box with a compatible collector at its
/// near edge.
///
/// Returns the position of the box if one is found whose inner near edge has a compatible
/// collector. The emission will be injected into that box's body during rebuild so that the
/// recursive `absorb_scope` on the inner body naturally absorbs it.
fn find_descent_target(
    py: Python,
    src: &CircuitData,
    start: usize,
    direction: Direction,
    absorbed: &HashSet<usize>,
) -> Option<usize> {
    let spec = emission_spec(py, &src.data()[start])?;

    let range: Box<dyn Iterator<Item = usize>> = match direction {
        Direction::Right => Box::new((start + 1)..src.len()),
        Direction::Left => Box::new((0..start).rev()),
    };

    for i in range {
        if absorbed.contains(&i) {
            continue;
        }
        let inst = &src.data()[i];

        // Skip other emissions — cross-scope emissions pass through them.
        if emission_spec(py, inst).is_some() {
            continue;
        }

        // If we hit a collector at this scope level, this emission doesn't descend.
        if collect_annotation(py, inst).is_some() {
            return None;
        }

        // If we hit a box, check if there's a compatible collector at the near edge.
        if let Some(body) = box_body(src, inst) {
            if has_compatible_collector_at_edge(py, body, direction, &spec) {
                return Some(i);
            }
            // Box without a compatible collector at its near edge — emission can't descend here.
            return None;
        }

        // Skip gates — propagating emissions pass through them.
        continue;
    }
    None
}

/// Check if a box body has a compatible collector at the near edge (from the given direction).
///
/// "Near edge" means: if the emission travels Right, it enters the box from the Left edge
/// (position 0 onward). If it travels Left, it enters from the Right edge (last position backward).
/// We skip emissions at the edge and look for a collector. If we find a nested box instead,
/// recurse into it.
fn has_compatible_collector_at_edge(
    py: Python,
    body: &CircuitData,
    direction: Direction,
    spec: &crate::emission_circuit::EmitSpec,
) -> bool {
    let range: Box<dyn Iterator<Item = usize>> = match direction {
        Direction::Right => Box::new(0..body.len()),
        Direction::Left => Box::new((0..body.len()).rev()),
    };

    for j in range {
        let inst = &body.data()[j];

        // Skip emissions at the edge — they don't block descent.
        if emission_spec(py, inst).is_some() {
            continue;
        }

        // Found a collector at the near edge — check compatibility.
        if let Some(coll_spec) = collect_annotation(py, inst) {
            return coll_spec.accepts(spec.virtual_type());
        }

        // Nested box — recurse into it.
        if let Some(inner_body) = box_body(body, inst) {
            return has_compatible_collector_at_edge(py, inner_body, direction, spec);
        }

        // Gate or anything else at near edge — no compatible collector here.
        return false;
    }
    false
}

/// Rebuild the circuit: remove absorbed positions, rewrite collectors with new specs and bodies,
/// and inject cross-scope emissions into box bodies before recursing.
fn rebuild(
    py: Python,
    src: &CircuitData,
    absorbed: &HashSet<usize>,
    collector_data: &[(usize, Vec<CollectItem>, Vec<usize>)],
    injection_map: &[(usize, Vec<(usize, Direction)>)],
) -> PyResult<CircuitData> {
    let mut out = CircuitData::with_capacity(
        src.num_qubits() as u32,
        src.num_clbits() as u32,
        src.len(),
        Param::Float(0.0),
    )
    .into_py_result()?;

    for (i, inst) in src.data().iter().enumerate() {
        // Skip absorbed instructions (gates and emissions now owned by a collector).
        if absorbed.contains(&i) {
            continue;
        }

        // Rewrite collectors with their absorbed content.
        if let Some(spec) = collect_annotation(py, inst) {
            if let Some((_, items, body_positions)) =
                collector_data.iter().find(|(pos, _, _)| *pos == i)
            {
                write_collector(py, src, inst, &spec, items, body_positions, &mut out)?;
            } else {
                // Collector with nothing absorbed — copy through unchanged.
                copy_through(src, inst, &mut out, None)?;
            }
            continue;
        }

        // Recurse into boxes, injecting cross-scope emissions before recursing.
        if let Some(body) = box_body(src, inst) {
            let new_body = if let Some((_, injections)) =
                injection_map.iter().find(|(pos, _)| *pos == i)
            {
                // Build a modified body with injected emissions at the near edges.
                let injected_body = inject_emissions_into_body(py, src, body, injections)?;
                absorb_scope(py, &injected_body)?
            } else {
                absorb_scope(py, body)?
            };
            copy_through(src, inst, &mut out, Some(new_body))?;
            continue;
        }

        // Everything else (propagating emissions, hard boxes, etc.) copies through.
        copy_through(src, inst, &mut out, None)?;
    }

    Ok(out)
}

/// Inject emissions from the outer scope into a box body at the appropriate edge.
///
/// Emissions walking Right enter the box from the Left edge (prepended).
/// Emissions walking Left enter from the Right edge (appended).
fn inject_emissions_into_body(
    py: Python,
    src: &CircuitData,
    body: &CircuitData,
    injections: &[(usize, Direction)],
) -> PyResult<CircuitData> {
    let mut new_body = CircuitData::with_capacity(
        body.num_qubits() as u32,
        body.num_clbits() as u32,
        body.len() + injections.len(),
        Param::Float(0.0),
    )
    .into_py_result()?;

    // Emissions entering from the left edge (direction = Right → enters left side of box).
    let left_injections: Vec<&(usize, Direction)> =
        injections.iter().filter(|(_, d)| *d == Direction::Right).collect();
    // Emissions entering from the right edge (direction = Left → enters right side of box).
    let right_injections: Vec<&(usize, Direction)> =
        injections.iter().filter(|(_, d)| *d == Direction::Left).collect();

    // Prepend left-edge injections.
    for (emit_pos, _dir) in &left_injections {
        write_injected_emission(py, src, *emit_pos, body.num_qubits() as u32, &mut new_body)?;
    }

    // Copy existing body content.
    for inst in body.data() {
        copy_through(body, inst, &mut new_body, None)?;
    }

    // Append right-edge injections.
    for (emit_pos, _dir) in &right_injections {
        write_injected_emission(py, src, *emit_pos, body.num_qubits() as u32, &mut new_body)?;
    }

    Ok(new_body)
}

/// Write a single emission instruction (from the outer scope) into the inner body.
///
/// The emission's qubits are remapped: in the outer scope it acts on the same qubits as the box,
/// so qubit indices map 1:1 from the box's qargs position to body qubit indices.
fn write_injected_emission(
    _py: Python,
    src: &CircuitData,
    emit_pos: usize,
    body_num_qubits: u32,
    out: &mut CircuitData,
) -> PyResult<()> {
    let emit_inst = &src.data()[emit_pos];

    // The emission in the outer scope acts on some subset of qubits. In the inner body,
    // we need to figure out which body qubits correspond. For simplicity and correctness:
    // the emission acts on the same qubits as the box (full width), so we map position-to-position.
    // The box's i-th qubit arg in the outer scope → body qubit i.
    let qargs: Vec<Qubit> = (0..body_num_qubits).map(Qubit).collect();

    // Copy the operation as-is (it's a PyCustom Emit instruction).
    let params = if !emit_inst.params_view().is_empty() {
        Some(Parameters::Params(
            emit_inst.params_view().iter().cloned().collect(),
        ))
    } else {
        None
    };

    out.push_packed_operation(emit_inst.op.clone(), params, &qargs, &[])
        .into_py_result()
}

/// Write a collector with its newly absorbed items and body.
fn write_collector(
    py: Python,
    src: &CircuitData,
    inst: &PackedInstruction,
    original_spec: &CollectSpec,
    items: &[CollectItem],
    body_positions: &[usize],
    out: &mut CircuitData,
) -> PyResult<()> {
    use qiskit_circuit::annotation::PyAnnotation;
    use smallvec::SmallVec;

    // Build the new body from the absorbed gate positions.
    let width = src.qargs_interner().get(inst.qubits).len();
    let mut body = CircuitData::with_capacity(width as u32, 0, body_positions.len(), Param::Float(0.0))
        .into_py_result()?;

    // The collector's qargs in this scope.
    let collector_qargs: Vec<usize> = src
        .qargs_interner()
        .get(inst.qubits)
        .iter()
        .map(|q| q.index())
        .collect();

    for &pos in body_positions {
        let gate = &src.data()[pos];
        // Remap gate qubits from scope-frame to collector-body-frame.
        let gate_qargs: Vec<Qubit> = src
            .qargs_interner()
            .get(gate.qubits)
            .iter()
            .map(|q| {
                let scope_idx = q.index();
                let body_idx = collector_qargs
                    .iter()
                    .position(|&cq| cq == scope_idx)
                    .unwrap_or(0);
                Qubit(body_idx as u32)
            })
            .collect();
        let params: Option<Parameters<_>> = (!gate.params_view().is_empty()).then(|| {
            Parameters::Params(gate.params_view().iter().cloned().collect::<SmallVec<[Param; 3]>>())
        });
        body.push_packed_operation(gate.op.clone(), params, &gate_qargs, &[])
            .into_py_result()?;
    }

    // Build new spec with accumulated items.
    let new_spec = CollectSpec {
        items: items.to_vec(),
        partition: original_spec.partition.clone(),
        parts: original_spec.parts.clone(),
    };

    debug_assert_eq!(
        new_spec.gate_count(),
        body.len(),
        "absorbed gates must match items Gates counts"
    );

    let qargs: Vec<Qubit> = src.qargs_interner().get(inst.qubits).to_vec();
    let cargs: Vec<Clbit> = src.cargs_interner().get(inst.clbits).to_vec();

    let annotation = Py::new(py, (Collect::new_from_spec(new_spec), PyAnnotation))?;
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
