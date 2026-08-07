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

//! Merge collectors: emission circuit (IR2) → emission circuit (IR2).
//!
//! The build pass is local, so every annotated box gets its own two collectors. This pass applies the
//! contextual collection model of `.notebooks/design/contextual_collection.md`: adjacent boxes sharing a
//! synthesizer share a *middle* collector, so N boxes in a row need N+1 dressing layers rather than 2N.
//!
//! **It rebuilds rather than mutates**, which is what makes it able to widen a collector — a
//! `DAGCircuit` box cannot be widened in place, since `substitute_node_with_dag` maps the replacement's
//! qubits into the replaced node's existing qargs.
//!
//! **Everything is per qubit.** Each open collector carries a *frontier* — the wires on which nothing
//! has happened since its position, so a later collector on them can still commute back and fuse. Real
//! content, emissions and independently-positioned collectors all do the same thing to it: release the
//! qubits they touch. Nothing latches the whole scope. That is what lets box A's right collector stay
//! available on q2-3 after box B has claimed q0-1, so box C's left factor still merges into it — worth
//! stating because the earlier version cleared the whole frontier on real content and silently dropped
//! that merge whenever a box was right-dressed (a right dressing puts the hard box *before* the
//! emissions, so it arrived while the frontier was still whole).
//!
//! Two phases, because a merged collector stops growing later than it must be *placed*: by the time box
//! C's left factor fuses into box A's right collector, box B's own right collector has already been
//! opened. So phase 1 walks into a buffer where a collector is a slot into an arena, and phase 2
//! resolves the slots in position order.
//!
//! **Siblings only.** Merging across a box boundary — promoting an inner collector out of its box to
//! fuse with an outer one — is deliberately declined. It is sound as a manoeuvre but it takes the
//! promoted gates off the circuit spine, so an enclosing emission's propagation through them would have
//! to be recorded, which needs segment structure on `CollectSpec`. See the nesting section of
//! `SAMPLEX_IR_DESIGN.md`. Each scope is walked with its own state, so nothing merges across a boundary
//! by accident.

use hashbrown::HashSet;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use qiskit_circuit::annotation::PyAnnotation;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{ControlFlow, ControlFlowInstruction, OperationRef, Param};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::Qubit;

use super::utils::{
    IntoPyResult, block_body, collect_annotation, copy_through, params_of, qubit_indices,
    to_circuit, to_dag,
};
use crate::annotated_circuit::SynthesizerType;
use crate::emission_circuit::{Collect, CollectItem, CollectPart, CollectSpec};
use crate::partition::Partition;

/// One contribution of absorbed gates, with the qubits its body is expressed over.
///
/// Merging concatenates bodies of different widths — a 2-qubit collector fusing into a 4-qubit one —
/// so each contribution has to remember its own frame until the merged width is known.
struct Absorbed {
    /// Body-local qubit → circuit qubit.
    frame: Vec<usize>,
    body: CircuitData,
}

/// A collector that may still accept contributions.
struct OpenCollector {
    /// Qubits on which nothing has happened since this collector's position, so a later collector on
    /// them can still commute back here and fuse.
    ///
    /// Initialised to *every* qubit in the scope rather than to `span`, because a merge may widen onto
    /// qubits this collector never covered and those wires need tracking too. A collector is dead once
    /// `frontier` and `span` no longer intersect, which [`find_mergeable`] tests for implicitly.
    frontier: HashSet<usize>,
    /// Every qubit this collector covers. Monotonic — it determines the emitted box's width, and a
    /// collector whose wires have all been released must still be wide enough for everything it
    /// collected.
    span: HashSet<usize>,
    /// Per-part descriptors accumulated from merged contributions.
    partition: Partition,
    parts: Vec<CollectPart>,
    /// Composition order, one contribution's run after another. A run's `Gates` counts refer to that
    /// contribution's body, and [`write_collector`] concatenates bodies in the same order, so counts
    /// stay valid without any offsetting — which is the whole reason they are counts.
    items: Vec<CollectItem>,
    absorbed: Vec<Absorbed>,
}

impl OpenCollector {
    fn qubits(&self) -> Vec<usize> {
        let mut qubits: Vec<usize> = self.span.iter().copied().collect();
        qubits.sort_unstable();
        qubits
    }
}

/// One entry in the phase-1 buffer.
enum Item {
    /// Reserves a collector's position; the arena entry may still grow.
    Collector(usize),
    /// An instruction copied through unchanged, by index into the source circuit.
    Copy(usize),
    /// A box whose body was recursively merged, by index into the source plus the new body.
    /// Boxed to keep this enum small — a `CircuitData` dwarfs the other variants.
    Rebuilt(usize, Box<CircuitData>),
}

/// Merge adjacent collectors throughout an emission circuit, in place.
#[pyfunction]
#[pyo3(name = "merge_collectors")]
pub fn py_merge_collectors(py: Python, dag: &mut DAGCircuit) -> PyResult<()> {
    let src = to_circuit(dag)?;
    let out = merge_collectors(py, &src)?;
    *dag = to_dag(&out)?;
    Ok(())
}

/// Merge adjacent collectors throughout an emission circuit.
pub fn merge_collectors(py: Python, circuit: &CircuitData) -> PyResult<CircuitData> {
    merge_scope(py, circuit)
}

/// Merge collectors within one scope, recursing into box bodies with fresh state.
fn merge_scope(py: Python, src: &CircuitData) -> PyResult<CircuitData> {
    let mut items: Vec<Item> = Vec::with_capacity(src.len());
    let mut open: Vec<OpenCollector> = Vec::new();
    let num_qubits = src.num_qubits();

    for (index, inst) in src.data().iter().enumerate() {
        if let Some(spec) = collect_annotation(py, inst) {
            let qubits = qubit_indices(src, inst);
            let absorbed = Absorbed {
                frame: qubits.clone(),
                body: block_body(src, inst)?.cloned().unwrap_or(new_body(qubits.len())?),
            };
            match find_mergeable(&open, &qubits, spec.synthesizer()) {
                Some(idx) => {
                    // Fuse into the open collector: it keeps its position, and gains this one's
                    // emissions, absorbed gates and qubits. Nothing is released — a merged
                    // contribution has no position of its own to get in anything's way.
                    // Items and bodies both append, and in the same order, so the two stay in step.
                    // The resulting sequence is right: A's outermost element ends up adjacent to B's
                    // outermost, which is how the two layers meet in circuit order.
                    let target = &mut open[idx];
                    target.items.extend_from_slice(&spec.items);
                    target.absorbed.push(absorbed);
                    target.span.extend(qubits.iter().copied());
                    // Widen the partition to cover both collectors' qubits.
                    target.partition =
                        Partition::union(&[&target.partition, &spec.partition])
                            .unwrap_or_else(|_| spec.partition.clone());
                    // Rebuild parts to match the widened partition. find_mergeable ensures all parts
                    // share the same synthesizer, so we replicate uniformly.
                    let synth = target.parts[0].synthesizer;
                    target.parts = (0..target.partition.len())
                        .map(|_| CollectPart { synthesizer: synth })
                        .collect();
                }
                None => {
                    // Nothing compatible is open on these qubits, so this collector gets a position of
                    // its own — and becomes a synth layer, i.e. real gates in the template. That
                    // blocks any later collector from reaching back past it on these wires.
                    release(&mut open, &qubits);
                    items.push(Item::Collector(open.len()));
                    open.push(OpenCollector {
                        frontier: (0..num_qubits).collect(),
                        span: qubits.iter().copied().collect(),
                        partition: spec.partition.clone(),
                        parts: spec.parts.clone(),
                        items: spec.items.clone(),
                        absorbed: vec![absorbed],
                    });
                }
            }
            continue;
        }

        // An emission is a twirl point, and real content ends absorption. Either way these wires stop
        // being at any open collector's frontier, so there is no distinction left to draw here.
        let qubits = qubit_indices(src, inst);
        release(&mut open, &qubits);

        match inst.op.view() {
            OperationRef::ControlFlow(cf) if matches!(cf.control_flow, ControlFlow::Box { .. }) => {
                // Recurse with fresh state, so a nested scope's collectors merge among themselves
                // but never across the boundary.
                let body = block_body(src, inst)?.ok_or_else(|| {
                    PyValueError::new_err("box instruction is missing its body")
                })?;
                items.push(Item::Rebuilt(index, Box::new(merge_scope(py, body)?)));
            }
            _ => items.push(Item::Copy(index)),
        }
    }

    materialize(py, src, &items, &open)
}

/// The open collector this one may fuse into, if any. First match wins, which keeps the result
/// deterministic since `open` is in the order collectors were encountered.
fn find_mergeable(
    open: &[OpenCollector],
    qubits: &[usize],
    synthesizer: SynthesizerType,
) -> Option<usize> {
    open.iter().position(|candidate| {
        candidate.parts.iter().all(|p| p.synthesizer == synthesizer)
            // A shared qubit is what gives the two collectors a temporal order to follow. Two
            // collectors on disjoint qubits are *concurrent*: their relative position in this circuit
            // is an artifact of whichever topological order `build` happened to walk, so fusing them
            // would make the output depend on an arbitrary choice. An overlap fixes the order in every
            // topological order, which is why it is required rather than merely usual.
            && qubits.iter().any(|q| candidate.span.contains(q))
            // Every qubit has to still be at the frontier, not just the shared one. A wire something
            // has already touched cannot commute back to this collector's position for free — the
            // emission's walk would pick up a conjugation by whatever that was, which at best costs a
            // propagation step and at worst has no rule and gets refused.
            //
            // This also excludes a dead collector without a flag: a shared qubit must be in both
            // `span` and `frontier`, so if they no longer intersect nothing can match.
            && qubits.iter().all(|q| candidate.frontier.contains(q))
    })
}

/// Take `qubits` off every open collector's frontier.
///
/// Something now sits between those wires and every open collector's position, so a later collector on
/// them can no longer commute back to fuse. This is the *only* closing rule — real content, emissions
/// and independently-positioned collectors all do exactly this, per qubit.
fn release(open: &mut [OpenCollector], qubits: &[usize]) {
    for collector in open.iter_mut() {
        for q in qubits {
            collector.frontier.remove(q);
        }
    }
}

/// Phase 2: write the buffer out in position order, resolving each collector slot.
fn materialize(
    py: Python,
    src: &CircuitData,
    items: &[Item],
    open: &[OpenCollector],
) -> PyResult<CircuitData> {
    let mut out = CircuitData::with_capacity(
        src.num_qubits() as u32,
        src.num_clbits() as u32,
        items.len(),
        Param::Float(0.0),
    )
    .into_py_result()?;

    for item in items {
        match item {
            Item::Collector(idx) => write_collector(py, &mut out, &open[*idx])?,
            Item::Copy(index) => copy_through(src, &src.data()[*index], &mut out, None)?,
            Item::Rebuilt(index, body) => {
                copy_through(src, &src.data()[*index], &mut out, Some((**body).clone()))?
            }
        }
    }
    Ok(out)
}

/// Emit one merged collector: a box over its full span, holding every absorbed contribution remapped
/// into the merged frame.
fn write_collector(py: Python, out: &mut CircuitData, collector: &OpenCollector) -> PyResult<()> {
    let qubits = collector.qubits();

    let mut body = new_body(qubits.len())?;
    for absorbed in &collector.absorbed {
        for inst in absorbed.body.data() {
            let qargs: Vec<Qubit> = absorbed
                .body
                .qargs_interner()
                .get(inst.qubits)
                .iter()
                .map(|q| {
                    let global = absorbed.frame[q.index()];
                    let local = qubits.iter().position(|x| *x == global).ok_or_else(|| {
                        PyValueError::new_err(format!("qubit {global} outside the merged span"))
                    })?;
                    Ok(Qubit(local as u32))
                })
                .collect::<PyResult<_>>()?;
            let params = params_of(inst);
            body.push_packed_operation(inst.op.clone(), params, &qargs, &[])
                .into_py_result()?;
        }
    }

    let annotation = Py::new(
        py,
        (
            Collect::new_from_spec(CollectSpec {
                items: collector.items.clone(),
                partition: collector.partition.clone(),
                parts: collector.parts.clone(),
            }),
            PyAnnotation,
        ),
    )?;
    let qargs: Vec<Qubit> = qubits.iter().map(|q| Qubit(*q as u32)).collect();
    let op = PackedOperation::from_control_flow(Box::new(ControlFlowInstruction {
        control_flow: ControlFlow::Box {
            duration: None,
            annotations: vec![annotation.into_any()],
        },
        num_qubits: qargs.len() as u32,
        num_clbits: 0,
    }));
    let block = out.add_block(body);
    out.push_packed_operation(op, Some(Parameters::Blocks(vec![block])), &qargs, &[])
        .into_py_result()
}

fn new_body(num_qubits: usize) -> PyResult<CircuitData> {
    super::utils::new_circuit_body(num_qubits, 0, 0)
}

