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

//! Merge collectors: emission circuit (IR2) → emission circuit (IR2), in place.
//!
//! The build pass is local, so every annotated box gets its own two collectors. This pass applies the
//! contextual collection model of `.notebooks/design/contextual_collection.md`: adjacent boxes sharing a
//! synthesizer share a *middle* collector, so N boxes in a row need N+1 dressing layers rather than 2N.
//!
//! **Merging is one contraction per group.** [`DAGCircuit::replace_block`] contracts a set of nodes
//! into a single node whose qargs is the *union* of theirs, so a merged collector is wider than any of
//! its members without anything being rebuilt — the widening this pass exists to do is a primitive of
//! the representation. Contraction also derives the merged node's *position* from the members' edges,
//! which is why the group can go on growing after the point it will end up occupying: there is no
//! placement decision to make early, so there is no buffer to make it into.
//!
//! **Everything is per qubit.** Each group carries a *frontier* — the wires on which nothing has
//! happened since its position, so a later collector on them can still commute back and fuse. Real
//! content, emissions and independently-positioned collectors all do the same thing to it: release the
//! qubits they touch. Nothing latches the whole scope. That is what lets box A's right collector stay
//! available on q2-3 after box B has claimed q0-1, so box C's left factor still merges into it — worth
//! stating because the earlier version cleared the whole frontier on real content and silently dropped
//! that merge whenever a box was right-dressed (a right dressing puts the hard box *before* the
//! emissions, so it arrived while the frontier was still whole).
//!
//! The frontier is a stronger condition than "the contraction is legal", and deliberately so.
//! Contraction is refused only when something is forced *between* two members, which says nothing about
//! a wire that merely one of them covers: an unrelated gate on q2 is concurrent with a collector on
//! q0-1, so contracting the two would be perfectly legal. It is still refused, because widening a
//! dressing layer onto a wire that something has already crossed would make an emission's walk pick up
//! a conjugation by that gate. `cycle_check` is therefore passed as a consistency check on the frontier
//! rule rather than as the rule itself.
//!
//! **Siblings only.** Merging across a box boundary — promoting an inner collector out of its box to
//! fuse with an outer one — is deliberately declined. It is sound as a manoeuvre but it takes the
//! promoted gates off the circuit spine, so an enclosing emission's propagation through them would have
//! to be recorded, which needs segment structure on `CollectSpec`. See the nesting section of
//! `SAMPLEX_IR_DESIGN.md`. Each scope is walked with its own state, so nothing merges across a boundary
//! by accident.

use hashbrown::{HashMap, HashSet};
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use pyo3::prelude::*;
use qiskit_circuit::annotation::PyAnnotation;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{ControlFlow, ControlFlowInstruction, OperationRef};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::{Clbit, Qubit};

use super::utils::{IntoPyResult, collect_annotation, new_dag_body, params_of};
use crate::annotated_circuit::SynthesizerType;
use crate::emission_circuit::{Collect, CollectItem, CollectPart, CollectSpec};
use crate::partition::Partition;

/// Collectors that will fuse into one, and the state deciding what else may join them.
struct Group {
    /// The collector nodes to contract, in the order their contributions compose. A group of one is
    /// left alone: there is nothing to fuse, and re-emitting it would only replace it with a copy.
    members: Vec<NodeIndex>,
    /// Qubits on which nothing has happened since this group's position, so a later collector on them
    /// can still commute back here and fuse.
    ///
    /// Initialised to *every* qubit in the scope rather than to `span`, because a merge may widen onto
    /// qubits no member ever covered and those wires need tracking too. A group is dead once
    /// `frontier` and `span` no longer intersect, which [`find_mergeable`] tests for implicitly.
    frontier: HashSet<Qubit>,
    /// Every qubit this group covers. Monotonic — it is the width the contracted box will have, and a
    /// group whose wires have all been released must still be wide enough for everything it collected.
    span: HashSet<Qubit>,
    /// Per-part descriptors accumulated from merged contributions.
    partition: Partition,
    parts: Vec<CollectPart>,
    /// Every annotated box whose emissions the contracted collector may consume — the union over the
    /// members, since a merged collector answers for all of their boxes.
    owned: Vec<u32>,
    /// Composition order, one contribution's run after another. A run's `Gates` counts refer to that
    /// contribution's body, and [`merged_body`] concatenates bodies in the same order, so counts stay
    /// valid without any offsetting — which is the whole reason they are counts.
    items: Vec<CollectItem>,
}

impl Group {
    /// The group's qubits in the frame the contracted node will use.
    ///
    /// Sorted, because [`DAGCircuit::replace_block`] orders the contracted node's qargs by the
    /// position map it is given, and this pass maps each qubit to its own index.
    fn frame(&self) -> Vec<Qubit> {
        let mut qubits: Vec<Qubit> = self.span.iter().copied().collect();
        qubits.sort_unstable();
        qubits
    }
}

/// Merge adjacent collectors throughout an emission circuit, in place.
#[pyfunction]
#[pyo3(name = "merge_collectors")]
pub fn py_merge_collectors(py: Python, dag: &mut DAGCircuit) -> PyResult<()> {
    merge_collectors(py, dag)
}

/// Merge adjacent collectors throughout an emission circuit, in place.
pub fn merge_collectors(py: Python, dag: &mut DAGCircuit) -> PyResult<()> {
    merge_scope(py, dag)
}

/// Merge collectors within one scope, then recurse into box bodies with fresh state.
fn merge_scope(py: Python, dag: &mut DAGCircuit) -> PyResult<()> {
    let all: Vec<Qubit> = (0..dag.num_qubits() as u32).map(Qubit).collect();
    let mut groups: Vec<Group> = Vec::new();
    let mut bodies: Vec<qiskit_circuit::Block> = Vec::new();

    // Grouping reads the DAG and contracting mutates it, so the sweep records node indices and the
    // rewriting happens afterwards. `StableDiGraph` keeps the indices valid in between.
    for node in dag.topological_op_nodes(false).collect::<Vec<_>>() {
        let inst = dag.dag()[node].unwrap_operation();
        let qubits: Vec<Qubit> = dag.qargs_interner().get(inst.qubits).to_vec();

        if let Some(spec) = collect_annotation(py, inst) {
            match find_mergeable(&groups, &qubits, spec.synthesizer()) {
                // Fuse into the open group: it keeps its position, and gains this collector's
                // emissions, absorbed gates and qubits. Nothing is released — a merged contribution
                // has no position of its own to get in anything's way. Items and bodies both append,
                // and in the same order, so the two stay in step. The resulting sequence is right:
                // A's outermost element ends up adjacent to B's outermost, which is how the two
                // layers meet in circuit order.
                Some(index) => join(&mut groups[index], node, &spec, &qubits),
                None => {
                    // Nothing compatible is open on these qubits, so this collector gets a position
                    // of its own — and becomes a synth layer, i.e. real gates in the template. That
                    // blocks any later collector from reaching back past it on these wires.
                    release(&mut groups, &qubits);
                    groups.push(Group {
                        members: vec![node],
                        frontier: all.iter().copied().collect(),
                        span: qubits.iter().copied().collect(),
                        partition: spec.partition.clone(),
                        parts: spec.parts.clone(),
                        owned: spec.owned.clone(),
                        items: spec.items.clone(),
                    });
                }
            }
            continue;
        }

        // An emission is a twirl point, and real content ends absorption. Either way these wires stop
        // being at any open group's frontier, so there is no distinction left to draw here.
        release(&mut groups, &qubits);
        if let [block] = *inst.blocks_view()
            && is_box(inst)
        {
            bodies.push(block);
        }
    }

    for group in &groups {
        if group.members.len() > 1 {
            fuse(py, dag, group)?;
        }
    }

    // Recurse with fresh state, so a nested scope's collectors merge among themselves but never
    // across the boundary.
    for block in bodies {
        merge_scope(py, dag.view_block_mut(block))?;
    }
    Ok(())
}

/// Whether an instruction is a `box`.
fn is_box(inst: &PackedInstruction) -> bool {
    matches!(inst.op.view(), OperationRef::ControlFlow(cf) if matches!(cf.control_flow, ControlFlow::Box { .. }))
}

/// The open group this collector may fuse into, if any. First match wins, which keeps the result
/// deterministic since `groups` is in the order collectors were encountered.
fn find_mergeable(
    groups: &[Group],
    qubits: &[Qubit],
    synthesizer: SynthesizerType,
) -> Option<usize> {
    groups.iter().position(|candidate| {
        candidate.parts.iter().all(|p| p.synthesizer == synthesizer)
            // A shared qubit is what gives the two collectors a temporal order to follow. Two
            // collectors on disjoint qubits are *concurrent*: their relative position in this circuit
            // is an artifact of whichever topological order the walk happened to take, so fusing them
            // would make the output depend on an arbitrary choice. An overlap fixes the order in every
            // topological order, which is why it is required rather than merely usual.
            && qubits.iter().any(|q| candidate.span.contains(q))
            // Every qubit has to still be at the frontier, not just the shared one. A wire something
            // has already touched cannot commute back to this group's position for free — the
            // emission's walk would pick up a conjugation by whatever that was, which at best costs a
            // propagation step and at worst has no rule and gets refused.
            //
            // This also excludes a dead group without a flag: a shared qubit must be in both `span`
            // and `frontier`, so if they no longer intersect nothing can match.
            && qubits.iter().all(|q| candidate.frontier.contains(q))
    })
}

/// Add a collector's contribution to an open group.
fn join(group: &mut Group, node: NodeIndex, spec: &CollectSpec, qubits: &[Qubit]) {
    group.members.push(node);
    group.items.extend_from_slice(&spec.items);
    group.span.extend(qubits.iter().copied());
    // The contracted collector stands in for both boxes, so it may consume either one's emissions.
    // Sorted and deduplicated, so the set does not depend on the order members were visited in.
    group.owned.extend_from_slice(&spec.owned);
    group.owned.sort_unstable();
    group.owned.dedup();
    // Widen the partition to cover both collectors' qubits.
    group.partition = Partition::union(&[&group.partition, &spec.partition])
        .unwrap_or_else(|_| spec.partition.clone());
    // Rebuild parts to match the widened partition. `find_mergeable` ensures all parts share the same
    // synthesizer, so we replicate uniformly.
    let synthesizer = group.parts[0].synthesizer;
    group.parts = (0..group.partition.len())
        .map(|_| CollectPart { synthesizer })
        .collect();
}

/// Take `qubits` off every open group's frontier.
///
/// Something now sits between those wires and every open group's position, so a later collector on them
/// can no longer commute back to fuse. This is the *only* closing rule — real content, emissions and
/// independently-positioned collectors all do exactly this, per qubit.
fn release(groups: &mut [Group], qubits: &[Qubit]) {
    for group in groups.iter_mut() {
        for qubit in qubits {
            group.frontier.remove(qubit);
        }
    }
}

/// Contract one group into a single collector over its full span.
fn fuse(py: Python, dag: &mut DAGCircuit, group: &Group) -> PyResult<()> {
    let frame = group.frame();
    let clbits = merged_clbits(dag, group);
    let body = merged_body(dag, group, &frame, clbits.len())?;
    let op = merged_op(py, group, frame.len(), clbits.len())?;

    // `replace_block` derives the contracted node's qargs from the members and orders it by these
    // maps, so mapping every wire to its own index makes it agree with `frame` by construction.
    let qubit_pos: HashMap<Qubit, usize> = (0..dag.num_qubits())
        .map(|index| (Qubit(index as u32), index))
        .collect();
    let clbit_pos: HashMap<Clbit, usize> = (0..dag.num_clbits())
        .map(|index| (Clbit(index as u32), index))
        .collect();

    let block = dag.add_block(body);
    // `cycle_check` asserts what the frontier rule already established — that nothing lies between the
    // members — so a refusal here is an inconsistency rather than a merge to skip, and is reported.
    dag.replace_block(
        &group.members,
        op,
        Some(Box::new(Parameters::Blocks(vec![block]))),
        None,
        true,
        &qubit_pos,
        &clbit_pos,
    )
    .into_py_result()?;
    Ok(())
}

/// The clbits the contracted node will cover, in its frame.
///
/// Collectors carry no classical wires today, so this is empty in practice; it is derived rather than
/// assumed so that the body's width keeps matching the node's if they ever do.
fn merged_clbits(dag: &DAGCircuit, group: &Group) -> Vec<Clbit> {
    let mut clbits: Vec<Clbit> = group
        .members
        .iter()
        .flat_map(|node| {
            let inst = dag.dag()[*node].unwrap_operation();
            dag.cargs_interner().get(inst.clbits).to_vec()
        })
        .collect();
    clbits.sort_unstable();
    clbits.dedup();
    clbits
}

/// Concatenate the members' bodies into one, remapped from each member's frame into the merged frame.
///
/// Members are visited in composition order, the same order their items were concatenated in, so the
/// `Gates` counts keep pointing at the right instructions.
fn merged_body(
    dag: &DAGCircuit,
    group: &Group,
    frame: &[Qubit],
    num_clbits: usize,
) -> PyResult<DAGCircuit> {
    let mut body = new_dag_body(frame.len(), num_clbits, group.items.len())?.into_builder();

    for node in &group.members {
        let inst = dag.dag()[*node].unwrap_operation();
        let member: Vec<Qubit> = dag.qargs_interner().get(inst.qubits).to_vec();
        let [block] = *inst.blocks_view() else {
            // A collector that has not been through the absorb pass has no body to contribute.
            continue;
        };
        let contribution = &dag.blocks()[block];

        // In written order, not topological order: a collector body is a *sequence*, since the items'
        // `Gates` counts index into it, and a topological read of a body of single-qubit gates would
        // group them by wire instead. A body is built once and never edited, so its node indices are
        // in the order they were appended.
        for (_, gate) in contribution.op_nodes(true) {
            let qargs: Vec<Qubit> = contribution
                .qargs_interner()
                .get(gate.qubits)
                .iter()
                .map(|local| {
                    // The member's span is part of the group's, so its wires are all in the frame.
                    let global = member[local.index()];
                    let merged = frame
                        .iter()
                        .position(|q| *q == global)
                        .expect("a member's qubits are part of the group's span");
                    Qubit(merged as u32)
                })
                .collect();
            super::utils::append(&mut body, gate.op.clone(), params_of(gate), &qargs, &[])?;
        }
    }
    Ok(body.build())
}

/// The collector operation carrying the merged descriptors.
fn merged_op(
    py: Python,
    group: &Group,
    num_qubits: usize,
    num_clbits: usize,
) -> PyResult<PackedOperation> {
    let spec = CollectSpec {
        items: group.items.clone(),
        owned: group.owned.clone(),
        partition: group.partition.clone(),
        parts: group.parts.clone(),
    };
    let annotation = Py::new(py, (Collect::new_from_spec(spec), PyAnnotation))?;
    Ok(PackedOperation::from_control_flow(Box::new(
        ControlFlowInstruction {
            control_flow: ControlFlow::Box {
                duration: None,
                annotations: vec![annotation.into_any()],
            },
            num_qubits: num_qubits as u32,
            num_clbits: num_clbits as u32,
        },
    )))
}
