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
//! Adjacent boxes that share a synthesizer come to share a *middle* collector, so N boxes in a row
//! need N+1 dressing layers rather than 2N. One [`DAGCircuit::replace_block`] contraction per
//! group, which takes the union of its members' qargs and derives its position from their edges.
//!
//! Two rules govern what may fuse, both per qubit: a candidate must overlap a group's `span`, and
//! every one of its qubits must still be at that group's `frontier`. **Siblings only** — promoting
//! a collector out of its box is declined, and each scope is walked with its own state.

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
use crate::emission_circuit::{Collect, CollectPart, CollectSpec};
use crate::partition::Partition;

/// Collectors that will fuse into one, and the state deciding what else may join them.
struct Group {
    /// The collector nodes to contract, in the order their contributions compose. A group of one is
    /// left alone.
    members: Vec<NodeIndex>,
    /// Qubits on which nothing has happened since this group's position, so a later collector on
    /// them can still commute back here and fuse.
    ///
    /// Initialised to every qubit in the scope, not to `span`, because a merge may widen onto
    /// qubits no member covered. A group is dead once this no longer intersects `span`.
    frontier: HashSet<Qubit>,
    /// Every qubit this group covers, monotonically: the width the contracted box will have.
    span: HashSet<Qubit>,
    /// The members' subsystems, as scope-frame qubits, in the order they were contributed.
    ///
    /// Qubits rather than indices because the members index into their own qargs, which is a
    /// different frame each; [`partition`](Self::partition) puts them back into the contracted
    /// node's.
    subsystems: Vec<Vec<Qubit>>,
    /// The descriptors the members share, taken from the first of them. [`find_mergeable`] admits a
    /// member only if it agrees on the synthesizer, which is all these carry.
    parts: Vec<CollectPart>,
    /// Every annotated box whose emissions the contracted collector may consume: the union over the
    /// members.
    owned: Vec<u32>,
}

impl Group {
    /// The group's qubits in the frame the contracted node will use.
    ///
    /// Sorted, to agree with the position map [`fuse`] hands [`DAGCircuit::replace_block`].
    fn frame(&self) -> Vec<Qubit> {
        let mut qubits: Vec<Qubit> = self.span.iter().copied().collect();
        qubits.sort_unstable();
        qubits
    }

    /// How the contracted collector groups its span, as indices into [`frame`](Self::frame).
    ///
    /// Members can overlap, and two subsystems sharing a qubit cannot be told apart afterwards —
    /// whatever samples them jointly is one draw — so overlapping subsystems coarsen into one part.
    /// Parts come out in ascending order of their lowest index, so the result does not depend on the
    /// order the members were visited in.
    fn partition(&self) -> Partition {
        let frame = self.frame();
        let position: HashMap<Qubit, usize> = frame
            .iter()
            .enumerate()
            .map(|(index, qubit)| (*qubit, index))
            .collect();

        // Union-find over frame positions, each set rooted at its lowest member.
        fn find(root: &mut [usize], mut index: usize) -> usize {
            while root[index] != index {
                root[index] = root[root[index]];
                index = root[index];
            }
            index
        }
        let mut root: Vec<usize> = (0..frame.len()).collect();
        for subsystem in &self.subsystems {
            let mut members = subsystem.iter().map(|qubit| position[qubit]);
            let Some(first) = members.next() else {
                continue;
            };
            for member in members {
                let (left, right) = (find(&mut root, first), find(&mut root, member));
                root[left.max(right)] = left.min(right);
            }
        }

        let mut part_of: HashMap<usize, usize> = HashMap::new();
        let mut parts: Vec<Vec<usize>> = Vec::new();
        for index in 0..frame.len() {
            let set = find(&mut root, index);
            let part = *part_of.entry(set).or_insert_with(|| {
                parts.push(Vec::new());
                parts.len() - 1
            });
            parts[part].push(index);
        }
        Partition::new(parts.into_iter().map(Vec::into_boxed_slice))
            .expect("every index of the frame lands in exactly one part")
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
    // rewriting happens afterwards; `StableDiGraph` keeps them valid in between.
    for node in dag.topological_op_nodes(false).collect::<Vec<_>>() {
        let inst = dag.dag()[node].unwrap_operation();
        let qubits: Vec<Qubit> = dag.qargs_interner().get(inst.qubits).to_vec();

        if let Some(spec) = collect_annotation(py, inst) {
            match find_mergeable(&groups, &qubits, spec.synthesizer()) {
                // Fuse into the open group, which keeps its position and gains this collector's
                // content and qubits. Nothing is released: a merged contribution has no position of
                // its own. Bodies append in encounter order, which is what makes A's outermost
                // element end up adjacent to B's outermost.
                Some(index) => join(&mut groups[index], node, &spec, &qubits),
                None => {
                    // Nothing compatible is open on these qubits, so this collector gets a position
                    // of its own and becomes a synth layer, blocking later collectors on these
                    // wires.
                    release(&mut groups, &qubits);
                    groups.push(Group {
                        members: vec![node],
                        frontier: all.iter().copied().collect(),
                        span: qubits.iter().copied().collect(),
                        subsystems: spec.partition.groups(&qubits),
                        parts: spec.parts.clone(),
                        owned: spec.owned.clone(),
                    });
                }
            }
            continue;
        }

        // An emission is a twirl point and real content ends absorption; either way these wires
        // leave every open group's frontier, so there is no distinction left to draw here.
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
            // A shared qubit is required, not merely usual: two collectors on disjoint qubits are
            // concurrent, so their relative position is an artifact of the topological order the
            // walk took, and fusing them would make the output depend on an arbitrary choice.
            && qubits.iter().any(|q| candidate.span.contains(q))
            // *Every* qubit must still be at the frontier, not just the shared one. This also
            // excludes a dead group without a flag, since a match needs a qubit in both `span` and
            // `frontier`.
            && qubits.iter().all(|q| candidate.frontier.contains(q))
    })
}

/// Add a collector's contribution to an open group.
fn join(group: &mut Group, node: NodeIndex, spec: &CollectSpec, qubits: &[Qubit]) {
    group.members.push(node);
    group.span.extend(qubits.iter().copied());
    // Sorted and deduplicated, so the set does not depend on the order members were visited in.
    group.owned.extend_from_slice(&spec.owned);
    group.owned.sort_unstable();
    group.owned.dedup();
    group.subsystems.extend(spec.partition.groups(qubits));
}

/// Take `qubits` off every open group's frontier, so a later collector on them can no longer fuse.
///
/// The only closing rule: real content, emissions and independently-positioned collectors all do
/// this.
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

    // `replace_block` orders the contracted node's qargs by these maps, so mapping every wire to
    // its own index makes it agree with `frame` by construction.
    let qubit_pos: HashMap<Qubit, usize> = (0..dag.num_qubits())
        .map(|index| (Qubit(index as u32), index))
        .collect();
    let clbit_pos: HashMap<Clbit, usize> = (0..dag.num_clbits())
        .map(|index| (Clbit(index as u32), index))
        .collect();

    let block = dag.add_block(body);
    // `cycle_check` asserts what the frontier rule already established, so a refusal here is an
    // inconsistency rather than a merge to skip, and is reported.
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
/// Empty in practice — collectors carry no classical wires — but derived rather than assumed, so
/// the body's width keeps matching the node's if they ever do.
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

/// Concatenate the members' bodies into one, remapped from each member's frame into the merged
/// frame.
///
/// Members are visited in composition order, so the result's composition order matches theirs.
fn merged_body(
    dag: &DAGCircuit,
    group: &Group,
    frame: &[Qubit],
    num_clbits: usize,
) -> PyResult<DAGCircuit> {
    let capacity: usize = group
        .members
        .iter()
        .map(|node| {
            let inst = dag.dag()[*node].unwrap_operation();
            match *inst.blocks_view() {
                [block] => dag.blocks()[block].op_nodes(true).count(),
                _ => 0,
            }
        })
        .sum();
    let mut body = new_dag_body(frame.len(), num_clbits, capacity)?.into_builder();

    for node in &group.members {
        let inst = dag.dag()[*node].unwrap_operation();
        let member: Vec<Qubit> = dag.qargs_interner().get(inst.qubits).to_vec();
        let [block] = *inst.blocks_view() else {
            // A collector that has not been through the absorb pass has no body to contribute.
            continue;
        };
        let contribution = &dag.blocks()[block];

        // In written order, not topological order: a collector body is a *sequence*. A body is
        // built once and never edited, so its node indices are already in the order they were
        // appended.
        for (_, gate) in contribution.op_nodes(true) {
            let qargs: Vec<Qubit> = contribution
                .qargs_interner()
                .get(gate.qubits)
                .iter()
                .map(|local| {
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
    let partition = group.partition();
    // `find_mergeable` has established that every member shares one synthesizer, so the merged
    // descriptors are that one replicated across however many parts the span came out as.
    let synthesizer = group.parts[0].synthesizer;
    let parts = (0..partition.len())
        .map(|_| CollectPart { synthesizer })
        .collect();
    let spec = CollectSpec {
        owned: group.owned.clone(),
        partition,
        parts,
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
