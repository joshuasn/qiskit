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
//! Two ways for two collectors to become one, and they need different machinery.
//!
//! **Siblings, in one scope.** Adjacent boxes that share a synthesizer come to share a *middle*
//! collector, so N boxes in a row need N+1 collector layers rather than 2N. One
//! [`DAGCircuit::replace_block`] contraction per group, which takes the union of its members' qargs
//! and derives its position from their edges. Two rules govern what may fuse, both per qubit: a
//! candidate must overlap a group's `span`, and every one of its qubits must still be at that group's
//! `frontier`. Each scope is walked with its own state.
//!
//! **A collector leaving its box**, folded into the one just outside it — one collector layer fewer per
//! nesting level. This is *not* a contraction: a nested collector covers a subset of its box's qubits,
//! which are a subset of the enclosing collector's, so nothing is ever widened. `replace_block` cannot
//! contract across DAGs anyway. It is a body transfer plus a node deletion, and it runs to a fixed
//! point before the sibling sweep, since one collector escaping exposes the next level down.
//!
//! An escape refuses on hazards, not on which pass has run. Before absorption every collector body is
//! empty, so an escape is purely structural then — delete a collector, re-point whatever it was
//! catching — and an unabsorbed circuit is a valid IR2 rather than an unfinished one. What geometry
//! *does* differ by pass order is adjacency: absorption removes content from between two collectors, so
//! an escape it refuses before may be available after.

use hashbrown::{HashMap, HashSet};
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use pyo3::prelude::*;
use qiskit_circuit::dag_circuit::{DAGCircuit, DAGCircuitBuilder};
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::{Clbit, Qubit};

use super::utils::{
    IntoPyResult, Site, WireCursor, block_body, collect_annotation, collect_op, emission_spec,
    is_box, is_collector, lift_wires, new_dag_body, params_of, scope_dag, scope_dag_mut,
};
use crate::annotated_circuit::SynthesizerType;
use crate::emission_circuit::{CollectPart, CollectSpec};
use crate::partition::Partition;
use crate::sampling_graph::Direction;

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
pub fn py_merge_collectors(dag: &mut DAGCircuit) -> PyResult<()> {
    merge_collectors(dag)
}

/// Merge adjacent collectors throughout an emission circuit, in place.
pub fn merge_collectors(dag: &mut DAGCircuit) -> PyResult<()> {
    // Escapes first, to a fixed point: taking one collector out of its box can leave the next level
    // down as its box's new head. Each round re-plans against the rewritten circuit rather than reusing
    // sites, and rounds are bounded by nesting depth.
    while escape_round(dag)? {}
    merge_scope(dag)
}

// --- Escape: a collector leaving its box --------------------------------------------------------

/// One collector folding into the collector just outside its box.
struct Escape {
    /// The collector that stays, and gains a body. Its width and spec do not change: its partition
    /// already covers the qubits of anything nested inside its box.
    outer: Site,
    /// The collector that goes.
    inner: Site,
    /// Which way `outer` had to walk to reach `inner`, which is what puts the two bodies in circuit
    /// order: rightward means `outer`'s content comes first.
    direction: Direction,
}

/// Do one round of escapes, reporting whether anything moved.
fn escape_round(dag: &mut DAGCircuit) -> PyResult<bool> {
    let plans = plan_escapes(dag)?;
    for plan in &plans {
        escape(dag, plan)?;
    }
    Ok(!plans.is_empty())
}

/// Find every escape available in the circuit as it stands.
///
/// A collector takes part in at most one escape per round, in either role: a middle collector in a
/// two-level nest is both somebody's `inner` and somebody else's `outer`, and doing both at once would
/// substitute a node this round also deletes. The fixed-point loop picks up the rest.
fn plan_escapes(root: &DAGCircuit) -> PyResult<Vec<Escape>> {
    let mut plans: Vec<Escape> = Vec::new();
    let mut claimed: HashSet<Site> = HashSet::new();
    plan_escapes_in(root, &mut Vec::new(), &mut plans, &mut claimed)?;
    Ok(plans)
}

fn plan_escapes_in(
    root: &DAGCircuit,
    path: &mut Vec<NodeIndex>,
    plans: &mut Vec<Escape>,
    claimed: &mut HashSet<Site>,
) -> PyResult<()> {
    let nodes: Vec<NodeIndex> = scope_dag(root, path)?.topological_op_nodes(false).collect();

    for node in &nodes {
        let dag = scope_dag(root, path)?;
        let inst = dag.dag()[*node].unwrap_operation();
        if !is_collector(inst) {
            continue;
        }
        let outer = Site {
            scope: path.clone(),
            node: *node,
        };
        if claimed.contains(&outer) {
            continue;
        }
        for direction in [Direction::Left, Direction::Right] {
            let Some(inner) = escapable(root, &outer, direction)? else {
                continue;
            };
            if claimed.contains(&inner) {
                continue;
            }
            claimed.insert(outer.clone());
            claimed.insert(inner.clone());
            plans.push(Escape {
                outer: outer.clone(),
                inner,
                direction,
            });
            break;
        }
    }

    // Recurse, so a collector two levels down is considered against the collector one level down.
    for node in &nodes {
        let dag = scope_dag(root, path)?;
        let inst = dag.dag()[*node].unwrap_operation();
        if !is_box(inst) || is_collector(inst) {
            continue;
        }
        path.push(*node);
        plan_escapes_in(root, path, plans, claimed)?;
        path.pop();
    }
    Ok(())
}

/// The collector that may leave its box and fold into `outer`, walking `direction`, if any.
fn escapable(root: &DAGCircuit, outer: &Site, direction: Direction) -> PyResult<Option<Site>> {
    let dag = scope_dag(root, &outer.scope)?;
    let inst = dag.dag()[outer.node].unwrap_operation();
    let spec = collect_annotation(inst).expect("only asked of a collector");
    let wires: Vec<Qubit> = dag.qargs_interner().get(inst.qubits).to_vec();

    // Every distinct collector-inside-a-box any of `outer`'s wires reaches first. More than one is
    // ordinary — two narrow boxes side by side inside one content box — so each is judged on its own
    // and the first that qualifies wins; a later round can take the others.
    let mut candidates: Vec<Site> = Vec::new();
    for wire in &wires {
        let Some(site) = first_site(root, outer, *wire, direction)? else {
            continue;
        };
        // A collector in the same scope is a sibling, which the contraction sweep handles.
        if site.scope.len() <= outer.scope.len() {
            continue;
        }
        let reached = scope_dag(root, &site.scope)?.dag()[site.node].unwrap_operation();
        if !is_collector(reached) {
            continue;
        }
        if !candidates.contains(&site) {
            candidates.push(site);
        }
    }

    for inner in candidates {
        let inner_dag = scope_dag(root, &inner.scope)?;
        let inner_inst = inner_dag.dag()[inner.node].unwrap_operation();
        let inner_spec = collect_annotation(inner_inst).expect("checked above");
        // Every part of both, not just the first: the two are about to become one layer, and a part
        // that synthesizes differently could not be expressed by it. Same test `find_mergeable` makes.
        let synthesizer = spec.synthesizer();
        if !inner_spec
            .parts
            .iter()
            .chain(spec.parts.iter())
            .all(|part| part.synthesizer == synthesizer)
        {
            continue;
        }
        let inner_wires = lift_to(root, &outer.scope, &inner, inner_dag, inner_inst)?;
        // Always true for a nested collector, since its box sits inside `outer`'s — checked because
        // the transfer would silently drop content if it were not.
        if !inner_wires.iter().all(|wire| wires.contains(wire)) {
            continue;
        }
        // P1: nothing between them, on every wire the inner collector covers. Wires of `outer` beyond
        // it may reach anything at all; they carry none of the content being moved.
        let mut adjacent = true;
        for wire in &inner_wires {
            if first_site(root, outer, *wire, direction)? != Some(inner.clone()) {
                adjacent = false;
                break;
            }
        }
        if !adjacent {
            continue;
        }
        // P2, as two hazards rather than one blanket refusal. `outward` is the way an emission would
        // have to travel to end up at `outer`, and after the transfer every such emission does, since
        // the collector that was catching them is gone.
        let outward = match direction {
            Direction::Right => Direction::Left,
            Direction::Left => Direction::Right,
        };
        let arriving = count_emissions_towards(scope_dag(root, &inner.scope)?, outward)?;
        if arriving > 0 && !body_is_empty(inner_dag, inner_inst)? {
            // The content being moved sits on such an emission's path: crossed today, with the inner
            // collector foreign to it, and composed inside its target afterwards.
            continue;
        }
        if arriving > 1 {
            // Two would arrive at `outer` from the same side, and which composes nearer its edge is a
            // question only the incoming-placement rule answers.
            continue;
        }
        return Ok(Some(inner));
    }
    Ok(None)
}

/// The first site one of a collector's wires reaches, descending into boxes but not into collectors.
fn first_site(
    root: &DAGCircuit,
    from: &Site,
    wire: Qubit,
    direction: Direction,
) -> PyResult<Option<Site>> {
    let descend = |inst: &PackedInstruction| is_box(inst) && !is_collector(inst);
    let mut cursor = WireCursor::new(from.scope.clone(), from.node, wire);
    cursor.advance(root, direction, &descend)
}

/// How many emissions in this body, at any depth, are still travelling in `direction`.
///
/// These are the ones that end up at the collector being folded into, since the transfer deletes the
/// collector that was catching them. The count is what separates an escape's two hazards from each
/// other: one of them needs any such emission plus content to move, the other needs two of them.
///
/// A local emission is not counted — it has resolved in place and travels nowhere.
fn count_emissions_towards(dag: &DAGCircuit, direction: Direction) -> PyResult<usize> {
    let mut total = 0;
    for (_, inst) in dag.op_nodes(true) {
        if let Some(spec) = emission_spec(inst) {
            if spec.direction == Some(direction) {
                total += 1;
            }
            continue;
        }
        if let Some(body) = block_body(dag, inst)? {
            total += count_emissions_towards(body, direction)?;
        }
    }
    Ok(total)
}

/// Whether a collector has nothing in its body to move.
///
/// Before absorption every body is empty, which is why an escape pre-absorption is purely structural:
/// it deletes a collector and re-points whatever that collector was catching, moving no content at all.
fn body_is_empty(dag: &DAGCircuit, inst: &PackedInstruction) -> PyResult<bool> {
    Ok(block_body(dag, inst)?.is_none_or(|body| body.num_ops() == 0))
}

/// A nested collector's qargs, lifted out of the boxes between it and `base` into that frame.
fn lift_to(
    root: &DAGCircuit,
    base: &[NodeIndex],
    site: &Site,
    site_dag: &DAGCircuit,
    inst: &PackedInstruction,
) -> PyResult<Vec<Qubit>> {
    let local = site_dag.qargs_interner().get(inst.qubits).to_vec();
    lift_wires(scope_dag(root, base)?, &site.scope[base.len()..], &local)
}

/// Move one collector's body into the collector outside its box, and delete it.
fn escape(root: &mut DAGCircuit, plan: &Escape) -> PyResult<()> {
    // Everything is read before anything is written: the merged body is built while both collectors
    // are still in place.
    let (op, body) = {
        let outer_dag = scope_dag(root, &plan.outer.scope)?;
        let outer_inst = outer_dag.dag()[plan.outer.node].unwrap_operation();
        let frame: Vec<Qubit> = outer_dag.qargs_interner().get(outer_inst.qubits).to_vec();
        let num_clbits = outer_dag.cargs_interner().get(outer_inst.clbits).len();
        let spec = collect_annotation(outer_inst).expect("planned from a collector");
        let outer_body = block_body(outer_dag, outer_inst)?;

        let inner_dag = scope_dag(root, &plan.inner.scope)?;
        let inner_inst = inner_dag.dag()[plan.inner.node].unwrap_operation();
        let inner_wires = lift_to(root, &plan.outer.scope, &plan.inner, inner_dag, inner_inst)?;
        let inner_body = block_body(inner_dag, inner_inst)?;

        let capacity =
            outer_body.map_or(0, |b| b.num_ops()) + inner_body.map_or(0, |b| b.num_ops());
        let mut body = new_dag_body(frame.len(), num_clbits, capacity)?.into_builder();
        // Circuit order, which is what keeps the composition order of the two contributions right: the
        // outer collector's content comes first exactly when it sits first.
        let outer_first = matches!(plan.direction, Direction::Right);
        for (contribution, wires) in order_contributions(
            (outer_body, &frame),
            (inner_body, &inner_wires),
            outer_first,
        ) {
            if let Some(contribution) = contribution {
                append_contribution(&mut body, contribution, wires, &frame)?;
            }
        }
        let op = collect_op(spec, frame.len(), num_clbits);
        (op, body.build())
    };

    let outer_scope = scope_dag_mut(root, &plan.outer.scope)?;
    let block = outer_scope.add_block(body);
    outer_scope
        .substitute_op(
            plan.outer.node,
            op,
            Some(Parameters::Blocks(vec![block])),
            None,
        )
        .into_py_result()?;
    scope_dag_mut(root, &plan.inner.scope)?.remove_op_node(plan.inner.node);
    Ok(())
}

/// The two contributions in the order they compose.
fn order_contributions<'a>(
    outer: (Option<&'a DAGCircuit>, &'a [Qubit]),
    inner: (Option<&'a DAGCircuit>, &'a [Qubit]),
    outer_first: bool,
) -> [(Option<&'a DAGCircuit>, &'a [Qubit]); 2] {
    if outer_first {
        [outer, inner]
    } else {
        [inner, outer]
    }
}

/// Merge collectors within one scope, then recurse into box bodies with fresh state.
fn merge_scope(dag: &mut DAGCircuit) -> PyResult<()> {
    let all: Vec<Qubit> = (0..dag.num_qubits() as u32).map(Qubit).collect();
    let mut groups: Vec<Group> = Vec::new();
    let mut bodies: Vec<qiskit_circuit::Block> = Vec::new();

    // Grouping reads the DAG and contracting mutates it, so the sweep records node indices and the
    // rewriting happens afterwards; `StableDiGraph` keeps them valid in between.
    for node in dag.topological_op_nodes(false).collect::<Vec<_>>() {
        let inst = dag.dag()[node].unwrap_operation();
        let qubits: Vec<Qubit> = dag.qargs_interner().get(inst.qubits).to_vec();

        if let Some(spec) = collect_annotation(inst) {
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
            fuse(dag, group)?;
        }
    }

    // Recurse with fresh state, so a nested scope's collectors merge among themselves but never
    // across the boundary.
    for block in bodies {
        merge_scope(dag.view_block_mut(block))?;
    }
    Ok(())
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
fn fuse(dag: &mut DAGCircuit, group: &Group) -> PyResult<()> {
    let frame = group.frame();
    let clbits = merged_clbits(dag, group);
    let body = merged_body(dag, group, &frame, clbits.len())?;
    let op = merged_op(group, frame.len(), clbits.len());

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
        append_contribution(&mut body, &dag.blocks()[block], &member, frame)?;
    }
    Ok(body.build())
}

/// Append one collector's body to a merged body, remapping its wires into `frame`.
///
/// `wires` maps the contribution's body-local wires into the frame `frame` is expressed in: a sibling's
/// own qargs, or an escaped collector's qargs lifted out through the boxes between.
///
/// In written order, not topological order: a collector body is a *sequence*. A body is built once and
/// never edited, so its node indices are already in the order they were appended.
fn append_contribution(
    out: &mut DAGCircuitBuilder,
    contribution: &DAGCircuit,
    wires: &[Qubit],
    frame: &[Qubit],
) -> PyResult<()> {
    for (_, gate) in contribution.op_nodes(true) {
        let qargs: Vec<Qubit> = contribution
            .qargs_interner()
            .get(gate.qubits)
            .iter()
            .map(|local| {
                let outer = wires[local.index()];
                let merged = frame
                    .iter()
                    .position(|q| *q == outer)
                    .expect("a contribution's qubits are part of the merged frame");
                Qubit(merged as u32)
            })
            .collect();
        super::utils::append(out, gate.op.clone(), params_of(gate), &qargs, &[])?;
    }
    Ok(())
}

/// The collector operation carrying the merged descriptors.
fn merged_op(group: &Group, num_qubits: usize, num_clbits: usize) -> PackedOperation {
    let partition = group.partition();
    // `find_mergeable` has established that every member shares one synthesizer, so the merged
    // descriptors are that one replicated across however many parts the span came out as.
    let synthesizer = group.parts[0].synthesizer;
    let parts = (0..partition.len())
        .map(|_| CollectPart { synthesizer })
        .collect();
    let spec = CollectSpec { partition, parts };
    collect_op(spec, num_qubits, num_clbits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotated_circuit::{DistributionType, Dressing, TwirlSpec};
    use qiskit_circuit::annotation::Annotation;
    use qiskit_circuit::operations::StandardGate;
    use std::sync::Arc;

    use super::super::build::build;
    use super::super::utils::{append, is_box, new_dag_body, write_box};

    fn twirl() -> Arc<dyn Annotation> {
        Arc::new(TwirlSpec {
            distribution: DistributionType::UniformPauli,
            dressing: Dressing::Left,
            decomposition: SynthesizerType::RzSx,
        })
    }

    /// A one-qubit annotated circuit of `count` twirled boxes in a row, each holding one gate.
    fn twirled_chain(count: usize) -> DAGCircuit {
        Python::initialize();
        let mut out = new_dag_body(1, 0, count * 2).unwrap().into_builder();
        for _ in 0..count {
            let mut body = new_dag_body(1, 0, 1).unwrap().into_builder();
            append(&mut body, StandardGate::H.into(), None, &[Qubit(0)], &[]).unwrap();
            write_box(
                &mut out,
                body.build(),
                vec![twirl()],
                None,
                &[Qubit(0)],
                &[],
            )
            .unwrap();
        }
        out.build()
    }

    fn collector_count(dag: &DAGCircuit) -> usize {
        dag.topological_op_nodes(false)
            .map(|node| dag.dag()[node].unwrap_operation())
            .filter(|inst| is_collector(inst))
            .count()
    }

    fn box_count(dag: &DAGCircuit) -> usize {
        dag.topological_op_nodes(false)
            .map(|node| dag.dag()[node].unwrap_operation())
            .filter(|inst| is_box(inst))
            .count()
    }

    #[test]
    fn test_merge_joins_adjacent_collectors_sharing_synthesizer() {
        // `build` is deliberately local: it writes a collector on each edge of each box and never looks
        // sideways. Two twirled boxes in a row therefore meet back to back with two collectors between
        // them, and joining that pair into one is this pass's whole reason to exist.
        let (mut dag, _table) = build(&twirled_chain(2)).unwrap();
        assert_eq!(collector_count(&dag), 4, "build collects on all four edges");
        assert_eq!(box_count(&dag), 6, "two content boxes among them");

        merge_collectors(&mut dag).unwrap();

        assert_eq!(
            collector_count(&dag),
            3,
            "the two collectors that met in the middle fuse into one"
        );
        assert_eq!(
            box_count(&dag),
            5,
            "the content boxes are untouched: merging joins collectors, not content"
        );
    }

    #[test]
    fn test_merge_leaves_a_lone_pair_alone() {
        // Nothing to fuse, so nothing may change. Guards against a merge that widens or drops a
        // collector just because it walked past one.
        let (mut dag, _table) = build(&twirled_chain(1)).unwrap();
        let before = (collector_count(&dag), box_count(&dag));
        merge_collectors(&mut dag).unwrap();
        assert_eq!((collector_count(&dag), box_count(&dag)), before);
    }
}
