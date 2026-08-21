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
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{Clbit, Qubit};

use crate::annotated_circuit::SynthesizerType;
use crate::emission_circuit::{Collect, CollectPart};
use crate::emission_circuit_navigation::{
    EmissionTally, ScopeOrder, Site, WireCursor, append_instruction, collect_annotation,
    collect_op, collectors, is_box, new_dag_body, scope_dag,
};
use crate::error::Result;
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
    Ok(merge_collectors(dag)?)
}

/// Merge adjacent collectors throughout an emission circuit, in place.
pub fn merge_collectors(dag: &mut DAGCircuit) -> Result<()> {
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
fn escape_round(dag: &mut DAGCircuit) -> Result<bool> {
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
fn plan_escapes(root: &DAGCircuit) -> Result<Vec<Escape>> {
    let mut plans: Vec<Escape> = Vec::new();
    let mut claimed: HashSet<Site> = HashSet::new();

    // `ScopeOrder::Outermost` is what pairs the levels up correctly: a collector is judged before the
    // collectors nested below it, so a two-level nest resolves from the outside in.
    for outer in collectors(root, ScopeOrder::Outermost)? {
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
    Ok(plans)
}

/// The collector that may leave its box and fold into `outer`, walking `direction`, if any.
fn escapable(root: &DAGCircuit, outer: &Site, direction: Direction) -> Result<Option<Site>> {
    let spec = outer.collector(root)?.expect("only asked of a collector");
    let wires: Vec<Qubit> = outer.qubits(root)?;

    // Every distinct collector-inside-a-box any of `outer`'s wires reaches first. More than one is
    // ordinary — two narrow boxes side by side inside one content box — so each is judged on its own
    // and the first that qualifies wins; a later round can take the others.
    let mut candidates: Vec<Site> = Vec::new();
    for wire in &wires {
        let Some(site) = first_site(root, outer, *wire, direction)? else {
            continue;
        };
        // A collector in the same scope is a sibling, which the contraction sweep handles.
        if !site.deeper_than(&outer.scope) {
            continue;
        }
        if site.collector(root)?.is_none() {
            continue;
        }
        if !candidates.contains(&site) {
            candidates.push(site);
        }
    }

    for inner in candidates {
        let inner_spec = inner.collector(root)?.expect("checked above");
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
        let inner_wires = inner.qubits_in(root, &outer.scope)?;
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
        // P2. `outward` is the way an emission would have to travel to end up at `outer`, and after the
        // transfer every such emission does, since the collector that was catching them is gone.
        let outward = match direction {
            Direction::Right => Direction::Left,
            Direction::Left => Direction::Right,
        };
        let arriving = EmissionTally::subtree(scope_dag(root, &inner.scope)?)?;
        if !hazards_clear(&arriving, outward, !body_is_empty(root, &inner)?) {
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
) -> Result<Option<Site>> {
    WireCursor::at(from, wire).advance(root, direction)
}

/// Whether an escape's two hazards are clear, given what the inner collector's box holds.
///
/// P2 as two hazards rather than one blanket refusal. `arriving` is a tally of the box the escaping
/// collector is leaving, and `outward` the way an emission has to travel to end up at the collector
/// being folded into — every one of those does end up there after the transfer, since the collector
/// that was catching them is gone. `moves_content` says whether the escaping collector has a body to
/// take with it.
///
/// - Content plus even one arriving emission refuses: the content being moved sits on that emission's
///   path — crossed today, with the escaping collector foreign to it, and composed inside its target
///   afterwards.
/// - Two arriving refuses whether anything moves or not: both would reach the same collector from the
///   same side, and which composes nearer its edge is a question only the incoming-placement rule
///   answers.
///
/// A tally rather than a body, so the hazards are decided on plain data. Which emissions are in the
/// tally is [`EmissionTally::subtree`]'s business, and it counts at any depth: an emission nested two
/// boxes down still arrives at `outer`, and a local one arrives nowhere.
fn hazards_clear(arriving: &EmissionTally, outward: Direction, moves_content: bool) -> bool {
    let arriving = arriving.towards(outward);
    if arriving > 0 && moves_content {
        return false;
    }
    arriving <= 1
}

/// Whether a collector has nothing in its body to move.
///
/// Before absorption every body is empty, which is why an escape pre-absorption is purely structural:
/// it deletes a collector and re-points whatever that collector was catching, moving no content at all.
fn body_is_empty(root: &DAGCircuit, site: &Site) -> Result<bool> {
    Ok(site.body(root)?.is_none_or(|body| body.num_ops() == 0))
}

/// Move one collector's body into the collector outside its box, and delete it.
fn escape(root: &mut DAGCircuit, plan: &Escape) -> Result<()> {
    // Everything is read before anything is written: the merged body is built while both collectors
    // are still in place.
    let (op, body) = {
        let frame: Vec<Qubit> = plan.outer.qubits(root)?;
        let num_clbits = plan.outer.num_clbits(root)?;
        let spec = plan
            .outer
            .collector(root)?
            .expect("planned from a collector");
        let outer_body = plan.outer.body(root)?;

        let inner_wires = plan.inner.qubits_in(root, &plan.outer.scope)?;
        let inner_body = plan.inner.body(root)?;

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

    plan.outer.substitute(root, op, body)?;
    plan.inner.remove(root)?;
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
fn merge_scope(dag: &mut DAGCircuit) -> Result<()> {
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
fn join(group: &mut Group, node: NodeIndex, spec: &Collect, qubits: &[Qubit]) {
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
fn fuse(dag: &mut DAGCircuit, group: &Group) -> Result<()> {
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
    )?;
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
) -> Result<DAGCircuit> {
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
) -> Result<()> {
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
        append_instruction(out, gate, &qargs, &[])?;
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
    let spec = Collect { partition, parts };
    collect_op(spec, num_qubits, num_clbits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotated_circuit::{DistributionType, Dressing, Twirl};
    use qiskit_circuit::annotation::Annotation;
    use qiskit_circuit::operations::StandardGate;
    use std::sync::Arc;

    use super::super::build::build;
    use crate::emission_circuit_navigation::{
        Sighting, append, is_collector, new_dag_body, write_box,
    };

    fn twirl() -> Arc<dyn Annotation> {
        Arc::new(Twirl {
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

    /// The tally of a box holding `count` emissions travelling `outward`, all of which arrive at the
    /// collector an escape would fold into.
    ///
    /// Where in the box they were written does not appear here, which is the point of taking a tally:
    /// [`EmissionTally::subtree`] counts them at any depth, and the hazards below cannot tell — nor
    /// should they, since an emission two boxes down arrives just the same.
    fn arriving(count: usize, outward: Direction) -> EmissionTally {
        (0..count)
            .map(|_| Sighting::Emission(Some(outward)))
            .collect()
    }

    #[test]
    fn test_an_escape_with_content_to_move_refuses_any_arriving_emission() {
        assert!(!hazards_clear(
            &arriving(1, Direction::Left),
            Direction::Left,
            true
        ));
        assert!(
            hazards_clear(&arriving(1, Direction::Left), Direction::Left, false),
            "an unabsorbed collector moves nothing, so its escape is purely structural"
        );
    }

    #[test]
    fn test_an_escape_refuses_two_arriving_emissions_whatever_moves() {
        for moves_content in [false, true] {
            assert!(!hazards_clear(
                &arriving(2, Direction::Left),
                Direction::Left,
                moves_content
            ));
        }
    }

    #[test]
    fn test_only_the_emissions_that_would_arrive_are_hazards() {
        // Travelling the other way, or resolved in place: neither ends up at the collector being folded
        // into, so neither refuses the escape even with a body to move.
        let elsewhere: EmissionTally = [
            Sighting::Emission(Some(Direction::Right)),
            Sighting::Emission(Some(Direction::Right)),
            Sighting::Emission(None),
        ]
        .into_iter()
        .collect();
        assert!(hazards_clear(&elsewhere, Direction::Left, true));
    }
}
