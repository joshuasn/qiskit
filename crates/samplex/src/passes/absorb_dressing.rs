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

//! Absorb dressing into collectors: emission circuit (IR2) → emission circuit (IR2), in place.
//!
//! Build leaves every emission and every gate where it was and every collector empty. This pass
//! walks outward from each collector along each of its own wires independently, writing what it
//! takes over into the collector's body:
//!
//! - A single-qubit standard gate *from inside its own box* → copied in as-is.
//! - An emission facing it, adjacent on *every* wire it covers → rewritten to `direction: None` and
//!   placed in the body.
//! - Anything else → that wire is done, and only that wire.
//!
//! **A dressing folds in only its own box's gates.** A gate in the collector's own scope belongs to
//! whatever encloses the box, so it stays where it is even when adjacent. That is a statement about
//! what a dressing *is* rather than a safety rule — an emission always arrives at a collector from its
//! content side, so a gate on the outward side is never on the path of one targeting it. It costs the
//! folding of single-qubit gates sitting on the spine beside a box: those stay in the template.
//!
//! What the resulting sequence guarantees is per-wire order, not circuit order; see
//! [`Collect::steps`].
//!
//! **A walk descends but never ascends.** It steps into a box it reaches and carries on along the
//! same wire inside, so a collector can fold in the gates at the edge of the content box beside it —
//! which is the only way they get folded in at all, since build no longer hoists them out. It will
//! climb back out of a box it descended into, but never above the scope it started in. That
//! asymmetry is load-bearing: a collector inside a box must not reach out and take an enclosing
//! box's emission, which would undo that box's randomization with none of its content in between.
//!
//! Two more things keep the walk honest, both by construction rather than by checking:
//!
//! - **An emission facing away is a barrier.** So the left collector's walk stops at the propagating
//!   emission sitting against the content box and can never descend past it, which is exactly the
//!   content it must not pull across.
//! - **Every box fences its own emissions**, because build writes its collector before them. So a
//!   descending collector meets an inner box's collector before any of that box's emissions, and can
//!   only ever take *gates* out of it.
//!
//! [`Collect::steps`]: crate::sampling_graph::Collect::steps

use hashbrown::{HashMap, HashSet};

use pyo3::prelude::*;
use qiskit_circuit::Qubit;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::operations::OperationRef;
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};

use crate::emission_circuit::{Collect, Emit};
use crate::emission_circuit_navigation::{
    ScopeOrder, Site, WireCursor, append, append_instruction, collect_op, collectors,
    emission_spec, new_dag_body, next_on_wire,
};
use crate::sampling_graph::Direction;

/// Absorb dressing into every collector, in place.
#[pyfunction]
#[pyo3(name = "absorb_dressing")]
pub fn py_absorb_dressing(dag: &mut DAGCircuit) -> PyResult<()> {
    absorb_dressing(dag)
}

/// Absorb dressing into every collector, in place.
///
/// Planning reads the whole circuit and rewriting mutates it, so the two are separate sweeps: every
/// plan is made against the original, and `StableDiGraph` keeps the sites carried between them valid.
pub fn absorb_dressing(dag: &mut DAGCircuit) -> PyResult<()> {
    let plans = plan_absorptions(dag)?;
    for plan in &plans {
        let body = build_body(dag, plan)?;
        let op = absorbed_op(dag, plan)?;
        plan.collector.substitute(dag, op, body)?;
    }
    // Kept as a sequence as well as a set, so that removal order is fixed rather than whatever the
    // set happens to iterate in.
    let consumed: Vec<Site> = plans
        .iter()
        .flat_map(|plan| plan.consumed.iter().cloned())
        .collect();
    for site in &consumed {
        site.remove(dag)?;
    }
    Ok(())
}

/// One thing a collector absorbed, in the order it will sit in the collector's body.
enum BodyOp {
    /// An absorbed gate, by site, with the wires it covers in the collector's frame.
    Gate(Site, Vec<Qubit>),
    /// A local emission, already rewritten to `direction: None`, with its wires in the collector's
    /// frame.
    Local(PackedOperation, Vec<Qubit>),
}

/// What one collector takes over from around it.
struct Absorption {
    collector: Site,
    /// The collector's existing descriptors, which absorption does not change.
    spec: Collect,
    /// Everything absorbed, in composition order — becomes the collector's body verbatim.
    content: Vec<BodyOp>,
    /// Every node this collector took over, to be deleted from wherever it was.
    consumed: Vec<Site>,
}

/// What one direction of one collector's walk found.
struct Walk {
    content: Vec<BodyOp>,
    consumed: Vec<Site>,
}

/// Plan every collector's absorption.
///
/// First come, first served: a claimed site is a barrier to the next collector, so nothing is absorbed
/// twice. [`ScopeOrder::Innermost`] is what makes that fair — a collector inside a box gets first
/// refusal on the content in there, being nearer to it than anything outside, and an outer collector
/// reaching in would starve it.
fn plan_absorptions(root: &DAGCircuit) -> PyResult<Vec<Absorption>> {
    let mut plans: Vec<Absorption> = Vec::new();
    let mut claimed: HashSet<Site> = HashSet::new();

    for collector in collectors(root, ScopeOrder::Innermost)? {
        let spec = collector
            .collector(root)?
            .expect("a swept site is a collector");
        let qubits = collector.qubits(root)?;

        // Walking leftward visits the outermost content last, so it comes back reversed.
        let mut left = walk_absorb(root, &collector, Direction::Left, &qubits, &claimed)?;
        left.content.reverse();
        let right = walk_absorb(root, &collector, Direction::Right, &qubits, &claimed)?;

        let mut content = left.content;
        content.extend(right.content);
        let mut consumed = left.consumed;
        consumed.extend(right.consumed);

        claimed.extend(consumed.iter().cloned());
        plans.push(Absorption {
            collector,
            spec,
            content,
            consumed,
        });
    }
    Ok(plans)
}

/// Whether a collector can absorb this instruction: a single-qubit standard gate.
///
/// Which wire it is on need not be checked — the walk only offers nodes adjacent along one of the
/// collector's own wires, so a single-qubit gate reached that way is on that wire by construction.
fn is_absorbable_gate(dag: &DAGCircuit, inst: &PackedInstruction) -> bool {
    matches!(inst.op.view(), OperationRef::StandardGate(_))
        && dag.qargs_interner().get(inst.qubits).len() == 1
}

/// Walk outward from a collector along its own wires, absorbing what it reaches.
///
/// Each wire carries its own cursor — the last site absorbed on it — so a blocked wire stops moving
/// rather than ending the walk, and a wire that has descended into a box goes on while another has
/// not. Each round drains the adjacent single-qubit gates then takes at most one emission layer; the
/// walk ends when no layer is available.
fn walk_absorb(
    root: &DAGCircuit,
    collector: &Site,
    direction: Direction,
    qubits: &[Qubit],
    claimed: &HashSet<Site>,
) -> PyResult<Walk> {
    // The direction an emission must have to face this collector.
    let facing = match direction {
        Direction::Right => Direction::Left,
        Direction::Left => Direction::Right,
    };
    let mut cursors: HashMap<Qubit, WireCursor> = qubits
        .iter()
        .map(|qubit| (*qubit, WireCursor::at(collector, *qubit)))
        .collect();
    let mut walk = Walk {
        content: Vec::new(),
        consumed: Vec::new(),
    };

    loop {
        // Drain the single-qubit gates now adjacent. Absorbing one can only expose another on that
        // same wire, so one pass per wire is enough.
        for qubit in qubits {
            while let Some((probe, site)) = unclaimed(root, &cursors[qubit], direction, claimed)? {
                let dag = site.scope_dag(root)?;
                let inst = dag.dag()[site.node].unwrap_operation();
                if !is_absorbable_gate(dag, inst) {
                    break;
                }
                // A dressing is its own box's. A gate in the collector's own scope belongs to whatever
                // encloses that box, not to it, so only what the walk descended into is on offer — and
                // of that, only the dressing side of the twirl point.
                if !site.deeper_than(&collector.scope)
                    || !on_dressing_side(dag, &site, direction, facing)?
                {
                    break;
                }
                let wires = site.qubits_in(root, &collector.scope)?;
                walk.content.push(BodyOp::Gate(site.clone(), wires));
                walk.consumed.push(site);
                cursors.insert(*qubit, probe);
            }
        }

        // Take one emission layer. Requiring every wire also confines it to the collector's own
        // qubits, since a wire outside them has no cursor to be adjacent on.
        let mut layer: Option<(Site, Emit, Vec<Qubit>)> = None;
        for qubit in qubits {
            let Some((_, site)) = unclaimed(root, &cursors[qubit], direction, claimed)? else {
                continue;
            };
            let Some(spec) = emission_spec(site.instruction(root)?) else {
                continue;
            };
            if spec.direction != Some(facing) {
                continue;
            }
            // Facing and adjacent is the whole test. Nothing checks whose box the emission came from:
            // an emission propagating out of an enclosing box faces the collectors of every box nested
            // inside it, and the nearest one takes it. That is deliberate but provisional — it
            // terminates the enclosing twirl at the inner dressing, with none of the enclosing box's
            // content in between. The discrimination belongs in *compatibility*, not position: once a
            // collector can be typed as unable to collect a given emission, it will decline and the
            // emission will carry on to one that can. See `lower::compatible`.
            let wires = site.qubits_in(root, &collector.scope)?;
            // An emission comes off as a whole layer or not at all: taking it on some wires only
            // would pull content from the far side of it into the body ahead of where it belongs.
            let mut every = true;
            for wire in &wires {
                let reached = match cursors.get(wire) {
                    Some(cursor) => {
                        unclaimed(root, cursor, direction, claimed)?.map(|(_, reached)| reached)
                    }
                    None => None,
                };
                if reached.as_ref() != Some(&site) {
                    every = false;
                    break;
                }
            }
            if every {
                layer = Some((site, spec, wires));
                break;
            }
        }
        let Some((site, spec, wires)) = layer else {
            break;
        };
        let local_spec = Emit {
            direction: None,
            partition: spec.partition.clone(),
            parts: spec.parts.clone(),
        };
        // The qargs are resolved later, when the emission is placed into its collector's body.
        let op = PackedOperation::from_custom_operation(Box::new(local_spec));
        walk.content.push(BodyOp::Local(op, wires.clone()));
        walk.consumed.push(site);
        for wire in &wires {
            let mut probe = cursors[wire].clone();
            probe.advance(root, direction)?;
            cursors.insert(*wire, probe);
        }
    }

    Ok(walk)
}

/// What this wire sees next, unless another collector has already claimed it.
///
/// A claimed site is this pass's own barrier laid on top of the walk's: first come, first served, so a
/// site already spoken for ends the wire as surely as the end of a body does.
fn unclaimed(
    root: &DAGCircuit,
    cursor: &WireCursor,
    direction: Direction,
    claimed: &HashSet<Site>,
) -> PyResult<Option<(WireCursor, Site)>> {
    match cursor.peek(root, direction)? {
        Some((probe, site)) if !claimed.contains(&site) => Ok(Some((probe, site))),
        _ => Ok(None),
    }
}

/// Whether a gate inside a box is on the dressing side of that box's twirl point, as seen by a
/// collector walking `direction`.
///
/// **A box with no emission in it has no twirl point to be on the wrong side of**, so its whole
/// absorbable run is fair game. That is the ordinary shape of a box carrying only a `ChangeBasis`: the
/// frame change names the box's edge, so it is written on the spine outside, and nothing propagates
/// within the body at all. Nothing there can be on an emission's path, so nothing there can be taken
/// off one. An enclosing box's emission crossing the body is unaffected — this collector is foreign to
/// it, so the gates are still crossed rather than composed.
///
/// Otherwise scan on along the gate's own wire, over the absorbable run it belongs to. An emission
/// facing this collector at the end of that run means the run lies between the collector and the twirl
/// point: those gates multiply into this dressing and nothing propagating ever crosses them. Anything
/// else — content, a collector, the end of the body — means the run is on the far side of the twirl
/// point, so it is content that an emission travelling this way is conjugated by. Folding that in would
/// take it off the propagation path, which is sound only if the incoming emission is composed on the far
/// side of it, and nothing implements that yet.
fn on_dressing_side(
    dag: &DAGCircuit,
    site: &Site,
    direction: Direction,
    facing: Direction,
) -> PyResult<bool> {
    if !holds_any_emission(dag)? {
        return Ok(true);
    }
    let inst = dag.dag()[site.node].unwrap_operation();
    // Absorbable gates are single-qubit, so the run this one belongs to is on one wire.
    let Some(wire) = dag.qargs_interner().get(inst.qubits).first().copied() else {
        return Ok(false);
    };
    let mut at = site.node;
    while let Some(next) = next_on_wire(dag, at, wire, direction) {
        let ahead = dag.dag()[next].unwrap_operation();
        if let Some(spec) = emission_spec(ahead) {
            return Ok(spec.direction == Some(facing));
        }
        if !is_absorbable_gate(dag, ahead) {
            return Ok(false);
        }
        at = next;
    }
    Ok(false)
}

/// Whether this body holds an emission of its own.
///
/// This scope only, deliberately. A nested box's emissions are fenced by that box's own collectors —
/// build writes them before its emissions on either side — so they resolve inside it and never reach a
/// gate in the enclosing body. Descending would count them anyway and cost the fold: a `ChangeBasis` box
/// wrapping a twirled one would look occupied by the inner box's twirl point and stop folding its own
/// absorbable run, which is exactly the case this is here to allow.
fn holds_any_emission(dag: &DAGCircuit) -> PyResult<bool> {
    Ok(dag
        .op_nodes(true)
        .any(|(_, inst)| emission_spec(inst).is_some()))
}

/// Build a collector's body from what it absorbed, remapped into its own frame.
fn build_body(root: &DAGCircuit, plan: &Absorption) -> PyResult<DAGCircuit> {
    let frame = plan.collector.qubits(root)?;
    let num_clbits = plan.collector.num_clbits(root)?;
    let mut body = new_dag_body(frame.len(), num_clbits, plan.content.len())?.into_builder();

    // The walk only ever offers content on the collector's own wires, so this always resolves.
    let remap = |wires: &[Qubit]| -> Vec<Qubit> {
        wires
            .iter()
            .map(|wire| {
                let local = frame
                    .iter()
                    .position(|q| q == wire)
                    .expect("absorbed content is on one of its collector's wires");
                Qubit(local as u32)
            })
            .collect()
    };

    for op in &plan.content {
        match op {
            BodyOp::Gate(site, wires) => {
                append_instruction(&mut body, site.instruction(root)?, &remap(wires), &[])?;
            }
            BodyOp::Local(local_op, wires) => {
                append(&mut body, local_op.clone(), None, &remap(wires), &[])?;
            }
        }
    }
    Ok(body.build())
}

/// The collector operation carrying the newly absorbed body.
fn absorbed_op(root: &DAGCircuit, plan: &Absorption) -> PyResult<PackedOperation> {
    let annotation = Collect {
        // Absorption changes only what a collector composes, not what it is.
        partition: plan.spec.partition.clone(),
        parts: plan.spec.parts.clone(),
    };
    Ok(collect_op(
        annotation,
        plan.collector.qubits(root)?.len(),
        plan.collector.num_clbits(root)?,
    ))
}
