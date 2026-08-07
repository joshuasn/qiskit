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
//! After the build pass, every emission and every easy gate sits on the spine and every collector
//! starts empty (no items, no body). This pass walks outward from each collector and takes over what
//! it reaches:
//!
//! - A single-qubit standard gate → part of a `CollectItem::Gates` run
//! - An emission facing this collector → a `CollectItem::Emission`
//! - Anything else (propagating emission, multi-qubit gate, box, another collector) → that wire is
//!   done, and only that wire
//!
//! **Absorption is per wire.** A collector reaches along each of its own qubits independently, so an
//! entangler on q0 stops the walk on q0 and q1 without saying anything about q2. The wire is also
//! what makes the gate's frame unambiguous: a single-qubit gate reached along wire `q` is *on* `q`, so
//! it maps into the collector body at `q`'s position, with nothing to guess.
//!
//! **Emissions come off in layers.** An emission is absorbed only when it is the next node on *every*
//! one of its wires at once. That is what keeps the item order meaningful: `Gates(n)` says the next
//! `n` body instructions compose on one side of the emission that follows, which is only true if the
//! emission is a barrier across all of them. Absorbing one wire's half of an emission layer early
//! would put a gate from the far side of it into the near run.
//!
//! The walk's order IS the composition order, so no `EmitSource`-based classification is needed.
//! Cross-scope absorption — an emission whose collector is inside an adjacent box — is handled by
//! injecting it at that box's near edge and letting the recursive descent absorb it there.

use hashbrown::{HashMap, HashSet};
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use pyo3::prelude::*;
use qiskit_circuit::Qubit;
use qiskit_circuit::annotation::PyAnnotation;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{ControlFlow, ControlFlowInstruction, OperationRef};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};

use super::utils::{
    IntoPyResult, collect_annotation, emission_spec, new_dag_body, next_on_wire, params_of,
};
use crate::emission_circuit::{Collect, CollectItem, CollectSpec, EmitSpec, LocalEmission};
use crate::virtual_flow_graph::Direction;

/// Absorb dressing into every collector, in place.
#[pyfunction]
#[pyo3(name = "absorb_dressing")]
pub fn py_absorb_dressing(py: Python, dag: &mut DAGCircuit) -> PyResult<()> {
    absorb_dressing(py, dag)
}

/// Absorb dressing into every collector, in place.
pub fn absorb_dressing(py: Python, dag: &mut DAGCircuit) -> PyResult<()> {
    absorb_scope(py, dag)
}

/// What one collector takes over from the spine.
struct Absorption {
    collector: NodeIndex,
    /// The collector's existing descriptors, which absorption does not change.
    spec: CollectSpec,
    /// Composition order: gate runs interleaved with the emissions between them.
    items: Vec<CollectItem>,
    /// Absorbed gate nodes in the order the `Gates` counts assume, so the body can be built by
    /// walking this list.
    gates: Vec<NodeIndex>,
    /// Every node this collector took over, gates and emissions alike, to be deleted from the spine.
    consumed: Vec<NodeIndex>,
}

/// Absorb one scope in place, then recurse into its boxes.
///
/// Planning reads the DAG and rewriting mutates it, so the two are separate sweeps. What is carried
/// between them is node indices, not circuits — `StableDiGraph` keeps indices valid across the
/// removals and substitutions below.
fn absorb_scope(py: Python, dag: &mut DAGCircuit) -> PyResult<()> {
    let plans = plan_absorptions(py, dag);
    // Kept as a sequence as well as a set, so that removal order is fixed rather than whatever the
    // set happens to iterate in.
    let consumed: Vec<NodeIndex> = plans
        .iter()
        .flat_map(|plan| plan.consumed.iter().copied())
        .collect();
    let claimed: HashSet<NodeIndex> = consumed.iter().copied().collect();
    let descents = plan_descents(py, dag, &claimed);

    for plan in &plans {
        let body = build_body(dag, plan)?;
        let op = collect_op(py, dag, plan)?;
        let block = dag.add_block(body);
        dag.substitute_op(
            plan.collector,
            op,
            Some(Parameters::Blocks(vec![block])),
            None,
        )
        .into_py_result()?;
    }
    for node in &consumed {
        dag.remove_op_node(*node);
    }

    // An emission that descends moves out of this scope and into the box's body, where the recursion
    // below finds it at the near edge and absorbs it like any local one.
    for (box_node, injections) in &descents {
        inject(dag, *box_node, injections)?;
        for (node, _) in injections {
            dag.remove_op_node(*node);
        }
    }

    // Recurse into every box that is not itself a collector — a collector's body holds only the
    // single-qubit gates just absorbed into it, so there is nothing there to absorb.
    let bodies: Vec<_> = dag
        .topological_op_nodes(false)
        .filter_map(|node| {
            let inst = dag.dag()[node].unwrap_operation();
            if collect_annotation(py, inst).is_some() {
                return None;
            }
            match inst.blocks_view() {
                [block] if is_box(inst) => Some(*block),
                _ => None,
            }
        })
        .collect();
    for block in bodies {
        absorb_scope(py, dag.view_block_mut(block))?;
    }
    Ok(())
}

/// Whether an instruction is a `box`.
fn is_box(inst: &PackedInstruction) -> bool {
    matches!(inst.op.view(), OperationRef::ControlFlow(cf) if matches!(cf.control_flow, ControlFlow::Box { .. }))
}

/// Whether a collector can absorb this instruction: a single-qubit standard gate.
///
/// Nothing here checks *which* wire it is on. It does not have to: the walk only ever offers a node
/// that is adjacent along one of the collector's own qubits, and a single-qubit gate reached that way
/// is on that qubit by construction.
fn is_absorbable_gate(dag: &DAGCircuit, inst: &PackedInstruction) -> bool {
    matches!(inst.op.view(), OperationRef::StandardGate(_))
        && dag.qargs_interner().get(inst.qubits).len() == 1
}

/// Plan every collector's absorption, in topological order.
///
/// Collectors are handled first-come-first-served: a node one collector has claimed is a barrier to
/// the next, so nothing is absorbed twice. Topological order makes which one wins deterministic.
fn plan_absorptions(py: Python, dag: &DAGCircuit) -> Vec<Absorption> {
    let mut plans: Vec<Absorption> = Vec::new();
    let mut claimed: HashSet<NodeIndex> = HashSet::new();

    for collector in dag.topological_op_nodes(false) {
        let inst = dag.dag()[collector].unwrap_operation();
        let Some(spec) = collect_annotation(py, inst) else {
            continue;
        };
        let qubits: Vec<Qubit> = dag.qargs_interner().get(inst.qubits).to_vec();

        // Walking leftward visits the outermost content last, so both lists come back reversed.
        // Reversing the gates too keeps each run's body order matching its count.
        let mut left = walk_absorb(py, dag, collector, Direction::Left, &qubits, &claimed);
        left.items.reverse();
        left.gates.reverse();
        let right = walk_absorb(py, dag, collector, Direction::Right, &qubits, &claimed);

        let mut items = left.items;
        items.extend(right.items);
        let mut gates = left.gates;
        gates.extend(right.gates);
        let mut consumed = left.consumed;
        consumed.extend(right.consumed);

        claimed.extend(consumed.iter().copied());
        plans.push(Absorption {
            collector,
            spec,
            items,
            gates,
            consumed,
        });
    }
    plans
}

/// What one direction of one collector's walk found.
struct Walk {
    items: Vec<CollectItem>,
    gates: Vec<NodeIndex>,
    consumed: Vec<NodeIndex>,
}

/// Walk outward from a collector along its own wires, absorbing what it reaches.
///
/// Each wire carries a cursor, the last node absorbed on it, so wires advance independently and a
/// blocked one simply stops moving rather than ending the walk. Each round drains the adjacent
/// single-qubit gates, then takes at most one emission layer; when no layer is available nothing can
/// change and the walk is over.
fn walk_absorb(
    py: Python,
    dag: &DAGCircuit,
    collector: NodeIndex,
    direction: Direction,
    qubits: &[Qubit],
    claimed: &HashSet<NodeIndex>,
) -> Walk {
    // The direction an emission must have to face this collector.
    let facing = match direction {
        Direction::Right => Direction::Left,
        Direction::Left => Direction::Right,
    };
    let mut cursor: HashMap<Qubit, NodeIndex> =
        qubits.iter().map(|qubit| (*qubit, collector)).collect();
    let mut walk = Walk {
        items: Vec::new(),
        gates: Vec::new(),
        consumed: Vec::new(),
    };
    let mut run: usize = 0;

    loop {
        // Drain the single-qubit gates now adjacent. A run on one wire comes off one at a time, and
        // absorbing one can only ever expose another on that same wire, so one pass per wire is enough.
        for qubit in qubits {
            while let Some(next) = adjacent(dag, &cursor, *qubit, direction, claimed) {
                if !is_absorbable_gate(dag, dag.dag()[next].unwrap_operation()) {
                    break;
                }
                run += 1;
                walk.gates.push(next);
                walk.consumed.push(next);
                cursor.insert(*qubit, next);
            }
        }

        // Take one emission layer: an emission that faces this collector and is adjacent on every one
        // of its wires. Requiring every wire also confines it to the collector's own qubits, since a
        // wire outside them has no cursor to be adjacent on.
        let layer = qubits.iter().find_map(|qubit| {
            let node = adjacent(dag, &cursor, *qubit, direction, claimed)?;
            let inst = dag.dag()[node].unwrap_operation();
            let spec = emission_spec(py, inst)?;
            if spec.direction != facing {
                return None;
            }
            dag.qargs_interner()
                .get(inst.qubits)
                .iter()
                .all(|wire| adjacent(dag, &cursor, *wire, direction, claimed) == Some(node))
                .then_some((node, spec))
        });
        let Some((node, spec)) = layer else {
            break;
        };
        if run > 0 {
            walk.items.push(CollectItem::Gates(run));
            run = 0;
        }
        walk.items.push(CollectItem::Emission(LocalEmission {
            partition: spec.partition.clone(),
            parts: spec.parts.clone(),
        }));
        walk.consumed.push(node);
        let inst = dag.dag()[node].unwrap_operation();
        for wire in dag.qargs_interner().get(inst.qubits) {
            cursor.insert(*wire, node);
        }
    }

    if run > 0 {
        walk.items.push(CollectItem::Gates(run));
    }
    walk
}

/// The node this wire sees next, or `None` if the wire is not the collector's, has ended, or runs
/// into something another collector already claimed.
fn adjacent(
    dag: &DAGCircuit,
    cursor: &HashMap<Qubit, NodeIndex>,
    qubit: Qubit,
    direction: Direction,
    claimed: &HashSet<NodeIndex>,
) -> Option<NodeIndex> {
    let from = *cursor.get(&qubit)?;
    let next = next_on_wire(dag, from, qubit, direction)?;
    (!claimed.contains(&next)).then_some(next)
}

/// Build a collector's body from the gates it absorbed, remapped into its own frame.
fn build_body(dag: &DAGCircuit, plan: &Absorption) -> PyResult<DAGCircuit> {
    let inst = dag.dag()[plan.collector].unwrap_operation();
    let frame: Vec<Qubit> = dag.qargs_interner().get(inst.qubits).to_vec();
    let num_clbits = dag.cargs_interner().get(inst.clbits).len();
    let mut body = new_dag_body(frame.len(), num_clbits, plan.gates.len())?.into_builder();

    for node in &plan.gates {
        let gate = dag.dag()[*node].unwrap_operation();
        let qargs: Vec<Qubit> = dag
            .qargs_interner()
            .get(gate.qubits)
            .iter()
            .map(|wire| {
                // The walk only offers gates on the collector's own wires, so this always resolves.
                let local = frame
                    .iter()
                    .position(|q| q == wire)
                    .expect("an absorbed gate is on one of its collector's wires");
                Qubit(local as u32)
            })
            .collect();
        super::utils::append(&mut body, gate.op.clone(), params_of(gate), &qargs, &[])?;
    }
    Ok(body.build())
}

/// The collector operation carrying the newly absorbed items.
fn collect_op(py: Python, dag: &DAGCircuit, plan: &Absorption) -> PyResult<PackedOperation> {
    let inst = dag.dag()[plan.collector].unwrap_operation();
    let spec = CollectSpec {
        items: plan.items.clone(),
        partition: plan.spec.partition.clone(),
        parts: plan.spec.parts.clone(),
    };
    debug_assert_eq!(
        spec.gate_count(),
        plan.gates.len(),
        "absorbed gates must match the items' `Gates` counts"
    );
    let annotation = Py::new(py, (Collect::new_from_spec(spec), PyAnnotation))?;
    Ok(PackedOperation::from_control_flow(Box::new(
        ControlFlowInstruction {
            control_flow: ControlFlow::Box {
                duration: None,
                annotations: vec![annotation.into_any()],
            },
            num_qubits: dag.qargs_interner().get(inst.qubits).len() as u32,
            num_clbits: dag.cargs_interner().get(inst.clbits).len() as u32,
        },
    )))
}

// --- Cross-scope absorption ---------------------------------------------------------------------

/// An emission moving into a box, and the direction it was travelling.
type Injection = (NodeIndex, Direction);

/// Find the emissions no collector in this scope absorbed that can descend into an adjacent box.
fn plan_descents(
    py: Python,
    dag: &DAGCircuit,
    claimed: &HashSet<NodeIndex>,
) -> Vec<(NodeIndex, Vec<Injection>)> {
    let mut descents: Vec<(NodeIndex, Vec<Injection>)> = Vec::new();
    for node in dag.topological_op_nodes(false) {
        if claimed.contains(&node) {
            continue;
        }
        let Some(spec) = emission_spec(py, dag.dag()[node].unwrap_operation()) else {
            continue;
        };
        let Some(target) = descent_target(py, dag, node, spec.direction) else {
            continue;
        };
        match descents
            .iter_mut()
            .find(|(box_node, _)| *box_node == target)
        {
            Some((_, injections)) => injections.push((node, spec.direction)),
            None => descents.push((target, vec![(node, spec.direction)])),
        }
    }
    descents
}

/// The box this emission descends into, if any.
///
/// Scanning is along the emission's own wires, and unlike the absorption walk it passes *through*
/// what it meets: a propagating emission crosses gates and other emissions by definition, including
/// ones a collector has already claimed. Only two things end the scan — a collector at this level,
/// meaning the emission belongs here rather than deeper, and a box, which it either enters or does
/// not.
fn descent_target(
    py: Python,
    dag: &DAGCircuit,
    start: NodeIndex,
    direction: Direction,
) -> Option<NodeIndex> {
    let inst = dag.dag()[start].unwrap_operation();
    let spec = emission_spec(py, inst)?;
    let qubits: Vec<Qubit> = dag.qargs_interner().get(inst.qubits).to_vec();
    let mut cursor: HashMap<Qubit, NodeIndex> =
        qubits.iter().map(|qubit| (*qubit, start)).collect();

    loop {
        // Advance whichever wire still has something on it; a wire that has run out just stops
        // contributing, and when none is left the emission has nowhere to descend to.
        let advance = qubits
            .iter()
            .find_map(|qubit| next_on_wire(dag, *cursor.get(qubit)?, *qubit, direction))?;
        let inst = dag.dag()[advance].unwrap_operation();
        if collect_annotation(py, inst).is_some() {
            return None;
        }
        if is_box(inst) {
            let [block] = *inst.blocks_view() else {
                return None;
            };
            let covered = dag.qargs_interner().get(inst.qubits);
            if !qubits.iter().all(|q| covered.contains(q)) {
                // The box does not span the emission, so there is no frame to inject it into.
                return None;
            }
            let body = dag.blocks().get(block)?;
            return has_compatible_collector_at_edge(py, body, direction, &spec).then_some(advance);
        }
        for wire in dag.qargs_interner().get(inst.qubits) {
            if cursor.contains_key(wire) {
                cursor.insert(*wire, advance);
            }
        }
    }
}

/// Whether a box body has a collector at its near edge that accepts this emission.
///
/// "Near edge" is the side the emission enters from: travelling right it enters at the body's left,
/// so the first thing on each wire; travelling left, the last. Emissions already at that edge do not
/// block it, and a nested box is descended into.
fn has_compatible_collector_at_edge(
    py: Python,
    body: &DAGCircuit,
    direction: Direction,
    spec: &EmitSpec,
) -> bool {
    for node in body.topological_op_nodes(matches!(direction, Direction::Left)) {
        let inst = body.dag()[node].unwrap_operation();
        if emission_spec(py, inst).is_some() {
            continue;
        }
        if let Some(collector) = collect_annotation(py, inst) {
            return collector.accepts(spec.virtual_type());
        }
        if is_box(inst) {
            return match inst.blocks_view() {
                [block] => body.blocks().get(*block).is_some_and(|inner| {
                    has_compatible_collector_at_edge(py, inner, direction, spec)
                }),
                _ => false,
            };
        }
        return false;
    }
    false
}

/// Write descending emissions into a box's body at the edge they enter from.
fn inject(dag: &mut DAGCircuit, box_node: NodeIndex, injections: &[Injection]) -> PyResult<()> {
    let inst = dag.dag()[box_node].unwrap_operation();
    let covered: Vec<Qubit> = dag.qargs_interner().get(inst.qubits).to_vec();
    let [block] = *inst.blocks_view() else {
        return Ok(());
    };

    // Each emission's operation and its qubits in the body's frame, resolved before the body is
    // borrowed mutably.
    let mut ops: Vec<(PackedOperation, Vec<Qubit>, Direction)> = Vec::new();
    for (node, direction) in injections {
        let emit = dag.dag()[*node].unwrap_operation();
        let qargs: Vec<Qubit> = dag
            .qargs_interner()
            .get(emit.qubits)
            .iter()
            .map(|wire| {
                // `descent_target` refuses a box that does not span the emission.
                let local = covered
                    .iter()
                    .position(|q| q == wire)
                    .expect("a descending emission lies within its box");
                Qubit(local as u32)
            })
            .collect();
        ops.push((emit.op.clone(), qargs, *direction));
    }

    let body = dag.view_block_mut(block);
    // An emission travelling right enters at the body's left edge, so it goes to the front — in
    // reverse, so that the first of several ends up outermost.
    for (op, qargs, _) in ops.iter().filter(|(_, _, d)| *d == Direction::Right).rev() {
        body.apply_operation_front(
            op.clone(),
            qargs,
            &[],
            None,
            None,
            #[cfg(feature = "cache_pygates")]
            None,
        )
        .into_py_result()?;
    }
    for (op, qargs, _) in ops.iter().filter(|(_, _, d)| *d == Direction::Left) {
        body.apply_operation_back(
            op.clone(),
            qargs,
            &[],
            None,
            None,
            #[cfg(feature = "cache_pygates")]
            None,
        )
        .into_py_result()?;
    }
    Ok(())
}
