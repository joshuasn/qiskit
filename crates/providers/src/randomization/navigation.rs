// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Reading a circuit being randomized one wire and one scope at a time.
//!
//! Every pass here works within a single scope. A box that survives lowering is opaque, so the only
//! way into a body is [`in_every_scope`].
//!
//! What a walk finds is offered as data rather than as a circuit. A [`ContentKind`] is what one
//! position looks like to a rule and [`WireContent`] is a wire's worth of them.

use rustworkx_core::petgraph::Direction as PetDirection;
use rustworkx_core::petgraph::stable_graph::NodeIndex;
use rustworkx_core::petgraph::visit::EdgeRef;

use qiskit_circuit::Qubit;
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType, Wire};
use qiskit_circuit::operations::{Operation, OperationRef};
use qiskit_circuit::packed_instruction::PackedInstruction;

use super::Direction;
use super::emissions::{Collect, Emit, Hem};

/// The [`Collect`] this instruction is if it is one.
pub fn as_collect(inst: &PackedInstruction) -> Option<&Collect> {
    match inst.op.view() {
        OperationRef::CustomOperation(op) => op.downcast_ref::<Collect>(),
        _ => None,
    }
}

/// The [`Emit`] this instruction is if it is one.
pub fn as_emit(inst: &PackedInstruction) -> Option<&Emit> {
    match inst.op.view() {
        OperationRef::CustomOperation(op) => op.downcast_ref::<Emit>(),
        _ => None,
    }
}

/// The [`Hem`] this instruction is if it is one.
pub fn as_hem(inst: &PackedInstruction) -> Option<&Hem> {
    match inst.op.view() {
        OperationRef::CustomOperation(op) => op.downcast_ref::<Hem>(),
        _ => None,
    }
}

/// The instruction at a node.
pub fn op_at(dag: &DAGCircuit, node: NodeIndex) -> &PackedInstruction {
    dag.dag()[node].unwrap_operation()
}

/// The collect at a node.
///
/// Panics if the node is not a collect. Ask [`as_collect`] where that is still the question.
pub fn collect_at(dag: &DAGCircuit, node: NodeIndex) -> &Collect {
    as_collect(op_at(dag, node)).expect("only asked of a collect")
}

/// Run `pass` on this circuit and on every box body nested in it at any depth.
pub fn in_every_scope(dag: &mut DAGCircuit, pass: &mut impl FnMut(&mut DAGCircuit)) {
    pass(dag);
    let blocks: Vec<_> = dag
        .op_nodes(true)
        .flat_map(|(_, inst)| inst.blocks_view().iter().copied())
        .collect();
    for block in blocks {
        in_every_scope(dag.view_block_mut(block), pass);
    }
}

/// The fallible twin of [`in_every_scope`], for a pass that can refuse.
///
/// The first scope to refuse stops the walk.
pub fn in_every_scope_result<E>(
    dag: &mut DAGCircuit,
    pass: &mut impl FnMut(&mut DAGCircuit) -> Result<(), E>,
) -> Result<(), E> {
    pass(dag)?;
    let blocks: Vec<_> = dag
        .op_nodes(true)
        .flat_map(|(_, inst)| inst.blocks_view().iter().copied())
        .collect();
    for block in blocks {
        in_every_scope_result(dag.view_block_mut(block), pass)?;
    }
    Ok(())
}

/// Every collect in this scope in circuit order.
pub fn collectors(dag: &DAGCircuit) -> Vec<NodeIndex> {
    dag.topological_op_nodes(false)
        .filter(|node| as_collect(op_at(dag, *node)).is_some())
        .collect()
}

/// The next operation node along one wire or `None` at the end of it.
///
/// A box is the node reported rather than something stepped into.
pub fn next_on_wire(
    dag: &DAGCircuit,
    from: NodeIndex,
    qubit: Qubit,
    direction: Direction,
) -> Option<NodeIndex> {
    let (search, wire) = match direction {
        Direction::Forward => (PetDirection::Outgoing, Wire::Qubit(qubit)),
        Direction::Backward => (PetDirection::Incoming, Wire::Qubit(qubit)),
    };
    let next = dag
        .dag()
        .edges_directed(from, search)
        .find(|edge| *edge.weight() == wire)
        .map(|edge| match direction {
            Direction::Forward => edge.target(),
            Direction::Backward => edge.source(),
        })?;
    matches!(dag.dag()[next], NodeType::Operation(_)).then_some(next)
}

/// The wires an instruction is on.
pub fn wires_of(dag: &DAGCircuit, node: NodeIndex) -> &[Qubit] {
    dag.qargs_interner().get(op_at(dag, node).qubits)
}

/// The one node every one of `wires` reaches next travelling `direction`.
///
/// `None` unless all of the wires arrive at the same node.
pub fn common_next(
    dag: &DAGCircuit,
    from: NodeIndex,
    wires: &[Qubit],
    direction: Direction,
) -> Option<NodeIndex> {
    let mut common: Option<NodeIndex> = None;
    for wire in wires {
        let next = next_on_wire(dag, from, *wire, direction)?;
        match common {
            None => common = Some(next),
            Some(seen) if seen == next => {}
            Some(_) => return None,
        }
    }
    common
}

/// The kind of content at a given position.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ContentKind {
    /// A gate that can belong to an easy run.
    Easy,
    /// An emit.
    Emit,
    /// A collect.
    Collect,
    /// A hem. Nothing absorbs it and nothing moves across it.
    Hem,
    /// Hard content. No collect absorbs it and nothing is moved across it.
    Hard,
}

impl ContentKind {
    /// What this instruction is.
    pub fn of(inst: &PackedInstruction) -> Self {
        if as_emit(inst).is_some() {
            ContentKind::Emit
        } else if as_collect(inst).is_some() {
            ContentKind::Collect
        } else if as_hem(inst).is_some() {
            ContentKind::Hem
        } else if matches!(inst.op.view(), OperationRef::StandardGate(gate) if gate.num_qubits() == 1)
        {
            ContentKind::Easy
        } else {
            // Every other custom operation included and a box above all. One the user asked for
            // is opaque so it is content nothing reaches into and nothing moves across.
            ContentKind::Hard
        }
    }

    /// Whether a dressing could fold this in.
    pub fn is_absorbable(self) -> bool {
        matches!(self, ContentKind::Easy)
    }
}

/// The content ahead of a position on one wire in the order that wire meets it.
pub struct WireContent<'a> {
    dag: &'a DAGCircuit,
    at: NodeIndex,
    wire: Qubit,
    direction: Direction,
}

impl<'a> WireContent<'a> {
    /// The content `wire` meets ahead of the node `from` travelling `direction`.
    pub fn ahead(dag: &'a DAGCircuit, from: NodeIndex, wire: Qubit, direction: Direction) -> Self {
        WireContent {
            dag,
            at: from,
            wire,
            direction,
        }
    }
}

impl Iterator for WireContent<'_> {
    type Item = ContentKind;

    fn next(&mut self) -> Option<ContentKind> {
        self.at = next_on_wire(self.dag, self.at, self.wire, self.direction)?;
        Some(ContentKind::of(self.dag.dag()[self.at].unwrap_operation()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use qiskit_circuit::bit::QuantumRegister;
    use qiskit_circuit::operations::StandardInstruction;
    use qiskit_circuit::packed_instruction::PackedOperation;

    use crate::randomization::SynthesizerKind;
    use crate::randomization::distributions::{Distribution, UniformPauli};
    use crate::randomization::partition::Partition;

    fn emit_instruction(direction: Direction) -> PackedInstruction {
        let factorization = Partition::singletons(0..1).expect("a range of qubits has no repeats");
        let distribution: Arc<dyn Distribution> = Arc::new(
            UniformPauli::new(factorization).expect("uniform Paulis resolve over singletons"),
        );
        PackedInstruction::from_custom_operation(
            Emit::new(direction, distribution, [0], false)
                .expect("the one factor of a one-qubit Pauli"),
            Default::default(),
            Default::default(),
            None,
        )
    }

    fn collect_instruction() -> PackedInstruction {
        PackedInstruction::from_custom_operation(
            Collect::new(SynthesizerKind::RzSx.resolve(1)),
            Default::default(),
            Default::default(),
            None,
        )
    }

    fn hem_instruction() -> PackedInstruction {
        PackedInstruction::from_custom_operation(Hem, Default::default(), Default::default(), None)
    }

    #[test]
    fn test_what_an_instruction_is() {
        // An emit is an emit whichever way it travels. Which way is read off the `Emit` by
        // whatever cares, so the kind does not carry it.
        for direction in [Direction::Forward, Direction::Backward] {
            assert_eq!(
                ContentKind::of(&emit_instruction(direction)),
                ContentKind::Emit,
                "{direction:?}"
            );
        }
        assert_eq!(
            ContentKind::of(&collect_instruction()),
            ContentKind::Collect
        );
        assert_eq!(ContentKind::of(&hem_instruction()), ContentKind::Hem);
    }

    #[test]
    fn test_a_user_barrier_is_hard_content_rather_than_a_hem() {
        let mut dag = DAGCircuit::new();
        dag.add_qreg(QuantumRegister::new_owning("q", 1))
            .expect("a fresh register");
        let mut builder = dag.into_builder();
        builder
            .apply_operation_back(
                PackedOperation::from_standard_instruction(StandardInstruction::Barrier(1)),
                &[Qubit(0)],
                &[],
                None,
                None,
            )
            .expect("a barrier on a real wire");
        let dag = builder.build();
        let node = dag
            .topological_op_nodes(false)
            .next()
            .expect("the barrier is in there");
        assert_eq!(ContentKind::of(op_at(&dag, node)), ContentKind::Hard);
    }
}
