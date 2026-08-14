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

use qiskit_circuit::standard_gate::StandardGate;

use crate::annotated_circuit::{DistributionType, SynthesizerType};
use crate::distributions::DistKey;
use crate::virtual_flow_graph::*;

pub fn emit_node(qubits: &[usize]) -> Node {
    emission_node(qubits, DistKey(0))
}

pub fn emission_node(qubits: &[usize], key: DistKey) -> Node {
    Node::singletons(
        qubits.to_vec(),
        NodeKind::Emission(Emission {
            key,
            direction: Direction::Right,
            virtual_type: VirtualType::Pauli,
        }),
    )
}

pub fn typed_emit_node(qubits: &[usize], distribution: DistributionType) -> Node {
    Node::singletons(
        qubits.to_vec(),
        NodeKind::Emission(Emission {
            key: DistKey(0),
            direction: Direction::Right,
            virtual_type: distribution.virtual_type(),
        }),
    )
}

pub fn propagate_node(qubits: &[usize]) -> Node {
    propagate_node_with(qubits, StandardGate::CX, Direction::Right)
}

pub fn propagate_node_with(qubits: &[usize], gate: StandardGate, direction: Direction) -> Node {
    Node::joint(
        qubits.to_vec(),
        NodeKind::Propagate(Propagate { gate, direction }),
    )
}

pub fn collect_node(qubits: &[usize]) -> Node {
    Node::singletons(
        qubits.to_vec(),
        NodeKind::Collect(Collect {
            synthesizer: SynthesizerType::RzSx,
            param_indices: vec![],
            steps: Vec::new(),
        }),
    )
}

/// A collector with one absorbed gate in its body, so it has angles of its own to synthesize.
pub fn collect_node_with_gate(qubits: &[usize], gate: StandardGate, gate_qubit: usize) -> Node {
    Node::singletons(
        qubits.to_vec(),
        NodeKind::Collect(Collect {
            synthesizer: SynthesizerType::RzSx,
            param_indices: vec![],
            steps: vec![CollectStep::Gate(AbsorbedGate {
                gate,
                qubits: vec![gate_qubit],
                params: vec![],
            })],
        }),
    )
}

pub fn measure_node(qubits: &[usize]) -> Node {
    Node::singletons(
        qubits.to_vec(),
        NodeKind::Measure(Measure {
            clbit_indices: vec![0],
        }),
    )
}
