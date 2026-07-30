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

//! The algebraic type of a virtual gate.
//!
//! This is an object rather than part of any one IR: the annotation vocabulary resolves *to* it, the
//! emission circuit carries it on each emission, and the sampling graph infers it along edges. It
//! previously lived in the sampling-graph module, which forced the annotation module to import from
//! the graph — a dependency pointing the wrong way.

use qiskit_circuit::operations::{Operation, StandardGate};

/// The group a virtual gate belongs to, which determines how it composes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VirtualType {
    Pauli,
    C1,
    U2,
    Z2,
}

impl VirtualType {
    /// The lowercase name used in Python-facing readouts.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Pauli => "pauli",
            Self::C1 => "c1",
            Self::U2 => "u2",
            Self::Z2 => "z2",
        }
    }
}

/// Whether a virtual gate of this type stays in its group when conjugated by `gate`.
///
/// This is the real limit on which circuits samplomatic supports — not box nesting, not control flow.
/// A Pauli stays a Pauli through a Clifford, which is what samplomatic's `PAULI_PAST_CLIFFORD` tables
/// encode; conjugating a Pauli by a non-Clifford leaves the group entirely and there is no rule to
/// apply. So a gate on an emission's propagation path with no rule for that emission's type must be a
/// hard error rather than a silently wrong answer.
///
/// Deliberately an allowlist: a gate absent from it is refused, so adding gate support is a conscious
/// act rather than something that happens by accident.
pub fn propagates(virtual_type: VirtualType, gate: StandardGate) -> bool {
    use StandardGate::*;
    // Conjugating by a single-qubit gate keeps a single-qubit group closed, so any 1Q Clifford works
    // for the finite groups and any 1Q gate at all works for U2.
    let clifford_1q = matches!(gate, H | S | Sdg | SX | SXdg | X | Y | Z);
    // Two-qubit entanglers with propagation rules. `RZZ` is here because samplomatic supports it as a
    // fractional entangler under Pauli twirling.
    let clifford_2q = matches!(gate, CX | CZ | CY | ECR | Swap | DCX | RZZ);

    match virtual_type {
        // A Pauli or a Z2 sign stays in its group through Cliffords only.
        VirtualType::Pauli | VirtualType::Z2 => clifford_1q || clifford_2q,
        // Local C1 elements propagate through the same entanglers, per the local-C1 tables.
        VirtualType::C1 => clifford_1q || clifford_2q,
        // A U2 element conjugated by a single-qubit gate is still a U2 element, but an entangler takes
        // it out of the local group, so there is nothing to propagate.
        VirtualType::U2 => gate.num_qubits() == 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pauli_survives_cliffords() {
        for gate in [
            StandardGate::H,
            StandardGate::S,
            StandardGate::SX,
            StandardGate::CX,
            StandardGate::CZ,
            StandardGate::ECR,
        ] {
            assert!(propagates(VirtualType::Pauli, gate), "{gate:?}");
        }
    }

    #[test]
    fn test_pauli_does_not_survive_non_cliffords() {
        // These are exactly the cases that would otherwise produce a silently wrong randomization.
        for gate in [StandardGate::T, StandardGate::RZ, StandardGate::RX] {
            assert!(!propagates(VirtualType::Pauli, gate), "{gate:?}");
        }
    }

    #[test]
    fn test_u2_only_survives_single_qubit_gates() {
        assert!(propagates(VirtualType::U2, StandardGate::RZ));
        assert!(propagates(VirtualType::U2, StandardGate::H));
        // an entangler takes a local U2 element out of the local group
        assert!(!propagates(VirtualType::U2, StandardGate::CX));
    }

    #[test]
    fn test_c1_matches_the_pauli_gate_set() {
        for gate in [StandardGate::H, StandardGate::CX, StandardGate::RZZ] {
            assert_eq!(
                propagates(VirtualType::C1, gate),
                propagates(VirtualType::Pauli, gate),
                "{gate:?}"
            );
        }
    }
}
