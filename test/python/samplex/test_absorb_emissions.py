# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the absorb_emissions pass: local emissions become inline data on collectors.

After build, every emission is a standalone Emit instruction and every collector references it as
("incoming", id). This pass absorbs emissions that are local — adjacent to their collector with the
direction pointing toward it — into ("local", 0) entries, removing the standalone instruction.

Only the far twirl half (which propagates through the hard box) remains as a standalone Emit.
"""

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit._accelerate.samplex import (
    ChangeBasis,
    InjectNoise,
    Twirl,
    absorb_emissions,
    build_lowered,
    merge_collectors,
)

from test import QiskitTestCase


def build(circuit):
    """Build and absorb, returning the emission circuit as a QuantumCircuit."""
    data, _table = build_lowered(circuit_to_dag(circuit))
    data = absorb_emissions(data)
    return QuantumCircuit._from_circuit_data(data)


def collectors(circuit):
    """The (annotation, body, qubit indices) of each collect box, in circuit order."""
    out = []
    for inst in circuit.data:
        annotations = getattr(inst.operation, "annotations", None)
        if annotations:
            out.append(
                (
                    annotations[0],
                    inst.operation.blocks[0] if inst.operation.blocks else None,
                    [circuit.find_bit(q).index for q in inst.qubits],
                )
            )
    return out


def emits(circuit):
    """All standalone Emit instructions remaining in the circuit."""
    return [inst for inst in circuit.data if inst.operation.name.startswith("samplex_emit_")]


class TestLocalAbsorption(QiskitTestCase):
    """Local emissions are absorbed; far twirl halves remain standalone."""

    def test_twirl_left_dressed_absorbs_near_half(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            circuit.cx(0, 1)
        ir2 = build(circuit)
        colls = collectors(ir2)
        self.assertEqual(len(colls), 2)
        left, right = colls
        # The left collector absorbs the left-directed twirl half (local)
        self.assertIn(("local", 0), left[0].items)
        # The right collector still references an incoming emit (the far half)
        self.assertTrue(any(tag == "incoming" for tag, _ in right[0].items))

    def test_twirl_left_dressed_one_standalone_emit_remains(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            circuit.cx(0, 1)
        ir2 = build(circuit)
        # Only the right-directed (far) twirl half remains as a standalone Emit
        remaining = emits(ir2)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].operation.direction, "right")

    def test_twirl_right_dressed_absorbs_near_half(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="right")]):
            circuit.cx(0, 1)
        ir2 = build(circuit)
        colls = collectors(ir2)
        right = colls[1]
        self.assertIn(("local", 0), right[0].items)
        remaining = emits(ir2)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].operation.direction, "left")

    def test_change_basis_is_fully_absorbed(self):
        circuit = QuantumCircuit(2)
        with circuit.box([ChangeBasis("b", placement="start")]):
            circuit.cx(0, 1)
        ir2 = build(circuit)
        # A ChangeBasis never propagates — it is always local to its collector
        remaining = emits(ir2)
        self.assertEqual(len(remaining), 0)
        colls = collectors(ir2)
        # The left collector should have absorbed it
        self.assertIn(("local", 0), colls[0][0].items)

    def test_inject_noise_is_absorbed(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left"), InjectNoise("n0", "after")]):
            circuit.cx(0, 1)
        ir2 = build(circuit)
        # Only the far twirl half remains; the noise injection is absorbed
        remaining = emits(ir2)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].operation.source, "twirl")

    def test_twirl_with_basis_change_absorbs_all_local(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left"), ChangeBasis("b", placement="start")]):
            circuit.cx(0, 1)
        ir2 = build(circuit)
        # ChangeBasis + the near twirl half are absorbed, far twirl half remains
        remaining = emits(ir2)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].operation.direction, "right")

    def test_collects_list_shows_only_incoming_ids(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            circuit.cx(0, 1)
        ir2 = build(circuit)
        colls = collectors(ir2)
        left, right = colls
        # Left collector has no incoming emissions (local was absorbed)
        self.assertEqual(left[0].collects, [])
        # Right collector has one incoming
        self.assertEqual(len(right[0].collects), 1)


class TestAbsorptionWithMerge(QiskitTestCase):
    """Absorption composes correctly with merge_collectors."""

    def test_absorb_then_merge(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="right")]):
            circuit.cx(0, 1)
        with circuit.box([Twirl(dressing="left")]):
            circuit.cx(0, 1)
        data, _table = build_lowered(circuit_to_dag(circuit))
        data = absorb_emissions(data)
        data = merge_collectors(data)
        ir2 = QuantumCircuit._from_circuit_data(data)
        # Should still have 3 collectors (left outer, middle merged, right outer)
        colls = collectors(ir2)
        self.assertEqual(len(colls), 3)

    def test_merge_then_absorb(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="right")]):
            circuit.cx(0, 1)
        with circuit.box([Twirl(dressing="left")]):
            circuit.cx(0, 1)
        data, _table = build_lowered(circuit_to_dag(circuit))
        data = merge_collectors(data)
        data = absorb_emissions(data)
        ir2 = QuantumCircuit._from_circuit_data(data)
        colls = collectors(ir2)
        self.assertEqual(len(colls), 3)

    def test_order_independence_emit_count(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="right")]):
            circuit.cx(0, 1)
        with circuit.box([Twirl(dressing="left")]):
            circuit.cx(0, 1)
        data, _table = build_lowered(circuit_to_dag(circuit))

        # absorb then merge
        am = absorb_emissions(merge_collectors(data))
        # merge then absorb
        ma = merge_collectors(absorb_emissions(data))

        ir2_am = QuantumCircuit._from_circuit_data(am)
        ir2_ma = QuantumCircuit._from_circuit_data(ma)

        # Same number of standalone emits remain either way
        self.assertEqual(len(emits(ir2_am)), len(emits(ir2_ma)))


class TestAbsorptionPreservesCompositionOrder(QiskitTestCase):
    """The items ordering is unchanged by absorption — only the tag kind changes."""

    def test_left_collector_order_preserved(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left"), ChangeBasis("b", placement="start")]):
            circuit.h(0)
            circuit.cx(0, 1)
        ir2 = build(circuit)
        colls = collectors(ir2)
        left = colls[0]
        items = left[0].items
        # Should have: local (basis), gates, local (inject/twirl near half)
        tags = [tag for tag, _ in items]
        # All emissions on the left collector are local (no incoming)
        self.assertNotIn("incoming", tags)
        # Gates are still present at their correct position
        self.assertIn("gates", tags)
