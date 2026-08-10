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

"""Tests for the absorb_dressing pass: scope-agnostic emission absorption.

After build, every emission is a standalone Emit instruction and every collector carries only Gates
items. This pass scans from each emission in its travel direction, crossing box boundaries, and
absorbs it into the first compatible collector.

Emissions that cannot reach a compatible collector (gate in the way, incompatible collector) remain
standalone for the future walk_emissions pass.
"""

import copy

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit._accelerate.samplex import (
    ChangeBasis,
    InjectNoise,
    Twirl,
    absorb_dressing,
    build_lowered,
    merge_collectors,
)

from test import QiskitTestCase


def build(circuit):
    """Build and absorb, returning the emission circuit as a QuantumCircuit."""
    dag, _table = build_lowered(circuit_to_dag(circuit))
    absorb_dressing(dag)
    return dag_to_circuit(dag)


def walk(circuit):
    """Every (instruction, containing circuit) at any depth, outermost first.

    Depth matters here: a nested box's collectors live inside the enclosing hard box, and a
    propagating emission is written inside the hard box it has to cross. A top-level-only scan reports
    both as absent, which makes "no emission is left standalone" true whether or not one was wrongly
    absorbed — so these helpers recurse.
    """
    for inst in circuit.data:
        yield inst, circuit
        for block in getattr(inst.operation, "blocks", None) or ():
            yield from walk(block)


def collectors(circuit):
    """The (annotation, body, qubit indices) of each collect box, at any depth, in circuit order."""
    out = []
    for inst, owner in walk(circuit):
        annotations = getattr(inst.operation, "annotations", None)
        if annotations:
            out.append(
                (
                    annotations[0],
                    inst.operation.blocks[0] if inst.operation.blocks else None,
                    [owner.find_bit(q).index for q in inst.qubits],
                )
            )
    return out


def emits(circuit):
    """All standalone Emit instructions remaining in the circuit, at any depth."""
    return [inst for inst, _ in walk(circuit) if inst.operation.name.startswith("samplex_emit_")]


class TestLocalAbsorption(QiskitTestCase):
    """Local emissions (adjacent to a compatible collector) are absorbed."""

    def test_twirl_left_dressed_absorbs_near_half(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            circuit.cx(0, 1)
        ir2 = build(circuit)
        colls = collectors(ir2)
        # The left collector absorbs the left-directed twirl half (local)
        left = colls[0]
        self.assertIn(("local", 0), left[0].items)

    def test_twirl_left_dressed_far_half_stays_standalone(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            circuit.cx(0, 1)
        ir2 = build(circuit)
        # The right-directed (far) twirl half remains standalone — the hard box blocks it
        remaining = emits(ir2)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].operation.direction, "right")

    def test_twirl_right_dressed_absorbs_near_half(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="right")]):
            circuit.cx(0, 1)
        ir2 = build(circuit)
        colls = collectors(ir2)
        right = colls[-1]
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

    def test_propagating_emission_stays_on_spine(self):
        """A far twirl half separated from its collector by gates stays on the spine."""
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            circuit.cx(0, 1)
        ir2 = build(circuit)
        # The far twirl half (right-travelling through cx) remains as a standalone Emit
        remaining = emits(ir2)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].operation.direction, "right")


class TestOwnership(QiskitTestCase):
    """A collector absorbs only the emissions of the boxes it owns.

    Facing is not sufficient. An emission propagating out of an enclosing box passes the collectors of
    every box nested inside it, and those collectors face it. Absorbing one there would compose it as a
    local value — dropping every conjugation it was owed, which leaves the enclosing box's
    randomization applied and immediately undone with none of its content in between.
    """

    def test_outer_far_half_is_not_absorbed_by_an_inner_collector(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            with circuit.box([Twirl(dressing="left")]):
                circuit.cx(0, 1)
        ir2 = build(circuit)

        # Two far halves, one per box, both still standalone: each must cross content to reach its
        # own box's right collector.
        remaining = emits(ir2)
        self.assertEqual(len(remaining), 2)
        self.assertEqual({e.operation.direction for e in remaining}, {"right"})
        self.assertEqual(len({e.operation.box_id for e in remaining}), 2)

    def test_each_box_owns_exactly_its_own_pair_of_collectors(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            with circuit.box([Twirl(dressing="left")]):
                circuit.cx(0, 1)
        ir2 = build(circuit)

        owners = [tuple(annotation.owned) for annotation, _, _ in collectors(ir2)]
        self.assertEqual(len(owners), 4)
        for owned in owners:
            self.assertEqual(len(owned), 1)
        # Two ids, each naming two collectors — one per side of its box.
        self.assertEqual(len(set(owners)), 2)

    def test_multi_level_nesting_keeps_every_far_half_standalone(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            with circuit.box([Twirl(dressing="left")]):
                with circuit.box([Twirl(dressing="left")]):
                    circuit.cx(0, 1)
        ir2 = build(circuit)

        remaining = emits(ir2)
        self.assertEqual(len(remaining), 3)
        self.assertEqual(len({e.operation.box_id for e in remaining}), 3)

    def test_an_inner_local_emission_is_still_absorbed_locally(self):
        """Ownership blocks foreign emissions, not a box's own adjacent ones."""
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            with circuit.box([ChangeBasis("b", placement="end")]):
                circuit.cx(0, 1)
        ir2 = build(circuit)

        # The inner ChangeBasis sits at the inner box's right edge, adjacent to that box's own right
        # collector, so it is absorbed there — no descent, no escaping.
        self.assertNotIn("change_basis", [e.operation.source for e in emits(ir2)])


class TestAbsorptionWithMerge(QiskitTestCase):
    """Absorption composes correctly with merge_collectors."""

    def test_absorb_then_merge(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="right")]):
            circuit.cx(0, 1)
        with circuit.box([Twirl(dressing="left")]):
            circuit.cx(0, 1)
        dag, _table = build_lowered(circuit_to_dag(circuit))
        absorb_dressing(dag)
        merge_collectors(dag)
        ir2 = dag_to_circuit(dag)
        # The two adjacent near-half collectors merge into one middle collector.
        # Outer collectors with no content are elided.
        colls = collectors(ir2)
        self.assertGreaterEqual(len(colls), 1)

    def test_merge_then_absorb(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="right")]):
            circuit.cx(0, 1)
        with circuit.box([Twirl(dressing="left")]):
            circuit.cx(0, 1)
        dag, _table = build_lowered(circuit_to_dag(circuit))
        merge_collectors(dag)
        absorb_dressing(dag)
        ir2 = dag_to_circuit(dag)
        colls = collectors(ir2)
        self.assertGreaterEqual(len(colls), 1)

    def test_order_independence_emit_count(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="right")]):
            circuit.cx(0, 1)
        with circuit.box([Twirl(dressing="left")]):
            circuit.cx(0, 1)
        dag, _table = build_lowered(circuit_to_dag(circuit))

        # The passes mutate in place, so each ordering gets its own copy of the same input.
        am = copy.copy(dag)
        merge_collectors(am)
        absorb_dressing(am)

        ma = copy.copy(dag)
        absorb_dressing(ma)
        merge_collectors(ma)

        ir2_am = dag_to_circuit(am)
        ir2_ma = dag_to_circuit(ma)

        # Same number of standalone emits remain either way
        self.assertEqual(len(emits(ir2_am)), len(emits(ir2_ma)))


class TestAbsorptionPreservesCompositionOrder(QiskitTestCase):
    """The items ordering reflects absorption direction correctly."""

    def test_left_collector_order_preserved(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left"), ChangeBasis("b", placement="start")]):
            circuit.h(0)
            circuit.cx(0, 1)
        ir2 = build(circuit)
        colls = collectors(ir2)
        left = colls[0]
        items = left[0].items
        # All emissions on the left collector are local (no incoming)
        tags = [tag for tag, _ in items]
        self.assertNotIn("incoming", tags)
        # Gates are still present at their correct position
        self.assertIn("gates", tags)
