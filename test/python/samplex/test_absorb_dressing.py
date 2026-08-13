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

After build, every emission is a standalone Emit instruction and every collector's body is empty.
This pass scans from each emission in its travel direction, crossing box boundaries, and absorbs it
into the first compatible collector — as a real instruction inserted into that collector's body.

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
    """Every emission still *travelling*, at any depth — the ones absorption did not resolve.

    Absorbed emissions are real `Emit` instructions too, sitting in their collector's body with
    `direction == "local"`, so a name-only filter would count them as unabsorbed and make every
    "nothing is left standalone" assertion vacuous. What is left standalone is what still has a
    direction.
    """
    return [
        inst
        for inst, _ in walk(circuit)
        if inst.operation.name.startswith("emit") and inst.operation.direction != "local"
    ]


def body_locals(body):
    """Absorbed local emissions in a collector body, in body order."""
    return [
        inst.operation
        for inst in body.data
        if inst.operation.name.startswith("emit") and inst.operation.direction == "local"
    ]


def real_gates(body):
    """Gate names in a collector body, excluding absorbed local emissions."""
    return [inst.operation.name for inst in body.data if not inst.operation.name.startswith("emit")]


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
        self.assertTrue(body_locals(left[1]))

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
        self.assertTrue(body_locals(right[1]))
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
        self.assertTrue(body_locals(colls[0][1]))

    def test_inject_noise_is_absorbed(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left"), InjectNoise("n0", "after")]):
            circuit.cx(0, 1)
        dag, table = build_lowered(circuit_to_dag(circuit))
        absorb_dressing(dag)
        ir2 = dag_to_circuit(dag)
        # Only the far twirl half remains; the noise injection is absorbed
        remaining = emits(ir2)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].operation.source(table), "twirl")

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


class TestNearestCompatibleCollectorWins(QiskitTestCase):
    """Absorption asks whether a collector *can* take an emission, not whose box it came from.

    An emission propagating out of an enclosing box passes the collectors of every box nested inside
    it, and those collectors face it, so the nearest one takes it. Nothing consulted here says
    otherwise: there is no id on an emission naming the box it came from, and scope does not separate
    the cases either — an enclosing box's propagating emission sits in the *same* scope as the
    collectors of every box nested inside it.

    **This is provisional, and these tests pin it deliberately.** For a nested twirl of the same group
    it terminates the enclosing randomization at the inner dressing, with none of the enclosing box's
    content in between — invisible to a round-trip test, since the circuit still evaluates to the same
    unitary. The discrimination is meant to live in *compatibility* rather than in position: once an
    emission carries a type an inner collector can decline, that collector will decline and the
    emission will carry on to one that can take it. Pinning the current answer means the change of rule
    shows up as a test change rather than silently. See `lower::compatible`.
    """

    def test_an_inner_collector_takes_the_enclosing_far_half(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            with circuit.box([Twirl(dressing="left")]):
                circuit.cx(0, 1)
        ir2 = build(circuit)

        # Only the innermost far half is left travelling: it is the one with content in the way. The
        # enclosing box's far half was adjacent to the inner box's left collector, which took it.
        remaining = emits(ir2)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].operation.direction, "right")

        # Two locals in one body is the tell: the inner box's left collector holds its own near half and
        # the enclosing box's far half, which should have crossed the inner box's content.
        local_counts = [len(body_locals(body)) for _, body, _ in collectors(ir2) if body is not None]
        self.assertEqual(sorted(local_counts), [0, 0, 1, 2])

    def test_every_level_of_nesting_hands_its_far_half_inward(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            with circuit.box([Twirl(dressing="left")]):
                with circuit.box([Twirl(dressing="left")]):
                    circuit.cx(0, 1)
        ir2 = build(circuit)

        # One per level under the old rule; one in total under this one.
        remaining = emits(ir2)
        self.assertEqual(len(remaining), 1)

    def test_an_inner_local_emission_is_still_absorbed_locally(self):
        """A box's own adjacent emissions are absorbed as before."""
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            with circuit.box([ChangeBasis("b", placement="end")]):
                circuit.cx(0, 1)
        dag, table = build_lowered(circuit_to_dag(circuit))
        absorb_dressing(dag)
        ir2 = dag_to_circuit(dag)

        # The inner ChangeBasis sits at the inner box's right edge, adjacent to that box's own right
        # collector, so it is absorbed there — no descent, no escaping.
        self.assertNotIn("change_basis", [e.operation.source(table) for e in emits(ir2)])


class TestAbsorptionWithMerge(QiskitTestCase):
    """Absorption composes with `merge_collectors` either way round, but not equally well.

    Both orders produce valid IR2 — an unmerged or unabsorbed circuit is unoptimized, not wrong. What
    differs is how much merging finds: two collectors merge only with nothing between them, and
    absorption is what clears what is. So absorbing first can only help, and on a nested box it halves
    the dressing layers.
    """

    def two_siblings(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="right")]):
            circuit.cx(0, 1)
        with circuit.box([Twirl(dressing="left")]):
            circuit.cx(0, 1)
        return circuit

    def nested(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            with circuit.box([Twirl(dressing="left")]):
                circuit.cx(0, 1)
        return circuit

    def both_orders(self, circuit):
        """`(absorb-then-merge, merge-then-absorb)` for the same input."""
        dag, _table = build_lowered(circuit_to_dag(circuit))
        # The passes mutate in place, so each ordering gets its own copy.
        first = copy.copy(dag)
        absorb_dressing(first)
        merge_collectors(first)
        second = copy.copy(dag)
        merge_collectors(second)
        absorb_dressing(second)
        return dag_to_circuit(first), dag_to_circuit(second)

    def test_the_orders_agree_without_nesting(self):
        absorbed_first, merged_first = self.both_orders(self.two_siblings())
        self.assertEqual(len(collectors(absorbed_first)), len(collectors(merged_first)))

    def test_absorbing_first_merges_at_least_as_much(self):
        for name, circuit in (("siblings", self.two_siblings()), ("nested", self.nested())):
            with self.subTest(shape=name):
                absorbed_first, merged_first = self.both_orders(circuit)
                self.assertLessEqual(
                    len(collectors(absorbed_first)),
                    len(collectors(merged_first)),
                    "absorbing first cannot merge less: it only removes things from between collectors",
                )

    def test_nesting_is_where_the_order_costs_something(self):
        absorbed_first, merged_first = self.both_orders(self.nested())
        self.assertEqual(len(collectors(absorbed_first)), 2)
        self.assertEqual(len(collectors(merged_first)), 4)

    def test_the_standalone_emission_count_is_order_independent(self):
        """Merging changes how many collectors there are, never how many emissions still travel."""
        for name, circuit in (("siblings", self.two_siblings()), ("nested", self.nested())):
            with self.subTest(shape=name):
                absorbed_first, merged_first = self.both_orders(circuit)
                self.assertEqual(len(emits(absorbed_first)), len(emits(merged_first)))


class TestAbsorptionPreservesCompositionOrder(QiskitTestCase):
    """The body ordering reflects absorption direction correctly."""

    def test_left_collector_order_preserved(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left"), ChangeBasis("b", placement="start")]):
            circuit.h(0)
            circuit.cx(0, 1)
        ir2 = build(circuit)
        colls = collectors(ir2)
        left = colls[0]
        # The absorbed hard-content gate and the absorbed local emission both land in the left
        # collector's body; nothing propagating (incoming) is absorbed here.
        self.assertEqual(real_gates(left[1]), ["h"])
        self.assertTrue(body_locals(left[1]))
