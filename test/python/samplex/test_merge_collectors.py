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

"""Tests for the samplex merge_collectors pass: emission circuit (IR2) -> emission circuit (IR2).

Build is local, so every annotated box gets its own two collectors. This pass applies the contextual
collection model, letting adjacent boxes that share a synthesizer share a middle collector.
"""

import copy

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit._accelerate.samplex import (
    ChangeBasis,
    Twirl,
    absorb_dressing,
    build_lowered,
    merge_collectors,
)

from test import QiskitTestCase
from test.python.samplex.test_build import collectors, emissions, gate_names, hard_boxes, real_gates


def build(circuit):
    """The emission circuit before merging."""
    dag, table = build_lowered(circuit_to_dag(circuit))
    return dag_to_circuit(dag), table


def merged(circuit):
    """The emission circuit after merging (absorb happens after merge)."""
    dag, table = build_lowered(circuit_to_dag(circuit))
    merge_collectors(dag)
    absorb_dressing(dag)
    return dag_to_circuit(dag), table


def notebook_circuit():
    """A wide box followed by two narrow ones covering its halves."""
    circuit = QuantumCircuit(4)
    with circuit.box([Twirl(), ChangeBasis("ref")]):
        circuit.noop(*range(4))
    with circuit.box([Twirl()]):
        circuit.noop(0, 1)
    with circuit.box([Twirl()]):
        circuit.noop(2, 3)
    return circuit


class TestSharedMiddleCollector(QiskitTestCase):
    """Adjacent boxes sharing a synthesizer share a collector."""

    def test_two_boxes_share_one_middle_collector(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)

        self.assertEqual(len(collectors(build(circuit)[0])), 4)
        out, _ = merged(circuit)
        got = collectors(out)
        # Adjacent empty collectors merge: left-outer + right-A + left-B merge into one, etc.
        # With empty collectors (no Incoming), all same-synthesizer adjacent collectors fuse.
        # Result: the 4 collectors reduce to 3 (left, merged-middle, right).
        self.assertEqual(len(got), 3)

    def test_wide_box_then_two_narrow_share_a_full_width_collector(self):
        # The wide box's right collector stays open on q2-3 after the first narrow box claims q0-1,
        # so the second narrow box still merges into it. This is what detaching (rather than
        # closing) on an emission buys.
        self.assertEqual(len(collectors(build(notebook_circuit())[0])), 6)
        out, _ = merged(notebook_circuit())
        got = collectors(out)
        self.assertEqual(len(got), 4)

        middle = got[1]
        self.assertEqual(middle[2], [0, 1, 2, 3])

    def test_wide_then_two_narrow_merges_with_real_content(self):
        # The same shape as above but with gates, and with the middle box right-dressed. That layout is
        # `COLLECT, HARD, EMIT`, so the hard box reaches the wide collector while its frontier is still
        # whole — the case an earlier version got wrong by clearing the whole frontier instead of
        # releasing q0-1, which silently cost a collector.
        circuit = QuantumCircuit(4)
        with circuit.box([Twirl(), ChangeBasis("ref")]):
            circuit.cx(0, 1)
            circuit.cx(2, 3)
        with circuit.box([Twirl(dressing="right")]):
            circuit.cx(0, 1)
        with circuit.box([Twirl()]):
            circuit.cx(2, 3)

        self.assertEqual(len(collectors(build(circuit)[0])), 6)
        out, _ = merged(circuit)
        got = collectors(out)
        self.assertEqual(len(got), 4)
        self.assertEqual([q for _, _, q in got], [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1], [2, 3]])

    def test_widening_onto_a_touched_wire_is_refused(self):
        # Merging must not reach back onto a wire something already crossed: the emission's walk would
        # pick up a conjugation by that gate, which costs a propagation step and, for a non-Clifford,
        # would be refused outright. Overlap on q0-1 alone is not enough to license widening onto q2.
        circuit = QuantumCircuit(4)
        with circuit.box([Twirl()]):
            circuit.noop(0, 1)
        circuit.h(2)
        with circuit.box([Twirl()]):
            circuit.noop(0, 1, 2, 3)
        out, _ = merged(circuit)

        self.assertEqual([q for _, _, q in collectors(out)], [[0, 1], [0, 1], [0, 1, 2, 3], [0, 1, 2, 3]])

    def test_merging_widens_the_collector(self):
        # Narrow-then-wider: the merged collector must cover the union, not just where it started.
        circuit = QuantumCircuit(4)
        with circuit.box([Twirl()]):
            circuit.noop(0, 1)
        with circuit.box([Twirl()]):
            circuit.noop(*range(4))
        out, _ = merged(circuit)
        got = collectors(out)

        self.assertEqual([q for _, _, q in got], [[0, 1], [0, 1, 2, 3], [0, 1, 2, 3]])

    def test_absorbed_gates_survive_a_widening_merge(self):
        # The two contributions have different widths, so each body has to be remapped into the
        # merged frame rather than copied.
        circuit = QuantumCircuit(4)
        with circuit.box([Twirl(dressing="right")]):
            circuit.cx(0, 1)
            circuit.s(0)
        # overlaps the first box on q0-1, so the two collectors fuse and the span widens 2 -> 4
        with circuit.box([Twirl(dressing="left")]):
            circuit.h(3)
            circuit.cx(2, 3)
            circuit.cx(0, 1)
        out, _ = merged(circuit)

        bodies = [real_gates(body) for _, body, _ in collectors(out)]
        # first box's absorbed `s` and second box's absorbed `h` land in the same middle collector
        self.assertIn(["s", "h"], bodies)
        middle = next(c for c in collectors(out) if real_gates(c[1]) == ["s", "h"])
        self.assertEqual(middle[2], [0, 1, 2, 3])
        # and they are still on the right qubits after remapping
        body = middle[1]
        self.assertEqual(
            [
                (i.operation.name, [body.find_bit(b).index for b in i.qubits])
                for i in body.data
                if not i.operation.name.startswith("samplex_emit")
            ],
            [("s", [0]), ("h", [3])],
        )


class TestMergeBarriers(QiskitTestCase):
    """What stops two collectors fusing."""

    def test_a_boxs_own_two_collectors_never_merge(self):
        # They sit either side of the twirl point, so fusing them would cancel the randomization.
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
        out, _ = merged(circuit)
        self.assertEqual(len(collectors(out)), 2)

    def test_empty_bodied_box_still_keeps_two_collectors(self):
        # Nothing between the emissions, so this is the case most at risk of over-merging.
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl()]):
            circuit.noop(0, 1)
        out, _ = merged(circuit)
        self.assertEqual(len(collectors(out)), 2)

    def test_bare_gate_between_boxes_blocks_merging(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
        circuit.h(0)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
        out, _ = merged(circuit)
        self.assertEqual(len(collectors(out)), 4)

    def test_incompatible_synthesizers_do_not_merge(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(decomposition="rzsx")]):
            circuit.cx(0, 1)
        with circuit.box([Twirl(decomposition="rzrx")]):
            circuit.cx(0, 1)
        out, _ = merged(circuit)
        got = collectors(out)
        self.assertEqual(len(got), 4)
        self.assertEqual([c.synthesizer for c, _, _ in got], ["rzsx", "rzsx", "rzrx", "rzrx"])

    def test_disjoint_qubits_do_not_merge(self):
        circuit = QuantumCircuit(4)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
        with circuit.box([Twirl()]):
            circuit.cx(2, 3)
        out, _ = merged(circuit)
        self.assertEqual(len(collectors(out)), 4)


class TestScopes(QiskitTestCase):
    """Merging is confined to one box scope."""

    def test_nested_collector_is_not_promoted_out_of_its_box(self):
        # Cross-boundary merging is deliberately out of scope: it would take the inner box's
        # absorbed gates off the spine, so the outer factor's propagation through them would have
        # to be recorded.
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl()]):
            with circuit.box([Twirl()]):
                circuit.cx(0, 1)
        out, _ = merged(circuit)

        self.assertEqual(len(collectors(out)), 2)
        (hard,) = hard_boxes(out)
        self.assertEqual(len(collectors(hard)), 2)

    def test_siblings_inside_a_box_still_merge(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl()]):
            with circuit.box([Twirl()]):
                circuit.cx(0, 1)
            with circuit.box([Twirl()]):
                circuit.cx(0, 1)
        out, _ = merged(circuit)

        (hard,) = hard_boxes(out)
        # the two inner boxes share a middle collector, just as siblings do at the top level
        self.assertEqual(len(collectors(hard)), 3)


class TestPreservation(QiskitTestCase):
    """What merging must not change."""

    def test_emissions_and_content_are_untouched(self):
        # Propagating emissions (those not absorbed) survive merge unchanged.
        circuit = notebook_circuit()
        dag, _ = build_lowered(circuit_to_dag(circuit))
        no_merge = copy.copy(dag)
        absorb_dressing(no_merge)
        with_merge = copy.copy(dag)
        merge_collectors(with_merge)
        absorb_dressing(with_merge)

        before = dag_to_circuit(no_merge)
        after = dag_to_circuit(with_merge)

        # Only propagating (standalone) emissions have a position the test can pin down. An absorbed
        # local emission lives in a collector body alongside other content on disjoint qubits, where
        # relative order is deliberately unconstrained — see lower.rs's topological-order argument.
        self.assertEqual(
            [(e.source, e.direction, tuple(e.qubits)) for e in emissions(before) if e.direction != "local"],
            [(e.source, e.direction, tuple(e.qubits)) for e in emissions(after) if e.direction != "local"],
        )

    def test_emissions_are_preserved_through_merge(self):
        # Whether or not merge runs, the same propagating emissions remain standalone.
        circuit = notebook_circuit()
        dag, _ = build_lowered(circuit_to_dag(circuit))
        no_merge = copy.copy(dag)
        absorb_dressing(no_merge)
        with_merge = copy.copy(dag)
        merge_collectors(with_merge)
        absorb_dressing(with_merge)

        before = dag_to_circuit(no_merge)
        after = dag_to_circuit(with_merge)

        self.assertEqual(
            [(e.source, e.direction) for e in emissions(before)],
            [(e.source, e.direction) for e in emissions(after)],
        )

    def test_hard_content_survives(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
        with circuit.box([Twirl()]):
            circuit.cx(1, 0)
        out, _ = merged(circuit)
        self.assertEqual([real_gates(h) for h in hard_boxes(out)], [["cx"], ["cx"]])

    def test_merge_is_deterministic(self):
        runs = []
        for _ in range(3):
            out, table = merged(notebook_circuit())
            runs.append(
                (
                    gate_names(out),
                    [(c.synthesizer, tuple(gate_names(b)), tuple(q)) for c, b, q in collectors(out)],
                    table.entries(),
                )
            )
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(runs[0], runs[2])

    def test_merging_is_idempotent(self):
        once, _ = build_lowered(circuit_to_dag(notebook_circuit()))
        merge_collectors(once)
        twice = copy.copy(once)
        merge_collectors(twice)

        def shape(dag):
            circuit = dag_to_circuit(dag)
            return [(c.synthesizer, tuple(gate_names(b)), tuple(q)) for c, b, q in collectors(circuit)]

        self.assertEqual(shape(once), shape(twice))


class TestOwnership(QiskitTestCase):
    """A merged collector answers for every box that contributed to it."""

    def test_a_shared_middle_collector_owns_both_boxes(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
        out, _ = merged(circuit)

        owners = [list(annotation.owned) for annotation, _, _ in collectors(out)]
        # Three collectors for two boxes: the middle one is shared, so it may take either box's
        # emissions. Anything narrower would leave one of them unable to be collected.
        self.assertEqual(len(owners), 3)
        self.assertEqual([len(o) for o in owners], [1, 2, 1])
        self.assertEqual(owners[1], sorted(set(owners[0] + owners[2])))

    def test_merging_order_does_not_change_the_owned_set(self):
        """Sorted and deduplicated, so two runs produce identical IR2."""
        circuit = QuantumCircuit(4)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
        with circuit.box([Twirl()]):
            circuit.cx(1, 2)
        with circuit.box([Twirl()]):
            circuit.cx(2, 3)
        runs = [[tuple(a.owned) for a, _, _ in collectors(merged(circuit)[0])] for _ in range(3)]
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(runs[0], runs[2])
        for owned in runs[0]:
            self.assertEqual(list(owned), sorted(set(owned)))
