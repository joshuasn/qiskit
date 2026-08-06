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

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit._accelerate.samplex import (
    ChangeBasis,
    Twirl,
    build_lowered,
    merge_collectors,
)

from test import QiskitTestCase
from test.python.samplex.test_build import collectors, emissions, gate_names, hard_boxes


def build(circuit):
    """The emission circuit before merging."""
    data, table = build_lowered(circuit_to_dag(circuit))
    return QuantumCircuit._from_circuit_data(data), table


def merged(circuit):
    """The emission circuit after merging."""
    data, table = build_lowered(circuit_to_dag(circuit))
    return QuantumCircuit._from_circuit_data(merge_collectors(data)), table


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

        bodies = [gate_names(body) for _, body, _ in collectors(out)]
        # first box's absorbed `s` and second box's absorbed `h` land in the same middle collector
        self.assertIn(["s", "h"], bodies)
        middle = next(c for c in collectors(out) if gate_names(c[1]) == ["s", "h"])
        self.assertEqual(middle[2], [0, 1, 2, 3])
        # and they are still on the right qubits after remapping
        body = middle[1]
        self.assertEqual(
            [(i.operation.name, [body.find_bit(b).index for b in i.qubits]) for i in body.data],
            [("s", [0]), ("h", [3])],
        )


    def test_a_merged_collector_holds_one_run_per_contribution(self):
        # Items and bodies both append, in the same order, so the counts stay valid without any
        # offsetting — which is why they are counts rather than index ranges. The resulting sequence is
        # right: the first box's outermost element ends up adjacent to the second box's outermost.
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="right")]):
            circuit.cx(0, 1)
            circuit.s(0)
        with circuit.box([Twirl(dressing="left")]):
            circuit.h(1)
            circuit.cx(0, 1)
        out, _ = merged(circuit)

        middle = next(c for c, body, _ in collectors(out) if len(body.data) > 1)
        # After build: collectors have only Gates items. The merged middle holds gates from both boxes.
        self.assertEqual(middle.items, [("gates", 1), ("gates", 1)])

    def test_merging_keeps_items_and_bodies_in_step(self):
        circuit = notebook_circuit()
        out, _ = merged(circuit)
        for annotation, body, _ in collectors(out):
            counted = sum(n for kind, n in annotation.items if kind == "gates")
            self.assertEqual(counted, len(body.data), repr(annotation))


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
        # to be recorded. See the nesting section of SAMPLEX_IR_DESIGN.md.
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
        circuit = notebook_circuit()
        before, _ = build(circuit)
        after, _ = merged(circuit)

        self.assertEqual(
            [(e.source, e.direction, tuple(e.qubits)) for e in emissions(before)],
            [(e.source, e.direction, tuple(e.qubits)) for e in emissions(after)],
        )

    def test_emissions_are_preserved_through_merge(self):
        circuit = notebook_circuit()
        before, _ = build(circuit)
        out, _ = merged(circuit)

        # All emissions remain in the circuit (merge doesn't touch them)
        self.assertEqual(
            [(e.source, e.direction) for e in emissions(before)],
            [(e.source, e.direction) for e in emissions(out)],
        )

    def test_hard_content_survives(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
        with circuit.box([Twirl()]):
            circuit.cx(1, 0)
        out, _ = merged(circuit)
        self.assertEqual([gate_names(h) for h in hard_boxes(out)], [["cx"], ["cx"]])

    def test_merge_is_deterministic(self):
        runs = []
        for _ in range(3):
            out, table = merged(notebook_circuit())
            runs.append(
                (
                    gate_names(out),
                    [(c.synthesizer, tuple(c.items), tuple(q)) for c, _, q in collectors(out)],
                    table.entries(),
                )
            )
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(runs[0], runs[2])

    def test_merging_is_idempotent(self):
        data, _ = build_lowered(circuit_to_dag(notebook_circuit()))
        once = merge_collectors(data)
        twice = merge_collectors(once)

        def shape(circuit_data):
            circuit = QuantumCircuit._from_circuit_data(circuit_data)
            return [
                (c.synthesizer, tuple(c.items), tuple(q)) for c, _, q in collectors(circuit)
            ]

        self.assertEqual(shape(once), shape(twice))
