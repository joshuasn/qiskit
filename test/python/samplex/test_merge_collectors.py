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
    lower,
    merge_collectors,
)

from test import QiskitTestCase
from test.python.samplex.test_build import (
    body_locals,
    collectors,
    content_boxes,
    emissions,
    emissions_with_qubits,
    gate_names,
    is_collector,
    real_gates,
)


def all_collectors(circuit):
    """Every collect box at any depth, in circuit order.

    `collectors` reports one scope. Promotion moves a collector *between* scopes, so counting one scope
    would show a promotion as a collector vanishing and a refusal as one appearing.
    """
    out = []
    for inst in circuit.data:
        if is_collector(inst.operation):
            out.append(inst.operation)
        for block in getattr(inst.operation, "blocks", None) or []:
            out.extend(all_collectors(block))
    return out


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


def dressed_then_merged(circuit):
    """The emission circuit in the documented pass order: absorb, then merge.

    Promotion only has anything to do in this order. It refuses to move a collector while an emission
    inside the box is still travelling towards the one it would fold into, and before absorption *every*
    emission is still travelling — so merging an unabsorbed circuit promotes nothing.
    """
    dag, table = build_lowered(circuit_to_dag(circuit))
    absorb_dressing(dag)
    merge_collectors(dag)
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
                if not i.operation.name.startswith("emit")
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
        (hard,) = content_boxes(out)
        self.assertEqual(len(collectors(hard)), 2)

    def test_siblings_inside_a_box_still_merge(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl()]):
            with circuit.box([Twirl()]):
                circuit.cx(0, 1)
            with circuit.box([Twirl()]):
                circuit.cx(0, 1)
        out, _ = merged(circuit)

        (hard,) = content_boxes(out)
        # the two inner boxes share a middle collector, just as siblings do at the top level
        self.assertEqual(len(collectors(hard)), 3)


class TestPreservation(QiskitTestCase):
    """What merging must not change."""

    def test_emissions_and_content_are_untouched(self):
        # Propagating emissions (those not absorbed) survive merge unchanged.
        circuit = notebook_circuit()
        dag, table = build_lowered(circuit_to_dag(circuit))
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
        def propagating(circuit):
            return [
                (e.source(table), e.direction, tuple(q))
                for e, q in emissions_with_qubits(circuit)
                if e.direction != "local"
            ]

        self.assertEqual(propagating(before), propagating(after))

    def test_emissions_are_preserved_through_merge(self):
        # Whether or not merge runs, the same propagating emissions remain standalone.
        circuit = notebook_circuit()
        dag, table = build_lowered(circuit_to_dag(circuit))
        no_merge = copy.copy(dag)
        absorb_dressing(no_merge)
        with_merge = copy.copy(dag)
        merge_collectors(with_merge)
        absorb_dressing(with_merge)

        before = dag_to_circuit(no_merge)
        after = dag_to_circuit(with_merge)

        self.assertEqual(
            [(e.source(table), e.direction) for e in emissions(before)],
            [(e.source(table), e.direction) for e in emissions(after)],
        )

    def test_hard_content_survives(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
        with circuit.box([Twirl()]):
            circuit.cx(1, 0)
        out, _ = merged(circuit)
        self.assertEqual([real_gates(h) for h in content_boxes(out)], [["cx"], ["cx"]])

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


class TestSharedMiddleIsWideEnough(QiskitTestCase):
    """A merged collector has to be able to take either box's emissions.

    Nothing records which boxes contributed to it — a collector is what it covers and what it can
    synthesize — so what a merge has to preserve is that the shared middle is wide enough and
    compatible enough for both. Anything narrower would leave one of them unable to be collected.
    """

    def test_a_shared_middle_collector_covers_both_boxes(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
        out, _ = merged(circuit)

        colls = collectors(out)
        # Three collectors for two boxes: the middle one is shared.
        self.assertEqual(len(colls), 3)
        qubits = [tuple(q) for _, _, q in colls]
        self.assertEqual(qubits, [(0, 1), (0, 1), (0, 1)])
        # And it synthesizes the same way, or it could not stand in for either.
        self.assertEqual({a.synthesizer for a, _, _ in colls}, {"rzsx"})

    def test_merging_order_does_not_change_the_result(self):
        """Two runs produce identical IR2, so nothing depends on the order the walk visited members."""
        circuit = QuantumCircuit(4)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
        with circuit.box([Twirl()]):
            circuit.cx(1, 2)
        with circuit.box([Twirl()]):
            circuit.cx(2, 3)
        runs = [
            [(a.synthesizer, tuple(q)) for a, _, q in collectors(merged(circuit)[0])]
            for _ in range(3)
        ]
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(runs[0], runs[2])


class TestPromotion(QiskitTestCase):
    """A collector leaving its box, folded into the one just outside it.

    Not a contraction: a nested collector covers a subset of its box's qubits, which are a subset of the
    enclosing collector's, so nothing is widened. The outer collector keeps its width and gains a body,
    and the inner one is deleted — one dressing layer fewer per nesting level.

    Two conditions gate it. Nothing may lie between the two collectors on any wire the inner one covers,
    or the transfer would reorder its content against whatever does. And nothing inside the box may still
    be travelling towards the outer collector, because the gates being moved sit on such an emission's
    path: they are crossed today, and afterwards they would be inside its target and composed instead.
    """

    def test_a_box_with_nothing_propagating_promotes_its_inner_collector(self):
        circuit = QuantumCircuit(2)
        with circuit.box([ChangeBasis("b")]):  # no twirl, so nothing of its own propagates
            with circuit.box([Twirl()]):
                circuit.cx(0, 1)

        before, _ = build(circuit)
        after, _ = dressed_then_merged(circuit)
        self.assertEqual(len(all_collectors(before)), 4)
        self.assertEqual(len(all_collectors(after)), 3)
        # The content box stays: what is left in one is what could not be absorbed, so it still says
        # something even after a collector has left it.
        self.assertEqual(len(content_boxes(after)), 1)

    def test_the_promoted_body_holds_both_contributions_in_order(self):
        circuit = QuantumCircuit(2)
        # `placement="start"` puts the basis change on the left edge, so the outer left collector has a
        # body of its own for the promoted one to be appended to.
        with circuit.box([ChangeBasis("b", placement="start")]):
            with circuit.box([Twirl()]):
                circuit.cx(0, 1)
        after, table = dressed_then_merged(circuit)

        outer_left = collectors(after)[0]
        locals_ = body_locals(outer_left[1])
        # The outer collector's own content first and the promoted content after it, because that is the
        # order the two sat in: its basis change, then the inner box's near half.
        self.assertEqual([op.source(table) for op in locals_], ["change_basis", "twirl"])

    def test_both_sides_promote_when_nothing_propagates_at_all(self):
        """With no twirl anywhere there is nothing travelling, so neither side is refused.

        This is also the only shape that exercises a *leftward* promotion, where the inner collector sits
        first and so its content has to be composed first.
        """
        circuit = QuantumCircuit(2)
        with circuit.box([ChangeBasis("outer", placement="end")]):
            with circuit.box([ChangeBasis("inner", placement="end")]):
                circuit.noop(0, 1)

        before, _ = build(circuit)
        after, _ = dressed_then_merged(circuit)
        self.assertEqual(len(all_collectors(before)), 4)
        self.assertEqual(len(all_collectors(after)), 2)

        # Both basis changes end up in the outer right collector, innermost first: the inner collector
        # sat before the outer one, so its contribution composes before it. Build interns the enclosing
        # box's distribution first, so the inner one's key is the higher of the two.
        outer_right = collectors(after)[-1]
        keys = [op.distribution_key for op in body_locals(outer_right[1])]
        self.assertEqual(keys, [1, 0])

    def test_a_nested_twirl_promotes_on_its_dressing_side(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            with circuit.box([Twirl(dressing="left")]):
                circuit.cx(0, 1)

        after, _ = dressed_then_merged(circuit)
        # Four collectors become three. The inner box's left collector had nothing travelling towards the
        # outer left one — both near halves are already absorbed and the inner far half travels the other
        # way — so it can leave.
        self.assertEqual(len(all_collectors(after)), 3)

    def test_the_far_half_blocks_promotion_on_the_side_it_travels_towards(self):
        """The inner right collector stays: the inner far half is heading for the outer right one.

        Promoting it would re-point that emission at the outer collector and move the gates it crosses
        into it, so the conjugation it is owed would be composed rather than crossed.
        """
        circuit = QuantumCircuit(2)
        with circuit.box([ChangeBasis("b")]):
            with circuit.box([Twirl(dressing="left")]):
                circuit.cx(0, 1)
        after, _ = dressed_then_merged(circuit)

        # Three, not two: the left side promoted and the right side did not.
        self.assertEqual(len(all_collectors(after)), 3)
        # And the inner far half is still travelling.
        self.assertEqual(len([e for e in emissions(after) if e.direction != "local"]), 1)

    def test_an_emission_in_between_blocks_promotion(self):
        """The outer box's own far half is standalone and sits between the two collectors."""
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            circuit.cx(0, 1)  # keeps the outer far half from being absorbed
            with circuit.box([Twirl(dressing="left")]):
                circuit.cx(0, 1)

        after, _ = dressed_then_merged(circuit)
        self.assertEqual(len(all_collectors(after)), 4)

    def test_promotion_is_transitive_across_two_levels(self):
        """Taking one collector out exposes the next level down, which the fixed point then takes."""
        circuit = QuantumCircuit(2)
        with circuit.box([ChangeBasis("b0")]):
            with circuit.box([ChangeBasis("b1")]):
                with circuit.box([Twirl()]):
                    circuit.cx(0, 1)

        before, _ = build(circuit)
        after, _ = dressed_then_merged(circuit)
        self.assertEqual(len(all_collectors(before)), 6)
        # Both left-hand collectors reached the outermost one, one round each.
        self.assertEqual(len(all_collectors(after)), 4)

    def test_promotion_keeps_every_conjugation(self):
        """The check that matters: a promotion must not take a gate off an emission's path.

        The template evaluates to the same unitary either way, so a lost conjugation shows up only as a
        missing `Propagate` node.
        """
        circuit = QuantumCircuit(2)
        with circuit.box([ChangeBasis("b")]):
            with circuit.box([Twirl(dressing="left")]):
                circuit.h(0)
                circuit.cx(0, 1)

        def propagates(merge):
            dag, table = build_lowered(circuit_to_dag(circuit))
            absorb_dressing(dag)
            if merge:
                merge_collectors(dag)
            _, graph, _ = lower(dag, table)
            return sorted(node[0] for node in graph.nodes() if node[0].startswith("propagate:"))

        self.assertEqual(propagates(merge=True), propagates(merge=False))
        # And there was something to preserve.
        self.assertTrue(propagates(merge=False))
