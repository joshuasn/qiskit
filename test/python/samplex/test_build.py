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

"""Tests for the samplex build pass: annotated circuit (IR1) -> emission circuit (IR2).

The build pass is deliberately *local*: every annotated box yields two collect boxes consuming only
its own emissions. Merging adjacent collectors belongs to a later pass, so these tests assert the
unmerged shape. They assert on structure rather than rendered output, because rendered graph output
is not a stable artifact.
"""

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit._accelerate.samplex import (
    ChangeBasis,
    InjectLocalClifford,
    InjectNoise,
    Tag,
    Twirl,
    absorb_dressing,
    build_lowered,
)

from test import QiskitTestCase


def lower(circuit):
    """Build the emission circuit, returning it as a QuantumCircuit plus the distribution table."""
    dag, table = build_lowered(circuit_to_dag(circuit))
    return dag_to_circuit(dag), table


def lower_absorbed(circuit):
    """Build and absorb, returning as a QuantumCircuit plus the distribution table."""
    dag, table = build_lowered(circuit_to_dag(circuit))
    absorb_dressing(dag)
    return dag_to_circuit(dag), table


def collectors(circuit):
    """The (annotation, body, qubit indices) of each collect box, in circuit order."""
    out = []
    for inst in circuit.data:
        annotations = getattr(inst.operation, "annotations", None)
        if annotations:
            out.append(
                (
                    annotations[0],
                    inst.operation.blocks[0],
                    [circuit.find_bit(b).index for b in inst.qubits],
                )
            )
    return out


def emissions(circuit):
    """Every Emit instruction, recursing into boxes, in circuit order."""
    out = []
    for inst in circuit.data:
        op = inst.operation
        if op.name.startswith("samplex_emit"):
            out.append(op)
        for block in getattr(op, "blocks", None) or []:
            out.extend(emissions(block))
    return out


def emissions_in_scope(circuit):
    """The Emit operations of one scope, in order, without recursing."""
    return [inst.operation for inst in circuit.data if inst.operation.name.startswith("samplex_emit")]


def hard_boxes(circuit):
    """Boxes carrying no annotation — the ones holding propagating content."""
    return [
        inst.operation.blocks[0]
        for inst in circuit.data
        if inst.operation.name == "box" and not getattr(inst.operation, "annotations", None)
    ]


def gate_names(circuit):
    """Every instruction name in one scope, in order — `Emit` markers included."""
    return [inst.operation.name for inst in circuit.data]


def hard_gates(circuit):
    """The real gates of a hard body, without the `Emit` markers written inside it.

    A propagating emission lives *inside* the hard box, at the edge it starts from, because the hard
    content is exactly what conjugates it on the way to the far collector. It is a marker rather than
    something that executes, so it is not part of what the easy/hard split put here.
    """
    return [name for name in gate_names(circuit) if not name.startswith("samplex_emit")]


class TestBuildShape(QiskitTestCase):
    """The shape a single annotated box lowers to."""

    def test_twirl_yields_two_collectors_and_a_pair(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
        lowered, table = lower(circuit)

        self.assertEqual(len(collectors(lowered)), 2)
        emits = emissions(lowered)
        self.assertEqual(len(emits), 2)
        # The inverse pair shares one table entry and carries opposite directions; inversion is
        # implied by the direction rather than recorded.
        self.assertEqual(emits[0].distribution_key, emits[1].distribution_key)
        self.assertEqual({e.direction for e in emits}, {"left", "right"})
        self.assertEqual(len(table), 1)

    def test_each_side_has_its_own_collector(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
        lowered, _ = lower(circuit)

        # Build produces two collectors (left and right), each with only Gates items
        left, right = collectors(lowered)
        left_tags = [tag for tag, _ in left[0].items]
        right_tags = [tag for tag, _ in right[0].items]
        self.assertTrue(all(t == "gates" for t in left_tags))
        self.assertTrue(all(t == "gates" for t in right_tags))

    def test_noise_and_basis_land_on_the_side_their_placement_names(self):
        circuit = QuantumCircuit(2)
        with circuit.box(
            [
                Twirl(),
                InjectNoise("n0", "after"),
                ChangeBasis("b0", placement="start"),
            ]
        ):
            circuit.cx(0, 1)
        lowered, table = lower(circuit)

        all_emits = list(emissions(lowered))
        noise = next(e for e in all_emits if e.source == "inject_noise")
        basis = next(e for e in all_emits if e.source == "change_basis")
        self.assertEqual(noise.direction, "right")  # site="after"
        self.assertEqual(basis.direction, "left")  # placement="start"

        # Emissions are placed on the correct side of the hard box (positionally).
        # Verify via spine ordering: basis is before the hard box, noise is after.
        names = gate_names(lowered)
        box_positions = [i for i, n in enumerate(names) if n == "box"]
        # The hard box is the non-annotated box in the middle
        hard_pos = box_positions[0] if len(box_positions) == 1 else box_positions[1]
        basis_pos = next(
            i
            for i, inst in enumerate(lowered.data)
            if inst.operation.name.startswith("samplex_emit") and inst.operation.source == "change_basis"
        )
        noise_pos = next(
            i
            for i, inst in enumerate(lowered.data)
            if inst.operation.name.startswith("samplex_emit") and inst.operation.source == "inject_noise"
        )
        self.assertLess(basis_pos, hard_pos)  # basis on left edge
        self.assertGreater(noise_pos, hard_pos)  # noise on right edge
        # twirl distribution + noise ref + basis ref
        self.assertEqual(len(table), 3)

    def test_noise_and_basis_sit_outside_the_hard_box(self):
        """A basis change or noise injection is written at the edge its placement names.

        Not at the dressing edge, which is where the twirl pair goes. When the two differ — a
        `placement="end"` on a left-dressed box, say — writing it on the dressing edge would leave the
        hard box between it and the collector consuming it, so the propagation walk would conjugate it
        by content it is supposed to sit outside of.
        """
        for dressing in ("left", "right"):
            for placement, site, side in (("start", "before", "left"), ("end", "after", "right")):
                for annotation in (
                    ChangeBasis("b0", placement=placement),
                    InjectNoise("n0", site),
                ):
                    with self.subTest(dressing=dressing, side=side, kind=type(annotation).__name__):
                        circuit = QuantumCircuit(2)
                        with circuit.box([Twirl(dressing=dressing), annotation]):
                            circuit.cx(0, 1)
                        lowered, _ = lower(circuit)

                        names = gate_names(lowered)
                        # the hard box is the middle `box`; the collectors bracket everything
                        box_at = [i for i, n in enumerate(names) if n == "box"][1]
                        # Located on the spine, which is itself the claim: one of these on the spine at
                        # all means it is outside the hard box. Only a *propagating* emission is
                        # written inside it, and neither of these ever propagates.
                        kind = (
                            "change_basis" if "ChangeBasis" in type(annotation).__name__ else "inject_noise"
                        )
                        at = next(
                            i
                            for i, inst in enumerate(lowered.data)
                            if inst.operation.name.startswith("samplex_emit")
                            and inst.operation.source == kind
                        )
                        self.assertEqual(
                            at < box_at,
                            side == "left",
                            f"{kind} landed on the wrong side of the hard box",
                        )

    def test_emissions_nest_by_how_close_they_are_to_the_content(self):
        """Within one edge: twirl innermost, injections next, basis change outermost.

        The vocabulary implies the order. A twirl *is* the easy/hard boundary; an injection happens to
        the hard content so it sits just outside the twirl; a basis change applies to the box as a whole
        so it wraps everything. `InjectLocalClifford` is an injection despite resolving to the same
        `ResolvedBasis` a `ChangeBasis` does — which is the only thing that distinguishes the two.
        """
        circuit = QuantumCircuit(2)
        with circuit.box(
            [
                Twirl(),  # left-dressed by default
                ChangeBasis("b", placement="end"),
                InjectNoise("n", "after"),
            ]
        ):
            circuit.h(0)
            circuit.cx(0, 1)
        lowered, _ = lower(circuit)

        # The right-edge emissions sit after the hard box in innermost-first order.
        right_emits = []
        past_hard = False
        for inst in lowered.data:
            if inst.operation.name == "box" and not getattr(inst.operation, "annotations", None):
                past_hard = True
                continue
            if past_hard and inst.operation.name.startswith("samplex_emit"):
                right_emits.append(inst.operation.source)
        # noise then basis change (innermost-first); the far twirl half is on the dressing (left) edge
        self.assertEqual(right_emits, ["inject_noise", "change_basis"])

    def test_a_local_clifford_flanks_the_content_but_a_basis_change_wraps_it(self):
        # The two resolve identically except for placement, so this is the only observable difference —
        # and `mode` cannot stand in for it, since ChangeBasis(mode="local_clifford") is legal.
        def right_edge_order(annotation):
            circuit = QuantumCircuit(2)
            with circuit.box([Twirl(), annotation, InjectNoise("n", "after")]):
                circuit.cx(0, 1)
            lowered, _ = lower(circuit)
            # Collect right-edge emission sources in spine order (after hard box)
            right_emits = []
            past_hard = False
            for inst in lowered.data:
                if inst.operation.name == "box" and not getattr(inst.operation, "annotations", None):
                    past_hard = True
                    continue
                if past_hard and inst.operation.name.startswith("samplex_emit"):
                    right_emits.append(inst.operation.source)
            return right_emits

        # an injection sits inside the noise's own depth band, so construction order decides between them
        self.assertEqual(
            right_edge_order(InjectLocalClifford("f", "after")),
            ["change_basis", "inject_noise"],
        )
        # a basis change is outermost, so it lands after the noise
        self.assertEqual(
            right_edge_order(ChangeBasis("b", placement="end")),
            ["inject_noise", "change_basis"],
        )

    def test_collectors_start_empty_after_build(self):
        """After build, collectors are empty — gates and emissions live on the spine.

        The absorb_dressing pass later walks from each collector to populate items and body.
        """
        circuit = QuantumCircuit(3)
        with circuit.box(
            [
                Twirl(dressing="left"),
                ChangeBasis("b", placement="start"),
                InjectNoise("n", "before"),
            ]
        ):
            circuit.h(0)
            circuit.cx(1, 2)
        dag, _ = build_lowered(circuit_to_dag(circuit))
        colls = collectors(dag_to_circuit(dag))
        for coll in colls:
            self.assertEqual(coll[0].items, [])

    def test_absorbed_gates_sit_inside_the_basis_change(self):
        """After absorb_dressing, the collector holds gates and emissions in composition order."""
        left = QuantumCircuit(3)
        with left.box(
            [
                Twirl(dressing="left"),
                ChangeBasis("b", placement="start"),
                InjectNoise("n", "before"),
            ]
        ):
            left.h(0)
            left.cx(1, 2)
        dag, _ = build_lowered(circuit_to_dag(left))
        absorb_dressing(dag)
        left_coll = collectors(dag_to_circuit(dag))[0]
        self.assertIn(("gates", 1), left_coll[0].items)
        self.assertEqual(gate_names(left_coll[1]), ["h"])

        right = QuantumCircuit(3)
        with right.box(
            [
                Twirl(dressing="right"),
                ChangeBasis("b", placement="end"),
                InjectNoise("n", "after"),
            ]
        ):
            right.cx(1, 2)
            right.h(0)
        dag, _ = build_lowered(circuit_to_dag(right))
        absorb_dressing(dag)
        right_coll = collectors(dag_to_circuit(dag))[-1]
        self.assertIn(("gates", 1), right_coll[0].items)
        self.assertEqual(gate_names(right_coll[1]), ["h"])

    def test_gates_counts_account_for_exactly_the_body(self):
        # The invariant that keeps items and bodies in step. A merge that concatenated one but not the
        # other would show up here rather than as a wrong answer much later.
        circuit = QuantumCircuit(4)
        with circuit.box([Twirl(), ChangeBasis("b", placement="start")]):
            circuit.h(0)
            circuit.s(2)
            circuit.cx(0, 1)
            circuit.cx(2, 3)
        with circuit.box([Twirl(dressing="right"), InjectNoise("n", "after")]):
            circuit.cx(0, 1)
            circuit.s(0)
        lowered, _ = lower(circuit)

        for annotation, body, _ in collectors(lowered):
            counted = sum(n for kind, n in annotation.items if kind == "gates")
            self.assertEqual(counted, len(body.data), f"{annotation!r} vs {gate_names(body)}")

    def test_inject_local_clifford_resolves_to_a_basis_change(self):
        circuit = QuantumCircuit(1)
        with circuit.box([InjectLocalClifford("c3", "before")]):
            circuit.h(0)
        lowered, table = lower(circuit)

        (emit,) = emissions(lowered)
        self.assertEqual(emit.source, "change_basis")
        self.assertEqual(emit.direction, "left")
        self.assertEqual(emit.virtual_type, "c1")
        self.assertIn("local_cliffords.c3", table.entries()[0])

    def test_tag_only_box_is_transparent(self):
        circuit = QuantumCircuit(1)
        with circuit.box([Tag()]):
            circuit.h(0)
        lowered, table = lower(circuit)

        self.assertEqual(emissions(lowered), [])
        self.assertEqual(collectors(lowered), [])
        self.assertEqual(len(table), 0)
        # flattened, so the gate is at the top level
        self.assertEqual(gate_names(lowered), ["h"])

    def test_unannotated_box_is_transparent(self):
        circuit = QuantumCircuit(2)
        with circuit.box():
            circuit.cx(0, 1)
        lowered, _ = lower(circuit)
        self.assertEqual(gate_names(lowered), ["cx"])


class TestEasyHardSplit(QiskitTestCase):
    """Which gates the dressing absorbs, and where the emissions sit."""

    def test_left_dressing_absorbs_from_the_left(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            circuit.h(0)
            circuit.cx(0, 1)
        lowered, _ = lower_absorbed(circuit)

        left, right = collectors(lowered)
        self.assertEqual(gate_names(left[1]), ["h"])
        self.assertEqual(gate_names(right[1]), [])
        (hard,) = hard_boxes(lowered)
        self.assertEqual(hard_gates(hard), ["cx"])

    def test_right_dressing_absorbs_from_the_right(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="right")]):
            circuit.cx(0, 1)
            circuit.s(0)
            circuit.h(1)
        lowered, _ = lower_absorbed(circuit)

        left, right = collectors(lowered)
        self.assertEqual(gate_names(left[1]), [])
        # absorbed gates keep circuit order even though the sweep runs backwards
        self.assertEqual(gate_names(right[1]), ["s", "h"])
        (hard,) = hard_boxes(lowered)
        self.assertEqual(hard_gates(hard), ["cx"])

    def test_emissions_sit_on_the_dressing_edge(self):
        # Placement is load-bearing: the factor the easy gates absorb must reach its collector
        # without crossing the hard content, and the other factor must cross it.
        for dressing, expect_before in (("left", True), ("right", False)):
            with self.subTest(dressing=dressing):
                circuit = QuantumCircuit(2)
                with circuit.box([Twirl(dressing=dressing)]):
                    circuit.cx(0, 1)
                lowered, _ = lower(circuit)

                names = gate_names(lowered)
                emit_at = names.index("samplex_emit_twirl")
                box_at = names.index("box", 1)  # the hard box, after the first collector
                self.assertEqual(emit_at < box_at, expect_before)


class TestPerQubitAbsorption(QiskitTestCase):
    """Absorption poisons wires rather than latching the whole body.

    A single-qubit gate on a wire no multi-qubit gate has touched is still at the dressing edge *on its
    own wire*, so it commutes out even when it sits after an entangler elsewhere. Poisoning over a
    topological order is DAG ancestry: a gate is absorbable exactly when all of its ancestors were.
    """

    def test_clean_wire_behind_a_two_qubit_gate_is_absorbed(self):
        circuit = QuantumCircuit(3)
        with circuit.box([Twirl(dressing="left")]):
            circuit.cx(0, 1)
            circuit.h(2)
        lowered, _ = lower_absorbed(circuit)

        left, _ = collectors(lowered)
        # q2 is untouched by the cx, so h(2) is at the dressing edge on its own wire
        self.assertEqual(gate_names(left[1]), ["h"])
        (hard,) = hard_boxes(lowered)
        self.assertEqual(hard_gates(hard), ["cx"])

    def test_a_run_on_a_clean_wire_is_absorbed(self):
        circuit = QuantumCircuit(3)
        with circuit.box([Twirl(dressing="left")]):
            circuit.cx(0, 1)
            circuit.h(2)
            circuit.s(2)
        lowered, _ = lower_absorbed(circuit)
        self.assertEqual(gate_names(collectors(lowered)[0][1]), ["h", "s"])

    def test_a_poisoned_wire_is_left_alone(self):
        # h(0) really is behind the cx on its own wire, so it cannot be commuted out.
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            circuit.cx(0, 1)
            circuit.h(0)
        lowered, _ = lower_absorbed(circuit)

        self.assertEqual(gate_names(collectors(lowered)[0][1]), [])
        (hard,) = hard_boxes(lowered)
        self.assertEqual(hard_gates(hard), ["cx", "h"])

    def test_poison_spreads_transitively(self):
        # q2 is clean until cx(1,2), which is itself poisoned by cx(0,1) — so s(2) is content. This is
        # what makes a linear scan equivalent to walking DAG ancestry.
        circuit = QuantumCircuit(3)
        with circuit.box([Twirl(dressing="left")]):
            circuit.cx(0, 1)
            circuit.cx(1, 2)
            circuit.s(2)
        lowered, _ = lower_absorbed(circuit)

        self.assertEqual(gate_names(collectors(lowered)[0][1]), [])
        (hard,) = hard_boxes(lowered)
        self.assertEqual(hard_gates(hard), ["cx", "cx", "s"])

    def test_right_dressing_sweeps_from_the_other_end(self):
        circuit = QuantumCircuit(3)
        with circuit.box([Twirl(dressing="right")]):
            circuit.h(2)
            circuit.cx(0, 1)
        lowered, _ = lower_absorbed(circuit)

        left, right = collectors(lowered)
        self.assertEqual(gate_names(left[1]), [])
        # absorbed into the right collector, which is where a right dressing sits
        self.assertEqual(gate_names(right[1]), ["h"])
        (hard,) = hard_boxes(lowered)
        self.assertEqual(hard_gates(hard), ["cx"])

    def test_a_fully_absorbed_box_has_no_hard_box(self):
        # An empty box carries no information once propagation is derived from placement.
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            circuit.h(0)
            circuit.s(1)
        lowered, _ = lower_absorbed(circuit)

        self.assertEqual(hard_boxes(lowered), [])
        self.assertEqual([gate_names(b) for _, b, _ in collectors(lowered)], [["h", "s"], []])

    def test_a_nested_box_is_split_the_same_way(self):
        circuit = QuantumCircuit(3)
        with circuit.box([Twirl()]):
            with circuit.box([Twirl(dressing="left")]):
                circuit.cx(0, 1)
                circuit.h(2)
        lowered, _ = lower_absorbed(circuit)

        (outer_hard,) = hard_boxes(lowered)
        self.assertEqual([gate_names(b) for _, b, _ in collectors(outer_hard)], [["h"], []])
        self.assertEqual([hard_gates(b) for b in hard_boxes(outer_hard)], [["cx"]])

    def test_no_gate_is_lost_or_duplicated(self):
        circuit = QuantumCircuit(3)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
            circuit.h(2)
            circuit.s(2)
            circuit.cx(1, 2)
        lowered, _ = lower_absorbed(circuit)

        names = [n for _, body, _ in collectors(lowered) for n in gate_names(body)]
        names += [n for body in hard_boxes(lowered) for n in hard_gates(body)]
        self.assertEqual(sorted(names), ["cx", "cx", "h", "s"])

    def test_an_absorbed_gate_keeps_its_qubit(self):
        circuit = QuantumCircuit(3)
        with circuit.box([Twirl(dressing="left")]):
            circuit.cx(0, 1)
            circuit.h(2)
        lowered, _ = lower_absorbed(circuit)

        _, body, qubits = collectors(lowered)[0]
        self.assertEqual(qubits, [0, 1, 2])
        (gate,) = body.data
        self.assertEqual(gate.operation.name, "h")
        self.assertEqual([body.find_bit(b).index for b in gate.qubits], [2])


class TestFidelity(QiskitTestCase):
    """Things the emission circuit must not lose."""

    def test_gate_parameters_survive(self):
        theta = Parameter("theta")
        circuit = QuantumCircuit(1)
        with circuit.box([Twirl()]):
            circuit.rz(theta, 0)
        lowered, _ = lower_absorbed(circuit)

        left, _ = collectors(lowered)
        (rz,) = left[1].data
        self.assertEqual(rz.operation.name, "rz")
        self.assertEqual(rz.operation.params, [theta])

    def test_reordered_noncontiguous_qargs_round_trip(self):
        # A box applied to reordered, non-contiguous qubits: body position 1 is global qubit 2.
        circuit = QuantumCircuit(6)
        with circuit.box([Twirl()]):
            circuit.noop(5, 2)
        lowered, _ = lower(circuit)

        emits = emissions(lowered)
        for emit in emits:
            self.assertEqual(sorted(emit.qubits), [2, 5])
        for _, _, qubits in collectors(lowered):
            self.assertEqual(sorted(qubits), [2, 5])

    def test_top_level_gates_and_measures_survive(self):
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])
        lowered, _ = lower(circuit)

        names = gate_names(lowered)
        self.assertEqual(names[0], "h")
        self.assertEqual(names.count("measure"), 2)

    def test_build_is_deterministic(self):
        circuit = QuantumCircuit(4)
        with circuit.box([Twirl(), ChangeBasis("ref")]):
            circuit.noop(*range(4))
        with circuit.box([Twirl()]):
            circuit.noop(0, 1)
        runs = []
        for _ in range(3):
            lowered, table = lower(circuit)
            runs.append(
                (
                    gate_names(lowered),
                    [(c.synthesizer, tuple(c.items)) for c, _, _ in collectors(lowered)],
                    [tuple(q) for _, _, q in collectors(lowered)],
                    [(e.direction, tuple(e.qubits)) for e in emissions(lowered)],
                    table.entries(),
                )
            )
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(runs[0], runs[2])


class TestNesting(QiskitTestCase):
    """Annotated boxes inside twirled boxes."""

    def test_nested_twirl_lowers_inside_the_hard_box(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl()]):
            with circuit.box([Twirl()]):
                circuit.cx(0, 1)
            circuit.cx(1, 0)
        lowered, _ = lower(circuit)

        # outer: two collectors at the top level, inner: two inside the hard box
        self.assertEqual(len(collectors(lowered)), 2)
        (hard,) = hard_boxes(lowered)
        self.assertEqual(len(collectors(hard)), 2)
        # four emissions in total: an inverse pair per twirl
        self.assertEqual(len(emissions(lowered)), 4)

    def test_nested_box_absorbs_into_its_own_dressing(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            with circuit.box([Twirl(dressing="left")]):
                circuit.h(0)
                circuit.cx(0, 1)
        lowered, _ = lower_absorbed(circuit)

        (outer_hard,) = hard_boxes(lowered)
        inner_left, inner_right = collectors(outer_hard)
        self.assertEqual(gate_names(inner_left[1]), ["h"])
        self.assertEqual(gate_names(inner_right[1]), [])
        (inner_hard,) = hard_boxes(outer_hard)
        self.assertEqual(hard_gates(inner_hard), ["cx"])

    def test_outermost_box_still_absorbs(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            circuit.h(0)
            with circuit.box([Twirl()]):
                circuit.cx(0, 1)
        lowered, _ = lower_absorbed(circuit)

        left, _ = collectors(lowered)
        self.assertEqual(gate_names(left[1]), ["h"])

    def test_nested_change_basis_only_box(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl()]):
            with circuit.box([ChangeBasis("b1", placement="end")]):
                circuit.cx(0, 1)
        lowered, _ = lower(circuit)

        sources = sorted(e.source for e in emissions(lowered))
        self.assertEqual(sources, ["change_basis", "twirl", "twirl"])

    def test_nested_unannotated_box_is_flattened_into_hard_content(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl()]):
            with circuit.box():
                circuit.cx(0, 1)
        lowered, _ = lower(circuit)

        (hard,) = hard_boxes(lowered)
        self.assertEqual(hard_gates(hard), ["cx"])


class TestRejections(QiskitTestCase):
    """What the pass refuses rather than silently mangling."""

    def test_non_box_control_flow_is_rejected(self):
        circuit = QuantumCircuit(1, 1)
        with circuit.if_test((circuit.clbits[0], True)):
            circuit.x(0)
        with self.assertRaisesRegex(ValueError, "Unsupported control flow"):
            lower(circuit)

    def test_inject_noise_without_twirl_is_rejected(self):
        circuit = QuantumCircuit(1)
        with circuit.box([InjectNoise("n0")]):
            circuit.h(0)
        with self.assertRaisesRegex(ValueError, "InjectNoise requires a Twirl"):
            lower(circuit)

    def test_change_basis_and_inject_local_clifford_conflict(self):
        circuit = QuantumCircuit(1)
        with circuit.box([ChangeBasis("b0"), InjectLocalClifford("c0")]):
            circuit.h(0)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            lower(circuit)


class TestPropagatingEmissionPlacement(QiskitTestCase):
    """A propagating emission is written inside the hard box; a local one is not.

    The hard content is exactly what conjugates the far half on its way to the far collector, so putting
    the emission outside the box would place a scope boundary between it and the gates it has to cross —
    which is what made a later pass have to move it there. Writing it in the right place to begin with is
    what makes that machinery unnecessary.
    """

    def test_the_far_half_is_written_inside_the_hard_box(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            circuit.cx(0, 1)
        lowered, _ = lower(circuit)

        spine = [op.direction for op in emissions_in_scope(lowered)]
        (hard,) = hard_boxes(lowered)
        inside = [op.direction for op in emissions_in_scope(hard)]
        # The near half stays on the spine, where its collector can absorb it; the far half goes inside.
        self.assertEqual(spine, ["left"])
        self.assertEqual(inside, ["right"])

    def test_right_dressing_puts_it_at_the_other_end(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="right")]):
            circuit.cx(0, 1)
        lowered, _ = lower(circuit)

        (hard,) = hard_boxes(lowered)
        # Travelling left, so it starts at the hard content's right-hand edge: the back of the body.
        self.assertEqual(gate_names(hard), ["cx", "samplex_emit_twirl"])
        self.assertEqual([op.direction for op in emissions_in_scope(hard)], ["left"])

    def test_a_box_with_no_hard_content_keeps_it_on_the_spine(self):
        """With nothing to cross there is no hard box, and no conjugation is due."""
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            circuit.h(0)
            circuit.s(1)
        lowered, _ = lower(circuit)

        self.assertEqual(hard_boxes(lowered), [])
        self.assertEqual(sorted(op.direction for op in emissions_in_scope(lowered)), ["left", "right"])

    def test_a_leading_hard_gate_is_not_absorbed_across_the_twirl_point(self):
        """The hard box is still the barrier that keeps content out of the dressing.

        Right dressing puts the twirl point at the *right* edge, so a single-qubit gate that the sweep
        classified as hard must not fold into the left collector — the far half travels leftward through
        it, and a target collector's own absorbed gates are not crossed on the way in.
        """
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="right")]):
            circuit.s(0)  # hard: poisoned by the cx below it
            circuit.cx(0, 1)
        lowered, _ = lower_absorbed(circuit)

        left, _ = collectors(lowered)
        self.assertEqual(gate_names(left[1]), [])
        (hard,) = hard_boxes(lowered)
        self.assertEqual(hard_gates(hard), ["s", "cx"])
