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
        if is_collector(inst.operation):
            out.append(
                (
                    inst.operation.annotations[0],
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
        if op.name.startswith("emit"):
            out.append(op)
        for block in getattr(op, "blocks", None) or []:
            out.extend(emissions(block))
    return out


def emissions_with_qubits(circuit, frame=None):
    """Every Emit instruction paired with the global qubits it landed on, in circuit order.

    An `Emit` operation holds only how its qubits group into subsystems, by index into its own
    qargs — the wires themselves are on the instruction, since one operation can be shared by
    several placements. So the qubits have to come from the walk, mapped out through each enclosing
    box's own qargs.
    """
    frame = list(range(circuit.num_qubits)) if frame is None else frame
    out = []
    for inst in circuit.data:
        op = inst.operation
        qubits = [frame[circuit.find_bit(b).index] for b in inst.qubits]
        if op.name.startswith("samplex_emit"):
            out.append((op, qubits))
        for block in getattr(op, "blocks", None) or []:
            out.extend(emissions_with_qubits(block, qubits))
    return out


def emissions_in_scope(circuit):
    """The Emit operations of one scope, in order, without recursing."""
    return [inst.operation for inst in circuit.data if inst.operation.name.startswith("emit")]


def content_boxes(circuit):
    """The bodies of the content boxes in one scope — every box that is not a collector.

    One per emitting box, always written, even when empty. After absorption its body holds exactly
    what could not be absorbed, which is what makes an empty one meaningful.
    """
    return [
        inst.operation.blocks[0]
        for inst in circuit.data
        if inst.operation.name == "box" and not is_collector(inst.operation)
    ]


def is_collector(op):
    """Whether a box operation carries a `Collect` annotation.

    Not merely "carries an annotation": a content box carries whatever annotations we do not act on,
    so the two are told apart by what the annotation is.
    """
    return any(hasattr(a, "synthesizer") for a in getattr(op, "annotations", None) or [])


def gate_names(circuit):
    """Every instruction name in one scope, in order — `Emit` markers included."""
    return [inst.operation.name for inst in circuit.data]


def all_gate_names(circuit):
    """Every instruction name, descending into boxes, in order.

    The box itself is reported too, so a caller can see where the nesting is.
    """
    out = []
    for inst in circuit.data:
        out.append(inst.operation.name)
        for block in getattr(inst.operation, "blocks", None) or []:
            out.extend(all_gate_names(block))
    return out


def unfolded(circuit, table):
    """Every emission and gate in physical order, descending into content boxes.

    Emissions are labelled `emit:<source>`. The twirl point is *inside* the content box, so the spine
    on its own no longer shows how emissions nest around the content; unfolding the box does. Collect
    boxes are not descended into — what is in one has already been taken off the spine.
    """
    out = []
    for inst in circuit.data:
        op = inst.operation
        if op.name.startswith("emit"):
            out.append(f"emit:{op.source(table)}")
        elif op.name == "box" and not is_collector(op):
            out.extend(unfolded(op.blocks[0], table))
        else:
            out.append(op.name)
    return out


def real_gates(body):
    """Gate names in a body, excluding the `Emit` markers sitting in it.

    Both kinds of body have them: a collector body holds the local emissions it absorbed, and a
    content box holds the twirl point's emissions. Neither executes, so neither is a gate.
    """
    return [inst.operation.name for inst in body.data if not inst.operation.name.startswith("emit")]


def body_locals(body):
    """Absorbed local emissions in a collector body, in body order."""
    return [
        inst.operation
        for inst in body.data
        if inst.operation.name.startswith("emit") and inst.operation.direction == "local"
    ]


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

        # Build produces two collectors (left and right), each with an empty body — nothing has
        # been absorbed into them yet.
        left, right = collectors(lowered)
        self.assertEqual(len(left[1].data), 0)
        self.assertEqual(len(right[1].data), 0)

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
        noise = next(e for e in all_emits if e.source(table) == "inject_noise")
        basis = next(e for e in all_emits if e.source(table) == "change_basis")
        self.assertEqual(noise.direction, "right")  # site="after"
        self.assertEqual(basis.direction, "left")  # placement="start"

        # Positionally: the basis change is before the content, the noise after it. The twirl pair sits
        # at the twirl point in between, which for a left dressing is against the content's near side.
        self.assertEqual(
            unfolded(lowered, table),
            ["box", "emit:change_basis", "emit:twirl", "emit:twirl", "cx", "emit:inject_noise", "box"],
        )
        # twirl distribution + noise ref + basis ref
        self.assertEqual(len(table), 3)

    def test_noise_and_basis_sit_outside_the_content(self):
        """A basis change or noise injection is written at the edge its placement names.

        Not at the dressing edge, which is where the twirl pair goes. When the two differ — a
        `placement="end"` on a left-dressed box, say — writing it on the dressing edge would leave the
        content between it and the collector consuming it, so the propagation walk would conjugate it
        by content it is supposed to sit outside of.

        The claim is about the *content*, not about which box: a basis change is on the spine outside
        the content box while an injection is written inside it, ahead of the content, and both are
        equally "outside the content".
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
                        lowered, table = lower(circuit)

                        kind = (
                            "change_basis" if "ChangeBasis" in type(annotation).__name__ else "inject_noise"
                        )
                        order = unfolded(lowered, table)
                        at = order.index(f"emit:{kind}")
                        content_at = order.index("cx")
                        self.assertEqual(
                            at < content_at,
                            side == "left",
                            f"{kind} landed on the wrong side of the content",
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
        lowered, table = lower(circuit)

        # The right-edge emissions sit after the content in innermost-first order: the injection is
        # written inside the content box against the content, the basis change on the spine outside it.
        order = unfolded(lowered, table)
        right_emits = [name[5:] for name in order[order.index("cx") :] if name.startswith("emit:")]
        # noise then basis change (innermost-first); the twirl pair is on the dressing (left) edge
        self.assertEqual(right_emits, ["inject_noise", "change_basis"])

    def test_a_local_clifford_flanks_the_content_but_a_basis_change_wraps_it(self):
        # The two resolve identically except for placement, so this is the only observable difference —
        # and `mode` cannot stand in for it, since ChangeBasis(mode="local_clifford") is legal.
        def right_edge_order(annotation):
            circuit = QuantumCircuit(2)
            with circuit.box([Twirl(), annotation, InjectNoise("n", "after")]):
                circuit.cx(0, 1)
            lowered, table = lower(circuit)
            # Emission sources after the content, in physical order. An injection is written inside the
            # content box against the content and a basis change on the spine outside it, so the order
            # only shows up once the box is unfolded.
            order = unfolded(lowered, table)
            return [name[5:] for name in order[order.index("cx") :] if name.startswith("emit:")]

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

        The absorb_dressing pass later walks from each collector to populate its body.
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
            self.assertEqual(len(coll[1].data), 0)

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
        self.assertEqual(real_gates(left_coll[1]), ["h"])
        self.assertTrue(body_locals(left_coll[1]))

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
        self.assertEqual(real_gates(right_coll[1]), ["h"])
        self.assertTrue(body_locals(right_coll[1]))

    def test_inject_local_clifford_resolves_to_a_basis_change(self):
        circuit = QuantumCircuit(1)
        with circuit.box([InjectLocalClifford("c3", "before")]):
            circuit.h(0)
        lowered, table = lower(circuit)

        (emit,) = emissions(lowered)
        self.assertEqual(emit.source(table), "change_basis")
        self.assertEqual(emit.direction, "left")
        self.assertEqual(emit.virtual_type(table), "c1")
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
        self.assertEqual(real_gates(left[1]), ["h"])
        self.assertEqual(real_gates(right[1]), [])
        (hard,) = content_boxes(lowered)
        self.assertEqual(real_gates(hard), ["cx"])

    def test_right_dressing_absorbs_from_the_right(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="right")]):
            circuit.cx(0, 1)
            circuit.s(0)
            circuit.h(1)
        lowered, _ = lower_absorbed(circuit)

        left, right = collectors(lowered)
        self.assertEqual(real_gates(left[1]), [])
        # absorbed gates keep circuit order even though the sweep runs backwards
        self.assertEqual(real_gates(right[1]), ["s", "h"])
        (hard,) = content_boxes(lowered)
        self.assertEqual(real_gates(hard), ["cx"])

    def test_emissions_sit_on_the_dressing_edge(self):
        # Placement is load-bearing: the factor the easy gates absorb must reach its collector
        # without crossing the hard content, and the other factor must cross it.
        for dressing, expect_before in (("left", True), ("right", False)):
            with self.subTest(dressing=dressing):
                circuit = QuantumCircuit(2)
                with circuit.box([Twirl(dressing=dressing)]):
                    circuit.cx(0, 1)
                lowered, table = lower(circuit)

                # Both halves are at the twirl point, inside the content box, so the side they are on
                # is their position relative to the content rather than to the box.
                order = unfolded(lowered, table)
                self.assertEqual(order.count("emit:twirl"), 2)
                emit_at = order.index("emit:twirl")
                content_at = order.index("cx")
                self.assertEqual(emit_at < content_at, expect_before)


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
        self.assertEqual(real_gates(left[1]), ["h"])
        (hard,) = content_boxes(lowered)
        self.assertEqual(real_gates(hard), ["cx"])

    def test_a_run_on_a_clean_wire_is_absorbed(self):
        circuit = QuantumCircuit(3)
        with circuit.box([Twirl(dressing="left")]):
            circuit.cx(0, 1)
            circuit.h(2)
            circuit.s(2)
        lowered, _ = lower_absorbed(circuit)
        self.assertEqual(real_gates(collectors(lowered)[0][1]), ["h", "s"])

    def test_a_poisoned_wire_is_left_alone(self):
        """h(0) really is behind the cx on its own wire, so it cannot be commuted out.

        Nor can the collector on the other side reach in and take it: it is on the far side of the twirl
        point, so the far half is conjugated by it on the way past, and only the dressing side of that
        point folds into a collector.
        """
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            circuit.cx(0, 1)
            circuit.h(0)
        lowered, _ = lower_absorbed(circuit)

        self.assertEqual(real_gates(collectors(lowered)[0][1]), [])
        (content,) = content_boxes(lowered)
        self.assertEqual(real_gates(content), ["cx", "h"])

    def test_poison_spreads_transitively(self):
        # q2 is clean until cx(1,2), which is itself poisoned by cx(0,1) — so s(2) is content. This is
        # what makes a linear scan equivalent to walking DAG ancestry.
        circuit = QuantumCircuit(3)
        with circuit.box([Twirl(dressing="left")]):
            circuit.cx(0, 1)
            circuit.cx(1, 2)
            circuit.s(2)
        lowered, _ = lower_absorbed(circuit)

        self.assertEqual(real_gates(collectors(lowered)[0][1]), [])
        (content,) = content_boxes(lowered)
        self.assertEqual(real_gates(content), ["cx", "cx", "s"])

    def test_right_dressing_sweeps_from_the_other_end(self):
        circuit = QuantumCircuit(3)
        with circuit.box([Twirl(dressing="right")]):
            circuit.h(2)
            circuit.cx(0, 1)
        lowered, _ = lower_absorbed(circuit)

        left, right = collectors(lowered)
        self.assertEqual(real_gates(left[1]), [])
        # absorbed into the right collector, which is where a right dressing sits
        self.assertEqual(real_gates(right[1]), ["h"])
        (hard,) = content_boxes(lowered)
        self.assertEqual(real_gates(hard), ["cx"])

    def test_a_fully_absorbed_box_keeps_an_empty_content_box(self):
        """The content box is always written, and an empty one is a statement rather than noise.

        What is left in a content box after absorption is exactly what could not be absorbed, so an
        empty body says nothing here was hard.
        """
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            circuit.h(0)
            circuit.s(1)
        lowered, _ = lower_absorbed(circuit)

        (content,) = content_boxes(lowered)
        self.assertEqual(real_gates(content), [])
        self.assertEqual([real_gates(b) for _, b, _ in collectors(lowered)], [["h", "s"], []])

    def test_a_nested_box_is_split_the_same_way(self):
        circuit = QuantumCircuit(3)
        with circuit.box([Twirl()]):
            with circuit.box([Twirl(dressing="left")]):
                circuit.cx(0, 1)
                circuit.h(2)
        lowered, _ = lower_absorbed(circuit)

        (outer_hard,) = content_boxes(lowered)
        self.assertEqual([real_gates(b) for _, b, _ in collectors(outer_hard)], [["h"], []])
        self.assertEqual([real_gates(b) for b in content_boxes(outer_hard)], [["cx"]])

    def test_no_gate_is_lost_or_duplicated(self):
        circuit = QuantumCircuit(3)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
            circuit.h(2)
            circuit.s(2)
            circuit.cx(1, 2)
        lowered, _ = lower_absorbed(circuit)

        names = [n for _, body, _ in collectors(lowered) for n in real_gates(body)]
        names += [n for body in content_boxes(lowered) for n in real_gates(body)]
        self.assertEqual(sorted(names), ["cx", "cx", "h", "s"])

    def test_an_absorbed_gate_keeps_its_qubit(self):
        circuit = QuantumCircuit(3)
        with circuit.box([Twirl(dressing="left")]):
            circuit.cx(0, 1)
            circuit.h(2)
        lowered, _ = lower_absorbed(circuit)

        _, body, qubits = collectors(lowered)[0]
        self.assertEqual(qubits, [0, 1, 2])
        (gate,) = [i for i in body.data if not i.operation.name.startswith("emit")]
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
        (rz,) = [i for i in left[1].data if not i.operation.name.startswith("emit")]
        self.assertEqual(rz.operation.name, "rz")
        self.assertEqual(rz.operation.params, [theta])

    def test_reordered_noncontiguous_qargs_round_trip(self):
        # A box applied to reordered, non-contiguous qubits: body position 1 is global qubit 2.
        circuit = QuantumCircuit(6)
        with circuit.box([Twirl()]):
            circuit.noop(5, 2)
        lowered, _ = lower(circuit)

        for _, qubits in emissions_with_qubits(lowered):
            self.assertEqual(sorted(qubits), [2, 5])
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
                    [(c.synthesizer, tuple(gate_names(b))) for c, b, _ in collectors(lowered)],
                    [tuple(q) for _, _, q in collectors(lowered)],
                    [(e.direction, tuple(q)) for e, q in emissions_with_qubits(lowered)],
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
        (hard,) = content_boxes(lowered)
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

        (outer_hard,) = content_boxes(lowered)
        inner_left, inner_right = collectors(outer_hard)
        self.assertEqual(real_gates(inner_left[1]), ["h"])
        self.assertEqual(real_gates(inner_right[1]), [])
        (inner_hard,) = content_boxes(outer_hard)
        self.assertEqual(real_gates(inner_hard), ["cx"])

    def test_outermost_box_still_absorbs(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            circuit.h(0)
            with circuit.box([Twirl()]):
                circuit.cx(0, 1)
        lowered, _ = lower_absorbed(circuit)

        left, _ = collectors(lowered)
        self.assertEqual(real_gates(left[1]), ["h"])

    def test_nested_change_basis_only_box(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl()]):
            with circuit.box([ChangeBasis("b1", placement="end")]):
                circuit.cx(0, 1)
        lowered, table = lower(circuit)

        sources = sorted(e.source(table) for e in emissions(lowered))
        self.assertEqual(sources, ["change_basis", "twirl", "twirl"])

    def test_nested_unannotated_box_is_flattened_into_hard_content(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl()]):
            with circuit.box():
                circuit.cx(0, 1)
        lowered, _ = lower(circuit)

        (hard,) = content_boxes(lowered)
        self.assertEqual(real_gates(hard), ["cx"])


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
    """Both halves of the pair are written inside the content box, at the twirl point.

    The twirl point is *not* the box boundary: it sits after the absorbable run, so those gates are on
    the near side of it and multiply into the dressing instead of being crossed. Putting it at the
    boundary would leave every gate in the body to be crossed by the far half, and a Pauli cannot cross
    a non-Clifford at all — an `rz` in a twirled box would stop being expressible.

    The pair stays together for the same reason it shares one draw: two halves about one point. Splitting
    them across the boundary would compose the near half on the far side of gates the far half is
    conjugated by.
    """

    def test_the_pair_is_written_inside_the_content_box(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            circuit.cx(0, 1)
        lowered, _ = lower(circuit)

        self.assertEqual(emissions_in_scope(lowered), [])
        (content,) = content_boxes(lowered)
        # Near half first — it faces the collector it will be absorbed into — then the far half, which
        # is what the hard content conjugates.
        self.assertEqual([op.direction for op in emissions_in_scope(content)], ["left", "right"])
        self.assertEqual(gate_names(content), ["emit", "emit", "cx"])

    def test_right_dressing_puts_it_at_the_other_end(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="right")]):
            circuit.cx(0, 1)
        lowered, _ = lower(circuit)

        (content,) = content_boxes(lowered)
        # The twirl point is the content's right-hand edge, so the pair is at the back of the body: the
        # far half travelling left first, then the near half facing its own collector.
        self.assertEqual(gate_names(content), ["cx", "emit", "emit"])
        self.assertEqual([op.direction for op in emissions_in_scope(content)], ["left", "right"])

    def test_a_box_with_no_hard_content_still_has_a_twirl_point(self):
        """With nothing to cross, the twirl point is simply past the whole absorbable run."""
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            circuit.h(0)
            circuit.s(1)
        lowered, _ = lower(circuit)

        self.assertEqual(emissions_in_scope(lowered), [])
        (content,) = content_boxes(lowered)
        self.assertEqual(gate_names(content), ["h", "s", "emit", "emit"])
        self.assertEqual(sorted(op.direction for op in emissions_in_scope(content)), ["left", "right"])

    def test_a_leading_hard_gate_is_not_absorbed_across_the_twirl_point(self):
        """The twirl point is the barrier that keeps content out of the dressing.

        Right dressing puts it at the *right* edge, so a single-qubit gate the sweep classified as hard
        must fold into neither collector — the far half travels leftward through it, and a target
        collector's own absorbed gates are not crossed on the way in.
        """
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="right")]):
            circuit.s(0)  # hard: poisoned by the cx below it
            circuit.cx(0, 1)
        lowered, _ = lower_absorbed(circuit)

        left, right = collectors(lowered)
        self.assertEqual(gate_names(left[1]), [])
        self.assertEqual(gate_names(right[1]), ["emit"])  # the near half, nothing else
        (content,) = content_boxes(lowered)
        self.assertEqual(real_gates(content), ["s", "cx"])
