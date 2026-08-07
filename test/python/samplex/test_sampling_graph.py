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

"""Tests for sampling graph construction: emission circuit (IR2) -> template + graph.

The graph says how to compute the template's angles. Both are read off the same IR2 circuit, so the
graph's parameter ranges are exactly the ones the template minted.

The load-bearing claim here is that a collector *owns* the gates it absorbed, rather than them being
separate nodes chained from a particular emission. That is what avoids having to attribute each
absorbed gate to the emission it multiplies into, which is impossible for a merged collector without
segment structure.
"""

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit._accelerate.samplex import (
    ChangeBasis,
    InjectNoise,
    Twirl,
    absorb_emissions,
    build_lowered,
    lower,
    merge_collectors,
)

from test import QiskitTestCase


def graph_of(circuit, optimize=True):
    """The sampling graph, with the IR2 optimizations optionally applied first."""
    dag, table = build_lowered(circuit_to_dag(circuit))
    if optimize:
        merge_collectors(dag)
        absorb_emissions(dag)
    _, graph = lower(dag, table)
    return graph


def kinds(graph):
    return [node[0] for node in graph.nodes()]


def of_kind(graph, prefix):
    return [node for node in graph.nodes() if node[0].startswith(prefix)]


def wiring(graph):
    """Edges as (source kind, target kind, direction)."""
    nodes = graph.nodes()
    return [(nodes[a][0], nodes[b][0], d) for a, b, d in graph.edges()]


def gates(node):
    """A collector's absorbed gates, dropping the emissions interleaved between them."""
    return [step for step in node[3] if step[0] != "emit"]


class TestCollectorOwnsAbsorbedGates(QiskitTestCase):
    """Absorbed gates are steps on the Collect node, not nodes of their own.

    A collector's `steps` are one *ordered* sequence mixing the emissions it consumes with the gates it
    absorbed, because where a gate sits in the layer matters — a `ChangeBasis` wraps the absorbed gates
    while a twirl composes inside them.
    """

    def test_absorbed_gate_is_recorded_on_its_collector(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            circuit.h(0)
            circuit.cx(0, 1)
        graph = graph_of(circuit)

        collects = of_kind(graph, "collect:")
        self.assertEqual(len(collects), 2)
        left, right = collects
        # the h was absorbed into the left dressing, on qubit 0, and composes before the twirl factor
        # the local near twirl half shows its partition qubits [0, 1]
        self.assertEqual(left[3], [("h", [0]), ("emit", [0, 1])])
        self.assertEqual(gates(right), [])

    def test_a_basis_change_composes_outside_the_absorbed_gates(self):
        # The ordering that two independent lists could not express: the basis change applies to the box
        # as a whole, so it wraps the easy gates, while the twirl factor composes inside them.
        # After absorption both are local, showing their partition qubits.
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left"), ChangeBasis("b", placement="start")]):
            circuit.h(0)
            circuit.cx(0, 1)
        graph = graph_of(circuit)

        left = of_kind(graph, "collect:")[0]
        self.assertEqual(left[3], [("emit", [0, 1]), ("h", [0]), ("emit", [0, 1])])

    def test_absorbed_gates_are_not_propagate_nodes(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            circuit.h(0)
            circuit.cx(0, 1)
        graph = graph_of(circuit)

        # only the cx conjugates anything; the h is folded into a layer instead
        self.assertEqual([n[0] for n in of_kind(graph, "propagate:")], ["propagate:cx"])

    def test_a_run_of_absorbed_gates_keeps_its_order(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            circuit.h(0)
            circuit.s(0)
            circuit.cx(0, 1)
        graph = graph_of(circuit)
        left = of_kind(graph, "collect:")[0]
        self.assertEqual(gates(left), [("h", [0]), ("s", [0])])

    def test_a_merged_collector_owns_every_contribution(self):
        # This is the case that has no per-emission attribution: after merging, the collector holds
        # absorbed gates from more than one box.
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="right")]):
            circuit.cx(0, 1)
            circuit.s(0)
        with circuit.box([Twirl(dressing="left")]):
            circuit.h(1)
            circuit.cx(0, 1)
        graph = graph_of(circuit)

        middle = next(c for c in of_kind(graph, "collect:") if len(gates(c)) > 1)
        # each box's own run, first box then second — so the two layers meet outermost-to-outermost
        # local emissions show their partition qubits after absorption
        self.assertEqual(
            middle[3],
            [("emit", [0, 1]), ("s", [0]), ("h", [1]), ("emit", [0, 1])],
        )


class TestPropagation(QiskitTestCase):
    """Which gates conjugate which emission, derived from placement."""

    def test_left_factor_reaches_its_collector_directly(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            circuit.h(0)
            circuit.cx(0, 1)
        graph = graph_of(circuit)

        # The near (left) factor is local — absorbed, no graph edge. Only the far (right) factor
        # propagates through the hard box: emit -> cx -> right collector.
        self.assertIn(("emit:UniformPauli", "propagate:cx", "right"), wiring(graph))
        self.assertIn(("propagate:cx", "collect:RzSx", "right"), wiring(graph))

    def test_right_dressing_propagates_the_left_factor(self):
        # Mirrored: with the dressing on the right, the hard content conjugates the *left* factor.
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="right")]):
            circuit.cx(0, 1)
            circuit.s(0)
        graph = graph_of(circuit)

        directions = {d for src, _, d in wiring(graph) if src == "propagate:cx"}
        self.assertEqual(directions, {"left"})

    def test_a_twirl_produces_one_incoming_emission_after_absorption(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
        graph = graph_of(circuit)
        # After absorption, only the far (incoming) twirl half remains as a VFG node.
        # The near half is local, absorbed into its collector.
        self.assertEqual(len(of_kind(graph, "emit:")), 1)
        emitted = {d for src, _, d in wiring(graph) if src.startswith("emit:")}
        self.assertEqual(emitted, {"right"})

    def test_a_chain_of_hard_gates_is_sequential(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            circuit.cx(0, 1)
            circuit.cx(1, 0)
        graph = graph_of(circuit)

        propagates = of_kind(graph, "propagate:")
        self.assertEqual(len(propagates), 2)
        # emit -> cx -> cx -> collector, not two parallel branches off the emit
        edges = wiring(graph)
        self.assertIn(("propagate:cx", "propagate:cx", "right"), edges)


class TestDirectionLivesOnNodes(QiskitTestCase):
    """Direction is carried by the nodes a flow passes through, not by the edges.

    `edges()` reads it off the source node. The consequence that matters is that a `Propagate` node is
    created per handedness, so one node never sits on paths running both ways.
    """

    def test_a_gate_crossed_both_ways_becomes_two_nodes(self):
        # Adjacent boxes with opposite dressings: the left-dressed box's far (right) half propagates
        # rightward through its cx, while the right-dressed box's far (left) half propagates leftward
        # through its cx. Conjugating one virtual gate leftward and rightward are different operations,
        # so a single fused node could not be evaluated.
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            circuit.cx(0, 1)
        with circuit.box([Twirl(dressing="right")]):
            circuit.cx(0, 1)
        graph = graph_of(circuit)

        self.assertEqual(len([n for n in graph.nodes() if n[0] == "propagate:cx"]), 2)
        directions = {d for src, _, d in wiring(graph) if src == "propagate:cx"}
        self.assertEqual(directions, {"left", "right"})

    def test_every_edge_of_a_real_graph_reports_a_direction(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(), ChangeBasis("ref"), InjectNoise("n0", "after")]):
            circuit.h(0)
            circuit.cx(0, 1)
        graph = graph_of(circuit)
        self.assertNotIn("none", {d for _, _, d in graph.edges()})


class TestOtherEmissionKinds(QiskitTestCase):
    """Basis changes and noise are emissions too, distinguished by their label rather than a kind."""

    def test_basis_change_and_noise_are_absorbed(self):
        # After absorption, basis changes and noise injections are local — they become steps on
        # their collector rather than independent VFG nodes.
        circuit = QuantumCircuit(2)
        with circuit.box(
            [Twirl(), ChangeBasis("b0", placement="start"), InjectNoise("n0", "after")]
        ):
            circuit.cx(0, 1)
        graph = graph_of(circuit)

        self.assertEqual(of_kind(graph, "change_basis:"), [])
        self.assertEqual(of_kind(graph, "inject_noise:"), [])
        # Only the incoming twirl half remains as a VFG node
        self.assertEqual(len(of_kind(graph, "emit:")), 1)

    def test_a_basis_change_is_never_conjugated_by_box_content(self):
        # It happens at the edge its placement names, so it reaches its collector without crossing the
        # hard content — for either dressing and either placement. Only the twirl factor is conjugated.
        for dressing in ("left", "right"):
            for placement in ("start", "end"):
                with self.subTest(dressing=dressing, placement=placement):
                    circuit = QuantumCircuit(2)
                    with circuit.box(
                        [Twirl(dressing=dressing), ChangeBasis("b", placement=placement)]
                    ):
                        circuit.cx(0, 1)
                    graph = graph_of(circuit)

                    conjugated = {src for src, tgt, _ in wiring(graph) if tgt == "propagate:cx"}
                    self.assertEqual(conjugated, {"emit:UniformPauli"})

    def test_injected_noise_is_never_conjugated_by_box_content(self):
        for dressing in ("left", "right"):
            for site in ("before", "after"):
                with self.subTest(dressing=dressing, site=site):
                    circuit = QuantumCircuit(2)
                    with circuit.box([Twirl(dressing=dressing), InjectNoise("n0", site)]):
                        circuit.cx(0, 1)
                    graph = graph_of(circuit)

                    conjugated = {src for src, tgt, _ in wiring(graph) if tgt == "propagate:cx"}
                    self.assertEqual(conjugated, {"emit:UniformPauli"})

    def test_measure_and_reset_become_nodes(self):
        circuit = QuantumCircuit(2, 2)
        circuit.reset(0)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])
        graph = graph_of(circuit)

        self.assertEqual(len(of_kind(graph, "reset")), 1)
        self.assertEqual(len(of_kind(graph, "measure")), 2)


class TestAgreementWithTheTemplate(QiskitTestCase):
    """The graph and the template must describe the same parameter vector."""

    def test_collect_nodes_carry_the_templates_parameter_ranges(self):
        circuit = QuantumCircuit(4)
        with circuit.box([Twirl(), ChangeBasis("ref")]):
            circuit.h(0)
            circuit.cx(0, 1)
            circuit.cx(2, 3)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
        dag, table = build_lowered(circuit_to_dag(circuit))
        absorb_emissions(dag)
        merge_collectors(dag)
        template, graph = lower(dag, table)

        allocated = sorted(i for node in graph.nodes() for i in node[2])
        self.assertEqual(allocated, list(range(len(allocated))))
        self.assertEqual(len(allocated), QuantumCircuit._from_circuit_data(template).num_parameters)

    def test_three_params_per_qubit_per_collector(self):
        circuit = QuantumCircuit(3)
        with circuit.box([Twirl()]):
            circuit.noop(0, 1, 2)
        graph = graph_of(circuit)
        for node in of_kind(graph, "collect:"):
            self.assertEqual(len(node[2]), 3 * len(node[1]))


class TestUnoptimisedIsStillValid(QiskitTestCase):
    """Lowering unmerged IR2 is correct, just larger."""

    def test_every_emission_reaches_a_collector_either_way(self):
        circuit = QuantumCircuit(4)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
        # Unoptimized: all 4 emissions are standalone VFG nodes
        graph = graph_of(circuit, optimize=False)
        self.assertEqual(len(of_kind(graph, "emit:")), 4)
        sources = {a for a, _, _ in graph.edges()}
        emit_positions = {
            i for i, node in enumerate(graph.nodes()) if node[0].startswith("emit:")
        }
        self.assertTrue(emit_positions <= sources)

        # Optimized (with absorption): only incoming (far) halves remain as VFG nodes
        graph = graph_of(circuit, optimize=True)
        self.assertEqual(len(of_kind(graph, "emit:")), 2)
        sources = {a for a, _, _ in graph.edges()}
        emit_positions = {
            i for i, node in enumerate(graph.nodes()) if node[0].startswith("emit:")
        }
        self.assertTrue(emit_positions <= sources)

    def test_optimizing_shrinks_the_graph(self):
        circuit = QuantumCircuit(4)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
        self.assertLess(
            graph_of(circuit, optimize=True).num_nodes,
            graph_of(circuit, optimize=False).num_nodes,
        )


class TestDeterminism(QiskitTestCase):
    def test_graph_is_stable(self):
        circuit = QuantumCircuit(4)
        with circuit.box([Twirl(), ChangeBasis("ref")]):
            circuit.h(0)
            circuit.cx(0, 1)
        with circuit.box([Twirl()]):
            circuit.cx(2, 3)
        runs = []
        for _ in range(3):
            graph = graph_of(circuit)
            runs.append((sorted(graph.nodes()), sorted(graph.edges())))
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(runs[0], runs[2])


class TestVirtualTypePreservation(QiskitTestCase):
    """The real limit on supported circuits: propagation must stay inside the virtual group."""

    def test_pauli_through_cliffords_is_accepted(self):
        for entangler in ("cx", "cz", "ecr"):
            with self.subTest(entangler=entangler):
                circuit = QuantumCircuit(2)
                with circuit.box([Twirl()]):
                    getattr(circuit, entangler)(0, 1)
                self.assertGreater(graph_of(circuit).num_nodes, 0)

    def test_pauli_through_a_fractional_entangler_is_accepted(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl()]):
            circuit.rzz(0.3, 0, 1)
        self.assertGreater(graph_of(circuit).num_nodes, 0)

    def test_pauli_through_a_non_clifford_is_refused(self):
        # Conjugating a Pauli by a T leaves the Pauli group, so there is no rule to apply. Refusing is
        # the point: the alternative is a randomization that silently does not cancel.
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
            circuit.t(0)
        with self.assertRaisesRegex(ValueError, "cannot propagate a pauli virtual gate through 't'"):
            graph_of(circuit)

    def test_local_u2_through_an_entangler_is_refused(self):
        # A local U2 element stays local under single-qubit gates but not under an entangler.
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(distribution="haar_u2")]):
            circuit.cx(0, 1)
        with self.assertRaisesRegex(ValueError, "cannot propagate a u2 virtual gate"):
            graph_of(circuit)

    def test_absorbed_non_cliffords_are_fine(self):
        # An absorbed gate is multiplied into the collector, not conjugated through, so it never needs
        # a propagation rule. An rz on a clean wire is absorbable and must not be refused.
        circuit = QuantumCircuit(1)
        with circuit.box([Twirl()]):
            circuit.rz(0.3, 0)
        graph = graph_of(circuit)
        self.assertEqual(gates(of_kind(graph, "collect:")[0]), [("rz", [0])])
