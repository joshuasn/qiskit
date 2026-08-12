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
from qiskit.circuit import Parameter, ParameterVector
from qiskit.converters import circuit_to_dag
from qiskit._accelerate.samplex import (
    ChangeBasis,
    InjectNoise,
    Twirl,
    absorb_dressing,
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
        absorb_dressing(dag)
    _, graph, _ = lower(dag, table)
    return graph


def graph_and_table_of(circuit, optimize=True):
    """The sampling graph and its distribution table, for tests that need to resolve emissions."""
    dag, table = build_lowered(circuit_to_dag(circuit))
    if optimize:
        merge_collectors(dag)
        absorb_dressing(dag)
    _, graph, _ = lower(dag, table)
    return graph, table


def artifacts(circuit, optimize=True):
    """All three lowering outputs, for the tests that need the parameter table too."""
    dag, table = build_lowered(circuit_to_dag(circuit))
    if optimize:
        merge_collectors(dag)
        absorb_dressing(dag)
    return lower(dag, table)


def kinds(graph, table=None):
    return [node[0] for node in graph.nodes(table)]


def of_kind(graph, prefix, table=None):
    return [node for node in graph.nodes(table) if node[0].startswith(prefix)]


def wiring(graph, table=None):
    """Edges as (source kind, target kind, direction)."""
    nodes = graph.nodes(table)
    return [(nodes[a][0], nodes[b][0], d) for a, b, d in graph.edges()]


def gates(node):
    """A collector's absorbed gates, dropping the emissions interleaved between them."""
    return [step for step in node[3] if step[0] != "emit"]


def on_wire(node, qubit):
    """A collector's steps projected onto one qubit, as ``(name, angles)`` pairs.

    This is the projection that ``steps`` actually guarantees. The flat sequence is one linear
    extension of the body among several -- steps on disjoint wires come back lowest-qubit-first
    whatever order they were written in -- so a wire's own subsequence is what a consumer may rely on.
    """
    return [(name, angles) for name, qubits, angles in node[3] if qubit in qubits]


class TestCollectorOwnsAbsorbedGates(QiskitTestCase):
    """Absorbed gates are steps on the Collect node, not nodes of their own.

    A collector's `steps` are one sequence mixing the emissions it consumes with the gates it absorbed,
    because where a gate sits *relative to the same wire* matters — a `ChangeBasis` wraps the absorbed
    gates while a twirl composes inside them. Order between disjoint wires carries no meaning; see
    TestStepsOrderIsPerWire.
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
        self.assertEqual(left[3], [("h", [0], []), ("emit", [0, 1], [])])
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
        self.assertEqual(left[3], [("emit", [0, 1], []), ("h", [0], []), ("emit", [0, 1], [])])

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
        self.assertEqual(gates(left), [("h", [0], []), ("s", [0], [])])

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
        # both contributions are there, and on each wire the absorbed gate sits between the two local
        # emissions — the two layers meeting outermost-to-outermost. Asserted per wire rather than as a
        # flat list: `s` and `h` are on disjoint qubits, so their relative position in the sequence is
        # an artifact of the topological read. See TestStepsOrderIsPerWire.
        self.assertEqual(on_wire(middle, 0), [("emit", []), ("s", []), ("emit", [])])
        self.assertEqual(on_wire(middle, 1), [("emit", []), ("h", []), ("emit", [])])


class TestStepsOrderIsPerWire(QiskitTestCase):
    """`steps` is a linear extension of the body's per-qubit order, not circuit order.

    Lowering reads a collector body with ``topological_op_nodes``, whose tie-break is lexicographic on
    ``(qargs, cargs)`` -- so two steps on disjoint wires come back lowest-qubit-first however they were
    written. The field used to be documented as circuit order, which is false, and an assertion on the
    flat sequence pins an arbitrary choice rather than a guarantee.

    Nothing is broken by this: a collector synthesizes three angles *per qubit*, every absorbed gate is
    single-qubit, and single-qubit gates on distinct qubits commute, so every linear extension of one
    body evaluates identically. These tests pin what is guaranteed, and pin the divergence itself so it
    stays a known property rather than a surprise.
    """

    @staticmethod
    def diverging():
        """A merged collector whose reported order is provably not circuit order.

        The first box contributes ``s`` on the *higher* wire and the second ``h`` on the lower one, so
        circuit order is ``s`` then ``h`` and the lexicographic tie-break reverses them.
        """
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="right")]):
            circuit.cx(0, 1)
            circuit.s(1)
        with circuit.box([Twirl(dressing="left")]):
            circuit.h(0)
            circuit.cx(0, 1)
        return circuit

    def merged(self):
        graph = graph_of(self.diverging())
        return next(c for c in of_kind(graph, "collect:") if len(gates(c)) > 1)

    def test_the_flat_sequence_is_not_circuit_order(self):
        # `s` is from the first box and `h` from the second, so circuit order is s, h -- and this is
        # the reverse. Pinned deliberately: it is the thing a consumer must not depend on.
        self.assertEqual([step[0] for step in gates(self.merged())], ["h", "s"])

    def test_each_wire_keeps_its_own_order(self):
        # What is guaranteed. Each wire sees its absorbed gate between the two local emissions, which
        # is what makes "the twirl factor composes inside the easy gates" well defined.
        middle = self.merged()
        self.assertEqual([name for name, _ in on_wire(middle, 0)], ["emit", "h", "emit"])
        self.assertEqual([name for name, _ in on_wire(middle, 1)], ["emit", "s", "emit"])

    def test_a_local_emission_is_a_barrier_on_every_wire_it_covers(self):
        # A local emission spans all its qubits, so no linear extension can move an absorbed gate
        # across it. That is what survives the re-read, and what the per-wire guarantee rests on.
        middle = self.merged()
        for qubit in (0, 1):
            names = [name for name, _ in on_wire(middle, qubit)]
            self.assertEqual(names[0], "emit")
            self.assertEqual(names[-1], "emit")
            self.assertEqual(names.count("emit"), 2)

    def test_order_is_stable_across_runs(self):
        # Arbitrary between wires, but not *random*: the same circuit lowers to the same sequence, so
        # the choice is reproducible even though it is not meaningful.
        runs = [[c[3] for c in of_kind(graph_of(self.diverging()), "collect:")] for _ in range(3)]
        self.assertEqual(runs, [runs[0]] * 3)


class TestAbsorbedAngles(QiskitTestCase):
    """An absorbed gate's angle travels with the graph.

    A collector folds its absorbed gates into the angles it synthesizes, so those gates are deliberately
    *not* written into the template. That makes the graph the only place their angles can live: dropping
    them left an absorbed ``rz(0.3)`` in neither artifact, so no binding of the template could reproduce
    the circuit. Bound angles ride inline on the step; symbolic ones are keys into the parameter table,
    whose ``free_parameters`` are what a caller still has to supply.
    """

    def test_a_bound_angle_rides_inline(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl()]):
            circuit.rz(0.3, 0)
            circuit.cx(0, 1)
        _, graph, params = artifacts(circuit)

        absorbed = [step for c in of_kind(graph, "collect:") for step in gates(c)]
        self.assertEqual(absorbed, [("rz", [0], ["0.3"])])
        # nothing symbolic, so nothing for the caller to bind
        self.assertEqual(len(params), 0)
        self.assertEqual(params.free_parameters, [])

    def test_a_symbolic_angle_becomes_a_key_the_caller_must_bind(self):
        theta = Parameter("t")
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl()]):
            circuit.rz(theta, 0)
            circuit.cx(0, 1)
        _, graph, params = artifacts(circuit)

        absorbed = [step for c in of_kind(graph, "collect:") for step in gates(c)]
        self.assertEqual(absorbed, [("rz", [0], ["#0"])])
        self.assertEqual(params.entries(), ["t"])
        self.assertEqual(params.free_parameters, ["t"])

    def test_angles_absorbed_from_the_spine_survive_too(self):
        # The case with the widest reach: `absorb_dressing` pulls in single-qubit gates from *outside*
        # the box, so a rotation the user never put in a box still loses its angle if it is not read.
        theta = Parameter("t")
        circuit = QuantumCircuit(2)
        circuit.rz(theta, 0)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
        circuit.rz(0.5, 1)
        _, graph, params = artifacts(circuit)

        absorbed = [step for c in of_kind(graph, "collect:") for step in gates(c)]
        self.assertEqual(sorted(absorbed), [("rz", [0], ["#0"]), ("rz", [1], ["0.5"])])
        self.assertEqual(params.free_parameters, ["t"])

    def test_a_merged_collector_keeps_both_contributions_angles(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="right")]):
            circuit.cx(0, 1)
            circuit.rz(0.25, 0)
        with circuit.box([Twirl(dressing="left")]):
            circuit.rz(0.75, 1)
            circuit.cx(0, 1)
        _, graph, _ = artifacts(circuit)

        middle = next(c for c in of_kind(graph, "collect:") if len(gates(c)) > 1)
        self.assertEqual(sorted(gates(middle)), [("rz", [0], ["0.25"]), ("rz", [1], ["0.75"])])

    def test_one_symbol_on_two_gates_is_one_entry(self):
        theta = Parameter("t")
        circuit = QuantumCircuit(2)
        circuit.rz(theta, 0)
        circuit.rz(theta, 1)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
        _, graph, params = artifacts(circuit)

        absorbed = [step for c in of_kind(graph, "collect:") for step in gates(c)]
        self.assertEqual(sorted(absorbed), [("rz", [0], ["#0"]), ("rz", [1], ["#0"])])
        self.assertEqual(len(params), 1)
        self.assertEqual(params.free_parameters, ["t"])

    def test_a_fully_bound_expression_is_not_a_free_parameter(self):
        # `2 * t` bound to a value is arithmetic, not something to supply, so it folds to a plain angle
        # and the table stays empty. That is what keeps `free_parameters` meaning "still needed".
        theta = Parameter("t")
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl()]):
            circuit.rz(2 * theta, 0)
            circuit.cx(0, 1)
        circuit = circuit.assign_parameters({theta: 0.5})
        _, graph, params = artifacts(circuit)

        absorbed = [step for c in of_kind(graph, "collect:") for step in gates(c)]
        self.assertEqual(absorbed, [("rz", [0], ["1"])])
        self.assertEqual(len(params), 0)
        self.assertEqual(params.free_parameters, [])

    def test_parameter_vector_elements_are_named_individually(self):
        # A vector element's bare `name` is the shared vector name, so listing by it would collapse the
        # elements into one. The caller binds `v[0]` and `v[1]` separately.
        vector = ParameterVector("v", 2)
        circuit = QuantumCircuit(2)
        circuit.rz(vector[0], 0)
        circuit.rz(vector[1], 1)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
        _, _, params = artifacts(circuit)

        self.assertEqual(params.free_parameters, ["v[0]", "v[1]"])

    def test_the_table_is_deterministic(self):
        # `iter_symbols` walks a HashMap, so an unsorted free list would differ run to run.
        vector = ParameterVector("w", 4)
        circuit = QuantumCircuit(4)
        for index in range(4):
            circuit.rz(vector[index], index)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
            circuit.cx(2, 3)

        runs = [artifacts(circuit)[2] for _ in range(3)]
        self.assertEqual(
            [run.free_parameters for run in runs],
            [["w[0]", "w[1]", "w[2]", "w[3]"]] * 3,
        )
        self.assertEqual([run.entries() for run in runs], [runs[0].entries()] * 3)


class TestPropagation(QiskitTestCase):
    """Which gates conjugate which emission, derived from placement."""

    def test_left_factor_reaches_its_collector_directly(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            circuit.h(0)
            circuit.cx(0, 1)
        graph, table = graph_and_table_of(circuit)

        # The near (left) factor is local — absorbed, no graph edge. Only the far (right) factor
        # propagates through the hard box: emit -> cx -> right collector.
        self.assertIn(("emit:UniformPauli", "propagate:cx", "right"), wiring(graph, table))
        self.assertIn(("propagate:cx", "collect:RzSx", "right"), wiring(graph, table))

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
        with circuit.box([Twirl(), ChangeBasis("b0", placement="start"), InjectNoise("n0", "after")]):
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
                    with circuit.box([Twirl(dressing=dressing), ChangeBasis("b", placement=placement)]):
                        circuit.cx(0, 1)
                    graph, table = graph_and_table_of(circuit)

                    conjugated = {
                        src for src, tgt, _ in wiring(graph, table) if tgt == "propagate:cx"
                    }
                    self.assertEqual(conjugated, {"emit:UniformPauli"})

    def test_injected_noise_is_never_conjugated_by_box_content(self):
        for dressing in ("left", "right"):
            for site in ("before", "after"):
                with self.subTest(dressing=dressing, site=site):
                    circuit = QuantumCircuit(2)
                    with circuit.box([Twirl(dressing=dressing), InjectNoise("n0", site)]):
                        circuit.cx(0, 1)
                    graph, table = graph_and_table_of(circuit)

                    conjugated = {
                        src for src, tgt, _ in wiring(graph, table) if tgt == "propagate:cx"
                    }
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
        absorb_dressing(dag)
        merge_collectors(dag)
        template, graph, _ = lower(dag, table)

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
        emit_positions = {i for i, node in enumerate(graph.nodes()) if node[0].startswith("emit:")}
        self.assertTrue(emit_positions <= sources)

        # Optimized (with absorption): only incoming (far) halves remain as VFG nodes
        graph = graph_of(circuit, optimize=True)
        self.assertEqual(len(of_kind(graph, "emit:")), 2)
        sources = {a for a, _, _ in graph.edges()}
        emit_positions = {i for i, node in enumerate(graph.nodes()) if node[0].startswith("emit:")}
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
        # and its angle comes along, because the collector folds it into what it synthesizes
        self.assertEqual(gates(of_kind(graph, "collect:")[0]), [("rz", [0], ["0.3"])])


class TestNestedPropagation(QiskitTestCase):
    """Where a nested box's collectors leave an enclosing emission, under the current rule.

    The asymmetry to keep straight: a box's own emissions split its body — the absorbable run multiplies
    into the near factor, the rest propagates the far one — while an *enclosing* emission ought to treat
    all of it as one unit, because every part of the inner box sits inside the outer twirl point.

    **That is not what happens yet, and these tests pin what does.** The nearest compatible collector
    wins, so the enclosing box's far half is taken by the first nested collector it passes and composed
    there with no conjugation at all: the outer randomization is applied and immediately undone with none
    of its content in between. The unitary is unchanged, so only the randomization is lost, which is why
    it has to be asserted on the graph rather than via a round trip.

    The fix belongs in compatibility, not in position or in an id: once an emission carries a type an
    inner collector can decline, it will propagate past and reach its own. When that lands, these
    assertions are the ones that should change. See `lower::compatible`.
    """

    def circuit(self):
        """Left-dressed outer box over a left-dressed inner box plus a gate of its own."""
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            with circuit.box([Twirl(dressing="left")]):
                circuit.cx(0, 1)  # inner content
            circuit.cx(1, 0)  # outer content, after the nested box
        return circuit

    def test_only_the_innermost_far_half_travels(self):
        graph = graph_of(self.circuit())

        # The enclosing box's far half was absorbed by the inner box's left collector, so it is a local
        # table read there rather than a node with a path. Only the inner far half is left travelling.
        self.assertEqual(len(of_kind(graph, "emit:")), 1)
        # And it crosses only the inner content, so there is one conjugation rather than two.
        self.assertEqual(len(of_kind(graph, "propagate:")), 1)

    def test_the_outer_box_right_collector_receives_nothing(self):
        """The cost of the current rule, stated plainly so a change of rule is visible."""
        graph = graph_of(self.circuit())
        nodes = graph.nodes()
        collectors = [i for i, node in enumerate(nodes) if node[0].startswith("collect:")]
        # Circuit order, so the last collector is the outer box's right-hand one.
        outer_right = collectors[-1]

        incoming = [(a, b) for a, b, _ in graph.edges() if b == outer_right]
        self.assertFalse(
            incoming,
            "the outer box's right collector received a value, which means the enclosing far half "
            "reached it — the rule has changed and this test should now assert that it does",
        )

    def test_an_unnested_box_is_unchanged(self):
        """The common case keeps exactly one conjugation per hard gate."""
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            circuit.cx(0, 1)
        graph = graph_of(circuit)

        self.assertEqual(len(of_kind(graph, "emit:")), 1)
        self.assertEqual(len(of_kind(graph, "propagate:")), 1)
        self.assertEqual(len(of_kind(graph, "collect:")), 2)
