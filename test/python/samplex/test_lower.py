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

"""Tests for template construction: emission circuit (IR2) -> template circuit.

Each collector becomes the parametric fragment its angles drive; absorbed gates are discarded because
they fold into those angles; emissions disappear; hard boxes flatten. Parameters are minted here and
nowhere earlier.
"""

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit._accelerate.samplex import (
    ChangeBasis,
    Twirl,
    absorb_emissions,
    build_lowered,
    build_template,
    merge_collectors,
)

from test import QiskitTestCase
from test.python.samplex.test_build import gate_names


def emission_circuit(circuit):
    data, _ = build_lowered(circuit_to_dag(circuit))
    return data


def template(circuit_data):
    data, collectors = build_template(circuit_data)
    return QuantumCircuit._from_circuit_data(data), collectors


def notebook_circuit():
    circuit = QuantumCircuit(4)
    with circuit.box([Twirl(), ChangeBasis("ref")]):
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(2, 3)
    with circuit.box([Twirl()]):
        circuit.cx(0, 1)
    with circuit.box([Twirl()]):
        circuit.cx(2, 3)
    return circuit


class TestTemplateShape(QiskitTestCase):
    """What ends up in the template."""

    def test_emissions_and_boxes_are_gone(self):
        out, _ = template(emission_circuit(notebook_circuit()))
        names = set(gate_names(out))
        self.assertNotIn("box", names)
        self.assertFalse({n for n in names if n.startswith("samplex")})

    def test_hard_content_survives_flattened(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl()]):
            circuit.cx(0, 1)
        out, _ = template(emission_circuit(circuit))
        self.assertEqual(gate_names(out).count("cx"), 1)

    def test_absorbed_gates_are_discarded(self):
        # The absorbed `h` folds into the collector's synthesized angles, so it must not also be
        # executed. If it appeared here it would be applied twice.
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl(dressing="left")]):
            circuit.h(0)
            circuit.cx(0, 1)
        out, _ = template(emission_circuit(circuit))
        self.assertNotIn("h", gate_names(out))
        self.assertIn("cx", gate_names(out))

    def test_rzsx_fragment(self):
        circuit = QuantumCircuit(1)
        with circuit.box([Twirl(decomposition="rzsx")]):
            circuit.noop(0)
        out, collectors = template(emission_circuit(circuit))
        self.assertEqual(gate_names(out), ["rz", "sx", "rz", "sx", "rz"] * 2)
        self.assertEqual([len(c[3]) for c in collectors], [3, 3])

    def test_rzrx_fragment(self):
        circuit = QuantumCircuit(1)
        with circuit.box([Twirl(decomposition="rzrx")]):
            circuit.noop(0)
        out, _ = template(emission_circuit(circuit))
        self.assertEqual(gate_names(out), ["rz", "rx", "rz"] * 2)

    def test_fragment_is_written_on_every_collector_qubit(self):
        circuit = QuantumCircuit(3)
        with circuit.box([Twirl()]):
            circuit.noop(0, 1, 2)
        out, collectors = template(emission_circuit(circuit))
        # three qubits x three angles, for each of the two collectors
        self.assertEqual([len(c[3]) for c in collectors], [9, 9])
        self.assertEqual(out.num_parameters, 18)


class TestParameterLabelling(QiskitTestCase):
    """Parameters are minted here, in circuit order."""

    def test_indices_are_contiguous_and_ordered(self):
        _, collectors = template(emission_circuit(notebook_circuit()))
        allocated = [i for _, _, _, params in collectors for i in params]
        self.assertEqual(allocated, list(range(len(allocated))))

    def test_count_is_three_per_qubit_per_collector(self):
        out, collectors = template(emission_circuit(notebook_circuit()))
        expected = sum(3 * len(qubits) for qubits, _, _, _ in collectors)
        self.assertEqual(out.num_parameters, expected)

    def test_merging_reduces_the_parameter_vector(self):
        # This is why labels cannot be minted before merging: the count and every subsequent index
        # would shift. Lowering unmerged is correct, only suboptimal.
        data = emission_circuit(notebook_circuit())
        unmerged, unmerged_collectors = template(data)
        merged, merged_collectors = template(merge_collectors(absorb_emissions(data)))

        self.assertEqual((len(unmerged_collectors), unmerged.num_parameters), (6, 48))
        self.assertEqual((len(merged_collectors), merged.num_parameters), (4, 36))

    def test_lowering_unmerged_is_still_complete(self):
        # Every emission still has exactly one collector responsible for it.
        data = emission_circuit(notebook_circuit())
        _, collectors = template(data)
        collected = [i for _, _, collects, _ in collectors for i in collects]
        self.assertEqual(sorted(collected), list(range(len(collected))))

    def test_parameter_names_are_zero_padded(self):
        # Zero padding makes lexicographic order match numeric order, as samplomatic's ParamIter does.
        out, _ = template(emission_circuit(notebook_circuit()))
        names = sorted(p.name for p in out.parameters)
        self.assertEqual(names, sorted(names, key=lambda n: int(n[1:])))


class TestDeterminism(QiskitTestCase):
    """Repeated runs agree on structure, though not on Parameter identity."""

    def test_structure_is_stable(self):
        runs = []
        for _ in range(3):
            out, collectors = template(emission_circuit(notebook_circuit()))
            runs.append(
                (
                    gate_names(out),
                    sorted(p.name for p in out.parameters),
                    [(tuple(q), s, tuple(c), tuple(p)) for q, s, c, p in collectors],
                )
            )
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(runs[0], runs[2])

    def test_parameters_are_distinct_objects_across_runs(self):
        # Freshly minted parameters carry fresh uuids, matching Python's Parameter semantics. So
        # determinism means same names and indices, not equal objects.
        first, _ = template(emission_circuit(notebook_circuit()))
        second, _ = template(emission_circuit(notebook_circuit()))
        self.assertEqual(
            sorted(p.name for p in first.parameters),
            sorted(p.name for p in second.parameters),
        )
        self.assertNotEqual(set(first.parameters), set(second.parameters))


class TestNesting(QiskitTestCase):
    """Nested collectors are lowered too."""

    def test_nested_collectors_get_parameters(self):
        circuit = QuantumCircuit(2)
        with circuit.box([Twirl()]):
            with circuit.box([Twirl()]):
                circuit.cx(0, 1)
        out, collectors = template(emission_circuit(circuit))
        # two collectors for the outer box, two for the inner
        self.assertEqual(len(collectors), 4)
        self.assertEqual(out.num_parameters, 4 * 2 * 3)
        self.assertNotIn("box", gate_names(out))
