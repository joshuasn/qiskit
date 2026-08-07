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

"""Tests for the samplex transpiler pass wrappers.

The wrappers exist so the lowering pipeline composes in a `PassManager`. What is worth testing is that
composing them reproduces what calling the Rust functions directly does, and that the property-set
handoff from build to lower actually works -- not the lowering semantics, which the other modules in
this directory cover.
"""

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.samplex import (
    SamplexAbsorbEmissions,
    SamplexBuild,
    SamplexLower,
    SamplexMergeCollectors,
)
from qiskit._accelerate.samplex import (
    ChangeBasis,
    Twirl,
    absorb_emissions,
    build_lowered,
    lower,
    merge_collectors,
)

from test import QiskitTestCase


def notebook_circuit():
    """A wide box followed by two narrow ones covering its halves."""
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


def pipeline(merge=True):
    passes = [SamplexBuild()]
    if merge:
        passes.append(SamplexMergeCollectors())
    passes += [SamplexAbsorbEmissions(), SamplexLower()]
    return PassManager(passes)


def directly(circuit, merge=True):
    """The same lowering, called straight through the Rust entry points."""
    dag, table = build_lowered(circuit_to_dag(circuit))
    if merge:
        merge_collectors(dag)
    absorb_emissions(dag)
    return lower(dag, table)


class TestSamplexPasses(QiskitTestCase):
    """The wrappers compose, and agree with calling the passes directly."""

    def test_pipeline_matches_direct_calls(self):
        for merge in (False, True):
            with self.subTest(merge=merge):
                manager = pipeline(merge)
                manager.run(notebook_circuit())
                template = manager.property_set["samplex_template"]
                graph = manager.property_set["samplex_flow_graph"]

                expected_template, expected_graph = directly(notebook_circuit(), merge)
                self.assertEqual(
                    [inst.operation.name for inst in template.data],
                    [
                        inst.operation.name
                        for inst in QuantumCircuit._from_circuit_data(expected_template).data
                    ],
                )
                self.assertEqual(
                    [node[0] for node in graph.nodes()],
                    [node[0] for node in expected_graph.nodes()],
                )

    def test_merging_reduces_collectors_and_parameters(self):
        # The point of the optional merge pass, visible through the pass manager: fewer, wider dressing
        # layers means fewer template parameters for the same circuit.
        unmerged = pipeline(merge=False)
        unmerged.run(notebook_circuit())
        merged = pipeline(merge=True)
        merged.run(notebook_circuit())

        self.assertLess(
            merged.property_set["samplex_template"].num_parameters,
            unmerged.property_set["samplex_template"].num_parameters,
        )

    def test_build_publishes_the_distribution_table(self):
        manager = PassManager([SamplexBuild()])
        manager.run(notebook_circuit())
        self.assertIsNotNone(manager.property_set["samplex_distribution_table"])

    def test_lower_without_build_is_refused(self):
        # Lowering needs the table build produces, and says so rather than failing obscurely.
        dag, _table = build_lowered(circuit_to_dag(notebook_circuit()))
        with self.assertRaisesRegex(TranspilerError, "SamplexBuild"):
            SamplexLower().run(dag)

    def test_in_place_passes_return_the_same_dag(self):
        # The two IR2 -> IR2 passes mutate rather than rebuild, so a pass manager sees one DAG all the
        # way through. Worth pinning: it is the property that made the migration to DAGCircuit
        # worthwhile.
        dag, _table = build_lowered(circuit_to_dag(notebook_circuit()))
        self.assertIs(SamplexMergeCollectors().run(dag), dag)
        self.assertIs(SamplexAbsorbEmissions().run(dag), dag)
