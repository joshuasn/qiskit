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

"""Lower a samplex emission circuit (IR2) into a template circuit and sampling graph."""

from qiskit._accelerate.samplex import lower
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError

from .build import DISTRIBUTION_TABLE

#: Property set key for the parametric template circuit.
TEMPLATE = "samplex_template"
#: Property set key for the sampling graph that computes the template's angles.
FLOW_GRAPH = "samplex_flow_graph"
#: Property set key for the table of symbolic angles absorbed into collectors.
PARAMETERS = "samplex_parameters"


class SamplexLower(AnalysisPass):
    """Produce the template circuit and sampling graph from an emission circuit.

    Each collector becomes a *synth template*: the fixed parametric fragment whose angles the sampling
    graph will fill in. The gates absorbed into its body are not executed separately, because they fold
    into those angles -- which is the whole point of having absorbed them -- but their angles travel on
    the graph, since whatever computes those angles needs them. ``Emit`` markers are markers only and
    disappear, and hard boxes were a grouping so their content flattens out.

    All three outputs are read off the same IR2 circuit, so the graph's parameter ranges are exactly the
    ones the template minted rather than two things that have to be kept in agreement.

    This is an analysis pass: the emission circuit is left untouched and the three artifacts go into the
    property set, as ``"samplex_template"``, ``"samplex_flow_graph"`` and ``"samplex_parameters"``. The
    template is a terminal product -- a flat parametric circuit for execution, not something further
    passes transform -- which is why it does not replace the DAG in the pipeline.

    The parameter table is empty unless some absorbed gate carried a symbolic angle. When it is not, its
    ``free_parameters`` are a *required input to sampling*: a collector's angles are a function of them,
    so they must be bound before drawing.

    Requires the distribution table :class:`.SamplexBuild` leaves in the property set. Because
    parameters are minted here, every pass that changes the number or width of collectors must have run
    already; see :mod:`qiskit.transpiler.passes.samplex`.
    """

    def run(self, dag):
        """Run the SamplexLower pass on ``dag``.

        Args:
            dag (DAGCircuit): the emission circuit (IR2).

        Raises:
            TranspilerError: if the distribution table is not in the property set, meaning
                :class:`.SamplexBuild` has not run.
            ValueError: if an emission cannot be propagated to its collector, or if IR2 is internally
                inconsistent.
        """
        table = self.property_set[DISTRIBUTION_TABLE]
        if table is None:
            raise TranspilerError(
                "SamplexLower requires the distribution table that SamplexBuild puts in the property "
                f"set under {DISTRIBUTION_TABLE!r}; run SamplexBuild first."
            )
        template, graph, parameters = lower(dag, table)
        self.property_set[TEMPLATE] = QuantumCircuit._from_circuit_data(template)
        self.property_set[FLOW_GRAPH] = graph
        self.property_set[PARAMETERS] = parameters
