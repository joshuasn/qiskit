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

"""Build the samplex emission circuit (IR2) from an annotated circuit (IR1)."""

from qiskit._accelerate.samplex import build_lowered
from qiskit.transpiler.basepasses import TransformationPass

#: Property set key under which the distribution table travels from build to lower.
DISTRIBUTION_TABLE = "samplex_distribution_table"


class SamplexBuild(TransformationPass):
    """Turn an annotated circuit (IR1) into an emission circuit (IR2).

    Each annotated box becomes a *hard box* holding the original content, flanked by two *collectors*
    -- the dressing layers whose angles will be synthesized -- with ``Emit`` markers recording where
    randomization enters. Build is local: every annotated box gets its own pair of collectors, which
    is what leaves :class:`.SamplexMergeCollectors` something to do.

    The pass also produces a distribution table, the registry of what each ``Emit`` draws from. It has
    nowhere to live on the DAG, so it goes into the property set under ``"samplex_distribution_table"``
    for :class:`.SamplexLower` to pick up. That makes this a transformation pass that also writes the
    property set, which the base class's one-line description does not anticipate but the pass manager
    permits.

    Unlike the other three passes here, this one returns a *new* DAG rather than mutating its input:
    IR1 and IR2 are different representations that happen to share a type.
    """

    def run(self, dag):
        """Run the SamplexBuild pass on ``dag``.

        Args:
            dag (DAGCircuit): the annotated circuit (IR1).

        Returns:
            DAGCircuit: the emission circuit (IR2).

        Raises:
            ValueError: if the circuit uses control flow other than ``box``, or an annotation samplex
                does not recognize.
        """
        out, table = build_lowered(dag)
        self.property_set[DISTRIBUTION_TABLE] = table
        return out
