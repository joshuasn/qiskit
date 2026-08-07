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

"""Absorb dressing gates and emission markers into samplex collectors."""

from qiskit._accelerate.samplex import absorb_dressing
from qiskit.transpiler.basepasses import TransformationPass


class SamplexAbsorbDressing(TransformationPass):
    """Take the dressing around each collector into the collector itself.

    After build, every emission marker and every easy single-qubit gate sits on the circuit spine and
    every collector is empty. This pass walks outward from each collector along its own wires and takes
    over what it reaches: single-qubit gates become part of the collector's body, and emissions facing
    it become steps it composes. Absorption is per wire, so an entangler on q0 stops the walk on q0 and
    q1 without saying anything about q2.

    Absorbing is what lets the collector's angles account for those gates instead of the template
    executing them separately, so it shrinks the template as well as recording the composition order
    the sampling graph needs. An emission whose collector sits inside an adjacent box descends into
    that box and is absorbed there.

    Runs in place and may be composed either side of :class:`.SamplexMergeCollectors`.
    """

    def run(self, dag):
        """Run the SamplexAbsorbDressing pass on ``dag``, in place.

        Args:
            dag (DAGCircuit): the emission circuit (IR2).

        Returns:
            DAGCircuit: the same ``dag``, with dressing absorbed into its collectors.
        """
        absorb_dressing(dag)
        return dag
