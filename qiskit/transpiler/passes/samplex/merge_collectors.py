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

"""Merge adjacent samplex collectors so neighbouring boxes share a dressing layer."""

from qiskit._accelerate.samplex import merge_collectors
from qiskit.transpiler.basepasses import TransformationPass


class SamplexMergeCollectors(TransformationPass):
    """Let adjacent boxes sharing a synthesizer share a dressing layer.

    Build is local, so N boxes in a row arrive with 2N collectors. Where two are adjacent on a shared
    wire with nothing in between, they contract into one collector spanning the union of their qubits
    -- so N boxes need N+1 dressing layers rather than 2N. On the four-qubit notebook circuit that is
    six collectors and 48 template parameters down to four and 36. A collector nested inside a box is
    folded into the one just outside it by the same reasoning, which is a layer per nesting level.

    This is an optimization, not a correctness requirement: :class:`.SamplexLower` handles unmerged
    IR2 perfectly well, just with more dressing layers than necessary. It is only invalid the other way
    round -- merging after lowering would invalidate parameter labels already minted -- so the pass is
    freely omitted but must precede lowering.

    Runs in place, and is best run *after* :class:`.SamplexAbsorbDressing`. Two collectors merge only
    with nothing between them, and absorption is what clears what is: run first instead and this pass
    finds less to do -- on a nested box, four collectors and 24 parameters where the other order gives
    two and 12. Freely omitted, then, but not freely reordered.
    """

    def run(self, dag):
        """Run the SamplexMergeCollectors pass on ``dag``, in place.

        Args:
            dag (DAGCircuit): the emission circuit (IR2).

        Returns:
            DAGCircuit: the same ``dag``, with mergeable collectors contracted.

        Raises:
            ValueError: if a group of collectors selected for merging cannot be contracted, which
                indicates the merge conditions and the circuit's structure disagree.
        """
        merge_collectors(dag)
        return dag
