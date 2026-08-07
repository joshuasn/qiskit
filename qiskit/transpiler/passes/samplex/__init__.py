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

"""Transpiler passes for the samplex lowering pipeline.

Samplex compiles a circuit carrying randomization annotations into an executable pair: a *template*
circuit whose angles are left parametric, and a *sampling graph* that says how to draw those angles.
It gets there through three intermediate representations::

    IR1 annotated circuit  --SamplexBuild-->  IR2 emission circuit  --SamplexLower-->  template + graph

IR1 and IR2 are both ``DAGCircuit``, so the middle of the pipeline is ordinary in-place transpiler
work and the passes below compose in a :class:`~qiskit.transpiler.PassManager`::

    PassManager([
        SamplexBuild(),
        SamplexMergeCollectors(),   # optional; fewer, wider dressing layers
        SamplexAbsorbDressing(),
        SamplexLower(),
    ])

Two ordering constraints are real rather than conventional. :class:`SamplexLower` mints the
template's parameters, so anything that changes how many collectors exist or how wide they are has to
run before it -- lowering *unmerged* IR2 is correct but uses more dressing layers than it needs,
while merging after lowering invalidates the labels already assigned. And :class:`SamplexLower`
consumes the distribution table :class:`SamplexBuild` puts in the property set, so it cannot run
without it.

These passes are deliberately not re-exported from :mod:`qiskit.transpiler.passes`: samplex has no
stable Python surface yet, and importing from this module is what keeps that explicit.
"""

from .absorb_dressing import SamplexAbsorbDressing
from .build import SamplexBuild
from .lower import SamplexLower
from .merge_collectors import SamplexMergeCollectors

__all__ = [
    "SamplexAbsorbDressing",
    "SamplexBuild",
    "SamplexLower",
    "SamplexMergeCollectors",
]
