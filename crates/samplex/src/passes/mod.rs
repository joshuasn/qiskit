// This code is a Qiskit project.
//
// (C) Copyright IBM 2025, 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! The transforms, one file per pass. See the crate doc for the chain and the ordering constraints.

use pyo3::prelude::*;

pub mod absorb_dressing;
pub mod build;
pub mod lower;
pub mod merge_collectors;
mod merge_parallel_nodes;
mod prune;
mod set_virtual_types;

#[cfg(test)]
pub(crate) mod test_fixtures;

pub use merge_parallel_nodes::merge_parallel_nodes;
pub use prune::{prune_unreachable_from_sinks, prune_unreachable_from_sources};
pub use set_virtual_types::set_virtual_types;

use crate::sampling_graph::SamplingGraph;

#[pyfunction]
#[pyo3(name = "merge_parallel_nodes")]
pub fn py_merge_parallel(sg: &mut SamplingGraph) {
    merge_parallel_nodes(sg);
}

#[pyfunction]
#[pyo3(name = "prune_unreachable_from_sources")]
pub fn py_prune_from_sources(sg: &mut SamplingGraph) {
    prune_unreachable_from_sources(sg);
}

#[pyfunction]
#[pyo3(name = "prune_unreachable_from_sinks")]
pub fn py_prune_from_sinks(sg: &mut SamplingGraph) {
    prune_unreachable_from_sinks(sg);
}

#[pyfunction]
#[pyo3(name = "set_virtual_types")]
pub fn py_set_virtual_types(sg: &mut SamplingGraph) {
    set_virtual_types(sg);
}
