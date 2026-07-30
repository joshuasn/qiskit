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

use pyo3::prelude::*;

pub mod build;
pub mod merge_collectors;
pub mod lower;
mod merge_parallel_nodes;
mod prune;
mod set_virtual_types;
mod utils;

pub use merge_parallel_nodes::merge_parallel_nodes;
pub use prune::{prune_unreachable_from_sinks, prune_unreachable_from_sources};
pub use set_virtual_types::set_virtual_types;

use crate::virtual_flow_graph::VirtualFlowGraph;

#[pyfunction]
pub fn py_merge_parallel(vfg: &mut VirtualFlowGraph) {
    merge_parallel_nodes(vfg);
}

#[pyfunction]
pub fn py_prune_from_sources(vfg: &mut VirtualFlowGraph) {
    prune_unreachable_from_sources(vfg);
}

#[pyfunction]
pub fn py_prune_from_sinks(vfg: &mut VirtualFlowGraph) {
    prune_unreachable_from_sinks(vfg);
}

#[pyfunction]
pub fn py_set_virtual_types(vfg: &mut VirtualFlowGraph) {
    set_virtual_types(vfg);
}
