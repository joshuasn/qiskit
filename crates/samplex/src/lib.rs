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

pub mod annotated_circuit;
pub mod distributions;
pub mod error;
pub mod emission_circuit;
pub mod partition;
pub mod passes;
pub mod virtual_flow_graph;
pub mod virtual_type;

pub fn samplex_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<annotated_circuit::Twirl>()?;
    m.add_class::<annotated_circuit::ChangeBasis>()?;
    m.add_class::<annotated_circuit::InjectLocalClifford>()?;
    m.add_class::<annotated_circuit::InjectNoise>()?;
    m.add_class::<annotated_circuit::Tag>()?;
    m.add_class::<emission_circuit::Collect>()?;
    m.add_class::<distributions::DistributionTable>()?;
    m.add_class::<emission_circuit::Emit>()?;
    m.add_class::<virtual_flow_graph::VirtualFlowGraph>()?;
    m.add_wrapped(wrap_pyfunction!(passes::build::py_build))?;
    m.add_wrapped(wrap_pyfunction!(passes::merge_collectors::py_merge_collectors))?;
    m.add_wrapped(wrap_pyfunction!(passes::lower::py_build_template))?;
    m.add_wrapped(wrap_pyfunction!(passes::lower::py_lower))?;
    m.add_wrapped(wrap_pyfunction!(passes::py_merge_parallel))?;
    m.add_wrapped(wrap_pyfunction!(passes::py_prune_from_sources))?;
    m.add_wrapped(wrap_pyfunction!(passes::py_prune_from_sinks))?;
    m.add_wrapped(wrap_pyfunction!(passes::py_set_virtual_types))?;
    Ok(())
}
