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

//! Sampling randomizations of quantum circuits, as a chain of circuit dialects.
//!
//! ```text
//! IR1  annotated circuit    DAGCircuit + samplex annotations on boxes    annotated_circuit
//!   --build-->
//! IR2  emission circuit     Emit instructions, Collect boxes, hard       emission_circuit
//!      + DistributionTable  boxes                                        distributions
//!   --absorb_dressing, merge_collectors-->
//! IR3  sampling graph       dataflow over virtual state                  virtual_flow_graph
//!      + template circuit   parameterized circuit
//!      + ParameterTable     the symbolic absorbed angles                 parameters
//!   --merge_parallel_nodes, prune, set_virtual_types-->
//! ```
//!
//! Each root module holds one stage's vocabulary; each file under [`passes`] is one transform.
//! `absorb_dressing` must precede `merge_collectors` (merging concatenates collector bodies), and
//! both must precede `lower` (which mints the template's parameters).
//!
//! IR3 is plain data: the IR3 passes hold no `Python` token, and lowering pays the GIL once at the
//! IR2 boundary.

use pyo3::prelude::*;

macro_rules! parse_enum {
    ($vis:vis $fn_name:ident, $enum_type:ty, $label:literal,
     { $($str:literal => $variant:ident),+ $(,)? }) => {
        $vis fn $fn_name(s: &str) -> pyo3::PyResult<$enum_type> {
            match s {
                $( $str => Ok(<$enum_type>::$variant), )+
                _ => Err(pyo3::exceptions::PyValueError::new_err(
                    format!(concat!("Unknown ", $label, ": '{}'"), s)
                )),
            }
        }
    };
}

pub mod annotated_circuit;
pub mod distributions;
pub mod error;
pub mod emission_circuit;
pub mod parameters;
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
    m.add_class::<parameters::ParameterTable>()?;
    m.add_class::<emission_circuit::Emit>()?;
    m.add_class::<virtual_flow_graph::VirtualFlowGraph>()?;
    m.add_wrapped(wrap_pyfunction!(passes::build::py_build))?;
    m.add_wrapped(wrap_pyfunction!(passes::absorb_emissions::py_absorb_dressing))?;
    m.add_wrapped(wrap_pyfunction!(passes::merge_collectors::py_merge_collectors))?;
    m.add_wrapped(wrap_pyfunction!(passes::lower::py_build_template))?;
    m.add_wrapped(wrap_pyfunction!(passes::lower::py_lower))?;
    m.add_wrapped(wrap_pyfunction!(passes::py_merge_parallel))?;
    m.add_wrapped(wrap_pyfunction!(passes::py_prune_from_sources))?;
    m.add_wrapped(wrap_pyfunction!(passes::py_prune_from_sinks))?;
    m.add_wrapped(wrap_pyfunction!(passes::py_set_virtual_types))?;
    Ok(())
}
