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
//! IR3  sampling graph       dataflow over virtual state                  sampling_graph
//!      + template circuit   parameterized circuit
//!      + ParameterTable     the symbolic absorbed angles                 parameters
//!   --merge_parallel_nodes, prune, set_virtual_types-->
//! ```
//!
//! Most root modules hold one stage's vocabulary, and each file under [`passes`] is one transform.
//! Two root modules are not stages but *readings* of IR2: [`emission_circuit_navigation`] is how a
//! pass walks a nested emission circuit, and [`spine`] is the flat reading of one that lowering
//! resolves propagations along. Both optimizations must precede `lower`, which mints the template's
//! parameters.
//!
//! `absorb_dressing` should precede `merge_collectors`, and the reason is what merging needs rather
//! than what it does to bodies: two collectors merge only with nothing between them, and absorption is
//! what clears what is — the twirl pairs, frame changes and foldable gates that otherwise sit there.
//! Run the other way round and merging finds less to do; on a nested box, four collectors and 24
//! template parameters where this order gives two and 12. Both orders produce valid IR2, so this is a
//! cost rather than a rule, and merging is freely omitted either way.
//!
//! IR3 is plain data: the IR3 passes hold no `Python` token, and lowering pays the GIL once at the
//! IR2 boundary. Which passes those are is readable off their signatures: everything that needs no
//! token returns [`error::Result`] rather than `PyResult`, and `From<SamplexError> for PyErr` is the
//! single seam where a refusal becomes a Python exception.

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
pub mod emission_circuit;
pub mod emission_circuit_navigation;
pub mod error;
pub mod parameters;
pub mod partition;
pub mod passes;
pub mod sampling_graph;
pub mod spine;
pub mod virtual_type;

pub fn samplex_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<annotated_circuit::PyTwirl>()?;
    m.add_class::<annotated_circuit::PyChangeBasis>()?;
    m.add_class::<annotated_circuit::PyInjectLocalClifford>()?;
    m.add_class::<annotated_circuit::PyInjectNoise>()?;
    m.add_class::<annotated_circuit::PyTag>()?;
    m.add_class::<emission_circuit::PyCollect>()?;
    m.add_class::<distributions::DistributionTable>()?;
    m.add_class::<parameters::ParameterTable>()?;
    m.add_class::<emission_circuit::PyEmit>()?;
    m.add_class::<sampling_graph::SamplingGraph>()?;
    m.add_wrapped(wrap_pyfunction!(passes::build::py_build))?;
    m.add_wrapped(wrap_pyfunction!(
        passes::absorb_dressing::py_absorb_dressing
    ))?;
    m.add_wrapped(wrap_pyfunction!(
        passes::merge_collectors::py_merge_collectors
    ))?;
    m.add_wrapped(wrap_pyfunction!(passes::lower::py_build_template))?;
    m.add_wrapped(wrap_pyfunction!(passes::lower::py_lower))?;
    m.add_wrapped(wrap_pyfunction!(passes::py_merge_parallel))?;
    m.add_wrapped(wrap_pyfunction!(passes::py_prune_from_sources))?;
    m.add_wrapped(wrap_pyfunction!(passes::py_prune_from_sinks))?;
    m.add_wrapped(wrap_pyfunction!(passes::py_set_virtual_types))?;
    Ok(())
}
