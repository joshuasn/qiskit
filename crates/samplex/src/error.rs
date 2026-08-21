// This code is a Qiskit project.
//
// (C) Copyright IBM 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Errors raised while lowering an annotated circuit.

use pyo3::PyResult;
use pyo3::exceptions::PyValueError;
use thiserror::Error;

use crate::annotated_circuit::AnnotationKind;

/// Extension trait that converts any `Result<T, E: Display>` into `PyResult<T>` via `PyValueError`.
///
/// The crate's one adapter between a Rust error and a Python one. Samplex's own rules are written as
/// pure functions returning plain `Result`s, and the `DAGCircuit` methods they call have errors of
/// their own; this is where both become the `PyResult` a pass hands back.
pub trait IntoPyResult<T> {
    fn into_py_result(self) -> PyResult<T>;
}

impl<T, E: std::fmt::Display> IntoPyResult<T> for Result<T, E> {
    fn into_py_result(self) -> PyResult<T> {
        self.map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LowerError {
    #[error("Duplicate annotation type {0:?} on a single box.")]
    DuplicateAnnotation(AnnotationKind),
    #[error("InjectNoise requires a Twirl on the same box.")]
    InjectNoiseWithoutTwirl,
    #[error("InjectLocalClifford requires a Twirl on the same box.")]
    InjectLocalCliffordWithoutTwirl,
    #[error("ChangeBasis and InjectLocalClifford are mutually exclusive on the same box.")]
    ChangeBasisConflict,
}
