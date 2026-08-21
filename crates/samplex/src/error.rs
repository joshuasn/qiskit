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

//! What samplex refuses to do, and the one seam where a refusal becomes a Python exception.
//!
//! Samplex is written as plain Rust over plain data: the IR2 walks and every pass but the two that
//! materialize Python objects hold no `Python` token, so they could not build a `PyErr` even to report
//! a failure. [`SamplexError`] is what they return instead, and `From<SamplexError> for PyErr` is
//! where one becomes the `PyValueError` a `#[pyfunction]` hands back.
//!
//! One enum covers the whole crate rather than one per module, because there is exactly one consumer:
//! nothing branches on a variant for control flow, and the only variant inspection anywhere is a
//! handful of tests asserting which failure they got. A per-module partition would be a seam with no
//! second adapter behind it to justify it.
//!
//! Every variant carries the values its message interpolates rather than a pre-formatted `String`, so
//! a test can name the failure it expects instead of matching prose. The foreign errors samplex calls
//! into — `DAGCircuit`'s and `CircuitData`'s — arrive through `#[from]` variants, so `?` works
//! directly on those calls.

use num_complex::Complex64;
use pyo3::PyResult;
use pyo3::exceptions::PyValueError;
use thiserror::Error;

use qiskit_circuit::circuit_data::CircuitDataError;
use qiskit_circuit::dag_circuit::{DAGError, DuplicateWireError};
use qiskit_circuit::operations::{Operation, StandardGate};
use qiskit_circuit::parameter::parameter_expression::ParameterError;

use crate::annotated_circuit::AnnotationKind;
use crate::distributions::DistKey;
use crate::emission_circuit_navigation::Site;
use crate::sampling_graph::Direction;
use crate::virtual_type::VirtualType;

/// Extension trait that converts any `Result<T, E: Display>` into `PyResult<T>` via `PyValueError`.
///
/// The adapter for a *foreign* error reaching a function that does hold a `Python` token. Samplex's
/// own failures are [`SamplexError`], which becomes a `PyErr` through its `From` impl instead, and the
/// foreign errors samplex actually meets now arrive through that enum's `#[from]` variants — so this
/// has no callers left in the crate, and is kept for the next `Display` error that has no variant.
pub trait IntoPyResult<T> {
    fn into_py_result(self) -> PyResult<T>;
}

impl<T, E: std::fmt::Display> IntoPyResult<T> for std::result::Result<T, E> {
    fn into_py_result(self) -> PyResult<T> {
        self.map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// Anything samplex refuses to do, and what about the input or the IR made it refuse.
#[derive(Debug, Error)]
pub enum SamplexError {
    // --- Annotations that do not resolve into one box's declaration (IR1) ------------------------
    #[error("Duplicate annotation type {0:?} on a single box.")]
    DuplicateAnnotation(AnnotationKind),
    #[error("InjectNoise requires a Twirl on the same box.")]
    InjectNoiseWithoutTwirl,
    #[error("InjectLocalClifford requires a Twirl on the same box.")]
    InjectLocalCliffordWithoutTwirl,
    #[error("ChangeBasis and InjectLocalClifford are mutually exclusive on the same box.")]
    ChangeBasisConflict,

    // --- Malformed IR2 structure, met while navigating it ----------------------------------------
    /// A `box` carrying no body, or more than one, where a walk needs the single body it declares.
    #[error("a box instruction should have exactly one body")]
    BoxWithoutOneBody,
    /// The same, met while descending a scope path in order to write at the end of it.
    #[error("a scope on the path should have exactly one body")]
    ScopeWithoutOneBody,
    /// The same, met while descending a scope path in order to read.
    #[error("a scope on the path has no body")]
    ScopeWithoutBody,
    #[error("cannot descend into a box with no body")]
    DescentIntoBodylessBox,
    /// A site asked to report its wires in the frame of a scope nested below the site itself, which no
    /// remapping could answer.
    #[error("a site cannot be lifted into a scope deeper than itself")]
    LiftIntoDeeperScope,
    /// A body holds content on a wire its box does not cover, so lifting that content out has nowhere
    /// to put it.
    #[error("content sits on a wire outside its box")]
    ContentOffItsBoxWires,

    // --- Input samplex cannot build an emission circuit from -------------------------------------
    #[error("Unsupported control flow in a samplex circuit: '{0}'. Only `box` is supported.")]
    UnsupportedControlFlow(String),
    /// The same refusal, met inside a box body rather than at the top level.
    #[error("Unsupported control flow in a samplex circuit: '{0}'.")]
    UnsupportedControlFlowInBox(String),
    #[error("box instruction is missing its body")]
    BoxMissingBody,
    #[error("qubit {0} out of scope")]
    QubitOutOfScope(usize),
    #[error("clbit {0} out of scope")]
    ClbitOutOfScope(usize),
    /// An emission covers its box's full width, so writing one on a different number of wires would
    /// leave its partition indexing wires that are not there.
    #[error("an emission on {qubits} qubits cannot be written on {wires} of them")]
    EmissionWidthMismatch { qubits: usize, wires: usize },

    // --- A propagation with no rule to apply -----------------------------------------------------
    /// Conjugating this virtual type by this gate leaves its group, so there is no closed form for a
    /// collector to undo it with.
    #[error(
        "cannot propagate a {} virtual gate through '{}': no propagation rule exists for that \
         combination, so the randomization could not be undone. Only Cliffords (and RZZ) admit \
         Pauli and local-C1 propagation; a local U2 element admits single-qubit gates only.",
        .virtual_type.name(),
        .gate.name(),
    )]
    NoPropagationRule {
        virtual_type: VirtualType,
        gate: StandardGate,
    },

    // --- Lowering IR2 into a template and a sampling graph ---------------------------------------
    #[error("cannot lower '{0}' into a template: it carries a body but is not a `box`")]
    BodyOnNonBox(String),
    /// A randomization with nothing ahead of it able to undo it.
    #[error(
        "emission on qubits {qubits:?} travelling {direction:?} has no compatible collector ahead of \
         it; its randomization could not be undone"
    )]
    EmissionWithoutCollector {
        qubits: Vec<usize>,
        direction: Direction,
    },
    /// The template reported one collect box twice, so the join onto it has no defined answer.
    #[error("the template reported two parameter ranges for the collector at {0:?}")]
    DuplicateCollectorParams(Site),
    /// A collector the graph walk saw and the template did not.
    #[error(
        "the graph walk found a collector at {0:?} that the template did not; the two must be built \
         from the same circuit"
    )]
    CollectorNotInTemplate(Site),
    /// The other direction: angles in the template with nothing in the graph computing them.
    #[error(
        "the template minted parameters for {count} collector(s) the graph walk did not find, one at \
         {site:?}; the two must be built from the same circuit"
    )]
    CollectorsNotInGraph { count: usize, site: Site },
    #[error("cannot absorb a gate whose angle evaluates to the complex value {0}")]
    ComplexAbsorbedAngle(Complex64),
    #[error(
        "cannot absorb a gate whose parameter is an opaque Python object: the sampling graph is read \
         without the GIL, so it cannot carry one"
    )]
    OpaqueAbsorbedParameter,
    #[error("emission (dist={}) references a missing table entry", .dist.0)]
    MissingTableEntry { dist: DistKey },

    // --- Foreign errors, so that `?` works on the calls that raise them --------------------------
    #[error(transparent)]
    Dag(#[from] DAGError),
    #[error(transparent)]
    DuplicateWire(#[from] DuplicateWireError),
    #[error(transparent)]
    Circuit(#[from] CircuitDataError),
    #[error(transparent)]
    Parameter(#[from] ParameterError),
}

/// The crate's one adapter from a samplex failure to a Python one.
///
/// Always a `ValueError`, because every variant says the same kind of thing: the circuit handed in, or
/// the IR built from it, is not something samplex can randomize. The message is the variant's
/// `Display`, so a rule's wording lives with the rule rather than at this seam.
impl From<SamplexError> for pyo3::PyErr {
    fn from(error: SamplexError) -> Self {
        PyValueError::new_err(error.to_string())
    }
}

/// The result of anything that can fail the way samplex fails.
pub type Result<T> = std::result::Result<T, SamplexError>;
