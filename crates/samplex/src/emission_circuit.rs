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

//! IR2 vocabulary: the `Emit` instruction and the `Collect` annotation.
//!
//! One `Emit` stands in for one emission. A `Twirl` produces two — the inverse pair, sharing a
//! [`DistKey`] and its draw slots, with opposite [`Direction`]s; `InjectNoise` and `ChangeBasis` /
//! `InjectLocalClifford` produce one each.
//!
//! An emission is Rust-native: [`Emit`] *is* the operation, implementing
//! [`CustomOperation`] so it lands in a circuit as a `PackedOperation` with no Python object at
//! rest, and Rust reads it back with `downcast_ref::<Emit>()`. The [`PyEmit`] pyclass is a
//! read-only view, built on demand by [`Emit::create_py_op`] whenever Python asks a circuit for
//! the operation — which is what keeps a lowered circuit inspectable and drawable.

use std::sync::Arc;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyString;
use qiskit_circuit::annotation::{Annotation, PyAnnotation};
use qiskit_circuit::operations::{CustomOperation, Operation, Param};
use smallvec::SmallVec;

use crate::annotated_circuit::{SynthesizerType, parse_decomposition};
use crate::distributions::{DistEntry, DistKey, DistributionTable};
use crate::partition::Partition;
use crate::sampling_graph::Direction;
use crate::virtual_type::VirtualType;

/// The instruction name reported to Qiskit for every emission, regardless of kind. Which kind an
/// emission is comes from the [`DistEntry`] its `dist` key points at; see [`PyEmit::source`].
pub const EMIT_NAME: &str = "emit";

/// Per-part descriptor for an emission, parallel with its partition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmitPart {
    /// The distribution this part draws from.
    pub dist: DistKey,
    /// Index into this part's `dist` key's sample array.
    pub draw: u32,
    /// Whether to take the adjoint of the sampled value before composing or propagating. True for
    /// the far half of a twirl pair, false everywhere else.
    pub adjoint: bool,
}

/// An emission: the operation itself, with [`PyEmit`] as Python's read-only view of one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Emit {
    /// Which way the emitted virtual state flows, or `None` if it has already resolved in place —
    /// owned directly by the collector body it sits in, rather than propagating towards one.
    pub direction: Option<Direction>,
    /// How the emission's qubits group into subsystems, by index into its own qargs.
    pub partition: Partition,
    /// Per-part descriptors, parallel with `partition.iter()`.
    pub parts: Vec<EmitPart>,
}

impl Emit {
    /// The distribution key of the first part. Convenience for the common uniform case where all
    /// parts share the same distribution.
    pub fn dist(&self) -> DistKey {
        self.parts[0].dist
    }

    /// The virtual type of the first part, resolved via `table`. Convenience for the common uniform
    /// case where all parts share the same virtual type.
    pub fn virtual_type(&self, table: &DistributionTable) -> VirtualType {
        table
            .get(self.dist())
            .expect("an Emit's dist key always resolves in the table it was built from")
            .virtual_type()
    }
}

impl Operation for Emit {
    fn name(&self) -> &str {
        EMIT_NAME
    }

    fn num_qubits(&self) -> u32 {
        self.partition.num_qubits() as u32
    }

    fn num_clbits(&self) -> u32 {
        0
    }

    fn num_params(&self) -> u32 {
        0
    }

    fn directive(&self) -> bool {
        false
    }
}

impl CustomOperation for Emit {
    // An emission is a marker for a later stage to consume, not a gate: it has no matrix and no
    // definition, so it cannot be decomposed or transpiled through.
    fn is_unitary(&self) -> bool {
        false
    }

    /// Hand Python a read-only [`PyEmit`] view of this emission.
    fn create_py_op(
        &self,
        py: Python,
        _params: Option<SmallVec<[Param; 3]>>,
        _label: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        ensure_registered(py)?;
        Ok(Py::new(py, PyEmit::new(self.clone()))?.into_any())
    }
}

/// A read-only view onto one [`Emit`] in a lowered circuit.
///
/// Never the storage — this is materialized on demand by [`Emit::create_py_op`], so there is no
/// way to build one from Python and append it. That is deliberate: a Python-constructed `Emit` would
/// land as a `PyInstruction`, which the `downcast_ref::<Emit>()` readers cannot see.
#[pyclass(
    name = "Emit",
    module = "qiskit._accelerate.samplex",
    frozen,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub struct PyEmit {
    pub(crate) inner: Emit,
}

impl PyEmit {
    /// Wrap a spec.
    pub fn new(inner: Emit) -> Self {
        PyEmit { inner }
    }

    /// The wrapped spec.
    pub fn spec(&self) -> &Emit {
        &self.inner
    }
}

#[pymethods]
impl PyEmit {
    // --- the `qiskit.circuit.Operation` interface ---

    #[getter]
    fn name(&self) -> &'static str {
        EMIT_NAME
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.partition.num_qubits()
    }

    #[getter]
    fn num_clbits(&self) -> usize {
        0
    }

    // --- payload readouts, for inspection from Python ---

    /// Which kind of annotation this emission stands in for — `"twirl"`, `"inject_noise"`, or
    /// `"change_basis"` — resolved from `table`'s entry for this emission's distribution key.
    /// `None` if no table is given, or if the key has no entry in it.
    #[pyo3(signature = (table=None))]
    fn source(&self, table: Option<&DistributionTable>) -> Option<&'static str> {
        let entry = table.and_then(|t| t.get(self.inner.dist()))?;
        Some(match entry {
            DistEntry::Distribution(_) => "twirl",
            DistEntry::Basis { .. } => "change_basis",
            DistEntry::Noise { .. } => "inject_noise",
        })
    }

    #[getter]
    fn distribution_key(&self) -> u32 {
        self.inner.dist().0
    }

    #[getter]
    fn direction(&self) -> &'static str {
        match self.inner.direction {
            Some(Direction::Left) => "left",
            Some(Direction::Right) => "right",
            None => "local",
        }
    }

    fn virtual_type(&self, table: &DistributionTable) -> &'static str {
        match self.inner.virtual_type(table) {
            VirtualType::Pauli => "pauli",
            VirtualType::C1 => "c1",
            VirtualType::U2 => "u2",
            VirtualType::Z2 => "z2",
        }
    }

    /// The subsystems this emission groups its qubits into, as indices into its own qargs.
    ///
    /// Read the qubits off the instruction — `circuit_instruction.qubits` — and index into them;
    /// there is deliberately no `qubits` readout here, since the operation is shared by every
    /// placement of it and so cannot know which wires it landed on.
    #[getter]
    fn subsystems(&self) -> Vec<Vec<usize>> {
        self.inner
            .partition
            .iter()
            .map(|part| part.to_vec())
            .collect()
    }

    /// The draw indices for each part, parallel with `subsystems`.
    #[getter]
    fn draws(&self) -> Vec<u32> {
        self.inner.parts.iter().map(|p| p.draw).collect()
    }

    /// Whether each part takes the adjoint of its sampled value, parallel with `subsystems`.
    #[getter]
    fn adjoints(&self) -> Vec<bool> {
        self.inner.parts.iter().map(|p| p.adjoint).collect()
    }

    fn __repr__(&self) -> String {
        let draws: Vec<u32> = self.inner.parts.iter().map(|p| p.draw).collect();
        let adjoint_marker = if self.inner.parts.iter().any(|p| p.adjoint) {
            ", adj"
        } else {
            ""
        };
        format!(
            "Emit(dist=#{}, {}, {}, draws={:?}{})",
            self.inner.dist().0,
            self.direction(),
            self.inner.partition,
            draws,
            adjoint_marker,
        )
    }

    fn __eq__(&self, other: &PyEmit) -> bool {
        self.inner == other.inner
    }
}

static REGISTERED: PyOnceLock<()> = PyOnceLock::new();

/// Register [`PyEmit`] as an `abc` virtual subclass of `qiskit.circuit.Operation`, once.
///
/// **Must not run while `qiskit._accelerate` is still initialising**, since importing
/// `qiskit.circuit` that early fails. [`Emit::create_py_op`] is the only caller, which keeps it
/// safe by construction: a view is only ever built when Python asks a circuit for an operation, long
/// after import.
pub fn ensure_registered(py: Python) -> PyResult<()> {
    REGISTERED.get_or_try_init::<_, PyErr>(py, || {
        qiskit_circuit::imports::OPERATION
            .get_bound(py)
            .call_method1("register", (py.get_type::<PyEmit>(),))?;
        Ok(())
    })?;
    Ok(())
}

// --- Collect ------------------------------------------------------------------------------------
//
// `Collect` is deliberately kept out of the IR1 vocabulary: those five annotations are what a user
// writes, while this one is written by the build pass. Keeping them apart is what makes a lowered
// circuit distinguishable from an annotated one, and it is why `Collect` declares a child namespace
// rather than the flat `samplex` the input annotations share.

/// Per-part descriptor for a collector, parallel with its partition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectPart {
    /// How the collected virtual gates on this part will be synthesized.
    pub synthesizer: SynthesizerType,
}

/// A collect annotation: marks a box whose body holds what a dressing absorbed, to be replaced by
/// a synthesizer template during lowering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Collect {
    /// How the collector's qubits group into subsystems, by index into the box's own qargs.
    pub partition: Partition,
    /// Per-part descriptors, parallel with `partition.iter()`.
    pub parts: Vec<CollectPart>,
}

impl Collect {
    /// The synthesizer of the first part. Convenience for the common uniform case where all parts
    /// share the same synthesizer.
    pub fn synthesizer(&self) -> SynthesizerType {
        self.parts[0].synthesizer
    }

    /// Whether the given virtual type is accepted by all parts of this collector.
    pub fn accepts(&self, vt: VirtualType) -> bool {
        self.parts.iter().all(|part| part.synthesizer.accepts(vt))
    }
}

/// The namespace a collector declares. A child of the IR1 namespace, so a single `samplex` handler
/// still catches it by the dispatch chain's parent fallback, while a handler that only cares about
/// what samplex *emitted* can name it exactly.
const NAMESPACE: &str = "samplex.collect";

impl Annotation for Collect {
    fn namespace(&self) -> &str {
        NAMESPACE
    }

    fn create_py_annotation(&self, py: Python) -> PyResult<Py<PyAny>> {
        Ok(Py::new(py, PyCollect::init(self.clone()))?.into_any())
    }
}

/// Python's view of a [`Collect`].
#[pyclass(name = "Collect", module = "qiskit._accelerate.samplex", frozen, extends = PyAnnotation)]
pub struct PyCollect {
    inner: Arc<Collect>,
}

impl PyCollect {
    /// Build the initializer, base and subclass sharing one allocation.
    ///
    /// The base *must* carry the native value: without it a Python round trip comes back as an
    /// opaque `PythonAnnotation`, `emission_circuit_navigation::collect_annotation` stops seeing the
    /// box as a collector, and the pass walks quietly treat it as ordinary content. This is the same
    /// hazard the `Emit` note above records, and it fails silently in exactly the same way.
    fn init(spec: Collect) -> PyClassInitializer<Self> {
        let inner = Arc::new(spec);
        PyClassInitializer::from(PyAnnotation::new(inner.clone())).add_subclass(PyCollect { inner })
    }
}

#[pymethods]
impl PyCollect {
    /// Construct a `Collect` annotation covering no qubits.
    ///
    /// The partition is empty because a bare annotation has no box to take its width from yet, while
    /// the one part is what `synthesizer` reads. Build writes an empty body too; `absorb_dressing` is
    /// what fills one in.
    #[new]
    #[pyo3(signature = (synthesizer="rzsx"))]
    fn new(synthesizer: &str) -> PyResult<PyClassInitializer<Self>> {
        let synth = parse_decomposition(synthesizer)?;
        Ok(PyCollect::init(Collect {
            partition: Partition::singletons(0),
            parts: vec![CollectPart { synthesizer: synth }],
        }))
    }

    #[classattr]
    fn namespace(py: Python) -> Py<PyString> {
        intern!(py, NAMESPACE).clone().unbind()
    }

    #[getter]
    fn synthesizer(&self) -> &'static str {
        match self.inner.synthesizer() {
            SynthesizerType::RzSx => "rzsx",
            SynthesizerType::RzRx => "rzrx",
        }
    }

    fn __repr__(&self) -> String {
        format!("Collect({:?})", self.inner.synthesizer())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qiskit_circuit::annotation::extract_annotation;

    fn spec() -> Collect {
        Collect {
            partition: Partition::singletons(2),
            parts: vec![
                CollectPart {
                    synthesizer: SynthesizerType::RzSx,
                },
                CollectPart {
                    synthesizer: SynthesizerType::RzSx,
                },
            ],
        }
    }

    #[test]
    fn test_collect_round_trips_through_python() {
        // The collector is the one annotation samplex both writes and reads back, so this round trip
        // is the one that keeps the IR2 passes able to recognise their own output.
        let original = spec();
        Python::initialize();
        Python::attach(|py| {
            let object = original.create_py_annotation(py).unwrap();
            let recovered = extract_annotation(object.bind(py));
            assert_eq!(recovered.downcast_ref::<Collect>(), Some(&original));
        });
    }

    #[test]
    fn test_python_constructed_collect_is_native() {
        // `Collect(...)` from Python goes through `#[new]`, a different path to `create_py_annotation`.
        // If that path left the base empty, a user-written collector would be invisible to every Rust
        // reader while looking perfectly correct from Python.
        Python::initialize();
        Python::attach(|py| {
            let object = Py::new(py, PyCollect::new("rzrx").unwrap()).unwrap();
            let recovered = extract_annotation(object.bind(py).as_any());
            let spec = recovered
                .downcast_ref::<Collect>()
                .expect("a Python-constructed collector must still be a native one");
            assert_eq!(spec.synthesizer(), SynthesizerType::RzRx);
        });
    }

    #[test]
    fn test_collect_declares_a_child_namespace() {
        // Samplex's *output* vocabulary, so a namespace of its own — but a child of the input one, so a
        // single `samplex` handler still catches it by parent fallback.
        assert_eq!(spec().namespace(), "samplex.collect");
        assert!(
            spec()
                .namespace()
                .starts_with(crate::annotated_circuit::NAMESPACE)
        );
        Python::initialize();
        Python::attach(|py| {
            assert_eq!(
                PyCollect::namespace(py).extract::<String>(py).unwrap(),
                NAMESPACE
            );
        });
    }
}
