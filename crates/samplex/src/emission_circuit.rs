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
//! An emission is Rust-native: [`EmitSpec`] *is* the operation, implementing
//! [`CustomOperation`] so it lands in a circuit as a `PackedOperation` with no Python object at
//! rest, and Rust reads it back with `downcast_ref::<EmitSpec>()`. The [`Emit`] pyclass is a
//! read-only view, built on demand by [`EmitSpec::create_py_op`] whenever Python asks a circuit for
//! the operation — which is what keeps a lowered circuit inspectable and drawable.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyString;
use qiskit_circuit::annotation::PyAnnotation;
use qiskit_circuit::operations::{CustomOperation, Operation, Param};
use smallvec::SmallVec;

use crate::annotated_circuit::{SynthesizerType, parse_decomposition};
use crate::distributions::{DistEntry, DistKey, DistributionTable};
use crate::partition::Partition;
use crate::sampling_graph::Direction;
use crate::virtual_type::VirtualType;

/// The instruction name reported to Qiskit for every emission, regardless of kind. Which kind an
/// emission is comes from the [`DistEntry`] its `dist` key points at; see [`Emit::source`].
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

/// The payload of an [`Emit`] instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmitSpec {
    /// Which way the emitted virtual state flows, or `None` if it has already resolved in place —
    /// owned directly by the collector body it sits in, rather than propagating towards one.
    pub direction: Option<Direction>,
    /// How the emission's qubits group into subsystems, by index into its own qargs.
    pub partition: Partition,
    /// Per-part descriptors, parallel with `partition.iter()`.
    pub parts: Vec<EmitPart>,
}

impl EmitSpec {
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
            .expect("an EmitSpec's dist key always resolves in the table it was built from")
            .virtual_type()
    }
}

impl Operation for EmitSpec {
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

impl CustomOperation for EmitSpec {
    // An emission is a marker for a later stage to consume, not a gate: it has no matrix and no
    // definition, so it cannot be decomposed or transpiled through.
    fn is_unitary(&self) -> bool {
        false
    }

    /// Hand Python a read-only [`Emit`] view of this emission.
    fn create_py_op(
        &self,
        py: Python,
        _params: Option<SmallVec<[Param; 3]>>,
        _label: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        ensure_registered(py)?;
        Ok(Py::new(py, Emit::new(self.clone()))?.into_any())
    }
}

/// A read-only view onto one [`EmitSpec`] in a lowered circuit.
///
/// Never the storage — this is materialized on demand by [`EmitSpec::create_py_op`], so there is no
/// way to build one from Python and append it. That is deliberate: a Python-constructed `Emit` would
/// land as a `PyInstruction`, which the `downcast_ref::<EmitSpec>()` readers cannot see.
#[pyclass(module = "qiskit._accelerate.samplex", frozen, skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct Emit {
    pub(crate) inner: EmitSpec,
}

impl Emit {
    /// Wrap a spec.
    pub fn new(inner: EmitSpec) -> Self {
        Emit { inner }
    }

    /// The wrapped spec.
    pub fn spec(&self) -> &EmitSpec {
        &self.inner
    }
}

#[pymethods]
impl Emit {
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

    fn __eq__(&self, other: &Emit) -> bool {
        self.inner == other.inner
    }
}

static REGISTERED: PyOnceLock<()> = PyOnceLock::new();

/// Register [`Emit`] as an `abc` virtual subclass of `qiskit.circuit.Operation`, once.
///
/// **Must not run while `qiskit._accelerate` is still initialising**, since importing
/// `qiskit.circuit` that early fails. [`EmitSpec::create_py_op`] is the only caller, which keeps it
/// safe by construction: a view is only ever built when Python asks a circuit for an operation, long
/// after import.
pub fn ensure_registered(py: Python) -> PyResult<()> {
    REGISTERED.get_or_try_init::<_, PyErr>(py, || {
        qiskit_circuit::imports::OPERATION
            .get_bound(py)
            .call_method1("register", (py.get_type::<Emit>(),))?;
        Ok(())
    })?;
    Ok(())
}

// --- Collect ------------------------------------------------------------------------------------
//
// `Collect` is deliberately not a `BoxAnnotation` variant: that enum is the *input* vocabulary,
// while this is written by the build pass. Keeping them apart is what makes a lowered circuit
// distinguishable from an annotated one.

/// Per-part descriptor for a collector, parallel with its partition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectPart {
    /// How the collected virtual gates on this part will be synthesized.
    pub synthesizer: SynthesizerType,
}

/// The payload of a [`Collect`] annotation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectSpec {
    /// How the collector's qubits group into subsystems, by index into the box's own qargs.
    pub partition: Partition,
    /// Per-part descriptors, parallel with `partition.iter()`.
    pub parts: Vec<CollectPart>,
}

impl CollectSpec {
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

/// Marks a box whose body holds what a dressing absorbed, to be replaced by a synthesizer template
/// during lowering.
#[pyclass(module = "qiskit._accelerate.samplex", frozen, extends = PyAnnotation)]
pub struct Collect {
    pub(crate) inner: CollectSpec,
}

impl Collect {
    /// Wrap a spec.
    pub fn new_from_spec(inner: CollectSpec) -> Self {
        Collect { inner }
    }
}

#[pymethods]
impl Collect {
    /// Construct a `Collect` annotation covering no qubits.
    ///
    /// The partition is empty because a bare annotation has no box to take its width from yet, while
    /// the one part is what `synthesizer` reads. Build writes an empty body too; `absorb_dressing` is
    /// what fills one in.
    #[new]
    #[pyo3(signature = (synthesizer="rzsx"))]
    fn new(synthesizer: &str) -> PyResult<PyClassInitializer<Self>> {
        let synth = parse_decomposition(synthesizer)?;
        Ok(
            PyClassInitializer::from(PyAnnotation).add_subclass(Collect {
                inner: CollectSpec {
                    partition: Partition::singletons(0),
                    parts: vec![CollectPart { synthesizer: synth }],
                },
            }),
        )
    }

    #[classattr]
    fn namespace(py: Python) -> Py<PyString> {
        intern!(py, "samplex").clone().unbind()
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

/// Try to extract a [`CollectSpec`] from a Python annotation object.
pub fn extract_collect(obj: &Bound<'_, PyAny>) -> Option<CollectSpec> {
    obj.cast::<Collect>().ok().map(|c| c.get().inner.clone())
}
