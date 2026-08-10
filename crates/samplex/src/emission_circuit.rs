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

//! The `Emit` instruction: a stand-in, in a lowered circuit, for a source of virtual gates.
//!
//! One `Emit` stands in for one emission. A `Twirl` produces *two* of them — the inverse pair — which
//! share a single [`DistKey`] and carry opposite [`Direction`]s; the inversion is implied by the
//! direction rather than being recorded separately. `InjectNoise` and `ChangeBasis` /
//! `InjectLocalClifford` each produce exactly one.
//!
//! `Emit` is a plain `#[pyclass]` registered as an `abc` virtual subclass of
//! `qiskit.circuit.Operation` (see [`register_operation`]). That is enough for Qiskit to accept it
//! into a circuit: `PyOpKind::from_type` classifies via `issubclass`, which honours virtual
//! registration, and `OperationFromPython` then reads only `name`, `num_qubits` and `num_clbits`
//! (plus an optional `params`). It therefore lands in a `CircuitData` as a `PyInstruction`, which
//! means Python can see and draw it, while Rust reads the payload back with a typed `cast::<Emit>()`
//! rather than attribute lookups.

use pyo3::IntoPyObjectExt;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyString;
use qiskit_circuit::annotation::PyAnnotation;

use crate::annotated_circuit::{SynthesizerType, parse_decomposition};
use crate::distributions::DistKey;
use crate::partition::Partition;
use crate::virtual_flow_graph::Direction;
use crate::virtual_type::VirtualType;

/// Which kind of annotation an [`Emit`] stands in for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmitSource {
    /// One half of a twirl's inverse pair.
    Twirl,
    /// Noise drawn from a referenced Pauli-Lindblad map.
    InjectNoise,
    /// A deterministic frame change (`ChangeBasis` or `InjectLocalClifford`).
    ChangeBasis,
}

impl EmitSource {
    /// The instruction name reported to Qiskit for this kind of emission.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Twirl => "samplex_emit_twirl",
            Self::InjectNoise => "samplex_emit_noise",
            Self::ChangeBasis => "samplex_emit_basis",
        }
    }
}

parse_enum!(parse_source, EmitSource, "emit source", {
    "twirl" => Twirl,
    "inject_noise" => InjectNoise,
    "change_basis" => ChangeBasis,
});

parse_enum!(parse_direction, Direction, "direction", {
    "left" => Left,
    "right" => Right,
});

/// Parse a direction that may also be `"local"`, meaning the emission resolves in place rather
/// than propagating.
fn parse_direction_opt(s: &str) -> PyResult<Option<Direction>> {
    match s {
        "local" => Ok(None),
        _ => parse_direction(s).map(Some),
    }
}

parse_enum!(parse_virtual_type, VirtualType, "virtual type", {
    "pauli" => Pauli,
    "c1" => C1,
    "u2" => U2,
    "z2" => Z2,
});

/// Per-part descriptor for an emission: what distribution is drawn on this subsystem, what
/// algebraic type results, and which slot in the distribution's sample array it reads from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmitPart {
    /// The distribution this part draws from.
    pub dist: DistKey,
    /// The algebraic type of the emitted virtual gates on this part.
    pub virtual_type: VirtualType,
    /// Index into this part's `dist` key's sample array.
    pub draw: u32,
    /// Whether to take the adjoint of the sampled value before composing/propagating.
    ///
    /// For a twirl pair the near half is `false` (uses the drawn element directly) while the far
    /// half is `true` (starts from the adjoint, then propagation conjugates it through gates).
    /// Non-twirl emissions are always `false`.
    pub adjoint: bool,
}

/// The payload of an [`Emit`] instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmitSpec {
    /// Which annotated box this emission came from.
    ///
    /// The collector that consumes it is one of *that box's* collectors, and no other. Adjacency
    /// cannot express this: an emission propagating through a nested box passes a collector that
    /// faces it, and without an owner to check against, that collector absorbs it as a local value
    /// and the conjugation it was owed silently never happens.
    ///
    /// This is **not** the emission id that earlier designs used and dropped. That was a handle on a
    /// graph node, invalidated by `merge_parallel_nodes` and `prune`; this names a *box*, is a
    /// build-time constant, and nothing downstream can invalidate it. See
    /// [`CollectSpec::owned`](CollectSpec::owned).
    pub box_id: u32,
    /// Which annotation this emission stands in for.
    pub source: EmitSource,
    /// Which way the emitted virtual state flows, or `None` if it has already resolved in place —
    /// owned directly by the collector body it sits in, rather than propagating towards one.
    pub direction: Option<Direction>,
    /// Subsystem grouping over the emission's qubits, in the *global* circuit frame.
    ///
    /// The instruction's qargs are body-local (a box body is its own circuit indexed `0..width`),
    /// so this is the one place the global frame is recorded; the graph reader relies on it rather
    /// than re-deriving it through enclosing box qargs.
    pub partition: Partition,
    /// Per-part descriptors, parallel with `partition.iter()`.
    pub parts: Vec<EmitPart>,
}

impl EmitSpec {
    /// The qubits this emission acts on, in the global frame, in ascending order.
    pub fn qubits(&self) -> Vec<usize> {
        let mut qubits: Vec<usize> = self.partition.all_elements().iter().copied().collect();
        qubits.sort_unstable();
        qubits
    }

    /// The distribution key of the first part. Convenience for the common uniform case where all
    /// parts share the same distribution.
    pub fn dist(&self) -> DistKey {
        self.parts[0].dist
    }

    /// The virtual type of the first part. Convenience for the common uniform case where all parts
    /// share the same virtual type.
    pub fn virtual_type(&self) -> VirtualType {
        self.parts[0].virtual_type
    }
}

/// A stand-in instruction for a source of virtual gates in a lowered circuit.
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
    /// Construct an `Emit` directly.
    ///
    /// The lowering builds these in Rust; this constructor exists so the instruction can be
    /// exercised from Python and from tests without running a full lowering.
    #[new]
    #[pyo3(signature = (subsystems, source="twirl", distribution_key=0, direction="left", virtual_type="pauli", draw_start=0, adjoint=false, box_id=0))]
    #[allow(clippy::too_many_arguments)]
    fn py_new(
        py: Python,
        subsystems: Vec<Vec<usize>>,
        source: &str,
        distribution_key: u32,
        direction: &str,
        virtual_type: &str,
        draw_start: u32,
        adjoint: bool,
        box_id: u32,
    ) -> PyResult<Self> {
        ensure_registered(py)?;
        if subsystems.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Emit requires at least one subsystem.",
            ));
        }
        let num_parts = subsystems.len();
        let partition =
            Partition::with_parts(subsystems.into_iter().map(|part| part.into_boxed_slice()))
                .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
        let dist = DistKey(distribution_key);
        let vt = parse_virtual_type(virtual_type)?;
        let parts = (0..num_parts)
            .map(|i| EmitPart {
                dist,
                virtual_type: vt,
                draw: draw_start + i as u32,
                adjoint,
            })
            .collect();
        Ok(Emit {
            inner: EmitSpec {
                box_id,
                source: parse_source(source)?,
                direction: parse_direction_opt(direction)?,
                partition,
                parts,
            },
        })
    }

    // --- the `qiskit.circuit.Operation` interface ---

    #[getter]
    fn name(&self) -> &'static str {
        self.inner.source.name()
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.partition.all_elements().len()
    }

    #[getter]
    fn num_clbits(&self) -> usize {
        0
    }

    // --- payload readouts, for inspection from Python ---

    #[getter]
    fn source(&self) -> &'static str {
        match self.inner.source {
            EmitSource::Twirl => "twirl",
            EmitSource::InjectNoise => "inject_noise",
            EmitSource::ChangeBasis => "change_basis",
        }
    }

    #[getter]
    fn distribution_key(&self) -> u32 {
        self.inner.dist().0
    }

    /// The annotated box this emission came from; only that box's collectors may consume it.
    #[getter]
    fn box_id(&self) -> u32 {
        self.inner.box_id
    }

    #[getter]
    fn direction(&self) -> &'static str {
        match self.inner.direction {
            Some(Direction::Left) => "left",
            Some(Direction::Right) => "right",
            None => "local",
        }
    }

    #[getter]
    fn virtual_type(&self) -> &'static str {
        match self.inner.virtual_type() {
            VirtualType::Pauli => "pauli",
            VirtualType::C1 => "c1",
            VirtualType::U2 => "u2",
            VirtualType::Z2 => "z2",
        }
    }

    /// The qubits this emission acts on, in the global circuit frame.
    #[getter]
    fn qubits(&self) -> Vec<usize> {
        self.inner.qubits()
    }

    /// The subsystems this emission acts on, in the global circuit frame.
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
            "Emit({}, dist={}, {}, {:?}, draws={:?}{})",
            self.source(),
            self.inner.dist().0,
            self.direction(),
            self.inner.qubits(),
            draws,
            adjoint_marker,
        )
    }

    fn __eq__(&self, other: &Emit) -> bool {
        self.inner == other.inner
    }

    // `circuit_to_dag(copy_operations=True)` deep-copies every operation, so an `Emit` that cannot
    // be copied cannot survive a DAG round-trip. The instruction is immutable, so both copies are
    // shallow; `__reduce__` additionally makes it picklable.
    fn __copy__(&self) -> Emit {
        self.clone()
    }

    /// Qiskit's operation-copying protocol (`PythonOperation::py_copy` calls this by name), used by
    /// the control-flow builder when it captures a box body.
    #[pyo3(signature = (name=None))]
    fn copy(&self, name: Option<String>) -> PyResult<Emit> {
        if name.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Emit instructions cannot be renamed; their name is derived from the emission kind.",
            ));
        }
        Ok(self.clone())
    }

    #[pyo3(signature = (_memo=None))]
    fn __deepcopy__(&self, _memo: Option<Bound<'_, PyAny>>) -> Emit {
        self.clone()
    }

    fn __reduce__(&self, py: Python) -> PyResult<Py<PyAny>> {
        (
            py.get_type::<Emit>(),
            (
                self.subsystems(),
                self.source(),
                self.inner.dist().0,
                self.direction(),
                self.virtual_type(),
                self.inner.parts[0].draw,
                self.inner.parts[0].adjoint,
                self.inner.box_id,
            ),
        )
            .into_py_any(py)
    }
}

static REGISTERED: PyOnceLock<()> = PyOnceLock::new();

/// Register [`Emit`] as an `abc` virtual subclass of `qiskit.circuit.Operation`, once.
///
/// Without this, `QuantumCircuit.append` rejects the instruction: Qiskit classifies operations with
/// `issubclass` against `Gate` / `Instruction` / `Operation`, and `Emit` inherits from none of them.
///
/// This must not run while the `qiskit._accelerate` module is still initialising — importing
/// `qiskit.circuit` that early fails, because `qiskit/__init__.py` has not yet installed the
/// `sys.modules["qiskit._accelerate.*"]` aliases that `qiskit.circuit` itself imports from. So it is
/// deferred to the first point where an `Emit` actually comes into existence.
pub fn ensure_registered(py: Python) -> PyResult<()> {
    REGISTERED.get_or_try_init::<_, PyErr>(py, || {
        qiskit_circuit::imports::OPERATION
            .get_bound(py)
            .call_method1("register", (py.get_type::<Emit>(),))?;
        Ok(())
    })?;
    Ok(())
}

// --- Lowering output vocabulary -----------------------------------------------------------------
//
// `Collect` is deliberately *not* a `BoxAnnotation` variant. `BoxAnnotation` is the input vocabulary
// — what a user writes on a circuit before lowering — whereas `Collect` is written *by* the lowering
// onto the boxes it creates. Keeping them apart means `extract_annotation` never has to consider a
// lowering artefact, and a lowered circuit is distinguishable from an annotated one.

/// Try to read an [`EmitSpec`] back out of a Python object.
pub fn extract_emit(obj: &Bound<'_, PyAny>) -> Option<EmitSpec> {
    obj.cast::<Emit>().ok().map(|e| e.get().inner.clone())
}

/// An emission owned directly by its collector — adjacent to it, never propagating through gates.
///
/// At sampling time the collector reads the sampled value from the distribution table and composes
/// it at the position its local `Emit` instruction (`direction: None`) sits at in the collector body.
/// No VFG `Emission` node — the local emission is folded straight into the collector's own step
/// list.
///
/// Neither direction nor source is stored: position in the collector body IS the composition order,
/// and the distribution table entry (reachable through [`EmitPart::dist`]) already encodes the
/// emission kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalEmission {
    pub partition: Partition,
    /// Per-part descriptors, parallel with `partition.iter()`.
    pub parts: Vec<EmitPart>,
}

/// Per-part descriptor for a collector: what synthesizer to use on this subsystem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectPart {
    /// How the collected virtual gates on this part will be synthesized.
    pub synthesizer: SynthesizerType,
}

/// The payload of a [`Collect`] annotation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectSpec {
    /// The annotated boxes whose emissions this collector may consume, ascending.
    ///
    /// Build gives each of a box's two collectors that box's own id. Merging unions them, so a shared
    /// middle collector owns every box that contributed to it — which is exactly the set of emissions
    /// it is allowed to take.
    ///
    /// A collector may own several boxes, and (with control flow) a box's id may appear on several
    /// collectors, provided they lie on mutually exclusive paths. See the control-flow section of
    /// `SAMPLEX_IR_DESIGN.md`.
    pub owned: Vec<u32>,
    /// Subsystem grouping over the collector's qubits, in the *global* circuit frame.
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

    /// Whether this collector may consume emissions from the box with this id.
    pub fn owns(&self, box_id: u32) -> bool {
        self.owned.contains(&box_id)
    }

    /// Take on another collector's ownership, keeping the set sorted and duplicate-free.
    ///
    /// Sorted so that a merged collector's set does not depend on the order the merge happened to
    /// visit its members in; two runs must produce identical IR2.
    pub fn absorb_ownership(&mut self, other: &[u32]) {
        self.owned.extend_from_slice(other);
        self.owned.sort_unstable();
        self.owned.dedup();
    }
}

/// Marks a box whose body holds the "easy" gates absorbed into a dressing, to be replaced by a
/// synthesizer template in a later stage.
#[pyclass(module = "qiskit._accelerate.samplex", frozen, extends = PyAnnotation)]
pub struct Collect {
    pub(crate) inner: CollectSpec,
}

impl Collect {
    /// Wrap a spec. The `#[new]` constructor below is the Python-facing equivalent.
    pub fn new_from_spec(inner: CollectSpec) -> Self {
        Collect { inner }
    }
}

#[pymethods]
impl Collect {
    /// Construct a `Collect` annotation.
    ///
    /// From Python this produces an empty collector, with no body yet. The build pass populates
    /// the body directly in Rust.
    #[new]
    #[pyo3(signature = (synthesizer="rzsx"))]
    fn new(synthesizer: &str) -> PyResult<PyClassInitializer<Self>> {
        let synth = parse_decomposition(synthesizer)?;
        Ok(
            PyClassInitializer::from(PyAnnotation).add_subclass(Collect {
                inner: CollectSpec {
                    owned: Vec::new(),
                    partition: Partition::new(),
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

    /// The annotated boxes whose emissions this collector may consume, ascending.
    #[getter]
    fn owned(&self) -> Vec<u32> {
        self.inner.owned.clone()
    }

    fn __repr__(&self) -> String {
        format!("Collect({:?})", self.inner.synthesizer())
    }
}

/// Try to extract a [`CollectSpec`] from a Python annotation object.
pub fn extract_collect(obj: &Bound<'_, PyAny>) -> Option<CollectSpec> {
    obj.cast::<Collect>().ok().map(|c| c.get().inner.clone())
}
