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

parse_enum!(parse_virtual_type, VirtualType, "virtual type", {
    "pauli" => Pauli,
    "c1" => C1,
    "u2" => U2,
    "z2" => Z2,
});

/// The payload of an [`Emit`] instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmitSpec {
    /// Identity of this emission within one lowered circuit.
    ///
    /// This is what a collect box's [`Collect`](crate::annotated_circuit::CollectSpec) annotation refers to
    /// when it names the emissions it consumes.
    pub id: u32,
    /// Which annotation this emission stands in for.
    pub source: EmitSource,
    /// The distribution this emission draws from. The two halves of a twirl share this key.
    pub dist: DistKey,
    /// Which way the emitted virtual state flows.
    pub direction: Direction,
    /// The algebraic type of the emitted virtual gates.
    pub virtual_type: VirtualType,
    /// Subsystem grouping over the emission's qubits, in the *global* circuit frame.
    ///
    /// The instruction's qargs are body-local (a box body is its own circuit indexed `0..width`),
    /// so this is the one place the global frame is recorded; the graph reader relies on it rather
    /// than re-deriving it through enclosing box qargs.
    pub partition: Partition,
}

impl EmitSpec {
    /// The qubits this emission acts on, in the global frame, in ascending order.
    pub fn qubits(&self) -> Vec<usize> {
        let mut qubits: Vec<usize> = self.partition.all_elements().iter().copied().collect();
        qubits.sort_unstable();
        qubits
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
    #[pyo3(signature = (subsystems, id=0, source="twirl", distribution_key=0, direction="left", virtual_type="pauli"))]
    fn py_new(
        py: Python,
        subsystems: Vec<Vec<usize>>,
        id: u32,
        source: &str,
        distribution_key: u32,
        direction: &str,
        virtual_type: &str,
    ) -> PyResult<Self> {
        ensure_registered(py)?;
        if subsystems.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Emit requires at least one subsystem.",
            ));
        }
        let partition =
            Partition::with_parts(subsystems.into_iter().map(|part| part.into_boxed_slice()))
                .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
        Ok(Emit {
            inner: EmitSpec {
                id,
                source: parse_source(source)?,
                dist: DistKey(distribution_key),
                direction: parse_direction(direction)?,
                virtual_type: parse_virtual_type(virtual_type)?,
                partition,
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
    fn id(&self) -> u32 {
        self.inner.id
    }

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
        self.inner.dist.0
    }

    #[getter]
    fn direction(&self) -> &'static str {
        match self.inner.direction {
            Direction::Left => "left",
            Direction::Right => "right",
        }
    }

    #[getter]
    fn virtual_type(&self) -> &'static str {
        match self.inner.virtual_type {
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

    fn __repr__(&self) -> String {
        format!(
            "Emit(#{}, {}, dist={}, {}, {:?})",
            self.inner.id,
            self.source(),
            self.inner.dist.0,
            self.direction(),
            self.inner.qubits(),
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
                self.inner.id,
                self.source(),
                self.inner.dist.0,
                self.direction(),
                self.virtual_type(),
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
/// it at the position its [`CollectItem`] list dictates. No standalone `Emit` instruction, no VFG
/// `Emission` node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalEmission {
    pub source: EmitSource,
    pub dist: DistKey,
    pub direction: Direction,
    pub virtual_type: VirtualType,
    pub partition: Partition,
}

/// One step in what a collector composes, in circuit order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CollectItem {
    /// A local emission owned by this collector. No standalone instruction needed.
    Emission(LocalEmission),
    /// The next `n` gates of the collect box's body.
    ///
    /// A **count**, not an index range. Merging concatenates bodies, and a count needs no offsetting
    /// when it does: the reader walks the items with a cursor into the body, so concatenating items and
    /// concatenating bodies is all a merge has to do. The counts of a well-formed collector sum to its
    /// body length.
    Gates(usize),
    /// A propagating emission (far twirl half) that arrives via graph edges after being conjugated
    /// by intervening gates. References a standalone [`Emit`] instruction by id.
    Incoming(u32),
}

/// The payload of a [`Collect`] annotation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectSpec {
    /// How the collected virtual gates will be synthesized into the box body.
    pub synthesizer: SynthesizerType,
    /// Everything this collector composes, in **circuit order** — which is outermost-first on the left
    /// of a box and innermost-first on the right.
    ///
    /// Ordered, and interleaved with the absorbed gates, because position in the layer is meaningful: a
    /// `ChangeBasis` wraps the whole box and so composes outside the easy gates the dressing absorbed,
    /// whereas an injection or a twirl attaches to the hard content and composes inside them. Two
    /// independent lists could not express that.
    ///
    /// Each `Emit` carries its own direction, so no per-reference metadata is needed here. Recording the
    /// references explicitly is what lets the graph reader avoid re-deriving the contextual collection
    /// rules (shared middle collectors, growing qubit sets) a second time.
    pub items: Vec<CollectItem>,
}

impl CollectSpec {
    /// The IDs of incoming (propagating) emissions, in composition order.
    pub fn incoming_ids(&self) -> Vec<u32> {
        self.items
            .iter()
            .filter_map(|item| match item {
                CollectItem::Incoming(id) => Some(*id),
                _ => None,
            })
            .collect()
    }

    /// How many body gates the items account for. Should equal the collect box's body length.
    pub fn gate_count(&self) -> usize {
        self.items
            .iter()
            .map(|item| match item {
                CollectItem::Gates(count) => *count,
                _ => 0,
            })
            .sum()
    }

    /// Whether this collector consumes no emissions at all (neither local nor incoming).
    pub fn collects_nothing(&self) -> bool {
        !self.items.iter().any(|item| {
            matches!(item, CollectItem::Emission(_) | CollectItem::Incoming(_))
        })
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
    /// Construct a `Collect` naming the incoming emissions it consumes, in order.
    ///
    /// Absorbed-gate positions cannot be given here — a hand-built annotation has no body to slice —
    /// so this produces incoming items only. The build pass constructs the interleaved form directly.
    #[new]
    #[pyo3(signature = (collects, synthesizer="rzsx"))]
    fn new(collects: Vec<u32>, synthesizer: &str) -> PyResult<PyClassInitializer<Self>> {
        Ok(
            PyClassInitializer::from(PyAnnotation).add_subclass(Collect {
                inner: CollectSpec {
                    synthesizer: parse_decomposition(synthesizer)?,
                    items: collects.into_iter().map(CollectItem::Incoming).collect(),
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
        match self.inner.synthesizer {
            SynthesizerType::RzSx => "rzsx",
            SynthesizerType::RzRx => "rzrx",
        }
    }

    /// The incoming emission IDs consumed, in composition order. Local emissions are not shown
    /// here — see [`items`](Self::items).
    #[getter]
    fn collects(&self) -> Vec<u32> {
        self.inner.incoming_ids()
    }

    /// Everything composed, in order: `("local", 0)` for a local emission, `("incoming", id)` for
    /// a propagating emission, `("gates", n)` for the next `n` gates of this box's body.
    #[getter]
    fn items(&self) -> Vec<(&'static str, usize)> {
        self.inner
            .items
            .iter()
            .map(|item| match item {
                CollectItem::Emission(_) => ("local", 0),
                CollectItem::Incoming(id) => ("incoming", *id as usize),
                CollectItem::Gates(count) => ("gates", *count),
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        let items = self
            .inner
            .items
            .iter()
            .map(|item| match item {
                CollectItem::Emission(_) => "~".to_string(),
                CollectItem::Incoming(id) => format!("#{id}"),
                CollectItem::Gates(count) => format!("{count}g"),
            })
            .collect::<Vec<_>>()
            .join(" ");
        format!("Collect({:?}, [{}])", self.inner.synthesizer, items)
    }
}

/// Try to extract a [`CollectSpec`] from a Python annotation object.
pub fn extract_collect(obj: &Bound<'_, PyAny>) -> Option<CollectSpec> {
    obj.cast::<Collect>().ok().map(|c| c.get().inner.clone())
}
