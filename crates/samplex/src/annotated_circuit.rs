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

//! IR1 vocabulary: the annotations a user puts on a box, and what they resolve to.
//!
//! `Twirl`, `ChangeBasis`, `InjectLocalClifford`, `InjectNoise` and `Tag` are the Python-facing
//! annotations; `resolve_annotations` folds a box's set of them into one `ResolvedBox`. The shared
//! enums the later IRs borrow live here too, so nothing downstream imports vocabulary from a pass.

use std::sync::Arc;

use hashbrown::HashSet;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyString;
use qiskit_circuit::annotation::{Annotation, PyAnnotation};

use crate::error::LowerError;

use crate::virtual_type::VirtualType;

/// The namespace every IR1 annotation declares. Flat, because these are one vocabulary: the one a
/// user writes. Samplex's own *output* annotation, `Collect`, sits under a child namespace instead.
pub(crate) const NAMESPACE: &str = "samplex";

// --- Annotation-level enums (shared with sampling_graph) ------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SynthesizerType {
    RzSx,
    RzRx,
}

impl SynthesizerType {
    pub fn accepts(&self, _vt: VirtualType) -> bool {
        match self {
            Self::RzSx | Self::RzRx => true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DistributionType {
    UniformPauli,
    BalancedUniformPauli,
    UniformPauliSubset,
    UniformC1,
    UniformLocalC1,
    HaarU2,
}

impl DistributionType {
    pub fn virtual_type(&self) -> VirtualType {
        match self {
            Self::UniformPauli | Self::BalancedUniformPauli | Self::UniformPauliSubset => {
                VirtualType::Pauli
            }
            Self::UniformC1 | Self::UniformLocalC1 => VirtualType::C1,
            Self::HaarU2 => VirtualType::U2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChangeBasisMode {
    PreparePauli,
    MeasurePauli,
    LocalClifford,
}

impl ChangeBasisMode {
    pub fn virtual_type(&self) -> VirtualType {
        match self {
            Self::PreparePauli | Self::MeasurePauli => VirtualType::Pauli,
            Self::LocalClifford => VirtualType::C1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dressing {
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Placement {
    Start,
    End,
}

/// Which annotation a [`ResolvedBasis`] came from.
///
/// `ChangeBasis` and `InjectLocalClifford` resolve to the same thing except for where the emission
/// is placed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BasisOrigin {
    /// A frame change for the box as a whole. Sits at the box's outer boundary.
    ChangeBasis,
    /// A local-Clifford injection. Flanks the hard content.
    InjectLocalClifford,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InjectionSite {
    Before,
    After,
}

/// Which samplex annotation this is, or `None` if samplex does not own it.
///
/// Every annotation on a box arrives as an `Arc<dyn Annotation>`, whether samplex wrote it or a
/// stranger did. This is the crate's only classifier over that: it names the ones in our vocabulary
/// and stays silent about the rest, which is what lets a foreign annotation ride through untouched.
pub fn annotation_kind(annotation: &dyn Annotation) -> Option<AnnotationKind> {
    if annotation.downcast_ref::<TwirlSpec>().is_some() {
        Some(AnnotationKind::Twirl)
    } else if annotation.downcast_ref::<ChangeBasisSpec>().is_some() {
        Some(AnnotationKind::ChangeBasis)
    } else if annotation
        .downcast_ref::<InjectLocalCliffordSpec>()
        .is_some()
    {
        Some(AnnotationKind::InjectLocalClifford)
    } else if annotation.downcast_ref::<InjectNoiseSpec>().is_some() {
        Some(AnnotationKind::InjectNoise)
    } else if annotation.downcast_ref::<TagSpec>().is_some() {
        Some(AnnotationKind::Tag)
    } else {
        None
    }
}

/// Annotation type discriminant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnnotationKind {
    Twirl,
    ChangeBasis,
    InjectLocalClifford,
    InjectNoise,
    Tag,
}

/// A twirl annotation.
///
/// The `group` (e.g. Pauli, local-C1) is resolved to a [`DistributionType`] at construction time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TwirlSpec {
    pub distribution: DistributionType,
    pub dressing: Dressing,
    pub decomposition: SynthesizerType,
}

/// A basis-change annotation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChangeBasisSpec {
    pub mode: ChangeBasisMode,
    pub reference: String,
    pub placement: Placement,
    pub decomposition: SynthesizerType,
}

/// An inject-local-Clifford annotation (a local-Clifford frame change).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InjectLocalCliffordSpec {
    pub reference: String,
    pub site: InjectionSite,
}

/// A noise-injection annotation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InjectNoiseSpec {
    pub reference: String,
    pub modifier: Option<String>,
    pub site: InjectionSite,
}

/// A tag annotation. Carries nothing: it marks a box without asking for any emission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TagSpec;

parse_enum!(parse_distribution, DistributionType, "distribution", {
    "uniform_pauli" => UniformPauli,
    "balanced_uniform_pauli" => BalancedUniformPauli,
    "uniform_pauli_subset" => UniformPauliSubset,
    "uniform_c1" => UniformC1,
    "uniform_local_c1" => UniformLocalC1,
    "haar_u2" => HaarU2,
});

parse_enum!(parse_dressing, Dressing, "dressing", {
    "left" => Left,
    "right" => Right,
});

parse_enum!(pub(crate) parse_decomposition, SynthesizerType, "decomposition", {
    "rzsx" => RzSx,
    "rzrx" => RzRx,
});

parse_enum!(parse_change_basis_mode, ChangeBasisMode, "change basis mode", {
    "prepare_pauli" => PreparePauli,
    "measure_pauli" => MeasurePauli,
    "local_clifford" => LocalClifford,
});

parse_enum!(parse_placement, Placement, "placement", {
    "start" => Start,
    "end" => End,
});

parse_enum!(parse_injection_site, InjectionSite, "injection site", {
    "before" => Before,
    "after" => After,
});

impl From<Placement> for crate::sampling_graph::Direction {
    fn from(placement: Placement) -> Self {
        match placement {
            Placement::Start => Self::Left,
            Placement::End => Self::Right,
        }
    }
}

impl From<InjectionSite> for crate::sampling_graph::Direction {
    fn from(site: InjectionSite) -> Self {
        match site {
            InjectionSite::Before => Self::Left,
            InjectionSite::After => Self::Right,
        }
    }
}

/// A basis change, after resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedBasis {
    pub origin: BasisOrigin,
    pub mode: ChangeBasisMode,
    pub ref_id: String,
    pub placement: Placement,
}

/// What a single annotated box resolves to.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResolvedBox {
    pub twirl: Option<TwirlSpec>,
    pub change_basis: Option<ResolvedBasis>,
    pub inject_noise: Option<InjectNoiseSpec>,
    pub synthesizer: Option<SynthesizerType>,
    pub dressing: Option<Dressing>,
}

impl ResolvedBox {
    /// Whether this box produces any emissions at all. A `Tag`-only or unannotated box does not.
    pub fn is_emitting(&self) -> bool {
        self.twirl.is_some() || self.change_basis.is_some() || self.inject_noise.is_some()
    }
}

/// Resolve a box's annotations, applying the vocabulary's validation rules.
pub fn resolve_annotations(annotations: &[Arc<dyn Annotation>]) -> Result<ResolvedBox, LowerError> {
    // `seen` is a membership test only, never iterated, so its `HashSet` cannot leak an ordering
    // into anything downstream.
    let mut seen: HashSet<AnnotationKind> = HashSet::new();
    for annotation in annotations {
        // A foreign annotation is not a duplicate of anything: two of them on one box is fine.
        let Some(kind) = annotation_kind(annotation.as_ref()) else {
            continue;
        };
        if !seen.insert(kind) {
            return Err(LowerError::DuplicateAnnotation(kind));
        }
    }
    if seen.contains(&AnnotationKind::ChangeBasis)
        && seen.contains(&AnnotationKind::InjectLocalClifford)
    {
        return Err(LowerError::ChangeBasisConflict);
    }
    // Both injections happen *to* a twirled box's content, sitting just outside its twirl point, so
    // neither means anything without one. A `ChangeBasis` is the only annotation that stands alone: it
    // names a frame change for the box as a whole.
    if seen.contains(&AnnotationKind::InjectNoise) && !seen.contains(&AnnotationKind::Twirl) {
        return Err(LowerError::InjectNoiseWithoutTwirl);
    }
    if seen.contains(&AnnotationKind::InjectLocalClifford) && !seen.contains(&AnnotationKind::Twirl)
    {
        return Err(LowerError::InjectLocalCliffordWithoutTwirl);
    }

    let mut resolved = ResolvedBox::default();
    for annotation in annotations {
        let annotation = annotation.as_ref();
        if let Some(twirl) = annotation.downcast_ref::<TwirlSpec>() {
            resolved.dressing = Some(twirl.dressing);
            resolved.twirl = Some(twirl.clone());
        } else if let Some(cb) = annotation.downcast_ref::<ChangeBasisSpec>() {
            resolved.change_basis = Some(ResolvedBasis {
                origin: BasisOrigin::ChangeBasis,
                mode: cb.mode,
                ref_id: format!("basis_changes.{}", cb.reference),
                placement: cb.placement,
            });
        } else if let Some(ilc) = annotation.downcast_ref::<InjectLocalCliffordSpec>() {
            resolved.change_basis = Some(ResolvedBasis {
                origin: BasisOrigin::InjectLocalClifford,
                mode: ChangeBasisMode::LocalClifford,
                ref_id: format!("local_cliffords.{}", ilc.reference),
                placement: match ilc.site {
                    InjectionSite::Before => Placement::Start,
                    InjectionSite::After => Placement::End,
                },
            });
        } else if let Some(inj) = annotation.downcast_ref::<InjectNoiseSpec>() {
            resolved.inject_noise = Some(inj.clone());
        }
    }

    // A dressing is the edge a box's absorbable run folds to. A `Twirl` names it outright, since the
    // dressing edge is where its pair sits; a `ChangeBasis` standing alone names it by its placement,
    // which is the only edge such a box has. Without one, nothing in the body could fold anywhere:
    // `classify_body` treats an undressed box as all content.
    if resolved.dressing.is_none() {
        resolved.dressing = resolved
            .change_basis
            .as_ref()
            .map(|basis| match basis.placement {
                Placement::Start => Dressing::Left,
                Placement::End => Dressing::Right,
            });
    }

    // Only `Twirl` and `ChangeBasis` name a synthesizer; `InjectLocalClifford` does not, which is part
    // of why it needs a `Twirl` beside it. Every *emitting* box therefore names one — an emission needs
    // a twirl, a frame change, or a noise injection, and the first two carry a synthesizer while the
    // third requires a twirl — so a consumer's fallback is unreachable rather than merely unused.
    resolved.synthesizer = annotations
        .iter()
        .find_map(|a| {
            a.as_ref()
                .downcast_ref::<TwirlSpec>()
                .map(|t| t.decomposition)
        })
        .or_else(|| {
            annotations.iter().find_map(|a| {
                a.as_ref()
                    .downcast_ref::<ChangeBasisSpec>()
                    .map(|cb| cb.decomposition)
            })
        });
    Ok(resolved)
}

impl Annotation for TwirlSpec {
    fn namespace(&self) -> &str {
        NAMESPACE
    }

    fn create_py_annotation(&self, py: Python) -> PyResult<Py<PyAny>> {
        Ok(Py::new(py, Twirl::init(self.clone()))?.into_any())
    }
}

#[pyclass(module = "qiskit._accelerate.samplex", frozen, extends = PyAnnotation)]
pub struct Twirl {
    inner: Arc<TwirlSpec>,
}

impl Twirl {
    /// Build the initializer, base and subclass sharing one allocation.
    ///
    /// The base *must* carry the native value: without it a Python round trip comes back as an
    /// opaque `PythonAnnotation` and every `downcast_ref::<TwirlSpec>()` reader silently sees `None`.
    fn init(spec: TwirlSpec) -> PyClassInitializer<Self> {
        let inner = Arc::new(spec);
        PyClassInitializer::from(PyAnnotation::new(inner.clone())).add_subclass(Twirl { inner })
    }
}

#[pymethods]
impl Twirl {
    #[new]
    #[pyo3(signature = (distribution="uniform_pauli", dressing="left", decomposition="rzsx"))]
    fn new(
        distribution: &str,
        dressing: &str,
        decomposition: &str,
    ) -> PyResult<PyClassInitializer<Self>> {
        Ok(Twirl::init(TwirlSpec {
            distribution: parse_distribution(distribution)?,
            dressing: parse_dressing(dressing)?,
            decomposition: parse_decomposition(decomposition)?,
        }))
    }

    #[classattr]
    fn namespace(py: Python) -> Py<PyString> {
        intern!(py, NAMESPACE).clone().unbind()
    }

    fn __repr__(&self) -> String {
        format!(
            "Twirl({:?}, {:?}, {:?})",
            self.inner.distribution, self.inner.dressing, self.inner.decomposition
        )
    }
}

impl Annotation for ChangeBasisSpec {
    fn namespace(&self) -> &str {
        NAMESPACE
    }

    fn create_py_annotation(&self, py: Python) -> PyResult<Py<PyAny>> {
        Ok(Py::new(py, ChangeBasis::init(self.clone()))?.into_any())
    }
}

#[pyclass(module = "qiskit._accelerate.samplex", frozen, extends = PyAnnotation)]
pub struct ChangeBasis {
    inner: Arc<ChangeBasisSpec>,
}

impl ChangeBasis {
    /// Build the initializer, base and subclass sharing one allocation.
    ///
    /// The base *must* carry the native value: without it a Python round trip comes back as an
    /// opaque `PythonAnnotation` and every `downcast_ref::<ChangeBasisSpec>()` reader silently sees `None`.
    fn init(spec: ChangeBasisSpec) -> PyClassInitializer<Self> {
        let inner = Arc::new(spec);
        PyClassInitializer::from(PyAnnotation::new(inner.clone()))
            .add_subclass(ChangeBasis { inner })
    }
}

#[pymethods]
impl ChangeBasis {
    #[new]
    #[pyo3(signature = (reference, mode="measure_pauli", placement="end", decomposition="rzsx"))]
    fn new(
        reference: String,
        mode: &str,
        placement: &str,
        decomposition: &str,
    ) -> PyResult<PyClassInitializer<Self>> {
        Ok(ChangeBasis::init(ChangeBasisSpec {
            mode: parse_change_basis_mode(mode)?,
            reference,
            placement: parse_placement(placement)?,
            decomposition: parse_decomposition(decomposition)?,
        }))
    }

    #[classattr]
    fn namespace(py: Python) -> Py<PyString> {
        intern!(py, NAMESPACE).clone().unbind()
    }

    fn __repr__(&self) -> String {
        format!(
            "ChangeBasis({:?}, {:?})",
            self.inner.mode, self.inner.reference
        )
    }
}

impl Annotation for InjectLocalCliffordSpec {
    fn namespace(&self) -> &str {
        NAMESPACE
    }

    fn create_py_annotation(&self, py: Python) -> PyResult<Py<PyAny>> {
        Ok(Py::new(py, InjectLocalClifford::init(self.clone()))?.into_any())
    }
}

#[pyclass(module = "qiskit._accelerate.samplex", frozen, extends = PyAnnotation)]
pub struct InjectLocalClifford {
    inner: Arc<InjectLocalCliffordSpec>,
}

impl InjectLocalClifford {
    /// Build the initializer, base and subclass sharing one allocation.
    ///
    /// The base *must* carry the native value: without it a Python round trip comes back as an
    /// opaque `PythonAnnotation` and every `downcast_ref::<InjectLocalCliffordSpec>()` reader silently sees `None`.
    fn init(spec: InjectLocalCliffordSpec) -> PyClassInitializer<Self> {
        let inner = Arc::new(spec);
        PyClassInitializer::from(PyAnnotation::new(inner.clone()))
            .add_subclass(InjectLocalClifford { inner })
    }
}

#[pymethods]
impl InjectLocalClifford {
    #[new]
    #[pyo3(signature = (reference, site="before"))]
    fn new(reference: String, site: &str) -> PyResult<PyClassInitializer<Self>> {
        Ok(InjectLocalClifford::init(InjectLocalCliffordSpec {
            reference,
            site: parse_injection_site(site)?,
        }))
    }

    #[classattr]
    fn namespace(py: Python) -> Py<PyString> {
        intern!(py, NAMESPACE).clone().unbind()
    }

    fn __repr__(&self) -> String {
        format!(
            "InjectLocalClifford({:?}, {:?})",
            self.inner.reference, self.inner.site
        )
    }
}

impl Annotation for InjectNoiseSpec {
    fn namespace(&self) -> &str {
        NAMESPACE
    }

    fn create_py_annotation(&self, py: Python) -> PyResult<Py<PyAny>> {
        Ok(Py::new(py, InjectNoise::init(self.clone()))?.into_any())
    }
}

#[pyclass(module = "qiskit._accelerate.samplex", frozen, extends = PyAnnotation)]
pub struct InjectNoise {
    inner: Arc<InjectNoiseSpec>,
}

impl InjectNoise {
    /// Build the initializer, base and subclass sharing one allocation.
    ///
    /// The base *must* carry the native value: without it a Python round trip comes back as an
    /// opaque `PythonAnnotation` and every `downcast_ref::<InjectNoiseSpec>()` reader silently sees `None`.
    fn init(spec: InjectNoiseSpec) -> PyClassInitializer<Self> {
        let inner = Arc::new(spec);
        PyClassInitializer::from(PyAnnotation::new(inner.clone()))
            .add_subclass(InjectNoise { inner })
    }
}

#[pymethods]
impl InjectNoise {
    #[new]
    #[pyo3(signature = (reference, site="after", modifier=None))]
    fn new(
        reference: String,
        site: &str,
        modifier: Option<String>,
    ) -> PyResult<PyClassInitializer<Self>> {
        Ok(InjectNoise::init(InjectNoiseSpec {
            reference,
            modifier,
            site: parse_injection_site(site)?,
        }))
    }

    #[classattr]
    fn namespace(py: Python) -> Py<PyString> {
        intern!(py, NAMESPACE).clone().unbind()
    }

    fn __repr__(&self) -> String {
        format!(
            "InjectNoise({:?}, {:?})",
            self.inner.reference, self.inner.site
        )
    }
}

impl Annotation for TagSpec {
    fn namespace(&self) -> &str {
        NAMESPACE
    }

    fn create_py_annotation(&self, py: Python) -> PyResult<Py<PyAny>> {
        Ok(Py::new(py, Tag::init())?.into_any())
    }
}

#[pyclass(module = "qiskit._accelerate.samplex", frozen, extends = PyAnnotation)]
pub struct Tag;

impl Tag {
    /// Build the initializer. `Tag` has no payload, but the base still carries the native value so
    /// that a round-tripped `Tag` downcasts back to a `TagSpec`.
    fn init() -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyAnnotation::new(Arc::new(TagSpec))).add_subclass(Tag)
    }
}

#[pymethods]
impl Tag {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        Tag::init()
    }

    #[classattr]
    fn namespace(py: Python) -> Py<PyString> {
        intern!(py, NAMESPACE).clone().unbind()
    }

    fn __repr__(&self) -> String {
        "Tag()".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qiskit_circuit::annotation::extract_annotation;

    /// Erase a spec to the trait object a box actually stores.
    fn annotation<A: Annotation>(spec: A) -> Arc<dyn Annotation> {
        Arc::new(spec)
    }

    fn twirl(distribution: DistributionType) -> Arc<dyn Annotation> {
        annotation(TwirlSpec {
            distribution,
            dressing: Dressing::Left,
            decomposition: SynthesizerType::RzSx,
        })
    }

    /// Assert a spec survives a Python round trip as itself.
    ///
    /// This pins the port's one silent failure mode. `create_py_annotation` must put the native value
    /// on the `PyAnnotation` base; if it does not, `extract_annotation` degrades the annotation to an
    /// opaque `PythonAnnotation`, every `downcast_ref` reader in the crate starts seeing `None`, and
    /// nothing errors — the box simply stops being recognised as ours.
    fn assert_round_trips<A>(spec: A)
    where
        A: Annotation + Clone + PartialEq + std::fmt::Debug,
    {
        Python::initialize();
        Python::attach(|py| {
            let object = spec.create_py_annotation(py).unwrap();
            let recovered = extract_annotation(object.bind(py));
            assert_eq!(
                recovered.downcast_ref::<A>(),
                Some(&spec),
                "{spec:?} did not come back as itself"
            );
        });
    }

    fn change_basis(reference: &str, placement: Placement) -> Arc<dyn Annotation> {
        annotation(ChangeBasisSpec {
            mode: ChangeBasisMode::MeasurePauli,
            reference: reference.to_string(),
            placement,
            decomposition: SynthesizerType::RzSx,
        })
    }

    #[test]
    fn test_resolve_twirl_implies_collect() {
        let resolved = resolve_annotations(&[twirl(DistributionType::UniformPauli)]).unwrap();
        assert!(resolved.twirl.is_some());
        assert_eq!(resolved.dressing, Some(Dressing::Left));
        assert!(resolved.synthesizer.is_some());
    }

    #[test]
    fn test_resolve_change_basis_ref_namespacing() {
        let resolved = resolve_annotations(&[change_basis("0", Placement::Start)]).unwrap();
        let cb = resolved.change_basis.unwrap();
        assert_eq!(cb.ref_id, "basis_changes.0");
        assert_eq!(cb.mode, ChangeBasisMode::MeasurePauli);
        assert_eq!(cb.placement, Placement::Start);
    }

    #[test]
    fn test_resolve_inject_local_clifford() {
        // Needs a twirl beside it: an injection happens *to* a twirled box's content.
        let resolved = resolve_annotations(&[
            twirl(DistributionType::UniformPauli),
            annotation(InjectLocalCliffordSpec {
                reference: "c3".to_string(),
                site: InjectionSite::Before,
            }),
        ])
        .unwrap();
        let cb = resolved.change_basis.unwrap();
        assert_eq!(cb.mode, ChangeBasisMode::LocalClifford);
        assert_eq!(cb.ref_id, "local_cliffords.c3");
        assert_eq!(cb.placement, Placement::Start);
    }

    #[test]
    fn test_resolve_inject_local_clifford_without_twirl_errors() {
        let err = resolve_annotations(&[annotation(InjectLocalCliffordSpec {
            reference: "c3".to_string(),
            site: InjectionSite::Before,
        })])
        .unwrap_err();
        assert_eq!(err, LowerError::InjectLocalCliffordWithoutTwirl);
    }

    #[test]
    fn test_resolve_change_basis_alone_dresses_by_placement() {
        // The one annotation that stands alone, so the one that has to name its own dressing edge.
        for (placement, expected) in [
            (Placement::Start, Dressing::Left),
            (Placement::End, Dressing::Right),
        ] {
            let resolved = resolve_annotations(&[change_basis("b", placement)]).unwrap();
            assert_eq!(resolved.dressing, Some(expected));
        }
    }

    #[test]
    fn test_a_twirl_names_the_dressing_over_any_placement() {
        let resolved = resolve_annotations(&[
            twirl(DistributionType::UniformPauli),
            change_basis("b", Placement::End),
        ])
        .unwrap();
        // The twirl's own edge wins: that is where its pair sits, and the pair is what the dressing is
        // for. A `ChangeBasis` only names it when there is no twirl to.
        assert_eq!(resolved.dressing, Some(Dressing::Left));
    }

    #[test]
    fn test_resolve_tag_only_produces_nothing() {
        let resolved = resolve_annotations(&[annotation(TagSpec)]).unwrap();
        assert!(resolved.twirl.is_none());
        assert!(resolved.change_basis.is_none());
        assert!(resolved.inject_noise.is_none());
        assert!(resolved.synthesizer.is_none());
    }

    #[test]
    fn test_resolve_duplicate_annotation_errors() {
        let err = resolve_annotations(&[
            twirl(DistributionType::UniformPauli),
            twirl(DistributionType::HaarU2),
        ])
        .unwrap_err();
        assert_eq!(err, LowerError::DuplicateAnnotation(AnnotationKind::Twirl));
    }

    #[test]
    fn test_resolve_inject_noise_without_twirl_errors() {
        let err = resolve_annotations(&[annotation(InjectNoiseSpec {
            reference: "r0".to_string(),
            modifier: None,
            site: InjectionSite::Before,
        })])
        .unwrap_err();
        assert_eq!(err, LowerError::InjectNoiseWithoutTwirl);
    }

    #[test]
    fn test_resolve_inject_noise_with_twirl_ok() {
        let resolved = resolve_annotations(&[
            twirl(DistributionType::UniformPauli),
            annotation(InjectNoiseSpec {
                reference: "r0".to_string(),
                modifier: Some("m1".to_string()),
                site: InjectionSite::After,
            }),
        ])
        .unwrap();
        let inj = resolved.inject_noise.unwrap();
        assert_eq!(inj.reference, "r0");
        assert_eq!(inj.modifier.as_deref(), Some("m1"));
        assert_eq!(inj.site, InjectionSite::After);
    }

    #[test]
    fn test_resolve_change_basis_conflict_errors() {
        let err = resolve_annotations(&[
            change_basis("0", Placement::Start),
            annotation(InjectLocalCliffordSpec {
                reference: "0".to_string(),
                site: InjectionSite::Before,
            }),
        ])
        .unwrap_err();
        assert_eq!(err, LowerError::ChangeBasisConflict);
    }

    #[test]
    fn test_twirl_round_trips_through_python() {
        assert_round_trips(TwirlSpec {
            distribution: DistributionType::UniformPauli,
            dressing: Dressing::Left,
            decomposition: SynthesizerType::RzSx,
        });
    }

    #[test]
    fn test_change_basis_round_trips_through_python() {
        assert_round_trips(ChangeBasisSpec {
            mode: ChangeBasisMode::MeasurePauli,
            reference: "0".to_string(),
            placement: Placement::Start,
            decomposition: SynthesizerType::RzSx,
        });
    }

    #[test]
    fn test_inject_local_clifford_round_trips_through_python() {
        assert_round_trips(InjectLocalCliffordSpec {
            reference: "c3".to_string(),
            site: InjectionSite::Before,
        });
    }

    #[test]
    fn test_inject_noise_round_trips_through_python() {
        assert_round_trips(InjectNoiseSpec {
            reference: "r0".to_string(),
            modifier: Some("m1".to_string()),
            site: InjectionSite::After,
        });
    }

    #[test]
    fn test_tag_round_trips_through_python() {
        // Carries no payload, so the round trip is entirely about identity: a tag has to come back as
        // a `TagSpec` and not as an opaque annotation that happens to have the right namespace.
        assert_round_trips(TagSpec);
    }

    #[test]
    fn test_ir1_annotations_share_the_flat_namespace() {
        // One vocabulary, the one a user writes, so one namespace. `CollectSpec` is the exception and
        // says why in `emission_circuit`.
        for annotation in [
            twirl(DistributionType::UniformPauli),
            change_basis("0", Placement::Start),
            annotation(TagSpec),
            annotation(InjectLocalCliffordSpec {
                reference: "c3".to_string(),
                site: InjectionSite::Before,
            }),
            annotation(InjectNoiseSpec {
                reference: "r0".to_string(),
                modifier: None,
                site: InjectionSite::After,
            }),
        ] {
            assert_eq!(annotation.namespace(), NAMESPACE);
        }
        assert_eq!(NAMESPACE, "samplex");
    }

    #[test]
    fn test_python_visible_namespace_matches_the_native_one() {
        // The class attribute is what a Python-side dispatch table keys on, and the trait method is
        // what Rust reads. They are separate surfaces over one const, and this is what keeps them so.
        Python::initialize();
        Python::attach(|py| {
            for declared in [
                Twirl::namespace(py),
                ChangeBasis::namespace(py),
                InjectLocalClifford::namespace(py),
                InjectNoise::namespace(py),
                Tag::namespace(py),
            ] {
                assert_eq!(declared.extract::<String>(py).unwrap(), NAMESPACE);
            }
        });
    }
}
