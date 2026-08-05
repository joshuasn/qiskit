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

use hashbrown::HashSet;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyString;
use qiskit_circuit::annotation::PyAnnotation;

use crate::error::LowerError;

use crate::virtual_type::VirtualType;

// --- Annotation-level enums (shared with virtual_flow_graph) ------------------------------------

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
/// `ChangeBasis` and `InjectLocalClifford` resolve to the same thing except for where the emission is
/// placed.
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

/// A single annotation attached to an annotated box.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoxAnnotation {
    Twirl(TwirlSpec),
    ChangeBasis(ChangeBasisSpec),
    InjectLocalClifford(InjectLocalCliffordSpec),
    InjectNoise(InjectNoiseSpec),
    Tag,
}

impl BoxAnnotation {
    pub fn kind(&self) -> AnnotationKind {
        match self {
            BoxAnnotation::Twirl(_) => AnnotationKind::Twirl,
            BoxAnnotation::ChangeBasis(_) => AnnotationKind::ChangeBasis,
            BoxAnnotation::InjectLocalClifford(_) => AnnotationKind::InjectLocalClifford,
            BoxAnnotation::InjectNoise(_) => AnnotationKind::InjectNoise,
            BoxAnnotation::Tag => AnnotationKind::Tag,
        }
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

impl From<Placement> for crate::virtual_flow_graph::Direction {
    fn from(placement: Placement) -> Self {
        match placement {
            Placement::Start => Self::Left,
            Placement::End => Self::Right,
        }
    }
}

impl From<InjectionSite> for crate::virtual_flow_graph::Direction {
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
pub fn resolve_annotations(annotations: &[BoxAnnotation]) -> Result<ResolvedBox, LowerError> {
    let mut seen: HashSet<AnnotationKind> = HashSet::new();
    for annotation in annotations {
        if !seen.insert(annotation.kind()) {
            return Err(LowerError::DuplicateAnnotation(annotation.kind()));
        }
    }
    if seen.contains(&AnnotationKind::ChangeBasis)
        && seen.contains(&AnnotationKind::InjectLocalClifford)
    {
        return Err(LowerError::ChangeBasisConflict);
    }
    if seen.contains(&AnnotationKind::InjectNoise) && !seen.contains(&AnnotationKind::Twirl) {
        return Err(LowerError::InjectNoiseWithoutTwirl);
    }

    let mut resolved = ResolvedBox::default();
    for annotation in annotations {
        match annotation {
            BoxAnnotation::Twirl(twirl) => {
                resolved.dressing = Some(twirl.dressing);
                resolved.twirl = Some(twirl.clone());
            }
            BoxAnnotation::ChangeBasis(cb) => {
                resolved.change_basis = Some(ResolvedBasis {
                    origin: BasisOrigin::ChangeBasis,
                    mode: cb.mode,
                    ref_id: format!("basis_changes.{}", cb.reference),
                    placement: cb.placement,
                });
            }
            BoxAnnotation::InjectLocalClifford(ilc) => {
                resolved.change_basis = Some(ResolvedBasis {
                    origin: BasisOrigin::InjectLocalClifford,
                    mode: ChangeBasisMode::LocalClifford,
                    ref_id: format!("local_cliffords.{}", ilc.reference),
                    placement: match ilc.site {
                        InjectionSite::Before => Placement::Start,
                        InjectionSite::After => Placement::End,
                    },
                });
            }
            BoxAnnotation::InjectNoise(inj) => resolved.inject_noise = Some(inj.clone()),
            BoxAnnotation::Tag => {}
        }
    }

    // Only `Twirl` and `ChangeBasis` name a synthesizer; `InjectLocalClifford` does not, so a box
    // carrying only that one leaves this `None` and the consumer picks its own default.
    resolved.synthesizer = annotations
        .iter()
        .find_map(|a| match a {
            BoxAnnotation::Twirl(t) => Some(t.decomposition),
            _ => None,
        })
        .or_else(|| {
            annotations.iter().find_map(|a| match a {
                BoxAnnotation::ChangeBasis(cb) => Some(cb.decomposition),
                _ => None,
            })
        });
    Ok(resolved)
}

#[pyclass(module = "qiskit._accelerate.samplex", frozen, extends = PyAnnotation)]
pub struct Twirl {
    pub(crate) inner: TwirlSpec,
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
        Ok(PyClassInitializer::from(PyAnnotation).add_subclass(Twirl {
            inner: TwirlSpec {
                distribution: parse_distribution(distribution)?,
                dressing: parse_dressing(dressing)?,
                decomposition: parse_decomposition(decomposition)?,
            },
        }))
    }

    #[classattr]
    fn namespace(py: Python) -> Py<PyString> {
        intern!(py, "samplex").clone().unbind()
    }

    fn __repr__(&self) -> String {
        format!(
            "Twirl({:?}, {:?}, {:?})",
            self.inner.distribution, self.inner.dressing, self.inner.decomposition
        )
    }
}

#[pyclass(module = "qiskit._accelerate.samplex", frozen, extends = PyAnnotation)]
pub struct ChangeBasis {
    pub(crate) inner: ChangeBasisSpec,
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
        Ok(
            PyClassInitializer::from(PyAnnotation).add_subclass(ChangeBasis {
                inner: ChangeBasisSpec {
                    mode: parse_change_basis_mode(mode)?,
                    reference,
                    placement: parse_placement(placement)?,
                    decomposition: parse_decomposition(decomposition)?,
                },
            }),
        )
    }

    #[classattr]
    fn namespace(py: Python) -> Py<PyString> {
        intern!(py, "samplex").clone().unbind()
    }

    fn __repr__(&self) -> String {
        format!(
            "ChangeBasis({:?}, {:?})",
            self.inner.mode, self.inner.reference
        )
    }
}

#[pyclass(module = "qiskit._accelerate.samplex", frozen, extends = PyAnnotation)]
pub struct InjectLocalClifford {
    pub(crate) inner: InjectLocalCliffordSpec,
}

#[pymethods]
impl InjectLocalClifford {
    #[new]
    #[pyo3(signature = (reference, site="before"))]
    fn new(reference: String, site: &str) -> PyResult<PyClassInitializer<Self>> {
        Ok(
            PyClassInitializer::from(PyAnnotation).add_subclass(InjectLocalClifford {
                inner: InjectLocalCliffordSpec {
                    reference,
                    site: parse_injection_site(site)?,
                },
            }),
        )
    }

    #[classattr]
    fn namespace(py: Python) -> Py<PyString> {
        intern!(py, "samplex").clone().unbind()
    }

    fn __repr__(&self) -> String {
        format!(
            "InjectLocalClifford({:?}, {:?})",
            self.inner.reference, self.inner.site
        )
    }
}

#[pyclass(module = "qiskit._accelerate.samplex", frozen, extends = PyAnnotation)]
pub struct InjectNoise {
    pub(crate) inner: InjectNoiseSpec,
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
        Ok(
            PyClassInitializer::from(PyAnnotation).add_subclass(InjectNoise {
                inner: InjectNoiseSpec {
                    reference,
                    modifier,
                    site: parse_injection_site(site)?,
                },
            }),
        )
    }

    #[classattr]
    fn namespace(py: Python) -> Py<PyString> {
        intern!(py, "samplex").clone().unbind()
    }

    fn __repr__(&self) -> String {
        format!(
            "InjectNoise({:?}, {:?})",
            self.inner.reference, self.inner.site
        )
    }
}

#[pyclass(module = "qiskit._accelerate.samplex", frozen, extends = PyAnnotation)]
pub struct Tag;

#[pymethods]
impl Tag {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyAnnotation).add_subclass(Tag)
    }

    #[classattr]
    fn namespace(py: Python) -> Py<PyString> {
        intern!(py, "samplex").clone().unbind()
    }

    fn __repr__(&self) -> String {
        "Tag()".to_string()
    }
}

/// Try to extract a Python annotation object into a `BoxAnnotation`.
pub fn extract_annotation(obj: &Bound<'_, PyAny>) -> PyResult<BoxAnnotation> {
    if let Ok(t) = obj.cast::<Twirl>() {
        Ok(BoxAnnotation::Twirl(t.get().inner.clone()))
    } else if let Ok(cb) = obj.cast::<ChangeBasis>() {
        Ok(BoxAnnotation::ChangeBasis(cb.get().inner.clone()))
    } else if let Ok(ilc) = obj.cast::<InjectLocalClifford>() {
        Ok(BoxAnnotation::InjectLocalClifford(ilc.get().inner.clone()))
    } else if let Ok(inj) = obj.cast::<InjectNoise>() {
        Ok(BoxAnnotation::InjectNoise(inj.get().inner.clone()))
    } else if obj.cast::<Tag>().is_ok() {
        Ok(BoxAnnotation::Tag)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "Unknown annotation type: {}",
            obj.get_type().name()?
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn twirl(distribution: DistributionType) -> BoxAnnotation {
        BoxAnnotation::Twirl(TwirlSpec {
            distribution,
            dressing: Dressing::Left,
            decomposition: SynthesizerType::RzSx,
        })
    }

    fn change_basis(reference: &str, placement: Placement) -> BoxAnnotation {
        BoxAnnotation::ChangeBasis(ChangeBasisSpec {
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
        let resolved = resolve_annotations(&[BoxAnnotation::InjectLocalClifford(
            InjectLocalCliffordSpec {
                reference: "c3".to_string(),
                site: InjectionSite::Before,
            },
        )])
        .unwrap();
        let cb = resolved.change_basis.unwrap();
        assert_eq!(cb.mode, ChangeBasisMode::LocalClifford);
        assert_eq!(cb.ref_id, "local_cliffords.c3");
        assert_eq!(cb.placement, Placement::Start);
    }

    #[test]
    fn test_resolve_tag_only_produces_nothing() {
        let resolved = resolve_annotations(&[BoxAnnotation::Tag]).unwrap();
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
        let err = resolve_annotations(&[BoxAnnotation::InjectNoise(InjectNoiseSpec {
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
            BoxAnnotation::InjectNoise(InjectNoiseSpec {
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
            BoxAnnotation::InjectLocalClifford(InjectLocalCliffordSpec {
                reference: "0".to_string(),
                site: InjectionSite::Before,
            }),
        ])
        .unwrap_err();
        assert_eq!(err, LowerError::ChangeBasisConflict);
    }
}
