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

//! IR2's side object: the deduplicated distributions its `Emit` instructions draw from.
//!
//! Produced by the build pass and immutable thereafter, so it is an *input* to lowering.
//! `draw_counts` sizes each entry's sample array.

use hashbrown::{HashMap, HashSet};
use pyo3::prelude::*;

use crate::annotated_circuit::{ChangeBasisMode, DistributionType};
use crate::virtual_type::VirtualType;

/// A key into a [`DistributionTable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DistKey(pub u32);

impl DistKey {
    /// The index this key refers to.
    pub fn index(&self) -> usize {
        self.0 as usize
    }
}

/// A single entry in a [`DistributionTable`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DistEntry {
    /// A randomization distribution, e.g. from a `Twirl`.
    Distribution(DistributionType),
    /// A reference to a Pauli-Lindblad noise map, with an optional modifier.
    Noise {
        reference: String,
        modifier: Option<String>,
    },
    /// A deterministic frame change, referencing user-supplied basis data.
    Basis {
        mode: ChangeBasisMode,
        ref_id: String,
    },
}

impl DistEntry {
    /// A short human-readable label, used in `__repr__` and in graph rendering.
    pub fn label(&self) -> String {
        match self {
            Self::Distribution(distribution) => format!("{distribution:?}"),
            Self::Noise {
                reference,
                modifier: Some(modifier),
            } => format!("Noise({reference}, {modifier})"),
            Self::Noise {
                reference,
                modifier: None,
            } => format!("Noise({reference})"),
            Self::Basis { mode, ref_id } => format!("Basis({mode:?}, {ref_id})"),
        }
    }

    /// The algebraic type of the virtual gates drawn from this entry.
    pub fn virtual_type(&self) -> VirtualType {
        match self {
            Self::Distribution(distribution) => distribution.virtual_type(),
            Self::Noise { .. } => VirtualType::Pauli,
            Self::Basis { mode, .. } => mode.virtual_type(),
        }
    }
}

/// The deduplicated table of distributions referenced by a lowered circuit's `Emit` instructions.
#[pyclass(module = "qiskit._accelerate.samplex", skip_from_py_object)]
#[derive(Debug, Clone, Default)]
pub struct DistributionTable {
    entries: Vec<DistEntry>,
    lookup: HashMap<DistEntry, DistKey>,
    /// Modifier references grouped by the noise reference they modify.
    noise_modifiers: HashMap<String, HashSet<String>>,
    /// How many subsystem draws are allocated for each entry (indexed by `DistKey`).
    draw_counts: Vec<u32>,
}

impl DistributionTable {
    /// Construct an empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert `entry`, returning the key for it. Returns the existing key if it is already present.
    pub fn intern(&mut self, entry: DistEntry) -> DistKey {
        if let Some(key) = self.lookup.get(&entry) {
            return *key;
        }
        if let DistEntry::Noise {
            reference,
            modifier: Some(modifier),
        } = &entry
        {
            self.noise_modifiers
                .entry(reference.clone())
                .or_default()
                .insert(modifier.clone());
        }
        let key = DistKey(self.entries.len() as u32);
        self.entries.push(entry.clone());
        self.draw_counts.push(0);
        self.lookup.insert(entry, key);
        key
    }

    /// Set the total number of subsystem draws for `key`. Called by the build pass after walking
    /// the full circuit to finalize how large each key's sample array needs to be.
    pub fn set_draw_count(&mut self, key: DistKey, count: u32) {
        self.draw_counts[key.index()] = count;
    }

    /// Look up an entry by key.
    pub fn get(&self, key: DistKey) -> Option<&DistEntry> {
        self.entries.get(key.index())
    }

    /// All entries, in key order.
    pub fn entries(&self) -> &[DistEntry] {
        &self.entries
    }

    /// The number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The modifier references associated with each noise reference.
    pub fn noise_modifiers(&self) -> &HashMap<String, HashSet<String>> {
        &self.noise_modifiers
    }
}

#[pymethods]
impl DistributionTable {
    fn __len__(&self) -> usize {
        self.len()
    }

    fn __repr__(&self) -> String {
        let entries = self
            .entries
            .iter()
            .enumerate()
            .map(|(idx, entry)| format!("{idx}: {}", entry.label()))
            .collect::<Vec<_>>()
            .join(", ");
        format!("DistributionTable({entries})")
    }

    /// The table's entries as labels, in key order.
    #[pyo3(name = "entries")]
    fn py_entries(&self) -> Vec<String> {
        self.entries.iter().map(|entry| entry.label()).collect()
    }

    /// The modifier references associated with each noise reference.
    #[getter]
    fn get_noise_modifiers(&self) -> HashMap<String, Vec<String>> {
        self.noise_modifiers
            .iter()
            .map(|(reference, modifiers)| {
                let mut modifiers: Vec<String> = modifiers.iter().cloned().collect();
                modifiers.sort();
                (reference.clone(), modifiers)
            })
            .collect()
    }

    /// How many subsystem draws each entry needs, in key order.
    #[getter]
    fn get_draw_counts(&self) -> Vec<u32> {
        self.draw_counts.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intern_dedups_identical_entries() {
        let mut table = DistributionTable::new();
        let a = table.intern(DistEntry::Distribution(DistributionType::UniformPauli));
        let b = table.intern(DistEntry::Distribution(DistributionType::UniformPauli));
        assert_eq!(a, b);
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn test_intern_distinguishes_distributions() {
        let mut table = DistributionTable::new();
        let a = table.intern(DistEntry::Distribution(DistributionType::UniformPauli));
        let b = table.intern(DistEntry::Distribution(DistributionType::HaarU2));
        assert_ne!(a, b);
        assert_eq!(table.len(), 2);
        assert_eq!(
            table.get(a),
            Some(&DistEntry::Distribution(DistributionType::UniformPauli))
        );
        assert_eq!(
            table.get(b),
            Some(&DistEntry::Distribution(DistributionType::HaarU2))
        );
    }

    #[test]
    fn test_noise_modifiers_are_grouped() {
        let mut table = DistributionTable::new();
        table.intern(DistEntry::Noise {
            reference: "n0".to_string(),
            modifier: Some("m1".to_string()),
        });
        table.intern(DistEntry::Noise {
            reference: "n0".to_string(),
            modifier: Some("m2".to_string()),
        });
        table.intern(DistEntry::Noise {
            reference: "n1".to_string(),
            modifier: None,
        });
        assert_eq!(table.len(), 3);
        let modifiers = table.noise_modifiers();
        assert_eq!(modifiers["n0"].len(), 2);
        assert!(modifiers["n0"].contains("m1"));
        assert!(modifiers["n0"].contains("m2"));
        assert!(!modifiers.contains_key("n1"));
    }

    #[test]
    fn test_basis_entries_distinguish_mode_and_ref() {
        let mut table = DistributionTable::new();
        let a = table.intern(DistEntry::Basis {
            mode: ChangeBasisMode::MeasurePauli,
            ref_id: "basis_changes.0".to_string(),
        });
        let b = table.intern(DistEntry::Basis {
            mode: ChangeBasisMode::PreparePauli,
            ref_id: "basis_changes.0".to_string(),
        });
        let c = table.intern(DistEntry::Basis {
            mode: ChangeBasisMode::MeasurePauli,
            ref_id: "basis_changes.1".to_string(),
        });
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_eq!(table.len(), 3);
    }
}
