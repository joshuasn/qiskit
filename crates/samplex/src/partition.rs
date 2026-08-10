// This code is part of Qiskit.
//
// (C) Copyright IBM 2025, 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Grouping of qubits into non-overlapping subsystems, shared by every IR.
//!
//! Order-preserving: `from_elements` keeps its iterator's order, so building one from a hash
//! container makes downstream node and parameter indices vary run to run.

use std::fmt;

use hashbrown::HashSet;
use qiskit_util::IndexMap;
use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum BuildError {
    #[error("Cannot act on partially overlapping parts.")]
    PartialOverlap,
    #[error("A part must contain at least one element.")]
    EmptyPart,
    #[error("Could not find all indices of the other partition in this one.")]
    IndicesNotFound,
    #[error("At least one subsystem is required.")]
    EmptyUnion,
    #[error("Cannot take intersection that partly overlaps on elements {0:?}.")]
    StrictIntersectionOverlap(Vec<usize>),
    #[error("Cannot union when some partitions are partially overlapping or reordered on {0:?}.")]
    UnionPartialOverlap(Vec<usize>),
}

/// A partition of a sequence of `usize` elements into non-overlapping subsets (parts).
///
/// Parts may have different sizes.
#[derive(Clone, PartialEq, Eq)]
pub struct Partition {
    all_elements: HashSet<usize>,
    parts: IndexMap<Box<[usize]>, usize>,
}

impl Partition {
    /// Create an empty partition.
    pub fn new() -> Self {
        Partition {
            all_elements: HashSet::new(),
            parts: IndexMap::default(),
        }
    }

    /// Create a partition initialized with the given parts.
    pub fn with_parts(
        parts: impl IntoIterator<Item = Box<[usize]>>,
    ) -> Result<Self, BuildError> {
        let mut partition = Self::new();
        for part in parts {
            partition.add(part)?;
        }
        Ok(partition)
    }

    /// Construct a partition where each element is its own singleton part.
    pub fn from_elements(elements: impl IntoIterator<Item = usize>) -> Self {
        let mut partition = Self::new();
        for element in elements {
            // Single-element parts cannot partially overlap, so unwrap is safe.
            partition.add(Box::new([element])).unwrap();
        }
        partition
    }

    pub fn all_elements(&self) -> &HashSet<usize> {
        &self.all_elements
    }

    pub fn len(&self) -> usize {
        self.parts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// Whether all parts have the same length (or the partition is empty).
    pub fn is_uniform(&self) -> bool {
        let mut iter = self.parts.keys();
        let Some(first) = iter.next() else {
            return true;
        };
        let size = first.len();
        iter.all(|p| p.len() == size)
    }

    /// Check if the given part is in this partition.
    pub fn contains(&self, part: &[usize]) -> bool {
        self.parts.contains_key(part)
    }

    /// Add a new part to this partition.
    ///
    /// Idempotent if the part already exists. Errors on empty parts or partial overlap.
    pub fn add(&mut self, part: Box<[usize]>) -> Result<(), BuildError> {
        if part.is_empty() {
            return Err(BuildError::EmptyPart);
        }
        if self.parts.contains_key(&*part) {
            return Ok(());
        }
        if part.iter().any(|e| self.all_elements.contains(e)) {
            return Err(BuildError::PartialOverlap);
        }
        self.all_elements.extend(part.iter().copied());
        let idx = self.parts.len();
        self.parts.insert(part, idx);
        Ok(())
    }

    /// Whether any of the given elements overlap with this partition's elements.
    pub fn overlaps_with(&self, elements: impl IntoIterator<Item = usize>) -> bool {
        elements.into_iter().any(|e| self.all_elements.contains(&e))
    }

    /// Get the indices of the parts of `other` in this partition.
    pub fn get_indices(&self, other: &Partition) -> Result<Vec<usize>, BuildError> {
        other
            .parts
            .keys()
            .map(|part| {
                self.parts
                    .get(&**part)
                    .copied()
                    .ok_or(BuildError::IndicesNotFound)
            })
            .collect()
    }

    /// Restrict to those parts fully contained in the required set.
    pub fn restrict(&self, required: &HashSet<usize>) -> Self {
        let mut result = Self::new();
        for part in self.parts.keys() {
            if part.iter().all(|e| required.contains(e)) {
                result.all_elements.extend(part.iter().copied());
                let idx = result.parts.len();
                result.parts.insert(part.clone(), idx);
            }
        }
        result
    }

    /// Return a new partition keeping only parts disjoint from `subtracted`.
    pub fn difference(&self, subtracted: &HashSet<usize>) -> Self {
        let mut result = Self::new();
        for part in self.parts.keys() {
            if part.iter().all(|e| !subtracted.contains(e)) {
                result.all_elements.extend(part.iter().copied());
                let idx = result.parts.len();
                result.parts.insert(part.clone(), idx);
            }
        }
        result
    }

    /// Return a new partition that is the intersection with the other.
    ///
    /// The order of this partition is maintained. If `strict` is true, errors on partial overlaps.
    pub fn intersection(&self, other: &Partition, strict: bool) -> Result<Self, BuildError> {
        let mut result = Self::new();
        for part in self.parts.keys() {
            if strict && other.overlaps_with(part.iter().copied()) && !other.contains(part) {
                let overlap: Vec<usize> = part
                    .iter()
                    .filter(|e| other.all_elements.contains(*e))
                    .copied()
                    .collect();
                return Err(BuildError::StrictIntersectionOverlap(overlap));
            }
            if other.contains(part) {
                result.all_elements.extend(part.iter().copied());
                let idx = result.parts.len();
                result.parts.insert(part.clone(), idx);
            }
        }
        Ok(result)
    }

    /// Take the union of one or more partitions.
    ///
    /// Order is maintained with earlier partitions taking precedence.
    pub fn union(partitions: &[&Partition]) -> Result<Self, BuildError> {
        if partitions.is_empty() {
            return Err(BuildError::EmptyUnion);
        }
        let mut result = partitions[0].clone();

        for partition in &partitions[1..] {
            for part in partition.parts.keys() {
                if !result.parts.contains_key(&**part) {
                    if part.iter().any(|e| result.all_elements.contains(e)) {
                        let mut overlap: Vec<usize> = part
                            .iter()
                            .filter(|e| result.all_elements.contains(*e))
                            .copied()
                            .collect();
                        overlap.sort_unstable();
                        return Err(BuildError::UnionPartialOverlap(overlap));
                    }
                    result.all_elements.extend(part.iter().copied());
                    let idx = result.parts.len();
                    result.parts.insert(part.clone(), idx);
                }
            }
        }
        Ok(result)
    }

    /// Iterate over parts in insertion order.
    pub fn iter(&self) -> PartitionIter<'_> {
        PartitionIter {
            inner: self.parts.keys(),
        }
    }
}

impl Default for Partition {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Partition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<&[usize]> = self.parts.keys().map(|p| &**p).collect();
        write!(f, "Partition(parts={:?})", parts)
    }
}

impl fmt::Display for Partition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

pub struct PartitionIter<'a> {
    inner: indexmap::map::Keys<'a, Box<[usize]>, usize>,
}

impl<'a> Iterator for PartitionIter<'a> {
    type Item = &'a [usize];

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|k| &**k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl ExactSizeIterator for PartitionIter<'_> {}

impl<'a> IntoIterator for &'a Partition {
    type Item = &'a [usize];
    type IntoIter = PartitionIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxed(slice: &[usize]) -> Box<[usize]> {
        slice.into()
    }

    #[test]
    fn test_construction() {
        let partition = Partition::new();
        assert!(partition.all_elements().is_empty());
        assert!(partition.is_empty());

        let partition = Partition::with_parts(vec![boxed(&[0]), boxed(&[2])]).unwrap();
        assert_eq!(partition.all_elements(), &HashSet::from([0, 2]));
    }

    #[test]
    fn test_contains() {
        let partition = Partition::with_parts(vec![boxed(&[0, 1]), boxed(&[2, 3])]).unwrap();
        assert!(partition.contains(&[0, 1]));
        assert!(partition.contains(&[2, 3]));
        assert!(!partition.contains(&[1, 2]));
        assert!(!partition.contains(&[1, 0]));
    }

    #[test]
    fn test_insertion_ordered() {
        let mut partition =
            Partition::with_parts(vec![boxed(&[6, 7]), boxed(&[2, 3])]).unwrap();
        partition.add(boxed(&[1, 0])).unwrap();
        let parts: Vec<&[usize]> = partition.iter().collect();
        assert_eq!(parts, vec![&[6, 7][..], &[2, 3][..], &[1, 0][..]]);
    }

    #[test]
    fn test_iteration() {
        let partition = Partition::with_parts(vec![boxed(&[0, 1]), boxed(&[2, 3])]).unwrap();
        let parts: HashSet<&[usize]> = partition.iter().collect();
        assert!(parts.contains(&[0, 1][..]));
        assert!(parts.contains(&[2, 3][..]));
        assert_eq!(parts.len(), 2);
    }

    #[test]
    fn test_len() {
        let partition = Partition::with_parts(vec![boxed(&[0, 1]), boxed(&[2, 3])]).unwrap();
        assert_eq!(partition.len(), 2);
        assert!(!partition.is_empty());

        let empty = Partition::new();
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_overlaps_with() {
        let partition = Partition::with_parts(vec![boxed(&[0, 1]), boxed(&[2, 3])]).unwrap();
        assert!(partition.overlaps_with([1]));
        assert!(partition.overlaps_with([1, 18]));
        assert!(partition.overlaps_with([1, 0]));

        assert!(!partition.overlaps_with([]));
        assert!(!partition.overlaps_with([5, 8]));
    }

    #[test]
    fn test_add() {
        let mut partition =
            Partition::with_parts(vec![boxed(&[0, 1]), boxed(&[2, 3])]).unwrap();
        assert_eq!(partition.iter().collect::<HashSet<&[usize]>>().len(), 2);

        // Idempotent re-add
        partition.add(boxed(&[0, 1])).unwrap();
        assert_eq!(partition.len(), 2);

        // New part (different size is now allowed)
        partition.add(boxed(&[4, 5, 6])).unwrap();
        assert_eq!(partition.len(), 3);
        assert!(partition.contains(&[4, 5, 6]));
    }

    #[test]
    fn test_add_empty_part() {
        let mut partition = Partition::new();
        let err = partition.add(boxed(&[])).unwrap_err();
        assert!(matches!(err, BuildError::EmptyPart));
        assert!(partition.is_empty());
    }

    #[test]
    fn test_add_partial_overlap() {
        let mut partition =
            Partition::with_parts(vec![boxed(&[0, 1]), boxed(&[2, 3])]).unwrap();
        let err = partition.add(boxed(&[3, 4])).unwrap_err();
        assert!(matches!(err, BuildError::PartialOverlap));
        assert_eq!(partition.len(), 2);
    }

    #[test]
    fn test_mixed_size_parts() {
        let partition = Partition::with_parts(vec![
            boxed(&[0, 1]),
            boxed(&[2]),
            boxed(&[3, 4, 5]),
        ])
        .unwrap();
        assert_eq!(partition.len(), 3);
        assert_eq!(partition.all_elements().len(), 6);
        assert!(partition.contains(&[0, 1]));
        assert!(partition.contains(&[2]));
        assert!(partition.contains(&[3, 4, 5]));
        assert!(!partition.is_uniform());
    }

    #[test]
    fn test_is_uniform() {
        let empty = Partition::new();
        assert!(empty.is_uniform());

        let uniform = Partition::with_parts(vec![boxed(&[0, 1]), boxed(&[2, 3])]).unwrap();
        assert!(uniform.is_uniform());

        let mixed = Partition::with_parts(vec![boxed(&[0, 1]), boxed(&[2])]).unwrap();
        assert!(!mixed.is_uniform());
    }

    #[test]
    fn test_get_indices() {
        let partition =
            Partition::with_parts(vec![boxed(&[0, 1]), boxed(&[2, 3]), boxed(&[4, 5])]).unwrap();

        // Identity
        assert_eq!(partition.get_indices(&partition).unwrap(), vec![0, 1, 2]);

        // Reordered
        let other =
            Partition::with_parts(vec![boxed(&[2, 3]), boxed(&[0, 1]), boxed(&[4, 5])]).unwrap();
        assert_eq!(partition.get_indices(&other).unwrap(), vec![1, 0, 2]);

        // Subset
        let other = Partition::with_parts(vec![boxed(&[4, 5]), boxed(&[0, 1])]).unwrap();
        assert_eq!(partition.get_indices(&other).unwrap(), vec![2, 0]);
    }

    #[test]
    fn test_get_indices_not_found() {
        let partition =
            Partition::with_parts(vec![boxed(&[0, 1]), boxed(&[2, 3]), boxed(&[4, 5])]).unwrap();

        let other = Partition::with_parts(vec![boxed(&[1, 2])]).unwrap();
        assert!(matches!(
            partition.get_indices(&other).unwrap_err(),
            BuildError::IndicesNotFound
        ));

        let other = Partition::with_parts(vec![boxed(&[8, 9])]).unwrap();
        assert!(matches!(
            partition.get_indices(&other).unwrap_err(),
            BuildError::IndicesNotFound
        ));
    }

    #[test]
    fn test_restrict() {
        let partition =
            Partition::with_parts(vec![boxed(&[0, 1]), boxed(&[2, 3]), boxed(&[4, 5])]).unwrap();

        let required = HashSet::from([0, 1, 4, 5, 99]);
        let restricted = partition.restrict(&required);
        assert_eq!(restricted.len(), 2);
        assert!(restricted.contains(&[0, 1]));
        assert!(restricted.contains(&[4, 5]));
        assert!(!restricted.contains(&[2, 3]));
    }

    #[test]
    fn test_difference() {
        let partition =
            Partition::with_parts(vec![boxed(&[0, 1]), boxed(&[2, 3]), boxed(&[4, 5])]).unwrap();

        let subtracted = HashSet::from([2, 3]);
        let diff = partition.difference(&subtracted);
        assert_eq!(diff.len(), 2);
        assert!(diff.contains(&[0, 1]));
        assert!(diff.contains(&[4, 5]));
        assert!(!diff.contains(&[2, 3]));
    }

    #[test]
    fn test_intersection() {
        let a =
            Partition::with_parts(vec![boxed(&[0, 1]), boxed(&[2, 3]), boxed(&[4, 5])]).unwrap();
        let b = Partition::with_parts(vec![boxed(&[2, 3]), boxed(&[4, 5])]).unwrap();

        let result = a.intersection(&b, false).unwrap();
        assert_eq!(result.len(), 2);
        assert!(result.contains(&[2, 3]));
        assert!(result.contains(&[4, 5]));

        // Order preserved from self (a)
        let parts: Vec<&[usize]> = result.iter().collect();
        assert_eq!(parts, vec![&[2, 3][..], &[4, 5][..]]);
    }

    #[test]
    fn test_intersection_mixed_sizes() {
        let a = Partition::with_parts(vec![boxed(&[0, 1]), boxed(&[2])]).unwrap();
        let b = Partition::with_parts(vec![boxed(&[2]), boxed(&[3, 4, 5])]).unwrap();

        let result = a.intersection(&b, false).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result.contains(&[2]));
    }

    #[test]
    fn test_intersection_strict() {
        let a = Partition::with_parts(vec![boxed(&[0, 1]), boxed(&[2, 3])]).unwrap();
        // other contains element 1 but not part [0, 1]
        let b = Partition::with_parts(vec![boxed(&[1, 4])]).unwrap();

        // Non-strict: no error, just empty result
        let result = a.intersection(&b, false).unwrap();
        assert!(result.is_empty());

        // Strict: errors on partial overlap
        assert!(matches!(
            a.intersection(&b, true).unwrap_err(),
            BuildError::StrictIntersectionOverlap(_)
        ));
    }

    #[test]
    fn test_from_elements() {
        let partition = Partition::from_elements([2, 3, 5, 7, 11]);
        assert!(partition.is_uniform());
        let parts: Vec<&[usize]> = partition.iter().collect();
        assert_eq!(
            parts,
            vec![&[2][..], &[3][..], &[5][..], &[7][..], &[11][..]]
        );
    }

    #[test]
    fn test_union() {
        let p0 = Partition::with_parts(vec![boxed(&[0, 1]), boxed(&[2, 3])]).unwrap();
        let p1 = Partition::with_parts(vec![boxed(&[2, 3]), boxed(&[4, 5])]).unwrap();
        let p2 = Partition::with_parts(vec![boxed(&[6, 7])]).unwrap();

        let union = Partition::union(&[&p0, &p1, &p2]).unwrap();
        assert_eq!(union.len(), 4);
        assert!(union.contains(&[0, 1]));
        assert!(union.contains(&[2, 3]));
        assert!(union.contains(&[4, 5]));
        assert!(union.contains(&[6, 7]));
    }

    #[test]
    fn test_union_mixed_sizes() {
        let p0 = Partition::with_parts(vec![boxed(&[0, 1])]).unwrap();
        let p1 = Partition::with_parts(vec![boxed(&[2, 3, 4])]).unwrap();
        let p2 = Partition::with_parts(vec![boxed(&[5])]).unwrap();

        let union = Partition::union(&[&p0, &p1, &p2]).unwrap();
        assert_eq!(union.len(), 3);
        assert!(union.contains(&[0, 1]));
        assert!(union.contains(&[2, 3, 4]));
        assert!(union.contains(&[5]));
        assert!(!union.is_uniform());
    }

    #[test]
    fn test_union_empty() {
        assert!(matches!(
            Partition::union(&[]).unwrap_err(),
            BuildError::EmptyUnion
        ));
    }

    #[test]
    fn test_union_partial_overlap() {
        let p0 = Partition::with_parts(vec![boxed(&[0, 1]), boxed(&[2, 3])]).unwrap();
        let p1 = Partition::with_parts(vec![boxed(&[4, 5]), boxed(&[1, 10])]).unwrap();
        assert!(matches!(
            Partition::union(&[&p0, &p1]).unwrap_err(),
            BuildError::UnionPartialOverlap(_)
        ));
    }

    #[test]
    fn test_union_reordered() {
        let p0 = Partition::with_parts(vec![boxed(&[0, 1]), boxed(&[2, 3])]).unwrap();
        let p1 = Partition::with_parts(vec![boxed(&[4, 5]), boxed(&[1, 0])]).unwrap();
        assert!(matches!(
            Partition::union(&[&p0, &p1]).unwrap_err(),
            BuildError::UnionPartialOverlap(_)
        ));
    }

    #[test]
    fn test_clone() {
        let mut partition =
            Partition::with_parts(vec![boxed(&[0, 1]), boxed(&[2, 3])]).unwrap();
        let copy = partition.clone();

        assert_eq!(
            partition.iter().collect::<Vec<_>>(),
            copy.iter().collect::<Vec<_>>()
        );

        partition.add(boxed(&[4, 5])).unwrap();
        assert!(partition.contains(&[4, 5]));
        assert!(!copy.contains(&[4, 5]));
    }

    #[test]
    fn test_eq() {
        let a = Partition::with_parts(vec![boxed(&[0, 1]), boxed(&[2, 3])]).unwrap();
        let b = Partition::with_parts(vec![boxed(&[0, 1]), boxed(&[2, 3])]).unwrap();
        let c = Partition::with_parts(vec![boxed(&[2, 3]), boxed(&[0, 1])]).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
