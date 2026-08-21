// This code is a Qiskit project.
//
// (C) Copyright IBM 2025, 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! How an item's qubits group into jointly-sampled subsystems.

use std::fmt;
use thiserror::Error;

/// Why a partition could not be built.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PartitionError {
    #[error("a part must cover at least one qubit")]
    EmptyPart,
    #[error("index {index} is out of range for a partition of {num_qubits} qubits")]
    IndexOutOfRange { index: usize, num_qubits: usize },
    #[error("index {0} appears in more than one part")]
    DuplicateIndex(usize),
}

/// A partition of the first *n* integers, grouping an item's qubits into subsystems.
///
/// The parts hold **indices into the qubits of whatever carries the partition** — an
/// [`Emit`](crate::emission_circuit::Emit)'s qargs, a
/// [`Collect`](crate::emission_circuit::Collect) box's qargs, a sampling-graph
/// [`Node`](crate::sampling_graph::Node)'s own qubit list. Every one of those already names its
/// qubits, so recording them here too would be a second copy of that list, free to drift out of
/// agreement with it and with the instruction's width; indices cannot. Pair them back up with
/// [`groups`](Self::groups).
///
/// So `[[0], [2, 1]]` over qargs `[3, 5, 4]` is one subsystem on qubit 3 and one joint subsystem on
/// qubits 4 and 5. Parts need not be consecutive runs and need not be sorted: **the order is the
/// caller's**, because per-part descriptors — [`EmitPart`](crate::emission_circuit::EmitPart),
/// [`CollectPart`](crate::emission_circuit::CollectPart) — are parallel with
/// [`iter`](Self::iter), so reordering here would silently repoint them.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Partition {
    /// The parts, each a non-empty set of indices; together exactly the indices `0..n`.
    parts: Vec<Box<[usize]>>,
}

impl Partition {
    /// One subsystem per qubit: the common case, where each qubit is sampled on its own.
    pub fn singletons(n: usize) -> Self {
        Partition {
            parts: (0..n).map(|index| vec![index].into_boxed_slice()).collect(),
        }
    }

    /// All `n` qubits in one joint subsystem. Empty when `n` is zero — a part covers at least one
    /// qubit, so there is nothing to hold.
    pub fn whole(n: usize) -> Self {
        Partition {
            parts: if n == 0 {
                Vec::new()
            } else {
                vec![(0..n).collect()]
            },
        }
    }

    /// Group qubits explicitly, by index.
    ///
    /// The parts must be a genuine partition: no empty part, no index twice, and — since *n* is read
    /// off the parts themselves — no index at or beyond the total they cover, which leaves no room
    /// for a gap.
    pub fn new(parts: impl IntoIterator<Item = Box<[usize]>>) -> Result<Self, PartitionError> {
        let parts: Vec<Box<[usize]>> = parts.into_iter().collect();
        let num_qubits: usize = parts.iter().map(|part| part.len()).sum();
        let mut seen = vec![false; num_qubits];
        for part in &parts {
            if part.is_empty() {
                return Err(PartitionError::EmptyPart);
            }
            for index in part.iter() {
                let slot = seen
                    .get_mut(*index)
                    .ok_or(PartitionError::IndexOutOfRange {
                        index: *index,
                        num_qubits,
                    })?;
                if *slot {
                    return Err(PartitionError::DuplicateIndex(*index));
                }
                *slot = true;
            }
        }
        Ok(Partition { parts })
    }

    /// How many subsystems there are. Parallel with the item's per-part descriptors.
    pub fn len(&self) -> usize {
        self.parts.len()
    }

    /// Whether there are no subsystems at all, which is also to say no qubits.
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// How many qubits the partition covers: the *n* it is a partition of, and so the width of the
    /// item carrying it.
    pub fn num_qubits(&self) -> usize {
        self.parts.iter().map(|part| part.len()).sum()
    }

    /// The parts, in order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &[usize]> {
        self.parts.iter().map(|part| &**part)
    }

    /// Whether every subsystem is a single qubit, which is how most partitions come out.
    pub fn is_singletons(&self) -> bool {
        self.parts.iter().all(|part| part.len() == 1)
    }

    /// Resolve the indices against the qubits they point into, one group per subsystem.
    ///
    /// Panics if `qubits` is not [`num_qubits`](Self::num_qubits) long — a partition and the qubits it
    /// describes travel together, so a mismatch is a bug in whatever put them side by side.
    pub fn groups<T: Copy>(&self, qubits: &[T]) -> Vec<Vec<T>> {
        assert_eq!(
            qubits.len(),
            self.num_qubits(),
            "a partition of {} qubits cannot describe {} of them",
            self.num_qubits(),
            qubits.len(),
        );
        self.parts
            .iter()
            .map(|part| part.iter().map(|index| qubits[*index]).collect())
            .collect()
    }
}

impl fmt::Debug for Partition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Partition({:?})", self.parts)
    }
}

impl fmt::Display for Partition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.parts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parts(parts: &[&[usize]]) -> Partition {
        Partition::new(parts.iter().map(|part| part.to_vec().into_boxed_slice())).unwrap()
    }

    #[test]
    fn test_singletons() {
        let partition = Partition::singletons(3);
        assert_eq!(partition.len(), 3);
        assert_eq!(partition.num_qubits(), 3);
        assert!(partition.is_singletons());
        assert!(!partition.is_empty());
        assert_eq!(
            partition.iter().collect::<Vec<_>>(),
            vec![&[0][..], &[1][..], &[2][..]]
        );
    }

    #[test]
    fn test_whole() {
        let partition = Partition::whole(2);
        assert_eq!(partition.len(), 1);
        assert_eq!(partition.num_qubits(), 2);
        assert!(!partition.is_singletons());
        // A one-qubit joint subsystem is a singleton, and a zero-qubit one does not exist.
        assert!(Partition::whole(1).is_singletons());
        assert!(Partition::whole(0).is_empty());
        assert_eq!(Partition::whole(0).num_qubits(), 0);
    }

    #[test]
    fn test_groups_resolves_indices_against_qubits() {
        // The partition carries no qubits of its own, so this is the only way to recover which qubit
        // is in which subsystem — and it reads them through the item's own list, in the order the
        // part gives, not sorted.
        let partition = parts(&[&[0], &[2, 1]]);
        assert_eq!(partition.groups(&[3, 5, 4]), vec![vec![3], vec![4, 5]]);
        assert_eq!(
            Partition::singletons(3).groups(&[3, 5, 4]),
            vec![vec![3], vec![5], vec![4]]
        );
    }

    #[test]
    fn test_part_order_is_the_callers() {
        // Per-part descriptors are parallel with `iter`, so construction must not reorder or sort.
        let partition = parts(&[&[2, 1], &[0]]);
        assert_eq!(
            partition.iter().collect::<Vec<_>>(),
            vec![&[2, 1][..], &[0][..]]
        );
    }

    #[test]
    fn test_groups_rejects_a_width_mismatch() {
        let result = std::panic::catch_unwind(|| Partition::singletons(2).groups(&[0, 1, 2]));
        assert!(result.is_err());
    }

    #[test]
    fn test_new_rejects_what_is_not_a_partition() {
        let boxed = |part: &[usize]| part.to_vec().into_boxed_slice();
        assert_eq!(
            Partition::new([boxed(&[0]), boxed(&[])]).unwrap_err(),
            PartitionError::EmptyPart
        );
        assert_eq!(
            Partition::new([boxed(&[0]), boxed(&[0])]).unwrap_err(),
            PartitionError::DuplicateIndex(0)
        );
        // Two parts cover two qubits, so index 2 has nothing to point at — which is also how a gap
        // is caught, there being no room for one otherwise.
        assert_eq!(
            Partition::new([boxed(&[0]), boxed(&[2])]).unwrap_err(),
            PartitionError::IndexOutOfRange {
                index: 2,
                num_qubits: 2
            }
        );
    }

    #[test]
    fn test_display_shows_the_parts() {
        assert_eq!(parts(&[&[0], &[2, 1]]).to_string(), "[[0], [2, 1]]");
        assert_eq!(
            format!("{:?}", Partition::singletons(2)),
            "Partition([[0], [1]])"
        );
    }
}
