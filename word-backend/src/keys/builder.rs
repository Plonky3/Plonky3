//! Exact two-pass construction of per-segment key storage.

use alloc::vec;
use alloc::vec::Vec;

use p3_word::{OperandRole, Segment, Word};

use super::code::{KeyCode, ShiftSequenceCode};
use super::{
    CompiledSegment, ConstraintReference, FAMILIES, InstanceBlock, KeyCompileError,
    LayoutComponent, Reference, StoredKey,
};

/// First-pass storage for reference counts and distinct key discovery.
pub(super) struct CountingSegment {
    /// The visibility class being compiled.
    segment: Segment,
    /// One count slot after every word's future offset.
    counts: Vec<usize>,
    /// Every encountered key before sorting and deduplication.
    key_codes: Vec<KeyCode>,
}

impl CountingSegment {
    /// Creates an empty compiler for one visibility segment.
    pub(super) fn new(word_count: usize, segment: Segment) -> Result<Self, KeyCompileError> {
        // The trailing slot stores the total after prefix summation.
        let offset_count = word_count
            .checked_add(1)
            .ok_or(KeyCompileError::LayoutTooLarge {
                segment,
                component: LayoutComponent::WordOffsets,
                len: word_count,
            })?;
        Ok(Self {
            segment,
            counts: vec![0; offset_count],
            key_codes: Vec::new(),
        })
    }

    /// Records one reference during the counting pass.
    pub(super) fn count(&mut self, reference: Reference) -> Result<(), KeyCompileError> {
        // Index one slot ahead so an in-place prefix sum produces start offsets.
        self.counts[reference.word + 1] = self.counts[reference.word + 1].checked_add(1).ok_or(
            KeyCompileError::LayoutTooLarge {
                segment: self.segment,
                component: LayoutComponent::References,
                len: usize::MAX,
            },
        )?;
        self.key_codes.push(reference.key_code);
        Ok(())
    }

    /// Allocates exact second-pass storage from the collected counts.
    pub(super) fn prepare(mut self) -> Result<PreparedSegment, KeyCompileError> {
        // Convert per-word counts into CSR start offsets.
        for word in 0..self.counts.len().saturating_sub(1) {
            self.counts[word + 1] = self.counts[word + 1].checked_add(self.counts[word]).ok_or(
                KeyCompileError::LayoutTooLarge {
                    segment: self.segment,
                    component: LayoutComponent::References,
                    len: usize::MAX,
                },
            )?;
        }

        // Every stored range endpoint must fit its compact representation.
        let reference_count = self.counts.last().copied().unwrap_or(0);
        if u32::try_from(reference_count).is_err() {
            return Err(KeyCompileError::LayoutTooLarge {
                segment: self.segment,
                component: LayoutComponent::References,
                len: reference_count,
            });
        }

        // Each cursor begins at its word's exact reference span.
        let word_count = self.counts.len().saturating_sub(1);
        let cursors = self.counts[..word_count].to_vec();
        Ok(PreparedSegment {
            segment: self.segment,
            references: vec![PackedReference::default(); reference_count],
            offsets: self.counts,
            cursors,
            key_codes: self.key_codes,
        })
    }
}

/// Exact second-pass storage for one visibility segment.
pub(super) struct PreparedSegment {
    /// The visibility class being compiled.
    segment: Segment,
    /// The start offset of each word and one trailing total.
    offsets: Vec<usize>,
    /// The next vacant slot inside each word's reference span.
    cursors: Vec<usize>,
    /// The references placed in word-major order.
    references: Vec<PackedReference>,
    /// Every encountered key before sorting and deduplication.
    key_codes: Vec<KeyCode>,
}

impl PreparedSegment {
    /// Places one reference into its preallocated word span.
    pub(super) fn insert(&mut self, reference: Reference) {
        // The counting pass reserved exactly one slot for this occurrence.
        let cursor = &mut self.cursors[reference.word];
        self.references[*cursor] = PackedReference {
            key_code: reference.key_code,
            constraint: reference.constraint,
        };
        *cursor += 1;
    }

    /// Compacts the populated reference table into immutable per-word keys.
    pub(super) fn compile<W: Word>(mut self) -> Result<CompiledSegment<W>, KeyCompileError> {
        // Dense key and shift tables make repeated spellings share one index.
        self.key_codes.sort_unstable();
        self.key_codes.dedup();
        let mut sequence_codes = self
            .key_codes
            .iter()
            .copied()
            .map(KeyCode::sequence)
            .collect::<Vec<_>>();
        sequence_codes.sort_unstable();
        sequence_codes.dedup();
        let shifts = sequence_codes
            .iter()
            .copied()
            .map(ShiftSequenceCode::shifts)
            .collect::<Vec<_>>();

        // Build every word's compact key span in first-occurrence order.
        let mut keys = Vec::new();
        let mut word_keys = Vec::with_capacity(self.offsets.len().saturating_sub(1));
        let mut flat_references = Vec::with_capacity(self.references.len());
        let mut slot_of = vec![usize::MAX; self.key_codes.len()];

        for word in 0..self.offsets.len().saturating_sub(1) {
            let start = u32::try_from(keys.len()).map_err(|_| KeyCompileError::LayoutTooLarge {
                segment: self.segment,
                component: LayoutComponent::Keys,
                len: keys.len(),
            })?;
            let mut groups: Vec<(usize, Vec<ConstraintReference>)> = Vec::new();
            for reference in &self.references[self.offsets[word]..self.offsets[word + 1]] {
                let id = self
                    .key_codes
                    .binary_search(&reference.key_code)
                    .expect("the dense key table covers every counted reference");
                let slot = slot_of[id];
                if slot == usize::MAX {
                    slot_of[id] = groups.len();
                    groups.push((id, vec![reference.constraint]));
                } else {
                    groups[slot].1.push(reference.constraint);
                }
            }

            // Flatten each group into one immutable contiguous reference span.
            for (id, constraints) in groups {
                slot_of[id] = usize::MAX;
                let code = self.key_codes[id];
                let reference_start = u32::try_from(flat_references.len()).map_err(|_| {
                    KeyCompileError::LayoutTooLarge {
                        segment: self.segment,
                        component: LayoutComponent::References,
                        len: flat_references.len(),
                    }
                })?;
                flat_references.extend(constraints);
                let reference_end = u32::try_from(flat_references.len()).map_err(|_| {
                    KeyCompileError::LayoutTooLarge {
                        segment: self.segment,
                        component: LayoutComponent::References,
                        len: flat_references.len(),
                    }
                })?;
                let sequence = code.sequence();
                let shift = sequence_codes
                    .binary_search(&sequence)
                    .expect("the dense shift table covers every compact key");
                let shift = u32::try_from(shift)
                    .expect("two nine-bit shift codes fit in a 32-bit dense index");
                keys.push(StoredKey {
                    operation: code.operation(),
                    shift,
                    references: reference_start..reference_end,
                });
            }

            // The trailing endpoint closes this word's key span.
            let end = u32::try_from(keys.len()).map_err(|_| KeyCompileError::LayoutTooLarge {
                segment: self.segment,
                component: LayoutComponent::Keys,
                len: keys.len(),
            })?;
            word_keys.push(start..end);
        }

        // A flat statement is one block of one instance: slots are its words.
        let slots =
            u32::try_from(word_keys.len()).map_err(|_| KeyCompileError::LayoutTooLarge {
                segment: self.segment,
                component: LayoutComponent::WordOffsets,
                len: word_keys.len(),
            })?;
        let words = word_keys.len();
        let blocks = (slots != 0)
            .then_some(InstanceBlock {
                word_base: 0,
                slots,
                instances: 1,
                slot_base: 0,
                strides: [0; FAMILIES],
                constraint_base: [0; FAMILIES],
            })
            .into_iter()
            .collect();

        Ok(CompiledSegment {
            shifts,
            keys,
            slot_keys: word_keys,
            references: flat_references,
            blocks,
            words,
        })
    }
}

/// One compact reference before equal-key groups are flattened.
#[derive(Clone, Copy)]
struct PackedReference {
    /// The encoded operation and shift sequence.
    key_code: KeyCode,
    /// The relation consumer of the shifted word.
    constraint: ConstraintReference,
}

impl Default for PackedReference {
    fn default() -> Self {
        // Every placeholder is overwritten during the exact placement pass.
        Self {
            key_code: KeyCode::default(),
            constraint: ConstraintReference {
                operand: OperandRole::Value,
                constraint: 0,
            },
        }
    }
}
