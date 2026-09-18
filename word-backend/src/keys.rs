use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

use p3_word::{ConstraintKind, ConstraintSystem, Operand, Segment, Shift, ShiftKind, Word};
use thiserror::Error;

const SHIFT_BITS: u32 = 9;
const SEQUENCE_BITS: u32 = 2 * SHIFT_BITS;
const SEQUENCE_MASK: u32 = (1 << SEQUENCE_BITS) - 1;

/// A compact index into one operation family.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ConstraintReference {
    operand: u8,
    constraint: u32,
}

impl ConstraintReference {
    /// Returns the operand position within the operation.
    #[inline]
    pub const fn operand(self) -> u8 {
        self.operand
    }

    /// Returns the constraint position within the operation family.
    #[inline]
    pub const fn constraint(self) -> u32 {
        self.constraint
    }
}

/// One word's references under a fixed operation and shift sequence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledKey {
    operation: ConstraintKind,
    shift: u16,
    references: Range<u32>,
}

impl CompiledKey {
    /// Returns the relation family using the word.
    #[inline]
    pub const fn operation(&self) -> ConstraintKind {
        self.operation
    }
}

/// Metadata compiled independently for one visibility segment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledSegment<W: Word> {
    shifts: Vec<[Shift<W>; 2]>,
    keys: Vec<CompiledKey>,
    word_keys: Vec<Range<u32>>,
    references: Vec<ConstraintReference>,
}

impl<W: Word> CompiledSegment<W> {
    /// Returns the number of indexed words.
    #[inline]
    pub const fn len(&self) -> usize {
        self.word_keys.len()
    }

    /// Returns whether the segment has no words.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.word_keys.is_empty()
    }

    /// Returns the keys attached to a word.
    pub fn keys(&self, word: usize) -> Option<&[CompiledKey]> {
        let range = self.word_keys.get(word)?;
        self.keys.get(range.start as usize..range.end as usize)
    }

    /// Returns a key's canonical inner and outer shifts.
    pub fn shifts(&self, key: &CompiledKey) -> Option<[Shift<W>; 2]> {
        self.shifts.get(key.shift as usize).copied()
    }

    /// Returns a key's references in operand-major order.
    pub fn references(&self, key: &CompiledKey) -> Option<&[ConstraintReference]> {
        self.references
            .get(key.references.start as usize..key.references.end as usize)
    }
}

/// Public and committed shift metadata, kept in separate index spaces.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledKeyLayout<W: Word> {
    public: CompiledSegment<W>,
    witness: CompiledSegment<W>,
}

impl<W: Word> CompiledKeyLayout<W> {
    /// Compiles every shifted reference in deterministic reduction order.
    pub fn new(system: &ConstraintSystem<W>) -> Result<Self, KeyCompileError> {
        check_constraint_counts(system)?;

        let mut public_counts = word_offsets(system.public_len(), Segment::Public)?;
        let mut witness_counts = word_offsets(system.witness_len(), Segment::Witness)?;
        let mut public_codes = Vec::new();
        let mut witness_codes = Vec::new();

        for_each_reference(system, |reference| {
            let (counts, codes) = match reference.segment {
                Segment::Public => (&mut public_counts, &mut public_codes),
                Segment::Witness => (&mut witness_counts, &mut witness_codes),
            };
            counts[reference.word + 1] = counts[reference.word + 1].checked_add(1).ok_or(
                KeyCompileError::LayoutTooLarge {
                    segment: reference.segment,
                    component: LayoutComponent::References,
                    len: usize::MAX,
                },
            )?;
            codes.push(reference.key_code);
            Ok(())
        })?;

        prefix_sum(&mut public_counts, Segment::Public)?;
        prefix_sum(&mut witness_counts, Segment::Witness)?;

        let mut public_references =
            vec![PackedReference::default(); public_counts.last().copied().unwrap_or(0)];
        let mut witness_references =
            vec![PackedReference::default(); witness_counts.last().copied().unwrap_or(0)];
        let mut public_cursors = public_counts[..system.public_len()].to_vec();
        let mut witness_cursors = witness_counts[..system.witness_len()].to_vec();

        for_each_reference(system, |reference| {
            let (references, cursors) = match reference.segment {
                Segment::Public => (&mut public_references, &mut public_cursors),
                Segment::Witness => (&mut witness_references, &mut witness_cursors),
            };
            let cursor = &mut cursors[reference.word];
            references[*cursor] = PackedReference {
                key_code: reference.key_code,
                constraint: reference.constraint,
            };
            *cursor += 1;
            Ok(())
        })?;

        Ok(Self {
            public: build_segment(
                Segment::Public,
                &public_counts,
                &public_references,
                public_codes,
            )?,
            witness: build_segment(
                Segment::Witness,
                &witness_counts,
                &witness_references,
                witness_codes,
            )?,
        })
    }

    /// Returns metadata for verifier-known words.
    #[inline]
    pub const fn public(&self) -> &CompiledSegment<W> {
        &self.public
    }

    /// Returns metadata for committed words.
    #[inline]
    pub const fn witness(&self) -> &CompiledSegment<W> {
        &self.witness
    }
}

/// A bounded part of the compact key layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayoutComponent {
    /// Constraint references.
    References,
    /// Per-word keys.
    Keys,
    /// Distinct shift sequences.
    Shifts,
    /// Word-offset table entries.
    WordOffsets,
}

/// A constraint system too large for the compact key representation.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum KeyCompileError {
    /// A family cannot be addressed by a 32-bit constraint index.
    #[error("{kind:?} has {len} constraints, exceeding u32::MAX")]
    TooManyConstraints {
        /// The oversized family.
        kind: ConstraintKind,
        /// The rejected length.
        len: usize,
    },
    /// A segment component cannot be addressed by its compact index.
    #[error("{segment:?} {component:?} has unsupported length {len}")]
    LayoutTooLarge {
        /// The oversized segment.
        segment: Segment,
        /// The oversized component.
        component: LayoutComponent,
        /// The rejected length.
        len: usize,
    },
}

#[derive(Clone, Copy, Default)]
struct PackedReference {
    key_code: u32,
    constraint: ConstraintReference,
}

#[derive(Clone, Copy)]
struct Reference {
    segment: Segment,
    word: usize,
    key_code: u32,
    constraint: ConstraintReference,
}

fn check_constraint_counts<W: Word>(system: &ConstraintSystem<W>) -> Result<(), KeyCompileError> {
    for (kind, len) in [
        (ConstraintKind::Zero, system.zero_constraints().len()),
        (ConstraintKind::And, system.and_constraints().len()),
        (
            ConstraintKind::IntegerMul,
            system.integer_mul_constraints().len(),
        ),
    ] {
        if u32::try_from(len).is_err() {
            return Err(KeyCompileError::TooManyConstraints { kind, len });
        }
    }
    Ok(())
}

fn word_offsets(len: usize, segment: Segment) -> Result<Vec<usize>, KeyCompileError> {
    let len = len.checked_add(1).ok_or(KeyCompileError::LayoutTooLarge {
        segment,
        component: LayoutComponent::WordOffsets,
        len,
    })?;
    Ok(vec![0; len])
}

fn for_each_reference<W: Word>(
    system: &ConstraintSystem<W>,
    mut visit: impl FnMut(Reference) -> Result<(), KeyCompileError>,
) -> Result<(), KeyCompileError> {
    for (constraint, relation) in system.zero_constraints().iter().enumerate() {
        visit_operand(
            ConstraintKind::Zero,
            0,
            u32::try_from(constraint).expect("constraint counts were bounded before compilation"),
            relation.value(),
            &mut visit,
        )?;
    }
    for operand in 0..3 {
        for (constraint, relation) in system.and_constraints().iter().enumerate() {
            let value = match operand {
                0 => relation.left(),
                1 => relation.right(),
                _ => relation.output(),
            };
            visit_operand(
                ConstraintKind::And,
                operand,
                u32::try_from(constraint)
                    .expect("constraint counts were bounded before compilation"),
                value,
                &mut visit,
            )?;
        }
    }
    for operand in 0..4 {
        for (constraint, relation) in system.integer_mul_constraints().iter().enumerate() {
            let value = match operand {
                0 => relation.left(),
                1 => relation.right(),
                2 => relation.low(),
                _ => relation.high(),
            };
            visit_operand(
                ConstraintKind::IntegerMul,
                operand,
                u32::try_from(constraint)
                    .expect("constraint counts were bounded before compilation"),
                value,
                &mut visit,
            )?;
        }
    }
    Ok(())
}

fn visit_operand<W: Word>(
    operation: ConstraintKind,
    operand: u8,
    constraint: u32,
    value: &Operand<W>,
    visit: &mut impl FnMut(Reference) -> Result<(), KeyCompileError>,
) -> Result<(), KeyCompileError> {
    for term in value.terms() {
        let index = term.index();
        visit(Reference {
            segment: index.segment(),
            word: index.position() as usize,
            key_code: key_code(operation, [term.inner(), term.outer()]),
            constraint: ConstraintReference {
                operand,
                constraint,
            },
        })?;
    }
    Ok(())
}

fn prefix_sum(offsets: &mut [usize], segment: Segment) -> Result<(), KeyCompileError> {
    for word in 0..offsets.len().saturating_sub(1) {
        offsets[word + 1] = offsets[word + 1].checked_add(offsets[word]).ok_or(
            KeyCompileError::LayoutTooLarge {
                segment,
                component: LayoutComponent::References,
                len: usize::MAX,
            },
        )?;
    }
    let len = offsets.last().copied().unwrap_or(0);
    if u32::try_from(len).is_err() {
        return Err(KeyCompileError::LayoutTooLarge {
            segment,
            component: LayoutComponent::References,
            len,
        });
    }
    Ok(())
}

fn build_segment<W: Word>(
    segment: Segment,
    offsets: &[usize],
    references: &[PackedReference],
    mut key_codes: Vec<u32>,
) -> Result<CompiledSegment<W>, KeyCompileError> {
    key_codes.sort_unstable();
    key_codes.dedup();
    let mut sequence_codes = key_codes
        .iter()
        .map(|code| code & SEQUENCE_MASK)
        .collect::<Vec<_>>();
    sequence_codes.sort_unstable();
    sequence_codes.dedup();
    if sequence_codes.len() > u16::MAX as usize + 1 {
        return Err(KeyCompileError::LayoutTooLarge {
            segment,
            component: LayoutComponent::Shifts,
            len: sequence_codes.len(),
        });
    }
    let shifts = sequence_codes
        .iter()
        .copied()
        .map(decode_sequence)
        .collect::<Vec<_>>();

    let mut keys = Vec::new();
    let mut word_keys = Vec::with_capacity(offsets.len().saturating_sub(1));
    let mut flat_references = Vec::with_capacity(references.len());
    let mut slot_of = vec![usize::MAX; key_codes.len()];

    for word in 0..offsets.len().saturating_sub(1) {
        let start = u32::try_from(keys.len()).map_err(|_| KeyCompileError::LayoutTooLarge {
            segment,
            component: LayoutComponent::Keys,
            len: keys.len(),
        })?;
        let mut groups: Vec<(usize, Vec<ConstraintReference>)> = Vec::new();
        for reference in &references[offsets[word]..offsets[word + 1]] {
            let id = key_codes
                .binary_search(&reference.key_code)
                .expect("the dense key list covers every reference");
            let slot = slot_of[id];
            if slot == usize::MAX {
                slot_of[id] = groups.len();
                groups.push((id, vec![reference.constraint]));
            } else {
                groups[slot].1.push(reference.constraint);
            }
        }

        for (id, constraints) in groups {
            slot_of[id] = usize::MAX;
            let code = key_codes[id];
            let reference_start = u32::try_from(flat_references.len()).map_err(|_| {
                KeyCompileError::LayoutTooLarge {
                    segment,
                    component: LayoutComponent::References,
                    len: flat_references.len(),
                }
            })?;
            flat_references.extend(constraints);
            let reference_end = u32::try_from(flat_references.len()).map_err(|_| {
                KeyCompileError::LayoutTooLarge {
                    segment,
                    component: LayoutComponent::References,
                    len: flat_references.len(),
                }
            })?;
            let sequence = code & SEQUENCE_MASK;
            let shift = sequence_codes
                .binary_search(&sequence)
                .expect("the dense shift list covers every key");
            let shift = u16::try_from(shift)
                .expect("the dense shift count was bounded before key construction");
            keys.push(CompiledKey {
                operation: decode_operation(code),
                shift,
                references: reference_start..reference_end,
            });
        }

        let end = u32::try_from(keys.len()).map_err(|_| KeyCompileError::LayoutTooLarge {
            segment,
            component: LayoutComponent::Keys,
            len: keys.len(),
        })?;
        word_keys.push(start..end);
    }

    Ok(CompiledSegment {
        shifts,
        keys,
        word_keys,
        references: flat_references,
    })
}

#[inline]
fn key_code<W: Word>(operation: ConstraintKind, shifts: [Shift<W>; 2]) -> u32 {
    operation_code(operation) << SEQUENCE_BITS
        | u32::from(shift_code(shifts[1])) << SHIFT_BITS
        | u32::from(shift_code(shifts[0]))
}

#[inline]
const fn operation_code(operation: ConstraintKind) -> u32 {
    match operation {
        ConstraintKind::Zero => 0,
        ConstraintKind::And => 1,
        ConstraintKind::IntegerMul => 2,
    }
}

#[inline]
const fn decode_operation(code: u32) -> ConstraintKind {
    match code >> SEQUENCE_BITS {
        0 => ConstraintKind::Zero,
        1 => ConstraintKind::And,
        2 => ConstraintKind::IntegerMul,
        _ => unreachable!(),
    }
}

#[inline]
fn shift_code<W: Word>(shift: Shift<W>) -> u16 {
    (kind_code(shift.kind()) << 6) | u16::from(shift.amount())
}

#[inline]
const fn kind_code(kind: ShiftKind) -> u16 {
    match kind {
        ShiftKind::LogicalLeft => 0,
        ShiftKind::LogicalRight => 1,
        ShiftKind::ArithmeticRight => 2,
        ShiftKind::RotateRight => 3,
        ShiftKind::Lane32LogicalLeft => 4,
        ShiftKind::Lane32LogicalRight => 5,
        ShiftKind::Lane32ArithmeticRight => 6,
        ShiftKind::Lane32RotateRight => 7,
    }
}

fn decode_sequence<W: Word>(code: u32) -> [Shift<W>; 2] {
    [decode_shift(code), decode_shift(code >> SHIFT_BITS)]
}

fn decode_shift<W: Word>(code: u32) -> Shift<W> {
    let code = code & ((1 << SHIFT_BITS) - 1);
    let kind = match code >> 6 {
        0 => ShiftKind::LogicalLeft,
        1 => ShiftKind::LogicalRight,
        2 => ShiftKind::ArithmeticRight,
        3 => ShiftKind::RotateRight,
        4 => ShiftKind::Lane32LogicalLeft,
        5 => ShiftKind::Lane32LogicalRight,
        6 => ShiftKind::Lane32ArithmeticRight,
        7 => ShiftKind::Lane32RotateRight,
        _ => unreachable!(),
    };
    Shift::new(kind, (code & 0x3f) as usize).expect("compiled shifts originated in checked terms")
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_word::{
        AndConstraint, IntegerMulConstraint, Operand, ShiftedValue, ValueIndex, Word32, Word64,
        ZeroConstraint,
    };

    use super::*;

    #[test]
    fn compilation_keeps_segments_and_groups_equal_keys() {
        let public = ValueIndex::public(0).unwrap();
        let witness = ValueIndex::witness(1).unwrap();
        let rotate = Shift::new(ShiftKind::RotateRight, 11).unwrap();
        let public_term = ShiftedValue::<Word64>::single(public, rotate);
        let witness_term = ShiftedValue::<Word64>::plain(witness);
        let left = Operand::new(vec![witness_term, public_term, witness_term]);
        let right = Operand::single(public_term);
        let output = Operand::single(witness_term);
        let system = ConstraintSystem::new(
            1,
            2,
            vec![ZeroConstraint::new(left.clone())],
            vec![AndConstraint::new(left, right, output)],
            vec![],
        )
        .unwrap();

        let layout = CompiledKeyLayout::new(&system).unwrap();
        let public_keys = layout.public().keys(0).unwrap();
        assert_eq!(public_keys.len(), 2);
        assert_eq!(
            layout.public().shifts(&public_keys[0]),
            Some([rotate, Shift::identity()])
        );
        assert_eq!(
            layout.public().references(&public_keys[1]).unwrap(),
            &[
                ConstraintReference {
                    operand: 0,
                    constraint: 0,
                },
                ConstraintReference {
                    operand: 1,
                    constraint: 0,
                },
            ]
        );

        assert!(layout.witness().keys(0).unwrap().is_empty());
        let witness_keys = layout.witness().keys(1).unwrap();
        assert_eq!(witness_keys.len(), 2);
        assert_eq!(
            layout.witness().references(&witness_keys[0]).unwrap(),
            &[
                ConstraintReference {
                    operand: 0,
                    constraint: 0,
                },
                ConstraintReference {
                    operand: 0,
                    constraint: 0,
                },
            ]
        );
    }

    #[test]
    fn operation_and_operand_order_matches_the_reduction_order() {
        let index = ValueIndex::witness(0).unwrap();
        let term = ShiftedValue::<Word32>::plain(index);
        let value = Operand::single(term);
        let system = ConstraintSystem::new(
            0,
            1,
            vec![ZeroConstraint::new(value.clone())],
            vec![AndConstraint::new(
                value.clone(),
                value.clone(),
                value.clone(),
            )],
            vec![IntegerMulConstraint::new(
                value.clone(),
                value.clone(),
                value.clone(),
                value,
            )],
        )
        .unwrap();

        let layout = CompiledKeyLayout::new(&system).unwrap();
        let keys = layout.witness().keys(0).unwrap();
        assert_eq!(
            keys.iter().map(CompiledKey::operation).collect::<Vec<_>>(),
            [
                ConstraintKind::Zero,
                ConstraintKind::And,
                ConstraintKind::IntegerMul,
            ]
        );
        assert_eq!(
            layout.witness().references(&keys[1]).unwrap(),
            &[
                ConstraintReference {
                    operand: 0,
                    constraint: 0,
                },
                ConstraintReference {
                    operand: 1,
                    constraint: 0,
                },
                ConstraintReference {
                    operand: 2,
                    constraint: 0,
                },
            ]
        );
    }

    #[test]
    fn dense_shifts_keep_word64_lane_semantics_distinct() {
        let index = ValueIndex::witness(0).unwrap();
        let full = Shift::new(ShiftKind::LogicalRight, 5).unwrap();
        let lanes = Shift::new(ShiftKind::Lane32LogicalRight, 5).unwrap();
        let system = ConstraintSystem::new(
            0,
            1,
            vec![ZeroConstraint::new(Operand::new(vec![
                ShiftedValue::<Word64>::single(index, full),
                ShiftedValue::<Word64>::single(index, lanes),
            ]))],
            vec![],
            vec![],
        )
        .unwrap();

        let layout = CompiledKeyLayout::new(&system).unwrap();
        let keys = layout.witness().keys(0).unwrap();
        assert_eq!(keys.len(), 2);
        assert_ne!(
            layout.witness().shifts(&keys[0]),
            layout.witness().shifts(&keys[1])
        );
    }

    #[test]
    fn two_shift_sequences_round_trip_through_the_dense_encoding() {
        let index = ValueIndex::witness(0).unwrap();
        let inner = Shift::new(ShiftKind::LogicalLeft, 9).unwrap();
        let outer = Shift::new(ShiftKind::RotateRight, 17).unwrap();
        let pair = ShiftedValue::<Word64>::pair(index, inner, outer).unwrap();
        let system = ConstraintSystem::new(
            0,
            1,
            vec![ZeroConstraint::new(Operand::single(pair))],
            vec![],
            vec![],
        )
        .unwrap();

        let layout = CompiledKeyLayout::new(&system).unwrap();
        let key = &layout.witness().keys(0).unwrap()[0];
        assert_eq!(layout.witness().shifts(key), Some([inner, outer]));
    }

    #[test]
    fn word_offset_length_overflow_is_rejected() {
        assert_eq!(
            word_offsets(usize::MAX, Segment::Witness),
            Err(KeyCompileError::LayoutTooLarge {
                segment: Segment::Witness,
                component: LayoutComponent::WordOffsets,
                len: usize::MAX,
            })
        );
    }
}
