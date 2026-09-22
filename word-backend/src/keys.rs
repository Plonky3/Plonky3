//! Compiled word-to-relation metadata for backend reductions.

use alloc::vec::Vec;
use core::ops::Range;

use p3_word::{
    ComponentCall, Composition, ConstraintKind, ConstraintSystem, ConstraintTerm, OperandRole,
    Segment, Shift, Word,
};
use thiserror::Error;

mod builder;
mod code;

use builder::CountingSegment;
use code::{KeyCode, ShiftSequenceCode};

/// Number of relation families a statement can declare.
const FAMILIES: usize = 3;

/// A compact index into one operation family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstraintReference {
    /// The operand's semantic role within its relation.
    operand: OperandRole,
    /// The position within its homogeneous relation family.
    constraint: u32,
}

impl ConstraintReference {
    /// Returns the operand's semantic role within the relation.
    #[inline]
    pub const fn operand(self) -> OperandRole {
        self.operand
    }

    /// Returns the constraint position within the operation family.
    #[inline]
    pub const fn constraint(self) -> u32 {
        self.constraint
    }
}

/// One word's resolved references under a fixed operation and shift sequence.
///
/// The stored references are the ones one gadget slot carries.
///
/// An instanced segment stores them once and adds the instance's offset on the way out.
///
/// That is what keeps the stored metadata independent of the instance count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledKey<'a, W: Word> {
    /// The relation family consuming the word.
    operation: ConstraintKind,
    /// The dense index of the shift sequence.
    shift_index: u32,
    /// The inner and outer movements in evaluation order.
    shifts: [Shift<W>; 2],
    /// The relation occurrences consuming the shifted word, in slot-local form.
    references: &'a [ConstraintReference],
    /// The instance's first relation position within this key's family.
    constraint_offset: u32,
}

impl<W: Word> CompiledKey<'_, W> {
    /// Returns the relation family using the word.
    #[inline]
    pub const fn operation(&self) -> ConstraintKind {
        self.operation
    }

    /// Returns the dense shift-sequence index.
    #[inline]
    pub const fn shift_index(&self) -> u32 {
        self.shift_index
    }

    /// Returns the inner and outer movements in evaluation order.
    #[inline]
    pub const fn shifts(&self) -> [Shift<W>; 2] {
        self.shifts
    }

    /// Returns the number of relation occurrences grouped under this key.
    #[inline]
    pub const fn num_references(&self) -> usize {
        self.references.len()
    }

    /// Returns every relation occurrence in reduction order.
    ///
    /// Positions are already resolved to the composed statement.
    #[inline]
    pub fn references(
        &self,
    ) -> impl ExactSizeIterator<Item = ConstraintReference> + Clone + use<'_, W> {
        let offset = self.constraint_offset;
        self.references.iter().map(move |reference| {
            // The stored position is slot-local, so the instance is added here.
            ConstraintReference {
                operand: reference.operand,
                constraint: reference.constraint + offset,
            }
        })
    }
}

/// One stored key before its references are resolved to borrowed slices.
#[derive(Clone, Debug, Eq, PartialEq)]
struct StoredKey {
    /// The relation family consuming the word.
    operation: ConstraintKind,
    /// The dense index of the shift sequence.
    shift: u32,
    /// The contiguous span of relation references.
    references: Range<u32>,
}

/// One call's slot storage, repeated across its instances.
///
/// A flat statement is the degenerate case.
///
/// It is one block of one instance whose slots are the segment's own words.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InstanceBlock {
    /// The block's first word in the composed segment.
    word_base: u32,
    /// Words one instance occupies, which is also the instance stride.
    slots: u32,
    /// Live instances of the call.
    instances: u32,
    /// The block's first entry in the shared slot-key table.
    slot_base: u32,
    /// Relations one instance adds to each family, in family-code order.
    strides: [u32; FAMILIES],
    /// The first relation position of instance zero, in family-code order.
    constraint_base: [u32; FAMILIES],
}

impl InstanceBlock {
    /// Returns the number of words the block spans.
    #[inline]
    const fn width(&self) -> u32 {
        // Both factors were bounded when the composition was laid out.
        self.slots * self.instances
    }

    /// Returns the instance and slot a composed word belongs to.
    #[inline]
    const fn split(&self, word: u32) -> (u32, u32) {
        let local = word - self.word_base;
        (local / self.slots, local % self.slots)
    }
}

/// Storage the compiled metadata of one segment occupies.
///
/// An instanced layout keeps these counts flat as the instance count grows.
///
/// They are therefore the quantity a scaling measurement should report.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LayoutFootprint {
    /// Stored relation occurrences.
    pub references: usize,
    /// Stored operation-and-shift groups.
    pub keys: usize,
    /// Stored per-slot key spans.
    pub slots: usize,
    /// Stored distinct shift spellings.
    pub shift_sequences: usize,
    /// Stored instance blocks.
    pub blocks: usize,
}

impl LayoutFootprint {
    /// Returns the total of every stored entry.
    #[must_use]
    pub const fn entries(&self) -> usize {
        self.references + self.keys + self.slots + self.shift_sequences + self.blocks
    }

    /// Adds another segment's footprint.
    const fn add(self, other: Self) -> Self {
        Self {
            references: self.references + other.references,
            keys: self.keys + other.keys,
            slots: self.slots + other.slots,
            shift_sequences: self.shift_sequences + other.shift_sequences,
            blocks: self.blocks + other.blocks,
        }
    }
}

/// Metadata compiled independently for one visibility segment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledSegment<W: Word> {
    /// The distinct shift-sequence spellings.
    shifts: Vec<[Shift<W>; 2]>,
    /// The operation and shift groups across all slots.
    keys: Vec<StoredKey>,
    /// The contiguous key span assigned to each slot.
    slot_keys: Vec<Range<u32>>,
    /// The relation consumers grouped by key, in slot-local positions.
    references: Vec<ConstraintReference>,
    /// The calls whose slots repeat across instances, in word order.
    blocks: Vec<InstanceBlock>,
    /// The number of composed words the blocks cover.
    words: usize,
}

impl<W: Word> CompiledSegment<W> {
    /// Returns the number of indexed words.
    #[inline]
    pub const fn len(&self) -> usize {
        self.words
    }

    /// Returns whether the segment has no words.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.words == 0
    }

    /// Returns the storage this segment's metadata occupies.
    #[must_use]
    pub const fn footprint(&self) -> LayoutFootprint {
        LayoutFootprint {
            references: self.references.len(),
            keys: self.keys.len(),
            slots: self.slot_keys.len(),
            shift_sequences: self.shifts.len(),
            blocks: self.blocks.len(),
        }
    }

    /// Returns the resolved keys attached to one word.
    pub fn keys(
        &self,
        word: usize,
    ) -> Option<impl ExactSizeIterator<Item = CompiledKey<'_, W>> + Clone + '_> {
        // Keep storage-local indices inside this segment.
        let word = u32::try_from(word).ok()?;
        let block = self.block_of(word)?;
        let (instance, slot) = block.split(word);
        let range = self.slot_keys.get((block.slot_base + slot) as usize)?;
        let keys = self.keys.get(range.start as usize..range.end as usize)?;

        // Every family shifts by this instance's own share of that family.
        let offsets = [
            block.constraint_base[0] + instance * block.strides[0],
            block.constraint_base[1] + instance * block.strides[1],
            block.constraint_base[2] + instance * block.strides[2],
        ];
        Some(keys.iter().map(move |key| self.resolve(key, offsets)))
    }

    /// Returns the dense shift-sequence table.
    #[inline]
    pub fn shift_sequences(&self) -> &[[Shift<W>; 2]] {
        &self.shifts
    }

    /// Returns the block owning one composed word.
    fn block_of(&self, word: u32) -> Option<&InstanceBlock> {
        // Blocks are stored in increasing word order and never overlap.
        let index = self
            .blocks
            .partition_point(|block| block.word_base <= word)
            .checked_sub(1)?;
        let block = &self.blocks[index];
        (word < block.word_base + block.width()).then_some(block)
    }

    fn resolve(&self, key: &StoredKey, offsets: [u32; FAMILIES]) -> CompiledKey<'_, W> {
        // Construction bounds every dense index and contiguous reference span.
        CompiledKey {
            operation: key.operation,
            shift_index: key.shift,
            shifts: self.shifts[key.shift as usize],
            references: &self.references
                [key.references.start as usize..key.references.end as usize],
            constraint_offset: offsets[key.operation.code() as usize],
        }
    }
}

/// Public and committed shift metadata, kept in separate index spaces.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledKeyLayout<W: Word> {
    /// The metadata for verifier-known words.
    public: CompiledSegment<W>,
    /// The metadata for prover-committed words.
    witness: CompiledSegment<W>,
}

impl<W: Word> CompiledKeyLayout<W> {
    /// Compiles every shifted reference in deterministic reduction order.
    ///
    /// - Families use zero, AND, then integer-multiplication order.
    /// - AND roles use left, right, then output order.
    /// - Multiplication roles use left, right, low, then high order.
    /// - Constraint positions ascend within each role.
    /// - Terms retain their supplied order and multiplicity.
    /// - Each word's keys retain the first occurrence of each group.
    pub fn new(system: &ConstraintSystem<W>) -> Result<Self, KeyCompileError> {
        // The first pass counts exact per-word storage without nested allocations.
        let mut public = CountingSegment::new(system.public_len(), Segment::Public)?;
        let mut witness = CountingSegment::new(system.witness_len(), Segment::Witness)?;
        for term in system.terms() {
            let reference = Reference::from(term);
            match reference.segment {
                Segment::Public => public.count(reference)?,
                Segment::Witness => witness.count(reference)?,
            }
        }

        // Prefix sums allocate one contiguous reference table per segment.
        let mut public = public.prepare()?;
        let mut witness = witness.prepare()?;

        // The second pass places every reference into its reserved word span.
        for term in system.terms() {
            let reference = Reference::from(term);
            match reference.segment {
                Segment::Public => public.insert(reference),
                Segment::Witness => witness.insert(reference),
            }
        }

        Ok(Self {
            public: public.compile()?,
            witness: witness.compile()?,
        })
    }

    /// Compiles each component once and repeats it across its instances.
    ///
    /// A call's relations are declared exactly once here, whatever its instance count.
    ///
    /// The stored metadata is therefore the size of the gadgets, not of the statement.
    ///
    /// Compiling the lowered statement instead would give an indistinguishable result.
    ///
    /// That is because the composed addressing is the affine map this one reverses.
    ///
    /// # Errors
    ///
    /// Returns an error when a gadget or the merged layout outgrows the compact key.
    pub fn compose(composition: &Composition<W>) -> Result<Self, KeyCompileError> {
        let mut public = Vec::with_capacity(composition.calls().len());
        let mut witness = Vec::with_capacity(composition.calls().len());
        let mut public_base = 0_u32;
        let mut witness_base = 0_u32;
        let mut constraint_base = [0_u32; FAMILIES];

        for (call, index) in composition.calls().iter().zip(0_usize..) {
            // One compilation of the body serves every instance of the call.
            let component = call.component();
            let compiled = Self::new(component.body())?;
            let instances = bound(call.instances(), Segment::Witness, LayoutComponent::Keys)?;
            let strides = relation_strides(call)?;

            public.push(SegmentPiece {
                segment: compiled.public,
                word_base: public_base,
                instances,
                strides,
                constraint_base,
            });
            witness.push(SegmentPiece {
                segment: compiled.witness,
                word_base: witness_base,
                instances,
                strides,
                constraint_base,
            });

            // The next call starts where this one's instance-packed block ends.
            public_base = advance(
                public_base,
                component.interface_slots(),
                instances,
                Segment::Public,
            )?;
            witness_base = advance(
                witness_base,
                component.local_slots(),
                instances,
                Segment::Witness,
            )?;
            for (family, stride) in strides.into_iter().enumerate() {
                let kind = ConstraintKind::from_code(family as u8)
                    .expect("relation strides are indexed by assigned family codes");
                constraint_base[family] =
                    constraint_base[family]
                        .checked_add(stride.checked_mul(instances).ok_or(
                            KeyCompileError::UnrepresentableComposition { call: index, kind },
                        )?)
                        .ok_or(KeyCompileError::UnrepresentableComposition { call: index, kind })?;
            }
        }

        Ok(Self {
            public: merge(public, Segment::Public)?,
            witness: merge(witness, Segment::Witness)?,
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

    /// Returns the storage both segments' metadata occupies.
    #[must_use]
    pub const fn footprint(&self) -> LayoutFootprint {
        self.public.footprint().add(self.witness.footprint())
    }
}

/// One call's compiled component before the segment blocks are merged.
struct SegmentPiece<W: Word> {
    /// The component body compiled once, in slot-local positions.
    segment: CompiledSegment<W>,
    /// The block's first word in the composed segment.
    word_base: u32,
    /// Live instances of the call.
    instances: u32,
    /// Relations one instance adds to each family.
    strides: [u32; FAMILIES],
    /// The first relation position of instance zero.
    constraint_base: [u32; FAMILIES],
}

/// Merges one compiled component per call into a single instanced segment.
fn merge<W: Word>(
    pieces: Vec<SegmentPiece<W>>,
    segment: Segment,
) -> Result<CompiledSegment<W>, KeyCompileError> {
    // Shift spellings are shared, so the merged table is the sorted union.
    let mut codes = pieces
        .iter()
        .flat_map(|piece| piece.segment.shifts.iter().copied().map(sequence_code))
        .collect::<Vec<_>>();
    codes.sort_unstable();
    codes.dedup();
    let shifts = codes
        .iter()
        .copied()
        .map(ShiftSequenceCode::shifts)
        .collect::<Vec<_>>();

    let mut keys = Vec::new();
    let mut slot_keys = Vec::new();
    let mut references = Vec::new();
    let mut blocks = Vec::new();
    let mut words = 0_usize;

    for piece in pieces {
        let slot_base = bound(slot_keys.len(), segment, LayoutComponent::WordOffsets)?;
        let key_base = bound(keys.len(), segment, LayoutComponent::Keys)?;
        let reference_base = bound(references.len(), segment, LayoutComponent::References)?;

        // Dense shift indices are rebased onto the merged table.
        let remap = piece
            .segment
            .shifts
            .iter()
            .copied()
            .map(|entry| {
                let code = sequence_code(entry);
                codes
                    .binary_search(&code)
                    .expect("the merged shift table covers every component spelling")
                    as u32
            })
            .collect::<Vec<_>>();

        references.extend(piece.segment.references.iter().copied());
        keys.extend(piece.segment.keys.iter().map(|key| StoredKey {
            operation: key.operation,
            shift: remap[key.shift as usize],
            references: (key.references.start + reference_base)
                ..(key.references.end + reference_base),
        }));
        slot_keys.extend(
            piece
                .segment
                .slot_keys
                .iter()
                .map(|range| (range.start + key_base)..(range.end + key_base)),
        );

        // A call with no words of this segment holds no addressable block.
        let slots = bound(
            piece.segment.slot_keys.len(),
            segment,
            LayoutComponent::WordOffsets,
        )?;
        let width = slots
            .checked_mul(piece.instances)
            .ok_or(KeyCompileError::LayoutTooLarge {
                segment,
                component: LayoutComponent::WordOffsets,
                len: usize::MAX,
            })?;
        if width != 0 {
            blocks.push(InstanceBlock {
                word_base: piece.word_base,
                slots,
                instances: piece.instances,
                slot_base,
                strides: piece.strides,
                constraint_base: piece.constraint_base,
            });
            words += width as usize;
        }
    }

    // Bound every merged endpoint before the segment is handed out.
    bound(keys.len(), segment, LayoutComponent::Keys)?;
    bound(references.len(), segment, LayoutComponent::References)?;
    bound(slot_keys.len(), segment, LayoutComponent::WordOffsets)?;

    Ok(CompiledSegment {
        shifts,
        keys,
        slot_keys,
        references,
        blocks,
        words,
    })
}

/// Returns the operation-independent code of one shift spelling.
fn sequence_code<W: Word>(shifts: [Shift<W>; 2]) -> ShiftSequenceCode {
    // The family tag is discarded, so any operation gives the same sequence.
    KeyCode::new(ConstraintKind::Zero, shifts).sequence()
}

/// Returns the relations one instance of a call adds to each family.
fn relation_strides<W: Word>(call: &ComponentCall<W>) -> Result<[u32; FAMILIES], KeyCompileError> {
    let mut strides = [0_u32; FAMILIES];
    for (family, count) in call.component().relation_counts().into_iter().enumerate() {
        let kind = ConstraintKind::from_code(family as u8)
            .expect("relation counts are indexed by assigned family codes");
        strides[family] = u32::try_from(count)
            .map_err(|_| KeyCompileError::UnrepresentableComposition { call: 0, kind })?;
    }
    Ok(strides)
}

/// Bounds one merged component length to its compact index.
fn bound(len: usize, segment: Segment, component: LayoutComponent) -> Result<u32, KeyCompileError> {
    u32::try_from(len).map_err(|_| KeyCompileError::LayoutTooLarge {
        segment,
        component,
        len,
    })
}

/// Advances a segment base past one instance-packed block.
fn advance(
    base: u32,
    slots: usize,
    instances: u32,
    segment: Segment,
) -> Result<u32, KeyCompileError> {
    let too_large = || KeyCompileError::LayoutTooLarge {
        segment,
        component: LayoutComponent::WordOffsets,
        len: usize::MAX,
    };
    let slots = u32::try_from(slots).map_err(|_| too_large())?;
    slots
        .checked_mul(instances)
        .and_then(|width| base.checked_add(width))
        .ok_or_else(too_large)
}

/// A bounded part of the compact key layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayoutComponent {
    /// Constraint references.
    References,
    /// Per-word keys.
    Keys,
    /// Word-offset table entries.
    WordOffsets,
}

/// A constraint system no key can represent or prove.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum KeyCompileError {
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
    /// The system declares a relation family the compiled protocol does not prove.
    #[error("{count} unsigned integer product relations are not proved by this protocol")]
    UnprovedRelation {
        /// Number of declared relations in the unsupported family.
        count: usize,
    },
    /// A call's instances do not fit the compact relation address space.
    #[error("call {call} repeats its {kind:?} relations past the compact address space")]
    UnrepresentableComposition {
        /// The call whose block cannot be addressed.
        call: usize,
        /// The relation family that overflows.
        kind: ConstraintKind,
    },
}

/// One semantic term translated into backend storage coordinates.
#[derive(Clone, Copy)]
struct Reference {
    /// The visibility class containing the source word.
    segment: Segment,
    /// The source offset within its visibility class.
    word: usize,
    /// The encoded operation and shift sequence.
    key_code: KeyCode,
    /// The relation consumer of the shifted word.
    constraint: ConstraintReference,
}

impl<W: Word> From<ConstraintTerm<'_, W>> for Reference {
    fn from(value: ConstraintTerm<'_, W>) -> Self {
        // The checked semantic term already carries its family provenance.
        let term = value.term();
        let index = term.index();
        Self {
            segment: index.segment(),
            word: index.position() as usize,
            key_code: KeyCode::new(value.kind(), term.shifts()),
            constraint: ConstraintReference {
                operand: value.role(),
                constraint: value.constraint(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_word::{
        AndConstraint, IntegerMulConstraint, Operand, ShiftKind, ShiftedValue, ValueIndex, Word32,
        Word64, ZeroConstraint,
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
        let public_keys = layout.public().keys(0).unwrap().collect::<Vec<_>>();
        assert_eq!(public_keys.len(), 2);
        assert_eq!(public_keys[0].shifts(), [rotate, Shift::identity()]);
        assert_eq!(
            public_keys[1].references().collect::<Vec<_>>(),
            [
                ConstraintReference {
                    operand: OperandRole::Left,
                    constraint: 0,
                },
                ConstraintReference {
                    operand: OperandRole::Right,
                    constraint: 0,
                },
            ]
        );

        assert_eq!(layout.witness().keys(0).unwrap().len(), 0);
        let witness_keys = layout.witness().keys(1).unwrap().collect::<Vec<_>>();
        assert_eq!(witness_keys.len(), 2);
        assert_eq!(
            witness_keys[0].references().collect::<Vec<_>>(),
            [
                ConstraintReference {
                    operand: OperandRole::Value,
                    constraint: 0,
                },
                ConstraintReference {
                    operand: OperandRole::Value,
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
        let keys = layout.witness().keys(0).unwrap().collect::<Vec<_>>();
        assert_eq!(
            keys.iter().map(|key| key.operation()).collect::<Vec<_>>(),
            [
                ConstraintKind::Zero,
                ConstraintKind::And,
                ConstraintKind::IntegerMul,
            ]
        );
        assert_eq!(
            keys[1].references().collect::<Vec<_>>(),
            [
                ConstraintReference {
                    operand: OperandRole::Left,
                    constraint: 0,
                },
                ConstraintReference {
                    operand: OperandRole::Right,
                    constraint: 0,
                },
                ConstraintReference {
                    operand: OperandRole::Output,
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
        let keys = layout.witness().keys(0).unwrap().collect::<Vec<_>>();
        assert_eq!(keys.len(), 2);
        assert_ne!(keys[0].shifts(), keys[1].shifts());
        assert_ne!(keys[0].shift_index(), keys[1].shift_index());
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
        let key = layout.witness().keys(0).unwrap().next().unwrap();
        assert_eq!(key.shifts(), [inner, outer]);
        assert_eq!(
            layout.witness().shift_sequences()[key.shift_index() as usize],
            [inner, outer]
        );
    }

    #[test]
    fn resolved_keys_cannot_cross_segment_boundaries() {
        // Public and committed words deliberately use different movements at the same position.
        let public = ValueIndex::public(0).unwrap();
        let witness = ValueIndex::witness(0).unwrap();
        let rotate = Shift::new(ShiftKind::RotateRight, 3).unwrap();
        let left = Shift::new(ShiftKind::LogicalLeft, 5).unwrap();
        let system = ConstraintSystem::new(
            1,
            1,
            vec![ZeroConstraint::new(Operand::new(vec![
                ShiftedValue::<Word64>::single(public, rotate),
                ShiftedValue::<Word64>::single(witness, left),
            ]))],
            vec![],
            vec![],
        )
        .unwrap();

        // Each iterator resolves indices only through the segment that owns them.
        let public_key = layout_key(&system, Segment::Public);
        let witness_key = layout_key(&system, Segment::Witness);
        assert_eq!(public_key, [rotate, Shift::identity()]);
        assert_eq!(witness_key, [left, Shift::identity()]);
    }

    fn layout_key(system: &ConstraintSystem<Word64>, segment: Segment) -> [Shift<Word64>; 2] {
        // Select the independent table whose key must be resolved.
        let layout = CompiledKeyLayout::new(system).unwrap();
        let compiled = match segment {
            Segment::Public => layout.public(),
            Segment::Witness => layout.witness(),
        };
        compiled.keys(0).unwrap().next().unwrap().shifts()
    }

    #[test]
    fn more_than_u16_shift_sequences_compile() {
        // Enumerate every checked movement once before forming irreducible pairs.
        let index = ValueIndex::witness(0).unwrap();
        let mut shifts = vec![Shift::<Word64>::identity()];
        for kind in [
            ShiftKind::LogicalLeft,
            ShiftKind::LogicalRight,
            ShiftKind::ArithmeticRight,
            ShiftKind::RotateRight,
            ShiftKind::Lane32LogicalLeft,
            ShiftKind::Lane32LogicalRight,
            ShiftKind::Lane32ArithmeticRight,
            ShiftKind::Lane32RotateRight,
        ] {
            let width = if kind.is_lane32() { 32 } else { 64 };
            for amount in 1..width {
                shifts.push(Shift::new(kind, amount).unwrap());
            }
        }

        // Fixture state: 65,537 distinct spellings exceed the former 16-bit index space.
        let target = u16::MAX as usize + 2;
        let mut terms = shifts
            .iter()
            .copied()
            .map(|shift| ShiftedValue::single(index, shift))
            .collect::<Vec<_>>();
        'outer: for &inner in &shifts {
            for &outer in &shifts {
                if let Ok(term) = ShiftedValue::pair(index, inner, outer) {
                    terms.push(term);
                    if terms.len() == target {
                        break 'outer;
                    }
                }
            }
        }
        assert_eq!(terms.len(), target);

        // A 32-bit dense index preserves every valid shift spelling.
        let system = ConstraintSystem::new(
            0,
            1,
            vec![ZeroConstraint::new(Operand::new(terms))],
            vec![],
            vec![],
        )
        .unwrap();
        let layout = CompiledKeyLayout::new(&system).unwrap();
        let keys = layout.witness().keys(0).unwrap();
        assert_eq!(keys.len(), target);
        assert_eq!(layout.witness().shift_sequences().len(), target);
        assert!(keys.map(|key| key.shift_index()).max().unwrap() > u32::from(u16::MAX));
    }

    #[test]
    fn word_offset_length_overflow_is_rejected() {
        assert!(matches!(
            CountingSegment::new(usize::MAX, Segment::Witness),
            Err(KeyCompileError::LayoutTooLarge {
                segment: Segment::Witness,
                component: LayoutComponent::WordOffsets,
                len: usize::MAX,
            })
        ));
    }
}
