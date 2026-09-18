//! Packed statement words in the bit order consumed by the Boolean PCS.

use alloc::vec::Vec;
use core::fmt::Debug;
use core::ops::AddAssign;

use p3_binary_field::{PackedGf2x32, PackedGf2x64};
use p3_word::{ConstraintSystem, Segment, ShapeError, ValueIndex, Word, Word32, Word64};

/// A word whose bits occupy one binary-field packing.
pub trait PackedWord: Word {
    /// One field lane per bit, from least to most significant.
    type Packing: AddAssign + Copy + Debug + Eq;

    /// Packs the word without changing its bit order.
    fn pack(self) -> Self::Packing;

    /// Recovers the word from its bit lanes.
    fn unpack(value: Self::Packing) -> Self;
}

impl PackedWord for Word32 {
    type Packing = PackedGf2x32;

    #[inline]
    fn pack(self) -> Self::Packing {
        PackedGf2x32::new(self.get())
    }

    #[inline]
    fn unpack(value: Self::Packing) -> Self {
        Self::new(value.to_bits())
    }
}

impl PackedWord for Word64 {
    type Packing = PackedGf2x64;

    #[inline]
    fn pack(self) -> Self::Packing {
        PackedGf2x64::new(self.get())
    }

    #[inline]
    fn unpack(value: Self::Packing) -> Self {
        Self::new(value.to_bits())
    }
}

/// A witness whose words retain the bit order consumed by the Boolean PCS.
///
/// Bit `i` of word `w` occupies binary lane `word_bits * w + i`.
/// Multilinear points place the within-word index in the trailing coordinates.
/// The least-significant coordinate is last under the big-endian convention.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedWitness<W: PackedWord> {
    /// The verifier-known words in least-significant-bit-first packing.
    public: Vec<W::Packing>,
    /// The committed words in least-significant-bit-first packing.
    witness: Vec<W::Packing>,
}

/// A witness that does not match its checked statement shape.
pub type WitnessError = ShapeError;

impl<W: PackedWord> PackedWitness<W> {
    /// Packs an exactly shaped public and committed word vector.
    pub fn new(
        system: &ConstraintSystem<W>,
        public: &[W],
        witness: &[W],
    ) -> Result<Self, WitnessError> {
        // Packing preserves the statement's two independent segment lengths.
        system.check_shape(public.len(), witness.len())?;

        Ok(Self {
            public: public.iter().copied().map(W::pack).collect(),
            witness: witness.iter().copied().map(W::pack).collect(),
        })
    }

    /// Takes ownership of exactly shaped buffers that are already bit-packed.
    pub fn from_packed(
        system: &ConstraintSystem<W>,
        public: Vec<W::Packing>,
        witness: Vec<W::Packing>,
    ) -> Result<Self, WitnessError> {
        // Prepacked buffers obey the same checked statement shape.
        system.check_shape(public.len(), witness.len())?;
        Ok(Self { public, witness })
    }

    pub(crate) const fn check_shape(
        &self,
        system: &ConstraintSystem<W>,
    ) -> Result<(), WitnessError> {
        // Revalidation protects backend entry points that receive stored witnesses.
        system.check_shape(self.public.len(), self.witness.len())
    }

    /// Returns the packed public words.
    #[inline]
    pub fn public(&self) -> &[W::Packing] {
        &self.public
    }

    /// Returns the packed committed words.
    #[inline]
    pub fn witness(&self) -> &[W::Packing] {
        &self.witness
    }

    /// Reads a word through its checked segment index.
    #[inline]
    pub fn get(&self, index: ValueIndex) -> Option<W> {
        let segment = match index.segment() {
            Segment::Public => &self.public,
            Segment::Witness => &self.witness,
        };
        segment
            .get(index.position() as usize)
            .copied()
            .map(W::unpack)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_word::ConstraintSystem;
    use proptest::prelude::*;

    use super::*;

    proptest! {
        #[test]
        fn word32_packing_preserves_every_bit(words in prop::collection::vec(any::<u32>(), 0..64)) {
            let words = words.into_iter().map(Word32::new).collect::<Vec<_>>();
            let system = ConstraintSystem::new(0, words.len(), vec![], vec![], vec![]).unwrap();
            let packed = PackedWitness::new(&system, &[], &words).unwrap();

            prop_assert_eq!(
                packed.witness().iter().map(|word| word.to_bits()).collect::<Vec<_>>(),
                words.iter().map(|word| word.get()).collect::<Vec<_>>(),
            );
        }

        #[test]
        fn word64_packing_preserves_every_bit(words in prop::collection::vec(any::<u64>(), 0..64)) {
            let words = words.into_iter().map(Word64::new).collect::<Vec<_>>();
            let system = ConstraintSystem::new(0, words.len(), vec![], vec![], vec![]).unwrap();
            let packed = PackedWitness::new(&system, &[], &words).unwrap();

            prop_assert_eq!(
                packed.witness().iter().map(|word| word.to_bits()).collect::<Vec<_>>(),
                words.iter().map(|word| word.get()).collect::<Vec<_>>(),
            );
        }
    }

    #[test]
    fn rejects_a_short_segment() {
        let system = ConstraintSystem::<Word32>::new(1, 1, vec![], vec![], vec![]).unwrap();
        assert_eq!(
            PackedWitness::new(&system, &[], &[Word32::new(1)]),
            Err(WitnessError {
                segment: Segment::Public,
                expected: 1,
                actual: 0,
            })
        );
    }

    #[test]
    fn packed_buffers_are_taken_without_reallocation() {
        let system = ConstraintSystem::<Word64>::new(1, 1, vec![], vec![], vec![]).unwrap();
        let mut public = Vec::with_capacity(5);
        public.push(PackedGf2x64::new(7));
        let mut witness = Vec::with_capacity(9);
        witness.push(PackedGf2x64::new(11));
        let public_ptr = public.as_ptr();
        let public_capacity = public.capacity();
        let witness_ptr = witness.as_ptr();
        let witness_capacity = witness.capacity();

        let packed = PackedWitness::from_packed(&system, public, witness).unwrap();

        assert_eq!(packed.public.as_ptr(), public_ptr);
        assert_eq!(packed.public.capacity(), public_capacity);
        assert_eq!(packed.witness.as_ptr(), witness_ptr);
        assert_eq!(packed.witness.capacity(), witness_capacity);
    }

    #[test]
    fn packed_buffers_still_require_the_exact_shape() {
        let system = ConstraintSystem::<Word32>::new(0, 1, vec![], vec![], vec![]).unwrap();
        assert_eq!(
            PackedWitness::from_packed(&system, vec![], vec![]),
            Err(WitnessError {
                segment: Segment::Witness,
                expected: 1,
                actual: 0,
            })
        );
    }
}
