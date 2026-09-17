//! Fixed-width blocks of bits, 8 to 512 wide, with no field structure attached.
//!
//! A block is addressed as an array of machine words, lowest bits in word `0`.
//! Nothing above this layer names a register: it asks for words, and the compiler chooses.
//!
//! # Why an aligned word array, not a vector intrinsic
//!
//! The wide blocks hold plain 64-bit words, aligned to the register that fits them.
//! That is the convention the crate's packed `GF(2^128)` already follows.
//!
//! It keeps a block constructible in a constant, and derivable for equality and hashing.
//! It is also one representation on every target, with no per-feature `cfg` to get wrong.
//!
//! Exclusive or, conjunction and shifts still reach the widest register the build enables.
//!
//! # Divisibility
//!
//! A wide block is laid out as narrow ones side by side, so reading it narrow moves no bits:
//!
//! ```text
//!     M512  ->  [M128; 4]  ->  [u64; 8]  ->  [u8; 64]
//! ```
//!
//! That is what makes a width change a borrow, and what lets the transpose sweep flat.

use core::fmt::Debug;
use core::hash::Hash;
use core::ops::{BitAnd, BitOr, BitXor, Not, Shl, Shr};
use core::slice;

use rand::{Rng, RngExt};

/// Keeps the storage traits closed to this crate.
///
/// Their safety rests on the exact layout of the few types below.
/// No outside implementation could be held to it.
pub(crate) mod private {
    pub trait Sealed {}
}

use private::Sealed;

impl Sealed for u8 {}
impl Sealed for u16 {}
impl Sealed for u32 {}
impl Sealed for u64 {}

/// The machine integer a bit kernel shifts and masks.
///
/// Every operation inside a word goes through this, so a kernel is written once per shape.
pub trait Word:
    Sealed
    + Copy
    + Default
    + Eq
    + Ord
    + Hash
    + Debug
    + Send
    + Sync
    + 'static
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + BitXor<Output = Self>
    + Not<Output = Self>
    + Shl<usize, Output = Self>
    + Shr<usize, Output = Self>
{
    /// Bits in the word.
    const BITS: usize;

    /// Every bit clear.
    const ZERO: Self;

    /// Every bit set.
    const ONES: Self;

    /// How many bits are set.
    fn count_ones(self) -> u32;

    /// The word widened to 64 bits, zero extended.
    fn to_u64(self) -> u64;

    /// The low bits of a 64-bit value, with anything above the word's width dropped.
    fn from_u64(value: u64) -> Self;
}

/// Implement the word trait for one primitive integer.
macro_rules! impl_word {
    ($word:ty) => {
        impl Word for $word {
            const BITS: usize = <$word>::BITS as usize;
            const ZERO: Self = 0;
            const ONES: Self = <$word>::MAX;

            #[inline]
            fn count_ones(self) -> u32 {
                Self::count_ones(self)
            }

            #[inline]
            fn to_u64(self) -> u64 {
                u64::from(self)
            }

            #[inline]
            fn from_u64(value: u64) -> Self {
                value as Self
            }
        }
    };
}

impl_word!(u8);
impl_word!(u16);
impl_word!(u32);
impl_word!(u64);

/// A fixed-width block of bits, addressed as an array of words.
///
/// Word `0` holds the lowest bits of the block, and bit `0` of a word is the lowest of all.
///
/// # Safety
/// A value must be exactly its words and nothing else:
/// - its size must equal the word count times the size of one word,
/// - its alignment must be at least a word's.
///
/// Reinterpreting a run of values as a run of words relies on both.
pub unsafe trait Underlier:
    Copy + Default + Eq + Hash + Debug + Send + Sync + 'static
{
    /// The word the block is addressed in.
    type Word: Word;

    /// How many words make up the block.
    const WORDS: usize;

    /// Bits in the whole block.
    const BITS: usize = Self::WORDS * <Self::Word as Word>::BITS;

    /// Every bit clear.
    const ZERO: Self;

    /// Every bit set.
    const ONES: Self;

    /// The words of the block, lowest bits first.
    fn words(&self) -> &[Self::Word];

    /// The words of the block, lowest bits first.
    fn words_mut(&mut self) -> &mut [Self::Word];

    /// Build a block word by word, lowest bits first.
    fn from_words_fn(f: impl FnMut(usize) -> Self::Word) -> Self;

    /// A block whose bits are uniformly random and independent.
    fn random<R: Rng + ?Sized>(rng: &mut R) -> Self;

    /// Combine two blocks word by word.
    #[inline]
    fn zip(self, other: Self, op: impl Fn(Self::Word, Self::Word) -> Self::Word) -> Self {
        Self::from_words_fn(|i| op(self.words()[i], other.words()[i]))
    }
}

/// Implement the underlier trait for a primitive integer, which is a block of one word.
macro_rules! impl_scalar_underlier {
    ($word:ty) => {
        // SAFETY: a value is its own single word, so the size and alignment match exactly.
        unsafe impl Underlier for $word {
            type Word = Self;

            const WORDS: usize = 1;
            const ZERO: Self = 0;
            const ONES: Self = <$word>::MAX;

            #[inline]
            fn words(&self) -> &[Self::Word] {
                slice::from_ref(self)
            }

            #[inline]
            fn words_mut(&mut self) -> &mut [Self::Word] {
                slice::from_mut(self)
            }

            #[inline]
            fn from_words_fn(mut f: impl FnMut(usize) -> Self::Word) -> Self {
                f(0)
            }

            #[inline]
            fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
                rng.random()
            }
        }
    };
}

impl_scalar_underlier!(u8);
impl_scalar_underlier!(u16);
impl_scalar_underlier!(u32);
impl_scalar_underlier!(u64);

/// Define a wide block of 64-bit words, aligned to the register that holds all of it.
///
/// That alignment also starts the block on the boundary every narrower block inside needs.
macro_rules! wide_underlier {
    ($name:ident, $words:literal, $align:literal, $doc:literal) => {
        #[doc = $doc]
        ///
        /// Word `0` holds the lowest bits.
        #[derive(Clone, Copy, Default, PartialEq, Eq, Hash, Debug)]
        #[repr(C, align($align))]
        pub struct $name([u64; $words]);

        impl Sealed for $name {}

        impl $name {
            /// A block from its words, lowest bits first.
            #[inline]
            #[must_use]
            pub const fn from_words(words: [u64; $words]) -> Self {
                Self(words)
            }

            /// The words of the block, lowest bits first.
            #[inline]
            #[must_use]
            pub const fn to_words(self) -> [u64; $words] {
                self.0
            }
        }

        // SAFETY: `repr(C)` over the word array gives the array's size, `align` only raises.
        unsafe impl Underlier for $name {
            type Word = u64;

            const WORDS: usize = $words;
            const ZERO: Self = Self([0; $words]);
            const ONES: Self = Self([u64::MAX; $words]);

            #[inline]
            fn words(&self) -> &[Self::Word] {
                &self.0
            }

            #[inline]
            fn words_mut(&mut self) -> &mut [Self::Word] {
                &mut self.0
            }

            #[inline]
            fn from_words_fn(f: impl FnMut(usize) -> Self::Word) -> Self {
                Self(core::array::from_fn(f))
            }

            #[inline]
            fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
                Self(rng.random())
            }
        }
    };
}

wide_underlier!(M128, 2, 16, "A 128-bit block of bits.");
wide_underlier!(M256, 4, 32, "A 256-bit block of bits.");
wide_underlier!(M512, 8, 64, "A 512-bit block of bits.");

/// A block that is exactly several narrower blocks laid end to end, part `0` lowest.
///
/// # Safety
/// The trait is sealed, and every implementation is checked below at compile time:
/// - the wide size is the ratio times the narrow size,
/// - the wide alignment is at least the narrow alignment,
/// - the target is little-endian whenever the narrow word is the shorter one.
///
/// The three together are what make the reinterpretations sound.
pub unsafe trait Divisible<Narrow: Underlier>: Underlier {
    /// How many narrow blocks fit in one wide block.
    const RATIO: usize = Self::BITS / Narrow::BITS;

    /// The narrow blocks this one is made of, lowest bits first.
    #[inline]
    fn parts(&self) -> &[Narrow] {
        const { Self::CHECK_LAYOUT };

        // SAFETY: the check pins one wide block at exactly `RATIO` narrow ones.
        // They are suitably aligned, and in the order the lane map promises.
        unsafe { slice::from_raw_parts(core::ptr::from_ref(self).cast(), Self::RATIO) }
    }

    /// The narrow blocks this one is made of, lowest bits first.
    #[inline]
    fn parts_mut(&mut self) -> &mut [Narrow] {
        const { Self::CHECK_LAYOUT };

        // SAFETY: as above, and the exclusive borrow is not duplicated.
        unsafe { slice::from_raw_parts_mut(core::ptr::from_mut(self).cast(), Self::RATIO) }
    }

    /// A run of wide blocks read as one run of narrow ones, lowest bits first.
    #[inline]
    fn split_slice(slice: &[Self]) -> &[Narrow] {
        const { Self::CHECK_LAYOUT };

        // SAFETY: wide blocks are contiguous and each covers exactly `RATIO` narrow ones.
        unsafe { slice::from_raw_parts(slice.as_ptr().cast(), slice.len() * Self::RATIO) }
    }

    /// A run of wide blocks read as one run of narrow ones, lowest bits first.
    #[inline]
    fn split_slice_mut(slice: &mut [Self]) -> &mut [Narrow] {
        const { Self::CHECK_LAYOUT };

        // SAFETY: as above, and the exclusive borrow is not duplicated.
        let len = slice.len() * Self::RATIO;
        unsafe { slice::from_raw_parts_mut(slice.as_mut_ptr().cast(), len) }
    }

    /// What every reinterpretation above rests on, rejected at compile time if it fails.
    ///
    /// Reading a block at a narrower word is a memory-order view.
    /// A lane is defined on word values, so the two agree only on a little-endian target.
    ///
    /// Views that keep the word width are pure reindexing, and hold either way.
    const CHECK_LAYOUT: () = {
        assert!(size_of::<Self>() == Self::RATIO * size_of::<Narrow>());
        assert!(align_of::<Self>() >= align_of::<Narrow>());
        assert!(
            cfg!(target_endian = "little") || size_of::<Narrow::Word>() == size_of::<Self::Word>(),
            "reading a block at a narrower word needs a little-endian target"
        );
    };
}

/// Declare that one block is an array of another.
///
/// The layout is checked by the trait itself, at every reinterpretation.
macro_rules! divisible {
    ($wide:ty, $narrow:ty) => {
        // SAFETY: the trait's own layout check rejects a pair that does not line up.
        unsafe impl Divisible<$narrow> for $wide {
            const RATIO: usize = <$wide as Underlier>::BITS / <$narrow as Underlier>::BITS;
        }
    };
}

divisible!(M512, M256);
divisible!(M512, M128);
divisible!(M512, u64);
divisible!(M512, u32);
divisible!(M512, u16);
divisible!(M512, u8);

divisible!(M256, M128);
divisible!(M256, u64);
divisible!(M256, u32);
divisible!(M256, u16);
divisible!(M256, u8);

divisible!(M128, u64);
divisible!(M128, u32);
divisible!(M128, u16);
divisible!(M128, u8);

divisible!(u64, u32);
divisible!(u64, u16);
divisible!(u64, u8);

divisible!(u32, u16);
divisible!(u32, u8);

divisible!(u16, u8);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_block_is_its_words() {
        // Invariant: word 0 holds the lowest bits, and the constants are all-clear, all-set.
        assert_eq!(M512::ZERO.words(), &[0u64; 8]);
        assert_eq!(M512::ONES.words(), &[u64::MAX; 8]);
        assert_eq!(M512::BITS, 512);
        assert_eq!(M256::BITS, 256);
        assert_eq!(M128::BITS, 128);
        assert_eq!(<u8 as Underlier>::BITS, 8);

        // Building word by word and reading back is a round trip.
        let block = M256::from_words_fn(|i| i as u64);
        assert_eq!(block.words(), &[0u64, 1, 2, 3]);
        assert_eq!(block.to_words(), [0u64, 1, 2, 3]);
        assert_eq!(M256::from_words([0, 1, 2, 3]), block);
    }

    #[test]
    fn a_wide_block_is_an_array_of_narrow_ones() {
        // Fixture: the 512-bit block whose word `i` is `i`.
        let block = M512::from_words_fn(|i| i as u64);

        // As 64-bit words, the parts are the words themselves.
        let words: &[u64] = block.parts();
        assert_eq!(words, &[0u64, 1, 2, 3, 4, 5, 6, 7]);

        // As 128-bit blocks, part `k` holds words `2k` and `2k + 1`.
        let halves: &[M128] = block.parts();
        assert_eq!(halves.len(), 4);
        assert_eq!(halves[0].to_words(), [0, 1]);
        assert_eq!(halves[3].to_words(), [6, 7]);

        // As 256-bit blocks, part 0 holds the low four words.
        let quarters: &[M256] = block.parts();
        assert_eq!(quarters.len(), 2);
        assert_eq!(quarters[0].to_words(), [0, 1, 2, 3]);
        assert_eq!(quarters[1].to_words(), [4, 5, 6, 7]);

        // As bytes, byte 0 is the lowest byte of word 0, so word 1 lands at byte 8.
        let bytes: &[u8] = block.parts();
        assert_eq!(bytes.len(), 64);
        assert_eq!(bytes[0], 0);
        assert_eq!(bytes[8], 1);
        assert_eq!(bytes[16], 2);
    }

    #[test]
    fn writing_through_a_narrow_view_changes_the_wide_block() {
        // The view borrows the same bytes, so a write through it is a write to the block.
        let mut block = M128::ZERO;
        {
            let bytes: &mut [u8] = block.parts_mut();
            bytes[0] = 0xff;
            bytes[15] = 1;
        }

        // Byte 0 is the lowest byte of word 0, and byte 15 the highest of word 1.
        assert_eq!(block.to_words(), [0xff, 1 << 56]);
    }

    #[test]
    fn a_word_widens_and_truncates() {
        // Widening is zero extension, narrowing keeps the low bits.
        assert_eq!(<u8 as Word>::to_u64(0xff), 255);
        assert_eq!(<u8 as Word>::from_u64(0x1234), 0x34);
        assert_eq!(<u16 as Word>::from_u64(0x1_2345), 0x2345);
        assert_eq!(<u64 as Word>::ONES, u64::MAX);
        assert_eq!(<u32 as Word>::count_ones(0b1011), 3);
    }

    #[test]
    fn combining_runs_word_by_word() {
        // Fixture: two blocks whose words differ in every position.
        let a = M128::from_words([0b1100, 0b1010]);
        let b = M128::from_words([0b1010, 0b0110]);

        assert_eq!(
            a.zip(b, |x, y| x ^ y).to_words(),
            [0b1100 ^ 0b1010, 0b1010 ^ 0b0110]
        );
        assert_eq!(
            a.zip(b, |x, y| x & y).to_words(),
            [0b1100 & 0b1010, 0b1010 & 0b0110]
        );
    }
}
