//! The storage layer the bit-sliced types are built on: fixed-width blocks of bits with no
//! field structure attached, and the relation that makes a wide block an array of narrow ones.
//!
//! # What an underlier is
//!
//! A block of `BITS` bits, addressed as a fixed number of machine words.
//!
//! Nothing above this layer knows how wide a register the target has; it asks the underlier
//! for words and the compiler picks the register:
//!
//! ```text
//!     M512   8 words of 64 bits    ->  one 512-bit register, or two 256, or four 128
//! ```
//!
//! # Why the representation is an aligned word array
//!
//! The wide types hold `[u64; n]` at the alignment of the register that fits them, rather
//! than a vector intrinsic type.
//!
//! This is the convention the crate's packed `GF(2^128)` already uses: hold the array, and
//! view it as a register at the point where an intrinsic needs one.
//!
//! It keeps the types constructible in a constant, comparable and hashable by derive, and
//! identical on every target, while exclusive or, conjunction and shifts lower to the widest
//! register the build actually enables.
//!
//! A vector intrinsic type in the field would instead have to be selected by `cfg` per target
//! feature, because holding a 512-bit value in a build without those registers splits it
//! anyway.
//!
//! # The divisibility relation
//!
//! A wide block is laid out as several narrow ones side by side, so reading it at the narrow
//! width moves no bits:
//!
//! ```text
//!     M512  ->  [M128; 4]  ->  [u64; 8]  ->  [u8; 64]
//! ```
//!
//! That relation is what lets a bit witness committed at one width be read at another for
//! free, and what lets the transpose sweep a whole run of rows as one flat word slice.

use core::fmt::Debug;
use core::hash::Hash;
use core::ops::{BitAnd, BitOr, BitXor, Not, Shl, Shr};
use core::slice;

use rand::{Rng, RngExt};

/// The machine integer a bit kernel does its shifting and masking on.
///
/// Every operation the bit-sliced layer performs inside a word goes through this trait, so
/// the kernels are written once and instantiated at each width.
pub trait Word:
    Copy
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
/// Word `0` holds the lowest bits of the block, and within a word bit `0` is the lowest.
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

/// Define a wide block of 64-bit words.
///
/// The alignment is the width of the register that holds the whole block, so a buffer of them
/// is laid out the way a vectorised pass wants to read it, and a wide block starts on the
/// boundary every narrower block inside it needs.
macro_rules! wide_underlier {
    ($name:ident, $words:literal, $align:literal, $doc:literal) => {
        #[doc = $doc]
        ///
        /// Word `0` holds the lowest bits.
        #[derive(Clone, Copy, Default, PartialEq, Eq, Hash, Debug)]
        #[repr(C, align($align))]
        pub struct $name([u64; $words]);

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

        // SAFETY: the type is `repr(C)` over exactly its word array, so its size is the array's,
        // and `repr(align)` only raises the alignment above a word's.
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

/// A block that is exactly several narrower blocks laid end to end.
///
/// The narrow blocks appear in order of increasing significance, so part `0` holds the lowest
/// bits of the wide block.
///
/// # Safety
/// An implementation asserts that the wide block's size is the ratio times the narrow one's
/// and that its alignment is at least the narrow one's, which is what makes reinterpreting a
/// run of wide blocks as a run of narrow ones sound.
pub unsafe trait Divisible<Narrow: Underlier>: Underlier {
    /// How many narrow blocks fit in one wide block.
    const RATIO: usize = Self::BITS / Narrow::BITS;

    /// The narrow blocks this one is made of, lowest bits first.
    #[inline]
    fn parts(&self) -> &[Narrow] {
        // SAFETY: the layout assertions below hold for every implementation, so one wide
        // block covers exactly `RATIO` narrow ones at a suitable alignment, and every bit
        // pattern of a narrow block is valid.
        unsafe { slice::from_raw_parts(core::ptr::from_ref(self).cast(), Self::RATIO) }
    }

    /// The narrow blocks this one is made of, lowest bits first.
    #[inline]
    fn parts_mut(&mut self) -> &mut [Narrow] {
        // SAFETY: as above, and the exclusive borrow is not duplicated.
        unsafe { slice::from_raw_parts_mut(core::ptr::from_mut(self).cast(), Self::RATIO) }
    }

    /// A run of wide blocks read as one run of narrow ones, lowest bits first.
    #[inline]
    fn split_slice(slice: &[Self]) -> &[Narrow] {
        // SAFETY: consecutive wide blocks are contiguous and each is exactly `RATIO` narrow
        // ones, so the run covers `len` narrow blocks and no more.
        unsafe { slice::from_raw_parts(slice.as_ptr().cast(), slice.len() * Self::RATIO) }
    }

    /// A run of wide blocks read as one run of narrow ones, lowest bits first.
    #[inline]
    fn split_slice_mut(slice: &mut [Self]) -> &mut [Narrow] {
        // SAFETY: as above, and the exclusive borrow is not duplicated.
        let len = slice.len() * Self::RATIO;
        unsafe { slice::from_raw_parts_mut(slice.as_mut_ptr().cast(), len) }
    }
}

/// Declare that one block is an array of another, with the layout checked at compile time.
macro_rules! divisible {
    ($wide:ty, $narrow:ty) => {
        // SAFETY: the constants below are rejected at compile time unless the wide block is
        // exactly `RATIO` narrow blocks and is at least as aligned as one.
        unsafe impl Divisible<$narrow> for $wide {
            const RATIO: usize = {
                let ratio = <$wide as Underlier>::BITS / <$narrow as Underlier>::BITS;
                assert!(size_of::<$wide>() == ratio * size_of::<$narrow>());
                assert!(align_of::<$wide>() >= align_of::<$narrow>());
                ratio
            };
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
        // Invariant: word `0` holds the lowest bits, and the constants are all-clear and
        // all-set.
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
        //
        //     words:  0 1 2 3 4 5 6 7
        let block = M512::from_words_fn(|i| i as u64);

        // Read as 64-bit words, the parts are the words themselves.
        let words: &[u64] = block.parts();
        assert_eq!(words, &[0u64, 1, 2, 3, 4, 5, 6, 7]);

        // Read as 128-bit blocks, part `k` holds words `2k` and `2k + 1`.
        let halves: &[M128] = block.parts();
        assert_eq!(halves.len(), 4);
        assert_eq!(halves[0].to_words(), [0, 1]);
        assert_eq!(halves[3].to_words(), [6, 7]);

        // Read as 256-bit blocks, part `0` holds the low four words.
        let quarters: &[M256] = block.parts();
        assert_eq!(quarters.len(), 2);
        assert_eq!(quarters[0].to_words(), [0, 1, 2, 3]);
        assert_eq!(quarters[1].to_words(), [4, 5, 6, 7]);

        // Read as bytes, the first byte is the lowest byte of word 0.
        //
        //     word 1 = 1  ->  byte 8 = 1
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

        // Byte 0 is the lowest byte of word 0; byte 15 is the highest byte of word 1.
        assert_eq!(block.to_words(), [0xff, 1 << 56]);
    }

    #[test]
    fn a_word_widens_and_truncates() {
        // Widening is zero extension, and narrowing keeps the low bits.
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
