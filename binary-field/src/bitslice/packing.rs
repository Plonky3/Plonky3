//! The bit-sliced `GF(2)` packings, eight through five hundred and twelve lanes wide.

use core::fmt::{self, Debug, Formatter};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::slice;

use p3_field::{Algebra, Field, PrimeCharacteristicRing};
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};

use super::underlier::{Divisible, M128, M256, M512, Underlier, Word};
use crate::Gf2;
use crate::gf2::characteristic_two_methods;

/// Masks selecting the low `s` bits of every `2s`-bit block, indexed by `log2(s)`.
///
/// The interleave and the transpose swap a high block against the low one beside it:
///
/// ```text
///     s = 1   ...0101 0101      every second bit
///     s = 2   ...0011 0011      every second pair
///     s = 4   ...00001111       every second nibble
/// ```
///
/// Six entries suffice, since `s` never exceeds half a `u64`.
const BLOCK_MASKS: [u64; 6] = [
    0x5555_5555_5555_5555,
    0x3333_3333_3333_3333,
    0x0f0f_0f0f_0f0f_0f0f,
    0x00ff_00ff_00ff_00ff,
    0x0000_ffff_0000_ffff,
    0x0000_0000_ffff_ffff,
];

/// The mask selecting the low `block` bits of every `2 * block`-bit block.
///
/// The pattern repeats with period `2 * block`, and every word width is a multiple of it.
/// So truncating the result to a narrower word keeps it intact.
///
/// # Panics
/// Panics if `block` is not a power of two in `1 ..= 32`.
#[inline]
pub(super) const fn block_mask(block: usize) -> u64 {
    // `trailing_zeros` is `log2` exactly because the caller passes a power of two.
    assert!(block.is_power_of_two() && block <= 32, "block out of range");
    BLOCK_MASKS[block.trailing_zeros() as usize]
}

/// Several independent elements of `GF(2)`, one per bit of a block of storage.
///
/// Lane `i` is bit `i mod B` of word `i / B`, counting from the least significant bit.
/// Read as bytes, that puts lane `i` at bit `i mod 8` of byte `i / 8`.
///
/// This is the order a bit witness already has on the wire, so packing one moves no bits.
///
/// The arithmetic is what the field does to one bit, applied to the whole block at once:
///
/// ```text
///     a + b   ->  a XOR b
///     a * b   ->  a AND b
///     a^2     ->  a
/// ```
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
#[must_use]
pub struct PackedGf2<U: Underlier>(U);

/// Eight elements of `GF(2)`, one per bit of a byte.
pub type PackedGf2x8 = PackedGf2<u8>;

/// Sixteen elements of `GF(2)`, one per bit of a 16-bit word.
pub type PackedGf2x16 = PackedGf2<u16>;

/// Thirty-two elements of `GF(2)`, one per bit of a 32-bit word.
pub type PackedGf2x32 = PackedGf2<u32>;

/// Sixty-four elements of `GF(2)`, one per bit of a 64-bit word.
pub type PackedGf2x64 = PackedGf2<u64>;

/// One hundred and twenty-eight elements of `GF(2)`, one per bit of a 128-bit block.
pub type PackedGf2x128 = PackedGf2<M128>;

/// Two hundred and fifty-six elements of `GF(2)`, one per bit of a 256-bit block.
pub type PackedGf2x256 = PackedGf2<M256>;

/// Five hundred and twelve elements of `GF(2)`, one per bit of a 512-bit block.
pub type PackedGf2x512 = PackedGf2<M512>;

impl<U: Underlier> PackedGf2<U> {
    /// How many independent elements one value holds.
    pub const WIDTH: usize = U::BITS;

    /// Bits in one backing word.
    pub(super) const WORD_BITS: usize = <U::Word as Word>::BITS;

    /// The packing over a block of storage, read lane for bit.
    #[inline]
    pub const fn new(bits: U) -> Self {
        Self(bits)
    }

    /// The block of storage behind the packing.
    #[inline]
    pub const fn to_bits(self) -> U {
        self.0
    }

    /// The backing words, lowest lanes first.
    #[inline]
    #[must_use]
    pub fn words(&self) -> &[U::Word] {
        self.0.words()
    }

    /// The backing words, lowest lanes first.
    ///
    /// Every bit pattern is a valid packing, so writing here cannot break an invariant.
    #[inline]
    pub fn words_mut(&mut self) -> &mut [U::Word] {
        self.0.words_mut()
    }

    /// The element in the given lane.
    ///
    /// # Panics
    /// Panics if the lane index is not below the width.
    #[inline]
    pub fn get(&self, lane: usize) -> Gf2 {
        assert!(lane < Self::WIDTH, "lane index out of range");

        // Bring the lane's bit down to the bottom of its word and keep only it.
        let word = self.0.words()[lane / Self::WORD_BITS];
        let bit = (word >> (lane % Self::WORD_BITS)) & U::Word::from_u64(1);
        Gf2::new(bit.to_u64() as u8)
    }

    /// Overwrite the element in the given lane.
    ///
    /// # Panics
    /// Panics if the lane index is not below the width.
    #[inline]
    pub fn set(&mut self, lane: usize, value: Gf2) {
        assert!(lane < Self::WIDTH, "lane index out of range");
        let index = lane / Self::WORD_BITS;
        let shift = lane % Self::WORD_BITS;

        // Clear the lane, then drop the new bit in.
        // Branch-free, so the stored bit never steers control flow.
        let word = self.0.words()[index];
        let cleared = word & !(U::Word::from_u64(1) << shift);
        let bit = U::Word::from_u64(u64::from(value.to_bit())) << shift;
        self.0.words_mut()[index] = cleared | bit;
    }

    /// Build a value lane by lane.
    #[inline]
    pub fn from_fn(mut f: impl FnMut(usize) -> Gf2) -> Self {
        let mut out = Self(U::ZERO);

        // One shift-and-insert per lane, lowest lane first.
        for lane in 0..Self::WIDTH {
            out.set(lane, f(lane));
        }
        out
    }

    /// The same element in every lane.
    #[inline]
    pub fn broadcast(value: Gf2) -> Self {
        // Negating the bit spreads it over the whole word: one becomes all ones, zero none.
        // Branch-free, so a witness bit never steers control flow.
        let word = U::Word::from_u64(0u64.wrapping_sub(u64::from(value.to_bit())));
        Self(U::from_words_fn(|_| word))
    }

    /// The lanes, lowest first.
    #[inline]
    pub fn lanes(self) -> impl Iterator<Item = Gf2> {
        (0..Self::WIDTH).map(move |lane| self.get(lane))
    }

    /// How many lanes hold one.
    #[inline]
    #[must_use]
    pub fn count_ones(&self) -> u32 {
        // Popcount is per word, so the totals add.
        self.0.words().iter().map(|word| word.count_ones()).sum()
    }

    /// The lane-wise inverse, with zero sent to zero.
    ///
    /// One is the only invertible element of `GF(2)` and it is its own inverse.
    ///
    /// Extending the map by `0^-1 = 0` therefore makes it the identity on every lane.
    #[inline]
    pub const fn invert_or_zero(self) -> Self {
        self
    }

    /// This value read as a run of narrower packings, lowest lanes first.
    ///
    /// A wide block is narrow ones side by side: lane `i` is lane `i mod w` of part `i / w`.
    #[inline]
    pub fn narrow<V: Underlier>(&self) -> &[PackedGf2<V>]
    where
        U: Divisible<V>,
    {
        wrap(self.0.parts())
    }

    /// This value read as a run of narrower packings, lowest lanes first.
    #[inline]
    pub fn narrow_mut<V: Underlier>(&mut self) -> &mut [PackedGf2<V>]
    where
        U: Divisible<V>,
    {
        wrap_mut(self.0.parts_mut())
    }

    /// A run of these packings read as a run of narrower ones, lowest lanes first.
    ///
    /// The same bytes carry the same lanes in order, so a narrower read copies nothing.
    #[inline]
    pub fn narrow_slice<V: Underlier>(slice: &[Self]) -> &[PackedGf2<V>]
    where
        U: Divisible<V>,
    {
        wrap(<U as Divisible<V>>::split_slice(unwrap(slice)))
    }

    /// A run of these packings read as a run of narrower ones, lowest lanes first.
    #[inline]
    pub fn narrow_slice_mut<V: Underlier>(slice: &mut [Self]) -> &mut [PackedGf2<V>]
    where
        U: Divisible<V>,
    {
        wrap_mut(<U as Divisible<V>>::split_slice_mut(unwrap_mut(slice)))
    }

    /// A run of these packings read as bytes, lowest lanes first.
    ///
    /// Byte `k` holds lanes `8k .. 8k + 8`, lowest lane at the lowest bit.
    #[inline]
    #[must_use]
    pub const fn as_bytes(slice: &[Self]) -> &[u8] {
        // A packing is exactly its words, so the byte count is the lane count over eight.
        //
        // A lane is a bit of a word value, and a byte index is a memory offset.
        // The two orders agree only on a little-endian target.
        const {
            assert!(size_of::<Self>() * 8 == U::BITS);
            assert!(
                cfg!(target_endian = "little"),
                "the byte view of a packing needs a little-endian target"
            );
        }
        let len = slice.len() * (U::BITS / 8);

        // SAFETY: the assertion pins the run at exactly `len` initialised bytes.
        // The byte slice borrows the same region for the same lifetime.
        unsafe { slice::from_raw_parts(slice.as_ptr().cast::<u8>(), len) }
    }

    /// Cut both operands into chunks of `block_len` lanes and interleave the chunks.
    ///
    /// Cut the stack of the two into two-by-two matrices of chunks, and transpose each:
    ///
    /// ```text
    ///     A = [x0, y0, x1, y1]
    ///     B = [x2, y2, x3, y3]
    ///
    ///     block_len = 1  ->  ([x0, x2, x1, x3], [y0, y2, y1, y3])
    /// ```
    ///
    /// A chunk as wide as the value leaves both operands untouched.
    ///
    /// This is the convention the packed-field trait fixes.
    /// So a caller can swap a bit-sliced packing for any other and see one permutation.
    ///
    /// # Panics
    /// Panics unless the chunk length is a power of two no larger than the width.
    pub fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        assert!(
            block_len.is_power_of_two() && block_len <= Self::WIDTH,
            "block length must be a power of two dividing the width"
        );

        // One chunk each means there is nothing to pair up.
        if block_len == Self::WIDTH {
            return (*self, other);
        }
        (
            self.interleave_from(other, block_len, 0),
            self.interleave_from(other, block_len, 1),
        )
    }

    /// One output value of the interleave: the first when `take` is zero, the second when one.
    ///
    /// Output chunk `c` comes from chunk `2 * (c / 2) + take`.
    /// Of the left operand when `c` is even, and of the right operand when it is odd.
    fn interleave_from(&self, other: Self, block_len: usize, take: usize) -> Self {
        let bits = Self::WORD_BITS;
        let (left, right) = (self.0.words(), other.0.words());

        Self(U::from_words_fn(|w| {
            // The first output lane this word carries.
            let first = w * bits;

            if block_len >= bits {
                // A chunk spans whole words, so the word is copied, not rearranged.
                let chunk = first / block_len;
                let source_lane = (2 * (chunk / 2) + take) * block_len + first % block_len;
                let source = if chunk.is_multiple_of(2) { left } else { right };
                source[source_lane / bits]
            } else {
                // Chunks sit inside a word.
                // This word takes every other chunk of each operand, starting at `take`.
                let mask = U::Word::from_u64(block_mask(block_len));
                let x = (left[w] >> (take * block_len)) & mask;
                let y = (right[w] >> (take * block_len)) & mask;
                x | (y << block_len)
            }
        }))
    }
}

/// A run of blocks read as a run of packings over them.
///
/// A packing is a transparent wrapper, so the two have one layout and nothing moves.
#[inline]
const fn wrap<V: Underlier>(parts: &[V]) -> &[PackedGf2<V>] {
    // SAFETY: `repr(transparent)` gives one size and alignment, and every block is valid.
    unsafe { slice::from_raw_parts(parts.as_ptr().cast(), parts.len()) }
}

/// A run of blocks read as a run of packings over them.
#[inline]
const fn wrap_mut<V: Underlier>(parts: &mut [V]) -> &mut [PackedGf2<V>] {
    // SAFETY: as above, and the exclusive borrow is not duplicated.
    unsafe { slice::from_raw_parts_mut(parts.as_mut_ptr().cast(), parts.len()) }
}

/// A run of packings read as a run of the blocks behind them.
#[inline]
const fn unwrap<V: Underlier>(packings: &[PackedGf2<V>]) -> &[V] {
    // SAFETY: the inverse of the transparent wrapping above.
    unsafe { slice::from_raw_parts(packings.as_ptr().cast(), packings.len()) }
}

/// A run of packings read as a run of the blocks behind them.
#[inline]
const fn unwrap_mut<V: Underlier>(packings: &mut [PackedGf2<V>]) -> &mut [V] {
    // SAFETY: the inverse of the transparent wrapping above.
    unsafe { slice::from_raw_parts_mut(packings.as_mut_ptr().cast(), packings.len()) }
}

impl<U: Underlier> Debug for PackedGf2<U> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // Words print in lane order, so the leftmost word holds the lowest lanes.
        write!(f, "PackedGf2x{}(", U::BITS)?;
        for (i, word) in self.0.words().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            let digits = Self::WORD_BITS / 4;
            write!(f, "{:#0width$x}", word.to_u64(), width = 2 + digits)?;
        }
        write!(f, ")")
    }
}

impl<U: Underlier> Distribution<PackedGf2<U>> for StandardUniform {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> PackedGf2<U> {
        PackedGf2(U::random(rng))
    }
}

impl<U: Underlier> From<Gf2> for PackedGf2<U> {
    #[inline]
    fn from(value: Gf2) -> Self {
        Self::broadcast(value)
    }
}

impl<U: Underlier> Add for PackedGf2<U> {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self {
        // Addition in `GF(2)` is exclusive or, lane by lane.
        Self(self.0.zip(rhs.0, |a, b| a ^ b))
    }
}

impl<U: Underlier> Sub for PackedGf2<U> {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self {
        // Subtraction coincides with addition in characteristic 2.
        self + rhs
    }
}

impl<U: Underlier> Neg for PackedGf2<U> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        // `-x = x` in characteristic 2.
        self
    }
}

impl<U: Underlier> Mul for PackedGf2<U> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // Multiplication in `GF(2)` is conjunction, lane by lane.
        Self(self.0.zip(rhs.0, |a, b| a & b))
    }
}

impl<U: Underlier> Add<Gf2> for PackedGf2<U> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Gf2) -> Self {
        self + Self::broadcast(rhs)
    }
}

impl<U: Underlier> Sub<Gf2> for PackedGf2<U> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Gf2) -> Self {
        self - Self::broadcast(rhs)
    }
}

impl<U: Underlier> Mul<Gf2> for PackedGf2<U> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Gf2) -> Self {
        self * Self::broadcast(rhs)
    }
}

impl<U: Underlier> Div<Gf2> for PackedGf2<U> {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Gf2) -> Self {
        // One is the only invertible element, and inverting zero panics.
        self * Self::broadcast(rhs.inverse())
    }
}

impl<U: Underlier> DivAssign<Gf2> for PackedGf2<U> {
    #[inline]
    fn div_assign(&mut self, rhs: Gf2) {
        *self = *self / rhs;
    }
}

impl<U: Underlier, T: Into<Self>> AddAssign<T> for PackedGf2<U> {
    #[inline]
    fn add_assign(&mut self, rhs: T) {
        *self = *self + rhs.into();
    }
}

impl<U: Underlier, T: Into<Self>> SubAssign<T> for PackedGf2<U> {
    #[inline]
    fn sub_assign(&mut self, rhs: T) {
        *self = *self - rhs.into();
    }
}

impl<U: Underlier, T: Into<Self>> MulAssign<T> for PackedGf2<U> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs.into();
    }
}

impl<U: Underlier> Sum for PackedGf2<U> {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x + y).unwrap_or(Self::ZERO)
    }
}

impl<U: Underlier> Product for PackedGf2<U> {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::ONE)
    }
}

impl<U: Underlier> PrimeCharacteristicRing for PackedGf2<U> {
    type PrimeSubfield = Gf2;

    const ZERO: Self = Self(U::ZERO);
    const ONE: Self = Self(U::ONES);
    // The characteristic is 2, so `TWO = ONE + ONE = ZERO`.
    const TWO: Self = Self(U::ZERO);
    // The characteristic is 2, so `NEG_ONE = ONE`.
    const NEG_ONE: Self = Self(U::ONES);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        Self::broadcast(f)
    }

    #[inline]
    fn from_bool(b: bool) -> Self {
        Self::broadcast(Gf2::from_bool(b))
    }

    characteristic_two_methods!();

    #[inline]
    fn square(&self) -> Self {
        // `0^2 = 0` and `1^2 = 1`, so squaring fixes every lane.
        *self
    }

    #[inline]
    fn andn(&self, y: &Self) -> Self {
        // `(1 - x) * y` is `NOT x AND y`, one instruction per word.
        Self(self.0.zip(y.0, |a, b| !a & b))
    }

    #[inline]
    fn bool_check(&self) -> Self {
        // Every element of `GF(2)` is a bit, so the booleanity residue is always zero.
        Self::ZERO
    }
}

impl<U: Underlier> Algebra<Gf2> for PackedGf2<U> {}

#[cfg(test)]
mod tests {
    use super::*;

    /// One test module per packing width.
    ///
    /// Each scenario is stated once, against a reference built from the lane accessors.
    macro_rules! packing_tests {
        ($module:ident, $alias:ident, $underlier:ty, $word:ty, $words:literal, $width:literal) => {
            mod $module {
                use alloc::vec;
                use alloc::vec::Vec;

                use p3_field::PrimeCharacteristicRing;
                use p3_field_testing::test_ring_with_eq_char2;
                use proptest::prelude::*;

                use crate::bitslice::underlier::Underlier;
                use crate::{Gf2, $alias};

                /// A packing over the given backing words, lowest lanes first.
                fn from_words(words: [$word; $words]) -> $alias {
                    $alias::new(<$underlier>::from_words_fn(|i| words[i]))
                }

                /// The lane-wise sum, built one lane at a time.
                fn reference_add(a: $alias, b: $alias) -> $alias {
                    $alias::from_fn(|i| a.get(i) + b.get(i))
                }

                /// The lane-wise product, built one lane at a time.
                fn reference_mul(a: $alias, b: $alias) -> $alias {
                    $alias::from_fn(|i| a.get(i) * b.get(i))
                }

                /// The interleave written straight from its definition.
                ///
                /// Chunk `2s` of each operand pairs with chunk `2s + 1`, transposed.
                /// The first output takes both chunks `2s`, the second both `2s + 1`.
                fn reference_interleave(a: $alias, b: $alias, block: usize) -> ($alias, $alias) {
                    if block == $width {
                        return (a, b);
                    }
                    let mut first = vec![Gf2::ZERO; $width];
                    let mut second = vec![Gf2::ZERO; $width];

                    for s in 0..($width / block / 2) {
                        for t in 0..block {
                            first[(2 * s) * block + t] = a.get((2 * s) * block + t);
                            first[(2 * s + 1) * block + t] = b.get((2 * s) * block + t);
                            second[(2 * s) * block + t] = a.get((2 * s + 1) * block + t);
                            second[(2 * s + 1) * block + t] = b.get((2 * s + 1) * block + t);
                        }
                    }
                    (
                        $alias::from_fn(|i| first[i]),
                        $alias::from_fn(|i| second[i]),
                    )
                }

                /// The bit patterns a random search is unlikely to reach.
                ///
                /// Each one drives the lane map to an extreme:
                ///
                /// ```text
                ///     all zeros     no lane set
                ///     all ones      every lane set
                ///     lane 0        the lowest bit of the first word
                ///     lane W-1      the highest bit of the last word
                /// ```
                fn specials() -> Vec<$alias> {
                    let mut lowest = $alias::ZERO;
                    lowest.set(0, Gf2::ONE);
                    let mut highest = $alias::ZERO;
                    highest.set($width - 1, Gf2::ONE);
                    vec![$alias::ZERO, $alias::ONE, lowest, highest]
                }

                #[test]
                fn lane_i_is_bit_i() {
                    // Setting one lane lights exactly one bit, at `i mod B` of word `i / B`.
                    for lane in 0..$width {
                        let mut value = $alias::ZERO;
                        value.set(lane, Gf2::ONE);

                        let bits = <$word>::BITS as usize;
                        let mut expected = [0 as $word; $words];
                        expected[lane / bits] = 1 << (lane % bits);
                        assert_eq!(value.words(), expected, "lane {lane}");

                        // And reading it back names the same lane, with the rest still zero.
                        for other in 0..$width {
                            let want = if other == lane { Gf2::ONE } else { Gf2::ZERO };
                            assert_eq!(value.get(other), want, "lane {lane}, read {other}");
                        }
                    }
                }

                #[test]
                fn lane_i_is_bit_i_mod_8_of_byte_i_div_8() {
                    // The byte view: eight lanes to the byte, lowest lane at the lowest bit.
                    //
                    //     | lane:  0 1 2 3 4 5 6 7 | 8 9 ...
                    //     | byte:  <--- byte 0 ---> <- byte 1 ...
                    for lane in 0..$width {
                        let mut value = $alias::ZERO;
                        value.set(lane, Gf2::ONE);

                        let mut expected = [0u8; $width / 8];
                        expected[lane / 8] = 1 << (lane % 8);
                        assert_eq!($alias::as_bytes(&[value]), expected, "lane {lane}");
                    }
                }

                #[test]
                fn constants_and_broadcast_agree_with_the_lane_view() {
                    // Zero is every lane clear, one is every lane set.
                    assert_eq!($alias::ZERO.count_ones(), 0);
                    assert_eq!($alias::ONE.count_ones(), $width);
                    assert_eq!($alias::WIDTH, $width);

                    // Characteristic 2: `1 + 1 = 0` and `-1 = 1`.
                    assert_eq!($alias::TWO, $alias::ZERO);
                    assert_eq!($alias::NEG_ONE, $alias::ONE);

                    // Broadcasting a scalar puts it in every lane.
                    assert_eq!($alias::broadcast(Gf2::ZERO), $alias::ZERO);
                    assert_eq!($alias::broadcast(Gf2::ONE), $alias::ONE);
                    assert_eq!($alias::from(Gf2::ONE), $alias::ONE);

                    // The lane iterator walks lanes in order, lowest first.
                    let mut alternating = $alias::ZERO;
                    for lane in (0..$width).step_by(2) {
                        alternating.set(lane, Gf2::ONE);
                    }
                    let seen: Vec<Gf2> = alternating.lanes().collect();
                    assert_eq!(seen.len(), $width);
                    for (lane, value) in seen.into_iter().enumerate() {
                        let want = if lane % 2 == 0 { Gf2::ONE } else { Gf2::ZERO };
                        assert_eq!(value, want, "lane {lane}");
                    }
                    assert_eq!(alternating.count_ones() as usize, $width / 2);
                }

                #[test]
                fn from_fn_inverts_get() {
                    // Rebuilding from its own lanes returns the same value.
                    for value in specials() {
                        assert_eq!($alias::from_fn(|i| value.get(i)), value);
                    }
                }

                #[test]
                fn every_operation_matches_the_lane_view_at_the_corners() {
                    // The kernels are lane-local, so the corners match lane by lane.
                    for a in specials() {
                        for b in specials() {
                            assert_eq!(a + b, reference_add(a, b));
                            assert_eq!(a - b, reference_add(a, b));
                            assert_eq!(a * b, reference_mul(a, b));
                            assert_eq!(a.xor(&b), reference_add(a, b));
                            assert_eq!(
                                a.andn(&b),
                                $alias::from_fn(|i| (Gf2::ONE - a.get(i)) * b.get(i))
                            );
                        }

                        // Negation, doubling and squaring are settled by the characteristic.
                        assert_eq!(-a, a);
                        assert_eq!(a.double(), $alias::ZERO);
                        assert_eq!(a.square(), a);
                        assert_eq!(a.invert_or_zero(), a);
                        assert_eq!(a.bool_check(), $alias::ZERO);

                        // Multiplying by a power of two collapses everything above the zeroth.
                        assert_eq!(a.mul_2exp_u64(0), a);
                        assert_eq!(a.mul_2exp_u64(1), $alias::ZERO);
                    }
                }

                #[test]
                fn scalar_operands_broadcast() {
                    // Mixing a scalar in must act as if it had been broadcast first.
                    for a in specials() {
                        for scalar in [Gf2::ZERO, Gf2::ONE] {
                            let broadcast = $alias::broadcast(scalar);
                            assert_eq!(a + scalar, a + broadcast);
                            assert_eq!(a - scalar, a - broadcast);
                            assert_eq!(a * scalar, a * broadcast);
                        }

                        // Dividing by one is the identity; one is the only invertible scalar.
                        assert_eq!(a / Gf2::ONE, a);
                    }
                }

                #[test]
                fn interleave_matches_the_reference_at_every_block_size() {
                    // Distinguishable lanes, so a misplaced one shows:
                    //
                    //     | a: lane i set iff i is even
                    //     | b: lane i set iff i is a multiple of three
                    let a = $alias::from_fn(|i| Gf2::from_bool(i % 2 == 0));
                    let b = $alias::from_fn(|i| Gf2::from_bool(i % 3 == 0));

                    let mut block = 1;
                    while block <= $width {
                        assert_eq!(
                            a.interleave(b, block),
                            reference_interleave(a, b, block),
                            "block length {block}"
                        );
                        block *= 2;
                    }

                    // A block as wide as the value leaves both operands where they were.
                    assert_eq!(a.interleave(b, $width), (a, b));
                }

                #[test]
                #[should_panic = "lane index out of range"]
                fn get_rejects_a_lane_beyond_the_width() {
                    let _lane = $alias::ZERO.get($width);
                }

                #[test]
                #[should_panic = "lane index out of range"]
                fn set_rejects_a_lane_beyond_the_width() {
                    let mut value = $alias::ZERO;
                    value.set($width, Gf2::ONE);
                }

                #[test]
                #[should_panic = "block length must be a power of two dividing the width"]
                fn interleave_rejects_a_block_wider_than_the_value() {
                    let _pair = $alias::ZERO.interleave($alias::ZERO, 2 * $width);
                }

                proptest! {
                    #![proptest_config(ProptestConfig::with_cases(64))]

                    #[test]
                    fn operations_match_the_lane_view(
                        left in prop::array::uniform::<_, $words>(any::<$word>()),
                        right in prop::array::uniform::<_, $words>(any::<$word>()),
                    ) {
                        let a = from_words(left);
                        let b = from_words(right);

                        // Every binary operation is the scalar one applied lane by lane.
                        prop_assert_eq!(a + b, reference_add(a, b));
                        prop_assert_eq!(a - b, reference_add(a, b));
                        prop_assert_eq!(a * b, reference_mul(a, b));
                        prop_assert_eq!(a.xor(&b), reference_add(a, b));

                        // And every unary one is fixed by the characteristic.
                        prop_assert_eq!(-a, a);
                        prop_assert_eq!(a.square(), a);
                        prop_assert_eq!(a.double(), $alias::ZERO);

                        // Words in, words out.
                        prop_assert_eq!(a.words(), left);

                        // Popcount over the lanes is popcount over the words.
                        let set = (0..$width).filter(|&i| a.get(i) == Gf2::ONE).count();
                        prop_assert_eq!(a.count_ones() as usize, set);
                    }

                    #[test]
                    fn interleave_matches_the_reference(
                        left in prop::array::uniform::<_, $words>(any::<$word>()),
                        right in prop::array::uniform::<_, $words>(any::<$word>()),
                        log_block in 0usize..=($width as usize).trailing_zeros() as usize,
                    ) {
                        let a = from_words(left);
                        let b = from_words(right);
                        let block = 1 << log_block;

                        prop_assert_eq!(
                            a.interleave(b, block),
                            reference_interleave(a, b, block)
                        );
                    }
                }

                test_ring_with_eq_char2!(
                    crate::$alias,
                    &[crate::$alias::ZERO],
                    &[crate::$alias::ONE]
                );
            }
        };
    }

    packing_tests!(x8, PackedGf2x8, u8, u8, 1, 8);
    packing_tests!(x16, PackedGf2x16, u16, u16, 1, 16);
    packing_tests!(x32, PackedGf2x32, u32, u32, 1, 32);
    packing_tests!(x64, PackedGf2x64, u64, u64, 1, 64);
    packing_tests!(x128, PackedGf2x128, crate::M128, u64, 2, 128);
    packing_tests!(x256, PackedGf2x256, crate::M256, u64, 4, 256);
    packing_tests!(x512, PackedGf2x512, crate::M512, u64, 8, 512);

    #[test]
    fn interleave_follows_the_packed_field_convention() {
        // Invariant: the permutation the packed-field trait fixes for every packing.
        //
        // Output chunk `c` is chunk `2 * (c / 2)` of the left operand when `c` is even.
        // Of the right operand when `c` is odd.
        //
        // The second output takes the odd source chunks the same way.
        //
        // The expectations below are worked out by hand from those sentences.
        // So they share no model with the implementation.
        //
        //     | a = 0xd3 = lanes 0 1 . . 4 . 6 7
        //     | b = 0x2c = lanes . . 2 3 . 5 . .
        let a = PackedGf2x8::new(0xd3);
        let b = PackedGf2x8::new(0x2c);

        // Chunks of one lane:
        //
        //     | first  = a0 b0 a2 b2 a4 b4 a6 b6 = 1 0 0 1 1 0 1 0 = 0x59
        //     | second = a1 b1 a3 b3 a5 b5 a7 b7 = 1 0 0 1 0 1 1 0 = 0x69
        assert_eq!(pair(a.interleave(b, 1)), (0x59, 0x69));

        // Chunks of two lanes, so whole pairs move:
        //
        //     | first  = a[0..2] b[0..2] a[4..6] b[4..6] = 11 00 10 01 = 0x93
        //     | second = a[2..4] b[2..4] a[6..8] b[6..8] = 00 11 11 00 = 0x3c
        assert_eq!(pair(a.interleave(b, 2)), (0x93, 0x3c));

        // Chunks of four lanes: one half of each operand into each output.
        //
        //     | first  = a[0..4] b[0..4] = 1100 0011 = 0xc3
        //     | second = a[4..8] b[4..8] = 1011 0100 = 0x2d
        assert_eq!(pair(a.interleave(b, 4)), (0xc3, 0x2d));

        // One chunk each leaves both operands alone.
        assert_eq!(pair(a.interleave(b, 8)), (0xd3, 0x2c));
    }

    /// The backing bytes of an interleaved pair, for comparison against literals.
    fn pair(values: (PackedGf2x8, PackedGf2x8)) -> (u8, u8) {
        (values.0.words()[0], values.1.words()[0])
    }

    #[test]
    fn widths_are_the_advertised_sizes() {
        // One bit per lane and nothing else, aligned to the register that holds it:
        //
        //     | width   bytes   alignment
        //     |     8       1        1
        //     |   512      64       64
        assert_eq!(size_of::<PackedGf2x8>(), PackedGf2x8::WIDTH / 8);
        assert_eq!(size_of::<PackedGf2x16>(), PackedGf2x16::WIDTH / 8);
        assert_eq!(size_of::<PackedGf2x32>(), PackedGf2x32::WIDTH / 8);
        assert_eq!(size_of::<PackedGf2x64>(), PackedGf2x64::WIDTH / 8);
        assert_eq!(size_of::<PackedGf2x128>(), PackedGf2x128::WIDTH / 8);
        assert_eq!(size_of::<PackedGf2x256>(), PackedGf2x256::WIDTH / 8);
        assert_eq!(size_of::<PackedGf2x512>(), PackedGf2x512::WIDTH / 8);

        assert_eq!(align_of::<PackedGf2x128>(), 16);
        assert_eq!(align_of::<PackedGf2x256>(), 32);
        assert_eq!(align_of::<PackedGf2x512>(), 64);
    }

    #[test]
    fn a_wide_packing_is_a_run_of_narrow_ones() {
        // Invariant: narrowing moves no bits, so a 512-lane value splits as
        //
        //     | lane 0    -> 64-lane packing 0, lane 0
        //     | lane 65   -> 64-lane packing 1, lane 1
        //     | lane 511  -> 64-lane packing 7, lane 63
        let mut wide = PackedGf2x512::ZERO;
        for lane in [0usize, 65, 511] {
            wide.set(lane, Gf2::ONE);
        }

        let narrow: &[PackedGf2x64] = wide.narrow();
        assert_eq!(narrow.len(), 8);
        assert_eq!(narrow[0].get(0), Gf2::ONE);
        assert_eq!(narrow[1].get(1), Gf2::ONE);
        assert_eq!(narrow[7].get(63), Gf2::ONE);
        assert_eq!(
            narrow.iter().map(PackedGf2x64::count_ones).sum::<u32>(),
            wide.count_ones()
        );

        // Every intermediate width sees the same lanes in the same order.
        let bytes: &[PackedGf2x8] = wide.narrow();
        assert_eq!(bytes.len(), 64);
        assert_eq!(bytes[0].get(0), Gf2::ONE);
        assert_eq!(bytes[8].get(1), Gf2::ONE);
        assert_eq!(bytes[63].get(7), Gf2::ONE);

        // Writing through a narrow view writes the wide value.
        wide.narrow_mut::<u64>()[3].set(5, Gf2::ONE);
        assert_eq!(wide.get(3 * 64 + 5), Gf2::ONE);
    }

    #[test]
    fn a_run_of_wide_packings_is_a_run_of_narrow_ones() {
        // The same relation across a whole buffer, which is what makes a re-read free.
        //
        //     | 512 lanes  ->  8 packings of 64 lanes  ->  64 packings of 8 lanes
        let wide: [PackedGf2x256; 2] = core::array::from_fn(|w| {
            PackedGf2x256::from_fn(|i| Gf2::from_bool((w * 256 + i) % 5 == 0))
        });

        let narrow: &[PackedGf2x64] = PackedGf2x256::narrow_slice(&wide);
        assert_eq!(narrow.len(), 8);
        for lane in 0..512 {
            let want = Gf2::from_bool(lane % 5 == 0);
            assert_eq!(narrow[lane / 64].get(lane % 64), want, "lane {lane}");
        }

        // And the byte view of the same buffer agrees with the lane order.
        let bytes = PackedGf2x256::as_bytes(&wide);
        assert_eq!(bytes.len(), 64);
        for lane in 0..512 {
            let bit = (bytes[lane / 8] >> (lane % 8)) & 1;
            assert_eq!(bit == 1, lane % 5 == 0, "lane {lane}");
        }
    }

    #[test]
    fn debug_prints_words_in_lane_order() {
        // The lowest lanes print first, so the text reads in lane order:
        //
        //     | lane 0 set   -> the first word ends in one
        //     | lane 64 set  -> the second word ends in one
        let mut value = PackedGf2x128::ZERO;
        value.set(0, Gf2::ONE);
        value.set(64, Gf2::ONE);
        assert_eq!(
            alloc::format!("{value:?}"),
            "PackedGf2x128(0x0000000000000001, 0x0000000000000001)"
        );
    }

    #[test]
    #[should_panic = "block out of range"]
    fn block_mask_rejects_a_block_wider_than_half_a_word() {
        let _mask = block_mask(64);
    }
}
