use core::fmt::Debug;
use core::hash::Hash;

pub(crate) mod sealed {
    /// Internal arithmetic needed by the scalar relation evaluator.
    pub trait Sealed: Sized {
        fn xor(self, rhs: Self) -> Self;
        fn and(self, rhs: Self) -> Self;
        fn logical_left(self, amount: u32) -> Self;
        fn logical_right(self, amount: u32) -> Self;
        fn arithmetic_right(self, amount: u32) -> Self;
        fn rotate_right(self, amount: u32) -> Self;
        fn lane32_left(self, amount: u32) -> Option<Self>;
        fn lane32_right(self, amount: u32) -> Option<Self>;
        fn lane32_arithmetic_right(self, amount: u32) -> Option<Self>;
        fn lane32_rotate_right(self, amount: u32) -> Option<Self>;
        fn wide_mul(self, rhs: Self) -> (Self, Self);
    }
}

/// A native word supported by the relation language.
///
/// The trait is sealed so protocol code can rely on exactly two widths.
pub trait Word: sealed::Sealed + Copy + Debug + Eq + Hash + Send + Sync + 'static {
    /// The number of bits in one word.
    const BITS: u32;

    /// The additive identity for XOR operands.
    const ZERO: Self;

    /// Returns the word as an unsigned 64-bit integer.
    fn to_u64(self) -> u64;

    /// Creates a word from the low bits of an unsigned representation.
    fn from_low_bits(value: u64) -> Self;
}

/// A 32-bit word.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Word32(
    /// The underlying 32-bit pattern.
    u32,
);

impl Word32 {
    /// Creates a word from its bit pattern.
    #[inline]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the underlying bit pattern.
    #[inline]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl Word for Word32 {
    const BITS: u32 = u32::BITS;
    const ZERO: Self = Self(0);

    #[inline]
    fn to_u64(self) -> u64 {
        self.0.into()
    }

    #[inline]
    fn from_low_bits(value: u64) -> Self {
        // Truncation retains exactly the low 32 bits named by the word width.
        Self(value as u32)
    }
}

impl sealed::Sealed for Word32 {
    #[inline]
    fn xor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }

    #[inline]
    fn and(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }

    #[inline]
    fn logical_left(self, amount: u32) -> Self {
        Self(self.0 << amount)
    }

    #[inline]
    fn logical_right(self, amount: u32) -> Self {
        Self(self.0 >> amount)
    }

    #[inline]
    fn arithmetic_right(self, amount: u32) -> Self {
        Self(((self.0 as i32) >> amount) as u32)
    }

    #[inline]
    fn rotate_right(self, amount: u32) -> Self {
        Self(self.0.rotate_right(amount))
    }

    #[inline]
    fn lane32_left(self, _amount: u32) -> Option<Self> {
        None
    }

    #[inline]
    fn lane32_right(self, _amount: u32) -> Option<Self> {
        None
    }

    #[inline]
    fn lane32_arithmetic_right(self, _amount: u32) -> Option<Self> {
        None
    }

    #[inline]
    fn lane32_rotate_right(self, _amount: u32) -> Option<Self> {
        None
    }

    #[inline]
    fn wide_mul(self, rhs: Self) -> (Self, Self) {
        let product = u64::from(self.0) * u64::from(rhs.0);
        (Self(product as u32), Self((product >> 32) as u32))
    }
}

/// A 64-bit word.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Word64(
    /// The underlying 64-bit pattern.
    u64,
);

impl Word64 {
    /// Creates a word from its bit pattern.
    #[inline]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the underlying bit pattern.
    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl Word for Word64 {
    const BITS: u32 = u64::BITS;
    const ZERO: Self = Self(0);

    #[inline]
    fn to_u64(self) -> u64 {
        self.0
    }

    #[inline]
    fn from_low_bits(value: u64) -> Self {
        // Every input bit fits the 64-bit representation.
        Self(value)
    }
}

impl sealed::Sealed for Word64 {
    #[inline]
    fn xor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }

    #[inline]
    fn and(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }

    #[inline]
    fn logical_left(self, amount: u32) -> Self {
        Self(self.0 << amount)
    }

    #[inline]
    fn logical_right(self, amount: u32) -> Self {
        Self(self.0 >> amount)
    }

    #[inline]
    fn arithmetic_right(self, amount: u32) -> Self {
        Self(((self.0 as i64) >> amount) as u64)
    }

    #[inline]
    fn rotate_right(self, amount: u32) -> Self {
        Self(self.0.rotate_right(amount))
    }

    #[inline]
    fn lane32_left(self, amount: u32) -> Option<Self> {
        let low = (self.0 as u32) << amount;
        let high = ((self.0 >> 32) as u32) << amount;
        Some(Self(u64::from(low) | (u64::from(high) << 32)))
    }

    #[inline]
    fn lane32_right(self, amount: u32) -> Option<Self> {
        let low = (self.0 as u32) >> amount;
        let high = ((self.0 >> 32) as u32) >> amount;
        Some(Self(u64::from(low) | (u64::from(high) << 32)))
    }

    #[inline]
    fn lane32_arithmetic_right(self, amount: u32) -> Option<Self> {
        let low = ((self.0 as u32 as i32) >> amount) as u32;
        let high = (((self.0 >> 32) as u32 as i32) >> amount) as u32;
        Some(Self(u64::from(low) | (u64::from(high) << 32)))
    }

    #[inline]
    fn lane32_rotate_right(self, amount: u32) -> Option<Self> {
        let low = (self.0 as u32).rotate_right(amount);
        let high = ((self.0 >> 32) as u32).rotate_right(amount);
        Some(Self(u64::from(low) | (u64::from(high) << 32)))
    }

    #[inline]
    fn wide_mul(self, rhs: Self) -> (Self, Self) {
        let product = u128::from(self.0) * u128::from(rhs.0);
        (Self(product as u64), Self((product >> 64) as u64))
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        #[test]
        fn low_bit_conversion_matches_each_word_width(value in any::<u64>()) {
            // The generic constructor makes its truncation rule explicit for narrow words.
            prop_assert_eq!(Word32::from_low_bits(value), Word32::new(value as u32));
            prop_assert_eq!(Word64::from_low_bits(value), Word64::new(value));
        }

        #[test]
        fn word32_integer_product_matches_u64(left in any::<u32>(), right in any::<u32>()) {
            // A 64-bit product contains both 32-bit result limbs.
            let product = u64::from(left) * u64::from(right);
            let actual = sealed::Sealed::wide_mul(Word32::new(left), Word32::new(right));

            prop_assert_eq!(
                actual,
                (
                    Word32::new(product as u32),
                    Word32::new((product >> 32) as u32),
                )
            );
        }

        #[test]
        fn word64_integer_product_matches_u128(left in any::<u64>(), right in any::<u64>()) {
            // A 128-bit product contains both 64-bit result limbs.
            let product = u128::from(left) * u128::from(right);
            let actual = sealed::Sealed::wide_mul(Word64::new(left), Word64::new(right));

            prop_assert_eq!(
                actual,
                (
                    Word64::new(product as u64),
                    Word64::new((product >> 64) as u64),
                )
            );
        }
    }
}
