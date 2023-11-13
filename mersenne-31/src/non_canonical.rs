use core::ops::{Add, AddAssign, Mul, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign};
use crate::{Mersenne31};
use p3_field::{AbstractField, PrimeField32, IntegerLike, NonCanonicalPrimeField32, Canonicalize};

/// Roughly a wrapper for i64's but treated as non canonical representatives of elements in the Mersenne31 field.
#[derive(Debug, Copy, Clone, Default, Eq, Hash, PartialEq)]
pub struct Mersenne31NonCanonical {
    value: i64,
}

impl Mersenne31NonCanonical {
    /// create a new `Mersenne31NonCanonical` from an `i64`.
    #[inline]
    pub(crate) const fn new(n: i64) -> Self {
        Self { value: n }
    }
}

impl Add for Mersenne31NonCanonical {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            value: self.value + rhs.value,
        }
    }
}

impl AddAssign for Mersenne31NonCanonical {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Mersenne31NonCanonical {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            value: self.value - rhs.value,
        }
    }
}

impl SubAssign for Mersenne31NonCanonical {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Shl<usize> for Mersenne31NonCanonical {
    type Output = Self;

    #[inline]
    fn shl(self, rhs: usize) -> Self {
        Self {
            value: self.value << rhs,
        }
    }
}

impl ShlAssign<usize> for Mersenne31NonCanonical {
    #[inline]
    fn shl_assign(&mut self, rhs: usize) {
        *self = *self << rhs;
    }
}

impl Shr<usize> for Mersenne31NonCanonical {
    type Output = Self;

    #[inline]
    fn shr(self, rhs: usize) -> Self {
        Self {
            value: self.value >> rhs,
        }
    }
}

impl ShrAssign<usize> for Mersenne31NonCanonical {
    #[inline]
    fn shr_assign(&mut self, rhs: usize) {
        *self = *self >> rhs;
    }
}

// Multiplying 2 generic elements of this field will result in an i128.
impl Mul for Mersenne31NonCanonical {
    type Output = i128;

    #[inline]
    fn mul(self, rhs: Self) -> i128 {
        (self.value as i128) * (rhs.value as i128)
    }
}

// Given a non canonical field element and an element of the base field we implement a product such that |output| < max(2^40, |non canonical input|)
impl Mul<Mersenne31> for Mersenne31NonCanonical {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Mersenne31) -> Self {
        const LOWMASK: i128 = (1 << 31) - 1; // Gets the bits lower than 31.
        let val = (self.value as i128) * (rhs.value as i128); // Output has is at most 96 bits.

        let small = (val & LOWMASK) as i64; // This has 31 bits.
        let mid = ((val >> 31) & LOWMASK) as i64;  // This has 31 bits.
        let high = (val >> 62) as i64;  // This has 34 bits.

        return Self::from_i64(small + mid + high)
    }
}

impl IntegerLike for Mersenne31NonCanonical {}

impl NonCanonicalPrimeField32 for Mersenne31NonCanonical {
    const ORDER_U32: u32 = (1 << 31) - 1; // Mersenne31 Prime

    #[inline]
    fn to_i64(self) -> i64 {
        self.value
    }

    #[inline]
    fn from_i64(input: i64) -> Self {
        Self::new(input)
    }

    /// Given x, an i128 representing a field element with |x| < 2**80 computes x' satisfying:
    /// |x'| < 2**50
    /// x' = x mod p
    /// x' = x mod 2^10
    #[inline]
    fn from_small_i128(input: i128) -> Self {
        const LOWMASK: i128 = (1 << 42) - 1; // Gets the bits lower than 42.
        const HIGHMASK: i128 = !(LOWMASK); // Gets all bits higher than 42.

        let low_bits = (input & LOWMASK) as i64; // low_bits < 2**42
        let high_bits = ((input & HIGHMASK) >> 31) as i64; // |high_bits| < 2**(n - 31)

        // We quickly prove that low_bits + high_bits is what we want.

        // The individual bounds clearly show that low_bits + high_bits < 2**(n - 30).
        // Next observe that low_bits + high_bits = input - (2**31 - 1)(high_bits) = input mod P.
        // Finally note that 2**11 divides high_bits and so low_bits + high_bits = low_bits mod 2**11 = input mod 2**11.

        Self::new(low_bits + high_bits)
    }
}

impl Canonicalize<Mersenne31> for Mersenne31NonCanonical {
    #[inline]
    fn to_canonical(self) -> Mersenne31 {
        todo!()
    }

    #[inline]
    fn from_canonical(input: Mersenne31) -> Self {
        Self::from_u32(input.as_canonical_u32())
    }

    #[inline]
    fn from_canonical_to_i31(input: Mersenne31) -> Self {
        let val = input.as_canonical_u32();
        if val & (1 << 30) != 0 {
            Self::from_u32(val)
        } else {
            Self::from_i32((val as i32) - (Mersenne31::ORDER_U32 as i32))
        }
    }

    /// Reduces an i64 in the range -2^61 <= x < 2^61 to its (almost) canonical representative mod 2^31 - 1.
    /// Not technically canonical as we allow P as an output.
    #[inline]
    fn to_canonical_i_small(self) -> Mersenne31 {
        debug_assert!((-(1 << 61)..(1 << 61)).contains(&self.value));

        const MASK: i64 = (1 << 31) - 1;

        // Morally, our value is a i62 not a i64 as the top 3 bits are garunteed to be equal.
        let low_bits = Mersenne31::from_canonical_u32((self.value & MASK) as u32); // Get the bottom 31 bits, 0 <= low_bits < 2**31.
        let high_bits = ((self.value >> 31) & MASK) as i32; // Get the top 31 bits. 0 <= high_bits < 2**31.
        let sign_bits = (self.value >> 62) as i32; // sign_bits = 0 or -1

        // Note that high_bits + sign_bits > 0 as by assumption b[63] = b[61].

        let high = Mersenne31::from_canonical_u32((high_bits + sign_bits) as u32); // 0 <= high <= P so we can do our usual algorithm from here.

        low_bits + high
    }

    /// Reduces an i64 in the range 0 <= x < 2^62 to its (almost) canonical representative mod 2^31 - 1.
    /// Not technically canonical as we allow P as an output.
    #[inline]
    fn to_canonical_u_small(self) -> Mersenne31 {
        debug_assert!((0..(1 << 62)).contains(&self.value));

        const MASK: i64 = (1 << 31) - 1;

        let low_bits = Mersenne31::from_canonical_u32((self.value & MASK) as u32); // Get the bottom 31 bits, 0 <= low_bits <= P
        let high_bits = Mersenne31::from_canonical_u32((self.value >> 31) as u32); // Get the top 31 bits, 0 <= high_bits <= P.

        low_bits + high_bits
    }
}