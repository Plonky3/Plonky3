use core::ops::{Add, AddAssign, Mul, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign};
use crate::field::PrimeField32;

// A collection of methods able to be appied to simple integers.
pub trait IntegerLike:
    Sized
    + Default
    + Copy
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + ShrAssign<usize>
    + Shr<usize, Output = Self>
    + ShlAssign<usize>
    + Shl<usize, Output = Self>
{
}

/// A implementation of Prime Fields where values are stored Non Canonically.
/// This allows for more control over when to perform modulo reductions.
/// Potentially should come with an "unsafe" label as improper use will lead to wraparound and cause errors.
/// Ensuring algorithms are correct is entirely left to the programmer.
/// Should only be used with small fields.
pub trait NonCanonicalPrimeField32: IntegerLike + Mul<Output = i128> // Multiplication of 2 field elements will yeild an i128.
{
    /// The order of the field.
    const ORDER_U32: u32;

    /// Return the internal field element value
    fn to_i64(self) -> i64;

    /// Produce a new non canonical field element
    fn from_i64(input: i64) -> Self;

    /// Produce a new non canonical field element from some other types
    #[inline]
    fn from_i32(input: i32) -> Self {
        Self::from_i64(input as i64)
    }

    #[inline]
    fn from_u32(input: u32) -> Self {
        Self::from_i64(input as i64)
    }

    /// Return the zero of the Field
    #[inline]
    fn zero() -> Self {
        Self::from_i64(0)
    }

    /// Given x, an i128 representing a field element with |x| < 2**80 computes x' satisfying:
    /// |x'| < 2**50
    /// x' = x mod p
    /// x' = x mod 2^10
    /// This is important for large convolutions.
    /// UNSAFE as if |x| > 2**80 there are no garuntees regarding the answer.
    unsafe fn from_small_i128(input: i128) -> Self;

    /// If we are sure a product will not overflow we don't need to pass to i128s.
    /// UNSAFE as there are no garuntees on correctness if overflow occurs.
    #[inline]
    unsafe fn mul_small(lhs: Self, rhs: Self) -> Self {
        Self::from_i64(Self::to_i64(lhs) * Self::to_i64(rhs))
    }

    /// If we are sure a dot product will not overflow we don't need to pass to i128s.
    /// UNSAFE as there are no garuntees on correctness if overflow occurs.
    #[inline]
    unsafe fn dot_small(lhs: &[Self], rhs: &[Self]) -> Self {
        debug_assert_eq!(lhs.len(), rhs.len());

        let n = lhs.len();

        let mut output = Self::mul_small(lhs[0], rhs[0]);

        for i in 1..n {
            output += Self::mul_small(lhs[i], rhs[i]);
        }

        output
    }

    /// Compute a dot product, passing to i128's to prevent overflow.
    /// UNSAFE as there are no garuntees on correctness if overflow of i128s occurs.
    #[inline]
    unsafe fn dot_large(lhs: &[Self], rhs: &[Self]) -> i128 {
        debug_assert_eq!(lhs.len(), rhs.len());

        let n = lhs.len();

        let mut output = lhs[0]*rhs[0];

        for i in 1..n {
            output += lhs[i]*rhs[i];
        }

        output
    }
}

/// This lets us pass from our non Canonical representatives back to canonical field elements.
/// We implement a couple of different methods to be used in different situtations.
pub trait Canonicalize<Base: PrimeField32>: NonCanonicalPrimeField32 + 
    Mul<Base, Output = Self> // Given a non canonical field element and an element of the base field we implement a product such that |output| < max(2^40, |non canonical input|)
{
    /// Given a generic non canonical field element, produce a canonical one.
    fn to_canonical(self) -> Base;

    /// Canonical elements embed into the field in the obvious way.
    fn from_canonical(val: Base) -> Self;

    /// Given an element x in a 31 bit field, return the unique element x' = x mod P with |x'| < 2**30.
    /// This can be handy to prevent overflow.
    fn from_canonical_to_i31(val: Base) -> Self;

    /// Given a non canonical field element garunteed to be < n < 64 bits, produce a canonical one.
    /// The precise value of n will depend on the field. Should be faster than to_canonical for some fields.
    fn to_canonical_i_small(self) -> Base;

    /// Given a non canonical field element garunteed to be positive and < n < 64 bits, produce a canonical one.
    /// The precise value of n will depend on the field. Should be faster than to_canonical for some fields.
    fn to_canonical_u_small(self) -> Base;
}