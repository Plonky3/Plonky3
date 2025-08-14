use alloc::vec::Vec;
use core::arch::aarch64::{self, uint32x4_t};
use core::iter::{Product, Sum};
use core::mem::transmute;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::exponentiation::exp_1717986917;
use p3_field::interleave::{interleave_u32, interleave_u64};
use p3_field::op_assign_macros::{
    impl_add_assign, impl_add_base_field, impl_div_methods, impl_mul_base_field, impl_mul_methods,
    impl_packed_value, impl_rng, impl_sub_assign, impl_sub_base_field, impl_sum_prod_base_field,
    ring_sum,
};
use p3_field::{
    Algebra, Field, InjectiveMonomial, PackedField, PackedFieldPow2, PackedValue,
    PermutationMonomial, PrimeCharacteristicRing, impl_packed_field_pow_2, uint32x4_mod_add,
    uint32x4_mod_sub,
};
use p3_util::reconstitute_from_base;
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};

use crate::Mersenne31;

const WIDTH: usize = 4;
const P: uint32x4_t = unsafe { transmute::<[u32; WIDTH], _>([0x7fffffff; WIDTH]) };

/// Vectorized NEON implementation of `Mersenne31` arithmetic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)] // Needed to make `transmute`s safe.
#[must_use]
pub struct PackedMersenne31Neon(pub [Mersenne31; WIDTH]);

impl PackedMersenne31Neon {
    #[inline]
    #[must_use]
    /// Get an arch-specific vector representing the packed values.
    fn to_vector(self) -> uint32x4_t {
        unsafe {
            // Safety: `Mersenne31` is `repr(transparent)` so it can be transmuted to `u32`. It
            // follows that `[Mersenne31; WIDTH]` can be transmuted to `[u32; WIDTH]`, which can be
            // transmuted to `uint32x4_t`, since arrays are guaranteed to be contiguous in memory.
            // Finally `PackedMersenne31Neon` is `repr(transparent)` so it can be transmuted to
            // `[Mersenne31; WIDTH]`.
            transmute(self)
        }
    }

    #[inline]
    /// Make a packed field vector from an arch-specific vector.
    ///
    /// SAFETY: The caller must ensure that each element of `vector` represents a valid
    /// `Mersenne31`.  In particular, each element of vector must be in `0..=P` (i.e. it fits in 31
    /// bits).
    unsafe fn from_vector(vector: uint32x4_t) -> Self {
        // Safety: It is up to the user to ensure that elements of `vector` represent valid
        // `Mersenne31` values. We must only reason about memory representations. `uint32x4_t` can
        // be transmuted to `[u32; WIDTH]` (since arrays elements are contiguous in memory), which
        // can be transmuted to `[Mersenne31; WIDTH]` (since `Mersenne31` is `repr(transparent)`),
        // which in turn can be transmuted to `PackedMersenne31Neon` (since `PackedMersenne31Neon`
        // is also `repr(transparent)`).
        unsafe { transmute(vector) }
    }

    /// Copy `value` to all positions in a packed vector. This is the same as
    /// `From<Mersenne31>::from`, but `const`.
    #[inline]
    const fn broadcast(value: Mersenne31) -> Self {
        Self([value; WIDTH])
    }
}

impl From<Mersenne31> for PackedMersenne31Neon {
    #[inline]
    fn from(value: Mersenne31) -> Self {
        Self::broadcast(value)
    }
}

impl Add for PackedMersenne31Neon {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let lhs = self.to_vector();
        let rhs = rhs.to_vector();
        let res = uint32x4_mod_add(lhs, rhs, P);
        unsafe {
            // Safety: `uint32x4_mod_add` returns valid values when given valid values.
            Self::from_vector(res)
        }
    }
}

impl Sub for PackedMersenne31Neon {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let lhs = self.to_vector();
        let rhs = rhs.to_vector();
        let res = uint32x4_mod_sub(lhs, rhs, P);
        unsafe {
            // Safety: `uint32x4_mod_sub` returns valid values when given valid values.
            Self::from_vector(res)
        }
    }
}

impl Neg for PackedMersenne31Neon {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let val = self.to_vector();
        let res = neg(val);
        unsafe {
            // Safety: `neg` returns valid values when given valid values.
            Self::from_vector(res)
        }
    }
}

impl Mul for PackedMersenne31Neon {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let lhs = self.to_vector();
        let rhs = rhs.to_vector();
        let res = mul(lhs, rhs);
        unsafe {
            // Safety: `mul` returns valid values when given valid values.
            Self::from_vector(res)
        }
    }
}

impl_add_assign!(PackedMersenne31Neon);
impl_sub_assign!(PackedMersenne31Neon);
impl_mul_methods!(PackedMersenne31Neon);
ring_sum!(PackedMersenne31Neon);
impl_rng!(PackedMersenne31Neon);

impl PrimeCharacteristicRing for PackedMersenne31Neon {
    type PrimeSubfield = Mersenne31;

    const ZERO: Self = Self::broadcast(Mersenne31::ZERO);
    const ONE: Self = Self::broadcast(Mersenne31::ONE);
    const TWO: Self = Self::broadcast(Mersenne31::TWO);
    const NEG_ONE: Self = Self::broadcast(Mersenne31::NEG_ONE);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        f.into()
    }

    // TODO: Add a custom implementation of `halve` that uses NEON intrinsics
    // and avoids the multiplication.

    #[inline(always)]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY: this is a repr(transparent) wrapper around an array.
        unsafe { reconstitute_from_base(Mersenne31::zero_vec(len * WIDTH)) }
    }
}

// Degree of the smallest permutation polynomial for Mersenne31.
//
// As p - 1 = 2×3^2×7×11×... the smallest choice for a degree D satisfying gcd(p - 1, D) = 1 is 5.
impl InjectiveMonomial<5> for PackedMersenne31Neon {}

impl PermutationMonomial<5> for PackedMersenne31Neon {
    /// In the field `Mersenne31`, `a^{1/5}` is equal to a^{1717986917}.
    ///
    /// This follows from the calculation `5 * 1717986917 = 4*(2^31 - 2) + 1 = 1 mod p - 1`.
    fn injective_exp_root_n(&self) -> Self {
        exp_1717986917(*self)
    }
}

impl_add_base_field!(PackedMersenne31Neon, Mersenne31);
impl_sub_base_field!(PackedMersenne31Neon, Mersenne31);
impl_mul_base_field!(PackedMersenne31Neon, Mersenne31);
impl_div_methods!(PackedMersenne31Neon, Mersenne31);
impl_sum_prod_base_field!(PackedMersenne31Neon, Mersenne31);

impl Algebra<Mersenne31> for PackedMersenne31Neon {}

/// Given a `val` in `0, ..., 2 P`, return a `res` in `0, ..., P` such that `res = val (mod P)`
#[inline]
#[must_use]
fn reduce_sum(val: uint32x4_t) -> uint32x4_t {
    // val is in 0, ..., 2 P. If val is in 0, ..., P - 1 then it is valid and
    // u := (val - P) mod 2^32 is in P <u 2^32 - P, ..., 2^32 - 1 and unsigned_min(val, u) = val as
    // desired. If val is in P + 1, ..., 2 P, then u is in 1, ..., P < P + 1 so u is valid, and
    // unsigned_min(val, u) = u as desired. The remaining case of val = P, u = 0 is trivial.

    unsafe {
        // Safety: If this code got compiled then NEON intrinsics are available.
        let u = aarch64::vsubq_u32(val, P);
        aarch64::vminq_u32(val, u)
    }
}

/// Multiply two 31-bit numbers to obtain a 62-bit immediate result, and return the high 31 bits of
/// that result. Results are arbitrary if the inputs do not fit in 31 bits.
#[inline]
#[must_use]
fn mul_31x31_to_hi_31(lhs: uint32x4_t, rhs: uint32x4_t) -> uint32x4_t {
    // This is just a wrapper around `aarch64::vqdmulhq_s32`, so we don't have to worry about the
    // casting elsewhere.
    unsafe {
        // Safety: If this code got compiled then NEON intrinsics are available.
        aarch64::vreinterpretq_u32_s32(aarch64::vqdmulhq_s32(
            aarch64::vreinterpretq_s32_u32(lhs),
            aarch64::vreinterpretq_s32_u32(rhs),
        ))
    }
}

/// Multiply vectors of Mersenne-31 field elements that fit in 31 bits.
/// If the inputs do not fit in 31 bits, the result is undefined.
#[inline]
#[must_use]
fn mul(lhs: uint32x4_t, rhs: uint32x4_t) -> uint32x4_t {
    // We want this to compile to:
    //      sqdmulh  prod_hi31.4s, lhs.4s, rhs.4s
    //      mul      t.4s, lhs.4s, rhs.4s
    //      mla      t.4s, prod_hi31.4s, P.4s
    //      sub      u.4s, t.4s, P.4s
    //      umin     res.4s, t.4s, u.4s
    // throughput: 1.25 cyc/vec (3.2 els/cyc)
    // latency: 10 cyc

    // We want to return res in 0, ..., P such that res = lhs * rhs (mod P).
    // Let prod := lhs * rhs. Break it up into prod = 2^31 prod_hi31 + prod_lo31, where both limbs
    // are in 0, ..., 2^31 - 1. Then prod = prod_hi31 + prod_lo31 (mod P), so let
    // t := prod_hi31 + prod_lo31.
    // Define prod_lo32 = prod mod 2^32 and observe that
    //   prod_lo32 = prod_lo31 + 2^31 (prod_hi31 mod 2)
    //             = prod_lo31 + 2^31 prod_hi31                                          (mod 2^32)
    // Then
    //   t = prod_lo32 - 2^31 prod_hi31 + prod_hi31                                      (mod 2^32)
    //     = prod_lo32 - (2^31 - 1) prod_hi31                                            (mod 2^32)
    //     = prod_lo32 - prod_hi31 * P                                                   (mod 2^32)
    //
    // t is in 0, ..., 2 P, so we apply reduce_sum to get the result.

    unsafe {
        // Safety: If this code got compiled then NEON intrinsics are available.
        let prod_hi31 = mul_31x31_to_hi_31(lhs, rhs);
        let prod_lo32 = aarch64::vmulq_u32(lhs, rhs);
        let t = aarch64::vmlsq_u32(prod_lo32, prod_hi31, P);
        reduce_sum(t)
    }
}

/// Negate a vector of Mersenne-31 field elements that fit in 31 bits.
/// If the inputs do not fit in 31 bits, the result is undefined.
#[inline]
#[must_use]
fn neg(val: uint32x4_t) -> uint32x4_t {
    // We want this to compile to:
    //      eor  res.16b, val.16b, P.16b
    // throughput: .25 cyc/vec (16 els/cyc)
    // latency: 2 cyc

    // val is in 0, ..., P, so res := P - val is also in 0, ..., P.

    unsafe {
        // Safety: If this code got compiled then NEON intrinsics are available.
        aarch64::vsubq_u32(P, val)
    }
}

impl_packed_value!(PackedMersenne31Neon, Mersenne31, WIDTH);

unsafe impl PackedField for PackedMersenne31Neon {
    type Scalar = Mersenne31;
}

impl_packed_field_pow_2!(
    PackedMersenne31Neon;
    [
        (1, interleave_u32),
        (2, interleave_u64)
    ],
    WIDTH
);

#[cfg(test)]
mod tests {
    use p3_field_testing::test_packed_field;

    use super::{Mersenne31, PackedMersenne31Neon};

    /// Zero has a redundant representation, so let's test both.
    const ZEROS: PackedMersenne31Neon = PackedMersenne31Neon(Mersenne31::new_array([
        0x00000000, 0x7fffffff, 0x00000000, 0x7fffffff,
    ]));

    const SPECIAL_VALS: PackedMersenne31Neon = PackedMersenne31Neon(Mersenne31::new_array([
        0x00000000, 0x00000001, 0x00000002, 0x7ffffffe,
    ]));

    test_packed_field!(
        crate::PackedMersenne31Neon,
        &[super::ZEROS],
        &[crate::PackedMersenne31Neon::ONE],
        super::SPECIAL_VALS
    );
}
