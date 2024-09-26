//! The prime field known as Goldilocks, defined as `F_p` where `p = 2^64 - 2^32 + 1`.

#![no_std]
#![cfg_attr(
    all(
        feature = "nightly-features",
        target_arch = "x86_64",
        target_feature = "avx512f"
    ),
    feature(stdarch_x86_avx512)
)]

extern crate alloc;

mod extension;
mod mds;
mod poseidon2;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(feature = "nightly-features", target_feature = "avx512f"))
))]
mod x86_64_avx2;

use alloc::vec;
use alloc::vec::Vec;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(feature = "nightly-features", target_feature = "avx512f"))
))]
pub use x86_64_avx2::*;

#[cfg(all(
    feature = "nightly-features",
    target_arch = "x86_64",
    target_feature = "avx512f"
))]
mod x86_64_avx512;

use core::fmt;
use core::fmt::{Debug, Display, Formatter};
use core::hash::{Hash, Hasher};
use core::intrinsics::transmute;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

pub use mds::*;
use num_bigint::BigUint;
use p3_field::{
    exp_10540996611094048183, exp_u64_by_squaring, halve_u64, AbstractField, Field, Packable,
    PrimeField, PrimeField64, TwoAdicField,
};
use p3_util::{assume, branch_hint};
pub use poseidon2::*;
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use serde::{Deserialize, Serialize};
#[cfg(all(
    feature = "nightly-features",
    target_arch = "x86_64",
    target_feature = "avx512f"
))]
pub use x86_64_avx512::*;

/// The Goldilocks prime
const P: u64 = 0xFFFF_FFFF_0000_0001;

/// The prime field known as Goldilocks, defined as `F_p` where `p = 2^64 - 2^32 + 1`.
#[derive(Copy, Clone, Default, Serialize, Deserialize)]
#[repr(transparent)] // Packed field implementations rely on this!
pub struct Goldilocks {
    /// Not necessarily canonical.
    value: u64,
}

impl Goldilocks {
    const fn new(value: u64) -> Self {
        Self { value }
    }

    /// Two's complement of `ORDER`, i.e. `2^64 - ORDER = 2^32 - 1`.
    const NEG_ORDER: u64 = Self::ORDER_U64.wrapping_neg();
}

impl PartialEq for Goldilocks {
    fn eq(&self, other: &Self) -> bool {
        self.as_canonical_u64() == other.as_canonical_u64()
    }
}

impl Eq for Goldilocks {}

impl Packable for Goldilocks {}

impl Hash for Goldilocks {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.as_canonical_u64());
    }
}

impl Ord for Goldilocks {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.as_canonical_u64().cmp(&other.as_canonical_u64())
    }
}

impl PartialOrd for Goldilocks {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Display for Goldilocks {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.value, f)
    }
}

impl Debug for Goldilocks {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.value, f)
    }
}

impl Distribution<Goldilocks> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Goldilocks {
        loop {
            let next_u64 = rng.next_u64();
            let is_canonical = next_u64 < Goldilocks::ORDER_U64;
            if is_canonical {
                return Goldilocks::new(next_u64);
            }
        }
    }
}

impl AbstractField for Goldilocks {
    type F = Self;

    fn zero() -> Self {
        Self::new(0)
    }
    fn one() -> Self {
        Self::new(1)
    }
    fn two() -> Self {
        Self::new(2)
    }
    fn neg_one() -> Self {
        Self::new(Self::ORDER_U64 - 1)
    }

    #[inline]
    fn from_f(f: Self::F) -> Self {
        f
    }

    fn from_bool(b: bool) -> Self {
        Self::new(u64::from(b))
    }

    fn from_canonical_u8(n: u8) -> Self {
        Self::new(u64::from(n))
    }

    fn from_canonical_u16(n: u16) -> Self {
        Self::new(u64::from(n))
    }

    fn from_canonical_u32(n: u32) -> Self {
        Self::new(u64::from(n))
    }

    #[inline(always)]
    fn from_canonical_u64(n: u64) -> Self {
        Self::new(n)
    }

    fn from_canonical_usize(n: usize) -> Self {
        Self::new(n as u64)
    }

    fn from_wrapped_u32(n: u32) -> Self {
        // A u32 must be canonical, plus we don't store canonical encodings anyway, so there's no
        // need for a reduction.
        Self::new(u64::from(n))
    }

    fn from_wrapped_u64(n: u64) -> Self {
        // There's no need to reduce `n` to canonical form, as our internal encoding is
        // non-canonical, so there's no need for a reduction.
        Self::new(n)
    }

    // Sage: GF(2^64 - 2^32 + 1).multiplicative_generator()
    fn generator() -> Self {
        Self::new(7)
    }
}

impl Field for Goldilocks {
    // TODO: Add cfg-guarded Packing for NEON

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(all(feature = "nightly-features", target_feature = "avx512f"))
    ))]
    type Packing = crate::PackedGoldilocksAVX2;

    #[cfg(all(
        feature = "nightly-features",
        target_arch = "x86_64",
        target_feature = "avx512f"
    ))]
    type Packing = crate::PackedGoldilocksAVX512;
    #[cfg(not(any(
        all(
            target_arch = "x86_64",
            target_feature = "avx2",
            not(all(feature = "nightly-features", target_feature = "avx512f"))
        ),
        all(
            feature = "nightly-features",
            target_arch = "x86_64",
            target_feature = "avx512f"
        ),
    )))]
    type Packing = Self;

    fn is_zero(&self) -> bool {
        self.value == 0 || self.value == Self::ORDER_U64
    }

    #[inline]
    fn exp_u64_generic<AF: AbstractField<F = Self>>(val: AF, power: u64) -> AF {
        match power {
            10540996611094048183 => exp_10540996611094048183(val), // used to compute x^{1/7}
            _ => exp_u64_by_squaring(val, power),
        }
    }

    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        // From Fermat's little theorem, in a prime field `F_p`, the inverse of `a` is `a^(p-2)`.
        //
        // compute a^(p - 2) using 72 multiplications
        // The exponent p - 2 is represented in binary as:
        // 0b1111111111111111111111111111111011111111111111111111111111111111
        // Adapted from: https://github.com/facebook/winterfell/blob/d238a1/math/src/field/f64/mod.rs#L136-L164

        // compute base^11
        let t2 = self.square() * *self;

        // compute base^111
        let t3 = t2.square() * *self;

        // compute base^111111 (6 ones)
        // repeatedly square t3 3 times and multiply by t3
        let t6 = exp_acc::<3>(t3, t3);
        let t60 = t6.square();
        let t7 = t60 * *self;

        // compute base^111111111111 (12 ones)
        // repeatedly square t6 6 times and multiply by t6
        let t12 = exp_acc::<5>(t60, t6);

        // compute base^111111111111111111111111 (24 ones)
        // repeatedly square t12 12 times and multiply by t12
        let t24 = exp_acc::<12>(t12, t12);

        // compute base^1111111111111111111111111111111 (31 ones)
        // repeatedly square t24 6 times and multiply by t6 first. then square t30 and
        // multiply by base
        let t31 = exp_acc::<7>(t24, t7);

        // compute base^111111111111111111111111111111101111111111111111111111111111111
        // repeatedly square t31 32 times and multiply by t31
        let t63 = exp_acc::<32>(t31, t31);

        // compute base^1111111111111111111111111111111011111111111111111111111111111111
        Some(t63.square() * *self)
    }

    #[inline]
    fn halve(&self) -> Self {
        Goldilocks::new(halve_u64::<P>(self.value))
    }

    #[inline]
    fn order() -> BigUint {
        P.into()
    }

    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY: repr(transparent) ensures transmutation safety.
        unsafe { transmute(vec![0u64; len]) }
    }
}

impl PrimeField for Goldilocks {
    fn as_canonical_biguint(&self) -> BigUint {
        <Self as PrimeField64>::as_canonical_u64(self).into()
    }
}

impl PrimeField64 for Goldilocks {
    const ORDER_U64: u64 = P;

    #[inline]
    fn as_canonical_u64(&self) -> u64 {
        let mut c = self.value;
        // We only need one condition subtraction, since 2 * ORDER would not fit in a u64.
        if c >= Self::ORDER_U64 {
            c -= Self::ORDER_U64;
        }
        c
    }
}

impl TwoAdicField for Goldilocks {
    const TWO_ADICITY: usize = 32;

    fn two_adic_generator(bits: usize) -> Self {
        assert!(bits <= Self::TWO_ADICITY);
        let base = Self::new(1_753_635_133_440_165_772); // generates the whole 2^TWO_ADICITY group
        base.exp_power_of_2(Self::TWO_ADICITY - bits)
    }
}

impl Add for Goldilocks {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let (sum, over) = self.value.overflowing_add(rhs.value);
        let (mut sum, over) = sum.overflowing_add(u64::from(over) * Self::NEG_ORDER);
        if over {
            // NB: self.value > Self::ORDER && rhs.value > Self::ORDER is necessary but not
            // sufficient for double-overflow.
            // This assume does two things:
            //  1. If compiler knows that either self.value or rhs.value <= ORDER, then it can skip
            //     this check.
            //  2. Hints to the compiler how rare this double-overflow is (thus handled better with
            //     a branch).
            assume(self.value > Self::ORDER_U64 && rhs.value > Self::ORDER_U64);
            branch_hint();
            sum += Self::NEG_ORDER; // Cannot overflow.
        }
        Self::new(sum)
    }
}

impl AddAssign for Goldilocks {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sum for Goldilocks {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        // This is faster than iter.reduce(|x, y| x + y).unwrap_or(Self::zero()) for iterators of length > 2.

        // This sum will not overflow so long as iter.len() < 2^64.
        let sum = iter.map(|x| x.value as u128).sum::<u128>();
        reduce128(sum)
    }
}

impl Sub for Goldilocks {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let (diff, under) = self.value.overflowing_sub(rhs.value);
        let (mut diff, under) = diff.overflowing_sub(u64::from(under) * Self::NEG_ORDER);
        if under {
            // NB: self.value < NEG_ORDER - 1 && rhs.value > ORDER is necessary but not
            // sufficient for double-underflow.
            // This assume does two things:
            //  1. If compiler knows that either self.value >= NEG_ORDER - 1 or rhs.value <= ORDER,
            //     then it can skip this check.
            //  2. Hints to the compiler how rare this double-underflow is (thus handled better
            //     with a branch).
            assume(self.value < Self::NEG_ORDER - 1 && rhs.value > Self::ORDER_U64);
            branch_hint();
            diff -= Self::NEG_ORDER; // Cannot underflow.
        }
        Self::new(diff)
    }
}

impl SubAssign for Goldilocks {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for Goldilocks {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(Self::ORDER_U64 - self.as_canonical_u64())
    }
}

impl Mul for Goldilocks {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        reduce128(u128::from(self.value) * u128::from(rhs.value))
    }
}

impl MulAssign for Goldilocks {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Product for Goldilocks {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::one())
    }
}

impl Div for Goldilocks {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        self * rhs.inverse()
    }
}

/// Squares the base N number of times and multiplies the result by the tail value.
#[inline(always)]
fn exp_acc<const N: usize>(base: Goldilocks, tail: Goldilocks) -> Goldilocks {
    base.exp_power_of_2(N) * tail
}

/// Reduces to a 64-bit value. The result might not be in canonical form; it could be in between the
/// field order and `2^64`.
#[inline]
pub(crate) fn reduce128(x: u128) -> Goldilocks {
    let (x_lo, x_hi) = split(x); // This is a no-op
    let x_hi_hi = x_hi >> 32;
    let x_hi_lo = x_hi & Goldilocks::NEG_ORDER;

    let (mut t0, borrow) = x_lo.overflowing_sub(x_hi_hi);
    if borrow {
        branch_hint(); // A borrow is exceedingly rare. It is faster to branch.
        t0 -= Goldilocks::NEG_ORDER; // Cannot underflow.
    }
    let t1 = x_hi_lo * Goldilocks::NEG_ORDER;
    let t2 = unsafe { add_no_canonicalize_trashing_input(t0, t1) };
    Goldilocks::new(t2)
}

#[inline]
#[allow(clippy::cast_possible_truncation)]
const fn split(x: u128) -> (u64, u64) {
    (x as u64, (x >> 64) as u64)
}

/// Fast addition modulo ORDER for x86-64.
/// This function is marked unsafe for the following reasons:
///   - It is only correct if x + y < 2**64 + ORDER = 0x1ffffffff00000001.
///   - It is only faster in some circumstances. In particular, on x86 it overwrites both inputs in
///     the registers, so its use is not recommended when either input will be used again.
#[inline(always)]
#[cfg(target_arch = "x86_64")]
unsafe fn add_no_canonicalize_trashing_input(x: u64, y: u64) -> u64 {
    let res_wrapped: u64;
    let adjustment: u64;
    core::arch::asm!(
        "add {0}, {1}",
        // Trick. The carry flag is set iff the addition overflowed.
        // sbb x, y does x := x - y - CF. In our case, x and y are both {1:e}, so it simply does
        // {1:e} := 0xffffffff on overflow and {1:e} := 0 otherwise. {1:e} is the low 32 bits of
        // {1}; the high 32-bits are zeroed on write. In the end, we end up with 0xffffffff in {1}
        // on overflow; this happens be NEG_ORDER.
        // Note that the CPU does not realize that the result of sbb x, x does not actually depend
        // on x. We must write the result to a register that we know to be ready. We have a
        // dependency on {1} anyway, so let's use it.
        "sbb {1:e}, {1:e}",
        inlateout(reg) x => res_wrapped,
        inlateout(reg) y => adjustment,
        options(pure, nomem, nostack),
    );
    assume(x != 0 || (res_wrapped == y && adjustment == 0));
    assume(y != 0 || (res_wrapped == x && adjustment == 0));
    // Add NEG_ORDER == subtract ORDER.
    // Cannot overflow unless the assumption if x + y < 2**64 + ORDER is incorrect.
    res_wrapped + adjustment
}

#[inline(always)]
#[cfg(not(target_arch = "x86_64"))]
unsafe fn add_no_canonicalize_trashing_input(x: u64, y: u64) -> u64 {
    let (res_wrapped, carry) = x.overflowing_add(y);
    // Below cannot overflow unless the assumption if x + y < 2**64 + ORDER is incorrect.
    res_wrapped + Goldilocks::NEG_ORDER * u64::from(carry)
}

/// Convert a constant u64 array into a constant Goldilocks array.
#[inline]
#[must_use]
pub(crate) const fn to_goldilocks_array<const N: usize>(input: [u64; N]) -> [Goldilocks; N] {
    let mut output = [Goldilocks { value: 0 }; N];
    let mut i = 0;
    loop {
        if i == N {
            break;
        }
        output[i].value = input[i];
        i += 1;
    }
    output
}

#[cfg(test)]
mod tests {
    use p3_field_testing::{test_field, test_field_dft, test_two_adic_field};

    use super::*;

    type F = Goldilocks;

    #[test]
    fn test_goldilocks() {
        let f = F::new(100);
        assert_eq!(f.as_canonical_u64(), 100);

        // Over the Goldilocks field, the following set of equations hold
        // p               = 0
        // 2^64 - 2^32 + 1 = 0
        // 2^64            = 2^32 - 1
        let f = F::new(u64::MAX);
        assert_eq!(f.as_canonical_u64(), u32::MAX as u64 - 1);

        let f = F::from_canonical_u64(u64::MAX);
        assert_eq!(f.as_canonical_u64(), u32::MAX as u64 - 1);

        let f = F::from_canonical_u64(0);
        assert!(f.is_zero());

        let f = F::from_canonical_u64(F::ORDER_U64);
        assert!(f.is_zero());

        assert_eq!(F::generator().as_canonical_u64(), 7_u64);

        let f_1 = F::new(1);
        let f_1_copy = F::new(1);

        let expected_result = F::zero();
        assert_eq!(f_1 - f_1_copy, expected_result);

        let expected_result = F::new(2);
        assert_eq!(f_1 + f_1_copy, expected_result);

        let f_2 = F::new(2);
        let expected_result = F::new(3);
        assert_eq!(f_1 + f_1_copy * f_2, expected_result);

        let expected_result = F::new(5);
        assert_eq!(f_1 + f_2 * f_2, expected_result);

        let f_p_minus_1 = F::from_canonical_u64(F::ORDER_U64 - 1);
        let expected_result = F::zero();
        assert_eq!(f_1 + f_p_minus_1, expected_result);

        let f_p_minus_2 = F::from_canonical_u64(F::ORDER_U64 - 2);
        let expected_result = F::from_canonical_u64(F::ORDER_U64 - 3);
        assert_eq!(f_p_minus_1 + f_p_minus_2, expected_result);

        let expected_result = F::new(1);
        assert_eq!(f_p_minus_1 - f_p_minus_2, expected_result);

        let expected_result = f_p_minus_1;
        assert_eq!(f_p_minus_2 - f_p_minus_1, expected_result);

        let expected_result = f_p_minus_2;
        assert_eq!(f_p_minus_1 - f_1, expected_result);

        let expected_result = F::new(3);
        assert_eq!(f_2 * f_2 - f_1, expected_result);

        // Generator check
        let expected_multiplicative_group_generator = F::new(7);
        assert_eq!(F::generator(), expected_multiplicative_group_generator);

        // Check on `reduce_u128`
        let x = u128::MAX;
        let y = reduce128(x);
        // The following equalitiy sequence holds, modulo p = 2^64 - 2^32 + 1
        // 2^128 - 1 = (2^64 - 1) * (2^64 + 1)
        //           = (2^32 - 1 - 1) * (2^32 - 1 + 1)
        //           = (2^32 - 2) * (2^32)
        //           = 2^64 - 2 * 2^32
        //           = 2^64 - 2^33
        //           = 2^32 - 1 - 2^33
        //           = - 2^32 - 1
        let expected_result = -F::new(2_u64.pow(32)) - F::new(1);
        assert_eq!(y, expected_result);

        assert_eq!(f.exp_u64(10540996611094048183).exp_const_u64::<7>(), f);
        assert_eq!(y.exp_u64(10540996611094048183).exp_const_u64::<7>(), y);
        assert_eq!(f_2.exp_u64(10540996611094048183).exp_const_u64::<7>(), f_2);
    }

    test_field!(crate::Goldilocks);
    test_two_adic_field!(crate::Goldilocks);

    test_field_dft!(radix2dit, crate::Goldilocks, p3_dft::Radix2Dit<_>);
    test_field_dft!(bowers, crate::Goldilocks, p3_dft::Radix2Bowers);
    test_field_dft!(parallel, crate::Goldilocks, p3_dft::Radix2DitParallel);
}
