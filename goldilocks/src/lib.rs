//! The prime field known as Goldilocks, defined as `F_p` where `p = 2^64 - 2^32 + 1`.

#![no_std]

use core::fmt;
use core::fmt::{Debug, Display, Formatter};
use core::hash::{Hash, Hasher};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};
use p3_field::field::{AbstractField, Field, PrimeField, TwoAdicField};
use p3_util::{assume, branch_hint};
use rand::distributions::{Distribution, Standard};
use rand::Rng;

/// The prime field known as Goldilocks, defined as `F_p` where `p = 2^64 - 2^32 + 1`.
#[derive(Copy, Clone, Default)]
pub struct Goldilocks {
    /// Not necessarily canonical.
    value: u64,
}

impl Goldilocks {
    pub const ORDER: u64 = 0xFFFFFFFF00000001;

    /// Two's complement of `ORDER`, i.e. `2^64 - ORDER = 2^32 - 1`.
    const NEG_ORDER: u64 = Self::ORDER.wrapping_neg();

    fn as_canonical_u64(&self) -> u64 {
        let mut c = self.value;
        // We only need one condition subtraction, since 2 * ORDER would not fit in a u64.
        if c >= Self::ORDER {
            c -= Self::ORDER;
        }
        c
    }
}

impl PartialEq for Goldilocks {
    fn eq(&self, other: &Self) -> bool {
        self.as_canonical_u64() == other.as_canonical_u64()
    }
}

impl Eq for Goldilocks {}

impl Hash for Goldilocks {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.as_canonical_u64())
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
            let is_canonical = next_u64 < Goldilocks::ORDER;
            if is_canonical {
                return Goldilocks { value: next_u64 };
            }
        }
    }
}

impl AbstractField<Self> for Goldilocks {
    const ZERO: Self = Self { value: 0 };
    const ONE: Self = Self { value: 1 };
    const TWO: Self = Self { value: 2 };
    const NEG_ONE: Self = Self {
        value: Self::ORDER - 1,
    };
    // Sage: GF(2^64 - 2^32 + 1).multiplicative_generator()
    const MULTIPLICATIVE_GROUP_GENERATOR: Self = Self { value: 7 };
}

impl Field for Goldilocks {
    // TODO: Add cfg-guarded Packing for AVX2, NEON, etc.
    type Packing = Self;

    fn is_zero(&self) -> bool {
        self.value == 0 || self.value == Self::ORDER
    }

    fn try_inverse(&self) -> Option<Self> {
        todo!()
    }
}

impl PrimeField for Goldilocks {}

impl TwoAdicField for Goldilocks {
    const TWO_ADICITY: usize = 32;
    const POWER_OF_TWO_GENERATOR: Self = Self {
        value: 1753635133440165772,
    };
}

impl Add<Self> for Goldilocks {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let (sum, over) = self.value.overflowing_add(rhs.value);
        let (mut sum, over) = sum.overflowing_add((over as u64) * Self::NEG_ORDER);
        if over {
            // NB: self.value > Self::ORDER && rhs.value > Self::ORDER is necessary but not
            // sufficient for double-overflow.
            // This assume does two things:
            //  1. If compiler knows that either self.value or rhs.value <= ORDER, then it can skip
            //     this check.
            //  2. Hints to the compiler how rare this double-overflow is (thus handled better with
            //     a branch).
            assume(self.value > Self::ORDER && rhs.value > Self::ORDER);
            branch_hint();
            sum += Self::NEG_ORDER; // Cannot overflow.
        }
        Self { value: sum }
    }
}

impl AddAssign<Self> for Goldilocks {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sum for Goldilocks {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x + y).unwrap_or(Self::ZERO)
    }
}

impl Sub<Self> for Goldilocks {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let (diff, under) = self.value.overflowing_sub(rhs.value);
        let (mut diff, under) = diff.overflowing_sub((under as u64) * Self::NEG_ORDER);
        if under {
            // NB: self.value < NEG_ORDER - 1 && rhs.value > ORDER is necessary but not
            // sufficient for double-underflow.
            // This assume does two things:
            //  1. If compiler knows that either self.value >= NEG_ORDER - 1 or rhs.value <= ORDER,
            //     then it can skip this check.
            //  2. Hints to the compiler how rare this double-underflow is (thus handled better
            //     with a branch).
            assume(self.value < Self::NEG_ORDER - 1 && rhs.value > Self::ORDER);
            branch_hint();
            diff -= Self::NEG_ORDER; // Cannot underflow.
        }
        Self { value: diff }
    }
}

impl SubAssign<Self> for Goldilocks {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for Goldilocks {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            value: Self::ORDER - self.as_canonical_u64(),
        }
    }
}

impl Mul<Self> for Goldilocks {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        reduce128((self.value as u128) * (rhs.value as u128))
    }
}

impl MulAssign<Self> for Goldilocks {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Product for Goldilocks {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::ONE)
    }
}

impl Div for Goldilocks {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        self * rhs.inverse()
    }
}

/// Reduces to a 64-bit value. The result might not be in canonical form; it could be in between the
/// field order and `2^64`.
#[inline]
fn reduce128(x: u128) -> Goldilocks {
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
    Goldilocks { value: t2 }
}

#[inline]
fn split(x: u128) -> (u64, u64) {
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
    res_wrapped + Goldilocks::NEG_ORDER * (carry as u64)
}
