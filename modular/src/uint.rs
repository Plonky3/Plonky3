//! Fixed-width unsigned integers on little-endian 64-bit limbs.

use alloc::vec::Vec;
use core::cmp::Ordering;

use num_bigint::BigUint;

/// `a + b + carry`, as the low limb and the carry out (0 or 1).
#[inline(always)]
pub(crate) const fn adc(a: u64, b: u64, carry: u64) -> (u64, u64) {
    let t = a as u128 + b as u128 + carry as u128;
    (t as u64, (t >> 64) as u64)
}

/// `a - b - borrow`, as the low limb and the borrow out (0 or 1).
#[inline(always)]
pub(crate) const fn sbb(a: u64, b: u64, borrow: u64) -> (u64, u64) {
    let (d, b1) = a.overflowing_sub(b);
    let (d, b2) = d.overflowing_sub(borrow);
    (d, (b1 | b2) as u64)
}

/// `a + b * c + carry`, as the low and high limbs.
///
/// The sum is at most `2^128 - 1`, so it never overflows.
#[inline(always)]
pub(crate) const fn mac(a: u64, b: u64, c: u64, carry: u64) -> (u64, u64) {
    let t = a as u128 + (b as u128) * (c as u128) + carry as u128;
    (t as u64, (t >> 64) as u64)
}

/// An unsigned integer below `2^(64 L)`, least significant limb first.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Uint<const L: usize>(pub [u64; L]);

impl<const L: usize> Default for Uint<L> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<const L: usize> Uint<L> {
    /// Zero.
    pub const ZERO: Self = Self([0; L]);

    /// One.
    pub const ONE: Self = Self::from_u64(1);

    /// `2^(64 L) - 1`.
    pub const MAX: Self = Self([u64::MAX; L]);

    /// The width in bits.
    pub const BITS: u32 = 64 * L as u32;

    /// The integer `x`.
    #[must_use]
    pub const fn from_u64(x: u64) -> Self {
        let mut limbs = [0; L];
        limbs[0] = x;
        Self(limbs)
    }

    /// Whether the value is zero.
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        let mut i = 0;
        while i < L {
            if self.0[i] != 0 {
                return false;
            }
            i += 1;
        }
        true
    }

    /// Whether the value is odd.
    #[must_use]
    pub const fn is_odd(&self) -> bool {
        self.0[0] & 1 == 1
    }

    /// The bit length: the position of the highest set bit plus one, or 0 for zero.
    #[must_use]
    pub const fn bits(&self) -> u32 {
        let mut i = L;
        while i > 0 {
            i -= 1;
            if self.0[i] != 0 {
                return 64 * i as u32 + 64 - self.0[i].leading_zeros();
            }
        }
        0
    }

    /// Bit `i`, or `false` when `i` is past the width.
    #[must_use]
    pub const fn bit(&self, i: u32) -> bool {
        let limb = (i / 64) as usize;
        limb < L && (self.0[limb] >> (i % 64)) & 1 == 1
    }

    /// `self + rhs` modulo `2^(64 L)`, and the carry out.
    #[inline(always)]
    #[must_use]
    pub const fn adc(&self, rhs: &Self) -> (Self, bool) {
        let mut out = [0; L];
        let mut carry = 0;
        let mut i = 0;
        while i < L {
            (out[i], carry) = adc(self.0[i], rhs.0[i], carry);
            i += 1;
        }
        (Self(out), carry == 1)
    }

    /// `self - rhs` modulo `2^(64 L)`, and the borrow out.
    #[inline(always)]
    #[must_use]
    pub const fn sbb(&self, rhs: &Self) -> (Self, bool) {
        let mut out = [0; L];
        let mut borrow = 0;
        let mut i = 0;
        while i < L {
            (out[i], borrow) = sbb(self.0[i], rhs.0[i], borrow);
            i += 1;
        }
        (Self(out), borrow == 1)
    }

    /// `self * rhs`, as the low and high halves of the `128 L`-bit product.
    #[inline(always)]
    #[must_use]
    pub const fn mul_wide(&self, rhs: &Self) -> (Self, Self) {
        let mut lo = [0; L];
        let mut hi = [0; L];
        let mut i = 0;
        while i < L {
            let mut carry = 0;
            let mut j = 0;
            while j < L {
                let k = i + j;
                if k < L {
                    (lo[k], carry) = mac(lo[k], self.0[i], rhs.0[j], carry);
                } else {
                    (hi[k - L], carry) = mac(hi[k - L], self.0[i], rhs.0[j], carry);
                }
                j += 1;
            }
            // Row i ends at limb i + L, which is always in the high half.
            hi[i] = carry;
            i += 1;
        }
        (Self(lo), Self(hi))
    }

    /// `self * rhs`, as the low `L` limbs and the limb above them.
    #[inline(always)]
    #[must_use]
    pub const fn mul_limb(&self, rhs: u64) -> (Self, u64) {
        let mut out = [0; L];
        let mut carry = 0;
        let mut i = 0;
        while i < L {
            (out[i], carry) = mac(0, self.0[i], rhs, carry);
            i += 1;
        }
        (Self(out), carry)
    }

    /// `self << n`, dropping the bits shifted past the width; zero when `n >= 64 L`.
    #[must_use]
    pub const fn shl(&self, n: u32) -> Self {
        let mut out = [0; L];
        let limbs = (n / 64) as usize;
        let bits = n % 64;
        let mut i = L;
        while i > limbs {
            i -= 1;
            let src = i - limbs;
            out[i] = self.0[src] << bits;
            if bits != 0 && src > 0 {
                out[i] |= self.0[src - 1] >> (64 - bits);
            }
        }
        Self(out)
    }

    /// `self >> n`; zero when `n >= 64 L`.
    #[must_use]
    pub const fn shr(&self, n: u32) -> Self {
        let mut out = [0; L];
        let limbs = (n / 64) as usize;
        let bits = n % 64;
        let mut i = 0;
        while i + limbs < L {
            let src = i + limbs;
            out[i] = self.0[src] >> bits;
            if bits != 0 && src + 1 < L {
                out[i] |= self.0[src + 1] << (64 - bits);
            }
            i += 1;
        }
        Self(out)
    }

    /// The comparison of `self` with `rhs`, usable in const contexts.
    #[must_use]
    pub const fn cmp_const(&self, rhs: &Self) -> Ordering {
        let mut i = L;
        while i > 0 {
            i -= 1;
            if self.0[i] != rhs.0[i] {
                return if self.0[i] < rhs.0[i] {
                    Ordering::Less
                } else {
                    Ordering::Greater
                };
            }
        }
        Ordering::Equal
    }

    /// The integer with these little-endian bytes, or `None` when it needs more than `L` limbs.
    #[must_use]
    pub fn from_le_bytes(bytes: &[u8]) -> Option<Self> {
        let (head, tail) = bytes.split_at(bytes.len().min(8 * L));
        if tail.iter().any(|&b| b != 0) {
            return None;
        }
        let mut out = [0; L];
        for (i, &b) in head.iter().enumerate() {
            out[i / 8] |= u64::from(b) << (8 * (i % 8));
        }
        Some(Self(out))
    }

    /// The integer with these big-endian bytes, or `None` when it needs more than `L` limbs.
    #[must_use]
    pub fn from_be_bytes(bytes: &[u8]) -> Option<Self> {
        let (head, tail) = bytes.split_at(bytes.len().saturating_sub(8 * L));
        if head.iter().any(|&b| b != 0) {
            return None;
        }
        let mut out = [0; L];
        for (i, &b) in tail.iter().rev().enumerate() {
            out[i / 8] |= u64::from(b) << (8 * (i % 8));
        }
        Some(Self(out))
    }

    /// Writes the `8 L` little-endian bytes into `out`.
    ///
    /// # Panics
    ///
    /// Panics when `out` is not `8 L` bytes long.
    pub fn write_le_bytes(&self, out: &mut [u8]) {
        assert_eq!(out.len(), 8 * L, "output is not 8 L bytes");
        for (chunk, limb) in out.as_chunks_mut::<8>().0.iter_mut().zip(self.0) {
            *chunk = limb.to_le_bytes();
        }
    }

    /// Writes the `8 L` big-endian bytes into `out`.
    ///
    /// # Panics
    ///
    /// Panics when `out` is not `8 L` bytes long.
    pub fn write_be_bytes(&self, out: &mut [u8]) {
        assert_eq!(out.len(), 8 * L, "output is not 8 L bytes");
        for (chunk, limb) in out
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .zip(self.0.iter().rev())
        {
            *chunk = limb.to_be_bytes();
        }
    }
}

impl<const L: usize> PartialOrd for Uint<L> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<const L: usize> Ord for Uint<L> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cmp_const(other)
    }
}

impl<const L: usize> From<Uint<L>> for BigUint {
    fn from(x: Uint<L>) -> Self {
        let mut digits = Vec::with_capacity(2 * L);
        for limb in x.0 {
            digits.push(limb as u32);
            digits.push((limb >> 32) as u32);
        }
        Self::new(digits)
    }
}

impl<const L: usize> TryFrom<&BigUint> for Uint<L> {
    type Error = ();

    fn try_from(x: &BigUint) -> Result<Self, ()> {
        if x.bits() > 64 * L as u64 {
            return Err(());
        }
        let mut out = [0; L];
        for (limb, digit) in out.iter_mut().zip(x.iter_u64_digits()) {
            *limb = digit;
        }
        Ok(Self(out))
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn big<const L: usize>(x: Uint<L>) -> BigUint {
        x.into()
    }

    fn arb<const L: usize>() -> impl Strategy<Value = Uint<L>> {
        prop::array::uniform::<_, L>(any::<u64>()).prop_map(Uint)
    }

    fn check_ops<const L: usize>(a: Uint<L>, b: Uint<L>, n: u32, x: u64) {
        let r = BigUint::from(1u8) << (64 * L);
        let (s, c) = a.adc(&b);
        assert_eq!(big(s) + (BigUint::from(c) << (64 * L)), big(a) + big(b));
        let (d, borrow) = a.sbb(&b);
        assert_eq!(big(d), (big(a) + &r - big(b)) % &r);
        assert_eq!(borrow, a < b);
        let (lo, hi) = a.mul_wide(&b);
        assert_eq!(big(lo) + (big(hi) << (64 * L)), big(a) * big(b));
        let (lo, hi) = a.mul_limb(x);
        assert_eq!(big(lo) + (BigUint::from(hi) << (64 * L)), big(a) * x);
        assert_eq!(big(a.shl(n)), (big(a) << n) % &r);
        assert_eq!(big(a.shr(n)), big(a) >> n);
        assert_eq!(u64::from(a.bits()), big(a).bits());
        assert_eq!(a.bit(n), big(a).bit(u64::from(n)));
        assert_eq!(a.cmp(&b), big(a).cmp(&big(b)));
        assert_eq!(Uint::<L>::try_from(&big(a)), Ok(a));

        let mut le = [0u8; 64];
        let mut be = [0u8; 64];
        a.write_le_bytes(&mut le[..8 * L]);
        a.write_be_bytes(&mut be[..8 * L]);
        assert_eq!(Uint::from_le_bytes(&le[..8 * L]), Some(a));
        assert_eq!(Uint::from_be_bytes(&be[..8 * L]), Some(a));
        assert_eq!(BigUint::from_bytes_le(&le[..8 * L]), big(a));
    }

    proptest! {
        #[test]
        fn ops_match_bigint_1(a in arb::<1>(), b in arb::<1>(), n in 0u32..80, x: u64) {
            check_ops(a, b, n, x);
        }

        #[test]
        fn ops_match_bigint_2(a in arb::<2>(), b in arb::<2>(), n in 0u32..140, x: u64) {
            check_ops(a, b, n, x);
        }

        #[test]
        fn ops_match_bigint_3(a in arb::<3>(), b in arb::<3>(), n in 0u32..200, x: u64) {
            check_ops(a, b, n, x);
        }

        #[test]
        fn ops_match_bigint_4(a in arb::<4>(), b in arb::<4>(), n in 0u32..260, x: u64) {
            check_ops(a, b, n, x);
        }
    }

    #[test]
    fn extremes_match_bigint() {
        for a in [Uint::<3>::ZERO, Uint::ONE, Uint::MAX] {
            for b in [Uint::<3>::ZERO, Uint::ONE, Uint::MAX] {
                check_ops(a, b, 0, u64::MAX);
                check_ops(a, b, 191, 0);
            }
        }
    }

    #[test]
    fn oversized_inputs_are_refused() {
        assert_eq!(Uint::<1>::from_le_bytes(&[0, 0, 0, 0, 0, 0, 0, 0, 1]), None);
        assert_eq!(Uint::<1>::from_be_bytes(&[1, 0, 0, 0, 0, 0, 0, 0, 0]), None);
        // Zero padding past the width is accepted.
        assert_eq!(
            Uint::<1>::from_le_bytes(&[7, 0, 0, 0, 0, 0, 0, 0, 0]),
            Some(Uint([7]))
        );
        assert_eq!(Uint::<1>::try_from(&(BigUint::from(1u8) << 64)), Err(()));
    }
}
