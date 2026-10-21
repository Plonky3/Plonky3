//! Montgomery arithmetic modulo a run-time odd modulus.

mod portable;
#[cfg(any(
    test,
    all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512ifma"
    )
))]
mod radix52;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512ifma"
))]
mod x86_64_ifma;

use alloc::vec::Vec;

use num_bigint::BigUint;
use p3_maybe_rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub(crate) use self::portable::Wide;
use crate::error::ModulusError;
use crate::prime::{self, PrimeSample};
use crate::ring::{ModField, ModRing, RingId, RingKind};
use crate::uint::Uint;

/// An element of `Z/m` in Montgomery form `x 2^(64 L) mod m`, always below `m`.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
pub struct Residue<const L: usize>(pub(crate) Uint<L>);

impl<const L: usize> Residue<L> {
    /// The stored Montgomery form, not the canonical value.
    #[must_use]
    pub const fn montgomery(&self) -> &Uint<L> {
        &self.0
    }
}

/// The canonical value of a residue in `[0, m)`, as little-endian limbs: the wire form.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct CanonicalResidue<const L: usize>(
    #[serde(with = "p3_util::array_serialization")] pub [u64; L],
);

/// The ring `Z/m` for an odd `m` with a spare top bit, in Montgomery form with `R = 2^(64 L)`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct MontModulus<const L: usize> {
    /// The modulus.
    m: Uint<L>,
    /// `-m^-1 mod 2^64`.
    m_neg_inv: u64,
    /// `R^2 mod m`.
    r2: Uint<L>,
    /// `R mod m`, the Montgomery form of one.
    one: Uint<L>,
    /// The bit length of `m`.
    bits: u32,
}

impl<const L: usize> MontModulus<L> {
    /// The ring modulo `m`.
    ///
    /// `m` must be odd, above 1, and below `2^(64 L - 1)`: the spare bit keeps lazy sums carry-free.
    pub const fn new(m: Uint<L>) -> Result<Self, ModulusError> {
        if L == 0 || m.bits() >= 64 * L as u32 {
            return Err(ModulusError::TooWide);
        }
        if !m.is_odd() {
            return Err(ModulusError::Even);
        }
        if m.bits() < 2 {
            return Err(ModulusError::TooSmall);
        }
        // R mod m by doubling 1 a total of 64 L times, then R^2 mod m by doubling R mod m as often.
        let mut one = Uint::ONE;
        let mut i = 0;
        while i < 64 * L {
            one = portable::double(&one, &m);
            i += 1;
        }
        let mut r2 = one;
        let mut i = 0;
        while i < 64 * L {
            r2 = portable::double(&r2, &m);
            i += 1;
        }
        Ok(Self {
            m,
            m_neg_inv: portable::neg_inv(m.0[0]),
            r2,
            one,
            bits: m.bits(),
        })
    }

    /// The modulus as limbs.
    #[must_use]
    pub const fn modulus_uint(&self) -> &Uint<L> {
        &self.m
    }

    /// The residue of `x`, for any `x < 2^(64 L)`.
    #[inline]
    #[must_use]
    pub const fn from_uint(&self, x: &Uint<L>) -> Residue<L> {
        Residue(portable::mul(x, &self.r2, &self.m, self.m_neg_inv))
    }

    /// The canonical value of `x` in `[0, m)`.
    #[inline]
    #[must_use]
    pub const fn to_uint(&self, x: Residue<L>) -> Uint<L> {
        portable::mul(&x.0, &Uint::ONE, &self.m, self.m_neg_inv)
    }

    /// The residue with this canonical value, or `None` when it is not below `m`.
    #[must_use]
    pub fn from_canonical(&self, x: &CanonicalResidue<L>) -> Option<Residue<L>> {
        let x = Uint(x.0);
        (x < self.m).then(|| self.from_uint(&x))
    }

    /// The canonical value of `x`.
    #[must_use]
    pub const fn to_canonical(&self, x: Residue<L>) -> CanonicalResidue<L> {
        CanonicalResidue(self.to_uint(x).0)
    }

    /// `x / 2`.
    #[must_use]
    pub const fn half(&self, x: Residue<L>) -> Residue<L> {
        // Halving commutes with the Montgomery factor.
        Residue(portable::half(&x.0, &self.m))
    }

    /// `base^e` for the exponent with these little-endian limbs.
    ///
    /// Runs in time that depends on the exponent, which must be public.
    #[must_use]
    pub fn pow(&self, base: Residue<L>, e: &[u64]) -> Residue<L> {
        let bits = e
            .iter()
            .rposition(|&w| w != 0)
            .map_or(0, |i| 64 * i as u32 + 64 - e[i].leading_zeros());
        let bit = |i: u32| (e[(i / 64) as usize] >> (i % 64)) & 1 == 1;
        // Sliding windows of up to 4 bits over the odd powers base^1, base^3, ..., base^15.
        let sq = self.square(base);
        let mut odd = [base; 8];
        for i in 1..8 {
            odd[i] = self.mul(odd[i - 1], sq);
        }
        let mut acc = self.one();
        let mut i = bits;
        while i > 0 {
            if !bit(i - 1) {
                acc = self.square(acc);
                i -= 1;
                continue;
            }
            let mut low = i.saturating_sub(4);
            while !bit(low) {
                low += 1;
            }
            let mut window = 0usize;
            for j in (low..i).rev() {
                acc = self.square(acc);
                window = (window << 1) | bit(j) as usize;
            }
            acc = self.mul(acc, odd[window >> 1]);
            i = low;
        }
        acc
    }

    /// The exact sum `sum_i a_i b_i` of the stored forms.
    fn dot_serial(&self, a: &[Residue<L>], b: &[Residue<L>]) -> Wide<L> {
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512ifma"
        ))]
        if let Some(sum) = x86_64_ifma::dot(a, b) {
            return sum;
        }
        // Two sums keep two carry chains in flight.
        let mut even = Wide::ZERO;
        let mut odd = Wide::ZERO;
        let (pa, ra) = a.as_chunks::<2>();
        let (pb, rb) = b.as_chunks::<2>();
        for (x, y) in pa.iter().zip(pb) {
            even.add_product(&x[0].0, &y[0].0);
            odd.add_product(&x[1].0, &y[1].0);
        }
        for (x, y) in ra.iter().zip(rb) {
            even.add_product(&x.0, &y.0);
        }
        even.add(&odd);
        even
    }

    /// The residue whose stored form is `S 2^(-64 L) mod m`, for the sum `S`.
    const fn reduce_wide(&self, sum: &Wide<L>) -> Residue<L> {
        let Wide { low, high, top } = *sum;
        // S R^-1 = low R^-1 + high + top R (mod m); each term is one Montgomery product.
        let a = portable::mul(&low, &Uint::ONE, &self.m, self.m_neg_inv);
        let b = portable::mul(&high, &self.one, &self.m, self.m_neg_inv);
        let c = portable::mul(&Uint::from_u64(top), &self.r2, &self.m, self.m_neg_inv);
        Residue(portable::add(&portable::add(&a, &b, &self.m), &c, &self.m))
    }

    fn mul_slices_serial(&self, a: &mut [Residue<L>], b: &[Residue<L>]) {
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512ifma"
        ))]
        if x86_64_ifma::mul_slices(self, a, b) {
            return;
        }
        for (x, y) in a.iter_mut().zip(b) {
            *x = self.mul(*x, *y);
        }
    }

    fn eq_step_serial(&self, lo: &mut [Residue<L>], hi: &mut [Residue<L>], x: Residue<L>) {
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512ifma"
        ))]
        if x86_64_ifma::eq_step(self, lo, hi, x) {
            return;
        }
        for (l, h) in lo.iter_mut().zip(hi) {
            *h = self.mul(*l, x);
            *l = self.sub(*l, *h);
        }
    }

    fn scale_add_serial(&self, acc: &mut [Residue<L>], x: &[Residue<L>], s: Residue<L>) {
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512ifma"
        ))]
        if x86_64_ifma::scale_add(self, acc, x, s) {
            return;
        }
        for (a, y) in acc.iter_mut().zip(x) {
            *a = self.add(*a, self.mul(s, *y));
        }
    }
}

/// Elements per parallel task of a slice kernel.
fn chunk_len<T>(len: usize) -> usize {
    min_task_len(len, 3 * size_of::<T>()).max(64)
}

impl<const L: usize> ModRing for MontModulus<L> {
    type Elem = Residue<L>;

    fn modulus_bits(&self) -> u32 {
        self.bits
    }

    fn modulus(&self) -> BigUint {
        self.m.into()
    }

    fn limbs(&self) -> usize {
        L
    }

    fn id(&self) -> RingId {
        RingId::new(RingKind::Residues, &self.m.into())
    }

    #[inline]
    fn zero(&self) -> Residue<L> {
        Residue(Uint::ZERO)
    }

    #[inline]
    fn one(&self) -> Residue<L> {
        Residue(self.one)
    }

    #[inline]
    fn add(&self, a: Residue<L>, b: Residue<L>) -> Residue<L> {
        Residue(portable::add(&a.0, &b.0, &self.m))
    }

    #[inline]
    fn sub(&self, a: Residue<L>, b: Residue<L>) -> Residue<L> {
        Residue(portable::sub(&a.0, &b.0, &self.m))
    }

    #[inline]
    fn neg(&self, a: Residue<L>) -> Residue<L> {
        Residue(portable::sub(&Uint::ZERO, &a.0, &self.m))
    }

    #[inline]
    fn mul(&self, a: Residue<L>, b: Residue<L>) -> Residue<L> {
        Residue(portable::mul(&a.0, &b.0, &self.m, self.m_neg_inv))
    }

    #[inline]
    fn square(&self, a: Residue<L>) -> Residue<L> {
        if L == 2 {
            let r = portable::square2(
                [a.0.0[0], a.0.0[1]],
                [self.m.0[0], self.m.0[1]],
                self.m_neg_inv,
            );
            return Residue(Uint(core::array::from_fn(|i| r[i])));
        }
        self.mul(a, a)
    }

    #[inline]
    fn from_u64(&self, x: u64) -> Residue<L> {
        self.from_uint(&Uint::from_u64(x))
    }

    fn reduce_limbs(&self, le: &[u64]) -> Residue<L> {
        // Horner over L-limb chunks from the top: Mont(V R + c) = Mont(V) R^2 R^-1 + c R^2 R^-1.
        let mut acc = self.zero();
        for chunk in le.chunks(L).rev() {
            let mut c = Uint::ZERO;
            c.0[..chunk.len()].copy_from_slice(chunk);
            let shifted = self.mul(acc, Residue(self.r2));
            acc = self.add(shifted, self.from_uint(&c));
        }
        acc
    }

    fn lift_limbs(&self, x: Residue<L>, out: &mut [u64]) {
        out.copy_from_slice(&self.to_uint(x).0);
    }

    fn encode(&self, x: Residue<L>, out: &mut Vec<u8>) {
        for limb in self.to_uint(x).0 {
            out.extend_from_slice(&limb.to_le_bytes());
        }
    }

    fn decode(&self, le: &[u8]) -> Option<Residue<L>> {
        if le.len() != 8 * L {
            return None;
        }
        let x = Uint::from_le_bytes(le)?;
        (x < self.m).then(|| self.from_uint(&x))
    }

    fn dot(&self, a: &[Residue<L>], b: &[Residue<L>]) -> Residue<L> {
        assert_eq!(a.len(), b.len(), "dot inputs differ in length");
        let chunk = chunk_len::<Residue<L>>(a.len());
        let sum = a
            .par_chunks(chunk)
            .zip(b.par_chunks(chunk))
            .par_fold_reduce(
                || Wide::ZERO,
                |mut acc, (a, b)| {
                    acc.add(&self.dot_serial(a, b));
                    acc
                },
                |mut x, y| {
                    x.add(&y);
                    x
                },
            );
        self.reduce_wide(&sum)
    }

    fn mul_slices(&self, a: &mut [Residue<L>], b: &[Residue<L>]) {
        assert_eq!(a.len(), b.len(), "product inputs differ in length");
        let chunk = chunk_len::<Residue<L>>(a.len());
        a.par_chunks_mut(chunk)
            .zip(b.par_chunks(chunk))
            .for_each(|(a, b)| self.mul_slices_serial(a, b));
    }

    fn scale_add(&self, acc: &mut [Residue<L>], x: &[Residue<L>], s: Residue<L>) {
        assert_eq!(acc.len(), x.len(), "scale-add inputs differ in length");
        let chunk = chunk_len::<Residue<L>>(acc.len());
        acc.par_chunks_mut(chunk)
            .zip(x.par_chunks(chunk))
            .for_each(|(acc, x)| self.scale_add_serial(acc, x, s));
    }

    fn eq_table(&self, point: &[Residue<L>]) -> Vec<Residue<L>> {
        let len = 1usize
            .checked_shl(point.len() as u32)
            .expect("point too long: 2^n overflows usize");
        let mut table = alloc::vec![self.zero(); len];
        table[0] = self.one();
        for (i, &x) in point.iter().rev().enumerate() {
            let (lo, hi) = table[..2 << i].split_at_mut(1 << i);
            let chunk = chunk_len::<Residue<L>>(lo.len());
            lo.par_chunks_mut(chunk)
                .zip(hi.par_chunks_mut(chunk))
                .for_each(|(lo, hi)| self.eq_step_serial(lo, hi, x));
        }
        table
    }
}

/// The prime field `F_p` in Montgomery form, constructible only for a prime `p`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct MontField<const L: usize>(MontModulus<L>);

impl<const L: usize> MontField<L> {
    /// The field modulo `p`, after BPSW and 64 Miller-Rabin rounds on bases from a fixed-seed stream.
    pub fn new(p: Uint<L>) -> Result<Self, ModulusError> {
        let ring = MontModulus::new(p)?;
        if prime::is_prime_fixed(&ring) {
            Ok(Self(ring))
        } else {
            Err(ModulusError::Composite)
        }
    }

    /// The field of a sampled prime, which the sampler has already tested.
    #[must_use]
    pub const fn from_sample(s: PrimeSample<L>) -> Self {
        Self(s.into_ring())
    }

    /// The underlying ring.
    #[must_use]
    pub const fn ring(&self) -> &MontModulus<L> {
        &self.0
    }
}

impl<const L: usize> ModRing for MontField<L> {
    type Elem = Residue<L>;

    fn modulus_bits(&self) -> u32 {
        self.0.modulus_bits()
    }

    fn modulus(&self) -> BigUint {
        self.0.modulus()
    }

    fn limbs(&self) -> usize {
        L
    }

    fn id(&self) -> RingId {
        RingId::new(RingKind::Field, &self.0.m.into())
    }

    #[inline]
    fn zero(&self) -> Residue<L> {
        self.0.zero()
    }

    #[inline]
    fn one(&self) -> Residue<L> {
        self.0.one()
    }

    #[inline]
    fn add(&self, a: Residue<L>, b: Residue<L>) -> Residue<L> {
        self.0.add(a, b)
    }

    #[inline]
    fn sub(&self, a: Residue<L>, b: Residue<L>) -> Residue<L> {
        self.0.sub(a, b)
    }

    #[inline]
    fn neg(&self, a: Residue<L>) -> Residue<L> {
        self.0.neg(a)
    }

    #[inline]
    fn mul(&self, a: Residue<L>, b: Residue<L>) -> Residue<L> {
        self.0.mul(a, b)
    }

    #[inline]
    fn square(&self, a: Residue<L>) -> Residue<L> {
        self.0.square(a)
    }

    #[inline]
    fn from_u64(&self, x: u64) -> Residue<L> {
        self.0.from_u64(x)
    }

    fn reduce_limbs(&self, le: &[u64]) -> Residue<L> {
        self.0.reduce_limbs(le)
    }

    fn lift_limbs(&self, x: Residue<L>, out: &mut [u64]) {
        self.0.lift_limbs(x, out);
    }

    fn encode(&self, x: Residue<L>, out: &mut Vec<u8>) {
        self.0.encode(x, out);
    }

    fn decode(&self, le: &[u8]) -> Option<Residue<L>> {
        self.0.decode(le)
    }

    fn dot(&self, a: &[Residue<L>], b: &[Residue<L>]) -> Residue<L> {
        self.0.dot(a, b)
    }

    fn mul_slices(&self, a: &mut [Residue<L>], b: &[Residue<L>]) {
        self.0.mul_slices(a, b);
    }

    fn scale_add(&self, acc: &mut [Residue<L>], x: &[Residue<L>], s: Residue<L>) {
        self.0.scale_add(acc, x, s);
    }

    fn eq_table(&self, point: &[Residue<L>]) -> Vec<Residue<L>> {
        self.0.eq_table(point)
    }
}

impl<const L: usize> ModField for MontField<L> {
    fn inverse(&self, x: Residue<L>) -> Option<Residue<L>> {
        if x == self.zero() {
            return None;
        }
        // Fermat: x^(p - 2); p - 2 does not borrow since p >= 3.
        let (e, _) = self.0.m.sbb(&Uint::from_u64(2));
        Some(self.0.pow(x, &e.0))
    }
}
