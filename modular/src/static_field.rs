//! Compile-time prime fields seen as run-time rings.

use core::fmt;
use core::marker::PhantomData;

use num_bigint::BigUint;
use p3_field::{PrimeField, PrimeField64};

use crate::ring::{ModField, ModRing, RingId, RingKind};

/// A prime field `F` below `2^64` as a ring with elements `F`.
pub struct StaticPrime<F>(PhantomData<fn() -> F>);

/// A prime field `F` of any size as a ring with elements `F`, lifted through its canonical integer.
pub struct StaticBigPrime<F>(PhantomData<fn() -> F>);

impl<F> StaticPrime<F> {
    /// The ring.
    #[must_use]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<F> StaticBigPrime<F> {
    /// The ring.
    #[must_use]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<F> Default for StaticPrime<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> Default for StaticBigPrime<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> Clone for StaticPrime<F> {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl<F> Clone for StaticBigPrime<F> {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl<F> fmt::Debug for StaticPrime<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("StaticPrime")
    }
}

impl<F> fmt::Debug for StaticBigPrime<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("StaticBigPrime")
    }
}

/// `2^64` in `F`.
fn two_pow_64<F: PrimeField>() -> F {
    F::from_u64(1 << 32).square()
}

impl<F: PrimeField64> ModRing for StaticPrime<F> {
    type Elem = F;

    fn modulus_bits(&self) -> u32 {
        64 - F::ORDER_U64.leading_zeros()
    }

    fn modulus(&self) -> BigUint {
        F::ORDER_U64.into()
    }

    fn limbs(&self) -> usize {
        1
    }

    fn id(&self) -> RingId {
        RingId::new(RingKind::Field, &F::ORDER_U64.into())
    }

    #[inline]
    fn zero(&self) -> F {
        F::ZERO
    }

    #[inline]
    fn one(&self) -> F {
        F::ONE
    }

    #[inline]
    fn add(&self, a: F, b: F) -> F {
        a + b
    }

    #[inline]
    fn sub(&self, a: F, b: F) -> F {
        a - b
    }

    #[inline]
    fn neg(&self, a: F) -> F {
        -a
    }

    #[inline]
    fn mul(&self, a: F, b: F) -> F {
        a * b
    }

    #[inline]
    fn square(&self, a: F) -> F {
        a.square()
    }

    #[inline]
    fn from_u64(&self, x: u64) -> F {
        F::from_u64(x)
    }

    fn reduce_limbs(&self, le: &[u64]) -> F {
        let radix = two_pow_64::<F>();
        le.iter()
            .rev()
            .fold(F::ZERO, |acc, &limb| acc * radix + F::from_u64(limb))
    }

    fn lift_limbs(&self, x: F, out: &mut [u64]) {
        out.copy_from_slice(&[x.as_canonical_u64()]);
    }

    fn decode(&self, le: &[u8]) -> Option<F> {
        let bytes: [u8; 8] = le.try_into().ok()?;
        F::from_canonical_checked(u64::from_le_bytes(bytes))
    }

    fn dot(&self, a: &[F], b: &[F]) -> F {
        assert_eq!(a.len(), b.len(), "dot inputs differ in length");
        // Chunks of 8 take the field's delayed-reduction dot product.
        let (ca, ra) = a.as_chunks::<8>();
        let (cb, rb) = b.as_chunks::<8>();
        let head = ca
            .iter()
            .zip(cb)
            .map(|(x, y)| F::dot_product(x, y))
            .sum::<F>();
        ra.iter().zip(rb).fold(head, |acc, (&x, &y)| acc + x * y)
    }
}

impl<F: PrimeField64> ModField for StaticPrime<F> {
    fn inverse(&self, x: F) -> Option<F> {
        x.try_inverse()
    }
}

impl<F: PrimeField> ModRing for StaticBigPrime<F> {
    type Elem = F;

    fn modulus_bits(&self) -> u32 {
        F::order().bits() as u32
    }

    fn modulus(&self) -> BigUint {
        F::order()
    }

    fn limbs(&self) -> usize {
        F::order().bits().div_ceil(64) as usize
    }

    fn id(&self) -> RingId {
        RingId::new(RingKind::Field, &F::order())
    }

    #[inline]
    fn zero(&self) -> F {
        F::ZERO
    }

    #[inline]
    fn one(&self) -> F {
        F::ONE
    }

    #[inline]
    fn add(&self, a: F, b: F) -> F {
        a + b
    }

    #[inline]
    fn sub(&self, a: F, b: F) -> F {
        a - b
    }

    #[inline]
    fn neg(&self, a: F) -> F {
        -a
    }

    #[inline]
    fn mul(&self, a: F, b: F) -> F {
        a * b
    }

    #[inline]
    fn square(&self, a: F) -> F {
        a.square()
    }

    #[inline]
    fn from_u64(&self, x: u64) -> F {
        F::from_u64(x)
    }

    fn reduce_limbs(&self, le: &[u64]) -> F {
        let radix = two_pow_64::<F>();
        le.iter()
            .rev()
            .fold(F::ZERO, |acc, &limb| acc * radix + F::from_u64(limb))
    }

    fn lift_limbs(&self, x: F, out: &mut [u64]) {
        out.fill(0);
        for (o, d) in out
            .iter_mut()
            .zip(x.as_canonical_biguint().iter_u64_digits())
        {
            *o = d;
        }
    }
}

impl<F: PrimeField> ModField for StaticBigPrime<F> {
    fn inverse(&self, x: F) -> Option<F> {
        x.try_inverse()
    }
}
