//! The ring `Z/2^k` for `k <= 192`.

use num_bigint::BigUint;

use crate::error::ModulusError;
use crate::ring::{ModRing, RingId, RingKind};
use crate::uint::Uint;

/// The ring `Z/2^k`, with elements kept as `Uint<3>` below `2^k`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PowerOfTwo {
    /// The exponent `k`.
    bits: u32,
    /// `2^k - 1`.
    mask: Uint<3>,
}

impl PowerOfTwo {
    /// The ring `Z/2^bits`, for `1 <= bits <= 192`.
    pub const fn new(bits: u32) -> Result<Self, ModulusError> {
        if bits == 0 {
            return Err(ModulusError::TooSmall);
        }
        if bits > Uint::<3>::BITS {
            return Err(ModulusError::TooWide);
        }
        Ok(Self {
            bits,
            mask: Uint::<3>::MAX.shr(Uint::<3>::BITS - bits),
        })
    }

    /// The exponent `k`.
    #[must_use]
    pub const fn bits(&self) -> u32 {
        self.bits
    }

    #[inline]
    fn wrap(&self, x: Uint<3>) -> Uint<3> {
        Uint(core::array::from_fn(|i| x.0[i] & self.mask.0[i]))
    }
}

impl ModRing for PowerOfTwo {
    type Elem = Uint<3>;

    fn modulus_bits(&self) -> u32 {
        self.bits + 1
    }

    fn modulus(&self) -> BigUint {
        BigUint::from(1u8) << self.bits
    }

    fn limbs(&self) -> usize {
        self.bits.div_ceil(64) as usize
    }

    fn id(&self) -> RingId {
        RingId::new(RingKind::Residues, &self.modulus())
    }

    #[inline]
    fn zero(&self) -> Uint<3> {
        Uint::ZERO
    }

    #[inline]
    fn one(&self) -> Uint<3> {
        Uint::ONE
    }

    #[inline]
    fn add(&self, a: Uint<3>, b: Uint<3>) -> Uint<3> {
        self.wrap(a.adc(&b).0)
    }

    #[inline]
    fn sub(&self, a: Uint<3>, b: Uint<3>) -> Uint<3> {
        self.wrap(a.sbb(&b).0)
    }

    #[inline]
    fn neg(&self, a: Uint<3>) -> Uint<3> {
        self.wrap(Uint::ZERO.sbb(&a).0)
    }

    #[inline]
    fn mul(&self, a: Uint<3>, b: Uint<3>) -> Uint<3> {
        self.wrap(a.mul_wide(&b).0)
    }

    #[inline]
    fn from_u64(&self, x: u64) -> Uint<3> {
        self.wrap(Uint::from_u64(x))
    }

    fn reduce_limbs(&self, le: &[u64]) -> Uint<3> {
        let mut x = Uint::ZERO;
        for (o, &w) in x.0.iter_mut().zip(le) {
            *o = w;
        }
        self.wrap(x)
    }

    fn lift_limbs(&self, x: Uint<3>, out: &mut [u64]) {
        out.copy_from_slice(&x.0[..self.limbs()]);
    }
}
