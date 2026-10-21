//! Rings `Z/M` whose modulus is a run-time value carried by a context.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;

use num_bigint::BigUint;

/// What a ring identifier names.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum RingKind {
    /// A prime field `F_p`.
    Field = 1,
    /// A residue ring `Z/M` with no claim that `M` is prime.
    Residues = 2,
}

/// The kind and modulus of a ring, for binding into a transcript.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct RingId {
    /// The kind tag.
    pub kind: RingKind,
    /// The modulus as minimal little-endian bytes.
    pub modulus: Vec<u8>,
}

impl RingId {
    /// The identifier of a ring of this kind and modulus.
    #[must_use]
    pub fn new(kind: RingKind, modulus: &BigUint) -> Self {
        Self {
            kind,
            modulus: modulus.to_bytes_le(),
        }
    }

    /// The kind byte, the modulus length as 4 little-endian bytes, then the modulus.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(5 + self.modulus.len());
        out.push(self.kind as u8);
        out.extend_from_slice(&(self.modulus.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.modulus);
        out
    }
}

/// A ring `Z/M`, with elements passed by value next to their context.
///
/// A run-time modulus cannot back the associated constants of a field type, so the context carries it instead.
pub trait ModRing: Clone + Debug + Send + Sync {
    /// An element, in whatever internal form the ring keeps.
    type Elem: Copy + Eq + Default + Debug + Send + Sync + 'static;

    /// The bit length of `M`.
    fn modulus_bits(&self) -> u32;

    /// The modulus `M`.
    fn modulus(&self) -> BigUint;

    /// The number of 64-bit limbs of a canonical lift.
    fn limbs(&self) -> usize;

    /// The identifier of this ring.
    fn id(&self) -> RingId;

    /// The additive identity.
    fn zero(&self) -> Self::Elem;

    /// The multiplicative identity.
    fn one(&self) -> Self::Elem;

    /// `a + b`.
    fn add(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem;

    /// `a - b`.
    fn sub(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem;

    /// `-a`.
    fn neg(&self, a: Self::Elem) -> Self::Elem;

    /// `a * b`.
    fn mul(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem;

    /// `a^2`.
    fn square(&self, a: Self::Elem) -> Self::Elem {
        self.mul(a, a)
    }

    /// The image of `x`.
    #[allow(clippy::wrong_self_convention)]
    fn from_u64(&self, x: u64) -> Self::Elem;

    /// The image of `x`.
    #[allow(clippy::wrong_self_convention)]
    fn from_i64(&self, x: i64) -> Self::Elem {
        let a = self.from_u64(x.unsigned_abs());
        if x < 0 { self.neg(a) } else { a }
    }

    /// The image of the nonnegative integer with these little-endian limbs, of any length.
    fn reduce_limbs(&self, le: &[u64]) -> Self::Elem;

    /// Writes the canonical lift in `[0, M)` into `out`, which holds exactly `limbs()` limbs.
    fn lift_limbs(&self, x: Self::Elem, out: &mut [u64]);

    /// Appends the canonical lift as `8 limbs()` little-endian bytes.
    fn encode(&self, x: Self::Elem, out: &mut Vec<u8>) {
        let mut limbs = vec![0; self.limbs()];
        self.lift_limbs(x, &mut limbs);
        for limb in limbs {
            out.extend_from_slice(&limb.to_le_bytes());
        }
    }

    /// Reads `8 limbs()` little-endian bytes, refusing any other length and any value `>= M`.
    fn decode(&self, le: &[u8]) -> Option<Self::Elem> {
        if le.len() != 8 * self.limbs() {
            return None;
        }
        let limbs: Vec<u64> = le
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&c| u64::from_le_bytes(c))
            .collect();
        let x = self.reduce_limbs(&limbs);
        // A value at or above `M` reduces to an element whose lift differs from it.
        let mut lift = vec![0; limbs.len()];
        self.lift_limbs(x, &mut lift);
        (lift == limbs).then_some(x)
    }

    /// `sum_i a_i b_i`.
    ///
    /// # Panics
    ///
    /// Panics when the lengths differ.
    fn dot(&self, a: &[Self::Elem], b: &[Self::Elem]) -> Self::Elem {
        assert_eq!(a.len(), b.len(), "dot inputs differ in length");
        a.iter()
            .zip(b)
            .fold(self.zero(), |acc, (&x, &y)| self.add(acc, self.mul(x, y)))
    }

    /// `a_i <- a_i b_i`.
    ///
    /// # Panics
    ///
    /// Panics when the lengths differ.
    fn mul_slices(&self, a: &mut [Self::Elem], b: &[Self::Elem]) {
        assert_eq!(a.len(), b.len(), "product inputs differ in length");
        for (x, &y) in a.iter_mut().zip(b) {
            *x = self.mul(*x, y);
        }
    }

    /// `acc_i <- acc_i + s x_i`.
    ///
    /// # Panics
    ///
    /// Panics when the lengths differ.
    fn scale_add(&self, acc: &mut [Self::Elem], x: &[Self::Elem], s: Self::Elem) {
        assert_eq!(acc.len(), x.len(), "scale-add inputs differ in length");
        for (a, &y) in acc.iter_mut().zip(x) {
            *a = self.add(*a, self.mul(s, y));
        }
    }

    /// The table `eq(point, i)` over `i` in `{0, 1}^n`, with `point[0]` on the most significant bit of `i`.
    ///
    /// # Panics
    ///
    /// Panics when `2^n` overflows `usize`.
    fn eq_table(&self, point: &[Self::Elem]) -> Vec<Self::Elem> {
        let len = 1usize
            .checked_shl(point.len() as u32)
            .expect("point too long: 2^n overflows usize");
        let mut table = vec![self.zero(); len];
        table[0] = self.one();
        for (i, &x) in point.iter().rev().enumerate() {
            let (lo, hi) = table[..2 << i].split_at_mut(1 << i);
            for (l, h) in lo.iter_mut().zip(hi) {
                *h = self.mul(*l, x);
                *l = self.sub(*l, *h);
            }
        }
        table
    }
}

/// A ring `Z/p` for a prime `p`.
pub trait ModField: ModRing {
    /// `x^-1`, or `None` for zero.
    fn inverse(&self, x: Self::Elem) -> Option<Self::Elem>;

    /// Inverts every nonzero entry in place, leaving zeros as they are.
    fn batch_inverse(&self, xs: &mut [Self::Elem]) {
        let zero = self.zero();
        // Prefix products over the nonzero entries, then one inversion unwinds them.
        let mut prefix = Vec::with_capacity(xs.len());
        let mut acc = self.one();
        for &x in xs.iter() {
            prefix.push(acc);
            if x != zero {
                acc = self.mul(acc, x);
            }
        }
        let Some(mut inv) = self.inverse(acc) else {
            return;
        };
        for (x, &p) in xs.iter_mut().zip(&prefix).rev() {
            if *x != zero {
                let next = self.mul(inv, *x);
                *x = self.mul(inv, p);
                inv = next;
            }
        }
    }
}
