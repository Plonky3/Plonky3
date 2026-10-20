//! Eight challenge-field elements in coefficient planes: quadword `l` of plane `k` is coefficient `k` of element `l`.
//!
//! A product needs no packing: CLMUL immediate 0x00 multiplies the even elements, 0x11 the odd ones.

use core::arch::x86_64::{
    __m512i, _mm_cvtsi128_si64, _mm_extract_epi64, _mm_xor_si128, _mm256_castsi256_si128,
    _mm256_extracti128_si256, _mm256_xor_si256, _mm512_castsi512_si256, _mm512_clmulepi64_epi128,
    _mm512_extracti64x4_epi64, _mm512_loadu_si512, _mm512_permutexvar_epi64, _mm512_set1_epi64,
    _mm512_setzero_si512, _mm512_slli_epi64, _mm512_srli_epi64, _mm512_ternarylogic_epi64,
    _mm512_unpackhi_epi64, _mm512_unpacklo_epi64, _mm512_xor_si512,
};

use p3_binary_field::{Poly64, Poly192};

/// Eight elements, one register per coefficient.
#[derive(Clone, Copy, Debug)]
pub(crate) struct F192x8(pub(crate) [__m512i; 3]);

/// A sum of products, unreduced: every 128-bit lane of plane `k` holds part of coefficient `k`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct F192x8Sum([__m512i; 3]);

/// Karatsuba recombination and the fold by `y^3 = y + 1`, as one XOR network.
#[inline(always)]
fn fold<V: Copy>(p0: V, p1: V, p2: V, p01: V, p02: V, p12: V, xor: impl Fn(V, V) -> V) -> [V; 3] {
    // Shared by the first two coefficients.
    let q = xor(p0, p12);
    [xor(xor(q, p1), p2), xor(q, p01), xor(xor(p0, p1), p02)]
}

/// Reduce `hi x^64 + lo` lane-wise modulo `x^64 + x^4 + x^3 + x + 1`.
#[inline(always)]
fn reduce_lanes(lo: __m512i, hi: __m512i) -> __m512i {
    // SAFETY: this module compiles only where the crate enables the needed target features.
    unsafe {
        // The bits of `hi * 0x1B` shifted past x^63.
        let spill = _mm512_xor_si512(
            _mm512_xor_si512(_mm512_srli_epi64::<63>(hi), _mm512_srli_epi64::<61>(hi)),
            _mm512_srli_epi64::<60>(hi),
        );
        let v = _mm512_xor_si512(hi, spill);
        let f = _mm512_xor_si512(
            _mm512_xor_si512(v, _mm512_slli_epi64::<1>(v)),
            _mm512_xor_si512(_mm512_slli_epi64::<3>(v), _mm512_slli_epi64::<4>(v)),
        );
        _mm512_xor_si512(lo, f)
    }
}

/// `x^64` in the committed field: `x^4 + x^3 + x + 1`.
const X64: Poly64 = Poly64::new(0x1B);

impl F192x8 {
    /// Eight elements, from coefficient planes.
    #[inline(always)]
    pub(crate) fn load(values: &[Poly192; 8]) -> Self {
        let planes: [[u64; 8]; 3] = core::array::from_fn(|k| {
            core::array::from_fn(|l| values[l].coefficients()[k].to_bits())
        });
        // SAFETY: each plane is eight quadwords, one register, and the target features are enabled.
        Self(planes.map(|p| unsafe { _mm512_loadu_si512(p.as_ptr().cast()) }))
    }

    /// One element in every lane.
    #[inline(always)]
    pub(crate) fn splat(value: Poly192) -> Self {
        // SAFETY: this module compiles only where the crate enables the needed target features.
        Self(
            value
                .coefficients()
                .map(|c| unsafe { _mm512_set1_epi64(c.to_bits() as i64) }),
        )
    }

    /// Element `l` in every lane.
    #[inline(always)]
    pub(crate) fn broadcast(self, l: usize) -> Self {
        // SAFETY: this module compiles only where the crate enables the needed target features.
        unsafe {
            let index = _mm512_set1_epi64(l as i64);
            Self(self.0.map(|p| _mm512_permutexvar_epi64(index, p)))
        }
    }

    /// The lane-wise products with base-field elements, one per lane.
    #[inline(always)]
    pub(crate) fn mul_base(self, base: __m512i) -> Self {
        // SAFETY: this module compiles only where the crate enables the needed target features.
        Self(self.0.map(|p| unsafe {
            // Immediates 0x00 and 0x11 take the even and the odd lanes of both operands.
            let even = _mm512_clmulepi64_epi128::<0x00>(p, base);
            let odd = _mm512_clmulepi64_epi128::<0x11>(p, base);
            reduce_lanes(
                _mm512_unpacklo_epi64(even, odd),
                _mm512_unpackhi_epi64(even, odd),
            )
        }))
    }

    /// The lane-wise sum.
    #[inline(always)]
    pub(crate) fn add(self, rhs: Self) -> Self {
        // SAFETY: this module compiles only where the crate enables the needed target features.
        Self([0, 1, 2].map(|k| unsafe { _mm512_xor_si512(self.0[k], rhs.0[k]) }))
    }

    /// The products of the even (`IMM = 0x00`) or odd (`IMM = 0x11`) elements, one per lane.
    #[inline(always)]
    fn karatsuba<const IMM: i32>(self, rhs: Self) -> [__m512i; 3] {
        let ([a0, a1, a2], [b0, b1, b2]) = (self.0, rhs.0);
        // SAFETY: this module compiles only where the crate enables the needed target features.
        let mul = |x, y| unsafe { _mm512_clmulepi64_epi128::<IMM>(x, y) };
        // SAFETY: as above.
        let xor = |x, y| unsafe { _mm512_xor_si512(x, y) };
        fold(
            mul(a0, b0),
            mul(a1, b1),
            mul(a2, b2),
            mul(xor(a0, a1), xor(b0, b1)),
            mul(xor(a0, a2), xor(b0, b2)),
            mul(xor(a1, a2), xor(b1, b2)),
            xor,
        )
    }

    /// The eight lane-wise products.
    #[inline(always)]
    pub(crate) fn mul(self, rhs: Self) -> Self {
        let (even, odd) = (self.karatsuba::<0x00>(rhs), self.karatsuba::<0x11>(rhs));
        // Lane j of `even` is element 2j's product and of `odd` element 2j + 1's.
        // SAFETY: this module compiles only where the crate enables the needed target features.
        Self([0, 1, 2].map(|k| unsafe {
            reduce_lanes(
                _mm512_unpacklo_epi64(even[k], odd[k]),
                _mm512_unpackhi_epi64(even[k], odd[k]),
            )
        }))
    }
}

impl F192x8Sum {
    /// The empty sum.
    #[inline(always)]
    pub(crate) fn zero() -> Self {
        // SAFETY: this module compiles only where the crate enables the needed target features.
        Self([unsafe { _mm512_setzero_si512() }; 3])
    }

    /// Add the eight lane-wise products of `a` and `b`.
    #[inline(always)]
    pub(crate) fn mul_add(&mut self, a: F192x8, b: F192x8) {
        let (even, odd) = (a.karatsuba::<0x00>(b), a.karatsuba::<0x11>(b));
        for k in 0..3 {
            // SAFETY: this module compiles only where the crate enables the needed target features.
            self.0[k] = unsafe { _mm512_ternarylogic_epi64::<0x96>(self.0[k], even[k], odd[k]) };
        }
    }

    /// The whole sum, its lanes folded together and reduced.
    #[inline(always)]
    pub(crate) fn total(self) -> Poly192 {
        // SAFETY: this module compiles only where the crate enables the needed target features.
        Poly192::new(self.0.map(|a| unsafe {
            let half =
                _mm256_xor_si256(_mm512_castsi512_si256(a), _mm512_extracti64x4_epi64::<1>(a));
            let q = _mm_xor_si128(
                _mm256_castsi256_si128(half),
                _mm256_extracti128_si256::<1>(half),
            );
            let (lo, hi) = (
                _mm_cvtsi128_si64(q) as u64,
                _mm_extract_epi64::<1>(q) as u64,
            );
            Poly64::new(lo) + Poly64::new(hi) * X64
        }))
    }
}

#[cfg(test)]
mod tests {
    use core::arch::x86_64::_mm512_storeu_si512;

    use proptest::prelude::*;

    use super::*;

    /// Eight elements from raw coefficients.
    fn elements(raw: [[u64; 3]; 8]) -> [Poly192; 8] {
        raw.map(|c| Poly192::new(c.map(Poly64::new)))
    }

    proptest! {
        #[test]
        fn planar_products_are_field_products(a: [[u64; 3]; 8], b: [[u64; 3]; 8], c: [[u64; 3]; 8]) {
            let (a, b, c) = (elements(a), elements(b), elements(c));
            let (pa, pb, pc) = (F192x8::load(&a), F192x8::load(&b), F192x8::load(&c));
            let mut sum = F192x8Sum::zero();
            sum.mul_add(pa.add(pc), pb);
            let prod = pa.mul(pb);
            let mut out = [0u64; 24];
            for k in 0..3 {
                // SAFETY: each store fills eight of the 24 quadwords.
                unsafe { _mm512_storeu_si512(out[8 * k..].as_mut_ptr().cast(), prod.0[k]) };
            }
            let want: Poly192 = (0..8).map(|l| (a[l] + c[l]) * b[l]).sum();
            prop_assert_eq!(sum.total(), want);
            for l in 0..8 {
                let got = Poly192::new([0, 1, 2].map(|k| Poly64::new(out[8 * k + l])));
                prop_assert_eq!(got, a[l] * b[l]);
            }
        }
    }
}
