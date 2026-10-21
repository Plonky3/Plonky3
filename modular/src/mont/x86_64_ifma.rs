//! Two-limb Montgomery slice kernels on AVX-512 IFMA, eight residues per vector in radix `2^52`.
//!
//! A residue below `2^127` takes three 52-bit digits, and REDC runs three digit rounds, dividing by `2^156`.
//!
//! The right operand enters as `b 2^28`, still three digits, so the quotient is `a b 2^-128` as the scalar path computes.

use core::arch::x86_64::{
    __m512i, _mm512_add_epi64, _mm512_and_si512, _mm512_cmplt_epi64_mask, _mm512_loadu_si512,
    _mm512_madd52hi_epu64, _mm512_madd52lo_epu64, _mm512_mask_blend_epi64, _mm512_or_si512,
    _mm512_set1_epi64, _mm512_setzero_si512, _mm512_slli_epi64, _mm512_srai_epi64,
    _mm512_srli_epi64, _mm512_storeu_si512, _mm512_sub_epi64, _mm512_unpackhi_epi64,
    _mm512_unpacklo_epi64,
};

use super::portable::Wide;
use super::{MontModulus, Residue};
use crate::ring::ModRing;
use crate::uint::{Uint, adc};

/// Residues per vector.
const LANES: usize = 8;

/// Products a column sums before it is flushed: each adds at most six 52-bit halves, so 512 stay below `2^64`.
const FLUSH: usize = 512;

/// The modulus digits and `-q^-1 mod 2^52`, broadcast.
struct Consts {
    q: [__m512i; 3],
    q_inv: __m512i,
    mask: __m512i,
}

impl Consts {
    #[inline]
    #[target_feature(enable = "avx512f,avx512ifma")]
    fn new(ring: &MontModulus<2>) -> Self {
        let m = ring.modulus_uint().0;
        let q = super::radix52::digits(m, 0).map(|d| _mm512_set1_epi64(d as i64));
        Self {
            q,
            q_inv: _mm512_set1_epi64((ring.m_neg_inv & super::radix52::MASK) as i64),
            mask: _mm512_set1_epi64(super::radix52::MASK as i64),
        }
    }
}

/// Views two-limb residues as limbs.
fn limbs(x: &[Residue<2>]) -> &[u64] {
    // SAFETY: `Residue<2>` and `Uint<2>` are transparent over `[u64; 2]`, so the slice is `2 len` contiguous limbs.
    unsafe { core::slice::from_raw_parts(x.as_ptr().cast::<u64>(), x.len() * 2) }
}

/// Views two-limb residues as mutable limbs.
fn limbs_mut(x: &mut [Residue<2>]) -> &mut [u64] {
    // SAFETY: as in `limbs`, and the unique borrow carries over to the limbs.
    unsafe { core::slice::from_raw_parts_mut(x.as_mut_ptr().cast::<u64>(), x.len() * 2) }
}

/// Eight residues as their low and high limbs, lanes in the order 0, 4, 1, 5, 2, 6, 3, 7.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn load(chunk: &[u64; 2 * LANES]) -> (__m512i, __m512i) {
    // SAFETY: the chunk holds 16 limbs, and the loads accept any alignment.
    let (v0, v1) = unsafe {
        (
            _mm512_loadu_si512(chunk.as_ptr().cast()),
            _mm512_loadu_si512(chunk.as_ptr().add(LANES).cast()),
        )
    };
    (_mm512_unpacklo_epi64(v0, v1), _mm512_unpackhi_epi64(v0, v1))
}

/// Stores what `load` read, inverting its lane order.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn store(chunk: &mut [u64; 2 * LANES], lo: __m512i, hi: __m512i) {
    let v0 = _mm512_unpacklo_epi64(lo, hi);
    let v1 = _mm512_unpackhi_epi64(lo, hi);
    // SAFETY: the chunk holds 16 limbs, and the stores accept any alignment.
    unsafe {
        _mm512_storeu_si512(chunk.as_mut_ptr().cast(), v0);
        _mm512_storeu_si512(chunk.as_mut_ptr().add(LANES).cast(), v1);
    }
}

/// The digits of `x` (shift 0) or of `x 2^28` (shift 28), for `x < 2^127`.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn split<const SHIFTED: bool>(lo: __m512i, hi: __m512i, mask: __m512i) -> [__m512i; 3] {
    if SHIFTED {
        [
            _mm512_and_si512(_mm512_slli_epi64::<28>(lo), mask),
            _mm512_and_si512(
                _mm512_or_si512(_mm512_srli_epi64::<24>(lo), _mm512_slli_epi64::<40>(hi)),
                mask,
            ),
            _mm512_srli_epi64::<12>(hi),
        ]
    } else {
        [
            _mm512_and_si512(lo, mask),
            _mm512_and_si512(
                _mm512_or_si512(_mm512_srli_epi64::<52>(lo), _mm512_slli_epi64::<12>(hi)),
                mask,
            ),
            _mm512_srli_epi64::<40>(hi),
        ]
    }
}

/// `a b' 2^-156 mod q`, canonical, as low and high limbs, for digits `a < q` and `b' < 2^156`.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn mont_mul(a: &[__m512i; 3], b: &[__m512i; 3], k: &Consts) -> (__m512i, __m512i) {
    let zero = _mm512_setzero_si512();
    // The full product in six unnormalized columns.
    let mut t = [zero; 6];
    for i in 0..3 {
        for j in 0..3 {
            t[i + j] = _mm512_madd52lo_epu64(t[i + j], a[i], b[j]);
            t[i + j + 1] = _mm512_madd52hi_epu64(t[i + j + 1], a[i], b[j]);
        }
    }
    // Three REDC rounds: each clears the low 52 bits of a column and carries the rest up.
    for i in 0..3 {
        let u = _mm512_madd52lo_epu64(zero, t[i], k.q_inv);
        for j in 0..3 {
            t[i + j] = _mm512_madd52lo_epu64(t[i + j], u, k.q[j]);
            t[i + j + 1] = _mm512_madd52hi_epu64(t[i + j + 1], u, k.q[j]);
        }
        t[i + 1] = _mm512_add_epi64(t[i + 1], _mm512_srli_epi64::<52>(t[i]));
    }
    // Normalize the quotient, below 2q < 2^128, into digits.
    let r1 = _mm512_add_epi64(t[4], _mm512_srli_epi64::<52>(t[3]));
    let r0 = _mm512_and_si512(t[3], k.mask);
    let r2 = _mm512_add_epi64(t[5], _mm512_srli_epi64::<52>(r1));
    let r1 = _mm512_and_si512(r1, k.mask);
    sub_if_ge(r0, r1, r2, k)
}

/// The digits `r < 2q` minus `q` when that is nonnegative, as low and high limbs.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn sub_if_ge(r0: __m512i, r1: __m512i, r2: __m512i, k: &Consts) -> (__m512i, __m512i) {
    let d0 = _mm512_sub_epi64(r0, k.q[0]);
    let d1 = _mm512_add_epi64(_mm512_sub_epi64(r1, k.q[1]), _mm512_srai_epi64::<52>(d0));
    let d2 = _mm512_add_epi64(_mm512_sub_epi64(r2, k.q[2]), _mm512_srai_epi64::<52>(d1));
    let keep = _mm512_cmplt_epi64_mask(d2, _mm512_setzero_si512());
    let s0 = _mm512_mask_blend_epi64(keep, _mm512_and_si512(d0, k.mask), r0);
    let s1 = _mm512_mask_blend_epi64(keep, _mm512_and_si512(d1, k.mask), r1);
    let s2 = _mm512_mask_blend_epi64(keep, d2, r2);
    (
        _mm512_or_si512(s0, _mm512_slli_epi64::<52>(s1)),
        _mm512_or_si512(_mm512_srli_epi64::<12>(s1), _mm512_slli_epi64::<40>(s2)),
    )
}

/// `a b mod q` for `a < q`: the Montgomery product, as the scalar path computes it.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn mul_vec(a: (__m512i, __m512i), b: (__m512i, __m512i), k: &Consts) -> (__m512i, __m512i) {
    mont_mul(
        &split::<false>(a.0, a.1, k.mask),
        &split::<true>(b.0, b.1, k.mask),
        k,
    )
}

/// `a + b mod q` for `a, b < q` below `2^127`.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn add_vec(a: (__m512i, __m512i), b: (__m512i, __m512i), k: &Consts) -> (__m512i, __m512i) {
    // Digit sums stay below 2^53 and the total below 2q < 2^128.
    let a = split::<false>(a.0, a.1, k.mask);
    let b = split::<false>(b.0, b.1, k.mask);
    let s0 = _mm512_add_epi64(a[0], b[0]);
    let s1 = _mm512_add_epi64(_mm512_add_epi64(a[1], b[1]), _mm512_srli_epi64::<52>(s0));
    let s2 = _mm512_add_epi64(_mm512_add_epi64(a[2], b[2]), _mm512_srli_epi64::<52>(s1));
    sub_if_ge(
        _mm512_and_si512(s0, k.mask),
        _mm512_and_si512(s1, k.mask),
        s2,
        k,
    )
}

/// `a - b mod q` for `a, b < q` below `2^127`, as `a + q - b`.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn sub_vec(a: (__m512i, __m512i), b: (__m512i, __m512i), k: &Consts) -> (__m512i, __m512i) {
    // a + q - b lies in (0, 2q), so arithmetic shifts carry the signed digit sums exactly.
    let a = split::<false>(a.0, a.1, k.mask);
    let b = split::<false>(b.0, b.1, k.mask);
    let s0 = _mm512_sub_epi64(_mm512_add_epi64(a[0], k.q[0]), b[0]);
    let s1 = _mm512_add_epi64(
        _mm512_sub_epi64(_mm512_add_epi64(a[1], k.q[1]), b[1]),
        _mm512_srai_epi64::<52>(s0),
    );
    let s2 = _mm512_add_epi64(
        _mm512_sub_epi64(_mm512_add_epi64(a[2], k.q[2]), b[2]),
        _mm512_srai_epi64::<52>(s1),
    );
    sub_if_ge(
        _mm512_and_si512(s0, k.mask),
        _mm512_and_si512(s1, k.mask),
        s2,
        k,
    )
}

/// Views a slice of `Residue<L>` as `Residue<2>`, for `L == 2`.
fn two<const L: usize>(x: &[Residue<L>]) -> Option<&[Residue<2>]> {
    // SAFETY: when `L == 2` the two slice types are the same.
    (L == 2).then(|| unsafe { core::slice::from_raw_parts(x.as_ptr().cast(), x.len()) })
}

/// Views a mutable slice of `Residue<L>` as `Residue<2>`, for `L == 2`.
fn two_mut<const L: usize>(x: &mut [Residue<L>]) -> Option<&mut [Residue<2>]> {
    // SAFETY: when `L == 2` the two slice types are the same, and the unique borrow carries over.
    (L == 2).then(|| unsafe { core::slice::from_raw_parts_mut(x.as_mut_ptr().cast(), x.len()) })
}

/// The exact sum `sum_i a_i b_i` of the stored forms, or `None` unless `L == 2`.
pub(super) fn dot<const L: usize>(a: &[Residue<L>], b: &[Residue<L>]) -> Option<Wide<L>> {
    let (a, b) = (two(a)?, two(b)?);
    // SAFETY: this module is compiled only for targets with AVX-512F and AVX-512 IFMA.
    let w = unsafe { dot2(a, b) };
    let mut sum = Wide::ZERO;
    sum.low.0.copy_from_slice(&w.low.0);
    sum.high.0.copy_from_slice(&w.high.0);
    sum.top = w.top;
    Some(sum)
}

#[target_feature(enable = "avx512f,avx512ifma")]
fn dot2(a: &[Residue<2>], b: &[Residue<2>]) -> Wide<2> {
    let mask = _mm512_set1_epi64(super::radix52::MASK as i64);
    let n = a.len().min(b.len()) / LANES * LANES;
    let mut cols = [0u128; 6];
    for (ca, cb) in limbs(&a[..n])
        .chunks(2 * LANES * FLUSH)
        .zip(limbs(&b[..n]).chunks(2 * LANES * FLUSH))
    {
        let mut t = [_mm512_setzero_si512(); 6];
        for (x, y) in ca
            .as_chunks::<{ 2 * LANES }>()
            .0
            .iter()
            .zip(cb.as_chunks::<{ 2 * LANES }>().0.iter())
        {
            let (xl, xh) = load(x);
            let (yl, yh) = load(y);
            let x = split::<false>(xl, xh, mask);
            let y = split::<false>(yl, yh, mask);
            for i in 0..3 {
                for j in 0..3 {
                    t[i + j] = _mm512_madd52lo_epu64(t[i + j], x[i], y[j]);
                    t[i + j + 1] = _mm512_madd52hi_epu64(t[i + j + 1], x[i], y[j]);
                }
            }
        }
        for (col, v) in cols.iter_mut().zip(t) {
            let mut lanes = [0u64; LANES];
            // SAFETY: the array holds one vector, and the store accepts any alignment.
            unsafe { _mm512_storeu_si512(lanes.as_mut_ptr().cast(), v) };
            *col += lanes.iter().map(|&x| x as u128).sum::<u128>();
        }
    }
    // Recombine the columns, weight 2^(52 k), into five limbs.
    let mut w = [0u64; 5];
    for (k, &col) in cols.iter().enumerate() {
        add_shifted(&mut w, col, 52 * k as u32);
    }
    let mut sum = Wide {
        low: Uint([w[0], w[1]]),
        high: Uint([w[2], w[3]]),
        top: w[4],
    };
    for (x, y) in a[n..].iter().zip(&b[n..]) {
        sum.add_product(&x.0, &y.0);
    }
    sum
}

/// `w += x 2^shift`, wrapping at five limbs.
fn add_shifted(w: &mut [u64; 5], x: u128, shift: u32) {
    let limb = (shift / 64) as usize;
    let bits = shift % 64;
    let parts = [
        (x as u64) << bits,
        if bits == 0 {
            (x >> 64) as u64
        } else {
            (x >> (64 - bits)) as u64
        },
        if bits == 0 {
            0
        } else {
            (x >> (128 - bits)) as u64
        },
    ];
    let mut carry = 0;
    for (i, slot) in w.iter_mut().enumerate().skip(limb) {
        let p = parts.get(i - limb).copied().unwrap_or(0);
        (*slot, carry) = adc(*slot, p, carry);
    }
}

/// `a_i <- a_i b_i`, or `false` unless `L == 2`.
pub(super) fn mul_slices<const L: usize>(
    ring: &MontModulus<L>,
    a: &mut [Residue<L>],
    b: &[Residue<L>],
) -> bool {
    let (Some(a), Some(b), Some(ring)) = (two_mut(a), two(b), two_ring(ring)) else {
        return false;
    };
    // SAFETY: this module is compiled only for targets with AVX-512F and AVX-512 IFMA.
    unsafe { mul_slices2(ring, a, b) };
    true
}

#[target_feature(enable = "avx512f,avx512ifma")]
fn mul_slices2(ring: &MontModulus<2>, a: &mut [Residue<2>], b: &[Residue<2>]) {
    let k = Consts::new(ring);
    let n = a.len() / LANES * LANES;
    for (x, y) in limbs_mut(&mut a[..n])
        .as_chunks_mut::<{ 2 * LANES }>()
        .0
        .iter_mut()
        .zip(limbs(&b[..n]).as_chunks::<{ 2 * LANES }>().0.iter())
    {
        let (lo, hi) = mul_vec(load(x), load(y), &k);
        store(x, lo, hi);
    }
    for (x, &y) in a[n..].iter_mut().zip(&b[n..]) {
        *x = ring.mul(*x, y);
    }
}

/// `acc_i <- acc_i + s x_i`, or `false` unless `L == 2`.
pub(super) fn scale_add<const L: usize>(
    ring: &MontModulus<L>,
    acc: &mut [Residue<L>],
    x: &[Residue<L>],
    s: Residue<L>,
) -> bool {
    let (Some(acc), Some(x), Some(ring)) = (two_mut(acc), two(x), two_ring(ring)) else {
        return false;
    };
    let s = Residue(Uint([s.0.0[0], s.0.0[1]]));
    // SAFETY: this module is compiled only for targets with AVX-512F and AVX-512 IFMA.
    unsafe { scale_add2(ring, acc, x, s) };
    true
}

#[target_feature(enable = "avx512f,avx512ifma")]
fn scale_add2(ring: &MontModulus<2>, acc: &mut [Residue<2>], x: &[Residue<2>], s: Residue<2>) {
    let k = Consts::new(ring);
    let sd = split::<true>(
        _mm512_set1_epi64(s.0.0[0] as i64),
        _mm512_set1_epi64(s.0.0[1] as i64),
        k.mask,
    );
    let n = acc.len() / LANES * LANES;
    for (a, y) in limbs_mut(&mut acc[..n])
        .as_chunks_mut::<{ 2 * LANES }>()
        .0
        .iter_mut()
        .zip(limbs(&x[..n]).as_chunks::<{ 2 * LANES }>().0.iter())
    {
        let (yl, yh) = load(y);
        let p = mont_mul(&split::<false>(yl, yh, k.mask), &sd, &k);
        let (lo, hi) = add_vec(load(a), p, &k);
        store(a, lo, hi);
    }
    for (a, &y) in acc[n..].iter_mut().zip(&x[n..]) {
        *a = ring.add(*a, ring.mul(s, y));
    }
}

/// One doubling of an equality table: `hi_i <- lo_i x`, then `lo_i <- lo_i - hi_i`, or `false` unless `L == 2`.
pub(super) fn eq_step<const L: usize>(
    ring: &MontModulus<L>,
    lo: &mut [Residue<L>],
    hi: &mut [Residue<L>],
    x: Residue<L>,
) -> bool {
    let (Some(lo), Some(hi), Some(ring)) = (two_mut(lo), two_mut(hi), two_ring(ring)) else {
        return false;
    };
    let x = Residue(Uint([x.0.0[0], x.0.0[1]]));
    // SAFETY: this module is compiled only for targets with AVX-512F and AVX-512 IFMA.
    unsafe { eq_step2(ring, lo, hi, x) };
    true
}

#[target_feature(enable = "avx512f,avx512ifma")]
fn eq_step2(ring: &MontModulus<2>, lo: &mut [Residue<2>], hi: &mut [Residue<2>], x: Residue<2>) {
    let k = Consts::new(ring);
    let xd = split::<true>(
        _mm512_set1_epi64(x.0.0[0] as i64),
        _mm512_set1_epi64(x.0.0[1] as i64),
        k.mask,
    );
    let n = lo.len().min(hi.len()) / LANES * LANES;
    for (l, h) in limbs_mut(&mut lo[..n])
        .as_chunks_mut::<{ 2 * LANES }>()
        .0
        .iter_mut()
        .zip(
            limbs_mut(&mut hi[..n])
                .as_chunks_mut::<{ 2 * LANES }>()
                .0
                .iter_mut(),
        )
    {
        let a = load(l);
        let p = mont_mul(&split::<false>(a.0, a.1, k.mask), &xd, &k);
        let (dl, dh) = sub_vec(a, p, &k);
        store(h, p.0, p.1);
        store(l, dl, dh);
    }
    for (l, h) in lo[n..].iter_mut().zip(&mut hi[n..]) {
        *h = ring.mul(*l, x);
        *l = ring.sub(*l, *h);
    }
}

/// The ring as its two-limb self, for `L == 2`.
fn two_ring<const L: usize>(ring: &MontModulus<L>) -> Option<&MontModulus<2>> {
    // SAFETY: when `L == 2` the two types are the same.
    (L == 2).then(|| unsafe { &*core::ptr::from_ref(ring).cast::<MontModulus<2>>() })
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use proptest::prelude::*;

    use super::*;

    fn ring(m: [u64; 2]) -> MontModulus<2> {
        let mut m = Uint(m);
        m.0[1] &= u64::MAX >> 1;
        m.0[0] |= 1;
        m.0[1] |= 1 << 40;
        MontModulus::new(m).expect("odd")
    }

    proptest! {
        #[test]
        fn vector_kernels_match_the_scalar_model(m: [u64; 2], seed: u64, len in 0usize..40) {
            let ring = ring(m);
            let mut s = seed;
            let mut draw = || {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ring.reduce_limbs(&[s, s.rotate_left(17), s.rotate_left(33)])
            };
            let a: Vec<Residue<2>> = (0..len).map(|_| draw()).collect();
            let b: Vec<Residue<2>> = (0..len).map(|_| draw()).collect();
            let c = draw();

            let mut got = a.clone();
            prop_assert!(mul_slices(&ring, &mut got, &b));
            for ((&g, &x), &y) in got.iter().zip(&a).zip(&b) {
                prop_assert_eq!(g.0.0, super::super::radix52::model::mul(x.0.0, y.0.0, &ring));
            }
            let mut got = a.clone();
            prop_assert!(scale_add(&ring, &mut got, &b, c));
            for ((&g, &x), &y) in got.iter().zip(&a).zip(&b) {
                prop_assert_eq!(g.0.0, super::super::radix52::model::scale_add(x.0.0, y.0.0, c.0.0, &ring));
            }
            let mut lo = a.clone();
            let mut hi = b.clone();
            prop_assert!(eq_step(&ring, &mut lo, &mut hi, c));
            for ((&l, &h), &x) in lo.iter().zip(&hi).zip(&a) {
                let (wl, wh) = super::super::radix52::model::eq_step(x.0.0, c.0.0, &ring);
                prop_assert_eq!((l.0.0, h.0.0), (wl, wh));
            }
            let got = dot(&a, &b).expect("two limbs");
            let want = super::super::radix52::model::dot(&a, &b);
            prop_assert_eq!((got.low, got.high, got.top), (want.low, want.high, want.top));
        }
    }

    #[test]
    fn dot_flushes_full_columns_exactly() {
        // The largest stored value under the largest modulus loads every lane past several flushes.
        let ring = MontModulus::new(Uint([u64::MAX, u64::MAX >> 1])).expect("odd");
        let top = Residue(Uint([u64::MAX - 1, u64::MAX >> 1]));
        for len in [LANES * FLUSH - 1, LANES * FLUSH, 3 * LANES * FLUSH + 5] {
            let a = vec![top; len];
            let got = dot(&a, &a).expect("two limbs");
            let want = super::super::radix52::model::dot(&a, &a);
            assert_eq!(
                (got.low, got.high, got.top),
                (want.low, want.high, want.top)
            );
            let reduced = ring.dot(&a, &a);
            let x = ring.to_uint(top);
            let want = ring.mul(ring.from_uint(&x), ring.from_uint(&x));
            assert_eq!(reduced, ring.mul(want, ring.from_u64(len as u64)));
        }
    }
}
