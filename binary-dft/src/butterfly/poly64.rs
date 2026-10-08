//! The butterfly of `GF(2^64)` in the polynomial basis, over wide carryless-multiply registers.
//!
//! The scalar field packs nothing, so each product would be one 64-bit carryless multiply.
//!
//! A wide register instead multiplies a whole lane group by the same twiddle in two instructions.

use p3_binary_field::Poly64;

use super::packed_butterfly;
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
mod neon;

/// The butterfly over two runs of `GF(2^64)` elements.
///
/// Whole registers take the wide kernel where the target has one.
///
/// The tail, and every run on other targets, takes the scalar kernel.
///
/// # Panics
///
/// Panics if the two runs have different lengths.
#[inline]
pub(super) fn butterfly<const INVERSE: bool>(lo: &mut [Poly64], hi: &mut [Poly64], t: Poly64) {
    // Both kernels stop at the shorter run, so a mismatch would silently drop work.
    assert_eq!(lo.len(), hi.len(), "butterfly lengths differ");

    // A zero twiddle is a bare addition, which the scalar kernel already shortcuts.
    let covered = if t.to_bits() == 0 {
        0
    } else {
        register_prefix::<INVERSE>(lo, hi, t)
    };
    packed_butterfly::<Poly64, INVERSE>(&mut lo[covered..], &mut hi[covered..], t);
}

/// The widest carryless-multiply register the build enables.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
type Wide = core::arch::x86_64::__m512i;

/// The widest carryless-multiply register the build enables.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "vpclmulqdq",
    not(target_feature = "avx512f")
))]
type Wide = core::arch::x86_64::__m256i;

/// Apply the butterfly to the whole registers of two runs.
///
/// # Returns
///
/// The number of elements covered, a multiple of the lane count.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "vpclmulqdq"
))]
#[inline]
fn register_prefix<const INVERSE: bool>(lo: &mut [Poly64], hi: &mut [Poly64], t: Poly64) -> usize {
    const LANES: usize = <Wide as ClmulRegister>::LANES;
    let t = Wide::splat(t.to_bits());

    // Whole registers only, so the remainder is the caller's to finish.
    let (lo, _) = lo.as_chunks_mut::<LANES>();
    let (hi, _) = hi.as_chunks_mut::<LANES>();
    let covered = lo.len() * LANES;

    for (lo, hi) in lo.iter_mut().zip(hi) {
        // SAFETY: each chunk is one register of transparent 64-bit words.
        unsafe {
            let u = Wide::load(lo.as_ptr().cast());
            let v = Wide::load(hi.as_ptr().cast());
            if INVERSE {
                // Recover the upper value, then remove its scaled copy from the lower one.
                let v = v.xor(u);
                Wide::store(hi.as_mut_ptr().cast(), v);
                Wide::store(lo.as_mut_ptr().cast(), u.xor(mul(v, t)));
            } else {
                // Scale the upper value into the lower one, then add the result to the upper one.
                let u = u.xor(mul(v, t));
                Wide::store(lo.as_mut_ptr().cast(), u);
                Wide::store(hi.as_mut_ptr().cast(), v.xor(u));
            }
        }
    }
    covered
}

/// The NEON kernel covers lane pairs, leaving an odd scalar tail.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
fn register_prefix<const INVERSE: bool>(lo: &mut [Poly64], hi: &mut [Poly64], t: Poly64) -> usize {
    neon::butterfly::<INVERSE>(lo, hi, t)
}

/// Without a wide carryless multiply there is no register prefix.
#[cfg(not(any(
    all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "vpclmulqdq"
    ),
    all(target_arch = "aarch64", target_feature = "aes")
)))]
#[inline]
const fn register_prefix<const INVERSE: bool>(
    _lo: &mut [Poly64],
    _hi: &mut [Poly64],
    _t: Poly64,
) -> usize {
    0
}

/// Multiply a register of elements by one broadcast twiddle, modulo `x^64 + x^4 + x^3 + x + 1`.
///
/// # Algorithm
///
/// The register kernel returns each 128-bit product as two words in element order.
///
/// The field polynomial gives `x^64 = r(x)` with `r = 1 + x + x^3 + x^4 = (1 + x)(1 + x^3)`.
///
/// So the high word folds back through the two factors, each a shift:
///
/// - The product is `lo + hi * x^64`, and `hi` has degree at most 62.
/// - `a = hi * (1 + x)` then has degree at most 63, so it fits one word.
/// - `a * (1 + x^3)` is `a + (a << 3)` in the word, plus `spill = a >> 61` times `x^64`.
/// - The spill has degree at most 2, so it folds back as `spill * (1 + x) * (1 + x^3)` with no bit lost.
/// - Writing `c = a + spill * (1 + x)`, the result is `lo + c + (c << 3)`.
///
/// Shifts stand in for more carryless multiplies, which are the scarcer unit.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "vpclmulqdq"
))]
#[inline(always)]
fn mul<L: ClmulRegister>(v: L, t: L) -> L {
    let (lo, hi) = v.products(t);

    // Multiplying by 1 + x: the product of two degree-63 polynomials leaves bit 63 of hi clear.
    let a = hi.xor(hi.double());

    // The top three bits of a, which the shift by three pushes out of the word.
    let spill = a.shr61();

    // Fold them back scaled by 1 + x, then apply the shared factor 1 + x^3 to both.
    let c = a.xor3(spill, spill.double());
    lo.xor3(c, c.shl3())
}

/// A register of 64-bit words with a carryless multiply.
///
/// Every operation acts on each word alone, except the product.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "vpclmulqdq"
))]
trait ClmulRegister: Copy {
    /// Words one register holds.
    const LANES: usize;

    /// Every word set to the same value.
    fn splat(word: u64) -> Self;

    /// Read one register, at any alignment.
    ///
    /// # Safety
    ///
    /// The address must be readable for one register.
    unsafe fn load(from: *const u64) -> Self;

    /// Write one register, at any alignment.
    ///
    /// # Safety
    ///
    /// The address must be writable for one register.
    unsafe fn store(to: *mut u64, value: Self);

    /// Bitwise exclusive or.
    fn xor(self, other: Self) -> Self;

    /// Three-way bitwise exclusive or.
    fn xor3(self, b: Self, c: Self) -> Self;

    /// Each word shifted left by one, as an addition.
    fn double(self) -> Self;

    /// Each word shifted right by 61.
    fn shr61(self) -> Self;

    /// Each word shifted left by 3.
    fn shl3(self) -> Self;

    /// The carryless products of every word with the broadcast twiddle.
    ///
    /// Returns the low halves and the high halves, each in element order.
    fn products(self, t: Self) -> (Self, Self);
}

/// The 512-bit register, over `AVX-512F` and `VPCLMULQDQ`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
mod avx512 {
    use core::arch::x86_64::{
        __m512i, _mm512_add_epi64, _mm512_clmulepi64_epi128, _mm512_loadu_si512, _mm512_set1_epi64,
        _mm512_slli_epi64, _mm512_srli_epi64, _mm512_storeu_si512, _mm512_ternarylogic_epi64,
        _mm512_unpackhi_epi64, _mm512_unpacklo_epi64, _mm512_xor_si512,
    };

    use super::ClmulRegister;

    // SAFETY: every intrinsic below needs only the instruction sets this module is gated on.
    impl ClmulRegister for __m512i {
        const LANES: usize = 8;

        #[inline(always)]
        fn splat(word: u64) -> Self {
            unsafe { _mm512_set1_epi64(word as i64) }
        }

        #[inline(always)]
        unsafe fn load(from: *const u64) -> Self {
            // SAFETY: the readability of the address is the caller's obligation.
            unsafe { _mm512_loadu_si512(from.cast()) }
        }

        #[inline(always)]
        unsafe fn store(to: *mut u64, value: Self) {
            // SAFETY: the writability of the address is the caller's obligation.
            unsafe { _mm512_storeu_si512(to.cast(), value) }
        }

        #[inline(always)]
        fn xor(self, other: Self) -> Self {
            unsafe { _mm512_xor_si512(self, other) }
        }

        #[inline(always)]
        fn xor3(self, b: Self, c: Self) -> Self {
            // 0x96 is the truth table of a three-way exclusive or.
            unsafe { _mm512_ternarylogic_epi64::<0x96>(self, b, c) }
        }

        #[inline(always)]
        fn double(self) -> Self {
            unsafe { _mm512_add_epi64(self, self) }
        }

        #[inline(always)]
        fn shr61(self) -> Self {
            unsafe { _mm512_srli_epi64::<61>(self) }
        }

        #[inline(always)]
        fn shl3(self) -> Self {
            unsafe { _mm512_slli_epi64::<3>(self) }
        }

        #[inline(always)]
        fn products(self, t: Self) -> (Self, Self) {
            unsafe {
                // The multiply reads one word of each 128-bit lane, so even and odd words take one each.
                //
                //     even = [ p_0 | p_2 | p_4 | p_6 ]      each p_i a 128-bit product (lo_i, hi_i)
                //     odd  = [ p_1 | p_3 | p_5 | p_7 ]
                let even = _mm512_clmulepi64_epi128::<0x00>(self, t);
                let odd = _mm512_clmulepi64_epi128::<0x01>(self, t);

                // Two unpacks sort the halves back into element order.
                //
                //     lo = [ lo_0 lo_1 | lo_2 lo_3 | ... ]
                //     hi = [ hi_0 hi_1 | hi_2 hi_3 | ... ]
                (
                    _mm512_unpacklo_epi64(even, odd),
                    _mm512_unpackhi_epi64(even, odd),
                )
            }
        }
    }
}

/// The 256-bit register, over `AVX2` and `VPCLMULQDQ`, for cores without `AVX-512`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "vpclmulqdq",
    not(target_feature = "avx512f")
))]
mod avx2 {
    use core::arch::x86_64::{
        __m256i, _mm256_add_epi64, _mm256_clmulepi64_epi128, _mm256_loadu_si256,
        _mm256_set1_epi64x, _mm256_slli_epi64, _mm256_srli_epi64, _mm256_storeu_si256,
        _mm256_unpackhi_epi64, _mm256_unpacklo_epi64, _mm256_xor_si256,
    };

    use super::ClmulRegister;

    // SAFETY: every intrinsic below needs only the instruction sets this module is gated on.
    impl ClmulRegister for __m256i {
        const LANES: usize = 4;

        #[inline(always)]
        fn splat(word: u64) -> Self {
            unsafe { _mm256_set1_epi64x(word as i64) }
        }

        #[inline(always)]
        unsafe fn load(from: *const u64) -> Self {
            // SAFETY: the readability of the address is the caller's obligation.
            unsafe { _mm256_loadu_si256(from.cast()) }
        }

        #[inline(always)]
        unsafe fn store(to: *mut u64, value: Self) {
            // SAFETY: the writability of the address is the caller's obligation.
            unsafe { _mm256_storeu_si256(to.cast(), value) }
        }

        #[inline(always)]
        fn xor(self, other: Self) -> Self {
            unsafe { _mm256_xor_si256(self, other) }
        }

        #[inline(always)]
        fn xor3(self, b: Self, c: Self) -> Self {
            // No ternary logic below AVX-512, so two plain exclusive ors.
            unsafe { _mm256_xor_si256(_mm256_xor_si256(self, b), c) }
        }

        #[inline(always)]
        fn double(self) -> Self {
            unsafe { _mm256_add_epi64(self, self) }
        }

        #[inline(always)]
        fn shr61(self) -> Self {
            unsafe { _mm256_srli_epi64::<61>(self) }
        }

        #[inline(always)]
        fn shl3(self) -> Self {
            unsafe { _mm256_slli_epi64::<3>(self) }
        }

        #[inline(always)]
        fn products(self, t: Self) -> (Self, Self) {
            unsafe {
                // Same layout as the 512-bit register, over two 128-bit lanes.
                let even = _mm256_clmulepi64_epi128::<0x00>(self, t);
                let odd = _mm256_clmulepi64_epi128::<0x01>(self, t);
                (
                    _mm256_unpacklo_epi64(even, odd),
                    _mm256_unpackhi_epi64(even, odd),
                )
            }
        }
    }
}
