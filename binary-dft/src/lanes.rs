//! The register of bytes the subfield kernels run on, and the backend behind it.

/// The byte positions congruent to `offset` modulo `group`, as a per-byte mask.
pub(crate) const fn group_mask(group: usize, offset: usize) -> u64 {
    let mut mask = 0u64;
    let mut position = offset;
    while position < 64 {
        mask |= 1 << position;
        position += group;
    }
    mask
}

/// The byte-register operations the subfield kernels are written against.
///
/// A matrix argument is one `8 x 8` block over `GF(2)`.
/// Byte `7 - i` of its quadword is the row producing output bit `i`.
pub(crate) trait ByteLanes: Copy {
    /// Bytes one register holds.
    const BYTES: usize;

    /// Read one register of bytes, at any alignment.
    ///
    /// # Safety
    /// The address must be readable for that many bytes.
    unsafe fn load(from: *const u8) -> Self;

    /// Write one register of bytes, at any alignment.
    ///
    /// # Safety
    /// The address must be writable for that many bytes.
    unsafe fn store(to: *mut u8, value: Self);

    /// Bitwise exclusive or.
    fn xor(self, other: Self) -> Self;

    /// Rotate every group of `GROUP` bytes, so a position takes the byte `shift` above it.
    ///
    /// # Panics
    /// Panics on a group size this backend cannot rotate, or a shift not below it.
    fn rotate_group<const GROUP: usize>(self, shift: usize) -> Self;

    /// The image of every byte under one block.
    fn affine(self, matrix: u64) -> Self;

    /// The image of another register where the mask selects, this one everywhere else.
    fn affine_merge(self, source: Self, matrix: u64, mask: u64) -> Self;
}

/// Apply a butterfly to whole registers of two byte runs, returning the bytes it covered.
///
/// Forward sends `(a, b)` to `(a + map(b), a + map(b) + b)`, and the flag inverts that.
///
/// # Panics
/// Panics if the two runs have different lengths.
// A register width is an associated constant, which a const-generic chunk size cannot take.
#[allow(clippy::chunks_exact_to_as_chunks)]
#[inline]
pub(crate) fn butterfly_run<L: ByteLanes, const INVERSE: bool>(
    lo: &mut [u8],
    hi: &mut [u8],
    map: impl Fn(L) -> L,
) -> usize {
    // Invariant: the two runs pair byte for byte, or the shorter one would drop work.
    assert_eq!(lo.len(), hi.len(), "butterfly lengths differ");

    // Whole registers only, so the remainder is the caller's to finish.
    let lo = lo.chunks_exact_mut(L::BYTES);
    let hi = hi.chunks_exact_mut(L::BYTES);
    let covered = lo.len() * L::BYTES;

    for (lo, hi) in lo.zip(hi) {
        // SAFETY: each chunk is exactly one register, and both accesses are unaligned forms.
        unsafe {
            let a = L::load(lo.as_ptr());
            let b = L::load(hi.as_ptr());

            if INVERSE {
                // Recover the upper half, then take the scaled result out of the lower one.
                let b = b.xor(a);
                L::store(hi.as_mut_ptr(), b);
                L::store(lo.as_mut_ptr(), a.xor(map(b)));
            } else {
                // Scale the upper half into the lower one, then sum both into the upper one.
                let a = a.xor(map(b));
                L::store(lo.as_mut_ptr(), a);
                L::store(hi.as_mut_ptr(), b.xor(a));
            }
        }
    }
    covered
}

/// The 512-bit register, over `GFNI` and `AVX-512`.
///
/// Its byte map shares one matrix across a quadword, and a write mask picks the byte positions.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
mod x86_64 {
    use core::arch::x86_64::{
        __m512i, _mm512_gf2p8affine_epi64_epi8, _mm512_loadu_si512,
        _mm512_mask_gf2p8affine_epi64_epi8, _mm512_ror_epi32, _mm512_set1_epi64,
        _mm512_shuffle_epi8, _mm512_storeu_si512, _mm512_xor_si512,
    };

    use super::ByteLanes;

    /// The byte each position takes when the two bytes of every pair are exchanged.
    static SWAP_PAIRS: [u8; 64] = {
        let mut pattern = [0u8; 64];
        let mut i = 0;
        while i < 64 {
            pattern[i] = (i ^ 1) as u8;
            i += 1;
        }
        pattern
    };

    // SAFETY: this module compiles only where the crate enables the needed target features.
    impl ByteLanes for __m512i {
        const BYTES: usize = 64;

        #[inline(always)]
        unsafe fn load(from: *const u8) -> Self {
            // SAFETY: the readability of the address is the caller's obligation.
            unsafe { _mm512_loadu_si512(from.cast()) }
        }

        #[inline(always)]
        unsafe fn store(to: *mut u8, value: Self) {
            // SAFETY: the writability of the address is the caller's obligation.
            unsafe { _mm512_storeu_si512(to.cast(), value) }
        }

        #[inline(always)]
        fn xor(self, other: Self) -> Self {
            unsafe { _mm512_xor_si512(self, other) }
        }

        #[inline(always)]
        fn rotate_group<const GROUP: usize>(self, shift: usize) -> Self {
            unsafe {
                match (GROUP, shift) {
                    (_, 0) => self,
                    // A pair has no rotate instruction, so a byte shuffle stands in.
                    (2, 1) => {
                        let pattern = _mm512_loadu_si512(SWAP_PAIRS.as_ptr().cast());
                        _mm512_shuffle_epi8(self, pattern)
                    }
                    // A four-byte group is a doubleword, which rotates by whole bytes.
                    (4, 1) => _mm512_ror_epi32::<8>(self),
                    (4, 2) => _mm512_ror_epi32::<16>(self),
                    (4, 3) => _mm512_ror_epi32::<24>(self),
                    _ => panic!("unsupported byte-group rotation"),
                }
            }
        }

        #[inline(always)]
        fn affine(self, matrix: u64) -> Self {
            unsafe { _mm512_gf2p8affine_epi64_epi8::<0>(self, _mm512_set1_epi64(matrix as i64)) }
        }

        #[inline(always)]
        fn affine_merge(self, source: Self, matrix: u64, mask: u64) -> Self {
            // Merge masking overwrites only the selected bytes, at no extra instruction.
            unsafe {
                _mm512_mask_gf2p8affine_epi64_epi8::<0>(
                    self,
                    mask,
                    source,
                    _mm512_set1_epi64(matrix as i64),
                )
            }
        }
    }
}
