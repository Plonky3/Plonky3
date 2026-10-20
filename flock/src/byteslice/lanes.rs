//! The register of 64 bytes the sliced kernels run on, and the backend behind it.

/// The byte moves the sliced kernels are written against.
///
/// An index register picks bytes or quadwords by position; a matrix register holds one `8 x 8` block per quadword.
pub(crate) trait SliceRegister: Copy {
    /// The all-zero register.
    fn zero() -> Self;

    /// One quadword in every quadword.
    fn splat(qword: u64) -> Self;

    /// Read one register of bytes, at any alignment.
    ///
    /// # Safety
    ///
    /// The address must be readable for 64 bytes.
    unsafe fn load(from: *const u8) -> Self;

    /// Write one register of bytes, at any alignment.
    ///
    /// # Safety
    ///
    /// The address must be writable for 64 bytes.
    unsafe fn store(to: *mut u8, value: Self);

    /// Bitwise exclusive or.
    fn xor(self, other: Self) -> Self;

    /// Bitwise exclusive or of three registers.
    fn xor3(self, a: Self, b: Self) -> Self;

    /// Byte `i` takes byte `index[i] mod 64` of this register.
    fn permute(self, index: Self) -> Self;

    /// Byte `i` takes byte `index[i] mod 64` of this register, or of `high` when bit 6 of the index is set.
    fn permute2(self, index: Self, high: Self) -> Self;

    /// Where the mask is set, byte `i` takes byte `index[i] mod 64` of `source`; elsewhere it keeps its own.
    fn permute_merge(self, mask: u64, index: Self, source: Self) -> Self;

    /// Quadword `i` takes quadword `index[i] mod 8` of this register, or of `high` when bit 3 of the index is set.
    fn permute2_qwords(self, index: Self, high: Self) -> Self;

    /// Where the mask is set, quadword `i` takes quadword `index[i] mod 8` of `source`; elsewhere it keeps its own.
    fn permute_merge_qwords(self, mask: u8, index: Self, source: Self) -> Self;

    /// Every byte through the block of its quadword, output bit `k` reading block byte `7 - k`.
    fn affine(self, matrices: Self) -> Self;
}

/// The 512-bit register, over AVX-512 and GFNI.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "gfni",
    target_feature = "vpclmulqdq"
))]
mod x86_64 {
    use core::arch::x86_64::{
        __m512i, _mm512_gf2p8affine_epi64_epi8, _mm512_loadu_si512, _mm512_mask_permutexvar_epi8,
        _mm512_mask_permutexvar_epi64, _mm512_permutex2var_epi8, _mm512_permutex2var_epi64,
        _mm512_permutexvar_epi8, _mm512_set1_epi64, _mm512_setzero_si512, _mm512_storeu_si512,
        _mm512_ternarylogic_epi64, _mm512_xor_si512,
    };

    use super::SliceRegister;

    // SAFETY: this module compiles only where the crate enables the needed target features.
    impl SliceRegister for __m512i {
        #[inline(always)]
        fn zero() -> Self {
            unsafe { _mm512_setzero_si512() }
        }

        #[inline(always)]
        fn splat(qword: u64) -> Self {
            unsafe { _mm512_set1_epi64(qword as i64) }
        }

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
        fn xor3(self, a: Self, b: Self) -> Self {
            unsafe { _mm512_ternarylogic_epi64::<0x96>(self, a, b) }
        }

        #[inline(always)]
        fn permute(self, index: Self) -> Self {
            unsafe { _mm512_permutexvar_epi8(index, self) }
        }

        #[inline(always)]
        fn permute2(self, index: Self, high: Self) -> Self {
            unsafe { _mm512_permutex2var_epi8(self, index, high) }
        }

        #[inline(always)]
        fn permute_merge(self, mask: u64, index: Self, source: Self) -> Self {
            unsafe { _mm512_mask_permutexvar_epi8(self, mask, index, source) }
        }

        #[inline(always)]
        fn permute2_qwords(self, index: Self, high: Self) -> Self {
            unsafe { _mm512_permutex2var_epi64(self, index, high) }
        }

        #[inline(always)]
        fn permute_merge_qwords(self, mask: u8, index: Self, source: Self) -> Self {
            unsafe { _mm512_mask_permutexvar_epi64(self, mask, index, source) }
        }

        #[inline(always)]
        fn affine(self, matrices: Self) -> Self {
            unsafe { _mm512_gf2p8affine_epi64_epi8::<0>(self, matrices) }
        }
    }
}
