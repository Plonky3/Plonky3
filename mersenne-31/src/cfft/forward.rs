//! Discrete Fourier Transform, in-place, decimation-in-frequency
//!
//! Straightforward recursive algorithm, "unrolled" up to size 256.
//!
//! Inspired by Bernstein's djbfft: https://cr.yp.to/djbfft.html

extern crate alloc;
use alloc::vec::Vec;

use itertools::izip;
use p3_field::PackedField;

use crate::{to_mersenne31_array, Mersenne31};

// the twiddle for the inner most layer is 2^15 (32768).
pub(crate) const _TWIDDLES_4: [Mersenne31; 2] = to_mersenne31_array([590768354, 978592373]);
pub(crate) const _TWIDDLES_8: [Mersenne31; 4] =
    to_mersenne31_array([1179735656, 1415090252, 1241207368, 2112881577]);
pub(crate) const _TWIDDLES_16: [Mersenne31; 8] = to_mersenne31_array([
    1567857810, 194696271, 505542828, 1133522282, 456695729, 567259857, 26164677, 1866536500,
]);
pub(crate) const _TWIDDLES_32: [Mersenne31; 16] = to_mersenne31_array([
    1774253895, 404685994, 212443077, 228509164, 262191051, 68458636, 883753057, 134155457,
    1309288441, 7144319, 1941424532, 2132953617, 408478793, 2137679949, 350742286, 1108537731,
]);

impl Mersenne31 {
    #[inline(always)]
    fn forward_butterfly<PF: PackedField<Scalar = Mersenne31>>(x: PF, y: PF, w: Self) -> (PF, PF) {
        let t = y * PF::from_f(w);
        (x + t, x - t)
    }

    #[inline]
    pub(crate) fn forward_pass<PF: PackedField<Scalar = Mersenne31>>(a: &mut [PF], roots: &[Self]) {
        let half_n = a.len() / 2;
        assert_eq!(roots.len(), half_n);

        // Safe because 0 <= half_n < a.len()
        let (top, tail) = unsafe { a.split_at_mut_unchecked(half_n) };

        izip!(top, tail, roots).for_each(|(hi, lo, &root)| {
            (*hi, *lo) = Self::forward_butterfly(*hi, *lo, root);
        });
    }

    #[inline(always)]
    fn forward_2<PF: PackedField<Scalar = Mersenne31>>(a: &mut [PF], root_table: &[Vec<Self>]) {
        assert_eq!(a.len(), 2);
        assert_eq!(root_table[0][0].value, 32768);

        let t = a[1] * root_table[0][0];
        let x = a[0] + t;
        let y = a[0] - t;
        a[0] = x;
        a[1] = y;
    }

    #[inline(always)]
    fn forward_4<PF: PackedField<Scalar = Mersenne31>>(a: &mut [PF], root_table: &[Vec<Self>]) {
        assert_eq!(a.len(), 4);

        // Safe because a.len() == 8
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::forward_2(a0, &root_table[1..]);
        Self::forward_2(a1, &root_table[1..]);

        Self::forward_pass(a, &root_table[0]);
    }

    #[inline(always)]
    fn forward_8<PF: PackedField<Scalar = Mersenne31>>(a: &mut [PF], root_table: &[Vec<Self>]) {
        assert_eq!(a.len(), 8);

        // Safe because a.len() == 8
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::forward_4(a0, &root_table[1..]);
        Self::forward_4(a1, &root_table[1..]);

        Self::forward_pass(a, &root_table[0]);
    }

    #[inline(always)]
    fn forward_16<PF: PackedField<Scalar = Mersenne31>>(a: &mut [PF], root_table: &[Vec<Self>]) {
        assert_eq!(a.len(), 16);

        // Safe because a.len() == 16
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::forward_8(a0, &root_table[1..]);
        Self::forward_8(a1, &root_table[1..]);

        Self::forward_pass(a, &root_table[0]);
    }

    #[inline(always)]
    fn forward_32<PF: PackedField<Scalar = Mersenne31>>(a: &mut [PF], root_table: &[Vec<Self>]) {
        assert_eq!(a.len(), 32);

        // Safe because a.len() == 32
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::forward_16(a0, &root_table[1..]);
        Self::forward_16(a1, &root_table[1..]);

        Self::forward_pass(a, &root_table[0]);
    }

    #[inline(always)]
    fn forward_64<PF: PackedField<Scalar = Mersenne31>>(a: &mut [PF], root_table: &[Vec<Self>]) {
        assert_eq!(a.len(), 64);

        // Safe because a.len() == 64
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::forward_32(a0, &root_table[1..]);
        Self::forward_32(a1, &root_table[1..]);

        Self::forward_pass(a, &root_table[0]);
    }

    #[inline(always)]
    fn forward_128<PF: PackedField<Scalar = Mersenne31>>(a: &mut [PF], root_table: &[Vec<Self>]) {
        assert_eq!(a.len(), 128);

        // Safe because a.len() == 128
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::forward_64(a0, &root_table[1..]);
        Self::forward_64(a1, &root_table[1..]);

        Self::forward_pass(a, &root_table[0]);
    }

    #[inline(always)]
    fn forward_256<PF: PackedField<Scalar = Mersenne31>>(a: &mut [PF], root_table: &[Vec<Self>]) {
        assert_eq!(a.len(), 256);

        // Safe because a.len() == 256
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::forward_128(a0, &root_table[1..]);
        Self::forward_128(a1, &root_table[1..]);

        Self::forward_pass(a, &root_table[0]);
    }

    #[inline]
    pub fn forward_fft<PF: PackedField<Scalar = Mersenne31>>(
        a: &mut [PF],
        twiddle_table: &[Vec<Self>],
    ) {
        let n = a.len();
        if n == 1 {
            return;
        }

        assert_eq!(n, 1 << twiddle_table.len());
        match n {
            256 => Self::forward_256(a, twiddle_table),
            128 => Self::forward_128(a, twiddle_table),
            64 => Self::forward_64(a, twiddle_table),
            32 => Self::forward_32(a, twiddle_table),
            16 => Self::forward_16(a, twiddle_table),
            8 => Self::forward_8(a, twiddle_table),
            4 => Self::forward_4(a, twiddle_table),
            2 => Self::forward_2(a, twiddle_table),
            _ => {
                debug_assert!(n > 64);

                // Safe because a.len() > 64
                let (a0, a1) = unsafe { a.split_at_mut_unchecked(n / 2) };

                Self::forward_fft(a0, &twiddle_table[1..]);
                Self::forward_fft(a1, &twiddle_table[1..]);

                Self::forward_pass(a, &twiddle_table[0]);
            }
        }
    }
}
