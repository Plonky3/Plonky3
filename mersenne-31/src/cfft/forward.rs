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
pub(crate) const TWIDDLES_4: [Mersenne31; 2] = to_mersenne31_array([590768354, 978592373]);
pub(crate) const TWIDDLES_8: [Mersenne31; 4] =
    to_mersenne31_array([1179735656, 1415090252, 1241207368, 2112881577]);
pub(crate) const TWIDDLES_16: [Mersenne31; 8] = to_mersenne31_array([
    1567857810, 194696271, 505542828, 1133522282, 456695729, 567259857, 26164677, 1866536500,
]);
pub(crate) const TWIDDLES_32: [Mersenne31; 16] = to_mersenne31_array([
    1774253895, 404685994, 212443077, 228509164, 262191051, 68458636, 883753057, 134155457,
    1309288441, 7144319, 1941424532, 2132953617, 408478793, 2137679949, 350742286, 1108537731,
]);
pub(crate) const TWIDDLES_64: [Mersenne31; 32] = to_mersenne31_array([
    736262640, 2098580229, 1334497267, 1093071961, 197700101, 849605071, 1977033713, 2066105389,
    1260750973, 1510607876, 1577470940, 445356670, 225856549, 1276547035, 125103457, 10472466,
    1553669210, 250538254, 2085743640, 1498890429, 1079800039, 1563928157, 2005527287, 1357626641,
    1362440376, 251924953, 236104903, 640817200, 1668363411, 1514613395, 1669530034, 1014093253,
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
    fn forward_2<PF: PackedField<Scalar = Mersenne31>>(a: &mut [PF]) {
        assert_eq!(a.len(), 2);

        let t = a[1].mul_2exp_u64(15);
        let x = a[0] + t;
        let y = a[0] - t;
        a[0] = x;
        a[1] = y;
    }

    #[inline(always)]
    fn forward_4<PF: PackedField<Scalar = Mersenne31>>(a: &mut [PF]) {
        assert_eq!(a.len(), 4);

        let a0 = a[0];
        let a1 = a[1].mul_2exp_u64(15);
        let a2 = a[2];
        let a3 = a[3].mul_2exp_u64(15);

        let a0_pos_01 = a0 + a1;
        let a1_neg_01 = a0 - a1;
        let a2_pos_23 = (a2 + a3) * TWIDDLES_4[0];
        let a3_neg_23 = (a2 - a3) * TWIDDLES_4[1];

        a[0] = a0_pos_01 + a2_pos_23;
        a[1] = a1_neg_01 + a3_neg_23;
        a[2] = a0_pos_01 - a2_pos_23;
        a[3] = a1_neg_01 - a3_neg_23;
    }

    #[inline(always)]
    fn forward_8<PF: PackedField<Scalar = Mersenne31>>(a: &mut [PF]) {
        assert_eq!(a.len(), 8);

        // Safe because a.len() == 8
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::forward_4(a0);
        Self::forward_4(a1);

        Self::forward_pass(a, &TWIDDLES_8);
    }

    #[inline(always)]
    fn forward_16<PF: PackedField<Scalar = Mersenne31>>(a: &mut [PF]) {
        assert_eq!(a.len(), 16);

        // Safe because a.len() == 16
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::forward_8(a0);
        Self::forward_8(a1);

        Self::forward_pass(a, &TWIDDLES_16);
    }

    #[inline(always)]
    fn forward_32<PF: PackedField<Scalar = Mersenne31>>(a: &mut [PF]) {
        assert_eq!(a.len(), 32);

        // Safe because a.len() == 32
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::forward_16(a0);
        Self::forward_16(a1);

        Self::forward_pass(a, &TWIDDLES_32);
    }

    #[inline(always)]
    fn forward_64<PF: PackedField<Scalar = Mersenne31>>(a: &mut [PF]) {
        assert_eq!(a.len(), 64);

        // Safe because a.len() == 64
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::forward_32(a0);
        Self::forward_32(a1);

        Self::forward_pass(a, &TWIDDLES_64);
    }

    #[inline(always)]
    fn forward_128<PF: PackedField<Scalar = Mersenne31>>(a: &mut [PF], root_table: &[Vec<Self>]) {
        assert_eq!(a.len(), 128);

        // Safe because a.len() == 128
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::forward_64(a0);
        Self::forward_64(a1);

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
            64 => Self::forward_64(a),
            32 => Self::forward_32(a),
            16 => Self::forward_16(a),
            8 => Self::forward_8(a),
            4 => Self::forward_4(a),
            2 => Self::forward_2(a),
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
