//! Discrete Fourier Transform, in-place, decimation-in-frequency
//!
//! Straightforward recursive algorithm, "unrolled" up to size 256.
//!
//! Inspired by Bernstein's djbfft: https://cr.yp.to/djbfft.html

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use itertools::{iterate, izip, Itertools};
use p3_field::{extension::ComplexExtendable, AbstractField, PackedField};

use crate::{to_mersenne31_array, Mersenne31};

// the twiddle for the inner most layer is 2^15 (32768).
pub(crate) const TWIDDLES_4: [Mersenne31; 2] = to_mersenne31_array([590768354, 1168891274]);
pub(crate) const TWIDDLES_8: [Mersenne31; 4] =
    to_mersenne31_array([1179735656, 1415090252, 34602070, 906276279]);
pub(crate) const TWIDDLES_16: [Mersenne31; 8] = to_mersenne31_array([
    1567857810, 505542828, 194696271, 1133522282, 280947147, 1580223790, 2121318970, 1690787918,
]);
pub(crate) const TWIDDLES_32: [Mersenne31; 16] = to_mersenne31_array([
    1774253895, 262191051, 212443077, 883753057, 404685994, 68458636, 228509164, 134155457,
    1038945916, 14530030, 9803698, 2140339328, 1796741361, 206059115, 1739004854, 838195206,
]);

impl Mersenne31 {
    /// NAME: TODO
    /// COMMENTS: TODO
    pub fn roots_of_unity_table(log_n: usize) -> Vec<Vec<Mersenne31>> {
        assert!(log_n > 6);
        let g = Mersenne31::circle_two_adic_generator(log_n);
        let shft = Mersenne31::circle_two_adic_generator(log_n + 1);

        let init_coset = iterate(shft, move |&p| p * g).take(1 << (log_n - 1));
        let (x_vals, y_vals): (Vec<_>, Vec<_>) = init_coset.map(|x| (x.real(), x.imag())).unzip();
        let mut twiddles = vec![y_vals];
        twiddles.push(x_vals.into_iter().step_by(2).collect_vec());

        for i in 0..(log_n - 2) {
            let prev = twiddles.last().unwrap();
            assert_eq!(prev.len(), 1 << (log_n - 2 - i));
            let next = prev
                .iter()
                .step_by(2)
                .map(|x| x.square().double() - Mersenne31::one())
                .collect_vec();
            twiddles.push(next);
        }
        twiddles
    }
}

impl Mersenne31 {
    #[inline(always)]
    fn forward_butterfly<PF: PackedField<Scalar = Mersenne31>>(x: PF, y: PF, w: Self) -> (PF, PF) {
        let t = y * PF::from_f(w);
        (x + t, x - t)
    }

    #[inline]
    fn forward_pass<PF: PackedField<Scalar = Mersenne31>>(a: &mut [PF], roots: &[Self]) {
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

        // Safe because a.len() == 8
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::forward_2(a0);
        Self::forward_2(a1);

        Self::forward_pass(a, &TWIDDLES_4);
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
    fn forward_64<PF: PackedField<Scalar = Mersenne31>>(a: &mut [PF], root_table: &[Vec<Self>]) {
        assert_eq!(a.len(), 64);

        // Safe because a.len() == 64
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::forward_32(a0);
        Self::forward_32(a1);

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
