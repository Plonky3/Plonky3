//! Discrete Fourier Transform, in-place, decimation-in-frequency
//!
//! Straightforward recursive algorithm, "unrolled" up to size 256.
//!
//! Inspired by Bernstein's djbfft: https://cr.yp.to/djbfft.html

extern crate alloc;
use alloc::vec::Vec;
use alloc::vec;

use itertools::{iterate, izip, Itertools};
use p3_field::{extension::ComplexExtendable, AbstractField, PackedField, TwoAdicField};
use p3_util::log2_strict_usize;

use crate::{to_mersenne31_array, Mersenne31};

// the twiddle for the inner most layer is 2^15 (32768).
pub(crate) const TWIDDLES_4: [Mersenne31; 2] = to_mersenne31_array([590768354, 1168891274]);
pub(crate) const TWIDDLES_8: [Mersenne31; 4] = to_mersenne31_array([1179735656, 1415090252, 34602070, 906276279]);
pub(crate) const TWIDDLES_16: [Mersenne31; 8] = to_mersenne31_array([1567857810, 505542828, 194696271, 1133522282, 280947147, 1580223790, 2121318970, 1690787918]);
pub(crate) const TWIDDLES_32: [Mersenne31; 16] = to_mersenne31_array([1774253895, 262191051, 212443077, 883753057, 404685994, 68458636, 228509164, 134155457, 1038945916, 14530030, 9803698, 2140339328, 1796741361, 206059115, 1739004854, 838195206]);

impl Mersenne31 {
    /// NAME: TODO
    /// COMMENTS: TODO
    pub fn roots_of_unity_table(log_n: usize) -> Vec<Vec<Mersenne31>> {
        assert!(log_n > 6);
        let g = Mersenne31::circle_two_adic_generator(log_n);
        let shft = Mersenne31::circle_two_adic_generator(log_n + 1);

        let init_coset = iterate(shft, move |&p| p*g).take(1 << (log_n - 1));
        let (x_vals, y_vals): (Vec<_>, Vec<_>) = init_coset.map(|x| (x.imag(), x.real())).unzip();
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
    fn forward_butterfly<PF: PackedField<Scalar = Mersenne31>>(
        x: PF,
        y: PF,
        w: Self,
    ) -> (PF, PF) {
        let t = x - y;
        (x + y, t * PF::from_f(w))
    }

    #[inline]
    fn forward_pass<PF: PackedField<Scalar = Mersenne31>>(a: &mut [PF], roots: &[Self]) {
        let half_n = a.len() / 2;
        assert_eq!(roots.len(), half_n - 1);

        // Safe because 0 <= half_n < a.len()
        let (top, tail) = unsafe { a.split_at_mut_unchecked(half_n) };

        let s = top[0] + tail[0];
        let t = top[0] - tail[0];
        top[0] = s;
        tail[0] = t;

        izip!(&mut top[1..], &mut tail[1..], roots).for_each(|(hi, lo, &root)| {
            (*hi, *lo) = Self::forward_butterfly(*hi, *lo, root);
        });
    }

    #[inline(always)]
    fn forward_2<PF: PackedField<Scalar = Mersenne31>>(a: &mut [PF]) {
        assert_eq!(a.len(), 2);

        let s = a[0] + a[1];
        let t = a[0] - a[1];
        a[0] = s;
        a[1] = t;
    }

    #[inline(always)]
    fn forward_4<PF: PackedField<Scalar = Mersenne31>>(a: &mut [PF]) {
        assert_eq!(a.len(), 4);

        // Expanding the calculation of t3 saves one instruction
        let t1 = a[1] - a[3];
        let t3 = t1; // TODO: Determine the small twiddles
        let t5 = a[1] + a[3];
        let t4 = a[0] + a[2];
        let t2 = a[0] - a[2];

        // Return in bit-reversed order
        a[0] = t4 + t5;
        a[1] = t4 - t5;
        a[2] = t2 + t3;
        a[3] = t2 - t3;
    }

    #[inline(always)]
    fn forward_8<PF: PackedField<Scalar = Mersenne31>>(a: &mut [PF]) {
        assert_eq!(a.len(), 8);

        Self::forward_pass(a, &TWIDDLES_8);

        // Safe because a.len() == 8
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::forward_4(a0);
        Self::forward_4(a1);
    }

    #[inline(always)]
    fn forward_16<PF: PackedField<Scalar = Mersenne31>>(a: &mut [PF]) {
        assert_eq!(a.len(), 16);

        Self::forward_pass(a, &TWIDDLES_16);

        // Safe because a.len() == 16
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::forward_8(a0);
        Self::forward_8(a1);
    }

    #[inline(always)]
    fn forward_32<PF: PackedField<Scalar = Mersenne31>>(
        a: &mut [PF],
    ) {
        assert_eq!(a.len(), 32);

        Self::forward_pass(a, &TWIDDLES_32);

        // Safe because a.len() == 32
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::forward_16(a0);
        Self::forward_16(a1);
    }

    #[inline(always)]
    fn forward_64<PF: PackedField<Scalar = Mersenne31>>(
        a: &mut [PF],
        root_table: &[Vec<Self>],
    ) {
        assert_eq!(a.len(), 64);

        Self::forward_pass(a, &root_table[0]);

        // Safe because a.len() == 64
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::forward_32(a0);
        Self::forward_32(a1);
    }

    #[inline(always)]
    fn forward_128<PF: PackedField<Scalar = Mersenne31>>(
        a: &mut [PF],
        root_table: &[Vec<Self>],
    ) {
        assert_eq!(a.len(), 128);

        Self::forward_pass(a, &root_table[0]);

        // Safe because a.len() == 128
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::forward_64(a0, &root_table[1..]);
        Self::forward_64(a1, &root_table[1..]);
    }

    #[inline(always)]
    fn forward_256<PF: PackedField<Scalar = Mersenne31>>(
        a: &mut [PF],
        root_table: &[Vec<Self>],
    ) {
        assert_eq!(a.len(), 256);

        Self::forward_pass(a, &root_table[0]);

        // Safe because a.len() == 256
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::forward_128(a0, &root_table[1..]);
        Self::forward_128(a1, &root_table[1..]);
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

        assert_eq!(n, 1 << (twiddle_table.len() + 1));
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
                Self::forward_pass(a, &twiddle_table[0]);

                // Safe because a.len() > 64
                let (a0, a1) = unsafe { a.split_at_mut_unchecked(n / 2) };

                Self::forward_fft(a0, &twiddle_table[1..]);
                Self::forward_fft(a1, &twiddle_table[1..]);
            }
        }
    }
}