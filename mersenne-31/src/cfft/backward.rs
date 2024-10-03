//! Discrete Fourier Transform, in-place, decimation-in-time
//!
//! Straightforward recursive algorithm, "unrolled" up to size 256.
//!
//! Inspired by Bernstein's djbfft: https://cr.yp.to/djbfft.html

extern crate alloc;
use alloc::vec::Vec;

use itertools::izip;
use p3_field::PackedField;

use crate::{to_mersenne31_array, Mersenne31};

const TWIDDLES_4: [Mersenne31; 2] = to_mersenne31_array([590768354, 1168891274]);

impl Mersenne31 {
    #[inline(always)]
    fn backward_butterfly<PF: PackedField<Scalar = Mersenne31>>(
        x: PF,
        y: PF,
        w: Self,
    ) -> (PF, PF) {
        let t = y * PF::from_f(w);
        (x + t, x - t)
    }

    #[inline]
    fn backward_pass<PF: PackedField<Scalar = Mersenne31>>(a: &mut [PF], roots: &[Self]) {
        let half_n = a.len() / 2;
        assert_eq!(roots.len(), half_n - 1);

        // Safe because 0 <= half_n < a.len()
        let (top, tail) = unsafe { a.split_at_mut_unchecked(half_n) };

        izip!(top.iter_mut(), tail.iter_mut(), roots).for_each(|(hi, lo, &root)| {
            (*hi, *lo) = Self::backward_butterfly(*hi, *lo, root);
        });
    }

    #[inline(always)]
    fn backward_2<PF: PackedField<Scalar = Mersenne31>>(a: &mut [PF]) {
        assert_eq!(a.len(), 2);

        let s = a[0] + a[1];
        let t = a[0] - a[1].mul_2exp_u64(15); // The smalleest t
        a[0] = s;
        a[1] = t;
    }

    #[inline(always)]
    fn backward_4<PF: PackedField<Scalar = Mersenne31>>(a: &mut [PF]) {
        assert_eq!(a.len(), 4);

        // Read in bit-reversed order
        let a0 = a[0];
        let a2 = a[1];
        let a1 = a[2];
        let a3 = a[3];

        // Expanding the calculation of t3 saves one instruction
        let t1 = a1 - a3;
        let t3 = t1; // TODO: Determine the small twiddles
        let t5 = a1 + a3;
        let t4 = a0 + a2;
        let t2 = a0 - a2;

        a[0] = t4 + t5;
        a[1] = t2 + t3;
        a[2] = t4 - t5;
        a[3] = t2 - t3;
    }

    #[inline(always)]
    fn backward_8<PF: PackedField<Scalar = Mersenne31>>(a: &mut [PF]) {
        assert_eq!(a.len(), 8);

        // Safe because a.len() == 8
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::backward_4(a0);
        Self::backward_4(a1);

        Self::backward_pass(a, todo!());
    }

    #[inline(always)]
    fn backward_16<PF: PackedField<Scalar = Mersenne31>>(a: &mut [PF]) {
        assert_eq!(a.len(), 16);

        // Safe because a.len() == 16
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::backward_8(a0);
        Self::backward_8(a1);

        Self::backward_pass(a, todo!());
    }

    #[inline(always)]
    fn backward_32<PF: PackedField<Scalar = Mersenne31>>(
        a: &mut [PF],
        root_table: &[Vec<Self>],
    ) {
        assert_eq!(a.len(), 32);

        // Safe because a.len() == 32
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::backward_16(a0);
        Self::backward_16(a1);

        Self::backward_pass(a, &root_table[0]);
    }

    #[inline(always)]
    fn backward_64<PF: PackedField<Scalar = Mersenne31>>(
        a: &mut [PF],
        root_table: &[Vec<Self>],
    ) {
        assert_eq!(a.len(), 64);

        // Safe because a.len() == 64
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::backward_32(a0, &root_table[1..]);
        Self::backward_32(a1, &root_table[1..]);

        Self::backward_pass(a, &root_table[0]);
    }

    #[inline(always)]
    fn backward_128<PF: PackedField<Scalar = Mersenne31>>(
        a: &mut [PF],
        root_table: &[Vec<Self>],
    ) {
        assert_eq!(a.len(), 128);

        // Safe because a.len() == 128
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::backward_64(a0, &root_table[1..]);
        Self::backward_64(a1, &root_table[1..]);

        Self::backward_pass(a, &root_table[0]);
    }

    #[inline(always)]
    fn backward_256<PF: PackedField<Scalar = Mersenne31>>(
        a: &mut [PF],
        root_table: &[Vec<Self>],
    ) {
        assert_eq!(a.len(), 256);

        // Safe because a.len() == 256
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::backward_128(a0, &root_table[1..]);
        Self::backward_128(a1, &root_table[1..]);

        Self::backward_pass(a, &root_table[0]);
    }

    #[inline]
    pub fn backward_fft<PF: PackedField<Scalar = Mersenne31>>(
        a: &mut [PF],
        root_table: &[Vec<Self>],
    ) {
        let n = a.len();
        if n == 1 {
            return;
        }

        assert_eq!(n, 1 << (root_table.len() + 1));
        match n {
            256 => Self::backward_256(a, root_table),
            128 => Self::backward_128(a, root_table),
            64 => Self::backward_64(a, root_table),
            32 => Self::backward_32(a, root_table),
            16 => Self::backward_16(a),
            8 => Self::backward_8(a),
            4 => Self::backward_4(a),
            2 => Self::backward_2(a),
            _ => {
                debug_assert!(n > 64);

                // Safe because a.len() > 64
                let (a0, a1) = unsafe { a.split_at_mut_unchecked(n / 2) };
                Self::backward_fft(a0, &root_table[1..]);
                Self::backward_fft(a1, &root_table[1..]);

                Self::backward_pass(a, &root_table[0]);
            }
        }
    }
}