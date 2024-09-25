//! Discrete Fourier Transform, in-place, decimation-in-frequency
//!
//! Straightforward recursive algorithm, "unrolled" up to size 256.
//!
//! Inspired by Bernstein's djbfft: https://cr.yp.to/djbfft.html

extern crate alloc;
use alloc::vec::Vec;

use itertools::izip;
use p3_field::{AbstractField, TwoAdicField};
use p3_util::log2_strict_usize;

use crate::{monty_reduce, FieldParameters, MontyField31, MontyParameters, TwoAdicData};

impl<MP: FieldParameters + TwoAdicData> MontyField31<MP> {
    /// Given a field element `gen` of order n where `n = 2^lg_n`,
    /// return a vector of vectors `table` where table[i] is the
    /// vector of twiddle factors for an fft of length n/2^i. The values
    /// gen^0 = 1 are skipped, as are g_i^k for k >= i/2 as these are
    /// just the negatives of the other roots (using g_i^{i/2} = -1).
    pub fn roots_of_unity_table(n: usize) -> Vec<Vec<Self>> {
        let lg_n = log2_strict_usize(n);
        let gen = Self::two_adic_generator(lg_n);
        let half_n = 1 << (lg_n - 1);
        // nth_roots = [g, g^2, g^3, ..., g^{n/2 - 1}]
        let nth_roots: Vec<_> = gen.powers().take(half_n).skip(1).collect();

        (0..(lg_n - 1))
            .map(|i| {
                nth_roots
                    .iter()
                    .skip((1 << i) - 1)
                    .step_by(1 << i)
                    .copied()
                    .collect()
            })
            .collect()
    }
}

impl<MP: MontyParameters + TwoAdicData> MontyField31<MP> {
    #[inline(always)]
    fn forward_butterfly(x: Self, y: Self, w: Self) -> (Self, Self) {
        let t = MP::PRIME + x.value - y.value;
        (
            x + y,
            Self::new_monty(monty_reduce::<MP>(t as u64 * w.value as u64)),
        )
    }

    #[inline]
    fn forward_pass(a: &mut [Self], roots: &[Self]) {
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
    fn forward_2(a: &mut [Self]) {
        assert_eq!(a.len(), 2);

        let s = a[0] + a[1];
        let t = a[0] - a[1];
        a[0] = s;
        a[1] = t;
    }

    #[inline(always)]
    fn forward_4(a: &mut [Self]) {
        assert_eq!(a.len(), 4);

        // Expanding the calculation of t3 saves one instruction
        let t1 = MP::PRIME + a[1].value - a[3].value;
        let t3 = MontyField31::new_monty(monty_reduce::<MP>(
            t1 as u64 * MP::ROOTS_8.as_ref()[1].value as u64,
        ));
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
    fn forward_8(a: &mut [Self]) {
        assert_eq!(a.len(), 8);

        Self::forward_pass(a, MP::ROOTS_8.as_ref());

        // Safe because a.len() == 8
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::forward_4(a0);
        Self::forward_4(a1);
    }

    #[inline(always)]
    fn forward_16(a: &mut [Self]) {
        assert_eq!(a.len(), 16);

        Self::forward_pass(a, MP::ROOTS_16.as_ref());

        // Safe because a.len() == 16
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::forward_8(a0);
        Self::forward_8(a1);
    }

    #[inline(always)]
    fn forward_32(a: &mut [Self], root_table: &[Vec<Self>]) {
        assert_eq!(a.len(), 32);

        Self::forward_pass(a, &root_table[0]);

        // Safe because a.len() == 32
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::forward_16(a0);
        Self::forward_16(a1);
    }

    #[inline(always)]
    fn forward_64(a: &mut [Self], root_table: &[Vec<Self>]) {
        assert_eq!(a.len(), 64);

        Self::forward_pass(a, &root_table[0]);

        // Safe because a.len() == 64
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::forward_32(a0, &root_table[1..]);
        Self::forward_32(a1, &root_table[1..]);
    }

    #[inline(always)]
    fn forward_128(a: &mut [Self], root_table: &[Vec<Self>]) {
        assert_eq!(a.len(), 128);

        Self::forward_pass(a, &root_table[0]);

        // Safe because a.len() == 128
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::forward_64(a0, &root_table[1..]);
        Self::forward_64(a1, &root_table[1..]);
    }

    #[inline(always)]
    fn forward_256(a: &mut [Self], root_table: &[Vec<Self>]) {
        assert_eq!(a.len(), 256);

        Self::forward_pass(a, &root_table[0]);

        // Safe because a.len() == 256
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::forward_128(a0, &root_table[1..]);
        Self::forward_128(a1, &root_table[1..]);
    }

    #[inline]
    pub fn forward_fft(a: &mut [Self], root_table: &[Vec<Self>]) {
        let n = a.len();
        if n == 1 {
            return;
        }

        assert_eq!(n, 1 << (root_table.len() + 1));
        match n {
            256 => Self::forward_256(a, root_table),
            128 => Self::forward_128(a, root_table),
            64 => Self::forward_64(a, root_table),
            32 => Self::forward_32(a, root_table),
            16 => Self::forward_16(a),
            8 => Self::forward_8(a),
            4 => Self::forward_4(a),
            2 => Self::forward_2(a),
            _ => {
                debug_assert!(n > 64);
                Self::forward_pass(a, &root_table[0]);

                // Safe because a.len() > 64
                let (a0, a1) = unsafe { a.split_at_mut_unchecked(n / 2) };

                Self::forward_fft(a0, &root_table[1..]);
                Self::forward_fft(a1, &root_table[1..]);
            }
        }
    }
}
