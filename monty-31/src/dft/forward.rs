//! Discrete Fourier Transform, in-place, decimation-in-frequency
//!
//! Straightforward recusive algorithm, "unrolled" up to size 256.
//!
//! Inspired by Bernstein's djbfft: https://cr.yp.to/djbfft

extern crate alloc;
use alloc::vec::Vec;

use p3_field::{AbstractField, Field, TwoAdicField};
use p3_util::{log2_strict_usize, split_at_mut_unchecked};

use crate::{
    monty_reduce, partial_monty_reduce, reduce_2p, reduce_4p, FieldParameters, MontyField31,
    MontyParameters, TwoAdicData,
};

impl<MP: FieldParameters + TwoAdicData> MontyField31<MP> {
    /// Given a field element `gen` of order n where `n = 2^lg_n`,
    /// return a vector of vectors `table` where table[i] is the
    /// vector of twiddle factors for an fft of length n/2^i. The values
    /// gen^0 = 1 are skipped, as are g_i^k for k >= i/2 as these are
    /// just the negatives of the other roots (using g_i^{i/2} = -1).
    fn make_table(gen: Self, lg_n: usize) -> Vec<Vec<Self>> {
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
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    pub fn roots_of_unity_table(n: usize) -> Vec<Vec<Self>> {
        let lg_n = log2_strict_usize(n);
        let gen = Self::two_adic_generator(lg_n);
        Self::make_table(gen, lg_n)
    }

    pub fn inv_roots_of_unity_table(n: usize) -> Vec<Vec<Self>> {
        let lg_n = log2_strict_usize(n);
        let gen = Self::two_adic_generator(lg_n).inverse();
        Self::make_table(gen, lg_n)
    }
}

impl<MP: MontyParameters + TwoAdicData> MontyField31<MP> {
    #[inline(always)]
    fn forward_butterfly(x: Self, y: Self, w: Self) -> (Self, Self) {
        let t = MP::PRIME + x.value - y.value;
        (
            Self::new_monty(reduce_2p::<MP>(x.value + y.value)),
            Self::new_monty(monty_reduce::<MP>(t as u64 * w.value as u64)),
        )
    }

    #[inline]
    fn forward_pass(a: &mut [Self], roots: &[Self]) {
        let half_n = a.len() / 2;
        assert_eq!(roots.len(), half_n - 1);

        let (top, tail) = unsafe { split_at_mut_unchecked(a, half_n) };

        let x = top[0];
        let y = tail[0];

        let s = reduce_2p::<MP>(x.value + y.value);
        let t = reduce_2p::<MP>(MP::PRIME + x.value - y.value);
        top[0].value = s;
        tail[0].value = t;

        for i in 1..half_n {
            (top[i], tail[i]) = Self::forward_butterfly(top[i], tail[i], roots[i - 1]);
        }
    }

    #[inline(always)]
    fn forward_2(a: &mut [Self]) {
        assert_eq!(a.len(), 2);

        let s = reduce_2p::<MP>(a[0].value + a[1].value);
        let t = reduce_2p::<MP>(MP::PRIME + a[0].value - a[1].value);
        a[0].value = s;
        a[1].value = t;
    }

    #[inline(always)]
    fn forward_4(a: &mut [Self]) {
        assert_eq!(a.len(), 4);

        let t1 = (MP::PRIME + a[1].value - a[3].value) as u64;
        let t5 = (a[1].value + a[3].value) as u64;
        let t3 = partial_monty_reduce::<MP>(t1 * MP::ROOTS_8.as_ref()[1].value as u64) as u64;
        let t4 = (a[0].value + a[2].value) as u64;
        let t2 = (MP::PRIME + a[0].value - a[2].value) as u64;

        // Return in bit-reversed order
        let a0 = reduce_4p::<MP>(t4 + t5); // b0
        let a2 = reduce_4p::<MP>(t2 + t3); // b1
        let a1 = reduce_4p::<MP>((2 * MP::PRIME as u64) + t4 - t5); // b2
        let a3 = reduce_4p::<MP>((2 * MP::PRIME as u64) + t2 - t3); // b3

        a[0].value = a0;
        a[2].value = a2;
        a[1].value = a1;
        a[3].value = a3;
    }

    #[inline(always)]
    fn forward_8(a: &mut [Self]) {
        assert_eq!(a.len(), 8);

        Self::forward_pass(a, MP::ROOTS_8.as_ref());

        let (a0, a1) = unsafe { split_at_mut_unchecked(a, a.len() / 2) };
        Self::forward_4(a0);
        Self::forward_4(a1);
    }

    #[inline(always)]
    fn forward_16(a: &mut [Self]) {
        assert_eq!(a.len(), 16);

        Self::forward_pass(a, MP::ROOTS_16.as_ref());

        let (a0, a1) = unsafe { split_at_mut_unchecked(a, a.len() / 2) };
        Self::forward_8(a0);
        Self::forward_8(a1);
    }

    #[inline(always)]
    fn forward_32(a: &mut [Self], root_table: &[Vec<Self>]) {
        assert_eq!(a.len(), 32);

        Self::forward_pass(a, &root_table[0]);

        let (a0, a1) = unsafe { split_at_mut_unchecked(a, a.len() / 2) };
        Self::forward_16(a0);
        Self::forward_16(a1);
    }

    #[inline(always)]
    fn forward_64(a: &mut [Self], root_table: &[Vec<Self>]) {
        assert_eq!(a.len(), 64);

        Self::forward_pass(a, &root_table[0]);

        let (a0, a1) = unsafe { split_at_mut_unchecked(a, a.len() / 2) };
        Self::forward_32(a0, &root_table[1..]);
        Self::forward_32(a1, &root_table[1..]);
    }

    #[inline(always)]
    fn forward_128(a: &mut [Self], root_table: &[Vec<Self>]) {
        assert_eq!(a.len(), 128);

        Self::forward_pass(a, &root_table[0]);

        let (a0, a1) = unsafe { split_at_mut_unchecked(a, a.len() / 2) };
        Self::forward_64(a0, &root_table[1..]);
        Self::forward_64(a1, &root_table[1..]);
    }

    #[inline(always)]
    fn forward_256(a: &mut [Self], root_table: &[Vec<Self>]) {
        assert_eq!(a.len(), 256);

        Self::forward_pass(a, &root_table[0]);

        let (a0, a1) = unsafe { split_at_mut_unchecked(a, a.len() / 2) };
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
                let (a0, a1) = unsafe { split_at_mut_unchecked(a, n / 2) };

                Self::forward_fft(a0, &root_table[1..]);
                Self::forward_fft(a1, &root_table[1..]);
            }
        }
    }
}
