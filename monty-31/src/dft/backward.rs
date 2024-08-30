//! Discrete Fourier Transform, in-place, decimation-in-time
//!
//! Straightforward recusive algorithm, "unrolled" up to size 256.
//!
//! Inspired by Bernstein's djbfft: https://cr.yp.to/djbfft

extern crate alloc;
use alloc::vec::Vec;

use super::split_at_mut_unchecked;
use crate::{
    monty_reduce, partial_monty_reduce, reduce_2p, reduce_4p, MontyField31, MontyParameters,
    TwoAdicData,
};

impl<MP: MontyParameters + TwoAdicData> MontyField31<MP> {
    #[inline(always)]
    fn backward_butterfly(x: Self, y: Self, w: Self) -> (Self, Self) {
        // TODO: See if doing a partial_monty_reduce followed by reduce_3p's is faster
        let t = monty_reduce::<MP>(y.value as u64 * w.value as u64);
        (
            Self::new_monty(reduce_2p::<MP>(x.value + t)),
            Self::new_monty(reduce_2p::<MP>(MP::PRIME + x.value - t)),
        )
    }

    #[inline]
    fn backward_pass(a: &mut [Self], roots: &[Self]) {
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
            (top[i], tail[i]) = Self::backward_butterfly(top[i], tail[i], roots[i - 1]);
        }
    }

    #[inline(always)]
    fn backward_2(a: &mut [Self]) {
        assert_eq!(a.len(), 2);

        let s = reduce_2p::<MP>(a[0].value + a[1].value);
        let t = reduce_2p::<MP>(MP::PRIME + a[0].value - a[1].value);
        a[0].value = s;
        a[1].value = t;
    }

    #[inline(always)]
    fn backward_4(a: &mut [Self]) {
        assert_eq!(a.len(), 4);

        // Read in bit-reversed order
        let a0 = a[0].value;
        let a2 = a[1].value;
        let a1 = a[2].value;
        let a3 = a[3].value;

        let t1 = (MP::PRIME + a1 - a3) as u64;
        let t5 = (a1 + a3) as u64;
        let t3 = partial_monty_reduce::<MP>(t1 * MP::INV_ROOTS_8.as_ref()[1].value as u64) as u64;
        let t4 = (a0 + a2) as u64;
        let t2 = (MP::PRIME + a0 - a2) as u64;

        a[0].value = reduce_4p::<MP>(t4 + t5);
        a[1].value = reduce_4p::<MP>(t2 + t3);
        a[2].value = reduce_4p::<MP>((2 * MP::PRIME as u64) + t4 - t5);
        a[3].value = reduce_4p::<MP>((2 * MP::PRIME as u64) + t2 - t3);
    }

    #[inline(always)]
    fn backward_8(a: &mut [Self]) {
        assert_eq!(a.len(), 8);

        let (a0, a1) = unsafe { split_at_mut_unchecked(a, a.len() / 2) };
        Self::backward_4(a0);
        Self::backward_4(a1);

        Self::backward_pass(a, MP::INV_ROOTS_8.as_ref());
    }

    #[inline(always)]
    fn backward_16(a: &mut [Self]) {
        assert_eq!(a.len(), 16);

        let (a0, a1) = unsafe { split_at_mut_unchecked(a, a.len() / 2) };
        Self::backward_8(a0);
        Self::backward_8(a1);

        Self::backward_pass(a, MP::INV_ROOTS_16.as_ref());
    }

    #[inline(always)]
    fn backward_32(a: &mut [Self], root_table: &[Vec<Self>]) {
        assert_eq!(a.len(), 32);

        let (a0, a1) = unsafe { split_at_mut_unchecked(a, a.len() / 2) };
        Self::backward_16(a0);
        Self::backward_16(a1);

        Self::backward_pass(a, &root_table[0]);
    }

    #[inline(always)]
    fn backward_64(a: &mut [Self], root_table: &[Vec<Self>]) {
        assert_eq!(a.len(), 64);

        let (a0, a1) = unsafe { split_at_mut_unchecked(a, a.len() / 2) };
        Self::backward_32(a0, &root_table[1..]);
        Self::backward_32(a1, &root_table[1..]);

        Self::backward_pass(a, &root_table[0]);
    }

    #[inline(always)]
    fn backward_128(a: &mut [Self], root_table: &[Vec<Self>]) {
        assert_eq!(a.len(), 128);

        let (a0, a1) = unsafe { split_at_mut_unchecked(a, a.len() / 2) };
        Self::backward_64(a0, &root_table[1..]);
        Self::backward_64(a1, &root_table[1..]);

        Self::backward_pass(a, &root_table[0]);
    }

    #[inline(always)]
    fn backward_256(a: &mut [Self], root_table: &[Vec<Self>]) {
        assert_eq!(a.len(), 256);

        let (a0, a1) = unsafe { split_at_mut_unchecked(a, a.len() / 2) };
        Self::backward_128(a0, &root_table[1..]);
        Self::backward_128(a1, &root_table[1..]);

        Self::backward_pass(a, &root_table[0]);
    }

    #[inline]
    pub fn backward_fft(a: &mut [Self], root_table: &[Vec<Self>]) {
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
                let (a0, a1) = unsafe { split_at_mut_unchecked(a, n / 2) };
                Self::backward_fft(a0, &root_table[1..]);
                Self::backward_fft(a1, &root_table[1..]);

                Self::backward_pass(a, &root_table[0]);
            }
        }
    }
}
