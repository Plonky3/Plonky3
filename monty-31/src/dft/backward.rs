extern crate alloc;
use alloc::vec::Vec;

use super::split_at_mut_unchecked;
use crate::{monty_reduce, MontyField31, MontyParameters, TwoAdicData};

impl<MP: MontyParameters + TwoAdicData> MontyField31<MP> {
    #[inline(always)]
    fn backward_butterfly(x: Self, y: Self, w: Self) -> (Self, Self) {
        // TODO: Could do a partial_monty_reduce followed by reduce_3p's
        let t = monty_reduce::<MP>(y.value as u64 * w.value as u64);
        (
            Self::new_monty(Self::reduce_2p(x.value + t)),
            Self::new_monty(Self::reduce_2p(MP::PRIME + x.value - t)),
        )
    }

    #[inline]
    fn backward_pass(a: &mut [Self], roots: &[Self]) {
        let half_n = a.len() / 2;
        assert_eq!(roots.len(), half_n - 1);

        let (top, tail) = unsafe { split_at_mut_unchecked(a, half_n) };

        let x = top[0];
        let y = tail[0];

        let s = Self::reduce_2p(x.value + y.value);
        let t = Self::reduce_2p(MP::PRIME + x.value - y.value);
        top[0].value = s;
        tail[0].value = t;

        for i in 1..half_n {
            (top[i], tail[i]) = Self::backward_butterfly(top[i], tail[i], roots[i - 1]);
        }
    }

    #[inline(always)]
    fn backward_2(a: &mut [Self]) {
        assert_eq!(a.len(), 2);

        let s = Self::reduce_2p(a[0].value + a[1].value);
        let t = Self::reduce_2p(MP::PRIME + a[0].value - a[1].value);
        a[0].value = s;
        a[1].value = t;
    }

    #[inline(always)]
    fn backward_4(a: &mut [Self], root_table: &[Vec<Self>]) {
        assert_eq!(a.len(), 4);

        // TODO: Unroll
        let (a0, a1) = unsafe { split_at_mut_unchecked(a, a.len() / 2) };
        Self::backward_2(a0);
        Self::backward_2(a1);

        // TODO: Use MP::ROOTS_8
        Self::backward_pass(a, &root_table[0]);
    }

    #[inline(always)]
    fn backward_8(a: &mut [Self], root_table: &[Vec<Self>]) {
        assert_eq!(a.len(), 8);

        let (a0, a1) = unsafe { split_at_mut_unchecked(a, a.len() / 2) };
        Self::backward_4(a0, &root_table[1..]);
        Self::backward_4(a1, &root_table[1..]);

        // TODO: Use MP::ROOTS_8
        Self::backward_pass(a, &root_table[0]);
    }

    #[inline(always)]
    fn backward_16(a: &mut [Self], root_table: &[Vec<Self>]) {
        assert_eq!(a.len(), 16);

        let (a0, a1) = unsafe { split_at_mut_unchecked(a, a.len() / 2) };
        Self::backward_8(a0, &root_table[1..]);
        Self::backward_8(a1, &root_table[1..]);

        // TODO: Use MP::ROOTS_16
        Self::backward_pass(a, &root_table[0]);
    }

    #[inline(always)]
    fn backward_32(a: &mut [Self], root_table: &[Vec<Self>]) {
        assert_eq!(a.len(), 32);

        let (a0, a1) = unsafe { split_at_mut_unchecked(a, a.len() / 2) };
        Self::backward_16(a0, &root_table[1..]);
        Self::backward_16(a1, &root_table[1..]);

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
            16 => Self::backward_16(a, root_table),
            8 => Self::backward_8(a, root_table),
            4 => Self::backward_4(a, root_table),
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
