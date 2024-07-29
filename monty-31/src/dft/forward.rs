extern crate alloc;
use alloc::vec::Vec;

use p3_field::{AbstractField, TwoAdicField};
use p3_util::log2_strict_usize;

use super::split_at_mut_unchecked;
use crate::{monty_reduce, FieldParameters, MontyField31, MontyParameters, TwoAdicData};

impl<MP: FieldParameters + TwoAdicData> MontyField31<MP> {
    /// FIXME: Document the structure of the return value
    pub fn roots_of_unity_table(n: usize) -> Vec<Vec<Self>> {
        // TODO: Consider following Hexl and storing the roots in a single
        // array in bit-reversed order, but with duplicates for certain roots
        // to avoid computing permutations in the inner loop.

        let lg_n = log2_strict_usize(n);
        let half_n = 1 << (lg_n - 1);
        // nth_roots = [g, g^2, g^3, ..., g^{n/2 - 1}]
        let gen = Self::two_adic_generator(lg_n);
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
}

impl<MP: MontyParameters + TwoAdicData> MontyField31<MP> {
    /// Given x in `0..P << MONTY_BITS`, return x mod P in [0, 2p).
    /// TODO: Double-check the ranges above.
    #[inline(always)]
    fn partial_monty_reduce(x: u64) -> u32 {
        let q = MP::MONTY_MU.wrapping_mul(x as u32);
        let h = ((q as u64 * MP::PRIME as u64) >> 32) as u32;
        MP::PRIME - h + (x >> 32) as u32
    }

    /// Given x in [0, 2p), return the representative of x mod p in [0, p)
    #[inline(always)]
    fn reduce_2p(x: u32) -> u32 {
        debug_assert!(x < 2 * MP::PRIME);

        if x < MP::PRIME {
            x
        } else {
            x - MP::PRIME
        }
    }

    /// Given x in [0, 4p), return the representative of x mod p in [0, p)
    #[inline(always)]
    fn reduce_4p(mut x: u64) -> u32 {
        debug_assert!(x < 4 * (MP::PRIME as u64));

        if x > (MP::PRIME as u64) {
            x -= MP::PRIME as u64;
        }
        if x > (MP::PRIME as u64) {
            x -= MP::PRIME as u64;
        }
        if x > (MP::PRIME as u64) {
            x -= MP::PRIME as u64;
        }
        x as u32
    }

    #[inline(always)]
    fn butterfly(x: Self, y: Self, w: Self) -> (Self, Self) {
        let t = MP::PRIME + x.value - y.value;
        (
            Self::new_monty(Self::reduce_2p(x.value + y.value)),
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

        let s = Self::reduce_2p(x.value + y.value);
        let t = Self::reduce_2p(MP::PRIME + x.value - y.value);
        top[0].value = s;
        tail[0].value = t;

        for i in 1..half_n {
            (top[i], tail[i]) = Self::butterfly(top[i], tail[i], roots[i - 1]);
        }
    }

    #[inline(always)]
    fn forward_2(a: &mut [Self]) {
        assert_eq!(a.len(), 2);

        let s = Self::reduce_2p(a[0].value + a[1].value);
        let t = Self::reduce_2p(MP::PRIME + a[0].value - a[1].value);
        a[0].value = s;
        a[1].value = t;
    }

    #[inline(always)]
    fn forward_4(a: &mut [Self]) {
        assert_eq!(a.len(), 4);

        let t1 = (MP::PRIME + a[1].value - a[3].value) as u64;
        let t5 = (a[1].value + a[3].value) as u64;
        let t3 = Self::partial_monty_reduce(t1 * MP::ROOTS_8.as_ref()[1].value as u64) as u64;
        let t4 = (a[0].value + a[2].value) as u64;
        let t2 = (MP::PRIME + a[0].value - a[2].value) as u64;

        // Return in bit-reversed order
        let a0 = Self::reduce_4p(t4 + t5); // b0
        let a2 = Self::reduce_4p(t2 + t3); // b1
        let a1 = Self::reduce_4p((2 * MP::PRIME as u64) + t4 - t5); // b2
        let a3 = Self::reduce_4p((2 * MP::PRIME as u64) + t2 - t3); // b3

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
        assert!(1 << (root_table.len() + 1) == n);

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
