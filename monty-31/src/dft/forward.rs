//! Discrete Fourier Transform, in-place, decimation-in-frequency
//!
//! Straightforward recursive algorithm, "unrolled" up to size 256.
//!
//! Inspired by Bernstein's djbfft: https://cr.yp.to/djbfft.html

extern crate alloc;

use alloc::vec::Vec;

use itertools::izip;
use p3_field::{AbstractField, Field, PackedField, PackedValue, TwoAdicField};
use p3_util::log2_strict_usize;

use crate::{monty_reduce, FieldParameters, MontyField31, TwoAdicData};

impl<MP: FieldParameters + TwoAdicData> MontyField31<MP> {
    /// Given a field element `gen` of order n where `n = 2^lg_n`,
    /// return a vector of vectors `table` where table[i] is the
    /// vector of twiddle factors for an fft of length n/2^i. The
    /// values g_i^k for k >= i/2 are skipped as these are just the
    /// negatives of the other roots (using g_i^{i/2} = -1).  The
    /// value gen^0 = 1 is included to aid consistency between the
    /// packed and non-packed variants.
    pub fn roots_of_unity_table(n: usize) -> Vec<Vec<Self>> {
        let lg_n = log2_strict_usize(n);
        let gen = Self::two_adic_generator(lg_n);
        let half_n = 1 << (lg_n - 1);
        // nth_roots = [1, g, g^2, g^3, ..., g^{n/2 - 1}]
        let nth_roots: Vec<_> = gen.powers().take(half_n).collect();

        (0..(lg_n - 1))
            .map(|i| nth_roots.iter().step_by(1 << i).copied().collect())
            .collect()
    }
}

impl<MP: FieldParameters + TwoAdicData> MontyField31<MP> {
    #[inline]
    fn forward_small_s0(a: &mut [Self], roots: &[Self]) {
        let n = a.len();
        // lg_m = lg_n - 1
        // s = 0
        // m = n/2
        // i = 0
        // offset = 0

        let packed_vec = <Self as Field>::Packing::pack_slice_mut(a);
        let packed_roots = <Self as Field>::Packing::pack_slice(roots);

        let m = n / 2;
        assert_eq!(m % <Self as Field>::Packing::WIDTH, 0);
        let m_elts = m / <Self as Field>::Packing::WIDTH;
        let (a0, a1) = unsafe { packed_vec.split_at_mut_unchecked(m_elts) };

        izip!(a0, a1, packed_roots).for_each(|(x, y, &root)| {
            let t = (*x - *y) * root;
            *x += *y;
            *y = t;
        });
    }

    #[inline]
    fn forward_small_s1(a: &mut [Self], roots: &[Self]) {
        let n = a.len();
        // s = 1
        // lg_m = lg_n - 2
        // m = n/4
        // i = 0, 1
        // offset = 0, n/2

        let (u, v) = unsafe { a.split_at_mut_unchecked(n / 2) };
        let (u0, u1) = unsafe { u.split_at_mut_unchecked(n / 4) };
        let (v0, v1) = unsafe { v.split_at_mut_unchecked(n / 4) };

        let m = n / 4;
        assert_eq!(m % <Self as Field>::Packing::WIDTH, 0);
        let u0 = <Self as Field>::Packing::pack_slice_mut(u0);
        let u1 = <Self as Field>::Packing::pack_slice_mut(u1);
        let v0 = <Self as Field>::Packing::pack_slice_mut(v0);
        let v1 = <Self as Field>::Packing::pack_slice_mut(v1);
        let packed_roots = <Self as Field>::Packing::pack_slice(roots);

        izip!(u0, u1, v0, v1, packed_roots).for_each(|(x, y, z, w, &root)| {
            let t1 = (*x - *y) * root;
            let t2 = (*z - *w) * root;
            *x += *y;
            *z += *w;
            *y = t1;
            *w = t2;
        });
    }

    #[inline]
    fn _forward_small_s0(a: &mut [Self], roots: &[Self]) {
        let n = a.len();
        // lg_m = lg_n - 1
        // s = 0
        // m = n/2
        // i = 0
        // offset = 0

        let m = n / 2;
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(m) };

        for k in 0..m {
            let x = a0[k];
            let y = a1[k];
            let t = MP::PRIME + x.value - y.value;
            a0[k] = x + y;
            a1[k] = Self::new_monty(monty_reduce::<MP>(t as u64 * roots[k].value as u64));
        }
    }

    #[inline]
    fn _forward_small_s1(a: &mut [Self], roots: &[Self]) {
        let n = a.len();
        // s = 1
        // lg_m = lg_n - 2
        // m = n/4
        // i = 0, 1
        // offset = 0, n/2

        let (a0, a1) = unsafe { a.split_at_mut_unchecked(n / 2) };

        // i = 0
        // offset = 0
        let m = n / 4;

        for k in 0..m {
            let x = a0[k];
            let y = a0[k + m];
            let t = MP::PRIME + x.value - y.value;
            a0[k] = x + y;
            a0[k + m] = Self::new_monty(monty_reduce::<MP>(t as u64 * roots[k].value as u64));
        }

        // i = 1
        // offset = n / 2
        for k in 0..m {
            let x = a1[k];
            let y = a1[k + m];
            let t = MP::PRIME + x.value - y.value;
            a1[k] = x + y;
            a1[k + m] = Self::new_monty(monty_reduce::<MP>(t as u64 * roots[k].value as u64));
        }
    }

    // FIXME: This might not work on AVX2 where WIDTH == 8
    #[inline]
    fn forward_small_t3(a: &mut [Self], roots: &[Self]) {
        let n = a.len();

        // TODO: Not sure this is the cleanest/fastest way to set r:
        // roots[0] == 1
        // r = [1, roots[1], 1, roots[1], ...]
        let r = <Self as Field>::Packing::from_fn(|i| roots[i % 8]);

        let packing_width = <Self as Field>::Packing::WIDTH;
        assert_eq!((n / 2) % packing_width, 0);
        assert!((n / 2) >= packing_width);
        let n_elts = (n / 2) / packing_width;
        let a = <Self as Field>::Packing::pack_slice_mut(a);

        for i in 0..n_elts {
            // lg_m = 1
            let offset = 2 * i;

            let x = a[offset];
            let y = a[offset + 1];
            let (mut x, y) = x.interleave(y, 8);
            let t = (x - y) * r;
            x += y;
            let (x, y) = x.interleave(t, 8);
            a[offset] = x;
            a[offset + 1] = y;
        }
    }

    #[inline]
    fn forward_small_t2(a: &mut [Self], roots: &[Self]) {
        let n = a.len();

        // TODO: Not sure this is the cleanest/fastest way to set r:
        // roots[0] == 1
        // r = [1, roots[1], 1, roots[1], ...]
        let r = <Self as Field>::Packing::from_fn(|i| roots[i % 4]);

        let packing_width = <Self as Field>::Packing::WIDTH;
        assert_eq!((n / 2) % packing_width, 0);
        assert!((n / 2) >= packing_width);
        let n_elts = (n / 2) / packing_width;
        let a = <Self as Field>::Packing::pack_slice_mut(a);

        for i in 0..n_elts {
            // lg_m = 1
            let offset = 2 * i;

            let x = a[offset];
            let y = a[offset + 1];
            let (mut x, y) = x.interleave(y, 4);
            let t = (x - y) * r;
            x += y;
            let (x, y) = x.interleave(t, 4);
            a[offset] = x;
            a[offset + 1] = y;
        }
    }

    #[inline]
    fn forward_small_t1(a: &mut [Self], roots: &[Self]) {
        // lg_m = 1
        // m = 2
        // s = lg_n - 2
        // i in 0..n/4
        // offset = 4*i
        // k in 0 .. 2
        let n = a.len();

        // TODO: Not sure this is the cleanest/fastest way to set r:
        // roots[0] == 1
        // r = [1, roots[1], 1, roots[1], ...]
        let r = <Self as Field>::Packing::from_fn(|i| roots[i % 2]);

        let packing_width = <Self as Field>::Packing::WIDTH;
        assert_eq!((n / 2) % packing_width, 0);
        assert!((n / 2) >= packing_width);
        // TODO: Store n_elts in caller and pass it in.
        let n_elts = (n / 2) / packing_width;
        let a = <Self as Field>::Packing::pack_slice_mut(a);

        for i in 0..n_elts {
            // lg_m = 1
            let offset = 2 * i;

            // m = 2
            let x = a[offset];
            let y = a[offset + 1];
            let (mut x, y) = x.interleave(y, 2);
            let t = (x - y) * r;
            x += y;
            let (x, y) = x.interleave(t, 2);
            a[offset] = x;
            a[offset + 1] = y;
        }
    }

    // TODO: use izip!
    // TODO: refactor all these functions using a const generic
    #[inline]
    fn forward_small_t0(a: &mut [Self]) {
        // lg_m = 0
        // m = 1
        // s = lg_n - 1
        // i in 0..n/2
        // offset = 2*i
        // k in 0 .. 1
        let packing_width = <Self as Field>::Packing::WIDTH;
        let n = a.len();
        let n_elts = (n / 2) / packing_width;
        let a = <Self as Field>::Packing::pack_slice_mut(a);
        for i in 0..n_elts {
            // lg_m = 0
            let offset = 2 * i;

            // k = 0
            // m = 1
            let x = a[offset];
            let y = a[offset + 1];
            let (mut x, y) = x.interleave(y, 1);
            let t = x - y; // roots[0] == 1
            x += y;
            let (x, y) = x.interleave(t, 1);
            a[offset] = x;
            a[offset + 1] = y;
        }
    }

    /// Breadth-first DIF FFT for small vectors
    #[inline]
    fn forward_small(a: &mut [Self], root_table: &[Vec<Self>]) {
        let n = a.len();
        let lg_n = log2_strict_usize(n);

        let packing_width = <Self as Field>::Packing::WIDTH;
        assert!(n >= 2 * packing_width);

        // Specialise the first few iterations; improves performance a little.

        // TODO: Need to avoid overlap with specialisation at the other end of the loop
        //Self::forward_small_s0(a, &root_table[0]); // lg_m == lg_n - 1, s == 0
        //Self::forward_small_s1(a, &root_table[1]); // lg_m == lg_n - 2, s == 1

        for lg_m in (4..lg_n).rev() {
            let s = lg_n - lg_m - 1;
            let m = 1 << lg_m;

            let roots = &root_table[s];
            assert_eq!(roots.len(), m);

            if packing_width <= n / (2 << s) {
                let packed_roots = <Self as Field>::Packing::pack_slice(roots);
                for i in 0..(1 << s) {
                    let offset = i << (lg_m + 1);

                    let b = &mut a[offset..];
                    let (b0, b1) = unsafe { b.split_at_mut_unchecked(m) };

                    let b0 = <Self as Field>::Packing::pack_slice_mut(b0);
                    let b1 = <Self as Field>::Packing::pack_slice_mut(b1);

                    izip!(b0, b1, packed_roots).for_each(|(x, y, &root)| {
                        let t = (*x - *y) * root;
                        *x += *y;
                        *y = t;
                    });
                }
            } else {
                panic!(
                    "shouldn't be here: s = {}; lg_m = {}; m = {}; n = {};",
                    s, lg_m, m, n
                );
            }
        }
        Self::forward_small_t3(a, &root_table[lg_n - 4]); // lg_m = 3; s = lg_n - 4
        Self::forward_small_t2(a, &root_table[lg_n - 3]); // lg_m = 2; s = lg_n - 3
        Self::forward_small_t1(a, &root_table[lg_n - 2]); // lg_m = 1; s = lg_n - 2
        Self::forward_small_t0(a); // lg_m = 0; s = lg_n - 1
    }

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
        assert_eq!(roots.len(), half_n);

        // Safe because 0 <= half_n < a.len()
        let (top, tail) = unsafe { a.split_at_mut_unchecked(half_n) };

        if half_n >= <Self as Field>::Packing::WIDTH {
            let top_packed = <Self as Field>::Packing::pack_slice_mut(top);
            let tail_packed = <Self as Field>::Packing::pack_slice_mut(tail);
            let roots_packed = <Self as Field>::Packing::pack_slice(roots);
            izip!(top_packed, tail_packed, roots_packed).for_each(|(x, y, &root)| {
                let t = (*x - *y) * root;
                *x += *y;
                *y = t;
            });
        } else {
            let s = top[0] + tail[0];
            let t = top[0] - tail[0];
            top[0] = s;
            tail[0] = t;

            izip!(&mut top[1..], &mut tail[1..], &roots[1..]).for_each(|(x, y, &root)| {
                (*x, *y) = Self::forward_butterfly(*x, *y, root);
            });
        }
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
            t1 as u64 * MP::ROOTS_8.as_ref()[2].value as u64,
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

    /// Assumes `a.len() > 8`
    #[inline]
    fn forward_fft_recur(a: &mut [Self], root_table: &[Vec<Self>]) {
        const ITERATIVE_FFT_THRESHOLD: usize = 1024;

        let n = a.len();
        if n <= ITERATIVE_FFT_THRESHOLD {
            Self::forward_small(a, root_table);
        } else {
            assert_eq!(n, 1 << (root_table.len() + 1));
            Self::forward_pass(a, &root_table[0]);

            // Safe because a.len() > ITERATIVE_FFT_THRESHOLD
            let (a0, a1) = unsafe { a.split_at_mut_unchecked(n / 2) };

            Self::forward_fft_recur(a0, &root_table[1..]);
            Self::forward_fft_recur(a1, &root_table[1..]);
        }
    }

    #[inline]
    pub fn forward_fft(a: &mut [Self], root_table: &[Vec<Self>]) {
        let n = a.len();
        if n == 1 {
            return;
        }
        assert_eq!(n, 1 << (root_table.len() + 1));
        match n {
            // TODO: Note that the limit is 8 for AVX2 and 16 (as imposed here) for AVX512
            16 => Self::forward_16(a),
            8 => Self::forward_8(a),
            4 => Self::forward_4(a),
            2 => Self::forward_2(a),
            _ => Self::forward_fft_recur(a, root_table),
        }
    }
}
