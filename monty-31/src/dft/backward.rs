//! Discrete Fourier Transform, in-place, decimation-in-time
//!
//! Straightforward recursive algorithm, "unrolled" up to size 256.
//!
//! Inspired by Bernstein's djbfft: https://cr.yp.to/djbfft.html

extern crate alloc;
use alloc::vec::Vec;

use itertools::izip;

use crate::{monty_reduce, FieldParameters, MontyField31, TwoAdicData};
use p3_field::{Field, PackedFieldPow2, PackedValue};
use p3_util::log2_strict_usize;

#[inline(always)]
fn backward_butterfly<T: PackedFieldPow2>(x: T, y: T, roots: T) -> (T, T) {
    let t = y * roots;
    (x + t, x - t)
}

#[inline(always)]
fn backward_butterfly_interleaved<const HALF_RADIX: usize, T: PackedFieldPow2>(
    x: T,
    y: T,
    roots: T,
) -> (T, T) {
    let (a, b) = x.interleave(y, HALF_RADIX);
    let (a, b) = backward_butterfly(a, b, roots);
    a.interleave(b, HALF_RADIX)
}

#[inline]
fn backward_iterative_layer_1<T: PackedFieldPow2>(input: &mut [T], roots: &[T::Scalar]) {
    let packed_roots = T::pack_slice(roots);
    let n = input.len();
    let (xs, ys) = unsafe { input.split_at_mut_unchecked(n / 2) };

    izip!(xs, ys, packed_roots)
        .for_each(|(x, y, &roots)| (*x, *y) = backward_butterfly(*x, *y, roots));
}

#[inline]
fn backward_iterative_layer_2<T: PackedFieldPow2>(input: &mut [T], roots: &[T::Scalar]) {
    let packed_roots = T::pack_slice(roots);
    let n = input.len();
    let (top_half, bottom_half) = unsafe { input.split_at_mut_unchecked(n / 2) };
    let (xs, ys) = unsafe { top_half.split_at_mut_unchecked(n / 4) };
    let (zs, ws) = unsafe { bottom_half.split_at_mut_unchecked(n / 4) };

    izip!(xs, ys, zs, ws, packed_roots).for_each(|(x, y, z, w, &root)| {
        (*x, *y) = backward_butterfly(*x, *y, root);
        (*z, *w) = backward_butterfly(*z, *w, root);
    });
}

#[inline]
fn backward_iterative_radix_r<const HALF_RADIX: usize, T: PackedFieldPow2>(
    input: &mut [T],
    roots: &[T::Scalar],
) {
    // roots[0] == 1
    // roots <-- [1, roots[1], ..., roots[HALF_RADIX-1], 1, roots[1], ...]
    let roots = T::from_fn(|i| roots[i % HALF_RADIX]);

    input.chunks_exact_mut(2).for_each(|pair| {
        let (x, y) = backward_butterfly_interleaved::<HALF_RADIX, _>(pair[0], pair[1], roots);
        pair[0] = x;
        pair[1] = y;
    });
}

#[inline]
fn backward_iterative_radix_2<T: PackedFieldPow2>(input: &mut [T]) {
    input.chunks_exact_mut(2).for_each(|pair| {
        let x = pair[0];
        let y = pair[1];
        let (mut x, y) = x.interleave(y, 1);
        let t = x - y; // roots[0] == 1
        x += y;
        let (x, y) = x.interleave(t, 1);
        pair[0] = x;
        pair[1] = y;
    });
}

impl<MP: FieldParameters + TwoAdicData> MontyField31<MP> {
    /// Breadth-first DIT FFT for smallish vectors (must be >= 64)
    #[inline]
    fn backward_iterative(input: &mut [Self], root_table: &[Vec<Self>]) {
        let n = input.len();
        let lg_n = log2_strict_usize(n);

        // Needed to avoid overlap with specialisation at the other end of the loop
        assert!(lg_n >= 6);

        let packing_width = <Self as Field>::Packing::WIDTH;
        assert!(n >= 2 * packing_width);

        let packed_input = <Self as Field>::Packing::pack_slice_mut(input);

        backward_iterative_radix_2(packed_input); // lg_m = 0; s = lg_n - 1
        backward_iterative_radix_r::<2, _>(packed_input, &root_table[lg_n - 2]); // lg_m = 1; s = lg_n - 2
        backward_iterative_radix_r::<4, _>(packed_input, &root_table[lg_n - 3]); // lg_m = 2; s = lg_n - 3
        backward_iterative_radix_r::<8, _>(packed_input, &root_table[lg_n - 4]); // lg_m = 3; s = lg_n - 4

        for lg_m in 4..(lg_n - 2) {
            let s = lg_n - lg_m - 1;
            let m = 1 << lg_m;

            let roots = &root_table[s];
            debug_assert_eq!(roots.len(), m);
            let packed_roots = <Self as Field>::Packing::pack_slice(roots);

            debug_assert!(packing_width <= n / (2 << s));
            for i in 0..(1 << s) {
                let offset = i << (lg_m + 1);

                // lg_m >= 4, so offset = 2^e * i with e >= 5, hence
                // packing_width divides offset
                let offset = offset / packing_width;

                // lg_m >= 4, so m = 2^lg_m >= 2^4, hence packing_width
                // divides m
                let m = m / packing_width;

                let block = &mut packed_input[offset..];
                let (xs, ys) = unsafe { block.split_at_mut_unchecked(m) };

                izip!(xs, ys, packed_roots)
                    .for_each(|(x, y, &root)| (*x, *y) = backward_butterfly(*x, *y, root));
            }
        }
        // Specialise the last few iterations; improves performance a little.
        backward_iterative_layer_2(packed_input, &root_table[1]); // lg_m == lg_n - 2, s == 1
        backward_iterative_layer_1(packed_input, &root_table[0]); // lg_m == lg_n - 1, s == 0
    }

    #[inline(always)]
    fn backward_butterfly(x: Self, y: Self, w: Self) -> (Self, Self) {
        let t = y * w;
        (x + t, x - t)
    }

    #[inline]
    fn backward_pass(input: &mut [Self], roots: &[Self]) {
        let half_n = input.len() / 2;
        assert_eq!(roots.len(), half_n);

        if half_n >= <Self as Field>::Packing::WIDTH {
            let packed_input = <Self as Field>::Packing::pack_slice_mut(input);
            backward_iterative_layer_1(packed_input, roots);
        } else {
            // Safe because 0 <= half_n < a.len()
            let (xs, ys) = unsafe { input.split_at_mut_unchecked(half_n) };

            let s = xs[0] + ys[0];
            let t = xs[0] - ys[0];
            xs[0] = s;
            ys[0] = t;

            izip!(&mut xs[1..], &mut ys[1..], &roots[1..]).for_each(|(x, y, &root)| {
                (*x, *y) = Self::backward_butterfly(*x, *y, root);
            });
        }
    }

    #[inline(always)]
    fn backward_2(a: &mut [Self]) {
        assert_eq!(a.len(), 2);

        let s = a[0] + a[1];
        let t = a[0] - a[1];
        a[0] = s;
        a[1] = t;
    }

    #[inline(always)]
    fn backward_4(a: &mut [Self]) {
        assert_eq!(a.len(), 4);

        // Read in bit-reversed order
        let a0 = a[0];
        let a2 = a[1];
        let a1 = a[2];
        let a3 = a[3];

        // Expanding the calculation of t3 saves one instruction
        let t1 = MP::PRIME + a1.value - a3.value;
        let t3 = MontyField31::new_monty(monty_reduce::<MP>(
            t1 as u64 * MP::INV_ROOTS_8.as_ref()[2].value as u64,
        ));
        let t5 = a1 + a3;
        let t4 = a0 + a2;
        let t2 = a0 - a2;

        a[0] = t4 + t5;
        a[1] = t2 + t3;
        a[2] = t4 - t5;
        a[3] = t2 - t3;
    }

    #[inline(always)]
    fn backward_8(a: &mut [Self]) {
        assert_eq!(a.len(), 8);

        // Safe because a.len() == 8
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::backward_4(a0);
        Self::backward_4(a1);

        Self::backward_pass(a, MP::INV_ROOTS_8.as_ref());
    }

    #[inline(always)]
    fn backward_16(a: &mut [Self]) {
        assert_eq!(a.len(), 16);

        // Safe because a.len() == 16
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::backward_8(a0);
        Self::backward_8(a1);

        Self::backward_pass(a, MP::INV_ROOTS_16.as_ref());
    }

    #[inline(always)]
    fn backward_32(a: &mut [Self], root_table: &[Vec<Self>]) {
        assert_eq!(a.len(), 32);

        // Safe because a.len() == 32
        let (a0, a1) = unsafe { a.split_at_mut_unchecked(a.len() / 2) };
        Self::backward_16(a0);
        Self::backward_16(a1);

        Self::backward_pass(a, &root_table[0]);
    }

    /// Assumes `a.len() > 8`
    #[inline]
    fn backward_fft_recur(input: &mut [Self], root_table: &[Vec<Self>]) {
        const ITERATIVE_FFT_THRESHOLD: usize = 2048;

        let n = input.len();
        if n <= ITERATIVE_FFT_THRESHOLD {
            Self::backward_iterative(input, root_table);
        } else {
            assert_eq!(n, 1 << (root_table.len() + 1));

            // Safe because a.len() > ITERATIVE_FFT_THRESHOLD
            let (a0, a1) = unsafe { input.split_at_mut_unchecked(n / 2) };
            Self::backward_fft_recur(a0, &root_table[1..]);
            Self::backward_fft_recur(a1, &root_table[1..]);

            Self::backward_pass(input, &root_table[0]);
        }
    }

    #[inline]
    pub fn backward_fft(a: &mut [Self], root_table: &[Vec<Self>]) {
        let n = a.len();
        if n == 1 {
            return;
        }

        assert_eq!(n, 1 << (root_table.len() + 1));
        match n {
            32 => Self::backward_32(a, root_table),
            16 => Self::backward_16(a),
            8 => Self::backward_8(a),
            4 => Self::backward_4(a),
            2 => Self::backward_2(a),
            _ => Self::backward_fft_recur(a, root_table),
        }
    }
}
