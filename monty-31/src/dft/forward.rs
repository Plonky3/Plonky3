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

#[inline(always)]
fn forward_butterfly<T: PackedField>(x: T, y: T, roots: T) -> (T, T) {
    let t = x - y;
    (x + y, t * roots)
}

#[inline(always)]
fn forward_butterfly_interleaved<const HALF_RADIX: usize, T: PackedField>(
    x: T,
    y: T,
    roots: T,
) -> (T, T) {
    let (a, b) = x.interleave(y, HALF_RADIX);
    let (a, b) = forward_butterfly(a, b, roots);
    a.interleave(b, HALF_RADIX)
}

#[inline]
fn forward_iterative_layer_1<T: PackedField>(input: &mut [T], roots: &[T::Scalar]) {
    let packed_roots = T::pack_slice(roots);
    let n = input.len();
    let (xs, ys) = unsafe { input.split_at_mut_unchecked(n / 2) };

    izip!(xs, ys, packed_roots)
        .for_each(|(x, y, &roots)| (*x, *y) = forward_butterfly(*x, *y, roots));
}

#[inline]
fn forward_iterative_layer_2<T: PackedField>(input: &mut [T], roots: &[T::Scalar]) {
    let packed_roots = T::pack_slice(roots);
    let n = input.len();
    let (top_half, bottom_half) = unsafe { input.split_at_mut_unchecked(n / 2) };
    let (xs, ys) = unsafe { top_half.split_at_mut_unchecked(n / 4) };
    let (zs, ws) = unsafe { bottom_half.split_at_mut_unchecked(n / 4) };

    izip!(xs, ys, zs, ws, packed_roots).for_each(|(x, y, z, w, &root)| {
        (*x, *y) = forward_butterfly(*x, *y, root);
        (*z, *w) = forward_butterfly(*z, *w, root);
    });
}

#[inline]
fn forward_iterative_radix_r<const HALF_RADIX: usize, T: PackedField>(
    input: &mut [T],
    roots: &[T::Scalar],
) {
    // roots[0] == 1
    // roots <-- [1, roots[1], ..., roots[HALF_RADIX-1], 1, roots[1], ...]
    let roots = T::from_fn(|i| roots[i % HALF_RADIX]);

    input.chunks_exact_mut(2).for_each(|pair| {
        let (x, y) = forward_butterfly_interleaved::<HALF_RADIX, _>(pair[0], pair[1], roots);
        pair[0] = x;
        pair[1] = y;
    });
}

#[inline]
fn forward_iterative_radix_2<T: PackedField>(input: &mut [T]) {
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
    /// Breadth-first DIF FFT for smallish vectors (must be >= 64)
    #[inline]
    fn forward_iterative(input: &mut [Self], root_table: &[Vec<Self>]) {
        let n = input.len();
        let lg_n = log2_strict_usize(n);

        // Needed to avoid overlap with specialisation at the other end of the loop
        assert!(lg_n >= 6);

        let packing_width = <Self as Field>::Packing::WIDTH;
        assert!(n >= 2 * packing_width);

        let packed_input = <Self as Field>::Packing::pack_slice_mut(input);

        // Specialise the first few iterations; improves performance a little.
        forward_iterative_layer_1(packed_input, &root_table[0]); // lg_m == lg_n - 1, s == 0
        forward_iterative_layer_2(packed_input, &root_table[1]); // lg_m == lg_n - 2, s == 1

        for lg_m in (4..(lg_n - 2)).rev() {
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
                    .for_each(|(x, y, &root)| (*x, *y) = forward_butterfly(*x, *y, root));
            }
        }
        forward_iterative_radix_r::<8, _>(packed_input, &root_table[lg_n - 4]); // lg_m = 3; s = lg_n - 4
        forward_iterative_radix_r::<4, _>(packed_input, &root_table[lg_n - 3]); // lg_m = 2; s = lg_n - 3
        forward_iterative_radix_r::<2, _>(packed_input, &root_table[lg_n - 2]); // lg_m = 1; s = lg_n - 2
        forward_iterative_radix_2(packed_input); // lg_m = 0; s = lg_n - 1
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
    fn forward_pass(input: &mut [Self], roots: &[Self]) {
        let half_n = input.len() / 2;
        assert_eq!(roots.len(), half_n);

        if half_n >= <Self as Field>::Packing::WIDTH {
            let packed_input = <Self as Field>::Packing::pack_slice_mut(input);
            forward_iterative_layer_1(packed_input, roots);
        } else {
            // Safe because 0 <= half_n < a.len()
            let (xs, ys) = unsafe { input.split_at_mut_unchecked(half_n) };

            let s = xs[0] + ys[0];
            let t = xs[0] - ys[0];
            xs[0] = s;
            ys[0] = t;

            izip!(&mut xs[1..], &mut ys[1..], &roots[1..]).for_each(|(x, y, &root)| {
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

    /// Assumes `a.len() > 8`
    #[inline]
    fn forward_fft_recur(input: &mut [Self], root_table: &[Vec<Self>]) {
        const ITERATIVE_FFT_THRESHOLD: usize = 2048;

        let n = input.len();
        if n <= ITERATIVE_FFT_THRESHOLD {
            Self::forward_iterative(input, root_table);
        } else {
            assert_eq!(n, 1 << (root_table.len() + 1));
            Self::forward_pass(input, &root_table[0]);

            // Safe because a.len() > ITERATIVE_FFT_THRESHOLD
            let (a0, a1) = unsafe { input.split_at_mut_unchecked(n / 2) };

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
            // Note that the limit is 8 for AVX2 and 16 (as imposed here) for AVX512
            32 => Self::forward_32(a, root_table),
            16 => Self::forward_16(a),
            8 => Self::forward_8(a),
            4 => Self::forward_4(a),
            2 => Self::forward_2(a),
            _ => Self::forward_fft_recur(a, root_table),
        }
    }
}
