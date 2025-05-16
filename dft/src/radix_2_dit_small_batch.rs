use core::{cell::RefCell, iter};

use alloc::vec::Vec;
use itertools::izip;
use p3_field::{Field, PackedFieldPow2, PackedValue, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::TwoAdicSubgroupDft;

/// A FFT algorithm which divides a butterfly network's layers into two halves.
///
/// Unlike other FFT algorithms, this algorithm is optimized for small batch sizes.
/// Hence it does not do any parallelization.
///
/// For the first half, we apply a butterfly network with smaller blocks in earlier layers,
/// i.e. either DIT or Bowers G. Then we bit-reverse, and for the second half, we continue executing
/// the same network but in bit-reversed order. This way we're always working with small blocks,
/// so within each half, we can have a certain amount of parallelism with no cross-thread
/// communication.
#[derive(Default, Clone, Debug)]
pub struct Radix2DitSmallBatch<F: Field> {
    /// Memoized twiddle factors for each length log_n.
    ///
    /// twiddles are stored in reverse order so twiddle[i]
    /// is a vector of length 2^{i + 1} designed to be used in the round
    /// of size 2^{i + 2}. E.g. twiddles[0] = vec![1, i] and will be
    /// used in the round of size 4. Twiddles are not stored
    /// for the final round of size 2, as the only twiddle is 1.
    ///
    /// TODO: The use of RefCell means this can't be shared across
    /// threads; consider using RwLock or finding a better design
    /// instead.
    twiddles: RefCell<Vec<Vec<F>>>,

    /// Memoized inverse twiddle factors for each length log_n.
    inv_twiddles: RefCell<Vec<Vec<F>>>,
}

impl<F: TwoAdicField> Radix2DitSmallBatch<F> {
    pub fn new(n: usize) -> Self {
        let res = Self {
            twiddles: RefCell::default(),
            inv_twiddles: RefCell::default(),
        };
        res.update_twiddles(n);
        res
    }

    /// Given a field element `gen` of order n where `n = 2^lg_n`,
    /// return a vector of vectors `table` where table[i] is the
    /// vector of twiddle factors for an fft of length n/2^i. The
    /// values g_i^k for k >= i/2 are skipped as these are just the
    /// negatives of the other roots (using g_i^{i/2} = -1).  The
    /// value gen^0 = 1 is included to aid consistency between the
    /// packed and non-packed variants.
    fn roots_of_unity_table(&self, n: usize) -> Vec<Vec<F>> {
        let lg_n = log2_strict_usize(n);
        let generator = F::two_adic_generator(lg_n);
        let half_n = 1 << (lg_n - 1);
        // nth_roots = [1, g, g^2, g^3, ..., g^{n/2 - 1}]
        let nth_roots: Vec<_> = generator.powers().take(half_n).collect();

        (0..(lg_n - 1))
            .rev()
            .map(|i| nth_roots.iter().step_by(1 << i).copied().collect())
            .collect()
    }

    /// Compute twiddle factors, or take memoized ones if already available.
    fn update_twiddles(&self, fft_len: usize) {
        // TODO: This recomputes the entire table from scratch if we
        // need it to be bigger, which is wasteful.

        // As we don't save the twiddles for the final layer where
        // the only twiddle is 1, roots_of_unity_table(fft_len)
        // returns a vector of twiddles of length log_2(fft_len) - 1.
        let curr_max_fft_len = 2 << self.twiddles.borrow().len();
        if fft_len > curr_max_fft_len {
            let new_twiddles = self.roots_of_unity_table(fft_len);
            // We can obtain the inverse twiddles by reversing and
            // negating the twiddles.
            let new_inv_twiddles = new_twiddles
                .iter()
                .map(|ts| {
                    // The first twiddle is still one, we reverse and negate the rest...
                    iter::once(F::ONE)
                        .chain(ts[1..].iter().rev().map(|&t| -t))
                        .collect()
                })
                .collect();
            self.twiddles.replace(new_twiddles);
            self.inv_twiddles.replace(new_inv_twiddles);
        }
    }
}

impl<F> TwoAdicSubgroupDft<F> for Radix2DitSmallBatch<F>
where
    F: TwoAdicField,
    F::Packing: PackedFieldPow2,
{
    type Evaluations = RowMajorMatrix<F>;

    fn dft(&self, mut vec: Vec<F>) -> Vec<F> {
        self.update_twiddles(vec.len());
        self.forward_fft(&mut vec);
        vec
    }

    fn dft_batch(&self, mat: RowMajorMatrix<F>) -> Self::Evaluations {
        let mut mat_transposed = mat.transpose();
        self.update_twiddles(mat.height());
        mat_transposed.par_rows_mut().for_each(|row| {
            self.forward_fft(row);
        });
        mat_transposed.transpose()
    }
}

impl<F: TwoAdicField> Radix2DitSmallBatch<F> {
    #[inline(always)]
    fn forward_2(&self, a: &mut [F]) {
        assert_eq!(a.len(), 2);

        let s = a[0] + a[1];
        let t = a[0] - a[1];
        a[0] = s;
        a[1] = t;
    }

    #[inline(always)]
    fn forward_4(&self, a: &mut [F], twiddle: &F) {
        assert_eq!(a.len(), 4);

        // Expanding the calculation of t3 saves one instruction
        let t3 = *twiddle * (a[1] - a[3]);
        let t2 = a[1] + a[3];
        let t0 = a[0] + a[2];
        let t1 = a[0] - a[2];

        // Return in bit-reversed order
        a[0] = t0 + t2;
        a[1] = t0 - t2;
        a[2] = t1 + t3;
        a[3] = t1 - t3;
    }

    #[inline]
    fn forward_iterative_layer(packed_input: &mut [F::Packing], roots: &[F], m: usize) {
        assert_eq!(roots.len(), m);
        let packed_roots = F::Packing::pack_slice(roots);

        // lg_m >= 4, so m = 2^lg_m >= 2^4, hence packing_width divides m
        let packed_m = m / F::Packing::WIDTH;
        packed_input
            .chunks_exact_mut(2 * packed_m)
            .for_each(|layer_chunk| {
                let (xs, ys) = unsafe { layer_chunk.split_at_mut_unchecked(packed_m) };

                izip!(xs, ys, packed_roots)
                    .for_each(|(x, y, &root)| (*x, *y) = forward_butterfly(*x, *y, root));
            });
    }

    #[inline]
    fn forward_iterative_packed_radix_16(&self, input: &mut [F::Packing])
    where
        F::Packing: PackedFieldPow2,
    {
        // Rather surprisingly, a version similar where the separate
        // loops in each call to forward_iterative_packed() are
        // combined into one, was not only not faster, but was
        // actually a bit slower.

        // Radix 16
        if F::Packing::WIDTH >= 16 {
            forward_iterative_packed::<8, _>(input, self.twiddles.borrow()[2].as_ref());
        } else {
            Self::forward_iterative_layer(input, self.twiddles.borrow()[2].as_ref(), 8);
        }

        // Radix 8
        if F::Packing::WIDTH >= 8 {
            forward_iterative_packed::<4, _>(input, self.twiddles.borrow()[1].as_ref());
        } else {
            Self::forward_iterative_layer(input, self.twiddles.borrow()[1].as_ref(), 4);
        }

        // Radix 4
        if F::Packing::WIDTH >= 4 {
            forward_iterative_packed::<2, _>(input, &self.twiddles.borrow()[0]);
        } else {
            Self::forward_iterative_layer(input, &self.twiddles.borrow()[0], 2);
        }

        // Radix 2
        forward_iterative_packed_radix_2(input);
    }

    /// Breadth-first DIF FFT for smallish vectors (must be >= 64)
    #[inline]
    fn forward_iterative(&self, packed_input: &mut [F::Packing])
    where
        F::Packing: PackedFieldPow2,
    {
        assert!(packed_input.len() >= 2);
        let packing_width = F::Packing::WIDTH;
        let n = packed_input.len() * packing_width;
        let lg_n = log2_strict_usize(n);

        // Stop loop early to do radix 16 separately. This value is determined by the largest
        // packing width we will encounter, which is 16 at the moment for AVX512. Specifically
        // it is log_2(max{possible packing widths}) = lg(16) = 4.
        const LAST_LOOP_LAYER: usize = 4;

        // How many layers have we specialised before the main loop
        const NUM_SPECIALISATIONS: usize = 2;

        // Needed to avoid overlap of the 2 specialisations at the start
        // with the radix-16 specialisation at the end of the loop
        assert!(lg_n >= LAST_LOOP_LAYER + NUM_SPECIALISATIONS);

        // Specialise the first NUM_SPECIALISATIONS iterations; improves performance a little.
        forward_pass_packed(packed_input, &self.twiddles.borrow()[lg_n - 1]); // lg_m == lg_n - 1, s == 0
        forward_iterative_layer_1(packed_input, &self.twiddles.borrow()[lg_n]); // lg_m == lg_n - 2, s == 1

        // loop from lg_n-2 down to 4.
        for lg_m in (LAST_LOOP_LAYER..(lg_n - NUM_SPECIALISATIONS)).rev() {
            let m = 1 << lg_m;

            let roots = &self.twiddles.borrow()[lg_m - 1];
            debug_assert_eq!(roots.len(), m);

            Self::forward_iterative_layer(packed_input, roots, m);
        }

        // Last 4 layers
        self.forward_iterative_packed_radix_16(packed_input);
    }

    /// Assumes `input.len() >= 64`.
    #[inline]
    fn forward_fft_recur(&self, input: &mut [F::Packing])
    where
        F::Packing: PackedFieldPow2,
    {
        const ITERATIVE_FFT_THRESHOLD: usize = 1024;

        let n = input.len() * F::Packing::WIDTH;
        let log_n = log2_strict_usize(n);
        if n <= ITERATIVE_FFT_THRESHOLD {
            self.forward_iterative(input);
        } else {
            forward_pass_packed(input, &self.twiddles.borrow()[log_n - 2]);

            // Safe because input.len() > ITERATIVE_FFT_THRESHOLD
            let (a0, a1) = unsafe { input.split_at_mut_unchecked(input.len() / 2) };

            self.forward_fft_recur(a0);
            self.forward_fft_recur(a1);
        }
    }

    #[inline]
    fn forward_fft(&self, input: &mut [F])
    where
        F::Packing: PackedFieldPow2,
    {
        let n = input.len();
        match n {
            4 => self.forward_4(input, &self.twiddles.borrow()[0][1]),
            2 => self.forward_2(input),
            1 => (),
            _ => {
                let packed_input = F::Packing::pack_slice_mut(input);
                self.forward_fft_recur(packed_input)
            }
        }
    }
}

#[inline]
fn forward_iterative_packed<const HALF_RADIX: usize, T: PackedFieldPow2>(
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

#[inline(always)]
fn forward_butterfly<T: PrimeCharacteristicRing + Copy>(x: T, y: T, roots: T) -> (T, T) {
    let t = x - y;
    (x + y, t * roots)
}

#[inline(always)]
fn forward_butterfly_interleaved<const HALF_RADIX: usize, T: PackedFieldPow2>(
    x: T,
    y: T,
    roots: T,
) -> (T, T) {
    let (x, y) = x.interleave(y, HALF_RADIX);
    let (x, y) = forward_butterfly(x, y, roots);
    x.interleave(y, HALF_RADIX)
}

#[inline]
fn forward_pass_packed<T: PackedFieldPow2>(input: &mut [T], roots: &[T::Scalar]) {
    assert!(roots.len() >= 8);
    let packed_roots = T::pack_slice(roots);
    let n = input.len();
    let (xs, ys) = unsafe { input.split_at_mut_unchecked(n / 2) };

    izip!(xs, ys, packed_roots)
        .for_each(|(x, y, &roots)| (*x, *y) = forward_butterfly(*x, *y, roots));
}

#[inline]
fn forward_iterative_layer_1<T: PackedFieldPow2>(input: &mut [T], twiddles: &[T::Scalar]) {
    assert!(twiddles.len() >= 8);
    let packed_twiddles = T::pack_slice(twiddles);
    let n = input.len();
    let (top_half, bottom_half) = unsafe { input.split_at_mut_unchecked(n / 2) };
    let (xs, ys) = unsafe { top_half.split_at_mut_unchecked(n / 4) };
    let (zs, ws) = unsafe { bottom_half.split_at_mut_unchecked(n / 4) };

    izip!(xs, ys, zs, ws, packed_twiddles).for_each(|(x, y, z, w, &root)| {
        (*x, *y) = forward_butterfly(*x, *y, root);
        (*z, *w) = forward_butterfly(*z, *w, root);
    });
}

#[inline]
fn forward_iterative_packed_radix_2<T: PackedFieldPow2>(input: &mut [T]) {
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
