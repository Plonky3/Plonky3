use alloc::vec::Vec;

use p3_dft::{reverse_matrix_index_bits, TwoAdicSubgroupDft};
use p3_field::{AbstractField, PrimeField64, TwoAdicField};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixViewMut};
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;

use crate::{Mersenne31, Mersenne31Complex};

type Base = Mersenne31;
type Ext = Mersenne31Complex<Base>;

#[derive(Default, Clone)]
pub struct Mersenne31ComplexRadix2Dit;

impl TwoAdicSubgroupDft<Ext> for Mersenne31ComplexRadix2Dit {
    fn dft_batch_bitrev<const IN_BITREV: bool, const OUT_BITREV: bool>(
        &self,
        mut mat: RowMajorMatrix<Ext>,
    ) -> RowMajorMatrix<Ext> {
        let h = mat.height();
        let log_h = log2_strict_usize(h);

        let root = Ext::two_adic_generator(log_h);
        let twiddles: Vec<Ext> = root.powers().take(h / 2).collect();

        if !IN_BITREV {
            reverse_matrix_index_bits(&mut mat);
        }
        // DIT butterfly
        for layer in 0..log_h {
            dit_layer(&mut mat.as_view_mut(), layer, &twiddles);
        }
        if OUT_BITREV {
            reverse_matrix_index_bits(&mut mat);
        }
        mat
    }
}

// NB: Most of what follows is copypasta from `dft/src/radix_2_dit.rs`.
// This is ugly, but the alternative is finding another way to "inject"
// the specialisation of the butterfly evaluation to Mersenne31Complex
// (in `dit_butterfly_inner()` below) into the existing structure.

/// One layer of a DIT butterfly network.
fn dit_layer(mat: &mut RowMajorMatrixViewMut<Ext>, layer: usize, twiddles: &[Ext]) {
    let h = mat.height();
    let log_h = log2_strict_usize(h);
    let layer_rev = log_h - 1 - layer;

    let half_block_size = 1 << layer;
    let block_size = half_block_size * 2;

    for j in (0..h).step_by(block_size) {
        // Unroll i=0 case
        let butterfly_hi = j;
        let butterfly_lo = butterfly_hi + half_block_size;
        twiddle_free_butterfly(mat, butterfly_hi, butterfly_lo);

        for i in 1..half_block_size {
            let butterfly_hi = j + i;
            let butterfly_lo = butterfly_hi + half_block_size;
            let twiddle = twiddles[i << layer_rev];
            dit_butterfly(mat, butterfly_hi, butterfly_lo, twiddle);
        }
    }
}

#[inline]
fn twiddle_free_butterfly(mat: &mut RowMajorMatrixViewMut<Ext>, row_1: usize, row_2: usize) {
    let ((prefix_1, shorts_1, suffix_1), (prefix_2, shorts_2, suffix_2)) =
        mat.packing_aligned_rows(row_1, row_2);

    // TODO: There's no special packing for Mersenne31Complex at the
    // time of writing; when there is we'll want to expand this out
    // into three separate loops.
    let row_1 = prefix_1.iter_mut().chain(shorts_1).chain(suffix_1);
    let row_2 = prefix_2.iter_mut().chain(shorts_2).chain(suffix_2);

    for (x, y) in row_1.zip(row_2) {
        let sum = *x + *y;
        let diff = *x - *y;
        *x = sum;
        *y = diff;
    }
}

#[inline]
fn dit_butterfly(mat: &mut RowMajorMatrixViewMut<Ext>, row_1: usize, row_2: usize, twiddle: Ext) {
    let ((prefix_1, shorts_1, suffix_1), (prefix_2, shorts_2, suffix_2)) =
        mat.packing_aligned_rows(row_1, row_2);

    // TODO: There's no special packing for Mersenne31Complex at the
    // time of writing; when there is we'll want to expand this out
    // into three separate loops.
    let row_1 = prefix_1.iter_mut().chain(shorts_1).chain(suffix_1);
    let row_2 = prefix_2.iter_mut().chain(shorts_2).chain(suffix_2);

    for (x, y) in row_1.zip(row_2) {
        dit_butterfly_inner(x, y, twiddle);
    }
}

/// Given x, y, and twiddle, return the "butterfly values"
/// x' = x + y*twiddle and y' = x - y*twiddle.
///
/// NB: At the time of writing, replacing the straight-forward
/// implementation
///
///    let sum = *x + *y * twiddle;
///    let diff = *x - *y * twiddle;
///    *x = sum;
///    *y = diff;
///
/// with the one below approximately halved the runtime of a DFT over
/// `Mersenne31Complex`.
#[inline]
fn dit_butterfly_inner(x: &mut Ext, y: &mut Ext, twiddle: Ext) {
    // Adding any multiple of P doesn't change the result modulo P;
    // we use this to ensure that the inputs to `from_wrapped_u64`
    // below are non-negative.
    const P_SQR: i64 = (Base::ORDER_U64 * Base::ORDER_U64) as i64;
    const TWO_P_SQR: i64 = 2 * P_SQR;

    // Unpack the inputs;
    //   x = x1 + i*x2
    //   y = y1 + i*y2
    //   twiddle = w1 + i*w2
    let (x1, x2) = (x.parts[0].value as i64, x.parts[1].value as i64);
    let (y1, y2) = (y.parts[0].value as i64, y.parts[1].value as i64);
    let (w1, w2) = (twiddle.parts[0].value as i64, twiddle.parts[1].value as i64);

    // x ± y*twiddle
    // = (x1 + i*x2) ± (y1 + i*y2)*(w1 + i*w2)
    // = (x1 ± (y1*w1 - y2*w2)) + i*(x2 ± (y2*w1 + y1*w2))
    // = (x1 ± z1) + i*(x2 ± z2)
    // where z1 + i*z2 = y*twiddle

    let z1 = y1 * w1 - y2 * w2; // -P^2 <= z1 <= P^2

    // NB: 2*P^2 + P < 2^63

    // -P^2 <= x1 + z1 <= P^2 + P
    let a1 = Base::from_wrapped_u64((P_SQR + x1 + z1) as u64);
    // -P^2 <= x1 - z1 <= P^2 + P
    let b1 = Base::from_wrapped_u64((P_SQR + x1 - z1) as u64);

    let z2 = y2 * w1 + y1 * w2; // 0 <= z2 <= 2*P^2

    // 0 <= x2 + z2 <= 2*P^2 + P
    let a2 = Base::from_wrapped_u64((x2 + z2) as u64);
    // -2*P^2 <= x2 - z2 <= P
    let b2 = Base::from_wrapped_u64((TWO_P_SQR + x2 - z2) as u64);

    *x = Ext::new(a1, a2);
    *y = Ext::new(b1, b2);
}
