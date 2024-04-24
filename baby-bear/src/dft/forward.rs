//use super::roots::{D1024, D128, D16, D2048, D256, D32, D4096, D512, D64, D8192};
use super::{split_at_mut_unchecked, Real, P};
use p3_field::{PrimeField64, TwoAdicField};
use p3_util::log2_strict_usize;

use alloc::vec::Vec;

/*
// a[0...8n-1], w[0...2n-2]; n >= 2
//
// TODO: Original comment is as above, but note that w should have
// length 2n-1, as is obvious from the original code, which addresses
// an odd number of elements of w.
fn forward_pass(a: &mut [Real], w: &[Real]) {
    debug_assert_eq!(a.len() % 8, 0);

    let n = a.len() / 8;

    debug_assert!(n >= 2);
    debug_assert_eq!(w.len(), 2 * n - 1);

    // Split a into four chunks of size 2*n.
    let (a0, a1) = unsafe { split_at_mut_unchecked(a, 2 * n) };
    let (a1, a2) = unsafe { split_at_mut_unchecked(a1, 2 * n) };
    let (a2, a3) = unsafe { split_at_mut_unchecked(a2, 2 * n) };

    transformzero(&mut a0[0], &mut a1[0], &mut a2[0], &mut a3[0]);

    // NB: The original version pulled the first iteration out of the
    // loop and unrolled the loop two iterations. When I did that all
    // the larger (>32) FFTs slowed down, most by about 25-35%!
    for i in 1..2 * n {
        transform(
            &mut a0[i],
            &mut a1[i],
            &mut a2[i],
            &mut a3[i],
            w[i - 1].re,
            w[i - 1].im,
        );
    }
}
*/

// copied from p3_dft::util::reverse_slice_index_bits;
#[inline]
const fn reverse_bits_len(x: usize, bit_len: usize) -> usize {
    // NB: The only reason we need overflowing_shr() here as opposed
    // to plain '>>' is to accommodate the case n == num_bits == 0,
    // which would become `0 >> 64`. Rust thinks that any shift of 64
    // bits causes overflow, even when the argument is zero.
    x.reverse_bits()
        .overflowing_shr(usize::BITS - bit_len as u32)
        .0
}
fn reverse_slice_index_bits<F>(vals: &mut [F]) {
    let n = vals.len();
    if n == 0 {
        return;
    }
    let log_n = log2_strict_usize(n);

    for i in 0..n {
        let j = reverse_bits_len(i, log_n);
        if i < j {
            vals.swap(i, j);
        }
    }
}

pub fn roots_of_unity_vector<F: PrimeField64 + TwoAdicField>(n: usize) -> Vec<Real> {
    let lg_n = log2_strict_usize(n);
    let rt = F::two_adic_generator(lg_n);

    let mut w = F::one();
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(w.as_canonical_u64() as i64);
        w *= rt;
    }
    //reverse_slice_index_bits(&mut v);
    v
}

#[inline]
fn forward_pass(a: &mut [Real], root: Real) {
    let half_n = a.len() / 2;
    let mut w = 1;
    for i in 0..half_n {
        let s = a[i];
        let t = a[i + half_n];
        a[i] = (s + t) % P;
        a[i + half_n] = ((P + s - t) * w) % P;
        w *= root;
        w %= P;
    }
}

#[inline]
pub fn forward_fft(a: &mut [Real], root: Real) {
    // roots: &[Real]) {
    let n = a.len();

    if n > 1 {
        forward_pass(a, root);
        let (a0, a1) = unsafe { split_at_mut_unchecked(a, n / 2) };

        let root_sqr = (root * root) % P;
        forward_fft(a0, root_sqr);
        forward_fft(a1, root_sqr);
    }
}
