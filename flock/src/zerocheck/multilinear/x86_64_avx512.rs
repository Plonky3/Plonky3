//! The non-storing bit pass on coefficient planes, eight quads to a register, so no product packs or unpacks.

use alloc::vec::Vec;
use core::arch::x86_64::__m512i;

use p3_binary_field::Poly192;

use super::{BitFold, LiveGroups, tasks};
use crate::byteslice::BLOCK;
use crate::zerocheck::EqHalves;
use crate::zerocheck::planar::{F192x8, F192x8Sum};

/// The eight pair sums of a bit pass over rows of `CHUNKS` bytes, one task per high eq index.
pub(super) fn bit_pass_sums<const CHUNKS: usize>(
    fold: &BitFold,
    rows: [&[[u8; CHUNKS]]; 2],
    eq: &EqHalves,
    live: LiveGroups,
    n_quads: usize,
) -> [Poly192; 8] {
    let lo_size = eq.low.len();
    // The eq table, eight consecutive weights a register per coefficient.
    let (groups, rest) = eq.low.as_chunks::<8>();
    assert!(rest.is_empty(), "whole groups of eight");
    let planes: Vec<F192x8> = groups.iter().map(F192x8::load).collect();
    tasks(n_quads.div_ceil(lo_size), None, 4 * lo_size, |hi, _| {
        quad_sums(fold, rows, hi * lo_size, &planes, live).map(|s| eq.high[hi] * s)
    })
}

/// The eight pair sums over the quads from `quad_first`, sixteen quads per pair of eq registers.
///
/// The eq weight multiplies the four `a` values first, so every term is a product of two operands.
#[inline]
fn quad_sums<const CHUNKS: usize>(
    fold: &BitFold,
    rows: [&[[u8; CHUNKS]]; 2],
    quad_first: usize,
    eq: &[F192x8],
    live: LiveGroups,
) -> [Poly192; 8] {
    let mut acc = [F192x8Sum::zero(); 8];
    for (b, eq) in eq.as_chunks::<2>().0.iter().enumerate() {
        let q0 = quad_first + (BLOCK / 4) * b;
        // A block wholly in padding folds to zero.
        if !(q0..q0 + BLOCK / 4).any(|q| live.contains(q)) {
            continue;
        }

        // Fold the block's `a`, `b` and `c = a AND b` rows straight into planes.
        let [ra, rb] = rows.map(|t| &t[4 * q0..].as_chunks::<BLOCK>().0[0]);
        let mut rc = [[0u8; CHUNKS]; BLOCK];
        super::and_rows(ra, rb, &mut rc);
        let [pa, pb, pc] = [ra, rb, &rc].map(|t| fold.fold_quads::<CHUNKS>(t));

        for (g, &e) in eq.iter().enumerate() {
            // Position `uv` of the eight quads `8g ..`.
            let at = |p: &[[__m512i; 8]; 3], uv: usize| {
                F192x8(p.each_ref().map(|plane| plane[uv + 4 * g]))
            };
            let [a0, a1, a2, a3] = [0, 1, 2, 3].map(|uv| e.mul(at(&pa, uv)));
            let [b0, b1, b2, b3] = [0, 1, 2, 3].map(|uv| at(&pb, uv));
            let [c1, c2, c3] = [1, 2, 3].map(|uv| at(&pc, uv));

            // The leading coefficients along `u` and along `v`.
            let (du0, du1, dv0, dv1) = (a0.add(a1), a2.add(a3), a0.add(a2), a1.add(a3));
            let (eu0, eu1, ev0, ev1) = (b0.add(b1), b2.add(b3), b0.add(b2), b1.add(b3));

            // The eight sums, in the slots the pair is read from.
            acc[0].mul_add(a1, b1);
            acc[0].mul_add(e, c1);
            acc[1].mul_add(a3, b3);
            acc[1].mul_add(e, c3);
            acc[2].mul_add(du0, eu0);
            acc[3].mul_add(du1, eu1);
            acc[4].mul_add(a2, b2);
            acc[4].mul_add(e, c2);
            acc[5].mul_add(dv0, ev0);
            acc[6].mul_add(dv1, ev1);
            acc[7].mul_add(du0.add(du1), eu0.add(eu1));
        }
    }
    acc.map(F192x8Sum::total)
}
