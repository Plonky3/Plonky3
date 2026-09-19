//! Multilinear-equality kernels in trace-index order.

use alloc::vec::Vec;

use p3_field::Field;

/// Evaluations of `eq(point, ·)` in most-significant-coordinate-first table order.
pub(super) fn equality_weights<F: Field>(point: &[F]) -> Vec<F> {
    let mut weights = alloc::vec![F::ONE];
    for &coordinate in point {
        let mut next = Vec::with_capacity(weights.len() * 2);
        for &weight in &weights {
            next.push(weight * (F::ONE - coordinate));
            next.push(weight * coordinate);
        }
        weights = next;
    }
    weights
}

/// Evaluation of `eq(point, vertex)` for a table-ordered Boolean vertex.
pub(super) fn equality_at_vertex<F: Field>(point: &[F], vertex: usize) -> F {
    point
        .iter()
        .enumerate()
        .map(|(coordinate, &challenge)| {
            let bit = (vertex >> (point.len() - 1 - coordinate)) & 1;
            if bit == 0 {
                F::ONE - challenge
            } else {
                challenge
            }
        })
        .product()
}

/// Evaluation of `eq(left, right)` for two equally sized field points.
pub(super) fn equality_evaluation<F: Field>(left: &[F], right: &[F]) -> F {
    debug_assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(&left, &right)| left * right + (F::ONE - left) * (F::ONE - right))
        .product()
}
