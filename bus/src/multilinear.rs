//! Multilinear equality helpers for public most-significant-variable-first points.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::Field;

/// Materialize equality weights in lexicographic Boolean-vertex order.
///
/// Coordinate zero binds the most significant bit of each output index.
pub(crate) fn equality_weights_msb<F: Field>(point: &[F]) -> Vec<F> {
    // The empty point addresses the sole vertex of a zero-variable cube.
    let mut weights = vec![F::ONE];

    for &coordinate in point {
        // Each coordinate splits every existing prefix into its zero and one children.
        let mut next = Vec::with_capacity(weights.len() * 2);
        for &weight in &weights {
            next.push(weight * (F::ONE - coordinate));
            next.push(weight * coordinate);
        }
        weights = next;
    }
    weights
}

/// Evaluate one lexicographic Boolean-vertex selector.
///
/// Coordinate zero binds the most significant bit of the vertex index.
pub(crate) fn equality_at_msb_vertex<F: Field>(point: &[F], vertex: usize) -> Option<F> {
    // Reject indices outside the Boolean cube before computing any bit shift.
    let shift = u32::try_from(point.len()).ok()?;
    let capacity = 1usize.checked_shl(shift)?;
    if vertex >= capacity {
        return None;
    }

    // Tensor the one-coordinate equality selectors in public coordinate order.
    Some(
        point
            .iter()
            .enumerate()
            .map(|(coordinate, &challenge)| {
                let shift = point.len() - 1 - coordinate;
                if (vertex >> shift) & 1 == 0 {
                    F::ONE - challenge
                } else {
                    challenge
                }
            })
            .product(),
    )
}
