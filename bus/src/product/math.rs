//! Multilinear arithmetic shared by product-tree proving and verification.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::Field;

/// Folds a fully materialized multilinear table in place.
pub(super) fn fold_dense<F: Field>(values: &mut Vec<F>, challenge: F) {
    let output_len = values.len() / 2;
    for row in 0..output_len {
        values[row] = interpolate_pair([values[2 * row], values[2 * row + 1]], challenge);
    }
    values.truncate(output_len);
}

/// Materializes equality weights over a low-variable-first Boolean cube.
pub(super) fn equality_weights<F: Field>(point: &[F]) -> Vec<F> {
    let mut weights = vec![F::ONE];
    for &coordinate in point {
        let old_len = weights.len();
        weights.resize(old_len * 2, F::ZERO);
        for index in 0..old_len {
            let weight = weights[index];
            weights[index] = weight * (F::ONE - coordinate);
            weights[old_len + index] = weight * coordinate;
        }
    }
    weights
}

/// Evaluates the multilinear equality polynomial at two points.
pub(super) fn equality_evaluation<F: Field>(left: &[F], right: &[F]) -> F {
    debug_assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(&left, &right)| (F::ONE - left) * (F::ONE - right) + left * right)
        .product()
}

/// Combines one claim per tree with consecutive powers of one challenge.
pub(super) fn combine<F: Field>(values: &[F], challenge: F) -> F {
    values
        .iter()
        .zip(challenge.powers())
        .map(|(&value, power)| value * power)
        .sum()
}

/// Interpolates a line whose endpoints are indexed by one Boolean variable.
pub(super) fn interpolate_pair<F: Field>(values: [F; 2], point: F) -> F {
    values[0] + point * (values[1] - values[0])
}

/// Interpolates a four-entry table at two low-order coordinates.
pub(super) fn interpolate_quad<F: Field>(values: [F; 4], point: [F; 2]) -> F {
    let low_zero = interpolate_pair([values[0], values[1]], point[0]);
    let low_one = interpolate_pair([values[2], values[3]], point[0]);
    interpolate_pair([low_zero, low_one], point[1])
}

/// Checks that degree-five interpolation has a valid six-point domain.
pub(super) fn has_distinct_round_nodes<F: Field>() -> bool {
    // Six distinct nodes cannot exist in a field with fewer than eight elements.
    // This guard avoids constructing interpolation nodes outside their domain.
    if F::bits() < 3 {
        return false;
    }

    // Pairwise comparison avoids allocating at the proof boundary.
    let nodes = core::array::from_fn::<_, 6, _>(F::interpolation_node);
    nodes
        .iter()
        .enumerate()
        .all(|(index, node)| !nodes[index + 1..].contains(node))
}
