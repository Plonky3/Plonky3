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

/// Evaluation of `eq(left, right)` for two equally sized field points.
///
/// Returns no value for points of different length.
pub(super) fn equality_evaluation<F: Field>(left: &[F], right: &[F]) -> Option<F> {
    // Truncating to the shorter point would return a well-formed but wrong weight.
    if left.len() != right.len() {
        return None;
    }
    Some(
        left.iter()
            .zip(right)
            .map(|(&left, &right)| left * right + (F::ONE - left) * (F::ONE - right))
            .product(),
    )
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    type F = BabyBear;

    #[test]
    fn mismatched_points_yield_no_equality_weight() {
        // A prefix product is well formed and wrong, so no value may come back at all.
        let left = [F::from_u8(2), F::from_u8(3)];
        assert_eq!(equality_evaluation(&left, &left[..1]), None);
        assert_eq!(equality_evaluation(&left[..1], &left), None);
        assert_eq!(equality_evaluation::<F>(&[], &left), None);
    }

    #[test]
    fn equal_points_agree_with_the_hand_written_product() {
        // Each coordinate contributes a * b + (1 - a) * (1 - b).
        let left = [F::from_u8(2), F::from_u8(3)];
        let right = [F::from_u8(5), F::from_u8(7)];
        let expected = (left[0] * right[0] + (F::ONE - left[0]) * (F::ONE - right[0]))
            * (left[1] * right[1] + (F::ONE - left[1]) * (F::ONE - right[1]));

        assert_eq!(equality_evaluation(&left, &right), Some(expected));
        assert_eq!(equality_evaluation::<F>(&[], &[]), Some(F::ONE));
    }

    #[test]
    fn equality_weights_tabulate_the_evaluation_at_every_vertex() {
        // Coordinate zero selects the half, so the table order is 00, 01, 10, 11.
        let a = F::from_u8(2);
        let b = F::from_u8(3);

        assert_eq!(
            equality_weights(&[a, b]),
            alloc::vec![
                (F::ONE - a) * (F::ONE - b),
                (F::ONE - a) * b,
                a * (F::ONE - b),
                a * b,
            ]
        );
    }
}
