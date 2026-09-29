//! The additive NTT domain: linear subspaces spanned by the Cantor basis.

use alloc::vec::Vec;

use p3_binary_field::TowerLevel;

/// The point of the domain at a given index, `sum_r (bit r of index) * v_r`.
///
/// - The subspace `S_l` is the span of the first `l` Cantor basis vectors `v_0, v_1, ...`.
/// - So the point is the same for every `l` above the index's bit length.
/// - It is also the same whichever tower level evaluates it.
///
/// # Panics
///
/// Panics if the index has a set bit at or above the bit width of the level.
#[must_use]
pub fn domain_point<F: TowerLevel>(index: usize) -> F {
    let mut point = F::ZERO;
    let mut remaining = index;
    let mut r = 0;

    // Add the basis vector of every set bit, lowest first.
    while remaining != 0 {
        if remaining & 1 == 1 {
            point += F::cantor_basis(r);
        }
        remaining >>= 1;
        r += 1;
    }
    point
}

/// The steps between consecutive even domain points, one per trailing-zero count.
///
/// - The domain point is linear in its index.
/// - Going from index `2(b - 1)` to `2b` flips the bits `1 ..= k + 1`, where `k` counts the trailing zeros of `b`.
/// - So the step is `sum_(r <= k) v_(r + 1)`, the entry `k` of the returned table.
/// - The `count` entries cover every `b` below `2^count`.
///
/// # Panics
///
/// Panics if `count` is at least the bit width of the level.
#[must_use]
pub fn domain_point_steps<F: TowerLevel>(count: usize) -> Vec<F> {
    // Each entry is the previous one plus the next basis vector up.
    let mut point = F::ZERO;
    (0..count)
        .map(|r| {
            point += F::cantor_basis(r + 1);
            point
        })
        .collect()
}

/// The subspace polynomial `W_j` of `S_j`, evaluated at `x`.
///
/// - The recurrence is `W_0(x) = x` and `W_j(x) = W_(j-1)(x)^2 + W_(j-1)(x)`.
/// - `W_j` is linear and vanishes exactly on `S_j`.
/// - For the Cantor basis `W_j(v_j) = 1`, so it is already normalised.
#[must_use]
pub fn subspace_polynomial<F: TowerLevel>(j: usize, x: F) -> F {
    let mut value = x;
    for _ in 0..j {
        value = value.square() + value;
    }
    value
}

#[cfg(test)]
mod tests {
    use p3_binary_field::{BinaryField8, BinaryField16, BinaryField128, TowerLevel};
    use p3_field::PrimeCharacteristicRing;

    use super::{domain_point, domain_point_steps, subspace_polynomial};

    /// Chaining the increments of a stage reproduces the points a full walk reaches.
    fn check_steps_chain_to_the_walk<F: TowerLevel>(count: usize) {
        let steps = domain_point_steps::<F>(count);
        let mut point = F::ZERO;
        assert_eq!(point, domain_point::<F>(0));
        for index in 1..1usize << count {
            point += steps[index.trailing_zeros() as usize];
            assert_eq!(point, domain_point::<F>(index << 1), "index={index}");
        }
    }

    /// The image of every single-bit index is the matching Cantor basis vector, which pins the
    /// basis itself and not merely some graded `F_2`-linear reparametrisation of it.
    #[test]
    fn domain_point_images_are_the_cantor_basis() {
        for r in 0..8 {
            assert_eq!(
                domain_point::<BinaryField8>(1 << r),
                BinaryField8::cantor_basis(r),
                "r={r}"
            );
        }
        for r in 0..usize::BITS as usize {
            assert_eq!(
                domain_point::<BinaryField128>(1 << r),
                BinaryField128::cantor_basis(r),
                "r={r}"
            );
        }
    }

    /// An index reaching past the basis of its level is rejected rather than silently masked.
    #[test]
    #[should_panic = "Cantor basis index out of range"]
    fn domain_point_rejects_an_index_past_the_basis() {
        let _ = domain_point::<BinaryField8>(1 << 8);
    }

    /// `domain_point` is `F_2`-linear in its index.
    #[test]
    fn domain_point_is_linear_in_the_index() {
        for a in 0..1usize << 8 {
            for b in 0..1usize << 8 {
                assert_eq!(
                    domain_point::<BinaryField8>(a ^ b),
                    domain_point::<BinaryField8>(a) + domain_point::<BinaryField8>(b)
                );
            }
        }
    }

    /// The increments are what a transform adds to a twiddle in place of walking the block
    /// index, so they have to agree with that walk at every index, not merely at the ends.
    #[test]
    fn domain_point_steps_chain_to_the_even_domain_points() {
        check_steps_chain_to_the_walk::<BinaryField8>(7);
        check_steps_chain_to_the_walk::<BinaryField16>(10);
        check_steps_chain_to_the_walk::<BinaryField128>(12);
    }

    /// `W_j` vanishes exactly on `S_j`, the span of the first `j` basis vectors.
    #[test]
    fn subspace_polynomial_vanishes_on_its_subspace() {
        for j in 0..=8 {
            for m in 0..1usize << 8 {
                let value = subspace_polynomial::<BinaryField8>(j, domain_point(m));
                assert_eq!(value == BinaryField8::ZERO, m < 1 << j, "j={j} m={m}");
            }
        }
    }

    /// The index-shift identity: `W_j(v_i) = v_(i - j)` for `i >= j`.
    #[test]
    fn subspace_polynomial_shifts_the_basis() {
        for j in 0..16 {
            for i in j..16 {
                assert_eq!(
                    subspace_polynomial::<BinaryField16>(j, BinaryField16::cantor_basis(i)),
                    BinaryField16::cantor_basis(i - j),
                    "j={j} i={i}"
                );
            }
        }
    }

    /// A point of the domain does not depend on the level it is computed at.
    #[test]
    fn domain_points_agree_across_levels() {
        for m in 0..1usize << 8 {
            let small: u128 = domain_point::<BinaryField8>(m).to_repr().into();
            let large: u128 = domain_point::<BinaryField128>(m).to_repr();
            assert_eq!(small, large, "index {m}");
        }
    }
}
