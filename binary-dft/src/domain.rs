//! The additive NTT domain: `F_2`-linear subspaces spanned by the Cantor basis.

use alloc::vec::Vec;

use p3_binary_field::TowerLevel;

/// The `index`-th point of the additive NTT domain, `Σ_r bit_r(index) · v_r`.
///
/// The domain `S_ℓ` is the span of the first `ℓ` Cantor basis vectors, so this is well
/// defined for any `ℓ > log2(index)` and does not depend on which level evaluates it.
///
/// # Panics
/// Panics if `index` has a set bit at or above the bit width of `F`.
#[must_use]
pub fn domain_point<F: TowerLevel>(index: usize) -> F {
    let mut point = F::ZERO;
    let mut remaining = index;
    let mut r = 0;
    while remaining != 0 {
        if remaining & 1 == 1 {
            point += F::cantor_basis(r);
        }
        remaining >>= 1;
        r += 1;
    }
    point
}

/// The increments of `domain_point(index << 1)` as `index` runs upwards from zero.
///
/// `domain_point` is `F_2`-linear in its index and `(index − 1) ^ index` is the mask of bits
/// `0..=k` for `k = index.trailing_zeros()`, so consecutive points differ by
/// `steps[k] = Σ_{r ≤ k} v_{r+1}`. The `count` entries returned cover every `index` below
/// `2^count`.
///
/// # Panics
/// Panics if `count` is at least the bit width of `F`.
#[must_use]
pub(crate) fn domain_point_steps<F: TowerLevel>(count: usize) -> Vec<F> {
    let mut point = F::ZERO;
    (0..count)
        .map(|r| {
            point += F::cantor_basis(r + 1);
            point
        })
        .collect()
}

/// The subspace polynomial `W_j` of `S_j`, defined by `W_0(x) = x` and
/// `W_j(x) = W_{j−1}(x)² + W_{j−1}(x)`.
///
/// `W_j` is `F_2`-linear, vanishes exactly on `S_j`, and satisfies `W_j(v_j) = v_0 = 1`, so it
/// is already the normalised subspace polynomial `Ŵ_j` for the Cantor basis (D8).
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

    use super::{domain_point, subspace_polynomial};

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

    /// D8's index-shift identity: `W_j(v_i) = v_{i−j}` for `i ≥ j`.
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
