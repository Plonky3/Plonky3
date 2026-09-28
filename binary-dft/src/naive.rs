//! Reference additive NTT, evaluated straight from the definition.

use core::marker::PhantomData;

use p3_binary_field::TowerLevel;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;

use crate::domain::domain_point;
use crate::traits::AdditiveNtt;

/// The reference additive NTT, evaluated straight from the definition.
///
/// - The forward transform evaluates `sum_i d_i * X_i(x)` point by point.
/// - Each subspace polynomial comes from its product definition, `prod_(s in S_j) (x - s)`, normalised at `v_j`.
/// - It relies on none of the recurrences the fast transforms use, which makes it their oracle.
/// - The cost is `O(n^2 log n)`, so tests keep it at or below `2^10` rows.
#[derive(Clone, Debug, Default)]
pub struct NaiveAdditiveNtt<F> {
    _marker: PhantomData<F>,
}

/// The subspace polynomial of `S_j` from its product definition, normalised to one at `v_j`.
fn normalised_subspace_poly<F: TowerLevel>(j: usize, x: F) -> F {
    let vanishing = |y: F| (0..1 << j).map(|m| y + domain_point::<F>(m)).product::<F>();
    vanishing(x)
        * vanishing(F::cantor_basis(j))
            .try_inverse()
            .expect("W_j does not vanish at v_j, which lies outside S_j")
}

impl<F: TowerLevel> AdditiveNtt<F> for NaiveAdditiveNtt<F> {
    fn shifted_ntt_batch(&self, mat: RowMajorMatrix<F>, shift: F) -> RowMajorMatrix<F> {
        let width = mat.width();
        let height = mat.height();
        let log_n = log2_strict_usize(height);

        let mut out = F::zero_vec(width * height);
        for row in 0..height {
            let x = shift + domain_point::<F>(row);
            // X_i(x) = prod_j W_j(x)^(bit j of i), built by doubling the filled prefix.
            let mut basis = F::zero_vec(height);
            basis[0] = F::ONE;
            for j in 0..log_n {
                let w = normalised_subspace_poly::<F>(j, x);
                for i in 0..1 << j {
                    basis[i + (1 << j)] = basis[i] * w;
                }
            }
            for col in 0..width {
                out[row * width + col] = (0..height)
                    .map(|i| basis[i] * mat.values[i * width + col])
                    .sum();
            }
        }
        RowMajorMatrix::new(out, width)
    }

    fn shifted_intt_batch(&self, mut mat: RowMajorMatrix<F>, shift: F) -> RowMajorMatrix<F> {
        let width = mat.width();
        let log_n = log2_strict_usize(mat.height());

        // Inverse LCH butterflies, with every twiddle taken from the product definition.
        for j in 0..log_n {
            let half = 1 << j;
            for (block, base) in (0..mat.height()).step_by(half << 1).enumerate() {
                let t =
                    normalised_subspace_poly::<F>(j, shift + domain_point::<F>(block << (j + 1)));
                for k in 0..half {
                    for col in 0..width {
                        let u = mat.values[(base + k) * width + col];
                        let v = mat.values[(base + k + half) * width + col];
                        let hi = u + v;
                        mat.values[(base + k) * width + col] = u + t * hi;
                        mat.values[(base + k + half) * width + col] = hi;
                    }
                }
            }
        }
        mat
    }
}

#[cfg(test)]
mod tests {
    use p3_binary_field::{BinaryField8, BinaryField16, TowerLevel};
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::NaiveAdditiveNtt;
    use crate::traits::AdditiveNtt;
    use crate::{domain_point, subspace_polynomial};

    /// Transforming the `k`-th novel-basis unit vector gives `X_k` evaluated on the domain.
    #[test]
    fn unit_vectors_evaluate_the_novel_basis() {
        const LOG_N: usize = 3;
        let ntt = NaiveAdditiveNtt::<BinaryField8>::default();

        for k in 0..1 << LOG_N {
            let mut coeffs = BinaryField8::zero_vec(1 << LOG_N);
            coeffs[k] = BinaryField8::ONE;
            let evals = ntt.ntt_batch(RowMajorMatrix::new_col(coeffs));

            for m in 0..1 << LOG_N {
                // X_k(x) = prod_j W_j(x)^(bit j of k), with each W_j normalised to one at v_j.
                let x = domain_point::<BinaryField8>(m);
                let expected = (0..LOG_N)
                    .filter(|j| k >> j & 1 == 1)
                    .map(|j| subspace_polynomial::<BinaryField8>(j, x))
                    .product::<BinaryField8>();
                assert_eq!(evals.values[m], expected, "k={k} m={m}");
            }
        }
    }

    /// The inverse transform undoes the forward transform, on a coset too.
    #[test]
    fn naive_round_trips() {
        let mut rng = SmallRng::seed_from_u64(1);
        let ntt = NaiveAdditiveNtt::<BinaryField16>::default();
        let coeffs = RowMajorMatrix::<BinaryField16>::rand(&mut rng, 1 << 6, 3);
        let shift = BinaryField16::from_repr(0x1234);

        let evals = ntt.shifted_ntt_batch(coeffs.clone(), shift);
        assert_eq!(ntt.shifted_intt_batch(evals, shift), coeffs);
    }

    /// The low-degree extension reproduces the input rows as its prefix.
    #[test]
    fn lde_extends_the_input() {
        let mut rng = SmallRng::seed_from_u64(2);
        let ntt = NaiveAdditiveNtt::<BinaryField16>::default();
        let evals = RowMajorMatrix::<BinaryField16>::rand(&mut rng, 1 << 5, 2);

        let extended = ntt.lde_batch(evals.clone(), 2);
        assert_eq!(extended.height(), evals.height() * 4);
        assert_eq!(&extended.values[..evals.values.len()], &evals.values[..]);
    }

    /// The Cantor basis makes the normalisation trivial. Assert that rather than believing it:
    /// The reference divides by `prod_(m in S_j) (v_j - m)`, and this pins that divisor to one.
    /// That holds independently of the squaring recurrence the fast transforms rely on.
    #[test]
    fn the_normalisation_divisor_is_one() {
        for j in 0..8 {
            let divisor = (0..1usize << j)
                .map(|m| BinaryField8::cantor_basis(j) + domain_point::<BinaryField8>(m))
                .product::<BinaryField8>();
            assert_eq!(divisor, BinaryField8::ONE, "j={j}");
        }
    }
}
