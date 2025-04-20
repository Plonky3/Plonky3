use alloc::vec;

use p3_field::TwoAdicField;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;

use crate::TwoAdicSubgroupDft;

#[derive(Default, Clone, Debug)]
pub struct NaiveDft;

impl<F: TwoAdicField> TwoAdicSubgroupDft<F> for NaiveDft {
    type Evaluations = RowMajorMatrix<F>;
    fn dft_batch(&self, mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        let w = mat.width();
        let h = mat.height();
        let log_h = log2_strict_usize(h);
        let g = F::two_adic_generator(log_h);

        let mut res = RowMajorMatrix::new(vec![F::ZERO; w * h], w);
        for (res_r, point) in g.powers().take(h).enumerate() {
            for (src_r, point_power) in point.powers().take(h).enumerate() {
                for c in 0..w {
                    res.values[res_r * w + c] += point_power * mat.values[src_r * w + c]
                }
            }
        }

        res
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};
    use p3_goldilocks::Goldilocks;
    use p3_matrix::dense::RowMajorMatrix;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use crate::{NaiveDft, TwoAdicSubgroupDft};

    #[test]
    fn basic() {
        type F = BabyBear;

        // A few polynomials:
        // 5 + 4x
        // 2 + 3x
        // 0
        let mat = RowMajorMatrix::new(
            vec![
                F::from_u8(5),
                F::from_u8(2),
                F::ZERO,
                F::from_u8(4),
                F::from_u8(3),
                F::ZERO,
            ],
            3,
        );

        let dft = NaiveDft.dft_batch(mat);
        // Expected evaluations on {1, -1}:
        // 9, 1
        // 5, -1
        // 0, 0
        assert_eq!(
            dft,
            RowMajorMatrix::new(
                vec![
                    F::from_u8(9),
                    F::from_u8(5),
                    F::ZERO,
                    F::ONE,
                    F::NEG_ONE,
                    F::ZERO,
                ],
                3,
            )
        )
    }

    #[test]
    fn dft_idft_consistency() {
        type F = Goldilocks;
        let mut rng = SmallRng::seed_from_u64(1);
        let original = RowMajorMatrix::<F>::rand(&mut rng, 8, 3);
        let dft = NaiveDft.dft_batch(original.clone());
        let idft = NaiveDft.idft_batch(dft);
        assert_eq!(original, idft);
    }

    #[test]
    fn coset_dft_idft_consistency() {
        type F = Goldilocks;
        let generator = F::GENERATOR;
        let mut rng = SmallRng::seed_from_u64(1);
        let original = RowMajorMatrix::<F>::rand(&mut rng, 8, 3);
        let dft = NaiveDft.coset_dft_batch(original.clone(), generator);
        let idft = NaiveDft.coset_idft_batch(dft, generator);
        assert_eq!(original, idft);
    }

    #[test]
    fn test_basic_dft_batch() {
        type F = BabyBear;

        // We are evaluating two polynomials, each with 2 coefficients:
        //
        // ```text
        //   f_0(x) = 1 + 2x
        //   f_1(x) = 3 + 4x
        // ```
        //
        // DFT is done over the 2-point subgroup H = { ω^0, ω^1 } = {1, -1}
        // with generator ω = -1.
        //
        // The coefficient matrix is:
        //
        // ```text
        //   [ c_{00}  c_{10} ]
        //   [ c_{01}  c_{11} ]
        // ```
        //
        // where:
        //   c_{00} = 1, c_{01} = 2,
        //   c_{10} = 3, c_{11} = 4.
        //
        // The DFT matrix (Vandermonde form) is:
        //
        // ```text
        //     [ ω^{0·0}   ω^{0·1} ]
        //     [ ω^{1·0}   ω^{1·1} ]
        // ```
        //
        // Full DFT matrix multiplication:
        //
        // ```text
        //     [ ω^{0·0}   ω^{0·1} ]   ×   [ c_{00}   c_{10} ]
        //     [ ω^{1·0}   ω^{1·1} ]       [ c_{01}   c_{11} ]
        //
        //   = [ f_0(ω^0)   f_1(ω^0) ]
        //     [ f_0(ω^1)   f_1(ω^1) ]
        // ```
        //
        // Manual computation of evaluations:
        //
        // ```text
        //   f_0(ω^0) = c_{00} + c_{01} · ω^0
        //   f_0(ω^1) = c_{00} + c_{01} · ω^1
        //
        //   f_1(ω^0) = c_{10} + c_{11} · ω^0
        //   f_1(ω^1) = c_{10} + c_{11} · ω^1
        // ```

        let c00 = F::from_u8(1);
        let c01 = F::from_u8(2);
        let c10 = F::from_u8(3);
        let c11 = F::from_u8(4);

        let mat = RowMajorMatrix::new(vec![c00, c10, c01, c11], 2);

        let omega = F::two_adic_generator(1); // ω = -1

        let eval00 = c00 + c01 * omega.exp_u64(0); // f_0(ω^0)
        let eval01 = c00 + c01 * omega.exp_u64(1); // f_0(ω^1)

        let eval10 = c10 + c11 * omega.exp_u64(0); // f_1(ω^0)
        let eval11 = c10 + c11 * omega.exp_u64(1); // f_1(ω^1)

        let expected = RowMajorMatrix::new(
            vec![
                eval00, eval10, // Row for ω^0
                eval01, eval11, // Row for ω^1
            ],
            2,
        );

        let dft = NaiveDft.dft_batch(mat);
        assert_eq!(dft, expected);
    }

    #[test]
    fn test_naive_dft_batch_size_4() {
        type F = BabyBear;

        // We evaluate two polynomials, each of degree 3:
        //
        // ```text
        //   f_0(x) = 1 + 2x + 3x^2 + 4x^3
        //   f_1(x) = 5 + 6x + 7x^2 + 8x^3
        // ```
        //
        // Coefficient matrix (column-wise layout):
        //
        // ```text
        //   [ c_{00}  c_{10} ]   ← constant terms
        //   [ c_{01}  c_{11} ]   ← x^1
        //   [ c_{02}  c_{12} ]   ← x^2
        //   [ c_{03}  c_{13} ]   ← x^3
        // ```

        let c00 = F::from_u8(1);
        let c01 = F::from_u8(2);
        let c02 = F::from_u8(3);
        let c03 = F::from_u8(4);

        let c10 = F::from_u8(5);
        let c11 = F::from_u8(6);
        let c12 = F::from_u8(7);
        let c13 = F::from_u8(8);

        let mat = RowMajorMatrix::new(
            vec![
                c00, c10, // [ c_{00}  c_{10} ]
                c01, c11, // [ c_{01}  c_{11} ]
                c02, c12, // [ c_{02}  c_{12} ]
                c03, c13, // [ c_{03}  c_{13} ]
            ],
            2,
        );

        let omega = F::two_adic_generator(2); // ω = primitive 4th root of unity

        // DFT domain points:
        //
        // ```text
        //   [ ω^0, ω^1, ω^2, ω^3 ]
        // ```
        //
        // The DFT matrix (Vandermonde-style, 4x4):
        //
        // ```text
        //   [ ω^{0·0}  ω^{0·1}  ω^{0·2}  ω^{0·3} ] = [ 1     1     1     1     ]
        //   [ ω^{1·0}  ω^{1·1}  ω^{1·2}  ω^{1·3} ] = [ 1     ω     ω^2   ω^3   ]
        //   [ ω^{2·0}  ω^{2·1}  ω^{2·2}  ω^{2·3} ] = [ 1     ω^2   ω^4   ω^6   ]
        //   [ ω^{3·0}  ω^{3·1}  ω^{3·2}  ω^{3·3} ] = [ 1     ω^3   ω^6   ω^9   ]
        // ```
        //
        // DFT computes:
        //
        // ```text
        //   f_j(ω^i) = c_{0j} + c_{1j}·ω^i + c_{2j}·ω^{2i} + c_{3j}·ω^{3i}
        // ```

        let omega1 = omega;
        let omega2 = omega * omega;
        let omega3 = omega * omega2;
        let omega4 = omega2 * omega2;
        let omega6 = omega2 * omega4;
        let omega9 = omega3 * omega6;

        // Polynomial f_0(x) = 1 + 2x + 3x^2 + 4x^3
        let f00 = c00 + c01 + c02 + c03; // ω^0
        let f01 = c00 + c01 * omega1 + c02 * omega2 + c03 * omega3; // ω^1
        let f02 = c00 + c01 * omega2 + c02 * omega4 + c03 * omega6; // ω^2
        let f03 = c00 + c01 * omega3 + c02 * omega6 + c03 * omega9; // ω^3

        // Polynomial f_1(x) = 5 + 6x + 7x^2 + 8x^3
        let f10 = c10 + c11 + c12 + c13;
        let f11 = c10 + c11 * omega1 + c12 * omega2 + c13 * omega3;
        let f12 = c10 + c11 * omega2 + c12 * omega4 + c13 * omega6;
        let f13 = c10 + c11 * omega3 + c12 * omega6 + c13 * omega9;

        let expected = RowMajorMatrix::new(
            vec![
                f00, f10, // Row for ω^0
                f01, f11, // Row for ω^1
                f02, f12, // Row for ω^2
                f03, f13, // Row for ω^3
            ],
            2,
        );

        let dft = NaiveDft.dft_batch(mat);
        assert_eq!(dft, expected);
    }

    #[test]
    fn test_basic_coset_dft_batch() {
        type F = BabyBear;

        // We evaluate two polynomials of degree 1:
        //
        // ```text
        //   f_0(x) = 1 + 2x
        //   f_1(x) = 3 + 4x
        // ```
        //
        // Coefficients laid out column-wise:
        //
        // ```text
        //   [ c_{00}  c_{10} ]
        //   [ c_{01}  c_{11} ]
        // ```

        let c00 = F::from_u8(1);
        let c01 = F::from_u8(2);
        let c10 = F::from_u8(3);
        let c11 = F::from_u8(4);

        let mat = RowMajorMatrix::new(
            vec![
                c00, c10, // [ c_{00}  c_{10} ]
                c01, c11, // [ c_{01}  c_{11} ]
            ],
            2,
        );

        let omega = F::two_adic_generator(1); // ω = -1
        let shift = F::from_u8(5); // arbitrary coset shift

        // Coset DFT is evaluated at domain:
        //
        // ```text
        //   [ ω^0 · shift, ω^1 · shift ] = [ shift, -shift ]
        // ```
        //
        // DFT matrix for 2-point subgroup:
        //
        // ```text
        //   [ ω^{0·0}  ω^{0·1} ]
        //   [ ω^{1·0}  ω^{1·1} ]
        // ```
        //
        // Before applying the matrix, we shift coefficients as:
        //
        // ```text
        //   c_{01} ← c_{01} · shift
        //   c_{11} ← c_{11} · shift
        // ```
        //
        // Shifted coefficient matrix:
        //
        // ```text
        //   [ c_{00}             c_{10}           ]
        //   [ c_{01} · shift     c_{11} · shift   ]
        // ```
        //
        // Full matrix multiplication:
        //
        // ```text
        //   [ f_0(shift)    f_1(shift)    ]
        //   [ f_0(-shift)   f_1(-shift)   ]
        //
        //   = [ ω^{0·0}  ω^{0·1} ] × [ c_{00}             c_{10}         ]
        //     [ ω^{1·0}  ω^{1·1} ]   [ c_{01} · shift     c_{11} · shift ]
        // ```
        //
        // Evaluations:
        //
        // ```text
        //   f_0( shift) = c_{00} + c_{01} · shift · ω^0
        //   f_0(-shift) = c_{00} + c_{01} · shift · ω^1
        //
        //   f_1( shift) = c_{10} + c_{11} · shift · ω^0
        //   f_1(-shift) = c_{10} + c_{11} · shift · ω^1
        // ```

        let c01_shifted = c01 * shift;
        let c11_shifted = c11 * shift;

        let eval00 = c00 + c01_shifted * omega.exp_u64(0); // f_0(shift)
        let eval01 = c00 + c01_shifted * omega.exp_u64(1); // f_0(-shift)

        let eval10 = c10 + c11_shifted * omega.exp_u64(0); // f_1(shift)
        let eval11 = c10 + c11_shifted * omega.exp_u64(1); // f_1(-shift)

        let expected = RowMajorMatrix::new(
            vec![
                eval00, eval10, // row for shift · ω^0
                eval01, eval11, // row for shift · ω^1
            ],
            2,
        );

        let coset_dft = NaiveDft.coset_dft_batch(mat, shift);
        assert_eq!(coset_dft, expected);
    }

    #[test]
    fn test_basic_coset_dft_batch_size_4() {
        type F = BabyBear;

        // We evaluate two polynomials of degree 3:
        //
        // ```text
        //   f_0(x) = 1 + 2x + 3x^2 + 4x^3
        //   f_1(x) = 5 + 6x + 7x^2 + 8x^3
        // ```
        //
        // Coefficients laid out column-wise:
        //
        // ```text
        //   [ c_{00}  c_{10} ]
        //   [ c_{01}  c_{11} ]
        //   [ c_{02}  c_{12} ]
        //   [ c_{03}  c_{13} ]
        // ```

        let c00 = F::from_u8(1);
        let c01 = F::from_u8(2);
        let c02 = F::from_u8(3);
        let c03 = F::from_u8(4);

        let c10 = F::from_u8(5);
        let c11 = F::from_u8(6);
        let c12 = F::from_u8(7);
        let c13 = F::from_u8(8);

        let mat = RowMajorMatrix::new(
            vec![
                c00, c10, // [ c_{00}  c_{10} ]
                c01, c11, // [ c_{01}  c_{11} ]
                c02, c12, // [ c_{02}  c_{12} ]
                c03, c13, // [ c_{03}  c_{13} ]
            ],
            2,
        );

        let omega = F::two_adic_generator(2); // ω = primitive 4th root of unity
        let shift = F::from_u8(5); // arbitrary coset shift

        // Coset DFT is evaluated at domain:
        //
        // ```text
        //   [ ω^0 · shift, ω^1 · shift, ω^2 · shift, ω^3 · shift ]
        // ```
        //
        // Vandermonde-style DFT matrix:
        //
        // ```text
        //   [ ω^{0·0}  ω^{0·1}  ω^{0·2}  ω^{0·3} ]
        //   [ ω^{1·0}  ω^{1·1}  ω^{1·2}  ω^{1·3} ]
        //   [ ω^{2·0}  ω^{2·1}  ω^{2·2}  ω^{2·3} ]
        //   [ ω^{3·0}  ω^{3·1}  ω^{3·2}  ω^{3·3} ]
        // ```
        //
        // Before applying this matrix, we shift each coefficient by powers of the coset shift:
        //
        // ```text
        //   c_{0j} ← c_{0j} · shift^j
        //   c_{1j} ← c_{1j} · shift^j
        // ```
        //
        // Resulting shifted coefficient matrix:
        //
        // ```text
        //   [ c_{00}             c_{10}           ]
        //   [ c_{01} · shift     c_{11} · shift   ]
        //   [ c_{02} · shift^2   c_{12} · shift^2 ]
        //   [ c_{03} · shift^3   c_{13} · shift^3 ]
        // ```
        //
        // Full matrix multiplication:
        //
        // ```text
        //   [ f_0(ω^0·s)  f_1(ω^0·s) ]
        //   [ f_0(ω^1·s)  f_1(ω^1·s) ]
        //   [ f_0(ω^2·s)  f_1(ω^2·s) ]
        //   [ f_0(ω^3·s)  f_1(ω^3·s) ]
        // ```

        let omega1 = omega;
        let omega2 = omega * omega;
        let omega3 = omega * omega2;
        let omega4 = omega2 * omega2;
        let omega6 = omega2 * omega4;
        let omega9 = omega3 * omega6;

        let s1 = shift;
        let s2 = shift * shift;
        let s3 = s2 * shift;

        let c01s = c01 * s1;
        let c02s = c02 * s2;
        let c03s = c03 * s3;

        let c11s = c11 * s1;
        let c12s = c12 * s2;
        let c13s = c13 * s3;

        // Manual evaluation using:
        //
        // ```text
        //   f_i(ω^k · s) = ∑_{j=0}^{3} c_{ij} · shift^j · ω^{k·j}
        // ```

        let f00 = c00 + c01s + c02s + c03s;
        let f01 = c00 + c01s * omega1 + c02s * omega2 + c03s * omega3;
        let f02 = c00 + c01s * omega2 + c02s * omega4 + c03s * omega6;
        let f03 = c00 + c01s * omega3 + c02s * omega6 + c03s * omega9;

        let f10 = c10 + c11s + c12s + c13s;
        let f11 = c10 + c11s * omega1 + c12s * omega2 + c13s * omega3;
        let f12 = c10 + c11s * omega2 + c12s * omega4 + c13s * omega6;
        let f13 = c10 + c11s * omega3 + c12s * omega6 + c13s * omega9;

        let expected = RowMajorMatrix::new(
            vec![
                f00, f10, // row for shift · ω^0
                f01, f11, // row for shift · ω^1
                f02, f12, // row for shift · ω^2
                f03, f13, // row for shift · ω^3
            ],
            2,
        );

        let coset_dft = NaiveDft.coset_dft_batch(mat, shift);
        assert_eq!(coset_dft, expected);
    }

    #[test]
    fn test_basic_idft_batch_dim2() {
        type F = BabyBear;

        // We start with evaluations of two polynomials at the 2-point
        // multiplicative subgroup H = {ω^0, ω^1} with ω = -1.
        //
        // Let the evaluation matrix be:
        //
        // ```text
        //   [ f_0(ω^0)   f_1(ω^0) ]   ← ω^0 =  1
        //   [ f_0(ω^1)   f_1(ω^1) ]   ← ω^1 = -1
        // ```
        //
        // Denote these evaluations as:
        //
        // ```text
        //   e_00 = f_0(ω^0) = c_00 + c_01 · ω^0
        //   e_01 = f_0(ω^1) = c_00 + c_01 · ω^1
        //
        //   e_10 = f_1(ω^0) = c_10 + c_11 · ω^0
        //   e_11 = f_1(ω^1) = c_10 + c_11 · ω^1
        // ```
        //
        // Our goal is to recover the coefficient matrix:
        //
        // ```text
        //   [ c_00  c_10 ]   ← constant terms
        //   [ c_01  c_11 ]   ← linear terms
        // ```
        //
        // Expected polynomials:
        //
        // ```text
        //   f_0(x) = 1 + 2x
        //   f_1(x) = 3 + 4x
        // ```

        let eval00 = F::from_u8(3); // f_0(ω^0) = e_00
        let eval01 = F::from_u8(7); // f_1(ω^0) = e_10
        let eval10 = F::from_i8(-1); // f_0(ω^1) = e_01
        let eval11 = F::from_i8(-1); // f_1(ω^1) = e_11

        let evals = RowMajorMatrix::new(
            vec![
                eval00, eval01, // [ e_00  e_10 ]
                eval10, eval11, // [ e_01  e_11 ]
            ],
            2,
        );

        // INVERSE DISCRETE FOURIER TRANSFORM (DFT^{-1})
        //
        // IDFT formula for coefficient c_i:
        //
        // ```text
        //   c_i = (1/n) ∑_{j=0}^{n-1} ω^{-i·j} · e_j
        // ```
        //
        // In our case, n = 2 and ω = -1, so:
        //
        // ```text
        //   IDFT matrix =
        //     [ ω^{-0·0}  ω^{-0·1} ]   = [ 1  1 ]
        //     [ ω^{-1·0}  ω^{-1·1} ]   = [ 1 -1 ]
        // ```
        //
        // Full matrix multiplication:
        //
        // ```text
        //   [ c_00  c_10 ]
        //   [ c_01  c_11 ]
        //
        // = 2^{-1} x
        //   [ 1  1 ]   ×   [ e_00  e_10 ]
        //   [ 1 -1 ]       [ e_01  e_11 ]
        // ```

        let omega = F::two_adic_generator(1); // ω = -1
        let omega_inv = omega.inverse();
        let inv2 = F::from_u8(2).inverse();

        // Compute coefficients for f_0(x)
        let c00 = (eval00 + eval10) * inv2; // constant term
        let c01 = (eval00 * omega_inv.exp_u64(0) + eval10 * omega_inv.exp_u64(1)) * inv2;

        // Compute coefficients for f_1(x)
        let c10 = (eval01 + eval11) * inv2; // constant term
        let c11 = (eval01 * omega_inv.exp_u64(0) + eval11 * omega_inv.exp_u64(1)) * inv2;

        let expected = RowMajorMatrix::new(
            vec![
                c00, c10, // [ c_00  c_10 ]
                c01, c11, // [ c_01  c_11 ]
            ],
            2,
        );

        let idft = NaiveDft.idft_batch(evals.clone());
        assert_eq!(idft, expected);

        // Round-trip: DFT(IDFT(evals)) = evals
        let roundtrip = NaiveDft.dft_batch(idft);
        assert_eq!(roundtrip, evals);
    }

    #[test]
    fn test_basic_idft_batch_dim4() {
        type F = BabyBear;

        // We evaluate two polynomials of degree 3 over the 4-point
        // subgroup H = {ω^0, ω^1, ω^2, ω^3} where ω is a primitive 4th root of unity.
        //
        // The evaluation matrix is:
        //
        // ```text
        //   [ f_0(ω^0)   f_1(ω^0) ]
        //   [ f_0(ω^1)   f_1(ω^1) ]
        //   [ f_0(ω^2)   f_1(ω^2) ]
        //   [ f_0(ω^3)   f_1(ω^3) ]
        // ```
        //
        // These evaluations correspond to:
        //
        // ```text
        //   e_00 = f_0(ω^0),  e_01 = f_0(ω^1),  e_02 = f_0(ω^2),  e_03 = f_0(ω^3)
        //   e_10 = f_1(ω^0),  e_11 = f_1(ω^1),  e_12 = f_1(ω^2),  e_13 = f_1(ω^3)
        // ```
        //
        // Goal: recover the coefficient matrix
        //
        // ```text
        //   [ c_00  c_10 ]
        //   [ c_01  c_11 ]
        //   [ c_02  c_12 ]
        //   [ c_03  c_13 ]
        // ```
        //
        // where:
        //   f_0(x) = c_00 + c_01·x + c_02·x^2 + c_03·x^3
        //   f_1(x) = c_10 + c_11·x + c_12·x^2 + c_13·x^3

        let eval00 = F::from_u8(10); // e_00 = f_0(ω^0)
        let eval01 = F::from_u8(0); // e_01 = f_0(ω^1)
        let eval02 = F::from_u8(4); // e_02 = f_0(ω^2)
        let eval03 = F::from_u8(8); // e_03 = f_0(ω^3)

        let eval10 = F::from_u8(26); // e_10 = f_1(ω^0)
        let eval11 = F::from_u8(4); // e_11 = f_1(ω^1)
        let eval12 = F::from_u8(18); // e_12 = f_1(ω^2)
        let eval13 = F::from_u8(4); // e_13 = f_1(ω^3)

        let evals = RowMajorMatrix::new(
            vec![
                eval00, eval10, // [ e_00  e_10 ]
                eval01, eval11, // [ e_01  e_11 ]
                eval02, eval12, // [ e_02  e_12 ]
                eval03, eval13, // [ e_03  e_13 ]
            ],
            2,
        );

        // INVERSE DISCRETE FOURIER TRANSFORM (DFT^{-1})
        //
        // IDFT formula:
        //
        // ```text
        //   c_i = (1/n) ∑_{j=0}^{n-1} ω^{-i·j} · e_j
        // ```
        //
        // For n = 4, the full inverse matrix is:
        //
        // ```text
        //   [ 1     1      1      1    ]  ← i = 0
        //   [ 1    ω^-1   ω^-2   ω^-3  ]  ← i = 1
        //   [ 1    ω^-2   ω^-4   ω^-6  ]  ← i = 2
        //   [ 1    ω^-3   ω^-6   ω^-9  ]  ← i = 3
        // ```
        //
        // Coefficient matrix is:
        //
        // ```text
        //   [ c_00  c_10 ]
        //   [ c_01  c_11 ]
        //   [ c_02  c_12 ]
        //   [ c_03  c_13 ]
        // = 4^{-1} × IDFT_Matrix × Evaluations
        // ```

        let omega = F::two_adic_generator(2); // ω, primitive 4th root
        let inv4 = F::from_u8(4).inverse();

        let omega_inv = omega.inverse();
        let omega2_inv = omega_inv * omega_inv;
        let omega3_inv = omega_inv * omega2_inv;

        // Coefficients for f_0(x)
        let c00 = (eval00 + eval01 + eval02 + eval03) * inv4;
        let c01 = (omega_inv.exp_u64(0) * eval00
            + omega_inv.exp_u64(1) * eval01
            + omega_inv.exp_u64(2) * eval02
            + omega_inv.exp_u64(3) * eval03)
            * inv4;
        let c02 = (omega2_inv.exp_u64(0) * eval00
            + omega2_inv.exp_u64(1) * eval01
            + omega2_inv.exp_u64(2) * eval02
            + omega2_inv.exp_u64(3) * eval03)
            * inv4;
        let c03 = (omega3_inv.exp_u64(0) * eval00
            + omega3_inv.exp_u64(1) * eval01
            + omega3_inv.exp_u64(2) * eval02
            + omega3_inv.exp_u64(3) * eval03)
            * inv4;

        // Coefficients for f_1(x)
        let c10 = (eval10 + eval11 + eval12 + eval13) * inv4;
        let c11 = (omega_inv.exp_u64(0) * eval10
            + omega_inv.exp_u64(1) * eval11
            + omega_inv.exp_u64(2) * eval12
            + omega_inv.exp_u64(3) * eval13)
            * inv4;
        let c12 = (omega2_inv.exp_u64(0) * eval10
            + omega2_inv.exp_u64(1) * eval11
            + omega2_inv.exp_u64(2) * eval12
            + omega2_inv.exp_u64(3) * eval13)
            * inv4;
        let c13 = (omega3_inv.exp_u64(0) * eval10
            + omega3_inv.exp_u64(1) * eval11
            + omega3_inv.exp_u64(2) * eval12
            + omega3_inv.exp_u64(3) * eval13)
            * inv4;

        let expected = RowMajorMatrix::new(
            vec![
                c00, c10, // [ c_00  c_10 ]
                c01, c11, // [ c_01  c_11 ]
                c02, c12, // [ c_02  c_12 ]
                c03, c13, // [ c_03  c_13 ]
            ],
            2,
        );

        let idft = NaiveDft.idft_batch(evals.clone());
        assert_eq!(idft, expected);

        // Round-trip: DFT(IDFT(evals)) = evals
        let roundtrip = NaiveDft.dft_batch(idft);
        assert_eq!(roundtrip, evals);
    }

    #[test]
    fn test_basic_lde_batch_dim2() {
        type F = BabyBear;

        let c00 = F::from_u8(1);
        let c01 = F::from_u8(2);
        let c10 = F::from_u8(3);
        let c11 = F::from_u8(4);

        // We evaluate two polynomials of degree 1:
        //
        // ```text
        //   f_0(x) = 1 + 2x
        //   f_1(x) = 3 + 4x
        // ```
        //
        // Evaluate each on the 2-point subgroup H = { ω^0, ω^1 } with ω = -1
        // Domain H = [1, -1]

        let omega = F::two_adic_generator(1); // ω = -1

        // Evaluations on H
        let eval00 = c00 + c01 * omega.exp_u64(0);
        let eval01 = c00 + c01 * omega.exp_u64(1);

        let eval10 = c10 + c11 * omega.exp_u64(0);
        let eval11 = c10 + c11 * omega.exp_u64(1);

        let evals = RowMajorMatrix::new(
            vec![
                eval00, eval10, // row for ω^0
                eval01, eval11, // row for ω^1
            ],
            2,
        );

        // We now want to extend these evaluations to the 4-point domain
        // Domain K = { ω^0, ω^1, ω^2, ω^3 } where ω is a primitive 4th root

        let added_bits = 1;
        let lde = NaiveDft.lde_batch(evals.clone(), added_bits);

        // Manual expected result: f_0(x) = 1 + 2x, f_1(x) = 3 + 4x
        let omega4 = F::two_adic_generator(2);

        let f00 = c00 + c01 * omega4.exp_u64(0);
        let f01 = c00 + c01 * omega4.exp_u64(1);
        let f02 = c00 + c01 * omega4.exp_u64(2);
        let f03 = c00 + c01 * omega4.exp_u64(3);

        let f10 = c10 + c11 * omega4.exp_u64(0);
        let f11 = c10 + c11 * omega4.exp_u64(1);
        let f12 = c10 + c11 * omega4.exp_u64(2);
        let f13 = c10 + c11 * omega4.exp_u64(3);

        let expected = RowMajorMatrix::new(
            vec![
                f00, f10, // row for ω^0
                f01, f11, // row for ω^1
                f02, f12, // row for ω^2
                f03, f13,
            ],
            2,
        );

        assert_eq!(lde, expected);
    }

    #[test]
    fn test_basic_lde_batch_dim4() {
        type F = BabyBear;

        let c00 = F::from_u8(1);
        let c01 = F::from_u8(2);
        let c02 = F::from_u8(3);
        let c03 = F::from_u8(4);

        let c10 = F::from_u8(5);
        let c11 = F::from_u8(6);
        let c12 = F::from_u8(7);
        let c13 = F::from_u8(8);

        // We evaluate two polynomials of degree 3:
        //
        // ```text
        //   f_0(x) = 1 + 2x + 3x^2 + 4x^3
        //   f_1(x) = 5 + 6x + 7x^2 + 8x^3
        // ```
        //
        // Evaluate each on the 4-point subgroup H = { ω^0, ω^1, ω^2, ω^3 }
        // where ω is a primitive 4th root of unity

        let omega = F::two_adic_generator(2); // ω

        // Evaluations on H
        let eval00 = c00 * omega.exp_u64(0 * 0)
            + c01 * omega.exp_u64(0 * 1)
            + c02 * omega.exp_u64(0 * 2)
            + c03 * omega.exp_u64(0 * 3);
        let eval01 = c00 * omega.exp_u64(1 * 0)
            + c01 * omega.exp_u64(1 * 1)
            + c02 * omega.exp_u64(1 * 2)
            + c03 * omega.exp_u64(1 * 3);
        let eval02 = c00 * omega.exp_u64(2 * 0)
            + c01 * omega.exp_u64(2 * 1)
            + c02 * omega.exp_u64(2 * 2)
            + c03 * omega.exp_u64(2 * 3);
        let eval03 = c00 * omega.exp_u64(3 * 0)
            + c01 * omega.exp_u64(3 * 1)
            + c02 * omega.exp_u64(3 * 2)
            + c03 * omega.exp_u64(3 * 3);

        let eval10 = c10 * omega.exp_u64(0 * 0)
            + c11 * omega.exp_u64(0 * 1)
            + c12 * omega.exp_u64(0 * 2)
            + c13 * omega.exp_u64(0 * 3);
        let eval11 = c10 * omega.exp_u64(1 * 0)
            + c11 * omega.exp_u64(1 * 1)
            + c12 * omega.exp_u64(1 * 2)
            + c13 * omega.exp_u64(1 * 3);
        let eval12 = c10 * omega.exp_u64(2 * 0)
            + c11 * omega.exp_u64(2 * 1)
            + c12 * omega.exp_u64(2 * 2)
            + c13 * omega.exp_u64(2 * 3);
        let eval13 = c10 * omega.exp_u64(3 * 0)
            + c11 * omega.exp_u64(3 * 1)
            + c12 * omega.exp_u64(3 * 2)
            + c13 * omega.exp_u64(3 * 3);

        let evals = RowMajorMatrix::new(
            vec![
                eval00, eval10, // row for ω^0
                eval01, eval11, // row for ω^1
                eval02, eval12, // row for ω^2
                eval03, eval13, // row for ω^3
            ],
            2,
        );

        // We now want to extend these evaluations to the 8-point domain
        // Domain K = { ω^0, ω^1, ..., ω^7 } where ω is a primitive 8th root

        let added_bits = 1;
        let lde = NaiveDft.lde_batch(evals.clone(), added_bits);

        // Manual expected result:
        //
        // ```text
        // f_0(x) = 1 + 2x + 3x^2 + 4x^3,
        // f_1(x) = 5 + 6x + 7x^2 + 8x^3
        // ```
        let omega8 = F::two_adic_generator(3); // ω for size-8 subgroup

        let f00 = c00 * omega8.exp_u64(0 * 0)
            + c01 * omega8.exp_u64(0 * 1)
            + c02 * omega8.exp_u64(0 * 2)
            + c03 * omega8.exp_u64(0 * 3);
        let f01 = c00 * omega8.exp_u64(1 * 0)
            + c01 * omega8.exp_u64(1 * 1)
            + c02 * omega8.exp_u64(1 * 2)
            + c03 * omega8.exp_u64(1 * 3);
        let f02 = c00 * omega8.exp_u64(2 * 0)
            + c01 * omega8.exp_u64(2 * 1)
            + c02 * omega8.exp_u64(2 * 2)
            + c03 * omega8.exp_u64(2 * 3);
        let f03 = c00 * omega8.exp_u64(3 * 0)
            + c01 * omega8.exp_u64(3 * 1)
            + c02 * omega8.exp_u64(3 * 2)
            + c03 * omega8.exp_u64(3 * 3);
        let f04 = c00 * omega8.exp_u64(4 * 0)
            + c01 * omega8.exp_u64(4 * 1)
            + c02 * omega8.exp_u64(4 * 2)
            + c03 * omega8.exp_u64(4 * 3);
        let f05 = c00 * omega8.exp_u64(5 * 0)
            + c01 * omega8.exp_u64(5 * 1)
            + c02 * omega8.exp_u64(5 * 2)
            + c03 * omega8.exp_u64(5 * 3);
        let f06 = c00 * omega8.exp_u64(6 * 0)
            + c01 * omega8.exp_u64(6 * 1)
            + c02 * omega8.exp_u64(6 * 2)
            + c03 * omega8.exp_u64(6 * 3);
        let f07 = c00 * omega8.exp_u64(7 * 0)
            + c01 * omega8.exp_u64(7 * 1)
            + c02 * omega8.exp_u64(7 * 2)
            + c03 * omega8.exp_u64(7 * 3);

        let g00 = c10 * omega8.exp_u64(0 * 0)
            + c11 * omega8.exp_u64(0 * 1)
            + c12 * omega8.exp_u64(0 * 2)
            + c13 * omega8.exp_u64(0 * 3);
        let g01 = c10 * omega8.exp_u64(1 * 0)
            + c11 * omega8.exp_u64(1 * 1)
            + c12 * omega8.exp_u64(1 * 2)
            + c13 * omega8.exp_u64(1 * 3);
        let g02 = c10 * omega8.exp_u64(2 * 0)
            + c11 * omega8.exp_u64(2 * 1)
            + c12 * omega8.exp_u64(2 * 2)
            + c13 * omega8.exp_u64(2 * 3);
        let g03 = c10 * omega8.exp_u64(3 * 0)
            + c11 * omega8.exp_u64(3 * 1)
            + c12 * omega8.exp_u64(3 * 2)
            + c13 * omega8.exp_u64(3 * 3);
        let g04 = c10 * omega8.exp_u64(4 * 0)
            + c11 * omega8.exp_u64(4 * 1)
            + c12 * omega8.exp_u64(4 * 2)
            + c13 * omega8.exp_u64(4 * 3);
        let g05 = c10 * omega8.exp_u64(5 * 0)
            + c11 * omega8.exp_u64(5 * 1)
            + c12 * omega8.exp_u64(5 * 2)
            + c13 * omega8.exp_u64(5 * 3);
        let g06 = c10 * omega8.exp_u64(6 * 0)
            + c11 * omega8.exp_u64(6 * 1)
            + c12 * omega8.exp_u64(6 * 2)
            + c13 * omega8.exp_u64(6 * 3);
        let g07 = c10 * omega8.exp_u64(7 * 0)
            + c11 * omega8.exp_u64(7 * 1)
            + c12 * omega8.exp_u64(7 * 2)
            + c13 * omega8.exp_u64(7 * 3);

        let expected = RowMajorMatrix::new(
            vec![
                f00, g00, // row for ω^0
                f01, g01, // row for ω^1
                f02, g02, // row for ω^2
                f03, g03, // row for ω^3
                f04, g04, // row for ω^4
                f05, g05, // row for ω^5
                f06, g06, // row for ω^6
                f07, g07, // row for ω^7
            ],
            2,
        );

        assert_eq!(lde, expected);
    }

    #[test]
    fn test_basic_coset_lde_batch_dim2() {
        type F = BabyBear;

        let c00 = F::from_u8(1);
        let c01 = F::from_u8(2);
        let c10 = F::from_u8(3);
        let c11 = F::from_u8(4);

        // We evaluate two polynomials of degree 1:
        //
        // ```text
        //   f_0(x) = 1 + 2x
        //   f_1(x) = 3 + 4x
        // ```
        //
        // We start by evaluating each on the 2-point subgroup
        // H = { ω^0, ω^1 } with ω = -1.
        //
        // Domain H = [1, -1]

        let omega = F::two_adic_generator(1); // ω = -1
        let shift = F::from_u8(5); // arbitrary coset shift

        // Evaluations on gH where g = 1 (so just H)
        let eval00 = c00 + c01 * omega.exp_u64(0); // f_0(1)
        let eval01 = c00 + c01 * omega.exp_u64(1); // f_0(-1)
        let eval10 = c10 + c11 * omega.exp_u64(0); // f_1(1)
        let eval11 = c10 + c11 * omega.exp_u64(1); // f_1(-1)

        let evals = RowMajorMatrix::new(
            vec![
                eval00, eval10, // row for ω^0
                eval01, eval11, // row for ω^1
            ],
            2,
        );

        // Now we want to compute the low-degree extension (LDE)
        // over the coset shift · K, where K is a 4-point subgroup.
        //
        // Domain K = [ω^0, ω^1, ω^2, ω^3] with ω a 4th root
        // Coset domain = [shift·ω^0, shift·ω^1, shift·ω^2, shift·ω^3]
        //
        // LDE is performed by:
        // 1. IDFT to get coefficients of f_0, f_1
        // 2. Padding with 0s (degree extension)
        // 3. Coset DFT on the 4-point domain

        let added_bits = 1;
        let lde = NaiveDft.coset_lde_batch(evals.clone(), added_bits, shift);

        // Manual computation of expected evaluations
        //
        // Reconstruct polynomials from coefficients:
        // f_0(x) = 1 + 2x
        // f_1(x) = 3 + 4x
        //
        // Evaluate on:
        //   [ shift·ω^0, shift·ω^1, shift·ω^2, shift·ω^3 ]

        let omega4 = F::two_adic_generator(2);

        let f00 = c00 + c01 * (shift * omega4.exp_u64(0));
        let f01 = c00 + c01 * (shift * omega4.exp_u64(1));
        let f02 = c00 + c01 * (shift * omega4.exp_u64(2));
        let f03 = c00 + c01 * (shift * omega4.exp_u64(3));

        let f10 = c10 + c11 * (shift * omega4.exp_u64(0));
        let f11 = c10 + c11 * (shift * omega4.exp_u64(1));
        let f12 = c10 + c11 * (shift * omega4.exp_u64(2));
        let f13 = c10 + c11 * (shift * omega4.exp_u64(3));

        let expected = RowMajorMatrix::new(
            vec![
                f00, f10, // row for shift·ω^0
                f01, f11, // row for shift·ω^1
                f02, f12, // row for shift·ω^2
                f03, f13, // row for shift·ω^3
            ],
            2,
        );

        assert_eq!(lde, expected);
    }

    #[test]
    fn test_basic_coset_lde_batch_dim4() {
        type F = BabyBear;

        let c00 = F::from_u8(1);
        let c01 = F::from_u8(2);
        let c02 = F::from_u8(3);
        let c03 = F::from_u8(4);

        let c10 = F::from_u8(5);
        let c11 = F::from_u8(6);
        let c12 = F::from_u8(7);
        let c13 = F::from_u8(8);

        // We evaluate two polynomials of degree 3:
        //
        // ```text
        //   f_0(x) = 1 + 2x + 3x^2 + 4x^3
        //   f_1(x) = 5 + 6x + 7x^2 + 8x^3
        // ```
        //
        // We start by evaluating each on the 4-point subgroup
        // H = { ω^0, ω^1, ω^2, ω^3 } where ω is a 4th root of unity.

        let omega = F::two_adic_generator(2); // ω
        let shift = F::from_u8(5); // arbitrary coset shift

        // Evaluations on gH where g = 1 (so just H)
        let eval00 = c00 * omega.exp_u64(0 * 0)
            + c01 * omega.exp_u64(0 * 1)
            + c02 * omega.exp_u64(0 * 2)
            + c03 * omega.exp_u64(0 * 3);
        let eval01 = c00 * omega.exp_u64(1 * 0)
            + c01 * omega.exp_u64(1 * 1)
            + c02 * omega.exp_u64(1 * 2)
            + c03 * omega.exp_u64(1 * 3);
        let eval02 = c00 * omega.exp_u64(2 * 0)
            + c01 * omega.exp_u64(2 * 1)
            + c02 * omega.exp_u64(2 * 2)
            + c03 * omega.exp_u64(2 * 3);
        let eval03 = c00 * omega.exp_u64(3 * 0)
            + c01 * omega.exp_u64(3 * 1)
            + c02 * omega.exp_u64(3 * 2)
            + c03 * omega.exp_u64(3 * 3);

        let eval10 = c10 * omega.exp_u64(0 * 0)
            + c11 * omega.exp_u64(0 * 1)
            + c12 * omega.exp_u64(0 * 2)
            + c13 * omega.exp_u64(0 * 3);
        let eval11 = c10 * omega.exp_u64(1 * 0)
            + c11 * omega.exp_u64(1 * 1)
            + c12 * omega.exp_u64(1 * 2)
            + c13 * omega.exp_u64(1 * 3);
        let eval12 = c10 * omega.exp_u64(2 * 0)
            + c11 * omega.exp_u64(2 * 1)
            + c12 * omega.exp_u64(2 * 2)
            + c13 * omega.exp_u64(2 * 3);
        let eval13 = c10 * omega.exp_u64(3 * 0)
            + c11 * omega.exp_u64(3 * 1)
            + c12 * omega.exp_u64(3 * 2)
            + c13 * omega.exp_u64(3 * 3);

        let evals = RowMajorMatrix::new(
            vec![
                eval00, eval10, // row for ω^0
                eval01, eval11, // row for ω^1
                eval02, eval12, // row for ω^2
                eval03, eval13, // row for ω^3
            ],
            2,
        );

        // Now we want to compute the low-degree extension (LDE)
        // over the coset shift · K, where K is an 8-point subgroup.
        //
        // Domain K = [ω^0, ω^1, ..., ω^7] with ω a 8th root
        // Coset domain = [shift·ω^0, shift·ω^1, ..., shift·ω^7]
        //
        // LDE is performed by:
        // 1. IDFT to get coefficients of f_0, f_1
        // 2. Padding with 0s (degree extension)
        // 3. Coset DFT on the 8-point domain

        let added_bits = 1;
        let lde = NaiveDft.coset_lde_batch(evals.clone(), added_bits, shift);

        // Manual computation of expected evaluations
        //
        // Reconstruct polynomials from coefficients:
        // f_0(x) = 1 + 2x + 3x^2 + 4x^3
        // f_1(x) = 5 + 6x + 7x^2 + 8x^3
        //
        // Evaluate on:
        //   [ shift·ω^0, shift·ω^1, ..., shift·ω^7 ]

        let omega8 = F::two_adic_generator(3);

        let f00 = c00 * (shift * omega8.exp_u64(0)).exp_u64(0)
            + c01 * (shift * omega8.exp_u64(0)).exp_u64(1)
            + c02 * (shift * omega8.exp_u64(0)).exp_u64(2)
            + c03 * (shift * omega8.exp_u64(0)).exp_u64(3);
        let f01 = c00 * (shift * omega8.exp_u64(1)).exp_u64(0)
            + c01 * (shift * omega8.exp_u64(1)).exp_u64(1)
            + c02 * (shift * omega8.exp_u64(1)).exp_u64(2)
            + c03 * (shift * omega8.exp_u64(1)).exp_u64(3);
        let f02 = c00 * (shift * omega8.exp_u64(2)).exp_u64(0)
            + c01 * (shift * omega8.exp_u64(2)).exp_u64(1)
            + c02 * (shift * omega8.exp_u64(2)).exp_u64(2)
            + c03 * (shift * omega8.exp_u64(2)).exp_u64(3);
        let f03 = c00 * (shift * omega8.exp_u64(3)).exp_u64(0)
            + c01 * (shift * omega8.exp_u64(3)).exp_u64(1)
            + c02 * (shift * omega8.exp_u64(3)).exp_u64(2)
            + c03 * (shift * omega8.exp_u64(3)).exp_u64(3);
        let f04 = c00 * (shift * omega8.exp_u64(4)).exp_u64(0)
            + c01 * (shift * omega8.exp_u64(4)).exp_u64(1)
            + c02 * (shift * omega8.exp_u64(4)).exp_u64(2)
            + c03 * (shift * omega8.exp_u64(4)).exp_u64(3);
        let f05 = c00 * (shift * omega8.exp_u64(5)).exp_u64(0)
            + c01 * (shift * omega8.exp_u64(5)).exp_u64(1)
            + c02 * (shift * omega8.exp_u64(5)).exp_u64(2)
            + c03 * (shift * omega8.exp_u64(5)).exp_u64(3);
        let f06 = c00 * (shift * omega8.exp_u64(6)).exp_u64(0)
            + c01 * (shift * omega8.exp_u64(6)).exp_u64(1)
            + c02 * (shift * omega8.exp_u64(6)).exp_u64(2)
            + c03 * (shift * omega8.exp_u64(6)).exp_u64(3);
        let f07 = c00 * (shift * omega8.exp_u64(7)).exp_u64(0)
            + c01 * (shift * omega8.exp_u64(7)).exp_u64(1)
            + c02 * (shift * omega8.exp_u64(7)).exp_u64(2)
            + c03 * (shift * omega8.exp_u64(7)).exp_u64(3);

        let g00 = c10 * (shift * omega8.exp_u64(0)).exp_u64(0)
            + c11 * (shift * omega8.exp_u64(0)).exp_u64(1)
            + c12 * (shift * omega8.exp_u64(0)).exp_u64(2)
            + c13 * (shift * omega8.exp_u64(0)).exp_u64(3);
        let g01 = c10 * (shift * omega8.exp_u64(1)).exp_u64(0)
            + c11 * (shift * omega8.exp_u64(1)).exp_u64(1)
            + c12 * (shift * omega8.exp_u64(1)).exp_u64(2)
            + c13 * (shift * omega8.exp_u64(1)).exp_u64(3);
        let g02 = c10 * (shift * omega8.exp_u64(2)).exp_u64(0)
            + c11 * (shift * omega8.exp_u64(2)).exp_u64(1)
            + c12 * (shift * omega8.exp_u64(2)).exp_u64(2)
            + c13 * (shift * omega8.exp_u64(2)).exp_u64(3);
        let g03 = c10 * (shift * omega8.exp_u64(3)).exp_u64(0)
            + c11 * (shift * omega8.exp_u64(3)).exp_u64(1)
            + c12 * (shift * omega8.exp_u64(3)).exp_u64(2)
            + c13 * (shift * omega8.exp_u64(3)).exp_u64(3);
        let g04 = c10 * (shift * omega8.exp_u64(4)).exp_u64(0)
            + c11 * (shift * omega8.exp_u64(4)).exp_u64(1)
            + c12 * (shift * omega8.exp_u64(4)).exp_u64(2)
            + c13 * (shift * omega8.exp_u64(4)).exp_u64(3);
        let g05 = c10 * (shift * omega8.exp_u64(5)).exp_u64(0)
            + c11 * (shift * omega8.exp_u64(5)).exp_u64(1)
            + c12 * (shift * omega8.exp_u64(5)).exp_u64(2)
            + c13 * (shift * omega8.exp_u64(5)).exp_u64(3);
        let g06 = c10 * (shift * omega8.exp_u64(6)).exp_u64(0)
            + c11 * (shift * omega8.exp_u64(6)).exp_u64(1)
            + c12 * (shift * omega8.exp_u64(6)).exp_u64(2)
            + c13 * (shift * omega8.exp_u64(6)).exp_u64(3);
        let g07 = c10 * (shift * omega8.exp_u64(7)).exp_u64(0)
            + c11 * (shift * omega8.exp_u64(7)).exp_u64(1)
            + c12 * (shift * omega8.exp_u64(7)).exp_u64(2)
            + c13 * (shift * omega8.exp_u64(7)).exp_u64(3);

        let expected = RowMajorMatrix::new(
            vec![
                f00, g00, // row for ω^0
                f01, g01, // row for ω^1
                f02, g02, // row for ω^2
                f03, g03, // row for ω^3
                f04, g04, // row for ω^4
                f05, g05, // row for ω^5
                f06, g06, // row for ω^6
                f07, g07, // row for ω^7
            ],
            2,
        );

        assert_eq!(lde, expected);
    }
}
