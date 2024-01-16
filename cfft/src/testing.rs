use p3_field::{Field, AbstractField, ComplexExtension};
use alloc::vec;
use alloc::vec::Vec;
use itertools::Itertools;

/// Unlike the standard DFT where both directions can be reinterpreted as polynomial evaluation,
/// In the CFFT only the iCFFT naturally corresponds to such an evaluation.
/// Thus instead of writing a "Naive" CFFT, we just give the polynomial evaluation and some auxillary functions.


/// Get the cfft polynomial basis.
/// The basis consists off all multi-linear products of: y, x, 2x^2 - 1, 2(2x^2 - 1)^2 - 1, ...
/// The ordering of these basis elements is the bit reversal of the sequence: 1, y, x, xy, (2x^2 - 1), (2x^2 - 1)y, ...
/// We also need to throw in a couple of negative signs for technical reasons.
fn cfft_poly_basis<Base: AbstractField + Field, Ext: ComplexExtension<Base>>(
    point: &Ext,
    n: u32,
) -> Vec<Base> {
    if n == 0 {
        return vec![Base::one()]; // Base case
    }

    let mut output = vec![Base::one(), point.imag()]; // The n = 1 case is also special as y only appears once.

    let mut current = point.real();

    for _ in 1..n {
        let new = output.clone().into_iter().map(|val| val * current); // New basis elements to add.

        output = output.into_iter().interleave(new).collect(); // Interleave the two basis together to keep the bit reversal ordering.

        current = (Base::two()) * current * current - Base::one();
        // Find the next basis vector.
    }

    // We need to handle the negatives which can appear in our cFFT method.
    // For the i'th basis element, we multiply it by -1 for every occurance of 11 in the binary decomposition of i.
    // There is almost certainly a better way to do this but this code is only here for cross checks and won't be used in production.
    for (i, val) in output.iter_mut().enumerate() {
        let mut last = false;
        for j in 0..n {
            let test_bit = 1 << j;
            let non_zero_test = i & test_bit != 0;
            if non_zero_test && last {
                *val *= -Base::one();
            }
            last = non_zero_test;
        }
    }

    output
}

/// Evaluate a polynomial with coefficents given in the CFFT basis at a point (x, y)
/// len(coeffs) needs to be a power of 2.
/// Gives a simple O(n^2) equivalent to check our CFFT against.
pub(crate)  fn evaluate_cfft_poly<Base: AbstractField + Field, Ext: ComplexExtension<Base>>(
    coeffs: &[Base],
    point: Ext,
) -> Base {
    let n = coeffs.len();

    debug_assert!(n.is_power_of_two()); // If n is not a power of 2 something has gone badly wrong.

    let log_n = n.trailing_zeros();

    let basis = cfft_poly_basis(&point, log_n); // Get the cfft polynomial basis evaluated at the point x.

    let mut output = Base::zero();

    for i in 0..n {
        output += coeffs[i] * basis[i] // Dot product the basis with the coefficients.
    }

    output
}

// pub(crate) fn test_dft_matches_naive<F, Dft>()
// where
//     F: TwoAdicField,
//     Standard: Distribution<F>,
//     Dft: TwoAdicSubgroupDft<F>,
// {
//     let dft = Dft::default();
//     let mut rng = thread_rng();
//     for log_h in 0..5 {
//         let h = 1 << log_h;
//         let mat = RowMajorMatrix::<F>::rand(&mut rng, h, 3);
//         let dft_naive = NaiveDft.dft_batch(mat.clone());
//         let dft_result = dft.dft_batch(mat);
//         assert_eq!(dft_naive, dft_result.to_row_major_matrix());
//     }
// }

// pub(crate) fn test_coset_dft_matches_naive<F, Dft>()
// where
//     F: TwoAdicField,
//     Standard: Distribution<F>,
//     Dft: TwoAdicSubgroupDft<F>,
// {
//     let dft = Dft::default();
//     let mut rng = thread_rng();
//     for log_h in 0..5 {
//         let h = 1 << log_h;
//         let mat = RowMajorMatrix::<F>::rand(&mut rng, h, 3);
//         let shift = F::generator();
//         let coset_dft_naive = NaiveDft.coset_dft_batch(mat.clone(), shift);
//         let coset_dft_result = dft.coset_dft_batch(mat, shift);
//         assert_eq!(coset_dft_naive, coset_dft_result.to_row_major_matrix());
//     }
// }

// pub(crate) fn test_idft_matches_naive<F, Dft>()
// where
//     F: TwoAdicField,
//     Standard: Distribution<F>,
//     Dft: TwoAdicSubgroupDft<F>,
// {
//     let dft = Dft::default();
//     let mut rng = thread_rng();
//     for log_h in 0..5 {
//         let h = 1 << log_h;
//         let mat = RowMajorMatrix::<F>::rand(&mut rng, h, 3);
//         let idft_naive = NaiveDft.idft_batch(mat.clone());
//         let idft_result = dft.idft_batch(mat);
//         assert_eq!(idft_naive, idft_result);
//     }
// }

// pub(crate) fn test_lde_matches_naive<F, Dft>()
// where
//     F: TwoAdicField,
//     Standard: Distribution<F>,
//     Dft: TwoAdicSubgroupDft<F>,
// {
//     let dft = Dft::default();
//     let mut rng = thread_rng();
//     for log_h in 0..5 {
//         let h = 1 << log_h;
//         let mat = RowMajorMatrix::<F>::rand(&mut rng, h, 3);
//         let lde_naive = NaiveDft.lde_batch(mat.clone(), 1);
//         let lde_result = dft.lde_batch(mat, 1);
//         assert_eq!(lde_naive, lde_result.to_row_major_matrix());
//     }
// }

// pub(crate) fn test_coset_lde_matches_naive<F, Dft>()
// where
//     F: TwoAdicField,
//     Standard: Distribution<F>,
//     Dft: TwoAdicSubgroupDft<F>,
// {
//     let dft = Dft::default();
//     let mut rng = thread_rng();
//     for log_h in 0..5 {
//         let h = 1 << log_h;
//         let mat = RowMajorMatrix::<F>::rand(&mut rng, h, 3);
//         let shift = F::generator();
//         let coset_lde_naive = NaiveDft.coset_lde_batch(mat.clone(), 1, shift);
//         let coset_lde_result = dft.coset_lde_batch(mat, 1, shift);
//         assert_eq!(coset_lde_naive, coset_lde_result.to_row_major_matrix());
//     }
// }

// pub(crate) fn test_dft_idft_consistency<F, Dft>()
// where
//     F: TwoAdicField,
//     Standard: Distribution<F>,
//     Dft: TwoAdicSubgroupDft<F>,
// {
//     let dft = Dft::default();
//     let mut rng = thread_rng();
//     for log_h in 0..5 {
//         let h = 1 << log_h;
//         let original = RowMajorMatrix::<F>::rand(&mut rng, h, 3);
//         let dft_output = dft.dft_batch(original.clone());
//         let idft_output = dft.idft_batch(dft_output.to_row_major_matrix());
//         assert_eq!(original, idft_output);
//     }
// }