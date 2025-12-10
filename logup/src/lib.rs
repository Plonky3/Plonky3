//! Generate a LogUp auxiliary trace for proving a permutation relationship between
//! the last two columns of a main trace.

use p3_field::{ExtensionField, Field, batch_multiplicative_inverse};
use p3_matrix::Matrix;
use p3_matrix::dense::DenseMatrix;

/// Generates an auxiliary trace for proving a permutation using the LogUp protocol.
///
/// This function implements the LogUp (Logarithmic Derivative) argument to prove that the
/// last two columns of the main trace form a permutation of each other. The LogUp technique
/// is a protocol that verifies set equality between two columns by accumulating fractional terms.
///
/// # Protocol Overview
///
/// Given two columns `A = [a₀, a₁, ..., aₙ₋₁]` and `B = [b₀, b₁, ..., bₙ₋₁]` that form a
/// permutation, and a random challenge `r`, the LogUp protocol constructs:
///
/// - `tᵢ = 1/(r - aᵢ)`
/// - `wᵢ = 1/(r - bᵢ)`
/// - Running sum: `Sᵢ = Σⱼ₌₀ⁱ (tⱼ - wⱼ)`
///
/// If A and B are truly a permutation, then `Sₙ₋₁ = 0` (all terms cancel out).
///
/// # Trace Layout
///
/// The generated auxiliary trace has 3 extension field columns (stored as `3 * EF::DIMENSION`
/// base field columns):
/// - Column 0: `tᵢ = 1/(r - main[i][width-2])`
/// - Column 1: `wᵢ = 1/(r - main[i][width-1])`
/// - Column 2: Running sum `Sᵢ = Sᵢ₋₁ + tᵢ - wᵢ` (with `S₀ = t₀ - w₀`)
///
/// # Parameters
///
/// * `main_trace` - The main execution trace. Must have width ≥ 2. The last two columns
///                  must form a permutation (verified in debug builds).
/// * `randomness` - A random challenge from the verifier, used to ensure soundness.
///
/// # Returns
///
/// A dense matrix containing the auxiliary trace with:
/// - Height: Same as `main_trace.height()`
/// - Width: `3 * EF::DIMENSION` (three extension field elements in base field representation)
///
/// # Panics
///
/// - If `main_trace.width() < 2` (permutation check requires at least 2 columns)
/// - In debug builds, if the last two columns don't form a valid permutation
///
/// # Example
///
/// ```ignore
/// use p3_baby_bear::BabyBear;
/// use p3_field::extension::BinomialExtensionField;
/// use p3_matrix::dense::DenseMatrix;
///
/// type F = BabyBear;
/// type EF = BinomialExtensionField<F, 4>;
///
/// // Create a trace where last two columns are [1,2,3,4] and [4,3,2,1]
/// let trace_values = vec![
///     F::from_u64(0), F::from_u64(1), F::from_u64(4),
///     F::from_u64(0), F::from_u64(2), F::from_u64(3),
///     F::from_u64(0), F::from_u64(3), F::from_u64(2),
///     F::from_u64(0), F::from_u64(4), F::from_u64(1),
/// ];
/// let main_trace = DenseMatrix::new(trace_values, 3);
/// let randomness = EF::from_u64(100);
///
/// let aux_trace = generate_logup_trace::<EF, _>(&main_trace, &randomness);
///
/// // The last element of the running sum column should be zero
/// // (proving the permutation is valid)
/// ```
///
/// # Security
///
/// The soundness error of this protocol is approximately `n/|F|` where `n` is the trace
/// length and `|F|` is the size of the extension field. The random challenge `r` ensures
/// that a malicious prover cannot construct fake witnesses that pass verification.
#[allow(clippy::doc_overindented_list_items)]
pub fn generate_logup_trace<EF, F>(main_trace: &DenseMatrix<F>, randomness: &EF) -> DenseMatrix<F>
where
    EF: ExtensionField<F>,
    F: Field,
{
    let len = main_trace.height();
    let width = main_trace.width();

    assert!(
        width >= 2,
        "Permutation check is not possible for main trace width ({width}) < 2"
    );

    let mut main_second_last_col = vec![];
    let mut main_last_col = vec![];

    for row_idx in 0..len {
        main_second_last_col.push(main_trace.get(row_idx, width - 2).unwrap());
        main_last_col.push(main_trace.get(row_idx, width - 1).unwrap());
    }

    // Sanity check that the last two columns are permutations.
    #[cfg(debug_assertions)]
    {
        assert!(
            is_permutation(&main_second_last_col, &main_last_col),
            "The last two columns of the main trace must form a permutation"
        );
    }

    // stores 1/(r - main[row_idx][width-2]) and 1/(r - main[row_idx][width-1])
    // Note that the inverse is taken over the extension field and later on only the final result is parsed as base field elements
    let r_sub_main_second_last_col = main_second_last_col
        .iter()
        .map(|&x| *randomness - EF::from(x))
        .collect::<Vec<EF>>();

    let r_sub_main_last_col = main_last_col
        .iter()
        .map(|&x| *randomness - EF::from(x))
        .collect::<Vec<EF>>();

    let aux_first_col = batch_multiplicative_inverse(&r_sub_main_second_last_col);
    let aux_second_col = batch_multiplicative_inverse(&r_sub_main_last_col);

    // stores
    // - t_i = 1/(r - main[row_idx][width-2])
    // - w_i = 1/(r - main[row_idx][width-1])
    // - running sum: sum(t_i - w_i)
    let mut aux_trace_values = vec![
        aux_first_col[0],
        aux_second_col[0],
        aux_first_col[0] - aux_second_col[0],
    ];
    for row_idx in 1..len {
        let tmp = aux_trace_values[(row_idx - 1) * 3 + 2] + aux_first_col[row_idx]
            - aux_second_col[row_idx];

        aux_trace_values.extend_from_slice(&[aux_first_col[row_idx], aux_second_col[row_idx], tmp]);
    }

    let aux_trace_base_values = aux_trace_values
        .iter()
        .flat_map(|r| r.as_basis_coefficients_slice())
        .cloned()
        .collect();

    DenseMatrix::new(aux_trace_base_values, 3 * EF::DIMENSION)
}

/// Check if two vectors contain the same multiset of elements (i.e., form a permutation).
///
/// Three possible approaches:
/// 1. HashMap - requires Hash trait (not available for Field)
/// 2. BTreeMap - requires Ord trait (not available for Field)
/// 3. O(n^2) algorithm - only requires Eq (available via Field -> Packable)
///
/// We use approach #3, and it is for debug-only assertions.
#[cfg(debug_assertions)]
fn is_permutation<F: Field>(col1: &[F], col2: &[F]) -> bool {
    if col1.len() != col2.len() {
        return false;
    }

    // For each element in col1, check it exists in col2 with the same count
    let mut col2_used = vec![false; col2.len()];

    for &elem1 in col1 {
        let mut found = false;
        for (i, &elem2) in col2.iter().enumerate() {
            if !col2_used[i] && elem1 == elem2 {
                col2_used[i] = true;
                found = true;
                break;
            }
        }
        if !found {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_goldilocks::Goldilocks;

    use super::*;

    type F = Goldilocks;
    type EF = BinomialExtensionField<F, 2>;

    #[test]
    fn test_simple_permutation() {
        // Create a simple trace with 4 rows and 3 columns
        // Last two columns form a permutation: [1,2,3,4] and [4,3,2,1]
        let trace_values = vec![
            F::from_u64(0),
            F::from_u64(1),
            F::from_u64(4),
            F::from_u64(0),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(0),
            F::from_u64(3),
            F::from_u64(2),
            F::from_u64(0),
            F::from_u64(4),
            F::from_u64(1),
        ];
        let main_trace = DenseMatrix::new(trace_values, 3);
        let randomness = EF::from_u64(100);

        let aux_trace = generate_logup_trace::<EF, _>(&main_trace, &randomness);

        // Check dimensions: 4 rows x 3 ext columns (each ext field element is 2 base field elements)
        assert_eq!(aux_trace.height(), 4);
        assert_eq!(aux_trace.width(), 6);

        // Verify the last running sum is zero (permutation property)
        // Running sum starts at column 4 (third ext field = columns 4-5)
        let last_running_sum = aux_trace.get(3, 4).unwrap();
        assert_eq!(last_running_sum, F::ZERO);
    }

    #[test]
    fn test_running_sum_initialization() {
        // Test that the first running sum is correctly initialized
        let trace_values = vec![
            F::from_u64(10),
            F::from_u64(5),
            F::from_u64(8),
            F::from_u64(20),
            F::from_u64(7),
            F::from_u64(5),
            F::from_u64(30),
            F::from_u64(8),
            F::from_u64(7),
        ];
        let main_trace = DenseMatrix::new(trace_values, 3);
        let randomness = EF::from_u64(50);

        let aux_trace = generate_logup_trace::<EF, _>(&main_trace, &randomness);

        // Verify dimensions (3 ext field columns x 2 base field elements per ext field = 6)
        assert_eq!(aux_trace.height(), 3);
        assert_eq!(aux_trace.width(), 6);

        // Verify first row initialization: running_sum[0] = t[0] - w[0]
        // t0 is at columns 0-1, w0 is at columns 2-3, running_sum_0 is at columns 4-5
        let t0 = aux_trace.get(0, 0).unwrap();
        let w0 = aux_trace.get(0, 2).unwrap();
        let running_sum_0 = aux_trace.get(0, 4).unwrap();
        assert_eq!(running_sum_0, t0 - w0);
    }

    #[test]
    #[cfg(debug_assertions)]
    fn test_is_permutation() {
        // Test the helper function
        let a = F::from_u64(1);
        let b = F::from_u64(2);
        let c = F::from_u64(3);
        let d = F::from_u64(4);

        // Valid permutation
        let col1 = vec![a, b, c, d];
        let col2 = vec![d, c, b, a];
        assert!(is_permutation(&col1, &col2));

        // Same elements with duplicates
        let col3 = vec![a, a, b, c];
        let col4 = vec![c, a, b, a];
        assert!(is_permutation(&col3, &col4));

        // Not a permutation (different elements)
        let col5 = vec![a, b, c, d];
        let col6 = vec![a, a, b, c];
        assert!(!is_permutation(&col5, &col6));

        // Not a permutation (different lengths)
        let col7 = vec![a, b];
        let col8 = vec![a, b, c];
        assert!(!is_permutation(&col7, &col8));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "The last two columns of the main trace must form a permutation")]
    fn test_invalid_permutation() {
        // Create a trace where last two columns are NOT a permutation
        let trace_values = vec![
            F::from_u64(0),
            F::from_u64(1),
            F::from_u64(4),
            F::from_u64(0),
            F::from_u64(2),
            F::from_u64(5), // 5 is not in second-to-last column
            F::from_u64(0),
            F::from_u64(3),
            F::from_u64(2),
            F::from_u64(0),
            F::from_u64(4),
            F::from_u64(1),
        ];
        let main_trace = DenseMatrix::new(trace_values, 3);
        let randomness = EF::from_u64(100);

        generate_logup_trace::<EF, _>(&main_trace, &randomness);
    }
}
