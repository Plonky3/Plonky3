//! Two-adic periodic column evaluator.
//!
//! This module provides `TwoAdicPeriodicEvaluator` for evaluating periodic columns
//! in two-adic STARK proofs. The implementation supports:
//!
//! - `eval_on_lde`: Evaluates periodic columns on the LDE domain using FFT extrapolation.
//!   All columns are padded to the maximum period, creating a rectangular matrix that
//!   stores only `max_period × blowup` rows with modular indexing for O(1) lookup.
//!
//! - `eval_at_point`: Evaluates periodic columns at arbitrary points using Lagrange
//!   interpolation. This is used by the verifier.
//!
//! ## Memory Efficiency
//!
//! Instead of materializing the full LDE-sized table, we store only `max_period × blowup`
//! rows. For a trace of size 2^20 with period-4 columns and blowup 4, this means storing
//! 16 rows instead of 4M rows per column.
//!
//! ## Complexity
//!
//! - `eval_on_lde`: O(max_period × blowup × log(max_period × blowup)) for FFT extrapolation,
//!   then O(1) per LDE point lookup using modular indexing.
//!
//! - `eval_at_point`: O(period) per column using direct Lagrange interpolation.
//!
//! Note: The current `eval_at_point` implementation is not optimized for multiple columns
//! with the same period. The Lagrange basis evaluations (or barycentric weights) at `point`
//! could be computed once per distinct period and reused across columns sharing that period.

use alloc::vec::Vec;

use p3_commit::{PeriodicEvaluator, PeriodicLdeTable};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;

/// Build the compact periodic LDE table using the two-adic evaluator.
///
/// This is a type-level helper so callers can use concrete `F` without
/// the compiler struggling to unify `F` with `PolynomialSpace::Val`.
pub fn build_periodic_lde_table_two_adic<F, Dft>(
    periodic_table: &[Vec<F>],
    trace_domain: &TwoAdicMultiplicativeCoset<F>,
    lde_domain: &TwoAdicMultiplicativeCoset<F>,
) -> PeriodicLdeTable<F>
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F> + Default,
{
    TwoAdicPeriodicEvaluator::<Dft>::eval_on_lde(periodic_table, trace_domain, lde_domain)
}

/// Evaluates periodic polynomials for two-adic multiplicative cosets.
///
/// For a periodic column with period `p` and trace length `n`, the periodic values
/// repeat every `n/p` rows. To evaluate at an arbitrary point `ζ`:
/// 1. Project `ζ` to the periodic subdomain: `ζ_periodic = ζ^(n/p)`
/// 2. Evaluate the degree-(p-1) polynomial at `ζ_periodic`
#[derive(Clone, Default)]
pub struct TwoAdicPeriodicEvaluator<Dft> {
    _phantom: core::marker::PhantomData<Dft>,
}

impl<Dft> TwoAdicPeriodicEvaluator<Dft> {
    pub fn new(_dft: Dft) -> Self {
        Self {
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<F, Dft> PeriodicEvaluator<F, TwoAdicMultiplicativeCoset<F>> for TwoAdicPeriodicEvaluator<Dft>
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    fn eval_on_lde(
        periodic_table: &[Vec<F>],
        trace_domain: &TwoAdicMultiplicativeCoset<F>,
        lde_domain: &TwoAdicMultiplicativeCoset<F>,
    ) -> PeriodicLdeTable<F> {
        if periodic_table.is_empty() {
            return PeriodicLdeTable::empty();
        }

        let trace_len = trace_domain.size();
        let lde_len = lde_domain.size();
        let lde_shift = lde_domain.shift();
        let blowup = lde_len / trace_len;
        let log_blowup = log2_strict_usize(blowup);

        // Find the maximum period and validate all columns
        let max_period = periodic_table
            .iter()
            .map(|col| {
                let period = col.len();
                debug_assert!(
                    period.is_power_of_two(),
                    "periodic column length must be a power of 2"
                );
                period
            })
            .max()
            .unwrap();

        let extended_height = max_period * blowup;
        let num_cols = periodic_table.len();

        // Compute the shift for the periodic subdomain at max_period.
        // This aligns the periodic domain with the LDE domain so modular indexing works.
        let periodic_shift = lde_shift.exp_u64((lde_len / extended_height) as u64);

        let dft = Dft::default();

        // Process each column: pad to max_period, then extrapolate
        // Build the result in column-major order first, then transpose to row-major
        let mut columns: Vec<Vec<F>> = Vec::with_capacity(num_cols);

        for col in periodic_table {
            let period = col.len();

            // Pad column to max_period by repeating values
            let padded: Vec<F> = if period == max_period {
                col.clone()
            } else {
                (0..max_period).map(|i| col[i % period]).collect()
            };

            // Extrapolate to extended_height using DFT
            let extended = dft.coset_lde(padded, log_blowup, periodic_shift);
            columns.push(extended);
        }

        // Convert from column-major to row-major storage
        let mut row_major_values = Vec::with_capacity(extended_height * num_cols);
        for row_idx in 0..extended_height {
            for col in &columns {
                row_major_values.push(col[row_idx]);
            }
        }

        PeriodicLdeTable::new(RowMajorMatrix::new(row_major_values, num_cols))
    }

    fn eval_at_point<EF: ExtensionField<F>>(
        periodic_table: &[Vec<F>],
        trace_domain: &TwoAdicMultiplicativeCoset<F>,
        point: EF,
    ) -> Vec<EF> {
        let trace_len = trace_domain.size();

        periodic_table
            .iter()
            .map(|col| {
                let period = col.len();
                debug_assert!(
                    period.is_power_of_two(),
                    "periodic column length must be a power of 2"
                );

                let exponent = (trace_len / period) as u64;

                // Project point to periodic subdomain: ζ^(n/p)
                let periodic_point = point.exp_u64(exponent);

                // Evaluate the periodic polynomial at the projected point
                eval_periodic_poly(col, periodic_point)
            })
            .collect()
    }
}

/// Evaluate a periodic polynomial at a single point using Lagrange interpolation.
///
/// The polynomial is the unique degree-(period-1) polynomial that interpolates
/// the given values over the subgroup of size `period`.
fn eval_periodic_poly<F: TwoAdicField, EF: ExtensionField<F>>(values: &[F], point: EF) -> EF {
    let period = values.len();
    if period == 0 {
        return EF::ZERO;
    }
    if period == 1 {
        return EF::from(values[0]);
    }

    let log_period = log2_strict_usize(period);
    let omega = F::two_adic_generator(log_period);

    // Compute Lagrange interpolation at `point`
    // L_i(x) = ∏_{j≠i} (x - ω^j) / (ω^i - ω^j)
    //
    // For efficiency, use the formula:
    // p(x) = (x^n - 1) / n * Σ_i (v_i / (x - ω^i))
    // where n = period, ω is the primitive n-th root of unity

    let n = EF::from(F::from_usize(period));
    let x_n_minus_1 = point.exp_u64(period as u64) - EF::ONE;

    // Handle case where point is on the subgroup (x^n - 1 = 0)
    // In this case, return the value at that point directly
    if x_n_minus_1.is_zero() {
        // Find which root of unity `point` equals
        let mut omega_i = F::ONE;
        for &val in values.iter() {
            if point == EF::from(omega_i) {
                return EF::from(val);
            }
            omega_i *= omega;
        }
        // If we get here, something is wrong
        return EF::ZERO;
    }

    // Compute Σ_i (v_i * ω^i / (x - ω^i))
    // Derived from Lagrange interpolation: L_i(x) = (x^n - 1) / (n * ω^{i(n-1)} * (x - ω^i))
    // Since ω^{-i(n-1)} = ω^i (as ω^n = 1), we get factor ω^i
    let mut sum = EF::ZERO;
    let mut omega_i = F::ONE;
    for &val in values.iter() {
        let denom = point - EF::from(omega_i);
        // denom should be non-zero since x^n - 1 ≠ 0
        sum += EF::from(val * omega_i) * denom.inverse();
        omega_i *= omega;
    }

    x_n_minus_1 * sum / n
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    type F = BabyBear;

    #[test]
    fn test_eval_periodic_poly_constant() {
        // Constant polynomial: all values are 5
        let values = vec![F::from_u64(5); 4];
        let point = F::from_u64(7);
        let result = eval_periodic_poly(&values, point);
        assert_eq!(result, F::from_u64(5));
    }

    #[test]
    fn test_eval_periodic_poly_at_roots() {
        // Values at roots of unity should interpolate correctly
        let values: Vec<F> = vec![1, 2, 3, 4].into_iter().map(F::from_u64).collect();
        let omega = F::two_adic_generator(2); // 4th root of unity

        for (i, &expected) in values.iter().enumerate() {
            let point = omega.exp_u64(i as u64);
            let result = eval_periodic_poly(&values, point);
            assert_eq!(result, expected, "Failed at root index {}", i);
        }
    }
}
