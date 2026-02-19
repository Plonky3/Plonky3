//! Periodic column support for Circle STARKs.
//!
//! This module provides `CirclePeriodicEvaluator` for evaluating periodic columns
//! in Circle STARK proofs. The implementation supports:
//!
//! - `eval_on_lde`: Evaluates periodic columns on the LDE domain using CFFT extrapolation.
//!   All columns are padded to the maximum period, creating a rectangular matrix that
//!   stores only `max_period × blowup` rows with modular indexing for O(1) lookup.
//!
//! - `eval_at_point`: Evaluates periodic columns at arbitrary points using polynomial
//!   evaluation with repeated doubling projection. This is used by the verifier.
//!
//! ## Memory Efficiency
//!
//! Instead of materializing the full LDE-sized table, we store only `max_period × blowup`
//! rows. For a trace of size 2^20 with period-4 columns and blowup 4, this means storing
//! 16 rows instead of 4M rows per column.
//!
//! ## Complexity
//!
//! - `eval_on_lde`: O(max_period × blowup × log(max_period × blowup)) for CFFT extrapolation,
//!   then O(1) per LDE point lookup using modular indexing.
//!
//! - `eval_at_point`: O(period) per column using polynomial evaluation.
//!
//! Note: The current `eval_at_point` implementation is not optimized for multiple columns
//! with the same period. The interpolation setup could be shared across columns with the
//! same period.

use alloc::vec::Vec;

use p3_commit::{PeriodicEvaluator, PeriodicLdeTable, PolynomialSpace};
use p3_field::ExtensionField;
use p3_field::extension::ComplexExtendable;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;

use crate::CircleEvaluations;
use crate::domain::CircleDomain;
use crate::point::Point;

/// Build the compact periodic LDE table using the circle evaluator.
///
/// This is a type-level helper so callers can use concrete `F` without
/// the compiler struggling to unify `F` with `PolynomialSpace::Val`.
pub fn build_periodic_lde_table_circle<F: ComplexExtendable>(
    periodic_table: &[Vec<F>],
    trace_domain: &CircleDomain<F>,
    lde_domain: &CircleDomain<F>,
) -> p3_commit::PeriodicLdeTable<F> {
    CirclePeriodicEvaluator::eval_on_lde(periodic_table, trace_domain, lde_domain)
}

/// Evaluates periodic polynomials for Circle STARKs.
///
/// For a periodic column with period `p` and trace length `n`, the periodic values
/// are interpolated on a Circle domain of size `p`. To evaluate at any point:
/// 1. Interpolate the periodic values on a small Circle domain of size `p`
/// 2. Project the query point to the periodic subdomain via repeated doubling
/// 3. Evaluate the polynomial at the projected point
#[derive(Clone, Copy, Debug, Default)]
pub struct CirclePeriodicEvaluator;

impl CirclePeriodicEvaluator {
    pub const fn new() -> Self {
        Self
    }
}

/// Compute parameters for periodic polynomial evaluation.
/// Returns (log_period, log_repetitions).
fn periodic_params(period: usize, trace_len: usize) -> (usize, usize) {
    debug_assert!(
        period.is_power_of_two(),
        "periodic column length must be a power of 2"
    );

    let log_period = log2_strict_usize(period);
    let log_repetitions = log2_strict_usize(trace_len / period);
    (log_period, log_repetitions)
}

impl<F: ComplexExtendable> PeriodicEvaluator<F, CircleDomain<F>> for CirclePeriodicEvaluator {
    fn eval_on_lde(
        periodic_table: &[Vec<F>],
        trace_domain: &CircleDomain<F>,
        lde_domain: &CircleDomain<F>,
    ) -> PeriodicLdeTable<F> {
        if periodic_table.is_empty() {
            return PeriodicLdeTable::empty();
        }

        let trace_len = trace_domain.size();
        let log_blowup = lde_domain.log_n - trace_domain.log_n;
        let blowup = 1 << log_blowup;

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

        let log_max_period = log2_strict_usize(max_period);
        let log_repetitions = log2_strict_usize(trace_len / max_period);
        let extended_height = max_period * blowup;
        let num_cols = periodic_table.len();

        // Compute the shift for the periodic subdomain at max_period.
        // This aligns the periodic domain with the LDE domain so modular indexing works.
        let extended_shift = lde_domain.shift.repeated_double(log_repetitions);
        let extended_periodic_domain =
            CircleDomain::new(log_max_period + log_blowup, extended_shift);

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

            // Interpolate on the max_period domain
            let periodic_domain = CircleDomain::standard(log_max_period);
            let evals = CircleEvaluations::from_natural_order(
                periodic_domain,
                RowMajorMatrix::new_col(padded),
            );

            // Extrapolate to extended_height using CFFT
            let extended_evals = evals.extrapolate(extended_periodic_domain);
            let extended_values = extended_evals.to_natural_order().to_row_major_matrix();
            columns.push(extended_values.values);
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
        trace_domain: &CircleDomain<F>,
        point: EF,
    ) -> Vec<EF> {
        let trace_len = trace_domain.size();

        periodic_table
            .iter()
            .map(|col| {
                let (log_period, log_repetitions) = periodic_params(col.len(), trace_len);
                let periodic_domain = CircleDomain::standard(log_period);

                let evals = CircleEvaluations::from_natural_order(
                    periodic_domain,
                    RowMajorMatrix::new_col(col.clone()),
                );

                // Project query point to periodic subdomain via repeated doubling
                let query_point = Point::<EF>::from_projective_line(point);
                let periodic_point = query_point.repeated_double(log_repetitions);
                evals.evaluate_at_point(periodic_point)[0]
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use hashbrown::HashMap;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_mersenne_31::Mersenne31;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type F = Mersenne31;
    type EF = BinomialExtensionField<F, 3>;

    #[test]
    fn test_periodic_eval_consistency_random_points() {
        // Test that eval_on_lde and eval_at_point define the same polynomial
        // by checking consistency at random out-of-domain points
        let log_n = 4;
        let log_blowup = 1;
        let trace_domain = CircleDomain::<F>::standard(log_n);
        let lde_domain = CircleDomain::<F>::standard(log_n + log_blowup);
        let lde_len = lde_domain.size();

        // Periodic column: [10, 20, 30, 40]
        let periodic_col = vec![
            F::from_u32(10),
            F::from_u32(20),
            F::from_u32(30),
            F::from_u32(40),
        ];
        let periodic_table = vec![periodic_col];

        // Evaluate on LDE domain
        let lde_table =
            CirclePeriodicEvaluator::eval_on_lde(&periodic_table, &trace_domain, &lde_domain);

        assert_eq!(lde_table.width(), 1);
        // Compact table has height = period * blowup = 4 * 2 = 8
        assert_eq!(lde_table.height(), 8);

        // Expand compact table to full LDE for interpolation test
        let full_lde: Vec<F> = (0..lde_len).map(|i| *lde_table.get(i, 0)).collect();

        // Interpolate the LDE result to get a polynomial we can evaluate anywhere
        let lde_evals =
            CircleEvaluations::from_natural_order(lde_domain, RowMajorMatrix::new_col(full_lde));

        // Test at random out-of-domain points
        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..10 {
            let random_point: EF = rng.random();

            // Evaluate the LDE polynomial at the random point
            let lde_at_point =
                lde_evals.evaluate_at_point(Point::from_projective_line(random_point))[0];

            // Evaluate using eval_at_point directly
            let eval_at_point_result = CirclePeriodicEvaluator::eval_at_point(
                &periodic_table,
                &trace_domain,
                random_point,
            );

            assert_eq!(
                lde_at_point, eval_at_point_result[0],
                "Mismatch at random point: LDE interpolation={:?}, eval_at_point={:?}",
                lde_at_point, eval_at_point_result[0]
            );
        }
    }

    #[test]
    fn test_periodic_eval_at_trace_domain_points() {
        // Test that evaluating the periodic polynomial at trace domain points
        // gives the expected periodic pattern
        let log_n = 4; // 16 rows
        let trace_domain = CircleDomain::<F>::standard(log_n);
        let trace_len = trace_domain.size();
        let period = 4;

        // Periodic column: [1, 2, 3, 4]
        let periodic_col = vec![
            F::from_u32(1),
            F::from_u32(2),
            F::from_u32(3),
            F::from_u32(4),
        ];
        let periodic_table = vec![periodic_col];

        // Evaluate on trace domain (same as LDE with blowup=1)
        let lde_table =
            CirclePeriodicEvaluator::eval_on_lde(&periodic_table, &trace_domain, &trace_domain);

        assert_eq!(lde_table.width(), 1);
        // Compact table has height = period * blowup = 4 * 1 = 4
        assert_eq!(lde_table.height(), 4);

        // Expand compact table to full trace
        let full_trace: Vec<F> = (0..trace_len).map(|i| *lde_table.get(i, 0)).collect();

        // The values should follow a periodic pattern with period 4
        // But the exact mapping depends on Circle domain structure.
        // Verify that we get exactly 4 distinct values, each appearing 4 times.
        let mut value_counts = HashMap::new();
        for &val in &full_trace {
            *value_counts.entry(val).or_insert(0) += 1;
        }
        assert_eq!(
            value_counts.len(),
            period,
            "Expected {} distinct values, got {}",
            period,
            value_counts.len()
        );
        for (val, count) in &value_counts {
            assert_eq!(
                *count, 4,
                "Value {:?} appears {} times, expected 4",
                val, count
            );
        }
    }

    #[test]
    fn test_cfft_extrapolation_matches_naive() {
        // Verify that the CFFT-based eval_on_lde matches point-by-point evaluation
        // using the naive repeated_double approach.
        for (log_n, log_blowup, log_period) in [(4, 1, 2), (5, 2, 2), (6, 1, 3), (8, 2, 4)] {
            let trace_domain = CircleDomain::<F>::standard(log_n);
            let lde_domain = CircleDomain::<F>::standard(log_n + log_blowup);
            let lde_len = lde_domain.size();
            let period = 1 << log_period;
            let log_repetitions = log_n - log_period;

            // Create a periodic column with distinct values
            let periodic_col: Vec<F> = (0..period).map(|i| F::from_u32(i as u32 + 1)).collect();
            let periodic_table = vec![periodic_col.clone()];

            // Evaluate using the optimized CFFT-based method
            let cfft_table =
                CirclePeriodicEvaluator::eval_on_lde(&periodic_table, &trace_domain, &lde_domain);

            // Expand compact table to full LDE
            let cfft_result: Vec<F> = (0..lde_len).map(|i| *cfft_table.get(i, 0)).collect();

            // Evaluate using the naive point-by-point method
            let periodic_domain = CircleDomain::standard(log_period);
            let evals = CircleEvaluations::from_natural_order(
                periodic_domain,
                RowMajorMatrix::new_col(periodic_col.clone()),
            );

            let naive_result: Vec<F> = (0..lde_len)
                .map(|lde_idx| {
                    let lde_point = lde_domain.nth_point(lde_idx);
                    let periodic_point = lde_point.repeated_double(log_repetitions);
                    evals.evaluate_at_point(periodic_point)[0]
                })
                .collect();

            assert_eq!(
                cfft_result, naive_result,
                "CFFT-based and naive methods disagree for log_n={}, log_blowup={}, log_period={}",
                log_n, log_blowup, log_period
            );
        }
    }
}
