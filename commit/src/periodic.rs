//! Periodic column evaluation support.
//!
//! Periodic columns are columns whose values repeat with a period that divides the trace length.
//! This module provides the `PeriodicEvaluator` trait for evaluating periodic polynomials
//! in a domain-agnostic way (supporting both two-adic and circle STARKs).
//!
//! ## Power-of-Two Requirement
//!
//! **All period lengths must be powers of two.** This is because:
//! - The trace domain is a multiplicative/additive group of order `n` (a power of 2)
//! - The periodic subdomain must be a subgroup of order `p`
//! - For `p` to divide `n` as group orders, `p` must also be a power of 2
//!
//! ## Mathematical Background
//!
//! A periodic column with period `p` and trace length `n` repeats every `p` rows:
//! `col[i] = col[i + p]` for all `i`.
//!
//! **The problem**: We have a polynomial `P` of degree `n-1` over the trace domain `H`,
//! but it only takes `p` distinct values. Can we work with a smaller polynomial instead?
//!
//! **Key observation**: We want `P(ω^i) = P(ω^{i+p})` for all `i`. So we need a map
//! `π: H → ?` that identifies points `p` apart: `π(ω^i) = π(ω^{i+p})`, i.e., `π` must
//! be constant on cosets of the subgroup `⟨ω^p⟩` of order `n/p`.
//!
//! **Finding π**: For cyclic groups, raising to the power `k` gives a homomorphism with
//! kernel of size `k`. Since we need `ker(π) = ⟨ω^p⟩` of order `n/p`, we set `π(x) = x^(n/p)`.
//! Indeed, `π(ω^{i+p}) = ω^{(i+p)·n/p} = ω^{i·n/p} · ω^n = π(ω^i)` since `ω^n = 1`.
//!
//! **Where π lands**: The image of `π` is `H_p = {1, ω^(n/p), ω^(2n/p), ...}`, a subgroup
//! of order `p`. Now we can factor `P = Q ∘ π` where `Q: H_p → F` is a degree `p-1`
//! polynomial interpolating the `p` periodic values.
//!
//! **Group-theoretic view**: `π: H → H_p` is a surjective homomorphism with kernel of
//! order `n/p`. By the first isomorphism theorem, `H/ker(π) ≅ H_p`. The periodic column
//! is constant on cosets of `ker(π)`, so it factors through `π`.
//!
//! **For Circle STARKs**: The same idea applies with `π(P) = (n/p)·P` (repeated doubling)
//! instead of exponentiation.
//!
//! **Evaluating at an out-of-domain point `ζ`**:
//! 1. Compute `π(ζ)` to get a point in `H_p`
//! 2. Evaluate `Q(π(ζ))` using Lagrange interpolation over `H_p`
//!
//! ## Memory-Efficient Storage
//!
//! Instead of materializing the full LDE-sized table (which would be wasteful for small periods),
//! we store only `max_period × blowup` rows in a [`PeriodicLdeTable`]. All periodic columns are
//! padded to the maximum period, creating a rectangular matrix that can be efficiently accessed
//! with modular indexing in the constraint evaluation hot loop.

use alloc::vec::Vec;

use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use thiserror::Error;

use crate::PolynomialSpace;

/// Why a declared periodic column cannot be laid over a trace of a given height.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum PeriodicColumnShapeError {
    /// A length with no subgroup of that order to interpolate over.
    #[error("periodic column {index} has length {length}, which is not a power of two")]
    LengthNotPowerOfTwo {
        /// Position of the offending column in the declared order.
        index: usize,
        /// How many values the column lists.
        length: usize,
    },
    /// A length that cannot tile the rows it has to cover.
    #[error(
        "periodic column {index} has length {length}, which does not divide the trace height {height}"
    )]
    LengthNotDividingHeight {
        /// Position of the offending column in the declared order.
        index: usize,
        /// How many values the column lists.
        length: usize,
        /// How many rows the column has to cover.
        height: usize,
    },
}

/// Periodic columns screened against the rows they have to cover.
///
/// A column of length `p` holds the evaluations of one polynomial over a subgroup of order `p`.
///
/// - Such a subgroup exists only when `p` is a power of two.
/// - It tiles the rows only when `p` divides the height.
///
/// ```text
///     height 8,  length 2   [0,1][0,1][0,1][0,1]       tiles
///     height 8,  length 16  [0,1,...,7|8,...,15]       truncated
///     height 12, length 8   [0,...,7][0,1,2,3|4,...]   partial repeat
/// ```
///
/// Row lookups wrap with `row mod p`, so an ill-shaped column still yields a value on every row.
/// Reading rows alone never reveals the mistake.
///
/// Evaluation is where it breaks.
/// Every path divides the height by the length and takes a base-two logarithm of the quotient.
/// Neither step means anything for a shape the rule rejects.
///
/// Holding this view is the evidence that the rule was applied.
#[derive(Debug)]
pub struct PeriodicColumns<'a, F> {
    /// One period of values per declared column, in declaration order.
    columns: &'a [Vec<F>],
    /// Rows the columns were screened against.
    height: usize,
}

// A shared slice and a row count are cheap to copy whatever the cell type is.
// Deriving would tie that to the cell type for no reason.
impl<F> Clone for PeriodicColumns<'_, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F> Copy for PeriodicColumns<'_, F> {}

impl<'a, F> PeriodicColumns<'a, F> {
    /// Screen the declared columns against the rows they have to cover.
    ///
    /// # Errors
    ///
    /// - A length that is not a power of two, zero included.
    /// - A length that does not divide the height.
    pub fn new(columns: &'a [Vec<F>], height: usize) -> Result<Self, PeriodicColumnShapeError> {
        for (index, column) in columns.iter().enumerate() {
            // The length is how many values the column lists before repeating.
            let length = column.len();

            // Powers of two are the orders for which a two-adic subgroup exists.
            // Zero fails here, ahead of the row lookup that would divide by it.
            if !length.is_power_of_two() {
                return Err(PeriodicColumnShapeError::LengthNotPowerOfTwo { index, length });
            }

            // Divisibility lands every repetition on a whole copy of that subgroup.
            if !height.is_multiple_of(length) {
                return Err(PeriodicColumnShapeError::LengthNotDividingHeight {
                    index,
                    length,
                    height,
                });
            }
        }

        Ok(Self { columns, height })
    }

    /// The screened columns, in declaration order.
    pub const fn as_slice(&self) -> &'a [Vec<F>] {
        self.columns
    }

    /// Rows the columns were screened against.
    pub const fn height(&self) -> usize {
        self.height
    }

    /// How many columns are declared.
    pub const fn len(&self) -> usize {
        self.columns.len()
    }

    /// True when the declaration is empty.
    pub const fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// Longest declared period, absent when the declaration is empty.
    ///
    /// Every period divides the height, so the longest one divides it too.
    /// Padding every column up to it yields one rectangular table over a single subgroup.
    pub fn max_period(&self) -> Option<usize> {
        self.columns.iter().map(Vec::len).max()
    }
}

/// Compact storage for periodic column values on the LDE domain.
///
/// Instead of materializing the full LDE-sized table, stores only `extended_height` rows
/// (where `extended_height = max_period × blowup`) and uses modular indexing to access values.
///
/// All periodic columns are padded to the maximum period before extrapolation, creating a
/// rectangular matrix for cache-friendly row-wise access.
///
/// # Invariants
///
/// - All periods must be powers of 2 (see module-level documentation)
/// - Height is always `max_period × blowup` (both powers of 2, so height is power of 2)
#[derive(Clone, Debug)]
pub struct PeriodicLdeTable<F> {
    /// Values in row-major form: height = extended_height, width = num_columns.
    /// Empty if there are no periodic columns.
    values: RowMajorMatrix<F>,
    /// Cached `values.values.len() / values.width` (`0` if `values.width == 0`).
    /// Guaranteed to be a power of two, so `get` can index with `& (height - 1)`
    /// instead of `%`.
    height: usize,
}

impl<F: Clone + Send + Sync> PeriodicLdeTable<F> {
    /// Create a new periodic LDE table from extrapolated values.
    ///
    /// The matrix should have height = `max_period × blowup` and width = `num_periodic_columns`.
    pub const fn new(values: RowMajorMatrix<F>) -> Self {
        let height = match values.values.len().checked_div(values.width) {
            Some(h) => h,
            None => 0,
        };
        debug_assert!(
            height == 0 || height.is_power_of_two(),
            "PeriodicLdeTable height must be a power of two for bitmask indexing"
        );
        Self { values, height }
    }

    /// Create an empty table (for AIRs without periodic columns).
    pub fn empty() -> Self {
        Self {
            values: RowMajorMatrix::new(Vec::new(), 0),
            height: 0,
        }
    }

    /// Returns true if there are no periodic columns.
    pub const fn is_empty(&self) -> bool {
        self.values.values.is_empty()
    }

    /// Number of periodic columns.
    pub const fn width(&self) -> usize {
        self.values.width
    }

    /// Height of the compact table (max_period × blowup).
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Number of distinct packed row groups when the LDE domain is read in groups of
    /// `pack_width` consecutive indices, group `g` starting at `g * pack_width`.
    ///
    /// [`get`](Self::get) reduces indices modulo `height`, and the group starts
    /// `g * pack_width mod height` repeat with period `height / gcd(height, pack_width)`.
    /// Group `g` therefore reads the same values as group `g % packed_group_period(pack_width)`.
    ///
    /// `pack_width` need not be a power of two or divide `height`. Returns `0` for an
    /// empty table.
    pub const fn packed_group_period(&self, pack_width: usize) -> usize {
        debug_assert!(pack_width > 0, "pack_width must be nonzero");
        // `height` is a power of two, so `gcd(height, pack_width)` is the largest power
        // of two dividing `pack_width`, capped at `height`.
        let log_gcd = if self.height.trailing_zeros() < pack_width.trailing_zeros() {
            self.height.trailing_zeros()
        } else {
            pack_width.trailing_zeros()
        };
        self.height >> log_gcd
    }

    /// Get a specific periodic column value for a given LDE index.
    #[inline]
    pub fn get(&self, lde_idx: usize, col_idx: usize) -> &F {
        let height = self.height;
        debug_assert!(height > 0, "cannot index into empty periodic table");
        let row_idx = lde_idx & (height - 1);
        &self.values.values[row_idx * self.values.width + col_idx]
    }
}

/// Evaluates periodic polynomials for a given domain system.
///
/// Periodic columns are defined by their values over one period. This trait
/// handles interpolation and evaluation, abstracting over the domain-specific
/// math (two-adic multiplicative groups vs circle groups).
///
/// # Power-of-Two Requirement
///
/// **All period lengths must be powers of two.** This ensures the periodic subdomain
/// is a valid subgroup of the trace domain. See module-level documentation for details.
///
/// # Type Parameters
/// - `F`: The base field type
/// - `D`: The polynomial space / domain type
pub trait PeriodicEvaluator<F: Field, D: PolynomialSpace<Val = F>> {
    /// Evaluate all periodic columns on the LDE domain, returning a compact table.
    ///
    /// This is used by the prover to compute periodic column values on the
    /// low-degree extension domain for constraint evaluation.
    ///
    /// The returned table stores only `max_period × blowup` rows. All columns are
    /// padded to the maximum period before extrapolation, creating a rectangular
    /// matrix for efficient row-wise access with modular indexing.
    ///
    /// # Arguments
    /// * `periodic_table` - Slice of periodic columns, each containing one period of values.
    ///   The length of each inner `Vec` is the period of that column (must be a power of 2).
    /// * `trace_domain` - The original trace domain
    /// * `lde_domain` - The low-degree extension domain
    ///
    /// # Returns
    /// A [`PeriodicLdeTable`] with height = `max_period × blowup` and width = number of columns.
    fn eval_on_lde(
        periodic_table: &[Vec<F>],
        trace_domain: &D,
        lde_domain: &D,
    ) -> PeriodicLdeTable<F>;

    /// Evaluate all periodic columns at a single point (for verification).
    ///
    /// This is used by the verifier to compute periodic column values at
    /// query points during constraint verification.
    ///
    /// # Arguments
    /// * `periodic_table` - Slice of periodic columns. Each column's length (period)
    ///   must be a power of 2.
    /// * `trace_domain` - The original trace domain
    /// * `point` - The query point (in extension field)
    ///
    /// # Returns
    /// `Vec<EF>` containing the evaluation of each periodic column at `point`
    fn eval_at_point<EF: ExtensionField<F>>(
        periodic_table: &[Vec<F>],
        trace_domain: &D,
        point: EF,
    ) -> Vec<EF>;
}

/// Unit type implements `PeriodicEvaluator` as a no-op.
///
/// This is used internally by `prove` and `verify` for AIRs without periodic columns.
/// Panics if any periodic columns are present.
impl<F: Field, D: PolynomialSpace<Val = F>> PeriodicEvaluator<F, D> for () {
    fn eval_on_lde(
        periodic_table: &[Vec<F>],
        _trace_domain: &D,
        _lde_domain: &D,
    ) -> PeriodicLdeTable<F> {
        assert!(
            periodic_table.is_empty(),
            "AIR has periodic columns but no PeriodicEvaluator was specified. \
             Use prove_with_periodic or verify_with_periodic with TwoAdicPeriodicEvaluator \
             or CirclePeriodicEvaluator."
        );
        PeriodicLdeTable::empty()
    }

    fn eval_at_point<EF: ExtensionField<F>>(
        periodic_table: &[Vec<F>],
        _trace_domain: &D,
        _point: EF,
    ) -> Vec<EF> {
        assert!(
            periodic_table.is_empty(),
            "AIR has periodic columns but no PeriodicEvaluator was specified. \
             Use prove_with_periodic or verify_with_periodic with TwoAdicPeriodicEvaluator \
             or CirclePeriodicEvaluator."
        );
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    // An AIR without periodic columns imposes nothing on the height.
    // Heights that no column could ever divide still pass.
    #[test]
    fn no_columns_accepts_any_height() {
        for height in [0, 1, 3, 7, 12] {
            let screened = PeriodicColumns::<u8>::new(&[], height).unwrap();

            assert!(screened.is_empty());
            assert_eq!(screened.len(), 0);
            assert_eq!(screened.height(), height);
            assert_eq!(screened.max_period(), None);
        }
    }

    // Fixture state: 8 rows, every power-of-two length up to the height.
    //
    //     length 1 -> 8 repeats, length 2 -> 4, length 4 -> 2, length 8 -> 1
    #[test]
    fn every_power_of_two_divisor_of_the_height_is_accepted() {
        for length in [1, 2, 4, 8] {
            let columns = vec![vec![0u8; length]];
            let screened = PeriodicColumns::new(&columns, 8).unwrap();

            assert_eq!(screened.max_period(), Some(length));
            assert_eq!(screened.as_slice(), columns.as_slice());
        }
    }

    // Padding to the longest period is what makes the columns one rectangular table.
    //
    //     lengths [2, 8, 4]  ->  longest 8
    #[test]
    fn the_longest_period_is_reported() {
        let columns = vec![vec![0u8; 2], vec![0u8; 8], vec![0u8; 4]];
        let screened = PeriodicColumns::new(&columns, 8).unwrap();

        assert_eq!(screened.len(), 3);
        assert_eq!(screened.max_period(), Some(8));
    }

    // Three values cannot be the evaluations of a polynomial over a two-adic subgroup.
    // The report names the column so an AIR with many of them stays diagnosable.
    #[test]
    fn non_power_of_two_length_is_rejected() {
        let columns = vec![vec![0u8; 3]];

        assert_eq!(
            PeriodicColumns::new(&columns, 8).unwrap_err(),
            PeriodicColumnShapeError::LengthNotPowerOfTwo {
                index: 0,
                length: 3
            }
        );
    }

    // An empty column would make the row lookup divide by zero.
    // Zero is not a power of two, so it is caught by the same arm.
    #[test]
    fn empty_column_is_rejected() {
        let columns: Vec<Vec<u8>> = vec![vec![]];

        assert_eq!(
            PeriodicColumns::new(&columns, 8).unwrap_err(),
            PeriodicColumnShapeError::LengthNotPowerOfTwo {
                index: 0,
                length: 0
            }
        );
    }

    // Mutation: 8 values over 12 rows.
    //
    //     [0,...,7][0,1,2,3|4,...]  <- the second repeat is cut in half
    //
    // Eight fits inside twelve, so a bound that only compares sizes would accept this.
    // Divisibility is the relation that matters, and it fails.
    #[test]
    fn length_that_fits_but_does_not_divide_is_rejected() {
        let columns = vec![vec![0u8; 8]];

        assert_eq!(
            PeriodicColumns::new(&columns, 12).unwrap_err(),
            PeriodicColumnShapeError::LengthNotDividingHeight {
                index: 0,
                length: 8,
                height: 12
            }
        );
    }

    // Columns are screened in declaration order, so the first bad one is the one reported.
    //
    //     column 0: length 4  ok
    //     column 1: length 6  not a power of two  <- reported
    //     column 2: length 5  never reached
    #[test]
    fn the_first_offending_column_is_the_one_reported() {
        let columns = vec![vec![0u8; 4], vec![0u8; 6], vec![0u8; 5]];

        assert_eq!(
            PeriodicColumns::new(&columns, 8).unwrap_err(),
            PeriodicColumnShapeError::LengthNotPowerOfTwo {
                index: 1,
                length: 6
            }
        );
    }

    // Both in-repo trace domains have a power-of-two size.
    // For p = 2^a and n = 2^b, p divides n iff a <= b iff p <= n.
    // So on such a height, "fits inside" and "divides" are the same predicate.
    #[test]
    fn on_a_power_of_two_height_fitting_and_dividing_agree() {
        for log_height in 0..16 {
            let height = 1usize << log_height;
            for log_length in 0..16 {
                let length = 1usize << log_length;
                let columns = vec![vec![0u8; length]];

                let fits = length <= height;
                let divides = PeriodicColumns::new(&columns, height).is_ok();

                assert_eq!(fits, divides, "height {height}, length {length}");
            }
        }
    }

    #[test]
    fn packed_group_period_matches_modular_indexing() {
        // (height, pack_width, expected period): widths that are not powers of two, or
        // that do not divide the height, visit every residue class before repeating.
        let cases = [
            (8, 3, 8),
            (8, 6, 4),
            (8, 1, 8),
            (8, 4, 2),
            (8, 8, 1),
            (4, 8, 1),
            (1, 3, 1),
        ];
        for (height, pack_width, expected) in cases {
            let values: Vec<u32> = (0..height).map(|i| i as u32).collect();
            let table = PeriodicLdeTable::new(RowMajorMatrix::new(values, 1));
            let period = table.packed_group_period(pack_width);
            assert_eq!(period, expected, "height {height}, pack_width {pack_width}");

            for group in 0..4 * height {
                let cached = group % period;
                for offset in 0..pack_width {
                    assert_eq!(
                        table.get(group * pack_width + offset, 0),
                        table.get(cached * pack_width + offset, 0),
                        "height {height}, pack_width {pack_width}, group {group}, offset {offset}"
                    );
                }
            }
        }

        assert_eq!(PeriodicLdeTable::<u32>::empty().packed_group_period(3), 0);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "PeriodicLdeTable height must be a power of two")]
    fn new_panics_on_non_power_of_two_height() {
        use alloc::vec;

        use p3_baby_bear::BabyBear;
        use p3_field::PrimeCharacteristicRing;

        use super::*;

        type F = BabyBear;

        let (a, b, c) = (F::ONE, F::TWO, F::from_u8(3));
        let _ = PeriodicLdeTable::new(RowMajorMatrix::new(vec![a, b, c], 1));
    }
}
