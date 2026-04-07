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

use crate::PolynomialSpace;

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
}

impl<F: Clone + Send + Sync> PeriodicLdeTable<F> {
    /// Create a new periodic LDE table from extrapolated values.
    ///
    /// The matrix should have height = `max_period × blowup` and width = `num_periodic_columns`.
    pub const fn new(values: RowMajorMatrix<F>) -> Self {
        Self { values }
    }

    /// Create an empty table (for AIRs without periodic columns).
    pub fn empty() -> Self {
        Self {
            values: RowMajorMatrix::new(Vec::new(), 0),
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
        if self.values.width == 0 {
            0
        } else {
            self.values.values.len() / self.values.width
        }
    }

    /// Get all periodic column values for a given LDE index using modular indexing.
    ///
    /// Returns a slice of length `width()` containing the value of each periodic column.
    #[inline]
    pub fn get_row(&self, lde_idx: usize) -> &[F] {
        let height = self.height();
        debug_assert!(height > 0, "cannot index into empty periodic table");
        let row_idx = lde_idx % height;
        let start = row_idx * self.values.width;
        let end = start + self.values.width;
        &self.values.values[start..end]
    }

    /// Get a specific periodic column value for a given LDE index.
    #[inline]
    pub fn get(&self, lde_idx: usize, col_idx: usize) -> &F {
        let height = self.height();
        debug_assert!(height > 0, "cannot index into empty periodic table");
        let row_idx = lde_idx % height;
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
