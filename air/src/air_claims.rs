use alloc::vec::Vec;

/// Structured output from AIR constraint evaluation / opening.
///
/// Separates **AIR-level semantics** from **proof-system internals**.
///
/// # What it captures
///
/// - Which trace columns were evaluated
/// - At which points (current row vs. next row)
/// - For both main and preprocessed traces
///
/// # What it excludes
///
/// - Quotient polynomial chunks
/// - PCS commitments
/// - Randomization columns
///
/// # Why
///
/// Without this type, opened values mix AIR semantics with proof mechanics
/// in a flat struct.
/// As the system grows to support multi-AIR composition and batch proving,
/// a clean separation reduces coupling and makes the data flow easier to
/// reason about.
///
/// # Compatibility
///
/// Conversion traits are provided alongside proof types.
/// Existing code can adopt this type incrementally without breaking changes.
#[derive(Debug, Clone)]
pub struct AirClaims<F> {
    /// Opened main trace column evaluations at the challenge point.
    ///
    /// One element per main trace column, in column order.
    pub main_evals: Vec<F>,

    /// Opened main trace column evaluations at the next-row point.
    ///
    /// `None` when:
    /// - the AIR has no transition constraints and
    /// - never accesses the next row.
    /// This avoids unnecessary polynomial openings in single-row AIRs.
    pub main_next_evals: Option<Vec<F>>,

    /// Opened preprocessed column evaluations at the challenge point.
    ///
    /// `None` when the AIR has no preprocessed (fixed) trace columns.
    pub preprocessed_evals: Option<Vec<F>>,

    /// Opened preprocessed column evaluations at the next-row point.
    ///
    /// `None` when:
    /// - the AIR has no preprocessed trace or,
    /// - never accesses the next preprocessed row.
    pub preprocessed_next_evals: Option<Vec<F>>,
}
