//! What the reduction sends, and the claims it leaves behind.

use alloc::vec::Vec;

use p3_multilinear_util::point::Point;
use p3_sumcheck::generic_degree::GenericDegreeProof;
use serde::{Deserialize, Serialize};

use crate::fractional_gkr::FractionGkrProof;

/// Everything the reduction sends.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogupStarProof<F, EF> {
    /// One pushforward per table, in statement order.
    ///
    /// These travel in the clear, since they are only as wide as the table.
    ///
    /// A verifier evaluates them wherever it needs to rather than opening them.
    pub pushforwards: Vec<Vec<EF>>,
    /// The reduction proving that each pushforward is the one the positions induce.
    pub fraction_gkr: FractionGkrProof<EF>,
    /// Each reader's position-column value at the point that reduction landed on.
    pub position_claims: Vec<EF>,
    /// The sumcheck binding every table to its pushforward.
    pub product: GenericDegreeProof<F, EF>,
    /// Each table's column values at the point that sumcheck landed on.
    pub column_claims: Vec<Vec<EF>>,
}

/// The evaluation claims one reduction hands back to its caller.
///
/// Nothing here is authenticated.
///
/// What is proved is that these claims and the incoming ones stand or fall together.
///
/// Discharging them against the commitments is the caller's job.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogupStarOutput<EF> {
    /// Point the position-column claims are drawn from, spanning the largest reader.
    ///
    /// A reader over `n` variables is claimed at the last `n` coordinates.
    pub position_point: Point<EF>,
    /// Point the table-column claims are drawn from, spanning the largest table.
    ///
    /// A table over `m` variables is claimed at the last `m` coordinates.
    pub table_point: Point<EF>,
    /// One entry per table, in the statement's table order.
    pub tables: Vec<TableOutput<EF>>,
}

/// The claims belonging to one table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableOutput<EF> {
    /// Claimed value of each table column, in column order.
    pub column_claims: Vec<EF>,
    /// Claimed value of each reader's position column, in the table's reader order.
    pub position_claims: Vec<EF>,
}
