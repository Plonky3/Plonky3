//! Indexed lookups through a pushforward, sound in every characteristic.
//!
//! A reader pulls values out of a table by naming the entry it reads.
//!
//! ```text
//!     X[i] = T[I[i]]
//! ```
//!
//! The caller arrives with a claim about `X` and leaves with claims about `T` and `I`.
//!
//! # The pulled values are never materialized
//!
//! Pullback and pushforward are adjoint, which turns the claim inside out:
//!
//! ```text
//!     X(r) = <X, eq_r> = <T, I_* eq_r> = <T, Y>
//! ```
//!
//! The left inner product runs over the reader's rows, the right one over the table's entries.
//!
//! So `Y` is the only new object, and it is as wide as the table rather than the reader.
//!
//! Two checks close it.
//!
//! One is that `Y` really is the pushforward, as a single sum of fractions:
//!
//! ```text
//!     sum_i eq_r(i) / (c - I(i))  -  sum_v Y(v) / (c - iota(v))  =  0
//! ```
//!
//! The other is `<T, Y> = e`, as one sumcheck over the table's entries.
//!
//! # Why this survives characteristic two
//!
//! A logarithmic-derivative lookup puts an integer count over each entry.
//!
//! Counts are read back modulo the characteristic, so modulo two they say only odd or even.
//!
//! Here the weight over an entry is `sum_{i : I[i] = v} eq_r(i)`.
//!
//! That is a field element, and no reduction modulo anything ever touches it.
//!
//! # References
//!
//! - Soukhanov. Logup*: faster, cheaper logup argument for small-table indexed lookups. <https://eprint.iacr.org/2025/946>

mod error;
mod plan;
pub mod position;
mod product;
mod proof;
mod prover;
pub mod transcript;
mod verifier;
mod witness;

#[cfg(test)]
mod tests;

pub use error::LogupStarError;
use p3_multilinear_util::point::Point;
pub use plan::LogupStarPlan;
pub use proof::{LogupStarOutput, LogupStarProof, TableOutput};

/// One reader's claim on the values it pulled out of a table.
#[derive(Clone, Copy, Debug)]
pub struct Reader<'a, EF> {
    /// Point at which every pulled column is claimed.
    pub point: &'a Point<EF>,
    /// Claimed value of each pulled column at that point, in table-column order.
    pub claims: &'a [EF],
}

/// One table together with every reader that pulls from it.
#[derive(Clone, Copy, Debug)]
pub struct TableLookup<'a, EF> {
    /// Base-two logarithm of the number of table entries.
    pub num_variables: usize,
    /// Every reader of this table, in an order both sides agree on.
    pub readers: &'a [Reader<'a, EF>],
}

impl<EF> TableLookup<'_, EF> {
    /// Number of columns each table entry carries.
    ///
    /// Every reader pulls the whole entry, so its claim count is that width.
    pub fn width(&self) -> usize {
        self.readers.first().map_or(0, |reader| reader.claims.len())
    }
}

/// Prover data behind one reader.
#[derive(Clone, Copy, Debug)]
pub struct ReaderWitness<'a> {
    /// Table entry each row pulls, one per row.
    ///
    /// The committed column is this list under the field's own enumeration of the integers.
    ///
    /// That is how the reduction rebuilds it.
    pub positions: &'a [usize],
}

/// Prover data behind one table.
#[derive(Clone, Copy, Debug)]
pub struct TableWitness<'a, F> {
    /// One column per value a table entry carries, each holding one value per entry.
    pub columns: &'a [&'a [F]],
    /// Prover data behind each reader, in the statement's reader order.
    pub readers: &'a [ReaderWitness<'a>],
}
