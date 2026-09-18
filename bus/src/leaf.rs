//! Direction-aware materialization of multiset fingerprint factors.
//!
//! A width-`2^s` tuple is interpreted as a Boolean-cube evaluation table.
//! Its fingerprint is the table's multilinear extension at a random point in `F^s`.
//!
//! For unequal multisets, subtract the push and pull products as a polynomial in both challenges.
//! Unique factorization makes this a nonzero polynomial.
//! Its total degree is at most `max(1, s) * N`.
//! Here `N` is the larger active multiset size.
//! Schwartz--Zippel gives collision probability at most `max(1, s) * N / |F|`.
//! Both challenges must be sampled after the tuple columns are committed.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::{ExtensionField, Field};
use thiserror::Error;

/// The side of a multiset equality to which an entry belongs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BusDirection {
    /// Add the entry to the multiset being produced.
    Push,
    /// Add the entry to the multiset being consumed.
    Pull,
}

/// Row activation for one bus declaration.
#[derive(Clone, Copy, Debug)]
pub enum BusSelector<'a, F> {
    /// Every row contributes one factor.
    Always,
    /// A Boolean-constrained column selects the rows that contribute factors.
    Boolean(&'a [F]),
}

/// One column-major family of equal-width bus tuples.
#[derive(Clone, Copy, Debug)]
pub struct BusLeafDeclaration<'a, F> {
    /// The multiset side receiving these rows.
    pub direction: BusDirection,
    /// Tuple columns in slot order.
    pub columns: &'a [&'a [F]],
    /// Row activation mode.
    pub selector: BusSelector<'a, F>,
}

/// Materialized product leaves, kept separate by direction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BusLeaves<EF> {
    /// Produced factors in declaration order and then row order.
    pub pushes: Vec<EF>,
    /// Consumed factors in declaration order and then row order.
    pub pulls: Vec<EF>,
}

/// Invalid declarations rejected before any product proof is built.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum BusLeafError {
    /// The fingerprint point describes a tuple width that cannot fit in memory.
    #[error("bus tuple width overflows usize")]
    TupleWidthOverflow,
    /// A declaration has a different tuple width from the fingerprint challenge.
    #[error("bus declaration {declaration} has width {actual}, expected {expected}")]
    TupleWidthMismatch {
        /// Position of the malformed declaration.
        declaration: usize,
        /// Width fixed by the challenge point.
        expected: usize,
        /// Width carried by the declaration.
        actual: usize,
    },
    /// Tuple columns within one declaration have different row counts.
    #[error("bus declaration {declaration} column {column} has {actual} rows, expected {expected}")]
    ColumnHeightMismatch {
        /// Position of the malformed declaration.
        declaration: usize,
        /// Position of the malformed column.
        column: usize,
        /// Height fixed by the first column.
        expected: usize,
        /// Height carried by the malformed column.
        actual: usize,
    },
    /// A selector does not cover the same rows as its tuple columns.
    #[error("bus declaration {declaration} selector has {actual} rows, expected {expected}")]
    SelectorHeightMismatch {
        /// Position of the malformed declaration.
        declaration: usize,
        /// Height fixed by the tuple columns.
        expected: usize,
        /// Height carried by the selector.
        actual: usize,
    },
    /// A selected-row marker is not zero or one.
    #[error("bus declaration {declaration} selector row {row} is not Boolean")]
    NonBooleanSelector {
        /// Position of the malformed declaration.
        declaration: usize,
        /// Position of the malformed row.
        row: usize,
    },
}

/// Materialize one fingerprint factor per active tuple.
///
/// The tuple fingerprint is its multilinear extension at the supplied point.
/// A factor is `offset - fingerprint`.
/// An inactive row contributes the multiplicative identity.
///
/// The tuple width is exactly `2^point.len()`.
/// This function does not separate distinct named buses.
/// A caller combining buses must reserve tuple slots for an injective domain separator.
///
/// Boolean selection is checked here for honest-prover diagnostics.
/// The surrounding AIR must also constrain every selector to be Boolean.
///
/// Each direction concatenates declarations without alignment padding.
/// The offset of one declaration is the sum of earlier row counts on that direction.
///
/// # Errors
///
/// Returns an error for inconsistent widths, heights, or non-Boolean selectors.
pub fn materialize_bus_leaves<F, EF>(
    declarations: &[BusLeafDeclaration<'_, F>],
    point: &[EF],
    offset: EF,
) -> Result<BusLeaves<EF>, BusLeafError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    // The point fixes the only tuple width that the fingerprint polynomial accepts.
    let shift = u32::try_from(point.len()).map_err(|_| BusLeafError::TupleWidthOverflow)?;
    let width = 1usize
        .checked_shl(shift)
        .ok_or(BusLeafError::TupleWidthOverflow)?;

    // Validate widths before allocating the challenge-sized equality table.
    for (declaration_index, declaration) in declarations.iter().enumerate() {
        if declaration.columns.len() != width {
            return Err(BusLeafError::TupleWidthMismatch {
                declaration: declaration_index,
                expected: width,
                actual: declaration.columns.len(),
            });
        }
    }

    // One equality table supplies the linear coefficient of every tuple slot.
    let weights = equality_weights(point);
    debug_assert_eq!(weights.len(), width);

    // Direction is metadata rather than a field sign.
    // This keeps push and pull distinct in characteristic two.
    let mut pushes = Vec::new();
    let mut pulls = Vec::new();

    for (declaration_index, declaration) in declarations.iter().enumerate() {
        // An empty tuple width is impossible because powers of two start at one.
        let height = declaration.columns[0].len();
        for (column_index, column) in declaration.columns.iter().enumerate().skip(1) {
            if column.len() != height {
                return Err(BusLeafError::ColumnHeightMismatch {
                    declaration: declaration_index,
                    column: column_index,
                    expected: height,
                    actual: column.len(),
                });
            }
        }

        // A selector covers exactly the rows whose tuples it activates.
        if let BusSelector::Boolean(selector) = declaration.selector
            && selector.len() != height
        {
            return Err(BusLeafError::SelectorHeightMismatch {
                declaration: declaration_index,
                expected: height,
                actual: selector.len(),
            });
        }

        let destination = match declaration.direction {
            BusDirection::Push => &mut pushes,
            BusDirection::Pull => &mut pulls,
        };
        destination.reserve(height);

        for row in 0..height {
            // The equality weights evaluate the tuple's multilinear extension.
            let fingerprint = declaration
                .columns
                .iter()
                .zip(&weights)
                .map(|(column, &weight)| weight * column[row])
                .sum::<EF>();
            let factor = offset - fingerprint;

            // Selection uses `1 + s * (factor - 1)`.
            // Both Boolean values therefore avoid a branch in the hot loop.
            let selected = match declaration.selector {
                BusSelector::Always => factor,
                BusSelector::Boolean(selector) => {
                    let selector = selector[row];
                    if selector * selector != selector {
                        return Err(BusLeafError::NonBooleanSelector {
                            declaration: declaration_index,
                            row,
                        });
                    }
                    EF::ONE + (factor - EF::ONE) * selector
                }
            };
            destination.push(selected);
        }
    }

    Ok(BusLeaves { pushes, pulls })
}

/// Evaluate the Boolean-cube equality polynomial at every tuple slot.
fn equality_weights<F: Field>(point: &[F]) -> Vec<F> {
    // The empty point addresses the sole slot of a width-one tuple.
    let mut weights = vec![F::ONE];

    for &coordinate in point {
        // Existing coordinates remain the low-order address bits.
        // The new coordinate selects between two complete halves.
        let old_len = weights.len();
        weights.resize(old_len * 2, F::ZERO);
        for index in 0..old_len {
            let weight = weights[index];
            weights[index] = weight * (F::ONE - coordinate);
            weights[old_len + index] = weight * coordinate;
        }
    }

    weights
}
