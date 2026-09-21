//! Shape rule every periodic column an AIR declares has to satisfy.

use alloc::vec::Vec;

use thiserror::Error;

/// Why a periodic column cannot be laid over a trace of a given height.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum PeriodicColumnError {
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

/// Reject periodic columns a trace of the given height cannot carry.
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
/// Reading rows alone never reveals the mistake, so the shape is screened up front.
///
/// # Errors
///
/// - A length that is not a power of two, zero included.
/// - A length that does not divide the height.
pub fn check_periodic_column_lengths<F>(
    columns: &[Vec<F>],
    height: usize,
) -> Result<(), PeriodicColumnError> {
    for (index, column) in columns.iter().enumerate() {
        // The length is how many values the column lists before repeating.
        let length = column.len();

        // Powers of two are the orders for which a two-adic subgroup exists.
        // An empty column fails here, ahead of the row lookup that would divide by its length.
        if !length.is_power_of_two() {
            return Err(PeriodicColumnError::LengthNotPowerOfTwo { index, length });
        }

        // Divisibility lands every repetition on a whole copy of that subgroup.
        if !height.is_multiple_of(length) {
            return Err(PeriodicColumnError::LengthNotDividingHeight {
                index,
                length,
                height,
            });
        }
    }

    Ok(())
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
            assert_eq!(check_periodic_column_lengths::<u8>(&[], height), Ok(()));
        }
    }

    // Fixture state: 8 rows, every power-of-two length up to the height.
    //
    //     length 1 -> 8 repeats, length 2 -> 4, length 4 -> 2, length 8 -> 1
    #[test]
    fn every_power_of_two_divisor_of_the_height_is_accepted() {
        for length in [1, 2, 4, 8] {
            let columns = vec![vec![0u8; length]];
            assert_eq!(check_periodic_column_lengths(&columns, 8), Ok(()));
        }
    }

    // Three values cannot be the evaluations of a polynomial over a two-adic subgroup.
    // The report names the column so an AIR with many of them stays diagnosable.
    #[test]
    fn non_power_of_two_length_is_rejected() {
        let columns = vec![vec![0u8; 3]];
        assert_eq!(
            check_periodic_column_lengths(&columns, 8),
            Err(PeriodicColumnError::LengthNotPowerOfTwo {
                index: 0,
                length: 3
            })
        );
    }

    // An empty column would make the row lookup divide by zero.
    // Zero is not a power of two, so it is caught by the same arm.
    #[test]
    fn empty_column_is_rejected() {
        let columns: Vec<Vec<u8>> = vec![vec![]];
        assert_eq!(
            check_periodic_column_lengths(&columns, 8),
            Err(PeriodicColumnError::LengthNotPowerOfTwo {
                index: 0,
                length: 0
            })
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
            check_periodic_column_lengths(&columns, 12),
            Err(PeriodicColumnError::LengthNotDividingHeight {
                index: 0,
                length: 8,
                height: 12
            })
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
            check_periodic_column_lengths(&columns, 8),
            Err(PeriodicColumnError::LengthNotPowerOfTwo {
                index: 1,
                length: 6
            })
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
                let divides = check_periodic_column_lengths(&columns, height).is_ok();

                assert_eq!(fits, divides, "height {height}, length {length}");
            }
        }
    }
}
