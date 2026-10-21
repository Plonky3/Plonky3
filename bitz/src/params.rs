//! The split of the committed bits into rows and columns, and the prover's knobs.

use p3_binary_pcs::whir::CommitmentParams;

use crate::error::ParamsError;

/// Row variables below which a row is not whole words.
const MIN_LOG_ROWS: usize = 6;

/// Row variables past which a fold accumulator lane could carry.
const MAX_LOG_ROWS: usize = 31;

/// `2^n` committed bits as `2^r1` rows of `2^r2` columns: bit `c 2^r1 + b` is row `b` of column `c`.
///
/// Rows are folded into integers, so a column is `2^r1` adjacent bits of whole words.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BitShape {
    /// `r1`.
    log_rows: usize,
    /// `r2`.
    log_columns: usize,
}

impl BitShape {
    /// The shape of `2^r1` rows and `2^r2` columns.
    ///
    /// # Errors
    ///
    /// A row is not whole words, a fold could carry, or the bits cannot be addressed.
    pub const fn new(log_rows: usize, log_columns: usize) -> Result<Self, ParamsError> {
        if log_rows < MIN_LOG_ROWS {
            return Err(ParamsError::RowsTooShort {
                log_rows,
                minimum: MIN_LOG_ROWS,
            });
        }
        if log_rows > MAX_LOG_ROWS {
            return Err(ParamsError::RowsTooLong {
                log_rows,
                maximum: MAX_LOG_ROWS,
            });
        }
        let num_variables = log_rows + log_columns;
        if num_variables >= usize::BITS as usize - 6 {
            return Err(ParamsError::TooManyVariables {
                num_variables,
                maximum: usize::BITS as usize - 7,
            });
        }
        Ok(Self {
            log_rows,
            log_columns,
        })
    }

    /// The fastest split of `2^n` bits, `r1 = ceil(0.6 n) - 1` within `[6, min(n, 31)]` (Section 7).
    ///
    /// # Errors
    ///
    /// Fewer than `2^6` bits, or bits that cannot be addressed.
    pub const fn balanced(num_variables: usize) -> Result<Self, ParamsError> {
        if num_variables < MIN_LOG_ROWS {
            return Err(ParamsError::RowsTooShort {
                log_rows: num_variables,
                minimum: MIN_LOG_ROWS,
            });
        }
        let target = (3 * num_variables).div_ceil(5).saturating_sub(1);
        let ceiling = if num_variables < MAX_LOG_ROWS {
            num_variables
        } else {
            MAX_LOG_ROWS
        };
        let log_rows = if target < MIN_LOG_ROWS {
            MIN_LOG_ROWS
        } else if target > ceiling {
            ceiling
        } else {
            target
        };
        Self::new(log_rows, num_variables.saturating_sub(log_rows))
    }

    /// `r1`, the variables of one column.
    #[must_use]
    pub const fn log_rows(&self) -> usize {
        self.log_rows
    }

    /// `r2`, the variables indexing the columns.
    #[must_use]
    pub const fn log_columns(&self) -> usize {
        self.log_columns
    }

    /// `n = r1 + r2`.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.log_rows + self.log_columns
    }

    /// `l1 = 2^r1`.
    #[must_use]
    pub const fn rows(&self) -> usize {
        1 << self.log_rows
    }

    /// `l2 = 2^r2`.
    #[must_use]
    pub const fn columns(&self) -> usize {
        1 << self.log_columns
    }

    /// Packed words of one column.
    #[must_use]
    pub const fn column_words(&self) -> usize {
        1 << (self.log_rows - MIN_LOG_ROWS)
    }

    /// Packed words of every column.
    #[must_use]
    pub const fn words(&self) -> usize {
        1 << (self.num_variables() - MIN_LOG_ROWS)
    }
}

/// Parameters of the commitment and of the reduction to it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BitzParams {
    /// WHIR parameters of the bit commitment; `security_bits` is the target for the whole proof.
    pub commitment: CommitmentParams,
    /// Grinding before every inner-product challenge.
    pub inner_pow_bits: usize,
}
