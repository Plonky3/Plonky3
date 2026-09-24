//! Starting contents of a timestamped memory.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;
use p3_util::log2_ceil_usize;

use super::{TimestampedMemory, TimestampedMemoryError};

/// Starting contents the verifier knows, as sparse runs of words over `2^log_cells` cells.
///
/// Every cell outside a run starts at zero.
///
/// A word holds `value_width` components, stored back to back inside its run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicImage<F> {
    /// Base-two logarithm of the number of cells the image covers.
    log_cells: usize,
    /// Number of field components in one word.
    value_width: usize,
    /// Runs in increasing cell order: the first cell, then the words.
    runs: Vec<(usize, Vec<F>)>,
}

impl<F: Field> PublicImage<F> {
    /// Builds an image of `2^log_cells` cells from runs of consecutive words.
    ///
    /// Each run is its first cell, then its words back to back.
    ///
    /// # Errors
    ///
    /// - A cell count that overflows a machine word.
    /// - An empty run, or one that does not hold whole words.
    /// - A run that starts before the previous one ends, or ends past the last cell.
    pub fn new<C: Field>(
        memory: &TimestampedMemory<C, F>,
        log_cells: usize,
        runs: Vec<(usize, Vec<F>)>,
    ) -> Result<Self, TimestampedMemoryError>
    where
        F: ExtensionField<C>,
    {
        let cells = u32::try_from(log_cells)
            .ok()
            .and_then(|bits| 1usize.checked_shl(bits))
            .ok_or(TimestampedMemoryError::ImageTooLarge { log_cells })?;
        let value_width = memory.value_width();

        // Runs are sorted and disjoint, so each cell has at most one starting word.
        let mut free = 0;
        for (run, (start, words)) in runs.iter().enumerate() {
            if words.is_empty() || words.len() % value_width != 0 {
                return Err(TimestampedMemoryError::ImageRunWidth {
                    run,
                    len: words.len(),
                    value_width,
                });
            }
            if *start < free {
                return Err(TimestampedMemoryError::OverlappingImageRuns { run });
            }
            free = start
                .checked_add(words.len() / value_width)
                .filter(|&end| end <= cells)
                .ok_or(TimestampedMemoryError::ImageRunOutOfRange { run, cells })?;
        }

        Ok(Self {
            log_cells,
            value_width,
            runs,
        })
    }

    /// Base-two logarithm of the number of cells the image covers.
    #[must_use]
    pub const fn log_cells(&self) -> usize {
        self.log_cells
    }

    /// Number of field components in one word.
    #[must_use]
    pub const fn value_width(&self) -> usize {
        self.value_width
    }

    /// Runs in increasing cell order: the first cell, then the words.
    #[must_use]
    pub fn runs(&self) -> &[(usize, Vec<F>)] {
        &self.runs
    }

    /// Every cell's starting value, one full column per word component.
    #[must_use]
    pub fn columns(&self) -> Vec<Vec<F>> {
        let mut columns = vec![F::zero_vec(1 << self.log_cells); self.value_width];
        for (start, words) in &self.runs {
            for (offset, word) in words.chunks_exact(self.value_width).enumerate() {
                for (column, &value) in columns.iter_mut().zip(word) {
                    column[start + offset] = value;
                }
            }
        }
        columns
    }

    /// Evaluates every component column's multilinear extension at `point`.
    ///
    /// Coordinate zero binds the most significant bit of the cell index.
    ///
    /// The zero cells cost nothing:
    ///
    /// ```text
    ///     value(r) = sum over run words z of  eq(z, r) * value(z)
    /// ```
    ///
    /// A run of `L` words costs `O(L + log_cells)`.
    ///
    /// # Panics
    ///
    /// Panics when `point` does not have one coordinate per cell-index bit.
    #[must_use]
    pub fn evaluate<EF: ExtensionField<F>>(&self, point: &[EF]) -> Vec<EF> {
        assert_eq!(point.len(), self.log_cells, "point dimension");
        let mut values = EF::zero_vec(self.value_width);
        for (start, words) in &self.runs {
            // Split the index at the run's power-of-two length.
            //
            //     eq(z, r) = eq(z_high, r_high) * eq(z_low, r_low)
            //
            // The low table costs at most twice the run length.
            let len = words.len() / self.value_width;
            let low_bits = log2_ceil_usize(len);
            let (high, low) = point.split_at(self.log_cells - low_bits);
            let low_eq = Point::new(low).equality_weights_msb();
            let low_mask = (1 << low_bits) - 1;

            // A run of at most `2^low_bits` cells crosses at most one window boundary.
            let mut window = None;
            let mut high_eq = EF::ZERO;
            for (offset, word) in words.chunks_exact(self.value_width).enumerate() {
                let cell = start + offset;
                if window != Some(cell >> low_bits) {
                    window = Some(cell >> low_bits);
                    high_eq = Point::new(high)
                        .equality_at_vertex(cell >> low_bits)
                        .expect("a checked run stays inside the image");
                }
                let weight = high_eq * low_eq[cell & low_mask];
                for (value, &component) in values.iter_mut().zip(word) {
                    *value += weight * component;
                }
            }
        }
        values
    }
}

/// Cells whose starting values only the prover knows.
///
/// The region covers `2^log_cells` consecutive cells from `first_cell`.
///
/// Its starting values are committed columns of the boundary block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrivateRegion {
    /// Index of the first cell.
    first_cell: usize,
    /// Base-two logarithm of the number of cells.
    log_cells: usize,
}

impl PrivateRegion {
    /// Builds the region of `2^log_cells` cells starting at cell `first_cell`.
    ///
    /// Returns `None` when the last cell index overflows a machine word.
    #[must_use]
    pub const fn new(first_cell: usize, log_cells: usize) -> Option<Self> {
        if log_cells >= usize::BITS as usize {
            return None;
        }
        match first_cell.checked_add((1 << log_cells) - 1) {
            Some(_) => Some(Self {
                first_cell,
                log_cells,
            }),
            None => None,
        }
    }

    /// Index of the last cell.
    #[must_use]
    pub const fn last_cell(&self) -> usize {
        // Add the span, not the length: a region may end exactly at `usize::MAX`.
        self.first_cell + ((1 << self.log_cells) - 1)
    }

    /// Index of the first cell.
    #[must_use]
    pub const fn first_cell(&self) -> usize {
        self.first_cell
    }

    /// Base-two logarithm of the number of cells.
    #[must_use]
    pub const fn log_cells(&self) -> usize {
        self.log_cells
    }
}
