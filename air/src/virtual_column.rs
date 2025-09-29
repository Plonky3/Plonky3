use alloc::vec;
use alloc::vec::Vec;
use core::ops::Mul;

use p3_field::{Field, PrimeCharacteristicRing};

/// An affine linear combination of columns in a PAIR (Preprocessed AIR).
///
/// This structure represents an affine function `f(x) = Σ(w_i * x_i) + c` where:
/// - `w_i` are the column weights
/// - `x_i` are the column values (either preprocessed or main trace columns)
/// - `c` is a constant term
///
/// Virtual columns are useful for creating derived values from existing trace columns
/// without explicitly storing them in the trace matrix, saving memory and computation.
#[derive(Clone, Debug)]
pub struct VirtualPairCol<F: Field> {
    /// Linear combination coefficients: pairs of (column, weight).
    column_weights: Vec<(PairCol, F)>,
    /// Constant term added to the linear combination.
    constant: F,
}

/// A reference to a column in a PAIR (Preprocessed AIR).
#[derive(Clone, Copy, Debug)]
pub enum PairCol {
    /// A preprocessed (fixed) column at the specified index.
    ///
    /// These columns contain values that are determined during the setup phase
    /// and remain constant across all proof generations.
    Preprocessed(usize),
    /// A main trace column at the specified index.
    ///
    /// These columns contain witness values that vary between different executions
    /// and are filled during trace generation.
    Main(usize),
}

impl PairCol {
    /// Retrieves the value from the appropriate column array based on the column type.
    ///
    /// # Arguments
    /// * `preprocessed` - Slice containing preprocessed column values
    /// * `main` - Slice containing main trace column values
    ///
    /// # Returns
    /// The value at the column index from the appropriate array
    ///
    /// # Panics
    /// Panics if the column index is out of bounds for the respective array.
    pub const fn get<T: Copy>(&self, preprocessed: &[T], main: &[T]) -> T {
        match self {
            Self::Preprocessed(i) => preprocessed[*i],
            Self::Main(i) => main[*i],
        }
    }
}

impl<F: Field> VirtualPairCol<F> {
    /// Creates a new virtual column with the specified column weights and constant term.
    ///
    /// # Arguments
    /// * `column_weights` - Vector of (column, weight) pairs defining the linear combination
    /// * `constant` - Constant term to add to the linear combination
    pub const fn new(column_weights: Vec<(PairCol, F)>, constant: F) -> Self {
        Self {
            column_weights,
            constant,
        }
    }

    /// Creates a virtual column as a linear combination of preprocessed columns.
    ///
    /// # Arguments
    /// * `column_weights` - Vector of (column_index, weight) pairs for preprocessed columns
    /// * `constant` - Constant term to add to the combination
    pub fn new_preprocessed(column_weights: Vec<(usize, F)>, constant: F) -> Self {
        Self::new(
            column_weights
                .into_iter()
                .map(|(i, w)| (PairCol::Preprocessed(i), w))
                .collect(),
            constant,
        )
    }

    /// Creates a virtual column as a linear combination of main trace columns.
    ///
    /// # Arguments
    /// * `column_weights` - Vector of (column_index, weight) pairs for main trace columns
    /// * `constant` - Constant term to add to the combination
    pub fn new_main(column_weights: Vec<(usize, F)>, constant: F) -> Self {
        Self::new(
            column_weights
                .into_iter()
                .map(|(i, w)| (PairCol::Main(i), w))
                .collect(),
            constant,
        )
    }

    /// A virtual column that always evaluates to the field element `1`.
    pub const ONE: Self = Self::constant(F::ONE);

    /// Creates a virtual column that evaluates to a constant value regardless of input.
    ///
    /// # Arguments
    /// * `x` - The constant field element this virtual column will always return
    #[must_use]
    pub const fn constant(x: F) -> Self {
        Self {
            column_weights: vec![],
            constant: x,
        }
    }

    /// Creates a virtual column representing a single column with weight 1.
    ///
    /// # Arguments
    /// * `column` - The column to represent
    ///
    /// # Returns
    /// A virtual column equivalent to `1 * column + 0`
    #[must_use]
    pub fn single(column: PairCol) -> Self {
        Self {
            column_weights: vec![(column, F::ONE)],
            constant: F::ZERO,
        }
    }

    /// Creates a virtual column representing a single preprocessed column.
    ///
    /// # Arguments
    /// * `column` - Index of the preprocessed column
    #[must_use]
    pub fn single_preprocessed(column: usize) -> Self {
        Self::single(PairCol::Preprocessed(column))
    }

    /// Creates a virtual column representing a single main trace column.
    ///
    /// # Arguments
    /// * `column` - Index of the main trace column
    #[must_use]
    pub fn single_main(column: usize) -> Self {
        Self::single(PairCol::Main(column))
    }

    /// Creates a virtual column that sums multiple main trace columns.
    ///
    /// # Arguments
    /// * `columns` - Vector of main trace column indices to sum
    ///
    /// # Returns
    /// A virtual column representing `∑ main[i]` where `i` are the provided indices
    #[must_use]
    pub fn sum_main(columns: Vec<usize>) -> Self {
        let column_weights = columns.into_iter().map(|col| (col, F::ONE)).collect();
        Self::new_main(column_weights, F::ZERO)
    }

    /// Creates a virtual column that sums multiple preprocessed columns.
    ///
    /// # Arguments
    /// * `columns` - Vector of preprocessed column indices to sum
    ///
    /// # Returns
    /// A virtual column representing `∑ preprocessed[i]` where `i` are the provided indices
    #[must_use]
    pub fn sum_preprocessed(columns: Vec<usize>) -> Self {
        let column_weights = columns.into_iter().map(|col| (col, F::ONE)).collect();
        Self::new_preprocessed(column_weights, F::ZERO)
    }

    /// Creates a virtual column representing the difference between two preprocessed columns.
    ///
    /// # Arguments
    /// * `a_col` - Index of the first preprocessed column
    /// * `b_col` - Index of the second preprocessed column
    ///
    /// # Returns
    /// A virtual column representing `preprocessed[a_col] - preprocessed[b_col]`
    #[must_use]
    pub fn diff_preprocessed(a_col: usize, b_col: usize) -> Self {
        Self::new_preprocessed(vec![(a_col, F::ONE), (b_col, F::NEG_ONE)], F::ZERO)
    }

    /// Creates a virtual column representing the difference between two main trace columns.
    ///
    /// # Arguments
    /// * `a_col` - Index of the first main trace column
    /// * `b_col` - Index of the second main trace column
    ///
    /// # Returns
    /// A virtual column representing `main[a_col] - main[b_col]`
    #[must_use]
    pub fn diff_main(a_col: usize, b_col: usize) -> Self {
        Self::new_main(vec![(a_col, F::ONE), (b_col, F::NEG_ONE)], F::ZERO)
    }

    /// Evaluates the virtual column by applying the affine linear combination to the given column values.
    ///
    /// This method computes `Σ(w_i * column_values[i]) + constant` where the column values
    /// are retrieved from the appropriate arrays based on their type (preprocessed or main).
    ///
    /// # Type Parameters
    /// * `Expr` - Expression type that supports field arithmetic
    /// * `Var` - Variable type that can be converted to expressions
    ///
    /// # Arguments
    /// * `preprocessed` - Array of preprocessed column values
    /// * `main` - Array of main trace column values
    ///
    /// # Returns
    /// The computed expression result.
    pub fn apply<Expr, Var>(&self, preprocessed: &[Var], main: &[Var]) -> Expr
    where
        F: Into<Expr>,
        Expr: PrimeCharacteristicRing + Mul<F, Output = Expr>,
        Var: Into<Expr> + Copy,
    {
        self.column_weights
            .iter()
            .fold(self.constant.into(), |acc, &(col, w)| {
                acc + col.get(preprocessed, main).into() * w
            })
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;

    use super::*;

    type F = BabyBear;

    #[test]
    fn test_pair_col_get_main_and_preprocessed() {
        let pre = [F::from_u8(10), F::from_u8(20)];
        let main = [F::from_u8(30), F::from_u8(40)];

        // Preprocessed(1) should return 20
        assert_eq!(PairCol::Preprocessed(1).get(&pre, &main), F::from_u8(20));

        // Main(0) should return 30
        assert_eq!(PairCol::Main(0).get(&pre, &main), F::from_u8(30));
    }

    #[test]
    fn test_constant_only_virtual_pair_col() {
        let col = VirtualPairCol::<F>::constant(F::from_u8(7));

        // Apply to any input: result should always be the constant
        let pre = [F::ONE];
        let main = [F::ONE];
        let result = col.apply::<F, F>(&pre, &main);

        assert_eq!(result, F::from_u8(7));
    }

    #[test]
    fn test_single_main_column() {
        let col = VirtualPairCol::<F>::single_main(1); // column index 1

        let main = [F::from_u8(9), F::from_u8(5)];
        let pre = [F::ZERO]; // ignored

        let result = col.apply::<F, F>(&pre, &main);

        // Since we used single_main(1), this should equal main[1] = 5
        assert_eq!(result, F::from_u8(5));
    }

    #[test]
    fn test_single_preprocessed_column() {
        let col = VirtualPairCol::<F>::single_preprocessed(0);

        let pre = [F::from_u8(12)];
        let main = [];

        let result = col.apply::<F, F>(&pre, &main);

        assert_eq!(result, F::from_u8(12));
    }

    #[test]
    fn test_sum_main_columns() {
        // This adds up main[0] + main[2]
        let col = VirtualPairCol::<F>::sum_main(vec![0, 2]);

        let main = [
            F::TWO,
            F::from_u8(99), // ignored
            F::from_u8(5),
        ];
        let pre = [];

        let result = col.apply::<F, F>(&pre, &main);

        assert_eq!(result, F::from_u8(2) + F::from_u8(5));
    }

    #[test]
    fn test_sum_preprocessed_columns() {
        let col = VirtualPairCol::<F>::sum_preprocessed(vec![1, 2]);

        let pre = [
            F::from_u8(3), // ignored
            F::from_u8(4),
            F::from_u8(6),
        ];
        let main = [];

        let result = col.apply::<F, F>(&pre, &main);

        assert_eq!(result, F::from_u8(4) + F::from_u8(6));
    }

    #[test]
    fn test_diff_main_columns() {
        // Computes main[2] - main[0]
        let col = VirtualPairCol::<F>::diff_main(2, 0);

        let main = [
            F::from_u8(7),
            F::ZERO, // ignored
            F::from_u8(10),
        ];
        let pre = [];

        let result = col.apply::<F, F>(&pre, &main);

        assert_eq!(result, F::from_u8(10) - F::from_u8(7));
    }

    #[test]
    fn test_diff_preprocessed_columns() {
        // Computes pre[1] - pre[0]
        let col = VirtualPairCol::<F>::diff_preprocessed(1, 0);

        let pre = [F::from_u8(4), F::from_u8(15)];
        let main = [];

        let result = col.apply::<F, F>(&pre, &main);

        assert_eq!(result, F::from_u8(15) - F::from_u8(4));
    }

    #[test]
    fn test_combination_with_constant_and_weights() {
        // Computes: 3 * main[1] + 2 * pre[0] + constant (5)
        let col = VirtualPairCol {
            column_weights: vec![
                (PairCol::Main(1), F::from_u8(3)),
                (PairCol::Preprocessed(0), F::TWO),
            ],
            constant: F::from_u8(5),
        };

        let main = [F::ZERO, F::from_u8(4)];
        let pre = [F::from_u8(6)];

        let result = col.apply::<F, F>(&pre, &main);

        // result = 3*4 + 2*6 + 5
        assert_eq!(result, F::from_u8(29));
    }

    #[test]
    fn test_virtual_pair_col_one_is_identity() {
        // VirtualPairCol::ONE should always evaluate to 1 regardless of input
        let col = VirtualPairCol::<F>::ONE;
        let pre = [F::from_u8(99)];
        let main = [F::from_u8(42)];

        let result = col.apply::<F, F>(&pre, &main);

        assert_eq!(result, F::ONE);
    }
}
