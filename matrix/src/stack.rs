use core::iter::Chain;
use core::ops::Deref;

use crate::Matrix;

/// A matrix composed by stacking two matrices vertically, one on top of the other.
///
/// Both matrices must have the same `width`.
/// The resulting matrix has dimensions:
/// - `width`: The same as the inputs.
/// - `height`: The sum of the `heights` of the input matrices.
///
/// Element access and iteration will first access the rows of the top matrix,
/// followed by the rows of the bottom matrix.
#[derive(Copy, Clone, Debug)]
pub struct VerticalPair<Top, Bottom> {
    /// The top matrix in the vertical composition.
    pub top: Top,
    /// The bottom matrix in the vertical composition.
    pub bottom: Bottom,
}

/// A matrix composed by placing two matrices side-by-side horizontally.
///
/// Both matrices must have the same `height`.
/// The resulting matrix has dimensions:
/// - `width`: The sum of the `widths` of the input matrices.
/// - `height`: The same as the inputs.
///
/// Element access and iteration for a given row `i` will first access the elements in the `i`'th row of the left matrix,
/// followed by elements in the `i'`th row of the right matrix.
#[derive(Copy, Clone, Debug)]
pub struct HorizontalPair<Left, Right> {
    /// The left matrix in the horizontal composition.
    pub left: Left,
    /// The right matrix in the horizontal composition.
    pub right: Right,
}

impl<Top, Bottom> VerticalPair<Top, Bottom> {
    /// Create a new `VerticalPair` by stacking two matrices vertically.
    ///
    /// # Panics
    /// Panics if the two matrices do not have the same width (i.e., number of columns),
    /// since vertical composition requires column alignment.
    ///
    /// # Returns
    /// A `VerticalPair` that represents the combined matrix.
    pub fn new<T>(top: Top, bottom: Bottom) -> Self
    where
        T: Send + Sync,
        Top: Matrix<T>,
        Bottom: Matrix<T>,
    {
        assert_eq!(top.width(), bottom.width());
        Self { top, bottom }
    }
}

impl<Left, Right> HorizontalPair<Left, Right> {
    /// Create a new `HorizontalPair` by joining two matrices side by side.
    ///
    /// # Panics
    /// Panics if the two matrices do not have the same height (i.e., number of rows),
    /// since horizontal composition requires row alignment.
    ///
    /// # Returns
    /// A `HorizontalPair` that represents the combined matrix.
    pub fn new<T>(left: Left, right: Right) -> Self
    where
        T: Send + Sync,
        Left: Matrix<T>,
        Right: Matrix<T>,
    {
        assert_eq!(left.height(), right.height());
        Self { left, right }
    }
}

impl<T: Send + Sync, Top: Matrix<T>, Bottom: Matrix<T>> Matrix<T> for VerticalPair<Top, Bottom> {
    fn width(&self) -> usize {
        self.top.width()
    }

    fn height(&self) -> usize {
        self.top.height() + self.bottom.height()
    }

    fn get(&self, r: usize, c: usize) -> T {
        if r < self.top.height() {
            self.top.get(r, c)
        } else {
            self.bottom.get(r - self.top.height(), c)
        }
    }

    type Row<'a>
        = EitherRow<Top::Row<'a>, Bottom::Row<'a>>
    where
        Self: 'a;

    fn row(&self, r: usize) -> Self::Row<'_> {
        if r < self.top.height() {
            EitherRow::Left(self.top.row(r))
        } else {
            EitherRow::Right(self.bottom.row(r - self.top.height()))
        }
    }

    fn row_slice(&self, r: usize) -> impl Deref<Target = [T]> {
        if r < self.top.height() {
            EitherRow::Left(self.top.row_slice(r))
        } else {
            EitherRow::Right(self.bottom.row_slice(r - self.top.height()))
        }
    }
}

impl<T: Send + Sync, Left: Matrix<T>, Right: Matrix<T>> Matrix<T> for HorizontalPair<Left, Right> {
    fn width(&self) -> usize {
        self.left.width() + self.right.width()
    }

    fn height(&self) -> usize {
        self.left.height()
    }

    fn get(&self, r: usize, c: usize) -> T {
        if c < self.left.width() {
            self.left.get(r, c)
        } else {
            self.right.get(r, c - self.left.width())
        }
    }

    type Row<'a>
        = Chain<Left::Row<'a>, Right::Row<'a>>
    where
        Self: 'a;

    fn row(&self, r: usize) -> Self::Row<'_> {
        self.left.row(r).chain(self.right.row(r))
    }
}

/// We use this to wrap both the row iterator and the row slice.
#[derive(Debug)]
pub enum EitherRow<L, R> {
    Left(L),
    Right(R),
}

impl<T, L, R> Iterator for EitherRow<L, R>
where
    L: Iterator<Item = T>,
    R: Iterator<Item = T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Left(l) => l.next(),
            Self::Right(r) => r.next(),
        }
    }
}

impl<T, L, R> Deref for EitherRow<L, R>
where
    L: Deref<Target = [T]>,
    R: Deref<Target = [T]>,
{
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Left(l) => l,
            Self::Right(r) => r,
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;
    use crate::RowMajorMatrix;

    #[test]
    fn test_vertical_pair_empty_top() {
        let top = RowMajorMatrix::new(vec![], 2); // 0x2
        let bottom = RowMajorMatrix::new(vec![1, 2, 3, 4], 2); // 2x2
        let vpair = VerticalPair::new::<i32>(top, bottom);
        assert_eq!(vpair.height(), 2);
        assert_eq!(vpair.get(1, 1), 4);
    }

    #[test]
    fn test_vertical_pair_composition() {
        let top = RowMajorMatrix::new(vec![1, 2, 3, 4], 2); // 2x2
        let bottom = RowMajorMatrix::new(vec![5, 6, 7, 8], 2); // 2x2
        let vertical = VerticalPair::new::<i32>(top, bottom);

        // Dimensions
        assert_eq!(vertical.width(), 2);
        assert_eq!(vertical.height(), 4);

        // Values from top
        assert_eq!(vertical.get(0, 0), 1);
        assert_eq!(vertical.get(1, 1), 4);

        // Values from bottom
        assert_eq!(vertical.get(2, 0), 5);
        assert_eq!(vertical.get(3, 1), 8);

        // Row iter from bottom
        let row = vertical.row(3);
        let values: Vec<_> = row.collect();
        assert_eq!(values, vec![7, 8]);

        // Row slice
        assert_eq!(vertical.row_slice(2).deref(), &[5, 6]);
    }

    #[test]
    fn test_horizontal_pair_composition() {
        let left = RowMajorMatrix::new(vec![1, 2, 3, 4], 2); // 2x2
        let right = RowMajorMatrix::new(vec![5, 6, 7, 8], 2); // 2x2
        let horizontal = HorizontalPair::new::<i32>(left, right);

        // Dimensions
        assert_eq!(horizontal.height(), 2);
        assert_eq!(horizontal.width(), 4);

        // Left values
        assert_eq!(horizontal.get(0, 0), 1);
        assert_eq!(horizontal.get(1, 1), 4);

        // Right values
        assert_eq!(horizontal.get(0, 2), 5);
        assert_eq!(horizontal.get(1, 3), 8);

        // Row iter
        let row = horizontal.row(0);
        let values: Vec<_> = row.collect();
        assert_eq!(values, vec![1, 2, 5, 6]);
    }

    #[test]
    fn test_either_row_iterator_behavior() {
        type Iter = alloc::vec::IntoIter<i32>;

        // Left variant
        let left: EitherRow<Iter, Iter> = EitherRow::Left(vec![10, 20].into_iter());
        assert_eq!(left.collect::<Vec<_>>(), vec![10, 20]);

        // Right variant
        let right: EitherRow<Iter, Iter> = EitherRow::Right(vec![30, 40].into_iter());
        assert_eq!(right.collect::<Vec<_>>(), vec![30, 40]);
    }

    #[test]
    fn test_either_row_deref_behavior() {
        let left: EitherRow<&[i32], &[i32]> = EitherRow::Left(&[1, 2, 3]);
        let right: EitherRow<&[i32], &[i32]> = EitherRow::Right(&[4, 5]);

        assert_eq!(&*left, &[1, 2, 3]);
        assert_eq!(&*right, &[4, 5]);
    }

    #[test]
    #[should_panic]
    fn test_vertical_pair_width_mismatch_should_panic() {
        let a = RowMajorMatrix::new(vec![1, 2, 3], 1); // 3x1
        let b = RowMajorMatrix::new(vec![4, 5], 2); // 1x2
        let _ = VerticalPair::new::<i32>(a, b);
    }

    #[test]
    #[should_panic]
    fn test_horizontal_pair_height_mismatch_should_panic() {
        let a = RowMajorMatrix::new(vec![1, 2, 3], 3); // 1x3
        let b = RowMajorMatrix::new(vec![4, 5], 1); // 2x1
        let _ = HorizontalPair::new::<i32>(a, b);
    }
}
