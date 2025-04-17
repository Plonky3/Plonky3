use core::ops::Deref;

use crate::Matrix;

/// A combination of two matrices, stacked together vertically.
#[derive(Copy, Clone, Debug)]
pub struct VerticalPair<First, Second> {
    pub first: First,
    pub second: Second,
}

/// A combination of two matrices, stacked together horizontally.
#[derive(Copy, Clone, Debug)]
pub struct HorizontalPair<First, Second> {
    pub first: First,
    pub second: Second,
}

impl<First, Second> VerticalPair<First, Second> {
    pub fn new<T>(first: First, second: Second) -> Self
    where
        T: Send + Sync,
        First: Matrix<T>,
        Second: Matrix<T>,
    {
        assert_eq!(first.width(), second.width());
        Self { first, second }
    }
}

impl<First, Second> HorizontalPair<First, Second> {
    pub fn new<T>(first: First, second: Second) -> Self
    where
        T: Send + Sync,
        First: Matrix<T>,
        Second: Matrix<T>,
    {
        assert_eq!(first.height(), second.height());
        Self { first, second }
    }
}

impl<T: Send + Sync, First: Matrix<T>, Second: Matrix<T>> Matrix<T>
    for VerticalPair<First, Second>
{
    fn width(&self) -> usize {
        self.first.width()
    }

    fn height(&self) -> usize {
        self.first.height() + self.second.height()
    }

    fn get(&self, r: usize, c: usize) -> Option<T> {
        if r < self.first.height() {
            self.first.get(r, c)
        } else {
            self.second.get(r - self.first.height(), c)
        }
    }

    fn row(
        &self,
        r: usize,
    ) -> Option<impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync>> {
        Some(if r < self.first.height() {
            unsafe {
                // Safety: We just checked that r < self.first.height()
                EitherRow::Left(self.first.row_unchecked(r).into_iter())
            }
        } else {
            EitherRow::Right(self.second.row(r - self.first.height())?.into_iter())
        })
    }

    unsafe fn row_unchecked(
        &self,
        r: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        unsafe {
            // Safety: The caller must ensure that r < self.height()
            if r < self.first.height() {
                EitherRow::Left(self.first.row_unchecked(r).into_iter())
            } else {
                EitherRow::Right(
                    self.second
                        .row_unchecked(r - self.first.height())
                        .into_iter(),
                )
            }
        }
    }

    unsafe fn row_subset_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        unsafe {
            // Safety: The caller must ensure that r < self.height() and start <= end <= self.width()
            if r < self.first.height() {
                EitherRow::Left(self.first.row_subset_unchecked(r, start, end).into_iter())
            } else {
                EitherRow::Right(
                    self.second
                        .row_subset_unchecked(r - self.first.height(), start, end)
                        .into_iter(),
                )
            }
        }
    }

    fn row_slice(&self, r: usize) -> Option<impl Deref<Target = [T]>> {
        Some(if r < self.first.height() {
            unsafe {
                // Safety: We just checked that r < self.first.height()
                EitherRow::Left(self.first.row_slice_unchecked(r))
            }
        } else {
            EitherRow::Right(self.second.row_slice(r - self.first.height())?)
        })
    }

    unsafe fn row_slice_unchecked(&self, r: usize) -> impl Deref<Target = [T]> {
        unsafe {
            // Safety: The caller must ensure that r < self.height()
            if r < self.first.height() {
                EitherRow::Left(self.first.row_slice_unchecked(r))
            } else {
                EitherRow::Right(self.second.row_slice_unchecked(r - self.first.height()))
            }
        }
    }

    unsafe fn row_subslice_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl Deref<Target = [T]> {
        unsafe {
            // Safety: The caller must ensure that r < self.height() and start <= end <= self.width()
            if r < self.first.height() {
                EitherRow::Left(self.first.row_subslice_unchecked(r, start, end))
            } else {
                EitherRow::Right(self.second.row_subslice_unchecked(
                    r - self.first.height(),
                    start,
                    end,
                ))
            }
        }
    }
}

impl<T: Send + Sync, First: Matrix<T>, Second: Matrix<T>> Matrix<T>
    for HorizontalPair<First, Second>
{
    fn width(&self) -> usize {
        self.first.width() + self.second.width()
    }

    fn height(&self) -> usize {
        self.first.height()
    }

    fn get(&self, r: usize, c: usize) -> Option<T> {
        if c < self.first.width() {
            self.first.get(r, c)
        } else {
            self.second.get(r, c - self.first.width())
        }
    }

    unsafe fn get_unchecked(&self, r: usize, c: usize) -> T {
        unsafe {
            // Safety: The caller must ensure that r < self.height() and c < self.width()
            if c < self.first.width() {
                self.first.get_unchecked(r, c)
            } else {
                self.second.get_unchecked(r, c - self.first.width())
            }
        }
    }

    fn row(
        &self,
        r: usize,
    ) -> Option<impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync>> {
        Some(self.first.row(r)?.into_iter().chain(self.second.row(r)?))
    }

    unsafe fn row_unchecked(
        &self,
        r: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        unsafe {
            // Safety: The caller must ensure that r < self.height()
            self.first
                .row_unchecked(r)
                .into_iter()
                .chain(self.second.row_unchecked(r))
        }
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

// TODO: Add tests for row and row_unchecked

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
        assert_eq!(vpair.get(1, 1), Some(4));
        unsafe {
            assert_eq!(vpair.get_unchecked(0, 0), 1);
        }
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
        assert_eq!(vertical.get(0, 0), Some(1));
        assert_eq!(vertical.get(1, 1), Some(4));

        // Values from bottom
        unsafe {
            assert_eq!(vertical.get_unchecked(2, 0), 5);
            assert_eq!(vertical.get_unchecked(3, 1), 8);
        }

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
        assert_eq!(horizontal.get(0, 0), Some(1));
        assert_eq!(horizontal.get(1, 1), Some(4));

        // Right values
        unsafe {
            assert_eq!(horizontal.get_unchecked(0, 2), 5);
            assert_eq!(horizontal.get_unchecked(1, 3), 8);
        }

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
