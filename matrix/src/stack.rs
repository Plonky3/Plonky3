use crate::Matrix;
use core::marker::PhantomData;

/// A combination of two matrices, stacked together vertically.
pub struct VerticalPair<T, First: Matrix<T>, Second: Matrix<T>> {
    first: First,
    second: Second,
    _phantom: PhantomData<T>,
}

impl<T, First: Matrix<T>, Second: Matrix<T>> VerticalPair<T, First, Second> {
    pub fn new(first: First, second: Second) -> Self {
        assert_eq!(first.width(), second.width());
        Self {
            first,
            second,
            _phantom: PhantomData,
        }
    }
}

impl<T, First: Matrix<T>, Second: Matrix<T>> Matrix<T> for VerticalPair<T, First, Second> {
    fn width(&self) -> usize {
        self.first.width()
    }

    fn height(&self) -> usize {
        self.first.height() + self.second.height()
    }

    fn row(&self, r: usize) -> &[T] {
        if r < self.first.height() {
            self.first.row(r)
        } else {
            self.second.row(r - self.first.height())
        }
    }
}
