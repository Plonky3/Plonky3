use core::ops::Deref;

use crate::Matrix;

/// A combination of two matrices, stacked together vertically.
#[derive(Copy, Clone, Debug)]
pub struct VerticalPair<First, Second> {
    first: First,
    second: Second,
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

impl<T: Send + Sync, First: Matrix<T>, Second: Matrix<T>> Matrix<T>
    for VerticalPair<First, Second>
{
    fn width(&self) -> usize {
        self.first.width()
    }

    fn height(&self) -> usize {
        self.first.height() + self.second.height()
    }

    type Row<'a> = EitherRow<First::Row<'a>, Second::Row<'a>> where Self: 'a;

    fn get(&self, r: usize, c: usize) -> T {
        if r < self.first.height() {
            self.first.get(r, c)
        } else {
            self.second.get(r - self.first.height(), c)
        }
    }

    fn row(&self, r: usize) -> Self::Row<'_> {
        if r < self.first.height() {
            EitherRow::Left(self.first.row(r))
        } else {
            EitherRow::Right(self.second.row(r - self.first.height()))
        }
    }

    fn row_slice(&self, r: usize) -> impl Deref<Target = [T]> {
        if r < self.first.height() {
            EitherRow::Left(self.first.row_slice(r))
        } else {
            EitherRow::Right(self.second.row_slice(r - self.first.height()))
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
            EitherRow::Left(l) => l.next(),
            EitherRow::Right(r) => r.next(),
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
            EitherRow::Left(l) => l,
            EitherRow::Right(r) => r,
        }
    }
}
