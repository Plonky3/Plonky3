use core::marker::PhantomData;

use crate::{Matrix, MatrixRows};

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
}

impl<T, First, Second> MatrixRows<T> for VerticalPair<T, First, Second>
where
    First: MatrixRows<T>,
    Second: MatrixRows<T>,
{
    type Row<'a> = EitherIterable<First::Row<'a>, Second::Row<'a>> where Self: 'a;

    fn row(&self, r: usize) -> Self::Row<'_> {
        if r < self.first.height() {
            EitherIterable::Left(self.first.row(r))
        } else {
            EitherIterable::Right(self.second.row(r - self.first.height()))
        }
    }
}

pub enum EitherIterable<L, R> {
    Left(L),
    Right(R),
}

pub enum EitherIterator<L, R> {
    Left(L),
    Right(R),
}

impl<T, L, R> IntoIterator for EitherIterable<L, R>
where
    L: IntoIterator<Item = T>,
    R: IntoIterator<Item = T>,
{
    type Item = T;
    type IntoIter = EitherIterator<L::IntoIter, R::IntoIter>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            EitherIterable::Left(l) => EitherIterator::Left(l.into_iter()),
            EitherIterable::Right(r) => EitherIterator::Right(r.into_iter()),
        }
    }
}

impl<T, L, R> Iterator for EitherIterator<L, R>
where
    L: Iterator<Item = T>,
    R: Iterator<Item = T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            EitherIterator::Left(l) => l.next(),
            EitherIterator::Right(r) => r.next(),
        }
    }
}
