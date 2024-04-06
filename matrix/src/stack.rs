use crate::Matrix;

/// A combination of two matrices, stacked together vertically.
#[derive(Debug)]
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

    type Row<'a> = EitherIterator<First::Row<'a>, Second::Row<'a>> where Self: 'a;

    fn get(&self, r: usize, c: usize) -> T {
        if r < self.first.height() {
            self.first.get(r, c)
        } else {
            self.second.get(r - self.first.height(), c)
        }
    }

    fn row(&self, r: usize) -> Self::Row<'_> {
        if r < self.first.height() {
            EitherIterator::Left(self.first.row(r))
        } else {
            EitherIterator::Right(self.second.row(r - self.first.height()))
        }
    }
}

#[derive(Debug)]
pub enum EitherIterator<L, R> {
    Left(L),
    Right(R),
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
