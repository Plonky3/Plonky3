#[cfg(feature = "parallel")]
pub use rayon::iter::Either;

#[cfg(not(feature = "parallel"))]
mod serial {
    use core::iter::FusedIterator;

    #[derive(Clone, Debug)]
    pub enum Either<L, R> {
        Left(L),
        Right(R),
    }

    impl<L, R> Iterator for Either<L, R>
    where
        L: Iterator,
        R: Iterator<Item = L::Item>,
    {
        type Item = L::Item;

        #[inline(always)]
        fn next(&mut self) -> Option<Self::Item> {
            match self {
                Self::Left(iter) => iter.next(),
                Self::Right(iter) => iter.next(),
            }
        }

        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            match self {
                Self::Left(iter) => iter.size_hint(),
                Self::Right(iter) => iter.size_hint(),
            }
        }

        #[inline]
        fn nth(&mut self, n: usize) -> Option<Self::Item> {
            match self {
                Self::Left(iter) => iter.nth(n),
                Self::Right(iter) => iter.nth(n),
            }
        }

        #[inline]
        fn last(self) -> Option<Self::Item> {
            match self {
                Self::Left(iter) => iter.last(),
                Self::Right(iter) => iter.last(),
            }
        }
    }

    impl<L, R> ExactSizeIterator for Either<L, R>
    where
        L: ExactSizeIterator,
        R: ExactSizeIterator<Item = L::Item>,
    {
        #[inline]
        fn len(&self) -> usize {
            match self {
                Self::Left(iter) => iter.len(),
                Self::Right(iter) => iter.len(),
            }
        }
    }

    impl<L, R> DoubleEndedIterator for Either<L, R>
    where
        L: DoubleEndedIterator,
        R: DoubleEndedIterator<Item = L::Item>,
    {
        #[inline]
        fn next_back(&mut self) -> Option<Self::Item> {
            match self {
                Self::Left(iter) => iter.next_back(),
                Self::Right(iter) => iter.next_back(),
            }
        }

        #[inline]
        fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
            match self {
                Self::Left(iter) => iter.nth_back(n),
                Self::Right(iter) => iter.nth_back(n),
            }
        }
    }

    impl<L, R> FusedIterator for Either<L, R>
    where
        L: FusedIterator,
        R: FusedIterator<Item = L::Item>,
    {
    }
}

#[cfg(not(feature = "parallel"))]
pub use serial::Either;
