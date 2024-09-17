#[cfg(feature = "parallel")]
pub mod prelude {
    pub use rayon::prelude::*;
    pub use rayon::{current_num_threads, join};

    pub trait SharedExt: ParallelIterator {
        fn par_fold_reduce<Acc, Id, F, R>(self, identity: Id, fold_op: F, reduce_op: R) -> Acc
        where
            Acc: Send,
            Id: Fn() -> Acc + Sync + Send,
            F: Fn(Acc, Self::Item) -> Acc + Sync + Send,
            R: Fn(Acc, Acc) -> Acc + Sync + Send;
    }

    impl<I: ParallelIterator> SharedExt for I {
        fn par_fold_reduce<Acc, Id, F, R>(self, identity: Id, fold_op: F, reduce_op: R) -> Acc
        where
            Acc: Send,
            Id: Fn() -> Acc + Sync + Send,
            F: Fn(Acc, Self::Item) -> Acc + Sync + Send,
            R: Fn(Acc, Acc) -> Acc + Sync + Send,
        {
            self.fold(&identity, fold_op).reduce(&identity, reduce_op)
        }
    }

    #[macro_export]
    macro_rules! par_izip {
        ( $first:expr $( , $rest:expr )+ $(,)* ) => {
            ($first $( , $rest )+ ).into_par_iter()
        };
    }
}

#[cfg(feature = "parallel")]
pub mod iter {
    pub use rayon::iter::{repeat, split};
}

#[cfg(not(feature = "parallel"))]
mod serial;

#[cfg(not(feature = "parallel"))]
pub mod prelude {
    pub use core::iter::{
        ExactSizeIterator as IndexedParallelIterator, Iterator as ParallelIterator,
    };

    pub use super::serial::*;

    pub trait SharedExt: ParallelIterator {
        fn par_fold_reduce<Acc, Id, F, R>(self, identity: Id, fold_op: F, reduce_op: R) -> Acc
        where
            Acc: Send,
            Id: Fn() -> Acc + Sync + Send,
            F: Fn(Acc, Self::Item) -> Acc + Sync + Send,
            R: Fn(Acc, Acc) -> Acc + Sync + Send;
    }

    impl<I: ParallelIterator> SharedExt for I {
        fn par_fold_reduce<Acc, Id, F, R>(self, identity: Id, fold_op: F, _reduce_op: R) -> Acc
        where
            Acc: Send,
            Id: Fn() -> Acc + Sync + Send,
            F: Fn(Acc, Self::Item) -> Acc + Sync + Send,
            R: Fn(Acc, Acc) -> Acc + Sync + Send,
        {
            self.fold(identity(), fold_op)
        }
    }

    #[macro_export]
    macro_rules! par_izip {
        ( $first:expr $( , $rest:expr )+ $(,)* ) => {
            itertools::izip!($first $(, $rest)+ )
        };
    }
}

#[cfg(not(feature = "parallel"))]
pub mod iter {
    pub use core::iter::repeat;

    pub fn split<D, S>(data: D, _splitter: S) -> core::iter::Once<D>
    where
        S: Fn(D) -> (D, Option<D>),
    {
        core::iter::once(data)
    }
}
