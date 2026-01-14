#![no_std]

#[cfg(not(feature = "parallel"))]
mod serial;

pub mod prelude {
    use core::marker::{Send, Sync};

    #[cfg(feature = "parallel")]
    pub use rayon::prelude::*;
    #[cfg(feature = "parallel")]
    pub use rayon::{current_num_threads, join};

    #[cfg(not(feature = "parallel"))]
    pub use core::iter::{
        ExactSizeIterator as IndexedParallelIterator, Iterator as ParallelIterator,
    };
    #[cfg(not(feature = "parallel"))]
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
        #[inline]
        fn par_fold_reduce<Acc, Id, F, R>(self, identity: Id, fold_op: F, reduce_op: R) -> Acc
        where
            Acc: Send,
            Id: Fn() -> Acc + Sync + Send,
            F: Fn(Acc, Self::Item) -> Acc + Sync + Send,
            R: Fn(Acc, Acc) -> Acc + Sync + Send,
        {
            #[cfg(feature = "parallel")]
            {
                self.fold(&identity, fold_op).reduce(&identity, reduce_op)
            }

            #[cfg(not(feature = "parallel"))]
            {
                let _ = reduce_op;
                self.fold(identity(), fold_op)
            }
        }
    }
}

pub mod iter {
    #[cfg(feature = "parallel")]
    pub use rayon::iter::{repeat, repeat_n};

    #[cfg(not(feature = "parallel"))]
    pub use core::iter::{repeat, repeat_n};
}
