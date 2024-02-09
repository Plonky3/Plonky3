#[cfg(feature = "parallel")]
pub mod prelude {
    pub use rayon::join;
    pub use rayon::prelude::*;
}

#[cfg(not(feature = "parallel"))]
mod serial;

#[cfg(not(feature = "parallel"))]
pub mod prelude {
    pub use core::iter::{
        ExactSizeIterator as IndexedParallelIterator, Iterator as ParallelIterator,
    };

    pub use super::serial::*;
}
