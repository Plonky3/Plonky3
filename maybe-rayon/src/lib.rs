#[cfg(feature = "parallel")]
pub mod prelude {
    pub use rayon::prelude::*;
}

#[cfg(not(feature = "parallel"))]
mod serial;

#[cfg(not(feature = "parallel"))]
pub mod prelude {
    pub use super::serial::*;
    pub use core::iter::{
        ExactSizeIterator as IndexedParallelIterator, Iterator as ParallelIterator,
    };
}
