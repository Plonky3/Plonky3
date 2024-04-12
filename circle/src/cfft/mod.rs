//! The Circle FFT and its inverse, as detailed in
//! Circle STARKs, Section 4.2 (page 14 of the first revision PDF)
//! This code is based on Angus Gruen's implementation, which uses a slightly
//! different cfft basis than that of the paper. Basically, it continues using the
//! same twiddles for the second half of the chunk, which only changes the sign of the
//! resulting basis. For a full explanation see the comments in `util::circle_basis`.
//! This alternate basis doesn't cause any change to the code apart from our testing functions.

#[cfg(feature = "multi_thread")]
pub mod multi_thread {
    use ::std::sync::{Arc, RwLock};

    include!("cfft.rs");
}

#[cfg(not(feature = "multi_thread"))]
pub mod single_thread {
    mod seal {
        use ::core::cell::{Ref, RefCell, RefMut};

        #[derive(Default, Debug)]
        pub struct RwLock<T>(RefCell<T>);

        // Mimic `RwLock`'s API
        impl<T> RwLock<T> {
            #[allow(dead_code)]
            pub fn new(value: T) -> Self {
                Self(RefCell::new(value))
            }

            #[allow(dead_code)]
            #[inline]
            pub fn read(&self) -> Result<Ref<'_, T>, ::core::convert::Infallible> {
                Ok(self.0.borrow())
            }

            #[inline]
            pub fn write(&self) -> Result<RefMut<'_, T>, ::core::convert::Infallible> {
                Ok(self.0.borrow_mut())
            }
        }
    }
    use ::alloc::rc::Rc as Arc;
    use seal::RwLock;

    include!("cfft.rs");
}

#[cfg(feature = "multi_thread")]
pub use multi_thread::*;
#[cfg(not(feature = "multi_thread"))]
pub use single_thread::*;
