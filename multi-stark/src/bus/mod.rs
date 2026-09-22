//! Binary-native bus planning and commitment binding.

pub(crate) mod composition;
mod context;
mod error;
pub(crate) mod transcript;

pub(crate) use context::BusContext;
pub use error::BusBindingError;
