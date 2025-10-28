//! Gadgets for enforcing interaction constraints.

pub mod logup;
mod traits;

pub use logup::LogUpGadget;
pub use traits::{GadgetConstraintContext, InteractionGadget};
