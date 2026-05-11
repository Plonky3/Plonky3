//! Circuit compilation and lowering subsystem.

mod lowerer;
mod optimizer;

pub use lowerer::{ExpressionLowerer, LoweringResult};
pub use optimizer::Optimizer;
