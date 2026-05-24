//! Generic-degree sumcheck driver.

mod error;
mod proof;
mod prover;
mod util;

pub use error::MultiRoundError;
pub use proof::MultiRoundProof;
pub use prover::RoundProver;
