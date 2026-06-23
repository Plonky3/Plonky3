//! Generic-degree sumcheck driver.

mod error;
mod proof;
mod prover;
mod util;

pub use error::GenericDegreeError;
pub use proof::GenericDegreeProof;
pub use prover::RoundProver;
