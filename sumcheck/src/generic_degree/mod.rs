//! Generic-degree sumcheck driver.

mod error;
mod proof;
mod prover;
mod transcript;
mod util;

pub use error::GenericDegreeError;
pub use proof::GenericDegreeProof;
pub use prover::RoundProver;
pub use transcript::{ProverTranscript, VerifierTranscript, domain_separator, pattern};
pub use util::RoundPolyInterpolator;
