//! Generic-degree sumcheck driver.

mod error;
mod proof;
mod prover;
pub(crate) mod transcript;
mod util;

pub use error::GenericDegreeError;
pub use proof::GenericDegreeProof;
pub use prover::RoundProver;
pub use transcript::{GenericDegreeShape, ProverTranscript, VerifierTranscript};
pub use util::RoundPolyInterpolator;
