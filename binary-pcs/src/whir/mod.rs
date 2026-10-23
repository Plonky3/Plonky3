//! WHIR over the additive binary domain, and the adapters that feed it narrow traces.

mod boolean;
mod budget;
mod commitment;
mod domain;
mod error;
pub mod preset;
mod profile;
mod proof;
mod region;
mod shape;

pub use boolean::{
    BooleanWhirData, BooleanWhirPcs, BooleanWhirProver, BooleanWhirTracePcs, WeightedClaim,
    WeightedOpening,
};
pub use budget::BinaryWhirBudget;
pub use commitment::{BitCommitment, CommitmentParams};
pub use domain::{BinaryWhirAlphabet, BinaryWhirDomain, BooleanWhirDomain, recommended_cap_height};
pub use error::{BooleanWhirError, BudgetError, CommitmentError, ProfileError};
pub use p3_sumcheck::ring_switch::bits::{BitTensor, SuffixTensors};
pub use profile::BinaryWhirProfile;
pub use proof::{BooleanWhirDeferredProof, BooleanWhirProof};
pub use region::{Region, RegionClaim, RegionLayout};
pub use shape::{ProofShape, ProofShapeError};
