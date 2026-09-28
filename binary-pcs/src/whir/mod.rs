//! WHIR over the additive binary domain, and the adapters that feed it narrow traces.

mod boolean;
mod budget;
mod domain;
mod error;
mod profile;
mod proof;
mod shape;

pub use boolean::{BooleanWhirData, BooleanWhirPcs, BooleanWhirProver, BooleanWhirTracePcs};
pub use budget::BinaryWhirBudget;
pub use domain::{BinaryWhirAlphabet, BinaryWhirDomain, BooleanWhirDomain, recommended_cap_height};
pub use error::{BooleanWhirError, BudgetError, ProfileError};
pub use profile::BinaryWhirProfile;
pub use proof::BooleanWhirProof;
pub use shape::ProofShape;
