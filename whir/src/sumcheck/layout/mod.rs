//! Stacked sumcheck layout.

mod opening;
mod prover;
mod verifier;
mod witness;

pub use opening::{
    MultiClaim, Opening, PrefixMultiClaim, PrefixOpening, PrefixVirtualClaim, SuffixMultiClaim,
    SuffixOpening, SuffixVirtualClaim, VerifierMultiClaim, VerifierOpening, VerifierVirtualClaim,
};
pub use prover::{PrefixProver, SuffixProver};
pub use verifier::{TableShape, Verifier};
pub use witness::{Selector, Table, TablePlacement, Witness};
