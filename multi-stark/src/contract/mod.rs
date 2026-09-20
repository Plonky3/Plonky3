//! The surface a separate machine builds against.
//!
//! A declaration fixes a statement, and an envelope frames a proof for it.
//!
//! Nothing here knows what the tables are for.

pub mod declaration;
mod digest;
pub mod envelope;
pub mod secrecy;

#[cfg(test)]
mod tests;

pub use declaration::{
    ColumnCounts, DeclarationError, FlushDeclaration, FlushDirection, HeightRange,
    LocalConstraints, MachineDeclaration, Run, TableDeclaration,
};
pub use envelope::{
    AcceptedProof, BODY_REVISION, ENVELOPE_VERSION, EnvelopeError, HEADER_LEN, MAGIC, SealedProof,
    SealedVerificationError, verify_sealed,
};
pub use secrecy::{BindingOnly, Hiding, Secrecy, SecrecyLevel};
