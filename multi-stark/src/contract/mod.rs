//! The surface a separate machine builds against.
//!
//! A declaration fixes a statement, and an envelope frames a proof for it.
//!
//! Nothing here knows what the tables are for.

mod digest;
mod envelope;
mod error;
mod machine;
mod run;
mod secrecy;
mod table;

pub use envelope::{
    AcceptedProof, BODY_REVISION, ENVELOPE_VERSION, HEADER_LEN, MAGIC, SealedProof,
};
pub use error::{DeclarationError, EnvelopeError, SealedVerificationError};
pub use machine::{MAX_POW_BITS, MAX_PROOF_BYTES, MAX_TABLES, MachineDeclaration};
pub use run::Run;
pub use secrecy::{BindingOnly, Hiding, Secrecy, SecrecyLevel};
pub use table::{
    ColumnCounts, FlushDeclaration, FlushDirection, HeightRange, LocalConstraints, MAX_COLUMNS,
    MAX_CONSTRAINT_DEGREE, MAX_CONSTRAINTS, MAX_FLUSHES, MAX_INDEXED, MAX_LOG_HEIGHT,
    MAX_TUPLE_WIDTH, TableDeclaration,
};
