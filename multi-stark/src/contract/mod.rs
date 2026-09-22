//! The surface a separate machine builds against.
//!
//! A declaration fixes a statement, and an envelope frames a proof for it.
//!
//! Nothing here knows what the tables are for.
//!
//! Nothing this backend accepts hides anything, so no proof framed here is zero-knowledge.

mod constraints;
mod cost;
mod digest;
mod envelope;
mod error;
mod machine;
mod run;
mod segment;
mod table;

pub use cost::{CostReport, TableCost};
pub use envelope::{
    AcceptedProof, BODY_REVISION, ENVELOPE_VERSION, HEADER_LEN, MAGIC, SealedProof,
};
pub use error::{DeclarationError, EnvelopeError, SealedVerificationError};
pub use machine::{
    MAX_POW_BITS, MAX_PROOF_BYTES, MAX_SECURITY_BITS, MAX_TABLES, MachineDeclaration,
};
pub use run::Run;
pub use segment::{
    ChainError, ChainedExecution, PublicSlot, SegmentClaim, SegmentInterface, chain,
};
pub use table::{
    ColumnCounts, FlushDeclaration, HeightRange, LocalConstraints, MAX_COLUMNS,
    MAX_CONSTRAINT_DEGREE, MAX_CONSTRAINTS, MAX_FLUSHES, MAX_INDEXED, MAX_LOG_HEIGHT,
    MAX_TUPLE_WIDTH, MIN_LOG_HEIGHT, TableDeclaration,
};
