//! Every way the contract refuses a statement, a run, or an encoded proof.

use core::fmt::Debug;

use thiserror::Error;

use crate::contract::envelope::HEADER_LEN;
use crate::verifier::VerificationError;

/// Why a declaration or a run of it is not usable.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum DeclarationError {
    /// A statement with no tables proves nothing.
    #[error("a statement must declare at least one table")]
    NoTables,
    /// A count exceeds the ceiling this module fixes for it.
    #[error("table {table}: {what} is {found}, above the limit of {limit}")]
    AboveLimit {
        /// Position of the offending table in declaration order.
        table: usize,
        /// Which count is out of range.
        what: &'static str,
        /// The declared value.
        found: usize,
        /// The largest accepted value.
        limit: usize,
    },
    /// A height range reaches below the smallest height the backend can prove.
    #[error("table {table}: height exponent floor {min} is below {floor}")]
    HeightBelowFloor {
        /// Position of the offending table in declaration order.
        table: usize,
        /// Smallest declared exponent.
        min: u32,
        /// Smallest exponent the backend accepts.
        floor: u32,
    },
    /// A height range excludes every height.
    #[error("table {table}: height range {min}..={max} is empty")]
    EmptyHeightRange {
        /// Position of the offending table in declaration order.
        table: usize,
        /// Smallest declared exponent.
        min: u32,
        /// Largest declared exponent.
        max: u32,
    },
    /// The declared proof-size budget is zero or above the hard ceiling.
    #[error("proof-size budget {found} is outside 1..={limit} bytes")]
    BudgetOutOfRange { found: usize, limit: usize },
    /// A run supplies a different number of heights than the statement has tables.
    #[error("the run supplies {found} heights for {expected} tables")]
    HeightCountMismatch {
        /// Number of declared tables.
        expected: usize,
        /// Number of heights the run supplied.
        found: usize,
    },
    /// A run picks a height the table never declared.
    #[error("table {table}: height exponent {found} is outside the declared {min}..={max}")]
    HeightNotDeclared {
        /// Position of the offending table in declaration order.
        table: usize,
        /// The exponent the run picked.
        found: u32,
        /// Smallest declared exponent.
        min: u32,
        /// Largest declared exponent.
        max: u32,
    },
    /// The declared security target is zero or above the hard ceiling.
    #[error("security target {found} is outside 1..={limit} bits")]
    SecurityOutOfRange { found: usize, limit: usize },
    /// A run requests more grinding than this module accepts.
    #[error("grinding difficulty {found} is above the limit of {limit}")]
    PowBitsAboveLimit {
        /// The requested difficulty.
        found: u32,
        /// The largest accepted difficulty.
        limit: u32,
    },
    /// A run belongs to a different statement than the one it is used with.
    #[error("the run describes a different statement")]
    ForeignRun,
}

/// Why a byte string is not an acceptable proof for a statement.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum EnvelopeError {
    /// The run does not belong to the statement it was used with.
    #[error("declaration: {0}")]
    Declaration(DeclarationError),
    /// The input is shorter than the fixed header.
    #[error("input is {found} bytes, shorter than the {HEADER_LEN}-byte header")]
    HeaderTooShort {
        /// Length of the input.
        found: usize,
    },
    /// The input does not start with the bytes every sealed proof starts with.
    #[error("input does not carry the expected opening bytes")]
    BadMagic,
    /// The framing revision is not the one this build speaks.
    #[error("framing revision {found} is not the supported {expected}")]
    EnvelopeVersion {
        /// Revision the input declares.
        found: u16,
        /// Revision this build speaks.
        expected: u16,
    },
    /// The body revision is not the one this build speaks.
    #[error("body revision {found} is not the supported {expected}")]
    BodyRevision {
        /// Revision the input declares.
        found: u16,
        /// Revision this build speaks.
        expected: u16,
    },
    /// The input was sealed against a different statement or a different run of it.
    #[error("the input was sealed against a different statement")]
    RunMismatch,
    /// The declared body length is above the statement's budget.
    #[error("declared body length {found} is above the budget of {budget} bytes")]
    BodyAboveBudget {
        /// Length the header declares.
        found: usize,
        /// Largest length the statement accepts.
        budget: usize,
    },
    /// The input stops before the body the header declares.
    #[error("header declares {declared} body bytes but only {available} follow")]
    Truncated {
        /// Length the header declares.
        declared: usize,
        /// Length actually present.
        available: usize,
    },
    /// The input continues past the body the header declares.
    #[error("{extra} bytes follow the declared body")]
    TrailingBytes {
        /// Number of bytes past the declared body.
        extra: usize,
    },
    /// The body is not a well-formed encoding.
    #[error("the body is not a well-formed encoding")]
    Malformed,
    /// The body decodes but leaves bytes behind.
    #[error("{remaining} body bytes were not consumed by the decoder")]
    UnreadBodyBytes {
        /// Number of bytes the decoder did not consume.
        remaining: usize,
    },
    /// A part of the proof is present when the statement declares none, or the reverse.
    #[error("the {section} part is present ({present}) against what the statement declares")]
    SectionMismatch {
        /// Which part disagrees.
        section: &'static str,
        /// Whether the proof carries it.
        present: bool,
    },
    /// A count inside the proof disagrees with the statement.
    #[error("the {section} count is {found} but the statement declares {expected}")]
    CountMismatch {
        /// Which count disagrees.
        section: &'static str,
        /// Count the statement declares.
        expected: usize,
        /// Count the proof carries.
        found: usize,
    },
    /// The encoded proof is longer than the statement's budget.
    #[error("the encoded proof is {found} bytes, above the budget of {budget}")]
    ProofAboveBudget {
        /// Length of the encoding.
        found: usize,
        /// Largest length the statement accepts.
        budget: usize,
    },
}

/// Why a sealed proof was not accepted.
#[derive(Debug, Error)]
pub enum SealedVerificationError<E: Debug> {
    /// The byte string never became a proof.
    #[error("envelope: {0}")]
    Envelope(EnvelopeError),
    /// The run and the instances describe different statements.
    #[error("the run and the instances disagree on {what}")]
    RunDisagreement {
        /// Which part disagrees.
        what: &'static str,
    },
    /// The statement and a constraint system describe different tables.
    #[error("table {table}: the statement and the constraint system disagree on {what}")]
    AirDisagreement {
        /// Position of the offending table in declaration order.
        table: usize,
        /// Which part disagrees.
        what: &'static str,
    },
    /// The proof was well framed but did not verify.
    #[error("verification: {0}")]
    Verification(VerificationError<E>),
}
