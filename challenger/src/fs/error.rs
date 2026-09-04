//! Errors returned by the transcript machinery.

use alloc::boxed::Box;

use thiserror::Error;

use crate::fs::pattern::Interaction;

/// Failures that can arise while validating a pattern or reading a proof.
///
/// Every variant is reachable.
///
/// Divergence between the recorded pattern and the code replaying it is a
/// programming bug, not malformed input, so it panics with a diff instead of
/// landing here.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum TranscriptError {
    /// An end marker was found with no matching opener in the sequence.
    #[error(transparent)]
    MissingBegin(Box<MissingBeginInfo>),

    /// A nested step uses a kind incompatible with the surrounding sub-protocol.
    #[error(transparent)]
    InvalidKind(Box<InvalidKindInfo>),

    /// A closer does not match the most recent opener.
    #[error(transparent)]
    MismatchedBeginEnd(Box<MismatchedBeginEndInfo>),

    /// An opener was never closed before the sequence ended.
    #[error(transparent)]
    MissingEnd(Box<MissingEndInfo>),

    /// Verifier-side parsing of the prover's serialized output failed.
    #[error("bad proof shape: {reason}")]
    BadProofShape {
        /// Short reason describing the parse failure.
        reason: &'static str,
    },
}

/// Payload describing an end marker that has no matching opener.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("missing begin for {end} at position {position}")]
pub struct MissingBeginInfo {
    /// Index of the orphan end-of-block interaction inside the recorded sequence.
    pub position: usize,
    /// The interaction that lacked a matching opener.
    pub end: Interaction,
}

/// Payload describing a nested step whose kind is incompatible with its container.
///
/// Raised for a leaf and for a nested opener alike.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error(
    "invalid kind {interaction} at {interaction_position} for sub-protocol {begin} \
     opened at {begin_position}"
)]
pub struct InvalidKindInfo {
    /// Index of the surrounding sub-protocol opener inside the recorded sequence.
    pub begin_position: usize,
    /// The sub-protocol opener whose declared kind is being violated.
    pub begin: Interaction,
    /// Index of the offending nested interaction.
    pub interaction_position: usize,
    /// The offending nested interaction: an atomic step or a nested opener.
    pub interaction: Interaction,
}

/// Payload describing a closer that does not match its opener.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("mismatched begin {begin} at {begin_position} versus end {end} at {end_position}")]
pub struct MismatchedBeginEndInfo {
    /// Index of the opener inside the recorded sequence.
    pub begin_position: usize,
    /// The opener that this closer was expected to match.
    pub begin: Interaction,
    /// Index of the closer inside the recorded sequence.
    pub end_position: usize,
    /// The closer that failed to match.
    pub end: Interaction,
}

/// Payload describing a sub-protocol that was opened but never closed.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("missing end for {begin} opened at position {position}")]
pub struct MissingEndInfo {
    /// Index of the opener inside the recorded sequence.
    pub position: usize,
    /// The opener that was left unclosed.
    pub begin: Interaction,
}
