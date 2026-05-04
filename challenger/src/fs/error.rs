//! Errors returned by the transcript machinery.

use alloc::boxed::Box;
use alloc::string::String;

use thiserror::Error;

use crate::fs::pattern::Interaction;

/// Failures that can arise while building or replaying a transcript.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum TranscriptError {
    /// An end marker was found with no matching opener in the sequence.
    #[error(transparent)]
    MissingBegin(Box<MissingBeginInfo>),

    /// An atomic step uses a kind incompatible with the surrounding sub-protocol.
    #[error(transparent)]
    InvalidKind(Box<InvalidKindInfo>),

    /// A closer does not match the most recent opener.
    #[error(transparent)]
    MismatchedBeginEnd(Box<MismatchedBeginEndInfo>),

    /// An opener was never closed before the sequence ended.
    #[error(transparent)]
    MissingEnd(Box<MissingEndInfo>),

    /// Playback ran past the end of the recorded sequence.
    #[error("transcript pattern exhausted")]
    PatternExhausted,

    /// Playback received a step that does not match the next recorded one.
    #[error(transparent)]
    PatternMismatch(Box<PatternMismatchInfo>),

    /// Verifier-side parsing of the prover's serialized output failed.
    #[error("bad proof shape: {reason}")]
    BadProofShape {
        /// Short reason describing the parse failure.
        reason: &'static str,
    },

    /// The salt supplied by the caller does not match the length declared by the recorded pattern.
    #[error("bad salt length: pattern declares {expected} bytes, caller supplied {got}")]
    BadSaltLen {
        /// Length recorded in the pattern, in bytes.
        expected: usize,
        /// Length supplied by the caller, in bytes.
        got: usize,
    },

    /// Free-form variant for failures best described in prose.
    #[error("{0}")]
    Other(String),
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

/// Payload describing an atomic step whose kind is incompatible with the surrounding sub-protocol.
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
    /// The offending nested interaction.
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

/// Payload describing a step that does not match the next recorded one.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("pattern mismatch: expected {expected}, got {got}")]
pub struct PatternMismatchInfo {
    /// The interaction that was expected, taken from the recorded sequence.
    pub expected: Interaction,
    /// The interaction that was actually requested by the caller.
    pub got: Interaction,
}
