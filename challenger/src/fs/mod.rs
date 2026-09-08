//! Spongefish-style Fiat–Shamir transcript framework.
//!
//! - IETF draft: <https://datatracker.ietf.org/doc/draft-irtf-cfrg-fiat-shamir/>
//! - Spongefish reference: <https://github.com/arkworks-rs/spongefish>

// Submodules are private: the public surface is the curated re-export list below.
mod bound;
mod codecs;
mod domain_separator;
mod error;
mod pattern;
mod state;
mod transcript_field;
mod unit;

pub use bound::TranscriptBound;
pub use codecs::{
    BytesToFieldCodec, Codec, ExtensionFieldCodec, FieldToFieldCodec, MIN_CHALLENGE_SECURITY_BITS,
};
pub use domain_separator::{DomainSeparator, PROTOCOL_ID_LEN};
pub use error::{
    InvalidKindInfo, MismatchedBeginEndInfo, MissingBeginInfo, MissingEndInfo, TranscriptError,
};
pub use pattern::{
    Hierarchy, Interaction, InteractionPattern, Kind, Label, Length, Pattern, PatternPlayer,
    PatternState, TypeTag,
};
pub use state::{ProverState, VerifierState};
pub use transcript_field::TranscriptField;
pub use unit::{FieldUnit, Unit};

/// Whether a drop-time completeness check may raise its panic.
///
/// A panic raised while another unwinds is not a second failure to report.
///
/// ```text
///     panic in flight  ->  panic in drop  ->  process abort, uncatchable
/// ```
///
/// The failure already in flight carries the diagnosis, so the check yields to it.
#[cfg(panic = "unwind")]
pub(crate) fn drop_check_may_panic() -> bool {
    !std::thread::panicking()
}

/// Whether a drop-time completeness check may raise its panic.
///
/// Without unwinding the first panic already ends the process.
/// No second one can follow it, so the check is always the only failure in flight.
#[cfg(not(panic = "unwind"))]
pub(crate) const fn drop_check_may_panic() -> bool {
    true
}
