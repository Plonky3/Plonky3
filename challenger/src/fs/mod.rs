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
pub use unit::{FieldUnit, Unit};
