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
pub use codecs::{BytesToFieldCodec, Codec, ExtensionFieldCodec, FieldToFieldCodec};
pub use domain_separator::{DomainSeparator, PROTOCOL_ID_LEN};
pub use error::{
    InvalidKindInfo, MismatchedBeginEndInfo, MissingBeginInfo, MissingEndInfo, PatternMismatchInfo,
    TranscriptError,
};
pub use pattern::{
    Hierarchy, Interaction, InteractionPattern, Kind, Label, Length, Pattern, PatternPlayer,
    PatternState,
};
pub use state::{ProverState, VerifierState};
pub use unit::Unit;
