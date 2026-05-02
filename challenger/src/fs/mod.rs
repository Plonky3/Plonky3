//! Spongefish-style Fiat–Shamir transcript framework.
//!
//! - IETF draft: <https://datatracker.ietf.org/doc/draft-irtf-cfrg-fiat-shamir/>
//! - Spongefish reference: <https://github.com/arkworks-rs/spongefish>

pub mod codecs;
pub mod domain_separator;
pub mod error;
pub mod pattern;
pub mod shake128;
pub mod state;
pub mod unit;

pub use codecs::{
    BytesToFieldCodec, Codec, ExtensionFieldCodec, FieldToBytesCodec, FieldToFieldCodec,
};
pub use domain_separator::{DomainSeparator, PROTOCOL_ID_LEN};
pub use error::{
    InvalidKindInfo, MismatchedBeginEndInfo, MissingBeginInfo, MissingEndInfo, PatternMismatchInfo,
    TranscriptError,
};
pub use pattern::{
    Hierarchy, Interaction, InteractionPattern, Kind, Label, Length, Pattern, PatternPlayer,
    PatternState,
};
pub use shake128::Shake128;
pub use state::{ProverState, VerifierState};
pub use unit::Unit;
