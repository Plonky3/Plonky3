//! Typed transcript-step vocabulary shared by the recorder and the player.
//!
//! # Overview
//!
//! Every transcript is described as a sequence of typed steps.
//!
//! The vocabulary is shared by two state machines:
//!
//! - One records a sequence step by step.
//! - One replays a finalised sequence against caller code.
//!
//! Both speak the same trait, so generic code can build a sub-protocol
//! once and run it against either side.
//!
//! # Lifecycle
//!
//! 1. Construct a recorder.
//! 2. Append steps via the trait helpers.
//! 3. Finalise the recorder to obtain a validated, hashable pattern.
//! 4. Wrap the pattern in a player.
//! 5. Replay each step in lockstep with prover or verifier code.
//! 6. Finalise the player to require full consumption.

mod player;
mod sequence;
mod state;
mod step;

pub use player::PatternPlayer;
pub use sequence::InteractionPattern;
pub use state::PatternState;
pub use step::{Hierarchy, Interaction, Kind, Label, Length};

/// Operations shared by every party that records or replays a transcript.
///
/// Two atomic primitives — an opener and a closer — cover all nesting.
/// Helpers are provided per semantic role.
///
/// Implementors enforce that every opener is matched by a closer with
/// the same kind, label, type, and length.
pub trait Pattern {
    /// Discard pending recording without enforcing the strict drop check.
    ///
    /// Idempotent: a second call is a no-op.
    fn abort(&mut self);

    /// Mark the start of a sub-protocol of arbitrary kind and length.
    fn begin<T: ?Sized>(&mut self, label: Label, kind: Kind, length: Length);

    /// Mark the end of a sub-protocol of arbitrary kind and length.
    fn end<T: ?Sized>(&mut self, label: Label, kind: Kind, length: Length);

    /// Open a protocol-kind sub-protocol with no carried length.
    fn begin_protocol<T: ?Sized>(&mut self, label: Label) {
        self.begin::<T>(label, Kind::Protocol, Length::None);
    }

    /// Close a protocol-kind sub-protocol.
    fn end_protocol<T: ?Sized>(&mut self, label: Label) {
        self.end::<T>(label, Kind::Protocol, Length::None);
    }

    /// Open a public-kind sub-protocol of the supplied length.
    fn begin_public<T: ?Sized>(&mut self, label: Label, length: Length) {
        self.begin::<T>(label, Kind::Public, length);
    }

    /// Close a public-kind sub-protocol.
    fn end_public<T: ?Sized>(&mut self, label: Label, length: Length) {
        self.end::<T>(label, Kind::Public, length);
    }

    /// Open a message-kind sub-protocol of the supplied length.
    fn begin_message<T: ?Sized>(&mut self, label: Label, length: Length) {
        self.begin::<T>(label, Kind::Message, length);
    }

    /// Close a message-kind sub-protocol.
    fn end_message<T: ?Sized>(&mut self, label: Label, length: Length) {
        self.end::<T>(label, Kind::Message, length);
    }

    /// Open a hint-kind sub-protocol of the supplied length.
    fn begin_hint<T: ?Sized>(&mut self, label: Label, length: Length) {
        self.begin::<T>(label, Kind::Hint, length);
    }

    /// Close a hint-kind sub-protocol.
    fn end_hint<T: ?Sized>(&mut self, label: Label, length: Length) {
        self.end::<T>(label, Kind::Hint, length);
    }

    /// Open a challenge-kind sub-protocol of the supplied length.
    fn begin_challenge<T: ?Sized>(&mut self, label: Label, length: Length) {
        self.begin::<T>(label, Kind::Challenge, length);
    }

    /// Close a challenge-kind sub-protocol.
    fn end_challenge<T: ?Sized>(&mut self, label: Label, length: Length) {
        self.end::<T>(label, Kind::Challenge, length);
    }
}
