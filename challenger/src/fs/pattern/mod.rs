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
pub use step::{Hierarchy, Interaction, Kind, Label, Length, TypeTag};

/// Operations shared by every party that records or replays a transcript.
///
/// One primitive — appending a step — covers everything.
/// Openers and closers are helpers on top of it.
///
/// Implementors enforce that every opener is matched by a closer with
/// the same kind, label, type, and length.
pub trait Pattern {
    /// Discard pending recording without enforcing the strict drop check.
    ///
    /// Idempotent: a second call is a no-op.
    fn abort(&mut self);

    /// Append one step, enforcing the structural rules of the side in play.
    fn interact(&mut self, interaction: Interaction);

    /// Mark the start of a sub-protocol of the given kind.
    fn begin<T: ?Sized>(&mut self, label: Label, kind: Kind) {
        self.interact(Interaction::marker::<T>(Hierarchy::Begin, kind, label));
    }

    /// Mark the end of a sub-protocol of the given kind.
    fn end<T: ?Sized>(&mut self, label: Label, kind: Kind) {
        self.interact(Interaction::marker::<T>(Hierarchy::End, kind, label));
    }

    /// Open a mixed container that accepts nested steps of any kind.
    fn begin_protocol<T: ?Sized>(&mut self, label: Label) {
        self.begin::<T>(label, Kind::Protocol);
    }

    /// Close a mixed container.
    fn end_protocol<T: ?Sized>(&mut self, label: Label) {
        self.end::<T>(label, Kind::Protocol);
    }
}
