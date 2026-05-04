//! Playback-side state machine for a transcript pattern.

use alloc::sync::Arc;

use super::Pattern;
use super::sequence::InteractionPattern;
use super::step::{Hierarchy, Interaction, Kind, Label, Length};

/// Walks a recorded sequence and matches each request against the next expected step.
///
/// # Overview
///
/// - Holds a shared reference to a finalised pattern and a cursor.
/// - Each call advances the cursor by one position.
/// - A mismatch raises a panic with a diff between expected and observed.
/// - Successful playback must consume every recorded step.
///
/// # Panics
///
/// - On drop without finalize.
/// - On a call that does not match the next recorded step.
/// - On finalize when steps remain unread.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PatternPlayer {
    /// Pattern under playback.
    ///
    /// Shared via `Arc` so prover and verifier sides reuse one allocation.
    pattern: Arc<InteractionPattern>,
    /// Cursor into the recorded sequence.
    position: usize,
    /// Whether playback has been finalised or aborted.
    finalized: bool,
}

impl PatternPlayer {
    /// Build a player positioned at the start of `pattern`.
    #[must_use]
    pub const fn new(pattern: Arc<InteractionPattern>) -> Self {
        Self {
            pattern,
            position: 0,
            finalized: false,
        }
    }

    /// Read-only access to the pattern under playback.
    #[must_use]
    pub fn pattern(&self) -> &InteractionPattern {
        &self.pattern
    }

    /// Cursor position inside the pattern.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.position
    }

    /// Total number of steps in the pattern.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pattern.interactions().len()
    }

    /// Whether the pattern has zero steps.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pattern.interactions().is_empty()
    }

    /// Number of steps not yet replayed.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.len().saturating_sub(self.position)
    }

    /// Whether the player has been finalised or aborted.
    #[must_use]
    pub const fn is_finalized(&self) -> bool {
        self.finalized
    }

    /// Confirm that every recorded step has been replayed.
    ///
    /// # Panics
    ///
    /// - Already finalised or aborted.
    /// - At least one recorded step is still un-played.
    pub fn finalize(mut self) {
        assert!(!self.finalized, "Player is already finalized.");
        // Mark finalised first to avoid a double-panic via drop.
        self.finalized = true;
        assert!(
            self.position >= self.len(),
            "Pattern not fully replayed, expecting {}",
            self.pattern.interactions()[self.position]
        );
    }

    /// Replay the next recorded step and require it matches `interaction`.
    ///
    /// # Panics
    ///
    /// - Already finalised or aborted.
    /// - Cursor has run past the end of the recorded sequence.
    /// - Supplied step does not match the next recorded step.
    pub fn interact(&mut self, interaction: Interaction) {
        assert!(!self.finalized, "Player is already finalized.");
        let Some(expected) = self.pattern.interactions().get(self.position) else {
            // Mark finalised first so drop does not also panic.
            self.finalized = true;
            panic!("No more recorded interactions, but received {interaction}");
        };
        // Whole-step compare so any differing field shows up in the diff.
        if expected != &interaction {
            self.finalized = true;
            panic!("Received interaction {interaction}, but expected {expected}");
        }
        self.position += 1;
    }
}

impl Drop for PatternPlayer {
    fn drop(&mut self) {
        // Loud failure surfaces forgot-to-finalize bugs.
        assert!(self.finalized, "Dropped unfinalized pattern player.");
    }
}

impl Pattern for PatternPlayer {
    fn abort(&mut self) {
        // Idempotent so wrappers can safely call abort from their own Drop.
        self.finalized = true;
    }

    fn begin<T: ?Sized>(&mut self, label: Label, kind: Kind, length: Length) {
        self.interact(Interaction::new::<T>(Hierarchy::Begin, kind, label, length));
    }

    fn end<T: ?Sized>(&mut self, label: Label, kind: Kind, length: Length) {
        self.interact(Interaction::new::<T>(Hierarchy::End, kind, label, length));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fs::pattern::state::PatternState;
    use crate::fs::pattern::step::{Hierarchy, Kind};

    /// Outer Protocol container with one nested Challenge atomic.
    fn build_simple_pattern() -> InteractionPattern {
        let mut s = PatternState::<u8>::new();
        s.begin_protocol::<()>("Example protocol");
        s.interact(Interaction::new::<u64>(
            Hierarchy::Atomic,
            Kind::Challenge,
            "nonce",
            Length::Scalar,
        ));
        s.end_protocol::<()>("Example protocol");
        s.finalize()
    }

    #[test]
    fn record_then_play_back_exactly() {
        // Verbatim replay walks the whole pattern and finalises cleanly.
        let pattern = Arc::new(build_simple_pattern());
        let mut p = PatternPlayer::new(pattern);
        p.begin_protocol::<()>("Example protocol");
        p.interact(Interaction::new::<u64>(
            Hierarchy::Atomic,
            Kind::Challenge,
            "nonce",
            Length::Scalar,
        ));
        p.end_protocol::<()>("Example protocol");
        p.finalize();
    }

    #[test]
    #[should_panic(expected = "Dropped unfinalized pattern player.")]
    fn drop_without_finalize_panics() {
        // Replay one step then drop without finalize.
        let pattern = Arc::new(build_simple_pattern());
        let mut p = PatternPlayer::new(pattern);
        p.begin_protocol::<()>("Example protocol");
    }

    #[test]
    #[should_panic(expected = "Received interaction")]
    fn type_mismatch_panics() {
        // f64 instead of the recorded u64.
        let pattern = Arc::new(build_simple_pattern());
        let mut p = PatternPlayer::new(pattern);
        p.begin_protocol::<()>("Example protocol");
        p.interact(Interaction::new::<f64>(
            Hierarchy::Atomic,
            Kind::Challenge,
            "nonce",
            Length::Scalar,
        ));
    }

    #[test]
    #[should_panic(expected = "Received interaction")]
    fn label_mismatch_panics() {
        // Wrong label on replay.
        let pattern = Arc::new(build_simple_pattern());
        let mut p = PatternPlayer::new(pattern);
        p.begin_protocol::<()>("Example protocol");
        p.interact(Interaction::new::<u64>(
            Hierarchy::Atomic,
            Kind::Challenge,
            "different-label",
            Length::Scalar,
        ));
    }

    #[test]
    #[should_panic(expected = "Received interaction")]
    fn kind_mismatch_panics() {
        // Message replayed where the recorded step is a Challenge.
        let pattern = Arc::new(build_simple_pattern());
        let mut p = PatternPlayer::new(pattern);
        p.begin_protocol::<()>("Example protocol");
        p.interact(Interaction::new::<u64>(
            Hierarchy::Atomic,
            Kind::Message,
            "nonce",
            Length::Scalar,
        ));
    }

    #[test]
    #[should_panic(expected = "Received interaction")]
    fn length_mismatch_panics() {
        // Fixed(1) replayed where the recorded step uses Scalar.
        let pattern = Arc::new(build_simple_pattern());
        let mut p = PatternPlayer::new(pattern);
        p.begin_protocol::<()>("Example protocol");
        p.interact(Interaction::new::<u64>(
            Hierarchy::Atomic,
            Kind::Challenge,
            "nonce",
            Length::Fixed(1),
        ));
    }

    #[test]
    #[should_panic(expected = "Pattern not fully replayed")]
    fn finalize_before_end_panics() {
        // Cursor stops one step short of the closer.
        let pattern = Arc::new(build_simple_pattern());
        let mut p = PatternPlayer::new(pattern);
        p.begin_protocol::<()>("Example protocol");
        p.interact(Interaction::new::<u64>(
            Hierarchy::Atomic,
            Kind::Challenge,
            "nonce",
            Length::Scalar,
        ));
        p.finalize();
    }

    #[test]
    #[should_panic(expected = "No more recorded interactions")]
    fn extra_interact_after_end_panics() {
        // One extra step beyond the recorded sequence.
        let pattern = Arc::new(build_simple_pattern());
        let mut p = PatternPlayer::new(pattern);
        p.begin_protocol::<()>("Example protocol");
        p.interact(Interaction::new::<u64>(
            Hierarchy::Atomic,
            Kind::Challenge,
            "nonce",
            Length::Scalar,
        ));
        p.end_protocol::<()>("Example protocol");
        p.interact(Interaction::new::<u64>(
            Hierarchy::Atomic,
            Kind::Challenge,
            "extra",
            Length::Scalar,
        ));
    }

    #[test]
    fn abort_skips_drop_check() {
        // Aborting an in-flight player suppresses the drop-time panic.
        let pattern = Arc::new(build_simple_pattern());
        let mut p = PatternPlayer::new(pattern);
        p.begin_protocol::<()>("Example protocol");
        p.abort();
    }
}
