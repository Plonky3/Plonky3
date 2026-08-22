//! Recording-side state machine for a transcript pattern.

use alloc::vec::Vec;
use core::marker::PhantomData;

use super::Pattern;
use super::sequence::InteractionPattern;
use super::step::{Hierarchy, Interaction, Kind, Label, Length};
use crate::fs::unit::Unit;

/// Records a transcript pattern step by step, validating as it goes.
///
/// Misuse is reported at the offending call site instead of as a generic error at finalisation.
///
/// # Panics
///
/// - On drop without finalize or abort.
/// - On a closer that does not match the most recent opener.
/// - On an atomic step whose kind is incompatible with the surrounding sub-protocol.
/// - On a closer with no matching opener.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PatternState<U = u8>
where
    U: Unit,
{
    /// Growing list of recorded steps.
    interactions: Vec<Interaction>,
    /// Whether the recorder has been finalised or aborted.
    finalized: bool,
    /// Type-level marker for the sponge alphabet.
    _unit: PhantomData<U>,
}

impl<U: Unit> Default for PatternState<U> {
    fn default() -> Self {
        Self::new()
    }
}

impl<U: Unit> PatternState<U> {
    /// Build an empty recorder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            interactions: Vec::new(),
            finalized: false,
            _unit: PhantomData,
        }
    }

    /// Finish recording and return the validated pattern.
    ///
    /// # Panics
    ///
    /// - Already finalised or aborted.
    /// - Final validation rejects the recorded sequence.
    #[must_use]
    pub fn finalize(mut self) -> InteractionPattern {
        // Mark finalised first to avoid a double-panic via drop.
        assert!(!self.finalized, "Pattern is already finalized.");
        self.finalized = true;
        // Move the buffer out so the destructor sees an empty vector.
        let recorded = core::mem::take(&mut self.interactions);
        match InteractionPattern::new(recorded) {
            Ok(pattern) => pattern,
            Err(e) => panic!("Error validating interaction pattern: {e}"),
        }
    }

    /// Append a single recorded step, enforcing the structural rules.
    pub fn interact(&mut self, interaction: Interaction) {
        assert!(!self.finalized, "Pattern is already finalized.");
        if let Some(open) = self.last_open_begin() {
            // Nested-kind compatibility: Protocol accepts any kind, others must match.
            if interaction.hierarchy() == Hierarchy::Atomic
                && !(open.kind() == Kind::Protocol || open.kind() == interaction.kind())
            {
                let surrounding = open.kind();
                let inner = interaction.kind();
                self.finalized = true;
                panic!(
                    "Invalid interaction kind: surrounding {surrounding} does not allow nested {inner}",
                );
            }
            // A closer must match the most recent opener on every field.
            if interaction.hierarchy() == Hierarchy::End && !interaction.closes(open) {
                let open_clone = *open;
                self.finalized = true;
                panic!("Mismatched begin and end: open {open_clone}, close {interaction}",);
            }
        } else if interaction.hierarchy() == Hierarchy::End {
            // No surrounding sub-protocol: the closer is an orphan.
            self.finalized = true;
            panic!("Missing begin for {interaction}");
        }

        self.interactions.push(interaction);
    }

    /// Find the closest unclosed opener, if any.
    ///
    /// Walks from newest to oldest, balancing closers against openers.
    fn last_open_begin(&self) -> Option<&Interaction> {
        let mut depth: usize = 0;
        for interaction in self.interactions.iter().rev() {
            match interaction.hierarchy() {
                // Each closer adds one to the "still owed" pile.
                Hierarchy::End => depth += 1,
                Hierarchy::Begin => {
                    // The first opener with no pending closer is the answer.
                    if depth == 0 {
                        return Some(interaction);
                    }
                    depth -= 1;
                }
                // Atomics do not affect the matching pile.
                Hierarchy::Atomic => {}
            }
        }
        None
    }
}

impl<U: Unit> Drop for PatternState<U> {
    fn drop(&mut self) {
        // Loud failure surfaces forgot-to-finalize bugs.
        assert!(self.finalized, "Dropped unfinalized pattern recorder.");
    }
}

impl<U: Unit> Pattern for PatternState<U> {
    fn abort(&mut self) {
        // Idempotent so wrappers can safely call abort from their own Drop.
        self.finalized = true;
        self.interactions.clear();
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
    use crate::fs::pattern::step::{Hierarchy, Kind};

    #[test]
    fn empty_recorder_finalizes_to_empty_pattern() {
        // Zero interact() calls is legal.
        let p = PatternState::<u8>::new().finalize();
        assert!(p.is_empty());
    }

    #[test]
    fn record_then_finalize_round_trip() {
        // Three steps record cleanly into a validated pattern of length 3.
        let mut s = PatternState::<u8>::new();
        s.begin_protocol::<()>("outer");
        s.interact(Interaction::new::<u64>(
            Hierarchy::Atomic,
            Kind::Challenge,
            "alpha",
            Length::Scalar,
        ));
        s.end_protocol::<()>("outer");
        let p = s.finalize();
        assert_eq!(p.len(), 3);
    }

    #[test]
    #[should_panic(expected = "Dropped unfinalized pattern recorder.")]
    fn drop_without_finalize_panics() {
        // Forgot-to-finalize must surface, not silently lose data.
        let mut s = PatternState::<u8>::new();
        s.interact(Interaction::new::<u64>(
            Hierarchy::Atomic,
            Kind::Challenge,
            "alpha",
            Length::Scalar,
        ));
    }

    #[test]
    fn abort_disables_drop_check() {
        // Abort is the explicit discard path; drop must stay quiet.
        let mut s = PatternState::<u8>::new();
        s.interact(Interaction::new::<u64>(
            Hierarchy::Atomic,
            Kind::Challenge,
            "alpha",
            Length::Scalar,
        ));
        s.abort();
    }

    #[test]
    #[should_panic(expected = "Mismatched begin and end")]
    fn mismatched_close_panics_on_record() {
        // Open "a" then close "b".
        let mut s = PatternState::<u8>::new();
        s.begin_protocol::<()>("a");
        s.end_protocol::<()>("b");
    }

    #[test]
    #[should_panic(expected = "Missing begin for")]
    fn orphan_close_panics_on_record() {
        // Closer with no matching opener.
        let mut s = PatternState::<u8>::new();
        s.end_protocol::<()>("x");
    }

    #[test]
    #[should_panic(expected = "Invalid interaction kind")]
    fn nested_kind_mismatch_panics_on_record() {
        // Challenge atomic inside a Message-kind sub-protocol.
        let mut s = PatternState::<u8>::new();
        s.begin_message::<()>("msg-block", Length::None);
        s.interact(Interaction::new::<u64>(
            Hierarchy::Atomic,
            Kind::Challenge,
            "alpha",
            Length::Scalar,
        ));
    }

    #[test]
    fn protocol_opener_allows_mixed_kinds() {
        // Protocol container takes a Message and a Challenge side by side.
        let mut s = PatternState::<u8>::new();
        s.begin_protocol::<()>("outer");
        s.interact(Interaction::new::<u32>(
            Hierarchy::Atomic,
            Kind::Message,
            "commit",
            Length::Scalar,
        ));
        s.interact(Interaction::new::<u64>(
            Hierarchy::Atomic,
            Kind::Challenge,
            "alpha",
            Length::Scalar,
        ));
        s.end_protocol::<()>("outer");
        let p = s.finalize();
        assert_eq!(p.len(), 4);
    }
}
