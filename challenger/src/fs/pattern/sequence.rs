//! A validated, hashable sequence of typed transcript steps.
//!
//! # Overview
//!
//! Holds a list of typed steps that has already passed structural validation.
//!
//! Acts as the source of truth for what a prover and verifier may do.
//!
//! Yields a stable 32-byte fingerprint suitable for use as a protocol identifier.
//!
//! # Validation
//!
//! Walk the steps left to right, maintaining a stack of openers:
//!
//! - On a Begin: push `(position, opener)` onto the stack.
//! - On an End: pop the top of the stack and require it to match.
//! - On an Atomic: if the stack is non-empty, require the inner kind
//!   to be compatible with the surrounding kind (Protocol-kind openers
//!   accept any inner kind).
//!
//! At the end of the walk the stack must be empty.
//!
//! # Hashing
//!
//! The 32-byte fingerprint is computed by:
//!
//! 1. Rendering the pattern in alternate-mode `Display`.
//! 2. Feeding the rendered bytes through Keccak-256.
//!
//! The alternate-mode rendering is type-name-free and length-prefixed,
//! so the fingerprint is stable across compiler versions.

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::{Display, Formatter, Write};

use p3_keccak::Keccak256Hash;
use p3_symmetric::CryptographicHasher;

use crate::fs::error::{
    InvalidKindInfo, MismatchedBeginEndInfo, MissingBeginInfo, MissingEndInfo, TranscriptError,
};
use crate::fs::pattern::step::{Hierarchy, Interaction, Kind};

/// A complete, validated transcript described as a sequence of typed steps.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Default)]
pub struct InteractionPattern {
    /// The validated sequence in transcript order.
    interactions: Vec<Interaction>,
}

impl InteractionPattern {
    /// Build a pattern from a sequence of steps after validating its structure.
    ///
    /// # Errors
    ///
    /// Returns a structured error when the sequence is malformed;
    /// See the variants of the error type for the exact failure modes.
    ///
    /// # Algorithm
    ///
    /// Walks the sequence once, maintaining a stack of currently-open sub-protocols.
    pub fn new(interactions: Vec<Interaction>) -> Result<Self, TranscriptError> {
        // Build first, validate second.
        //
        // This lets validation report the original positions of offending
        // steps in the recorded sequence.
        let result = Self { interactions };
        result.validate()?;
        Ok(result)
    }

    /// Read-only access to the recorded sequence.
    #[must_use]
    pub fn interactions(&self) -> &[Interaction] {
        &self.interactions
    }

    /// Number of steps in the pattern.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.interactions.len()
    }

    /// Whether the pattern has zero steps.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.interactions.is_empty()
    }

    /// Stable 32-byte fingerprint of the pattern.
    ///
    /// # Algorithm
    ///
    /// 1. Render the pattern via alternate-mode `Display`.
    /// 2. Feed the rendered bytes into Keccak-256.
    ///
    /// The rendering is type-name-free and length-prefixed, so the
    /// digest is stable across compiler versions and free of label
    /// collision shenanigans.
    ///
    /// # Performance
    ///
    /// One allocation for the rendered string and one Keccak-256 call.
    /// Intended for use at protocol setup, not in hot paths.
    #[must_use]
    pub fn pattern_hash(&self) -> [u8; 32] {
        // Render into one contiguous buffer in alternate mode.
        //
        // The rendering is deterministic and self-delimiting.
        let mut rendered = String::new();
        // Writing into a String can only fail on allocation, which would
        // already have aborted; the result is safe to unwrap.
        write!(&mut rendered, "{self:#}").expect("writing into a String never fails");
        // Hash the rendered bytes with Keccak-256.
        Keccak256Hash.hash_iter(rendered.bytes())
    }

    /// Run the structural validation pass.
    ///
    /// See the module-level overview for the algorithm.
    fn validate(&self) -> Result<(), TranscriptError> {
        // Stack holds (position, opener) for each currently-open sub-protocol.
        //
        // Bounded by nesting depth, not by sequence length.
        let mut stack: Vec<(usize, &Interaction)> = Vec::new();

        for (position, interaction) in self.interactions.iter().enumerate() {
            match interaction.hierarchy() {
                // Phase: open a sub-protocol.
                //
                // Push regardless of kind.
                // The kind compatibility check belongs to the next nested step, not the opener.
                Hierarchy::Begin => stack.push((position, interaction)),

                // Phase: close a sub-protocol.
                //
                // The top of the stack must match this closer.
                // - An empty stack means an orphan closer;
                // - A non-matching top means a misnested begin/end pair.
                Hierarchy::End => {
                    let Some((begin_position, begin)) = stack.pop() else {
                        return Err(TranscriptError::MissingBegin(Box::new(MissingBeginInfo {
                            position,
                            end: *interaction,
                        })));
                    };
                    if !interaction.closes(begin) {
                        return Err(TranscriptError::MismatchedBeginEnd(Box::new(
                            MismatchedBeginEndInfo {
                                begin_position,
                                begin: *begin,
                                end_position: position,
                                end: *interaction,
                            },
                        )));
                    }
                }

                // Phase: atomic step.
                //
                // If a sub-protocol is open, its kind must accept the inner kind.
                // - Protocol-kind openers act as mixed containers and accept everything;
                // - Every other kind requires an exact match.
                Hierarchy::Atomic => {
                    let Some(&(begin_position, begin)) = stack.last() else {
                        // No surrounding sub-protocol:
                        // Any kind is allowed at the top level.
                        continue;
                    };
                    let surrounding = begin.kind();
                    if surrounding != Kind::Protocol && surrounding != interaction.kind() {
                        return Err(TranscriptError::InvalidKind(Box::new(InvalidKindInfo {
                            begin_position,
                            begin: *begin,
                            interaction_position: position,
                            interaction: *interaction,
                        })));
                    }
                }
            }
        }

        // Final invariant: the stack must drain to empty.
        //
        // A non-empty stack means at least one opener was never closed.
        if let Some((position, begin)) = stack.pop() {
            return Err(TranscriptError::MissingEnd(Box::new(MissingEndInfo {
                position,
                begin: *begin,
            })));
        }
        Ok(())
    }
}

impl Display for InteractionPattern {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        // Header: include the step count first to guarantee no shorter
        // pattern is a prefix of a longer one.
        let length = self.interactions.len();
        // Pad the position field so columns line up in default mode.
        let width = length.saturating_sub(1).to_string().len();
        writeln!(f, "Plonky3 Fiat-Shamir Transcript ({length} interactions)")?;

        // Indentation tracks open sub-protocols.
        //
        // The closer for a Begin sits at the same indentation as its opener.
        // So we decrement before printing when we hit an End.
        let mut indentation: usize = 0;
        for (position, interaction) in self.interactions.iter().enumerate() {
            write!(f, "{position:0>width$} ")?;
            if interaction.hierarchy() == Hierarchy::End {
                // Close shifts left by one level;
                // The body of this iteration prints at the closer's level.
                indentation = indentation.saturating_sub(1);
            }
            for _ in 0..indentation {
                write!(f, "  ")?;
            }
            // Forward the alternate flag so the whole pattern is either:
            // - hash-stable or
            // - human-readable consistently.
            if f.alternate() {
                writeln!(f, "{interaction:#}")?;
            } else {
                writeln!(f, "{interaction}")?;
            }
            if interaction.hierarchy() == Hierarchy::Begin {
                // Open shifts subsequent lines right by one level.
                indentation += 1;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::{format, vec};

    use super::*;
    use crate::fs::pattern::step::{Hierarchy, Kind, Length};

    fn atomic<T: ?Sized>(kind: Kind, label: &'static str, length: Length) -> Interaction {
        Interaction::new::<T>(Hierarchy::Atomic, kind, label, length)
    }

    fn open<T: ?Sized>(kind: Kind, label: &'static str) -> Interaction {
        Interaction::new::<T>(Hierarchy::Begin, kind, label, Length::None)
    }

    fn close<T: ?Sized>(kind: Kind, label: &'static str) -> Interaction {
        Interaction::new::<T>(Hierarchy::End, kind, label, Length::None)
    }

    #[test]
    fn empty_pattern_validates() {
        // Boundary: a zero-step transcript is structurally legal.
        let p = InteractionPattern::new(vec![]).expect("empty pattern is legal");
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);
    }

    #[test]
    fn well_nested_protocol_validates() {
        // Fixture state: one outer protocol containing one challenge.
        // Mutation: build, validate. Expectation: no error.
        let p = InteractionPattern::new(vec![
            open::<()>(Kind::Protocol, "outer"),
            atomic::<u64>(Kind::Challenge, "alpha", Length::Scalar),
            close::<()>(Kind::Protocol, "outer"),
        ])
        .expect("well-nested protocol is legal");
        assert_eq!(p.len(), 3);
    }

    #[test]
    fn missing_begin_is_caught() {
        // Orphan closer at position 0.
        let err = InteractionPattern::new(vec![close::<()>(Kind::Protocol, "outer")])
            .expect_err("orphan close must be rejected");
        match err {
            TranscriptError::MissingBegin(info) => assert_eq!(
                *info,
                MissingBeginInfo {
                    position: 0,
                    end: close::<()>(Kind::Protocol, "outer"),
                }
            ),
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn missing_end_is_caught() {
        // Opener at position 0 never closed.
        let err = InteractionPattern::new(vec![open::<()>(Kind::Protocol, "outer")])
            .expect_err("missing close must be rejected");
        match err {
            TranscriptError::MissingEnd(info) => assert_eq!(
                *info,
                MissingEndInfo {
                    position: 0,
                    begin: open::<()>(Kind::Protocol, "outer"),
                }
            ),
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn mismatched_begin_end_is_caught() {
        // Close label "inner" does not match open label "outer".
        let err = InteractionPattern::new(vec![
            open::<()>(Kind::Protocol, "outer"),
            close::<()>(Kind::Protocol, "inner"),
        ])
        .expect_err("mismatched begin/end must be rejected");
        match err {
            TranscriptError::MismatchedBeginEnd(info) => assert_eq!(
                *info,
                MismatchedBeginEndInfo {
                    begin_position: 0,
                    begin: open::<()>(Kind::Protocol, "outer"),
                    end_position: 1,
                    end: close::<()>(Kind::Protocol, "inner"),
                }
            ),
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn protocol_kind_acts_as_a_mixed_container() {
        // Invariant: a Protocol-kind opener accepts any nested kind.
        //
        // Fixture state: a Protocol container with Message + Challenge inside.
        let p = InteractionPattern::new(vec![
            open::<()>(Kind::Protocol, "outer"),
            atomic::<u32>(Kind::Message, "commitment", Length::Scalar),
            atomic::<u64>(Kind::Challenge, "alpha", Length::Scalar),
            close::<()>(Kind::Protocol, "outer"),
        ])
        .expect("Protocol container accepts mixed kinds");
        assert_eq!(p.len(), 4);
    }

    #[test]
    fn non_protocol_sub_protocol_enforces_kind_match() {
        // Message-kind container rejects a Challenge-kind atomic.
        let err = InteractionPattern::new(vec![
            open::<()>(Kind::Message, "msg-block"),
            atomic::<u64>(Kind::Challenge, "alpha", Length::Scalar),
            close::<()>(Kind::Message, "msg-block"),
        ])
        .expect_err("non-Protocol container must enforce kind match");

        match err {
            TranscriptError::InvalidKind(info) => assert_eq!(
                *info,
                InvalidKindInfo {
                    begin_position: 0,
                    begin: open::<()>(Kind::Message, "msg-block"),
                    interaction_position: 1,
                    interaction: atomic::<u64>(Kind::Challenge, "alpha", Length::Scalar),
                }
            ),
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn pattern_hash_is_stable() {
        // Invariant: pattern_hash is a deterministic function of the alternate-mode rendering.
        //
        // Fixture state: a small protocol with one nested challenge.
        let p = InteractionPattern::new(vec![
            open::<()>(Kind::Protocol, "test"),
            atomic::<u64>(Kind::Challenge, "nonce", Length::Scalar),
            close::<()>(Kind::Protocol, "test"),
        ])
        .unwrap();

        // Rendered string matches a known layout.
        let rendered = format!("{p:#}");
        let expected = "Plonky3 Fiat-Shamir Transcript (3 interactions)\n\
                        0 Begin Protocol 4 test None\n\
                        1   Atomic Challenge 5 nonce Scalar\n\
                        2 End Protocol 4 test None\n";
        assert_eq!(rendered, expected);

        // The hash is deterministic, so two calls must agree.
        let h = p.pattern_hash();
        assert_eq!(h.len(), 32);
        assert_eq!(p.pattern_hash(), h);
    }

    #[test]
    fn pattern_hash_distinguishes_two_patterns() {
        // Invariant:
        //
        // Two patterns differing in even one byte of their rendering produce different hashes.
        let p1 = InteractionPattern::new(vec![atomic::<u64>(Kind::Challenge, "a", Length::Scalar)])
            .unwrap();
        let p2 = InteractionPattern::new(vec![atomic::<u64>(Kind::Challenge, "b", Length::Scalar)])
            .unwrap();
        assert_ne!(p1.pattern_hash(), p2.pattern_hash());
    }
}
