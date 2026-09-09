//! Shared checks and captures for the typed Fiat-Shamir transcript layer.
//!
//! # Overview
//!
//! Two things a transcript test needs and the protocol crates cannot build for themselves.
//!
//! - A sponge stand-in that keeps every absorbed value, in order.
//! - One comparable value standing for a whole transcript seed.
//! - The grinding difficulties a recorded transcript description demands.
//!
//! # Comparing two seeds
//!
//! Domain separation is a statement about the bytes a sponge absorbs at seeding time.
//!
//! ```text
//!     two seeds differ  ->  every later challenge is drawn from a different state
//!     two seeds agree   ->  the two runs are one protocol as far as Fiat-Shamir knows
//! ```
//!
//! Sampling a challenge to compare two seeds tests the sponge as well as the binding.
//! Comparing digests of the seed streams tests the binding alone.

use alloc::vec::Vec;
use core::fmt::{Debug, Display, Formatter, Result as FmtResult};

use p3_keccak::Keccak256Hash;
use p3_symmetric::CryptographicHasher;

use crate::CanObserve;
use crate::fs::{DomainSeparator, InteractionPattern, Kind, Label, Length, Unit};

/// Captures every value absorbed into it, in order.
///
/// Stands in for a sponge wherever a test needs the absorbed sequence itself.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Recorder<T> {
    /// Values absorbed so far, oldest first.
    absorbed: Vec<T>,
}

impl<T> Recorder<T> {
    /// An empty recorder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            absorbed: Vec::new(),
        }
    }

    /// The values absorbed so far, oldest first.
    #[must_use]
    pub fn absorbed(&self) -> &[T] {
        &self.absorbed
    }

    /// Consume the recorder and yield the values it absorbed.
    #[must_use]
    pub fn into_absorbed(self) -> Vec<T> {
        self.absorbed
    }
}

impl<T> Default for Recorder<T> {
    /// An empty recorder.
    ///
    /// Written by hand rather than derived, which would demand `T: Default`.
    fn default() -> Self {
        Self::new()
    }
}

impl<T> CanObserve<T> for Recorder<T> {
    fn observe(&mut self, value: T) {
        self.absorbed.push(value);
    }
}

/// Canonical fingerprint of one transcript seed.
///
/// Holds the Keccak-256 digest of the byte stream absorbed at seeding time.
/// Two separators share a digest exactly when they seed a sponge with the same bytes.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SeedDigest([u8; 32]);

impl Debug for SeedDigest {
    /// Render the digest as lowercase hex, so a failing assertion names both seeds.
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        self.0.iter().try_for_each(|byte| write!(f, "{byte:02x}"))
    }
}

/// Digest the whole byte stream a separator seeds its sponge with.
///
/// The stream is `[protocol_id | pattern_hash | len_be_4_bytes | label | domain_tag]`.
/// It is assembled before the alphabet packs it, so the digest does not depend on `U`.
///
/// This is the value that must be unique per protocol and configuration.
/// Every knob a protocol claims to bind has to move it.
#[must_use]
pub fn seed_digest<U: Unit>(separator: &DomainSeparator<U>) -> SeedDigest {
    SeedDigest(Keccak256Hash.hash_iter(separator.seed_bytes()))
}

/// Every grinding step a pattern describes, in transcript order.
///
/// A proof-of-work step carries its difficulty as a fixed length.
///
/// A grind repeated once per round is one entry per round.
///
/// This is the transcript half of a grinding-accounting check.
///
/// The security half reads the same difficulties out of a parameter set.
///
/// # Panics
///
/// When a proof-of-work step carries any other length variant.
#[must_use]
pub fn pow_difficulties(pattern: &InteractionPattern) -> Vec<(Label, usize)> {
    pattern
        .interactions()
        .iter()
        .filter(|interaction| interaction.kind() == Kind::Pow)
        .map(|interaction| match interaction.length() {
            Length::Fixed(bits) => (interaction.label(), bits),
            other => panic!(
                "the `{}` proof-of-work step carries `{other}`, not a `Fixed` difficulty",
                interaction.label(),
            ),
        })
        .collect()
}

/// Panic unless every labelled seed differs from every other one in the set.
///
/// Pairwise is the strong form of the separation claim.
///
/// ```text
///     each differs from one baseline  ->  two knobs may still share a seed
///     each differs from every other   ->  no two configurations share a seed
/// ```
///
/// Labels only appear in the panic message, so any printable type serves.
///
/// # Panics
///
/// When two entries share a digest, naming both labels and the digest they share.
pub fn assert_seeds_pairwise_distinct<L: Display>(seeds: &[(L, SeedDigest)]) {
    for (index, (left_label, left)) in seeds.iter().enumerate() {
        for (right_label, right) in &seeds[index + 1..] {
            assert_ne!(
                left, right,
                "`{left_label}` and `{right_label}` land on the same transcript seed",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;

    use super::*;
    use crate::fs::{FieldUnit, InteractionPattern};

    /// A separator over the byte alphabet, named and labelled by the caller.
    fn byte_separator(name: &[u8], label: &[u8]) -> DomainSeparator<u8> {
        let pattern = InteractionPattern::new(Vec::new()).unwrap();
        let mut separator = DomainSeparator::new(1, name, pattern);
        separator.instance(label);
        separator
    }

    #[test]
    fn a_recorder_keeps_every_absorbed_value_in_order() {
        // A recorder starts empty and grows by one entry per absorb.
        let mut recorder = Recorder::<u8>::new();
        recorder.observe_slice(&[7, 8]);
        recorder.observe(9);
        assert_eq!(recorder.absorbed(), &[7, 8, 9]);
        assert_eq!(recorder.into_absorbed(), vec![7, 8, 9]);
    }

    #[test]
    fn the_digest_is_the_digest_of_the_recorded_byte_stream() {
        // The byte alphabet absorbs the seed verbatim, so the two views must agree.
        let separator = byte_separator(b"proto", b"instance");
        let mut recorder = Recorder::<u8>::new();
        separator.seed(&mut recorder);
        let recorded = SeedDigest(Keccak256Hash.hash_iter(recorder.into_absorbed()));
        assert_eq!(seed_digest(&separator), recorded);
    }

    #[test]
    fn the_digest_ignores_the_sponge_alphabet() {
        // The stream is assembled before packing, so re-keying the alphabet cannot move it.
        let bytes: DomainSeparator<u8> = byte_separator(b"proto", b"instance");
        let pattern = InteractionPattern::new(Vec::new()).unwrap();
        let mut fields: DomainSeparator<FieldUnit<BabyBear>> =
            DomainSeparator::new(1, b"proto", pattern);
        fields.instance(b"instance");
        assert_eq!(seed_digest(&bytes), seed_digest(&fields));
    }

    #[test]
    fn a_differing_name_or_label_moves_the_digest() {
        // Both halves of the seed reach the digest.
        let base = seed_digest(&byte_separator(b"proto", b"instance"));
        assert_ne!(base, seed_digest(&byte_separator(b"other", b"instance")));
        assert_ne!(base, seed_digest(&byte_separator(b"proto", b"other")));
    }

    #[test]
    fn a_set_of_distinct_seeds_passes_the_pairwise_check() {
        // Three names, three seeds, no collision to report.
        let seeds = [
            ("a", seed_digest(&byte_separator(b"a", b"i"))),
            ("b", seed_digest(&byte_separator(b"b", b"i"))),
            ("c", seed_digest(&byte_separator(b"c", b"i"))),
        ];
        assert_seeds_pairwise_distinct(&seeds);
    }

    #[test]
    #[should_panic(expected = "`second` and `third` land on the same transcript seed")]
    fn the_pairwise_check_names_the_two_labels_that_collided() {
        // Invariant: the report identifies which pair failed, not merely that one did.
        //
        //     first  : name "a"
        //     second : name "b"
        //     third  : name "b"   -> collides with `second`, not with `first`
        let seeds = [
            (
                String::from("first"),
                seed_digest(&byte_separator(b"a", b"i")),
            ),
            (
                String::from("second"),
                seed_digest(&byte_separator(b"b", b"i")),
            ),
            (
                String::from("third"),
                seed_digest(&byte_separator(b"b", b"i")),
            ),
        ];
        assert_seeds_pairwise_distinct(&seeds);
    }
}
