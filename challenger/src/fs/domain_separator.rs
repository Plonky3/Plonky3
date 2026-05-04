//! Per-instance binding for a transcript.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::CanObserve;
use crate::fs::pattern::InteractionPattern;
use crate::fs::unit::Unit;

/// Protocol-identifier length from IETF draft §3, in bytes.
pub const PROTOCOL_ID_LEN: usize = 64;

/// Terminator that closes the seeding stream so no seed is a prefix of another.
const DOMAIN_TAG: u8 = 0x80;

/// Per-instance binding for a transcript.
///
/// Pairs a fixed-length protocol identifier with a variable-length
/// instance label and a finalised step sequence.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct DomainSeparator<U: Unit = u8> {
    /// 64-byte protocol identifier: `[version | name | zero padding]`.
    protocol_id: [u8; PROTOCOL_ID_LEN],
    /// Instance-binding bytes appended to the seed.
    instance_label: Vec<u8>,
    /// Validated step sequence published by this separator.
    pattern: Arc<InteractionPattern>,
    /// Type-level marker for the sponge alphabet.
    _u: PhantomData<U>,
}

impl<U: Unit> DomainSeparator<U> {
    /// Build a separator from a version byte, a protocol name, and a pattern.
    ///
    /// # Panics
    ///
    /// When the protocol name does not fit in 63 bytes.
    #[must_use]
    pub fn new(version: u8, protocol_name: &[u8], pattern: InteractionPattern) -> Self {
        // Index 0 holds the version, leaving 63 bytes for the name.
        assert!(
            protocol_name.len() < PROTOCOL_ID_LEN,
            "protocol_name must fit in {} bytes (version byte takes the first slot)",
            PROTOCOL_ID_LEN - 1,
        );
        // [version | name | zero padding].
        let mut protocol_id = [0u8; PROTOCOL_ID_LEN];
        protocol_id[0] = version;
        protocol_id[1..1 + protocol_name.len()].copy_from_slice(protocol_name);
        Self {
            protocol_id,
            instance_label: Vec::new(),
            pattern: Arc::new(pattern),
            _u: PhantomData,
        }
    }

    /// Append bytes to the instance label. Chainable.
    pub fn instance(&mut self, bytes: &[u8]) -> &mut Self {
        self.instance_label.extend_from_slice(bytes);
        self
    }

    /// Append the 32-byte pattern fingerprint to the instance label.
    ///
    /// Two patterns differing in shape (rounds, kinds, labels, lengths) produce distinct seeds.
    pub fn bind_pattern_hash(&mut self) -> &mut Self {
        let h = self.pattern.pattern_hash();
        self.instance_label.extend_from_slice(&h);
        self
    }

    /// Read-only access to the protocol identifier.
    #[must_use]
    pub const fn protocol_id(&self) -> &[u8; PROTOCOL_ID_LEN] {
        &self.protocol_id
    }

    /// Read-only access to the instance label bytes.
    #[must_use]
    pub fn instance_label(&self) -> &[u8] {
        &self.instance_label
    }

    /// Read-only access to the validated pattern.
    #[must_use]
    pub const fn pattern(&self) -> &Arc<InteractionPattern> {
        &self.pattern
    }

    /// Absorb the seed into a byte-shaped sponge.
    ///
    /// Wire layout: `[protocol_id | len_be_4_bytes | label | DOMAIN_TAG]`.
    pub fn seed_bytes<C>(&self, challenger: &mut C)
    where
        C: CanObserve<u8>,
    {
        challenger.observe_slice(&self.protocol_id);
        // Length prefix prevents two labels of different lengths from colliding.
        let len = self.instance_label.len() as u32;
        challenger.observe_slice(&len.to_be_bytes());
        challenger.observe_slice(&self.instance_label);
        // Terminator stops later absorbs from extending the seed.
        challenger.observe(DOMAIN_TAG);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;
    use crate::CanObserve;
    use crate::fs::pattern::{Hierarchy, Interaction, InteractionPattern, Kind, Length};

    /// Captures every absorbed byte in order.
    #[derive(Default)]
    struct ByteRecorder {
        buf: Vec<u8>,
    }

    impl CanObserve<u8> for ByteRecorder {
        fn observe(&mut self, value: u8) {
            self.buf.push(value);
        }
    }

    fn empty_pattern() -> InteractionPattern {
        InteractionPattern::new(Vec::new()).unwrap()
    }

    fn challenge_pattern() -> InteractionPattern {
        InteractionPattern::new(vec![Interaction::new::<u64>(
            Hierarchy::Atomic,
            Kind::Challenge,
            "alpha",
            Length::Scalar,
        )])
        .unwrap()
    }

    #[test]
    fn protocol_id_layout_carries_version_and_name() {
        // Identifier is `[version | name | zero pad]`, no trailing data.
        let ds: DomainSeparator<u8> = DomainSeparator::new(1, b"demo", empty_pattern());
        let id = ds.protocol_id();
        assert_eq!(id[0], 1);
        assert_eq!(&id[1..5], b"demo");
        assert!(id[5..].iter().all(|&b| b == 0));
    }

    #[test]
    #[should_panic(expected = "protocol_name must fit")]
    fn protocol_name_too_long_panics() {
        // 64-byte name leaves no room for the version at index 0.
        let too_long = [b'x'; PROTOCOL_ID_LEN];
        let _ = DomainSeparator::<u8>::new(0, &too_long, empty_pattern());
    }

    #[test]
    fn seed_bytes_layout_matches_dsfs_remark_2_1() {
        // Wire layout: [id | len_be(5) | "hello" | DOMAIN_TAG] = 64 + 4 + 5 + 1.
        let mut ds: DomainSeparator<u8> = DomainSeparator::new(7, b"p3", empty_pattern());
        ds.instance(b"hello");
        let mut rec = ByteRecorder::default();
        ds.seed_bytes(&mut rec);

        assert_eq!(rec.buf.len(), PROTOCOL_ID_LEN + 4 + 5 + 1);
        assert_eq!(rec.buf[0], 7);
        assert_eq!(&rec.buf[1..3], b"p3");
        assert!(rec.buf[3..PROTOCOL_ID_LEN].iter().all(|&b| b == 0));
        assert_eq!(
            &rec.buf[PROTOCOL_ID_LEN..PROTOCOL_ID_LEN + 4],
            &[0, 0, 0, 5]
        );
        assert_eq!(&rec.buf[PROTOCOL_ID_LEN + 4..PROTOCOL_ID_LEN + 9], b"hello");
        assert_eq!(rec.buf[PROTOCOL_ID_LEN + 9], DOMAIN_TAG);
    }

    #[test]
    fn distinct_protocol_names_seed_different_streams() {
        // Different name -> different identifier bytes -> different seed.
        let a: DomainSeparator<u8> = DomainSeparator::new(1, b"a", empty_pattern());
        let b: DomainSeparator<u8> = DomainSeparator::new(1, b"b", empty_pattern());
        let mut ra = ByteRecorder::default();
        let mut rb = ByteRecorder::default();
        a.seed_bytes(&mut ra);
        b.seed_bytes(&mut rb);
        assert_ne!(ra.buf, rb.buf);
    }

    #[test]
    fn distinct_instance_labels_seed_different_streams() {
        // Different label -> different label bytes -> different seed.
        let mut a: DomainSeparator<u8> = DomainSeparator::new(1, b"proto", empty_pattern());
        a.instance(b"instance_a");
        let mut b: DomainSeparator<u8> = DomainSeparator::new(1, b"proto", empty_pattern());
        b.instance(b"instance_b");
        let mut ra = ByteRecorder::default();
        let mut rb = ByteRecorder::default();
        a.seed_bytes(&mut ra);
        b.seed_bytes(&mut rb);
        assert_ne!(ra.buf, rb.buf);
    }

    #[test]
    fn bind_pattern_hash_appends_thirty_two_bytes() {
        // The pattern hash is always 32 bytes wide.
        let mut ds: DomainSeparator<u8> = DomainSeparator::new(0, b"p", challenge_pattern());
        let len_before = ds.instance_label().len();
        ds.bind_pattern_hash();
        assert_eq!(ds.instance_label().len(), len_before + 32);
    }

    #[test]
    fn distinct_patterns_lead_to_distinct_seeds_via_pattern_hash() {
        // Two patterns differing only in label produce different fingerprints.
        let pat_a = InteractionPattern::new(vec![Interaction::new::<u64>(
            Hierarchy::Atomic,
            Kind::Challenge,
            "alpha",
            Length::Scalar,
        )])
        .unwrap();
        let pat_b = InteractionPattern::new(vec![Interaction::new::<u64>(
            Hierarchy::Atomic,
            Kind::Challenge,
            "beta",
            Length::Scalar,
        )])
        .unwrap();

        let mut a: DomainSeparator<u8> = DomainSeparator::new(0, b"p", pat_a);
        let mut b: DomainSeparator<u8> = DomainSeparator::new(0, b"p", pat_b);
        a.bind_pattern_hash();
        b.bind_pattern_hash();

        let mut ra = ByteRecorder::default();
        let mut rb = ByteRecorder::default();
        a.seed_bytes(&mut ra);
        b.seed_bytes(&mut rb);
        assert_ne!(ra.buf, rb.buf);
    }
}
