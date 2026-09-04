//! Per-instance binding for a transcript.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::CanObserve;
use crate::fs::pattern::InteractionPattern;
use crate::fs::unit::Unit;

/// Protocol-identifier length from IETF draft §3, in bytes.
pub const PROTOCOL_ID_LEN: usize = 64;

/// Index of the name-length byte inside the protocol identifier.
/// - The version takes slot 0.
/// - The name length takes the final slot.
/// - The middle bytes hold the name itself.
const NAME_LEN_INDEX: usize = PROTOCOL_ID_LEN - 1;

/// Largest protocol name that fits: `[version | name | name_len]`.
const MAX_NAME_LEN: usize = PROTOCOL_ID_LEN - 2;

/// Terminator byte closing the seeding stream.
///
/// The length prefix on the instance label already makes the seed prefix-free.
/// This byte is therefore redundant for that purpose.
///
/// It is kept because the IETF draft ends its seeding stream with a domain tag.
/// Aligning the two later should not need a format change.
const DOMAIN_TAG: u8 = 0x80;

/// Per-instance binding for a transcript.
///
/// Pairs a fixed-length protocol identifier with a variable-length
/// instance label and a finalised step sequence.
///
/// The `U` parameter names the sponge alphabet the seed is absorbed into.
/// It is `u8` for a byte sponge and [`crate::fs::FieldUnit<F>`] for a sponge
/// that speaks a prime field natively.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct DomainSeparator<U: Unit = u8> {
    /// 64-byte protocol identifier: `[version | name | zero padding | name_len]`.
    protocol_id: [u8; PROTOCOL_ID_LEN],
    /// Instance-binding bytes appended to the seed, each chunk length-prefixed.
    instance_label: Vec<u8>,
    /// Validated step sequence published by this separator.
    pattern: Arc<InteractionPattern>,
    /// Type-level marker for the sponge alphabet.
    _u: PhantomData<U>,
}

impl<U: Unit> DomainSeparator<U> {
    /// Build a separator from a version byte, a protocol name, and a pattern.
    ///
    /// The identifier is laid out as `[version | name | zero padding | name_len]`.
    ///
    /// The final byte stores the name length.
    /// Without it, zero-padding maps `b"a"` and `b"a\0"` to the same identifier.
    /// Storing the length keeps such names on distinct seeds.
    ///
    /// # Panics
    ///
    /// - When the protocol name is empty.
    /// - When the protocol name is longer than `MAX_NAME_LEN` bytes.
    #[must_use]
    pub fn new(version: u8, protocol_name: &[u8], pattern: InteractionPattern) -> Self {
        // An empty name gives no domain separation at all between protocols.
        assert!(!protocol_name.is_empty(), "protocol_name must not be empty");
        // Slot 0 holds the version and the last slot holds the name length.
        assert!(
            protocol_name.len() <= MAX_NAME_LEN,
            "protocol_name must fit in {MAX_NAME_LEN} bytes \
             (version and length bytes take the first and last slots)",
        );
        // [version | name | zero padding | name_len].
        let mut protocol_id = [0u8; PROTOCOL_ID_LEN];
        protocol_id[0] = version;
        protocol_id[1..1 + protocol_name.len()].copy_from_slice(protocol_name);
        // The name length disambiguates names that share a zero-padded prefix.
        protocol_id[NAME_LEN_INDEX] = protocol_name.len() as u8;
        Self {
            protocol_id,
            instance_label: Vec::new(),
            pattern: Arc::new(pattern),
            _u: PhantomData,
        }
    }

    /// Append one delimited chunk to the instance label. Chainable.
    ///
    /// Each call writes `[len_be_4_bytes | bytes]`, so two calls never collapse
    /// into one:
    ///
    /// ```text
    ///     .instance(a).instance(b)  ->  [len(a) | a][len(b) | b]
    ///     .instance(a ++ b)         ->  [len(a) + len(b) | a b]
    /// ```
    ///
    /// A caller binding a public key and then a message gets real separation between the two.
    /// The label is not one opaque blob.
    ///
    /// # Panics
    ///
    /// When a single chunk is longer than `u32::MAX` bytes.
    pub fn instance(&mut self, bytes: &[u8]) -> &mut Self {
        let len = u32::try_from(bytes.len()).expect("instance chunk must fit in u32 bytes");
        self.instance_label.extend_from_slice(&len.to_be_bytes());
        self.instance_label.extend_from_slice(bytes);
        self
    }

    /// Read-only access to the protocol identifier.
    #[must_use]
    pub const fn protocol_id(&self) -> &[u8; PROTOCOL_ID_LEN] {
        &self.protocol_id
    }

    /// Read-only access to the delimited instance-label bytes.
    #[must_use]
    pub fn instance_label(&self) -> &[u8] {
        &self.instance_label
    }

    /// Read-only access to the validated pattern.
    #[must_use]
    pub const fn pattern(&self) -> &Arc<InteractionPattern> {
        &self.pattern
    }

    /// Absorb the seed into the sponge.
    ///
    /// Wire layout: `[protocol_id | pattern_hash | len_be_4_bytes | label | DOMAIN_TAG]`.
    ///
    /// The byte stream is built first, then handed to the alphabet:
    /// a byte sponge takes it verbatim, a field sponge packs it into elements.
    ///
    /// The pattern fingerprint is absorbed unconditionally.
    /// Two protocols differing only in transcript shape land on distinct states.
    /// Binding it here, not in an opt-in call, makes it impossible to forget or double-apply.
    pub fn seed<C>(&self, challenger: &mut C)
    where
        C: CanObserve<U::Item>,
    {
        let mut seed = Vec::with_capacity(PROTOCOL_ID_LEN + 32 + 4 + self.instance_label.len() + 1);
        seed.extend_from_slice(&self.protocol_id);
        // Pattern shape is part of the seed: distinct shapes cannot collide.
        seed.extend_from_slice(&self.pattern.pattern_hash());
        // Length prefix prevents two labels of different lengths from colliding.
        let len = self.instance_label.len() as u32;
        seed.extend_from_slice(&len.to_be_bytes());
        seed.extend_from_slice(&self.instance_label);
        // Terminator stops later absorbs from extending the seed.
        seed.push(DOMAIN_TAG);
        U::observe_bytes(challenger, &seed);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::CanObserve;
    use crate::fs::pattern::{Hierarchy, Interaction, InteractionPattern, Kind, Length};
    use crate::fs::unit::FieldUnit;

    /// Concrete field exercised in this module's tests.
    type F = BabyBear;

    /// Captures every absorbed value in order.
    #[derive(Default)]
    struct Recorder<T> {
        buf: Vec<T>,
    }

    impl<T> CanObserve<T> for Recorder<T> {
        fn observe(&mut self, value: T) {
            self.buf.push(value);
        }
    }

    fn empty_pattern() -> InteractionPattern {
        InteractionPattern::new(Vec::new()).unwrap()
    }

    #[test]
    fn protocol_id_layout_carries_version_name_and_length() {
        // Identifier is `[version | name | zero pad | name_len]`.
        let ds: DomainSeparator<u8> = DomainSeparator::new(1, b"demo", empty_pattern());
        let id = ds.protocol_id();
        assert_eq!(id[0], 1);
        assert_eq!(&id[1..5], b"demo");
        // Middle bytes are zero padding; the last byte carries the name length.
        assert!(id[5..NAME_LEN_INDEX].iter().all(|&b| b == 0));
        assert_eq!(id[NAME_LEN_INDEX], 4);
    }

    #[test]
    fn names_sharing_a_zero_padded_prefix_do_not_collide() {
        // Without the length byte, `b"a"` and `b"a\0"` would map to the same
        // zero-padded identifier. The length byte keeps them distinct.
        let a: DomainSeparator<u8> = DomainSeparator::new(1, b"a", empty_pattern());
        let b: DomainSeparator<u8> = DomainSeparator::new(1, b"a\0", empty_pattern());
        assert_ne!(a.protocol_id(), b.protocol_id());
    }

    #[test]
    #[should_panic(expected = "protocol_name must fit")]
    fn protocol_name_too_long_panics() {
        // A 63-byte name leaves no room for both the version and length bytes.
        let too_long = [b'x'; MAX_NAME_LEN + 1];
        let _ = DomainSeparator::<u8>::new(0, &too_long, empty_pattern());
    }

    #[test]
    #[should_panic(expected = "must not be empty")]
    fn empty_protocol_name_panics() {
        // An empty name provides no domain separation between protocols.
        let _ = DomainSeparator::<u8>::new(0, b"", empty_pattern());
    }

    #[test]
    fn seed_layout_matches_dsfs_remark_2_1() {
        // Wire layout: [id(64) | pattern_hash(32) | len_be(4) | label | DOMAIN_TAG].
        //
        // The label itself is one delimited chunk: [len_be(4) | "hello"].
        let pattern = empty_pattern();
        let expected_hash = pattern.pattern_hash();
        let mut ds: DomainSeparator<u8> = DomainSeparator::new(7, b"p3", pattern);
        ds.instance(b"hello");
        let mut rec = Recorder::<u8>::default();
        ds.seed(&mut rec);

        // Offsets into the seed stream.
        const HASH: usize = PROTOCOL_ID_LEN;
        const LEN: usize = HASH + 32;
        const LABEL: usize = LEN + 4;

        // Label region is 4 length bytes plus the 5 payload bytes.
        assert_eq!(rec.buf.len(), PROTOCOL_ID_LEN + 32 + 4 + 9 + 1);
        assert_eq!(rec.buf[0], 7);
        assert_eq!(&rec.buf[1..3], b"p3");
        assert!(rec.buf[3..NAME_LEN_INDEX].iter().all(|&b| b == 0));
        assert_eq!(rec.buf[NAME_LEN_INDEX], 2);
        // Pattern fingerprint is bound automatically right after the identifier.
        assert_eq!(&rec.buf[HASH..LEN], &expected_hash);
        // Big-endian length prefix of the whole label region.
        assert_eq!(&rec.buf[LEN..LABEL], &[0, 0, 0, 9]);
        // Delimiter of the single chunk, then its payload.
        assert_eq!(&rec.buf[LABEL..LABEL + 4], &[0, 0, 0, 5]);
        assert_eq!(&rec.buf[LABEL + 4..LABEL + 9], b"hello");
        assert_eq!(rec.buf[LABEL + 9], DOMAIN_TAG);
    }

    #[test]
    fn distinct_protocol_names_seed_different_streams() {
        // Different name -> different identifier bytes -> different seed.
        let a: DomainSeparator<u8> = DomainSeparator::new(1, b"a", empty_pattern());
        let b: DomainSeparator<u8> = DomainSeparator::new(1, b"b", empty_pattern());
        let mut ra = Recorder::<u8>::default();
        let mut rb = Recorder::<u8>::default();
        a.seed(&mut ra);
        b.seed(&mut rb);
        assert_ne!(ra.buf, rb.buf);
    }

    #[test]
    fn distinct_instance_labels_seed_different_streams() {
        // Different label -> different label bytes -> different seed.
        let mut a: DomainSeparator<u8> = DomainSeparator::new(1, b"proto", empty_pattern());
        a.instance(b"instance_a");
        let mut b: DomainSeparator<u8> = DomainSeparator::new(1, b"proto", empty_pattern());
        b.instance(b"instance_b");
        let mut ra = Recorder::<u8>::default();
        let mut rb = Recorder::<u8>::default();
        a.seed(&mut ra);
        b.seed(&mut rb);
        assert_ne!(ra.buf, rb.buf);
    }

    #[test]
    fn instance_chunks_are_delimited() {
        // Invariant: two chunks never collapse into their concatenation.
        //
        // A caller binding a key and then a message must not land on the same
        // seed as one binding the two glued together.
        let mut split: DomainSeparator<u8> = DomainSeparator::new(1, b"proto", empty_pattern());
        split.instance(b"key").instance(b"message");
        let mut glued: DomainSeparator<u8> = DomainSeparator::new(1, b"proto", empty_pattern());
        glued.instance(b"keymessage");

        assert_ne!(split.instance_label(), glued.instance_label());

        let mut rs = Recorder::<u8>::default();
        let mut rg = Recorder::<u8>::default();
        split.seed(&mut rs);
        glued.seed(&mut rg);
        assert_ne!(rs.buf, rg.buf);
    }

    #[test]
    fn distinct_patterns_lead_to_distinct_seeds_automatically() {
        // Two patterns differing only in a label seed different streams,
        // without any explicit binding call: `seed` always folds in
        // the pattern fingerprint.
        let pat_a = InteractionPattern::new(vec![Interaction::algebra::<F, F>(
            Hierarchy::Atomic,
            Kind::Challenge,
            "alpha",
            Length::Scalar,
        )])
        .unwrap();
        let pat_b = InteractionPattern::new(vec![Interaction::algebra::<F, F>(
            Hierarchy::Atomic,
            Kind::Challenge,
            "beta",
            Length::Scalar,
        )])
        .unwrap();

        let a: DomainSeparator<u8> = DomainSeparator::new(0, b"p", pat_a);
        let b: DomainSeparator<u8> = DomainSeparator::new(0, b"p", pat_b);

        let mut ra = Recorder::<u8>::default();
        let mut rb = Recorder::<u8>::default();
        a.seed(&mut ra);
        b.seed(&mut rb);
        assert_ne!(ra.buf, rb.buf);
    }

    #[test]
    fn field_alphabet_seeds_a_native_field_sponge() {
        // Invariant: the same seed bytes reach a field sponge as field elements.
        //
        // The alphabet decides the packing; the seed content is unchanged.
        let ds: DomainSeparator<FieldUnit<F>> = DomainSeparator::new(1, b"duplex", empty_pattern());
        let mut rec = Recorder::<F>::default();
        ds.seed(&mut rec);

        // Seed is 64 + 32 + 4 + 0 + 1 = 101 bytes, packed 3 bytes per element,
        // behind one leading length element.
        assert_eq!(rec.buf.len(), 1 + 101_usize.div_ceil(3));
        assert_eq!(rec.buf[0], F::from_u32(101));
    }

    #[test]
    fn field_alphabet_separates_distinct_protocols() {
        // Domain separation survives the byte-to-field packing.
        let a: DomainSeparator<FieldUnit<F>> = DomainSeparator::new(1, b"a", empty_pattern());
        let b: DomainSeparator<FieldUnit<F>> = DomainSeparator::new(1, b"b", empty_pattern());
        let mut ra = Recorder::<F>::default();
        let mut rb = Recorder::<F>::default();
        a.seed(&mut ra);
        b.seed(&mut rb);
        assert_ne!(ra.buf, rb.buf);
    }
}
