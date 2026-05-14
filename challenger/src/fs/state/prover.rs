//! Prover-side driver.

use alloc::vec::Vec;

use p3_field::{BasedVectorSpace, PrimeField};

use crate::fs::bound::TranscriptBound;
use crate::fs::codecs::Codec;
use crate::fs::codecs::decode_field::{encode_field_be, field_byte_size};
use crate::fs::codecs::length_prefix::{bound_byte_width, encode_len_be};
use crate::fs::domain_separator::DomainSeparator;
use crate::fs::pattern::{Hierarchy, Interaction, Kind, Label, Length, Pattern, PatternPlayer};
use crate::fs::unit::Unit;
use crate::{CanObserve, GrindingChallenger};

/// Drives a prover-side transcript in lockstep with a recorded pattern.
///
/// Each absorb or sample method advances the player by one step.
/// The serialised wire bytes are yielded only at finalisation.
pub struct ProverState<C, U: Unit = u8> {
    /// Underlying sponge that absorbs prover messages and yields challenges.
    challenger: C,
    /// Pattern player that validates each call against the recorded sequence.
    player: PatternPlayer,
    /// Accumulated wire bytes returned at finalisation.
    narg: Vec<u8>,
    /// Type-level marker for the sponge alphabet.
    _u: core::marker::PhantomData<U>,
}

impl<C, U: Unit> Drop for ProverState<C, U> {
    fn drop(&mut self) {
        // Abort the player on drop when the user did not finalise.
        //
        // This way, paths do not turn into double-panics during cleanup.
        if !self.player.is_finalized() {
            self.player.abort();
        }
    }
}

impl<C, U: Unit> ProverState<C, U> {
    /// Build a driver and seed the challenger from the domain separator.
    pub fn new(mut challenger: C, ds: &DomainSeparator<U>) -> Self
    where
        C: CanObserve<u8>,
    {
        // Seed first so two distinct (protocol, instance) pairs land on distinct sponge states.
        ds.seed_bytes(&mut challenger);
        let player = PatternPlayer::new(ds.pattern().clone());
        Self {
            challenger,
            player,
            narg: Vec::new(),
            _u: core::marker::PhantomData,
        }
    }

    /// Read-only access to the underlying challenger.
    pub const fn challenger(&self) -> &C {
        &self.challenger
    }

    /// Read-only access to the bytes buffered for the proof so far.
    pub fn narg(&self) -> &[u8] {
        &self.narg
    }

    /// Finalise the driver and return the serialised wire bytes.
    ///
    /// # Panics
    ///
    /// When the recorded pattern is not fully consumed.
    pub fn finalize(self) -> Vec<u8> {
        // The custom Drop impl prevents direct destructuring of `self`;
        // So move each field out by hand under ManuallyDrop.
        let this = core::mem::ManuallyDrop::new(self);
        // SAFETY: each field is moved out exactly once, then Drop never runs on `this`.
        let player = unsafe { core::ptr::read(&this.player) };
        let narg = unsafe { core::ptr::read(&this.narg) };
        let challenger = unsafe { core::ptr::read(&this.challenger) };
        // Drop the challenger explicitly: the wrapper's Drop is suppressed.
        drop(challenger);
        // Strict pattern-fully-consumed check.
        player.finalize();
        narg
    }

    /// Absorb a salt step and record its bytes in the wire buffer.
    pub fn add_salt(&mut self, salt_bytes: &[u8])
    where
        C: CanObserve<u8>,
    {
        // Validate: the next pattern step is a salt of the given length.
        self.player.interact(Interaction::new::<u8>(
            Hierarchy::Atomic,
            Kind::Salt,
            "salt",
            Length::Fixed(salt_bytes.len()),
        ));
        // Absorb so future samples depend on the salt.
        self.challenger.observe_slice(salt_bytes);
        // Mirror the same bytes onto the wire so the verifier can re-absorb them.
        self.narg.extend_from_slice(salt_bytes);
    }

    /// Absorb one prover scalar through the supplied codec.
    pub fn add_scalar<F, Cdc>(&mut self, label: Label, value: &F) -> TranscriptBound<F>
    where
        F: PrimeField,
        Cdc: Codec<C, F>,
    {
        // Validate: the next pattern step is a scalar message of type `F`.
        self.player.interact(Interaction::new::<F>(
            Hierarchy::Atomic,
            Kind::Message,
            label,
            Length::Scalar,
        ));
        // Absorb through the codec; the codec decides field vs. byte path.
        Cdc::observe(&mut self.challenger, value);
        // Append the canonical big-endian encoding to the wire.
        let bytes = encode_field_be::<F>(value);
        self.narg.extend_from_slice(&bytes);
        TranscriptBound::wrap(*value)
    }

    /// Absorb a known-length list of scalars under a single pattern step.
    ///
    /// No length prefix is written; the recorded pattern is the source of truth.
    pub fn add_scalars<F, Cdc>(&mut self, label: Label, values: &[F]) -> Vec<TranscriptBound<F>>
    where
        F: PrimeField,
        Cdc: Codec<C, F>,
    {
        // Validate: the next pattern step is a fixed-length list of scalars.
        self.player.interact(Interaction::new::<F>(
            Hierarchy::Atomic,
            Kind::Message,
            label,
            Length::Fixed(values.len()),
        ));
        // Absorb each value, mirror its canonical encoding to the wire, and bind it.
        let mut bound = Vec::with_capacity(values.len());
        for v in values {
            Cdc::observe(&mut self.challenger, v);
            let bytes = encode_field_be::<F>(v);
            self.narg.extend_from_slice(&bytes);
            bound.push(TranscriptBound::wrap(*v));
        }
        bound
    }

    /// Absorb a variable-length list of at most `max` scalars under a single step.
    ///
    /// # Wire format
    ///
    /// ```text
    ///     [len in W bytes, big-endian][canonical encoding of each value]
    /// ```
    ///
    /// # Sponge layout
    ///
    /// ```text
    ///     absorb: [the same length bytes][each value through the codec]
    /// ```
    ///
    /// # Why absorb the length first
    ///
    /// The sponge transcript stays prefix-free.
    ///
    /// No shorter run of this step is a prefix of a longer one.
    ///
    /// This matches the soundness condition from CO25 §6.2.
    ///
    /// # Panics
    ///
    /// When the supplied slice is longer than `max`.
    pub fn add_scalars_bounded<F, Cdc>(
        &mut self,
        label: Label,
        values: &[F],
        max: usize,
    ) -> Vec<TranscriptBound<F>>
    where
        F: PrimeField,
        C: CanObserve<u8>,
        Cdc: Codec<C, F>,
    {
        // Caller bug: writing more than the cap would diverge from the recorded pattern.
        assert!(
            values.len() <= max,
            "message length {} exceeds declared maximum {max}",
            values.len(),
        );
        // Validate against the recorded pattern step so shape divergence panics here.
        self.player.interact(Interaction::new::<F>(
            Hierarchy::Atomic,
            Kind::Message,
            label,
            Length::Bounded(max),
        ));
        // Prefix width is deterministic on both sides from the recorded bound.
        let width = bound_byte_width(max);
        let len_bytes = encode_len_be(values.len(), width);
        // Bind the actual count into the sponge before any value enters it.
        //
        // This is what keeps the transcript prefix-free for variable-length steps.
        self.challenger.observe_slice(&len_bytes[..width]);
        // Mirror the same prefix onto the wire so the verifier sees the count.
        self.narg.extend_from_slice(&len_bytes[..width]);
        // Absorb each value through the codec and write its canonical encoding.
        let mut bound = Vec::with_capacity(values.len());
        for v in values {
            // Sponge path: absorb the value in whichever shape the codec dictates.
            Cdc::observe(&mut self.challenger, v);
            // Wire path: write a canonical big-endian field encoding.
            let bytes = encode_field_be::<F>(v);
            self.narg.extend_from_slice(&bytes);
            // Hand back a binding witness so callers can prove the value was threaded through.
            bound.push(TranscriptBound::wrap(*v));
        }
        bound
    }

    /// Absorb one extension-field element coefficient by coefficient.
    pub fn add_extension<F, EF, Cdc>(&mut self, label: Label, value: &EF) -> TranscriptBound<EF>
    where
        F: PrimeField,
        EF: BasedVectorSpace<F> + Copy,
        Cdc: Codec<C, F>,
    {
        // Validate: the next pattern step is a scalar message of extension type.
        self.player.interact(Interaction::new::<EF>(
            Hierarchy::Atomic,
            Kind::Message,
            label,
            Length::Scalar,
        ));
        // Each coefficient travels through the same codec and wire format as a scalar.
        for coeff in value.as_basis_coefficients_slice() {
            Cdc::observe(&mut self.challenger, coeff);
            let bytes = encode_field_be::<F>(coeff);
            self.narg.extend_from_slice(&bytes);
        }
        TranscriptBound::wrap(*value)
    }

    /// Append a hint to the wire buffer without absorbing it into the sponge.
    ///
    /// Hint bytes are part of the wire format but never enter the sponge,
    /// so they cannot influence sampled challenges.
    pub fn add_hint(&mut self, label: Label, bytes: &[u8]) {
        self.player.interact(Interaction::new::<u8>(
            Hierarchy::Atomic,
            Kind::Hint,
            label,
            Length::Fixed(bytes.len()),
        ));
        self.narg.extend_from_slice(bytes);
    }

    /// Append a variable-length hint of at most `max` bytes.
    ///
    /// # Wire format
    ///
    /// ```text
    ///     [len in W bytes, big-endian][payload bytes]
    /// ```
    ///
    /// `W` is the minimum width that can encode `max`.
    ///
    /// # Sponge behaviour
    ///
    /// Neither the length prefix nor the payload enter the sponge.
    ///
    /// This step cannot influence any later challenge.
    ///
    /// # Panics
    ///
    /// When the supplied byte count exceeds `max`.
    pub fn add_hint_bounded(&mut self, label: Label, bytes: &[u8], max: usize) {
        // Caller bug: writing more than the cap would silently truncate on the verifier side.
        assert!(
            bytes.len() <= max,
            "hint length {} exceeds declared maximum {max}",
            bytes.len(),
        );
        // Validate against the recorded pattern step so any shape divergence panics here.
        self.player.interact(Interaction::new::<u8>(
            Hierarchy::Atomic,
            Kind::Hint,
            label,
            Length::Bounded(max),
        ));
        // Prefix width is deterministic on both sides from the recorded bound.
        let width = bound_byte_width(max);
        // Push the big-endian length onto the wire.
        let len_bytes = encode_len_be(bytes.len(), width);
        self.narg.extend_from_slice(&len_bytes[..width]);
        // Payload follows the prefix verbatim — never absorbed into the sponge.
        self.narg.extend_from_slice(bytes);
    }

    /// Sample one challenge scalar of type `F` via codec `Cdc`.
    pub fn challenge_scalar<F, Cdc>(&mut self, label: Label) -> TranscriptBound<F>
    where
        F: PrimeField,
        Cdc: Codec<C, F>,
    {
        self.player.interact(Interaction::new::<F>(
            Hierarchy::Atomic,
            Kind::Challenge,
            label,
            Length::Scalar,
        ));
        TranscriptBound::wrap(Cdc::sample(&mut self.challenger))
    }

    /// Sample `n` challenge scalars of type `F` under a single pattern step.
    pub fn challenge_scalars<F, Cdc>(&mut self, label: Label, n: usize) -> Vec<TranscriptBound<F>>
    where
        F: PrimeField,
        Cdc: Codec<C, F>,
    {
        self.player.interact(Interaction::new::<F>(
            Hierarchy::Atomic,
            Kind::Challenge,
            label,
            Length::Fixed(n),
        ));
        (0..n)
            .map(|_| TranscriptBound::wrap(Cdc::sample(&mut self.challenger)))
            .collect()
    }

    /// Sample one challenge extension-field element coefficient by coefficient.
    pub fn challenge_extension<F, EF, Cdc>(&mut self, label: Label) -> TranscriptBound<EF>
    where
        F: PrimeField,
        EF: BasedVectorSpace<F>,
        Cdc: Codec<C, F>,
    {
        self.player.interact(Interaction::new::<EF>(
            Hierarchy::Atomic,
            Kind::Challenge,
            label,
            Length::Scalar,
        ));
        TranscriptBound::wrap(EF::from_basis_coefficients_fn(|_| {
            Cdc::sample(&mut self.challenger)
        }))
    }

    /// Run a proof-of-work step and append the witness to the wire buffer.
    pub fn pow(&mut self, bits: usize)
    where
        C: GrindingChallenger,
        <C as GrindingChallenger>::Witness: PrimeField,
    {
        // Validate: the next pattern step is a proof-of-work step.
        self.player.interact(Interaction::new::<u64>(
            Hierarchy::Atomic,
            Kind::Pow,
            "pow",
            Length::Scalar,
        ));
        // Grind through the challenger's SIMD path.
        let witness = self.challenger.grind(bits);
        // Serialize as canonical big-endian, left-padded so width is constant.
        let target = field_byte_size::<<C as GrindingChallenger>::Witness>();
        let bytes = witness.as_canonical_biguint().to_bytes_be();
        if bytes.len() < target {
            self.narg
                .extend(core::iter::repeat_n(0u8, target - bytes.len()));
        }
        self.narg.extend_from_slice(&bytes);
    }

    /// Open a sub-protocol marker in the recorded pattern.
    pub fn begin_protocol<T: ?Sized>(&mut self, label: Label) {
        self.player.interact(Interaction::new::<T>(
            Hierarchy::Begin,
            Kind::Protocol,
            label,
            Length::None,
        ));
    }

    /// Close a sub-protocol marker in the recorded pattern.
    pub fn end_protocol<T: ?Sized>(&mut self, label: Label) {
        self.player.interact(Interaction::new::<T>(
            Hierarchy::End,
            Kind::Protocol,
            label,
            Length::None,
        ));
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, PrimeField32};

    use crate::fs::TranscriptBound;
    use crate::fs::codecs::BytesToFieldCodec;
    use crate::fs::domain_separator::DomainSeparator;
    use crate::fs::pattern::{Hierarchy, Interaction, InteractionPattern, Kind, Length};
    use crate::fs::shake128::Shake128;
    use crate::fs::state::{ProverState, VerifierState};

    /// Concrete field exercised in this module's tests.
    type F = BabyBear;

    /// Three messages followed by two challenges, each as one fixed-length step.
    fn small_pattern() -> InteractionPattern {
        InteractionPattern::new(alloc::vec![
            Interaction::new::<F>(Hierarchy::Atomic, Kind::Message, "msgs", Length::Fixed(3),),
            Interaction::new::<F>(
                Hierarchy::Atomic,
                Kind::Challenge,
                "challs",
                Length::Fixed(2),
            ),
        ])
        .unwrap()
    }

    #[test]
    fn prover_round_trip_with_shake128_and_bytes_to_field_codec() {
        // Shared pattern + DS; `bind_pattern_hash` separates protocols by shape.
        let pattern = small_pattern();
        let mut ds: DomainSeparator<u8> = DomainSeparator::new(0x01, b"test-protocol", pattern);
        ds.bind_pattern_hash();

        // Three field elements 1, 2, 3 fed as the "msgs" step.
        let messages: Vec<F> = (1u32..=3).map(F::from_u32).collect();

        // Prover walks the pattern and emits a wire payload.
        let prover_ch = Shake128::new(&[0u8; 64]);
        let mut prover = ProverState::<_, u8>::new(prover_ch, &ds);
        prover.add_scalars::<F, BytesToFieldCodec<F>>("msgs", &messages);
        let prover_challenges = prover.challenge_scalars::<F, BytesToFieldCodec<F>>("challs", 2);
        let narg = prover.finalize();

        // Verifier seeded identically replays the pattern over the wire.
        let verifier_ch = Shake128::new(&[0u8; 64]);
        let mut verifier = VerifierState::<_, u8>::new(verifier_ch, &ds, &narg);
        let read_messages = verifier
            .next_scalars::<F, BytesToFieldCodec<F>>("msgs", 3)
            .expect("verifier must accept the prover's NARG");
        let verifier_challenges =
            verifier.challenge_scalars::<F, BytesToFieldCodec<F>>("challs", 2);
        verifier.finalize().expect("NARG must be fully consumed");

        // Property 1: messages round-trip byte-for-byte through the wire.
        let read_inner: Vec<F> = read_messages
            .into_iter()
            .map(TranscriptBound::into_inner)
            .collect();
        assert_eq!(read_inner, messages);
        // Property 2: both sides derive the same challenge stream.
        assert_eq!(prover_challenges, verifier_challenges);
        // Property 3: every challenge lies in the canonical range [0, p).
        for c in &verifier_challenges {
            assert!(c.as_inner().as_canonical_u32() < F::ORDER_U32);
        }
    }

    #[test]
    fn salt_changes_subsequent_challenges() {
        // Pattern: 8-byte salt followed by one challenge scalar.
        let pattern = InteractionPattern::new(alloc::vec![
            Interaction::new::<u8>(Hierarchy::Atomic, Kind::Salt, "salt", Length::Fixed(8),),
            Interaction::new::<F>(Hierarchy::Atomic, Kind::Challenge, "alpha", Length::Scalar,),
        ])
        .unwrap();

        // Helper: run a prover with the given salt and return the bound challenge.
        let drive = |salt: &[u8]| -> TranscriptBound<F> {
            let mut ds: DomainSeparator<u8> = DomainSeparator::new(0, b"zk", pattern.clone());
            ds.bind_pattern_hash();
            let mut p = ProverState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds);
            p.add_salt(salt);
            let c = p.challenge_scalar::<F, BytesToFieldCodec<F>>("alpha");
            let _ = p.finalize();
            c
        };

        // Two salts that differ only in the lowest bit of byte 0.
        let salt_a = [0u8; 8];
        let mut salt_b = salt_a;
        salt_b[0] ^= 1;

        // Property: a single salt-bit flip propagates into a different challenge.
        assert_ne!(drive(&salt_a), drive(&salt_b));
    }

    #[test]
    fn salt_round_trips_through_verifier() {
        // Pattern: 16-byte salt followed by one challenge scalar.
        let pattern = InteractionPattern::new(alloc::vec![
            Interaction::new::<u8>(Hierarchy::Atomic, Kind::Salt, "salt", Length::Fixed(16),),
            Interaction::new::<F>(Hierarchy::Atomic, Kind::Challenge, "alpha", Length::Scalar,),
        ])
        .unwrap();

        let mut ds: DomainSeparator<u8> = DomainSeparator::new(0, b"zk", pattern);
        ds.bind_pattern_hash();

        // Fixed salt fixture so the test is deterministic.
        let salt = [0xa5u8; 16];

        // Prover absorbs the salt then samples the challenge.
        let mut p = ProverState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds);
        p.add_salt(&salt);
        let c_p = p.challenge_scalar::<F, BytesToFieldCodec<F>>("alpha");
        let narg = p.finalize();

        // Verifier seeded identically reads the salt back from the wire and re-samples.
        let mut v = VerifierState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds, &narg);
        let read_salt = v.next_salt(16).expect("verifier reads salt");
        let c_v = v.challenge_scalar::<F, BytesToFieldCodec<F>>("alpha");
        v.finalize().expect("NARG fully consumed");

        // Property 1: salt round-trips byte-for-byte.
        assert_eq!(read_salt, salt);
        // Property 2: same salt absorbed -> same challenge derived.
        assert_eq!(c_p, c_v);
    }

    #[test]
    fn hints_are_carried_in_narg_but_not_absorbed() {
        // Pattern: 4-byte hint followed by one challenge scalar.
        let pattern = InteractionPattern::new(alloc::vec![
            Interaction::new::<u8>(
                Hierarchy::Atomic,
                Kind::Hint,
                "merkle-path",
                Length::Fixed(4),
            ),
            Interaction::new::<F>(Hierarchy::Atomic, Kind::Challenge, "alpha", Length::Scalar,),
        ])
        .unwrap();
        let mut ds: DomainSeparator<u8> = DomainSeparator::new(0, b"hint-test", pattern);
        ds.bind_pattern_hash();

        // Helper: run a prover with the given hint bytes and return (bound challenge, wire).
        let drive_with_hint = |hint: &[u8; 4]| -> (TranscriptBound<F>, Vec<u8>) {
            let mut p = ProverState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds);
            p.add_hint("merkle-path", hint);
            let c = p.challenge_scalar::<F, BytesToFieldCodec<F>>("alpha");
            (c, p.finalize())
        };

        // Two runs that differ only in hint content.
        let (c_a, narg_a) = drive_with_hint(&[1, 2, 3, 4]);
        let (c_b, narg_b) = drive_with_hint(&[9, 9, 9, 9]);

        // Property 1: hint never enters the sponge -> challenges match.
        assert_eq!(c_a, c_b);
        // Property 2: hint bytes are still on the wire -> NARGs differ.
        assert_ne!(narg_a, narg_b);

        // Property 3: verifier reads back the original hint bytes.
        let mut v = VerifierState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds, &narg_a);
        let read_hint = v
            .next_hint("merkle-path", 4)
            .expect("verifier reads hint bytes");
        let _ = v.challenge_scalar::<F, BytesToFieldCodec<F>>("alpha");
        v.finalize().expect("NARG fully consumed");
        assert_eq!(read_hint, &[1, 2, 3, 4]);
    }

    #[test]
    fn bounded_hint_round_trips_with_short_payload() {
        // Invariant: a bounded hint round-trips its actual payload exactly.
        //
        // The hint never enters the sponge, so the verifier's challenge matches the prover's.

        // Fixture state: hint cap of 8 bytes followed by one challenge.
        let pattern = InteractionPattern::new(alloc::vec![
            Interaction::new::<u8>(
                Hierarchy::Atomic,
                Kind::Hint,
                "auth-path",
                Length::Bounded(8),
            ),
            Interaction::new::<F>(Hierarchy::Atomic, Kind::Challenge, "alpha", Length::Scalar,),
        ])
        .unwrap();
        let mut ds: DomainSeparator<u8> = DomainSeparator::new(0, b"bounded-hint", pattern);
        ds.bind_pattern_hash();

        // Mutation (prover): send 3 bytes — strictly less than the cap — then sample a challenge.
        let mut p = ProverState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds);
        p.add_hint_bounded("auth-path", &[0xaa, 0xbb, 0xcc], 8);
        let c_p = p.challenge_scalar::<F, BytesToFieldCodec<F>>("alpha");
        let narg = p.finalize();

        // Mutation (verifier): replay the same step, read the actual byte count, sample a challenge.
        let mut v = VerifierState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds, &narg);
        let read_hint = v.next_hint_bounded("auth-path", 8).expect("legal hint");
        let c_v = v.challenge_scalar::<F, BytesToFieldCodec<F>>("alpha");
        v.finalize().expect("NARG fully consumed");

        // Property 1: payload round-trips byte-for-byte.
        assert_eq!(read_hint, &[0xaa, 0xbb, 0xcc]);

        // Property 2: hint payload is not absorbed, so both sides derive the same challenge.
        assert_eq!(c_p, c_v);
    }

    #[test]
    fn bounded_hint_length_does_not_bind_subsequent_challenges() {
        // Invariant: hint payload and its length are wire-only.
        //
        // Two runs that share the recorded pattern always derive the same challenge.

        // Fixture state: hint cap of 8 bytes followed by one challenge.
        let pattern = InteractionPattern::new(alloc::vec![
            Interaction::new::<u8>(
                Hierarchy::Atomic,
                Kind::Hint,
                "auth-path",
                Length::Bounded(8),
            ),
            Interaction::new::<F>(Hierarchy::Atomic, Kind::Challenge, "alpha", Length::Scalar,),
        ])
        .unwrap();
        let mut ds: DomainSeparator<u8> = DomainSeparator::new(0, b"hint-iso", pattern);
        ds.bind_pattern_hash();

        // Helper: drive a prover with the supplied hint and return the sampled challenge.
        let drive = |hint: &[u8]| -> TranscriptBound<F> {
            let mut p = ProverState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds);
            p.add_hint_bounded("auth-path", hint, 8);
            let c = p.challenge_scalar::<F, BytesToFieldCodec<F>>("alpha");
            let _ = p.finalize();
            c
        };

        // Same challenge across an empty payload and a 3-byte payload.
        //
        // The hint content is not absorbed, so it cannot affect later samples.
        assert_eq!(drive(&[]), drive(&[1, 2, 3]));

        // Same challenge across two payloads of different lengths and contents.
        //
        // The length prefix is also not absorbed for hints.
        assert_eq!(drive(&[1, 2, 3]), drive(&[9; 7]));
    }

    #[test]
    fn bounded_scalars_round_trip() {
        // Invariant: a bounded scalar slice round-trips its values in order.
        //
        // Both sides absorb the same prefix and values, so challenges agree.

        // Fixture state: scalar slice with cap 5 followed by one challenge.
        let pattern = InteractionPattern::new(alloc::vec![
            Interaction::new::<F>(Hierarchy::Atomic, Kind::Message, "msgs", Length::Bounded(5),),
            Interaction::new::<F>(Hierarchy::Atomic, Kind::Challenge, "alpha", Length::Scalar,),
        ])
        .unwrap();
        let mut ds: DomainSeparator<u8> = DomainSeparator::new(0, b"bounded-msgs", pattern);
        ds.bind_pattern_hash();

        // Mutation (prover): send 3 values, strictly below the cap of 5.
        let msgs: alloc::vec::Vec<F> = (1u32..=3).map(F::from_u32).collect();
        let mut p = ProverState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds);
        p.add_scalars_bounded::<F, BytesToFieldCodec<F>>("msgs", &msgs, 5);
        let c_p = p.challenge_scalar::<F, BytesToFieldCodec<F>>("alpha");
        let narg = p.finalize();

        // Mutation (verifier): replay the step and sample the matching challenge.
        let mut v = VerifierState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds, &narg);
        let read = v
            .next_scalars_bounded::<F, BytesToFieldCodec<F>>("msgs", 5)
            .expect("legal scalars");
        let c_v = v.challenge_scalar::<F, BytesToFieldCodec<F>>("alpha");
        v.finalize().expect("NARG fully consumed");

        // Property 1: values round-trip in their original order.
        let read_vals: alloc::vec::Vec<F> =
            read.into_iter().map(TranscriptBound::into_inner).collect();
        assert_eq!(read_vals, msgs);

        // Property 2: both sides absorb the same prefix and values, so challenges agree.
        assert_eq!(c_p, c_v);
    }

    #[test]
    fn bounded_message_length_binds_subsequent_challenges() {
        // Invariant: the absorbed length prefix keeps the sponge transcript prefix-free.
        //
        // Two runs that share value content but differ in count derive different challenges.
        //
        // This matches the soundness condition from CO25 §6.2.

        // Fixture state: scalar slice with cap 5 followed by one challenge.
        let pattern = InteractionPattern::new(alloc::vec![
            Interaction::new::<F>(Hierarchy::Atomic, Kind::Message, "msgs", Length::Bounded(5),),
            Interaction::new::<F>(Hierarchy::Atomic, Kind::Challenge, "alpha", Length::Scalar,),
        ])
        .unwrap();
        let mut ds: DomainSeparator<u8> = DomainSeparator::new(0, b"len-bind", pattern);
        ds.bind_pattern_hash();

        // Helper: drive a prover with `n` zero scalars and return the sampled challenge.
        let drive = |n: usize| -> TranscriptBound<F> {
            let zeros: alloc::vec::Vec<F> = (0..n).map(|_| F::from_u32(0)).collect();
            let mut p = ProverState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds);
            p.add_scalars_bounded::<F, BytesToFieldCodec<F>>("msgs", &zeros, 5);
            let c = p.challenge_scalar::<F, BytesToFieldCodec<F>>("alpha");
            let _ = p.finalize();
            c
        };

        // Empty slice versus one zero scalar.
        //
        // Value content is identical past the prefix, but the prefix itself differs.
        assert_ne!(drive(0), drive(1));

        // One zero scalar versus two.
        //
        // The longer run is never a prefix of the shorter one on the sponge.
        assert_ne!(drive(1), drive(2));
    }

    #[test]
    fn pattern_hash_binds_bounded_max() {
        // Invariant: the bound is part of the pattern fingerprint.
        //
        // Two protocols differing only in capacity must seed with different bytes.

        // Fixture state: two patterns identical except for the cap (7 vs 8).
        let pat_a = InteractionPattern::new(alloc::vec![Interaction::new::<u8>(
            Hierarchy::Atomic,
            Kind::Hint,
            "auth",
            Length::Bounded(7),
        )])
        .unwrap();
        let pat_b = InteractionPattern::new(alloc::vec![Interaction::new::<u8>(
            Hierarchy::Atomic,
            Kind::Hint,
            "auth",
            Length::Bounded(8),
        )])
        .unwrap();

        // The fingerprint distinguishes the two capacities.
        assert_ne!(pat_a.pattern_hash(), pat_b.pattern_hash());
    }

    #[test]
    #[should_panic(expected = "exceeds declared maximum")]
    fn prover_panics_on_oversize_bounded_hint() {
        // Invariant: writing more than the recorded cap is a caller bug.
        //
        // The prover panics loudly rather than emitting a malformed wire frame.

        // Fixture state: a hint with a cap of 4 bytes.
        let pattern = InteractionPattern::new(alloc::vec![Interaction::new::<u8>(
            Hierarchy::Atomic,
            Kind::Hint,
            "auth",
            Length::Bounded(4),
        )])
        .unwrap();
        let mut ds: DomainSeparator<u8> = DomainSeparator::new(0, b"oversize", pattern);
        ds.bind_pattern_hash();

        // Mutation: feed 5 bytes into a cap of 4.
        let mut p = ProverState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds);
        p.add_hint_bounded("auth", &[0u8; 5], 4);

        // Drain the player so dropping the prover during the panic does not double-panic.
        let _ = p.finalize();
    }

    #[test]
    fn distinct_protocol_ids_yield_distinct_challenges() {
        // Three small messages reused across both runs.
        let messages: Vec<F> = alloc::vec![F::from_u32(7), F::from_u32(11), F::from_u32(13)];

        // Helper: drive a prover under the given protocol name and return its bound challenges.
        let drive = |name: &[u8]| -> Vec<TranscriptBound<F>> {
            let pattern = small_pattern();
            let mut ds: DomainSeparator<u8> = DomainSeparator::new(0x01, name, pattern);
            ds.bind_pattern_hash();
            let mut p = ProverState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds);
            p.add_scalars::<F, BytesToFieldCodec<F>>("msgs", &messages);
            let chs = p.challenge_scalars::<F, BytesToFieldCodec<F>>("challs", 2);
            let _ = p.finalize();
            chs
        };

        // Property: different protocol names -> different seeds -> different challenges.
        assert_ne!(drive(b"protocol-a"), drive(b"protocol-b"));
    }
}
