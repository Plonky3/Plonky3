//! Prover-side driver.

use alloc::vec::Vec;

use p3_field::{BasedVectorSpace, PrimeField};

use crate::fs::codecs::Codec;
use crate::fs::codecs::decode_field::{encode_field_be, field_byte_size};
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
    pub fn add_scalar<F, Cdc>(&mut self, label: Label, value: &F)
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
    }

    /// Absorb a known-length list of scalars under a single pattern step.
    ///
    /// No length prefix is written; the recorded pattern is the source of truth.
    pub fn add_scalars<F, Cdc>(&mut self, label: Label, values: &[F])
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
        // Absorb each value and append its canonical big-endian encoding to the wire.
        for v in values {
            Cdc::observe(&mut self.challenger, v);
            let bytes = encode_field_be::<F>(v);
            self.narg.extend_from_slice(&bytes);
        }
    }

    /// Absorb one extension-field element coefficient by coefficient.
    pub fn add_extension<F, EF, Cdc>(&mut self, label: Label, value: &EF)
    where
        F: PrimeField,
        EF: BasedVectorSpace<F>,
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

    /// Sample one challenge scalar of type `F` via codec `Cdc`.
    pub fn challenge_scalar<F, Cdc>(&mut self, label: Label) -> F
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
        Cdc::sample(&mut self.challenger)
    }

    /// Sample `n` challenge scalars of type `F` under a single pattern step.
    pub fn challenge_scalars<F, Cdc>(&mut self, label: Label, n: usize) -> Vec<F>
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
        (0..n).map(|_| Cdc::sample(&mut self.challenger)).collect()
    }

    /// Sample one challenge extension-field element coefficient by coefficient.
    pub fn challenge_extension<F, EF, Cdc>(&mut self, label: Label) -> EF
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
        EF::from_basis_coefficients_fn(|_| Cdc::sample(&mut self.challenger))
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
    use p3_field::PrimeField32;
    use p3_field::PrimeCharacteristicRing;

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
        assert_eq!(read_messages, messages);
        // Property 2: both sides derive the same challenge stream.
        assert_eq!(prover_challenges, verifier_challenges);
        // Property 3: every challenge lies in the canonical range [0, p).
        for c in &verifier_challenges {
            assert!(c.as_canonical_u32() < F::ORDER_U32);
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

        // Helper: run a prover with the given salt and return the challenge.
        let drive = |salt: &[u8]| -> F {
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

        // Helper: run a prover with the given hint bytes and return (challenge, wire).
        let drive_with_hint = |hint: &[u8; 4]| -> (F, Vec<u8>) {
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
    fn distinct_protocol_ids_yield_distinct_challenges() {
        // Three small messages reused across both runs.
        let messages: Vec<F> = alloc::vec![F::from_u32(7), F::from_u32(11), F::from_u32(13)];

        // Helper: drive a prover under the given protocol name and return its challenges.
        let drive = |name: &[u8]| -> Vec<F> {
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
