//! Prover-side driver.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::{BasedVectorSpace, Field, PrimeField64};

use crate::fs::bound::TranscriptBound;
use crate::fs::codecs::{
    Codec, ExtensionFieldCodec, bound_byte_width, encode_field_be, encode_len_be,
};
use crate::fs::domain_separator::DomainSeparator;
use crate::fs::pattern::{Hierarchy, Interaction, Kind, Label, Length, Pattern, PatternPlayer};
use crate::fs::state::assert_challenge_security;
use crate::fs::unit::Unit;
use crate::{CanObserve, CanSampleBits, GrindingChallenger};

/// Drives a prover-side transcript in lockstep with a recorded pattern.
///
/// Each absorb or sample method advances the player by one step.
/// The serialised wire bytes are yielded only at finalisation.
///
/// The `U` parameter names the sponge alphabet:
/// `u8` for a byte sponge such as `HashChallenger<u8, _, _>`,
/// [`crate::fs::FieldUnit<F>`] for a native field sponge such as
/// `DuplexChallenger` or `SerializingChallenger32/64`.
///
/// The wire format is bytes either way, so a proof does not depend on which
/// sponge produced it.
///
/// # Panics
///
/// On drop without [`Self::finalize`], to surface a transcript that was
/// abandoned halfway through.
pub struct ProverState<C, U: Unit = u8> {
    /// Underlying sponge that absorbs prover messages and yields challenges.
    challenger: C,
    /// Pattern player that validates each call against the recorded sequence.
    player: PatternPlayer,
    /// Accumulated wire bytes returned at finalisation.
    narg: Vec<u8>,
    /// Type-level marker for the sponge alphabet.
    _u: PhantomData<U>,
}

impl<C, U: Unit> Drop for ProverState<C, U> {
    fn drop(&mut self) {
        // Loud failure surfaces a transcript abandoned before finalisation.
        //
        // Every path that panics or bails marks the player aborted first,
        // so this check never fires during cleanup of another failure.
        if !self.player.is_finalized() {
            let remaining = self.player.remaining();
            // Release the player's own drop check so this panic stays single.
            self.player.abort();
            panic!(
                "Dropped unfinalized ProverState: {remaining} pattern step(s) were never played."
            );
        }
    }
}

impl<C, U: Unit> ProverState<C, U> {
    /// Build a driver and seed the challenger from the domain separator.
    pub fn new(mut challenger: C, ds: &DomainSeparator<U>) -> Self
    where
        C: CanObserve<U::Item>,
    {
        // Seed first so two distinct (protocol, instance) pairs land on distinct sponge states.
        ds.seed(&mut challenger);
        let player = PatternPlayer::new(ds.pattern().clone());
        Self {
            challenger,
            player,
            narg: Vec::new(),
            _u: PhantomData,
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
    pub fn finalize(mut self) -> Vec<u8> {
        // Strict pattern-fully-consumed check.
        //
        // `finalize` marks the player before asserting, so the `Drop` below
        // stays quiet whichever way this call goes.
        self.player.finalize();
        core::mem::take(&mut self.narg)
    }

    /// Report a caller bug after releasing the drop-time check.
    ///
    /// Marking the player first keeps the panic single.
    /// `Drop` would otherwise fire its own assertion while this one unwinds.
    fn fail(&mut self, message: core::fmt::Arguments<'_>) -> ! {
        self.player.abort();
        panic!("{message}")
    }

    /// Absorb a salt step and record its bytes in the wire buffer.
    pub fn add_salt(&mut self, label: Label, salt_bytes: &[u8])
    where
        C: CanObserve<U::Item>,
    {
        // Validate: the next pattern step is a salt of the given length.
        self.player.interact(Interaction::bytes(
            Hierarchy::Atomic,
            Kind::Salt,
            label,
            Length::Fixed(salt_bytes.len()),
        ));
        // Absorb so future samples depend on the salt.
        U::observe_bytes(&mut self.challenger, salt_bytes);
        // Mirror the same bytes onto the wire so the verifier can re-absorb them.
        self.narg.extend_from_slice(salt_bytes);
    }

    /// Absorb one shared public scalar through the supplied codec.
    ///
    /// Public values are known to both parties before the run.
    /// They are bound into the sponge but never written to the wire.
    /// The verifier re-absorbs the same value from its own inputs.
    pub fn add_public_scalar<F, Cdc>(&mut self, label: Label, value: &F) -> TranscriptBound<F>
    where
        F: PrimeField64,
        Cdc: Codec<C, F>,
    {
        // Validate: the next pattern step is a public scalar of type `F`.
        self.player.interact(Interaction::algebra::<F, F>(
            Hierarchy::Atomic,
            Kind::Public,
            label,
            Length::Scalar,
        ));
        // Bind into the sponge only; public data is not part of the wire.
        Cdc::observe(&mut self.challenger, value);
        TranscriptBound::wrap(*value)
    }

    /// Absorb a fixed-length list of shared public scalars under a single step.
    ///
    /// As with [`Self::add_public_scalar`], nothing is written to the wire.
    pub fn add_public_scalars<F, Cdc>(
        &mut self,
        label: Label,
        values: &[F],
    ) -> Vec<TranscriptBound<F>>
    where
        F: PrimeField64,
        Cdc: Codec<C, F>,
    {
        // Validate: the next pattern step is a fixed-length list of public scalars.
        self.player.interact(Interaction::algebra::<F, F>(
            Hierarchy::Atomic,
            Kind::Public,
            label,
            Length::Fixed(values.len()),
        ));
        // Bind each value into the sponge; none reaches the wire.
        values
            .iter()
            .map(|v| {
                Cdc::observe(&mut self.challenger, v);
                TranscriptBound::wrap(*v)
            })
            .collect()
    }

    /// Absorb one prover scalar through the supplied codec.
    pub fn add_scalar<F, Cdc>(&mut self, label: Label, value: &F) -> TranscriptBound<F>
    where
        F: PrimeField64,
        Cdc: Codec<C, F>,
    {
        // Validate: the next pattern step is a scalar message of type `F`.
        self.player.interact(Interaction::algebra::<F, F>(
            Hierarchy::Atomic,
            Kind::Message,
            label,
            Length::Scalar,
        ));
        // Sponge path and wire path both belong to the codec, so they cannot drift.
        Cdc::observe(&mut self.challenger, value);
        Cdc::encode(value, &mut self.narg);
        TranscriptBound::wrap(*value)
    }

    /// Absorb a known-length list of scalars under a single pattern step.
    ///
    /// No length prefix is written; the recorded pattern is the source of truth.
    pub fn add_scalars<F, Cdc>(&mut self, label: Label, values: &[F]) -> Vec<TranscriptBound<F>>
    where
        F: PrimeField64,
        Cdc: Codec<C, F>,
    {
        // Validate: the next pattern step is a fixed-length list of scalars.
        self.player.interact(Interaction::algebra::<F, F>(
            Hierarchy::Atomic,
            Kind::Message,
            label,
            Length::Fixed(values.len()),
        ));
        // Absorb each value, mirror its canonical encoding to the wire, and bind it.
        values
            .iter()
            .map(|v| {
                Cdc::observe(&mut self.challenger, v);
                Cdc::encode(v, &mut self.narg);
                TranscriptBound::wrap(*v)
            })
            .collect()
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
    /// The length is absorbed first so the sponge transcript stays prefix-free:
    /// no shorter run of this step is a prefix of a longer one.
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
        F: PrimeField64,
        C: CanObserve<U::Item>,
        Cdc: Codec<C, F>,
    {
        // Caller bug: writing more than the cap would diverge from the recorded pattern.
        if values.len() > max {
            self.fail(format_args!(
                "message length {} exceeds declared maximum {max}",
                values.len(),
            ));
        }
        // Validate against the recorded pattern step so shape divergence panics here.
        self.player.interact(Interaction::algebra::<F, F>(
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
        U::observe_bytes(&mut self.challenger, &len_bytes[..width]);
        // Mirror the same prefix onto the wire so the verifier sees the count.
        self.narg.extend_from_slice(&len_bytes[..width]);
        // Absorb each value through the codec and write its canonical encoding.
        values
            .iter()
            .map(|v| {
                Cdc::observe(&mut self.challenger, v);
                Cdc::encode(v, &mut self.narg);
                TranscriptBound::wrap(*v)
            })
            .collect()
    }

    /// Absorb one extension-field element coefficient by coefficient.
    ///
    /// Routed through [`ExtensionFieldCodec`].
    /// The coefficient layout therefore has one definition, shared with the verifier.
    pub fn add_extension<F, EF, Cdc>(&mut self, label: Label, value: &EF) -> TranscriptBound<EF>
    where
        F: PrimeField64,
        EF: Field + BasedVectorSpace<F>,
        Cdc: Codec<C, F>,
    {
        // Validate: the next pattern step is a scalar message of extension type.
        self.player.interact(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Message,
            label,
            Length::Scalar,
        ));
        ExtensionFieldCodec::<F, EF, Cdc>::observe(&mut self.challenger, value);
        ExtensionFieldCodec::<F, EF, Cdc>::encode(value, &mut self.narg);
        TranscriptBound::wrap(*value)
    }

    /// Absorb one extension-field message the caller carries itself.
    ///
    /// # Overview
    ///
    /// An adding method writes its value into the proof bytes this driver builds.
    /// An observing method does not.
    /// Both bind the value into the sponge the same way.
    ///
    /// # When to use this
    ///
    /// A protocol whose proof is its own type already carries the value.
    /// Writing it again here would ship it twice.
    ///
    /// # Wire accounting
    ///
    /// Nothing is written, so nothing is read on the other side.
    ///
    /// A prover that writes where its verifier observes leaves unread bytes.
    /// Finalisation rejects those.
    /// The reverse runs the verifier past the end, which fails too.
    ///
    /// So the two sides cannot silently disagree about who carries a value.
    pub fn observe_extension<F, EF, Cdc>(&mut self, label: Label, value: &EF) -> TranscriptBound<EF>
    where
        F: PrimeField64,
        EF: Field + BasedVectorSpace<F>,
        Cdc: Codec<C, F>,
    {
        // Validate: the next pattern step is a scalar message of extension type.
        self.player.interact(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Message,
            label,
            Length::Scalar,
        ));
        // Sponge only: the caller's own proof type carries the value.
        ExtensionFieldCodec::<F, EF, Cdc>::observe(&mut self.challenger, value);
        TranscriptBound::wrap(*value)
    }

    /// Absorb a fixed-length list of extension-field messages the caller carries itself.
    ///
    /// The whole list is one step.
    /// Its length comes from the recorded shape, never from the data.
    pub fn observe_extensions<F, EF, Cdc>(
        &mut self,
        label: Label,
        values: &[EF],
    ) -> Vec<TranscriptBound<EF>>
    where
        F: PrimeField64,
        EF: Field + BasedVectorSpace<F>,
        Cdc: Codec<C, F>,
    {
        // Validate: the next pattern step is a fixed-length list of extension messages.
        self.player.interact(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Message,
            label,
            Length::Fixed(values.len()),
        ));
        // Absorb in order and hand back one binding witness per value.
        values
            .iter()
            .map(|v| {
                ExtensionFieldCodec::<F, EF, Cdc>::observe(&mut self.challenger, v);
                TranscriptBound::wrap(*v)
            })
            .collect()
    }

    /// Absorb a value the challenger knows how to encode, carried by the caller.
    ///
    /// # When to use this
    ///
    /// A commitment is the usual case.
    /// Its width lives in the commitment scheme's configuration, not here.
    ///
    /// # What is bound
    ///
    /// The value's content, through the challenger's own encoding.
    /// Its width is not, so two runs differing only in that share a fingerprint.
    ///
    /// Bind such widths through the instance label where a protocol can see them.
    pub fn observe_opaque<T>(&mut self, label: Label, value: T) -> TranscriptBound<T>
    where
        T: Clone,
        C: CanObserve<T>,
    {
        // Validate: the next pattern step is an opaque message.
        self.player.interact(Interaction::opaque(
            Hierarchy::Atomic,
            Kind::Message,
            label,
            Length::Scalar,
        ));
        // The challenger owns the encoding, so hand the value over whole.
        self.challenger.observe(value.clone());
        TranscriptBound::wrap(value)
    }

    /// Sample `count` challenges of `width` uniform bits under one step.
    ///
    /// The width travels in the step.
    /// A verifier drawing narrower indices fails the shape check.
    ///
    /// Nothing is absorbed between draws, so one step describes them all.
    pub fn challenge_bits(
        &mut self,
        label: Label,
        width: usize,
        count: usize,
    ) -> Vec<TranscriptBound<usize>>
    where
        C: CanSampleBits<usize>,
    {
        self.player.interact(Interaction::bits(
            Hierarchy::Atomic,
            Kind::Challenge,
            label,
            width,
            Length::Fixed(count),
        ));
        (0..count)
            .map(|_| TranscriptBound::wrap(self.challenger.sample_bits(width)))
            .collect()
    }

    /// Absorb a fixed-length byte string as a prover message.
    ///
    /// The dominant prover message in this repository is a digest.
    /// A Merkle root or cap is a byte string, not a field element.
    ///
    /// Unlike a hint, these bytes enter the sponge.
    /// Later challenges therefore depend on the commitment, as Fiat-Shamir requires.
    pub fn add_bytes(&mut self, label: Label, bytes: &[u8])
    where
        C: CanObserve<U::Item>,
    {
        // Validate: the next pattern step is a fixed-length byte message.
        self.player.interact(Interaction::bytes(
            Hierarchy::Atomic,
            Kind::Message,
            label,
            Length::Fixed(bytes.len()),
        ));
        // Absorbed and mirrored: the verifier re-reads the same bytes off the wire.
        U::observe_bytes(&mut self.challenger, bytes);
        self.narg.extend_from_slice(bytes);
    }

    /// Absorb a variable-length byte message of at most `max` bytes.
    ///
    /// # Wire format
    ///
    /// ```text
    ///     [len in W bytes, big-endian][payload bytes]
    /// ```
    ///
    /// Both the prefix and the payload are absorbed, the prefix first, which
    /// keeps the sponge transcript prefix-free (CO25 §6.2).
    ///
    /// # Panics
    ///
    /// When the supplied byte count exceeds `max`.
    pub fn add_bytes_bounded(&mut self, label: Label, bytes: &[u8], max: usize)
    where
        C: CanObserve<U::Item>,
    {
        // Caller bug: writing more than the cap would diverge from the recorded pattern.
        if bytes.len() > max {
            self.fail(format_args!(
                "message length {} exceeds declared maximum {max}",
                bytes.len(),
            ));
        }
        self.player.interact(Interaction::bytes(
            Hierarchy::Atomic,
            Kind::Message,
            label,
            Length::Bounded(max),
        ));
        // Prefix width is deterministic on both sides from the recorded bound.
        let width = bound_byte_width(max);
        let len_bytes = encode_len_be(bytes.len(), width);
        // Length first, then payload, on both the sponge and the wire.
        U::observe_bytes(&mut self.challenger, &len_bytes[..width]);
        U::observe_bytes(&mut self.challenger, bytes);
        self.narg.extend_from_slice(&len_bytes[..width]);
        self.narg.extend_from_slice(bytes);
    }

    /// Append a hint to the wire buffer without absorbing it into the sponge.
    ///
    /// Hint bytes are part of the wire format but never enter the sponge,
    /// so they cannot influence sampled challenges.
    pub fn add_hint(&mut self, label: Label, bytes: &[u8]) {
        self.player.interact(Interaction::bytes(
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
    /// Neither the length prefix nor the payload enter the sponge, so this step
    /// cannot influence any later challenge.
    ///
    /// # Panics
    ///
    /// When the supplied byte count exceeds `max`.
    pub fn add_hint_bounded(&mut self, label: Label, bytes: &[u8], max: usize) {
        // Caller bug: writing more than the cap would silently truncate on the verifier side.
        if bytes.len() > max {
            self.fail(format_args!(
                "hint length {} exceeds declared maximum {max}",
                bytes.len(),
            ));
        }
        // Validate against the recorded pattern step so any shape divergence panics here.
        self.player.interact(Interaction::bytes(
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
        F: PrimeField64,
        Cdc: Codec<C, F>,
    {
        assert_challenge_security::<C, F, Cdc>();
        self.player.interact(Interaction::algebra::<F, F>(
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
        F: PrimeField64,
        Cdc: Codec<C, F>,
    {
        assert_challenge_security::<C, F, Cdc>();
        self.player.interact(Interaction::algebra::<F, F>(
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
        F: PrimeField64,
        EF: Field + BasedVectorSpace<F>,
        Cdc: Codec<C, F>,
    {
        assert_challenge_security::<C, F, Cdc>();
        self.player.interact(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Challenge,
            label,
            Length::Scalar,
        ));
        TranscriptBound::wrap(ExtensionFieldCodec::<F, EF, Cdc>::sample(
            &mut self.challenger,
        ))
    }

    /// Run a proof-of-work step and append the witness to the wire buffer.
    ///
    /// The difficulty is recorded as `Length::Fixed(bits)`.
    /// It is therefore part of the pattern, and of the seed derived from it.
    ///
    /// A verifier configured with a different `bits` hits a pattern mismatch.
    /// It cannot silently accept a cheaper grind.
    pub fn pow(&mut self, label: Label, bits: usize)
    where
        C: GrindingChallenger,
        <C as GrindingChallenger>::Witness: PrimeField64,
    {
        // Validate: the next pattern step is a proof-of-work step of this difficulty.
        self.player
            .interact(Interaction::algebra::<C::Witness, C::Witness>(
                Hierarchy::Atomic,
                Kind::Pow,
                label,
                Length::Fixed(bits),
            ));
        // Grind through the challenger's SIMD path.
        let witness = self.challenger.grind(bits);
        // Serialize as canonical big-endian, left-padded so width is constant.
        encode_field_be(&witness, &mut self.narg);
    }

    /// Run a proof-of-work step and hand the witness back to the caller.
    ///
    /// The difficulty is recorded exactly as for a wire-carried step.
    /// A verifier expecting a cheaper grind therefore fails the shape check.
    ///
    /// # Returns
    ///
    /// The witness the search found, for the caller to store in its own proof.
    pub fn observe_pow(&mut self, label: Label, bits: usize) -> C::Witness
    where
        C: GrindingChallenger,
        <C as GrindingChallenger>::Witness: PrimeField64,
    {
        // Validate: the next pattern step is a proof-of-work step of this difficulty.
        self.player
            .interact(Interaction::algebra::<C::Witness, C::Witness>(
                Hierarchy::Atomic,
                Kind::Pow,
                label,
                Length::Fixed(bits),
            ));
        // Grinding absorbs the winning witness, which is what advances the sponge.
        self.challenger.grind(bits)
    }

    /// Open a sub-protocol marker of the given kind in the recorded pattern.
    pub fn begin<T: ?Sized>(&mut self, label: Label, kind: Kind) {
        self.player
            .interact(Interaction::marker::<T>(Hierarchy::Begin, kind, label));
    }

    /// Close a sub-protocol marker of the given kind in the recorded pattern.
    pub fn end<T: ?Sized>(&mut self, label: Label, kind: Kind) {
        self.player
            .interact(Interaction::marker::<T>(Hierarchy::End, kind, label));
    }

    /// Open a mixed container that accepts nested steps of any kind.
    pub fn begin_protocol<T: ?Sized>(&mut self, label: Label) {
        self.begin::<T>(label, Kind::Protocol);
    }

    /// Close a mixed container.
    pub fn end_protocol<T: ?Sized>(&mut self, label: Label) {
        self.end::<T>(label, Kind::Protocol);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField32};
    use p3_keccak::Keccak256Hash;
    use p3_symmetric::{CryptographicPermutation, Permutation};

    use crate::fs::codecs::{BytesToFieldCodec, FieldToFieldCodec};
    use crate::fs::domain_separator::DomainSeparator;
    use crate::fs::pattern::{
        Hierarchy, Interaction, InteractionPattern, Kind, Length, Pattern, PatternState,
    };
    use crate::fs::state::{ProverState, VerifierState};
    use crate::fs::unit::FieldUnit;
    use crate::fs::{TranscriptBound, TranscriptError};
    use crate::{DuplexChallenger, HashChallenger, SerializingChallenger32};

    /// Concrete field exercised in this module's tests.
    type F = BabyBear;
    /// Degree-4 binomial extension over `F`.
    type EF4 = BinomialExtensionField<F, 4>;
    /// Byte codec used with a byte sponge.
    type ByteCodec = BytesToFieldCodec<F>;
    /// Field codec used with a native field sponge.
    type NativeCodec = FieldToFieldCodec<F>;

    /// Keccak-chained byte sponge backing the byte-alphabet transcripts.
    ///
    /// Each squeeze advances the chaining state.
    /// Consecutive samples therefore differ.
    fn byte_sponge() -> HashChallenger<u8, Keccak256Hash, 32> {
        HashChallenger::new(Vec::new(), Keccak256Hash)
    }

    /// Production field sponge: the Keccak byte sponge lifted to `F` elements.
    ///
    /// Implements `CanObserve<F>`, `CanSample<F>` and `GrindingChallenger`, so
    /// it drives the field alphabet, the native codec, and proof-of-work.
    fn field_sponge() -> SerializingChallenger32<F, HashChallenger<u8, Keccak256Hash, 32>> {
        SerializingChallenger32::from_hasher(Vec::new(), Keccak256Hash)
    }

    /// Sponge width used by the duplex challenger under test.
    const WIDTH: usize = 8;
    /// Rate used by the duplex challenger under test.
    const RATE: usize = 4;

    /// Cheap mixing permutation: enough to check the duplex wiring end to end.
    #[derive(Clone, Debug)]
    struct MixPermutation;

    impl<A: PrimeCharacteristicRing + Copy> Permutation<[A; WIDTH]> for MixPermutation {
        fn permute_mut(&self, state: &mut [A; WIDTH]) {
            // Rotate so every slot feeds a different slot on the next round.
            state.rotate_left(1);
            // Multiply-add by a slot-dependent constant so the map is not linear in position.
            for (i, x) in state.iter_mut().enumerate() {
                *x = *x * A::from_u64(i as u64 + 2) + A::ONE;
            }
        }
    }

    impl<A: PrimeCharacteristicRing + Copy> CryptographicPermutation<[A; WIDTH]> for MixPermutation {}

    /// Native duplex sponge over `F`.
    fn duplex_sponge() -> DuplexChallenger<F, MixPermutation, WIDTH, RATE> {
        DuplexChallenger::new(MixPermutation)
    }

    /// One atomic step over `F` with the given role and length.
    fn scalar_step(kind: Kind, label: &'static str, length: Length) -> Interaction {
        Interaction::algebra::<F, F>(Hierarchy::Atomic, kind, label, length)
    }

    /// One atomic byte step with the given role and length.
    fn byte_step(kind: Kind, label: &'static str, length: Length) -> Interaction {
        Interaction::bytes(Hierarchy::Atomic, kind, label, length)
    }

    /// Three messages followed by two challenges, each as one fixed-length step.
    fn small_pattern() -> InteractionPattern {
        InteractionPattern::new(vec![
            scalar_step(Kind::Message, "msgs", Length::Fixed(3)),
            scalar_step(Kind::Challenge, "challs", Length::Fixed(2)),
        ])
        .unwrap()
    }

    #[test]
    fn prover_round_trip_with_keccak_sponge_and_bytes_to_field_codec() {
        // Shared pattern + DS; the separator binds the pattern shape automatically.
        let pattern = small_pattern();
        let ds: DomainSeparator<u8> = DomainSeparator::new(0x01, b"test-protocol", pattern);

        // Three field elements 1, 2, 3 fed as the "msgs" step.
        let messages: Vec<F> = (1u32..=3).map(F::from_u32).collect();

        // Prover walks the pattern and emits a wire payload.
        let mut prover = ProverState::<_, u8>::new(byte_sponge(), &ds);
        prover.add_scalars::<F, ByteCodec>("msgs", &messages);
        let prover_challenges = prover.challenge_scalars::<F, ByteCodec>("challs", 2);
        let narg = prover.finalize();

        // Verifier seeded identically replays the pattern over the wire.
        let mut verifier = VerifierState::<_, u8>::new(byte_sponge(), &ds, &narg);
        let read_messages = verifier
            .next_scalars::<F, ByteCodec>("msgs", 3)
            .expect("verifier must accept the prover's NARG");
        let verifier_challenges = verifier.challenge_scalars::<F, ByteCodec>("challs", 2);
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
    fn round_trip_over_a_native_field_sponge() {
        // Invariant: the field alphabet drives an existing production challenger.
        //
        // `SerializingChallenger32` observes and samples `F`, never bytes, so it
        // only fits through `FieldUnit<F>` seeding and the native codec.
        let ds: DomainSeparator<FieldUnit<F>> =
            DomainSeparator::new(0x02, b"native-field", small_pattern());

        let messages: Vec<F> = (7u32..=9).map(F::from_u32).collect();

        let mut prover = ProverState::<_, FieldUnit<F>>::new(field_sponge(), &ds);
        prover.add_scalars::<F, NativeCodec>("msgs", &messages);
        let prover_challenges = prover.challenge_scalars::<F, NativeCodec>("challs", 2);
        let narg = prover.finalize();

        let mut verifier = VerifierState::<_, FieldUnit<F>>::new(field_sponge(), &ds, &narg);
        let read = verifier
            .next_scalars::<F, NativeCodec>("msgs", 3)
            .expect("verifier must accept the prover's NARG");
        let verifier_challenges = verifier.challenge_scalars::<F, NativeCodec>("challs", 2);
        verifier.finalize().expect("NARG must be fully consumed");

        // Property 1: the wire is alphabet-independent, so messages still round-trip.
        let read_inner: Vec<F> = read.into_iter().map(TranscriptBound::into_inner).collect();
        assert_eq!(read_inner, messages);
        // Property 2: both sides derive the same challenge stream.
        assert_eq!(prover_challenges, verifier_challenges);
        // Property 3: the wire holds three canonical 4-byte encodings and nothing else.
        assert_eq!(narg.len(), 3 * 4);
    }

    #[test]
    fn round_trip_over_a_duplex_sponge() {
        // Invariant: a native duplex sponge drives the layer through the same path.
        //
        // `DuplexChallenger` has no byte interface at all, so this only works
        // because seeding goes through the alphabet.
        let ds: DomainSeparator<FieldUnit<F>> =
            DomainSeparator::new(0x03, b"duplex", small_pattern());

        let messages: Vec<F> = (4u32..=6).map(F::from_u32).collect();

        let mut prover = ProverState::<_, FieldUnit<F>>::new(duplex_sponge(), &ds);
        prover.add_scalars::<F, NativeCodec>("msgs", &messages);
        let prover_challenges = prover.challenge_scalars::<F, NativeCodec>("challs", 2);
        let narg = prover.finalize();

        let mut verifier = VerifierState::<_, FieldUnit<F>>::new(duplex_sponge(), &ds, &narg);
        let read = verifier
            .next_scalars::<F, NativeCodec>("msgs", 3)
            .expect("verifier must accept the prover's NARG");
        let verifier_challenges = verifier.challenge_scalars::<F, NativeCodec>("challs", 2);
        verifier.finalize().expect("NARG must be fully consumed");

        let read_inner: Vec<F> = read.into_iter().map(TranscriptBound::into_inner).collect();
        assert_eq!(read_inner, messages);
        assert_eq!(prover_challenges, verifier_challenges);
    }

    #[test]
    fn proof_of_work_round_trips_and_binds_its_difficulty() {
        // Invariant: the prover grinds, the verifier checks, and both agree on `bits`.

        // Fixture state: an 8-bit proof of work followed by one challenge.
        let bits = 8;
        let pattern = InteractionPattern::new(vec![
            Interaction::algebra::<F, F>(
                Hierarchy::Atomic,
                Kind::Pow,
                "grind",
                Length::Fixed(bits),
            ),
            scalar_step(Kind::Challenge, "alpha", Length::Scalar),
        ])
        .unwrap();
        let ds: DomainSeparator<FieldUnit<F>> = DomainSeparator::new(0, b"pow", pattern);

        // Mutation (prover): grind, then sample past the absorbed witness.
        let mut p = ProverState::<_, FieldUnit<F>>::new(field_sponge(), &ds);
        p.pow("grind", bits);
        let c_p = p.challenge_scalar::<F, NativeCodec>("alpha");
        let narg = p.finalize();

        // The witness occupies one canonical field encoding on the wire.
        assert_eq!(narg.len(), 4);

        // Mutation (verifier): re-check the witness, then sample the matching challenge.
        let mut v = VerifierState::<_, FieldUnit<F>>::new(field_sponge(), &ds, &narg);
        v.check_pow("grind", bits).expect("witness must verify");
        let c_v = v.challenge_scalar::<F, NativeCodec>("alpha");
        v.finalize().expect("NARG fully consumed");

        // Property: absorbing the same witness leaves both sponges in the same state.
        assert_eq!(c_p, c_v);
    }

    #[test]
    fn proof_of_work_rejects_a_tampered_witness() {
        // Invariant: a witness that misses the target is malformed input, not a panic.
        let bits = 8;
        let pattern = InteractionPattern::new(vec![Interaction::algebra::<F, F>(
            Hierarchy::Atomic,
            Kind::Pow,
            "grind",
            Length::Fixed(bits),
        )])
        .unwrap();
        let ds: DomainSeparator<FieldUnit<F>> = DomainSeparator::new(0, b"pow-bad", pattern);

        // Mutation: flip the low byte of the witness the prover found.
        let mut p = ProverState::<_, FieldUnit<F>>::new(field_sponge(), &ds);
        p.pow("grind", bits);
        let mut narg = p.finalize();
        narg[3] ^= 0x01;

        let mut v = VerifierState::<_, FieldUnit<F>>::new(field_sponge(), &ds, &narg);
        assert_eq!(
            v.check_pow("grind", bits),
            Err(TranscriptError::BadProofShape {
                reason: "pow witness does not produce enough zero bits",
            })
        );
    }

    #[test]
    #[should_panic(expected = "Received interaction")]
    fn proof_of_work_difficulty_mismatch_is_caught_by_the_pattern() {
        // Invariant: `bits` lives in the recorded step, so it cannot drift.
        //
        // A verifier configured with a cheaper grind hits a pattern mismatch
        // rather than silently accepting the proof.
        let pattern = InteractionPattern::new(vec![Interaction::algebra::<F, F>(
            Hierarchy::Atomic,
            Kind::Pow,
            "grind",
            Length::Fixed(8),
        )])
        .unwrap();
        let ds: DomainSeparator<FieldUnit<F>> = DomainSeparator::new(0, b"pow-bits", pattern);

        let mut p = ProverState::<_, FieldUnit<F>>::new(field_sponge(), &ds);
        p.pow("grind", 8);
        let narg = p.finalize();

        let mut v = VerifierState::<_, FieldUnit<F>>::new(field_sponge(), &ds, &narg);
        let _ = v.check_pow("grind", 4);
    }

    #[test]
    fn extension_messages_and_challenges_round_trip() {
        // Invariant: extension steps travel as `DIMENSION` base encodings.
        //
        // Both sides route through the same `ExtensionFieldCodec`, so the layout
        // has one definition.
        let pattern = InteractionPattern::new(vec![
            Interaction::algebra::<F, EF4>(
                Hierarchy::Atomic,
                Kind::Message,
                "opening",
                Length::Scalar,
            ),
            Interaction::algebra::<F, EF4>(
                Hierarchy::Atomic,
                Kind::Challenge,
                "zeta",
                Length::Scalar,
            ),
        ])
        .unwrap();
        let ds: DomainSeparator<u8> = DomainSeparator::new(0, b"ext", pattern);

        let value =
            EF4::from_basis_coefficients_iter([11u32, 13, 17, 19].into_iter().map(F::from_u32))
                .expect("constructing an extension from its basis coefficients");

        let mut p = ProverState::<_, u8>::new(byte_sponge(), &ds);
        p.add_extension::<F, EF4, ByteCodec>("opening", &value);
        let zeta_p = p.challenge_extension::<F, EF4, ByteCodec>("zeta");
        let narg = p.finalize();

        // Four base-field coefficients at four bytes each.
        assert_eq!(narg.len(), 16);

        let mut v = VerifierState::<_, u8>::new(byte_sponge(), &ds, &narg);
        let read = v
            .next_extension::<F, EF4, ByteCodec>("opening")
            .expect("legal extension element");
        let zeta_v = v.challenge_extension::<F, EF4, ByteCodec>("zeta");
        v.finalize().expect("NARG fully consumed");

        assert_eq!(read.into_inner(), value);
        assert_eq!(zeta_p, zeta_v);
    }

    #[test]
    fn public_scalars_bind_the_sponge_without_touching_the_wire() {
        // Invariant: a public input changes the challenge but adds nothing to the proof.
        let pattern = InteractionPattern::new(vec![
            scalar_step(Kind::Public, "public-inputs", Length::Fixed(2)),
            scalar_step(Kind::Public, "index", Length::Scalar),
            scalar_step(Kind::Challenge, "alpha", Length::Scalar),
        ])
        .unwrap();
        let ds: DomainSeparator<u8> = DomainSeparator::new(0, b"public", pattern);

        // Helper: drive a prover over the given public inputs.
        let drive = |inputs: [u32; 2]| -> (TranscriptBound<F>, Vec<u8>) {
            let values: Vec<F> = inputs.iter().copied().map(F::from_u32).collect();
            let mut p = ProverState::<_, u8>::new(byte_sponge(), &ds);
            p.add_public_scalars::<F, ByteCodec>("public-inputs", &values);
            p.add_public_scalar::<F, ByteCodec>("index", &F::from_u32(5));
            let c = p.challenge_scalar::<F, ByteCodec>("alpha");
            (c, p.finalize())
        };

        let (c_a, narg_a) = drive([1, 2]);
        let (c_b, narg_b) = drive([1, 3]);

        // Property 1: public data is absorbed, so a different input moves the challenge.
        assert_ne!(c_a, c_b);
        // Property 2: public data never reaches the wire, so both proofs are empty.
        assert!(narg_a.is_empty());
        assert!(narg_b.is_empty());

        // Property 3: the verifier re-absorbs its own copy and lands on the same challenge.
        let mut v = VerifierState::<_, u8>::new(byte_sponge(), &ds, &narg_a);
        v.observe_public_scalars::<F, ByteCodec>(
            "public-inputs",
            &[F::from_u32(1), F::from_u32(2)],
        );
        v.observe_public_scalar::<F, ByteCodec>("index", &F::from_u32(5));
        let c_v = v.challenge_scalar::<F, ByteCodec>("alpha");
        v.finalize().expect("NARG fully consumed");
        assert_eq!(c_a, c_v);
    }

    #[test]
    fn byte_messages_are_absorbed_unlike_hints() {
        // Invariant: a digest-shaped message binds the sponge; a hint does not.
        //
        // This is the step a Merkle root travels on.
        let pattern = InteractionPattern::new(vec![
            byte_step(Kind::Message, "commitment", Length::Fixed(32)),
            byte_step(Kind::Message, "opening", Length::Bounded(64)),
            scalar_step(Kind::Challenge, "alpha", Length::Scalar),
        ])
        .unwrap();
        let ds: DomainSeparator<u8> = DomainSeparator::new(0, b"digest", pattern);

        // Helper: drive a prover over the given root and opening.
        let drive = |root: [u8; 32], opening: &[u8]| -> (TranscriptBound<F>, Vec<u8>) {
            let mut p = ProverState::<_, u8>::new(byte_sponge(), &ds);
            p.add_bytes("commitment", &root);
            p.add_bytes_bounded("opening", opening, 64);
            let c = p.challenge_scalar::<F, ByteCodec>("alpha");
            (c, p.finalize())
        };

        let (c_a, narg_a) = drive([0xaa; 32], &[1, 2, 3]);
        let (c_b, _) = drive([0xbb; 32], &[1, 2, 3]);
        let (c_c, _) = drive([0xaa; 32], &[1, 2, 3, 4]);

        // Property 1: changing the commitment moves the challenge.
        assert_ne!(c_a, c_b);
        // Property 2: the absorbed length prefix makes a longer opening a different transcript.
        assert_ne!(c_a, c_c);

        // Property 3: the verifier reads both back and lands on the same challenge.
        let mut v = VerifierState::<_, u8>::new(byte_sponge(), &ds, &narg_a);
        let root = v.next_bytes("commitment", 32).expect("legal digest");
        let opening = v.next_bytes_bounded("opening", 64).expect("legal opening");
        let c_v = v.challenge_scalar::<F, ByteCodec>("alpha");
        v.finalize().expect("NARG fully consumed");

        assert_eq!(root, [0xaa; 32]);
        assert_eq!(opening, &[1, 2, 3]);
        assert_eq!(c_a, c_v);
    }

    #[test]
    fn salt_changes_subsequent_challenges() {
        // Pattern: 8-byte salt followed by one challenge scalar.
        let pattern = InteractionPattern::new(vec![
            byte_step(Kind::Salt, "salt", Length::Fixed(8)),
            scalar_step(Kind::Challenge, "alpha", Length::Scalar),
        ])
        .unwrap();

        // Helper: run a prover with the given salt and return the bound challenge.
        let drive = |salt: &[u8]| -> TranscriptBound<F> {
            let ds: DomainSeparator<u8> = DomainSeparator::new(0, b"zk", pattern.clone());
            let mut p = ProverState::<_, u8>::new(byte_sponge(), &ds);
            p.add_salt("salt", salt);
            let c = p.challenge_scalar::<F, ByteCodec>("alpha");
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
        let pattern = InteractionPattern::new(vec![
            byte_step(Kind::Salt, "salt", Length::Fixed(16)),
            scalar_step(Kind::Challenge, "alpha", Length::Scalar),
        ])
        .unwrap();

        let ds: DomainSeparator<u8> = DomainSeparator::new(0, b"zk", pattern);

        // Fixed salt fixture so the test is deterministic.
        let salt = [0xa5u8; 16];

        // Prover absorbs the salt then samples the challenge.
        let mut p = ProverState::<_, u8>::new(byte_sponge(), &ds);
        p.add_salt("salt", &salt);
        let c_p = p.challenge_scalar::<F, ByteCodec>("alpha");
        let narg = p.finalize();

        // Verifier seeded identically reads the salt back from the wire and re-samples.
        let mut v = VerifierState::<_, u8>::new(byte_sponge(), &ds, &narg);
        let read_salt = v.next_salt("salt", 16).expect("verifier reads salt");
        let c_v = v.challenge_scalar::<F, ByteCodec>("alpha");
        v.finalize().expect("NARG fully consumed");

        // Property 1: salt round-trips byte-for-byte.
        assert_eq!(read_salt, salt);
        // Property 2: same salt absorbed -> same challenge derived.
        assert_eq!(c_p, c_v);
    }

    #[test]
    fn hints_are_carried_in_narg_but_not_absorbed() {
        // Pattern: 4-byte hint followed by one challenge scalar.
        let pattern = InteractionPattern::new(vec![
            byte_step(Kind::Hint, "merkle-path", Length::Fixed(4)),
            scalar_step(Kind::Challenge, "alpha", Length::Scalar),
        ])
        .unwrap();
        let ds: DomainSeparator<u8> = DomainSeparator::new(0, b"hint-test", pattern);

        // Helper: run a prover with the given hint bytes and return (bound challenge, wire).
        let drive_with_hint = |hint: &[u8; 4]| -> (TranscriptBound<F>, Vec<u8>) {
            let mut p = ProverState::<_, u8>::new(byte_sponge(), &ds);
            p.add_hint("merkle-path", hint);
            let c = p.challenge_scalar::<F, ByteCodec>("alpha");
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
        let mut v = VerifierState::<_, u8>::new(byte_sponge(), &ds, &narg_a);
        let read_hint = v
            .next_hint("merkle-path", 4)
            .expect("verifier reads hint bytes");
        let _ = v.challenge_scalar::<F, ByteCodec>("alpha");
        v.finalize().expect("NARG fully consumed");
        assert_eq!(read_hint, &[1, 2, 3, 4]);
    }

    #[test]
    fn bounded_hint_round_trips_with_short_payload() {
        // Invariant: a bounded hint round-trips its actual payload exactly.
        //
        // The hint never enters the sponge, so the verifier's challenge matches the prover's.

        // Fixture state: hint cap of 8 bytes followed by one challenge.
        let pattern = InteractionPattern::new(vec![
            byte_step(Kind::Hint, "auth-path", Length::Bounded(8)),
            scalar_step(Kind::Challenge, "alpha", Length::Scalar),
        ])
        .unwrap();
        let ds: DomainSeparator<u8> = DomainSeparator::new(0, b"bounded-hint", pattern);

        // Mutation (prover): send 3 bytes — strictly less than the cap — then sample a challenge.
        let mut p = ProverState::<_, u8>::new(byte_sponge(), &ds);
        p.add_hint_bounded("auth-path", &[0xaa, 0xbb, 0xcc], 8);
        let c_p = p.challenge_scalar::<F, ByteCodec>("alpha");
        let narg = p.finalize();

        // Mutation (verifier): replay the same step, read the actual byte count, sample a challenge.
        let mut v = VerifierState::<_, u8>::new(byte_sponge(), &ds, &narg);
        let read_hint = v.next_hint_bounded("auth-path", 8).expect("legal hint");
        let c_v = v.challenge_scalar::<F, ByteCodec>("alpha");
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
        let pattern = InteractionPattern::new(vec![
            byte_step(Kind::Hint, "auth-path", Length::Bounded(8)),
            scalar_step(Kind::Challenge, "alpha", Length::Scalar),
        ])
        .unwrap();
        let ds: DomainSeparator<u8> = DomainSeparator::new(0, b"hint-iso", pattern);

        // Helper: drive a prover with the supplied hint and return the sampled challenge.
        let drive = |hint: &[u8]| -> TranscriptBound<F> {
            let mut p = ProverState::<_, u8>::new(byte_sponge(), &ds);
            p.add_hint_bounded("auth-path", hint, 8);
            let c = p.challenge_scalar::<F, ByteCodec>("alpha");
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
        let pattern = InteractionPattern::new(vec![
            scalar_step(Kind::Message, "msgs", Length::Bounded(5)),
            scalar_step(Kind::Challenge, "alpha", Length::Scalar),
        ])
        .unwrap();
        let ds: DomainSeparator<u8> = DomainSeparator::new(0, b"bounded-msgs", pattern);

        // Mutation (prover): send 3 values, strictly below the cap of 5.
        let msgs: Vec<F> = (1u32..=3).map(F::from_u32).collect();
        let mut p = ProverState::<_, u8>::new(byte_sponge(), &ds);
        p.add_scalars_bounded::<F, ByteCodec>("msgs", &msgs, 5);
        let c_p = p.challenge_scalar::<F, ByteCodec>("alpha");
        let narg = p.finalize();

        // Mutation (verifier): replay the step and sample the matching challenge.
        let mut v = VerifierState::<_, u8>::new(byte_sponge(), &ds, &narg);
        let read = v
            .next_scalars_bounded::<F, ByteCodec>("msgs", 5)
            .expect("legal scalars");
        let c_v = v.challenge_scalar::<F, ByteCodec>("alpha");
        v.finalize().expect("NARG fully consumed");

        // Property 1: values round-trip in their original order.
        let read_vals: Vec<F> = read.into_iter().map(TranscriptBound::into_inner).collect();
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
        let pattern = InteractionPattern::new(vec![
            scalar_step(Kind::Message, "msgs", Length::Bounded(5)),
            scalar_step(Kind::Challenge, "alpha", Length::Scalar),
        ])
        .unwrap();
        let ds: DomainSeparator<u8> = DomainSeparator::new(0, b"len-bind", pattern);

        // Helper: drive a prover with `n` zero scalars and return the sampled challenge.
        let drive = |n: usize| -> TranscriptBound<F> {
            let zeros: Vec<F> = vec![F::ZERO; n];
            let mut p = ProverState::<_, u8>::new(byte_sponge(), &ds);
            p.add_scalars_bounded::<F, ByteCodec>("msgs", &zeros, 5);
            let c = p.challenge_scalar::<F, ByteCodec>("alpha");
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
        let pat_a =
            InteractionPattern::new(vec![byte_step(Kind::Hint, "auth", Length::Bounded(7))])
                .unwrap();
        let pat_b =
            InteractionPattern::new(vec![byte_step(Kind::Hint, "auth", Length::Bounded(8))])
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
        let pattern =
            InteractionPattern::new(vec![byte_step(Kind::Hint, "auth", Length::Bounded(4))])
                .unwrap();
        let ds: DomainSeparator<u8> = DomainSeparator::new(0, b"oversize", pattern);

        // Mutation: feed 5 bytes into a cap of 4.
        //
        // The failure path releases the drop-time check, so this panic stays single.
        let mut p = ProverState::<_, u8>::new(byte_sponge(), &ds);
        p.add_hint_bounded("auth", &[0u8; 5], 4);
    }

    #[test]
    #[should_panic(expected = "Dropped unfinalized ProverState")]
    fn prover_dropped_without_finalize_panics() {
        // Invariant: abandoning a transcript halfway is a bug, not a silent no-op.
        let ds: DomainSeparator<u8> = DomainSeparator::new(0, b"drop", small_pattern());
        let mut p = ProverState::<_, u8>::new(byte_sponge(), &ds);
        p.add_scalars::<F, ByteCodec>("msgs", &[F::ONE, F::ONE, F::ONE]);
    }

    #[test]
    fn distinct_protocol_ids_yield_distinct_challenges() {
        // Three small messages reused across both runs.
        let messages: Vec<F> = vec![F::from_u32(7), F::from_u32(11), F::from_u32(13)];

        // Helper: drive a prover under the given protocol name and return its bound challenges.
        let drive = |name: &[u8]| -> Vec<TranscriptBound<F>> {
            let ds: DomainSeparator<u8> = DomainSeparator::new(0x01, name, small_pattern());
            let mut p = ProverState::<_, u8>::new(byte_sponge(), &ds);
            p.add_scalars::<F, ByteCodec>("msgs", &messages);
            let chs = p.challenge_scalars::<F, ByteCodec>("challs", 2);
            let _ = p.finalize();
            chs
        };

        // Property: different protocol names -> different seeds -> different challenges.
        assert_ne!(drive(b"protocol-a"), drive(b"protocol-b"));
    }

    #[test]
    fn nested_sub_protocols_replay_in_lockstep() {
        // Invariant: every recordable step is playable by the drivers.
        //
        // Containers of a non-`Protocol` kind included.
        let mut recorder = PatternState::<u8>::new();
        recorder.begin_protocol::<()>("outer");
        recorder.begin::<()>("commitments", Kind::Message);
        recorder.interact(scalar_step(Kind::Message, "a", Length::Scalar));
        recorder.end::<()>("commitments", Kind::Message);
        recorder.interact(scalar_step(Kind::Challenge, "alpha", Length::Scalar));
        recorder.end_protocol::<()>("outer");
        let ds: DomainSeparator<u8> = DomainSeparator::new(0, b"nested", recorder.finalize());

        let mut p = ProverState::<_, u8>::new(byte_sponge(), &ds);
        p.begin_protocol::<()>("outer");
        p.begin::<()>("commitments", Kind::Message);
        p.add_scalar::<F, ByteCodec>("a", &F::from_u32(42));
        p.end::<()>("commitments", Kind::Message);
        let c_p = p.challenge_scalar::<F, ByteCodec>("alpha");
        p.end_protocol::<()>("outer");
        let narg = p.finalize();

        let mut v = VerifierState::<_, u8>::new(byte_sponge(), &ds, &narg);
        v.begin_protocol::<()>("outer");
        v.begin::<()>("commitments", Kind::Message);
        let read = v.next_scalar::<F, ByteCodec>("a").expect("legal scalar");
        v.end::<()>("commitments", Kind::Message);
        let c_v = v.challenge_scalar::<F, ByteCodec>("alpha");
        v.end_protocol::<()>("outer");
        v.finalize().expect("NARG fully consumed");

        assert_eq!(read.into_inner(), F::from_u32(42));
        assert_eq!(c_p, c_v);
    }

    #[test]
    fn end_to_end_wire_format_is_pinned() {
        // Invariant: the NARG bytes and the challenge stream are a fixed function
        // of (protocol id, instance label, pattern, messages).
        //
        // Nothing here is derived at run time from the implementation: the
        // expected values are literals, so a refactor that silently changes the
        // seeding layout, the absorb order, or the byte-to-field decoding
        // breaks this test rather than the next protocol to depend on it.
        //
        // Sponge: Keccak-256 hash chain over bytes.
        // Field:  BabyBear, 4-byte canonical big-endian on the wire.
        let pattern = InteractionPattern::new(vec![
            byte_step(Kind::Message, "commitment", Length::Fixed(4)),
            scalar_step(Kind::Message, "value", Length::Scalar),
            scalar_step(Kind::Challenge, "alpha", Length::Fixed(2)),
            byte_step(Kind::Hint, "aux", Length::Bounded(8)),
        ])
        .unwrap();
        let mut ds: DomainSeparator<u8> = DomainSeparator::new(0x07, b"p3-kat", pattern);
        ds.instance(b"vector-1");

        let mut p = ProverState::<_, u8>::new(byte_sponge(), &ds);
        p.add_bytes("commitment", &[0xde, 0xad, 0xbe, 0xef]);
        p.add_scalar::<F, ByteCodec>("value", &F::from_u32(1_234_567));
        let challenges = p.challenge_scalars::<F, ByteCodec>("alpha", 2);
        p.add_hint_bounded("aux", &[0x01, 0x02], 8);
        let narg = p.finalize();

        // Wire layout: [commitment(4) | value(4) | hint len(1) | hint(2)].
        assert_eq!(
            narg,
            vec![
                0xde, 0xad, 0xbe, 0xef, 0x00, 0x12, 0xd6, 0x87, 0x02, 0x01, 0x02
            ],
        );

        // Challenge stream pinned to the exact field elements.
        let drawn: Vec<u32> = challenges
            .iter()
            .map(|c| c.as_inner().as_canonical_u32())
            .collect();
        assert_eq!(drawn, vec![PINNED_ALPHA_0, PINNED_ALPHA_1]);

        // The verifier reproduces the same stream from the same wire.
        let mut v = VerifierState::<_, u8>::new(byte_sponge(), &ds, &narg);
        assert_eq!(
            v.next_bytes("commitment", 4).unwrap(),
            [0xde, 0xad, 0xbe, 0xef]
        );
        assert_eq!(
            v.next_scalar::<F, ByteCodec>("value").unwrap().into_inner(),
            F::from_u32(1_234_567),
        );
        let replayed: Vec<u32> = v
            .challenge_scalars::<F, ByteCodec>("alpha", 2)
            .iter()
            .map(|c| c.as_inner().as_canonical_u32())
            .collect();
        assert_eq!(v.next_hint_bounded("aux", 8).unwrap(), &[0x01, 0x02]);
        v.finalize().expect("NARG fully consumed");
        assert_eq!(replayed, drawn);
    }

    /// First challenge of the pinned end-to-end vector.
    const PINNED_ALPHA_0: u32 = 252_236_841;
    /// Second challenge of the pinned end-to-end vector.
    const PINNED_ALPHA_1: u32 = 884_894_143;
}
