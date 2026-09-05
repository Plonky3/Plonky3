//! Verifier-side driver.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::{BasedVectorSpace, Field, PrimeField64};

use crate::fs::bound::TranscriptBound;
use crate::fs::codecs::{
    Codec, ExtensionFieldCodec, bound_byte_width, decode_field_be_canonical, decode_len_be,
    encode_len_be, field_byte_size,
};
use crate::fs::domain_separator::DomainSeparator;
use crate::fs::error::TranscriptError;
use crate::fs::pattern::{Hierarchy, Interaction, Kind, Label, Length, Pattern, PatternPlayer};
use crate::fs::state::assert_challenge_security;
use crate::fs::unit::Unit;
use crate::{CanObserve, CanSampleBits, GrindingChallenger};

/// Drives a verifier-side transcript in lockstep with a recorded pattern.
///
/// Reads bytes from a caller-supplied slice through a cursor — no copies.
///
/// Wire-format problems return a structured error.
///
/// Pattern misuse panics with a diff message.
///
/// # Panics
///
/// On drop without [`Self::finalize`], unless a read has already failed.
///
/// Forgetting to finalise skips the pattern-fully-replayed check.
/// It also skips the trailing-bytes check.
/// Such a verifier accepts a proof with arbitrary appended data.
/// It also never notices an unconsumed step, a skipped proof-of-work among them.
///
/// A read that returns an error releases the check.
/// The caller is expected to reject the proof.
/// The structured error must reach them, not a drop-time panic on top of it.
pub struct VerifierState<'a, C, U: Unit = u8> {
    /// Underlying sponge, seeded identically to the prover.
    challenger: C,
    /// Pattern player that validates each call against the recorded sequence.
    player: PatternPlayer,
    /// Caller-supplied wire bytes consumed in order.
    narg: &'a [u8],
    /// Read position into the wire bytes.
    cursor: usize,
    /// Type-level marker for the sponge alphabet.
    _u: PhantomData<U>,
}

impl<C, U: Unit> Drop for VerifierState<'_, C, U> {
    fn drop(&mut self) {
        // Loud failure surfaces a verifier that never ran its final checks.
        //
        // Failing reads and pattern panics both mark the player aborted first,
        // so this never fires while another failure unwinds.
        if !self.player.is_finalized() {
            let steps = self.player.remaining();
            let bytes = self.remaining_narg();
            // Release the player's own drop check so this panic stays single.
            self.player.abort();
            panic!(
                "Dropped unfinalized VerifierState: {steps} pattern step(s) were never replayed \
                 and {bytes} NARG byte(s) were never checked."
            );
        }
    }
}

impl<'a, C, U: Unit> VerifierState<'a, C, U> {
    /// Build a driver and seed the challenger from the domain separator.
    pub fn new(mut challenger: C, ds: &DomainSeparator<U>, narg: &'a [u8]) -> Self
    where
        C: CanObserve<U::Item>,
    {
        // Seed identically to the prover so both sides land on the same sponge state.
        ds.seed(&mut challenger);
        let player = PatternPlayer::new(ds.pattern().clone());
        Self {
            challenger,
            player,
            narg,
            cursor: 0,
            _u: PhantomData,
        }
    }

    /// Read-only access to the underlying challenger.
    pub const fn challenger(&self) -> &C {
        &self.challenger
    }

    /// Number of wire bytes still ahead of the cursor.
    pub const fn remaining_narg(&self) -> usize {
        self.narg.len() - self.cursor
    }

    /// Finalise the driver.
    ///
    /// # Errors
    ///
    /// Returns an error when wire bytes remain unread.
    ///
    /// # Panics
    ///
    /// When the recorded pattern is not fully replayed.
    pub fn finalize(mut self) -> Result<(), TranscriptError> {
        // Pattern check: every recorded step must have been replayed.
        //
        // `finalize` marks the player before asserting, so `Drop` stays quiet
        // whichever way this call goes.
        self.player.finalize();
        // Wire check: trailing bytes mean the prover smuggled data the verifier never read.
        if self.cursor != self.narg.len() {
            return Err(TranscriptError::BadProofShape {
                reason: "trailing NARG bytes after final verifier step",
            });
        }
        Ok(())
    }

    /// Release the drop-time check when a read has failed.
    ///
    /// A failed read leaves the transcript mid-step.
    /// The caller is expected to reject the proof, so the error is what must surface.
    fn poison<T>(&mut self, result: Result<T, TranscriptError>) -> Result<T, TranscriptError> {
        if result.is_err() {
            self.player.abort();
        }
        result
    }

    /// Take `n` raw bytes from the wire cursor, or fail if out of bounds.
    ///
    /// The bound is computed with `checked_add` so the arithmetic is total.
    /// `n` is verifier-controlled here, but an unchecked sum in a parser is a habit worth not having.
    fn take_bytes(&mut self, n: usize) -> Result<&'a [u8], TranscriptError> {
        let end = self
            .cursor
            .checked_add(n)
            .filter(|&end| end <= self.narg.len())
            .ok_or(TranscriptError::BadProofShape {
                reason: "NARG ended before all expected bytes were read",
            })?;
        let slice = &self.narg[self.cursor..end];
        self.cursor = end;
        Ok(slice)
    }

    /// Replay a salt step by reading `byte_len` bytes from the wire.
    pub fn next_salt(&mut self, label: Label, byte_len: usize) -> Result<&'a [u8], TranscriptError>
    where
        C: CanObserve<U::Item>,
    {
        // The verifier must know the length up front:
        //
        // Reading it from the wire would let an attacker control how much data is consumed.
        self.player.interact(Interaction::bytes(
            Hierarchy::Atomic,
            Kind::Salt,
            label,
            Length::Fixed(byte_len),
        ));
        let bytes = self.take_bytes(byte_len);
        let bytes = self.poison(bytes)?;
        // Absorb so future samples depend on the prover's salt.
        U::observe_bytes(&mut self.challenger, bytes);
        Ok(bytes)
    }

    /// Replay an extension-field message the caller carries itself.
    ///
    /// # Overview
    ///
    /// A reading method takes the next value off the wire this driver consumes.
    /// An observing method takes it from the caller instead.
    /// Both bind the value into the sponge the same way.
    ///
    /// # When to use this
    ///
    /// A protocol whose proof is its own type has already deserialised the value.
    /// The prover must have used the matching observing method.
    ///
    /// # Trust
    ///
    /// The value is prover-chosen, so it is untrusted.
    /// Binding it stops the prover choosing it after seeing the next challenge.
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
        // Sponge only: the value came from the caller's own proof type.
        ExtensionFieldCodec::<F, EF, Cdc>::observe(&mut self.challenger, value);
        TranscriptBound::wrap(*value)
    }

    /// Replay a fixed-length list of extension-field messages the caller carries itself.
    ///
    /// # Panics
    ///
    /// When the supplied length differs from the recorded one.
    ///
    /// That length is attacker-controlled.
    /// A verifier must check it and reject a mismatch with an error first.
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

    /// Replay a proof-of-work step whose witness the caller carries itself.
    ///
    /// # Errors
    ///
    /// When the witness does not produce the required number of zero bits.
    pub fn observe_pow(
        &mut self,
        label: Label,
        bits: usize,
        witness: C::Witness,
    ) -> Result<(), TranscriptError>
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
        // Checking absorbs the witness, which is what keeps both sponges aligned.
        if !self.challenger.check_witness(bits, witness) {
            // Release the drop-time check so the rejection reaches the caller.
            self.player.abort();
            return Err(TranscriptError::BadProofShape {
                reason: "pow witness does not produce enough zero bits",
            });
        }
        Ok(())
    }

    /// Replay a value the challenger knows how to encode, carried by the caller.
    ///
    /// The value is prover-chosen, so it is untrusted.
    /// Binding it stops the prover choosing it after seeing the next challenge.
    ///
    /// Its width is not bound here.
    /// The component that owns the value rejects a wrong-shaped one.
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

    /// Sample `count` challenges of `width` uniform bits, in lockstep with the prover.
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

    /// Replay a public-scalar step by absorbing the caller-supplied value.
    ///
    /// Public values never travel on the wire.
    /// The verifier holds them as its own input and absorbs them as the prover did.
    pub fn observe_public_scalar<F, Cdc>(&mut self, label: Label, value: &F) -> TranscriptBound<F>
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
        // Bind into the sponge only; nothing is read from the wire.
        Cdc::observe(&mut self.challenger, value);
        TranscriptBound::wrap(*value)
    }

    /// Replay a fixed-length public-scalar list by absorbing the caller-supplied values.
    ///
    /// As with [`Self::observe_public_scalar`], nothing is read from the wire.
    pub fn observe_public_scalars<F, Cdc>(
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
        // Bind each value into the sponge; none is read from the wire.
        values
            .iter()
            .map(|v| {
                Cdc::observe(&mut self.challenger, v);
                TranscriptBound::wrap(*v)
            })
            .collect()
    }

    /// Replay an `add_scalar` step from the prover.
    pub fn next_scalar<F, Cdc>(
        &mut self,
        label: Label,
    ) -> Result<TranscriptBound<F>, TranscriptError>
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
        let value = self.read_value::<F, Cdc>();
        let value = self.poison(value)?;
        // Absorb through the same codec the prover used so both sides agree.
        Cdc::observe(&mut self.challenger, &value);
        Ok(TranscriptBound::wrap(value))
    }

    /// Replay an `add_scalars` step from the prover.
    pub fn next_scalars<F, Cdc>(
        &mut self,
        label: Label,
        n: usize,
    ) -> Result<Vec<TranscriptBound<F>>, TranscriptError>
    where
        F: PrimeField64,
        Cdc: Codec<C, F>,
    {
        // Validate: the next pattern step is a fixed-length list of `n` scalars.
        self.player.interact(Interaction::algebra::<F, F>(
            Hierarchy::Atomic,
            Kind::Message,
            label,
            Length::Fixed(n),
        ));
        let values = self.read_values::<F, Cdc>(n);
        self.poison(values)
    }

    /// Replay a bounded scalar-slice step.
    ///
    /// # Algorithm
    ///
    /// 1. Read the length prefix from the wire.
    /// 2. Reject any actual length above `max`.
    /// 3. Absorb the prefix bytes into the sponge, matching the prover.
    /// 4. Read and absorb that many scalars through the codec.
    ///
    /// # Errors
    ///
    /// - The length prefix runs past the end of the wire.
    /// - Any scalar runs past the end of the wire.
    /// - The decoded length exceeds `max`.
    /// - Any scalar encoding is non-canonical.
    pub fn next_scalars_bounded<F, Cdc>(
        &mut self,
        label: Label,
        max: usize,
    ) -> Result<Vec<TranscriptBound<F>>, TranscriptError>
    where
        F: PrimeField64,
        C: CanObserve<U::Item>,
        Cdc: Codec<C, F>,
    {
        // Validate against the recorded pattern step so shape divergence panics here.
        self.player.interact(Interaction::algebra::<F, F>(
            Hierarchy::Atomic,
            Kind::Message,
            label,
            Length::Bounded(max),
        ));
        let count = self.read_bounded_len(max, "message length exceeds declared maximum");
        let (actual, width) = self.poison(count)?;
        // Bind the count into the sponge before any value enters it.
        //
        // This keeps the transcript prefix-free, matching CO25 §6.2.
        let len_bytes = encode_len_be(actual, width);
        U::observe_bytes(&mut self.challenger, &len_bytes[..width]);
        let values = self.read_values::<F, Cdc>(actual);
        self.poison(values)
    }

    /// Replay an `add_extension` step from the prover.
    ///
    /// Routed through [`ExtensionFieldCodec`].
    /// The coefficient layout therefore has one definition, shared with the prover.
    pub fn next_extension<F, EF, Cdc>(
        &mut self,
        label: Label,
    ) -> Result<TranscriptBound<EF>, TranscriptError>
    where
        F: PrimeField64,
        EF: Field + BasedVectorSpace<F>,
        Cdc: Codec<C, F>,
    {
        // Validate: the next pattern step is a scalar message of extension type `EF`.
        self.player.interact(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Message,
            label,
            Length::Scalar,
        ));
        let value = self.read_value::<EF, ExtensionFieldCodec<F, EF, Cdc>>();
        let value = self.poison(value)?;
        ExtensionFieldCodec::<F, EF, Cdc>::observe(&mut self.challenger, &value);
        Ok(TranscriptBound::wrap(value))
    }

    /// Replay an `add_bytes` step: a fixed-length byte message.
    ///
    /// Unlike a hint, these bytes are absorbed, matching the prover.
    pub fn next_bytes(&mut self, label: Label, byte_len: usize) -> Result<&'a [u8], TranscriptError>
    where
        C: CanObserve<U::Item>,
    {
        self.player.interact(Interaction::bytes(
            Hierarchy::Atomic,
            Kind::Message,
            label,
            Length::Fixed(byte_len),
        ));
        let bytes = self.take_bytes(byte_len);
        let bytes = self.poison(bytes)?;
        U::observe_bytes(&mut self.challenger, bytes);
        Ok(bytes)
    }

    /// Replay an `add_bytes_bounded` step: a variable-length byte message.
    ///
    /// The length prefix is absorbed before the payload, matching the prover.
    ///
    /// # Errors
    ///
    /// - The length prefix or the payload runs past the end of the wire.
    /// - The decoded length exceeds `max`.
    pub fn next_bytes_bounded(
        &mut self,
        label: Label,
        max: usize,
    ) -> Result<&'a [u8], TranscriptError>
    where
        C: CanObserve<U::Item>,
    {
        self.player.interact(Interaction::bytes(
            Hierarchy::Atomic,
            Kind::Message,
            label,
            Length::Bounded(max),
        ));
        let count = self.read_bounded_len(max, "message length exceeds declared maximum");
        let (actual, width) = self.poison(count)?;
        let payload = self.take_bytes(actual);
        let payload = self.poison(payload)?;
        // Length first, then payload: the same order the prover absorbed them in.
        let len_bytes = encode_len_be(actual, width);
        U::observe_bytes(&mut self.challenger, &len_bytes[..width]);
        U::observe_bytes(&mut self.challenger, payload);
        Ok(payload)
    }

    /// Replay an `add_hint` step.
    ///
    /// Hint bytes are returned to the caller; they are never absorbed.
    pub fn next_hint(
        &mut self,
        label: Label,
        byte_len: usize,
    ) -> Result<&'a [u8], TranscriptError> {
        self.player.interact(Interaction::bytes(
            Hierarchy::Atomic,
            Kind::Hint,
            label,
            Length::Fixed(byte_len),
        ));
        let bytes = self.take_bytes(byte_len);
        self.poison(bytes)
    }

    /// Replay a bounded hint step.
    ///
    /// # Algorithm
    ///
    /// 1. Read the length prefix from the wire.
    /// 2. Reject any actual length above `max`.
    /// 3. Return the payload bytes as a borrowed slice.
    ///
    /// Nothing is absorbed into the sponge.
    ///
    /// # Errors
    ///
    /// - The length prefix runs past the end of the wire.
    /// - The payload runs past the end of the wire.
    /// - The decoded length exceeds `max`.
    pub fn next_hint_bounded(
        &mut self,
        label: Label,
        max: usize,
    ) -> Result<&'a [u8], TranscriptError> {
        // Validate against the recorded pattern step so shape divergence panics here.
        self.player.interact(Interaction::bytes(
            Hierarchy::Atomic,
            Kind::Hint,
            label,
            Length::Bounded(max),
        ));
        let count = self.read_bounded_len(max, "hint length exceeds declared maximum");
        let (actual, _) = self.poison(count)?;
        // Hand back the payload as a borrowed slice — no sponge absorption.
        let payload = self.take_bytes(actual);
        self.poison(payload)
    }

    /// Sample one challenge scalar in lockstep with the prover.
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

    /// Sample `n` challenge scalars in lockstep with the prover.
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

    /// Sample one challenge extension-field element in lockstep with the prover.
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

    /// Replay a proof-of-work step.
    ///
    /// - Reads the witness from the wire,
    /// - Rejects a non-canonical encoding,
    /// - Absorbs it through the challenger's PoW path.
    ///
    /// `bits` is part of the recorded step.
    /// A verifier configured with a different difficulty fails the pattern check.
    /// It cannot silently accept a cheaper grind.
    pub fn check_pow(&mut self, label: Label, bits: usize) -> Result<(), TranscriptError>
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
        // Read the witness from the wire.
        //
        // `decode_field_be_canonical` reads exactly `field_byte_size` bytes and
        // rejects anything at or above `p`, and a fixed-width big-endian
        // encoding of an integer below `2^(8*need)` is unique.
        // So a canonical decode already implies a canonical encoding.
        let need = field_byte_size::<<C as GrindingChallenger>::Witness>();
        let raw = self.take_bytes(need);
        let raw = self.poison(raw)?;
        let witness = decode_field_be_canonical::<<C as GrindingChallenger>::Witness>(raw);
        let witness = self.poison(witness)?;
        // Verify the witness produces the required number of zero bits.
        if !self.challenger.check_witness(bits, witness) {
            self.player.abort();
            return Err(TranscriptError::BadProofShape {
                reason: "pow witness does not produce enough zero bits",
            });
        }
        Ok(())
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

    /// Read one codec-shaped value off the wire without absorbing it.
    fn read_value<T, Cdc: Codec<C, T>>(&mut self) -> Result<T, TranscriptError> {
        let raw = self.take_bytes(Cdc::wire_len())?;
        Cdc::decode(raw)
    }

    /// Read `n` codec-shaped values off the wire, absorbing and binding each.
    fn read_values<T, Cdc: Codec<C, T>>(
        &mut self,
        n: usize,
    ) -> Result<Vec<TranscriptBound<T>>, TranscriptError> {
        // Refuse to preallocate for more values than the wire can hold.
        //
        // `n` is attacker-controlled for a bounded step, so a short wire with a
        // large declared count must not force a huge `with_capacity`.
        if n.checked_mul(Cdc::wire_len())
            .is_none_or(|bytes| bytes > self.remaining_narg())
        {
            return Err(TranscriptError::BadProofShape {
                reason: "step declares more values than the wire holds",
            });
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let value = self.read_value::<T, Cdc>()?;
            // Absorb through the same codec the prover used so both sides agree.
            Cdc::observe(&mut self.challenger, &value);
            out.push(TranscriptBound::wrap(value));
        }
        Ok(out)
    }

    /// Read the length prefix of a bounded step and check it against `max`.
    ///
    /// Returns the decoded count alongside the prefix width both sides derived.
    fn read_bounded_len(
        &mut self,
        max: usize,
        over_cap_reason: &'static str,
    ) -> Result<(usize, usize), TranscriptError> {
        // Prefix width is deterministic on both sides from the recorded bound.
        let width = bound_byte_width(max);
        let actual = decode_len_be(self.take_bytes(width)?, width);
        // A wire length above the cap is malformed input, not a panic-worthy bug.
        if actual > max {
            return Err(TranscriptError::BadProofShape {
                reason: over_cap_reason,
            });
        }
        Ok((actual, width))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_keccak::Keccak256Hash;

    use super::*;
    use crate::HashChallenger;
    use crate::fs::codecs::BytesToFieldCodec;
    use crate::fs::pattern::InteractionPattern;
    use crate::fs::state::ProverState;

    /// Concrete field exercised in this module's tests.
    type F = BabyBear;
    /// Byte codec used throughout this module.
    type ByteCodec = BytesToFieldCodec<F>;

    /// Keccak-chained byte sponge backing the transcript in tests.
    ///
    /// Each squeeze advances the chaining state.
    /// Consecutive samples therefore differ.
    fn sponge() -> HashChallenger<u8, Keccak256Hash, 32> {
        HashChallenger::new(Vec::new(), Keccak256Hash)
    }

    /// One atomic byte step with the given role and length.
    fn byte_step(kind: Kind, label: &'static str, length: Length) -> Interaction {
        Interaction::bytes(Hierarchy::Atomic, kind, label, length)
    }

    fn one_msg_pattern() -> InteractionPattern {
        InteractionPattern::new(vec![Interaction::algebra::<F, F>(
            Hierarchy::Atomic,
            Kind::Message,
            "msg",
            Length::Scalar,
        )])
        .unwrap()
    }

    #[test]
    fn truncated_narg_yields_bad_proof_shape() {
        // Pattern wants a 4-byte scalar; verifier gets 1 byte.
        let ds: DomainSeparator<u8> = DomainSeparator::new(0, b"trunc", one_msg_pattern());
        let narg = [0u8; 1];
        let mut v = VerifierState::<_, u8>::new(sponge(), &ds, &narg);
        let err = v
            .next_scalar::<F, ByteCodec>("msg")
            .expect_err("truncated NARG must error");
        assert_eq!(
            err,
            TranscriptError::BadProofShape {
                reason: "NARG ended before all expected bytes were read",
            }
        );
    }

    #[test]
    fn non_canonical_scalar_encoding_is_rejected() {
        // 0xFFFFFFFF > BabyBear order, so canonical decoding rejects it.
        let ds: DomainSeparator<u8> = DomainSeparator::new(0, b"non-canon", one_msg_pattern());
        let narg = [0xffu8; 4];
        let mut v = VerifierState::<_, u8>::new(sponge(), &ds, &narg);
        let err = v
            .next_scalar::<F, ByteCodec>("msg")
            .expect_err("non-canonical encoding must error");
        assert_eq!(
            err,
            TranscriptError::BadProofShape {
                reason: "field encoding outside canonical range",
            }
        );
    }

    #[test]
    fn trailing_narg_bytes_rejected_at_finalize() {
        // Pattern: one scalar message.
        let ds: DomainSeparator<u8> = DomainSeparator::new(0, b"trailing", one_msg_pattern());

        // Prover writes a valid NARG, then we smuggle one extra byte at the tail.
        let mut p = ProverState::<_, u8>::new(sponge(), &ds);
        p.add_scalar::<F, ByteCodec>("msg", &F::from_u32(7u32));
        let mut narg = p.finalize();
        narg.push(0x42);

        // Verifier consumes the legal scalar, then finalize must reject the leftover byte.
        let mut v = VerifierState::<_, u8>::new(sponge(), &ds, &narg);
        let _ = v.next_scalar::<F, ByteCodec>("msg").expect("legal scalar");
        let err = v.finalize().expect_err("trailing bytes must be rejected");

        // Property: finalize reports the exact "trailing bytes" reason.
        assert_eq!(
            err,
            TranscriptError::BadProofShape {
                reason: "trailing NARG bytes after final verifier step",
            }
        );
    }

    #[test]
    #[should_panic(expected = "Dropped unfinalized VerifierState")]
    fn verifier_dropped_without_finalize_panics() {
        // Invariant: skipping `finalize` skips both end-of-proof checks.
        //
        // A verifier that forgot it would accept the trailing byte appended
        // here, so the omission has to be loud.
        let ds: DomainSeparator<u8> = DomainSeparator::new(0, b"drop", one_msg_pattern());

        let mut p = ProverState::<_, u8>::new(sponge(), &ds);
        p.add_scalar::<F, ByteCodec>("msg", &F::from_u32(7u32));
        let mut narg = p.finalize();
        narg.push(0x42);

        let mut v = VerifierState::<_, u8>::new(sponge(), &ds, &narg);
        let _ = v.next_scalar::<F, ByteCodec>("msg").expect("legal scalar");
    }

    #[test]
    fn a_failed_read_releases_the_drop_check() {
        // Invariant: the structured error is what reaches the caller.
        //
        // A failing read leaves the transcript mid-step, so the drop-time
        // panic must stand down and let the rejection surface.
        let ds: DomainSeparator<u8> = DomainSeparator::new(0, b"poison", one_msg_pattern());
        let narg = [0u8; 1];
        let mut v = VerifierState::<_, u8>::new(sponge(), &ds, &narg);
        assert!(v.next_scalar::<F, ByteCodec>("msg").is_err());
        // Dropping `v` here must be quiet.
        drop(v);
    }

    #[test]
    fn bounded_hint_rejects_length_above_max() {
        // Invariant: a wire length above the recorded cap is malformed input.
        //
        // The verifier rejects it with a structured error rather than panicking.

        // Fixture state: hint cap of 4 bytes.
        let pat = InteractionPattern::new(vec![byte_step(Kind::Hint, "auth", Length::Bounded(4))])
            .unwrap();
        let ds: DomainSeparator<u8> = DomainSeparator::new(0, b"hint-oob", pat);

        // Mutation: hand-craft a wire frame whose prefix declares 5 bytes.
        //
        // ```text
        //     cap = 4 → prefix width = 1 byte
        //     wire   = [0x05, .., .., .., .., .., ..]
        //                ^^^^ declared count above cap
        // ```
        let narg = [5u8, 0, 0, 0, 0, 0, 0];

        // The verifier sees the over-cap prefix and surfaces a structured error.
        let mut v = VerifierState::<_, u8>::new(sponge(), &ds, &narg);
        let err = v
            .next_hint_bounded("auth", 4)
            .expect_err("length above max must be rejected");
        assert_eq!(
            err,
            TranscriptError::BadProofShape {
                reason: "hint length exceeds declared maximum",
            }
        );
    }

    #[test]
    fn bounded_hint_rejects_truncated_payload() {
        // Invariant: a wire frame that promises more bytes than it carries is malformed.

        // Fixture state: hint cap of 8 bytes.
        let pat = InteractionPattern::new(vec![byte_step(Kind::Hint, "auth", Length::Bounded(8))])
            .unwrap();
        let ds: DomainSeparator<u8> = DomainSeparator::new(0, b"hint-trunc", pat);

        // Mutation: declare 7 payload bytes but supply only 2.
        //
        // ```text
        //     wire = [0x07, 0xaa, 0xbb]
        //              ^^^^ declared
        //                    ^^^^^^^^^^ only 2 bytes follow
        // ```
        let narg = [7u8, 0xaa, 0xbb];

        // The verifier runs out of bytes mid-payload and reports a malformed wire.
        let mut v = VerifierState::<_, u8>::new(sponge(), &ds, &narg);
        let err = v
            .next_hint_bounded("auth", 8)
            .expect_err("truncated payload must be rejected");
        assert_eq!(
            err,
            TranscriptError::BadProofShape {
                reason: "NARG ended before all expected bytes were read",
            }
        );
    }

    #[test]
    fn bounded_scalars_rejects_length_above_max() {
        // Invariant: a wire length above the cap is malformed for messages too.

        // Fixture state: scalar slice with cap of 2.
        let pat = InteractionPattern::new(vec![Interaction::algebra::<F, F>(
            Hierarchy::Atomic,
            Kind::Message,
            "msgs",
            Length::Bounded(2),
        )])
        .unwrap();
        let ds: DomainSeparator<u8> = DomainSeparator::new(0, b"msg-oob", pat);

        // Mutation: declare 3 scalars where the cap is 2.
        //
        // ```text
        //     cap = 2 → prefix width = 1 byte
        //     wire   = [0x03]
        //               ^^^^ declared count above cap
        // ```
        let narg = [3u8];

        // The verifier surfaces a structured error before touching the sponge.
        let mut v = VerifierState::<_, u8>::new(sponge(), &ds, &narg);
        let err = v
            .next_scalars_bounded::<F, ByteCodec>("msgs", 2)
            .expect_err("length above max must be rejected");
        assert_eq!(
            err,
            TranscriptError::BadProofShape {
                reason: "message length exceeds declared maximum",
            }
        );
    }

    #[test]
    fn bounded_scalars_rejects_a_count_the_wire_cannot_hold() {
        // Invariant: a declared count is never trusted for allocation.
        //
        // `actual` is attacker-controlled up to the cap, so a short wire with a
        // large count must be rejected before any `with_capacity`.

        // Fixture state: cap of 200 scalars, so the prefix is one byte wide.
        let pat = InteractionPattern::new(vec![Interaction::algebra::<F, F>(
            Hierarchy::Atomic,
            Kind::Message,
            "msgs",
            Length::Bounded(200),
        )])
        .unwrap();
        let ds: DomainSeparator<u8> = DomainSeparator::new(0, b"msg-greedy", pat);

        // Mutation: declare 200 scalars but supply no payload at all.
        let narg = [200u8];

        let mut v = VerifierState::<_, u8>::new(sponge(), &ds, &narg);
        let err = v
            .next_scalars_bounded::<F, ByteCodec>("msgs", 200)
            .expect_err("an unbacked count must be rejected");
        assert_eq!(
            err,
            TranscriptError::BadProofShape {
                reason: "step declares more values than the wire holds",
            }
        );
    }

    #[test]
    #[should_panic(expected = "Received interaction")]
    fn pattern_mismatch_on_label_panics() {
        // Pattern declares "msg" but the caller asks for "different".
        let ds: DomainSeparator<u8> = DomainSeparator::new(0, b"mismatch", one_msg_pattern());
        let narg = [0u8; 4];
        let mut v = VerifierState::<_, u8>::new(sponge(), &ds, &narg);
        let _ = v.next_scalar::<F, ByteCodec>("different");
    }
}
