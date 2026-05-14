//! Verifier-side driver.

use alloc::vec::Vec;

use p3_field::{BasedVectorSpace, PrimeField};

use crate::fs::bound::TranscriptBound;
use crate::fs::codecs::Codec;
use crate::fs::codecs::decode_field::{
    decode_field_be_canonical, encode_field_be, field_byte_size,
};
use crate::fs::codecs::length_prefix::{bound_byte_width, decode_len_be};
use crate::fs::domain_separator::DomainSeparator;
use crate::fs::error::TranscriptError;
use crate::fs::pattern::{Hierarchy, Interaction, Kind, Label, Length, Pattern, PatternPlayer};
use crate::fs::unit::Unit;
use crate::{CanObserve, GrindingChallenger};

/// Drives a verifier-side transcript in lockstep with a recorded pattern.
///
/// Reads bytes from a caller-supplied slice through a cursor — no copies.
///
/// Wire-format problems return a structured error.
///
/// Pattern misuse panics with a diff message.
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
    _u: core::marker::PhantomData<U>,
}

impl<C, U: Unit> Drop for VerifierState<'_, C, U> {
    fn drop(&mut self) {
        // Abort the player on drop when the user did not finalise
        //
        // This way, error paths do not turn into double-panics during cleanup.
        if !self.player.is_finalized() {
            self.player.abort();
        }
    }
}

impl<'a, C, U: Unit> VerifierState<'a, C, U> {
    /// Build a driver and seed the challenger from the domain separator.
    pub fn new(mut challenger: C, ds: &DomainSeparator<U>, narg: &'a [u8]) -> Self
    where
        C: CanObserve<u8>,
    {
        // Seed identically to the prover so both sides land on the same sponge state.
        ds.seed_bytes(&mut challenger);
        let player = PatternPlayer::new(ds.pattern().clone());
        Self {
            challenger,
            player,
            narg,
            cursor: 0,
            _u: core::marker::PhantomData,
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
    pub fn finalize(self) -> Result<(), TranscriptError> {
        // Move fields out by hand: the wrapper's Drop must not run.
        let this = core::mem::ManuallyDrop::new(self);
        // SAFETY: each field is moved out exactly once and Drop never runs on `this`.
        let player = unsafe { core::ptr::read(&this.player) };
        let cursor = this.cursor;
        let narg_len = this.narg.len();
        let challenger = unsafe { core::ptr::read(&this.challenger) };
        drop(challenger);
        // Pattern check: every recorded step must have been replayed.
        player.finalize();
        // Wire check: trailing bytes mean the prover smuggled data the verifier never read.
        if cursor != narg_len {
            return Err(TranscriptError::BadProofShape {
                reason: "trailing NARG bytes after final verifier step",
            });
        }
        Ok(())
    }

    /// Take `n` raw bytes from the wire cursor, or fail if out of bounds.
    fn take_bytes(&mut self, n: usize) -> Result<&'a [u8], TranscriptError> {
        if self.cursor + n > self.narg.len() {
            return Err(TranscriptError::BadProofShape {
                reason: "NARG ended before all expected bytes were read",
            });
        }
        let slice = &self.narg[self.cursor..self.cursor + n];
        self.cursor += n;
        Ok(slice)
    }

    /// Replay a salt step by reading `byte_len` bytes from the wire.
    pub fn next_salt(&mut self, byte_len: usize) -> Result<Vec<u8>, TranscriptError>
    where
        C: CanObserve<u8>,
    {
        // The verifier must know the length up front:
        //
        // Reading it from the wire would let an attacker control how much data is consumed.
        self.player.interact(Interaction::new::<u8>(
            Hierarchy::Atomic,
            Kind::Salt,
            "salt",
            Length::Fixed(byte_len),
        ));
        let bytes = self.take_bytes(byte_len)?.to_vec();
        // Absorb so future samples depend on the prover's salt.
        self.challenger.observe_slice(&bytes);
        Ok(bytes)
    }

    /// Replay an `add_scalar` step from the prover.
    pub fn next_scalar<F, Cdc>(
        &mut self,
        label: Label,
    ) -> Result<TranscriptBound<F>, TranscriptError>
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
        // Read the canonical big-endian encoding from the wire.
        let need = field_byte_size::<F>();
        let raw = self.take_bytes(need)?;
        let value = decode_field_be_canonical::<F>(raw)?;
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
        F: PrimeField,
        Cdc: Codec<C, F>,
    {
        // Validate: the next pattern step is a fixed-length list of `n` scalars.
        self.player.interact(Interaction::new::<F>(
            Hierarchy::Atomic,
            Kind::Message,
            label,
            Length::Fixed(n),
        ));
        // Pull `n` canonical encodings from the wire, decoding, absorbing, and binding each.
        let need = field_byte_size::<F>();
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let raw = self.take_bytes(need)?;
            let v = decode_field_be_canonical::<F>(raw)?;
            Cdc::observe(&mut self.challenger, &v);
            out.push(TranscriptBound::wrap(v));
        }
        Ok(out)
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
        F: PrimeField,
        C: CanObserve<u8>,
        Cdc: Codec<C, F>,
    {
        // Validate against the recorded pattern step so shape divergence panics here.
        self.player.interact(Interaction::new::<F>(
            Hierarchy::Atomic,
            Kind::Message,
            label,
            Length::Bounded(max),
        ));
        // Prefix width is deterministic on both sides from the recorded bound.
        let width = bound_byte_width(max);
        // Copy the prefix out of the wire so the absorb call below does not borrow `self.narg`.
        let len_bytes: alloc::vec::Vec<u8> = self.take_bytes(width)?.to_vec();
        let actual = decode_len_be(&len_bytes, width);
        // A wire length above the cap is malformed input, not a panic-worthy bug.
        if actual > max {
            return Err(TranscriptError::BadProofShape {
                reason: "message length exceeds declared maximum",
            });
        }
        // Bind the count into the sponge before any value enters it.
        //
        // This keeps the transcript prefix-free, matching CO25 §6.2.
        self.challenger.observe_slice(&len_bytes);
        // One canonical field encoding per scalar.
        let need = field_byte_size::<F>();
        let mut out = Vec::with_capacity(actual);
        for _ in 0..actual {
            // Pull one scalar's bytes off the wire and reject non-canonical encodings.
            let raw = self.take_bytes(need)?;
            let v = decode_field_be_canonical::<F>(raw)?;
            // Absorb through the same codec the prover used so both sides agree.
            Cdc::observe(&mut self.challenger, &v);
            // Hand back a binding witness alongside the decoded value.
            out.push(TranscriptBound::wrap(v));
        }
        Ok(out)
    }

    /// Replay an `add_extension` step from the prover.
    pub fn next_extension<F, EF, Cdc>(
        &mut self,
        label: Label,
    ) -> Result<TranscriptBound<EF>, TranscriptError>
    where
        F: PrimeField,
        EF: BasedVectorSpace<F>,
        Cdc: Codec<C, F>,
    {
        // Validate: the next pattern step is a scalar message of extension type `EF`.
        self.player.interact(Interaction::new::<EF>(
            Hierarchy::Atomic,
            Kind::Message,
            label,
            Length::Scalar,
        ));
        // Read one base-field coefficient per basis index, decoding and absorbing each.
        let need = field_byte_size::<F>();
        let basis_len = EF::DIMENSION;
        let mut coeffs: Vec<F> = Vec::with_capacity(basis_len);
        for _ in 0..basis_len {
            let raw = self.take_bytes(need)?;
            let v = decode_field_be_canonical::<F>(raw)?;
            Cdc::observe(&mut self.challenger, &v);
            coeffs.push(v);
        }
        // Reconstruct in the same basis order the prover used.
        let value = EF::from_basis_coefficients_iter(coeffs.into_iter()).ok_or(
            TranscriptError::BadProofShape {
                reason: "extension element basis size mismatch",
            },
        )?;
        Ok(TranscriptBound::wrap(value))
    }

    /// Replay an `add_hint` step.
    ///
    /// Hint bytes are returned to the caller; they are never absorbed.
    pub fn next_hint(
        &mut self,
        label: Label,
        byte_len: usize,
    ) -> Result<&'a [u8], TranscriptError> {
        self.player.interact(Interaction::new::<u8>(
            Hierarchy::Atomic,
            Kind::Hint,
            label,
            Length::Fixed(byte_len),
        ));
        self.take_bytes(byte_len)
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
        self.player.interact(Interaction::new::<u8>(
            Hierarchy::Atomic,
            Kind::Hint,
            label,
            Length::Bounded(max),
        ));
        // Prefix width is deterministic on both sides from the recorded bound.
        let width = bound_byte_width(max);
        // Pull the prefix off the wire and decode it.
        let len_bytes = self.take_bytes(width)?;
        let actual = decode_len_be(len_bytes, width);
        // A wire length above the cap is malformed input, not a panic-worthy bug.
        if actual > max {
            return Err(TranscriptError::BadProofShape {
                reason: "hint length exceeds declared maximum",
            });
        }
        // Hand back the payload as a borrowed slice — no sponge absorption.
        self.take_bytes(actual)
    }

    /// Sample one challenge scalar in lockstep with the prover.
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

    /// Sample `n` challenge scalars in lockstep with the prover.
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

    /// Sample one challenge extension-field element in lockstep with the prover.
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

    /// Replay a proof-of-work step.
    ///
    /// - Reads the witness from the wire,
    /// - Checks its encoding is canonical,
    /// - Absorbs it through the challenger's PoW path.
    pub fn check_pow(&mut self, bits: usize) -> Result<(), TranscriptError>
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
        // Read the witness from the wire and decode it as a canonical field element.
        let need = field_byte_size::<<C as GrindingChallenger>::Witness>();
        let raw = self.take_bytes(need)?;
        let witness = decode_field_be_canonical::<<C as GrindingChallenger>::Witness>(raw)?;
        // Re-encode and compare so a malicious prover cannot smuggle a non-canonical encoding.
        let canonical = encode_field_be::<<C as GrindingChallenger>::Witness>(&witness);
        if canonical.as_slice() != raw {
            return Err(TranscriptError::BadProofShape {
                reason: "pow witness encoding is non-canonical",
            });
        }
        // Verify the witness produces the required number of leading zero bits.
        if !self.challenger.check_witness(bits, witness) {
            return Err(TranscriptError::BadProofShape {
                reason: "pow witness does not produce enough zero bits",
            });
        }
        Ok(())
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
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::fs::codecs::BytesToFieldCodec;
    use crate::fs::pattern::InteractionPattern;
    use crate::fs::shake128::Shake128;

    /// Concrete field exercised in this module's tests.
    type F = BabyBear;

    fn one_msg_pattern() -> InteractionPattern {
        InteractionPattern::new(vec![Interaction::new::<F>(
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
        let pat = one_msg_pattern();
        let mut ds: DomainSeparator<u8> = DomainSeparator::new(0, b"trunc", pat);
        ds.bind_pattern_hash();
        let narg = [0u8; 1];
        let mut v = VerifierState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds, &narg);
        let err = v
            .next_scalar::<F, BytesToFieldCodec<F>>("msg")
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
        let pat = one_msg_pattern();
        let mut ds: DomainSeparator<u8> = DomainSeparator::new(0, b"non-canon", pat);
        ds.bind_pattern_hash();
        let narg = [0xffu8; 4];
        let mut v = VerifierState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds, &narg);
        let err = v
            .next_scalar::<F, BytesToFieldCodec<F>>("msg")
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
        let pat = one_msg_pattern();
        let mut ds: DomainSeparator<u8> = DomainSeparator::new(0, b"trailing", pat);
        ds.bind_pattern_hash();

        // Prover writes a valid NARG, then we smuggle one extra byte at the tail.
        use crate::fs::state::ProverState;
        let mut p = ProverState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds);
        p.add_scalar::<F, BytesToFieldCodec<F>>("msg", &F::from_u32(7u32));
        let mut narg = p.finalize();
        narg.push(0x42);

        // Verifier consumes the legal scalar, then finalize must reject the leftover byte.
        let mut v = VerifierState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds, &narg);
        let _ = v
            .next_scalar::<F, BytesToFieldCodec<F>>("msg")
            .expect("legal scalar");
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
    fn bounded_hint_rejects_length_above_max() {
        // Invariant: a wire length above the recorded cap is malformed input.
        //
        // The verifier rejects it with a structured error rather than panicking.

        // Fixture state: hint cap of 4 bytes.
        let pat = InteractionPattern::new(vec![Interaction::new::<u8>(
            Hierarchy::Atomic,
            Kind::Hint,
            "auth",
            Length::Bounded(4),
        )])
        .unwrap();
        let mut ds: DomainSeparator<u8> = DomainSeparator::new(0, b"hint-oob", pat);
        ds.bind_pattern_hash();

        // Mutation: hand-craft a wire frame whose prefix declares 5 bytes.
        //
        // ```text
        //     cap = 4 → prefix width = 1 byte
        //     wire   = [0x05, .., .., .., .., .., ..]
        //                ^^^^ declared count above cap
        // ```
        let narg = [5u8, 0, 0, 0, 0, 0, 0];

        // The verifier sees the over-cap prefix and surfaces a structured error.
        let mut v = VerifierState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds, &narg);
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
        let pat = InteractionPattern::new(vec![Interaction::new::<u8>(
            Hierarchy::Atomic,
            Kind::Hint,
            "auth",
            Length::Bounded(8),
        )])
        .unwrap();
        let mut ds: DomainSeparator<u8> = DomainSeparator::new(0, b"hint-trunc", pat);
        ds.bind_pattern_hash();

        // Mutation: declare 7 payload bytes but supply only 2.
        //
        // ```text
        //     wire = [0x07, 0xaa, 0xbb]
        //              ^^^^ declared
        //                    ^^^^^^^^^^ only 2 bytes follow
        // ```
        let narg = [7u8, 0xaa, 0xbb];

        // The verifier runs out of bytes mid-payload and reports a malformed wire.
        let mut v = VerifierState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds, &narg);
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
        let pat = InteractionPattern::new(vec![Interaction::new::<F>(
            Hierarchy::Atomic,
            Kind::Message,
            "msgs",
            Length::Bounded(2),
        )])
        .unwrap();
        let mut ds: DomainSeparator<u8> = DomainSeparator::new(0, b"msg-oob", pat);
        ds.bind_pattern_hash();

        // Mutation: declare 3 scalars where the cap is 2.
        //
        // ```text
        //     cap = 2 → prefix width = 1 byte
        //     wire   = [0x03]
        //               ^^^^ declared count above cap
        // ```
        let narg = [3u8];

        // The verifier surfaces a structured error before touching the sponge.
        let mut v = VerifierState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds, &narg);
        let err = v
            .next_scalars_bounded::<F, BytesToFieldCodec<F>>("msgs", 2)
            .expect_err("length above max must be rejected");
        assert_eq!(
            err,
            TranscriptError::BadProofShape {
                reason: "message length exceeds declared maximum",
            }
        );
    }

    #[test]
    #[should_panic(expected = "Received interaction")]
    fn pattern_mismatch_on_label_panics() {
        // Pattern declares "msg" but the caller asks for "different".
        let pat = one_msg_pattern();
        let mut ds: DomainSeparator<u8> = DomainSeparator::new(0, b"mismatch", pat);
        ds.bind_pattern_hash();
        let narg = [0u8; 4];
        let mut v = VerifierState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds, &narg);
        let _ = v.next_scalar::<F, BytesToFieldCodec<F>>("different");
    }
}
