//! Steps both drivers play identically.

use alloc::vec::Vec;

use p3_field::{AlgebraIdentity, Field};

use crate::fs::TranscriptField;
use crate::fs::bound::TranscriptBound;
use crate::fs::codecs::Codec;
use crate::fs::pattern::Label;
use crate::fs::state::{ProverState, VerifierState};
use crate::fs::unit::Unit;
use crate::{CanObserve, CanSampleUniformBits};

mod sealed {
    pub trait Sealed {}
}

impl<C, U: Unit> sealed::Sealed for ProverState<C, U> {}
impl<C, U: Unit> sealed::Sealed for VerifierState<'_, C, U> {}

/// Steps a prover and a verifier play identically.
///
/// Each one touches the sponge and the recorded pattern, never the wire.
/// A body written against this trait therefore plays either side in lockstep.
///
/// Absorbing a proof message, grinding, and checking a witness differ per side.
/// Those stay on the concrete drivers.
///
/// Every step forwards to the inherent method of the same name on either driver.
/// [`ProverState`] and [`VerifierState`] carry what each step records and what it leaves out.
pub trait SymmetricSteps: sealed::Sealed {
    /// Sponge the driver absorbs into and samples from.
    type Challenger;

    /// Open a mixed container that accepts nested steps of any kind.
    fn begin_protocol<T: ?Sized>(&mut self, label: Label);

    /// Close a mixed container.
    fn end_protocol<T: ?Sized>(&mut self, label: Label);

    /// Absorb a value the challenger encodes itself, carried by the caller's own proof.
    fn observe_opaque<T>(&mut self, label: Label, value: T) -> TranscriptBound<T>
    where
        T: Clone,
        Self::Challenger: CanObserve<T>;

    /// Absorb one extension-field message the caller's own proof carries.
    fn observe_extension<F, EF, Cdc>(&mut self, label: Label, value: &EF) -> TranscriptBound<EF>
    where
        F: TranscriptField,
        EF: Field + AlgebraIdentity<F>,
        Cdc: Codec<Self::Challenger, F>;

    /// Sample one extension-field challenge.
    fn challenge_extension<F, EF, Cdc>(&mut self, label: Label) -> TranscriptBound<EF>
    where
        F: TranscriptField,
        EF: Field + AlgebraIdentity<F>,
        Cdc: Codec<Self::Challenger, F>;

    /// Sample `count` extension challenges the predicate accepts, under one step.
    ///
    /// [`ProverState::challenge_extensions_rejecting`] carries what the step does not record.
    fn challenge_extensions_rejecting<F, EF, Cdc>(
        &mut self,
        label: Label,
        count: usize,
        accept: impl FnMut(&EF, &[EF]) -> bool,
    ) -> Vec<TranscriptBound<EF>>
    where
        F: TranscriptField,
        EF: Field + AlgebraIdentity<F>,
        Cdc: Codec<Self::Challenger, F>;

    /// Sample `count` challenges of `width` bits with no modular bias, under one step.
    ///
    /// [`ProverState::challenge_uniform_bits`] carries why this draw differs from the biased one.
    fn challenge_uniform_bits<W>(
        &mut self,
        label: Label,
        width: usize,
        count: usize,
    ) -> Vec<TranscriptBound<usize>>
    where
        Self::Challenger: CanSampleUniformBits<W>;
}

impl<C, U: Unit> SymmetricSteps for ProverState<C, U> {
    type Challenger = C;

    #[inline]
    fn begin_protocol<T: ?Sized>(&mut self, label: Label) {
        Self::begin_protocol::<T>(self, label);
    }

    #[inline]
    fn end_protocol<T: ?Sized>(&mut self, label: Label) {
        Self::end_protocol::<T>(self, label);
    }

    #[inline]
    fn observe_opaque<T>(&mut self, label: Label, value: T) -> TranscriptBound<T>
    where
        T: Clone,
        C: CanObserve<T>,
    {
        Self::observe_opaque::<T>(self, label, value)
    }

    #[inline]
    fn observe_extension<F, EF, Cdc>(&mut self, label: Label, value: &EF) -> TranscriptBound<EF>
    where
        F: TranscriptField,
        EF: Field + AlgebraIdentity<F>,
        Cdc: Codec<C, F>,
    {
        Self::observe_extension::<F, EF, Cdc>(self, label, value)
    }

    #[inline]
    fn challenge_extension<F, EF, Cdc>(&mut self, label: Label) -> TranscriptBound<EF>
    where
        F: TranscriptField,
        EF: Field + AlgebraIdentity<F>,
        Cdc: Codec<C, F>,
    {
        Self::challenge_extension::<F, EF, Cdc>(self, label)
    }

    #[inline]
    fn challenge_extensions_rejecting<F, EF, Cdc>(
        &mut self,
        label: Label,
        count: usize,
        accept: impl FnMut(&EF, &[EF]) -> bool,
    ) -> Vec<TranscriptBound<EF>>
    where
        F: TranscriptField,
        EF: Field + AlgebraIdentity<F>,
        Cdc: Codec<C, F>,
    {
        Self::challenge_extensions_rejecting::<F, EF, Cdc>(self, label, count, accept)
    }

    #[inline]
    fn challenge_uniform_bits<W>(
        &mut self,
        label: Label,
        width: usize,
        count: usize,
    ) -> Vec<TranscriptBound<usize>>
    where
        C: CanSampleUniformBits<W>,
    {
        Self::challenge_uniform_bits::<W>(self, label, width, count)
    }
}

impl<C, U: Unit> SymmetricSteps for VerifierState<'_, C, U> {
    type Challenger = C;

    #[inline]
    fn begin_protocol<T: ?Sized>(&mut self, label: Label) {
        Self::begin_protocol::<T>(self, label);
    }

    #[inline]
    fn end_protocol<T: ?Sized>(&mut self, label: Label) {
        Self::end_protocol::<T>(self, label);
    }

    #[inline]
    fn observe_opaque<T>(&mut self, label: Label, value: T) -> TranscriptBound<T>
    where
        T: Clone,
        C: CanObserve<T>,
    {
        Self::observe_opaque::<T>(self, label, value)
    }

    #[inline]
    fn observe_extension<F, EF, Cdc>(&mut self, label: Label, value: &EF) -> TranscriptBound<EF>
    where
        F: TranscriptField,
        EF: Field + AlgebraIdentity<F>,
        Cdc: Codec<C, F>,
    {
        Self::observe_extension::<F, EF, Cdc>(self, label, value)
    }

    #[inline]
    fn challenge_extension<F, EF, Cdc>(&mut self, label: Label) -> TranscriptBound<EF>
    where
        F: TranscriptField,
        EF: Field + AlgebraIdentity<F>,
        Cdc: Codec<C, F>,
    {
        Self::challenge_extension::<F, EF, Cdc>(self, label)
    }

    #[inline]
    fn challenge_extensions_rejecting<F, EF, Cdc>(
        &mut self,
        label: Label,
        count: usize,
        accept: impl FnMut(&EF, &[EF]) -> bool,
    ) -> Vec<TranscriptBound<EF>>
    where
        F: TranscriptField,
        EF: Field + AlgebraIdentity<F>,
        Cdc: Codec<C, F>,
    {
        Self::challenge_extensions_rejecting::<F, EF, Cdc>(self, label, count, accept)
    }

    #[inline]
    fn challenge_uniform_bits<W>(
        &mut self,
        label: Label,
        width: usize,
        count: usize,
    ) -> Vec<TranscriptBound<usize>>
    where
        C: CanSampleUniformBits<W>,
    {
        Self::challenge_uniform_bits::<W>(self, label, width, count)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_keccak::Keccak256Hash;

    use super::SymmetricSteps;
    use crate::fs::TranscriptBound;
    use crate::fs::codecs::FieldToFieldCodec;
    use crate::fs::domain_separator::DomainSeparator;
    use crate::fs::pattern::{Hierarchy, Interaction, Kind, Length, Pattern, PatternState};
    use crate::fs::state::{ProverState, VerifierState};
    use crate::fs::unit::FieldUnit;
    use crate::{
        CanObserve, CanSample, CanSampleUniformBits, HashChallenger, SerializingChallenger32,
    };

    /// Concrete field exercised in this module's tests.
    type F = BabyBear;
    /// Degree-4 binomial extension over `F`.
    type EF4 = BinomialExtensionField<F, 4>;
    /// Field codec used with a native field sponge.
    type NativeCodec = FieldToFieldCodec<F>;

    /// Opaque value the block binds first.
    const COMMITMENT: u32 = 7;
    /// Extension message the block binds before any challenge.
    const VALUE: u32 = 9;

    /// Production field sponge: the Keccak byte sponge lifted to `F` elements.
    fn field_sponge() -> SerializingChallenger32<F, HashChallenger<u8, Keccak256Hash, 32>> {
        SerializingChallenger32::from_hasher(Vec::new(), Keccak256Hash)
    }

    /// One block holding every step the trait exposes, in the order `play` walks it.
    fn block_separator() -> DomainSeparator<FieldUnit<F>> {
        let mut recorder = PatternState::<FieldUnit<F>>::new();
        recorder.begin_protocol::<()>("block");
        recorder.interact(Interaction::opaque(
            Hierarchy::Atomic,
            Kind::Message,
            "commitment",
            Length::Scalar,
        ));
        recorder.interact(Interaction::algebra::<F, EF4>(
            Hierarchy::Atomic,
            Kind::Message,
            "value",
            Length::Scalar,
        ));
        recorder.interact(Interaction::algebra::<F, EF4>(
            Hierarchy::Atomic,
            Kind::Challenge,
            "point",
            Length::Scalar,
        ));
        recorder.interact(Interaction::algebra::<F, EF4>(
            Hierarchy::Atomic,
            Kind::Challenge,
            "points",
            Length::Fixed(2),
        ));
        recorder.interact(Interaction::uniform_bits(
            Hierarchy::Atomic,
            Kind::Challenge,
            "indices",
            4,
            Length::Fixed(3),
        ));
        recorder.end_protocol::<()>("block");
        DomainSeparator::new(0, b"symmetric-steps", recorder.finalize())
    }

    /// Walk the block through either driver.
    ///
    /// # Returns
    ///
    /// - The single point.
    /// - Two points, distinct from the single one and from each other.
    /// - Three indices.
    fn play<S>(state: &mut S) -> (EF4, Vec<EF4>, Vec<usize>)
    where
        S: SymmetricSteps,
        S::Challenger: CanObserve<F> + CanSample<F> + CanSampleUniformBits<F>,
    {
        state.begin_protocol::<()>("block");
        state.observe_opaque("commitment", F::from_u32(COMMITMENT));
        state.observe_extension::<F, EF4, NativeCodec>("value", &EF4::from_u32(VALUE));
        let point = state
            .challenge_extension::<F, EF4, NativeCodec>("point")
            .into_inner();
        let points = state
            .challenge_extensions_rejecting::<F, EF4, NativeCodec>(
                "points",
                2,
                |candidate, kept| *candidate != point && !kept.contains(candidate),
            )
            .into_iter()
            .map(TranscriptBound::into_inner)
            .collect();
        let indices = state
            .challenge_uniform_bits::<F>("indices", 4, 3)
            .into_iter()
            .map(TranscriptBound::into_inner)
            .collect();
        state.end_protocol::<()>("block");
        (point, points, indices)
    }

    #[test]
    fn both_drivers_play_one_body_to_one_stream() {
        // Invariant: one body written against the trait replays on either side.
        //
        // Nothing in the block touches the wire, so the verifier reads an empty one.
        let ds = block_separator();

        let mut prover_sponge = field_sponge();
        let mut prover = ProverState::new(&mut prover_sponge, &ds);
        let proved = play(&mut prover);
        assert!(prover.finalize().is_empty());

        let mut verifier_sponge = field_sponge();
        let mut verifier = VerifierState::new(&mut verifier_sponge, &ds, &[]);
        let replayed = play(&mut verifier);
        verifier
            .finalize()
            .expect("the block writes nothing, so nothing is left unread");

        // Both sides drew the same values and handed back the same sponge.
        assert_eq!(proved, replayed);
        let next: F = prover_sponge.sample();
        assert_eq!(next, verifier_sponge.sample());

        // The predicate reached the rejecting draw.
        let (point, points, _) = proved;
        assert!(!points.contains(&point));
        assert_ne!(points[0], points[1]);
    }

    #[test]
    fn the_trait_draws_what_the_inherent_methods_draw() {
        // Invariant: every step forwards to the inherent method of the same name.
        //
        // A forwarder that resolved to itself would overflow the stack here instead.
        let ds = block_separator();

        let mut through_trait = field_sponge();
        let mut state = ProverState::new(&mut through_trait, &ds);
        let played = play(&mut state);
        assert!(state.finalize().is_empty());

        // The same walk, spelled with inherent calls on the concrete driver.
        let mut inherent = field_sponge();
        let mut state = ProverState::new(&mut inherent, &ds);
        state.begin_protocol::<()>("block");
        state.observe_opaque("commitment", F::from_u32(COMMITMENT));
        state.observe_extension::<F, EF4, NativeCodec>("value", &EF4::from_u32(VALUE));
        let point = state
            .challenge_extension::<F, EF4, NativeCodec>("point")
            .into_inner();
        let points: Vec<EF4> = state
            .challenge_extensions_rejecting::<F, EF4, NativeCodec>(
                "points",
                2,
                |candidate, kept| *candidate != point && !kept.contains(candidate),
            )
            .into_iter()
            .map(TranscriptBound::into_inner)
            .collect();
        let indices: Vec<usize> = state
            .challenge_uniform_bits::<F>("indices", 4, 3)
            .into_iter()
            .map(TranscriptBound::into_inner)
            .collect();
        state.end_protocol::<()>("block");
        assert!(state.finalize().is_empty());

        assert_eq!(played, (point, points, indices));
        let next: F = through_trait.sample();
        assert_eq!(next, inherent.sample());
    }
}
