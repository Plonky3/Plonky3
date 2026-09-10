//! Optional PCS grinding after the opening claims and before their batching challenge.
//!
//! A zero-bit site preserves the existing transcript. An enabled site binds its name,
//! difficulty and challenge field through the typed Fiat-Shamir layer.

use alloc::vec;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, VerifierState,
};
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, PrimeField64};

use crate::error::{ProofShapeError, StirError};

const BATCH_POW: &str = "batch_pow";
const ALPHA: &str = "alpha";

/// Domain separator for an enabled PCS batching grind.
///
/// At zero difficulty the PCS skips this transcript and samples alpha directly.
pub fn batch_domain_separator<F: PrimeField64, EF: ExtensionField<F>>(
    bits: usize,
) -> DomainSeparator<FieldUnit<F>> {
    let pattern = InteractionPattern::new(vec![
        Interaction::algebra::<F, F>(Hierarchy::Atomic, Kind::Pow, BATCH_POW, Length::Fixed(bits)),
        Interaction::algebra::<F, EF>(Hierarchy::Atomic, Kind::Challenge, ALPHA, Length::Scalar),
    ])
    .expect("the batching phase has no nested steps");
    DomainSeparator::new(1, b"p3-stir-pcs-batch", pattern)
}

pub(crate) fn prove<F, EF, C>(challenger: &mut C, bits: usize) -> (EF, Option<F>)
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    C: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    if bits == 0 {
        return (challenger.sample_algebra_element(), None);
    }
    let mut state = ProverState::new(challenger, &batch_domain_separator::<F, EF>(bits));
    let witness = state.observe_pow(BATCH_POW, bits);
    let alpha = state
        .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ALPHA)
        .into_inner();
    assert!(state.finalize().is_empty(), "the PCS carries the witness");
    (alpha, Some(witness))
}

pub(crate) fn verify<F, EF, C, E, IE>(
    challenger: &mut C,
    bits: usize,
    witness: Option<F>,
) -> Result<EF, StirError<E, IE>>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    C: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    if witness.is_some() != (bits > 0) {
        return Err(ProofShapeError::BatchPowWitness {
            expected: bits > 0,
            got: witness.is_some(),
        }
        .into());
    }
    let Some(witness) = witness else {
        return Ok(challenger.sample_algebra_element());
    };
    let mut state = VerifierState::new(challenger, &batch_domain_separator::<F, EF>(bits), &[]);
    state
        .observe_pow(BATCH_POW, bits, witness)
        .map_err(|_| StirError::InvalidBatchPowWitness { bits })?;
    let alpha = state
        .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ALPHA)
        .into_inner();
    state
        .finalize()
        .expect("the batching phase reads an empty wire");
    Ok(alpha)
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::{CanObserve, DuplexChallenger};
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Ch = DuplexChallenger<F, Poseidon2BabyBear<16>, 16, 8>;

    fn challenger() -> Ch {
        Ch::new(Poseidon2BabyBear::new_from_rng_128(
            &mut SmallRng::seed_from_u64(12),
        ))
    }

    #[test]
    fn zero_bits_preserve_the_bare_batching_challenge_and_transcript() {
        let mut bare = challenger();
        let mut prover = bare.clone();
        let mut verifier = bare.clone();
        let expected: EF = bare.sample_algebra_element();
        let (alpha, witness) = prove::<F, EF, _>(&mut prover, 0);
        assert_eq!(alpha, expected);
        assert_eq!(witness, None);
        assert_eq!(
            verify::<F, EF, _, (), ()>(&mut verifier, 0, None),
            Ok(expected)
        );
        let next: EF = bare.sample_algebra_element();
        assert_eq!(prover.sample_algebra_element::<EF>(), next);
        assert_eq!(verifier.sample_algebra_element::<EF>(), next);
    }

    #[test]
    fn a_witness_meeting_only_a_lower_difficulty_is_rejected() {
        let mut base = challenger();
        base.observe(F::from_u64(123));
        let mut seeded = base.clone();
        batch_domain_separator::<F, EF>(8).seed(&mut seeded);
        let weak = (0..1024)
            .map(F::from_u64)
            .find(|&witness| {
                seeded.clone().check_witness(1, witness)
                    && !seeded.clone().check_witness(8, witness)
            })
            .expect("a witness clearing one bit but fewer than eight exists");
        assert_eq!(
            verify::<F, EF, _, (), ()>(&mut base, 8, Some(weak)),
            Err(StirError::InvalidBatchPowWitness { bits: 8 }),
        );
    }
}
