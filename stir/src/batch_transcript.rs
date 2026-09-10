//! Optional grinding after the opening claims and before their batching challenge.
//!
//! # Overview
//!
//! The batching challenge folds every opening claim of a batch into a single claim.
//!
//! It is drawn after all of them.
//!
//! Grinding in front of it raises the cost of steering it.
//!
//! # Shape
//!
//! ```text
//!     difficulty 0  ->  [ challenge ]
//!     difficulty b  ->  [ grind at b bits, challenge ]
//! ```
//!
//! # What the shape binds
//!
//! A fingerprint of the description enters the sponge before either step runs.
//!
//! The difficulty travels inside the grinding step, so the fingerprint covers it.
//!
//! Two sites at different difficulties therefore cannot share a transcript.
//!
//! An ungrounded site shares one with no ground site either.
//!
//! Its description omits the step entirely.

use alloc::vec::Vec;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, VerifierState,
};
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, PrimeField64};

use crate::error::{ProofShapeError, StirError};

/// Protocol name bound into this phase's transcript seed.
///
/// The security model keys its grinding table on this name.
const NAME: &[u8] = b"p3-stir-pcs-batch";

/// Step label of the grinding that guards the batching challenge.
const BATCH_POW: &str = "batch_pow";

/// Step label of the batching challenge itself.
const ALPHA: &str = "alpha";

/// Describe the batching phase at one difficulty.
///
/// # Arguments
///
/// - `bits`: grinding difficulty guarding the challenge, or zero to omit grinding.
fn separator<F: PrimeField64, EF: ExtensionField<F>>(bits: usize) -> DomainSeparator<FieldUnit<F>> {
    // The challenge is the only step every site plays, and grinding may precede it.
    let mut steps = Vec::with_capacity(2);

    // Grinding sits ahead of the challenge it protects.
    //
    // A zero difficulty describes no work.
    //
    // It contributes no step, not an empty one.
    if bits > 0 {
        steps.push(Interaction::algebra::<F, F>(
            Hierarchy::Atomic,
            Kind::Pow,
            BATCH_POW,
            Length::Fixed(bits),
        ));
    }

    // Every opening claim is already absorbed by the time this challenge is drawn.
    steps.push(Interaction::algebra::<F, EF>(
        Hierarchy::Atomic,
        Kind::Challenge,
        ALPHA,
        Length::Scalar,
    ));

    // A separator exists for every difficulty, zero included.
    //
    // Both drivers seed from it.
    //
    // No site draws from a bare sponge.
    DomainSeparator::new(
        1,
        NAME,
        InteractionPattern::new(steps)
            .expect("a flat sequence of leaf steps is always well formed"),
    )
}

/// Draw the batching challenge, grinding first when the difficulty is positive.
///
/// # Arguments
///
/// - `challenger`: sponge of the surrounding protocol, borrowed for the phase.
/// - `bits`: grinding difficulty guarding the challenge, or zero to omit grinding.
///
/// # Returns
///
/// The batching challenge, and the grinding witness when one was demanded.
pub(crate) fn prove<F, EF, C>(challenger: &mut C, bits: usize) -> (EF, Option<F>)
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    C: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // Seeding is unconditional.
    //
    // The challenge is bound to this phase at every difficulty.
    let mut state = ProverState::new(challenger, &separator::<F, EF>(bits));

    // A zero difficulty describes no grinding step, so there is no witness to search for.
    let witness = (bits > 0).then(|| state.observe_pow(BATCH_POW, bits));

    // The challenge lands downstream of the seed, and of the grind when there is one.
    let alpha = state
        .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ALPHA)
        .into_inner();

    // Nothing here writes to the wire.
    //
    // The witness travels in the proof the caller assembles.
    assert!(state.finalize().is_empty(), "the PCS carries the witness");

    (alpha, witness)
}

/// Replay the batching phase and recover the same challenge the prover drew.
///
/// # Arguments
///
/// - `challenger`: sponge of the surrounding protocol, borrowed for the phase.
/// - `bits`: grinding difficulty the configuration demands, or zero to omit grinding.
/// - `witness`: grinding witness from the proof, absent when none is demanded.
///
/// # Returns
///
/// The batching challenge, once the witness has been accepted.
///
/// # Errors
///
/// When the witness presence disagrees with the demanded difficulty.
///
/// When the witness fails to clear that difficulty.
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
    // Whether a witness exists is fixed by configuration, never read from the proof.
    //
    // A proof disagreeing with it is malformed, and is rejected before the sponge moves.
    if witness.is_some() != (bits > 0) {
        return Err(ProofShapeError::BatchPowWitness {
            expected: bits > 0,
            got: witness.is_some(),
        }
        .into());
    }

    // The verifier seeds from the same difficulty, so both sides describe one phase.
    let mut state = VerifierState::new(challenger, &separator::<F, EF>(bits), &[]);

    // Only a described grinding step is replayed, matching the prover's own branch.
    if let Some(witness) = witness {
        state
            .observe_pow(BATCH_POW, bits, witness)
            .map_err(|_| StirError::InvalidBatchPowWitness { bits })?;
    }

    let alpha = state
        .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ALPHA)
        .into_inner();

    // The phase reads nothing from the wire.
    //
    // Finalisation therefore depends on code, not on input.
    state
        .finalize()
        .expect("the batching phase reads an empty wire");

    Ok(alpha)
}

#[cfg(test)]
mod tests {
    use core::str::from_utf8;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::fs::PROTOCOL_ID_LEN;
    use p3_challenger::testing::{assert_seeds_pairwise_distinct, pow_difficulties, seed_digest};
    use p3_challenger::{CanObserve, DuplexChallenger};
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_security::GrindingSites;
    use p3_security::grinding::{
        GRINDING_VOCABULARY, GrindingBudget, GrindingSite, RecordedGrind, ZeroBitConvention,
        grinding_step,
    };
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
    fn an_ungrounded_site_still_seeds_its_phase() {
        // Invariant: the batching challenge is bound to this phase at every difficulty.
        //
        // Fixture state: one sponge, cloned three ways from the same state.
        //
        //     bare     : sample straight off the sponge, no seed
        //     prover   : seeds at difficulty 0, then samples
        //     verifier : seeds at difficulty 0, then samples
        let mut bare = challenger();
        let mut prover = bare.clone();
        let mut verifier = bare.clone();

        // A sponge that absorbed no seed lands somewhere the seeded phase must not reach.
        let unbound: EF = bare.sample_algebra_element();

        // Zero difficulty demands no witness.
        //
        // The challenge is still drawn through the driver.
        let (alpha, witness) = prove::<F, EF, _>(&mut prover, 0);
        assert_eq!(witness, None);
        assert_ne!(alpha, unbound);

        // Both sides describe the same phase, so both recover the same challenge.
        assert_eq!(
            verify::<F, EF, _, (), ()>(&mut verifier, 0, None),
            Ok(alpha)
        );

        // The two sponges stay in step for whatever the surrounding protocol draws next.
        assert_eq!(
            prover.sample_algebra_element::<EF>(),
            verifier.sample_algebra_element::<EF>(),
        );
    }

    #[test]
    fn a_ground_site_and_an_ungrounded_one_agree_with_their_own_verifier() {
        // Invariant: prover and verifier recover one challenge, at any difficulty.
        //
        // Fixture state: difficulty 0 demands no witness, difficulty 8 demands one.
        for bits in [0, 8] {
            let base = challenger();
            let mut prover = base.clone();
            let mut verifier = base.clone();

            let (alpha, witness) = prove::<F, EF, _>(&mut prover, bits);

            // A witness exists exactly when the difficulty is positive.
            assert_eq!(witness.is_some(), bits > 0);

            assert_eq!(
                verify::<F, EF, _, (), ()>(&mut verifier, bits, witness),
                Ok(alpha),
            );
        }
    }

    #[test]
    fn every_difficulty_lands_on_its_own_seed() {
        // Invariant: the difficulty is bound, so no two sites share a transcript seed.
        //
        // Fixture state: zero omits the grinding step.
        //
        // The rest carry it at their own width.
        //
        //     0   ->  [ challenge ]
        //     1   ->  [ pow(1),  challenge ]
        //     8   ->  [ pow(8),  challenge ]
        //     20  ->  [ pow(20), challenge ]
        let seeds: Vec<_> = [0, 1, 8, 20]
            .into_iter()
            .map(|bits| (bits, seed_digest(&separator::<F, EF>(bits))))
            .collect();

        assert_seeds_pairwise_distinct(&seeds);
    }

    #[test]
    fn a_witness_where_none_is_demanded_is_rejected() {
        // Invariant: witness presence comes from configuration, never from the proof.
        //
        // Fixture state: difficulty 0 describes no grinding step.
        //
        // Mutation: attach a witness anyway.
        //
        //     demanded: none      carried: Some(7)
        //     -> shape error before the sponge moves
        let mut challenger = challenger();

        assert_eq!(
            verify::<F, EF, _, (), ()>(&mut challenger, 0, Some(F::from_u64(7))),
            Err(ProofShapeError::BatchPowWitness {
                expected: false,
                got: true,
            }
            .into()),
        );
    }

    #[test]
    fn a_missing_witness_where_one_is_demanded_is_rejected() {
        // Invariant: witness presence comes from configuration, never from the proof.
        //
        // Fixture state: difficulty 8 describes a grinding step.
        //
        // Mutation: carry no witness.
        //
        //     demanded: 8 bits    carried: none
        //     -> shape error before the sponge moves
        let mut challenger = challenger();

        assert_eq!(
            verify::<F, EF, _, (), ()>(&mut challenger, 8, None),
            Err(ProofShapeError::BatchPowWitness {
                expected: true,
                got: false,
            }
            .into()),
        );
    }

    #[test]
    fn a_witness_meeting_only_a_lower_difficulty_is_rejected() {
        // Invariant: a witness must clear the difficulty the configuration demands.
        //
        // Fixture state: the sponge absorbs one value, then the phase seed for 8 bits.
        let mut base = challenger();
        base.observe(F::from_u64(123));
        let mut seeded = base.clone();
        separator::<F, EF>(8).seed(&mut seeded);

        // Mutation: search for a witness clearing one bit but fewer than eight.
        //
        //     clears 1 bit   -> would pass a cheaper site
        //     clears 8 bits  -> no
        //     -> rejected at the demanded difficulty
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

    fn protocol() -> &'static str {
        from_utf8(NAME).expect("the protocol name is ASCII")
    }

    #[test]
    fn the_grinding_vocabulary_maps_the_one_grind_this_protocol_describes() {
        // The security model keys its table on the name and the label bound here.
        let batch = grinding_step(protocol(), BATCH_POW).expect("the batching grind is mapped");
        assert_eq!(batch.site, GrindingSite::BatchCombination);

        // A zero difficulty contributes no step, which is the elided convention.
        assert_eq!(batch.zero_bits, ZeroBitConvention::Elided);

        // One grind described, so one row.
        assert_eq!(
            GRINDING_VOCABULARY
                .iter()
                .filter(|step| step.protocol == protocol())
                .count(),
            1,
        );
    }

    #[test]
    fn the_recorded_difficulty_matches_the_one_the_security_model_credits() {
        // Invariant: the bits this phase describes are the bits the model credits.
        //
        // The credited half is the batch-combination site of a parameter set.
        //
        // The recorded half is read back out of the description.
        //
        // Sweep: zero and two positive difficulties, so both sides of the
        // elided-at-zero convention are exercised.
        for bits in [0, 1, 10] {
            let recorded: Vec<_> = pow_difficulties(separator::<F, EF>(bits).pattern())
                .into_iter()
                .map(|(label, described)| RecordedGrind::new(protocol(), label, described))
                .collect();

            // A zero difficulty describes no step, so nothing is recorded to credit.
            assert_eq!(recorded.len(), usize::from(bits > 0));

            GrindingBudget::from_sites(&GrindingSites {
                batch_combination: bits,
                ..GrindingSites::NONE
            })
            .check(&[protocol()], &recorded)
            .unwrap_or_else(|mismatch| panic!("bits={bits}: {mismatch}"));
        }
    }

    #[test]
    fn this_name_is_separated_from_the_one_it_extends() {
        // Invariant: a name is separated from a shorter name it starts with.
        //
        // Fixture state: this phase's name extends the core protocol's.
        //
        //     [1 | p3-stir            | 0 .. 0 |  7]
        //     [1 | p3-stir-pcs-batch  | 0 .. 0 | 17]
        //
        // Zero padding alone cannot part the two.
        //
        // The last byte carries the name length, and that is what parts them.
        let batch = separator::<F, EF>(8);
        let core =
            DomainSeparator::<FieldUnit<F>>::new(1, b"p3-stir", batch.pattern().as_ref().clone());

        // The prefix relation is real, so the name bytes alone do not separate them.
        assert!(NAME.starts_with(b"p3-stir"));

        // Same version, same description, so only the name can differ.
        assert_ne!(core.protocol_id(), batch.protocol_id());

        // And the differing byte is the length.
        assert_ne!(
            core.protocol_id()[PROTOCOL_ID_LEN - 1],
            batch.protocol_id()[PROTOCOL_ID_LEN - 1],
        );
    }
}
