use alloc::string::ToString;
use alloc::vec::Vec;
use alloc::{format, vec};

use itertools::Itertools;
use p3_circuit::symbolic::ColumnsTargets;
use p3_circuit::{CircuitBuilder, CircuitBuilderError, NonPrimitiveOpId};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{BasedVectorSpace, ExtensionField, Field, PrimeCharacteristicRing, PrimeField64};
use p3_lookup::logup::LogUpGadget;
use p3_uni_stark::{StarkGenericConfig, Val};

use super::{ObservableCommitment, VerificationError, recompose_quotient_from_chunks_circuit};
use crate::Target;
use crate::challenger::CircuitChallenger;
use crate::challenger_perm::ChallengerPermConfig;
use crate::traits::{LookupMetadata, Recursive, RecursiveAir, RecursivePcs};
use crate::types::{
    CommitmentTargets, OpenedValuesTargets, OpenedValuesTargetsWithLookups, ProofTargets,
    StarkChallengeParams, StarkChallenges,
};

/// Type alias for PCS verifier parameters.
type PcsVerifierParams<SC, InputProof, OpeningProof, Comm> =
    <<SC as StarkGenericConfig>::Pcs as RecursivePcs<
        SC,
        InputProof,
        OpeningProof,
        Comm,
        <<SC as StarkGenericConfig>::Pcs as Pcs<
            <SC as StarkGenericConfig>::Challenge,
            <SC as StarkGenericConfig>::Challenger,
        >>::Domain,
    >>::VerifierParams;

type PcsDomain<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Domain;

/// Verifies a STARK proof within a circuit.
///
/// This function adds constraints to the circuit builder that verify a STARK proof.
///
/// # Parameters
/// - `config`: STARK configuration including PCS and challenger
/// - `air`: The Algebraic Intermediate Representation defining the computation
/// - `circuit`: Circuit builder to add verification constraints to
/// - `proof_targets`: Recursive representation of the proof
/// - `public_values`: Public input targets
/// - `pcs_params`: PCS-specific verifier parameters (e.g. FRI's log blowup / final poly size)
///
/// # Returns
/// `Ok(Vec<NonPrimitiveOpId>)` containing operation IDs that require private data
/// (e.g., Merkle sibling values for MMCS verification). The caller must set
/// private data for these operations before running the circuit.
/// `Err` if there was a structural error.
#[allow(clippy::too_many_arguments)]
pub fn verify_p3_uni_proof_circuit<
    A,
    SC: StarkGenericConfig,
    Comm: Recursive<
            SC::Challenge,
            Input = <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Commitment,
        > + Clone
        + ObservableCommitment,
    InputProof: Recursive<SC::Challenge>,
    OpeningProof: Recursive<SC::Challenge>,
    CP: ChallengerPermConfig,
    const WIDTH: usize,
    const RATE: usize,
>(
    config: &SC,
    air: &A,
    circuit: &mut CircuitBuilder<SC::Challenge>,
    proof_targets: &ProofTargets<SC, Comm, OpeningProof>,
    public_values: &[Target],
    preprocessed_commit: &Option<Comm>,
    pcs_params: &PcsVerifierParams<SC, InputProof, OpeningProof, Comm>,
    challenger_perm_config: CP,
) -> Result<Vec<NonPrimitiveOpId>, VerificationError>
where
    A: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
    <SC as StarkGenericConfig>::Pcs: RecursivePcs<
            SC,
            InputProof,
            OpeningProof,
            Comm,
            <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Domain,
        >,
    Val<SC>: PrimeField64,
    SC::Challenge: ExtensionField<Val<SC>> + PrimeCharacteristicRing,
{
    let ProofTargets {
        commitments_targets:
            CommitmentTargets {
                trace_targets,
                quotient_chunks_targets,
                random_commit,
                ..
            },
        opened_values_targets,
        opening_proof,
        degree_bits,
    } = proof_targets;

    let OpenedValuesTargets {
        trace_local_targets: opened_trace_local_targets,
        trace_next_targets: opened_trace_next_targets,
        preprocessed_local_targets: opt_opened_preprocessed_local_targets,
        preprocessed_next_targets: opt_opened_preprocessed_next_targets,
        quotient_chunks_targets: opened_quotient_chunks_targets,
        random_targets: opened_random,
        ..
    } = opened_values_targets;

    let degree = 1 << degree_bits;
    let lookup_gadget = LogUpGadget {};
    let preprocessed_width = opt_opened_preprocessed_local_targets
        .as_ref()
        .map_or(0, |p| p.len());

    // Lookups are not supported for recursive single STARK verification.
    let log_quotient_degree = A::get_log_num_quotient_chunks(
        air,
        preprocessed_width,
        &[],
        &[],
        config.is_zk(),
        &lookup_gadget,
    );
    let quotient_degree = 1 << (log_quotient_degree + config.is_zk());

    let pcs = config.pcs();
    let trace_domain = pcs.natural_domain_for_degree(degree);
    let init_trace_domain = pcs.natural_domain_for_degree(degree >> (config.is_zk()));

    let quotient_domain =
        pcs.create_disjoint_domain(trace_domain, 1 << (degree_bits + log_quotient_degree));
    let quotient_chunks_domains = pcs.split_domains(&quotient_domain, quotient_degree);

    let randomized_quotient_chunks_domains = quotient_chunks_domains
        .iter()
        .map(|domain| pcs.natural_domain_for_degree(pcs.size(domain) << (config.is_zk())))
        .collect_vec();

    // Generate all challenges (alpha, zeta, zeta_next, PCS challenges)
    let (challenge_targets, mut challenger) =
        get_circuit_challenges::<A, SC, Comm, InputProof, OpeningProof, CP, WIDTH, RATE>(
            air,
            config,
            proof_targets,
            public_values,
            preprocessed_width,
            preprocessed_commit,
            &init_trace_domain,
            circuit,
            pcs_params,
            challenger_perm_config,
        )?;

    // Validate ZK randomization consistency
    if (opened_random.is_some() != SC::Pcs::ZK) || (random_commit.is_some() != SC::Pcs::ZK) {
        return Err(VerificationError::RandomizationError);
    }

    // Validate proof shape
    validate_proof_shape::<A, SC, Comm>(
        air,
        opened_values_targets,
        preprocessed_width,
        preprocessed_commit,
        quotient_degree,
    )?;

    let alpha = challenge_targets[0];
    let zeta = challenge_targets[1];
    let zeta_next = challenge_targets[2];

    // Prepare commitments with their opening points for PCS verification
    let mut coms_to_verify = if let Some(r_commit) = &random_commit {
        let random_values = opened_random
            .as_ref()
            .ok_or(VerificationError::RandomizationError)?;
        vec![(
            r_commit.clone(),
            vec![(trace_domain, vec![(zeta, random_values.clone())])],
        )]
    } else {
        vec![]
    };

    coms_to_verify.extend([
        (
            trace_targets.clone(),
            vec![(
                trace_domain,
                vec![
                    (zeta, opened_trace_local_targets.clone()),
                    (zeta_next, opened_trace_next_targets.clone()),
                ],
            )],
        ),
        (
            quotient_chunks_targets.clone(),
            // Check the commitment on the randomized domains
            {
                if randomized_quotient_chunks_domains.len() != opened_quotient_chunks_targets.len()
                {
                    return Err(VerificationError::InvalidProofShape(
                        "Randomized quotient chunks length mismatch".to_string(),
                    ));
                }
                randomized_quotient_chunks_domains
                    .iter()
                    .zip(opened_quotient_chunks_targets)
                    .map(|(domain, values)| (*domain, vec![(zeta, values.clone())]))
                    .collect_vec()
            },
        ),
    ]);

    // Add preprocessed commitment verification if present
    if preprocessed_width > 0 {
        coms_to_verify.push((
            preprocessed_commit
                .clone()
                .expect("We checked in validate_proof_shape that the commit exists"),
            vec![(
                trace_domain,
                vec![
                    (zeta, opt_opened_preprocessed_local_targets.clone().unwrap()),
                    (
                        zeta_next,
                        opt_opened_preprocessed_next_targets.clone().unwrap(),
                    ),
                ],
            )],
        ));
    }

    // Verify polynomial openings using PCS
    let mmcs_op_ids = pcs.verify_circuit::<WIDTH, RATE, CP>(
        circuit,
        &challenge_targets[3..], // PCS challenges (after alpha, zeta, zeta_next)
        &mut challenger,
        &coms_to_verify,
        opening_proof,
        pcs_params,
    )?;

    // Compute quotient polynomial evaluation from chunks
    let quotient = recompose_quotient_from_chunks_circuit::<
        SC,
        InputProof,
        OpeningProof,
        Comm,
        PcsDomain<SC>,
    >(
        circuit,
        &quotient_chunks_domains,
        opened_quotient_chunks_targets,
        zeta,
        pcs,
    );

    // Evaluate AIR constraints at out-of-domain point
    // Note that lookups are not supported for recursive single STARK verification.
    let sels = pcs.selectors_at_point_circuit(circuit, &init_trace_domain, &zeta);
    let columns_targets = ColumnsTargets {
        challenges: &[],
        public_values,
        permutation_local_values: &[],
        permutation_next_values: &[],
        permutation_values: &[],
        local_prep_values: opt_opened_preprocessed_local_targets
            .as_ref()
            .map_or(&[], |p| p),
        next_prep_values: opt_opened_preprocessed_next_targets
            .as_ref()
            .map_or(&[], |p| p),
        local_values: opened_trace_local_targets,
        next_values: opened_trace_next_targets,
    };

    let dummy_lookup_metadata = LookupMetadata {
        contexts: &[],
        lookup_data: &[],
    };
    let folded_constraints = air.eval_folded_circuit(
        circuit,
        &sels,
        &alpha,
        &dummy_lookup_metadata,
        columns_targets,
        &lookup_gadget,
    );

    // Verify: constraints / Z_H(zeta) == quotient(zeta)
    let folded_mul = circuit.mul(folded_constraints, sels.inv_vanishing);
    circuit.connect(folded_mul, quotient);

    Ok(mmcs_op_ids)
}

/// Generate all challenges for STARK verification.
///
/// This includes:
/// - Base STARK challenges (alpha, zeta, zeta_next)
/// - PCS-specific challenges (e.g., FRI betas, query indices)
#[allow(clippy::too_many_arguments)]
fn get_circuit_challenges<
    A: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
    SC: StarkGenericConfig,
    Comm: Recursive<
            SC::Challenge,
            Input = <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Commitment,
        > + ObservableCommitment,
    InputProof: Recursive<SC::Challenge>,
    OpeningProof: Recursive<SC::Challenge>,
    CP: ChallengerPermConfig,
    const WIDTH: usize,
    const RATE: usize,
>(
    _air: &A,
    config: &SC,
    proof_targets: &ProofTargets<SC, Comm, OpeningProof>,
    public_values: &[Target],
    preprocessed_width: usize,
    preprocessed_commit: &Option<Comm>,
    init_trace_domain: &PcsDomain<SC>,
    circuit: &mut CircuitBuilder<SC::Challenge>,
    pcs_params: &PcsVerifierParams<SC, InputProof, OpeningProof, Comm>,
    challenger_perm_config: CP,
) -> Result<(Vec<Target>, CircuitChallenger<WIDTH, RATE, CP>), CircuitBuilderError>
where
    SC::Pcs: RecursivePcs<
            SC,
            InputProof,
            OpeningProof,
            Comm,
            <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Domain,
        >,
    Val<SC>: PrimeField64,
    SC::Challenge: ExtensionField<Val<SC>> + PrimeCharacteristicRing,
{
    let pcs = config.pcs();

    // Compute the trace domain generator for zeta_next = zeta * generator
    // The generator is the primitive n-th root of unity for the init_trace_domain
    let first_point = pcs.first_point(init_trace_domain);
    let next_point = init_trace_domain
        .next_point(first_point)
        .expect("init_trace_domain should have next_point");
    let trace_domain_generator = next_point * first_point.inverse();

    let mut challenger = CircuitChallenger::<WIDTH, RATE, CP>::new(challenger_perm_config);

    // Set up challenge parameters matching native challenger behavior
    let challenge_params = StarkChallengeParams {
        degree_bits: proof_targets.degree_bits,
        is_zk: config.is_zk(),
        preprocessed_width,
        preprocessed_commit,
        trace_domain_generator,
    };

    // Allocate base STARK challenges (alpha, zeta, zeta_next) using Fiat-Shamir
    let base_challenges = StarkChallenges::allocate::<SC, Comm, OpeningProof>(
        circuit,
        &mut challenger,
        proof_targets,
        public_values,
        &challenge_params,
    );

    let opened_values_no_lookups = OpenedValuesTargetsWithLookups {
        opened_values_no_lookups: proof_targets.opened_values_targets.clone(),
        permutation_local_targets: vec![],
        permutation_next_targets: vec![],
    };

    // Observe opened values before getting PCS challenges.
    // For single-STARK with one instance, the standard observation order is correct.
    opened_values_no_lookups.observe(circuit, &mut challenger);

    // Get PCS-specific challenges (FRI betas, query indices, etc.)
    let pcs_challenges = SC::Pcs::get_challenges_circuit::<WIDTH, RATE, CP>(
        circuit,
        &mut challenger,
        &proof_targets.opening_proof,
        &opened_values_no_lookups,
        pcs_params,
    )?;

    // Return flat vector: [alpha, zeta, zeta_next, ...pcs_challenges] and challenger for PCS verification
    let mut all_challenges = base_challenges.to_vec();
    all_challenges.extend(pcs_challenges);
    Ok((all_challenges, challenger))
}

/// Validate the shape of the proof (dimensions, lengths).
fn validate_proof_shape<A, SC: StarkGenericConfig, Comm>(
    air: &A,
    opened_values: &OpenedValuesTargets<SC>,
    preprocessed_width: usize,
    preprocessed_commit: &Option<Comm>,
    quotient_degree: usize,
) -> Result<(), VerificationError>
where
    A: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
    SC::Challenge: PrimeCharacteristicRing,
{
    let air_width = A::width(air);

    if preprocessed_commit.is_some() && preprocessed_width == 0 {
        return Err(VerificationError::InvalidProofShape(
            "There is a preprocessed commit but no opening values provided.".to_string(),
        ));
    }

    if preprocessed_commit.is_none() && preprocessed_width > 0 {
        return Err(VerificationError::InvalidProofShape(
            "Preprocessed width is non-zero but no preprocessed commit provided.".to_string(),
        ));
    }

    let OpenedValuesTargets {
        trace_local_targets: opened_trace_local,
        trace_next_targets: opened_trace_next,
        preprocessed_local_targets: opened_prep_local,
        preprocessed_next_targets: opened_prep_next,
        quotient_chunks_targets: opened_quotient_chunks,
        random_targets: opened_random,
        ..
    } = opened_values;

    if opened_trace_local.len() != air_width || opened_trace_next.len() != air_width {
        return Err(VerificationError::InvalidProofShape(format!(
            "Expected opened_trace_local and opened_trace_next to have length {}, got {} and {}",
            air_width,
            opened_trace_local.len(),
            opened_trace_next.len()
        )));
    }

    let preprocessed_local_len = opened_prep_local.as_ref().map_or(0, |v| v.len());
    let preprocessed_next_len = opened_prep_next.as_ref().map_or(0, |v| v.len());
    if preprocessed_width != preprocessed_local_len || preprocessed_width != preprocessed_next_len {
        // Verifier expects preprocessed trace while proof does not have it, or vice versa
        return Err(VerificationError::InvalidProofShape(format!(
            "Expected preprocessed width {preprocessed_width} but local has length {preprocessed_local_len} and next has length {preprocessed_next_len}"
        )));
    }

    if opened_quotient_chunks.len() != quotient_degree {
        return Err(VerificationError::InvalidProofShape(format!(
            "Expected opened_quotient_chunks to have length {}, got {}",
            quotient_degree,
            opened_quotient_chunks.len()
        )));
    }

    if opened_quotient_chunks
        .iter()
        .any(|opened_chunk| opened_chunk.len() != SC::Challenge::DIMENSION)
    {
        return Err(VerificationError::InvalidProofShape(format!(
            "Invalid quotient chunk length: expected {}",
            SC::Challenge::DIMENSION
        )));
    }

    if let Some(r_comm) = &opened_random
        && r_comm.len() != SC::Challenge::DIMENSION
    {
        return Err(VerificationError::InvalidProofShape(format!(
            "Expected opened random values to have length {}, got {}",
            SC::Challenge::DIMENSION,
            r_comm.len()
        )));
    }

    Ok(())
}
