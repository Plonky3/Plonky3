use alloc::vec;
use alloc::vec::Vec;

use hashbrown::HashMap;
use p3_air::Air;
use p3_air::symbolic::AirLayout;
use p3_batch_stark::symbolic::get_log_num_quotient_chunks as get_batch_log_num_quotient_chunks;
use p3_batch_stark::{BatchProof, BatchTranscript, CommonData};
use p3_challenger::{CanObserve, CanSample, CanSampleBits, FieldChallenger, GrindingChallenger};
use p3_commit::{BatchOpening, Mmcs, OpenedValues, Pcs, PolynomialSpace};
use p3_field::{Algebra, BasedVectorSpace, PrimeCharacteristicRing, PrimeField, TwoAdicField};
use p3_fri::{FriProof, HidingFriPcs, TwoAdicFriPcs};
use p3_lookup::lookup_traits::{Kind, Lookup, LookupGadget};
use p3_uni_stark::{
    Domain, StarkGenericConfig, SymbolicAirBuilder, SymbolicExpression, SymbolicExpressionExt, Val,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GenerationError {
    #[error("Missing parameter for challenge generation")]
    MissingParameterError,

    #[error(
        "Invalid number of parameters provided for challenge generation: got {0}, expected {1}"
    )]
    InvalidParameterCount(usize, usize),

    #[error("The FRI batch randomization does not correspond to the ZK setting.")]
    RandomizationError,

    #[error("Witness check failed during challenge generation.")]
    InvalidPowWitness,

    #[error("Invalid proof shape: {0}")]
    InvalidProofShape(&'static str),
}

/// A type alias for a single opening point and its values.
type PointOpening<SC> = (
    <SC as StarkGenericConfig>::Challenge,
    Vec<<SC as StarkGenericConfig>::Challenge>,
);

/// A type alias for all openings within a specific domain.
type DomainOpenings<SC> = Vec<(Domain<SC>, Vec<PointOpening<SC>>)>;

/// A type alias for a commitment and its associated domain openings.
type CommitmentWithOpenings<SC> = (
    <<SC as StarkGenericConfig>::Pcs as Pcs<
        <SC as StarkGenericConfig>::Challenge,
        <SC as StarkGenericConfig>::Challenger,
    >>::Commitment,
    DomainOpenings<SC>,
);

/// The final type alias for a slice of commitments with their openings.
type ComsWithOpenings<SC> = [CommitmentWithOpenings<SC>];

/// Trait which defines the methods necessary
/// for a Pcs to generate challenge values.
pub trait PcsGeneration<SC: StarkGenericConfig, OpeningProof> {
    fn generate_challenges(
        &self,
        config: &SC,
        challenger: &mut SC::Challenger,
        coms_to_verify: &ComsWithOpenings<SC>,
        opening_proof: &OpeningProof,
        // Depending on the `OpeningProof`, we might need additional parameters. For example, for a `FriProof`, we need the `log_max_height` to sample query indices.
        extra_params: Option<&[usize]>,
    ) -> Result<Vec<SC::Challenge>, GenerationError>;

    fn num_challenges(
        opening_proof: &OpeningProof,
        extra_params: Option<&[usize]>,
    ) -> Result<usize, GenerationError>;
}

/// Generates the challenges used in the verification of a batch-STARK proof.
pub fn generate_batch_challenges<SC: StarkGenericConfig, A, LG: LookupGadget>(
    airs: &[A],
    config: &SC,
    proof: &BatchProof<SC>,
    public_values: &[Vec<Val<SC>>],
    extra_params: Option<&[usize]>,
    common_data: &CommonData<SC>,
    lookup_gadget: &LG,
) -> Result<Vec<SC::Challenge>, GenerationError>
where
    A: Air<SymbolicAirBuilder<Val<SC>, SC::Challenge>>,
    SC::Pcs: PcsGeneration<SC, <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Proof>,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    let all_lookups = &common_data.lookups;

    let BatchProof {
        commitments,
        opened_values,
        opening_proof,
        global_lookup_data,
        degree_bits,
    } = proof;

    // Check that the global lookup data is consistent with the lookups.
    all_lookups
        .iter()
        .zip(global_lookup_data)
        .try_for_each(|(lookups, global_lookups)| {
            let mut counter = 0;
            lookups.iter().try_for_each(|lookup| match &lookup.kind {
                Kind::Global(name) => {
                    if counter >= global_lookups.len() || global_lookups[counter].name != *name {
                        Err(GenerationError::InvalidProofShape(
                            "Global lookups are inconsistent with lookups",
                        ))
                    } else {
                        counter += 1;
                        Ok(())
                    }
                }
                Kind::Local => Ok(()),
            })?;
            if counter != global_lookups.len() {
                return Err(GenerationError::InvalidProofShape(
                    "Global lookups are inconsistent with lookups",
                ));
            }
            let is_sorted = global_lookups
                .windows(2)
                .all(|w| w[0].aux_idx <= w[1].aux_idx);
            if !is_sorted {
                return Err(GenerationError::InvalidProofShape(
                    "Expected cumulated values not sorted by auxiliary index",
                ));
            }
            Ok(())
        })?;

    let n_instances = airs.len();
    if n_instances == 0
        || opened_values.instances.len() != n_instances
        || public_values.len() != n_instances
        || degree_bits.len() != n_instances
    {
        return Err(GenerationError::InvalidProofShape(
            "instance metadata length mismatch",
        ));
    }

    // Check randomization consistency against the PCS ZK setting.
    if (opened_values
        .instances
        .iter()
        .any(|ov| ov.base_opened_values.random.is_some() != SC::Pcs::ZK))
        || (commitments.random.is_some() != SC::Pcs::ZK)
    {
        return Err(GenerationError::RandomizationError);
    }

    let pcs = config.pcs();
    let mut transcript = BatchTranscript::<SC>::new(config.initialise_challenger());

    transcript.observe_instance_count(n_instances);

    for inst in &opened_values.instances {
        if inst
            .base_opened_values
            .quotient_chunks
            .iter()
            .any(|c| c.len() != SC::Challenge::DIMENSION)
        {
            return Err(GenerationError::InvalidProofShape(
                "invalid quotient chunk length",
            ));
        }

        if inst
            .base_opened_values
            .random
            .as_ref()
            .is_some_and(|r_vals| r_vals.len() != SC::Challenge::DIMENSION)
        {
            return Err(GenerationError::RandomizationError);
        }
    }

    let mut preprocessed_widths = Vec::with_capacity(airs.len());
    let mut log_quotient_degrees = Vec::with_capacity(n_instances);
    let mut quotient_degrees = Vec::with_capacity(n_instances);
    for (i, air) in airs.iter().enumerate() {
        let pre_w = common_data
            .preprocessed
            .as_ref()
            .and_then(|g| g.instances[i].as_ref().map(|m| m.width))
            .unwrap_or(0);
        preprocessed_widths.push(pre_w);

        let batch_layout = AirLayout {
            preprocessed_width: pre_w,
            main_width: air.width(),
            num_public_values: air.num_public_values(),
            ..Default::default()
        };
        let log_qd = get_batch_log_num_quotient_chunks(
            air,
            batch_layout,
            &all_lookups[i],
            config.is_zk(),
            lookup_gadget,
        );
        let quotient_degree = 1 << (log_qd + config.is_zk());
        log_quotient_degrees.push(log_qd);
        quotient_degrees.push(quotient_degree);
    }

    for i in 0..n_instances {
        let ext_db = degree_bits[i];
        let base_db =
            ext_db
                .checked_sub(config.is_zk())
                .ok_or(GenerationError::InvalidProofShape(
                    "extended degree smaller than zk adjustment",
                ))?;

        transcript.observe_instance_binding(
            ext_db,
            base_db,
            A::width(&airs[i]),
            quotient_degrees[i],
        );
    }

    transcript.observe_main(&commitments.main, public_values);
    transcript.observe_preprocessed(&preprocessed_widths, common_data.preprocessed.as_ref());

    let is_lookup = commitments.permutation.is_some();

    // Fetch lookups and sample their challenges.
    // We use `get_different_perm_challenges` to ensure we only store the newly created challenges, in their order of sampling.
    let different_challenges = transcript
        .sample_perm_challenges(all_lookups, lookup_gadget)
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    // Then, observe the permutation tables, if any and sample the alpha challenge.
    let alpha = transcript
        .observe_perm_and_sample_alpha(commitments.permutation.as_ref(), global_lookup_data);

    transcript.observe_quotient_commitment(&commitments.quotient_chunks);
    if let Some(random_commit) = &commitments.random {
        transcript.observe_random_commitment(random_commit);
    }
    let zeta = transcript.sample_zeta();

    let trace_domains: Vec<_> = degree_bits
        .iter()
        .map(|&ext_db| {
            let base_db =
                ext_db
                    .checked_sub(config.is_zk())
                    .ok_or(GenerationError::InvalidProofShape(
                        "extended degree smaller than zk adjustment",
                    ))?;
            Ok(pcs.natural_domain_for_degree(1 << base_db))
        })
        .collect::<Result<Vec<_>, GenerationError>>()?;
    let ext_trace_domains: Vec<_> = degree_bits
        .iter()
        .map(|&ext_db| pcs.natural_domain_for_degree(1 << ext_db))
        .collect();

    // We have, in the typical lookup case, up to five rounds:
    // optional random, trace, quotient, optional preprocessed, and optional permutation.
    let mut coms_to_verify = Vec::with_capacity(5);

    if let Some(random_commit) = &commitments.random {
        let random_round = ext_trace_domains
            .iter()
            .zip(opened_values.instances.iter())
            .map(|(domain, inst)| {
                let random_vals = inst
                    .base_opened_values
                    .random
                    .as_ref()
                    .ok_or(GenerationError::RandomizationError)?;
                Ok((*domain, vec![(zeta, random_vals.clone())]))
            })
            .collect::<Result<Vec<_>, GenerationError>>()?;
        coms_to_verify.push((random_commit.clone(), random_round));
    }

    let trace_round = ext_trace_domains
        .iter()
        .zip(trace_domains.iter())
        .zip(opened_values.instances.iter())
        .map(|((ext_dom, trace_dom), inst)| {
            let zeta_next =
                trace_dom
                    .next_point(zeta)
                    .ok_or(GenerationError::InvalidProofShape(
                        "trace domain lacks next point",
                    ))?;
            Ok((
                *ext_dom,
                vec![
                    (zeta, inst.base_opened_values.trace_local.clone()),
                    (
                        zeta_next,
                        inst.base_opened_values
                            .trace_next
                            .clone()
                            .expect("trace_next is always present"),
                    ),
                ],
            ))
        })
        .collect::<Result<Vec<_>, GenerationError>>()?;
    coms_to_verify.push((commitments.main.clone(), trace_round));

    let quotient_domains: Vec<Vec<_>> = degree_bits
        .iter()
        .zip(ext_trace_domains.iter())
        .zip(log_quotient_degrees.iter())
        .map(
            |((&ext_db, ext_dom), &log_qd)| -> Result<Vec<_>, GenerationError> {
                let base_db = ext_db.checked_sub(config.is_zk()).ok_or(
                    GenerationError::InvalidProofShape(
                        "extended degree smaller than zk adjustment",
                    ),
                )?;
                let q_domain =
                    ext_dom.create_disjoint_domain(1 << (base_db + log_qd + config.is_zk()));
                Ok(q_domain.split_domains(1 << (log_qd + config.is_zk())))
            },
        )
        .collect::<Result<Vec<_>, GenerationError>>()?;

    let randomized_quotient_domains: Vec<Vec<_>> = quotient_domains
        .iter()
        .map(|domains| {
            domains
                .iter()
                .map(|domain| pcs.natural_domain_for_degree(domain.size() << config.is_zk()))
                .collect()
        })
        .collect();

    let mut quotient_round = Vec::with_capacity(
        randomized_quotient_domains
            .iter()
            .map(|domains| domains.len())
            .sum(),
    );
    for (domains, inst) in randomized_quotient_domains
        .iter()
        .zip(opened_values.instances.iter())
    {
        if inst.base_opened_values.quotient_chunks.len() != domains.len() {
            return Err(GenerationError::InvalidProofShape(
                "quotient chunk count mismatch",
            ));
        }
        for (domain, values) in domains
            .iter()
            .zip(inst.base_opened_values.quotient_chunks.iter())
        {
            quotient_round.push((*domain, vec![(zeta, values.clone())]));
        }
    }
    coms_to_verify.push((commitments.quotient_chunks.clone(), quotient_round));

    if let Some(global) = &common_data.preprocessed {
        let mut pre_round = Vec::with_capacity(global.matrix_to_instance.len());

        for (matrix_index, &inst_idx) in global.matrix_to_instance.iter().enumerate() {
            let pre_w = preprocessed_widths[inst_idx];
            if pre_w == 0 {
                return Err(GenerationError::InvalidProofShape(
                    "preprocessed width is zero but commitment exists",
                ));
            }

            let inst = &opened_values.instances[inst_idx];
            let local = inst.base_opened_values.preprocessed_local.as_ref().ok_or(
                GenerationError::InvalidProofShape("preprocessed local values should exist"),
            )?;
            let next = inst.base_opened_values.preprocessed_next.as_ref().ok_or(
                GenerationError::InvalidProofShape("preprocessed next values should exist"),
            )?;

            // Validate that the preprocessed data's degree metadata matches this instance.
            let ext_db = degree_bits[inst_idx];

            let meta =
                global.instances[inst_idx]
                    .as_ref()
                    .ok_or(GenerationError::InvalidProofShape(
                        "Missing preprocessed instance metadata",
                    ))?;
            if meta.matrix_index != matrix_index || meta.degree_bits != ext_db {
                return Err(GenerationError::InvalidProofShape(
                    "Preprocessed instance metadata mismatch",
                ));
            }

            let base_db = meta.degree_bits;
            let pre_domain = pcs.natural_domain_for_degree(1 << base_db);
            let zeta_next_i = trace_domains[inst_idx].next_point(zeta).ok_or(
                GenerationError::InvalidProofShape("Preprocessed domain lacks next point"),
            )?;

            pre_round.push((
                pre_domain,
                vec![(zeta, local.clone()), (zeta_next_i, next.clone())],
            ));
        }

        coms_to_verify.push((global.commitment.clone(), pre_round));
    }

    if is_lookup {
        let permutation_commit = commitments.permutation.clone().unwrap();
        let mut permutation_round = Vec::with_capacity(ext_trace_domains.len());
        for (i, (ext_dom, inst_opened_vals)) in ext_trace_domains
            .iter()
            .zip(opened_values.instances.iter())
            .enumerate()
        {
            if inst_opened_vals.permutation_local.len() != inst_opened_vals.permutation_next.len() {
                return Err(GenerationError::InvalidProofShape(
                    "Permutation opened values length mismatch",
                ));
            }
            if !inst_opened_vals.permutation_local.is_empty() {
                let zeta_next =
                    trace_domains[i]
                        .next_point(zeta)
                        .ok_or(GenerationError::InvalidProofShape(
                            "Missing preprocessed instance metadata",
                        ))?;
                permutation_round.push((
                    *ext_dom,
                    vec![
                        (zeta, inst_opened_vals.permutation_local.clone()),
                        (zeta_next, inst_opened_vals.permutation_next.clone()),
                    ],
                ));
            }
        }
        coms_to_verify.push((permutation_commit, permutation_round));
    }

    let pcs_challenges = pcs.generate_challenges(
        config,
        &mut transcript.challenger,
        &coms_to_verify,
        opening_proof,
        extra_params,
    )?;

    let mut challenges = Vec::with_capacity(2 + pcs_challenges.len());
    challenges.extend(different_challenges);
    challenges.push(alpha);
    challenges.push(zeta);
    challenges.extend(pcs_challenges);

    Ok(challenges)
}

type InnerFriProof<SC, InputMmcs, FriMmcs> = FriProof<
    <SC as StarkGenericConfig>::Challenge,
    FriMmcs,
    Val<SC>,
    Vec<BatchOpening<Val<SC>, InputMmcs>>,
>;

impl<SC: StarkGenericConfig, Dft, InputMmcs: Mmcs<Val<SC>>, FriMmcs: Mmcs<SC::Challenge>>
    PcsGeneration<SC, InnerFriProof<SC, InputMmcs, FriMmcs>>
    for TwoAdicFriPcs<Val<SC>, Dft, InputMmcs, FriMmcs>
where
    Val<SC>: TwoAdicField + PrimeField,
    SC::Challenger: FieldChallenger<Val<SC>>
        + GrindingChallenger<Witness = Val<SC>>
        + CanObserve<FriMmcs::Commitment>,
{
    fn generate_challenges(
        &self,
        _config: &SC,
        challenger: &mut SC::Challenger,
        coms_to_verify: &ComsWithOpenings<SC>,
        opening_proof: &InnerFriProof<SC, InputMmcs, FriMmcs>,
        extra_params: Option<&[usize]>,
    ) -> Result<Vec<SC::Challenge>, GenerationError> {
        let num_challenges =
            <Self as PcsGeneration<SC, InnerFriProof<SC, InputMmcs, FriMmcs>>>::num_challenges(
                opening_proof,
                None,
            )?;
        let mut challenges = Vec::with_capacity(num_challenges);

        // Observe all openings.
        for (_, round) in coms_to_verify {
            for (_, mat) in round {
                for (_, point) in mat {
                    point
                        .iter()
                        .for_each(|&opening| challenger.observe_algebra_element(opening));
                }
            }
        }

        challenges.push(challenger.sample_algebra_element());

        // Get `beta` challenges for the FRI rounds.
        opening_proof
            .commit_phase_commits
            .iter()
            .zip(&opening_proof.commit_pow_witnesses)
            .for_each(|(comm, pow_witness)| {
                // To match with the prover (and for security purposes),
                // we observe the commitment before sampling the challenge.
                challenger.observe(comm.clone());
                challenger.observe(*pow_witness);
                // Sample a challenge as H(transcript || pow_witness). The circuit later
                // verifies that the challenge begins with the required number of leading zeros.
                let rand_f: Val<SC> = challenger.sample();
                let rand_usize = rand_f.as_canonical_biguint().to_u64_digits()[0] as usize;
                challenges.push(SC::Challenge::from_usize(rand_usize));

                challenges.push(challenger.sample_algebra_element());
            });

        // Observe all coefficients of the final polynomial.
        opening_proof
            .final_poly
            .iter()
            .for_each(|x| challenger.observe_algebra_element(*x));

        // Bind the variable-arity schedule into the transcript before query grinding,
        // matching the native FRI verifier in Plonky3.
        if let Some(first_qp) = opening_proof.query_proofs.first() {
            for step in &first_qp.commit_phase_openings {
                challenger.observe(Val::<SC>::from_usize(step.log_arity as usize));
            }
        }

        let params = extra_params.ok_or(GenerationError::MissingParameterError)?;

        if params.len() != 2 {
            return Err(GenerationError::InvalidParameterCount(params.len(), 2));
        }

        // Check PoW witness.
        challenger.observe(opening_proof.query_pow_witness);

        // Sample a challenge as H(transcript || pow_witness). The circuit later
        // verifies that the challenge begins with the required number of leading zeros.
        let rand_f: Val<SC> = challenger.sample();
        let rand_usize = rand_f.as_canonical_biguint().to_u64_digits()[0] as usize;
        challenges.push(SC::Challenge::from_usize(rand_usize));

        let log_height_max = params[1];
        let log_global_max_height = opening_proof.commit_phase_commits.len() + log_height_max;
        for _ in &opening_proof.query_proofs {
            // For each query proof, we start by generating the random index.
            challenges.push(SC::Challenge::from_usize(
                challenger.sample_bits(log_global_max_height),
            ));
        }

        Ok(challenges)
    }

    fn num_challenges(
        opening_proof: &InnerFriProof<SC, InputMmcs, FriMmcs>,
        _extra_params: Option<&[usize]>,
    ) -> Result<usize, GenerationError> {
        let num_challenges =
            1 + opening_proof.commit_phase_commits.len() + opening_proof.query_proofs.len();

        Ok(num_challenges)
    }
}

type HidingInnerFriProof<SC, InputMmcs, FriMmcs> = (
    OpenedValues<<SC as StarkGenericConfig>::Challenge>,
    InnerFriProof<SC, InputMmcs, FriMmcs>,
);

impl<SC: StarkGenericConfig, Dft, InputMmcs: Mmcs<Val<SC>>, FriMmcs: Mmcs<SC::Challenge>, R>
    PcsGeneration<SC, HidingInnerFriProof<SC, InputMmcs, FriMmcs>>
    for HidingFriPcs<Val<SC>, Dft, InputMmcs, FriMmcs, R>
where
    Val<SC>: TwoAdicField + PrimeField,
    SC::Challenger: FieldChallenger<Val<SC>>
        + GrindingChallenger<Witness = Val<SC>>
        + CanObserve<FriMmcs::Commitment>,
{
    fn generate_challenges(
        &self,
        _config: &SC,
        challenger: &mut SC::Challenger,
        coms_to_verify: &ComsWithOpenings<SC>,
        opening_proof: &HidingInnerFriProof<SC, InputMmcs, FriMmcs>,
        extra_params: Option<&[usize]>,
    ) -> Result<Vec<SC::Challenge>, GenerationError> {
        let inner_proof = &opening_proof.1;
        let num_challenges = <Self as PcsGeneration<
            SC,
            HidingInnerFriProof<SC, InputMmcs, FriMmcs>,
        >>::num_challenges(opening_proof, None)?;
        let mut challenges = Vec::with_capacity(num_challenges);

        for (_, round) in coms_to_verify {
            for (_, mat) in round {
                for (_, point) in mat {
                    point
                        .iter()
                        .for_each(|&opening| challenger.observe_algebra_element(opening));
                }
            }
        }

        challenges.push(challenger.sample_algebra_element());

        inner_proof
            .commit_phase_commits
            .iter()
            .zip(&inner_proof.commit_pow_witnesses)
            .for_each(|(comm, pow_witness)| {
                challenger.observe(comm.clone());
                challenger.observe(*pow_witness);
                let rand_f: Val<SC> = challenger.sample();
                let rand_usize = rand_f.as_canonical_biguint().to_u64_digits()[0] as usize;
                challenges.push(SC::Challenge::from_usize(rand_usize));
                challenges.push(challenger.sample_algebra_element());
            });

        inner_proof
            .final_poly
            .iter()
            .for_each(|x| challenger.observe_algebra_element(*x));

        if let Some(first_qp) = inner_proof.query_proofs.first() {
            for step in &first_qp.commit_phase_openings {
                challenger.observe(Val::<SC>::from_usize(step.log_arity as usize));
            }
        }

        let params = extra_params.ok_or(GenerationError::MissingParameterError)?;
        if params.len() != 2 {
            return Err(GenerationError::InvalidParameterCount(params.len(), 2));
        }

        challenger.observe(inner_proof.query_pow_witness);
        let rand_f: Val<SC> = challenger.sample();
        let rand_usize = rand_f.as_canonical_biguint().to_u64_digits()[0] as usize;
        challenges.push(SC::Challenge::from_usize(rand_usize));

        let log_height_max = params[1];
        let log_global_max_height = inner_proof.commit_phase_commits.len() + log_height_max;
        for _ in &inner_proof.query_proofs {
            challenges.push(SC::Challenge::from_usize(
                challenger.sample_bits(log_global_max_height),
            ));
        }

        Ok(challenges)
    }

    fn num_challenges(
        opening_proof: &HidingInnerFriProof<SC, InputMmcs, FriMmcs>,
        _extra_params: Option<&[usize]>,
    ) -> Result<usize, GenerationError> {
        let inner_proof = &opening_proof.1;
        Ok(1 + inner_proof.commit_phase_commits.len() + inner_proof.query_proofs.len())
    }
}

/// Samples all the different permutation challenges in the right order.
/// This method is used to generate values for the challenge public values, and so it must store
/// only the newly created challenges, in their order of sampling.
pub fn get_different_perm_challenges<SC: StarkGenericConfig, LG: LookupGadget>(
    challenger: &mut SC::Challenger,
    all_lookups: &[Vec<Lookup<Val<SC>>>],
    lookup_gadget: &LG,
) -> Vec<SC::Challenge> {
    let num_challenges_per_lookup = lookup_gadget.num_challenges();
    let approx_global_names: usize = all_lookups.iter().map(|contexts| contexts.len()).sum();
    let approx_total_challenges: usize = all_lookups
        .iter()
        .map(|contexts| contexts.len() * num_challenges_per_lookup)
        .sum();
    let mut global_perm_names = HashMap::with_capacity(approx_global_names);
    let mut different_challenges = Vec::with_capacity(approx_total_challenges);

    all_lookups.iter().for_each(|contexts| {
        for context in contexts {
            match &context.kind {
                Kind::Global(name) => {
                    // Get or create the global challenges.
                    // We only store the newly created challenges, in `different_challenges`.
                    // `global_perm_challenges` is just used to track the names already encountered, so
                    // we only insert `(name, ())` when a new name is encountered.
                    global_perm_names.entry(name).or_insert_with(|| {
                        (0..num_challenges_per_lookup).for_each(|_| {
                            let sampled = challenger.sample_algebra_element();
                            different_challenges.push(sampled);
                        });
                    });
                }
                Kind::Local => {
                    different_challenges.extend(
                        (0..num_challenges_per_lookup)
                            .map(|_| challenger.sample_algebra_element::<SC::Challenge>()),
                    );
                }
            }
        }
    });
    different_challenges
}
