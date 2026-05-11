#![allow(clippy::upper_case_acronyms)]

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::{format, vec};

use hashbrown::HashMap;
use p3_air::{Air as P3Air, BaseAir as P3BaseAir, SymbolicAirBuilder};
use p3_batch_stark::CommonData;
use p3_circuit::symbolic::ColumnsTargets;
use p3_circuit::{CircuitBuilder, NonPrimitiveOpId};
use p3_circuit_prover::air::{AluAir, ConstAir, PublicAir};
use p3_circuit_prover::batch_stark_prover::{
    AirVariant, DynamicAirEntry, NUM_PRIMITIVE_TABLES, PrimitiveTable, RowCounts, TableProver,
};
use p3_circuit_prover::field_params::ExtractBinomialW;
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{
    Algebra, BasedVectorSpace, ExtensionField, Field, PrimeCharacteristicRing, PrimeField64,
};
use p3_lookup::LookupAir;
use p3_lookup::lookup_traits::{Kind, Lookup, LookupData, LookupGadget};
use p3_uni_stark::{StarkGenericConfig, SymbolicExpression, SymbolicExpressionExt, Val};

use super::{ObservableCommitment, VerificationError, recompose_quotient_from_chunks_circuit};
use crate::challenger::CircuitChallenger;
use crate::challenger_perm::ChallengerPermConfig;
use crate::traits::{
    LookupMetadata, Recursive, RecursiveAir, RecursiveChallenger, RecursiveLookupGadget,
    RecursivePcs,
};
use crate::types::{
    BatchProofTargets, CommonDataTargets, OpenedValuesTargets, OpenedValuesTargetsWithLookups,
};
use crate::{BatchStarkVerifierInputsBuilder, Target};

/// Type alias for PCS verifier parameters.
pub type PcsVerifierParams<SC, InputProof, OpeningProof, Comm> =
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

/// Type-erased recursive AIR entry for non-primitive tables.
pub type DynRecursionAirEntry<SC> = DynamicAirEntry<SC>;

/// Wrapper enum for heterogeneous circuit table AIRs used by circuit-prover tables.
pub enum CircuitTablesAir<SC: StarkGenericConfig, const D: usize> {
    Const(ConstAir<Val<SC>, D>),
    Public(PublicAir<Val<SC>, D>),
    Alu(AluAir<Val<SC>, D>),
    Dynamic(DynRecursionAirEntry<SC>),
}

impl<SC, const D: usize> P3BaseAir<Val<SC>> for CircuitTablesAir<SC, D>
where
    SC: StarkGenericConfig,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SymbolicExpression<Val<SC>>>,
{
    fn width(&self) -> usize {
        match self {
            Self::Const(a) => P3BaseAir::width(a),
            Self::Public(a) => P3BaseAir::width(a),
            Self::Alu(a) => P3BaseAir::width(a),
            Self::Dynamic(a) => P3BaseAir::width(a),
        }
    }

    fn num_public_values(&self) -> usize {
        match self {
            Self::Const(a) => P3BaseAir::num_public_values(a),
            Self::Public(a) => P3BaseAir::num_public_values(a),
            Self::Alu(a) => P3BaseAir::num_public_values(a),
            Self::Dynamic(a) => P3BaseAir::num_public_values(a),
        }
    }
}

impl<SC, const D: usize> P3Air<SymbolicAirBuilder<Val<SC>, <SC as StarkGenericConfig>::Challenge>>
    for CircuitTablesAir<SC, D>
where
    SC: StarkGenericConfig,
    Val<SC>: PrimeField64,
    <SC as StarkGenericConfig>::Challenge: ExtensionField<Val<SC>>,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    fn eval(
        &self,
        builder: &mut SymbolicAirBuilder<Val<SC>, <SC as StarkGenericConfig>::Challenge>,
    ) {
        match self {
            Self::Const(a) => P3Air::eval(a, builder),
            Self::Public(a) => P3Air::eval(a, builder),
            Self::Alu(a) => P3Air::eval(a, builder),
            Self::Dynamic(inner) => P3Air::eval(inner, builder),
        }
    }
}

impl<SC, const D: usize> LookupAir<Val<SC>> for CircuitTablesAir<SC, D>
where
    SC: StarkGenericConfig,
    Val<SC>: PrimeField64,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        match self {
            Self::Const(a) => LookupAir::<Val<SC>>::add_lookup_columns(a),
            Self::Public(a) => LookupAir::<Val<SC>>::add_lookup_columns(a),
            Self::Alu(a) => LookupAir::<Val<SC>>::add_lookup_columns(a),
            Self::Dynamic(inner) => LookupAir::<Val<SC>>::add_lookup_columns(inner),
        }
    }

    #[allow(clippy::missing_transmute_annotations)]
    fn get_lookups(&mut self) -> Vec<Lookup<Val<SC>>> {
        match self {
            Self::Const(a) => LookupAir::<Val<SC>>::get_lookups(a),
            Self::Public(a) => LookupAir::<Val<SC>>::get_lookups(a),
            Self::Alu(a) => LookupAir::<Val<SC>>::get_lookups(a),
            Self::Dynamic(inner) => LookupAir::<Val<SC>>::get_lookups(inner),
        }
    }
}

/// Create an AluAir with the appropriate constructor based on TRACE_D.
///
/// For D=1 (base field), uses `new_with_preprocessed` with zeroed lane prep.
/// For D=5 with `alu_quintic_trinomial`, uses `new_quintic_trinomial_with_preprocessed`.
/// Otherwise for D>1, uses `new_binomial_with_preprocessed` with `W` from `EF`.
/// `horner_packed_steps` must match `BatchStarkProof.table_packing.horner_packed_steps` from the proof.
fn create_alu_air<F, EF, const TRACE_D: usize>(
    num_ops: usize,
    lanes: usize,
    horner_packed_steps: usize,
    alu_quintic_trinomial: bool,
) -> AluAir<F, TRACE_D>
where
    F: Field + PrimeCharacteristicRing + Copy,
    EF: ExtensionField<F> + ExtractBinomialW<F>,
{
    if TRACE_D == 1 {
        let preprocessed = if num_ops == 0 {
            Vec::new()
        } else {
            vec![F::ZERO; num_ops * AluAir::<F, TRACE_D>::preprocessed_lane_width()]
        };
        AluAir::<F, TRACE_D>::new_with_preprocessed(
            num_ops,
            lanes,
            preprocessed,
            horner_packed_steps,
        )
    } else if TRACE_D == 5 && alu_quintic_trinomial {
        let preprocessed = if num_ops == 0 {
            Vec::new()
        } else {
            vec![F::ZERO; num_ops * AluAir::<F, TRACE_D>::preprocessed_lane_width()]
        };
        AluAir::<F, TRACE_D>::new_quintic_trinomial_with_preprocessed(
            num_ops,
            lanes,
            preprocessed,
            horner_packed_steps,
        )
    } else {
        let w = binomial_w_for_alu::<F, EF>();
        let preprocessed = if num_ops == 0 {
            Vec::new()
        } else {
            vec![F::ZERO; num_ops * AluAir::<F, TRACE_D>::preprocessed_lane_width()]
        };
        AluAir::<F, TRACE_D>::new_binomial_with_preprocessed(
            num_ops,
            lanes,
            w,
            preprocessed,
            horner_packed_steps,
        )
    }
}

fn binomial_w_for_alu<F: Field, EF: ExtensionField<F> + ExtractBinomialW<F>>() -> F {
    EF::extract_w().expect("extension field must provide binomial W for ALU AIR")
}

/// Build and attach a recursive verifier circuit for a circuit-prover [`BatchStarkProof`].
///
/// This reconstructs the circuit table AIRs from the proof metadata (rows + packing) so callers
/// don't need to pass `circuit_airs` explicitly. Returns the allocated input builder to pack
/// public inputs afterwards.
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
pub fn verify_p3_batch_proof_circuit<
    SC: StarkGenericConfig + 'static,
    Comm: Recursive<
            SC::Challenge,
            Input = <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Commitment,
        > + Clone
        + ObservableCommitment,
    InputProof: Recursive<SC::Challenge>,
    OpeningProof: Recursive<SC::Challenge, Input = <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Proof>,
    LG: RecursiveLookupGadget<SC::Challenge>,
    CP: ChallengerPermConfig,
    const WIDTH: usize,
    const RATE: usize,
    const TRACE_D: usize,
>(
    config: &SC,
    circuit: &mut CircuitBuilder<SC::Challenge>,
    proof: &p3_circuit_prover::batch_stark_prover::BatchStarkProof<SC>,
    pcs_params: &PcsVerifierParams<SC, InputProof, OpeningProof, Comm>,
    common_data: &CommonData<SC>,
    lookup_gadget: &LG,
    challenger_perm_config: CP,
    non_primitive_provers: &[Box<dyn TableProver<SC>>],
) -> Result<
    (
        BatchStarkVerifierInputsBuilder<SC, Comm, OpeningProof>,
        Vec<NonPrimitiveOpId>,
    ),
    VerificationError,
>
where
    <SC as StarkGenericConfig>::Pcs: RecursivePcs<
            SC,
            InputProof,
            OpeningProof,
            Comm,
            <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Domain,
        >,
    Val<SC>: PrimeField64,
    SC::Challenge: ExtensionField<Val<SC>> + PrimeCharacteristicRing + ExtractBinomialW<Val<SC>>,
    <<SC as StarkGenericConfig>::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Domain: Clone,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    assert_eq!(proof.ext_degree, TRACE_D, "trace extension degree mismatch");
    let rows: RowCounts = proof.rows;
    let packing = proof.table_packing.clone();
    let public_lanes = packing.public_lanes();
    let alu_lanes = packing.alu_lanes();

    // Create AluAir with appropriate constructor based on TRACE_D and the stored
    // primitive ALU variant used during proving.
    // For now both variants share the same AIR type; this hook allows us to swap
    // in a different ALU AIR in the future based on `proof.alu_variant`.
    let alu_air = match proof.alu_variant {
        AirVariant::Baseline | AirVariant::Optimized => {
            create_alu_air::<Val<SC>, SC::Challenge, TRACE_D>(
                rows[PrimitiveTable::Alu],
                alu_lanes,
                packing.horner_packed_steps(),
                proof.alu_quintic_trinomial,
            )
        }
    };

    let mut circuit_airs: Vec<CircuitTablesAir<SC, TRACE_D>> = vec![
        CircuitTablesAir::Const(ConstAir::<Val<SC>, TRACE_D>::new(
            rows[PrimitiveTable::Const],
        )),
        CircuitTablesAir::Public(PublicAir::<Val<SC>, TRACE_D>::new(
            rows[PrimitiveTable::Public],
            public_lanes,
        )),
        CircuitTablesAir::Alu(alu_air),
    ];

    for entry in &proof.non_primitives {
        let plugin = non_primitive_provers
            .iter()
            .find(|p| TableProver::op_type(p.as_ref()) == entry.op_type)
            .ok_or_else(|| {
                VerificationError::InvalidProofShape(format!(
                    "unknown non-primitive op: {:?}",
                    entry.op_type
                ))
            })?;
        let air = plugin
            .batch_air_from_table_entry(config, TRACE_D, proof.ext_degree as u32, entry)
            .map_err(VerificationError::InvalidProofShape)?;
        circuit_airs.push(CircuitTablesAir::Dynamic(air));
    }

    let mut air_public_counts = vec![0usize; NUM_PRIMITIVE_TABLES];
    for entry in &proof.non_primitives {
        air_public_counts.push(entry.public_values.len());
    }
    let verifier_inputs = BatchStarkVerifierInputsBuilder::<SC, Comm, OpeningProof>::allocate(
        circuit,
        &proof.proof,
        common_data,
        &air_public_counts,
    );

    let common = &verifier_inputs.common_data;

    let mmcs_op_ids = verify_batch_circuit::<
        CircuitTablesAir<SC, TRACE_D>,
        SC,
        Comm,
        InputProof,
        OpeningProof,
        LG,
        CP,
        WIDTH,
        RATE,
    >(
        config,
        &circuit_airs,
        circuit,
        &verifier_inputs.proof_targets,
        &verifier_inputs.air_public_targets,
        pcs_params,
        common,
        lookup_gadget,
        challenger_perm_config,
    )?;

    Ok((verifier_inputs, mmcs_op_ids))
}

/// Verify a batch-STARK proof inside a recursive circuit.
///
/// # Returns
/// `Ok(Vec<NonPrimitiveOpId>)` containing operation IDs that require private data
/// (e.g., Merkle sibling values for MMCS verification). The caller must set
/// private data for these operations before running the circuit.
/// `Err` if there was a structural error.
#[allow(clippy::too_many_arguments)]
pub fn verify_batch_circuit<
    A,
    SC: StarkGenericConfig,
    Comm: Recursive<
            SC::Challenge,
            Input = <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Commitment,
        > + Clone
        + ObservableCommitment,
    InputProof: Recursive<SC::Challenge>,
    OpeningProof: Recursive<SC::Challenge>,
    LG: RecursiveLookupGadget<SC::Challenge>,
    CP: ChallengerPermConfig,
    const WIDTH: usize,
    const RATE: usize,
>(
    config: &SC,
    airs: &[A],
    circuit: &mut CircuitBuilder<SC::Challenge>,
    proof_targets: &BatchProofTargets<SC, Comm, OpeningProof>,
    public_values: &[Vec<Target>],
    pcs_params: &PcsVerifierParams<SC, InputProof, OpeningProof, Comm>,
    common: &CommonDataTargets<SC, Comm>,
    lookup_gadget: &LG,
    challenger_perm_config: CP,
) -> Result<Vec<NonPrimitiveOpId>, VerificationError>
where
    A: RecursiveAir<Val<SC>, SC::Challenge, LG>,
    <SC as StarkGenericConfig>::Pcs: RecursivePcs<
            SC,
            InputProof,
            OpeningProof,
            Comm,
            <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Domain,
        >,
    Val<SC>: PrimeField64,
    SC::Challenge: ExtensionField<Val<SC>> + PrimeCharacteristicRing,
    <<SC as StarkGenericConfig>::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Domain: Clone,
{
    let BatchProofTargets {
        commitments_targets,
        flattened_opened_values_targets: flattened,
        opened_values_targets,
        opening_proof,
        global_lookup_data,
        degree_bits,
    } = proof_targets;
    let instances = &opened_values_targets.instances;

    if airs.is_empty() {
        return Err(VerificationError::InvalidProofShape(
            "batch-STARK verification requires at least one instance".to_string(),
        ));
    }

    if airs.len() != instances.len()
        || airs.len() != public_values.len()
        || airs.len() != proof_targets.degree_bits.len()
    {
        return Err(VerificationError::InvalidProofShape(
            "Mismatch between number of AIRs, instances, public values, or degree bits".to_string(),
        ));
    }

    let all_lookups = &common.lookups;

    let pcs = config.pcs();

    let n_instances = airs.len();

    // Check randomization consistency against the PCS ZK setting.
    if (instances
        .iter()
        .any(|inst| inst.opened_values_no_lookups.random_targets.is_some() != SC::Pcs::ZK))
        || (commitments_targets.random_commit.is_some() != SC::Pcs::ZK)
    {
        return Err(VerificationError::RandomizationError);
    }

    // Pre-compute per-instance quotient degrees and preprocessed widths, and validate proof shape.
    let mut preprocessed_widths = Vec::with_capacity(airs.len());
    let mut log_quotient_degrees = Vec::with_capacity(n_instances);
    let mut quotient_degrees = Vec::with_capacity(n_instances);
    for (i, ((air, instance), public_vals)) in airs
        .iter()
        .zip(instances.iter())
        .zip(public_values)
        .enumerate()
    {
        let OpenedValuesTargets {
            trace_local_targets,
            trace_next_targets,
            preprocessed_local_targets,
            preprocessed_next_targets,
            quotient_chunks_targets,
            random_targets,
            ..
        } = &instance.opened_values_no_lookups;

        let pre_w = common
            .preprocessed
            .as_ref()
            .and_then(|g| g.instances.instances[i].as_ref().map(|m| m.width))
            .unwrap_or(0);
        preprocessed_widths.push(pre_w);

        let local_prep_len = preprocessed_local_targets.as_ref().map_or(0, |v| v.len());
        let next_prep_len = preprocessed_next_targets.as_ref().map_or(0, |v| v.len());
        if local_prep_len != pre_w || next_prep_len != pre_w {
            return Err(VerificationError::InvalidProofShape(format!(
                "Instance has incorrect preprocessed width: expected {pre_w}, got {local_prep_len} / {next_prep_len}"
            )));
        }
        let air_width = A::width(air);
        if trace_local_targets.len() != air_width || trace_next_targets.len() != air_width {
            return Err(VerificationError::InvalidProofShape(format!(
                "Instance has incorrect trace width: expected {}, got {} / {}",
                air_width,
                trace_local_targets.len(),
                trace_next_targets.len()
            )));
        }

        let expected_global_count = all_lookups[i]
            .iter()
            .filter(|l| matches!(&l.kind, Kind::Global(_)))
            .count();
        let actual_global_count = global_lookup_data[i].len();
        if actual_global_count < expected_global_count {
            return Err(VerificationError::InvalidProofShape(
                "Expected cumulated value missing".to_string(),
            ));
        }
        if actual_global_count > expected_global_count {
            return Err(VerificationError::InvalidProofShape(
                "Too many expected cumulated values provided".to_string(),
            ));
        }

        let lookups_i = &all_lookups[i];
        let global_lookups_i = &global_lookup_data[i];
        let mut global_name_idx = 0;
        for lookup in lookups_i {
            match &lookup.kind {
                Kind::Global(name) => {
                    if global_name_idx >= global_lookups_i.len()
                        || global_lookups_i[global_name_idx].name != *name
                    {
                        return Err(VerificationError::InvalidProofShape(
                            "Global lookups are inconsistent with lookups".to_string(),
                        ));
                    }
                    global_name_idx += 1;
                }
                Kind::Local => {}
            }
        }
        if global_name_idx != global_lookups_i.len() {
            return Err(VerificationError::InvalidProofShape(
                "Global lookups are inconsistent with lookups".to_string(),
            ));
        }

        let is_sorted_by_aux_idx = global_lookup_data[i]
            .windows(2)
            .all(|w| w[0].aux_idx <= w[1].aux_idx);
        if !is_sorted_by_aux_idx {
            return Err(VerificationError::InvalidProofShape(
                "Expected cumulated values not sorted by auxiliary index".to_string(),
            ));
        }

        let log_qd = A::get_log_num_quotient_chunks(
            air,
            pre_w,
            &all_lookups[i],
            &lookup_data_to_pv_index(&global_lookup_data[i], public_vals.len()),
            config.is_zk(),
            lookup_gadget,
        );
        let quotient_degree = 1 << (log_qd + config.is_zk());

        if quotient_chunks_targets.len() != quotient_degree {
            return Err(VerificationError::InvalidProofShape(format!(
                "Instance quotient chunk count mismatch: expected {}, got {}",
                quotient_degree,
                quotient_chunks_targets.len()
            )));
        }

        if quotient_chunks_targets
            .iter()
            .any(|chunk| chunk.len() != SC::Challenge::DIMENSION)
        {
            return Err(VerificationError::InvalidProofShape(format!(
                "Invalid quotient chunk length: expected {}",
                SC::Challenge::DIMENSION
            )));
        }

        if random_targets
            .as_ref()
            .is_some_and(|r_vals| r_vals.len() != SC::Challenge::DIMENSION)
        {
            return Err(VerificationError::RandomizationError);
        }

        log_quotient_degrees.push(log_qd);
        quotient_degrees.push(quotient_degree);
    }

    // Challenger initialisation mirrors the native batch-STARK verifier transcript.
    // Native uses observe_base_as_algebra_element which decomposes to D coefficients,
    // so we use observe_ext to match.
    let mut challenger = CircuitChallenger::<WIDTH, RATE, CP>::new(challenger_perm_config);
    let inst_count_target = circuit.alloc_const(
        SC::Challenge::from_usize(n_instances),
        "number of instances",
    );
    challenger.observe_ext(circuit, inst_count_target);

    for ((&ext_db, quotient_degree), air) in degree_bits
        .iter()
        .zip(quotient_degrees.iter())
        .zip(airs.iter())
    {
        let base_db = ext_db.checked_sub(config.is_zk()).ok_or_else(|| {
            VerificationError::InvalidProofShape(
                "Extended degree bits smaller than ZK adjustment".to_string(),
            )
        })?;
        let base_db_target =
            circuit.alloc_const(SC::Challenge::from_usize(base_db), "base degree bits");
        let ext_db_target =
            circuit.alloc_const(SC::Challenge::from_usize(ext_db), "extended degree bits");
        let width_target =
            circuit.alloc_const(SC::Challenge::from_usize(A::width(air)), "air width");
        let quotient_chunks_target = circuit.alloc_const(
            SC::Challenge::from_usize(*quotient_degree),
            "quotient chunk count",
        );

        // Native uses observe_base_as_algebra_element (via observe_instance_binding),
        // so we use observe_ext to match by decomposing to D base coefficients.
        challenger.observe_ext(circuit, ext_db_target);
        challenger.observe_ext(circuit, base_db_target);
        challenger.observe_ext(circuit, width_target);
        challenger.observe_ext(circuit, quotient_chunks_target);
    }

    challenger.observe_slice(
        circuit,
        &commitments_targets.trace_targets.to_observation_targets(),
    );
    for pv in public_values {
        challenger.observe_slice(circuit, pv);
    }

    // Observe preprocessed widths for each instance. If a global
    // preprocessed commitment exists, observe it once.
    // Native uses observe_base_as_algebra_element, so we use observe_ext.
    for &pre_w in preprocessed_widths.iter() {
        let pre_w_target =
            circuit.alloc_const(SC::Challenge::from_usize(pre_w), "preprocessed width");
        challenger.observe_ext(circuit, pre_w_target);
    }
    if let Some(global) = &common.preprocessed {
        challenger.observe_slice(circuit, &global.commitment.to_observation_targets());
    }

    // Validate shape of the lookup commitment.
    let is_lookup = proof_targets
        .commitments_targets
        .permutation_targets
        .is_some();
    if is_lookup != all_lookups.iter().any(|c| !c.is_empty()) {
        return Err(VerificationError::InvalidProofShape(
            "Mismatch between lookup commitment and lookup data".to_string(),
        ));
    }

    // Fetch lookups and sample their challenges.
    let challenges_per_instance = get_perm_challenges::<SC, CP, WIDTH, RATE, LG>(
        circuit,
        &mut challenger,
        all_lookups,
        lookup_gadget,
    );

    // Then, observe the permutation tables, if any.
    if is_lookup {
        challenger.observe_slice(
            circuit,
            &commitments_targets
                .permutation_targets
                .clone()
                .expect("We checked that the commitment exists")
                .to_observation_targets(),
        );
        for instance_data in global_lookup_data {
            for ld in instance_data {
                challenger.observe_ext(circuit, ld.expected_cumulated);
            }
        }
    }

    // Sample alpha challenge (extension field element)
    let alpha = challenger.sample_ext(circuit);

    challenger.observe_slice(
        circuit,
        &commitments_targets
            .quotient_chunks_targets
            .to_observation_targets(),
    );
    if let Some(random_commit) = &commitments_targets.random_commit {
        challenger.observe_slice(circuit, &random_commit.to_observation_targets());
    }
    // Sample zeta challenge (extension field element)
    let zeta = challenger.sample_ext(circuit);

    // Build per-instance domains.
    let mut trace_domains = Vec::with_capacity(n_instances);
    let mut ext_trace_domains = Vec::with_capacity(n_instances);
    for &ext_db in degree_bits {
        let base_db = ext_db.checked_sub(config.is_zk()).ok_or_else(|| {
            VerificationError::InvalidProofShape(
                "Extended degree bits smaller than ZK adjustment".to_string(),
            )
        })?;
        trace_domains.push(pcs.natural_domain_for_degree(1 << base_db));
        ext_trace_domains.push(pcs.natural_domain_for_degree(1 << ext_db));
    }

    // Collect commitments with opening points for PCS verification.
    // We have, in the typical lookup case, up to five rounds:
    // optional random, trace, quotient, optional preprocessed, and optional permutation.
    let mut coms_to_verify = Vec::with_capacity(5);

    if let Some(random_commit) = &commitments_targets.random_commit {
        coms_to_verify.push((
            random_commit.clone(),
            ext_trace_domains
                .iter()
                .zip(instances.iter())
                .map(|(domain, inst)| {
                    let random_vals = inst
                        .opened_values_no_lookups
                        .random_targets
                        .as_ref()
                        .ok_or(VerificationError::RandomizationError)?;
                    Ok((*domain, vec![(zeta, random_vals.clone())]))
                })
                .collect::<Result<Vec<_>, VerificationError>>()?,
        ));
    }

    let trace_round: Vec<_> = ext_trace_domains
        .iter()
        .zip(trace_domains.iter())
        .zip(instances.iter())
        .map(|((ext_dom, trace_dom), inst)| {
            let first_point = pcs.first_point(trace_dom);
            let next_point = trace_dom.next_point(first_point).ok_or_else(|| {
                VerificationError::InvalidProofShape(
                    "Trace domain does not provide next point".to_string(),
                )
            })?;
            let generator = next_point * first_point.inverse();
            let generator_const = circuit.define_const(generator);
            let zeta_next = circuit.mul(zeta, generator_const);
            Ok((
                *ext_dom,
                vec![
                    (
                        zeta,
                        inst.opened_values_no_lookups.trace_local_targets.clone(),
                    ),
                    (
                        zeta_next,
                        inst.opened_values_no_lookups.trace_next_targets.clone(),
                    ),
                ],
            ))
        })
        .collect::<Result<_, VerificationError>>()?;
    coms_to_verify.push((commitments_targets.trace_targets.clone(), trace_round));

    let quotient_domains: Vec<Vec<_>> = degree_bits
        .iter()
        .zip(ext_trace_domains.iter())
        .zip(log_quotient_degrees.iter())
        .map(
            |((&ext_db, ext_dom), &log_qd)| -> Result<Vec<_>, VerificationError> {
                let base_db = ext_db.checked_sub(config.is_zk()).ok_or_else(|| {
                    VerificationError::InvalidProofShape(
                        "Extended degree bits smaller than ZK adjustment".to_string(),
                    )
                })?;
                let q_domain =
                    ext_dom.create_disjoint_domain(1 << (base_db + log_qd + config.is_zk()));
                Ok(q_domain.split_domains(1 << (log_qd + config.is_zk())))
            },
        )
        .collect::<Result<Vec<_>, VerificationError>>()?;

    let randomized_quotient_domains: Vec<Vec<_>> = quotient_domains
        .iter()
        .map(|domains| {
            domains
                .iter()
                .map(|domain| pcs.natural_domain_for_degree(pcs.size(domain) << config.is_zk()))
                .collect()
        })
        .collect();

    let mut quotient_round = Vec::with_capacity(
        randomized_quotient_domains
            .iter()
            .map(|domains| domains.len())
            .sum(),
    );
    for (domains, inst) in randomized_quotient_domains.iter().zip(instances.iter()) {
        if domains.len() != inst.opened_values_no_lookups.quotient_chunks_targets.len() {
            return Err(VerificationError::InvalidProofShape(
                "Quotient chunk count mismatch across domains".to_string(),
            ));
        }
        for (domain, values) in domains
            .iter()
            .zip(inst.opened_values_no_lookups.quotient_chunks_targets.iter())
        {
            quotient_round.push((*domain, vec![(zeta, values.clone())]));
        }
    }
    coms_to_verify.push((
        commitments_targets.quotient_chunks_targets.clone(),
        quotient_round,
    ));

    if let Some(global) = &common.preprocessed {
        let mut pre_round = Vec::with_capacity(global.matrix_to_instance.len());

        for (matrix_index, &inst_idx) in global.matrix_to_instance.iter().enumerate() {
            let pre_w = preprocessed_widths[inst_idx];
            if pre_w == 0 {
                return Err(VerificationError::InvalidProofShape(
                    "Instance has preprocessed columns with zero width".to_string(),
                ));
            }

            let inst = &instances[inst_idx];
            let local = inst
                .opened_values_no_lookups
                .preprocessed_local_targets
                .as_ref()
                .ok_or_else(|| {
                    VerificationError::InvalidProofShape(
                        "Missing preprocessed local columns".to_string(),
                    )
                })?;
            let next = inst
                .opened_values_no_lookups
                .preprocessed_next_targets
                .as_ref()
                .ok_or_else(|| {
                    VerificationError::InvalidProofShape(
                        "Missing preprocessed next columns".to_string(),
                    )
                })?;
            // Validate that the preprocessed data's degree metadata matches this instance.
            let ext_db = degree_bits[inst_idx];

            let meta = global.instances.instances[inst_idx]
                .as_ref()
                .ok_or_else(|| {
                    VerificationError::InvalidProofShape(
                        "Missing preprocessed instance metadata".to_string(),
                    )
                })?;
            if meta.matrix_index != matrix_index || meta.degree_bits != ext_db {
                return Err(VerificationError::InvalidProofShape(
                    "Preprocessed instance metadata mismatch".to_string(),
                ));
            }

            // Compute base preprocessed domain (matching prover in generation.rs)
            let pre_domain = pcs.natural_domain_for_degree(1 << meta.degree_bits);

            // Use the base trace domain for zeta_next computation.
            let trace_dom = &trace_domains[inst_idx];
            let first_point = pcs.first_point(trace_dom);
            let next_point = trace_dom.next_point(first_point).ok_or_else(|| {
                VerificationError::InvalidProofShape(
                    "Preprocessed domain does not provide next point".to_string(),
                )
            })?;
            let generator = next_point * first_point.inverse();
            let generator_const = circuit.define_const(generator);
            let zeta_next = circuit.mul(zeta, generator_const);

            pre_round.push((
                pre_domain,
                vec![(zeta, local.clone()), (zeta_next, next.clone())],
            ));
        }

        coms_to_verify.push((global.commitment.clone(), pre_round));
    }

    if is_lookup {
        let permutation_commit = commitments_targets
            .permutation_targets
            .clone()
            .expect("We checked that the commitment exists");

        let mut permutation_round = Vec::with_capacity(ext_trace_domains.len());

        for (i, ext_dom) in ext_trace_domains.iter().enumerate() {
            let inst = &instances[i];
            let permutation_local = &inst.permutation_local_targets;
            let permutation_next = &inst.permutation_next_targets;

            if permutation_local.len() != permutation_next.len() {
                return Err(VerificationError::InvalidProofShape(
                    "Mismatch between the lengths of permutation local and next opened values"
                        .to_string(),
                ));
            }

            if !permutation_local.is_empty() {
                let trace_dom = &trace_domains[i];
                let first_point = pcs.first_point(trace_dom);
                let next_point = trace_dom.next_point(first_point).ok_or_else(|| {
                    VerificationError::InvalidProofShape(
                        "Trace domain does not provide next point".to_string(),
                    )
                })?;
                let generator = next_point * first_point.inverse();
                let generator_const = circuit.define_const(generator);
                let zeta_next = circuit.mul(zeta, generator_const);

                permutation_round.push((
                    *ext_dom,
                    vec![
                        (zeta, permutation_local.clone()),
                        (zeta_next, permutation_next.clone()),
                    ],
                ));
            }
        }

        coms_to_verify.push((permutation_commit, permutation_round));
    }

    // Observe opened values in the correct order (matching native).
    // For HidingFriPcs, the native verifier merges FRI-level random opened values into
    // each point's values before observing. We must do the same here to keep the
    // Fiat-Shamir transcript in sync with the prover/verifier.
    let fri_random_rounds = SC::Pcs::get_fri_random_opened_values(&proof_targets.opening_proof);
    observe_opened_values_circuit::<SC, CP, WIDTH, RATE>(
        circuit,
        &mut challenger,
        instances,
        &quotient_degrees,
        fri_random_rounds,
        common.preprocessed.is_some(),
        is_lookup,
    );

    let pcs_challenges = SC::Pcs::get_challenges_circuit::<WIDTH, RATE, CP>(
        circuit,
        &mut challenger,
        &proof_targets.opening_proof,
        flattened,
        pcs_params,
    )?;

    let mmcs_op_ids = pcs.verify_circuit::<WIDTH, RATE, CP>(
        circuit,
        &pcs_challenges,
        &mut challenger,
        &coms_to_verify,
        opening_proof,
        pcs_params,
    )?;

    // Verify AIR constraints per instance.
    for i in 0..n_instances {
        let air = &airs[i];
        let inst = &instances[i];
        let trace_domain = &trace_domains[i];
        let public_values = &public_values[i];
        let domains = &quotient_domains[i];

        let quotient = recompose_quotient_from_chunks_circuit::<SC, _, _, _, _>(
            circuit,
            domains,
            &inst.opened_values_no_lookups.quotient_chunks_targets,
            zeta,
            pcs,
        );

        // Recompose permutation openings from base-flattened columns into extension field columns.
        // The permutation commitment is a base-flattened matrix with `width = aux_width * DIMENSION`.
        // For constraint evaluation, we need an extension field matrix with width `aux_width``.
        let aux_width = all_lookups[i]
            .iter()
            .flat_map(|ctx| ctx.columns.iter().copied())
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);

        let recompose = |circuit: &mut CircuitBuilder<SC::Challenge>,
                         flat: &[Target]|
         -> Vec<Target> {
            if aux_width == 0 {
                return vec![];
            }
            let ext_degree = SC::Challenge::DIMENSION;
            debug_assert!(
                flat.len() == aux_width * ext_degree,
                "flattened permutation opening length ({}) must equal aux_width ({}) * DIMENSION ({})",
                flat.len(),
                aux_width,
                ext_degree
            );
            // Chunk the flattened coefficients into groups of size `dim`.
            // Each chunk represents the coefficients of one extension field element.
            flat.chunks_exact(ext_degree)
                .map(|coeffs| {
                    let mut sum = circuit.define_const(SC::Challenge::ZERO);
                    // Dot product: sum(coeff_j * basis_j)
                    coeffs.iter().enumerate().for_each(|(j, &coeff)| {
                        let e_i = circuit.define_const(
                            SC::Challenge::ith_basis_element(j)
                                .expect("Basis element should exist"),
                        );
                        sum = circuit.mul_add(coeff, e_i, sum);
                    });
                    sum
                })
                .collect()
        };

        let local_permutation_values = recompose(circuit, &inst.permutation_local_targets);
        let next_permutation_values = recompose(circuit, &inst.permutation_next_targets);

        let local_prep_values = match inst
            .opened_values_no_lookups
            .preprocessed_local_targets
            .as_ref()
        {
            Some(v) => v.as_slice(),
            None => &[],
        };
        let next_prep_values = match inst
            .opened_values_no_lookups
            .preprocessed_next_targets
            .as_ref()
        {
            Some(v) => v.as_slice(),
            None => &[],
        };

        let expected_cumulated_values: Vec<Target> = global_lookup_data[i]
            .iter()
            .map(|ld| ld.expected_cumulated)
            .collect();
        let sels = pcs.selectors_at_point_circuit(circuit, trace_domain, &zeta);
        let columns_targets = ColumnsTargets {
            challenges: &challenges_per_instance[i],
            public_values,
            permutation_local_values: &local_permutation_values,
            permutation_next_values: &next_permutation_values,
            permutation_values: &expected_cumulated_values,
            local_prep_values,
            next_prep_values,
            local_values: &inst.opened_values_no_lookups.trace_local_targets,
            next_values: &inst.opened_values_no_lookups.trace_next_targets,
        };

        let lookup_metadata = LookupMetadata {
            contexts: &all_lookups[i],
            lookup_data: &lookup_data_to_pv_index(&global_lookup_data[i], public_values.len()),
        };
        let folded_constraints = air.eval_folded_circuit(
            circuit,
            &sels,
            &alpha,
            &lookup_metadata,
            columns_targets,
            lookup_gadget,
        );

        let folded_mul = circuit.mul(folded_constraints, sels.inv_vanishing);
        circuit.connect(folded_mul, quotient);

        // Check that the global lookup cumulative values accumulate to the expected value.
        let mut global_cumulative = HashMap::<&String, Vec<_>>::new();
        for data in global_lookup_data.iter().flatten() {
            global_cumulative
                .entry(&data.name)
                .or_default()
                .push(data.expected_cumulated);
        }

        for all_expected_cumulative in global_cumulative.values() {
            lookup_gadget.verify_global_final_value_circuit(circuit, all_expected_cumulative);
        }
    }

    Ok(mmcs_op_ids)
}

pub(crate) fn get_perm_challenges<
    SC: StarkGenericConfig,
    CP: ChallengerPermConfig,
    const WIDTH: usize,
    const RATE: usize,
    LG: LookupGadget,
>(
    circuit: &mut CircuitBuilder<SC::Challenge>,
    challenger: &mut CircuitChallenger<WIDTH, RATE, CP>,
    all_lookups: &[Vec<Lookup<Val<SC>>>],
    lookup_gadget: &LG,
) -> Vec<Vec<Target>>
where
    Val<SC>: PrimeField64,
    SC::Challenge: ExtensionField<Val<SC>>,
{
    let num_challenges_per_lookup = lookup_gadget.num_challenges();
    let approx_global_names: usize = all_lookups.iter().map(|contexts| contexts.len()).sum();
    let mut global_perm_challenges = HashMap::with_capacity(approx_global_names);

    all_lookups
        .iter()
        .map(|contexts| {
            // Pre-allocate for the instance's challenges.
            let num_challenges = contexts.len() * num_challenges_per_lookup;
            let mut instance_challenges = Vec::with_capacity(num_challenges);

            for context in contexts {
                match &context.kind {
                    Kind::Global(name) => {
                        // Get or create the global challenges (extension field elements).
                        let challenges: &mut Vec<Target> =
                            global_perm_challenges.entry(name).or_insert_with(|| {
                                (0..num_challenges_per_lookup)
                                    .map(|_| challenger.sample_ext(circuit))
                                    .collect()
                            });
                        instance_challenges.extend_from_slice(challenges);
                    }
                    Kind::Local => {
                        // Local challenges are extension field elements.
                        instance_challenges.extend(
                            (0..num_challenges_per_lookup).map(|_| challenger.sample_ext(circuit)),
                        );
                    }
                }
            }
            instance_challenges
        })
        .collect()
}

fn lookup_data_to_pv_index(
    global_lookup_data: &[LookupData<Target>],
    public_values_len: usize,
) -> Vec<LookupData<usize>> {
    global_lookup_data
        .iter()
        .enumerate()
        .map(|(index, ld)| LookupData {
            name: ld.name.clone(),
            aux_idx: ld.aux_idx,
            expected_cumulated: public_values_len + index,
        })
        .collect::<Vec<_>>()
}

/// Observe opened values in the circuit in the correct order to match native.
///
/// For `HidingFriPcs`, the native verifier merges FRI-level random opened values into
/// each point's values before observing them. `fri_random_rounds` carries those extra
/// values (layout: `rounds[round][mat][point]`) and must be interleaved here to keep
/// the Fiat-Shamir transcript in sync. For `TwoAdicFriPcs`, pass an empty slice.
///
/// Observation order (matching native batch-STARK verifier):
/// 1. Random round (if ZK): for each instance, observe random opened values (+ FRI random)
/// 2. Trace round: for each instance, observe trace_local (+ FRI random) then trace_next (+ FRI random)
/// 3. Quotient round: for each chunk, observe quotient values (+ FRI random)
/// 4. Preprocessed round (if present): for each matrix, observe prep_local (+ FRI random) then prep_next (+ FRI random)
/// 5. Permutation round (if present): for each instance, observe perm_local (+ FRI random) then perm_next (+ FRI random)
#[allow(clippy::too_many_arguments)]
fn observe_opened_values_circuit<
    SC,
    CP: ChallengerPermConfig,
    const WIDTH: usize,
    const RATE: usize,
>(
    circuit: &mut CircuitBuilder<SC::Challenge>,
    challenger: &mut CircuitChallenger<WIDTH, RATE, CP>,
    instances: &[OpenedValuesTargetsWithLookups<SC>],
    quotient_degrees: &[usize],
    fri_random_rounds: &[Vec<Vec<Vec<Target>>>],
    has_preprocessed: bool,
    is_lookup: bool,
) where
    SC: StarkGenericConfig,
    Val<SC>: PrimeField64,
    SC::Challenge: ExtensionField<Val<SC>>,
{
    // Helper: observe a point's original values followed by any FRI random values.
    let observe_point = |circuit: &mut CircuitBuilder<SC::Challenge>,
                         challenger: &mut CircuitChallenger<WIDTH, RATE, CP>,
                         original: &[Target],
                         fri_random: Option<&Vec<Target>>| {
        challenger.observe_ext_slice(circuit, original);
        if let Some(rand_vals) = fri_random {
            challenger.observe_ext_slice(circuit, rand_vals);
        }
    };

    // Track which round index within `fri_random_rounds` we are at.
    let mut round_idx: usize = 0;

    // 1. Random round (if ZK): for each instance (= mat), one point at zeta.
    let has_random_round = instances
        .first()
        .is_some_and(|i| i.opened_values_no_lookups.random_targets.is_some());
    if has_random_round {
        let rand_round = fri_random_rounds.get(round_idx);
        for (mat_idx, inst) in instances.iter().enumerate() {
            if let Some(random_vals) = &inst.opened_values_no_lookups.random_targets {
                let fri_rand = rand_round
                    .and_then(|r| r.get(mat_idx))
                    .and_then(|m| m.first());
                observe_point(circuit, challenger, random_vals, fri_rand);
            }
        }
        round_idx += 1;
    }

    // 2. Trace round: for each instance (= mat), two points (zeta, zeta_next).
    {
        let rand_round = fri_random_rounds.get(round_idx);
        for (mat_idx, inst) in instances.iter().enumerate() {
            let fri_rand_local = rand_round
                .and_then(|r| r.get(mat_idx))
                .and_then(|m| m.first());
            let fri_rand_next = rand_round
                .and_then(|r| r.get(mat_idx))
                .and_then(|m| m.get(1));
            observe_point(
                circuit,
                challenger,
                &inst.opened_values_no_lookups.trace_local_targets,
                fri_rand_local,
            );
            observe_point(
                circuit,
                challenger,
                &inst.opened_values_no_lookups.trace_next_targets,
                fri_rand_next,
            );
        }
        round_idx += 1;
    }

    // 3. Quotient round: mats are flattened chunks across all instances, one point each.
    {
        let rand_round = fri_random_rounds.get(round_idx);
        let mut flat_mat_idx: usize = 0;
        for (inst, &qd) in instances.iter().zip(quotient_degrees.iter()) {
            for chunk_values in inst
                .opened_values_no_lookups
                .quotient_chunks_targets
                .iter()
                .take(qd)
            {
                let fri_rand = rand_round
                    .and_then(|r| r.get(flat_mat_idx))
                    .and_then(|m| m.first());
                observe_point(circuit, challenger, chunk_values, fri_rand);
                flat_mat_idx += 1;
            }
        }
        round_idx += 1;
    }

    // 4. Preprocessed round (if present): mats are indexed by matrix_to_instance order.
    if has_preprocessed {
        let rand_round = fri_random_rounds.get(round_idx);
        let mut mat_idx: usize = 0;
        for inst in instances {
            if let Some(prep_local) = &inst.opened_values_no_lookups.preprocessed_local_targets {
                let fri_rand_local = rand_round
                    .and_then(|r| r.get(mat_idx))
                    .and_then(|m| m.first());
                let fri_rand_next = rand_round
                    .and_then(|r| r.get(mat_idx))
                    .and_then(|m| m.get(1));
                observe_point(circuit, challenger, prep_local, fri_rand_local);
                if let Some(prep_next) = &inst.opened_values_no_lookups.preprocessed_next_targets {
                    observe_point(circuit, challenger, prep_next, fri_rand_next);
                }
                mat_idx += 1;
            }
        }
        round_idx += 1;
    }

    // 5. Permutation round (if present): for each instance with non-empty permutation.
    if is_lookup {
        let rand_round = fri_random_rounds.get(round_idx);
        let mut mat_idx: usize = 0;
        for inst in instances {
            if !inst.permutation_local_targets.is_empty() {
                let fri_rand_local = rand_round
                    .and_then(|r| r.get(mat_idx))
                    .and_then(|m| m.first());
                let fri_rand_next = rand_round
                    .and_then(|r| r.get(mat_idx))
                    .and_then(|m| m.get(1));
                observe_point(
                    circuit,
                    challenger,
                    &inst.permutation_local_targets,
                    fri_rand_local,
                );
                if !inst.permutation_next_targets.is_empty() {
                    observe_point(
                        circuit,
                        challenger,
                        &inst.permutation_next_targets,
                        fri_rand_next,
                    );
                }
                mat_idx += 1;
            }
        }
    }
}
