//! WHIR-native circuit table proof for recursive verifier traces.
//!
//! This module intentionally does not route through the univariate STARK/FRI
//! prover. It commits each circuit table as an extension-field multilinear
//! WHIR oracle, then proves table-local zero checks over those committed
//! oracles. The verifier does not receive the trace payload; it checks only
//! public circuit shape, table metadata, commitments, local sumchecks, and WHIR
//! openings.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_air::{Air, AirBuilder, BaseAir, RowWindow};
use p3_baby_bear::BabyBear;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_circuit::ops::{
    NpoPrivateData, Op, Poseidon2CircuitRow, Poseidon2Config, Poseidon2Trace, RecomposeTrace,
    RecomposeTraceKind,
};
use p3_circuit::{AluOpKind, Circuit, Traces, WitnessId};
use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, PrimeField64, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_poseidon2_circuit_air::{BabyBearD4Width16, extract_preprocessed_from_operations};
use p3_whir::parameters::{ProtocolParameters, SumcheckStrategy};
use p3_whir::pcs::proof::WhirProof;
use p3_whir::pcs::{WhirExtensionDeferredProverData, WhirPcs};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::batch_stark_prover::BABY_BEAR_MODULUS;
use crate::whir_native_sumcheck::{
    WhirNativeSumcheckProof, point_from_prefix_current_suffix, prove_sumcheck, verify_sumcheck,
};

const DIGEST_MIX: u64 = 1_099_511_627_761;
const TABLE_LAYOUT_VERSION: u32 = 1;

/// Prover/verifier knobs for the WHIR-native table proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WhirNativeCircuitOptions {
    /// Number of Fiat-Shamir sampled WHIR openings per table oracle.
    pub openings_per_table: usize,
    /// Minimum oracle arity. This must be at least the configured WHIR folding
    /// factor, otherwise very small tables cannot be opened by the current WHIR
    /// implementation.
    pub min_num_variables: usize,
}

impl Default for WhirNativeCircuitOptions {
    fn default() -> Self {
        Self {
            openings_per_table: 2,
            min_num_variables: 4,
        }
    }
}

/// Public table categories committed by the proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WhirNativeTableKind {
    Witness,
    Const,
    Public,
    Alu,
    Poseidon2,
    Recompose,
}

impl WhirNativeTableKind {
    const fn tag(self) -> u64 {
        match self {
            Self::Witness => 1,
            Self::Const => 2,
            Self::Public => 3,
            Self::Alu => 4,
            Self::Poseidon2 => 5,
            Self::Recompose => 6,
        }
    }
}

/// Shape information bound into the transcript before a table is committed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhirNativeTableMetadata {
    pub kind: WhirNativeTableKind,
    pub op_type: String,
    pub width: usize,
    pub padded_width: usize,
    pub active_rows: usize,
    pub padded_height: usize,
    pub num_variables: usize,
    pub column_layout_version: u32,
}

/// One table commitment.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "Comm: Serialize", deserialize = "Comm: Deserialize<'de>"))]
pub struct WhirNativeTableCommitment<Comm> {
    pub metadata: WhirNativeTableMetadata,
    pub commitment: Comm,
}

/// Constraint-batch claim for one source table.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "EF: Serialize", deserialize = "EF: Deserialize<'de>"))]
pub struct WhirNativeConstraintSumcheckProof<EF> {
    pub table_index: usize,
    pub checked_constraints: usize,
    pub claimed_zero_sum: EF,
    pub local_proof: Option<WhirNativeLocalConstraintProof<EF>>,
}

/// WHIR opening proof for one table oracle.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize + Send + Sync + Clone, EF: Serialize, MT::Commitment: Serialize, MT::Proof: Serialize",
    deserialize = "F: Deserialize<'de> + Send + Sync + Clone, EF: Deserialize<'de>, MT::Commitment: Deserialize<'de>, MT::Proof: Deserialize<'de>"
))]
pub struct WhirNativeTableOpeningProof<F: Send + Sync + Clone, EF, MT: Mmcs<F>> {
    pub table_index: usize,
    pub opening_claims: Vec<(Vec<EF>, EF)>,
    pub proof: WhirProof<F, EF, MT>,
}

/// Local multilinear constraint family proven by a WHIR-native sumcheck.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WhirNativeLocalConstraintKind {
    Witness,
    KnownRows,
    Alu,
    Poseidon2Air,
    Recompose,
}

impl WhirNativeLocalConstraintKind {
    const fn tag(self) -> u64 {
        match self {
            Self::Witness => 1,
            Self::KnownRows => 2,
            Self::Alu => 3,
            Self::Poseidon2Air => 4,
            Self::Recompose => 5,
        }
    }
}

/// One terminal table-column opening needed to finish a local sumcheck check.
///
/// The point is the full packed table-oracle point, i.e. row coordinates
/// followed by column selector coordinates.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WhirNativeTerminalColumnClaim<EF> {
    pub table_index: usize,
    pub column: usize,
    pub point: Vec<EF>,
    pub value: EF,
}

/// Sumcheck proof for a table-local constraint batch.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "EF: Serialize", deserialize = "EF: Deserialize<'de>"))]
pub struct WhirNativeLocalConstraintProof<EF> {
    pub table_index: usize,
    pub kind: WhirNativeLocalConstraintKind,
    pub degree: usize,
    pub constraint_challenge: EF,
    pub zerocheck_point: Vec<EF>,
    pub terminal_row_point: Vec<EF>,
    pub terminal_claim: EF,
    pub sumcheck: WhirNativeSumcheckProof<EF>,
    pub terminal_openings: Vec<WhirNativeTerminalColumnClaim<EF>>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct WhirNativeConstRow<EF> {
    pub witness_id: u32,
    pub value: EF,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WhirNativePublicRow<EF> {
    pub witness_id: u32,
    pub value: EF,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WhirNativeAluRow<EF> {
    pub kind: u8,
    pub indices: [u32; 4],
    pub values: [EF; 4],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WhirNativePoseidon2Table<F> {
    pub op_type: String,
    pub config: Poseidon2Config,
    pub rows: Vec<WhirNativePoseidon2Row<F>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WhirNativePoseidon2Row<F> {
    pub new_start: bool,
    pub merkle_path: bool,
    pub mmcs_bit: bool,
    pub mmcs_index_sum: F,
    pub input_values: Vec<F>,
    pub in_ctl: Vec<bool>,
    pub input_indices: Vec<u32>,
    pub out_ctl: Vec<bool>,
    pub output_indices: Vec<u32>,
    pub mmcs_index_sum_idx: u32,
    pub mmcs_ctl_enabled: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WhirNativeRecomposeTable<F> {
    pub op_type: String,
    pub kind: u8,
    pub rows: Vec<WhirNativeRecomposeRow<F>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WhirNativeRecomposeRow<F> {
    pub input_wids: Vec<u32>,
    pub output_wid: u32,
    pub values: Vec<F>,
}

/// Full table payload bound to the WHIR commitments.
#[derive(Clone, Serialize, Deserialize)]
pub struct WhirNativeTracePayload<F, EF> {
    pub witness_values: Vec<EF>,
    pub const_rows: Vec<WhirNativeConstRow<EF>>,
    pub public_rows: Vec<WhirNativePublicRow<EF>>,
    pub alu_rows: Vec<WhirNativeAluRow<EF>>,
    pub poseidon2_tables: Vec<WhirNativePoseidon2Table<F>>,
    pub recompose_tables: Vec<WhirNativeRecomposeTable<F>>,
}

/// Oracle-only WHIR-native circuit proof.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize + Send + Sync + Clone, EF: Serialize, MT::Commitment: Serialize, MT::Proof: Serialize",
    deserialize = "F: Deserialize<'de> + Send + Sync + Clone, EF: Deserialize<'de>, MT::Commitment: Deserialize<'de>, MT::Proof: Deserialize<'de>"
))]
pub struct WhirNativeCircuitProof<F: Send + Sync + Clone, EF, MT: Mmcs<F>> {
    pub table_commitments: Vec<WhirNativeTableCommitment<MT::Commitment>>,
    pub constraint_sumcheck_proofs: Vec<WhirNativeConstraintSumcheckProof<EF>>,
    pub opening_proofs: Vec<WhirNativeTableOpeningProof<F, EF, MT>>,
    pub public_io_digest: Vec<F>,
    pub shape_digest: Vec<F>,
}

/// Diagnostic payload-revealing proof wrapper.
///
/// This is useful for local tests and debugging trace extraction, but it is not
/// used by the recursive comparison benchmark path.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize + Send + Sync + Clone, EF: Serialize, MT::Commitment: Serialize, MT::Proof: Serialize",
    deserialize = "F: Deserialize<'de> + Send + Sync + Clone, EF: Deserialize<'de>, MT::Commitment: Deserialize<'de>, MT::Proof: Deserialize<'de>"
))]
pub struct WhirNativeDiagnosticTraceProof<F: Send + Sync + Clone, EF, MT: Mmcs<F>> {
    pub oracle_proof: WhirNativeCircuitProof<F, EF, MT>,
    pub trace_payload: WhirNativeTracePayload<F, EF>,
}

#[derive(Clone, Debug)]
struct WhirNativeTableData<EF> {
    metadata: WhirNativeTableMetadata,
    values: Vec<EF>,
}

#[derive(Debug, Error)]
pub enum WhirNativeCircuitError {
    #[error("unsupported non-primitive trace {0}")]
    UnsupportedNonPrimitiveTrace(String),
    #[error("unsupported WHIR-native oracle-only proof component: {0}")]
    UnsupportedSoundComponent(String),
    #[error("shape digest mismatch")]
    ShapeDigestMismatch,
    #[error("public input digest mismatch")]
    PublicIoDigestMismatch,
    #[error("table count mismatch: expected {expected}, got {got}")]
    TableCountMismatch { expected: usize, got: usize },
    #[error("table metadata mismatch at table {table_index}")]
    TableMetadataMismatch { table_index: usize },
    #[error("table commitment mismatch at table {table_index}")]
    TableCommitmentMismatch { table_index: usize },
    #[error("opening claim mismatch at table {table_index}, opening {opening_index}")]
    OpeningClaimMismatch {
        table_index: usize,
        opening_index: usize,
    },
    #[error("WHIR opening verification failed at table {table_index}: {details}")]
    WhirVerificationFailed { table_index: usize, details: String },
    #[error("constraint violation: {0}")]
    ConstraintViolation(String),
}

fn unsupported_poseidon2_component(op_type: &str) -> WhirNativeCircuitError {
    WhirNativeCircuitError::UnsupportedSoundComponent(format!(
        "non-primitive table `{op_type}` requires a WHIR-native Poseidon2/MMCS AIR proof before it can enter comparison timing"
    ))
}

/// Build and prove a WHIR-native circuit table proof.
#[allow(clippy::too_many_arguments)]
pub fn prove_whir_native_circuit<
    F,
    EF,
    MT,
    Challenger,
    Dft,
    MakePcs,
    MakeChallenger,
    PoseidonEval,
    const DIGEST_ELEMS: usize,
>(
    circuit: &Circuit<EF>,
    public_inputs: &[EF],
    private_inputs: &[EF],
    _npo_private_data: &[NpoPrivateData],
    traces: &Traces<EF>,
    options: WhirNativeCircuitOptions,
    make_pcs: MakePcs,
    make_challenger: MakeChallenger,
    _poseidon_eval: PoseidonEval,
) -> Result<WhirNativeCircuitProof<F, EF, MT>, WhirNativeCircuitError>
where
    F: TwoAdicField + PrimeField64 + Ord + Send + Sync + Serialize + for<'de> Deserialize<'de>,
    EF: ExtensionField<F>
        + ExtensionField<BabyBear>
        + TwoAdicField
        + Send
        + Sync
        + Serialize
        + for<'de> Deserialize<'de>,
    MT: Mmcs<F>,
    MT::Commitment: Clone + PartialEq + Serialize + for<'de> Deserialize<'de>,
    MT::Proof: Clone + Serialize + for<'de> Deserialize<'de>,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanObserve<F>
        + CanObserve<MT::Commitment>
        + Clone,
    Dft: TwoAdicSubgroupDft<F>,
    MakePcs: Fn(usize) -> WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>,
    MakeChallenger: Fn() -> Challenger,
    PoseidonEval: Fn(Poseidon2Config, &[EF]) -> Option<Vec<EF>>,
{
    if private_inputs.len() != circuit.private_flat_len {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "private input length mismatch: expected {}, got {}",
            circuit.private_flat_len,
            private_inputs.len()
        )));
    }
    ensure_oracle_only_supported(circuit)?;

    let payload = trace_payload_from_traces::<F, EF>(circuit, traces)?;
    let public_io_digest = compute_public_io_digest::<F, EF>(public_inputs);
    let shape_digest = compute_shape_digest::<F, EF>(circuit);

    let tables = build_tables(&payload, options)?;
    let expected_metadata = expected_table_metadata::<F, EF>(circuit, public_inputs, options)?;
    if tables.len() != expected_metadata.len() {
        return Err(WhirNativeCircuitError::TableCountMismatch {
            expected: expected_metadata.len(),
            got: tables.len(),
        });
    }
    for (table_index, (table, metadata)) in tables.iter().zip(&expected_metadata).enumerate() {
        if &table.metadata != metadata {
            return Err(WhirNativeCircuitError::TableMetadataMismatch { table_index });
        }
    }

    let mut table_commitments = Vec::with_capacity(tables.len());
    let mut table_prover_data = Vec::with_capacity(tables.len());
    let mut table_challengers = Vec::with_capacity(tables.len());

    for (table_index, table) in tables.iter().enumerate() {
        let pcs = make_pcs(table.metadata.num_variables);
        let mut challenger = make_challenger();
        observe_table_context(
            &mut challenger,
            &public_io_digest,
            &shape_digest,
            table_index,
            &table.metadata,
            options,
        );
        let (commitment, prover_data) = pcs.commit_extension_deferred(
            RowMajorMatrix::new(table.values.clone(), 1),
            &mut challenger,
        );
        table_commitments.push(WhirNativeTableCommitment {
            metadata: table.metadata.clone(),
            commitment,
        });
        table_prover_data.push(prover_data);
        table_challengers.push(challenger);
    }

    let mut constraint_sumcheck_proofs = Vec::with_capacity(tables.len());
    let mut terminal_claims_by_table = vec![Vec::new(); tables.len()];
    for (table_index, table) in tables.iter().enumerate() {
        let local_proof = prove_table_local_constraints::<F, EF, MT, Challenger, MakeChallenger>(
            circuit,
            public_inputs,
            table_index,
            &tables,
            &table_commitments,
            &public_io_digest,
            &shape_digest,
            options,
            &make_challenger,
        )
        .map_err(|err| {
            WhirNativeCircuitError::ConstraintViolation(format!(
                "proving local constraints for table {table_index} ({:?} `{}`) failed: {err}",
                table.metadata.kind, table.metadata.op_type
            ))
        })?;
        for (claim_table_index, point, value) in
            local_proof_terminal_claims::<EF>(local_proof.as_ref(), &expected_metadata)?
        {
            terminal_claims_by_table[claim_table_index].push((point, value));
        }
        constraint_sumcheck_proofs.push(WhirNativeConstraintSumcheckProof {
            table_index,
            checked_constraints: table.metadata.active_rows,
            claimed_zero_sum: EF::ZERO,
            local_proof,
        });
    }

    let mut opening_proofs = Vec::with_capacity(tables.len());
    for (table_index, ((table, prover_data), mut challenger)) in tables
        .iter()
        .zip(table_prover_data)
        .zip(table_challengers)
        .enumerate()
    {
        let pcs = make_pcs(table.metadata.num_variables);
        let mut points = sample_table_opening_points(
            &mut challenger,
            table.metadata.num_variables,
            table_index,
            options.openings_per_table,
        );
        points.extend(
            terminal_claims_by_table[table_index]
                .iter()
                .map(|(point, _)| point.clone()),
        );
        let (opened_values, proof) = pcs.open_extension_deferred(
            prover_data,
            core::slice::from_ref(&points),
            &mut challenger,
        );
        let opening_claims = points
            .into_iter()
            .zip(opened_values[0].iter().copied())
            .map(|(point, value)| (point.as_slice().to_vec(), value))
            .collect();

        opening_proofs.push(WhirNativeTableOpeningProof {
            table_index,
            opening_claims,
            proof,
        });
    }

    let proof = WhirNativeCircuitProof {
        table_commitments,
        constraint_sumcheck_proofs,
        opening_proofs,
        public_io_digest,
        shape_digest,
    };

    verify_whir_native_circuit_proof::<
        F,
        EF,
        MT,
        Challenger,
        Dft,
        MakePcs,
        MakeChallenger,
        PoseidonEval,
        DIGEST_ELEMS,
    >(
        circuit,
        public_inputs,
        options,
        &proof,
        make_pcs,
        make_challenger,
        _poseidon_eval,
    )?;

    Ok(proof)
}

/// Build an oracle proof and include the extracted trace payload for diagnostics.
///
/// This intentionally stays out of benchmark/comparison paths. It is useful for
/// local debugging because it validates the payload against the trace extractor,
/// but verifier soundness for [`WhirNativeCircuitProof`] must come only from
/// commitments, local proofs, and WHIR openings.
#[allow(clippy::too_many_arguments)]
pub fn prove_whir_native_diagnostic_trace<
    F,
    EF,
    MT,
    Challenger,
    Dft,
    MakePcs,
    MakeChallenger,
    PoseidonEval,
    const DIGEST_ELEMS: usize,
>(
    circuit: &Circuit<EF>,
    public_inputs: &[EF],
    private_inputs: &[EF],
    npo_private_data: &[NpoPrivateData],
    traces: &Traces<EF>,
    options: WhirNativeCircuitOptions,
    make_pcs: MakePcs,
    make_challenger: MakeChallenger,
    poseidon_eval: PoseidonEval,
) -> Result<WhirNativeDiagnosticTraceProof<F, EF, MT>, WhirNativeCircuitError>
where
    F: TwoAdicField + PrimeField64 + Ord + Send + Sync + Serialize + for<'de> Deserialize<'de>,
    EF: ExtensionField<F>
        + ExtensionField<BabyBear>
        + TwoAdicField
        + Send
        + Sync
        + Serialize
        + for<'de> Deserialize<'de>,
    MT: Mmcs<F>,
    MT::Commitment: Clone + PartialEq + Serialize + for<'de> Deserialize<'de>,
    MT::Proof: Clone + Serialize + for<'de> Deserialize<'de>,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanObserve<F>
        + CanObserve<MT::Commitment>
        + Clone,
    Dft: TwoAdicSubgroupDft<F>,
    MakePcs: Fn(usize) -> WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>,
    MakeChallenger: Fn() -> Challenger,
    PoseidonEval: Fn(Poseidon2Config, &[EF]) -> Option<Vec<EF>>,
{
    let trace_payload = trace_payload_from_traces::<F, EF>(circuit, traces)?;
    validate_trace_payload(circuit, public_inputs, &trace_payload, &poseidon_eval)?;
    let oracle_proof = prove_whir_native_circuit::<
        F,
        EF,
        MT,
        Challenger,
        Dft,
        MakePcs,
        MakeChallenger,
        PoseidonEval,
        DIGEST_ELEMS,
    >(
        circuit,
        public_inputs,
        private_inputs,
        npo_private_data,
        traces,
        options,
        make_pcs,
        make_challenger,
        poseidon_eval,
    )?;
    Ok(WhirNativeDiagnosticTraceProof {
        oracle_proof,
        trace_payload,
    })
}

/// Verify a WHIR-native circuit table proof.
#[allow(clippy::too_many_arguments)]
pub fn verify_whir_native_circuit_proof<
    F,
    EF,
    MT,
    Challenger,
    Dft,
    MakePcs,
    MakeChallenger,
    PoseidonEval,
    const DIGEST_ELEMS: usize,
>(
    circuit: &Circuit<EF>,
    public_inputs: &[EF],
    options: WhirNativeCircuitOptions,
    proof: &WhirNativeCircuitProof<F, EF, MT>,
    make_pcs: MakePcs,
    make_challenger: MakeChallenger,
    _poseidon_eval: PoseidonEval,
) -> Result<(), WhirNativeCircuitError>
where
    F: TwoAdicField + PrimeField64 + Ord + Send + Sync + Serialize + for<'de> Deserialize<'de>,
    EF: ExtensionField<F>
        + ExtensionField<BabyBear>
        + TwoAdicField
        + Send
        + Sync
        + Serialize
        + for<'de> Deserialize<'de>,
    MT: Mmcs<F>,
    MT::Commitment: Clone + PartialEq + Serialize + for<'de> Deserialize<'de>,
    MT::Proof: Clone + Serialize + for<'de> Deserialize<'de>,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanObserve<F>
        + CanObserve<MT::Commitment>
        + Clone,
    Dft: TwoAdicSubgroupDft<F>,
    MakePcs: Fn(usize) -> WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>,
    MakeChallenger: Fn() -> Challenger,
    PoseidonEval: Fn(Poseidon2Config, &[EF]) -> Option<Vec<EF>>,
{
    let expected_public_digest = compute_public_io_digest::<F, EF>(public_inputs);
    if proof.public_io_digest != expected_public_digest {
        return Err(WhirNativeCircuitError::PublicIoDigestMismatch);
    }

    let expected_shape_digest = compute_shape_digest::<F, EF>(circuit);
    if proof.shape_digest != expected_shape_digest {
        return Err(WhirNativeCircuitError::ShapeDigestMismatch);
    }
    ensure_oracle_only_supported(circuit)?;

    let expected_metadata = expected_table_metadata::<F, EF>(circuit, public_inputs, options)?;
    if proof.table_commitments.len() != expected_metadata.len() {
        return Err(WhirNativeCircuitError::TableCountMismatch {
            expected: expected_metadata.len(),
            got: proof.table_commitments.len(),
        });
    }
    if proof.opening_proofs.len() != expected_metadata.len() {
        return Err(WhirNativeCircuitError::TableCountMismatch {
            expected: expected_metadata.len(),
            got: proof.opening_proofs.len(),
        });
    }
    if proof.constraint_sumcheck_proofs.len() != expected_metadata.len() {
        return Err(WhirNativeCircuitError::TableCountMismatch {
            expected: expected_metadata.len(),
            got: proof.constraint_sumcheck_proofs.len(),
        });
    }
    for (table_index, metadata) in expected_metadata.iter().enumerate() {
        let table_commitment = &proof.table_commitments[table_index];
        if &table_commitment.metadata != metadata {
            return Err(WhirNativeCircuitError::TableMetadataMismatch { table_index });
        }
    }

    let mut terminal_claims_by_table = vec![Vec::new(); expected_metadata.len()];
    for (table_index, metadata) in expected_metadata.iter().enumerate() {
        let constraint_proof = &proof.constraint_sumcheck_proofs[table_index];
        if constraint_proof.table_index != table_index
            || constraint_proof.claimed_zero_sum != EF::ZERO
            || constraint_proof.checked_constraints != metadata.active_rows
        {
            return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                "bad constraint summary for table {table_index}"
            )));
        }
        for (claim_table_index, point, value) in
            verify_table_local_constraints::<F, EF, MT, Challenger, MakeChallenger>(
                circuit,
                public_inputs,
                table_index,
                &expected_metadata,
                &proof.table_commitments,
                &proof.public_io_digest,
                &proof.shape_digest,
                options,
                constraint_proof,
                &make_challenger,
            )?
        {
            terminal_claims_by_table[claim_table_index].push((point, value));
        }
    }

    for (table_index, metadata) in expected_metadata.iter().enumerate() {
        let pcs = make_pcs(metadata.num_variables);
        verify_table_opening(
            &pcs,
            metadata,
            &proof.table_commitments[table_index],
            &proof.opening_proofs[table_index],
            table_index,
            &terminal_claims_by_table[table_index],
            &proof.public_io_digest,
            &proof.shape_digest,
            options,
            &make_challenger,
        )?;
    }

    Ok(())
}

/// Convenience constructor for a WHIR PCS factory.
pub fn whir_native_pcs_factory<EF, F, MT, Challenger, Dft, const DIGEST_ELEMS: usize>(
    protocol_params: ProtocolParameters<MT>,
    dft: Dft,
    sumcheck_strategy: SumcheckStrategy,
) -> impl Fn(usize) -> WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F>,
    Dft: TwoAdicSubgroupDft<F> + Clone,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    move |num_variables| {
        WhirPcs::new(
            num_variables,
            protocol_params.clone(),
            dft.clone(),
            sumcheck_strategy,
        )
    }
}

fn trace_payload_from_traces<F, EF>(
    circuit: &Circuit<EF>,
    traces: &Traces<EF>,
) -> Result<WhirNativeTracePayload<F, EF>, WhirNativeCircuitError>
where
    F: Field + Send + Sync + 'static,
    EF: ExtensionField<F> + Send + Sync + 'static,
{
    let witness_values = (0..traces.witness_trace.num_rows())
        .map(|i| {
            *traces
                .witness_trace
                .get_value(WitnessId(i as u32))
                .expect("witness row must exist")
        })
        .collect();
    let const_rows = traces
        .const_trace
        .index
        .iter()
        .zip(&traces.const_trace.values)
        .map(|(&witness_id, &value)| WhirNativeConstRow {
            witness_id: witness_id.0,
            value,
        })
        .collect();
    let public_rows = traces
        .public_trace
        .index
        .iter()
        .zip(&traces.public_trace.values)
        .map(|(&witness_id, &value)| WhirNativePublicRow {
            witness_id: witness_id.0,
            value,
        })
        .collect();
    let expected_alu_count = expected_alu_rows_from_circuit(circuit).len();
    let mut alu_rows = traces
        .alu_trace
        .op_kind
        .iter()
        .zip(&traces.alu_trace.indices)
        .zip(&traces.alu_trace.values)
        .map(|((&kind, indices), &values)| WhirNativeAluRow {
            kind: alu_kind_to_tag(kind),
            indices: [indices[0].0, indices[1].0, indices[2].0, indices[3].0],
            values,
        })
        .collect::<Vec<_>>();
    if expected_alu_count == 0
        && alu_rows.len() == 1
        && alu_rows[0].kind == alu_kind_to_tag(AluOpKind::Add)
        && alu_rows[0].indices == [0, 0, 0, 0]
        && alu_rows[0].values == [EF::ZERO; 4]
    {
        alu_rows.clear();
    }

    let mut poseidon2_tables = Vec::new();
    let mut recompose_tables = Vec::new();
    let mut npo_entries = traces.non_primitive_traces.iter().collect::<Vec<_>>();
    npo_entries.sort_by(|(left, _), (right, _)| left.as_str().cmp(right.as_str()));

    for (op_type, trace) in npo_entries {
        if let Some(poseidon2) = trace.as_any().downcast_ref::<Poseidon2Trace<F>>() {
            let config = poseidon2_config_from_op_type(op_type.as_str()).ok_or_else(|| {
                WhirNativeCircuitError::UnsupportedNonPrimitiveTrace(op_type.to_string())
            })?;
            poseidon2_tables.push(WhirNativePoseidon2Table {
                op_type: op_type.as_str().to_string(),
                config,
                rows: poseidon2
                    .operations
                    .iter()
                    .map(|row| WhirNativePoseidon2Row {
                        new_start: row.new_start,
                        merkle_path: row.merkle_path,
                        mmcs_bit: row.mmcs_bit,
                        mmcs_index_sum: row.mmcs_index_sum,
                        input_values: row.input_values.clone(),
                        in_ctl: row.in_ctl.clone(),
                        input_indices: row.input_indices.clone(),
                        out_ctl: row.out_ctl.clone(),
                        output_indices: row.output_indices.clone(),
                        mmcs_index_sum_idx: row.mmcs_index_sum_idx,
                        mmcs_ctl_enabled: row.mmcs_ctl_enabled,
                    })
                    .collect(),
            });
        } else if let Some(recompose) = trace.as_any().downcast_ref::<RecomposeTrace<F>>() {
            let kind = recompose_kind_to_tag(recompose.kind);
            recompose_tables.push(WhirNativeRecomposeTable {
                op_type: op_type.as_str().to_string(),
                kind,
                rows: recompose
                    .operations
                    .iter()
                    .map(|row| WhirNativeRecomposeRow {
                        input_wids: row.input_wids.iter().map(|wid| wid.0).collect(),
                        output_wid: row.output_wid.0,
                        values: row.values.clone(),
                    })
                    .collect(),
            });
        } else {
            return Err(WhirNativeCircuitError::UnsupportedNonPrimitiveTrace(
                op_type.to_string(),
            ));
        }
    }

    Ok(WhirNativeTracePayload {
        witness_values,
        const_rows,
        public_rows,
        alu_rows,
        poseidon2_tables,
        recompose_tables,
    })
}

fn validate_trace_payload<F, EF, PoseidonEval>(
    circuit: &Circuit<EF>,
    public_inputs: &[EF],
    payload: &WhirNativeTracePayload<F, EF>,
    poseidon_eval: &PoseidonEval,
) -> Result<(), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F> + ExtensionField<BabyBear> + Send + Sync,
    PoseidonEval: Fn(Poseidon2Config, &[EF]) -> Option<Vec<EF>>,
{
    if public_inputs.len() != circuit.public_flat_len {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "public input length mismatch: expected {}, got {}",
            circuit.public_flat_len,
            public_inputs.len()
        )));
    }
    if payload.witness_values.len() != circuit.witness_count as usize {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "witness table height mismatch: expected {}, got {}",
            circuit.witness_count,
            payload.witness_values.len()
        )));
    }

    validate_const_public_and_alu(circuit, public_inputs, payload)?;
    validate_recompose_tables(circuit, payload)?;
    validate_poseidon2_tables(circuit, payload, poseidon_eval)?;
    Ok(())
}

fn validate_const_public_and_alu<F, EF>(
    circuit: &Circuit<EF>,
    public_inputs: &[EF],
    payload: &WhirNativeTracePayload<F, EF>,
) -> Result<(), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F> + ExtensionField<BabyBear> + Send + Sync,
{
    let mut const_idx = 0;
    let mut public_idx = 0;
    let mut alu_idx = 0;

    for op in &circuit.ops {
        match op {
            Op::Const { out, val } => {
                let row = payload.const_rows.get(const_idx).ok_or_else(|| {
                    WhirNativeCircuitError::ConstraintViolation("missing const row".to_string())
                })?;
                if row.witness_id != out.0 || row.value != *val || witness(payload, *out)? != *val {
                    return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                        "const row {const_idx} does not bind WitnessId({})",
                        out.0
                    )));
                }
                const_idx += 1;
            }
            Op::Public { out, public_pos } => {
                let row = payload.public_rows.get(public_idx).ok_or_else(|| {
                    WhirNativeCircuitError::ConstraintViolation("missing public row".to_string())
                })?;
                let expected = public_inputs.get(*public_pos).copied().ok_or_else(|| {
                    WhirNativeCircuitError::ConstraintViolation(format!(
                        "public input position {public_pos} out of range"
                    ))
                })?;
                if row.witness_id != out.0
                    || row.value != expected
                    || witness(payload, *out)? != expected
                {
                    return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                        "public row {public_idx} does not bind WitnessId({})",
                        out.0
                    )));
                }
                public_idx += 1;
            }
            Op::Alu {
                kind,
                a,
                b,
                c,
                out,
                intermediate_out,
            } => {
                let row = payload.alu_rows.get(alu_idx).ok_or_else(|| {
                    WhirNativeCircuitError::ConstraintViolation("missing ALU row".to_string())
                })?;
                let expected_c = c.unwrap_or(WitnessId(0));
                let expected_indices = [a.0, b.0, expected_c.0, out.0];
                if row.kind != alu_kind_to_tag(*kind) || row.indices != expected_indices {
                    return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                        "ALU row {alu_idx} shape mismatch"
                    )));
                }

                let a_val = witness(payload, *a)?;
                let b_val = witness(payload, *b)?;
                let c_val = if *kind == AluOpKind::BoolCheck {
                    a_val
                } else if let Some(wid) = c {
                    witness(payload, *wid)?
                } else {
                    EF::ZERO
                };
                let out_val = witness(payload, *out)?;
                if row.values != [a_val, b_val, c_val, out_val] {
                    return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                        "ALU row {alu_idx} witness bus mismatch"
                    )));
                }

                let ok = match kind {
                    AluOpKind::Add => a_val + b_val == out_val,
                    AluOpKind::Mul => a_val * b_val == out_val,
                    AluOpKind::BoolCheck => {
                        a_val * (a_val - EF::ONE) == EF::ZERO && out_val == a_val
                    }
                    AluOpKind::MulAdd => a_val * b_val + c_val == out_val,
                    AluOpKind::HornerAcc => {
                        let acc = if let Some(acc) = intermediate_out {
                            witness(payload, *acc)?
                        } else if alu_idx > 0 {
                            payload.alu_rows[alu_idx - 1].values[3]
                        } else {
                            EF::ZERO
                        };
                        acc * b_val + c_val - a_val == out_val
                    }
                };
                if !ok {
                    return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                        "ALU row {alu_idx} arithmetic failed"
                    )));
                }
                alu_idx += 1;
            }
            Op::Hint { .. } | Op::NonPrimitiveOpWithExecutor { .. } => {}
        }
    }

    if const_idx != payload.const_rows.len() {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "extra const rows".to_string(),
        ));
    }
    if public_idx != payload.public_rows.len() {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "extra public rows".to_string(),
        ));
    }
    if alu_idx != payload.alu_rows.len() {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "extra ALU rows".to_string(),
        ));
    }

    Ok(())
}

fn validate_recompose_tables<F, EF>(
    circuit: &Circuit<EF>,
    payload: &WhirNativeTracePayload<F, EF>,
) -> Result<(), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F> + ExtensionField<BabyBear> + Send + Sync,
{
    for table in &payload.recompose_tables {
        let ops = npo_ops_for_type(circuit, &table.op_type);
        if ops.len() != table.rows.len() {
            return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                "recompose table {} row count mismatch: expected {}, got {}",
                table.op_type,
                ops.len(),
                table.rows.len()
            )));
        }

        for (row_index, (op, row)) in ops.iter().zip(&table.rows).enumerate() {
            let Op::NonPrimitiveOpWithExecutor {
                inputs, outputs, ..
            } = op
            else {
                unreachable!();
            };
            if inputs.len() != 1 || outputs.len() != 1 || outputs[0].len() != 1 {
                return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                    "recompose op {row_index} has malformed IO"
                )));
            }
            let input_wids = inputs[0].iter().map(|wid| wid.0).collect::<Vec<_>>();
            if row.input_wids != input_wids || row.output_wid != outputs[0][0].0 {
                return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                    "recompose row {row_index} witness ids mismatch"
                )));
            }
            for (&wid, &value) in inputs[0].iter().zip(&row.values) {
                if witness(payload, wid)? != EF::from(value) {
                    return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                        "recompose row {row_index} input WitnessId({}) mismatch",
                        wid.0
                    )));
                }
            }
            let recomposed = EF::from_basis_coefficients_slice(&row.values).ok_or_else(|| {
                WhirNativeCircuitError::ConstraintViolation(format!(
                    "recompose row {row_index} has wrong coefficient count"
                ))
            })?;
            if witness(payload, outputs[0][0])? != recomposed {
                return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                    "recompose row {row_index} output mismatch"
                )));
            }
        }
    }
    Ok(())
}

fn validate_poseidon2_tables<F, EF, PoseidonEval>(
    circuit: &Circuit<EF>,
    payload: &WhirNativeTracePayload<F, EF>,
    poseidon_eval: &PoseidonEval,
) -> Result<(), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F> + ExtensionField<BabyBear> + Send + Sync,
    PoseidonEval: Fn(Poseidon2Config, &[EF]) -> Option<Vec<EF>>,
{
    for table in &payload.poseidon2_tables {
        let ops = npo_ops_for_type(circuit, &table.op_type);
        if ops.len() != table.rows.len() {
            return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                "Poseidon2 table {} row count mismatch: expected {}, got {}",
                table.op_type,
                ops.len(),
                table.rows.len()
            )));
        }

        let mut last_normal: Option<Vec<EF>> = None;
        let mut last_merkle: Option<Vec<EF>> = None;

        for (row_index, (op, row)) in ops.iter().zip(&table.rows).enumerate() {
            validate_poseidon2_row_shape(table.config, row_index, row)?;
            validate_poseidon2_op_flags(row_index, op, row)?;

            let state = poseidon2_state_from_base::<F, EF>(table.config, &row.input_values)?;
            let mut pre_swap_state = state.clone();
            let rate_ext = table.config.rate_ext();
            if row.merkle_path && row.mmcs_bit {
                for i in 0..rate_ext {
                    pre_swap_state.swap(i, rate_ext + i);
                }
            }

            let Op::NonPrimitiveOpWithExecutor {
                inputs, outputs, ..
            } = op
            else {
                unreachable!();
            };
            validate_poseidon2_bus(
                table.config,
                row_index,
                row,
                inputs,
                outputs,
                &pre_swap_state,
                payload,
            )?;
            validate_poseidon2_chain(
                table.config,
                row_index,
                row,
                &pre_swap_state,
                last_normal.as_deref(),
                last_merkle.as_deref(),
            )?;

            let output = poseidon_eval(table.config, &state).ok_or_else(|| {
                WhirNativeCircuitError::ConstraintViolation(format!(
                    "no Poseidon2 evaluator for {:?}",
                    table.config
                ))
            })?;
            if output.len() != table.config.width_ext() {
                return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                    "Poseidon2 row {row_index} evaluator returned {} limbs, expected {}",
                    output.len(),
                    table.config.width_ext()
                )));
            }
            validate_poseidon2_outputs(table.config, row_index, row, outputs, &output, payload)?;

            if row.merkle_path {
                last_merkle = Some(output);
            } else {
                last_normal = Some(output);
            }
        }
    }
    Ok(())
}

fn validate_poseidon2_row_shape<F>(
    config: Poseidon2Config,
    row_index: usize,
    row: &WhirNativePoseidon2Row<F>,
) -> Result<(), WhirNativeCircuitError>
where
    F: Field,
{
    if row.input_values.len() != config.width()
        || row.in_ctl.len() != config.width_ext()
        || row.input_indices.len() != config.width_ext()
        || row.out_ctl.len() != config.rate_ext()
        || row.output_indices.len() != config.rate_ext()
    {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "Poseidon2 row {row_index} shape mismatch"
        )));
    }
    if row.mmcs_bit && !row.merkle_path {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "Poseidon2 row {row_index} has mmcs_bit outside Merkle mode"
        )));
    }
    Ok(())
}

fn validate_poseidon2_op_flags<F, EF>(
    row_index: usize,
    op: &Op<EF>,
    row: &WhirNativePoseidon2Row<F>,
) -> Result<(), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let Op::NonPrimitiveOpWithExecutor { executor, .. } = op else {
        unreachable!();
    };
    let debug = format!("{executor:?}");
    if debug.contains("new_start: true") && !row.new_start
        || debug.contains("new_start: false") && row.new_start
    {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "Poseidon2 row {row_index} new_start flag mismatch"
        )));
    }
    if debug.contains("merkle_path: true") && !row.merkle_path
        || debug.contains("merkle_path: false") && row.merkle_path
    {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "Poseidon2 row {row_index} merkle_path flag mismatch"
        )));
    }
    Ok(())
}

fn validate_poseidon2_bus<F, EF>(
    config: Poseidon2Config,
    row_index: usize,
    row: &WhirNativePoseidon2Row<F>,
    inputs: &[Vec<WitnessId>],
    outputs: &[Vec<WitnessId>],
    pre_swap_state: &[EF],
    payload: &WhirNativeTracePayload<F, EF>,
) -> Result<(), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let width_ext = config.width_ext();
    if inputs.len() != width_ext + 2 && !(config.d() == 1 && inputs.len() == config.width()) {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "Poseidon2 row {row_index} input arity mismatch"
        )));
    }
    if outputs.len() != config.rate_ext() && outputs.len() != config.width_ext() {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "Poseidon2 row {row_index} output arity mismatch"
        )));
    }

    for i in 0..width_ext {
        match inputs.get(i).map(Vec::as_slice).unwrap_or(&[]) {
            [] => {
                if row.in_ctl[i] {
                    return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                        "Poseidon2 row {row_index} input ctl {i} unexpectedly enabled"
                    )));
                }
            }
            [wid] => {
                if !row.in_ctl[i] || row.input_indices[i] != wid.0 {
                    return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                        "Poseidon2 row {row_index} input ctl {i} mismatch"
                    )));
                }
                if witness(payload, *wid)? != pre_swap_state[i] {
                    return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                        "Poseidon2 row {row_index} input WitnessId({}) mismatch",
                        wid.0
                    )));
                }
            }
            _ => {
                return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                    "Poseidon2 row {row_index} input {i} has multiple witnesses"
                )));
            }
        }
    }

    if inputs.len() >= width_ext + 2 {
        match inputs[width_ext].as_slice() {
            [] if row.mmcs_ctl_enabled => {
                return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                    "Poseidon2 row {row_index} mmcs index ctl unexpectedly enabled"
                )));
            }
            [wid] => {
                if !row.mmcs_ctl_enabled || row.mmcs_index_sum_idx != wid.0 {
                    return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                        "Poseidon2 row {row_index} mmcs index ctl mismatch"
                    )));
                }
                if witness(payload, *wid)? != EF::from(row.mmcs_index_sum) {
                    return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                        "Poseidon2 row {row_index} mmcs index value mismatch"
                    )));
                }
            }
            _ => {}
        }
        match inputs[width_ext + 1].as_slice() {
            [] => {
                if row.merkle_path {
                    return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                        "Poseidon2 row {row_index} missing Merkle direction bit"
                    )));
                }
            }
            [wid] => {
                let bit = witness(payload, *wid)?;
                let expected = if row.mmcs_bit { EF::ONE } else { EF::ZERO };
                if bit != expected {
                    return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                        "Poseidon2 row {row_index} direction bit mismatch"
                    )));
                }
            }
            _ => {
                return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                    "Poseidon2 row {row_index} direction slot has multiple witnesses"
                )));
            }
        }
    }

    Ok(())
}

fn validate_poseidon2_chain<F, EF>(
    config: Poseidon2Config,
    row_index: usize,
    row: &WhirNativePoseidon2Row<F>,
    pre_swap_state: &[EF],
    last_normal: Option<&[EF]>,
    last_merkle: Option<&[EF]>,
) -> Result<(), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let width_ext = config.width_ext();
    let rate_ext = config.rate_ext();

    if row.merkle_path {
        for i in 0..rate_ext {
            if row.in_ctl[i] {
                continue;
            }
            let expected = if row.new_start {
                EF::ZERO
            } else {
                *last_merkle.and_then(|prev| prev.get(i)).ok_or_else(|| {
                    WhirNativeCircuitError::ConstraintViolation(format!(
                        "Poseidon2 row {row_index} missing Merkle chain state"
                    ))
                })?
            };
            if pre_swap_state[i] != expected {
                return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                    "Poseidon2 row {row_index} Merkle chain input {i} mismatch"
                )));
            }
        }
    } else {
        for i in 0..width_ext {
            if row.in_ctl[i] {
                continue;
            }
            let expected = if row.new_start {
                EF::ZERO
            } else {
                *last_normal.and_then(|prev| prev.get(i)).ok_or_else(|| {
                    WhirNativeCircuitError::ConstraintViolation(format!(
                        "Poseidon2 row {row_index} missing normal chain state"
                    ))
                })?
            };
            if pre_swap_state[i] != expected {
                return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                    "Poseidon2 row {row_index} normal chain input {i} mismatch"
                )));
            }
        }
    }
    Ok(())
}

fn validate_poseidon2_outputs<F, EF>(
    config: Poseidon2Config,
    row_index: usize,
    row: &WhirNativePoseidon2Row<F>,
    outputs: &[Vec<WitnessId>],
    output: &[EF],
    payload: &WhirNativeTracePayload<F, EF>,
) -> Result<(), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    for i in 0..config.rate_ext() {
        match outputs.get(i).map(Vec::as_slice).unwrap_or(&[]) {
            [] => {
                if row.out_ctl[i] {
                    return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                        "Poseidon2 row {row_index} output ctl {i} unexpectedly enabled"
                    )));
                }
            }
            [wid] => {
                if !row.out_ctl[i] || row.output_indices[i] != wid.0 {
                    return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                        "Poseidon2 row {row_index} output ctl {i} mismatch"
                    )));
                }
                if witness(payload, *wid)? != output[i] {
                    return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                        "Poseidon2 row {row_index} output WitnessId({}) mismatch",
                        wid.0
                    )));
                }
            }
            _ => {
                return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                    "Poseidon2 row {row_index} output {i} has multiple witnesses"
                )));
            }
        }
    }
    Ok(())
}

fn poseidon2_state_from_base<F, EF>(
    config: Poseidon2Config,
    input_values: &[F],
) -> Result<Vec<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if config.d() == 1 {
        return Ok(input_values.iter().copied().map(EF::from).collect());
    }
    input_values
        .chunks_exact(config.d())
        .map(|chunk| {
            EF::from_basis_coefficients_slice(chunk).ok_or_else(|| {
                WhirNativeCircuitError::ConstraintViolation(format!(
                    "cannot reconstruct {:?} extension limb",
                    config
                ))
            })
        })
        .collect()
}

fn npo_ops_for_type<'a, EF>(circuit: &'a Circuit<EF>, op_type: &str) -> Vec<&'a Op<EF>>
where
    EF: Field,
{
    circuit
        .ops
        .iter()
        .filter(|op| {
            matches!(
                op,
                Op::NonPrimitiveOpWithExecutor { executor, .. }
                    if executor.op_type().as_str() == op_type
            )
        })
        .collect()
}

fn ensure_oracle_only_supported<EF>(circuit: &Circuit<EF>) -> Result<(), WhirNativeCircuitError>
where
    EF: Field,
{
    for op in &circuit.ops {
        match op {
            Op::NonPrimitiveOpWithExecutor { executor, .. } => {
                let op_type = executor.op_type();
                if let Some(config) = poseidon2_config_from_op_type(op_type.as_str()) {
                    if config == Poseidon2Config::BabyBearD4Width16 {
                        continue;
                    }
                    return Err(unsupported_poseidon2_component(op_type.as_str()));
                }
                if !is_recompose_op_type(op_type.as_str()) {
                    return Err(WhirNativeCircuitError::UnsupportedSoundComponent(format!(
                        "non-primitive table `{}` needs an expanded WHIR-native table constraint proof",
                        op_type.as_str()
                    )));
                }
            }
            Op::Const { .. } | Op::Public { .. } | Op::Alu { .. } | Op::Hint { .. } => {}
        }
    }
    Ok(())
}

fn witness<F, EF>(
    payload: &WhirNativeTracePayload<F, EF>,
    wid: WitnessId,
) -> Result<EF, WhirNativeCircuitError>
where
    F: Field,
    EF: Field,
{
    payload
        .witness_values
        .get(wid.0 as usize)
        .copied()
        .ok_or_else(|| {
            WhirNativeCircuitError::ConstraintViolation(format!(
                "WitnessId({}) out of range",
                wid.0
            ))
        })
}

fn build_tables<F, EF>(
    payload: &WhirNativeTracePayload<F, EF>,
    options: WhirNativeCircuitOptions,
) -> Result<Vec<WhirNativeTableData<EF>>, WhirNativeCircuitError>
where
    F: Field + PrimeField64,
    EF: ExtensionField<F>,
{
    let mut tables = Vec::new();

    tables.push(pack_rows(
        WhirNativeTableKind::Witness,
        String::new(),
        payload
            .witness_values
            .iter()
            .enumerate()
            .map(|(i, &value)| vec![ef_from_u64::<F, EF>(i as u64), value])
            .collect(),
        Some(2),
        options,
    ));

    tables.push(pack_rows(
        WhirNativeTableKind::Const,
        String::new(),
        payload
            .const_rows
            .iter()
            .map(|row| vec![ef_from_u64::<F, EF>(row.witness_id as u64), row.value])
            .collect(),
        Some(2),
        options,
    ));

    tables.push(pack_rows(
        WhirNativeTableKind::Public,
        String::new(),
        payload
            .public_rows
            .iter()
            .map(|row| vec![ef_from_u64::<F, EF>(row.witness_id as u64), row.value])
            .collect(),
        Some(2),
        options,
    ));

    tables.push(pack_rows(
        WhirNativeTableKind::Alu,
        String::new(),
        payload
            .alu_rows
            .iter()
            .map(|row| {
                let mut values = vec![ef_from_u64::<F, EF>(row.kind as u64)];
                values.extend(
                    row.indices
                        .iter()
                        .map(|&idx| ef_from_u64::<F, EF>(idx as u64)),
                );
                values.extend_from_slice(&row.values);
                values
            })
            .collect(),
        Some(ALU_WIDTH),
        options,
    ));

    for table in &payload.poseidon2_tables {
        tables.push(build_poseidon2_air_table::<F, EF>(table, options)?);
    }

    for table in &payload.recompose_tables {
        tables.push(pack_rows(
            WhirNativeTableKind::Recompose,
            table.op_type.clone(),
            table
                .rows
                .iter()
                .map(|row| {
                    let mut values = vec![
                        ef_from_u64::<F, EF>(table.kind as u64),
                        ef_from_u64::<F, EF>(row.output_wid as u64),
                        ef_from_u64::<F, EF>(row.input_wids.len() as u64),
                    ];
                    values.extend(
                        row.input_wids
                            .iter()
                            .map(|&wid| ef_from_u64::<F, EF>(wid as u64)),
                    );
                    values.extend(row.values.iter().copied().map(EF::from));
                    values
                })
                .collect(),
            None,
            options,
        ));
    }

    Ok(tables)
}

fn build_poseidon2_air_table<F, EF>(
    table: &WhirNativePoseidon2Table<F>,
    options: WhirNativeCircuitOptions,
) -> Result<WhirNativeTableData<EF>, WhirNativeCircuitError>
where
    F: Field + PrimeField64,
    EF: ExtensionField<F>,
{
    let Poseidon2Config::BabyBearD4Width16 = table.config else {
        return Err(unsupported_poseidon2_component(&table.op_type));
    };
    ensure_babybear_base_field::<F>()?;

    let mut ops = table
        .rows
        .iter()
        .enumerate()
        .map(|(row_index, row)| poseidon2_row_to_babybear_op::<F>(table.config, row_index, row))
        .collect::<Result<Vec<_>, _>>()?;
    let padded_rows = ops.len().max(1).next_power_of_two();
    ops.resize(padded_rows, poseidon2_babybear_d4_width16_filler_row());

    let constants = BabyBearD4Width16::round_constants();
    let air = BabyBearD4Width16::default_air();
    let trace = air.generate_trace_rows(&ops, &constants, 0);
    if trace.width != P2_BB_D4_WIDTH16_AIR_WIDTH {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "Poseidon2 AIR trace width mismatch: expected {}, got {}",
            P2_BB_D4_WIDTH16_AIR_WIDTH, trace.width
        )));
    }
    let rows = trace
        .values
        .chunks_exact(trace.width)
        .map(|row| {
            row.iter()
                .copied()
                .map(babybear_to_ef::<F, EF>)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let rows = append_cyclic_shifted_columns(rows, trace.width)?;

    Ok(pack_rows(
        WhirNativeTableKind::Poseidon2,
        table.op_type.clone(),
        rows,
        Some(P2_BB_D4_WIDTH16_TABLE_WIDTH),
        options,
    ))
}

fn append_cyclic_shifted_columns<EF>(
    rows: Vec<Vec<EF>>,
    width: usize,
) -> Result<Vec<Vec<EF>>, WhirNativeCircuitError>
where
    EF: Field,
{
    if rows.is_empty() {
        return Ok(rows);
    }
    if rows.iter().any(|row| row.len() != width) {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "cannot append shifted columns to ragged rows".to_string(),
        ));
    }
    Ok((0..rows.len())
        .map(|row_index| {
            let mut row = rows[row_index].clone();
            row.extend(rows[(row_index + 1) % rows.len()].iter().copied());
            row
        })
        .collect())
}

fn ensure_babybear_base_field<F>() -> Result<(), WhirNativeCircuitError>
where
    F: Field,
{
    if F::from_u64(BABY_BEAR_MODULUS) != F::ZERO {
        return Err(WhirNativeCircuitError::UnsupportedSoundComponent(
            "BabyBear D4 Width16 Poseidon2 AIR requires BabyBear as the WHIR-native base field"
                .to_string(),
        ));
    }
    Ok(())
}

fn babybear_to_f<F>(value: BabyBear) -> F
where
    F: Field,
{
    F::from_u64(value.as_canonical_u64())
}

fn babybear_to_ef<F, EF>(value: BabyBear) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    EF::from(babybear_to_f::<F>(value))
}

fn f_to_babybear<F>(value: F) -> BabyBear
where
    F: PrimeField64,
{
    BabyBear::from_u64(value.as_canonical_u64())
}

fn poseidon2_row_to_babybear_op<F>(
    config: Poseidon2Config,
    row_index: usize,
    row: &WhirNativePoseidon2Row<F>,
) -> Result<Poseidon2CircuitRow<BabyBear>, WhirNativeCircuitError>
where
    F: Field + PrimeField64,
{
    validate_poseidon2_row_shape(config, row_index, row)?;
    Ok(Poseidon2CircuitRow {
        new_start: row.new_start,
        merkle_path: row.merkle_path,
        mmcs_bit: row.mmcs_bit,
        mmcs_index_sum: f_to_babybear(row.mmcs_index_sum),
        input_values: row
            .input_values
            .iter()
            .copied()
            .map(f_to_babybear)
            .collect(),
        in_ctl: row.in_ctl.clone(),
        input_indices: row.input_indices.clone(),
        out_ctl: row.out_ctl.clone(),
        output_indices: row.output_indices.clone(),
        mmcs_index_sum_idx: row.mmcs_index_sum_idx,
        mmcs_ctl_enabled: row.mmcs_ctl_enabled,
    })
}

fn poseidon2_babybear_d4_width16_filler_row() -> Poseidon2CircuitRow<BabyBear> {
    Poseidon2CircuitRow {
        new_start: true,
        merkle_path: false,
        mmcs_bit: false,
        mmcs_index_sum: BabyBear::ZERO,
        input_values: BabyBear::zero_vec(P2_BB_D4_WIDTH16_WIDTH),
        in_ctl: vec![false; P2_BB_D4_WIDTH16_WIDTH_EXT],
        input_indices: vec![0; P2_BB_D4_WIDTH16_WIDTH_EXT],
        out_ctl: vec![false; P2_BB_D4_WIDTH16_RATE_EXT],
        output_indices: vec![0; P2_BB_D4_WIDTH16_RATE_EXT],
        mmcs_index_sum_idx: 0,
        mmcs_ctl_enabled: false,
    }
}

fn pack_rows<EF>(
    kind: WhirNativeTableKind,
    op_type: String,
    rows: Vec<Vec<EF>>,
    fixed_width: Option<usize>,
    options: WhirNativeCircuitOptions,
) -> WhirNativeTableData<EF>
where
    EF: Field,
{
    let active_rows = rows.len();
    let width = fixed_width.unwrap_or_else(|| rows.iter().map(Vec::len).max().unwrap_or(1).max(1));
    debug_assert!(rows.iter().all(|row| row.len() <= width));
    let padded_width = width.next_power_of_two();
    let mut padded_height = active_rows.max(1).next_power_of_two();
    while padded_width * padded_height < (1usize << options.min_num_variables) {
        padded_height *= 2;
    }
    let num_variables = (padded_width * padded_height).ilog2() as usize;
    let mut values = EF::zero_vec(padded_width * padded_height);
    for (row_idx, row) in rows.iter().enumerate() {
        for (col_idx, &value) in row.iter().enumerate() {
            values[row_idx * padded_width + col_idx] = value;
        }
    }

    WhirNativeTableData {
        metadata: WhirNativeTableMetadata {
            kind,
            op_type,
            width,
            padded_width,
            active_rows,
            padded_height,
            num_variables,
            column_layout_version: TABLE_LAYOUT_VERSION,
        },
        values,
    }
}

fn expected_table_metadata<F, EF>(
    circuit: &Circuit<EF>,
    public_inputs: &[EF],
    options: WhirNativeCircuitOptions,
) -> Result<Vec<WhirNativeTableMetadata>, WhirNativeCircuitError>
where
    F: Field + PrimeField64,
    EF: ExtensionField<F>,
{
    let const_rows = expected_const_rows_from_circuit::<F, EF>(circuit);
    let public_rows = expected_public_rows_from_circuit::<F, EF>(circuit, public_inputs)?;
    let alu_rows = expected_alu_rows_from_circuit(circuit);

    let mut metadata = vec![
        metadata_for_shape(
            WhirNativeTableKind::Witness,
            String::new(),
            circuit.witness_count as usize,
            2,
            options,
        ),
        metadata_for_shape(
            WhirNativeTableKind::Const,
            String::new(),
            const_rows.len(),
            2,
            options,
        ),
        metadata_for_shape(
            WhirNativeTableKind::Public,
            String::new(),
            public_rows.len(),
            2,
            options,
        ),
        metadata_for_shape(
            WhirNativeTableKind::Alu,
            String::new(),
            alu_rows.len(),
            ALU_WIDTH,
            options,
        ),
    ];

    for op_type in expected_nonprimitive_op_types(circuit) {
        if is_recompose_op_type(&op_type) {
            let expected_rows = expected_recompose_rows_from_circuit(circuit, &op_type)?;
            metadata.push(metadata_for_shape(
                WhirNativeTableKind::Recompose,
                op_type,
                expected_rows.len(),
                recompose_width_from_expected_rows(&expected_rows)?,
                options,
            ));
        } else if let Some(config) = poseidon2_config_from_op_type(&op_type) {
            if config != Poseidon2Config::BabyBearD4Width16 {
                return Err(unsupported_poseidon2_component(&op_type));
            }
            ensure_babybear_base_field::<F>()?;
            let expected_rows = expected_poseidon2_rows_from_circuit::<EF>(circuit, &op_type)?;
            metadata.push(metadata_for_shape(
                WhirNativeTableKind::Poseidon2,
                op_type,
                expected_rows.len().max(1).next_power_of_two(),
                P2_BB_D4_WIDTH16_TABLE_WIDTH,
                options,
            ));
        } else {
            return Err(WhirNativeCircuitError::UnsupportedNonPrimitiveTrace(
                op_type,
            ));
        }
    }

    Ok(metadata)
}

fn metadata_for_shape(
    kind: WhirNativeTableKind,
    op_type: String,
    active_rows: usize,
    width: usize,
    options: WhirNativeCircuitOptions,
) -> WhirNativeTableMetadata {
    let width = width.max(1);
    let padded_width = width.next_power_of_two();
    let mut padded_height = active_rows.max(1).next_power_of_two();
    while padded_width * padded_height < (1usize << options.min_num_variables) {
        padded_height *= 2;
    }
    let num_variables = (padded_width * padded_height).ilog2() as usize;
    WhirNativeTableMetadata {
        kind,
        op_type,
        width,
        padded_width,
        active_rows,
        padded_height,
        num_variables,
        column_layout_version: TABLE_LAYOUT_VERSION,
    }
}

/// Number of row-address variables for a packed WHIR-native table.
pub fn whir_native_table_row_variables(metadata: &WhirNativeTableMetadata) -> usize {
    metadata.padded_height.ilog2() as usize
}

/// Number of column-selector variables for a packed WHIR-native table.
pub fn whir_native_table_column_variables(metadata: &WhirNativeTableMetadata) -> usize {
    metadata.padded_width.ilog2() as usize
}

/// Build the packed table-oracle point for one logical column at a row MLE point.
///
/// The table packing is row-major: `row * padded_width + column`. Therefore
/// the point is `[row_bits..., column_bits...]`.
pub fn whir_native_table_column_point<F, EF>(
    metadata: &WhirNativeTableMetadata,
    row_point: &Point<EF>,
    column: usize,
) -> Result<Point<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let row_vars = whir_native_table_row_variables(metadata);
    let col_vars = whir_native_table_column_variables(metadata);
    if row_point.num_variables() != row_vars {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "row point arity mismatch for table {:?}: expected {row_vars}, got {}",
            metadata.kind,
            row_point.num_variables()
        )));
    }
    if column >= metadata.padded_width {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "column {column} out of range for padded width {}",
            metadata.padded_width
        )));
    }

    let mut point = Point::new(row_point.as_slice().to_vec());
    point.extend(&Point::<EF>::hypercube(column, col_vars));
    Ok(point)
}

#[allow(dead_code)]
fn eval_table_column_at_row_point<F, EF>(
    table: &WhirNativeTableData<EF>,
    row_point: &Point<EF>,
    column: usize,
) -> Result<EF, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let point = whir_native_table_column_point::<F, EF>(&table.metadata, row_point, column)?;
    Ok(Poly::new(table.values.clone()).eval_ext::<F>(&point))
}

/// Compose static per-row witness identifiers into a witness-table row point.
///
/// For a source table row point `r`, this evaluates each bit of the static
/// `WitnessId` column as a known multilinear polynomial over `r`. The returned
/// point can be used to open the witness table's value column, proving a
/// witness-bus read/write without revealing the source row.
pub fn whir_native_witness_address_point<F, EF>(
    witness_metadata: &WhirNativeTableMetadata,
    source_padded_height: usize,
    source_row_point: &Point<EF>,
    witness_ids_by_source_row: &[u32],
) -> Result<Point<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if witness_metadata.kind != WhirNativeTableKind::Witness {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "witness address point requires witness table metadata".to_string(),
        ));
    }
    if source_padded_height == 0 || !source_padded_height.is_power_of_two() {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "bad source padded height {source_padded_height}"
        )));
    }
    if source_row_point.num_variables() != source_padded_height.ilog2() as usize {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "source row point arity mismatch: expected {}, got {}",
            source_padded_height.ilog2(),
            source_row_point.num_variables()
        )));
    }
    if witness_ids_by_source_row.len() > source_padded_height {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "too many witness ids: {} for source height {source_padded_height}",
            witness_ids_by_source_row.len()
        )));
    }

    let witness_row_vars = whir_native_table_row_variables(witness_metadata);
    let mut coords = Vec::with_capacity(witness_row_vars);
    for bit_index in 0..witness_row_vars {
        let shift = witness_row_vars - 1 - bit_index;
        let mut bit_evals = EF::zero_vec(source_padded_height);
        for (row, &wid) in witness_ids_by_source_row.iter().enumerate() {
            let bit = (wid >> shift) & 1 == 1;
            bit_evals[row] = ef_from_bool::<F, EF>(bit);
        }
        coords.push(Poly::new(bit_evals).eval_ext::<F>(source_row_point));
    }
    Ok(Point::new(coords))
}

/// Build the packed witness-table value-column opening point for a source row.
pub fn whir_native_witness_value_opening_point<F, EF>(
    witness_metadata: &WhirNativeTableMetadata,
    source_padded_height: usize,
    source_row_point: &Point<EF>,
    witness_ids_by_source_row: &[u32],
) -> Result<Point<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let address = whir_native_witness_address_point::<F, EF>(
        witness_metadata,
        source_padded_height,
        source_row_point,
        witness_ids_by_source_row,
    )?;
    whir_native_table_column_point::<F, EF>(witness_metadata, &address, 1)
}

const WITNESS_TABLE_INDEX: usize = 0;
const WITNESS_LOCAL_DEGREE: usize = 3;
const ALU_LOCAL_DEGREE: usize = 4;
const WITNESS_WIDTH: usize = 2;
const WITNESS_COLUMNS: [usize; WITNESS_WIDTH] = [0, 1];
const MISSING_WITNESS_ID: u32 = u32::MAX;
const ALU_WIDTH: usize = 9;
const ALU_SHAPE_WIDTH: usize = 5;
const ALU_COLUMNS: [usize; ALU_WIDTH] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
const ALU_WITNESS_PORTS: [usize; 5] = [0, 1, 2, 3, 4];
const ALU_ACC_WITNESS_PORT: usize = 4;
const P2_BB_D4_WIDTH16_D: usize = 4;
const P2_BB_D4_WIDTH16_WIDTH: usize = 16;
const P2_BB_D4_WIDTH16_WIDTH_EXT: usize = 4;
const P2_BB_D4_WIDTH16_RATE_EXT: usize = 2;
const P2_BB_D4_WIDTH16_PERM_WIDTH: usize = 298;
const P2_BB_D4_WIDTH16_OUTPUT_OFFSET: usize = 282;
const P2_BB_D4_WIDTH16_MMCS_INDEX_SUM_COL: usize = P2_BB_D4_WIDTH16_PERM_WIDTH + 1;
const P2_BB_D4_WIDTH16_AIR_WIDTH: usize = P2_BB_D4_WIDTH16_PERM_WIDTH + 2;
const P2_BB_D4_WIDTH16_SHIFTED_OFFSET: usize = P2_BB_D4_WIDTH16_AIR_WIDTH;
const P2_BB_D4_WIDTH16_TABLE_WIDTH: usize = P2_BB_D4_WIDTH16_AIR_WIDTH * 2;
const P2_BB_D4_WIDTH16_PREPROCESSED_WIDTH: usize = 24;
const P2_BB_D4_WIDTH16_WITNESS_PORTS: usize =
    P2_BB_D4_WIDTH16_WIDTH_EXT + P2_BB_D4_WIDTH16_RATE_EXT + 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WhirNativeExpectedAluRow {
    kind: AluOpKind,
    indices: [u32; 4],
    acc_index: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct WhirNativeExpectedRecomposeRow {
    kind: u8,
    output_wid: u32,
    input_wids: Vec<u32>,
}

#[allow(dead_code)]
fn expected_alu_rows_from_circuit<EF>(circuit: &Circuit<EF>) -> Vec<WhirNativeExpectedAluRow> {
    circuit
        .ops
        .iter()
        .filter_map(|op| {
            let Op::Alu {
                kind,
                a,
                b,
                c,
                out,
                intermediate_out,
            } = op
            else {
                return None;
            };
            Some(WhirNativeExpectedAluRow {
                kind: *kind,
                indices: [a.0, b.0, c.unwrap_or(WitnessId(0)).0, out.0],
                acc_index: intermediate_out.map(|wid| wid.0),
            })
        })
        .collect()
}

fn expected_nonprimitive_op_types<EF>(circuit: &Circuit<EF>) -> Vec<String>
where
    EF: Field,
{
    let mut op_types = circuit
        .ops
        .iter()
        .filter_map(|op| {
            let Op::NonPrimitiveOpWithExecutor { executor, .. } = op else {
                return None;
            };
            Some(executor.op_type().as_str().to_string())
        })
        .collect::<Vec<_>>();
    op_types.sort();
    op_types.dedup();
    op_types
}

fn is_recompose_op_type(op_type: &str) -> bool {
    matches!(op_type, "recompose" | "recompose/coeff")
}

fn recompose_kind_from_op_type(op_type: &str) -> Result<u8, WhirNativeCircuitError> {
    match op_type {
        "recompose" => Ok(recompose_kind_to_tag(RecomposeTraceKind::Standard)),
        "recompose/coeff" => Ok(recompose_kind_to_tag(RecomposeTraceKind::WithCoeffLookups)),
        _ => Err(WhirNativeCircuitError::UnsupportedNonPrimitiveTrace(
            op_type.to_string(),
        )),
    }
}

fn expected_recompose_rows_from_circuit<EF>(
    circuit: &Circuit<EF>,
    op_type: &str,
) -> Result<Vec<WhirNativeExpectedRecomposeRow>, WhirNativeCircuitError>
where
    EF: Field,
{
    let kind = recompose_kind_from_op_type(op_type)?;
    npo_ops_for_type(circuit, op_type)
        .into_iter()
        .enumerate()
        .map(|(row_index, op)| {
            let Op::NonPrimitiveOpWithExecutor {
                inputs, outputs, ..
            } = op
            else {
                unreachable!();
            };
            if inputs.len() != 1 || outputs.len() != 1 || outputs[0].len() != 1 {
                return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                    "recompose op {row_index} has malformed IO"
                )));
            }
            Ok(WhirNativeExpectedRecomposeRow {
                kind,
                output_wid: outputs[0][0].0,
                input_wids: inputs[0].iter().map(|wid| wid.0).collect(),
            })
        })
        .collect()
}

fn expected_poseidon2_rows_from_circuit<EF>(
    circuit: &Circuit<EF>,
    op_type: &str,
) -> Result<Vec<Poseidon2CircuitRow<BabyBear>>, WhirNativeCircuitError>
where
    EF: Field,
{
    let config = poseidon2_config_from_op_type(op_type)
        .ok_or_else(|| WhirNativeCircuitError::UnsupportedNonPrimitiveTrace(op_type.to_string()))?;
    if config != Poseidon2Config::BabyBearD4Width16 {
        return Err(unsupported_poseidon2_component(op_type));
    }

    npo_ops_for_type(circuit, op_type)
        .into_iter()
        .enumerate()
        .map(|(row_index, op)| expected_poseidon2_row_from_op(config, row_index, op))
        .collect()
}

fn expected_poseidon2_row_from_op<EF>(
    config: Poseidon2Config,
    row_index: usize,
    op: &Op<EF>,
) -> Result<Poseidon2CircuitRow<BabyBear>, WhirNativeCircuitError>
where
    EF: Field,
{
    let Op::NonPrimitiveOpWithExecutor {
        inputs,
        outputs,
        executor,
        ..
    } = op
    else {
        unreachable!();
    };
    let width_ext = config.width_ext();
    let rate_ext = config.rate_ext();
    if inputs.len() != width_ext + 2 && !(config.d() == 1 && inputs.len() == config.width()) {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "Poseidon2 op {row_index} input arity mismatch"
        )));
    }
    if outputs.len() != rate_ext && outputs.len() != width_ext {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "Poseidon2 op {row_index} output arity mismatch"
        )));
    }

    let mut in_ctl = vec![false; width_ext];
    let mut input_indices = vec![0; width_ext];
    for i in 0..width_ext {
        match inputs.get(i).map(Vec::as_slice).unwrap_or(&[]) {
            [] => {}
            [wid] => {
                in_ctl[i] = true;
                input_indices[i] = wid.0;
            }
            _ => {
                return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                    "Poseidon2 op {row_index} input {i} has multiple witnesses"
                )));
            }
        }
    }

    let mut out_ctl = vec![false; rate_ext];
    let mut output_indices = vec![0; rate_ext];
    for i in 0..rate_ext {
        match outputs.get(i).map(Vec::as_slice).unwrap_or(&[]) {
            [] => {}
            [wid] => {
                out_ctl[i] = true;
                output_indices[i] = wid.0;
            }
            _ => {
                return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                    "Poseidon2 op {row_index} output {i} has multiple witnesses"
                )));
            }
        }
    }

    let (mut mmcs_index_sum_idx, mut mmcs_ctl_enabled) = (0, false);
    if let Some(slot) = inputs.get(width_ext) {
        match slot.as_slice() {
            [] => {}
            [wid] => {
                mmcs_index_sum_idx = wid.0;
                mmcs_ctl_enabled = true;
            }
            _ => {
                return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                    "Poseidon2 op {row_index} MMCS index slot has multiple witnesses"
                )));
            }
        }
    }

    let debug = format!("{executor:?}");
    let new_start = if debug.contains("new_start: true") {
        true
    } else if debug.contains("new_start: false") {
        false
    } else {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "Poseidon2 op {row_index} missing new_start flag"
        )));
    };
    let merkle_path = if debug.contains("merkle_path: true") {
        true
    } else if debug.contains("merkle_path: false") {
        false
    } else {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "Poseidon2 op {row_index} missing merkle_path flag"
        )));
    };

    Ok(Poseidon2CircuitRow {
        new_start,
        merkle_path,
        mmcs_bit: false,
        mmcs_index_sum: BabyBear::ZERO,
        input_values: BabyBear::zero_vec(config.width()),
        in_ctl,
        input_indices,
        out_ctl,
        output_indices,
        mmcs_index_sum_idx,
        mmcs_ctl_enabled,
    })
}

fn expected_poseidon2_direction_bit_witness_ids_from_circuit<EF>(
    circuit: &Circuit<EF>,
    op_type: &str,
) -> Result<Vec<u32>, WhirNativeCircuitError>
where
    EF: Field,
{
    let config = poseidon2_config_from_op_type(op_type)
        .ok_or_else(|| WhirNativeCircuitError::UnsupportedNonPrimitiveTrace(op_type.to_string()))?;
    if config != Poseidon2Config::BabyBearD4Width16 {
        return Err(unsupported_poseidon2_component(op_type));
    }

    npo_ops_for_type(circuit, op_type)
        .into_iter()
        .enumerate()
        .map(|(row_index, op)| {
            let Op::NonPrimitiveOpWithExecutor { inputs, .. } = op else {
                unreachable!();
            };
            let width_ext = config.width_ext();
            match inputs.get(width_ext + 1).map(Vec::as_slice) {
                None | Some([]) => Ok(MISSING_WITNESS_ID),
                Some([wid]) => Ok(wid.0),
                Some(_) => Err(WhirNativeCircuitError::ConstraintViolation(format!(
                    "Poseidon2 op {row_index} direction-bit slot has multiple witnesses"
                ))),
            }
        })
        .collect()
}

fn recompose_width_from_expected_rows(
    rows: &[WhirNativeExpectedRecomposeRow],
) -> Result<usize, WhirNativeCircuitError> {
    let Some(first) = rows.first() else {
        return Ok(1);
    };
    let d = first.input_wids.len();
    if d == 0 {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "recompose rows must have at least one coefficient".to_string(),
        ));
    }
    if rows.iter().any(|row| row.input_wids.len() != d) {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "recompose table rows have inconsistent coefficient counts".to_string(),
        ));
    }
    Ok(3 + d + d)
}

fn expected_const_rows_from_circuit<F, EF>(circuit: &Circuit<EF>) -> Vec<Vec<EF>>
where
    F: Field,
    EF: ExtensionField<F>,
{
    circuit
        .ops
        .iter()
        .filter_map(|op| {
            let Op::Const { out, val } = op else {
                return None;
            };
            Some(vec![ef_from_u64::<F, EF>(out.0 as u64), *val])
        })
        .collect()
}

fn expected_const_witness_ids_from_circuit<EF>(circuit: &Circuit<EF>) -> Vec<u32> {
    circuit
        .ops
        .iter()
        .filter_map(|op| {
            let Op::Const { out, .. } = op else {
                return None;
            };
            Some(out.0)
        })
        .collect()
}

fn expected_public_rows_from_circuit<F, EF>(
    circuit: &Circuit<EF>,
    public_inputs: &[EF],
) -> Result<Vec<Vec<EF>>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    circuit
        .ops
        .iter()
        .filter_map(|op| {
            let Op::Public { out, public_pos } = op else {
                return None;
            };
            Some(
                public_inputs
                    .get(*public_pos)
                    .copied()
                    .ok_or_else(|| {
                        WhirNativeCircuitError::ConstraintViolation(format!(
                            "public input position {public_pos} out of range"
                        ))
                    })
                    .map(|value| vec![ef_from_u64::<F, EF>(out.0 as u64), value]),
            )
        })
        .collect()
}

fn expected_public_witness_ids_from_circuit<EF>(circuit: &Circuit<EF>) -> Vec<u32> {
    circuit
        .ops
        .iter()
        .filter_map(|op| {
            let Op::Public { out, .. } = op else {
                return None;
            };
            Some(out.0)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn prove_table_local_constraints<F, EF, MT, Challenger, MakeChallenger>(
    circuit: &Circuit<EF>,
    public_inputs: &[EF],
    table_index: usize,
    tables: &[WhirNativeTableData<EF>],
    table_commitments: &[WhirNativeTableCommitment<MT::Commitment>],
    public_io_digest: &[F],
    shape_digest: &[F],
    options: WhirNativeCircuitOptions,
    make_challenger: &MakeChallenger,
) -> Result<Option<WhirNativeLocalConstraintProof<EF>>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F> + ExtensionField<BabyBear> + Send + Sync,
    MT: Mmcs<F>,
    MT::Commitment: Clone,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanObserve<F>
        + CanObserve<MT::Commitment>,
    MakeChallenger: Fn() -> Challenger,
{
    let table = tables.get(table_index).ok_or_else(|| {
        WhirNativeCircuitError::ConstraintViolation(format!("missing table {table_index}"))
    })?;
    let witness_table = tables.first().ok_or_else(|| {
        WhirNativeCircuitError::ConstraintViolation("missing witness table".to_string())
    })?;
    let mut challenger = make_challenger();
    observe_circuit_constraint_context(
        &mut challenger,
        public_io_digest,
        shape_digest,
        options,
        table_commitments,
    );
    match table.metadata.kind {
        WhirNativeTableKind::Witness => prove_witness_local_constraints::<F, EF, Challenger>(
            table_index,
            table,
            &mut challenger,
        )
        .map(Some),
        WhirNativeTableKind::Const => {
            let expected_rows = expected_const_rows_from_circuit::<F, EF>(circuit);
            let expected_witness_ids = expected_const_witness_ids_from_circuit(circuit);
            prove_known_rows_local_constraints::<F, EF, Challenger>(
                table_index,
                table,
                witness_table,
                &expected_rows,
                &expected_witness_ids,
                &mut challenger,
            )
            .map(Some)
        }
        WhirNativeTableKind::Public => {
            let expected_rows = expected_public_rows_from_circuit::<F, EF>(circuit, public_inputs)?;
            let expected_witness_ids = expected_public_witness_ids_from_circuit(circuit);
            prove_known_rows_local_constraints::<F, EF, Challenger>(
                table_index,
                table,
                witness_table,
                &expected_rows,
                &expected_witness_ids,
                &mut challenger,
            )
            .map(Some)
        }
        WhirNativeTableKind::Alu => {
            let expected_rows = expected_alu_rows_from_circuit(circuit);
            prove_alu_local_constraints::<F, EF, Challenger>(
                table_index,
                table,
                witness_table,
                &expected_rows,
                &mut challenger,
            )
            .map(Some)
        }
        WhirNativeTableKind::Recompose => {
            let expected_rows =
                expected_recompose_rows_from_circuit(circuit, &table.metadata.op_type)?;
            prove_recompose_local_constraints::<F, EF, Challenger>(
                table_index,
                table,
                witness_table,
                &expected_rows,
                &mut challenger,
            )
            .map(Some)
        }
        WhirNativeTableKind::Poseidon2 => {
            let expected_rows =
                expected_poseidon2_rows_from_circuit(circuit, &table.metadata.op_type)?;
            let direction_bit_witness_ids =
                expected_poseidon2_direction_bit_witness_ids_from_circuit(
                    circuit,
                    &table.metadata.op_type,
                )?;
            prove_poseidon2_air_constraints::<F, EF, Challenger>(
                table_index,
                table,
                witness_table,
                &expected_rows,
                &direction_bit_witness_ids,
                &mut challenger,
            )
            .map(Some)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn verify_table_local_constraints<F, EF, MT, Challenger, MakeChallenger>(
    circuit: &Circuit<EF>,
    public_inputs: &[EF],
    table_index: usize,
    metadata: &[WhirNativeTableMetadata],
    table_commitments: &[WhirNativeTableCommitment<MT::Commitment>],
    public_io_digest: &[F],
    shape_digest: &[F],
    options: WhirNativeCircuitOptions,
    proof: &WhirNativeConstraintSumcheckProof<EF>,
    make_challenger: &MakeChallenger,
) -> Result<Vec<(usize, Point<EF>, EF)>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F> + ExtensionField<BabyBear> + Send + Sync,
    MT: Mmcs<F>,
    MT::Commitment: Clone,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanObserve<F>
        + CanObserve<MT::Commitment>,
    MakeChallenger: Fn() -> Challenger,
{
    let table_metadata = metadata.get(table_index).ok_or_else(|| {
        WhirNativeCircuitError::ConstraintViolation(format!(
            "missing metadata for table {table_index}"
        ))
    })?;
    let witness_metadata = metadata.first().ok_or_else(|| {
        WhirNativeCircuitError::ConstraintViolation("missing witness metadata".to_string())
    })?;
    let mut challenger = make_challenger();
    observe_circuit_constraint_context(
        &mut challenger,
        public_io_digest,
        shape_digest,
        options,
        table_commitments,
    );
    match table_metadata.kind {
        WhirNativeTableKind::Witness => {
            let local_proof = proof.local_proof.as_ref().ok_or_else(|| {
                WhirNativeCircuitError::ConstraintViolation(format!(
                    "missing witness local proof for table {table_index}"
                ))
            })?;
            verify_witness_local_constraints::<F, EF, Challenger>(
                local_proof,
                table_index,
                table_metadata,
                &mut challenger,
            )
        }
        WhirNativeTableKind::Const => {
            let local_proof = proof.local_proof.as_ref().ok_or_else(|| {
                WhirNativeCircuitError::ConstraintViolation(format!(
                    "missing const local proof for table {table_index}"
                ))
            })?;
            let expected_rows = expected_const_rows_from_circuit::<F, EF>(circuit);
            let expected_witness_ids = expected_const_witness_ids_from_circuit(circuit);
            verify_known_rows_local_constraints::<F, EF, Challenger>(
                local_proof,
                table_index,
                table_metadata,
                witness_metadata,
                &expected_rows,
                &expected_witness_ids,
                &mut challenger,
            )
        }
        WhirNativeTableKind::Public => {
            let local_proof = proof.local_proof.as_ref().ok_or_else(|| {
                WhirNativeCircuitError::ConstraintViolation(format!(
                    "missing public local proof for table {table_index}"
                ))
            })?;
            let expected_rows = expected_public_rows_from_circuit::<F, EF>(circuit, public_inputs)?;
            let expected_witness_ids = expected_public_witness_ids_from_circuit(circuit);
            verify_known_rows_local_constraints::<F, EF, Challenger>(
                local_proof,
                table_index,
                table_metadata,
                witness_metadata,
                &expected_rows,
                &expected_witness_ids,
                &mut challenger,
            )
        }
        WhirNativeTableKind::Alu => {
            let local_proof = proof.local_proof.as_ref().ok_or_else(|| {
                WhirNativeCircuitError::ConstraintViolation(format!(
                    "missing ALU local proof for table {table_index}"
                ))
            })?;
            let expected_rows = expected_alu_rows_from_circuit(circuit);
            verify_alu_local_constraints::<F, EF, Challenger>(
                local_proof,
                table_index,
                table_metadata,
                witness_metadata,
                &expected_rows,
                &mut challenger,
            )
        }
        WhirNativeTableKind::Recompose => {
            let local_proof = proof.local_proof.as_ref().ok_or_else(|| {
                WhirNativeCircuitError::ConstraintViolation(format!(
                    "missing recompose local proof for table {table_index}"
                ))
            })?;
            let expected_rows =
                expected_recompose_rows_from_circuit(circuit, &table_metadata.op_type)?;
            verify_recompose_local_constraints::<F, EF, Challenger>(
                local_proof,
                table_index,
                table_metadata,
                witness_metadata,
                &expected_rows,
                &mut challenger,
            )
        }
        WhirNativeTableKind::Poseidon2 => {
            let local_proof = proof.local_proof.as_ref().ok_or_else(|| {
                WhirNativeCircuitError::ConstraintViolation(format!(
                    "missing Poseidon2 AIR local proof for table {table_index}"
                ))
            })?;
            let expected_rows =
                expected_poseidon2_rows_from_circuit(circuit, &table_metadata.op_type)?;
            let direction_bit_witness_ids =
                expected_poseidon2_direction_bit_witness_ids_from_circuit(
                    circuit,
                    &table_metadata.op_type,
                )?;
            verify_poseidon2_air_constraints::<F, EF, Challenger>(
                local_proof,
                table_index,
                table_metadata,
                witness_metadata,
                &expected_rows,
                &direction_bit_witness_ids,
                &mut challenger,
            )
        }
    }
}

fn local_proof_terminal_claims<EF>(
    proof: Option<&WhirNativeLocalConstraintProof<EF>>,
    metadata: &[WhirNativeTableMetadata],
) -> Result<Vec<(usize, Point<EF>, EF)>, WhirNativeCircuitError>
where
    EF: Field,
{
    let Some(proof) = proof else {
        return Ok(Vec::new());
    };
    proof
        .terminal_openings
        .iter()
        .enumerate()
        .map(|(opening_index, claim)| {
            let Some(claim_metadata) = metadata.get(claim.table_index) else {
                return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                    "terminal opening {opening_index} table index out of range"
                )));
            };
            if claim.column >= claim_metadata.padded_width {
                return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                    "terminal opening {opening_index} column out of range"
                )));
            }
            let point = Point::new(claim.point.clone());
            if point.num_variables() != claim_metadata.num_variables {
                return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                    "terminal opening {opening_index} arity mismatch"
                )));
            }
            Ok((claim.table_index, point, claim.value))
        })
        .collect()
}

#[allow(dead_code)]
fn prove_witness_local_constraints<F, EF, Challenger>(
    table_index: usize,
    table: &WhirNativeTableData<EF>,
    challenger: &mut Challenger,
) -> Result<WhirNativeLocalConstraintProof<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<F>,
{
    validate_witness_metadata(&table.metadata)?;
    observe_local_constraint_context::<F, EF, Challenger>(
        challenger,
        table_index,
        &table.metadata,
        WhirNativeLocalConstraintKind::Witness,
        WITNESS_LOCAL_DEGREE,
        table.metadata.active_rows,
    );
    let (constraint_challenge, zerocheck_point) =
        sample_local_constraint_challenges::<F, EF, Challenger>(
            challenger,
            whir_native_table_row_variables(&table.metadata),
        );

    let (sumcheck, terminal_row_point, terminal_claim) = prove_sumcheck::<F, EF, Challenger, _>(
        whir_native_table_row_variables(&table.metadata),
        WITNESS_LOCAL_DEGREE,
        EF::ZERO,
        challenger,
        |round, prefix, t, suffix| {
            let row_point = point_from_prefix_current_suffix::<F, EF>(
                prefix,
                t,
                suffix,
                whir_native_table_row_variables(&table.metadata) - round - 1,
            );
            let constraint = eval_witness_constraint_from_table::<F, EF>(
                table,
                &row_point,
                constraint_challenge,
            )
            .expect("witness local constraint inputs were validated");
            eq_eval_ext(&zerocheck_point, &row_point) * constraint
        },
    )?;
    let terminal_constraint = eval_witness_constraint_from_table::<F, EF>(
        table,
        &terminal_row_point,
        constraint_challenge,
    )?;
    if eq_eval_ext(&zerocheck_point, &terminal_row_point) * terminal_constraint != terminal_claim {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "witness local prover terminal claim is inconsistent".to_string(),
        ));
    }

    let terminal_openings = terminal_column_claims_for_table::<F, EF>(
        table_index,
        table,
        &terminal_row_point,
        &WITNESS_COLUMNS,
    )?;

    Ok(WhirNativeLocalConstraintProof {
        table_index,
        kind: WhirNativeLocalConstraintKind::Witness,
        degree: WITNESS_LOCAL_DEGREE,
        constraint_challenge,
        zerocheck_point: zerocheck_point.as_slice().to_vec(),
        terminal_row_point: terminal_row_point.as_slice().to_vec(),
        terminal_claim,
        sumcheck,
        terminal_openings,
    })
}

#[allow(dead_code)]
fn verify_witness_local_constraints<F, EF, Challenger>(
    proof: &WhirNativeLocalConstraintProof<EF>,
    table_index: usize,
    metadata: &WhirNativeTableMetadata,
    challenger: &mut Challenger,
) -> Result<Vec<(usize, Point<EF>, EF)>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<F>,
{
    validate_witness_metadata(metadata)?;
    verify_local_constraint_header(
        proof,
        table_index,
        WhirNativeLocalConstraintKind::Witness,
        WITNESS_LOCAL_DEGREE,
    )?;
    observe_local_constraint_context::<F, EF, Challenger>(
        challenger,
        table_index,
        metadata,
        WhirNativeLocalConstraintKind::Witness,
        WITNESS_LOCAL_DEGREE,
        metadata.active_rows,
    );
    let (constraint_challenge, zerocheck_point) =
        sample_local_constraint_challenges::<F, EF, Challenger>(
            challenger,
            whir_native_table_row_variables(metadata),
        );
    if proof.constraint_challenge != constraint_challenge
        || proof.zerocheck_point.as_slice() != zerocheck_point.as_slice()
    {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "witness local proof challenge mismatch".to_string(),
        ));
    }

    let (terminal_row_point, terminal_claim) = verify_sumcheck::<F, EF, Challenger>(
        &proof.sumcheck,
        whir_native_table_row_variables(metadata),
        EF::ZERO,
        challenger,
    )?;
    if proof.terminal_row_point.as_slice() != terminal_row_point.as_slice()
        || proof.terminal_claim != terminal_claim
    {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "witness local proof terminal mismatch".to_string(),
        ));
    }
    if proof.terminal_openings.len() != WITNESS_COLUMNS.len() {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "witness terminal opening count mismatch: expected {}, got {}",
            WITNESS_COLUMNS.len(),
            proof.terminal_openings.len()
        )));
    }

    let (column_values, opening_claims) = extract_terminal_column_values::<F, EF>(
        proof,
        metadata,
        &terminal_row_point,
        &WITNESS_COLUMNS,
    )?;
    let constraint = eval_witness_constraint_from_values::<F, EF>(
        metadata,
        &terminal_row_point,
        &column_values,
        constraint_challenge,
    )?;
    let expected_terminal = eq_eval_ext(&zerocheck_point, &terminal_row_point) * constraint;
    if expected_terminal != terminal_claim {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "witness local proof terminal claim is inconsistent".to_string(),
        ));
    }

    Ok(opening_claims
        .into_iter()
        .map(|(point, value)| (table_index, point, value))
        .collect())
}

#[allow(dead_code)]
fn prove_known_rows_local_constraints<F, EF, Challenger>(
    table_index: usize,
    table: &WhirNativeTableData<EF>,
    witness_table: &WhirNativeTableData<EF>,
    expected_rows: &[Vec<EF>],
    expected_witness_ids: &[u32],
    challenger: &mut Challenger,
) -> Result<WhirNativeLocalConstraintProof<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<F>,
{
    validate_known_rows_local_inputs(table, expected_rows, expected_witness_ids)?;
    validate_witness_metadata(&witness_table.metadata)?;
    let degree = known_rows_local_degree(&witness_table.metadata);
    observe_local_constraint_context::<F, EF, Challenger>(
        challenger,
        table_index,
        &table.metadata,
        WhirNativeLocalConstraintKind::KnownRows,
        degree,
        expected_rows.len(),
    );
    observe_expected_row_values::<F, EF, Challenger>(challenger, expected_rows);
    let (constraint_challenge, zerocheck_point) =
        sample_local_constraint_challenges::<F, EF, Challenger>(
            challenger,
            whir_native_table_row_variables(&table.metadata),
        );

    let (sumcheck, terminal_row_point, terminal_claim) = prove_sumcheck::<F, EF, Challenger, _>(
        whir_native_table_row_variables(&table.metadata),
        degree,
        EF::ZERO,
        challenger,
        |round, prefix, t, suffix| {
            let row_point = point_from_prefix_current_suffix::<F, EF>(
                prefix,
                t,
                suffix,
                whir_native_table_row_variables(&table.metadata) - round - 1,
            );
            let constraint = eval_known_rows_constraint_from_table::<F, EF>(
                table,
                witness_table,
                expected_rows,
                expected_witness_ids,
                &row_point,
                constraint_challenge,
            )
            .expect("known-row local constraint inputs were validated");
            eq_eval_ext(&zerocheck_point, &row_point) * constraint
        },
    )?;
    let terminal_constraint = eval_known_rows_constraint_from_table::<F, EF>(
        table,
        witness_table,
        expected_rows,
        expected_witness_ids,
        &terminal_row_point,
        constraint_challenge,
    )?;
    if eq_eval_ext(&zerocheck_point, &terminal_row_point) * terminal_constraint != terminal_claim {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "known-row local prover terminal claim is inconsistent".to_string(),
        ));
    }

    let columns = logical_columns(&table.metadata);
    let mut terminal_openings = terminal_column_claims_for_table::<F, EF>(
        table_index,
        table,
        &terminal_row_point,
        &columns,
    )?;
    terminal_openings.push(terminal_witness_value_claim_for_source_port::<F, EF>(
        WITNESS_TABLE_INDEX,
        witness_table,
        table.metadata.padded_height,
        &terminal_row_point,
        expected_witness_ids,
    )?);

    Ok(WhirNativeLocalConstraintProof {
        table_index,
        kind: WhirNativeLocalConstraintKind::KnownRows,
        degree,
        constraint_challenge,
        zerocheck_point: zerocheck_point.as_slice().to_vec(),
        terminal_row_point: terminal_row_point.as_slice().to_vec(),
        terminal_claim,
        sumcheck,
        terminal_openings,
    })
}

#[allow(dead_code)]
fn verify_known_rows_local_constraints<F, EF, Challenger>(
    proof: &WhirNativeLocalConstraintProof<EF>,
    table_index: usize,
    metadata: &WhirNativeTableMetadata,
    witness_metadata: &WhirNativeTableMetadata,
    expected_rows: &[Vec<EF>],
    expected_witness_ids: &[u32],
    challenger: &mut Challenger,
) -> Result<Vec<(usize, Point<EF>, EF)>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<F>,
{
    validate_known_rows_metadata(metadata, expected_rows, expected_witness_ids)?;
    validate_witness_metadata(witness_metadata)?;
    let degree = known_rows_local_degree(witness_metadata);
    verify_local_constraint_header(
        proof,
        table_index,
        WhirNativeLocalConstraintKind::KnownRows,
        degree,
    )?;
    observe_local_constraint_context::<F, EF, Challenger>(
        challenger,
        table_index,
        metadata,
        WhirNativeLocalConstraintKind::KnownRows,
        degree,
        expected_rows.len(),
    );
    observe_expected_row_values::<F, EF, Challenger>(challenger, expected_rows);
    let (constraint_challenge, zerocheck_point) =
        sample_local_constraint_challenges::<F, EF, Challenger>(
            challenger,
            whir_native_table_row_variables(metadata),
        );
    if proof.constraint_challenge != constraint_challenge
        || proof.zerocheck_point.as_slice() != zerocheck_point.as_slice()
    {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "known-row local proof challenge mismatch".to_string(),
        ));
    }

    let (terminal_row_point, terminal_claim) = verify_sumcheck::<F, EF, Challenger>(
        &proof.sumcheck,
        whir_native_table_row_variables(metadata),
        EF::ZERO,
        challenger,
    )?;
    if proof.terminal_row_point.as_slice() != terminal_row_point.as_slice()
        || proof.terminal_claim != terminal_claim
    {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "known-row local proof terminal mismatch".to_string(),
        ));
    }

    let columns = logical_columns(metadata);
    if proof.terminal_openings.len() != columns.len() + 1 {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "known-row terminal opening count mismatch: expected {}, got {}",
            columns.len() + 1,
            proof.terminal_openings.len()
        )));
    }
    let (column_values, opening_claims) =
        extract_terminal_column_values::<F, EF>(proof, metadata, &terminal_row_point, &columns)?;
    let witness_claim_index = columns.len();
    let (witness_value, witness_opening_claim) = extract_terminal_witness_value::<F, EF>(
        proof,
        witness_metadata,
        witness_claim_index,
        metadata.padded_height,
        &terminal_row_point,
        expected_witness_ids,
    )?;
    let constraint = eval_known_rows_constraint_from_values::<F, EF>(
        metadata,
        expected_rows,
        expected_witness_ids,
        &terminal_row_point,
        &column_values,
        witness_value,
        constraint_challenge,
    )?;
    let expected_terminal = eq_eval_ext(&zerocheck_point, &terminal_row_point) * constraint;
    if expected_terminal != terminal_claim {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "known-row local proof terminal claim is inconsistent".to_string(),
        ));
    }

    let mut all_claims = opening_claims
        .into_iter()
        .map(|(point, value)| (table_index, point, value))
        .collect::<Vec<_>>();
    all_claims.push((
        WITNESS_TABLE_INDEX,
        witness_opening_claim.0,
        witness_opening_claim.1,
    ));
    Ok(all_claims)
}

#[allow(dead_code)]
fn prove_alu_local_constraints<F, EF, Challenger>(
    table_index: usize,
    table: &WhirNativeTableData<EF>,
    witness_table: &WhirNativeTableData<EF>,
    expected_rows: &[WhirNativeExpectedAluRow],
    challenger: &mut Challenger,
) -> Result<WhirNativeLocalConstraintProof<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<F>,
{
    validate_alu_local_inputs(table, expected_rows)?;
    validate_witness_metadata(&witness_table.metadata)?;
    let degree = alu_local_degree(&witness_table.metadata);
    observe_local_constraint_context::<F, EF, Challenger>(
        challenger,
        table_index,
        &table.metadata,
        WhirNativeLocalConstraintKind::Alu,
        degree,
        expected_rows.len(),
    );
    observe_expected_alu_rows::<F, Challenger>(challenger, expected_rows);
    let (constraint_challenge, zerocheck_point) =
        sample_local_constraint_challenges::<F, EF, Challenger>(
            challenger,
            whir_native_table_row_variables(&table.metadata),
        );

    let (sumcheck, terminal_row_point, terminal_claim) = prove_sumcheck::<F, EF, Challenger, _>(
        whir_native_table_row_variables(&table.metadata),
        degree,
        EF::ZERO,
        challenger,
        |round, prefix, t, suffix| {
            let row_point = point_from_prefix_current_suffix::<F, EF>(
                prefix,
                t,
                suffix,
                whir_native_table_row_variables(&table.metadata) - round - 1,
            );
            let constraint = eval_alu_constraint_from_table::<F, EF>(
                table,
                witness_table,
                expected_rows,
                &row_point,
                constraint_challenge,
            )
            .expect("ALU local constraint inputs were validated");
            eq_eval_ext(&zerocheck_point, &row_point) * constraint
        },
    )?;
    let terminal_constraint = eval_alu_constraint_from_table::<F, EF>(
        table,
        witness_table,
        expected_rows,
        &terminal_row_point,
        constraint_challenge,
    )?;
    if eq_eval_ext(&zerocheck_point, &terminal_row_point) * terminal_constraint != terminal_claim {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "ALU local prover terminal claim is inconsistent".to_string(),
        ));
    }

    let mut terminal_openings = terminal_column_claims_for_table::<F, EF>(
        table_index,
        table,
        &terminal_row_point,
        &ALU_COLUMNS,
    )?;
    for port in ALU_WITNESS_PORTS {
        terminal_openings.push(terminal_witness_value_claim_for_source_port::<F, EF>(
            WITNESS_TABLE_INDEX,
            witness_table,
            table.metadata.padded_height,
            &terminal_row_point,
            &alu_port_witness_ids(expected_rows, port),
        )?);
    }

    Ok(WhirNativeLocalConstraintProof {
        table_index,
        kind: WhirNativeLocalConstraintKind::Alu,
        degree,
        constraint_challenge,
        zerocheck_point: zerocheck_point.as_slice().to_vec(),
        terminal_row_point: terminal_row_point.as_slice().to_vec(),
        terminal_claim,
        sumcheck,
        terminal_openings,
    })
}

#[allow(dead_code)]
fn verify_alu_local_constraints<F, EF, Challenger>(
    proof: &WhirNativeLocalConstraintProof<EF>,
    table_index: usize,
    metadata: &WhirNativeTableMetadata,
    witness_metadata: &WhirNativeTableMetadata,
    expected_rows: &[WhirNativeExpectedAluRow],
    challenger: &mut Challenger,
) -> Result<Vec<(usize, Point<EF>, EF)>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<F>,
{
    validate_alu_metadata(metadata, expected_rows)?;
    validate_witness_metadata(witness_metadata)?;
    let degree = alu_local_degree(witness_metadata);
    verify_local_constraint_header(
        proof,
        table_index,
        WhirNativeLocalConstraintKind::Alu,
        degree,
    )?;
    observe_local_constraint_context::<F, EF, Challenger>(
        challenger,
        table_index,
        metadata,
        WhirNativeLocalConstraintKind::Alu,
        degree,
        expected_rows.len(),
    );
    observe_expected_alu_rows::<F, Challenger>(challenger, expected_rows);
    let (constraint_challenge, zerocheck_point) =
        sample_local_constraint_challenges::<F, EF, Challenger>(
            challenger,
            whir_native_table_row_variables(metadata),
        );
    if proof.constraint_challenge != constraint_challenge
        || proof.zerocheck_point.as_slice() != zerocheck_point.as_slice()
    {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "ALU local proof challenge mismatch".to_string(),
        ));
    }

    let (terminal_row_point, terminal_claim) = verify_sumcheck::<F, EF, Challenger>(
        &proof.sumcheck,
        whir_native_table_row_variables(metadata),
        EF::ZERO,
        challenger,
    )?;
    if proof.terminal_row_point.as_slice() != terminal_row_point.as_slice()
        || proof.terminal_claim != terminal_claim
    {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "ALU local proof terminal mismatch".to_string(),
        ));
    }
    if proof.terminal_openings.len() != ALU_COLUMNS.len() + ALU_WITNESS_PORTS.len() {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "ALU terminal opening count mismatch: expected {}, got {}",
            ALU_COLUMNS.len() + ALU_WITNESS_PORTS.len(),
            proof.terminal_openings.len()
        )));
    }

    let (column_values, opening_claims) = extract_terminal_column_values::<F, EF>(
        proof,
        metadata,
        &terminal_row_point,
        &ALU_COLUMNS,
    )?;
    let constraint = eval_alu_constraint_from_values::<F, EF>(
        metadata,
        expected_rows,
        &terminal_row_point,
        &column_values,
        &extract_terminal_alu_witness_values::<F, EF>(
            proof,
            witness_metadata,
            ALU_COLUMNS.len(),
            metadata.padded_height,
            &terminal_row_point,
            expected_rows,
        )?
        .0,
        constraint_challenge,
    )?;
    let expected_terminal = eq_eval_ext(&zerocheck_point, &terminal_row_point) * constraint;
    if expected_terminal != terminal_claim {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "ALU local proof terminal claim is inconsistent".to_string(),
        ));
    }

    let (_, witness_opening_claims) = extract_terminal_alu_witness_values::<F, EF>(
        proof,
        witness_metadata,
        ALU_COLUMNS.len(),
        metadata.padded_height,
        &terminal_row_point,
        expected_rows,
    )?;
    let mut all_claims = opening_claims
        .into_iter()
        .map(|(point, value)| (table_index, point, value))
        .collect::<Vec<_>>();
    all_claims.extend(
        witness_opening_claims
            .into_iter()
            .map(|(point, value)| (WITNESS_TABLE_INDEX, point, value)),
    );
    Ok(all_claims)
}

#[allow(dead_code)]
fn prove_recompose_local_constraints<F, EF, Challenger>(
    table_index: usize,
    table: &WhirNativeTableData<EF>,
    witness_table: &WhirNativeTableData<EF>,
    expected_rows: &[WhirNativeExpectedRecomposeRow],
    challenger: &mut Challenger,
) -> Result<WhirNativeLocalConstraintProof<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<F>,
{
    validate_recompose_local_inputs(table, expected_rows)?;
    validate_witness_metadata(&witness_table.metadata)?;
    let degree = recompose_local_degree(&witness_table.metadata);
    observe_local_constraint_context::<F, EF, Challenger>(
        challenger,
        table_index,
        &table.metadata,
        WhirNativeLocalConstraintKind::Recompose,
        degree,
        expected_rows.len(),
    );
    observe_expected_recompose_rows::<F, Challenger>(challenger, expected_rows);
    let (constraint_challenge, zerocheck_point) =
        sample_local_constraint_challenges::<F, EF, Challenger>(
            challenger,
            whir_native_table_row_variables(&table.metadata),
        );

    let (sumcheck, terminal_row_point, terminal_claim) = prove_sumcheck::<F, EF, Challenger, _>(
        whir_native_table_row_variables(&table.metadata),
        degree,
        EF::ZERO,
        challenger,
        |round, prefix, t, suffix| {
            let row_point = point_from_prefix_current_suffix::<F, EF>(
                prefix,
                t,
                suffix,
                whir_native_table_row_variables(&table.metadata) - round - 1,
            );
            let constraint = eval_recompose_constraint_from_table::<F, EF>(
                table,
                witness_table,
                expected_rows,
                &row_point,
                constraint_challenge,
            )
            .expect("recompose local constraint inputs were validated");
            eq_eval_ext(&zerocheck_point, &row_point) * constraint
        },
    )?;
    let terminal_constraint = eval_recompose_constraint_from_table::<F, EF>(
        table,
        witness_table,
        expected_rows,
        &terminal_row_point,
        constraint_challenge,
    )?;
    if eq_eval_ext(&zerocheck_point, &terminal_row_point) * terminal_constraint != terminal_claim {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "recompose local prover terminal claim is inconsistent".to_string(),
        ));
    }

    let columns = logical_columns(&table.metadata);
    let mut terminal_openings = terminal_column_claims_for_table::<F, EF>(
        table_index,
        table,
        &terminal_row_point,
        &columns,
    )?;
    for port in 0..recompose_degree(expected_rows)? {
        terminal_openings.push(terminal_witness_value_claim_for_source_port::<F, EF>(
            WITNESS_TABLE_INDEX,
            witness_table,
            table.metadata.padded_height,
            &terminal_row_point,
            &recompose_input_witness_ids(expected_rows, port),
        )?);
    }
    terminal_openings.push(terminal_witness_value_claim_for_source_port::<F, EF>(
        WITNESS_TABLE_INDEX,
        witness_table,
        table.metadata.padded_height,
        &terminal_row_point,
        &recompose_output_witness_ids(expected_rows),
    )?);

    Ok(WhirNativeLocalConstraintProof {
        table_index,
        kind: WhirNativeLocalConstraintKind::Recompose,
        degree,
        constraint_challenge,
        zerocheck_point: zerocheck_point.as_slice().to_vec(),
        terminal_row_point: terminal_row_point.as_slice().to_vec(),
        terminal_claim,
        sumcheck,
        terminal_openings,
    })
}

#[allow(dead_code)]
fn verify_recompose_local_constraints<F, EF, Challenger>(
    proof: &WhirNativeLocalConstraintProof<EF>,
    table_index: usize,
    metadata: &WhirNativeTableMetadata,
    witness_metadata: &WhirNativeTableMetadata,
    expected_rows: &[WhirNativeExpectedRecomposeRow],
    challenger: &mut Challenger,
) -> Result<Vec<(usize, Point<EF>, EF)>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<F>,
{
    validate_recompose_metadata(metadata, expected_rows)?;
    validate_witness_metadata(witness_metadata)?;
    let degree = recompose_local_degree(witness_metadata);
    verify_local_constraint_header(
        proof,
        table_index,
        WhirNativeLocalConstraintKind::Recompose,
        degree,
    )?;
    observe_local_constraint_context::<F, EF, Challenger>(
        challenger,
        table_index,
        metadata,
        WhirNativeLocalConstraintKind::Recompose,
        degree,
        expected_rows.len(),
    );
    observe_expected_recompose_rows::<F, Challenger>(challenger, expected_rows);
    let (constraint_challenge, zerocheck_point) =
        sample_local_constraint_challenges::<F, EF, Challenger>(
            challenger,
            whir_native_table_row_variables(metadata),
        );
    if proof.constraint_challenge != constraint_challenge
        || proof.zerocheck_point.as_slice() != zerocheck_point.as_slice()
    {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "recompose local proof challenge mismatch".to_string(),
        ));
    }

    let (terminal_row_point, terminal_claim) = verify_sumcheck::<F, EF, Challenger>(
        &proof.sumcheck,
        whir_native_table_row_variables(metadata),
        EF::ZERO,
        challenger,
    )?;
    if proof.terminal_row_point.as_slice() != terminal_row_point.as_slice()
        || proof.terminal_claim != terminal_claim
    {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "recompose local proof terminal mismatch".to_string(),
        ));
    }

    let d = recompose_degree(expected_rows)?;
    let columns = logical_columns(metadata);
    if proof.terminal_openings.len() != columns.len() + d + 1 {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "recompose terminal opening count mismatch: expected {}, got {}",
            columns.len() + d + 1,
            proof.terminal_openings.len()
        )));
    }

    let (column_values, opening_claims) =
        extract_terminal_column_values::<F, EF>(proof, metadata, &terminal_row_point, &columns)?;
    let (input_witness_values, input_witness_claims) =
        extract_terminal_recompose_input_witness_values::<F, EF>(
            proof,
            witness_metadata,
            columns.len(),
            metadata.padded_height,
            &terminal_row_point,
            expected_rows,
        )?;
    let (output_witness_value, output_witness_claim) = extract_terminal_witness_value::<F, EF>(
        proof,
        witness_metadata,
        columns.len() + d,
        metadata.padded_height,
        &terminal_row_point,
        &recompose_output_witness_ids(expected_rows),
    )?;

    let constraint = eval_recompose_constraint_from_values::<F, EF>(
        metadata,
        expected_rows,
        &terminal_row_point,
        &column_values,
        &input_witness_values,
        output_witness_value,
        constraint_challenge,
    )?;
    let expected_terminal = eq_eval_ext(&zerocheck_point, &terminal_row_point) * constraint;
    if expected_terminal != terminal_claim {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "recompose local proof terminal claim is inconsistent".to_string(),
        ));
    }

    let mut all_claims = opening_claims
        .into_iter()
        .map(|(point, value)| (table_index, point, value))
        .collect::<Vec<_>>();
    all_claims.extend(
        input_witness_claims
            .into_iter()
            .map(|(point, value)| (WITNESS_TABLE_INDEX, point, value)),
    );
    all_claims.push((
        WITNESS_TABLE_INDEX,
        output_witness_claim.0,
        output_witness_claim.1,
    ));
    Ok(all_claims)
}

#[allow(dead_code)]
fn prove_poseidon2_air_constraints<F, EF, Challenger>(
    table_index: usize,
    table: &WhirNativeTableData<EF>,
    witness_table: &WhirNativeTableData<EF>,
    expected_rows: &[Poseidon2CircuitRow<BabyBear>],
    direction_bit_witness_ids: &[u32],
    challenger: &mut Challenger,
) -> Result<WhirNativeLocalConstraintProof<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F> + ExtensionField<BabyBear> + Send + Sync,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<F>,
{
    validate_poseidon2_air_metadata::<F, EF>(&table.metadata, expected_rows)?;
    validate_poseidon2_direction_bit_witness_ids(expected_rows, direction_bit_witness_ids)?;
    validate_witness_metadata(&witness_table.metadata)?;
    let preprocessed = poseidon2_expected_preprocessed_values::<F, EF>(
        expected_rows,
        table.metadata.padded_height,
    )?;
    let shifted_preprocessed =
        cyclic_shift_row_major_values(&preprocessed, P2_BB_D4_WIDTH16_PREPROCESSED_WIDTH)?;
    let degree = poseidon2_air_local_degree(&table.metadata, &witness_table.metadata);
    observe_local_constraint_context::<F, EF, Challenger>(
        challenger,
        table_index,
        &table.metadata,
        WhirNativeLocalConstraintKind::Poseidon2Air,
        degree,
        expected_rows.len(),
    );
    observe_expected_poseidon2_rows::<F, Challenger>(challenger, expected_rows);
    let (constraint_challenge, zerocheck_point) =
        sample_local_constraint_challenges::<F, EF, Challenger>(
            challenger,
            whir_native_table_row_variables(&table.metadata),
        );

    let (sumcheck, terminal_row_point, terminal_claim) = prove_sumcheck::<F, EF, Challenger, _>(
        whir_native_table_row_variables(&table.metadata),
        degree,
        EF::ZERO,
        challenger,
        |round, prefix, t, suffix| {
            let row_point = point_from_prefix_current_suffix::<F, EF>(
                prefix,
                t,
                suffix,
                whir_native_table_row_variables(&table.metadata) - round - 1,
            );
            let constraint = eval_poseidon2_air_constraint_from_table::<F, EF>(
                table,
                witness_table,
                expected_rows,
                direction_bit_witness_ids,
                &preprocessed,
                &shifted_preprocessed,
                &row_point,
                constraint_challenge,
            )
            .expect("Poseidon2 AIR local constraint inputs were validated");
            eq_eval_ext(&zerocheck_point, &row_point) * constraint
        },
    )?;
    let terminal_constraint = eval_poseidon2_air_constraint_from_table::<F, EF>(
        table,
        witness_table,
        expected_rows,
        direction_bit_witness_ids,
        &preprocessed,
        &shifted_preprocessed,
        &terminal_row_point,
        constraint_challenge,
    )?;
    if eq_eval_ext(&zerocheck_point, &terminal_row_point) * terminal_constraint != terminal_claim {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "Poseidon2 AIR local prover terminal claim is inconsistent".to_string(),
        ));
    }

    let local_columns = poseidon2_main_columns();
    let shifted_columns = poseidon2_shifted_columns();
    let next_columns = poseidon2_main_columns();
    let mut terminal_openings = terminal_poseidon2_transition_claims::<F, EF>(
        table_index,
        table,
        &terminal_row_point,
        &local_columns,
        &shifted_columns,
        &next_columns,
    )?;
    for witness_ids in poseidon2_witness_port_ids(expected_rows, direction_bit_witness_ids) {
        terminal_openings.push(terminal_witness_value_claim_for_source_port::<F, EF>(
            WITNESS_TABLE_INDEX,
            witness_table,
            table.metadata.padded_height,
            &terminal_row_point,
            &witness_ids,
        )?);
    }

    Ok(WhirNativeLocalConstraintProof {
        table_index,
        kind: WhirNativeLocalConstraintKind::Poseidon2Air,
        degree,
        constraint_challenge,
        zerocheck_point: zerocheck_point.as_slice().to_vec(),
        terminal_row_point: terminal_row_point.as_slice().to_vec(),
        terminal_claim,
        sumcheck,
        terminal_openings,
    })
}

#[allow(dead_code)]
fn verify_poseidon2_air_constraints<F, EF, Challenger>(
    proof: &WhirNativeLocalConstraintProof<EF>,
    table_index: usize,
    metadata: &WhirNativeTableMetadata,
    witness_metadata: &WhirNativeTableMetadata,
    expected_rows: &[Poseidon2CircuitRow<BabyBear>],
    direction_bit_witness_ids: &[u32],
    challenger: &mut Challenger,
) -> Result<Vec<(usize, Point<EF>, EF)>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F> + ExtensionField<BabyBear> + Send + Sync,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<F>,
{
    validate_poseidon2_air_metadata::<F, EF>(metadata, expected_rows)?;
    validate_poseidon2_direction_bit_witness_ids(expected_rows, direction_bit_witness_ids)?;
    validate_witness_metadata(witness_metadata)?;
    let degree = poseidon2_air_local_degree(metadata, witness_metadata);
    verify_local_constraint_header(
        proof,
        table_index,
        WhirNativeLocalConstraintKind::Poseidon2Air,
        degree,
    )?;
    observe_local_constraint_context::<F, EF, Challenger>(
        challenger,
        table_index,
        metadata,
        WhirNativeLocalConstraintKind::Poseidon2Air,
        degree,
        expected_rows.len(),
    );
    observe_expected_poseidon2_rows::<F, Challenger>(challenger, expected_rows);
    let (constraint_challenge, zerocheck_point) =
        sample_local_constraint_challenges::<F, EF, Challenger>(
            challenger,
            whir_native_table_row_variables(metadata),
        );
    if proof.constraint_challenge != constraint_challenge
        || proof.zerocheck_point.as_slice() != zerocheck_point.as_slice()
    {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "Poseidon2 AIR local proof challenge mismatch".to_string(),
        ));
    }

    let (terminal_row_point, terminal_claim) = verify_sumcheck::<F, EF, Challenger>(
        &proof.sumcheck,
        whir_native_table_row_variables(metadata),
        EF::ZERO,
        challenger,
    )?;
    if proof.terminal_row_point.as_slice() != terminal_row_point.as_slice()
        || proof.terminal_claim != terminal_claim
    {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "Poseidon2 AIR local proof terminal mismatch".to_string(),
        ));
    }

    let local_columns = poseidon2_main_columns();
    let shifted_columns = poseidon2_shifted_columns();
    let next_columns = poseidon2_main_columns();
    let expected_openings = local_columns.len()
        + shifted_columns.len()
        + next_columns.len()
        + P2_BB_D4_WIDTH16_WITNESS_PORTS;
    if proof.terminal_openings.len() != expected_openings {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "Poseidon2 AIR terminal opening count mismatch: expected {expected_openings}, got {}",
            proof.terminal_openings.len()
        )));
    }

    let (local_values, shifted_values, original_next_values, mut opening_claims) =
        extract_terminal_poseidon2_transition_values::<F, EF>(
            proof,
            metadata,
            &terminal_row_point,
            &local_columns,
            &shifted_columns,
            &next_columns,
        )?;
    let (witness_values, witness_claims) = extract_terminal_poseidon2_witness_values::<F, EF>(
        proof,
        witness_metadata,
        local_columns.len() + shifted_columns.len() + next_columns.len(),
        metadata.padded_height,
        &terminal_row_point,
        expected_rows,
        direction_bit_witness_ids,
    )?;
    let preprocessed =
        poseidon2_expected_preprocessed_values::<F, EF>(expected_rows, metadata.padded_height)?;
    let shifted_preprocessed =
        cyclic_shift_row_major_values(&preprocessed, P2_BB_D4_WIDTH16_PREPROCESSED_WIDTH)?;
    let prep_local = eval_poseidon2_preprocessed_row::<F, EF>(&preprocessed, &terminal_row_point)?;
    let prep_next =
        eval_poseidon2_preprocessed_row::<F, EF>(&shifted_preprocessed, &terminal_row_point)?;
    let constraint = eval_poseidon2_air_constraint_from_values::<F, EF>(
        metadata,
        expected_rows,
        &terminal_row_point,
        &local_values,
        &shifted_values,
        &original_next_values,
        &prep_local,
        &prep_next,
        &witness_values,
        constraint_challenge,
    )?;
    let expected_terminal = eq_eval_ext(&zerocheck_point, &terminal_row_point) * constraint;
    if expected_terminal != terminal_claim {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "Poseidon2 AIR local proof terminal claim is inconsistent".to_string(),
        ));
    }

    let mut all_claims = opening_claims
        .drain(..)
        .map(|(point, value)| (table_index, point, value))
        .collect::<Vec<_>>();
    all_claims.extend(
        witness_claims
            .into_iter()
            .map(|(point, value)| (WITNESS_TABLE_INDEX, point, value)),
    );
    Ok(all_claims)
}

fn validate_poseidon2_air_metadata<F, EF>(
    metadata: &WhirNativeTableMetadata,
    expected_rows: &[Poseidon2CircuitRow<BabyBear>],
) -> Result<(), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if metadata.kind != WhirNativeTableKind::Poseidon2 {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "Poseidon2 AIR proof requires Poseidon2 table, got {:?}",
            metadata.kind
        )));
    }
    if EF::DIMENSION != P2_BB_D4_WIDTH16_D {
        return Err(WhirNativeCircuitError::UnsupportedSoundComponent(format!(
            "BabyBear D4 Width16 Poseidon2 AIR requires extension degree {}, got {}",
            P2_BB_D4_WIDTH16_D,
            EF::DIMENSION
        )));
    }
    if metadata.width != P2_BB_D4_WIDTH16_TABLE_WIDTH {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "Poseidon2 AIR table width mismatch: expected {}, got {}",
            P2_BB_D4_WIDTH16_TABLE_WIDTH, metadata.width
        )));
    }
    if metadata.active_rows != expected_rows.len().max(1).next_power_of_two() {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "Poseidon2 AIR active row mismatch: expected {}, got {}",
            expected_rows.len().max(1).next_power_of_two(),
            metadata.active_rows
        )));
    }
    Ok(())
}

fn validate_poseidon2_direction_bit_witness_ids(
    expected_rows: &[Poseidon2CircuitRow<BabyBear>],
    direction_bit_witness_ids: &[u32],
) -> Result<(), WhirNativeCircuitError> {
    if direction_bit_witness_ids.len() != expected_rows.len() {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "Poseidon2 direction-bit witness id count mismatch: expected {}, got {}",
            expected_rows.len(),
            direction_bit_witness_ids.len()
        )));
    }
    for (row_index, (row, &wid)) in expected_rows
        .iter()
        .zip(direction_bit_witness_ids)
        .enumerate()
    {
        if row.merkle_path && wid == MISSING_WITNESS_ID {
            return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                "Poseidon2 Merkle row {row_index} is missing a direction-bit witness"
            )));
        }
    }
    Ok(())
}

fn poseidon2_air_local_degree(
    metadata: &WhirNativeTableMetadata,
    witness_metadata: &WhirNativeTableMetadata,
) -> usize {
    let witness_vars = whir_native_table_row_variables(witness_metadata);
    // Poseidon2 transition gates use current columns and committed shifted
    // columns evaluated at the same row point, so each table/preprocessed term
    // is multilinear in the sumcheck variable. Shifted-column binding
    // constraints also open the original main columns at `next(x)`, whose
    // degree is bounded by the row-variable count. The final `eq(r, x)` zero
    // check contributes one more degree.
    let row_vars = whir_native_table_row_variables(metadata).max(1);
    let transition_degree = 4;
    let shifted_binding_degree = row_vars + 1;
    // Witness bindings open the witness table at a static-address MLE derived
    // from the source row. Keep the conservative bound used by the direct
    // witness-opening adapter.
    let witness_binding_degree = row_vars + witness_vars + 2;
    transition_degree
        .max(shifted_binding_degree)
        .max(witness_binding_degree)
        .max(4)
}

fn poseidon2_expected_preprocessed_values<F, EF>(
    expected_rows: &[Poseidon2CircuitRow<BabyBear>],
    padded_height: usize,
) -> Result<Vec<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let preprocessed = extract_preprocessed_from_operations::<
        P2_BB_D4_WIDTH16_WIDTH_EXT,
        P2_BB_D4_WIDTH16_RATE_EXT,
        BabyBear,
        BabyBear,
    >(expected_rows, P2_BB_D4_WIDTH16_D as u32, P2_BB_D4_WIDTH16_D);
    let air = BabyBearD4Width16::default_air_with_preprocessed(preprocessed, padded_height);
    let matrix = BaseAir::<BabyBear>::preprocessed_trace(&air).ok_or_else(|| {
        WhirNativeCircuitError::ConstraintViolation(
            "Poseidon2 AIR did not produce preprocessed trace".to_string(),
        )
    })?;
    if matrix.width != P2_BB_D4_WIDTH16_PREPROCESSED_WIDTH || matrix.height() != padded_height {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "Poseidon2 AIR preprocessed shape mismatch".to_string(),
        ));
    }
    Ok(matrix
        .values
        .into_iter()
        .map(babybear_to_ef::<F, EF>)
        .collect())
}

fn cyclic_shift_row_major_values<EF>(
    values: &[EF],
    width: usize,
) -> Result<Vec<EF>, WhirNativeCircuitError>
where
    EF: Field,
{
    if width == 0 || !values.len().is_multiple_of(width) {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "row-major values cannot be cyclically shifted".to_string(),
        ));
    }
    let height = values.len() / width;
    if height == 0 {
        return Ok(Vec::new());
    }
    let mut shifted = Vec::with_capacity(values.len());
    for row in 0..height {
        let next = (row + 1) % height;
        shifted.extend_from_slice(&values[next * width..(next + 1) * width]);
    }
    Ok(shifted)
}

struct WhirNativePoseidon2EvalBuilder<'a, EF> {
    main: RowWindow<'a, EF>,
    preprocessed: RowWindow<'a, EF>,
    is_first_row: EF,
    is_last_row: EF,
    is_transition: EF,
    constraints: Vec<EF>,
    _marker: PhantomData<BabyBear>,
}

impl<'a, EF> WhirNativePoseidon2EvalBuilder<'a, EF> {
    fn new(
        local_values: &'a [EF],
        shifted_values: &'a [EF],
        prep_local: &'a [EF],
        prep_next: &'a [EF],
        is_first_row: EF,
        is_last_row: EF,
        is_transition: EF,
    ) -> Self {
        Self {
            main: RowWindow::from_two_rows(local_values, shifted_values),
            preprocessed: RowWindow::from_two_rows(prep_local, prep_next),
            is_first_row,
            is_last_row,
            is_transition,
            constraints: Vec::new(),
            _marker: PhantomData,
        }
    }
}

impl<'a, EF> AirBuilder for WhirNativePoseidon2EvalBuilder<'a, EF>
where
    EF: ExtensionField<BabyBear> + Send + Sync,
{
    type F = BabyBear;
    type Expr = EF;
    type Var = EF;
    type PreprocessedWindow = RowWindow<'a, EF>;
    type MainWindow = RowWindow<'a, EF>;
    type PublicVar = EF;

    fn main(&self) -> Self::MainWindow {
        self.main
    }

    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        &self.preprocessed
    }

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        assert!(size <= 2, "only two-row windows are supported, got {size}");
        self.is_transition
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.constraints.push(x.into());
    }
}

fn eval_poseidon2_air_constraint_from_table<F, EF>(
    table: &WhirNativeTableData<EF>,
    witness_table: &WhirNativeTableData<EF>,
    expected_rows: &[Poseidon2CircuitRow<BabyBear>],
    direction_bit_witness_ids: &[u32],
    preprocessed: &[EF],
    shifted_preprocessed: &[EF],
    row_point: &Point<EF>,
    alpha: EF,
) -> Result<EF, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F> + ExtensionField<BabyBear> + Send + Sync,
{
    let local_values = (0..P2_BB_D4_WIDTH16_AIR_WIDTH)
        .map(|column| eval_table_column_at_row_point::<F, EF>(table, row_point, column))
        .collect::<Result<Vec<_>, _>>()?;
    let shifted_values = (0..P2_BB_D4_WIDTH16_AIR_WIDTH)
        .map(|column| {
            eval_table_column_at_row_point::<F, EF>(
                table,
                row_point,
                P2_BB_D4_WIDTH16_SHIFTED_OFFSET + column,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let next_row_point = cyclic_next_row_point::<EF>(row_point);
    let original_next_values = (0..P2_BB_D4_WIDTH16_AIR_WIDTH)
        .map(|column| eval_table_column_at_row_point::<F, EF>(table, &next_row_point, column))
        .collect::<Result<Vec<_>, _>>()?;
    let prep_local = eval_poseidon2_preprocessed_row::<F, EF>(preprocessed, row_point)?;
    let prep_next = eval_poseidon2_preprocessed_row::<F, EF>(shifted_preprocessed, row_point)?;
    let witness_values = poseidon2_witness_port_ids(expected_rows, direction_bit_witness_ids)
        .into_iter()
        .map(|witness_ids| {
            eval_witness_table_value_for_source_port::<F, EF>(
                witness_table,
                table.metadata.padded_height,
                row_point,
                &witness_ids,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    eval_poseidon2_air_constraint_from_values::<F, EF>(
        &table.metadata,
        expected_rows,
        row_point,
        &local_values,
        &shifted_values,
        &original_next_values,
        &prep_local,
        &prep_next,
        &witness_values,
        alpha,
    )
}

#[allow(clippy::too_many_arguments)]
fn eval_poseidon2_air_constraint_from_values<F, EF>(
    metadata: &WhirNativeTableMetadata,
    expected_rows: &[Poseidon2CircuitRow<BabyBear>],
    row_point: &Point<EF>,
    local_values: &[EF],
    shifted_values: &[EF],
    original_next_values: &[EF],
    prep_local: &[EF],
    prep_next: &[EF],
    witness_values: &[EF],
    alpha: EF,
) -> Result<EF, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F> + ExtensionField<BabyBear> + Send + Sync,
{
    if local_values.len() != P2_BB_D4_WIDTH16_AIR_WIDTH
        || shifted_values.len() != P2_BB_D4_WIDTH16_AIR_WIDTH
        || original_next_values.len() != P2_BB_D4_WIDTH16_AIR_WIDTH
    {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "Poseidon2 AIR terminal main width mismatch".to_string(),
        ));
    }
    if prep_local.len() != P2_BB_D4_WIDTH16_PREPROCESSED_WIDTH
        || prep_next.len() != P2_BB_D4_WIDTH16_PREPROCESSED_WIDTH
    {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "Poseidon2 AIR terminal preprocessed width mismatch".to_string(),
        ));
    }
    if witness_values.len() != P2_BB_D4_WIDTH16_WITNESS_PORTS {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "Poseidon2 AIR terminal witness port count mismatch".to_string(),
        ));
    }

    let is_first_row = eq_eval_ext(&Point::hypercube(0, row_point.num_variables()), row_point);
    let is_last_row = eq_eval_ext(
        &Point::hypercube(metadata.padded_height - 1, row_point.num_variables()),
        row_point,
    );
    let is_transition = EF::ONE - is_last_row;
    let mut builder = WhirNativePoseidon2EvalBuilder::new(
        local_values,
        shifted_values,
        prep_local,
        prep_next,
        is_first_row,
        is_last_row,
        is_transition,
    );
    let air = BabyBearD4Width16::default_air();
    Air::eval(&air, &mut builder);
    let mut constraints = builder.constraints;
    constraints.extend(poseidon2_witness_binding_constraints::<F, EF>(
        prep_local,
        prep_next,
        local_values,
        witness_values,
    )?);
    constraints.extend(
        shifted_values
            .iter()
            .zip(original_next_values)
            .map(|(&shifted, &original_next)| shifted - original_next),
    );

    let _ = expected_rows;
    Ok(batch_constraints(constraints, alpha))
}

fn poseidon2_witness_binding_constraints<F, EF>(
    prep_local: &[EF],
    prep_next: &[EF],
    local_values: &[EF],
    witness_values: &[EF],
) -> Result<Vec<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let mut constraints = Vec::new();
    let tail = poseidon2_prep_tail_offset();
    let merkle_path = prep_local[tail + 3];
    let not_merkle = EF::ONE - merkle_path;

    for limb in 0..P2_BB_D4_WIDTH16_WIDTH_EXT {
        let mult = prep_local[poseidon2_prep_input_ctl_col(limb)] * not_merkle;
        let witness = witness_values[limb];
        let local_limb = extension_from_base_mle_limbs::<F, EF>(
            &local_values[limb * P2_BB_D4_WIDTH16_D..(limb + 1) * P2_BB_D4_WIDTH16_D],
        );
        constraints.push(mult * (local_limb - witness));
    }

    let output_witness_offset = P2_BB_D4_WIDTH16_WIDTH_EXT;
    for limb in 0..P2_BB_D4_WIDTH16_RATE_EXT {
        let mult = prep_local[poseidon2_prep_output_ctl_col(limb)];
        let witness = witness_values[output_witness_offset + limb];
        let start = P2_BB_D4_WIDTH16_OUTPUT_OFFSET + limb * P2_BB_D4_WIDTH16_D;
        let local_limb = extension_from_base_mle_limbs::<F, EF>(
            &local_values[start..start + P2_BB_D4_WIDTH16_D],
        );
        constraints.push(mult * (local_limb - witness));
    }

    let mmcs_witness = witness_values[P2_BB_D4_WIDTH16_WIDTH_EXT + P2_BB_D4_WIDTH16_RATE_EXT];
    let mmcs_mult = prep_local[tail + 1] * prep_next[tail + 2];
    constraints
        .push(mmcs_mult * (local_values[P2_BB_D4_WIDTH16_MMCS_INDEX_SUM_COL] - mmcs_witness));

    let direction_witness =
        witness_values[P2_BB_D4_WIDTH16_WIDTH_EXT + P2_BB_D4_WIDTH16_RATE_EXT + 1];
    constraints.push(merkle_path * (local_values[P2_BB_D4_WIDTH16_PERM_WIDTH] - direction_witness));
    Ok(constraints)
}

fn extension_from_base_mle_limbs<F, EF>(limbs: &[EF]) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    debug_assert_eq!(limbs.len(), EF::DIMENSION);
    limbs
        .iter()
        .enumerate()
        .fold(EF::ZERO, |acc, (idx, &limb)| {
            let basis = EF::from_basis_coefficients_fn(|j| F::from_bool(j == idx));
            acc + limb * basis
        })
}

fn poseidon2_main_columns() -> Vec<usize> {
    (0..P2_BB_D4_WIDTH16_AIR_WIDTH).collect()
}

fn poseidon2_shifted_columns() -> Vec<usize> {
    (P2_BB_D4_WIDTH16_SHIFTED_OFFSET..P2_BB_D4_WIDTH16_SHIFTED_OFFSET + P2_BB_D4_WIDTH16_AIR_WIDTH)
        .collect()
}

fn terminal_poseidon2_transition_claims<F, EF>(
    table_index: usize,
    table: &WhirNativeTableData<EF>,
    row_point: &Point<EF>,
    local_columns: &[usize],
    shifted_columns: &[usize],
    next_columns: &[usize],
) -> Result<Vec<WhirNativeTerminalColumnClaim<EF>>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let mut claims =
        terminal_column_claims_for_table::<F, EF>(table_index, table, row_point, local_columns)?;
    claims.extend(terminal_column_claims_for_table::<F, EF>(
        table_index,
        table,
        row_point,
        shifted_columns,
    )?);
    let next_row_point = cyclic_next_row_point::<EF>(row_point);
    claims.extend(terminal_column_claims_for_table::<F, EF>(
        table_index,
        table,
        &next_row_point,
        next_columns,
    )?);
    Ok(claims)
}

fn extract_terminal_poseidon2_transition_values<F, EF>(
    proof: &WhirNativeLocalConstraintProof<EF>,
    metadata: &WhirNativeTableMetadata,
    terminal_row_point: &Point<EF>,
    local_columns: &[usize],
    shifted_columns: &[usize],
    next_columns: &[usize],
) -> Result<(Vec<EF>, Vec<EF>, Vec<EF>, Vec<(Point<EF>, EF)>), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let mut local_values = EF::zero_vec(P2_BB_D4_WIDTH16_AIR_WIDTH);
    let mut shifted_values = EF::zero_vec(P2_BB_D4_WIDTH16_AIR_WIDTH);
    let mut original_next_values = EF::zero_vec(P2_BB_D4_WIDTH16_AIR_WIDTH);
    let mut claims =
        Vec::with_capacity(local_columns.len() + shifted_columns.len() + next_columns.len());

    for (column_offset, &column) in local_columns.iter().enumerate() {
        let claim = proof.terminal_openings.get(column_offset).ok_or_else(|| {
            WhirNativeCircuitError::ConstraintViolation(format!(
                "missing Poseidon2 local terminal opening {column_offset}"
            ))
        })?;
        if claim.table_index != proof.table_index || claim.column != column {
            return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                "Poseidon2 local terminal opening {column_offset} metadata mismatch"
            )));
        }
        let expected_point =
            whir_native_table_column_point::<F, EF>(metadata, terminal_row_point, column)?;
        if claim.point.as_slice() != expected_point.as_slice() {
            return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                "Poseidon2 local terminal opening {column_offset} point mismatch"
            )));
        }
        local_values[column] = claim.value;
        claims.push((expected_point, claim.value));
    }

    for (column_offset, &column) in shifted_columns.iter().enumerate() {
        let opening_index = local_columns.len() + column_offset;
        let claim = proof.terminal_openings.get(opening_index).ok_or_else(|| {
            WhirNativeCircuitError::ConstraintViolation(format!(
                "missing Poseidon2 shifted terminal opening {opening_index}"
            ))
        })?;
        if claim.table_index != proof.table_index || claim.column != column {
            return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                "Poseidon2 shifted terminal opening {opening_index} metadata mismatch"
            )));
        }
        let expected_point =
            whir_native_table_column_point::<F, EF>(metadata, terminal_row_point, column)?;
        if claim.point.as_slice() != expected_point.as_slice() {
            return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                "Poseidon2 shifted terminal opening {opening_index} point mismatch"
            )));
        }
        shifted_values[column - P2_BB_D4_WIDTH16_SHIFTED_OFFSET] = claim.value;
        claims.push((expected_point, claim.value));
    }

    let next_row_point = cyclic_next_row_point::<EF>(terminal_row_point);
    for (column_offset, &column) in next_columns.iter().enumerate() {
        let opening_index = local_columns.len() + shifted_columns.len() + column_offset;
        let claim = proof.terminal_openings.get(opening_index).ok_or_else(|| {
            WhirNativeCircuitError::ConstraintViolation(format!(
                "missing Poseidon2 next-binding terminal opening {opening_index}"
            ))
        })?;
        if claim.table_index != proof.table_index || claim.column != column {
            return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                "Poseidon2 next-binding terminal opening {opening_index} metadata mismatch"
            )));
        }
        let expected_point =
            whir_native_table_column_point::<F, EF>(metadata, &next_row_point, column)?;
        if claim.point.as_slice() != expected_point.as_slice() {
            return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                "Poseidon2 next-binding terminal opening {opening_index} point mismatch"
            )));
        }
        original_next_values[column] = claim.value;
        claims.push((expected_point, claim.value));
    }

    Ok((local_values, shifted_values, original_next_values, claims))
}

fn extract_terminal_poseidon2_witness_values<F, EF>(
    proof: &WhirNativeLocalConstraintProof<EF>,
    witness_metadata: &WhirNativeTableMetadata,
    opening_offset: usize,
    source_padded_height: usize,
    source_row_point: &Point<EF>,
    expected_rows: &[Poseidon2CircuitRow<BabyBear>],
    direction_bit_witness_ids: &[u32],
) -> Result<(Vec<EF>, Vec<(Point<EF>, EF)>), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let mut values = Vec::with_capacity(P2_BB_D4_WIDTH16_WITNESS_PORTS);
    let mut claims = Vec::with_capacity(P2_BB_D4_WIDTH16_WITNESS_PORTS);
    for (port, witness_ids) in poseidon2_witness_port_ids(expected_rows, direction_bit_witness_ids)
        .into_iter()
        .enumerate()
    {
        let (value, claim) = extract_terminal_witness_value::<F, EF>(
            proof,
            witness_metadata,
            opening_offset + port,
            source_padded_height,
            source_row_point,
            &witness_ids,
        )?;
        values.push(value);
        claims.push(claim);
    }
    Ok((values, claims))
}

fn eval_poseidon2_preprocessed_row<F, EF>(
    preprocessed: &[EF],
    row_point: &Point<EF>,
) -> Result<Vec<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    (0..P2_BB_D4_WIDTH16_PREPROCESSED_WIDTH)
        .map(|column| {
            eval_row_major_column_at_row_point::<F, EF>(
                preprocessed,
                P2_BB_D4_WIDTH16_PREPROCESSED_WIDTH,
                row_point,
                column,
            )
        })
        .collect()
}

fn eval_row_major_column_at_row_point<F, EF>(
    values: &[EF],
    width: usize,
    row_point: &Point<EF>,
    column: usize,
) -> Result<EF, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if column >= width {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "column {column} out of row-major width {width}"
        )));
    }
    if !values.len().is_multiple_of(width) {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "row-major values length is not divisible by width".to_string(),
        ));
    }
    let height = values.len() / width;
    validate_row_point(height, row_point)?;
    let column_values = values
        .chunks_exact(width)
        .map(|row| row[column])
        .collect::<Vec<_>>();
    Ok(Poly::new(column_values).eval_ext::<F>(row_point))
}

fn cyclic_next_row_point<EF>(row_point: &Point<EF>) -> Point<EF>
where
    EF: Field,
{
    let bits = row_point.as_slice();
    let mut next = Vec::with_capacity(bits.len());
    for i in 0..bits.len() {
        let carry = bits[i + 1..].iter().copied().product::<EF>();
        let bit = bits[i];
        next.push(bit + carry - bit.double() * carry);
    }
    Point::new(next)
}

fn poseidon2_witness_port_ids(
    expected_rows: &[Poseidon2CircuitRow<BabyBear>],
    direction_bit_witness_ids: &[u32],
) -> Vec<Vec<u32>> {
    let mut ports = Vec::with_capacity(P2_BB_D4_WIDTH16_WITNESS_PORTS);
    for limb in 0..P2_BB_D4_WIDTH16_WIDTH_EXT {
        ports.push(
            expected_rows
                .iter()
                .map(|row| row.input_indices[limb])
                .collect(),
        );
    }
    for limb in 0..P2_BB_D4_WIDTH16_RATE_EXT {
        ports.push(
            expected_rows
                .iter()
                .map(|row| row.output_indices[limb])
                .collect(),
        );
    }
    ports.push(
        expected_rows
            .iter()
            .map(|row| row.mmcs_index_sum_idx)
            .collect(),
    );
    ports.push(direction_bit_witness_ids.to_vec());
    ports
}

fn observe_expected_poseidon2_rows<F, Challenger>(
    challenger: &mut Challenger,
    expected_rows: &[Poseidon2CircuitRow<BabyBear>],
) where
    F: Field,
    Challenger: CanObserve<F>,
{
    challenger.observe(F::from_u64(expected_rows.len() as u64));
    for row in expected_rows {
        challenger.observe(F::from_bool(row.new_start));
        challenger.observe(F::from_bool(row.merkle_path));
        challenger.observe(F::from_u64(row.input_indices.len() as u64));
        for (ctl, &idx) in row.in_ctl.iter().zip(&row.input_indices) {
            challenger.observe(F::from_bool(*ctl));
            challenger.observe(F::from_u64(idx as u64));
        }
        challenger.observe(F::from_u64(row.output_indices.len() as u64));
        for (ctl, &idx) in row.out_ctl.iter().zip(&row.output_indices) {
            challenger.observe(F::from_bool(*ctl));
            challenger.observe(F::from_u64(idx as u64));
        }
        challenger.observe(F::from_u64(row.mmcs_index_sum_idx as u64));
        challenger.observe(F::from_bool(row.mmcs_ctl_enabled));
    }
}

const fn poseidon2_prep_input_ctl_col(limb: usize) -> usize {
    limb * 4 + 1
}

const fn poseidon2_prep_output_ctl_col(limb: usize) -> usize {
    P2_BB_D4_WIDTH16_WIDTH_EXT * 4 + limb * 2 + 1
}

const fn poseidon2_prep_tail_offset() -> usize {
    P2_BB_D4_WIDTH16_WIDTH_EXT * 4 + P2_BB_D4_WIDTH16_RATE_EXT * 2
}

fn validate_known_rows_local_inputs<EF>(
    table: &WhirNativeTableData<EF>,
    expected_rows: &[Vec<EF>],
    expected_witness_ids: &[u32],
) -> Result<(), WhirNativeCircuitError>
where
    EF: Field,
{
    validate_known_rows_metadata(&table.metadata, expected_rows, expected_witness_ids)
}

fn validate_known_rows_metadata<EF>(
    metadata: &WhirNativeTableMetadata,
    expected_rows: &[Vec<EF>],
    expected_witness_ids: &[u32],
) -> Result<(), WhirNativeCircuitError>
where
    EF: Field,
{
    if !matches!(
        metadata.kind,
        WhirNativeTableKind::Const | WhirNativeTableKind::Public
    ) {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "known-row local proof requires const/public table, got {:?}",
            metadata.kind
        )));
    }
    if metadata.active_rows != expected_rows.len() {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "known-row expected row count mismatch: expected {}, got {}",
            metadata.active_rows,
            expected_rows.len()
        )));
    }
    if metadata.active_rows != expected_witness_ids.len() {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "known-row witness id count mismatch: expected {}, got {}",
            metadata.active_rows,
            expected_witness_ids.len()
        )));
    }
    for (row_index, row) in expected_rows.iter().enumerate() {
        if row.len() != metadata.width {
            return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                "known-row {row_index} width mismatch: expected {}, got {}",
                metadata.width,
                row.len()
            )));
        }
    }
    Ok(())
}

fn validate_alu_local_inputs<EF>(
    table: &WhirNativeTableData<EF>,
    expected_rows: &[WhirNativeExpectedAluRow],
) -> Result<(), WhirNativeCircuitError>
where
    EF: Field,
{
    validate_alu_metadata(&table.metadata, expected_rows)
}

fn validate_alu_metadata(
    metadata: &WhirNativeTableMetadata,
    expected_rows: &[WhirNativeExpectedAluRow],
) -> Result<(), WhirNativeCircuitError> {
    if metadata.kind != WhirNativeTableKind::Alu {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "ALU local proof requires ALU table, got {:?}",
            metadata.kind
        )));
    }
    if metadata.width != ALU_WIDTH {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "ALU local proof requires width {ALU_WIDTH}, got {}",
            metadata.width
        )));
    }
    if metadata.active_rows != expected_rows.len() {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "ALU expected row count mismatch: expected {}, got {}",
            metadata.active_rows,
            expected_rows.len()
        )));
    }
    Ok(())
}

fn validate_recompose_local_inputs<EF>(
    table: &WhirNativeTableData<EF>,
    expected_rows: &[WhirNativeExpectedRecomposeRow],
) -> Result<(), WhirNativeCircuitError>
where
    EF: Field,
{
    validate_recompose_metadata(&table.metadata, expected_rows)
}

fn validate_recompose_metadata(
    metadata: &WhirNativeTableMetadata,
    expected_rows: &[WhirNativeExpectedRecomposeRow],
) -> Result<(), WhirNativeCircuitError> {
    if metadata.kind != WhirNativeTableKind::Recompose {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "recompose local proof requires recompose table, got {:?}",
            metadata.kind
        )));
    }
    if metadata.active_rows != expected_rows.len() {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "recompose expected row count mismatch: expected {}, got {}",
            metadata.active_rows,
            expected_rows.len()
        )));
    }
    let expected_width = recompose_width_from_expected_rows(expected_rows)?;
    if metadata.width != expected_width {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "recompose table width mismatch: expected {expected_width}, got {}",
            metadata.width
        )));
    }
    Ok(())
}

fn validate_witness_metadata(
    metadata: &WhirNativeTableMetadata,
) -> Result<(), WhirNativeCircuitError> {
    if metadata.kind != WhirNativeTableKind::Witness {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "witness local proof requires witness table, got {:?}",
            metadata.kind
        )));
    }
    if metadata.width != WITNESS_WIDTH {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "witness table width mismatch: expected {WITNESS_WIDTH}, got {}",
            metadata.width
        )));
    }
    Ok(())
}

fn known_rows_local_degree(witness_metadata: &WhirNativeTableMetadata) -> usize {
    3.max(whir_native_table_row_variables(witness_metadata) + 2)
}

fn alu_local_degree(witness_metadata: &WhirNativeTableMetadata) -> usize {
    ALU_LOCAL_DEGREE.max(whir_native_table_row_variables(witness_metadata) + 4)
}

fn recompose_local_degree(witness_metadata: &WhirNativeTableMetadata) -> usize {
    3.max(whir_native_table_row_variables(witness_metadata) + 2)
}

fn verify_local_constraint_header<EF>(
    proof: &WhirNativeLocalConstraintProof<EF>,
    table_index: usize,
    kind: WhirNativeLocalConstraintKind,
    degree: usize,
) -> Result<(), WhirNativeCircuitError> {
    if proof.table_index != table_index || proof.kind != kind || proof.degree != degree {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "bad local constraint proof header for table {table_index}"
        )));
    }
    if proof.sumcheck.degree != degree {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "bad local sumcheck degree for table {table_index}"
        )));
    }
    Ok(())
}

fn observe_local_constraint_context<F, EF, Challenger>(
    challenger: &mut Challenger,
    table_index: usize,
    metadata: &WhirNativeTableMetadata,
    kind: WhirNativeLocalConstraintKind,
    degree: usize,
    expected_rows: usize,
) where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: CanObserve<F>,
{
    challenger.observe(F::from_u64(0x5748_4952_4c4f_434c));
    challenger.observe(F::from_u64(table_index as u64));
    challenger.observe(F::from_u64(kind.tag()));
    challenger.observe(F::from_u64(degree as u64));
    challenger.observe(F::from_u64(expected_rows as u64));
    challenger.observe(F::from_u64(metadata.kind.tag()));
    observe_string(challenger, &metadata.op_type);
    challenger.observe(F::from_u64(metadata.width as u64));
    challenger.observe(F::from_u64(metadata.padded_width as u64));
    challenger.observe(F::from_u64(metadata.active_rows as u64));
    challenger.observe(F::from_u64(metadata.padded_height as u64));
    challenger.observe(F::from_u64(metadata.num_variables as u64));
    challenger.observe(F::from_u64(metadata.column_layout_version as u64));
}

fn observe_circuit_constraint_context<F, Comm, Challenger>(
    challenger: &mut Challenger,
    public_io_digest: &[F],
    shape_digest: &[F],
    options: WhirNativeCircuitOptions,
    table_commitments: &[WhirNativeTableCommitment<Comm>],
) where
    F: Field,
    Comm: Clone,
    Challenger: CanObserve<F> + CanObserve<Comm>,
{
    challenger.observe(F::from_u64(0x5748_4952_474c_4f42));
    for &value in public_io_digest {
        challenger.observe(value);
    }
    for &value in shape_digest {
        challenger.observe(value);
    }
    challenger.observe(F::from_u64(options.openings_per_table as u64));
    challenger.observe(F::from_u64(options.min_num_variables as u64));
    challenger.observe(F::from_u64(table_commitments.len() as u64));
    for (table_index, table_commitment) in table_commitments.iter().enumerate() {
        challenger.observe(F::from_u64(table_index as u64));
        let metadata = &table_commitment.metadata;
        challenger.observe(F::from_u64(metadata.kind.tag()));
        observe_string::<F, Challenger>(challenger, &metadata.op_type);
        challenger.observe(F::from_u64(metadata.width as u64));
        challenger.observe(F::from_u64(metadata.padded_width as u64));
        challenger.observe(F::from_u64(metadata.active_rows as u64));
        challenger.observe(F::from_u64(metadata.padded_height as u64));
        challenger.observe(F::from_u64(metadata.num_variables as u64));
        challenger.observe(F::from_u64(metadata.column_layout_version as u64));
        challenger.observe(table_commitment.commitment.clone());
    }
}

fn observe_expected_row_values<F, EF, Challenger>(
    challenger: &mut Challenger,
    expected_rows: &[Vec<EF>],
) where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: CanObserve<F>,
{
    challenger.observe(F::from_u64(expected_rows.len() as u64));
    for row in expected_rows {
        challenger.observe(F::from_u64(row.len() as u64));
        for &value in row {
            for &coeff in value.as_basis_coefficients_slice() {
                challenger.observe(coeff);
            }
        }
    }
}

fn observe_expected_alu_rows<F, Challenger>(
    challenger: &mut Challenger,
    expected_rows: &[WhirNativeExpectedAluRow],
) where
    F: Field,
    Challenger: CanObserve<F>,
{
    challenger.observe(F::from_u64(expected_rows.len() as u64));
    for row in expected_rows {
        challenger.observe(F::from_u64(alu_kind_to_tag(row.kind) as u64));
        for &index in &row.indices {
            challenger.observe(F::from_u64(index as u64));
        }
        challenger.observe(F::from_u64(
            row.acc_index.map_or(u32::MAX, |index| index) as u64
        ));
    }
}

fn observe_expected_recompose_rows<F, Challenger>(
    challenger: &mut Challenger,
    expected_rows: &[WhirNativeExpectedRecomposeRow],
) where
    F: Field,
    Challenger: CanObserve<F>,
{
    challenger.observe(F::from_u64(expected_rows.len() as u64));
    for row in expected_rows {
        challenger.observe(F::from_u64(row.kind as u64));
        challenger.observe(F::from_u64(row.output_wid as u64));
        challenger.observe(F::from_u64(row.input_wids.len() as u64));
        for &wid in &row.input_wids {
            challenger.observe(F::from_u64(wid as u64));
        }
    }
}

fn sample_local_constraint_challenges<F, EF, Challenger>(
    challenger: &mut Challenger,
    row_variables: usize,
) -> (EF, Point<EF>)
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
{
    let constraint_challenge = challenger.sample_algebra_element();
    let row_point = (0..row_variables)
        .map(|_| challenger.sample_algebra_element())
        .collect();
    (constraint_challenge, Point::new(row_point))
}

fn eval_known_rows_constraint_from_table<F, EF>(
    table: &WhirNativeTableData<EF>,
    witness_table: &WhirNativeTableData<EF>,
    expected_rows: &[Vec<EF>],
    expected_witness_ids: &[u32],
    row_point: &Point<EF>,
    alpha: EF,
) -> Result<EF, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let values = logical_columns(&table.metadata)
        .into_iter()
        .map(|column| eval_table_column_at_row_point::<F, EF>(table, row_point, column))
        .collect::<Result<Vec<_>, _>>()?;
    let witness_value = eval_witness_table_value_for_source_port::<F, EF>(
        witness_table,
        table.metadata.padded_height,
        row_point,
        expected_witness_ids,
    )?;
    eval_known_rows_constraint_from_values::<F, EF>(
        &table.metadata,
        expected_rows,
        expected_witness_ids,
        row_point,
        &values,
        witness_value,
        alpha,
    )
}

fn eval_known_rows_constraint_from_values<F, EF>(
    metadata: &WhirNativeTableMetadata,
    expected_rows: &[Vec<EF>],
    expected_witness_ids: &[u32],
    row_point: &Point<EF>,
    values: &[EF],
    witness_value: EF,
    alpha: EF,
) -> Result<EF, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if values.len() != metadata.width {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "known-row terminal width mismatch: expected {}, got {}",
            metadata.width,
            values.len()
        )));
    }
    let active =
        active_selector_eval::<F, EF>(metadata.active_rows, metadata.padded_height, row_point)?;
    let inactive = EF::ONE - active;
    let mut constraints = Vec::with_capacity(metadata.width * 2);
    for (column, &value) in values.iter().enumerate() {
        let expected = known_rows_column_eval::<F, EF>(
            metadata.padded_height,
            expected_rows,
            column,
            row_point,
        )?;
        constraints.push(active * (value - expected));
        constraints.push(inactive * value);
    }
    let source_value = values.get(1).copied().ok_or_else(|| {
        WhirNativeCircuitError::ConstraintViolation("missing known-row value column".to_string())
    })?;
    let witness_id =
        static_u32_column_eval::<F, EF>(metadata.padded_height, expected_witness_ids, row_point)?;
    constraints.push(active * (values[0] - witness_id));
    constraints.push(active * (source_value - witness_value));
    Ok(batch_constraints(constraints, alpha))
}

fn eval_alu_constraint_from_table<F, EF>(
    table: &WhirNativeTableData<EF>,
    witness_table: &WhirNativeTableData<EF>,
    expected_rows: &[WhirNativeExpectedAluRow],
    row_point: &Point<EF>,
    alpha: EF,
) -> Result<EF, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let values = ALU_COLUMNS
        .iter()
        .map(|&column| eval_table_column_at_row_point::<F, EF>(table, row_point, column))
        .collect::<Result<Vec<_>, _>>()?;
    let witness_values = ALU_WITNESS_PORTS
        .iter()
        .map(|&port| {
            eval_witness_table_value_for_source_port::<F, EF>(
                witness_table,
                table.metadata.padded_height,
                row_point,
                &alu_port_witness_ids(expected_rows, port),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    eval_alu_constraint_from_values::<F, EF>(
        &table.metadata,
        expected_rows,
        row_point,
        &values,
        &witness_values,
        alpha,
    )
}

fn eval_alu_constraint_from_values<F, EF>(
    metadata: &WhirNativeTableMetadata,
    expected_rows: &[WhirNativeExpectedAluRow],
    row_point: &Point<EF>,
    values: &[EF],
    witness_values: &[EF],
    alpha: EF,
) -> Result<EF, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if values.len() != ALU_WIDTH {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "ALU terminal width mismatch: expected {ALU_WIDTH}, got {}",
            values.len()
        )));
    }
    if witness_values.len() != ALU_WITNESS_PORTS.len() {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "ALU witness port count mismatch: expected {}, got {}",
            ALU_WITNESS_PORTS.len(),
            witness_values.len()
        )));
    }
    let active =
        active_selector_eval::<F, EF>(metadata.active_rows, metadata.padded_height, row_point)?;
    let inactive = EF::ONE - active;
    let mut constraints = Vec::with_capacity(24);

    for (column, &value) in values.iter().take(ALU_SHAPE_WIDTH).enumerate() {
        let expected = expected_alu_column_eval::<F, EF>(
            metadata.padded_height,
            expected_rows,
            column,
            row_point,
        )?;
        constraints.push(active * (value - expected));
    }
    for &value in values {
        constraints.push(inactive * value);
    }

    let a = values[5];
    let b = values[6];
    let c = values[7];
    let out = values[8];
    let sel_add = expected_alu_selector_eval::<F, EF>(
        metadata.padded_height,
        expected_rows,
        AluOpKind::Add,
        row_point,
    )?;
    let sel_mul = expected_alu_selector_eval::<F, EF>(
        metadata.padded_height,
        expected_rows,
        AluOpKind::Mul,
        row_point,
    )?;
    let sel_bool = expected_alu_selector_eval::<F, EF>(
        metadata.padded_height,
        expected_rows,
        AluOpKind::BoolCheck,
        row_point,
    )?;
    let sel_muladd = expected_alu_selector_eval::<F, EF>(
        metadata.padded_height,
        expected_rows,
        AluOpKind::MulAdd,
        row_point,
    )?;
    let sel_horner = expected_alu_selector_eval::<F, EF>(
        metadata.padded_height,
        expected_rows,
        AluOpKind::HornerAcc,
        row_point,
    )?;

    constraints.push(sel_add * (a + b - out));
    constraints.push(sel_add * c);
    constraints.push(sel_mul * (a * b - out));
    constraints.push(sel_mul * c);
    constraints.push(sel_bool * a * (a - EF::ONE));
    constraints.push(sel_bool * (out - a));
    constraints.push(sel_bool * (c - a));
    constraints.push(sel_muladd * (a * b + c - out));
    constraints.push(sel_horner * (witness_values[ALU_ACC_WITNESS_PORT] * b + c - a - out));
    constraints.push(active * (a - witness_values[0]));
    constraints.push(active * (b - witness_values[1]));
    constraints.push((sel_muladd + sel_horner) * (c - witness_values[2]));
    constraints.push(active * (out - witness_values[3]));

    Ok(batch_constraints(constraints, alpha))
}

fn eval_recompose_constraint_from_table<F, EF>(
    table: &WhirNativeTableData<EF>,
    witness_table: &WhirNativeTableData<EF>,
    expected_rows: &[WhirNativeExpectedRecomposeRow],
    row_point: &Point<EF>,
    alpha: EF,
) -> Result<EF, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let values = logical_columns(&table.metadata)
        .into_iter()
        .map(|column| eval_table_column_at_row_point::<F, EF>(table, row_point, column))
        .collect::<Result<Vec<_>, _>>()?;
    let input_witness_values = (0..recompose_degree(expected_rows)?)
        .map(|port| {
            eval_witness_table_value_for_source_port::<F, EF>(
                witness_table,
                table.metadata.padded_height,
                row_point,
                &recompose_input_witness_ids(expected_rows, port),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let output_witness_value = eval_witness_table_value_for_source_port::<F, EF>(
        witness_table,
        table.metadata.padded_height,
        row_point,
        &recompose_output_witness_ids(expected_rows),
    )?;
    eval_recompose_constraint_from_values::<F, EF>(
        &table.metadata,
        expected_rows,
        row_point,
        &values,
        &input_witness_values,
        output_witness_value,
        alpha,
    )
}

fn eval_recompose_constraint_from_values<F, EF>(
    metadata: &WhirNativeTableMetadata,
    expected_rows: &[WhirNativeExpectedRecomposeRow],
    row_point: &Point<EF>,
    values: &[EF],
    input_witness_values: &[EF],
    output_witness_value: EF,
    alpha: EF,
) -> Result<EF, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let d = recompose_degree(expected_rows)?;
    let expected_width = 3 + d + d;
    if values.len() != expected_width {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "recompose terminal width mismatch: expected {expected_width}, got {}",
            values.len()
        )));
    }
    if input_witness_values.len() != d {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "recompose witness port count mismatch: expected {d}, got {}",
            input_witness_values.len()
        )));
    }
    if d != EF::DIMENSION {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "recompose coefficient count {d} does not match extension dimension {}",
            EF::DIMENSION
        )));
    }

    let active =
        active_selector_eval::<F, EF>(metadata.active_rows, metadata.padded_height, row_point)?;
    let inactive = EF::ONE - active;
    let mut constraints = Vec::with_capacity(values.len() * 2 + d + 1);

    for &value in values {
        constraints.push(inactive * value);
    }

    let shape_width = 3 + d;
    for column in 0..shape_width {
        let expected = expected_recompose_shape_column_eval::<F, EF>(
            metadata.padded_height,
            expected_rows,
            column,
            row_point,
        )?;
        constraints.push(active * (values[column] - expected));
    }

    let value_start = 3 + d;
    for coeff in 0..d {
        constraints.push(active * (values[value_start + coeff] - input_witness_values[coeff]));
    }

    let mut recomposed = EF::ZERO;
    for coeff in 0..d {
        let basis = EF::ith_basis_element(coeff).ok_or_else(|| {
            WhirNativeCircuitError::ConstraintViolation(format!(
                "missing extension basis element {coeff}"
            ))
        })?;
        recomposed += values[value_start + coeff] * basis;
    }
    constraints.push(active * (recomposed - output_witness_value));

    Ok(batch_constraints(constraints, alpha))
}

fn eval_witness_constraint_from_table<F, EF>(
    table: &WhirNativeTableData<EF>,
    row_point: &Point<EF>,
    alpha: EF,
) -> Result<EF, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let values = WITNESS_COLUMNS
        .iter()
        .map(|&column| eval_table_column_at_row_point::<F, EF>(table, row_point, column))
        .collect::<Result<Vec<_>, _>>()?;
    eval_witness_constraint_from_values::<F, EF>(&table.metadata, row_point, &values, alpha)
}

fn eval_witness_constraint_from_values<F, EF>(
    metadata: &WhirNativeTableMetadata,
    row_point: &Point<EF>,
    values: &[EF],
    alpha: EF,
) -> Result<EF, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if values.len() != WITNESS_WIDTH {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "witness terminal width mismatch: expected {WITNESS_WIDTH}, got {}",
            values.len()
        )));
    }
    let active =
        active_selector_eval::<F, EF>(metadata.active_rows, metadata.padded_height, row_point)?;
    let inactive = EF::ONE - active;
    let row_index = row_index_eval::<F, EF>(metadata.padded_height, row_point)?;
    Ok(batch_constraints(
        vec![
            active * (values[0] - row_index),
            inactive * values[0],
            inactive * values[1],
        ],
        alpha,
    ))
}

fn active_selector_eval<F, EF>(
    active_rows: usize,
    padded_height: usize,
    row_point: &Point<EF>,
) -> Result<EF, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    validate_row_point(padded_height, row_point)?;
    if active_rows > padded_height {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "active row count {active_rows} exceeds padded height {padded_height}"
        )));
    }
    let mut evals = vec![EF::ZERO; padded_height];
    for value in evals.iter_mut().take(active_rows) {
        *value = EF::ONE;
    }
    Ok(Poly::new(evals).eval_ext::<F>(row_point))
}

fn known_rows_column_eval<F, EF>(
    padded_height: usize,
    expected_rows: &[Vec<EF>],
    column: usize,
    row_point: &Point<EF>,
) -> Result<EF, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    validate_row_point(padded_height, row_point)?;
    if expected_rows.len() > padded_height {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "too many expected rows: {} for padded height {padded_height}",
            expected_rows.len()
        )));
    }
    let mut evals = vec![EF::ZERO; padded_height];
    for (row_index, row) in expected_rows.iter().enumerate() {
        let Some(&value) = row.get(column) else {
            return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                "expected row {row_index} missing column {column}"
            )));
        };
        evals[row_index] = value;
    }
    Ok(Poly::new(evals).eval_ext::<F>(row_point))
}

fn row_index_eval<F, EF>(
    padded_height: usize,
    row_point: &Point<EF>,
) -> Result<EF, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    validate_row_point(padded_height, row_point)?;
    let evals = (0..padded_height)
        .map(|row| ef_from_u64::<F, EF>(row as u64))
        .collect::<Vec<_>>();
    Ok(Poly::new(evals).eval_ext::<F>(row_point))
}

fn static_u32_column_eval<F, EF>(
    padded_height: usize,
    values: &[u32],
    row_point: &Point<EF>,
) -> Result<EF, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    validate_row_point(padded_height, row_point)?;
    if values.len() > padded_height {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "too many static values: {} for padded height {padded_height}",
            values.len()
        )));
    }
    let mut evals = vec![EF::ZERO; padded_height];
    for (row, &value) in values.iter().enumerate() {
        evals[row] = ef_from_u64::<F, EF>(value as u64);
    }
    Ok(Poly::new(evals).eval_ext::<F>(row_point))
}

fn expected_alu_column_eval<F, EF>(
    padded_height: usize,
    expected_rows: &[WhirNativeExpectedAluRow],
    column: usize,
    row_point: &Point<EF>,
) -> Result<EF, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    validate_row_point(padded_height, row_point)?;
    if column >= ALU_SHAPE_WIDTH {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "ALU expected shape column {column} out of range"
        )));
    }
    if expected_rows.len() > padded_height {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "too many ALU expected rows: {} for padded height {padded_height}",
            expected_rows.len()
        )));
    }
    let mut evals = vec![EF::ZERO; padded_height];
    for (row_index, row) in expected_rows.iter().enumerate() {
        let value = if column == 0 {
            alu_kind_to_tag(row.kind) as u64
        } else {
            row.indices[column - 1] as u64
        };
        evals[row_index] = ef_from_u64::<F, EF>(value);
    }
    Ok(Poly::new(evals).eval_ext::<F>(row_point))
}

fn expected_alu_selector_eval<F, EF>(
    padded_height: usize,
    expected_rows: &[WhirNativeExpectedAluRow],
    kind: AluOpKind,
    row_point: &Point<EF>,
) -> Result<EF, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    validate_row_point(padded_height, row_point)?;
    if expected_rows.len() > padded_height {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "too many ALU expected rows: {} for padded height {padded_height}",
            expected_rows.len()
        )));
    }
    let mut evals = vec![EF::ZERO; padded_height];
    for (row_index, row) in expected_rows.iter().enumerate() {
        evals[row_index] = ef_from_bool::<F, EF>(row.kind == kind);
    }
    Ok(Poly::new(evals).eval_ext::<F>(row_point))
}

fn expected_recompose_shape_column_eval<F, EF>(
    padded_height: usize,
    expected_rows: &[WhirNativeExpectedRecomposeRow],
    column: usize,
    row_point: &Point<EF>,
) -> Result<EF, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    validate_row_point(padded_height, row_point)?;
    let d = recompose_degree(expected_rows)?;
    if column >= 3 + d {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "recompose expected shape column {column} out of range"
        )));
    }
    if expected_rows.len() > padded_height {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "too many recompose expected rows: {} for padded height {padded_height}",
            expected_rows.len()
        )));
    }
    let mut evals = vec![EF::ZERO; padded_height];
    for (row_index, row) in expected_rows.iter().enumerate() {
        let value = match column {
            0 => row.kind as u64,
            1 => row.output_wid as u64,
            2 => row.input_wids.len() as u64,
            _ => row.input_wids[column - 3] as u64,
        };
        evals[row_index] = ef_from_u64::<F, EF>(value);
    }
    Ok(Poly::new(evals).eval_ext::<F>(row_point))
}

fn terminal_column_claims_for_table<F, EF>(
    table_index: usize,
    table: &WhirNativeTableData<EF>,
    row_point: &Point<EF>,
    columns: &[usize],
) -> Result<Vec<WhirNativeTerminalColumnClaim<EF>>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    columns
        .iter()
        .map(|&column| {
            let point =
                whir_native_table_column_point::<F, EF>(&table.metadata, row_point, column)?;
            let value = eval_table_column_at_row_point::<F, EF>(table, row_point, column)?;
            Ok(WhirNativeTerminalColumnClaim {
                table_index,
                column,
                point: point.as_slice().to_vec(),
                value,
            })
        })
        .collect()
}

fn terminal_witness_value_claim_for_source_port<F, EF>(
    witness_table_index: usize,
    witness_table: &WhirNativeTableData<EF>,
    source_padded_height: usize,
    source_row_point: &Point<EF>,
    witness_ids_by_source_row: &[u32],
) -> Result<WhirNativeTerminalColumnClaim<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let point = whir_native_witness_value_opening_point::<F, EF>(
        &witness_table.metadata,
        source_padded_height,
        source_row_point,
        witness_ids_by_source_row,
    )?;
    let value = Poly::new(witness_table.values.clone()).eval_ext::<F>(&point);
    Ok(WhirNativeTerminalColumnClaim {
        table_index: witness_table_index,
        column: 1,
        point: point.as_slice().to_vec(),
        value,
    })
}

fn eval_witness_table_value_for_source_port<F, EF>(
    witness_table: &WhirNativeTableData<EF>,
    source_padded_height: usize,
    source_row_point: &Point<EF>,
    witness_ids_by_source_row: &[u32],
) -> Result<EF, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let point = whir_native_witness_value_opening_point::<F, EF>(
        &witness_table.metadata,
        source_padded_height,
        source_row_point,
        witness_ids_by_source_row,
    )?;
    Ok(Poly::new(witness_table.values.clone()).eval_ext::<F>(&point))
}

fn extract_terminal_column_values<F, EF>(
    proof: &WhirNativeLocalConstraintProof<EF>,
    metadata: &WhirNativeTableMetadata,
    terminal_row_point: &Point<EF>,
    columns: &[usize],
) -> Result<(Vec<EF>, Vec<(Point<EF>, EF)>), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if proof.terminal_openings.len() < columns.len() {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "terminal opening count mismatch: expected at least {}, got {}",
            columns.len(),
            proof.terminal_openings.len()
        )));
    }
    let mut values = Vec::with_capacity(columns.len());
    let mut opening_claims = Vec::with_capacity(columns.len());
    for (opening_index, (&column, claim)) in
        columns.iter().zip(&proof.terminal_openings).enumerate()
    {
        if claim.table_index != proof.table_index || claim.column != column {
            return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                "terminal opening {opening_index} metadata mismatch"
            )));
        }
        let expected_point =
            whir_native_table_column_point::<F, EF>(metadata, terminal_row_point, column)?;
        if claim.point.as_slice() != expected_point.as_slice() {
            return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                "terminal opening {opening_index} point mismatch"
            )));
        }
        values.push(claim.value);
        opening_claims.push((expected_point, claim.value));
    }
    Ok((values, opening_claims))
}

fn extract_terminal_witness_value<F, EF>(
    proof: &WhirNativeLocalConstraintProof<EF>,
    witness_metadata: &WhirNativeTableMetadata,
    opening_index: usize,
    source_padded_height: usize,
    source_row_point: &Point<EF>,
    witness_ids_by_source_row: &[u32],
) -> Result<(EF, (Point<EF>, EF)), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let claim = proof.terminal_openings.get(opening_index).ok_or_else(|| {
        WhirNativeCircuitError::ConstraintViolation(format!(
            "missing witness terminal opening {opening_index}"
        ))
    })?;
    if claim.table_index != WITNESS_TABLE_INDEX || claim.column != 1 {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "witness terminal opening {opening_index} metadata mismatch"
        )));
    }
    let expected_point = whir_native_witness_value_opening_point::<F, EF>(
        witness_metadata,
        source_padded_height,
        source_row_point,
        witness_ids_by_source_row,
    )?;
    if claim.point.as_slice() != expected_point.as_slice() {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "witness terminal opening {opening_index} point mismatch"
        )));
    }
    Ok((claim.value, (expected_point, claim.value)))
}

fn extract_terminal_alu_witness_values<F, EF>(
    proof: &WhirNativeLocalConstraintProof<EF>,
    witness_metadata: &WhirNativeTableMetadata,
    opening_offset: usize,
    source_padded_height: usize,
    source_row_point: &Point<EF>,
    expected_rows: &[WhirNativeExpectedAluRow],
) -> Result<(Vec<EF>, Vec<(Point<EF>, EF)>), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let mut values = Vec::with_capacity(ALU_WITNESS_PORTS.len());
    let mut claims = Vec::with_capacity(ALU_WITNESS_PORTS.len());
    for (port_index, port) in ALU_WITNESS_PORTS.iter().copied().enumerate() {
        let (value, claim) = extract_terminal_witness_value::<F, EF>(
            proof,
            witness_metadata,
            opening_offset + port_index,
            source_padded_height,
            source_row_point,
            &alu_port_witness_ids(expected_rows, port),
        )?;
        values.push(value);
        claims.push(claim);
    }
    Ok((values, claims))
}

fn extract_terminal_recompose_input_witness_values<F, EF>(
    proof: &WhirNativeLocalConstraintProof<EF>,
    witness_metadata: &WhirNativeTableMetadata,
    opening_offset: usize,
    source_padded_height: usize,
    source_row_point: &Point<EF>,
    expected_rows: &[WhirNativeExpectedRecomposeRow],
) -> Result<(Vec<EF>, Vec<(Point<EF>, EF)>), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let d = recompose_degree(expected_rows)?;
    let mut values = Vec::with_capacity(d);
    let mut claims = Vec::with_capacity(d);
    for port in 0..d {
        let (value, claim) = extract_terminal_witness_value::<F, EF>(
            proof,
            witness_metadata,
            opening_offset + port,
            source_padded_height,
            source_row_point,
            &recompose_input_witness_ids(expected_rows, port),
        )?;
        values.push(value);
        claims.push(claim);
    }
    Ok((values, claims))
}

fn alu_port_witness_ids(expected_rows: &[WhirNativeExpectedAluRow], port: usize) -> Vec<u32> {
    expected_rows
        .iter()
        .map(|row| {
            if port == ALU_ACC_WITNESS_PORT {
                row.acc_index.unwrap_or(0)
            } else {
                row.indices[port]
            }
        })
        .collect()
}

fn recompose_degree(
    expected_rows: &[WhirNativeExpectedRecomposeRow],
) -> Result<usize, WhirNativeCircuitError> {
    let Some(first) = expected_rows.first() else {
        return Ok(0);
    };
    let d = first.input_wids.len();
    if d == 0 || expected_rows.iter().any(|row| row.input_wids.len() != d) {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "recompose rows have invalid coefficient counts".to_string(),
        ));
    }
    Ok(d)
}

fn recompose_input_witness_ids(
    expected_rows: &[WhirNativeExpectedRecomposeRow],
    port: usize,
) -> Vec<u32> {
    expected_rows
        .iter()
        .map(|row| row.input_wids.get(port).copied().unwrap_or(0))
        .collect()
}

fn recompose_output_witness_ids(expected_rows: &[WhirNativeExpectedRecomposeRow]) -> Vec<u32> {
    expected_rows.iter().map(|row| row.output_wid).collect()
}

fn validate_row_point<EF>(
    padded_height: usize,
    row_point: &Point<EF>,
) -> Result<(), WhirNativeCircuitError>
where
    EF: Field,
{
    if padded_height == 0 || !padded_height.is_power_of_two() {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "bad padded height {padded_height}"
        )));
    }
    let expected = padded_height.ilog2() as usize;
    if row_point.num_variables() != expected {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "row point arity mismatch: expected {expected}, got {}",
            row_point.num_variables()
        )));
    }
    Ok(())
}

fn logical_columns(metadata: &WhirNativeTableMetadata) -> Vec<usize> {
    (0..metadata.width).collect()
}

fn eq_eval_ext<EF>(left: &Point<EF>, right: &Point<EF>) -> EF
where
    EF: Field,
{
    debug_assert_eq!(left.num_variables(), right.num_variables());
    left.as_slice()
        .iter()
        .zip(right.as_slice())
        .map(|(&l, &r)| (EF::ONE - l) * (EF::ONE - r) + l * r)
        .product()
}

fn batch_constraints<EF>(constraints: Vec<EF>, alpha: EF) -> EF
where
    EF: Field,
{
    let mut acc = EF::ZERO;
    let mut coeff = EF::ONE;
    for constraint in constraints {
        acc += coeff * constraint;
        coeff *= alpha;
    }
    acc
}

#[allow(clippy::too_many_arguments)]
fn verify_table_opening<F, EF, MT, Challenger, Dft, MakeChallenger, const DIGEST_ELEMS: usize>(
    pcs: &WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>,
    metadata: &WhirNativeTableMetadata,
    table_commitment: &WhirNativeTableCommitment<MT::Commitment>,
    opening_proof: &WhirNativeTableOpeningProof<F, EF, MT>,
    table_index: usize,
    local_opening_claims: &[(Point<EF>, EF)],
    public_io_digest: &[F],
    shape_digest: &[F],
    options: WhirNativeCircuitOptions,
    make_challenger: &MakeChallenger,
) -> Result<(), WhirNativeCircuitError>
where
    F: TwoAdicField + Ord + Send + Sync + Serialize + for<'de> Deserialize<'de>,
    EF: ExtensionField<F> + TwoAdicField + Send + Sync + Serialize + for<'de> Deserialize<'de>,
    MT: Mmcs<F>,
    MT::Commitment: Clone + PartialEq + Serialize + for<'de> Deserialize<'de>,
    MT::Proof: Clone + Serialize + for<'de> Deserialize<'de>,
    Challenger:
        FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment> + Clone,
    Dft: TwoAdicSubgroupDft<F>,
    MakeChallenger: Fn() -> Challenger,
{
    if opening_proof.table_index != table_index {
        return Err(WhirNativeCircuitError::OpeningClaimMismatch {
            table_index,
            opening_index: usize::MAX,
        });
    }

    let mut challenger = make_challenger();
    observe_table_context(
        &mut challenger,
        public_io_digest,
        shape_digest,
        table_index,
        metadata,
        options,
    );
    let parsed_commitment = pcs
        .parse_initial_commitment(
            &table_commitment.commitment,
            &opening_proof.proof,
            &mut challenger,
        )
        .map_err(|err| WhirNativeCircuitError::WhirVerificationFailed {
            table_index,
            details: format!("{err:?}"),
        })?;
    let opening_points = sample_table_opening_points(
        &mut challenger,
        metadata.num_variables,
        table_index,
        options.openings_per_table,
    );
    if opening_points.len() + local_opening_claims.len() != opening_proof.opening_claims.len() {
        return Err(WhirNativeCircuitError::OpeningClaimMismatch {
            table_index,
            opening_index: usize::MAX,
        });
    }

    let mut opening_claims = Vec::with_capacity(opening_points.len() + local_opening_claims.len());
    for (opening_index, (point, (stored_point, value))) in opening_points
        .into_iter()
        .zip(&opening_proof.opening_claims)
        .enumerate()
    {
        if point.as_slice() != stored_point.as_slice() {
            return Err(WhirNativeCircuitError::OpeningClaimMismatch {
                table_index,
                opening_index,
            });
        }
        opening_claims.push((point, *value));
    }
    let random_claim_count = opening_claims.len();
    for (local_index, (expected_point, expected_value)) in local_opening_claims.iter().enumerate() {
        let opening_index = random_claim_count + local_index;
        let Some((stored_point, stored_value)) = opening_proof.opening_claims.get(opening_index)
        else {
            return Err(WhirNativeCircuitError::OpeningClaimMismatch {
                table_index,
                opening_index,
            });
        };
        if expected_point.as_slice() != stored_point.as_slice() || expected_value != stored_value {
            return Err(WhirNativeCircuitError::OpeningClaimMismatch {
                table_index,
                opening_index,
            });
        }
        opening_claims.push((expected_point.clone(), *expected_value));
    }

    pcs.verify_deferred_after_commitment(
        &parsed_commitment,
        &[opening_claims],
        &opening_proof.proof,
        &mut challenger,
    )
    .map_err(|err| WhirNativeCircuitError::WhirVerificationFailed {
        table_index,
        details: format!("{err:?}"),
    })
}

#[allow(dead_code)]
fn open_table_at_points<F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>(
    pcs: &WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>,
    prover_data: WhirExtensionDeferredProverData<F, EF, MT, DIGEST_ELEMS>,
    table_index: usize,
    points: Vec<Point<EF>>,
    challenger: &mut Challenger,
) -> WhirNativeTableOpeningProof<F, EF, MT>
where
    F: TwoAdicField + Ord + Send + Sync + Clone,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment>,
    Dft: TwoAdicSubgroupDft<F>,
{
    let (opened_values, proof) =
        pcs.open_extension_deferred(prover_data, core::slice::from_ref(&points), challenger);
    let opening_claims = points
        .into_iter()
        .zip(opened_values[0].iter().copied())
        .map(|(point, value)| (point.as_slice().to_vec(), value))
        .collect();

    WhirNativeTableOpeningProof {
        table_index,
        opening_claims,
        proof,
    }
}

#[allow(dead_code)]
fn verify_table_opening_claims_after_context<
    F,
    EF,
    MT,
    Challenger,
    Dft,
    const DIGEST_ELEMS: usize,
>(
    pcs: &WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>,
    table_commitment: &WhirNativeTableCommitment<MT::Commitment>,
    opening_proof: &WhirNativeTableOpeningProof<F, EF, MT>,
    table_index: usize,
    expected_claims: &[(Point<EF>, EF)],
    challenger: &mut Challenger,
) -> Result<(), WhirNativeCircuitError>
where
    F: TwoAdicField + Ord + Send + Sync + Serialize + for<'de> Deserialize<'de>,
    EF: ExtensionField<F> + TwoAdicField + Send + Sync + Serialize + for<'de> Deserialize<'de>,
    MT: Mmcs<F>,
    MT::Commitment: Clone + PartialEq + Serialize + for<'de> Deserialize<'de>,
    MT::Proof: Clone + Serialize + for<'de> Deserialize<'de>,
    Challenger:
        FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment> + Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    if opening_proof.table_index != table_index
        || opening_proof.opening_claims.len() != expected_claims.len()
    {
        return Err(WhirNativeCircuitError::OpeningClaimMismatch {
            table_index,
            opening_index: usize::MAX,
        });
    }

    let parsed_commitment = pcs
        .parse_initial_commitment(
            &table_commitment.commitment,
            &opening_proof.proof,
            challenger,
        )
        .map_err(|err| WhirNativeCircuitError::WhirVerificationFailed {
            table_index,
            details: format!("{err:?}"),
        })?;

    let mut claims = Vec::with_capacity(expected_claims.len());
    for (opening_index, ((expected_point, expected_value), (stored_point, stored_value))) in
        expected_claims
            .iter()
            .zip(&opening_proof.opening_claims)
            .enumerate()
    {
        if expected_point.as_slice() != stored_point.as_slice() || expected_value != stored_value {
            return Err(WhirNativeCircuitError::OpeningClaimMismatch {
                table_index,
                opening_index,
            });
        }
        claims.push((expected_point.clone(), *expected_value));
    }

    pcs.verify_deferred_after_commitment(
        &parsed_commitment,
        &[claims],
        &opening_proof.proof,
        challenger,
    )
    .map_err(|err| WhirNativeCircuitError::WhirVerificationFailed {
        table_index,
        details: format!("{err:?}"),
    })
}

fn observe_table_context<F, Challenger>(
    challenger: &mut Challenger,
    public_io_digest: &[F],
    shape_digest: &[F],
    table_index: usize,
    metadata: &WhirNativeTableMetadata,
    options: WhirNativeCircuitOptions,
) where
    F: Field,
    Challenger: CanObserve<F>,
{
    challenger.observe(F::from_u64(0x5748_4952_4349_5243));
    for &value in public_io_digest {
        challenger.observe(value);
    }
    for &value in shape_digest {
        challenger.observe(value);
    }
    challenger.observe(F::from_u64(table_index as u64));
    challenger.observe(F::from_u64(metadata.kind.tag()));
    observe_string(challenger, &metadata.op_type);
    challenger.observe(F::from_u64(metadata.width as u64));
    challenger.observe(F::from_u64(metadata.padded_width as u64));
    challenger.observe(F::from_u64(metadata.active_rows as u64));
    challenger.observe(F::from_u64(metadata.padded_height as u64));
    challenger.observe(F::from_u64(metadata.num_variables as u64));
    challenger.observe(F::from_u64(metadata.column_layout_version as u64));
    challenger.observe(F::from_u64(options.openings_per_table as u64));
    challenger.observe(F::from_u64(options.min_num_variables as u64));
}

fn sample_table_opening_points<F, EF, Challenger>(
    challenger: &mut Challenger,
    num_variables: usize,
    table_index: usize,
    count: usize,
) -> Vec<Point<EF>>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
{
    let count = count.max(1);
    challenger.observe(F::from_u64(table_index as u64));
    challenger.observe(F::from_u64(num_variables as u64));
    challenger.observe(F::from_u64(count as u64));
    (0..count)
        .map(|i| {
            challenger.observe(F::from_u64(i as u64));
            Point::expand_from_univariate(challenger.sample_algebra_element(), num_variables)
        })
        .collect()
}

fn compute_public_io_digest<F, EF>(public_inputs: &[EF]) -> Vec<F>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let mut digest = new_digest::<F>();
    absorb_u64(&mut digest, 1);
    absorb_u64(&mut digest, public_inputs.len() as u64);
    for &value in public_inputs {
        absorb_ef(&mut digest, value);
    }
    digest
}

fn compute_shape_digest<F, EF>(circuit: &Circuit<EF>) -> Vec<F>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let mut digest = new_digest::<F>();
    absorb_u64(&mut digest, 2);
    absorb_u64(&mut digest, circuit.witness_count as u64);
    absorb_u64(&mut digest, circuit.public_flat_len as u64);
    absorb_u64(&mut digest, circuit.private_flat_len as u64);

    let mut enabled = circuit.enabled_ops.keys().collect::<Vec<_>>();
    enabled.sort_by(|left, right| left.as_str().cmp(right.as_str()));
    for op_type in enabled {
        absorb_string(&mut digest, op_type.as_str());
    }

    for op in &circuit.ops {
        match op {
            Op::Const { out, val } => {
                absorb_u64(&mut digest, 10);
                absorb_u64(&mut digest, out.0 as u64);
                absorb_ef(&mut digest, *val);
            }
            Op::Public { out, public_pos } => {
                absorb_u64(&mut digest, 11);
                absorb_u64(&mut digest, out.0 as u64);
                absorb_u64(&mut digest, *public_pos as u64);
            }
            Op::Alu {
                kind,
                a,
                b,
                c,
                out,
                intermediate_out,
            } => {
                absorb_u64(&mut digest, 12);
                absorb_u64(&mut digest, alu_kind_to_tag(*kind) as u64);
                absorb_u64(&mut digest, a.0 as u64);
                absorb_u64(&mut digest, b.0 as u64);
                absorb_u64(&mut digest, c.map_or(u32::MAX, |wid| wid.0) as u64);
                absorb_u64(&mut digest, out.0 as u64);
                absorb_u64(
                    &mut digest,
                    intermediate_out.map_or(u32::MAX, |wid| wid.0) as u64,
                );
            }
            Op::Hint {
                inputs, outputs, ..
            } => {
                absorb_u64(&mut digest, 13);
                absorb_wids(&mut digest, inputs);
                absorb_wids(&mut digest, outputs);
            }
            Op::NonPrimitiveOpWithExecutor {
                inputs,
                outputs,
                executor,
                op_id,
            } => {
                absorb_u64(&mut digest, 14);
                absorb_string(&mut digest, executor.op_type().as_str());
                absorb_string(&mut digest, &format!("{executor:?}"));
                absorb_u64(&mut digest, op_id.0 as u64);
                absorb_wid_groups(&mut digest, inputs);
                absorb_wid_groups(&mut digest, outputs);
            }
        }
    }

    digest
}

fn new_digest<F>() -> Vec<F>
where
    F: Field,
{
    vec![
        F::from_u64(0x9e37_79b9),
        F::from_u64(0x85eb_ca6b),
        F::from_u64(0xc2b2_ae35),
        F::from_u64(0x27d4_eb2f),
    ]
}

fn absorb_f<F>(digest: &mut [F], value: F)
where
    F: Field,
{
    for (i, slot) in digest.iter_mut().enumerate() {
        *slot = *slot * F::from_u64(DIGEST_MIX + i as u64 * 17) + value + F::from_u64(i as u64);
    }
}

fn absorb_u64<F>(digest: &mut [F], value: u64)
where
    F: Field,
{
    absorb_f(digest, F::from_u64(value));
}

fn absorb_ef<F, EF>(digest: &mut [F], value: EF)
where
    F: Field,
    EF: ExtensionField<F>,
{
    for &coeff in value.as_basis_coefficients_slice() {
        absorb_f(digest, coeff);
    }
}

fn absorb_string<F>(digest: &mut [F], value: &str)
where
    F: Field,
{
    absorb_u64(digest, value.len() as u64);
    for byte in value.bytes() {
        absorb_u64(digest, byte as u64);
    }
}

fn observe_string<F, Challenger>(challenger: &mut Challenger, value: &str)
where
    F: Field,
    Challenger: CanObserve<F>,
{
    challenger.observe(F::from_u64(value.len() as u64));
    for byte in value.bytes() {
        challenger.observe(F::from_u64(byte as u64));
    }
}

fn absorb_wids<F>(digest: &mut [F], wids: &[WitnessId])
where
    F: Field,
{
    absorb_u64(digest, wids.len() as u64);
    for wid in wids {
        absorb_u64(digest, wid.0 as u64);
    }
}

fn absorb_wid_groups<F>(digest: &mut [F], groups: &[Vec<WitnessId>])
where
    F: Field,
{
    absorb_u64(digest, groups.len() as u64);
    for group in groups {
        absorb_wids(digest, group);
    }
}

fn ef_from_u64<F, EF>(value: u64) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    EF::from(F::from_u64(value))
}

fn ef_from_bool<F, EF>(value: bool) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    EF::from(F::from_bool(value))
}

const fn alu_kind_to_tag(kind: AluOpKind) -> u8 {
    match kind {
        AluOpKind::Add => 1,
        AluOpKind::Mul => 2,
        AluOpKind::BoolCheck => 3,
        AluOpKind::MulAdd => 4,
        AluOpKind::HornerAcc => 5,
    }
}

fn recompose_kind_to_tag(kind: RecomposeTraceKind) -> u8 {
    match kind {
        RecomposeTraceKind::Standard => 1,
        RecomposeTraceKind::WithCoeffLookups => 2,
    }
}

fn poseidon2_config_from_op_type(op_type: &str) -> Option<Poseidon2Config> {
    op_type
        .strip_prefix("poseidon2_perm/")
        .and_then(Poseidon2Config::from_variant_name)
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear, default_babybear_poseidon2_16};
    use p3_challenger::{
        CanObserve, CanSample, CanSampleBits, DuplexChallenger, FieldChallenger, GrindingChallenger,
    };
    use p3_circuit::CircuitBuilder;
    use p3_circuit::ops::{
        NpoPrivateData, Poseidon2PermCall, Poseidon2PermPrivateData, generate_poseidon2_trace,
        generate_recompose_trace,
    };
    use p3_dft::Radix2DFTSmallBatch;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_poseidon2_circuit_air::BabyBearD4Width16;
    use p3_symmetric::{MerkleCap, PaddingFreeSponge, TruncatedPermutation};
    use p3_whir::parameters::{
        FoldingFactor, ProtocolParameters, SecurityAssumption, SumcheckStrategy,
    };

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type PackedF = <F as Field>::Packing;
    type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;
    type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;
    type MyDft = Radix2DFTSmallBatch<F>;
    type MyCommitment = MerkleCap<F, [F; DIGEST_ELEMS]>;
    type MyProof = WhirNativeCircuitProof<F, EF, MyMmcs>;

    const DIGEST_ELEMS: usize = 8;

    #[derive(Clone)]
    struct TestChallenger(MyChallenger);

    impl TestChallenger {
        fn new() -> Self {
            Self(MyChallenger::new(default_babybear_poseidon2_16()))
        }
    }

    impl CanObserve<F> for TestChallenger {
        fn observe(&mut self, value: F) {
            self.0.observe(value);
        }
    }

    impl CanObserve<MyCommitment> for TestChallenger {
        fn observe(&mut self, value: MyCommitment) {
            self.0.observe(value);
        }
    }

    impl CanObserve<Vec<MyCommitment>> for TestChallenger {
        fn observe(&mut self, value: Vec<MyCommitment>) {
            for commitment in value {
                self.0.observe(commitment);
            }
        }
    }

    impl CanSample<F> for TestChallenger {
        fn sample(&mut self) -> F {
            self.0.sample()
        }
    }

    impl CanSampleBits<usize> for TestChallenger {
        fn sample_bits(&mut self, bits: usize) -> usize {
            self.0.sample_bits(bits)
        }
    }

    impl FieldChallenger<F> for TestChallenger {}

    impl GrindingChallenger for TestChallenger {
        type Witness = F;

        fn grind(&mut self, bits: usize) -> Self::Witness {
            self.0.grind(bits)
        }
    }

    fn make_mmcs() -> MyMmcs {
        let perm = default_babybear_poseidon2_16();
        MyMmcs::new(MyHash::new(perm.clone()), MyCompress::new(perm), 0)
    }

    fn make_protocol_params(mmcs: &MyMmcs) -> ProtocolParameters<MyMmcs> {
        ProtocolParameters {
            security_level: 32,
            pow_bits: 0,
            rs_domain_initial_reduction_factor: 1,
            folding_factor: FoldingFactor::Constant(4),
            mmcs: mmcs.clone(),
            soundness_type: SecurityAssumption::JohnsonBound,
            starting_log_inv_rate: 1,
        }
    }

    fn make_pcs(
        mmcs: &MyMmcs,
        num_variables: usize,
    ) -> WhirPcs<EF, F, MyMmcs, TestChallenger, MyDft, DIGEST_ELEMS> {
        WhirPcs::new(
            num_variables,
            make_protocol_params(mmcs),
            MyDft::default(),
            SumcheckStrategy::Svo,
        )
    }

    fn no_poseidon(_: Poseidon2Config, _: &[EF]) -> Option<Vec<EF>> {
        None
    }

    #[test]
    fn benchmark_critical_path_has_no_count_only_witness_bus() {
        let source = include_str!("whir_native.rs");
        for forbidden in [
            concat!("WhirNative", "WitnessBusProof"),
            concat!("witness_", "bus_proof"),
            concat!("claimed_", "logup_sum"),
            concat!("build_", "witness_bus_summary"),
            concat!("verify_", "witness_bus_summary"),
        ] {
            assert!(
                !source.contains(forbidden),
                "benchmark-critical WHIR-native path contains placeholder `{forbidden}`"
            );
        }
    }

    fn simple_arithmetic_proof() -> (Circuit<EF>, Vec<EF>, MyMmcs, MyProof) {
        let mut builder = CircuitBuilder::<EF>::new();
        let x = builder.public_input();
        let y = builder.public_input();
        let sum = builder.add(x, y);
        let three = builder.define_const(EF::from(F::from_u64(3)));
        let diff = builder.sub(sum, three);
        builder.assert_zero(diff);
        let circuit = builder.build().expect("build simple circuit");

        let public_inputs = vec![EF::from(F::ONE), EF::from(F::TWO)];
        let mut runner = circuit.runner();
        runner
            .set_public_inputs(&public_inputs)
            .expect("set public inputs");
        let traces = runner.run().expect("run simple circuit");

        let mmcs = make_mmcs();
        let options = WhirNativeCircuitOptions {
            openings_per_table: 1,
            min_num_variables: 4,
        };
        let proof = prove_whir_native_circuit(
            &circuit,
            &public_inputs,
            &[],
            &[],
            &traces,
            options,
            |num_variables| make_pcs(&mmcs, num_variables),
            TestChallenger::new,
            no_poseidon,
        )
        .expect("prove simple arithmetic circuit");

        (circuit, public_inputs, mmcs, proof)
    }

    fn recompose_proof() -> (Circuit<EF>, Vec<EF>, MyMmcs, MyProof) {
        let mut builder = CircuitBuilder::<EF>::new();
        builder.enable_recompose::<F>(generate_recompose_trace::<F, EF>);
        let coeffs = (0..<EF as BasedVectorSpace<F>>::DIMENSION)
            .map(|_| builder.public_input())
            .collect::<Vec<_>>();
        let _recomposed = builder
            .recompose_base_coeffs_to_ext::<F>(&coeffs)
            .expect("add recompose op");
        let circuit = builder.build().expect("build recompose circuit");

        let public_inputs = (0..<EF as BasedVectorSpace<F>>::DIMENSION)
            .map(|i| EF::from(F::from_u64((i + 1) as u64)))
            .collect::<Vec<_>>();
        let mut runner = circuit.runner();
        runner
            .set_public_inputs(&public_inputs)
            .expect("set public inputs");
        let traces = runner.run().expect("run recompose circuit");

        let mmcs = make_mmcs();
        let options = WhirNativeCircuitOptions {
            openings_per_table: 1,
            min_num_variables: 4,
        };
        let proof = prove_whir_native_circuit(
            &circuit,
            &public_inputs,
            &[],
            &[],
            &traces,
            options,
            |num_variables| make_pcs(&mmcs, num_variables),
            TestChallenger::new,
            no_poseidon,
        )
        .expect("prove recompose circuit");

        (circuit, public_inputs, mmcs, proof)
    }

    #[test]
    fn poseidon2_air_proves_and_detects_terminal_tamper() {
        let mut builder = CircuitBuilder::<EF>::new();
        builder.enable_poseidon2_perm::<BabyBearD4Width16, _>(
            generate_poseidon2_trace::<EF, BabyBearD4Width16>,
            default_babybear_poseidon2_16(),
        );
        let config = Poseidon2Config::BabyBearD4Width16;
        let inputs = (0..config.width_ext())
            .map(|_| builder.public_input())
            .collect::<Vec<_>>();
        let _outputs = builder
            .add_poseidon2_perm_for_challenger(config, &inputs)
            .expect("add Poseidon2 challenger permutation");
        let circuit = builder.build().expect("build Poseidon2 circuit");

        let public_inputs = EF::zero_vec(config.width_ext());
        let mut runner = circuit.runner();
        runner
            .set_public_inputs(&public_inputs)
            .expect("set public inputs");
        let traces = runner.run().expect("run Poseidon2 circuit");

        let payload = trace_payload_from_traces::<F, EF>(&circuit, &traces)
            .expect("extract Poseidon2 trace payload");
        let tables = build_tables(
            &payload,
            WhirNativeCircuitOptions {
                openings_per_table: 1,
                min_num_variables: 4,
            },
        )
        .expect("build Poseidon2 WHIR-native tables");
        let op_type = format!("poseidon2_perm/{}", config.variant_name());
        let expected_rows = expected_poseidon2_rows_from_circuit(&circuit, &op_type)
            .expect("derive expected Poseidon2 rows");
        let direction_bit_witness_ids =
            expected_poseidon2_direction_bit_witness_ids_from_circuit(&circuit, &op_type)
                .expect("derive Poseidon2 direction-bit witness ids");

        let witness_table = &tables[WITNESS_TABLE_INDEX];
        let poseidon2_table_index = 4;
        let poseidon2_table = &tables[poseidon2_table_index];
        assert_eq!(
            poseidon2_table.metadata.kind,
            WhirNativeTableKind::Poseidon2
        );
        assert_eq!(poseidon2_table.metadata.width, P2_BB_D4_WIDTH16_TABLE_WIDTH);
        let proof = prove_poseidon2_air_constraints::<F, EF, TestChallenger>(
            poseidon2_table_index,
            poseidon2_table,
            witness_table,
            &expected_rows,
            &direction_bit_witness_ids,
            &mut TestChallenger::new(),
        )
        .expect("prove Poseidon2 AIR constraints");
        assert_eq!(proof.kind, WhirNativeLocalConstraintKind::Poseidon2Air);

        verify_poseidon2_air_constraints::<F, EF, TestChallenger>(
            &proof,
            poseidon2_table_index,
            &poseidon2_table.metadata,
            &witness_table.metadata,
            &expected_rows,
            &direction_bit_witness_ids,
            &mut TestChallenger::new(),
        )
        .expect("verify Poseidon2 AIR constraints");

        for (opening_index, label) in [
            (0, "current column"),
            (P2_BB_D4_WIDTH16_AIR_WIDTH, "shifted column"),
            (P2_BB_D4_WIDTH16_AIR_WIDTH * 2, "next-row binding"),
            (P2_BB_D4_WIDTH16_AIR_WIDTH * 3, "input witness"),
            (
                P2_BB_D4_WIDTH16_AIR_WIDTH * 3 + P2_BB_D4_WIDTH16_WIDTH_EXT,
                "output witness",
            ),
        ] {
            let mut tampered = proof.clone();
            tampered.terminal_openings[opening_index].value += EF::ONE;
            let err = verify_poseidon2_air_constraints::<F, EF, TestChallenger>(
                &tampered,
                poseidon2_table_index,
                &poseidon2_table.metadata,
                &witness_table.metadata,
                &expected_rows,
                &direction_bit_witness_ids,
                &mut TestChallenger::new(),
            );
            assert!(err.is_err(), "tampered Poseidon2 {label} opening must fail");
        }

        let mmcs = make_mmcs();
        prove_whir_native_circuit(
            &circuit,
            &public_inputs,
            &[],
            &[],
            &traces,
            WhirNativeCircuitOptions {
                openings_per_table: 1,
                min_num_variables: 4,
            },
            |num_variables| make_pcs(&mmcs, num_variables),
            TestChallenger::new,
            no_poseidon,
        )
        .expect("prove oracle-only Poseidon2 circuit");
    }

    #[test]
    fn poseidon2_merkle_witness_bindings_detect_terminal_tamper() {
        let mut builder = CircuitBuilder::<EF>::new();
        builder.enable_poseidon2_perm::<BabyBearD4Width16, _>(
            generate_poseidon2_trace::<EF, BabyBearD4Width16>,
            default_babybear_poseidon2_16(),
        );
        let config = Poseidon2Config::BabyBearD4Width16;
        let row0_inputs = (0..config.width_ext())
            .map(|i| builder.alloc_const(EF::from(F::from_u64((i + 1) as u64)), "row0"))
            .map(Some)
            .collect::<Vec<_>>();
        let bit0 = builder.alloc_const(EF::ZERO, "bit0");
        builder
            .add_poseidon2_perm(&Poseidon2PermCall {
                config,
                new_start: true,
                merkle_path: true,
                mmcs_bit: Some(bit0),
                inputs: row0_inputs,
                out_ctl: vec![false; config.rate_ext()],
                return_all_outputs: false,
                mmcs_index_sum: None,
            })
            .expect("add first Merkle Poseidon2 row");

        let bit1 = builder.alloc_const(EF::ONE, "bit1");
        let mmcs_index_sum = builder.public_input();
        let (row1_op_id, _) = builder
            .add_poseidon2_perm(&Poseidon2PermCall {
                config,
                new_start: false,
                merkle_path: true,
                mmcs_bit: Some(bit1),
                inputs: vec![None; config.width_ext()],
                out_ctl: vec![false; config.rate_ext()],
                return_all_outputs: false,
                mmcs_index_sum: Some(mmcs_index_sum),
            })
            .expect("add second Merkle Poseidon2 row");

        let circuit = builder.build().expect("build Merkle Poseidon2 circuit");
        let public_inputs = vec![EF::ONE];
        let mut runner = circuit.runner();
        runner
            .set_public_inputs(&public_inputs)
            .expect("set public inputs");
        runner
            .set_private_data(
                row1_op_id,
                NpoPrivateData::new(Poseidon2PermPrivateData {
                    sibling: vec![EF::from(F::from_u64(11)), EF::from(F::from_u64(12))],
                }),
            )
            .expect("set Merkle sibling");
        let traces = runner.run().expect("run Merkle Poseidon2 circuit");

        let payload = trace_payload_from_traces::<F, EF>(&circuit, &traces)
            .expect("extract Merkle Poseidon2 trace payload");
        let tables = build_tables(
            &payload,
            WhirNativeCircuitOptions {
                openings_per_table: 1,
                min_num_variables: 4,
            },
        )
        .expect("build Merkle Poseidon2 WHIR-native tables");
        let op_type = format!("poseidon2_perm/{}", config.variant_name());
        let expected_rows = expected_poseidon2_rows_from_circuit(&circuit, &op_type)
            .expect("derive expected Poseidon2 rows");
        let direction_bit_witness_ids =
            expected_poseidon2_direction_bit_witness_ids_from_circuit(&circuit, &op_type)
                .expect("derive Poseidon2 direction-bit witness ids");
        let witness_table = &tables[WITNESS_TABLE_INDEX];
        let poseidon2_table_index = 4;
        let poseidon2_table = &tables[poseidon2_table_index];
        let proof = prove_poseidon2_air_constraints::<F, EF, TestChallenger>(
            poseidon2_table_index,
            poseidon2_table,
            witness_table,
            &expected_rows,
            &direction_bit_witness_ids,
            &mut TestChallenger::new(),
        )
        .expect("prove Merkle Poseidon2 AIR constraints");

        for (port, label) in [
            (
                P2_BB_D4_WIDTH16_WIDTH_EXT + P2_BB_D4_WIDTH16_RATE_EXT,
                "MMCS index",
            ),
            (
                P2_BB_D4_WIDTH16_WIDTH_EXT + P2_BB_D4_WIDTH16_RATE_EXT + 1,
                "direction bit",
            ),
        ] {
            let mut tampered = proof.clone();
            tampered.terminal_openings[P2_BB_D4_WIDTH16_AIR_WIDTH * 3 + port].value += EF::ONE;
            let err = verify_poseidon2_air_constraints::<F, EF, TestChallenger>(
                &tampered,
                poseidon2_table_index,
                &poseidon2_table.metadata,
                &witness_table.metadata,
                &expected_rows,
                &direction_bit_witness_ids,
                &mut TestChallenger::new(),
            );
            assert!(err.is_err(), "tampered Poseidon2 {label} opening must fail");
        }
    }

    fn test_witness_table(
        values: &[(u32, EF)],
        min_num_variables: usize,
    ) -> WhirNativeTableData<EF> {
        let max_wid = values.iter().map(|(wid, _)| *wid).max().unwrap_or(0) as usize;
        let mut rows = (0..=max_wid)
            .map(|wid| vec![EF::from(F::from_u64(wid as u64)), EF::ZERO])
            .collect::<Vec<_>>();
        for &(wid, value) in values {
            rows[wid as usize][1] = value;
        }
        pack_rows::<EF>(
            WhirNativeTableKind::Witness,
            String::new(),
            rows,
            Some(WITNESS_WIDTH),
            WhirNativeCircuitOptions {
                openings_per_table: 1,
                min_num_variables,
            },
        )
    }

    #[test]
    fn valid_arithmetic_circuit_proves_and_verifies() {
        let (circuit, public_inputs, mmcs, proof) = simple_arithmetic_proof();
        verify_whir_native_circuit_proof(
            &circuit,
            &public_inputs,
            WhirNativeCircuitOptions {
                openings_per_table: 1,
                min_num_variables: 4,
            },
            &proof,
            |num_variables| make_pcs(&mmcs, num_variables),
            TestChallenger::new,
            no_poseidon,
        )
        .expect("verify simple arithmetic circuit");
    }

    #[test]
    fn tampered_public_input_fails() {
        let (circuit, mut public_inputs, mmcs, proof) = simple_arithmetic_proof();
        public_inputs[0] += EF::ONE;
        let err = verify_whir_native_circuit_proof(
            &circuit,
            &public_inputs,
            WhirNativeCircuitOptions {
                openings_per_table: 1,
                min_num_variables: 4,
            },
            &proof,
            |num_variables| make_pcs(&mmcs, num_variables),
            TestChallenger::new,
            no_poseidon,
        )
        .expect_err("tampered public input must fail");
        assert!(matches!(
            err,
            WhirNativeCircuitError::PublicIoDigestMismatch
        ));
    }

    #[test]
    fn tampered_witness_terminal_opening_fails() {
        let (circuit, public_inputs, mmcs, mut proof) = simple_arithmetic_proof();
        proof.constraint_sumcheck_proofs[0]
            .local_proof
            .as_mut()
            .expect("witness local proof")
            .terminal_openings[1]
            .value += EF::ONE;
        verify_whir_native_circuit_proof(
            &circuit,
            &public_inputs,
            WhirNativeCircuitOptions {
                openings_per_table: 1,
                min_num_variables: 4,
            },
            &proof,
            |num_variables| make_pcs(&mmcs, num_variables),
            TestChallenger::new,
            no_poseidon,
        )
        .expect_err("tampered witness terminal opening must fail");
    }

    #[test]
    fn primitive_local_constraints_are_part_of_table_openings() {
        let (_circuit, _public_inputs, _mmcs, proof) = simple_arithmetic_proof();

        assert!(
            proof.constraint_sumcheck_proofs[0].local_proof.is_some(),
            "witness table should have a local proof"
        );
        for table_index in [1, 2, 3] {
            assert!(
                proof.constraint_sumcheck_proofs[table_index]
                    .local_proof
                    .is_some(),
                "primitive table {table_index} should have a local proof"
            );
            assert!(
                proof.opening_proofs[table_index].opening_claims.len() > 1,
                "primitive table {table_index} should open random plus terminal claims"
            );
        }
    }

    #[test]
    fn recompose_table_proves_and_detects_terminal_tamper() {
        let (circuit, public_inputs, mmcs, mut proof) = recompose_proof();
        assert_eq!(
            proof.table_commitments[4].metadata.kind,
            WhirNativeTableKind::Recompose
        );
        assert!(
            proof.constraint_sumcheck_proofs[4].local_proof.is_some(),
            "recompose table should have a local proof"
        );

        verify_whir_native_circuit_proof(
            &circuit,
            &public_inputs,
            WhirNativeCircuitOptions {
                openings_per_table: 1,
                min_num_variables: 4,
            },
            &proof,
            |num_variables| make_pcs(&mmcs, num_variables),
            TestChallenger::new,
            no_poseidon,
        )
        .expect("verify recompose circuit");

        proof.constraint_sumcheck_proofs[4]
            .local_proof
            .as_mut()
            .expect("recompose local proof")
            .terminal_openings
            .last_mut()
            .expect("recompose output witness opening")
            .value += EF::ONE;
        verify_whir_native_circuit_proof(
            &circuit,
            &public_inputs,
            WhirNativeCircuitOptions {
                openings_per_table: 1,
                min_num_variables: 4,
            },
            &proof,
            |num_variables| make_pcs(&mmcs, num_variables),
            TestChallenger::new,
            no_poseidon,
        )
        .expect_err("tampered recompose terminal opening must fail");
    }

    #[test]
    fn tampered_primitive_local_sumcheck_fails() {
        let (circuit, public_inputs, mmcs, mut proof) = simple_arithmetic_proof();
        proof.constraint_sumcheck_proofs[3]
            .local_proof
            .as_mut()
            .expect("ALU local proof")
            .terminal_openings[8]
            .value += EF::ONE;

        verify_whir_native_circuit_proof(
            &circuit,
            &public_inputs,
            WhirNativeCircuitOptions {
                openings_per_table: 1,
                min_num_variables: 4,
            },
            &proof,
            |num_variables| make_pcs(&mmcs, num_variables),
            TestChallenger::new,
            no_poseidon,
        )
        .expect_err("tampered primitive local proof must fail");
    }

    #[test]
    fn tampered_primitive_terminal_opening_claim_fails() {
        let (circuit, public_inputs, mmcs, mut proof) = simple_arithmetic_proof();
        let alu_openings = &mut proof.opening_proofs[3].opening_claims;
        let last = alu_openings.last_mut().expect("ALU terminal opening");
        last.1 += EF::ONE;

        verify_whir_native_circuit_proof(
            &circuit,
            &public_inputs,
            WhirNativeCircuitOptions {
                openings_per_table: 1,
                min_num_variables: 4,
            },
            &proof,
            |num_variables| make_pcs(&mmcs, num_variables),
            TestChallenger::new,
            no_poseidon,
        )
        .expect_err("tampered primitive terminal opening must fail");
    }

    #[test]
    fn table_column_point_matches_row_major_packing() {
        let table = pack_rows::<EF>(
            WhirNativeTableKind::Alu,
            String::new(),
            vec![
                vec![EF::from(F::from_u64(10)), EF::from(F::from_u64(11))],
                vec![EF::from(F::from_u64(20)), EF::from(F::from_u64(21))],
            ],
            None,
            WhirNativeCircuitOptions {
                openings_per_table: 1,
                min_num_variables: 2,
            },
        );

        let row_zero = Point::hypercube(0, whir_native_table_row_variables(&table.metadata));
        let row_one = Point::hypercube(1, whir_native_table_row_variables(&table.metadata));

        assert_eq!(
            eval_table_column_at_row_point::<F, EF>(&table, &row_zero, 0).unwrap(),
            EF::from(F::from_u64(10))
        );
        assert_eq!(
            eval_table_column_at_row_point::<F, EF>(&table, &row_zero, 1).unwrap(),
            EF::from(F::from_u64(11))
        );
        assert_eq!(
            eval_table_column_at_row_point::<F, EF>(&table, &row_one, 0).unwrap(),
            EF::from(F::from_u64(20))
        );
        assert_eq!(
            eval_table_column_at_row_point::<F, EF>(&table, &row_one, 1).unwrap(),
            EF::from(F::from_u64(21))
        );
    }

    #[test]
    fn witness_address_point_composes_static_witness_ids() {
        let witness_table = pack_rows::<EF>(
            WhirNativeTableKind::Witness,
            String::new(),
            vec![
                vec![EF::ZERO, EF::from(F::from_u64(100))],
                vec![EF::ONE, EF::from(F::from_u64(101))],
                vec![EF::from(F::TWO), EF::from(F::from_u64(102))],
                vec![EF::from(F::from_u64(3)), EF::from(F::from_u64(103))],
            ],
            Some(WITNESS_WIDTH),
            WhirNativeCircuitOptions {
                openings_per_table: 1,
                min_num_variables: 3,
            },
        );
        let source_height: usize = 2;
        let row_zero = Point::hypercube(0, source_height.ilog2() as usize);
        let row_one = Point::hypercube(1, source_height.ilog2() as usize);

        let address_zero = whir_native_witness_address_point::<F, EF>(
            &witness_table.metadata,
            source_height,
            &row_zero,
            &[3, 1],
        )
        .unwrap();
        let address_one = whir_native_witness_address_point::<F, EF>(
            &witness_table.metadata,
            source_height,
            &row_one,
            &[3, 1],
        )
        .unwrap();

        assert_eq!(
            address_zero,
            Point::hypercube(3, whir_native_table_row_variables(&witness_table.metadata))
        );
        assert_eq!(
            address_one,
            Point::hypercube(1, whir_native_table_row_variables(&witness_table.metadata))
        );
        assert_eq!(
            eval_table_column_at_row_point::<F, EF>(&witness_table, &address_zero, 1).unwrap(),
            EF::from(F::from_u64(103))
        );
        assert_eq!(
            eval_table_column_at_row_point::<F, EF>(&witness_table, &address_one, 1).unwrap(),
            EF::from(F::from_u64(101))
        );

        let packed_value_point = whir_native_witness_value_opening_point::<F, EF>(
            &witness_table.metadata,
            source_height,
            &row_zero,
            &[3, 1],
        )
        .unwrap();
        assert_eq!(
            Poly::new(witness_table.values.clone()).eval_ext::<F>(&packed_value_point),
            EF::from(F::from_u64(103))
        );
    }

    #[test]
    fn whir_native_local_known_rows_proves_and_detects_terminal_tamper() {
        let rows = vec![
            vec![EF::from(F::from_u64(3)), EF::from(F::from_u64(30))],
            vec![EF::from(F::from_u64(7)), EF::from(F::from_u64(70))],
        ];
        let table = pack_rows::<EF>(
            WhirNativeTableKind::Public,
            String::new(),
            rows.clone(),
            Some(2),
            WhirNativeCircuitOptions {
                openings_per_table: 1,
                min_num_variables: 3,
            },
        );
        let witness_ids = vec![3, 7];
        let witness_table = test_witness_table(
            &[
                (3, EF::from(F::from_u64(30))),
                (7, EF::from(F::from_u64(70))),
            ],
            3,
        );

        let mut prover_challenger = TestChallenger::new();
        let proof = prove_known_rows_local_constraints::<F, EF, TestChallenger>(
            2,
            &table,
            &witness_table,
            &rows,
            &witness_ids,
            &mut prover_challenger,
        )
        .expect("prove known-row local constraints");

        let mut verifier_challenger = TestChallenger::new();
        let terminal_claims = verify_known_rows_local_constraints::<F, EF, TestChallenger>(
            &proof,
            2,
            &table.metadata,
            &witness_table.metadata,
            &rows,
            &witness_ids,
            &mut verifier_challenger,
        )
        .expect("verify known-row local constraints");
        assert_eq!(terminal_claims.len(), table.metadata.width + 1);

        let mut tampered = proof.clone();
        tampered.terminal_openings[1].value += EF::ONE;
        let mut verifier_challenger = TestChallenger::new();
        verify_known_rows_local_constraints::<F, EF, TestChallenger>(
            &tampered,
            2,
            &table.metadata,
            &witness_table.metadata,
            &rows,
            &witness_ids,
            &mut verifier_challenger,
        )
        .expect_err("tampered terminal opening must fail");
    }

    #[test]
    fn whir_native_local_known_rows_rejects_bad_table() {
        let rows = vec![vec![EF::from(F::from_u64(5)), EF::from(F::from_u64(55))]];
        let mut table = pack_rows::<EF>(
            WhirNativeTableKind::Const,
            String::new(),
            rows.clone(),
            Some(2),
            WhirNativeCircuitOptions {
                openings_per_table: 1,
                min_num_variables: 3,
            },
        );
        table.values[1] += EF::ONE;
        let witness_ids = vec![5];
        let witness_table = test_witness_table(&[(5, EF::from(F::from_u64(55)))], 3);

        let mut prover_challenger = TestChallenger::new();
        prove_known_rows_local_constraints::<F, EF, TestChallenger>(
            1,
            &table,
            &witness_table,
            &rows,
            &witness_ids,
            &mut prover_challenger,
        )
        .expect_err("bad known-row table must not prove");
    }

    #[test]
    fn whir_native_local_alu_proves_and_detects_terminal_tamper() {
        let expected_rows = vec![
            WhirNativeExpectedAluRow {
                kind: AluOpKind::Add,
                indices: [11, 1, 0, 2],
                acc_index: None,
            },
            WhirNativeExpectedAluRow {
                kind: AluOpKind::Mul,
                indices: [3, 4, 0, 5],
                acc_index: None,
            },
            WhirNativeExpectedAluRow {
                kind: AluOpKind::BoolCheck,
                indices: [6, 0, 0, 6],
                acc_index: None,
            },
            WhirNativeExpectedAluRow {
                kind: AluOpKind::MulAdd,
                indices: [7, 8, 9, 10],
                acc_index: None,
            },
        ];
        let rows = vec![
            vec![
                EF::from(F::from_u64(alu_kind_to_tag(AluOpKind::Add) as u64)),
                EF::from(F::from_u64(11)),
                EF::ONE,
                EF::ZERO,
                EF::from(F::TWO),
                EF::from(F::TWO),
                EF::from(F::from_u64(3)),
                EF::ZERO,
                EF::from(F::from_u64(5)),
            ],
            vec![
                EF::from(F::from_u64(alu_kind_to_tag(AluOpKind::Mul) as u64)),
                EF::from(F::from_u64(3)),
                EF::from(F::from_u64(4)),
                EF::ZERO,
                EF::from(F::from_u64(5)),
                EF::from(F::from_u64(4)),
                EF::from(F::from_u64(5)),
                EF::ZERO,
                EF::from(F::from_u64(20)),
            ],
            vec![
                EF::from(F::from_u64(alu_kind_to_tag(AluOpKind::BoolCheck) as u64)),
                EF::from(F::from_u64(6)),
                EF::ZERO,
                EF::ZERO,
                EF::from(F::from_u64(6)),
                EF::ONE,
                EF::ZERO,
                EF::ONE,
                EF::ONE,
            ],
            vec![
                EF::from(F::from_u64(alu_kind_to_tag(AluOpKind::MulAdd) as u64)),
                EF::from(F::from_u64(7)),
                EF::from(F::from_u64(8)),
                EF::from(F::from_u64(9)),
                EF::from(F::from_u64(10)),
                EF::from(F::TWO),
                EF::from(F::from_u64(4)),
                EF::from(F::from_u64(7)),
                EF::from(F::from_u64(15)),
            ],
        ];
        let table = pack_rows::<EF>(
            WhirNativeTableKind::Alu,
            String::new(),
            rows,
            Some(ALU_WIDTH),
            WhirNativeCircuitOptions {
                openings_per_table: 1,
                min_num_variables: 4,
            },
        );
        let witness_table = test_witness_table(
            &[
                (0, EF::ZERO),
                (1, EF::from(F::from_u64(3))),
                (2, EF::from(F::from_u64(5))),
                (3, EF::from(F::from_u64(4))),
                (4, EF::from(F::from_u64(5))),
                (5, EF::from(F::from_u64(20))),
                (6, EF::ONE),
                (7, EF::from(F::TWO)),
                (8, EF::from(F::from_u64(4))),
                (9, EF::from(F::from_u64(7))),
                (10, EF::from(F::from_u64(15))),
                (11, EF::from(F::TWO)),
            ],
            4,
        );

        let mut prover_challenger = TestChallenger::new();
        let proof = prove_alu_local_constraints::<F, EF, TestChallenger>(
            3,
            &table,
            &witness_table,
            &expected_rows,
            &mut prover_challenger,
        )
        .expect("prove ALU local constraints");

        let mut verifier_challenger = TestChallenger::new();
        let terminal_claims = verify_alu_local_constraints::<F, EF, TestChallenger>(
            &proof,
            3,
            &table.metadata,
            &witness_table.metadata,
            &expected_rows,
            &mut verifier_challenger,
        )
        .expect("verify ALU local constraints");
        assert_eq!(terminal_claims.len(), ALU_WIDTH + ALU_WITNESS_PORTS.len());

        let mut tampered = proof.clone();
        tampered.terminal_openings[8].value += EF::ONE;
        let mut verifier_challenger = TestChallenger::new();
        verify_alu_local_constraints::<F, EF, TestChallenger>(
            &tampered,
            3,
            &table.metadata,
            &witness_table.metadata,
            &expected_rows,
            &mut verifier_challenger,
        )
        .expect_err("tampered ALU terminal opening must fail");
    }

    #[test]
    fn whir_native_local_alu_rejects_bad_arithmetic() {
        let expected_rows = vec![WhirNativeExpectedAluRow {
            kind: AluOpKind::Mul,
            indices: [0, 1, 0, 2],
            acc_index: None,
        }];
        let rows = vec![vec![
            EF::from(F::from_u64(alu_kind_to_tag(AluOpKind::Mul) as u64)),
            EF::ZERO,
            EF::ONE,
            EF::ZERO,
            EF::from(F::TWO),
            EF::from(F::from_u64(6)),
            EF::from(F::from_u64(7)),
            EF::ZERO,
            EF::from(F::from_u64(41)),
        ]];
        let table = pack_rows::<EF>(
            WhirNativeTableKind::Alu,
            String::new(),
            rows,
            Some(ALU_WIDTH),
            WhirNativeCircuitOptions {
                openings_per_table: 1,
                min_num_variables: 4,
            },
        );
        let witness_table = test_witness_table(
            &[
                (0, EF::from(F::from_u64(6))),
                (1, EF::from(F::from_u64(7))),
                (2, EF::from(F::from_u64(41))),
            ],
            4,
        );

        let mut prover_challenger = TestChallenger::new();
        prove_alu_local_constraints::<F, EF, TestChallenger>(
            3,
            &table,
            &witness_table,
            &expected_rows,
            &mut prover_challenger,
        )
        .expect_err("bad ALU arithmetic must not prove");
    }

    #[test]
    fn whir_native_local_alu_proves_horner_acc() {
        let expected_rows = vec![WhirNativeExpectedAluRow {
            kind: AluOpKind::HornerAcc,
            indices: [0, 1, 2, 3],
            acc_index: Some(4),
        }];
        let rows = vec![vec![
            EF::from(F::from_u64(alu_kind_to_tag(AluOpKind::HornerAcc) as u64)),
            EF::ZERO,
            EF::ONE,
            EF::from(F::TWO),
            EF::from(F::from_u64(3)),
            EF::from(F::from_u64(4)),
            EF::from(F::from_u64(5)),
            EF::from(F::from_u64(6)),
            EF::from(F::from_u64(7)),
        ]];
        let table = pack_rows::<EF>(
            WhirNativeTableKind::Alu,
            String::new(),
            rows,
            Some(ALU_WIDTH),
            WhirNativeCircuitOptions {
                openings_per_table: 1,
                min_num_variables: 4,
            },
        );
        let witness_table = test_witness_table(
            &[
                (0, EF::from(F::from_u64(4))),
                (1, EF::from(F::from_u64(5))),
                (2, EF::from(F::from_u64(6))),
                (3, EF::from(F::from_u64(7))),
                (4, EF::ONE),
            ],
            4,
        );

        let mut prover_challenger = TestChallenger::new();
        let proof = prove_alu_local_constraints::<F, EF, TestChallenger>(
            3,
            &table,
            &witness_table,
            &expected_rows,
            &mut prover_challenger,
        )
        .expect("prove HornerAcc local constraints");

        let mut verifier_challenger = TestChallenger::new();
        verify_alu_local_constraints::<F, EF, TestChallenger>(
            &proof,
            3,
            &table.metadata,
            &witness_table.metadata,
            &expected_rows,
            &mut verifier_challenger,
        )
        .expect("verify HornerAcc local constraints");
    }
}
