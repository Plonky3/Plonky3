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
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

#[cfg(feature = "std")]
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

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
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_poseidon2_circuit_air::{BabyBearD4Width16, extract_preprocessed_from_operations};
use p3_whir::constraints::statement::{
    BatchedLinearSigmaProverOracle, BatchedLinearSigmaReductionProof, EqStatement,
    LinearSigmaConstraint, LinearSigmaStatement, prove_batched_linear_sigma_reduction,
    verify_batched_linear_sigma_reduction,
};
use p3_whir::parameters::{ProtocolParameters, SumcheckStrategy};
use p3_whir::pcs::proof::WhirProof;
use p3_whir::pcs::{
    WhirBatchedDeferredProverOracle, WhirBatchedDeferredVerifierOracle,
    WhirExtensionDeferredProverData, WhirPcs, WhirSharedExtensionDeferredProverData,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::batch_stark_prover::BABY_BEAR_MODULUS;
use crate::whir_native_sumcheck::{
    WhirNativeSumcheckProof, point_from_prefix_current_suffix, verify_sumcheck,
};

const DIGEST_MIX: u64 = 1_099_511_627_761;
const TABLE_LAYOUT_VERSION: u32 = 2;

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

fn whir_native_column_batching_enabled() -> bool {
    #[cfg(feature = "std")]
    {
        std::env::var("P3_WHIR_NATIVE_COLUMN_BATCH").as_deref() == Ok("1")
    }
    #[cfg(not(feature = "std"))]
    {
        false
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
    Poseidon2Shift,
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
            Self::Poseidon2Shift => 7,
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

/// Opening backend used by a WHIR-native circuit proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WhirNativeOpeningMode {
    /// Legacy path: one packed row+column WHIR oracle and proof per table.
    PerTable,
    /// Batched path: one shared WHIR root per row-domain arity group.
    ColumnBatched,
}

impl WhirNativeOpeningMode {
    const fn tag(self) -> u64 {
        match self {
            Self::PerTable => 1,
            Self::ColumnBatched => 2,
        }
    }
}

/// One row-domain column included in a shared WHIR column batch.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhirNativeColumnRef {
    pub table_index: usize,
    pub column: usize,
}

/// Commitment to several same-arity row-domain columns under one shared root.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "Comm: Serialize", deserialize = "Comm: Deserialize<'de>"))]
pub struct WhirNativeColumnBatchCommitment<Comm> {
    pub num_variables: usize,
    pub columns: Vec<WhirNativeColumnRef>,
    pub commitment: Comm,
}

/// WHIR opening proof for one shared row-domain column batch.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize + Send + Sync + Clone, EF: Serialize, MT::Commitment: Serialize, MT::Proof: Serialize",
    deserialize = "F: Deserialize<'de> + Send + Sync + Clone, EF: Deserialize<'de>, MT::Commitment: Deserialize<'de>, MT::Proof: Deserialize<'de>"
))]
pub struct WhirNativeColumnBatchOpeningProof<F: Send + Sync + Clone, EF, MT: Mmcs<F>> {
    pub batch_index: usize,
    pub random_opening_values: Vec<Vec<EF>>,
    pub reduction_proof: BatchedLinearSigmaReductionProof<F, EF>,
    pub proof: WhirProof<F, EF, MT>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WhirNativeReadBusSectionKind {
    WitnessSender,
    KnownRows,
    Alu,
    Recompose,
    Poseidon2,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "EF: Serialize", deserialize = "EF: Deserialize<'de>"))]
pub struct WhirNativeReadBusSectionProof<EF> {
    pub table_index: usize,
    pub kind: WhirNativeReadBusSectionKind,
    pub port: u32,
    pub active_reads: usize,
    pub cumulative: EF,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "EF: Serialize", deserialize = "EF: Deserialize<'de>"))]
pub struct WhirNativeReadBusProof<EF> {
    pub alpha: EF,
    pub beta: EF,
    pub witness_read_counts: Vec<u32>,
    pub sender_cumulative: EF,
    pub receiver_cumulative: EF,
    pub final_difference: EF,
    pub sections: Vec<WhirNativeReadBusSectionProof<EF>>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Send + Sync + Clone + Serialize, EF: Serialize, MT::Commitment: Serialize, MT::Proof: Serialize",
    deserialize = "F: Send + Sync + Clone + Deserialize<'de>, EF: Deserialize<'de>, MT::Commitment: Deserialize<'de>, MT::Proof: Deserialize<'de>"
))]
pub struct WhirNativePoseidon2ShiftBusProof<F: Send + Sync + Clone, EF, MT: Mmcs<F>> {
    pub theta: EF,
    pub alpha: EF,
    pub beta: EF,
    pub sections: Vec<WhirNativePoseidon2ShiftSectionProof<F, EF, MT>>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Send + Sync + Clone + Serialize, EF: Serialize, MT::Commitment: Serialize, MT::Proof: Serialize",
    deserialize = "F: Send + Sync + Clone + Deserialize<'de>, EF: Deserialize<'de>, MT::Commitment: Deserialize<'de>, MT::Proof: Deserialize<'de>"
))]
pub struct WhirNativePoseidon2ShiftSectionProof<F: Send + Sync + Clone, EF, MT: Mmcs<F>> {
    pub table_index: usize,
    pub active_rows: usize,
    pub sender_cumulative: EF,
    pub receiver_cumulative: EF,
    pub final_difference: EF,
    pub inverse_commitment: WhirNativeTableCommitment<MT::Commitment>,
    pub degree: usize,
    pub constraint_challenge: EF,
    pub sum_challenge: EF,
    pub zerocheck_point: Vec<EF>,
    pub terminal_row_point: Vec<EF>,
    pub terminal_claim: EF,
    pub sumcheck: WhirNativeSumcheckProof<EF>,
    pub terminal_main_openings: Vec<WhirNativeTerminalColumnClaim<EF>>,
    pub terminal_inverse_openings: Vec<WhirNativeTerminalColumnClaim<EF>>,
    pub inverse_opening_proof: WhirNativeTableOpeningProof<F, EF, MT>,
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
    pub acc_value: EF,
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
    pub opening_mode: WhirNativeOpeningMode,
    pub table_commitments: Vec<WhirNativeTableCommitment<MT::Commitment>>,
    pub column_batch_commitments: Vec<WhirNativeColumnBatchCommitment<MT::Commitment>>,
    pub read_bus_proof: WhirNativeReadBusProof<EF>,
    pub poseidon2_shift_bus_proof: WhirNativePoseidon2ShiftBusProof<F, EF, MT>,
    pub constraint_sumcheck_proofs: Vec<WhirNativeConstraintSumcheckProof<EF>>,
    pub opening_proofs: Vec<WhirNativeTableOpeningProof<F, EF, MT>>,
    pub column_batch_opening_proofs: Vec<WhirNativeColumnBatchOpeningProof<F, EF, MT>>,
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

#[derive(Clone, Debug)]
struct WhirNativeColumnBatchLayout {
    num_variables: usize,
    columns: Vec<WhirNativeColumnRef>,
}

struct WhirNativeColumnBatchProverData<F, EF, MT, const DIGEST_ELEMS: usize>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    layout: WhirNativeColumnBatchLayout,
    commitment: MT::Commitment,
    prover_data: Arc<WhirSharedExtensionDeferredProverData<F, EF, MT, DIGEST_ELEMS>>,
    polys: Vec<Poly<EF>>,
}

#[derive(Clone, Copy)]
enum WhirNativeCommitmentContext<'a, Comm> {
    PerTable(&'a [WhirNativeTableCommitment<Comm>]),
    ColumnBatched {
        metadata: &'a [WhirNativeTableMetadata],
        column_batches: &'a [WhirNativeColumnBatchCommitment<Comm>],
    },
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
    #[error("column batch mismatch at batch {batch_index}")]
    ColumnBatchMismatch { batch_index: usize },
    #[error("WHIR column batch verification failed at batch {batch_index}: {details}")]
    WhirColumnBatchVerificationFailed { batch_index: usize, details: String },
    #[error("constraint violation: {0}")]
    ConstraintViolation(String),
}

fn unsupported_poseidon2_component(op_type: &str) -> WhirNativeCircuitError {
    WhirNativeCircuitError::UnsupportedSoundComponent(format!(
        "non-primitive table `{op_type}` requires a WHIR-native Poseidon2/MMCS AIR proof before it can enter comparison timing"
    ))
}

#[cfg(feature = "std")]
type WhirNativeDiagnosticInstant = Option<std::time::Instant>;
#[cfg(not(feature = "std"))]
type WhirNativeDiagnosticInstant = ();

#[derive(Clone, Debug)]
struct WhirNativeDiagnostics {
    #[cfg(feature = "std")]
    print_phases: bool,
    #[cfg(feature = "std")]
    json_path: Option<String>,
    #[cfg(feature = "std")]
    context: WhirNativeDiagnosticContext,
}

#[cfg(feature = "std")]
#[derive(Clone, Debug)]
struct WhirNativeDiagnosticContext {
    k: Option<usize>,
    n: Option<usize>,
    arity: Option<usize>,
}

impl WhirNativeDiagnostics {
    fn from_env() -> Self {
        #[cfg(feature = "std")]
        {
            Self {
                print_phases: std::env::var("P3_WHIR_NATIVE_PHASES").as_deref() == Ok("1"),
                json_path: std::env::var("P3_WHIR_NATIVE_JSON")
                    .ok()
                    .filter(|path| !path.trim().is_empty()),
                context: WhirNativeDiagnosticContext {
                    k: read_usize_env("P3_WARP_SUMCHECK_K"),
                    n: read_usize_env("P3_WARP_SUMCHECK_N"),
                    arity: read_usize_env("P3_WARP_ARITY"),
                },
            }
        }
        #[cfg(not(feature = "std"))]
        {
            Self {}
        }
    }

    #[cfg(feature = "std")]
    fn active(&self) -> bool {
        self.print_phases || self.json_path.is_some()
    }

    fn start(&self) -> WhirNativeDiagnosticInstant {
        #[cfg(feature = "std")]
        {
            self.active().then(std::time::Instant::now)
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = self;
        }
    }

    fn record_phase(
        &self,
        phase: &str,
        start: WhirNativeDiagnosticInstant,
        table_count: Option<usize>,
        opening_claim_count: Option<usize>,
        proof_byte_count: Option<usize>,
    ) {
        #[cfg(feature = "std")]
        {
            let Some(start) = start else {
                return;
            };
            let elapsed = start.elapsed();
            if self.print_phases {
                eprintln!(
                    "[whir-native] phase={phase} elapsed={elapsed:?} table_count={} opening_claims={} proof_bytes={}",
                    text_optional_usize(table_count),
                    text_optional_usize(opening_claim_count),
                    text_optional_usize(proof_byte_count),
                );
            }
            if self.json_path.is_some() {
                self.append_jsonl(&format!(
                    concat!(
                        "{{",
                        "\"component\":\"whir_native_outer_proof\",",
                        "\"event\":\"phase\",",
                        "\"k\":{},",
                        "\"n\":{},",
                        "\"arity\":{},",
                        "\"phase\":{},",
                        "\"duration_nanos\":{},",
                        "\"table_count\":{},",
                        "\"opening_claim_count\":{},",
                        "\"proof_byte_count\":{}",
                        "}}"
                    ),
                    json_optional_usize(self.context.k),
                    json_optional_usize(self.context.n),
                    json_optional_usize(self.context.arity),
                    json_string(phase),
                    elapsed.as_nanos(),
                    json_optional_usize(table_count),
                    json_optional_usize(opening_claim_count),
                    json_optional_usize(proof_byte_count),
                ));
            }
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = (
                phase,
                start,
                table_count,
                opening_claim_count,
                proof_byte_count,
            );
        }
    }

    fn record_table(
        &self,
        phase: &str,
        start: WhirNativeDiagnosticInstant,
        table_index: usize,
        metadata: &WhirNativeTableMetadata,
        opening_claim_count: usize,
    ) {
        #[cfg(feature = "std")]
        {
            let Some(start) = start else {
                return;
            };
            let elapsed = start.elapsed();
            let op_type = if metadata.op_type.is_empty() {
                "-"
            } else {
                metadata.op_type.as_str()
            };
            if self.print_phases {
                eprintln!(
                    "[whir-native] table phase={phase} table_index={table_index} kind={:?} op_type={} active_rows={} padded_height={} padded_width={} num_variables={} opening_claims={} elapsed={elapsed:?}",
                    metadata.kind,
                    op_type,
                    metadata.active_rows,
                    metadata.padded_height,
                    metadata.padded_width,
                    metadata.num_variables,
                    opening_claim_count,
                );
            }
            if self.json_path.is_some() {
                self.append_jsonl(&format!(
                    concat!(
                        "{{",
                        "\"component\":\"whir_native_outer_proof\",",
                        "\"event\":\"table_subphase\",",
                        "\"k\":{},",
                        "\"n\":{},",
                        "\"arity\":{},",
                        "\"phase\":{},",
                        "\"duration_nanos\":{},",
                        "\"table_index\":{},",
                        "\"kind\":{},",
                        "\"op_type\":{},",
                        "\"active_rows\":{},",
                        "\"padded_height\":{},",
                        "\"padded_width\":{},",
                        "\"width\":{},",
                        "\"num_variables\":{},",
                        "\"opening_claim_count\":{}",
                        "}}"
                    ),
                    json_optional_usize(self.context.k),
                    json_optional_usize(self.context.n),
                    json_optional_usize(self.context.arity),
                    json_string(phase),
                    elapsed.as_nanos(),
                    table_index,
                    json_string(&format!("{:?}", metadata.kind)),
                    json_string(&metadata.op_type),
                    metadata.active_rows,
                    metadata.padded_height,
                    metadata.padded_width,
                    metadata.width,
                    metadata.num_variables,
                    opening_claim_count,
                ));
            }
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = (phase, start, table_index, metadata, opening_claim_count);
        }
    }

    #[cfg(feature = "std")]
    fn append_jsonl(&self, line: &str) {
        let Some(path) = &self.json_path else {
            return;
        };
        use std::io::Write as _;
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            Ok(mut file) => {
                if let Err(err) = writeln!(file, "{line}") {
                    eprintln!("[whir-native] failed to write JSONL `{path}`: {err}");
                }
            }
            Err(err) => eprintln!("[whir-native] failed to open JSONL `{path}`: {err}"),
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
enum WhirNativeLocalDiagMetric {
    TableColumn,
    StaticSelector,
    WitnessAddress,
    WitnessValue,
    Poseidon2NextRow,
    ConstraintBatch,
}

#[derive(Debug)]
struct WhirNativeLocalProofDiagnostics {
    kind: WhirNativeLocalConstraintKind,
    table_index: usize,
    #[cfg(feature = "std")]
    op_type: String,
    #[cfg(feature = "std")]
    enabled: bool,
    #[cfg(feature = "std")]
    json_path: Option<String>,
    #[cfg(feature = "std")]
    context: WhirNativeDiagnosticContext,
    #[cfg(feature = "std")]
    table_column_calls: AtomicUsize,
    #[cfg(feature = "std")]
    table_column_nanos: AtomicU64,
    #[cfg(feature = "std")]
    static_selector_calls: AtomicUsize,
    #[cfg(feature = "std")]
    static_selector_nanos: AtomicU64,
    #[cfg(feature = "std")]
    witness_address_calls: AtomicUsize,
    #[cfg(feature = "std")]
    witness_address_nanos: AtomicU64,
    #[cfg(feature = "std")]
    witness_value_calls: AtomicUsize,
    #[cfg(feature = "std")]
    witness_value_nanos: AtomicU64,
    #[cfg(feature = "std")]
    poseidon2_next_row_calls: AtomicUsize,
    #[cfg(feature = "std")]
    poseidon2_next_row_nanos: AtomicU64,
    #[cfg(feature = "std")]
    constraint_batch_calls: AtomicUsize,
    #[cfg(feature = "std")]
    constraint_batch_nanos: AtomicU64,
}

impl WhirNativeLocalProofDiagnostics {
    fn new(
        kind: WhirNativeLocalConstraintKind,
        table_index: usize,
        metadata: &WhirNativeTableMetadata,
    ) -> Self {
        #[cfg(feature = "std")]
        {
            let json_path = std::env::var("P3_WHIR_NATIVE_JSON")
                .ok()
                .filter(|path| !path.trim().is_empty());
            Self {
                kind,
                table_index,
                op_type: metadata.op_type.clone(),
                enabled: std::env::var("P3_WHIR_NATIVE_PHASES").as_deref() == Ok("1")
                    || json_path.is_some(),
                json_path,
                context: WhirNativeDiagnosticContext {
                    k: read_usize_env("P3_WARP_SUMCHECK_K"),
                    n: read_usize_env("P3_WARP_SUMCHECK_N"),
                    arity: read_usize_env("P3_WARP_ARITY"),
                },
                table_column_calls: AtomicUsize::new(0),
                table_column_nanos: AtomicU64::new(0),
                static_selector_calls: AtomicUsize::new(0),
                static_selector_nanos: AtomicU64::new(0),
                witness_address_calls: AtomicUsize::new(0),
                witness_address_nanos: AtomicU64::new(0),
                witness_value_calls: AtomicUsize::new(0),
                witness_value_nanos: AtomicU64::new(0),
                poseidon2_next_row_calls: AtomicUsize::new(0),
                poseidon2_next_row_nanos: AtomicU64::new(0),
                constraint_batch_calls: AtomicUsize::new(0),
                constraint_batch_nanos: AtomicU64::new(0),
            }
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = metadata;
            Self { kind, table_index }
        }
    }

    #[cfg(feature = "std")]
    fn record_count(&self, metric: WhirNativeLocalDiagMetric, calls: usize) {
        if !self.enabled || calls == 0 {
            return;
        }
        match metric {
            WhirNativeLocalDiagMetric::TableColumn => {
                self.table_column_calls.fetch_add(calls, Ordering::Relaxed);
            }
            WhirNativeLocalDiagMetric::StaticSelector => {
                self.static_selector_calls
                    .fetch_add(calls, Ordering::Relaxed);
            }
            WhirNativeLocalDiagMetric::WitnessAddress => {
                self.witness_address_calls
                    .fetch_add(calls, Ordering::Relaxed);
            }
            WhirNativeLocalDiagMetric::WitnessValue => {
                self.witness_value_calls.fetch_add(calls, Ordering::Relaxed);
            }
            WhirNativeLocalDiagMetric::Poseidon2NextRow => {
                self.poseidon2_next_row_calls
                    .fetch_add(calls, Ordering::Relaxed);
            }
            WhirNativeLocalDiagMetric::ConstraintBatch => {
                self.constraint_batch_calls
                    .fetch_add(calls, Ordering::Relaxed);
            }
        }
    }

    fn time<T>(&self, metric: WhirNativeLocalDiagMetric, calls: usize, f: impl FnOnce() -> T) -> T {
        #[cfg(feature = "std")]
        {
            if !self.enabled {
                return f();
            }
            let start = std::time::Instant::now();
            let result = f();
            let nanos = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            self.record_count(metric, calls);
            match metric {
                WhirNativeLocalDiagMetric::TableColumn => {
                    self.table_column_nanos.fetch_add(nanos, Ordering::Relaxed)
                }
                WhirNativeLocalDiagMetric::StaticSelector => self
                    .static_selector_nanos
                    .fetch_add(nanos, Ordering::Relaxed),
                WhirNativeLocalDiagMetric::WitnessAddress => self
                    .witness_address_nanos
                    .fetch_add(nanos, Ordering::Relaxed),
                WhirNativeLocalDiagMetric::WitnessValue => {
                    self.witness_value_nanos.fetch_add(nanos, Ordering::Relaxed)
                }
                WhirNativeLocalDiagMetric::Poseidon2NextRow => self
                    .poseidon2_next_row_nanos
                    .fetch_add(nanos, Ordering::Relaxed),
                WhirNativeLocalDiagMetric::ConstraintBatch => self
                    .constraint_batch_nanos
                    .fetch_add(nanos, Ordering::Relaxed),
            };
            result
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = (metric, calls);
            f()
        }
    }

    fn finish(&self) {
        #[cfg(feature = "std")]
        {
            if !self.enabled {
                return;
            }
            let op_type = if self.op_type.is_empty() {
                "-"
            } else {
                self.op_type.as_str()
            };
            let table_column_calls = self.table_column_calls.load(Ordering::Relaxed);
            let table_column_nanos = self.table_column_nanos.load(Ordering::Relaxed);
            let static_selector_calls = self.static_selector_calls.load(Ordering::Relaxed);
            let static_selector_nanos = self.static_selector_nanos.load(Ordering::Relaxed);
            let witness_address_calls = self.witness_address_calls.load(Ordering::Relaxed);
            let witness_address_nanos = self.witness_address_nanos.load(Ordering::Relaxed);
            let witness_value_calls = self.witness_value_calls.load(Ordering::Relaxed);
            let witness_value_nanos = self.witness_value_nanos.load(Ordering::Relaxed);
            let poseidon2_next_row_calls = self.poseidon2_next_row_calls.load(Ordering::Relaxed);
            let poseidon2_next_row_nanos = self.poseidon2_next_row_nanos.load(Ordering::Relaxed);
            let constraint_batch_calls = self.constraint_batch_calls.load(Ordering::Relaxed);
            let constraint_batch_nanos = self.constraint_batch_nanos.load(Ordering::Relaxed);
            eprintln!(
                "[whir-native] local-detail table_index={} kind={:?} op_type={} table_column_evals={} table_column_elapsed_ns={} static_selector_evals={} static_selector_elapsed_ns={} witness_address_evals={} witness_address_elapsed_ns={} witness_value_mles={} witness_value_elapsed_ns={} poseidon2_next_row_evals={} poseidon2_next_row_elapsed_ns={} constraint_batches={} constraint_batch_elapsed_ns={}",
                self.table_index,
                self.kind,
                op_type,
                table_column_calls,
                table_column_nanos,
                static_selector_calls,
                static_selector_nanos,
                witness_address_calls,
                witness_address_nanos,
                witness_value_calls,
                witness_value_nanos,
                poseidon2_next_row_calls,
                poseidon2_next_row_nanos,
                constraint_batch_calls,
                constraint_batch_nanos,
            );
            if self.json_path.is_some() {
                self.append_jsonl(&format!(
                    concat!(
                        "{{",
                        "\"component\":\"whir_native_outer_proof\",",
                        "\"event\":\"local_detail\",",
                        "\"k\":{},",
                        "\"n\":{},",
                        "\"arity\":{},",
                        "\"table_index\":{},",
                        "\"kind\":{},",
                        "\"op_type\":{},",
                        "\"table_column_evals\":{},",
                        "\"table_column_elapsed_ns\":{},",
                        "\"static_selector_evals\":{},",
                        "\"static_selector_elapsed_ns\":{},",
                        "\"witness_address_evals\":{},",
                        "\"witness_address_elapsed_ns\":{},",
                        "\"witness_value_mles\":{},",
                        "\"witness_value_elapsed_ns\":{},",
                        "\"poseidon2_next_row_evals\":{},",
                        "\"poseidon2_next_row_elapsed_ns\":{},",
                        "\"constraint_batches\":{},",
                        "\"constraint_batch_elapsed_ns\":{}",
                        "}}"
                    ),
                    json_optional_usize(self.context.k),
                    json_optional_usize(self.context.n),
                    json_optional_usize(self.context.arity),
                    self.table_index,
                    json_string(&format!("{:?}", self.kind)),
                    json_string(&self.op_type),
                    table_column_calls,
                    table_column_nanos,
                    static_selector_calls,
                    static_selector_nanos,
                    witness_address_calls,
                    witness_address_nanos,
                    witness_value_calls,
                    witness_value_nanos,
                    poseidon2_next_row_calls,
                    poseidon2_next_row_nanos,
                    constraint_batch_calls,
                    constraint_batch_nanos,
                ));
            }
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = (self.kind, self.table_index);
        }
    }

    #[cfg(feature = "std")]
    fn append_jsonl(&self, line: &str) {
        let Some(path) = &self.json_path else {
            return;
        };
        use std::io::Write as _;
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            Ok(mut file) => {
                if let Err(err) = writeln!(file, "{line}") {
                    eprintln!("[whir-native] failed to write JSONL `{path}`: {err}");
                }
            }
            Err(err) => eprintln!("[whir-native] failed to open JSONL `{path}`: {err}"),
        }
    }
}

#[cfg(feature = "std")]
fn read_usize_env(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

#[cfg(feature = "std")]
fn text_optional_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "-".to_string(), |value| value.to_string())
}

#[cfg(feature = "std")]
fn json_optional_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

#[cfg(feature = "std")]
fn json_string(value: &str) -> String {
    let mut out = String::from("\"");
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

#[cfg(feature = "std")]
fn whir_native_proof_byte_count<F, EF, MT>(
    proof: &WhirNativeCircuitProof<F, EF, MT>,
) -> Option<usize>
where
    F: Send + Sync + Clone + Serialize,
    EF: Serialize,
    MT: Mmcs<F>,
    MT::Commitment: Serialize,
    MT::Proof: Serialize,
{
    postcard::to_allocvec(proof).ok().map(|bytes| bytes.len())
}

#[cfg(not(feature = "std"))]
fn whir_native_proof_byte_count<F, EF, MT>(
    _proof: &WhirNativeCircuitProof<F, EF, MT>,
) -> Option<usize>
where
    F: Send + Sync + Clone,
    MT: Mmcs<F>,
{
    None
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

    let diagnostics = WhirNativeDiagnostics::from_env();
    let total_start = diagnostics.start();

    let phase_start = diagnostics.start();
    let payload = trace_payload_from_traces::<F, EF>(circuit, traces)?;
    diagnostics.record_phase("trace_payload_from_traces", phase_start, None, None, None);
    let public_io_digest = compute_public_io_digest::<F, EF>(public_inputs);
    let shape_digest = compute_shape_digest::<F, EF>(circuit);

    let phase_start = diagnostics.start();
    let tables = build_tables(&payload, options)?;
    diagnostics.record_phase("build_tables", phase_start, Some(tables.len()), None, None);
    let phase_start = diagnostics.start();
    let expected_metadata =
        whir_native_expected_table_metadata::<F, EF>(circuit, public_inputs, options)?;
    diagnostics.record_phase(
        "expected_table_metadata",
        phase_start,
        Some(expected_metadata.len()),
        None,
        None,
    );
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

    let opening_mode = if whir_native_column_batching_enabled() {
        WhirNativeOpeningMode::ColumnBatched
    } else {
        WhirNativeOpeningMode::PerTable
    };
    let column_batch_layouts = if opening_mode == WhirNativeOpeningMode::ColumnBatched {
        build_column_batch_layouts(&expected_metadata, options)
    } else {
        Vec::new()
    };
    let mut table_commitments = Vec::with_capacity(tables.len());
    let mut table_prover_data = Vec::with_capacity(tables.len());
    let mut table_challengers = Vec::with_capacity(tables.len());
    let mut column_batch_commitments = Vec::with_capacity(column_batch_layouts.len());
    let mut column_batch_prover_data = Vec::with_capacity(column_batch_layouts.len());

    match opening_mode {
        WhirNativeOpeningMode::PerTable => {
            let phase_start = diagnostics.start();
            for (table_index, table) in tables.iter().enumerate() {
                let table_start = diagnostics.start();
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
                diagnostics.record_table(
                    "table_commitment",
                    table_start,
                    table_index,
                    &table.metadata,
                    0,
                );
            }
            diagnostics.record_phase(
                "table_commitments",
                phase_start,
                Some(tables.len()),
                None,
                None,
            );
        }
        WhirNativeOpeningMode::ColumnBatched => {
            let phase_start = diagnostics.start();
            for layout in &column_batch_layouts {
                let pcs = make_pcs(layout.num_variables);
                let mut challenger = make_challenger();
                let mut matrices = Vec::with_capacity(layout.columns.len());
                let mut polys = Vec::with_capacity(layout.columns.len());
                for column_ref in &layout.columns {
                    let table = tables.get(column_ref.table_index).ok_or_else(|| {
                        WhirNativeCircuitError::ConstraintViolation(format!(
                            "column batch table {} out of range",
                            column_ref.table_index
                        ))
                    })?;
                    let values = column_batch_values::<F, EF>(
                        table,
                        column_ref.column,
                        layout.num_variables,
                    )?;
                    polys.push(Poly::new(values.clone()));
                    matrices.push(RowMajorMatrix::new(values, 1));
                }
                let (commitment, prover_data) =
                    pcs.commit_extension_batch_deferred(matrices, &mut challenger);
                column_batch_commitments.push(WhirNativeColumnBatchCommitment {
                    num_variables: layout.num_variables,
                    columns: layout.columns.clone(),
                    commitment: commitment.clone(),
                });
                column_batch_prover_data.push(WhirNativeColumnBatchProverData {
                    layout: layout.clone(),
                    commitment,
                    prover_data,
                    polys,
                });
            }
            diagnostics.record_phase(
                "column_batch_commitments",
                phase_start,
                Some(column_batch_layouts.len()),
                Some(
                    column_batch_layouts
                        .iter()
                        .map(|layout| layout.columns.len())
                        .sum(),
                ),
                None,
            );
        }
    }

    let commitment_context = match opening_mode {
        WhirNativeOpeningMode::PerTable => {
            WhirNativeCommitmentContext::PerTable(table_commitments.as_slice())
        }
        WhirNativeOpeningMode::ColumnBatched => WhirNativeCommitmentContext::ColumnBatched {
            metadata: expected_metadata.as_slice(),
            column_batches: column_batch_commitments.as_slice(),
        },
    };
    let phase_start = diagnostics.start();
    let mut read_bus_challenger = make_challenger();
    observe_read_bus_challenge_context(
        &mut read_bus_challenger,
        &public_io_digest,
        &shape_digest,
        options,
        commitment_context.clone(),
    );
    let read_bus_alpha = read_bus_challenger.sample_algebra_element();
    let read_bus_beta = read_bus_challenger.sample_algebra_element();
    let read_bus_proof = prove_read_bus::<F, EF>(
        circuit,
        &tables,
        &expected_metadata,
        read_bus_alpha,
        read_bus_beta,
    )?;
    diagnostics.record_phase(
        "read_bus_proof",
        phase_start,
        Some(tables.len()),
        Some(read_bus_proof.sections.len()),
        None,
    );
    let phase_start = diagnostics.start();
    let (poseidon2_shift_bus_proof, poseidon2_shift_terminal_claims) = prove_poseidon2_shift_bus::<
        F,
        EF,
        MT,
        Challenger,
        Dft,
        MakePcs,
        MakeChallenger,
        DIGEST_ELEMS,
    >(
        &tables,
        &expected_metadata,
        &public_io_digest,
        &shape_digest,
        options,
        commitment_context.clone(),
        &make_pcs,
        &make_challenger,
    )?;
    diagnostics.record_phase(
        "poseidon2_shift_bus_proof",
        phase_start,
        Some(poseidon2_shift_bus_proof.sections.len()),
        Some(poseidon2_shift_terminal_claims.len()),
        None,
    );
    let mut constraint_sumcheck_proofs = Vec::with_capacity(tables.len());
    let mut terminal_claims_by_table = vec![Vec::new(); tables.len()];
    let mut terminal_claims_by_batch = vec![Vec::new(); column_batch_layouts.len()];
    let mut total_terminal_claims = 0usize;
    for claim in poseidon2_shift_terminal_claims {
        let point = Point::new(claim.point.clone());
        terminal_claims_by_table[claim.table_index].push((point, claim.value));
        total_terminal_claims += 1;
        if opening_mode == WhirNativeOpeningMode::ColumnBatched {
            let (batch_index, column_offset) = column_batch_index_and_offset(
                &column_batch_layouts,
                claim.table_index,
                claim.column,
            )
            .ok_or_else(|| {
                WhirNativeCircuitError::ConstraintViolation(format!(
                    "missing column batch for Poseidon2 shift claim table {}, column {}",
                    claim.table_index, claim.column
                ))
            })?;
            terminal_claims_by_batch[batch_index].push((column_offset, claim));
        }
    }
    let phase_start = diagnostics.start();
    for (table_index, table) in tables.iter().enumerate() {
        let table_start = diagnostics.start();
        let local_proof = prove_table_local_constraints::<F, EF, MT, Challenger, MakeChallenger>(
            circuit,
            public_inputs,
            table_index,
            &tables,
            commitment_context.clone(),
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
        let terminal_claims =
            local_proof_terminal_claims::<EF>(local_proof.as_ref(), &expected_metadata)?;
        let table_terminal_claim_count = terminal_claims.len();
        total_terminal_claims += table_terminal_claim_count;
        for (claim_table_index, point, value) in terminal_claims {
            terminal_claims_by_table[claim_table_index].push((point, value));
        }
        if opening_mode == WhirNativeOpeningMode::ColumnBatched {
            for claim in
                local_proof_terminal_column_claims::<EF>(local_proof.as_ref(), &expected_metadata)?
            {
                let (batch_index, column_offset) = column_batch_index_and_offset(
                    &column_batch_layouts,
                    claim.table_index,
                    claim.column,
                )
                .ok_or_else(|| {
                    WhirNativeCircuitError::ConstraintViolation(format!(
                        "missing column batch for terminal claim table {}, column {}",
                        claim.table_index, claim.column
                    ))
                })?;
                terminal_claims_by_batch[batch_index].push((column_offset, claim));
            }
        }
        constraint_sumcheck_proofs.push(WhirNativeConstraintSumcheckProof {
            table_index,
            checked_constraints: table.metadata.active_rows,
            claimed_zero_sum: EF::ZERO,
            local_proof,
        });
        diagnostics.record_table(
            "local_constraint_proof",
            table_start,
            table_index,
            &table.metadata,
            table_terminal_claim_count,
        );
    }
    diagnostics.record_phase(
        "local_constraint_proofs",
        phase_start,
        Some(tables.len()),
        Some(total_terminal_claims),
        None,
    );

    let mut opening_proofs = Vec::new();
    let mut column_batch_opening_proofs = Vec::new();
    let mut total_opening_claims = 0usize;
    match opening_mode {
        WhirNativeOpeningMode::PerTable => {
            opening_proofs = Vec::with_capacity(tables.len());
            let phase_start = diagnostics.start();
            for (table_index, ((table, prover_data), mut challenger)) in tables
                .iter()
                .zip(table_prover_data)
                .zip(table_challengers)
                .enumerate()
            {
                let table_start = diagnostics.start();
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
                let opening_claims: Vec<(Vec<EF>, EF)> = points
                    .into_iter()
                    .zip(opened_values[0].iter().copied())
                    .map(|(point, value)| (point.as_slice().to_vec(), value))
                    .collect();
                let opening_claim_count = opening_claims.len();
                total_opening_claims += opening_claim_count;

                opening_proofs.push(WhirNativeTableOpeningProof {
                    table_index,
                    opening_claims,
                    proof,
                });
                diagnostics.record_table(
                    "whir_opening",
                    table_start,
                    table_index,
                    &table.metadata,
                    opening_claim_count,
                );
            }
            diagnostics.record_phase(
                "whir_openings",
                phase_start,
                Some(tables.len()),
                Some(total_opening_claims),
                None,
            );
        }
        WhirNativeOpeningMode::ColumnBatched => {
            column_batch_opening_proofs = Vec::with_capacity(column_batch_layouts.len());
            let phase_start = diagnostics.start();
            for (batch_index, batch) in column_batch_prover_data.into_iter().enumerate() {
                let pcs = make_pcs(batch.layout.num_variables);
                let batch_commitment = column_batch_commitments
                    .get(batch_index)
                    .ok_or_else(|| WhirNativeCircuitError::ColumnBatchMismatch { batch_index })?;
                if batch.commitment != batch_commitment.commitment {
                    return Err(WhirNativeCircuitError::ColumnBatchMismatch { batch_index });
                }
                let mut challenger = make_challenger();
                observe_column_batch_opening_context(
                    &mut challenger,
                    &public_io_digest,
                    &shape_digest,
                    batch_index,
                    batch_commitment,
                    &expected_metadata,
                    options,
                );
                let random_points = sample_column_batch_opening_points::<F, EF, Challenger>(
                    &mut challenger,
                    batch.layout.num_variables,
                    batch_index,
                    options.openings_per_table,
                );
                let random_opening_values = random_points
                    .iter()
                    .map(|point| {
                        batch
                            .polys
                            .iter()
                            .map(|poly| poly.eval_ext::<F>(point))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let statements = build_column_batch_statements::<F, EF>(
                    &batch.layout,
                    &expected_metadata,
                    &random_points,
                    &random_opening_values,
                    &terminal_claims_by_batch[batch_index],
                )?;
                let oracles = statements
                    .iter()
                    .zip(&batch.polys)
                    .map(|(statement, poly)| {
                        BatchedLinearSigmaProverOracle::extension(statement, poly)
                    })
                    .collect::<Vec<_>>();
                let (reduction_proof, residual_claim) =
                    prove_batched_linear_sigma_reduction::<F, EF, Challenger>(
                        &oracles,
                        &mut challenger,
                        0,
                    )
                    .map_err(|err| {
                        WhirNativeCircuitError::WhirColumnBatchVerificationFailed {
                            batch_index,
                            details: format!("{err:?}"),
                        }
                    })?;
                let proof = pcs
                    .open_grouped_batched_deferred(
                        vec![WhirBatchedDeferredProverOracle::SharedExtension {
                            coeffs: residual_claim.coeffs,
                            data: batch.prover_data,
                        }],
                        residual_claim.point,
                        residual_claim.value,
                        &mut challenger,
                    )
                    .map_err(
                        |err| WhirNativeCircuitError::WhirColumnBatchVerificationFailed {
                            batch_index,
                            details: format!("{err:?}"),
                        },
                    )?;
                total_opening_claims += random_opening_values.iter().map(Vec::len).sum::<usize>()
                    + terminal_claims_by_batch[batch_index].len();
                column_batch_opening_proofs.push(WhirNativeColumnBatchOpeningProof {
                    batch_index,
                    random_opening_values,
                    reduction_proof,
                    proof,
                });
            }
            diagnostics.record_phase(
                "column_batch_whir_openings",
                phase_start,
                Some(column_batch_layouts.len()),
                Some(total_opening_claims),
                None,
            );
        }
    }

    let phase_start = diagnostics.start();
    let proof = WhirNativeCircuitProof {
        opening_mode,
        table_commitments,
        column_batch_commitments,
        read_bus_proof,
        poseidon2_shift_bus_proof,
        constraint_sumcheck_proofs,
        opening_proofs,
        column_batch_opening_proofs,
        public_io_digest,
        shape_digest,
    };
    let proof_byte_count = whir_native_proof_byte_count(&proof);
    diagnostics.record_phase(
        "final_proof_assembly",
        phase_start,
        Some(tables.len()),
        Some(total_opening_claims),
        proof_byte_count,
    );

    let phase_start = diagnostics.start();
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
    diagnostics.record_phase(
        "verify_whir_native_circuit_proof",
        phase_start,
        Some(tables.len()),
        Some(total_opening_claims),
        None,
    );
    diagnostics.record_phase(
        "total",
        total_start,
        Some(tables.len()),
        Some(total_opening_claims),
        proof_byte_count,
    );

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

    let expected_metadata =
        whir_native_expected_table_metadata::<F, EF>(circuit, public_inputs, options)?;
    if proof.constraint_sumcheck_proofs.len() != expected_metadata.len() {
        return Err(WhirNativeCircuitError::TableCountMismatch {
            expected: expected_metadata.len(),
            got: proof.constraint_sumcheck_proofs.len(),
        });
    }

    let column_batch_layouts = build_column_batch_layouts(&expected_metadata, options);
    match proof.opening_mode {
        WhirNativeOpeningMode::PerTable => {
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
            if !proof.column_batch_commitments.is_empty()
                || !proof.column_batch_opening_proofs.is_empty()
            {
                return Err(WhirNativeCircuitError::ConstraintViolation(
                    "per-table proof carries column batch data".to_string(),
                ));
            }
            for (table_index, metadata) in expected_metadata.iter().enumerate() {
                let table_commitment = &proof.table_commitments[table_index];
                if &table_commitment.metadata != metadata {
                    return Err(WhirNativeCircuitError::TableMetadataMismatch { table_index });
                }
            }
        }
        WhirNativeOpeningMode::ColumnBatched => {
            if !proof.table_commitments.is_empty() || !proof.opening_proofs.is_empty() {
                return Err(WhirNativeCircuitError::ConstraintViolation(
                    "column-batched proof carries per-table opening data".to_string(),
                ));
            }
            if proof.column_batch_commitments.len() != column_batch_layouts.len() {
                return Err(WhirNativeCircuitError::TableCountMismatch {
                    expected: column_batch_layouts.len(),
                    got: proof.column_batch_commitments.len(),
                });
            }
            if proof.column_batch_opening_proofs.len() != column_batch_layouts.len() {
                return Err(WhirNativeCircuitError::TableCountMismatch {
                    expected: column_batch_layouts.len(),
                    got: proof.column_batch_opening_proofs.len(),
                });
            }
            for (batch_index, (layout, commitment)) in column_batch_layouts
                .iter()
                .zip(&proof.column_batch_commitments)
                .enumerate()
            {
                if commitment.num_variables != layout.num_variables
                    || commitment.columns != layout.columns
                {
                    return Err(WhirNativeCircuitError::ColumnBatchMismatch { batch_index });
                }
            }
        }
    }

    let commitment_context = match proof.opening_mode {
        WhirNativeOpeningMode::PerTable => {
            WhirNativeCommitmentContext::PerTable(proof.table_commitments.as_slice())
        }
        WhirNativeOpeningMode::ColumnBatched => WhirNativeCommitmentContext::ColumnBatched {
            metadata: expected_metadata.as_slice(),
            column_batches: proof.column_batch_commitments.as_slice(),
        },
    };
    let mut read_bus_challenger = make_challenger();
    observe_read_bus_challenge_context(
        &mut read_bus_challenger,
        &proof.public_io_digest,
        &proof.shape_digest,
        options,
        commitment_context.clone(),
    );
    let read_bus_alpha = read_bus_challenger.sample_algebra_element();
    let read_bus_beta = read_bus_challenger.sample_algebra_element();
    verify_read_bus_proof::<F, EF>(
        circuit,
        &expected_metadata,
        &proof.read_bus_proof,
        read_bus_alpha,
        read_bus_beta,
    )?;
    let poseidon2_shift_terminal_claims = verify_poseidon2_shift_bus::<
        F,
        EF,
        MT,
        Challenger,
        Dft,
        MakePcs,
        MakeChallenger,
        DIGEST_ELEMS,
    >(
        &expected_metadata,
        &proof.poseidon2_shift_bus_proof,
        &proof.public_io_digest,
        &proof.shape_digest,
        options,
        commitment_context.clone(),
        &make_pcs,
        &make_challenger,
    )?;
    let mut terminal_claims_by_table = vec![Vec::new(); expected_metadata.len()];
    let mut terminal_claims_by_batch = vec![Vec::new(); column_batch_layouts.len()];
    for claim in poseidon2_shift_terminal_claims {
        let point = Point::new(claim.point.clone());
        terminal_claims_by_table[claim.table_index].push((point, claim.value));
        if proof.opening_mode == WhirNativeOpeningMode::ColumnBatched {
            let (batch_index, column_offset) = column_batch_index_and_offset(
                &column_batch_layouts,
                claim.table_index,
                claim.column,
            )
            .ok_or_else(|| {
                WhirNativeCircuitError::ConstraintViolation(format!(
                    "missing column batch for Poseidon2 shift claim table {}, column {}",
                    claim.table_index, claim.column
                ))
            })?;
            terminal_claims_by_batch[batch_index].push((column_offset, claim));
        }
    }
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
                commitment_context.clone(),
                &proof.public_io_digest,
                &proof.shape_digest,
                options,
                constraint_proof,
                &make_challenger,
            )?
        {
            terminal_claims_by_table[claim_table_index].push((point, value));
        }
        if proof.opening_mode == WhirNativeOpeningMode::ColumnBatched {
            for claim in local_proof_terminal_column_claims::<EF>(
                constraint_proof.local_proof.as_ref(),
                &expected_metadata,
            )? {
                let (batch_index, column_offset) = column_batch_index_and_offset(
                    &column_batch_layouts,
                    claim.table_index,
                    claim.column,
                )
                .ok_or_else(|| {
                    WhirNativeCircuitError::ConstraintViolation(format!(
                        "missing column batch for terminal claim table {}, column {}",
                        claim.table_index, claim.column
                    ))
                })?;
                terminal_claims_by_batch[batch_index].push((column_offset, claim));
            }
        }
    }

    match proof.opening_mode {
        WhirNativeOpeningMode::PerTable => {
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
        }
        WhirNativeOpeningMode::ColumnBatched => {
            for (batch_index, layout) in column_batch_layouts.iter().enumerate() {
                let pcs = make_pcs(layout.num_variables);
                verify_column_batch_opening(
                    &pcs,
                    layout,
                    &proof.column_batch_commitments[batch_index],
                    &proof.column_batch_opening_proofs[batch_index],
                    batch_index,
                    &terminal_claims_by_batch[batch_index],
                    &expected_metadata,
                    &proof.public_io_digest,
                    &proof.shape_digest,
                    options,
                    &make_challenger,
                )?;
            }
        }
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
    let witness_values: Vec<EF> = (0..traces.witness_trace.num_rows())
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
    let expected_alu_rows = expected_alu_rows_from_circuit(circuit);
    let expected_alu_count = expected_alu_rows.len();
    let mut alu_rows = traces
        .alu_trace
        .op_kind
        .iter()
        .zip(&traces.alu_trace.indices)
        .zip(&traces.alu_trace.values)
        .enumerate()
        .map(|(row_index, ((&kind, indices), &values))| {
            let acc_value = expected_alu_rows
                .get(row_index)
                .and_then(|row| {
                    (row.kind == AluOpKind::HornerAcc)
                        .then_some(row.acc_index)
                        .flatten()
                })
                .and_then(|wid| witness_values.get(wid as usize).copied())
                .unwrap_or(EF::ZERO);
            WhirNativeAluRow {
                kind: alu_kind_to_tag(kind),
                indices: [indices[0].0, indices[1].0, indices[2].0, indices[3].0],
                values,
                acc_value,
            }
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
                let expected_acc = if *kind == AluOpKind::HornerAcc {
                    intermediate_out
                        .map(|wid| witness(payload, wid))
                        .transpose()?
                        .unwrap_or(EF::ZERO)
                } else {
                    EF::ZERO
                };
                if row.acc_value != expected_acc {
                    return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                        "ALU row {alu_idx} accumulator read mismatch"
                    )));
                }
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
            .map(|row| {
                vec![
                    ef_from_u64::<F, EF>(row.witness_id as u64),
                    row.value,
                    row.value,
                ]
            })
            .collect(),
        Some(KNOWN_ROWS_WIDTH),
        options,
    ));

    tables.push(pack_rows(
        WhirNativeTableKind::Public,
        String::new(),
        payload
            .public_rows
            .iter()
            .map(|row| {
                vec![
                    ef_from_u64::<F, EF>(row.witness_id as u64),
                    row.value,
                    row.value,
                ]
            })
            .collect(),
        Some(KNOWN_ROWS_WIDTH),
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
                values.push(row.acc_value);
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
                    let output_read =
                        EF::from_basis_coefficients_slice(&row.values).unwrap_or(EF::ZERO);
                    values.push(output_read);
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
        .enumerate()
        .map(|(row_index, row)| {
            let read_values = poseidon2_read_values_for_air_row::<F, EF>(
                table.config,
                table.rows.get(row_index),
                row,
            );
            let mut row_values = row
                .iter()
                .copied()
                .map(babybear_to_ef::<F, EF>)
                .collect::<Vec<_>>();
            row_values.extend(read_values);
            row_values
        })
        .collect::<Vec<_>>();
    let rows = append_cyclic_shifted_columns(rows, P2_BB_D4_WIDTH16_AIR_WIDTH)?;

    Ok(pack_rows(
        WhirNativeTableKind::Poseidon2,
        table.op_type.clone(),
        rows,
        Some(P2_BB_D4_WIDTH16_TABLE_WIDTH),
        options,
    ))
}

fn poseidon2_read_values_for_air_row<F, EF>(
    config: Poseidon2Config,
    row: Option<&WhirNativePoseidon2Row<F>>,
    air_row: &[BabyBear],
) -> Vec<EF>
where
    F: Field + PrimeField64,
    EF: ExtensionField<F>,
{
    let Some(row) = row else {
        return EF::zero_vec(P2_BB_D4_WIDTH16_WITNESS_PORTS);
    };
    let mut values = Vec::with_capacity(P2_BB_D4_WIDTH16_WITNESS_PORTS);
    for limb in 0..P2_BB_D4_WIDTH16_WIDTH_EXT {
        let value = if row.in_ctl[limb] {
            EF::from_basis_coefficients_fn(|coeff| {
                babybear_to_f::<F>(air_row[limb * config.d() + coeff])
            })
        } else {
            EF::ZERO
        };
        values.push(value);
    }
    for limb in 0..P2_BB_D4_WIDTH16_RATE_EXT {
        let start = P2_BB_D4_WIDTH16_OUTPUT_OFFSET + limb * config.d();
        let value = if row.out_ctl[limb] {
            EF::from_basis_coefficients_fn(|coeff| babybear_to_f::<F>(air_row[start + coeff]))
        } else {
            EF::ZERO
        };
        values.push(value);
    }
    values.push(if row.mmcs_ctl_enabled {
        EF::from(row.mmcs_index_sum)
    } else {
        EF::ZERO
    });
    values.push(if row.merkle_path {
        ef_from_bool::<F, EF>(row.mmcs_bit)
    } else {
        EF::ZERO
    });
    values
}

fn append_cyclic_shifted_columns<EF>(
    rows: Vec<Vec<EF>>,
    air_width: usize,
) -> Result<Vec<Vec<EF>>, WhirNativeCircuitError>
where
    EF: Field,
{
    if rows.is_empty() {
        return Ok(rows);
    }
    if rows
        .iter()
        .any(|row| row.len() != air_width + P2_BB_D4_WIDTH16_WITNESS_PORTS)
    {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "cannot append shifted columns to ragged Poseidon2 rows".to_string(),
        ));
    }
    Ok((0..rows.len())
        .map(|row_index| {
            let mut row = rows[row_index][..air_width].to_vec();
            row.extend(
                rows[(row_index + 1) % rows.len()][..air_width]
                    .iter()
                    .copied(),
            );
            row.extend(
                rows[row_index][air_width..air_width + P2_BB_D4_WIDTH16_WITNESS_PORTS]
                    .iter()
                    .copied(),
            );
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

/// Reconstruct the WHIR-native table metadata implied by a circuit and options.
pub fn whir_native_expected_table_metadata<F, EF>(
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
            KNOWN_ROWS_WIDTH,
            options,
        ),
        metadata_for_shape(
            WhirNativeTableKind::Public,
            String::new(),
            public_rows.len(),
            KNOWN_ROWS_WIDTH,
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
    validate_row_point(table.metadata.padded_height, row_point)?;
    let column_values = table_column_values(table, column)?;
    Ok(Poly::eval_ext_slice::<F>(&column_values, row_point))
}

fn table_column_values<EF>(
    table: &WhirNativeTableData<EF>,
    column: usize,
) -> Result<Vec<EF>, WhirNativeCircuitError>
where
    EF: Field,
{
    if column >= table.metadata.padded_width {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "column {column} out of range for padded width {}",
            table.metadata.padded_width
        )));
    }
    Ok((0..table.metadata.padded_height)
        .map(|row| table.values[row * table.metadata.padded_width + column])
        .collect())
}

fn build_column_batch_layouts(
    metadata: &[WhirNativeTableMetadata],
    options: WhirNativeCircuitOptions,
) -> Vec<WhirNativeColumnBatchLayout> {
    let mut layouts = Vec::<WhirNativeColumnBatchLayout>::new();
    for (table_index, metadata) in metadata.iter().enumerate() {
        let num_variables =
            whir_native_table_row_variables(metadata).max(options.min_num_variables);
        let layout_index = layouts
            .iter()
            .position(|layout| layout.num_variables == num_variables)
            .unwrap_or_else(|| {
                layouts.push(WhirNativeColumnBatchLayout {
                    num_variables,
                    columns: Vec::new(),
                });
                layouts.len() - 1
            });
        for column in 0..metadata.padded_width {
            layouts[layout_index].columns.push(WhirNativeColumnRef {
                table_index,
                column,
            });
        }
    }
    layouts.sort_by_key(|layout| layout.num_variables);
    layouts
}

fn column_batch_values<F, EF>(
    table: &WhirNativeTableData<EF>,
    column: usize,
    num_variables: usize,
) -> Result<Vec<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let row_vars = whir_native_table_row_variables(&table.metadata);
    if num_variables < row_vars {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "column batch arity {num_variables} is smaller than row arity {row_vars}"
        )));
    }
    let mut values = table_column_values(table, column)?;
    values.resize(1 << num_variables, EF::ZERO);
    Ok(values)
}

fn extend_row_point_for_column_batch<EF>(
    row_point: &Point<EF>,
    row_variables: usize,
    batch_variables: usize,
) -> Result<Point<EF>, WhirNativeCircuitError>
where
    EF: Field,
{
    if row_point.num_variables() != row_variables {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "row point arity mismatch: expected {row_variables}, got {}",
            row_point.num_variables()
        )));
    }
    if batch_variables < row_variables {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "batch arity {batch_variables} is smaller than row arity {row_variables}"
        )));
    }
    let mut coords = EF::zero_vec(batch_variables - row_variables);
    coords.extend_from_slice(row_point.as_slice());
    Ok(Point::new(coords))
}

fn row_point_from_terminal_column_claim<F, EF>(
    metadata: &WhirNativeTableMetadata,
    claim: &WhirNativeTerminalColumnClaim<EF>,
    batch_variables: usize,
) -> Result<Point<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if claim.column >= metadata.padded_width {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "terminal column {} out of range for padded width {}",
            claim.column, metadata.padded_width
        )));
    }
    if claim.point.len() != metadata.num_variables {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "terminal point arity mismatch: expected {}, got {}",
            metadata.num_variables,
            claim.point.len()
        )));
    }
    let row_variables = whir_native_table_row_variables(metadata);
    let column_variables = whir_native_table_column_variables(metadata);
    let expected_column = Point::<EF>::hypercube(claim.column, column_variables);
    if claim.point[row_variables..] != *expected_column.as_slice() {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "terminal column selector mismatch".to_string(),
        ));
    }
    let row_point = Point::new(claim.point[..row_variables].to_vec());
    extend_row_point_for_column_batch(&row_point, row_variables, batch_variables)
}

fn column_batch_index_and_offset(
    layouts: &[WhirNativeColumnBatchLayout],
    table_index: usize,
    column: usize,
) -> Option<(usize, usize)> {
    layouts
        .iter()
        .enumerate()
        .find_map(|(batch_index, layout)| {
            layout
                .columns
                .iter()
                .position(|column_ref| {
                    column_ref.table_index == table_index && column_ref.column == column
                })
                .map(|column_offset| (batch_index, column_offset))
        })
}

fn active_selector_values<EF>(
    active_rows: usize,
    padded_height: usize,
) -> Result<Vec<EF>, WhirNativeCircuitError>
where
    EF: Field,
{
    if active_rows > padded_height {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "active row count {active_rows} exceeds padded height {padded_height}"
        )));
    }
    let mut evals = EF::zero_vec(padded_height);
    for value in evals.iter_mut().take(active_rows) {
        *value = EF::ONE;
    }
    Ok(evals)
}

fn row_index_values<F, EF>(padded_height: usize) -> Vec<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    (0..padded_height)
        .map(|row| ef_from_u64::<F, EF>(row as u64))
        .collect()
}

fn static_u32_column_values<F, EF>(
    padded_height: usize,
    values: &[u32],
) -> Result<Vec<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if values.len() > padded_height {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "too many static values: {} for padded height {padded_height}",
            values.len()
        )));
    }
    let mut evals = EF::zero_vec(padded_height);
    for (row, &value) in values.iter().enumerate() {
        evals[row] = ef_from_u64::<F, EF>(value as u64);
    }
    Ok(evals)
}

fn witness_address_bit_columns<F, EF>(
    witness_metadata: &WhirNativeTableMetadata,
    source_padded_height: usize,
    witness_ids_by_source_row: &[u32],
) -> Result<Vec<Vec<EF>>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if witness_metadata.kind != WhirNativeTableKind::Witness {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "witness address point requires witness table metadata".to_string(),
        ));
    }
    if witness_ids_by_source_row.len() > source_padded_height {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "too many witness ids: {} for source height {source_padded_height}",
            witness_ids_by_source_row.len()
        )));
    }
    let witness_row_vars = whir_native_table_row_variables(witness_metadata);
    let mut columns = Vec::with_capacity(witness_row_vars);
    for bit_index in 0..witness_row_vars {
        let shift = witness_row_vars - 1 - bit_index;
        let mut bit_evals = EF::zero_vec(source_padded_height);
        for (row, &wid) in witness_ids_by_source_row.iter().enumerate() {
            bit_evals[row] = ef_from_bool::<F, EF>((wid >> shift) & 1 == 1);
        }
        columns.push(bit_evals);
    }
    Ok(columns)
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

    let bit_columns = witness_address_bit_columns::<F, EF>(
        witness_metadata,
        source_padded_height,
        witness_ids_by_source_row,
    )?;
    let coords = bit_columns
        .iter()
        .map(|bit_evals| Poly::eval_ext_slice::<F>(bit_evals, source_row_point))
        .collect();
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

#[derive(Clone, Debug)]
struct FoldedColumn<EF> {
    values: Vec<EF>,
}

impl<EF> FoldedColumn<EF>
where
    EF: Field,
{
    fn new(values: Vec<EF>) -> Self {
        debug_assert!(values.len().is_power_of_two());
        Self { values }
    }

    fn line_evals<F>(&self, suffix: usize, degree: usize) -> Vec<EF>
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        if self.values.len() == 1 {
            return vec![self.values[0]; degree + 1];
        }
        let half = self.values.len() / 2;
        debug_assert!(suffix < half);
        line_evals_zero_to_degree::<F, EF>(self.values[suffix], self.values[half + suffix], degree)
    }

    fn line_value<F>(&self, suffix: usize, point: usize) -> EF
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        if self.values.len() == 1 {
            return self.values[0];
        }
        let half = self.values.len() / 2;
        debug_assert!(suffix < half);
        let low = self.values[suffix];
        let high = self.values[half + suffix];
        low + (high - low) * EF::from(F::from_u64(point as u64))
    }

    fn fold(&mut self, challenge: EF) {
        if self.values.len() == 1 {
            return;
        }
        let half = self.values.len() / 2;
        let (lo, hi) = self.values.split_at_mut(half);
        for (lo, &hi) in lo.iter_mut().zip(hi.iter()) {
            *lo += (hi - *lo) * challenge;
        }
        self.values.truncate(half);
    }
}

fn folded_column_line_values<F, EF>(
    columns: &[FoldedColumn<EF>],
    suffix: usize,
    point: usize,
) -> Vec<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    columns
        .iter()
        .map(|column| column.line_value::<F>(suffix, point))
        .collect()
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct FoldedWitnessPort<EF> {
    address_bits: Vec<FoldedColumn<EF>>,
}

#[allow(dead_code)]
impl<EF> FoldedWitnessPort<EF>
where
    EF: Field,
{
    fn fold(&mut self, challenge: EF) {
        for bit in &mut self.address_bits {
            bit.fold(challenge);
        }
    }

    fn line_endpoints(&self, suffix: usize) -> (Vec<EF>, Vec<EF>) {
        let mut low = Vec::with_capacity(self.address_bits.len());
        let mut high = Vec::with_capacity(self.address_bits.len());
        for bit in &self.address_bits {
            if bit.values.len() == 1 {
                low.push(bit.values[0]);
                high.push(bit.values[0]);
            } else {
                let half = bit.values.len() / 2;
                debug_assert!(suffix < half);
                low.push(bit.values[suffix]);
                high.push(bit.values[half + suffix]);
            }
        }
        (low, high)
    }
}

#[allow(dead_code)]
fn prove_row_folded_sumcheck_by_suffix<F, EF, Challenger, State, Eval, Fold>(
    num_variables: usize,
    degree: usize,
    initial_claim: EF,
    challenger: &mut Challenger,
    state: &mut State,
    mut suffix_evals: Eval,
    mut fold_state: Fold,
) -> Result<(WhirNativeSumcheckProof<EF>, Point<EF>, EF), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    Eval: FnMut(&State, usize, usize, usize) -> Result<Vec<EF>, WhirNativeCircuitError>,
    Fold: FnMut(&mut State, EF),
{
    let mut proof = WhirNativeSumcheckProof::new(degree);
    let mut claim = initial_claim;
    let mut point = Vec::with_capacity(num_variables);

    for round in 0..num_variables {
        let suffix_vars = num_variables - round - 1;
        let suffix_count = 1usize << suffix_vars;
        let mut evals = EF::zero_vec(degree + 1);

        for suffix in 0..suffix_count {
            let values = suffix_evals(state, round, suffix, degree)?;
            if values.len() != degree + 1 {
                return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                    "sumcheck round {round} suffix evaluator returned {} points, expected {}",
                    values.len(),
                    degree + 1
                )));
            }
            for (acc, value) in evals.iter_mut().zip(values) {
                *acc += value;
            }
        }

        if evals[0] + evals[1] != claim {
            return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                "sumcheck round {round} inconsistent with claimed sum"
            )));
        }

        challenger.observe_algebra_slice(&evals);
        let challenge = challenger.sample_algebra_element();
        claim = crate::whir_native_sumcheck::lagrange_eval_on_zero_to_degree::<F, EF>(
            &evals, challenge,
        );
        point.push(challenge);
        proof.round_evals.push(evals);
        fold_state(state, challenge);
    }

    Ok((proof, Point::new(point), claim))
}

#[allow(dead_code)]
fn prove_row_folded_sumcheck_by_suffix_parallel<F, EF, Challenger, State, Eval, Fold>(
    num_variables: usize,
    degree: usize,
    initial_claim: EF,
    challenger: &mut Challenger,
    state: &mut State,
    suffix_evals: Eval,
    mut fold_state: Fold,
) -> Result<(WhirNativeSumcheckProof<EF>, Point<EF>, EF), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F> + Send + Sync,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    State: Sync,
    Eval: Fn(&State, usize, usize, usize) -> Result<Vec<EF>, WhirNativeCircuitError> + Sync,
    Fold: FnMut(&mut State, EF),
{
    let mut proof = WhirNativeSumcheckProof::new(degree);
    let mut claim = initial_claim;
    let mut point = Vec::with_capacity(num_variables);

    for round in 0..num_variables {
        let suffix_vars = num_variables - round - 1;
        let suffix_count = 1usize << suffix_vars;
        let suffix_results = (0..suffix_count)
            .into_par_iter()
            .map(|suffix| suffix_evals(state, round, suffix, degree))
            .collect::<Vec<_>>();
        let mut evals = EF::zero_vec(degree + 1);

        for values in suffix_results {
            let values = values?;
            if values.len() != degree + 1 {
                return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                    "sumcheck round {round} suffix evaluator returned {} points, expected {}",
                    values.len(),
                    degree + 1
                )));
            }
            for (acc, value) in evals.iter_mut().zip(values) {
                *acc += value;
            }
        }

        if evals[0] + evals[1] != claim {
            return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                "sumcheck round {round} inconsistent with claimed sum"
            )));
        }

        challenger.observe_algebra_slice(&evals);
        let challenge = challenger.sample_algebra_element();
        claim = crate::whir_native_sumcheck::lagrange_eval_on_zero_to_degree::<F, EF>(
            &evals, challenge,
        );
        point.push(challenge);
        proof.round_evals.push(evals);
        fold_state(state, challenge);
    }

    Ok((proof, Point::new(point), claim))
}

fn prove_row_folded_sumcheck_by_suffix_into_parallel<F, EF, Challenger, State, Eval, Fold>(
    num_variables: usize,
    degree: usize,
    initial_claim: EF,
    challenger: &mut Challenger,
    state: &mut State,
    suffix_evals: Eval,
    mut fold_state: Fold,
) -> Result<(WhirNativeSumcheckProof<EF>, Point<EF>, EF), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F> + Send + Sync,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    State: Sync,
    Eval: Fn(&State, usize, usize, usize, &mut [EF]) -> Result<(), WhirNativeCircuitError> + Sync,
    Fold: FnMut(&mut State, EF),
{
    let mut proof = WhirNativeSumcheckProof::new(degree);
    let mut claim = initial_claim;
    let mut point = Vec::with_capacity(num_variables);

    for round in 0..num_variables {
        let suffix_vars = num_variables - round - 1;
        let suffix_count = 1usize << suffix_vars;
        let suffix_results = (0..suffix_count)
            .into_par_iter()
            .map(|suffix| {
                let mut scratch = EF::zero_vec(degree + 1);
                suffix_evals(state, round, suffix, degree, &mut scratch)?;
                Ok(scratch)
            })
            .collect::<Vec<_>>();
        let mut evals = EF::zero_vec(degree + 1);

        for values in suffix_results {
            let values = values?;
            if values.len() != degree + 1 {
                return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                    "sumcheck round {round} suffix evaluator returned {} points, expected {}",
                    values.len(),
                    degree + 1
                )));
            }
            for (acc, value) in evals.iter_mut().zip(values) {
                *acc += value;
            }
        }

        if evals[0] + evals[1] != claim {
            return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                "sumcheck round {round} inconsistent with claimed sum"
            )));
        }

        challenger.observe_algebra_slice(&evals);
        let challenge = challenger.sample_algebra_element();
        claim = crate::whir_native_sumcheck::lagrange_eval_on_zero_to_degree::<F, EF>(
            &evals, challenge,
        );
        point.push(challenge);
        proof.round_evals.push(evals);
        fold_state(state, challenge);
    }

    Ok((proof, Point::new(point), claim))
}

fn line_evals_zero_to_degree<F, EF>(low: EF, high: EF, degree: usize) -> Vec<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let delta = high - low;
    (0..=degree)
        .map(|point| low + delta * EF::from(F::from_u64(point as u64)))
        .collect()
}

#[allow(dead_code)]
fn eval_mle_on_line_zero_to_degree<F, EF>(
    evals: &[EF],
    low_point: &[EF],
    high_point: &[EF],
    degree: usize,
) -> Vec<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    const MAX_BATCHED_LINE_ELEMENTS: usize = 1 << 20;

    debug_assert_eq!(low_point.len(), high_point.len());
    debug_assert_eq!(evals.len(), 1 << low_point.len());

    let lanes = degree + 1;
    if evals.len().saturating_mul(lanes) > MAX_BATCHED_LINE_ELEMENTS {
        return (0..=degree)
            .map(|point| {
                let t = EF::from(F::from_u64(point as u64));
                let line_point = low_point
                    .iter()
                    .zip(high_point)
                    .map(|(&low, &high)| low + (high - low) * t)
                    .collect();
                Poly::eval_ext_slice::<F>(evals, &Point::new(line_point))
            })
            .collect();
    }

    let mut work = Vec::with_capacity(evals.len() * lanes);
    for &value in evals {
        for _ in 0..lanes {
            work.push(value);
        }
    }
    let line_coords = low_point
        .iter()
        .zip(high_point)
        .map(|(&low, &high)| line_evals_zero_to_degree::<F, EF>(low, high, degree))
        .collect::<Vec<_>>();

    let mut current_len = evals.len();
    for coords in line_coords {
        if current_len == 1 {
            break;
        }
        let half = current_len / 2;
        for row in 0..half {
            let lo_base = row * lanes;
            let hi_base = (row + half) * lanes;
            for lane in 0..lanes {
                let low = work[lo_base + lane];
                let high = work[hi_base + lane];
                work[lo_base + lane] = low + (high - low) * coords[lane];
            }
        }
        current_len = half;
    }
    work.truncate(lanes);
    work
}

fn fold_columns<EF>(columns: &mut [FoldedColumn<EF>], challenge: EF)
where
    EF: Field,
{
    for column in columns {
        column.fold(challenge);
    }
}

const WITNESS_TABLE_INDEX: usize = 0;
const WITNESS_LOCAL_DEGREE: usize = 3;
const ALU_LOCAL_DEGREE: usize = 4;
const WITNESS_WIDTH: usize = 2;
const WITNESS_COLUMNS: [usize; WITNESS_WIDTH] = [0, 1];
const KNOWN_ROWS_WIDTH: usize = 3;
const KNOWN_ROWS_EXPECTED_WIDTH: usize = 2;
const KNOWN_ROW_WITNESS_ID_COL: usize = 0;
const KNOWN_ROW_VALUE_COL: usize = 1;
const KNOWN_ROW_READ_VALUE_COL: usize = 2;
const MISSING_WITNESS_ID: u32 = u32::MAX;
const ALU_WIDTH: usize = 10;
const ALU_SHAPE_WIDTH: usize = 5;
const ALU_COLUMNS: [usize; ALU_WIDTH] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
#[allow(dead_code)]
const ALU_WITNESS_PORTS: [usize; 5] = [0, 1, 2, 3, 4];
#[allow(dead_code)]
const ALU_ACC_WITNESS_PORT: usize = 4;
const ALU_READ_A_COL: usize = 5;
const ALU_READ_B_COL: usize = 6;
const ALU_READ_C_COL: usize = 7;
const ALU_READ_OUT_COL: usize = 8;
const ALU_READ_ACC_COL: usize = 9;
const P2_BB_D4_WIDTH16_D: usize = 4;
const P2_BB_D4_WIDTH16_WIDTH: usize = 16;
const P2_BB_D4_WIDTH16_WIDTH_EXT: usize = 4;
const P2_BB_D4_WIDTH16_RATE_EXT: usize = 2;
const P2_BB_D4_WIDTH16_PERM_WIDTH: usize = 298;
const P2_BB_D4_WIDTH16_OUTPUT_OFFSET: usize = 282;
const P2_BB_D4_WIDTH16_MMCS_INDEX_SUM_COL: usize = P2_BB_D4_WIDTH16_PERM_WIDTH + 1;
const P2_BB_D4_WIDTH16_AIR_WIDTH: usize = P2_BB_D4_WIDTH16_PERM_WIDTH + 2;
const P2_BB_D4_WIDTH16_SHIFTED_OFFSET: usize = P2_BB_D4_WIDTH16_AIR_WIDTH;
const P2_BB_D4_WIDTH16_READ_OFFSET: usize = P2_BB_D4_WIDTH16_AIR_WIDTH * 2;
const P2_BB_D4_WIDTH16_PREPROCESSED_WIDTH: usize = 24;
const P2_BB_D4_WIDTH16_WITNESS_PORTS: usize =
    P2_BB_D4_WIDTH16_WIDTH_EXT + P2_BB_D4_WIDTH16_RATE_EXT + 2;
const P2_BB_D4_WIDTH16_TABLE_WIDTH: usize =
    P2_BB_D4_WIDTH16_READ_OFFSET + P2_BB_D4_WIDTH16_WITNESS_PORTS;
const P2_SHIFT_AUX_WIDTH: usize = 2;
const P2_SHIFT_SENDER_INV_COL: usize = 0;
const P2_SHIFT_RECEIVER_INV_COL: usize = 1;
const P2_SHIFT_LOCAL_DEGREE: usize = 3;

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
    Ok(4 + d + d)
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
    commitment_context: WhirNativeCommitmentContext<'_, MT::Commitment>,
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
        commitment_context,
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
        WhirNativeTableKind::Poseidon2Shift => Err(WhirNativeCircuitError::ConstraintViolation(
            "Poseidon2 shift auxiliary tables are not main local-constraint tables".to_string(),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn verify_table_local_constraints<F, EF, MT, Challenger, MakeChallenger>(
    circuit: &Circuit<EF>,
    public_inputs: &[EF],
    table_index: usize,
    metadata: &[WhirNativeTableMetadata],
    commitment_context: WhirNativeCommitmentContext<'_, MT::Commitment>,
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
        commitment_context,
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
        WhirNativeTableKind::Poseidon2Shift => Err(WhirNativeCircuitError::ConstraintViolation(
            "Poseidon2 shift auxiliary tables are not main local-constraint tables".to_string(),
        )),
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

fn local_proof_terminal_column_claims<EF>(
    proof: Option<&WhirNativeLocalConstraintProof<EF>>,
    metadata: &[WhirNativeTableMetadata],
) -> Result<Vec<WhirNativeTerminalColumnClaim<EF>>, WhirNativeCircuitError>
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
            if claim.point.len() != claim_metadata.num_variables {
                return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                    "terminal opening {opening_index} arity mismatch"
                )));
            }
            Ok(claim.clone())
        })
        .collect()
}

#[derive(Clone, Debug)]
struct WitnessFoldedSumcheckState<EF> {
    eq: FoldedColumn<EF>,
    active: FoldedColumn<EF>,
    row_index: FoldedColumn<EF>,
    table_columns: Vec<FoldedColumn<EF>>,
    alpha: EF,
}

impl<EF> WitnessFoldedSumcheckState<EF>
where
    EF: Field,
{
    fn fold(&mut self, challenge: EF) {
        self.eq.fold(challenge);
        self.active.fold(challenge);
        self.row_index.fold(challenge);
        fold_columns(&mut self.table_columns, challenge);
    }
}

fn build_witness_folded_sumcheck_state<F, EF>(
    table: &WhirNativeTableData<EF>,
    zerocheck_point: &Point<EF>,
    alpha: EF,
) -> Result<WitnessFoldedSumcheckState<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let eq = FoldedColumn::new(
        Poly::<EF>::new_from_point(zerocheck_point.as_slice(), EF::ONE).into_evals(),
    );
    let active = FoldedColumn::new(active_selector_values::<EF>(
        table.metadata.active_rows,
        table.metadata.padded_height,
    )?);
    let row_index = FoldedColumn::new(row_index_values::<F, EF>(table.metadata.padded_height));
    let table_columns = WITNESS_COLUMNS
        .iter()
        .map(|&column| table_column_values(table, column).map(FoldedColumn::new))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(WitnessFoldedSumcheckState {
        eq,
        active,
        row_index,
        table_columns,
        alpha,
    })
}

fn eval_witness_constraint_from_folded_values<EF>(
    active: EF,
    row_index: EF,
    values: &[EF],
    alpha: EF,
) -> Result<EF, WhirNativeCircuitError>
where
    EF: Field,
{
    if values.len() != WITNESS_WIDTH {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "witness folded width mismatch: expected {WITNESS_WIDTH}, got {}",
            values.len()
        )));
    }
    let inactive = EF::ONE - active;
    Ok(batch_constraints(
        vec![
            active * (values[0] - row_index),
            inactive * values[0],
            inactive * values[1],
        ],
        alpha,
    ))
}

fn witness_folded_suffix_evals_into<F, EF>(
    state: &WitnessFoldedSumcheckState<EF>,
    suffix: usize,
    degree: usize,
    out: &mut [EF],
) -> Result<(), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if out.len() != degree + 1 {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "witness suffix output length mismatch: expected {}, got {}",
            degree + 1,
            out.len()
        )));
    }
    for (point, out) in out.iter_mut().enumerate() {
        let eq = state.eq.line_value::<F>(suffix, point);
        let active = state.active.line_value::<F>(suffix, point);
        let row_index = state.row_index.line_value::<F>(suffix, point);
        let values = folded_column_line_values::<F, EF>(&state.table_columns, suffix, point);
        let constraint =
            eval_witness_constraint_from_folded_values(active, row_index, &values, state.alpha)?;
        *out = eq * constraint;
    }
    Ok(())
}

#[allow(dead_code)]
fn prove_witness_local_constraints<F, EF, Challenger>(
    table_index: usize,
    table: &WhirNativeTableData<EF>,
    challenger: &mut Challenger,
) -> Result<WhirNativeLocalConstraintProof<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F> + Send + Sync,
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

    let mut folded_state = build_witness_folded_sumcheck_state::<F, EF>(
        table,
        &zerocheck_point,
        constraint_challenge,
    )?;
    let (sumcheck, terminal_row_point, terminal_claim) =
        prove_row_folded_sumcheck_by_suffix_into_parallel::<F, EF, Challenger, _, _, _>(
            whir_native_table_row_variables(&table.metadata),
            WITNESS_LOCAL_DEGREE,
            EF::ZERO,
            challenger,
            &mut folded_state,
            |state, _round, suffix, degree, out| {
                witness_folded_suffix_evals_into::<F, EF>(state, suffix, degree, out)
            },
            |state, challenge| state.fold(challenge),
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
    EF: ExtensionField<F> + Send + Sync,
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

#[derive(Clone, Debug)]
struct KnownRowsFoldedSumcheckState<EF> {
    eq: FoldedColumn<EF>,
    active: FoldedColumn<EF>,
    table_columns: Vec<FoldedColumn<EF>>,
    expected_columns: Vec<FoldedColumn<EF>>,
    witness_id: FoldedColumn<EF>,
    alpha: EF,
}

impl<EF> KnownRowsFoldedSumcheckState<EF>
where
    EF: Field,
{
    fn fold(&mut self, challenge: EF) {
        self.eq.fold(challenge);
        self.active.fold(challenge);
        fold_columns(&mut self.table_columns, challenge);
        fold_columns(&mut self.expected_columns, challenge);
        self.witness_id.fold(challenge);
    }
}

fn expected_known_row_column_values<EF>(
    padded_height: usize,
    expected_rows: &[Vec<EF>],
    column: usize,
) -> Result<Vec<EF>, WhirNativeCircuitError>
where
    EF: Field,
{
    if expected_rows.len() > padded_height {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "too many expected rows: {} for padded height {padded_height}",
            expected_rows.len()
        )));
    }
    let mut evals = EF::zero_vec(padded_height);
    for (row_index, row) in expected_rows.iter().enumerate() {
        let Some(&value) = row.get(column) else {
            return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                "expected row {row_index} missing column {column}"
            )));
        };
        evals[row_index] = value;
    }
    Ok(evals)
}

fn build_known_rows_folded_sumcheck_state<F, EF>(
    table: &WhirNativeTableData<EF>,
    expected_rows: &[Vec<EF>],
    expected_witness_ids: &[u32],
    zerocheck_point: &Point<EF>,
    alpha: EF,
) -> Result<KnownRowsFoldedSumcheckState<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let eq = FoldedColumn::new(
        Poly::<EF>::new_from_point(zerocheck_point.as_slice(), EF::ONE).into_evals(),
    );
    let active = FoldedColumn::new(active_selector_values::<EF>(
        table.metadata.active_rows,
        table.metadata.padded_height,
    )?);
    let table_columns = logical_columns(&table.metadata)
        .into_iter()
        .map(|column| table_column_values(table, column).map(FoldedColumn::new))
        .collect::<Result<Vec<_>, _>>()?;
    let expected_columns = (0..KNOWN_ROWS_EXPECTED_WIDTH)
        .map(|column| {
            expected_known_row_column_values(table.metadata.padded_height, expected_rows, column)
                .map(FoldedColumn::new)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let witness_id = FoldedColumn::new(static_u32_column_values::<F, EF>(
        table.metadata.padded_height,
        expected_witness_ids,
    )?);

    Ok(KnownRowsFoldedSumcheckState {
        eq,
        active,
        table_columns,
        expected_columns,
        witness_id,
        alpha,
    })
}

fn eval_known_rows_constraint_from_folded_values<EF>(
    active: EF,
    values: &[EF],
    expected_values: &[EF],
    witness_id: EF,
    alpha: EF,
) -> Result<EF, WhirNativeCircuitError>
where
    EF: Field,
{
    if values.len() != KNOWN_ROWS_WIDTH || expected_values.len() != KNOWN_ROWS_EXPECTED_WIDTH {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "known-row folded width mismatch: {} values, {} expected",
            values.len(),
            expected_values.len()
        )));
    }
    let source_value = values[KNOWN_ROW_VALUE_COL];
    let read_value = values[KNOWN_ROW_READ_VALUE_COL];
    let inactive = EF::ONE - active;
    let mut constraints = Vec::with_capacity(values.len() + expected_values.len() + 2);
    for &value in values {
        constraints.push(inactive * value);
    }
    for (&value, &expected) in values
        .iter()
        .take(KNOWN_ROWS_EXPECTED_WIDTH)
        .zip(expected_values)
    {
        constraints.push(active * (value - expected));
    }
    constraints.push(active * (values[KNOWN_ROW_WITNESS_ID_COL] - witness_id));
    constraints.push(active * (source_value - read_value));
    Ok(batch_constraints(constraints, alpha))
}

fn known_rows_folded_suffix_evals_into<F, EF>(
    state: &KnownRowsFoldedSumcheckState<EF>,
    suffix: usize,
    degree: usize,
    diagnostics: &WhirNativeLocalProofDiagnostics,
    out: &mut [EF],
) -> Result<(), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if out.len() != degree + 1 {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "known-row suffix output length mismatch: expected {}, got {}",
            degree + 1,
            out.len()
        )));
    }
    diagnostics.time(
        WhirNativeLocalDiagMetric::ConstraintBatch,
        degree + 1,
        || {
            for (point, out) in out.iter_mut().enumerate() {
                let eq = state.eq.line_value::<F>(suffix, point);
                let active = state.active.line_value::<F>(suffix, point);
                let values =
                    folded_column_line_values::<F, EF>(&state.table_columns, suffix, point);
                let expected_values =
                    folded_column_line_values::<F, EF>(&state.expected_columns, suffix, point);
                let witness_id = state.witness_id.line_value::<F>(suffix, point);
                let constraint = eval_known_rows_constraint_from_folded_values(
                    active,
                    &values,
                    &expected_values,
                    witness_id,
                    state.alpha,
                )?;
                *out = eq * constraint;
            }
            Ok(())
        },
    )
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
    EF: ExtensionField<F> + Send + Sync,
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

    let diagnostics = WhirNativeLocalProofDiagnostics::new(
        WhirNativeLocalConstraintKind::KnownRows,
        table_index,
        &table.metadata,
    );
    let mut folded_state = build_known_rows_folded_sumcheck_state::<F, EF>(
        table,
        expected_rows,
        expected_witness_ids,
        &zerocheck_point,
        constraint_challenge,
    )?;
    let (sumcheck, terminal_row_point, terminal_claim) =
        prove_row_folded_sumcheck_by_suffix_into_parallel::<F, EF, Challenger, _, _, _>(
            whir_native_table_row_variables(&table.metadata),
            degree,
            EF::ZERO,
            challenger,
            &mut folded_state,
            |state, _round, suffix, degree, out| {
                known_rows_folded_suffix_evals_into::<F, EF>(
                    state,
                    suffix,
                    degree,
                    &diagnostics,
                    out,
                )
            },
            |state, challenge| state.fold(challenge),
        )?;
    diagnostics.finish();
    let terminal_constraint = eval_known_rows_constraint_from_table::<F, EF>(
        table,
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
    let terminal_openings = terminal_column_claims_for_table::<F, EF>(
        table_index,
        table,
        &terminal_row_point,
        &columns,
    )?;

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
    if proof.terminal_openings.len() != columns.len() {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "known-row terminal opening count mismatch: expected {}, got {}",
            columns.len(),
            proof.terminal_openings.len()
        )));
    }
    let (column_values, opening_claims) =
        extract_terminal_column_values::<F, EF>(proof, metadata, &terminal_row_point, &columns)?;
    let constraint = eval_known_rows_constraint_from_values::<F, EF>(
        metadata,
        expected_rows,
        expected_witness_ids,
        &terminal_row_point,
        &column_values,
        constraint_challenge,
    )?;
    let expected_terminal = eq_eval_ext(&zerocheck_point, &terminal_row_point) * constraint;
    if expected_terminal != terminal_claim {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "known-row local proof terminal claim is inconsistent".to_string(),
        ));
    }

    let all_claims = opening_claims
        .into_iter()
        .map(|(point, value)| (table_index, point, value))
        .collect::<Vec<_>>();
    Ok(all_claims)
}

#[derive(Clone, Debug)]
struct AluFoldedSumcheckState<EF> {
    eq: FoldedColumn<EF>,
    active: FoldedColumn<EF>,
    table_columns: Vec<FoldedColumn<EF>>,
    shape_columns: Vec<FoldedColumn<EF>>,
    selectors: Vec<FoldedColumn<EF>>,
    alpha: EF,
}

impl<EF> AluFoldedSumcheckState<EF>
where
    EF: Field,
{
    fn fold(&mut self, challenge: EF) {
        self.eq.fold(challenge);
        self.active.fold(challenge);
        fold_columns(&mut self.table_columns, challenge);
        fold_columns(&mut self.shape_columns, challenge);
        fold_columns(&mut self.selectors, challenge);
    }
}

fn build_alu_folded_sumcheck_state<F, EF>(
    table: &WhirNativeTableData<EF>,
    expected_rows: &[WhirNativeExpectedAluRow],
    zerocheck_point: &Point<EF>,
    alpha: EF,
) -> Result<AluFoldedSumcheckState<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let eq = FoldedColumn::new(
        Poly::<EF>::new_from_point(zerocheck_point.as_slice(), EF::ONE).into_evals(),
    );
    let active = FoldedColumn::new(active_selector_values::<EF>(
        table.metadata.active_rows,
        table.metadata.padded_height,
    )?);
    let table_columns = ALU_COLUMNS
        .iter()
        .map(|&column| table_column_values(table, column).map(FoldedColumn::new))
        .collect::<Result<Vec<_>, _>>()?;
    let shape_columns = (0..ALU_SHAPE_WIDTH)
        .map(|column| {
            expected_alu_column_values::<F, EF>(table.metadata.padded_height, expected_rows, column)
                .map(FoldedColumn::new)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let selector_kinds = [
        AluOpKind::Add,
        AluOpKind::Mul,
        AluOpKind::BoolCheck,
        AluOpKind::MulAdd,
        AluOpKind::HornerAcc,
    ];
    let selectors = selector_kinds
        .iter()
        .map(|&kind| {
            expected_alu_selector_values::<F, EF>(table.metadata.padded_height, expected_rows, kind)
                .map(FoldedColumn::new)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(AluFoldedSumcheckState {
        eq,
        active,
        table_columns,
        shape_columns,
        selectors,
        alpha,
    })
}

fn alu_folded_suffix_evals<F, EF>(
    state: &AluFoldedSumcheckState<EF>,
    suffix: usize,
    degree: usize,
    diagnostics: &WhirNativeLocalProofDiagnostics,
) -> Result<Vec<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let eq_evals = state.eq.line_evals::<F>(suffix, degree);
    let active_evals = diagnostics.time(
        WhirNativeLocalDiagMetric::StaticSelector,
        degree + 1,
        || state.active.line_evals::<F>(suffix, degree),
    );
    let table_column_evals = diagnostics.time(
        WhirNativeLocalDiagMetric::TableColumn,
        state.table_columns.len() * (degree + 1),
        || {
            state
                .table_columns
                .iter()
                .map(|column| column.line_evals::<F>(suffix, degree))
                .collect::<Vec<_>>()
        },
    );
    let shape_evals = diagnostics.time(
        WhirNativeLocalDiagMetric::StaticSelector,
        state.shape_columns.len() * (degree + 1),
        || {
            state
                .shape_columns
                .iter()
                .map(|column| column.line_evals::<F>(suffix, degree))
                .collect::<Vec<_>>()
        },
    );
    let selector_evals = diagnostics.time(
        WhirNativeLocalDiagMetric::StaticSelector,
        state.selectors.len() * (degree + 1),
        || {
            state
                .selectors
                .iter()
                .map(|column| column.line_evals::<F>(suffix, degree))
                .collect::<Vec<_>>()
        },
    );
    diagnostics.time(
        WhirNativeLocalDiagMetric::ConstraintBatch,
        degree + 1,
        || {
            let mut evals = Vec::with_capacity(degree + 1);
            for point in 0..=degree {
                let values = table_column_evals
                    .iter()
                    .map(|column| column[point])
                    .collect::<Vec<_>>();
                let shape_values = shape_evals
                    .iter()
                    .map(|column| column[point])
                    .collect::<Vec<_>>();
                let selectors = selector_evals
                    .iter()
                    .map(|column| column[point])
                    .collect::<Vec<_>>();
                let constraint = eval_alu_constraint_from_folded_values(
                    active_evals[point],
                    &values,
                    &shape_values,
                    &selectors,
                    state.alpha,
                )?;
                evals.push(eq_evals[point] * constraint);
            }
            Ok(evals)
        },
    )
}

fn alu_folded_suffix_evals_into<F, EF>(
    state: &AluFoldedSumcheckState<EF>,
    suffix: usize,
    degree: usize,
    diagnostics: &WhirNativeLocalProofDiagnostics,
    out: &mut [EF],
) -> Result<(), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let values = alu_folded_suffix_evals::<F, EF>(state, suffix, degree, diagnostics)?;
    if values.len() != out.len() {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "ALU suffix output length mismatch: expected {}, got {}",
            out.len(),
            values.len()
        )));
    }
    out.copy_from_slice(&values);
    Ok(())
}

fn eval_alu_constraint_from_folded_values<EF>(
    active: EF,
    values: &[EF],
    shape_values: &[EF],
    selectors: &[EF],
    alpha: EF,
) -> Result<EF, WhirNativeCircuitError>
where
    EF: Field,
{
    if values.len() != ALU_WIDTH || shape_values.len() != ALU_SHAPE_WIDTH || selectors.len() != 5 {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "ALU folded evaluator width mismatch".to_string(),
        ));
    }
    let inactive = EF::ONE - active;
    let mut constraints = Vec::with_capacity(24);

    for (column, &expected) in shape_values.iter().enumerate() {
        constraints.push(active * (values[column] - expected));
    }
    for &value in values {
        constraints.push(inactive * value);
    }

    let a = values[ALU_READ_A_COL];
    let b = values[ALU_READ_B_COL];
    let c = values[ALU_READ_C_COL];
    let out = values[ALU_READ_OUT_COL];
    let acc = values[ALU_READ_ACC_COL];
    let sel_add = selectors[0];
    let sel_mul = selectors[1];
    let sel_bool = selectors[2];
    let sel_muladd = selectors[3];
    let sel_horner = selectors[4];

    constraints.push(sel_add * (a + b - out));
    constraints.push(sel_add * c);
    constraints.push(sel_mul * (a * b - out));
    constraints.push(sel_mul * c);
    constraints.push(sel_bool * a * (a - EF::ONE));
    constraints.push(sel_bool * (out - a));
    constraints.push(sel_bool * (c - a));
    constraints.push(sel_muladd * (a * b + c - out));
    constraints.push(sel_horner * (acc * b + c - a - out));
    constraints.push((EF::ONE - sel_horner) * acc);

    Ok(batch_constraints(constraints, alpha))
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
    EF: ExtensionField<F> + Send + Sync,
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

    let diagnostics = WhirNativeLocalProofDiagnostics::new(
        WhirNativeLocalConstraintKind::Alu,
        table_index,
        &table.metadata,
    );
    let mut folded_state = build_alu_folded_sumcheck_state::<F, EF>(
        table,
        expected_rows,
        &zerocheck_point,
        constraint_challenge,
    )?;
    let (sumcheck, terminal_row_point, terminal_claim) =
        prove_row_folded_sumcheck_by_suffix_into_parallel::<F, EF, Challenger, _, _, _>(
            whir_native_table_row_variables(&table.metadata),
            degree,
            EF::ZERO,
            challenger,
            &mut folded_state,
            |state, _round, suffix, degree, out| {
                alu_folded_suffix_evals_into::<F, EF>(state, suffix, degree, &diagnostics, out)
            },
            |state, challenge| state.fold(challenge),
        )?;
    diagnostics.finish();
    let terminal_constraint = eval_alu_constraint_from_table::<F, EF>(
        table,
        expected_rows,
        &terminal_row_point,
        constraint_challenge,
    )?;
    if eq_eval_ext(&zerocheck_point, &terminal_row_point) * terminal_constraint != terminal_claim {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "ALU local prover terminal claim is inconsistent".to_string(),
        ));
    }

    let terminal_openings = terminal_column_claims_for_table::<F, EF>(
        table_index,
        table,
        &terminal_row_point,
        &ALU_COLUMNS,
    )?;

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
    if proof.terminal_openings.len() != ALU_COLUMNS.len() {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "ALU terminal opening count mismatch: expected {}, got {}",
            ALU_COLUMNS.len(),
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
        constraint_challenge,
    )?;
    let expected_terminal = eq_eval_ext(&zerocheck_point, &terminal_row_point) * constraint;
    if expected_terminal != terminal_claim {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "ALU local proof terminal claim is inconsistent".to_string(),
        ));
    }

    let all_claims = opening_claims
        .into_iter()
        .map(|(point, value)| (table_index, point, value))
        .collect::<Vec<_>>();
    Ok(all_claims)
}

#[derive(Clone, Debug)]
struct RecomposeFoldedSumcheckState<EF> {
    eq: FoldedColumn<EF>,
    active: FoldedColumn<EF>,
    table_columns: Vec<FoldedColumn<EF>>,
    shape_columns: Vec<FoldedColumn<EF>>,
    basis: Vec<EF>,
    alpha: EF,
}

impl<EF> RecomposeFoldedSumcheckState<EF>
where
    EF: Field,
{
    fn fold(&mut self, challenge: EF) {
        self.eq.fold(challenge);
        self.active.fold(challenge);
        fold_columns(&mut self.table_columns, challenge);
        fold_columns(&mut self.shape_columns, challenge);
    }
}

fn expected_recompose_shape_column_values<F, EF>(
    padded_height: usize,
    expected_rows: &[WhirNativeExpectedRecomposeRow],
    column: usize,
) -> Result<Vec<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
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
    let mut evals = EF::zero_vec(padded_height);
    for (row_index, row) in expected_rows.iter().enumerate() {
        let value = match column {
            0 => row.kind as u64,
            1 => row.output_wid as u64,
            2 => row.input_wids.len() as u64,
            _ => row.input_wids[column - 3] as u64,
        };
        evals[row_index] = ef_from_u64::<F, EF>(value);
    }
    Ok(evals)
}

fn build_recompose_folded_sumcheck_state<F, EF>(
    table: &WhirNativeTableData<EF>,
    expected_rows: &[WhirNativeExpectedRecomposeRow],
    zerocheck_point: &Point<EF>,
    alpha: EF,
) -> Result<RecomposeFoldedSumcheckState<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let d = recompose_degree(expected_rows)?;
    let eq = FoldedColumn::new(
        Poly::<EF>::new_from_point(zerocheck_point.as_slice(), EF::ONE).into_evals(),
    );
    let active = FoldedColumn::new(active_selector_values::<EF>(
        table.metadata.active_rows,
        table.metadata.padded_height,
    )?);
    let table_columns = logical_columns(&table.metadata)
        .into_iter()
        .map(|column| table_column_values(table, column).map(FoldedColumn::new))
        .collect::<Result<Vec<_>, _>>()?;
    let shape_columns = (0..3 + d)
        .map(|column| {
            expected_recompose_shape_column_values::<F, EF>(
                table.metadata.padded_height,
                expected_rows,
                column,
            )
            .map(FoldedColumn::new)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let basis = (0..d)
        .map(|coeff| {
            EF::ith_basis_element(coeff).ok_or_else(|| {
                WhirNativeCircuitError::ConstraintViolation(format!(
                    "missing extension basis element {coeff}"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(RecomposeFoldedSumcheckState {
        eq,
        active,
        table_columns,
        shape_columns,
        basis,
        alpha,
    })
}

fn eval_recompose_constraint_from_folded_values<EF>(
    active: EF,
    values: &[EF],
    shape_values: &[EF],
    basis: &[EF],
    alpha: EF,
) -> Result<EF, WhirNativeCircuitError>
where
    EF: Field,
{
    let d = basis.len();
    let expected_width = 4 + d + d;
    if values.len() != expected_width {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "recompose folded width mismatch: expected {expected_width}, got {}",
            values.len()
        )));
    }
    if shape_values.len() != 3 + d {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "recompose folded auxiliary width mismatch".to_string(),
        ));
    }
    let inactive = EF::ONE - active;
    let mut constraints = Vec::with_capacity(values.len() * 2 + d + 1);
    for &value in values {
        constraints.push(inactive * value);
    }
    for column in 0..3 + d {
        constraints.push(active * (values[column] - shape_values[column]));
    }
    let value_start = 3 + d;
    let mut recomposed = EF::ZERO;
    for coeff in 0..d {
        recomposed += values[value_start + coeff] * basis[coeff];
    }
    constraints.push(active * (recomposed - values[value_start + d]));
    Ok(batch_constraints(constraints, alpha))
}

fn recompose_folded_suffix_evals_into<F, EF>(
    state: &RecomposeFoldedSumcheckState<EF>,
    suffix: usize,
    degree: usize,
    diagnostics: &WhirNativeLocalProofDiagnostics,
    out: &mut [EF],
) -> Result<(), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if out.len() != degree + 1 {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "recompose suffix output length mismatch: expected {}, got {}",
            degree + 1,
            out.len()
        )));
    }
    diagnostics.time(
        WhirNativeLocalDiagMetric::ConstraintBatch,
        degree + 1,
        || {
            for (point, out) in out.iter_mut().enumerate() {
                let eq = state.eq.line_value::<F>(suffix, point);
                let active = state.active.line_value::<F>(suffix, point);
                let values =
                    folded_column_line_values::<F, EF>(&state.table_columns, suffix, point);
                let shape_values =
                    folded_column_line_values::<F, EF>(&state.shape_columns, suffix, point);
                let constraint = eval_recompose_constraint_from_folded_values(
                    active,
                    &values,
                    &shape_values,
                    &state.basis,
                    state.alpha,
                )?;
                *out = eq * constraint;
            }
            Ok(())
        },
    )
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
    EF: ExtensionField<F> + Send + Sync,
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

    let diagnostics = WhirNativeLocalProofDiagnostics::new(
        WhirNativeLocalConstraintKind::Recompose,
        table_index,
        &table.metadata,
    );
    let mut folded_state = build_recompose_folded_sumcheck_state::<F, EF>(
        table,
        expected_rows,
        &zerocheck_point,
        constraint_challenge,
    )?;
    let (sumcheck, terminal_row_point, terminal_claim) =
        prove_row_folded_sumcheck_by_suffix_into_parallel::<F, EF, Challenger, _, _, _>(
            whir_native_table_row_variables(&table.metadata),
            degree,
            EF::ZERO,
            challenger,
            &mut folded_state,
            |state, _round, suffix, degree, out| {
                recompose_folded_suffix_evals_into::<F, EF>(
                    state,
                    suffix,
                    degree,
                    &diagnostics,
                    out,
                )
            },
            |state, challenge| state.fold(challenge),
        )?;
    diagnostics.finish();
    let terminal_constraint = eval_recompose_constraint_from_table::<F, EF>(
        table,
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
    let terminal_openings = terminal_column_claims_for_table::<F, EF>(
        table_index,
        table,
        &terminal_row_point,
        &columns,
    )?;

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

    let columns = logical_columns(metadata);
    if proof.terminal_openings.len() != columns.len() {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "recompose terminal opening count mismatch: expected {}, got {}",
            columns.len(),
            proof.terminal_openings.len()
        )));
    }

    let (column_values, opening_claims) =
        extract_terminal_column_values::<F, EF>(proof, metadata, &terminal_row_point, &columns)?;

    let constraint = eval_recompose_constraint_from_values::<F, EF>(
        metadata,
        expected_rows,
        &terminal_row_point,
        &column_values,
        constraint_challenge,
    )?;
    let expected_terminal = eq_eval_ext(&zerocheck_point, &terminal_row_point) * constraint;
    if expected_terminal != terminal_claim {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "recompose local proof terminal claim is inconsistent".to_string(),
        ));
    }

    let all_claims = opening_claims
        .into_iter()
        .map(|(point, value)| (table_index, point, value))
        .collect::<Vec<_>>();
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
    let diagnostics = WhirNativeLocalProofDiagnostics::new(
        WhirNativeLocalConstraintKind::Poseidon2Air,
        table_index,
        &table.metadata,
    );
    let mut folded_state = build_poseidon2_folded_sumcheck_state::<F, EF>(
        table,
        &preprocessed,
        &shifted_preprocessed,
        &zerocheck_point,
        constraint_challenge,
    )?;

    let (sumcheck, terminal_row_point, terminal_claim) =
        prove_row_folded_sumcheck_by_suffix_into_parallel::<F, EF, Challenger, _, _, _>(
            whir_native_table_row_variables(&table.metadata),
            degree,
            EF::ZERO,
            challenger,
            &mut folded_state,
            |state, _round, suffix, degree, out| {
                poseidon2_folded_suffix_evals_into::<F, EF>(
                    &table.metadata,
                    expected_rows,
                    state,
                    suffix,
                    degree,
                    &diagnostics,
                    out,
                )
            },
            |state, challenge| state.fold(challenge),
        )?;
    diagnostics.finish();
    let terminal_constraint = eval_poseidon2_air_constraint_from_table::<F, EF>(
        table,
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
    let read_columns = poseidon2_read_columns();
    let mut terminal_openings = terminal_column_claims_for_table::<F, EF>(
        table_index,
        table,
        &terminal_row_point,
        &local_columns,
    )?;
    terminal_openings.extend(terminal_column_claims_for_table::<F, EF>(
        table_index,
        table,
        &terminal_row_point,
        &shifted_columns,
    )?);
    terminal_openings.extend(terminal_column_claims_for_table::<F, EF>(
        table_index,
        table,
        &terminal_row_point,
        &read_columns,
    )?);

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
    let read_columns = poseidon2_read_columns();
    let expected_openings = local_columns.len() + shifted_columns.len() + read_columns.len();
    if proof.terminal_openings.len() != expected_openings {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "Poseidon2 AIR terminal opening count mismatch: expected {expected_openings}, got {}",
            proof.terminal_openings.len()
        )));
    }

    let (local_values, shifted_values, mut opening_claims) =
        extract_terminal_poseidon2_transition_values::<F, EF>(
            proof,
            metadata,
            &terminal_row_point,
            &local_columns,
            &shifted_columns,
        )?;
    let (read_values, read_claims) = extract_terminal_poseidon2_read_values::<F, EF>(
        proof,
        local_columns.len() + shifted_columns.len(),
        &terminal_row_point,
        metadata,
        &read_columns,
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
        &prep_local,
        &prep_next,
        &read_values,
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
        read_claims
            .into_iter()
            .map(|(point, value)| (table_index, point, value)),
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

#[derive(Clone, Debug)]
struct Poseidon2FoldedSumcheckState<EF> {
    eq: FoldedColumn<EF>,
    local_columns: Vec<FoldedColumn<EF>>,
    shifted_columns: Vec<FoldedColumn<EF>>,
    read_columns: Vec<FoldedColumn<EF>>,
    preprocessed_columns: Vec<FoldedColumn<EF>>,
    shifted_preprocessed_columns: Vec<FoldedColumn<EF>>,
    prefix: Vec<EF>,
    alpha: EF,
}

impl<EF> Poseidon2FoldedSumcheckState<EF>
where
    EF: Field,
{
    fn fold(&mut self, challenge: EF) {
        self.eq.fold(challenge);
        fold_columns(&mut self.local_columns, challenge);
        fold_columns(&mut self.shifted_columns, challenge);
        fold_columns(&mut self.read_columns, challenge);
        fold_columns(&mut self.preprocessed_columns, challenge);
        fold_columns(&mut self.shifted_preprocessed_columns, challenge);
        self.prefix.push(challenge);
    }
}

fn build_poseidon2_folded_sumcheck_state<F, EF>(
    table: &WhirNativeTableData<EF>,
    preprocessed: &[EF],
    shifted_preprocessed: &[EF],
    zerocheck_point: &Point<EF>,
    alpha: EF,
) -> Result<Poseidon2FoldedSumcheckState<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let eq = FoldedColumn::new(
        Poly::<EF>::new_from_point(zerocheck_point.as_slice(), EF::ONE).into_evals(),
    );
    let local_columns = (0..P2_BB_D4_WIDTH16_AIR_WIDTH)
        .map(|column| table_column_values(table, column).map(FoldedColumn::new))
        .collect::<Result<Vec<_>, _>>()?;
    let shifted_columns = (0..P2_BB_D4_WIDTH16_AIR_WIDTH)
        .map(|column| {
            table_column_values(table, P2_BB_D4_WIDTH16_SHIFTED_OFFSET + column)
                .map(FoldedColumn::new)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let read_columns = (0..P2_BB_D4_WIDTH16_WITNESS_PORTS)
        .map(|column| {
            table_column_values(table, P2_BB_D4_WIDTH16_READ_OFFSET + column).map(FoldedColumn::new)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let preprocessed_columns = (0..P2_BB_D4_WIDTH16_PREPROCESSED_WIDTH)
        .map(|column| {
            row_major_column_values(preprocessed, P2_BB_D4_WIDTH16_PREPROCESSED_WIDTH, column)
                .map(FoldedColumn::new)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let shifted_preprocessed_columns = (0..P2_BB_D4_WIDTH16_PREPROCESSED_WIDTH)
        .map(|column| {
            row_major_column_values(
                shifted_preprocessed,
                P2_BB_D4_WIDTH16_PREPROCESSED_WIDTH,
                column,
            )
            .map(FoldedColumn::new)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Poseidon2FoldedSumcheckState {
        eq,
        local_columns,
        shifted_columns,
        read_columns,
        preprocessed_columns,
        shifted_preprocessed_columns,
        prefix: Vec::new(),
        alpha,
    })
}

fn poseidon2_folded_suffix_evals_into<F, EF>(
    metadata: &WhirNativeTableMetadata,
    expected_rows: &[Poseidon2CircuitRow<BabyBear>],
    state: &Poseidon2FoldedSumcheckState<EF>,
    suffix: usize,
    degree: usize,
    diagnostics: &WhirNativeLocalProofDiagnostics,
    out: &mut [EF],
) -> Result<(), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F> + ExtensionField<BabyBear> + Send + Sync,
{
    if out.len() != degree + 1 {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "Poseidon2 suffix output length mismatch: expected {}, got {}",
            degree + 1,
            out.len()
        )));
    }
    let suffix_vars = whir_native_table_row_variables(metadata)
        .checked_sub(state.prefix.len() + 1)
        .ok_or_else(|| {
            WhirNativeCircuitError::ConstraintViolation(
                "Poseidon2 folded prefix is longer than row arity".to_string(),
            )
        })?;

    for (point, out) in out.iter_mut().enumerate() {
        let t = EF::from(F::from_u64(point as u64));
        let row_point =
            point_from_prefix_current_suffix::<F, EF>(&state.prefix, t, suffix, suffix_vars);
        let eq = state.eq.line_value::<F>(suffix, point);
        let local_values = diagnostics.time(
            WhirNativeLocalDiagMetric::TableColumn,
            P2_BB_D4_WIDTH16_AIR_WIDTH,
            || folded_column_line_values::<F, EF>(&state.local_columns, suffix, point),
        );
        let shifted_values = diagnostics.time(
            WhirNativeLocalDiagMetric::TableColumn,
            P2_BB_D4_WIDTH16_AIR_WIDTH,
            || folded_column_line_values::<F, EF>(&state.shifted_columns, suffix, point),
        );
        let prep_local = diagnostics.time(
            WhirNativeLocalDiagMetric::StaticSelector,
            P2_BB_D4_WIDTH16_PREPROCESSED_WIDTH,
            || folded_column_line_values::<F, EF>(&state.preprocessed_columns, suffix, point),
        );
        let prep_next = diagnostics.time(
            WhirNativeLocalDiagMetric::StaticSelector,
            P2_BB_D4_WIDTH16_PREPROCESSED_WIDTH,
            || {
                folded_column_line_values::<F, EF>(
                    &state.shifted_preprocessed_columns,
                    suffix,
                    point,
                )
            },
        );
        let read_values = diagnostics.time(
            WhirNativeLocalDiagMetric::TableColumn,
            P2_BB_D4_WIDTH16_WITNESS_PORTS,
            || folded_column_line_values::<F, EF>(&state.read_columns, suffix, point),
        );
        let constraint = diagnostics.time(WhirNativeLocalDiagMetric::ConstraintBatch, 1, || {
            eval_poseidon2_air_constraint_from_values::<F, EF>(
                metadata,
                expected_rows,
                &row_point,
                &local_values,
                &shifted_values,
                &prep_local,
                &prep_next,
                &read_values,
                state.alpha,
            )
        })?;
        *out = eq * constraint;
    }

    Ok(())
}

fn poseidon2_air_local_degree(
    _metadata: &WhirNativeTableMetadata,
    _witness_metadata: &WhirNativeTableMetadata,
) -> usize {
    // Poseidon2 AIR constraints, read bindings, and the static selector
    // transition window are all evaluated on committed same-row columns. The
    // shifted-column copy check is handled by the dedicated shift bus.
    5
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
    expected_rows: &[Poseidon2CircuitRow<BabyBear>],
    _direction_bit_witness_ids: &[u32],
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
    let prep_local = eval_poseidon2_preprocessed_row::<F, EF>(preprocessed, row_point)?;
    let prep_next = eval_poseidon2_preprocessed_row::<F, EF>(shifted_preprocessed, row_point)?;
    let read_values = (0..P2_BB_D4_WIDTH16_WITNESS_PORTS)
        .map(|column| {
            eval_table_column_at_row_point::<F, EF>(
                table,
                row_point,
                P2_BB_D4_WIDTH16_READ_OFFSET + column,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    eval_poseidon2_air_constraint_from_values::<F, EF>(
        &table.metadata,
        expected_rows,
        row_point,
        &local_values,
        &shifted_values,
        &prep_local,
        &prep_next,
        &read_values,
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

fn poseidon2_read_columns() -> Vec<usize> {
    (P2_BB_D4_WIDTH16_READ_OFFSET..P2_BB_D4_WIDTH16_READ_OFFSET + P2_BB_D4_WIDTH16_WITNESS_PORTS)
        .collect()
}

#[derive(Clone, Debug)]
struct Poseidon2ShiftFoldedSumcheckState<EF> {
    eq: FoldedColumn<EF>,
    active: FoldedColumn<EF>,
    row_id: FoldedColumn<EF>,
    next_row_id: FoldedColumn<EF>,
    local_columns: Vec<FoldedColumn<EF>>,
    shifted_columns: Vec<FoldedColumn<EF>>,
    sender_inv: FoldedColumn<EF>,
    receiver_inv: FoldedColumn<EF>,
    prefix: Vec<EF>,
    theta: EF,
    alpha_shift: EF,
    beta_shift: EF,
    constraint_challenge: EF,
    sum_challenge: EF,
}

impl<EF> Poseidon2ShiftFoldedSumcheckState<EF>
where
    EF: Field,
{
    fn fold(&mut self, challenge: EF) {
        self.eq.fold(challenge);
        self.active.fold(challenge);
        self.row_id.fold(challenge);
        self.next_row_id.fold(challenge);
        fold_columns(&mut self.local_columns, challenge);
        fold_columns(&mut self.shifted_columns, challenge);
        self.sender_inv.fold(challenge);
        self.receiver_inv.fold(challenge);
        self.prefix.push(challenge);
    }
}

fn poseidon2_shift_aux_metadata(
    source_metadata: &WhirNativeTableMetadata,
    options: WhirNativeCircuitOptions,
) -> WhirNativeTableMetadata {
    let mut padded_width = P2_SHIFT_AUX_WIDTH.next_power_of_two();
    while padded_width * source_metadata.padded_height < (1usize << options.min_num_variables) {
        padded_width *= 2;
    }
    WhirNativeTableMetadata {
        kind: WhirNativeTableKind::Poseidon2Shift,
        op_type: source_metadata.op_type.clone(),
        width: P2_SHIFT_AUX_WIDTH,
        padded_width,
        active_rows: source_metadata.active_rows,
        padded_height: source_metadata.padded_height,
        num_variables: (padded_width * source_metadata.padded_height).ilog2() as usize,
        column_layout_version: TABLE_LAYOUT_VERSION,
    }
}

fn pack_poseidon2_shift_aux_rows<EF>(
    source_metadata: &WhirNativeTableMetadata,
    rows: Vec<[EF; P2_SHIFT_AUX_WIDTH]>,
    options: WhirNativeCircuitOptions,
) -> Result<WhirNativeTableData<EF>, WhirNativeCircuitError>
where
    EF: Field,
{
    if rows.len() != source_metadata.active_rows {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "Poseidon2 shift inverse row count mismatch: expected {}, got {}",
            source_metadata.active_rows,
            rows.len()
        )));
    }
    let metadata = poseidon2_shift_aux_metadata(source_metadata, options);
    let mut values = EF::zero_vec(metadata.padded_width * metadata.padded_height);
    for (row_index, row) in rows.iter().enumerate() {
        let row_start = row_index * metadata.padded_width;
        values[row_start + P2_SHIFT_SENDER_INV_COL] = row[P2_SHIFT_SENDER_INV_COL];
        values[row_start + P2_SHIFT_RECEIVER_INV_COL] = row[P2_SHIFT_RECEIVER_INV_COL];
    }
    Ok(WhirNativeTableData { metadata, values })
}

fn poseidon2_shift_row_id_values<F, EF>(
    active_rows: usize,
    padded_height: usize,
) -> Result<Vec<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if active_rows == 0 || active_rows > padded_height {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "bad Poseidon2 shift active row count {active_rows} for height {padded_height}"
        )));
    }
    let mut values = EF::zero_vec(padded_height);
    for (row, value) in values.iter_mut().take(active_rows).enumerate() {
        *value = ef_from_u64::<F, EF>(row as u64);
    }
    Ok(values)
}

fn poseidon2_shift_next_row_id_values<F, EF>(
    active_rows: usize,
    padded_height: usize,
) -> Result<Vec<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if active_rows == 0 || active_rows > padded_height {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "bad Poseidon2 shift active row count {active_rows} for height {padded_height}"
        )));
    }
    let mut values = EF::zero_vec(padded_height);
    for (row, value) in values.iter_mut().take(active_rows).enumerate() {
        *value = ef_from_u64::<F, EF>(((row + 1) % active_rows) as u64);
    }
    Ok(values)
}

fn compressed_linear_combination<EF>(values: &[EF], theta: EF) -> EF
where
    EF: Field,
{
    let mut acc = EF::ZERO;
    let mut coeff = EF::ONE;
    for &value in values {
        acc += coeff * value;
        coeff *= theta;
    }
    acc
}

fn poseidon2_air_row_fold_from_table<EF>(
    table: &WhirNativeTableData<EF>,
    row: usize,
    offset: usize,
    theta: EF,
) -> Result<EF, WhirNativeCircuitError>
where
    EF: Field,
{
    if row >= table.metadata.active_rows
        || offset + P2_BB_D4_WIDTH16_AIR_WIDTH > table.metadata.width
    {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "Poseidon2 shift row fold out of range".to_string(),
        ));
    }
    let start = row * table.metadata.padded_width + offset;
    Ok(compressed_linear_combination(
        &table.values[start..start + P2_BB_D4_WIDTH16_AIR_WIDTH],
        theta,
    ))
}

fn build_poseidon2_shift_aux_table<F, EF>(
    table: &WhirNativeTableData<EF>,
    theta: EF,
    alpha_shift: EF,
    beta_shift: EF,
    options: WhirNativeCircuitOptions,
) -> Result<(WhirNativeTableData<EF>, EF, EF), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    validate_poseidon2_shift_source_metadata(&table.metadata)?;
    let active_rows = table.metadata.active_rows;
    let mut rows = Vec::with_capacity(active_rows);
    let mut sender_cumulative = EF::ZERO;
    let mut receiver_cumulative = EF::ZERO;
    for row in 0..active_rows {
        let row_id = ef_from_u64::<F, EF>(row as u64);
        let next_row_id = ef_from_u64::<F, EF>(((row + 1) % active_rows) as u64);
        let main_fold = poseidon2_air_row_fold_from_table(table, row, 0, theta)?;
        let shifted_fold =
            poseidon2_air_row_fold_from_table(table, row, P2_BB_D4_WIDTH16_SHIFTED_OFFSET, theta)?;
        let sender_key = row_id + beta_shift * main_fold;
        let receiver_key = next_row_id + beta_shift * shifted_fold;
        let sender_inv = (alpha_shift - sender_key).try_inverse().ok_or_else(|| {
            WhirNativeCircuitError::ConstraintViolation(
                "Poseidon2 shift sender challenge collision".to_string(),
            )
        })?;
        let receiver_inv = (alpha_shift - receiver_key).try_inverse().ok_or_else(|| {
            WhirNativeCircuitError::ConstraintViolation(
                "Poseidon2 shift receiver challenge collision".to_string(),
            )
        })?;
        sender_cumulative += sender_inv;
        receiver_cumulative += receiver_inv;
        rows.push([sender_inv, receiver_inv]);
    }
    let aux_table = pack_poseidon2_shift_aux_rows(&table.metadata, rows, options)?;
    Ok((aux_table, sender_cumulative, receiver_cumulative))
}

fn validate_poseidon2_shift_source_metadata(
    metadata: &WhirNativeTableMetadata,
) -> Result<(), WhirNativeCircuitError> {
    if metadata.kind != WhirNativeTableKind::Poseidon2
        || metadata.width != P2_BB_D4_WIDTH16_TABLE_WIDTH
        || metadata.active_rows == 0
    {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "Poseidon2 shift proof requires a Poseidon2 table".to_string(),
        ));
    }
    Ok(())
}

fn build_poseidon2_shift_folded_sumcheck_state<F, EF>(
    table: &WhirNativeTableData<EF>,
    aux_table: &WhirNativeTableData<EF>,
    zerocheck_point: &Point<EF>,
    theta: EF,
    alpha_shift: EF,
    beta_shift: EF,
    constraint_challenge: EF,
    sum_challenge: EF,
) -> Result<Poseidon2ShiftFoldedSumcheckState<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let eq = FoldedColumn::new(
        Poly::<EF>::new_from_point(zerocheck_point.as_slice(), EF::ONE).into_evals(),
    );
    let active = FoldedColumn::new(active_selector_values::<EF>(
        table.metadata.active_rows,
        table.metadata.padded_height,
    )?);
    let row_id = FoldedColumn::new(poseidon2_shift_row_id_values::<F, EF>(
        table.metadata.active_rows,
        table.metadata.padded_height,
    )?);
    let next_row_id = FoldedColumn::new(poseidon2_shift_next_row_id_values::<F, EF>(
        table.metadata.active_rows,
        table.metadata.padded_height,
    )?);
    let local_columns = (0..P2_BB_D4_WIDTH16_AIR_WIDTH)
        .map(|column| table_column_values(table, column).map(FoldedColumn::new))
        .collect::<Result<Vec<_>, _>>()?;
    let shifted_columns = (0..P2_BB_D4_WIDTH16_AIR_WIDTH)
        .map(|column| {
            table_column_values(table, P2_BB_D4_WIDTH16_SHIFTED_OFFSET + column)
                .map(FoldedColumn::new)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let sender_inv = FoldedColumn::new(table_column_values(aux_table, P2_SHIFT_SENDER_INV_COL)?);
    let receiver_inv =
        FoldedColumn::new(table_column_values(aux_table, P2_SHIFT_RECEIVER_INV_COL)?);
    Ok(Poseidon2ShiftFoldedSumcheckState {
        eq,
        active,
        row_id,
        next_row_id,
        local_columns,
        shifted_columns,
        sender_inv,
        receiver_inv,
        prefix: Vec::new(),
        theta,
        alpha_shift,
        beta_shift,
        constraint_challenge,
        sum_challenge,
    })
}

fn eval_poseidon2_shift_constraint_from_folds<EF>(
    eq: EF,
    active: EF,
    row_id: EF,
    next_row_id: EF,
    main_fold: EF,
    shifted_fold: EF,
    sender_inv: EF,
    receiver_inv: EF,
    alpha_shift: EF,
    beta_shift: EF,
    constraint_challenge: EF,
    sum_challenge: EF,
) -> EF
where
    EF: Field,
{
    let sender_key = row_id + beta_shift * main_fold;
    let receiver_key = next_row_id + beta_shift * shifted_fold;
    let sender_constraint = (alpha_shift - sender_key) * sender_inv - active;
    let receiver_constraint = (alpha_shift - receiver_key) * receiver_inv - active;
    eq * (sender_constraint + constraint_challenge * receiver_constraint)
        + sum_challenge * (sender_inv - receiver_inv)
}

fn poseidon2_shift_folded_suffix_evals_into<F, EF>(
    metadata: &WhirNativeTableMetadata,
    state: &Poseidon2ShiftFoldedSumcheckState<EF>,
    suffix: usize,
    degree: usize,
    out: &mut [EF],
) -> Result<(), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if out.len() != degree + 1 {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "Poseidon2 shift suffix output length mismatch: expected {}, got {}",
            degree + 1,
            out.len()
        )));
    }
    let suffix_vars = whir_native_table_row_variables(metadata)
        .checked_sub(state.prefix.len() + 1)
        .ok_or_else(|| {
            WhirNativeCircuitError::ConstraintViolation(
                "Poseidon2 shift folded prefix is longer than row arity".to_string(),
            )
        })?;
    for (point, out) in out.iter_mut().enumerate() {
        let t = EF::from(F::from_u64(point as u64));
        let row_point =
            point_from_prefix_current_suffix::<F, EF>(&state.prefix, t, suffix, suffix_vars);
        let eq = state.eq.line_value::<F>(suffix, point);
        let active = state.active.line_value::<F>(suffix, point);
        let row_id = state.row_id.line_value::<F>(suffix, point);
        let next_row_id = state.next_row_id.line_value::<F>(suffix, point);
        let local_values = folded_column_line_values::<F, EF>(&state.local_columns, suffix, point);
        let shifted_values =
            folded_column_line_values::<F, EF>(&state.shifted_columns, suffix, point);
        let main_fold = compressed_linear_combination(&local_values, state.theta);
        let shifted_fold = compressed_linear_combination(&shifted_values, state.theta);
        let sender_inv = state.sender_inv.line_value::<F>(suffix, point);
        let receiver_inv = state.receiver_inv.line_value::<F>(suffix, point);
        let _ = row_point;
        *out = eval_poseidon2_shift_constraint_from_folds(
            eq,
            active,
            row_id,
            next_row_id,
            main_fold,
            shifted_fold,
            sender_inv,
            receiver_inv,
            state.alpha_shift,
            state.beta_shift,
            state.constraint_challenge,
            state.sum_challenge,
        );
    }
    Ok(())
}

fn eval_poseidon2_shift_constraint_from_values<F, EF>(
    metadata: &WhirNativeTableMetadata,
    zerocheck_point: &Point<EF>,
    row_point: &Point<EF>,
    local_values: &[EF],
    shifted_values: &[EF],
    sender_inv: EF,
    receiver_inv: EF,
    theta: EF,
    alpha_shift: EF,
    beta_shift: EF,
    constraint_challenge: EF,
    sum_challenge: EF,
) -> Result<EF, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if local_values.len() != P2_BB_D4_WIDTH16_AIR_WIDTH
        || shifted_values.len() != P2_BB_D4_WIDTH16_AIR_WIDTH
    {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "Poseidon2 shift terminal width mismatch".to_string(),
        ));
    }
    let active =
        active_selector_eval::<F, EF>(metadata.active_rows, metadata.padded_height, row_point)?;
    let row_ids =
        poseidon2_shift_row_id_values::<F, EF>(metadata.active_rows, metadata.padded_height)?;
    let row_id = Poly::eval_ext_slice::<F>(&row_ids, row_point);
    let next_row_ids =
        poseidon2_shift_next_row_id_values::<F, EF>(metadata.active_rows, metadata.padded_height)?;
    let next_row_id = Poly::eval_ext_slice::<F>(&next_row_ids, row_point);
    let eq = eq_eval_ext(zerocheck_point, row_point);
    let main_fold = compressed_linear_combination(local_values, theta);
    let shifted_fold = compressed_linear_combination(shifted_values, theta);
    Ok(eval_poseidon2_shift_constraint_from_folds(
        eq,
        active,
        row_id,
        next_row_id,
        main_fold,
        shifted_fold,
        sender_inv,
        receiver_inv,
        alpha_shift,
        beta_shift,
        constraint_challenge,
        sum_challenge,
    ))
}

#[allow(clippy::too_many_arguments)]
fn prove_poseidon2_shift_bus<
    F,
    EF,
    MT,
    Challenger,
    Dft,
    MakePcs,
    MakeChallenger,
    const DIGEST_ELEMS: usize,
>(
    tables: &[WhirNativeTableData<EF>],
    metadata: &[WhirNativeTableMetadata],
    public_io_digest: &[F],
    shape_digest: &[F],
    options: WhirNativeCircuitOptions,
    commitment_context: WhirNativeCommitmentContext<'_, MT::Commitment>,
    make_pcs: &MakePcs,
    make_challenger: &MakeChallenger,
) -> Result<
    (
        WhirNativePoseidon2ShiftBusProof<F, EF, MT>,
        Vec<WhirNativeTerminalColumnClaim<EF>>,
    ),
    WhirNativeCircuitError,
>
where
    F: TwoAdicField + PrimeField64 + Ord + Send + Sync + Serialize + for<'de> Deserialize<'de>,
    EF: ExtensionField<F> + TwoAdicField + Send + Sync + Serialize + for<'de> Deserialize<'de>,
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
{
    let mut challenge_challenger = make_challenger();
    observe_poseidon2_shift_challenge_context(
        &mut challenge_challenger,
        public_io_digest,
        shape_digest,
        options,
        commitment_context.clone(),
    );
    let theta = challenge_challenger.sample_algebra_element();
    let alpha = challenge_challenger.sample_algebra_element();
    let beta = challenge_challenger.sample_algebra_element();

    let poseidon2_table_indices = metadata
        .iter()
        .enumerate()
        .filter_map(|(table_index, table_metadata)| {
            (table_metadata.kind == WhirNativeTableKind::Poseidon2).then_some(table_index)
        })
        .collect::<Vec<_>>();
    let mut sections = Vec::with_capacity(poseidon2_table_indices.len());
    let mut terminal_main_claims = Vec::new();
    for table_index in poseidon2_table_indices {
        let table = tables.get(table_index).ok_or_else(|| {
            WhirNativeCircuitError::ConstraintViolation(format!(
                "missing Poseidon2 shift source table {table_index}"
            ))
        })?;
        let table_metadata = metadata.get(table_index).ok_or_else(|| {
            WhirNativeCircuitError::ConstraintViolation(format!(
                "missing Poseidon2 shift source metadata {table_index}"
            ))
        })?;
        if &table.metadata != table_metadata {
            return Err(WhirNativeCircuitError::TableMetadataMismatch { table_index });
        }
        validate_poseidon2_shift_source_metadata(table_metadata)?;
        let (aux_table, sender_cumulative, receiver_cumulative) =
            build_poseidon2_shift_aux_table::<F, EF>(table, theta, alpha, beta, options)?;
        let pcs = make_pcs(aux_table.metadata.num_variables);
        let mut aux_challenger = make_challenger();
        observe_poseidon2_shift_aux_table_context(
            &mut aux_challenger,
            public_io_digest,
            shape_digest,
            options,
            table_index,
            table_metadata,
            &aux_table.metadata,
            theta,
            alpha,
            beta,
        );
        let (inverse_commitment_value, inverse_prover_data) = pcs.commit_extension_deferred(
            RowMajorMatrix::new(aux_table.values.clone(), 1),
            &mut aux_challenger,
        );
        let inverse_commitment = WhirNativeTableCommitment {
            metadata: aux_table.metadata.clone(),
            commitment: inverse_commitment_value,
        };

        let mut section_challenger = make_challenger();
        observe_poseidon2_shift_constraint_context(
            &mut section_challenger,
            public_io_digest,
            shape_digest,
            options,
            commitment_context.clone(),
            table_index,
            table_metadata,
            &inverse_commitment,
            theta,
            alpha,
            beta,
        );
        let (constraint_challenge, sum_challenge, zerocheck_point) =
            sample_poseidon2_shift_constraint_challenges::<F, EF, Challenger>(
                &mut section_challenger,
                whir_native_table_row_variables(table_metadata),
            );
        let mut folded_state = build_poseidon2_shift_folded_sumcheck_state::<F, EF>(
            table,
            &aux_table,
            &zerocheck_point,
            theta,
            alpha,
            beta,
            constraint_challenge,
            sum_challenge,
        )?;
        let claimed_sum = sum_challenge * (sender_cumulative - receiver_cumulative);
        let (sumcheck, terminal_row_point, terminal_claim) =
            prove_row_folded_sumcheck_by_suffix_into_parallel::<F, EF, Challenger, _, _, _>(
                whir_native_table_row_variables(table_metadata),
                P2_SHIFT_LOCAL_DEGREE,
                claimed_sum,
                &mut section_challenger,
                &mut folded_state,
                |state, _round, suffix, degree, out| {
                    poseidon2_shift_folded_suffix_evals_into::<F, EF>(
                        table_metadata,
                        state,
                        suffix,
                        degree,
                        out,
                    )
                },
                |state, challenge| state.fold(challenge),
            )?;

        let terminal_main_openings =
            terminal_poseidon2_shift_main_claims::<F, EF>(table_index, table, &terminal_row_point)?;
        let terminal_inverse_openings = terminal_column_claims_for_table::<F, EF>(
            table_index,
            &aux_table,
            &terminal_row_point,
            &[P2_SHIFT_SENDER_INV_COL, P2_SHIFT_RECEIVER_INV_COL],
        )?;
        let (local_values, shifted_values, _) = extract_poseidon2_shift_main_values::<F, EF>(
            table_index,
            table_metadata,
            &terminal_row_point,
            &terminal_main_openings,
        )?;
        let (sender_inv, receiver_inv, _) = extract_poseidon2_shift_inverse_values::<F, EF>(
            table_index,
            &aux_table.metadata,
            &terminal_row_point,
            &terminal_inverse_openings,
        )?;
        let terminal_constraint = eval_poseidon2_shift_constraint_from_values::<F, EF>(
            table_metadata,
            &zerocheck_point,
            &terminal_row_point,
            &local_values,
            &shifted_values,
            sender_inv,
            receiver_inv,
            theta,
            alpha,
            beta,
            constraint_challenge,
            sum_challenge,
        )?;
        if terminal_constraint != terminal_claim {
            return Err(WhirNativeCircuitError::ConstraintViolation(
                "Poseidon2 shift prover terminal claim is inconsistent".to_string(),
            ));
        }

        let inverse_points = terminal_inverse_openings
            .iter()
            .map(|claim| Point::new(claim.point.clone()))
            .collect::<Vec<_>>();
        let inverse_opening_proof = open_table_at_points::<F, EF, MT, Challenger, Dft, DIGEST_ELEMS>(
            &pcs,
            inverse_prover_data,
            table_index,
            inverse_points,
            &mut aux_challenger,
        );
        terminal_main_claims.extend(terminal_main_openings.iter().cloned());
        sections.push(WhirNativePoseidon2ShiftSectionProof {
            table_index,
            active_rows: table_metadata.active_rows,
            sender_cumulative,
            receiver_cumulative,
            final_difference: sender_cumulative - receiver_cumulative,
            inverse_commitment,
            degree: P2_SHIFT_LOCAL_DEGREE,
            constraint_challenge,
            sum_challenge,
            zerocheck_point: zerocheck_point.as_slice().to_vec(),
            terminal_row_point: terminal_row_point.as_slice().to_vec(),
            terminal_claim,
            sumcheck,
            terminal_main_openings,
            terminal_inverse_openings,
            inverse_opening_proof,
        });
    }

    Ok((
        WhirNativePoseidon2ShiftBusProof {
            theta,
            alpha,
            beta,
            sections,
        },
        terminal_main_claims,
    ))
}

#[allow(clippy::too_many_arguments)]
fn verify_poseidon2_shift_bus<
    F,
    EF,
    MT,
    Challenger,
    Dft,
    MakePcs,
    MakeChallenger,
    const DIGEST_ELEMS: usize,
>(
    metadata: &[WhirNativeTableMetadata],
    proof: &WhirNativePoseidon2ShiftBusProof<F, EF, MT>,
    public_io_digest: &[F],
    shape_digest: &[F],
    options: WhirNativeCircuitOptions,
    commitment_context: WhirNativeCommitmentContext<'_, MT::Commitment>,
    make_pcs: &MakePcs,
    make_challenger: &MakeChallenger,
) -> Result<Vec<WhirNativeTerminalColumnClaim<EF>>, WhirNativeCircuitError>
where
    F: TwoAdicField + PrimeField64 + Ord + Send + Sync + Serialize + for<'de> Deserialize<'de>,
    EF: ExtensionField<F> + TwoAdicField + Send + Sync + Serialize + for<'de> Deserialize<'de>,
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
{
    let mut challenge_challenger = make_challenger();
    observe_poseidon2_shift_challenge_context(
        &mut challenge_challenger,
        public_io_digest,
        shape_digest,
        options,
        commitment_context.clone(),
    );
    let theta = challenge_challenger.sample_algebra_element();
    let alpha = challenge_challenger.sample_algebra_element();
    let beta = challenge_challenger.sample_algebra_element();
    if proof.theta != theta || proof.alpha != alpha || proof.beta != beta {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "Poseidon2 shift challenge mismatch".to_string(),
        ));
    }
    let expected_table_indices = metadata
        .iter()
        .enumerate()
        .filter_map(|(table_index, table_metadata)| {
            (table_metadata.kind == WhirNativeTableKind::Poseidon2).then_some(table_index)
        })
        .collect::<Vec<_>>();
    if proof.sections.len() != expected_table_indices.len() {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "Poseidon2 shift section count mismatch: expected {}, got {}",
            expected_table_indices.len(),
            proof.sections.len()
        )));
    }
    let mut terminal_main_claims = Vec::new();
    for (section, expected_table_index) in proof.sections.iter().zip(expected_table_indices) {
        if section.table_index != expected_table_index {
            return Err(WhirNativeCircuitError::ConstraintViolation(
                "Poseidon2 shift section table mismatch".to_string(),
            ));
        }
        let table_metadata = metadata.get(section.table_index).ok_or_else(|| {
            WhirNativeCircuitError::ConstraintViolation(format!(
                "missing Poseidon2 shift metadata for table {}",
                section.table_index
            ))
        })?;
        validate_poseidon2_shift_source_metadata(table_metadata)?;
        let claims = verify_poseidon2_shift_section::<
            F,
            EF,
            MT,
            Challenger,
            Dft,
            MakePcs,
            MakeChallenger,
            DIGEST_ELEMS,
        >(
            section,
            table_metadata,
            public_io_digest,
            shape_digest,
            options,
            commitment_context.clone(),
            theta,
            alpha,
            beta,
            make_pcs,
            make_challenger,
        )?;
        terminal_main_claims.extend(claims);
    }
    Ok(terminal_main_claims)
}

#[allow(clippy::too_many_arguments)]
fn verify_poseidon2_shift_section<
    F,
    EF,
    MT,
    Challenger,
    Dft,
    MakePcs,
    MakeChallenger,
    const DIGEST_ELEMS: usize,
>(
    section: &WhirNativePoseidon2ShiftSectionProof<F, EF, MT>,
    table_metadata: &WhirNativeTableMetadata,
    public_io_digest: &[F],
    shape_digest: &[F],
    options: WhirNativeCircuitOptions,
    commitment_context: WhirNativeCommitmentContext<'_, MT::Commitment>,
    theta: EF,
    alpha: EF,
    beta: EF,
    make_pcs: &MakePcs,
    make_challenger: &MakeChallenger,
) -> Result<Vec<WhirNativeTerminalColumnClaim<EF>>, WhirNativeCircuitError>
where
    F: TwoAdicField + PrimeField64 + Ord + Send + Sync + Serialize + for<'de> Deserialize<'de>,
    EF: ExtensionField<F> + TwoAdicField + Send + Sync + Serialize + for<'de> Deserialize<'de>,
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
{
    let expected_aux_metadata = poseidon2_shift_aux_metadata(table_metadata, options);
    if section.active_rows != table_metadata.active_rows
        || section.inverse_commitment.metadata != expected_aux_metadata
        || section.degree != P2_SHIFT_LOCAL_DEGREE
        || section.final_difference != section.sender_cumulative - section.receiver_cumulative
        || section.final_difference != EF::ZERO
    {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "Poseidon2 shift section header mismatch".to_string(),
        ));
    }
    let mut section_challenger = make_challenger();
    observe_poseidon2_shift_constraint_context(
        &mut section_challenger,
        public_io_digest,
        shape_digest,
        options,
        commitment_context,
        section.table_index,
        table_metadata,
        &section.inverse_commitment,
        theta,
        alpha,
        beta,
    );
    let (constraint_challenge, sum_challenge, zerocheck_point) =
        sample_poseidon2_shift_constraint_challenges::<F, EF, Challenger>(
            &mut section_challenger,
            whir_native_table_row_variables(table_metadata),
        );
    if section.constraint_challenge != constraint_challenge
        || section.sum_challenge != sum_challenge
        || section.zerocheck_point.as_slice() != zerocheck_point.as_slice()
    {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "Poseidon2 shift section challenge mismatch".to_string(),
        ));
    }
    let claimed_sum = sum_challenge * (section.sender_cumulative - section.receiver_cumulative);
    let (terminal_row_point, terminal_claim) = verify_sumcheck::<F, EF, Challenger>(
        &section.sumcheck,
        whir_native_table_row_variables(table_metadata),
        claimed_sum,
        &mut section_challenger,
    )?;
    if section.terminal_row_point.as_slice() != terminal_row_point.as_slice()
        || section.terminal_claim != terminal_claim
    {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "Poseidon2 shift sumcheck terminal mismatch".to_string(),
        ));
    }
    let (local_values, shifted_values, main_claims) = extract_poseidon2_shift_main_values::<F, EF>(
        section.table_index,
        table_metadata,
        &terminal_row_point,
        &section.terminal_main_openings,
    )?;
    let (sender_inv, receiver_inv, inverse_claims) = extract_poseidon2_shift_inverse_values::<F, EF>(
        section.table_index,
        &section.inverse_commitment.metadata,
        &terminal_row_point,
        &section.terminal_inverse_openings,
    )?;
    let terminal_constraint = eval_poseidon2_shift_constraint_from_values::<F, EF>(
        table_metadata,
        &zerocheck_point,
        &terminal_row_point,
        &local_values,
        &shifted_values,
        sender_inv,
        receiver_inv,
        theta,
        alpha,
        beta,
        constraint_challenge,
        sum_challenge,
    )?;
    if terminal_constraint != terminal_claim {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "Poseidon2 shift terminal claim is inconsistent".to_string(),
        ));
    }

    let pcs = make_pcs(section.inverse_commitment.metadata.num_variables);
    let mut aux_challenger = make_challenger();
    observe_poseidon2_shift_aux_table_context(
        &mut aux_challenger,
        public_io_digest,
        shape_digest,
        options,
        section.table_index,
        table_metadata,
        &section.inverse_commitment.metadata,
        theta,
        alpha,
        beta,
    );
    verify_table_opening_claims_after_context::<F, EF, MT, Challenger, Dft, DIGEST_ELEMS>(
        &pcs,
        &section.inverse_commitment,
        &section.inverse_opening_proof,
        section.table_index,
        &inverse_claims,
        &mut aux_challenger,
    )?;
    Ok(main_claims)
}

fn terminal_poseidon2_shift_main_claims<F, EF>(
    table_index: usize,
    table: &WhirNativeTableData<EF>,
    row_point: &Point<EF>,
) -> Result<Vec<WhirNativeTerminalColumnClaim<EF>>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let local_columns = poseidon2_main_columns();
    let shifted_columns = poseidon2_shifted_columns();
    let mut claims =
        terminal_column_claims_for_table::<F, EF>(table_index, table, row_point, &local_columns)?;
    claims.extend(terminal_column_claims_for_table::<F, EF>(
        table_index,
        table,
        row_point,
        &shifted_columns,
    )?);
    Ok(claims)
}

fn extract_poseidon2_shift_main_values<F, EF>(
    table_index: usize,
    metadata: &WhirNativeTableMetadata,
    terminal_row_point: &Point<EF>,
    claims: &[WhirNativeTerminalColumnClaim<EF>],
) -> Result<(Vec<EF>, Vec<EF>, Vec<WhirNativeTerminalColumnClaim<EF>>), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let local_columns = poseidon2_main_columns();
    let shifted_columns = poseidon2_shifted_columns();
    let expected = local_columns.len() + shifted_columns.len();
    if claims.len() != expected {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "Poseidon2 shift main opening count mismatch: expected {expected}, got {}",
            claims.len()
        )));
    }
    let mut local_values = EF::zero_vec(P2_BB_D4_WIDTH16_AIR_WIDTH);
    let mut shifted_values = EF::zero_vec(P2_BB_D4_WIDTH16_AIR_WIDTH);
    for (opening_index, (&column, claim)) in local_columns.iter().zip(claims).enumerate() {
        validate_terminal_claim(
            claim,
            table_index,
            metadata,
            terminal_row_point,
            column,
            opening_index,
            "Poseidon2 shift local",
        )?;
        local_values[column] = claim.value;
    }
    for (offset, (&column, claim)) in shifted_columns
        .iter()
        .zip(&claims[local_columns.len()..])
        .enumerate()
    {
        let opening_index = local_columns.len() + offset;
        validate_terminal_claim(
            claim,
            table_index,
            metadata,
            terminal_row_point,
            column,
            opening_index,
            "Poseidon2 shift shifted",
        )?;
        shifted_values[column - P2_BB_D4_WIDTH16_SHIFTED_OFFSET] = claim.value;
    }
    Ok((local_values, shifted_values, claims.to_vec()))
}

fn extract_poseidon2_shift_inverse_values<F, EF>(
    table_index: usize,
    metadata: &WhirNativeTableMetadata,
    terminal_row_point: &Point<EF>,
    claims: &[WhirNativeTerminalColumnClaim<EF>],
) -> Result<(EF, EF, Vec<(Point<EF>, EF)>), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if claims.len() != P2_SHIFT_AUX_WIDTH {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "Poseidon2 shift inverse opening count mismatch: expected {P2_SHIFT_AUX_WIDTH}, got {}",
            claims.len()
        )));
    }
    let mut values = [EF::ZERO; P2_SHIFT_AUX_WIDTH];
    let mut opening_claims = Vec::with_capacity(P2_SHIFT_AUX_WIDTH);
    for (opening_index, claim) in claims.iter().enumerate() {
        validate_terminal_claim(
            claim,
            table_index,
            metadata,
            terminal_row_point,
            opening_index,
            opening_index,
            "Poseidon2 shift inverse",
        )?;
        let point =
            whir_native_table_column_point::<F, EF>(metadata, terminal_row_point, opening_index)?;
        values[opening_index] = claim.value;
        opening_claims.push((point, claim.value));
    }
    Ok((
        values[P2_SHIFT_SENDER_INV_COL],
        values[P2_SHIFT_RECEIVER_INV_COL],
        opening_claims,
    ))
}

fn validate_terminal_claim<F, EF>(
    claim: &WhirNativeTerminalColumnClaim<EF>,
    table_index: usize,
    metadata: &WhirNativeTableMetadata,
    row_point: &Point<EF>,
    column: usize,
    opening_index: usize,
    label: &str,
) -> Result<(), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if claim.table_index != table_index || claim.column != column {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "{label} terminal opening {opening_index} metadata mismatch"
        )));
    }
    let expected_point = whir_native_table_column_point::<F, EF>(metadata, row_point, column)?;
    if claim.point.as_slice() != expected_point.as_slice() {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "{label} terminal opening {opening_index} point mismatch"
        )));
    }
    Ok(())
}

fn extract_terminal_poseidon2_transition_values<F, EF>(
    proof: &WhirNativeLocalConstraintProof<EF>,
    metadata: &WhirNativeTableMetadata,
    terminal_row_point: &Point<EF>,
    local_columns: &[usize],
    shifted_columns: &[usize],
) -> Result<(Vec<EF>, Vec<EF>, Vec<(Point<EF>, EF)>), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let mut local_values = EF::zero_vec(P2_BB_D4_WIDTH16_AIR_WIDTH);
    let mut shifted_values = EF::zero_vec(P2_BB_D4_WIDTH16_AIR_WIDTH);
    let mut claims = Vec::with_capacity(local_columns.len() + shifted_columns.len());

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

    Ok((local_values, shifted_values, claims))
}

#[allow(dead_code)]
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

fn extract_terminal_poseidon2_read_values<F, EF>(
    proof: &WhirNativeLocalConstraintProof<EF>,
    opening_offset: usize,
    source_row_point: &Point<EF>,
    metadata: &WhirNativeTableMetadata,
    read_columns: &[usize],
) -> Result<(Vec<EF>, Vec<(Point<EF>, EF)>), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let mut values = Vec::with_capacity(read_columns.len());
    let mut claims = Vec::with_capacity(read_columns.len());
    for (offset, &column) in read_columns.iter().enumerate() {
        let opening_index = opening_offset + offset;
        let claim = proof.terminal_openings.get(opening_index).ok_or_else(|| {
            WhirNativeCircuitError::ConstraintViolation(format!(
                "missing Poseidon2 read terminal opening {opening_index}"
            ))
        })?;
        if claim.table_index != proof.table_index || claim.column != column {
            return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                "Poseidon2 read terminal opening {opening_index} metadata mismatch"
            )));
        }
        let expected_point =
            whir_native_table_column_point::<F, EF>(metadata, source_row_point, column)?;
        if claim.point.as_slice() != expected_point.as_slice() {
            return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                "Poseidon2 read terminal opening {opening_index} point mismatch"
            )));
        }
        values.push(claim.value);
        claims.push((expected_point, claim.value));
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
    let column_values = row_major_column_values(values, width, column)?;
    Ok(Poly::eval_ext_slice::<F>(&column_values, row_point))
}

fn row_major_column_values<EF>(
    values: &[EF],
    width: usize,
    column: usize,
) -> Result<Vec<EF>, WhirNativeCircuitError>
where
    EF: Field,
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
    Ok(values
        .chunks_exact(width)
        .map(|row| row[column])
        .collect::<Vec<_>>())
}

#[allow(dead_code)]
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
    if metadata.width != KNOWN_ROWS_WIDTH {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "known-row table width mismatch: expected {KNOWN_ROWS_WIDTH}, got {}",
            metadata.width
        )));
    }
    for (row_index, row) in expected_rows.iter().enumerate() {
        if row.len() != KNOWN_ROWS_EXPECTED_WIDTH {
            return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                "known-row {row_index} width mismatch: expected {}, got {}",
                KNOWN_ROWS_EXPECTED_WIDTH,
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

fn known_rows_local_degree(_witness_metadata: &WhirNativeTableMetadata) -> usize {
    3
}

fn alu_local_degree(_witness_metadata: &WhirNativeTableMetadata) -> usize {
    ALU_LOCAL_DEGREE
}

fn recompose_local_degree(_witness_metadata: &WhirNativeTableMetadata) -> usize {
    3
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

fn observe_table_metadata<F, Challenger>(
    challenger: &mut Challenger,
    table_index: usize,
    metadata: &WhirNativeTableMetadata,
) where
    F: Field,
    Challenger: CanObserve<F>,
{
    challenger.observe(F::from_u64(table_index as u64));
    challenger.observe(F::from_u64(metadata.kind.tag()));
    observe_string::<F, Challenger>(challenger, &metadata.op_type);
    challenger.observe(F::from_u64(metadata.width as u64));
    challenger.observe(F::from_u64(metadata.padded_width as u64));
    challenger.observe(F::from_u64(metadata.active_rows as u64));
    challenger.observe(F::from_u64(metadata.padded_height as u64));
    challenger.observe(F::from_u64(metadata.num_variables as u64));
    challenger.observe(F::from_u64(metadata.column_layout_version as u64));
}

fn observe_column_ref<F, Challenger>(challenger: &mut Challenger, column_ref: &WhirNativeColumnRef)
where
    F: Field,
    Challenger: CanObserve<F>,
{
    challenger.observe(F::from_u64(column_ref.table_index as u64));
    challenger.observe(F::from_u64(column_ref.column as u64));
}

fn observe_circuit_constraint_context<F, Comm, Challenger>(
    challenger: &mut Challenger,
    public_io_digest: &[F],
    shape_digest: &[F],
    options: WhirNativeCircuitOptions,
    commitment_context: WhirNativeCommitmentContext<'_, Comm>,
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
    match commitment_context {
        WhirNativeCommitmentContext::PerTable(table_commitments) => {
            challenger.observe(F::from_u64(WhirNativeOpeningMode::PerTable.tag()));
            challenger.observe(F::from_u64(table_commitments.len() as u64));
            for (table_index, table_commitment) in table_commitments.iter().enumerate() {
                observe_table_metadata::<F, Challenger>(
                    challenger,
                    table_index,
                    &table_commitment.metadata,
                );
                challenger.observe(table_commitment.commitment.clone());
            }
        }
        WhirNativeCommitmentContext::ColumnBatched {
            metadata,
            column_batches,
        } => {
            challenger.observe(F::from_u64(WhirNativeOpeningMode::ColumnBatched.tag()));
            challenger.observe(F::from_u64(metadata.len() as u64));
            for (table_index, metadata) in metadata.iter().enumerate() {
                observe_table_metadata::<F, Challenger>(challenger, table_index, metadata);
            }
            challenger.observe(F::from_u64(column_batches.len() as u64));
            for (batch_index, batch) in column_batches.iter().enumerate() {
                challenger.observe(F::from_u64(batch_index as u64));
                challenger.observe(F::from_u64(batch.num_variables as u64));
                challenger.observe(F::from_u64(batch.columns.len() as u64));
                for column_ref in &batch.columns {
                    observe_column_ref::<F, Challenger>(challenger, column_ref);
                }
                challenger.observe(batch.commitment.clone());
            }
        }
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

fn sample_poseidon2_shift_constraint_challenges<F, EF, Challenger>(
    challenger: &mut Challenger,
    row_variables: usize,
) -> (EF, EF, Point<EF>)
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
{
    let constraint_challenge = challenger.sample_algebra_element();
    let sum_challenge = challenger.sample_algebra_element();
    let row_point = (0..row_variables)
        .map(|_| challenger.sample_algebra_element())
        .collect();
    (constraint_challenge, sum_challenge, Point::new(row_point))
}

fn eval_known_rows_constraint_from_table<F, EF>(
    table: &WhirNativeTableData<EF>,
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
    eval_known_rows_constraint_from_values::<F, EF>(
        &table.metadata,
        expected_rows,
        expected_witness_ids,
        row_point,
        &values,
        alpha,
    )
}

fn eval_known_rows_constraint_from_values<F, EF>(
    metadata: &WhirNativeTableMetadata,
    expected_rows: &[Vec<EF>],
    expected_witness_ids: &[u32],
    row_point: &Point<EF>,
    values: &[EF],
    alpha: EF,
) -> Result<EF, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if values.len() != metadata.width || metadata.width != KNOWN_ROWS_WIDTH {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "known-row terminal width mismatch: expected {KNOWN_ROWS_WIDTH}, got {}",
            values.len()
        )));
    }
    let active =
        active_selector_eval::<F, EF>(metadata.active_rows, metadata.padded_height, row_point)?;
    let inactive = EF::ONE - active;
    let mut constraints = Vec::with_capacity(metadata.width + KNOWN_ROWS_EXPECTED_WIDTH + 2);
    for &value in values {
        constraints.push(inactive * value);
    }
    for (column, &value) in values.iter().take(KNOWN_ROWS_EXPECTED_WIDTH).enumerate() {
        let expected = known_rows_column_eval::<F, EF>(
            metadata.padded_height,
            expected_rows,
            column,
            row_point,
        )?;
        constraints.push(active * (value - expected));
    }
    let source_value = values[KNOWN_ROW_VALUE_COL];
    let read_value = values[KNOWN_ROW_READ_VALUE_COL];
    let witness_id =
        static_u32_column_eval::<F, EF>(metadata.padded_height, expected_witness_ids, row_point)?;
    constraints.push(active * (values[KNOWN_ROW_WITNESS_ID_COL] - witness_id));
    constraints.push(active * (source_value - read_value));
    Ok(batch_constraints(constraints, alpha))
}

fn eval_alu_constraint_from_table<F, EF>(
    table: &WhirNativeTableData<EF>,
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
    eval_alu_constraint_from_values::<F, EF>(
        &table.metadata,
        expected_rows,
        row_point,
        &values,
        alpha,
    )
}

fn eval_alu_constraint_from_values<F, EF>(
    metadata: &WhirNativeTableMetadata,
    expected_rows: &[WhirNativeExpectedAluRow],
    row_point: &Point<EF>,
    values: &[EF],
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

    let a = values[ALU_READ_A_COL];
    let b = values[ALU_READ_B_COL];
    let c = values[ALU_READ_C_COL];
    let out = values[ALU_READ_OUT_COL];
    let acc = values[ALU_READ_ACC_COL];
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
    constraints.push(sel_horner * (acc * b + c - a - out));
    constraints.push((EF::ONE - sel_horner) * acc);

    Ok(batch_constraints(constraints, alpha))
}

fn eval_recompose_constraint_from_table<F, EF>(
    table: &WhirNativeTableData<EF>,
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
    eval_recompose_constraint_from_values::<F, EF>(
        &table.metadata,
        expected_rows,
        row_point,
        &values,
        alpha,
    )
}

fn eval_recompose_constraint_from_values<F, EF>(
    metadata: &WhirNativeTableMetadata,
    expected_rows: &[WhirNativeExpectedRecomposeRow],
    row_point: &Point<EF>,
    values: &[EF],
    alpha: EF,
) -> Result<EF, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let d = recompose_degree(expected_rows)?;
    let expected_width = 4 + d + d;
    if values.len() != expected_width {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "recompose terminal width mismatch: expected {expected_width}, got {}",
            values.len()
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
    let mut recomposed = EF::ZERO;
    for coeff in 0..d {
        let basis = EF::ith_basis_element(coeff).ok_or_else(|| {
            WhirNativeCircuitError::ConstraintViolation(format!(
                "missing extension basis element {coeff}"
            ))
        })?;
        recomposed += values[value_start + coeff] * basis;
    }
    constraints.push(active * (recomposed - values[value_start + d]));

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
    let evals = active_selector_values::<EF>(active_rows, padded_height)?;
    Ok(Poly::eval_ext_slice::<F>(&evals, row_point))
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
    Ok(Poly::eval_ext_slice::<F>(&evals, row_point))
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
    let evals = row_index_values::<F, EF>(padded_height);
    Ok(Poly::eval_ext_slice::<F>(&evals, row_point))
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
    let evals = static_u32_column_values::<F, EF>(padded_height, values)?;
    Ok(Poly::eval_ext_slice::<F>(&evals, row_point))
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
    let evals = expected_alu_column_values::<F, EF>(padded_height, expected_rows, column)?;
    Ok(Poly::eval_ext_slice::<F>(&evals, row_point))
}

fn expected_alu_column_values<F, EF>(
    padded_height: usize,
    expected_rows: &[WhirNativeExpectedAluRow],
    column: usize,
) -> Result<Vec<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
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
    Ok(evals)
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
    let evals = expected_alu_selector_values::<F, EF>(padded_height, expected_rows, kind)?;
    Ok(Poly::eval_ext_slice::<F>(&evals, row_point))
}

fn expected_alu_selector_values<F, EF>(
    padded_height: usize,
    expected_rows: &[WhirNativeExpectedAluRow],
    kind: AluOpKind,
) -> Result<Vec<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
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
    Ok(evals)
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
    Ok(Poly::eval_ext_slice::<F>(&evals, row_point))
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

#[allow(dead_code)]
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
    let address = whir_native_witness_address_point::<F, EF>(
        &witness_table.metadata,
        source_padded_height,
        source_row_point,
        witness_ids_by_source_row,
    )?;
    let witness_value_column = table_column_values(witness_table, 1)?;
    let value = Poly::eval_ext_slice::<F>(&witness_value_column, &address);
    Ok(WhirNativeTerminalColumnClaim {
        table_index: witness_table_index,
        column: 1,
        point: point.as_slice().to_vec(),
        value,
    })
}

#[allow(dead_code)]
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
    let witness_row_vars = whir_native_table_row_variables(&witness_table.metadata);
    let address = Point::new(point.as_slice()[..witness_row_vars].to_vec());
    let witness_value_column = table_column_values(witness_table, 1)?;
    Ok(Poly::eval_ext_slice::<F>(&witness_value_column, &address))
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

#[allow(dead_code)]
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

#[allow(dead_code)]
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

#[allow(dead_code)]
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

fn add_eval_constraint_to_linear_sigma_statement<F, EF>(
    statement: &mut LinearSigmaStatement<EF>,
    point: Point<EF>,
    value: EF,
) where
    F: Field,
    EF: ExtensionField<F>,
{
    let mut eq = EqStatement::initialize(statement.num_variables());
    eq.add_evaluated_constraint(point, value);
    statement.add_constraint(LinearSigmaConstraint::from_eq_statement::<F>(&eq, EF::ONE));
}

fn build_column_batch_statements<F, EF>(
    layout: &WhirNativeColumnBatchLayout,
    metadata: &[WhirNativeTableMetadata],
    random_points: &[Point<EF>],
    random_values: &[Vec<EF>],
    terminal_claims: &[(usize, WhirNativeTerminalColumnClaim<EF>)],
) -> Result<Vec<LinearSigmaStatement<EF>>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if random_points.len() != random_values.len() {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "column batch random opening count mismatch: {} points, {} value rows",
            random_points.len(),
            random_values.len()
        )));
    }
    let mut statements = (0..layout.columns.len())
        .map(|_| LinearSigmaStatement::initialize(layout.num_variables))
        .collect::<Vec<_>>();

    for (random_index, (point, values)) in random_points.iter().zip(random_values).enumerate() {
        if point.num_variables() != layout.num_variables {
            return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                "column batch random point {random_index} arity mismatch"
            )));
        }
        if values.len() != layout.columns.len() {
            return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                "column batch random value row {random_index} width mismatch: expected {}, got {}",
                layout.columns.len(),
                values.len()
            )));
        }
        for (statement, &value) in statements.iter_mut().zip(values) {
            add_eval_constraint_to_linear_sigma_statement::<F, EF>(statement, point.clone(), value);
        }
    }

    for (column_offset, claim) in terminal_claims {
        let Some(column_ref) = layout.columns.get(*column_offset) else {
            return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                "terminal claim column offset {column_offset} out of range"
            )));
        };
        if column_ref.table_index != claim.table_index || column_ref.column != claim.column {
            return Err(WhirNativeCircuitError::ConstraintViolation(
                "terminal claim column mapping mismatch".to_string(),
            ));
        }
        let Some(claim_metadata) = metadata.get(claim.table_index) else {
            return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                "terminal claim table {} out of range",
                claim.table_index
            )));
        };
        let row_point = row_point_from_terminal_column_claim::<F, EF>(
            claim_metadata,
            claim,
            layout.num_variables,
        )?;
        add_eval_constraint_to_linear_sigma_statement::<F, EF>(
            &mut statements[*column_offset],
            row_point,
            claim.value,
        );
    }

    Ok(statements)
}

#[allow(dead_code)]
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

#[allow(dead_code)]
fn recompose_input_witness_ids(
    expected_rows: &[WhirNativeExpectedRecomposeRow],
    port: usize,
) -> Vec<u32> {
    expected_rows
        .iter()
        .map(|row| row.input_wids.get(port).copied().unwrap_or(0))
        .collect()
}

#[allow(dead_code)]
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

#[derive(Clone, Debug)]
struct ReadBusSectionSkeleton {
    table_index: usize,
    kind: WhirNativeReadBusSectionKind,
    port: u32,
    active_reads: usize,
}

fn observe_read_bus_challenge_context<F, Comm, Challenger>(
    challenger: &mut Challenger,
    public_io_digest: &[F],
    shape_digest: &[F],
    options: WhirNativeCircuitOptions,
    commitment_context: WhirNativeCommitmentContext<'_, Comm>,
) where
    F: Field,
    Comm: Clone,
    Challenger: CanObserve<F> + CanObserve<Comm>,
{
    challenger.observe(F::from_u64(0x5748_4952_5242_5553));
    observe_circuit_constraint_context(
        challenger,
        public_io_digest,
        shape_digest,
        options,
        commitment_context,
    );
}

fn prove_read_bus<F, EF>(
    circuit: &Circuit<EF>,
    tables: &[WhirNativeTableData<EF>],
    metadata: &[WhirNativeTableMetadata],
    alpha: EF,
    beta: EF,
) -> Result<WhirNativeReadBusProof<EF>, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F> + ExtensionField<BabyBear>,
{
    let (witness_read_counts, skeletons) =
        build_expected_read_bus_plan::<F, EF>(circuit, metadata)?;
    let witness_table = tables.first().ok_or_else(|| {
        WhirNativeCircuitError::ConstraintViolation("missing witness table".to_string())
    })?;
    validate_witness_metadata(&witness_table.metadata)?;

    let mut sender_cumulative = EF::ZERO;
    let mut sender_active_reads = 0usize;
    for (wid, &count) in witness_read_counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        if wid >= witness_table.metadata.active_rows {
            return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                "read count for WitnessId({wid}) exceeds witness table"
            )));
        }
        let value = witness_table.values[wid * witness_table.metadata.padded_width + 1];
        sender_cumulative += read_bus_contribution::<F, EF>(alpha, beta, wid as u32, value, count)?;
        sender_active_reads += count as usize;
    }

    let mut sections = Vec::with_capacity(skeletons.len());
    sections.push(WhirNativeReadBusSectionProof {
        table_index: WITNESS_TABLE_INDEX,
        kind: WhirNativeReadBusSectionKind::WitnessSender,
        port: 0,
        active_reads: sender_active_reads,
        cumulative: sender_cumulative,
    });

    let mut receiver_cumulative = EF::ZERO;
    for skeleton in skeletons {
        let cumulative =
            prove_read_bus_receiver_section::<F, EF>(circuit, tables, &skeleton, alpha, beta)?;
        receiver_cumulative += cumulative;
        sections.push(WhirNativeReadBusSectionProof {
            table_index: skeleton.table_index,
            kind: skeleton.kind,
            port: skeleton.port,
            active_reads: skeleton.active_reads,
            cumulative,
        });
    }
    let final_difference = sender_cumulative - receiver_cumulative;
    if final_difference != EF::ZERO {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "read bus cumulative mismatch".to_string(),
        ));
    }

    Ok(WhirNativeReadBusProof {
        alpha,
        beta,
        witness_read_counts,
        sender_cumulative,
        receiver_cumulative,
        final_difference,
        sections,
    })
}

fn verify_read_bus_proof<F, EF>(
    circuit: &Circuit<EF>,
    metadata: &[WhirNativeTableMetadata],
    proof: &WhirNativeReadBusProof<EF>,
    alpha: EF,
    beta: EF,
) -> Result<(), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F> + ExtensionField<BabyBear>,
{
    if proof.alpha != alpha || proof.beta != beta {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "read bus challenge mismatch".to_string(),
        ));
    }
    let (expected_counts, skeletons) = build_expected_read_bus_plan::<F, EF>(circuit, metadata)?;
    if proof.witness_read_counts != expected_counts {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "read bus witness read counts mismatch".to_string(),
        ));
    }
    if proof.sections.len() != skeletons.len() + 1 {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "read bus section count mismatch: expected {}, got {}",
            skeletons.len() + 1,
            proof.sections.len()
        )));
    }
    let sender_section = &proof.sections[0];
    let expected_sender_reads = expected_counts.iter().map(|&count| count as usize).sum();
    if sender_section.table_index != WITNESS_TABLE_INDEX
        || sender_section.kind != WhirNativeReadBusSectionKind::WitnessSender
        || sender_section.port != 0
        || sender_section.active_reads != expected_sender_reads
        || sender_section.cumulative != proof.sender_cumulative
    {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "read bus sender section mismatch".to_string(),
        ));
    }
    let mut receiver_cumulative = EF::ZERO;
    for (section, skeleton) in proof.sections.iter().skip(1).zip(&skeletons) {
        if section.table_index != skeleton.table_index
            || section.kind != skeleton.kind
            || section.port != skeleton.port
            || section.active_reads != skeleton.active_reads
        {
            return Err(WhirNativeCircuitError::ConstraintViolation(
                "read bus receiver section mismatch".to_string(),
            ));
        }
        receiver_cumulative += section.cumulative;
    }
    if receiver_cumulative != proof.receiver_cumulative {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "read bus receiver cumulative mismatch".to_string(),
        ));
    }
    if proof.final_difference != proof.sender_cumulative - proof.receiver_cumulative
        || proof.final_difference != EF::ZERO
    {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "read bus final cumulative mismatch".to_string(),
        ));
    }
    Ok(())
}

fn read_bus_contribution<F, EF>(
    alpha: EF,
    beta: EF,
    witness_id: u32,
    value: EF,
    multiplicity: u32,
) -> Result<EF, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if multiplicity == 0 {
        return Ok(EF::ZERO);
    }
    let key = ef_from_u64::<F, EF>(witness_id as u64) + beta * value;
    let denominator = alpha - key;
    let inverse = denominator.try_inverse().ok_or_else(|| {
        WhirNativeCircuitError::ConstraintViolation(
            "read bus challenge collision with witness key".to_string(),
        )
    })?;
    Ok(ef_from_u64::<F, EF>(multiplicity as u64) * inverse)
}

fn build_expected_read_bus_plan<F, EF>(
    circuit: &Circuit<EF>,
    metadata: &[WhirNativeTableMetadata],
) -> Result<(Vec<u32>, Vec<ReadBusSectionSkeleton>), WhirNativeCircuitError>
where
    F: Field,
    EF: Field,
{
    let mut counts = vec![0u32; circuit.witness_count as usize];
    let mut sections = Vec::new();
    for (table_index, table_metadata) in metadata.iter().enumerate() {
        match table_metadata.kind {
            WhirNativeTableKind::Witness => {}
            WhirNativeTableKind::Const => {
                let wids = expected_const_witness_ids_from_circuit(circuit);
                for &wid in &wids {
                    increment_witness_read_count(&mut counts, wid)?;
                }
                sections.push(ReadBusSectionSkeleton {
                    table_index,
                    kind: WhirNativeReadBusSectionKind::KnownRows,
                    port: 0,
                    active_reads: wids.len(),
                });
            }
            WhirNativeTableKind::Public => {
                let wids = expected_public_witness_ids_from_circuit(circuit);
                for &wid in &wids {
                    increment_witness_read_count(&mut counts, wid)?;
                }
                sections.push(ReadBusSectionSkeleton {
                    table_index,
                    kind: WhirNativeReadBusSectionKind::KnownRows,
                    port: 0,
                    active_reads: wids.len(),
                });
            }
            WhirNativeTableKind::Alu => {
                let rows = expected_alu_rows_from_circuit(circuit);
                for port in 0..=ALU_READ_ACC_COL - ALU_READ_A_COL {
                    let mut active_reads = 0usize;
                    for row in &rows {
                        if let Some(wid) = alu_active_read_witness_id(row, port) {
                            increment_witness_read_count(&mut counts, wid)?;
                            active_reads += 1;
                        }
                    }
                    sections.push(ReadBusSectionSkeleton {
                        table_index,
                        kind: WhirNativeReadBusSectionKind::Alu,
                        port: port as u32,
                        active_reads,
                    });
                }
            }
            WhirNativeTableKind::Recompose => {
                let rows = expected_recompose_rows_from_circuit(circuit, &table_metadata.op_type)?;
                let d = recompose_degree(&rows)?;
                for port in 0..=d {
                    let mut active_reads = 0usize;
                    for row in &rows {
                        let wid = if port < d {
                            row.input_wids[port]
                        } else {
                            row.output_wid
                        };
                        increment_witness_read_count(&mut counts, wid)?;
                        active_reads += 1;
                    }
                    sections.push(ReadBusSectionSkeleton {
                        table_index,
                        kind: WhirNativeReadBusSectionKind::Recompose,
                        port: port as u32,
                        active_reads,
                    });
                }
            }
            WhirNativeTableKind::Poseidon2 => {
                let rows = expected_poseidon2_rows_from_circuit(circuit, &table_metadata.op_type)?;
                let direction_bit_witness_ids =
                    expected_poseidon2_direction_bit_witness_ids_from_circuit(
                        circuit,
                        &table_metadata.op_type,
                    )?;
                for port in 0..P2_BB_D4_WIDTH16_WITNESS_PORTS {
                    let mut active_reads = 0usize;
                    for (row_index, row) in rows.iter().enumerate() {
                        if let Some(wid) = poseidon2_active_read_witness_id(
                            row,
                            &direction_bit_witness_ids,
                            row_index,
                            port,
                        ) {
                            increment_witness_read_count(&mut counts, wid)?;
                            active_reads += 1;
                        }
                    }
                    sections.push(ReadBusSectionSkeleton {
                        table_index,
                        kind: WhirNativeReadBusSectionKind::Poseidon2,
                        port: port as u32,
                        active_reads,
                    });
                }
            }
            WhirNativeTableKind::Poseidon2Shift => {}
        }
    }
    Ok((counts, sections))
}

fn increment_witness_read_count(
    counts: &mut [u32],
    witness_id: u32,
) -> Result<(), WhirNativeCircuitError> {
    if witness_id == MISSING_WITNESS_ID {
        return Ok(());
    }
    let count = counts.get_mut(witness_id as usize).ok_or_else(|| {
        WhirNativeCircuitError::ConstraintViolation(format!(
            "read bus WitnessId({witness_id}) out of range"
        ))
    })?;
    *count = count.checked_add(1).ok_or_else(|| {
        WhirNativeCircuitError::ConstraintViolation(
            "read bus witness read count overflow".to_string(),
        )
    })?;
    Ok(())
}

fn alu_active_read_witness_id(row: &WhirNativeExpectedAluRow, port: usize) -> Option<u32> {
    match port {
        0 => Some(row.indices[0]),
        1 => Some(row.indices[1]),
        2 if matches!(row.kind, AluOpKind::MulAdd | AluOpKind::HornerAcc) => Some(row.indices[2]),
        3 => Some(row.indices[3]),
        4 if row.kind == AluOpKind::HornerAcc => row.acc_index,
        _ => None,
    }
}

fn alu_read_column_for_port(port: usize) -> Result<usize, WhirNativeCircuitError> {
    match port {
        0 => Ok(ALU_READ_A_COL),
        1 => Ok(ALU_READ_B_COL),
        2 => Ok(ALU_READ_C_COL),
        3 => Ok(ALU_READ_OUT_COL),
        4 => Ok(ALU_READ_ACC_COL),
        _ => Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "bad ALU read port {port}"
        ))),
    }
}

fn poseidon2_active_read_witness_id(
    row: &Poseidon2CircuitRow<BabyBear>,
    direction_bit_witness_ids: &[u32],
    row_index: usize,
    port: usize,
) -> Option<u32> {
    if port < P2_BB_D4_WIDTH16_WIDTH_EXT {
        return (row.in_ctl[port] && !row.merkle_path).then_some(row.input_indices[port]);
    }
    let output_start = P2_BB_D4_WIDTH16_WIDTH_EXT;
    if port < output_start + P2_BB_D4_WIDTH16_RATE_EXT {
        let limb = port - output_start;
        return row.out_ctl[limb].then_some(row.output_indices[limb]);
    }
    let mmcs_port = P2_BB_D4_WIDTH16_WIDTH_EXT + P2_BB_D4_WIDTH16_RATE_EXT;
    if port == mmcs_port {
        return row.mmcs_ctl_enabled.then_some(row.mmcs_index_sum_idx);
    }
    if port == mmcs_port + 1 {
        return row
            .merkle_path
            .then(|| direction_bit_witness_ids[row_index]);
    }
    None
}

fn prove_read_bus_receiver_section<F, EF>(
    circuit: &Circuit<EF>,
    tables: &[WhirNativeTableData<EF>],
    skeleton: &ReadBusSectionSkeleton,
    alpha: EF,
    beta: EF,
) -> Result<EF, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F> + ExtensionField<BabyBear>,
{
    let table = tables.get(skeleton.table_index).ok_or_else(|| {
        WhirNativeCircuitError::ConstraintViolation(format!(
            "read bus table {} out of range",
            skeleton.table_index
        ))
    })?;
    match skeleton.kind {
        WhirNativeReadBusSectionKind::WitnessSender => Ok(EF::ZERO),
        WhirNativeReadBusSectionKind::KnownRows => {
            let witness_ids = match table.metadata.kind {
                WhirNativeTableKind::Const => expected_const_witness_ids_from_circuit(circuit),
                WhirNativeTableKind::Public => expected_public_witness_ids_from_circuit(circuit),
                _ => {
                    return Err(WhirNativeCircuitError::ConstraintViolation(
                        "known-row read section mapped to wrong table".to_string(),
                    ));
                }
            };
            read_bus_cumulative_for_rows::<F, EF>(
                table,
                &witness_ids,
                KNOWN_ROW_READ_VALUE_COL,
                alpha,
                beta,
            )
        }
        WhirNativeReadBusSectionKind::Alu => {
            let rows = expected_alu_rows_from_circuit(circuit);
            let port = skeleton.port as usize;
            let column = alu_read_column_for_port(port)?;
            let witness_ids = rows
                .iter()
                .map(|row| alu_active_read_witness_id(row, port).unwrap_or(MISSING_WITNESS_ID))
                .collect::<Vec<_>>();
            read_bus_cumulative_for_rows::<F, EF>(table, &witness_ids, column, alpha, beta)
        }
        WhirNativeReadBusSectionKind::Recompose => {
            let rows = expected_recompose_rows_from_circuit(circuit, &table.metadata.op_type)?;
            let d = recompose_degree(&rows)?;
            let port = skeleton.port as usize;
            let value_start = 3 + d;
            let column = if port < d {
                value_start + port
            } else if port == d {
                value_start + d
            } else {
                return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                    "bad recompose read port {port}"
                )));
            };
            let witness_ids = rows
                .iter()
                .map(|row| {
                    if port < d {
                        row.input_wids[port]
                    } else {
                        row.output_wid
                    }
                })
                .collect::<Vec<_>>();
            read_bus_cumulative_for_rows::<F, EF>(table, &witness_ids, column, alpha, beta)
        }
        WhirNativeReadBusSectionKind::Poseidon2 => {
            let rows = expected_poseidon2_rows_from_circuit(circuit, &table.metadata.op_type)?;
            let direction_bit_witness_ids =
                expected_poseidon2_direction_bit_witness_ids_from_circuit(
                    circuit,
                    &table.metadata.op_type,
                )?;
            let port = skeleton.port as usize;
            let column = P2_BB_D4_WIDTH16_READ_OFFSET + port;
            let witness_ids = rows
                .iter()
                .enumerate()
                .map(|(row_index, row)| {
                    poseidon2_active_read_witness_id(
                        row,
                        &direction_bit_witness_ids,
                        row_index,
                        port,
                    )
                    .unwrap_or(MISSING_WITNESS_ID)
                })
                .collect::<Vec<_>>();
            read_bus_cumulative_for_rows::<F, EF>(table, &witness_ids, column, alpha, beta)
        }
    }
}

fn read_bus_cumulative_for_rows<F, EF>(
    table: &WhirNativeTableData<EF>,
    witness_ids: &[u32],
    value_column: usize,
    alpha: EF,
    beta: EF,
) -> Result<EF, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if witness_ids.len() > table.metadata.active_rows {
        return Err(WhirNativeCircuitError::ConstraintViolation(
            "read bus section has more witness ids than active rows".to_string(),
        ));
    }
    if value_column >= table.metadata.width {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "read bus value column {value_column} out of width {}",
            table.metadata.width
        )));
    }
    let mut cumulative = EF::ZERO;
    for (row, &wid) in witness_ids.iter().enumerate() {
        if wid == MISSING_WITNESS_ID {
            continue;
        }
        let value = table.values[row * table.metadata.padded_width + value_column];
        cumulative += read_bus_contribution::<F, EF>(alpha, beta, wid, value, 1)?;
    }
    Ok(cumulative)
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

#[allow(clippy::too_many_arguments)]
fn verify_column_batch_opening<
    F,
    EF,
    MT,
    Challenger,
    Dft,
    MakeChallenger,
    const DIGEST_ELEMS: usize,
>(
    pcs: &WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>,
    layout: &WhirNativeColumnBatchLayout,
    batch_commitment: &WhirNativeColumnBatchCommitment<MT::Commitment>,
    opening_proof: &WhirNativeColumnBatchOpeningProof<F, EF, MT>,
    batch_index: usize,
    terminal_claims: &[(usize, WhirNativeTerminalColumnClaim<EF>)],
    metadata: &[WhirNativeTableMetadata],
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
    if opening_proof.batch_index != batch_index
        || batch_commitment.num_variables != layout.num_variables
        || batch_commitment.columns != layout.columns
    {
        return Err(WhirNativeCircuitError::ColumnBatchMismatch { batch_index });
    }

    let mut challenger = make_challenger();
    observe_column_batch_opening_context(
        &mut challenger,
        public_io_digest,
        shape_digest,
        batch_index,
        batch_commitment,
        metadata,
        options,
    );
    let random_points = sample_column_batch_opening_points::<F, EF, Challenger>(
        &mut challenger,
        layout.num_variables,
        batch_index,
        options.openings_per_table,
    );
    let statements = build_column_batch_statements::<F, EF>(
        layout,
        metadata,
        &random_points,
        &opening_proof.random_opening_values,
        terminal_claims,
    )?;
    let statement_refs = statements.iter().collect::<Vec<_>>();
    let residual_claim = verify_batched_linear_sigma_reduction::<F, EF, Challenger>(
        &statement_refs,
        &opening_proof.reduction_proof,
        &mut challenger,
        0,
    )
    .map_err(
        |err| WhirNativeCircuitError::WhirColumnBatchVerificationFailed {
            batch_index,
            details: format!("{err:?}"),
        },
    )?;

    pcs.verify_batched_deferred(
        &[WhirBatchedDeferredVerifierOracle::SharedExtension {
            coeffs: residual_claim.coeffs,
            commitment: batch_commitment.commitment.clone(),
        }],
        residual_claim.point,
        residual_claim.value,
        &opening_proof.proof,
        &mut challenger,
    )
    .map_err(
        |err| WhirNativeCircuitError::WhirColumnBatchVerificationFailed {
            batch_index,
            details: format!("{err:?}"),
        },
    )
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

fn observe_column_batch_opening_context<F, Comm, Challenger>(
    challenger: &mut Challenger,
    public_io_digest: &[F],
    shape_digest: &[F],
    batch_index: usize,
    batch: &WhirNativeColumnBatchCommitment<Comm>,
    metadata: &[WhirNativeTableMetadata],
    options: WhirNativeCircuitOptions,
) where
    F: Field,
    Comm: Clone,
    Challenger: CanObserve<F> + CanObserve<Comm>,
{
    challenger.observe(F::from_u64(0x5748_4952_4342_4154));
    for &value in public_io_digest {
        challenger.observe(value);
    }
    for &value in shape_digest {
        challenger.observe(value);
    }
    challenger.observe(F::from_u64(WhirNativeOpeningMode::ColumnBatched.tag()));
    challenger.observe(F::from_u64(options.openings_per_table as u64));
    challenger.observe(F::from_u64(options.min_num_variables as u64));
    challenger.observe(F::from_u64(batch_index as u64));
    challenger.observe(F::from_u64(batch.num_variables as u64));
    challenger.observe(F::from_u64(metadata.len() as u64));
    for (table_index, metadata) in metadata.iter().enumerate() {
        observe_table_metadata::<F, Challenger>(challenger, table_index, metadata);
    }
    challenger.observe(F::from_u64(batch.columns.len() as u64));
    for column_ref in &batch.columns {
        observe_column_ref::<F, Challenger>(challenger, column_ref);
    }
    challenger.observe(batch.commitment.clone());
}

fn observe_poseidon2_shift_challenge_context<F, Comm, Challenger>(
    challenger: &mut Challenger,
    public_io_digest: &[F],
    shape_digest: &[F],
    options: WhirNativeCircuitOptions,
    commitment_context: WhirNativeCommitmentContext<'_, Comm>,
) where
    F: Field,
    Comm: Clone,
    Challenger: CanObserve<F> + CanObserve<Comm>,
{
    challenger.observe(F::from_u64(0x5032_5348_4348_414c));
    observe_circuit_constraint_context(
        challenger,
        public_io_digest,
        shape_digest,
        options,
        commitment_context,
    );
}

#[allow(clippy::too_many_arguments)]
fn observe_poseidon2_shift_aux_table_context<F, EF, Challenger>(
    challenger: &mut Challenger,
    public_io_digest: &[F],
    shape_digest: &[F],
    options: WhirNativeCircuitOptions,
    table_index: usize,
    source_metadata: &WhirNativeTableMetadata,
    aux_metadata: &WhirNativeTableMetadata,
    theta: EF,
    alpha: EF,
    beta: EF,
) where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: CanObserve<F>,
{
    challenger.observe(F::from_u64(0x5032_5348_4155_5843));
    for &value in public_io_digest {
        challenger.observe(value);
    }
    for &value in shape_digest {
        challenger.observe(value);
    }
    challenger.observe(F::from_u64(options.openings_per_table as u64));
    challenger.observe(F::from_u64(options.min_num_variables as u64));
    challenger.observe(F::from_u64(table_index as u64));
    observe_table_metadata::<F, Challenger>(challenger, table_index, source_metadata);
    observe_table_metadata::<F, Challenger>(challenger, table_index, aux_metadata);
    observe_ef::<F, EF, Challenger>(challenger, theta);
    observe_ef::<F, EF, Challenger>(challenger, alpha);
    observe_ef::<F, EF, Challenger>(challenger, beta);
}

#[allow(clippy::too_many_arguments)]
fn observe_poseidon2_shift_constraint_context<F, EF, Comm, Challenger>(
    challenger: &mut Challenger,
    public_io_digest: &[F],
    shape_digest: &[F],
    options: WhirNativeCircuitOptions,
    commitment_context: WhirNativeCommitmentContext<'_, Comm>,
    table_index: usize,
    source_metadata: &WhirNativeTableMetadata,
    inverse_commitment: &WhirNativeTableCommitment<Comm>,
    theta: EF,
    alpha: EF,
    beta: EF,
) where
    F: Field,
    EF: ExtensionField<F>,
    Comm: Clone,
    Challenger: CanObserve<F> + CanObserve<Comm>,
{
    challenger.observe(F::from_u64(0x5032_5348_5a43_484b));
    observe_circuit_constraint_context(
        challenger,
        public_io_digest,
        shape_digest,
        options,
        commitment_context,
    );
    challenger.observe(F::from_u64(table_index as u64));
    observe_table_metadata::<F, Challenger>(challenger, table_index, source_metadata);
    observe_table_metadata::<F, Challenger>(challenger, table_index, &inverse_commitment.metadata);
    challenger.observe(inverse_commitment.commitment.clone());
    observe_ef::<F, EF, Challenger>(challenger, theta);
    observe_ef::<F, EF, Challenger>(challenger, alpha);
    observe_ef::<F, EF, Challenger>(challenger, beta);
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

fn sample_column_batch_opening_points<F, EF, Challenger>(
    challenger: &mut Challenger,
    num_variables: usize,
    batch_index: usize,
    count: usize,
) -> Vec<Point<EF>>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
{
    let count = count.max(1);
    challenger.observe(F::from_u64(batch_index as u64));
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

fn observe_ef<F, EF, Challenger>(challenger: &mut Challenger, value: EF)
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: CanObserve<F>,
{
    for &coeff in value.as_basis_coefficients_slice() {
        challenger.observe(coeff);
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

    fn verify_test_proof(
        circuit: &Circuit<EF>,
        public_inputs: &[EF],
        mmcs: &MyMmcs,
        proof: &MyProof,
    ) -> Result<(), WhirNativeCircuitError> {
        verify_whir_native_circuit_proof(
            circuit,
            public_inputs,
            WhirNativeCircuitOptions {
                openings_per_table: 1,
                min_num_variables: 4,
            },
            proof,
            |num_variables| make_pcs(mmcs, num_variables),
            TestChallenger::new,
            no_poseidon,
        )
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

    fn poseidon2_challenger_proof() -> (Circuit<EF>, Vec<EF>, MyMmcs, MyProof) {
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

        let mmcs = make_mmcs();
        let proof = prove_whir_native_circuit(
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
        .expect("prove Poseidon2 circuit");

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
            (P2_BB_D4_WIDTH16_AIR_WIDTH * 2, "input witness"),
            (
                P2_BB_D4_WIDTH16_AIR_WIDTH * 2 + P2_BB_D4_WIDTH16_WIDTH_EXT,
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
            tampered.terminal_openings[P2_BB_D4_WIDTH16_AIR_WIDTH * 2 + port].value += EF::ONE;
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

    #[test]
    fn poseidon2_shift_bus_detects_terminal_and_cumulative_tampering() {
        let (circuit, public_inputs, mmcs, proof) = poseidon2_challenger_proof();
        verify_test_proof(&circuit, &public_inputs, &mmcs, &proof)
            .expect("honest Poseidon2 shift proof verifies");
        assert!(
            !proof.poseidon2_shift_bus_proof.sections.is_empty(),
            "Poseidon2 proof should carry a shift-bus section"
        );

        let mut tampered = proof.clone();
        tampered.poseidon2_shift_bus_proof.sections[0].terminal_main_openings[0].value += EF::ONE;
        verify_test_proof(&circuit, &public_inputs, &mmcs, &tampered)
            .expect_err("tampered Poseidon2 main shift opening must fail");

        let mut tampered = proof.clone();
        tampered.poseidon2_shift_bus_proof.sections[0].terminal_main_openings
            [P2_BB_D4_WIDTH16_AIR_WIDTH]
            .value += EF::ONE;
        verify_test_proof(&circuit, &public_inputs, &mmcs, &tampered)
            .expect_err("tampered Poseidon2 shifted opening must fail");

        let mut tampered = proof.clone();
        tampered.poseidon2_shift_bus_proof.sections[0].terminal_inverse_openings[0].value +=
            EF::ONE;
        verify_test_proof(&circuit, &public_inputs, &mmcs, &tampered)
            .expect_err("tampered Poseidon2 shift inverse opening must fail");

        let mut tampered = proof.clone();
        tampered.poseidon2_shift_bus_proof.sections[0]
            .inverse_opening_proof
            .opening_claims[0]
            .1 += EF::ONE;
        verify_test_proof(&circuit, &public_inputs, &mmcs, &tampered)
            .expect_err("tampered Poseidon2 shift inverse PCS claim must fail");

        let mut tampered = proof.clone();
        tampered.poseidon2_shift_bus_proof.sections[0].sender_cumulative += EF::ONE;
        verify_test_proof(&circuit, &public_inputs, &mmcs, &tampered)
            .expect_err("tampered Poseidon2 shift cumulative must fail");
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
    fn tampered_read_bus_cumulative_fails() {
        let (circuit, public_inputs, mmcs, mut proof) = simple_arithmetic_proof();
        proof.read_bus_proof.receiver_cumulative += EF::ONE;

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
        .expect_err("tampered read-bus cumulative must fail");
    }

    #[test]
    fn tampered_witness_read_count_fails() {
        let (circuit, public_inputs, mmcs, mut proof) = simple_arithmetic_proof();
        let count = proof
            .read_bus_proof
            .witness_read_counts
            .first_mut()
            .expect("at least one witness read count");
        *count += 1;

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
        .expect_err("tampered witness read count must fail");
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
            vec![
                EF::from(F::from_u64(3)),
                EF::from(F::from_u64(30)),
                EF::from(F::from_u64(30)),
            ],
            vec![
                EF::from(F::from_u64(7)),
                EF::from(F::from_u64(70)),
                EF::from(F::from_u64(70)),
            ],
        ];
        let expected_rows = vec![
            vec![EF::from(F::from_u64(3)), EF::from(F::from_u64(30))],
            vec![EF::from(F::from_u64(7)), EF::from(F::from_u64(70))],
        ];
        let table = pack_rows::<EF>(
            WhirNativeTableKind::Public,
            String::new(),
            rows.clone(),
            Some(KNOWN_ROWS_WIDTH),
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
            &expected_rows,
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
            &expected_rows,
            &witness_ids,
            &mut verifier_challenger,
        )
        .expect("verify known-row local constraints");
        assert_eq!(terminal_claims.len(), table.metadata.width);

        let mut tampered = proof.clone();
        tampered.terminal_openings[1].value += EF::ONE;
        let mut verifier_challenger = TestChallenger::new();
        verify_known_rows_local_constraints::<F, EF, TestChallenger>(
            &tampered,
            2,
            &table.metadata,
            &witness_table.metadata,
            &expected_rows,
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
                EF::ZERO,
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
                EF::ZERO,
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
                EF::ZERO,
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
                EF::ZERO,
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
        assert_eq!(terminal_claims.len(), ALU_WIDTH);

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
            EF::ZERO,
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
            EF::ONE,
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
