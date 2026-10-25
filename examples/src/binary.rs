//! Batches of hashes proved as one multi-STARK circuit table, each hash an instance of its Flock circuit.
//!
//! Values live in `GF(2^64)`, challenges in `GF(2^192)`, and the witness is committed through WHIR.
//!
//! The proving key, the security assessment, the proof and its verification are all timed apart.

use core::fmt;
use std::time::Instant;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_binary_field::{BinaryChallenger, Poly64, Poly192};
use p3_binary_pcs::BooleanTraceCommitmentData;
use p3_binary_pcs::whir::{
    BinaryWhirBudget, BinaryWhirDomain, BinaryWhirProfile, BooleanWhirData, BooleanWhirError,
    BooleanWhirPcs, BooleanWhirProver, BooleanWhirTracePcs, BudgetError, ProfileError,
    recommended_cap_height,
};
use p3_blake3::Blake3;
use p3_challenger::HashChallenger;
use p3_commit::MultilinearPcs;
use p3_flock::{Circuit, Witness};
use p3_keccak::Keccak256Hash;
use p3_maybe_rayon::prelude::current_num_threads;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multi_stark::config::{MultiStarkConfig, PcsError, PcsProverError};
use p3_multi_stark::{
    CIRCUIT_OPENING_CLAIMS, MultiStarkProof, ProverInstance, ProverInstances, ProvingError,
    ProvingKey, SecurityError, VerificationError, VerifierInstance, VerifierInstances,
    VerifyingKey, prove, security_report, setup, verify,
};
use p3_sumcheck::TableShape;
use p3_sumcheck::layout::{Table, plan_stacked_layout};
use p3_sumcheck::ring_switch::bits::BitRingSwitch;
use p3_symmetric::{CompressionFunctionFromHasher, CryptographicHasher, SerializingHasher};
use serde::Serialize;

/// Committed values: sixty-four bits each, in the polynomial basis.
pub type Val = Poly64;

/// Challenges: the cubic extension of the committed values.
pub type Challenge = Poly192;

/// The byte hash every Merkle leaf, node and transcript absorb runs.
type Hash<H> = SerializingHasher<H>;

/// Binary Merkle node compression.
type Compress<H> = CompressionFunctionFromHasher<H, 2, 32>;

/// A binary Merkle tree over committed values.
type ValMmcs<H> = MerkleTreeMmcs<Val, u8, Hash<H>, Compress<H>, 2, 32>;

/// The transcript, speaking committed values.
type ValChallenger<H> = BinaryChallenger<Val, HashChallenger<u8, H, 32>>;

/// The additive evaluation domain of the committed values.
type Domain = BinaryWhirDomain<Val>;

/// The Boolean trace commitment at this field pair.
type WhirPcs<H> = BooleanWhirTracePcs<Val, Challenge, Domain, ValMmcs<H>, ValChallenger<H>>;

/// Widest postcard code of one committed value: a `u64` varint.
const ENCODED_BASE_ELEMENT_BYTES: usize = 10;

/// Widest postcard code of one challenge: three `u64` varints.
const ENCODED_EXTENSION_ELEMENT_BYTES: usize = 30;

/// Bytes one Merkle digest occupies.
const DIGEST_BYTES: usize = 32;

/// A byte hash the Merkle trees and the transcript run.
pub trait HarnessHash:
    CryptographicHasher<u8, [u8; 32]> + Clone + Send + Sync + fmt::Debug + 'static
{
    /// The hasher itself.
    const INSTANCE: Self;

    /// Collision resistance of its 32-byte digest, in bits.
    const COLLISION_RESISTANCE_BITS: usize;

    /// Wrap a failure from `setup` or proving.
    fn prove_error(error: ProvingError<PcsProverError<WhirStarkConfig<Self>>>) -> BinaryProofError;

    /// Wrap a failure from verification.
    fn verify_error(error: VerificationError<PcsError<WhirStarkConfig<Self>>>) -> BinaryProofError;
}

impl HarnessHash for Keccak256Hash {
    const INSTANCE: Self = Self;
    const COLLISION_RESISTANCE_BITS: usize = 128;

    fn prove_error(error: ProvingError<PcsProverError<WhirStarkConfig<Self>>>) -> BinaryProofError {
        BinaryProofError::Prove(ProveError::Keccak(error))
    }

    fn verify_error(error: VerificationError<PcsError<WhirStarkConfig<Self>>>) -> BinaryProofError {
        BinaryProofError::Verify(VerifyError::Keccak(error))
    }
}

impl HarnessHash for Blake3 {
    const INSTANCE: Self = Self;
    const COLLISION_RESISTANCE_BITS: usize = 128;

    fn prove_error(error: ProvingError<PcsProverError<WhirStarkConfig<Self>>>) -> BinaryProofError {
        BinaryProofError::Prove(ProveError::Blake3(error))
    }

    fn verify_error(error: VerificationError<PcsError<WhirStarkConfig<Self>>>) -> BinaryProofError {
        BinaryProofError::Verify(VerifyError::Blake3(error))
    }
}

/// The byte hash a run selects.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum HashFamily {
    /// Keccak-256.
    #[default]
    Keccak256,
    /// BLAKE3.
    Blake3,
}

impl fmt::Display for HashFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Keccak256 => "keccak-256",
            Self::Blake3 => "blake3",
        })
    }
}

/// The proximity regime the WHIR schedule is derived in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum WhirRegime {
    /// Unique decoding: one candidate polynomial.
    UniqueDecoding,
    /// List decoding up to the Johnson bound, proven.
    Johnson,
}

/// The WHIR schedule and the ceilings it must fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WhirOptions {
    /// Proximity regime.
    pub regime: WhirRegime,
    /// Per-term target the schedule is derived for.
    pub term_security_bits: usize,
    /// Query, payload and grinding ceilings.
    pub budget: BinaryWhirBudget,
    /// Grinding bits before each query phase, each one standing in for part of a query.
    pub query_grinding: usize,
    /// Variables the first round folds and domain bits it drops, or the profile's own folding.
    pub first_round: Option<(usize, usize)>,
}

/// What one run proves under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinaryProofOptions {
    /// The WHIR schedule.
    pub whir: WhirOptions,
    /// Log of the inverse code rate.
    pub log_inv_rate: usize,
    /// Composed security target of the whole proof, in bits.
    pub security_bits: usize,
    /// Variables each WHIR round after the first folds.
    pub folding: usize,
    /// Grinding bits demanded per sumcheck round.
    pub sumcheck_pow_bits: usize,
    /// Byte hash the Merkle trees and the transcript share.
    pub hash: HashFamily,
}

/// The WHIR schedule one run derived, for its report.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct WhirSummary {
    /// Proximity regime.
    pub regime: WhirRegime,
    /// Per-term target the schedule was derived for.
    pub term_security_bits: usize,
    /// Variables of the packed polynomial WHIR commits.
    pub packed_variables: usize,
    /// Merkle cap height.
    pub cap_height: usize,
    /// Codeword positions every query phase opens, summed.
    pub total_opened_positions: usize,
    /// Upper bound on the opening's serialized bytes.
    pub pcs_payload_bytes: usize,
    /// Largest grinding difficulty of any phase.
    pub max_grinding_bits: usize,
}

/// Measurements from one run.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct BinaryProofReport {
    /// Hashes proved: the circuit table's rows.
    pub rows: usize,
    /// Committed bits per row: one circuit instance.
    pub width: usize,
    /// Variables of the stacked bit witness.
    pub stacked_variables: usize,
    /// Byte hash the Merkle trees and the transcript ran.
    pub hash: HashFamily,
    /// Committed values each Merkle leaf of the base codeword packed.
    pub leaf_elements: usize,
    /// Serialized proof size, in bytes.
    pub proof_bytes: usize,
    /// Wall-clock time spent in `prove`.
    pub prove_seconds: f64,
    /// Wall-clock time spent in `verify`.
    pub verify_seconds: f64,
    /// Wall-clock time spent configuring, setting up, and assessing security.
    pub setup_seconds: f64,
    /// Composed security bits reported by the statement's assessment.
    pub security_bits: f64,
    /// The WHIR schedule.
    pub whir: WhirSummary,
    /// Wall-clock time to serialize the proof with postcard.
    pub serialize_seconds: f64,
    /// Wall-clock time to deserialize the proof back from those bytes.
    pub deserialize_seconds: f64,
    /// Worker threads the prover ran on, the Rayon pool size.
    pub threads: usize,
    /// Wall-clock time the caller spent generating the witness, if it measured it.
    pub witness_seconds: Option<f64>,
}

impl BinaryProofReport {
    /// Records the time the caller spent generating the witness.
    #[must_use]
    pub const fn with_witness_seconds(mut self, seconds: f64) -> Self {
        self.witness_seconds = Some(seconds);
        self
    }
}

impl fmt::Display for BinaryProofReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Hashes: {}", self.rows)?;
        writeln!(f, "Bits per hash: {}", self.width)?;
        writeln!(f, "Stacked variables: {}", self.stacked_variables)?;
        writeln!(f, "Fields: GF(2^64) values, GF(2^192) challenges")?;
        writeln!(f, "Hash: {}", self.hash)?;
        writeln!(
            f,
            "Merkle leaf: {} field elements ({} bytes)",
            self.leaf_elements,
            self.leaf_elements * size_of::<Val>()
        )?;
        writeln!(f, "Threads: {}", self.threads)?;
        if let Some(seconds) = self.witness_seconds {
            writeln!(f, "Witness time: {seconds:.3}s")?;
        }
        writeln!(f, "Proof size: {} bytes", self.proof_bytes)?;
        writeln!(f, "Prove time: {:.3}s", self.prove_seconds)?;
        writeln!(f, "Serialize time: {:.3}s", self.serialize_seconds)?;
        writeln!(f, "Deserialize time: {:.3}s", self.deserialize_seconds)?;
        writeln!(f, "Verify time: {:.3}s", self.verify_seconds)?;
        writeln!(f, "Setup/security time: {:.3}s", self.setup_seconds)?;
        writeln!(f, "Composed security: {:.2} bits", self.security_bits)?;
        writeln!(f, "WHIR regime: {:?}", self.whir.regime)?;
        writeln!(f, "WHIR term target: {} bits", self.whir.term_security_bits)?;
        writeln!(f, "WHIR packed variables: {}", self.whir.packed_variables)?;
        writeln!(
            f,
            "WHIR opened positions: {}",
            self.whir.total_opened_positions
        )?;
        writeln!(
            f,
            "WHIR PCS payload estimate: {} bytes",
            self.whir.pcs_payload_bytes
        )?;
        writeln!(f, "WHIR max grinding: {} bits", self.whir.max_grinding_bits)
    }
}

/// Why a run could not be configured, proved, or verified.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum BinaryProofError {
    /// The rate is one, so the code has no redundancy to test.
    #[error("the inverse rate must be at least two")]
    NonRedundantRate,
    /// A fold would bind more variables than the committed witness has.
    #[error("folding {requested} variables exceeds the {committed} the witness commits")]
    FoldingExceeds {
        /// Variables asked for.
        requested: usize,
        /// Variables there are.
        committed: usize,
    },
    /// The WHIR profile could not derive a schedule.
    #[error("WHIR profile configuration failed: {0}")]
    Profile(ProfileError),
    /// The WHIR Boolean commitment refused its configuration.
    #[error("WHIR Boolean commitment configuration failed: {0}")]
    Commitment(BooleanWhirError),
    /// The schedule or the proof exceeded a ceiling.
    #[error("WHIR budget check failed: {0}")]
    Budget(BudgetError),
    /// The statement's security assessment fell short of the target.
    #[error("security check failed: {0}")]
    Security(SecurityError),
    /// Setup or proving failed.
    #[error("proof generation failed: {0}")]
    Prove(ProveError),
    /// Verification failed.
    #[error("proof verification failed: {0}")]
    Verify(VerifyError),
}

/// A proving failure, through the hash the run selected.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ProveError {
    /// Through Keccak-256.
    #[error("Keccak-256: {0}")]
    Keccak(ProvingError<PcsProverError<WhirStarkConfig<Keccak256Hash>>>),
    /// Through BLAKE3.
    #[error("BLAKE3: {0}")]
    Blake3(ProvingError<PcsProverError<WhirStarkConfig<Blake3>>>),
}

/// A verification failure, through the hash the run selected.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum VerifyError {
    /// Through Keccak-256.
    #[error("Keccak-256: {0}")]
    Keccak(VerificationError<PcsError<WhirStarkConfig<Keccak256Hash>>>),
    /// Through BLAKE3.
    #[error("BLAKE3: {0}")]
    Blake3(VerificationError<PcsError<WhirStarkConfig<Blake3>>>),
}

/// A multi-STARK configuration committing bits through WHIR at this field pair.
pub struct WhirStarkConfig<H: HarnessHash = Keccak256Hash> {
    /// The commitment every witness bit is packed into.
    pcs: WhirPcs<H>,
    /// Committed values each Merkle leaf of the base codeword packs.
    leaf_elements: usize,
    /// The derived schedule.
    summary: WhirSummary,
    /// Ceilings the complete proof is checked against.
    budget: BinaryWhirBudget,
}

impl<H: HarnessHash> MultiStarkConfig for WhirStarkConfig<H> {
    type Val = Val;
    type Challenge = Challenge;
    type Challenger = ValChallenger<H>;
    type Pcs = WhirPcs<H>;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }

    fn collision_resistance_bits(&self) -> Option<usize> {
        Some(H::COLLISION_RESISTANCE_BITS)
    }

    fn min_num_variables(&self) -> usize {
        1
    }

    fn build_witness(&self, tables: Vec<Table<Val>>) -> Vec<Table<Val>> {
        tables
    }

    fn committed_table<'a>(
        &self,
        prover_data: &'a BooleanTraceCommitmentData<
            Val,
            BooleanWhirData<Val, Challenge, ValMmcs<H>>,
        >,
        table_index: usize,
    ) -> &'a Table<Val> {
        prover_data.table(table_index)
    }
}

/// The AIR of a lone circuit table: one column per port word, and no constraint of its own.
struct CircuitPorts(usize);

impl<F> BaseAir<F> for CircuitPorts {
    fn width(&self) -> usize {
        self.0
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        Vec::new()
    }
}

impl<AB: AirBuilder> Air<AB> for CircuitPorts {
    fn eval(&self, _builder: &mut AB) {}
}

/// The transcript both sides of one run start from.
fn challenger<H: HarnessHash>() -> ValChallenger<H> {
    ValChallenger::from_hasher(b"p3-examples-flock-hashes-v1".to_vec(), H::INSTANCE)
}

/// Build the WHIR configuration committing one shape, its opening batching `claims` claims.
///
/// # Errors
///
/// A rate or folding the shape cannot take, a schedule the profile cannot derive, or one over budget.
fn whir_config<H: HarnessHash>(
    shape: TableShape,
    claims: usize,
    options: BinaryProofOptions,
) -> Result<WhirStarkConfig<H>, BinaryProofError> {
    let whir = options.whir;
    if options.log_inv_rate == 0 {
        return Err(BinaryProofError::NonRedundantRate);
    }
    let (arity, _) = plan_stacked_layout(&[shape]);
    let absorbed = BitRingSwitch::<Val, Challenge>::ABSORBED;
    let packed_variables = arity
        .checked_sub(absorbed)
        .ok_or(BinaryProofError::FoldingExceeds {
            requested: absorbed,
            committed: arity,
        })?;
    if options.folding == 0 || options.folding > packed_variables {
        return Err(BinaryProofError::FoldingExceeds {
            requested: options.folding,
            committed: packed_variables,
        });
    }

    // The schedule: the regime's queries, ground down, then the first round's own folding.
    let domain = Domain::default();
    let profile = match whir.regime {
        WhirRegime::UniqueDecoding => BinaryWhirProfile::unique_decoding(
            whir.term_security_bits,
            options.log_inv_rate,
            options.folding,
        ),
        WhirRegime::Johnson => BinaryWhirProfile::proven_list_decoding(
            whir.term_security_bits,
            options.log_inv_rate,
            options.folding,
        ),
    }
    .with_query_grinding(whir.query_grinding);
    let profile = match whir.first_round {
        Some((folding, reduction)) => {
            profile.with_first_round(folding.min(packed_variables), reduction)
        }
        None => profile,
    };
    let config = profile
        .config::<Challenge, Val, ValChallenger<H>, _>(packed_variables, &domain)
        .map_err(BinaryProofError::Profile)?;
    let first_fold = config.folding_schedule()[0];
    let cap_height = recommended_cap_height(&config);
    let merkle = ValMmcs::<H>::new(
        Hash::new(H::INSTANCE),
        Compress::new(H::INSTANCE),
        cap_height,
    );
    let pcs = BooleanWhirPcs::new(BooleanWhirProver::new(config, domain, merkle), arity)
        .map_err(BinaryProofError::Commitment)?;

    // The schedule is graded against the budget before anything is committed.
    let proof_shape = pcs
        .proof_shape(claims, false)
        .map_err(|error| BinaryProofError::Budget(error.into()))?;
    whir.budget
        .check_shape(
            &proof_shape,
            ENCODED_BASE_ELEMENT_BYTES,
            ENCODED_EXTENSION_ELEMENT_BYTES,
            DIGEST_BYTES,
        )
        .map_err(BinaryProofError::Budget)?;
    let summary = WhirSummary {
        regime: whir.regime,
        term_security_bits: whir.term_security_bits,
        packed_variables,
        cap_height,
        total_opened_positions: proof_shape.stir_queries,
        pcs_payload_bytes: proof_shape
            .max_bytes(
                ENCODED_BASE_ELEMENT_BYTES,
                ENCODED_EXTENSION_ELEMENT_BYTES,
                DIGEST_BYTES,
            )
            .map_err(|error| BinaryProofError::Budget(error.into()))?,
        max_grinding_bits: proof_shape.grinding_bits,
    };
    Ok(WhirStarkConfig {
        pcs: WhirPcs::from_commitment(pcs),
        leaf_elements: 1 << first_fold,
        summary,
        budget: whir.budget,
    })
}

/// Prove and verify a batch of circuit instances as one circuit table, reporting size and timing.
///
/// # Errors
///
/// Any failure of the configuration, the security target, proving, or verification.
pub fn prove_circuit(
    circuit: &Circuit,
    witness: Witness,
    options: BinaryProofOptions,
) -> Result<BinaryProofReport, BinaryProofError> {
    match options.hash {
        HashFamily::Keccak256 => prove_circuit_with::<Keccak256Hash>(circuit, witness, options),
        HashFamily::Blake3 => prove_circuit_with::<Blake3>(circuit, witness, options),
    }
}

/// Assess `2^log_instances` instances of a circuit without proving: the composed security bits and the WHIR schedule.
///
/// # Errors
///
/// Any failure of the configuration or the security target.
pub fn preflight_circuit(
    circuit: &Circuit,
    log_instances: usize,
    options: BinaryProofOptions,
) -> Result<(f64, WhirSummary), BinaryProofError> {
    match options.hash {
        HashFamily::Keccak256 => {
            preflight_circuit_with::<Keccak256Hash>(circuit, log_instances, options)
        }
        HashFamily::Blake3 => preflight_circuit_with::<Blake3>(circuit, log_instances, options),
    }
}

/// The preflight over one hash.
fn preflight_circuit_with<H: HarnessHash>(
    circuit: &Circuit,
    log_instances: usize,
    options: BinaryProofOptions,
) -> Result<(f64, WhirSummary), BinaryProofError> {
    let ports = CircuitPorts(circuit.const_pos() / 64);
    let (config, _, security_bits) = assess::<H>(circuit, &ports, log_instances, options)?;
    Ok((security_bits, config.summary))
}

/// Configure a batch of circuit instances, set it up, and check it meets the security target.
fn assess<H: HarnessHash>(
    circuit: &Circuit,
    ports: &CircuitPorts,
    log_instances: usize,
    options: BinaryProofOptions,
) -> Result<(WhirStarkConfig<H>, Keys<H>, f64), BinaryProofError> {
    let shape = TableShape::new(circuit.k_log() + log_instances, 1);
    // The opening reads the witness through the circuit's weighted claims alone.
    let config = whir_config::<H>(shape, CIRCUIT_OPENING_CLAIMS, options)?;
    let (pk, vk) = setup(&config, &[ports], &mut challenger::<H>()).map_err(H::prove_error)?;
    let instances = VerifierInstances::new(vec![VerifierInstance::circuit(
        ports,
        circuit,
        &vk,
        log_instances,
        &[],
    )]);
    let report = security_report(&config, &instances).map_err(BinaryProofError::Security)?;
    report
        .require_security(options.security_bits)
        .map_err(BinaryProofError::Security)?;
    let security_bits = report
        .security_bits()
        .expect("require_security succeeded, so every component is assessed");
    Ok((config, (pk, vk), security_bits))
}

/// The proving and verifying keys of one configuration.
type Keys<H> = (
    ProvingKey<WhirStarkConfig<H>>,
    VerifyingKey<WhirStarkConfig<H>>,
);

/// One run over one hash: configure, assess, prove, serialize, verify, report.
fn prove_circuit_with<H: HarnessHash>(
    circuit: &Circuit,
    witness: Witness,
    options: BinaryProofOptions,
) -> Result<BinaryProofReport, BinaryProofError> {
    let setup_start = Instant::now();
    let log_instances = witness.log_instances();
    // The table's AIR reads the port words, and nothing else reads them in a lone batch.
    let ports = CircuitPorts(circuit.const_pos() / 64);

    // The statement is assessed once, before the timed phases.
    let (config, (pk, vk), security_bits) = assess::<H>(circuit, &ports, log_instances, options)?;
    let instances = || {
        VerifierInstances::new(vec![VerifierInstance::circuit(
            &ports,
            circuit,
            &vk,
            log_instances,
            &[],
        )])
    };
    let setup_seconds = setup_start.elapsed().as_secs_f64();

    let prove_start = Instant::now();
    let proof = prove(
        &config,
        ProverInstances::new(vec![ProverInstance::circuit(
            &ports,
            circuit,
            witness,
            &pk,
            &[],
        )]),
        options.sumcheck_pow_bits,
        &mut challenger::<H>(),
    )
    .map_err(H::prove_error)?;
    let prove_seconds = prove_start.elapsed().as_secs_f64();

    let serialize_start = Instant::now();
    let bytes = postcard::to_allocvec(&proof).expect("postcard serialization must not fail");
    let serialize_seconds = serialize_start.elapsed().as_secs_f64();
    config
        .budget
        .check_bytes(bytes.len())
        .map_err(BinaryProofError::Budget)?;

    let deserialize_start = Instant::now();
    let proof: MultiStarkProof<WhirStarkConfig<H>> =
        postcard::from_bytes(&bytes).expect("postcard round trip must not fail");
    let deserialize_seconds = deserialize_start.elapsed().as_secs_f64();

    let verify_start = Instant::now();
    verify(
        &config,
        instances(),
        &proof,
        options.sumcheck_pow_bits,
        &mut challenger::<H>(),
    )
    .map_err(H::verify_error)?;
    let verify_seconds = verify_start.elapsed().as_secs_f64();

    Ok(BinaryProofReport {
        rows: 1 << log_instances,
        width: 1 << circuit.k_log(),
        stacked_variables: config.pcs().num_vars(),
        hash: options.hash,
        leaf_elements: config.leaf_elements,
        proof_bytes: bytes.len(),
        prove_seconds,
        verify_seconds,
        setup_seconds,
        security_bits,
        whir: config.summary,
        serialize_seconds,
        deserialize_seconds,
        threads: current_num_threads(),
        witness_seconds: None,
    })
}
