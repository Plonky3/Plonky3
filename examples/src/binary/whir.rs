//! Configuration for the binary harness's Boolean WHIR commitment.

use p3_air::BaseAir;
use p3_air::symbolic::AirLayout;
use p3_binary_pcs::BooleanTraceCommitmentData;
use p3_binary_pcs::whir::{
    BinaryWhirBudget, BinaryWhirProfile, BooleanWhirData, BooleanWhirDomain, BooleanWhirPcs,
    BooleanWhirProver, BooleanWhirTracePcs, recommended_cap_height,
};
use p3_bus::BusSymbolicBuilder;
use p3_lookup::InteractionSymbolicBuilder;
use p3_multi_stark::config::MultiStarkConfig;
use p3_sumcheck::TableShape;
use p3_sumcheck::layout::{Table, plan_stacked_layout};
use p3_sumcheck::ring_switch::bits::BitRingSwitch;

use super::{
    BinaryAir, BinaryProofError, BinaryProofOptions, BooleanWhirProveError, BooleanWhirVerifyError,
    Challenger, F, HarnessHash, MerkleMmcs,
};

/// Bytes the encoder writes for one base-field element: the widest variable-length code
/// for a 16-byte `F`, not its in-memory width.
const ENCODED_BASE_ELEMENT_BYTES: usize = 19;

/// Bytes the encoder writes for one extension-field element.
const ENCODED_EXTENSION_ELEMENT_BYTES: usize = 19;

/// Bytes one Merkle digest occupies.
const DIGEST_BYTES: usize = 32;

/// The proximity regime used for the WHIR schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhirRegime {
    /// Decode from the unique-codeword radius.
    UniqueDecoding,
    /// Decode from the proven Johnson radius.
    Johnson,
}

/// User-selected WHIR schedule and budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WhirOptions {
    /// Proximity regime.
    pub regime: WhirRegime,
    /// Per-term target used while deriving the schedule.
    pub term_security_bits: usize,
    /// Query, payload, and grinding ceilings.
    pub budget: BinaryWhirBudget,
}

/// Boolean PCS selected by the binary harness.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BooleanPcsChoice {
    /// The existing folding-only Boolean commitment.
    #[default]
    Folding,
    /// The additive-domain WHIR commitment.
    Whir(WhirOptions),
}

/// Symbolic declaration families that are outside this one-opening WHIR adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhirInteractionFamily {
    /// Intra-AIR interactions.
    Local,
    /// Cross-AIR interactions.
    Global,
    /// Mutually-exclusive interactions.
    Exclusive,
    /// Indexed reads.
    IndexedRead,
    /// Indexed tables.
    IndexedTable,
    /// Binary-native bus declarations.
    BinaryBus,
}

impl core::fmt::Display for WhirInteractionFamily {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::Local => "local interactions",
            Self::Global => "global interactions",
            Self::Exclusive => "exclusive interactions",
            Self::IndexedRead => "indexed reads",
            Self::IndexedTable => "indexed tables",
            Self::BinaryBus => "binary bus interactions",
        })
    }
}

/// Typed refusals made before WHIR setup, pricing, or witness generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum WhirIncompatibility {
    /// WHIR currently uses binary authentication geometry.
    #[error("WHIR requires binary Merkle arity 2, got {actual}")]
    MerkleArity { actual: usize },
    /// WHIR derives its initial leaf width from the fold schedule.
    #[error("WHIR does not accept an explicit leaf_elements override ({actual:?})")]
    LeafElements { actual: Option<usize> },
    /// PCS proof-of-work is distinct from WHIR's internal grinding phases.
    #[error("WHIR requires pcs_pow_bits = 0, got {actual}")]
    PcsPowBits { actual: usize },
    /// The dense-field entry points cannot select the Boolean WHIR adapter.
    #[error("WHIR is available only for Boolean trace commitments")]
    DenseField,
    /// WHIR requires a redundant starting code.
    #[error("WHIR requires log_inv_rate >= 1, got {actual}")]
    NonRedundantRate { actual: usize },
    /// A zero-width fold has no valid first schedule round.
    #[error("WHIR folding must be positive")]
    ZeroFolding,
    /// WHIR does not silently clamp its requested fold width.
    #[error("WHIR folding width {requested} exceeds packed arity {committed}")]
    FoldingExceeds { requested: usize, committed: usize },
    /// The harness scope excludes public values.
    #[error("WHIR harness does not support {actual} public values")]
    PublicValues { actual: usize },
    /// The harness scope excludes preprocessed columns.
    #[error("WHIR harness does not support {actual} preprocessed columns")]
    PreprocessedColumns { actual: usize },
    /// The table shape and the AIR disagree on the main trace width.
    #[error("WHIR table shape width {shape} does not match AIR width {air}")]
    WidthMismatch { shape: usize, air: usize },
    /// An AIR declaration names a successor column outside its main width.
    #[error("WHIR successor column {column} is outside AIR width {width}")]
    SuccessorOutOfRange { column: usize, width: usize },
    /// The adapter requires successors in strictly increasing order.
    #[error("WHIR successor columns are not strictly increasing: {previous} then {current}")]
    SuccessorNotStrict { previous: usize, current: usize },
    /// The scoped adapter has no budget for symbolic lookup reductions.
    #[error("WHIR does not support {0}")]
    UnsupportedInteractions(WhirInteractionFamily),
    /// The options and the helper entry point disagree about the selected PCS.
    #[error(
        "boolean_whir_config requires matching WHIR options; embedded={embedded:?}, explicit={explicit:?}"
    )]
    PcsChoiceMismatch {
        embedded: Option<WhirOptions>,
        explicit: WhirOptions,
    },
    /// Stacking the requested table shape would overflow `usize`.
    #[error("WHIR table shape overflows usize: 2^{num_variables} rows × {width} columns")]
    ShapeOverflow { num_variables: usize, width: usize },
}

/// Small scalar metadata describing a derived WHIR schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WhirSummary {
    /// Proximity regime.
    pub regime: WhirRegime,
    /// Per-term schedule target.
    pub term_security_bits: usize,
    /// Variables in the packed field witness.
    pub packed_variables: usize,
    /// Binary Merkle cap height.
    pub cap_height: usize,
    /// Number of positions opened by the WHIR schedule.
    pub total_opened_positions: usize,
    /// Conservative serialized PCS payload estimate.
    pub pcs_payload_bytes: usize,
    /// Hardest derived proof-of-work phase.
    pub max_grinding_bits: usize,
}

/// A binary Boolean trace configuration backed by additive-domain WHIR.
pub struct BooleanWhirStarkConfig<H: HarnessHash = p3_keccak::Keccak256Hash> {
    pub(crate) pcs: BooleanWhirTracePcs<F, BooleanWhirDomain, MerkleMmcs<H, 2>, Challenger<H>>,
    pub(crate) leaf_elements: usize,
    /// Derived schedule metadata used by the harness and its config-only tests.
    pub summary: WhirSummary,
    /// Budget retained for complete-proof checks in the shared harness.
    pub(crate) budget: BinaryWhirBudget,
}

impl<H: HarnessHash> core::fmt::Debug for BooleanWhirStarkConfig<H> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BooleanWhirStarkConfig")
            .field("leaf_elements", &self.leaf_elements)
            .field("summary", &self.summary)
            .field("budget", &self.budget)
            .finish_non_exhaustive()
    }
}

impl<H: HarnessHash> BooleanWhirStarkConfig<H> {
    /// Budget retained for the shared harness's complete-proof byte check.
    #[must_use]
    pub const fn budget(&self) -> BinaryWhirBudget {
        self.budget
    }
}

impl<H> MultiStarkConfig for BooleanWhirStarkConfig<H>
where
    H: HarnessHash,
{
    type Val = F;
    type Challenge = F;
    type Challenger = Challenger<H>;
    type Pcs = BooleanWhirTracePcs<F, BooleanWhirDomain, MerkleMmcs<H, 2>, Challenger<H>>;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }

    fn collision_resistance_bits(&self) -> Option<usize> {
        Some(H::COLLISION_RESISTANCE_BITS)
    }

    fn min_num_variables(&self) -> usize {
        1
    }

    fn build_witness(&self, tables: Vec<Table<F>>) -> Vec<Table<F>> {
        tables
    }

    fn committed_table<'a>(
        &self,
        prover_data: &'a BooleanTraceCommitmentData<F, BooleanWhirData<F, MerkleMmcs<H, 2>>>,
        table_index: usize,
    ) -> &'a Table<F> {
        prover_data.table(table_index)
    }
}

pub(crate) trait WhirErrorProjection: HarnessHash {
    fn prove_error(
        error: p3_multi_stark::ProvingError<
            p3_multi_stark::config::PcsProverError<BooleanWhirStarkConfig<Self>>,
        >,
    ) -> BinaryProofError;

    fn verify_error(
        error: p3_multi_stark::VerificationError<
            p3_multi_stark::config::PcsError<BooleanWhirStarkConfig<Self>>,
        >,
    ) -> BinaryProofError;
}

impl WhirErrorProjection for p3_keccak::Keccak256Hash {
    fn prove_error(
        error: p3_multi_stark::ProvingError<
            p3_multi_stark::config::PcsProverError<BooleanWhirStarkConfig<Self>>,
        >,
    ) -> BinaryProofError {
        BinaryProofError::WhirProve(BooleanWhirProveError::Keccak(error))
    }

    fn verify_error(
        error: p3_multi_stark::VerificationError<
            p3_multi_stark::config::PcsError<BooleanWhirStarkConfig<Self>>,
        >,
    ) -> BinaryProofError {
        BinaryProofError::WhirVerify(BooleanWhirVerifyError::Keccak(error))
    }
}

impl WhirErrorProjection for p3_blake3::Blake3 {
    fn prove_error(
        error: p3_multi_stark::ProvingError<
            p3_multi_stark::config::PcsProverError<BooleanWhirStarkConfig<Self>>,
        >,
    ) -> BinaryProofError {
        BinaryProofError::WhirProve(BooleanWhirProveError::Blake3(error))
    }

    fn verify_error(
        error: p3_multi_stark::VerificationError<
            p3_multi_stark::config::PcsError<BooleanWhirStarkConfig<Self>>,
        >,
    ) -> BinaryProofError {
        BinaryProofError::WhirVerify(BooleanWhirVerifyError::Blake3(error))
    }
}

impl<H> super::HarnessConfig for BooleanWhirStarkConfig<H>
where
    H: WhirErrorProjection,
{
    fn leaf_elements(&self) -> usize {
        self.leaf_elements
    }

    fn prove_error(
        error: p3_multi_stark::ProvingError<p3_multi_stark::config::PcsProverError<Self>>,
    ) -> BinaryProofError {
        H::prove_error(error)
    }

    fn verify_error(
        error: p3_multi_stark::VerificationError<p3_multi_stark::config::PcsError<Self>>,
    ) -> BinaryProofError {
        H::verify_error(error)
    }

    fn check_proof_bytes(&self, bytes: usize) -> Result<(), BinaryProofError> {
        self.budget()
            .check_bytes(bytes)
            .map_err(BinaryProofError::WhirBudget)
    }

    fn whir_summary(&self) -> Option<WhirSummary> {
        Some(self.summary)
    }
}

/// Build a Boolean WHIR configuration from AIR metadata and a table shape.
pub fn boolean_whir_config<A: BinaryAir, H: HarnessHash>(
    air: &A,
    shape: TableShape,
    options: BinaryProofOptions,
    whir: WhirOptions,
) -> Result<BooleanWhirStarkConfig<H>, BinaryProofError> {
    boolean_whir_config_with_schedule(air, shape, options, whir, false)
}

pub(crate) fn boolean_whir_config_with_schedule<A: BinaryAir, H: HarnessHash>(
    air: &A,
    shape: TableShape,
    options: BinaryProofOptions,
    whir: WhirOptions,
    trace_schedule: bool,
) -> Result<BooleanWhirStarkConfig<H>, BinaryProofError> {
    let embedded = match options.pcs {
        BooleanPcsChoice::Whir(embedded) => embedded,
        BooleanPcsChoice::Folding => {
            return Err(BinaryProofError::WhirIncompatible(
                WhirIncompatibility::PcsChoiceMismatch {
                    embedded: None,
                    explicit: whir,
                },
            ));
        }
    };
    if embedded != whir {
        return Err(BinaryProofError::WhirIncompatible(
            WhirIncompatibility::PcsChoiceMismatch {
                embedded: Some(embedded),
                explicit: whir,
            },
        ));
    }
    let num_variables = shape.num_variables();
    let row_count = 1usize.checked_shl(num_variables as u32).ok_or_else(|| {
        BinaryProofError::WhirIncompatible(WhirIncompatibility::ShapeOverflow {
            num_variables,
            width: shape.width(),
        })
    })?;
    if shape.width().checked_mul(row_count).is_none() {
        return Err(BinaryProofError::WhirIncompatible(
            WhirIncompatibility::ShapeOverflow {
                num_variables,
                width: shape.width(),
            },
        ));
    }

    validate_options(air, shape, options)?;

    let (arity, _) = plan_stacked_layout(&[shape]);
    let absorbed = BitRingSwitch::<F>::ABSORBED;
    let packed_variables = arity
        .checked_sub(absorbed)
        .ok_or(BinaryProofError::WhirConfig(
            p3_binary_pcs::BooleanTraceCommitmentError::Boolean(
                p3_binary_pcs::whir::BooleanWhirError::WitnessTooNarrow {
                    needed: absorbed,
                    actual: arity,
                },
            ),
        ))?;

    if options.folding == 0 {
        return Err(BinaryProofError::WhirIncompatible(
            WhirIncompatibility::ZeroFolding,
        ));
    }
    if options.folding > packed_variables {
        return Err(BinaryProofError::WhirIncompatible(
            WhirIncompatibility::FoldingExceeds {
                requested: options.folding,
                committed: packed_variables,
            },
        ));
    }

    let domain = BooleanWhirDomain::default();
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
    };
    let whir_config = profile
        .config::<F, F, Challenger<H>, _>(packed_variables, &domain)
        .map_err(BinaryProofError::WhirProfile)?;
    if trace_schedule {
        tracing::debug!(
            target: "p3_examples::binary::whir",
            regime = ?whir.regime,
            term_security_bits = whir.term_security_bits,
            folding_schedule = ?whir_config.folding_schedule,
            starting_folding_pow_bits = whir_config.starting_folding_pow_bits,
            round_log_inv_rates = ?whir_config
                .round_parameters
                .iter()
                .map(|round| round.log_inv_rate)
                .collect::<Vec<_>>(),
            round_queries = ?whir_config
                .round_parameters
                .iter()
                .map(|round| round.num_queries)
                .collect::<Vec<_>>(),
            round_ood_samples = ?whir_config
                .round_parameters
                .iter()
                .map(|round| round.ood_samples)
                .collect::<Vec<_>>(),
            round_pow_bits = ?whir_config
                .round_parameters
                .iter()
                .map(|round| (round.pow_bits, round.folding_pow_bits))
                .collect::<Vec<_>>(),
            commitment_ood_samples = whir_config.commitment_ood_samples,
            final_queries = whir_config.final_queries,
            final_pow_bits = whir_config.final_pow_bits,
            final_folding_pow_bits = whir_config.final_folding_pow_bits,
            final_sumcheck_rounds = whir_config.final_sumcheck_rounds,
            final_direct_send_arity = whir_config.final_round_config().num_variables,
        );
    }
    let first_fold =
        whir_config
            .folding_schedule
            .first()
            .copied()
            .ok_or(BinaryProofError::WhirIncompatible(
                WhirIncompatibility::ZeroFolding,
            ))?;
    let leaf_elements =
        1usize
            .checked_shl(first_fold as u32)
            .ok_or(BinaryProofError::WhirIncompatible(
                WhirIncompatibility::FoldingExceeds {
                    requested: first_fold,
                    committed: packed_variables,
                },
            ))?;

    let cap_height = recommended_cap_height(&whir_config);
    let merkle = MerkleMmcs::<H, 2>::new(
        super::Hash::new(H::INSTANCE),
        super::Compress::<H, 2>::new(H::INSTANCE),
        cap_height,
    );
    let prover = BooleanWhirProver::new(whir_config, domain, merkle);
    let pcs = BooleanWhirPcs::new(prover, arity).map_err(|error| {
        BinaryProofError::WhirConfig(p3_binary_pcs::BooleanTraceCommitmentError::Boolean(error))
    })?;
    let (num_claims, successor_tensors) = claim_shape(air, shape);
    let shape = pcs.proof_shape(num_claims, successor_tensors);
    whir.budget
        .check_shape(
            &shape,
            ENCODED_BASE_ELEMENT_BYTES,
            ENCODED_EXTENSION_ELEMENT_BYTES,
            DIGEST_BYTES,
        )
        .map_err(BinaryProofError::WhirBudget)?;
    let pcs = BooleanWhirTracePcs::from_commitment(pcs);
    let summary = WhirSummary {
        regime: whir.regime,
        term_security_bits: whir.term_security_bits,
        packed_variables,
        cap_height,
        total_opened_positions: shape.stir_queries,
        pcs_payload_bytes: shape.max_bytes(
            ENCODED_BASE_ELEMENT_BYTES,
            ENCODED_EXTENSION_ELEMENT_BYTES,
            DIGEST_BYTES,
        ),
        max_grinding_bits: shape.grinding_bits,
    };
    Ok(BooleanWhirStarkConfig {
        pcs,
        leaf_elements,
        summary,
        budget: whir.budget,
    })
}

fn validate_options<A: BinaryAir>(
    air: &A,
    shape: TableShape,
    options: BinaryProofOptions,
) -> Result<(), BinaryProofError> {
    if options.merkle_arity != 2 {
        return Err(BinaryProofError::WhirIncompatible(
            WhirIncompatibility::MerkleArity {
                actual: options.merkle_arity,
            },
        ));
    }
    if options.leaf_elements.is_some() {
        return Err(BinaryProofError::WhirIncompatible(
            WhirIncompatibility::LeafElements {
                actual: options.leaf_elements,
            },
        ));
    }
    if options.pcs_pow_bits != 0 {
        return Err(BinaryProofError::WhirIncompatible(
            WhirIncompatibility::PcsPowBits {
                actual: options.pcs_pow_bits,
            },
        ));
    }
    if options.log_inv_rate == 0 {
        return Err(BinaryProofError::WhirIncompatible(
            WhirIncompatibility::NonRedundantRate {
                actual: options.log_inv_rate,
            },
        ));
    }
    if BaseAir::<F>::num_public_values(air) != 0 {
        return Err(BinaryProofError::WhirIncompatible(
            WhirIncompatibility::PublicValues {
                actual: BaseAir::<F>::num_public_values(air),
            },
        ));
    }
    if BaseAir::<F>::preprocessed_width(air) != 0 {
        return Err(BinaryProofError::WhirIncompatible(
            WhirIncompatibility::PreprocessedColumns {
                actual: BaseAir::<F>::preprocessed_width(air),
            },
        ));
    }
    if shape.width() != air.width() {
        return Err(BinaryProofError::WhirIncompatible(
            WhirIncompatibility::WidthMismatch {
                shape: shape.width(),
                air: air.width(),
            },
        ));
    }
    let layout = AirLayout::from_air(air);
    let symbolic = InteractionSymbolicBuilder::<F, F>::from_air(air, layout);
    if !symbolic.local_interactions().is_empty() {
        return Err(BinaryProofError::WhirIncompatible(
            WhirIncompatibility::UnsupportedInteractions(WhirInteractionFamily::Local),
        ));
    }
    if !symbolic.global_interactions().is_empty() {
        return Err(BinaryProofError::WhirIncompatible(
            WhirIncompatibility::UnsupportedInteractions(WhirInteractionFamily::Global),
        ));
    }
    if !symbolic.exclusive_interactions().is_empty() {
        return Err(BinaryProofError::WhirIncompatible(
            WhirIncompatibility::UnsupportedInteractions(WhirInteractionFamily::Exclusive),
        ));
    }
    if !symbolic.indexed_reads().is_empty() {
        return Err(BinaryProofError::WhirIncompatible(
            WhirIncompatibility::UnsupportedInteractions(WhirInteractionFamily::IndexedRead),
        ));
    }
    if !symbolic.indexed_tables().is_empty() {
        return Err(BinaryProofError::WhirIncompatible(
            WhirIncompatibility::UnsupportedInteractions(WhirInteractionFamily::IndexedTable),
        ));
    }
    let bus = BusSymbolicBuilder::<F, F>::from_air(air, layout);
    if !bus.interactions().is_empty() {
        return Err(BinaryProofError::WhirIncompatible(
            WhirIncompatibility::UnsupportedInteractions(WhirInteractionFamily::BinaryBus),
        ));
    }
    validate_successors(air)
}

fn validate_successors<A: BinaryAir>(air: &A) -> Result<(), BinaryProofError> {
    let width = air.width();
    let next = air.main_next_row_columns();
    for &column in &next {
        if column >= width {
            return Err(BinaryProofError::WhirIncompatible(
                WhirIncompatibility::SuccessorOutOfRange { column, width },
            ));
        }
    }
    for pair in next.windows(2) {
        if pair[0] >= pair[1] {
            return Err(BinaryProofError::WhirIncompatible(
                WhirIncompatibility::SuccessorNotStrict {
                    previous: pair[0],
                    current: pair[1],
                },
            ));
        }
    }
    Ok(())
}

fn claim_shape<A: BinaryAir>(air: &A, shape: TableShape) -> (usize, bool) {
    let width = shape.width();
    let next = air.main_next_row_columns();
    let full = next.len() == width && next.iter().copied().eq(0..width);
    let claims = if next.is_empty() || full { 1 } else { width };
    (
        claims,
        !next.is_empty() && shape.num_variables() > BitRingSwitch::<F>::ABSORBED,
    )
}
