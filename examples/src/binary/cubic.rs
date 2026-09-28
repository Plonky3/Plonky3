//! The Boolean WHIR harness at values in `GF(2^64)` and challenges in `GF(2^192)`.
//!
//! ```text
//!     GF(2^128)              one field for values and challenges
//!     GF(2^64), GF(2^192)    64 trace bits per committed element, 192-bit challenges
//! ```
//!
//! Every challenge-field term then sits near 190 bits.
//! The proximity schedule and the 32-byte hash are what bound the proof.
//!
//! The zerocheck runs through the generic backend.
//! The sliced `GF(4)` kernels are wired for `GF(2^128)` alone.

use std::time::Instant;

use p3_air::{Air, BaseAir};
use p3_binary_field::{BinaryChallenger, Poly64, Poly192};
use p3_binary_pcs::BooleanTraceCommitmentData;
use p3_binary_pcs::whir::{
    BinaryWhirBudget, BinaryWhirDomain, BinaryWhirProfile, BooleanWhirData, BooleanWhirPcs,
    BooleanWhirProver, BooleanWhirTracePcs, recommended_cap_height,
};
use p3_blake3::Blake3;
use p3_bus::BusSymbolicBuilder;
use p3_challenger::HashChallenger;
use p3_commit::MultilinearPcs;
use p3_keccak::Keccak256Hash;
use p3_lookup::InteractionSymbolicBuilder;
use p3_maybe_rayon::prelude::current_num_threads;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multi_stark::config::{MultiStarkConfig, PcsError, PcsProverError};
use p3_multi_stark::folder::{InteractionMultilinearFolder, MultilinearFolder};
use p3_multi_stark::packed_ext::PackedExt;
use p3_multi_stark::{
    MultiStarkProof, ProverInstance, ProverInstances, ProvingError, VerificationError,
    VerifierInstance, VerifierInstances, prove, security_report, setup, verify,
};
use p3_sumcheck::TableShape;
use p3_sumcheck::layout::{Table, plan_stacked_layout};
use p3_sumcheck::ring_switch::bits::BitRingSwitch;

use super::whir::{claim_shape, validate_options};
use super::{
    BinaryAir, BinaryFields, BinaryProofError, BinaryProofOptions, BinaryProofReport,
    BooleanPcsChoice, Compress, HarnessHash, Hash, HashFamily, PcsIdentity, WhirIncompatibility,
    WhirOptions, WhirRegime, WhirSummary,
};

/// Committed values: sixty-four bits each, in the polynomial basis.
pub type Val = Poly64;

/// Challenges: the cubic extension of [`Val`].
pub type Challenge = Poly192;

/// A binary Merkle tree over committed values.
type ValMmcs<H> = MerkleTreeMmcs<Val, u8, Hash<H>, Compress<H, 2>, 2, 32>;

/// The transcript, speaking committed values.
type ValChallenger<H> = BinaryChallenger<Val, HashChallenger<u8, H, 32>>;

/// The additive evaluation domain of the committed values.
type Domain = BinaryWhirDomain<Val>;

/// The Boolean trace commitment at this field pair.
type CubicPcs<H> = BooleanWhirTracePcs<Val, Challenge, Domain, ValMmcs<H>, ValChallenger<H>>;

/// Widest postcard code of one committed value: a `u64` varint.
const ENCODED_BASE_ELEMENT_BYTES: usize = 10;

/// Widest postcard code of one challenge: three `u64` varints.
const ENCODED_EXTENSION_ELEMENT_BYTES: usize = 30;

/// Bytes one Merkle digest occupies.
const DIGEST_BYTES: usize = 32;

/// A Boolean trace configuration with `GF(2^64)` values and `GF(2^192)` challenges.
pub struct CubicWhirStarkConfig<H: HarnessHash = Keccak256Hash> {
    /// The commitment every trace bit is packed into.
    pcs: CubicPcs<H>,
    /// Field elements each Merkle leaf of the base codeword packs.
    leaf_elements: usize,
    /// Derived schedule metadata.
    pub summary: WhirSummary,
    /// Budget the complete proof is checked against.
    budget: BinaryWhirBudget,
}

impl<H: HarnessHash> MultiStarkConfig for CubicWhirStarkConfig<H> {
    type Val = Val;
    type Challenge = Challenge;
    type Challenger = ValChallenger<H>;
    type Pcs = CubicPcs<H>;

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

/// AIR obligations of the generic prover at this field pair, each stated once.
///
/// `Poly64` is its own packing and `Poly192` its own extension packing.
/// So [`p3_multi_stark::folder::ProverAir`] names some folders twice, which this trait does not.
pub trait CubicAir:
    BaseAir<Val>
    + Air<InteractionSymbolicBuilder<Val, Challenge>>
    + Air<BusSymbolicBuilder<Val, Challenge>>
    + for<'a> Air<MultilinearFolder<'a, Val, Challenge, Challenge>>
    + for<'a> Air<InteractionMultilinearFolder<'a, Val, Challenge, Challenge>>
    + for<'a> Air<MultilinearFolder<'a, Val, Val, Challenge>>
    + for<'a> Air<MultilinearFolder<'a, Val, PackedExt<Val, Challenge>, PackedExt<Val, Challenge>>>
    + for<'a> Air<InteractionMultilinearFolder<'a, Val, Val, Challenge>>
    + for<'a> Air<
        InteractionMultilinearFolder<'a, Val, PackedExt<Val, Challenge>, PackedExt<Val, Challenge>>,
    >
{
}

impl<A> CubicAir for A where
    A: BaseAir<Val>
        + Air<InteractionSymbolicBuilder<Val, Challenge>>
        + Air<BusSymbolicBuilder<Val, Challenge>>
        + for<'a> Air<MultilinearFolder<'a, Val, Challenge, Challenge>>
        + for<'a> Air<InteractionMultilinearFolder<'a, Val, Challenge, Challenge>>
        + for<'a> Air<MultilinearFolder<'a, Val, Val, Challenge>>
        + for<'a> Air<
            MultilinearFolder<'a, Val, PackedExt<Val, Challenge>, PackedExt<Val, Challenge>>,
        > + for<'a> Air<InteractionMultilinearFolder<'a, Val, Val, Challenge>>
        + for<'a> Air<
            InteractionMultilinearFolder<
                'a,
                Val,
                PackedExt<Val, Challenge>,
                PackedExt<Val, Challenge>,
            >,
        >
{
}

/// The proving error at this field pair, through the two supported hashes.
#[derive(Debug, thiserror::Error)]
pub enum CubicProveError {
    /// Through Keccak-256.
    #[error("Keccak-256: {0}")]
    Keccak(ProvingError<PcsProverError<CubicWhirStarkConfig<Keccak256Hash>>>),
    /// Through BLAKE3.
    #[error("BLAKE3: {0}")]
    Blake3(ProvingError<PcsProverError<CubicWhirStarkConfig<Blake3>>>),
}

/// The verification error at this field pair, through the two supported hashes.
#[derive(Debug, thiserror::Error)]
pub enum CubicVerifyError {
    /// Through Keccak-256.
    #[error("Keccak-256: {0}")]
    Keccak(VerificationError<PcsError<CubicWhirStarkConfig<Keccak256Hash>>>),
    /// Through BLAKE3.
    #[error("BLAKE3: {0}")]
    Blake3(VerificationError<PcsError<CubicWhirStarkConfig<Blake3>>>),
}

/// How one hash's failures at this field pair surface as a [`BinaryProofError`].
trait CubicErrorProjection: HarnessHash {
    /// Wrap a failure from `setup` or proving.
    fn prove_error(
        error: ProvingError<PcsProverError<CubicWhirStarkConfig<Self>>>,
    ) -> BinaryProofError;

    /// Wrap a failure from verification.
    fn verify_error(
        error: VerificationError<PcsError<CubicWhirStarkConfig<Self>>>,
    ) -> BinaryProofError;
}

impl CubicErrorProjection for Keccak256Hash {
    fn prove_error(
        error: ProvingError<PcsProverError<CubicWhirStarkConfig<Self>>>,
    ) -> BinaryProofError {
        BinaryProofError::CubicProve(CubicProveError::Keccak(error))
    }

    fn verify_error(
        error: VerificationError<PcsError<CubicWhirStarkConfig<Self>>>,
    ) -> BinaryProofError {
        BinaryProofError::CubicVerify(CubicVerifyError::Keccak(error))
    }
}

impl CubicErrorProjection for Blake3 {
    fn prove_error(
        error: ProvingError<PcsProverError<CubicWhirStarkConfig<Self>>>,
    ) -> BinaryProofError {
        BinaryProofError::CubicProve(CubicProveError::Blake3(error))
    }

    fn verify_error(
        error: VerificationError<PcsError<CubicWhirStarkConfig<Self>>>,
    ) -> BinaryProofError {
        BinaryProofError::CubicVerify(CubicVerifyError::Blake3(error))
    }
}

/// The transcript both sides of one run start from.
fn challenger<H: HarnessHash>() -> ValChallenger<H> {
    ValChallenger::from_hasher(b"p3-examples-binary-hash-air-v1".to_vec(), H::INSTANCE)
}

/// Build the Boolean WHIR configuration at this field pair for one AIR and table shape.
///
/// The options are checked as the `GF(2^128)` adapter checks them.
///
/// # Errors
///
/// - The options or the AIR fall outside the adapter's scope.
/// - The profile cannot derive a schedule, or the schedule exceeds the budget.
pub fn cubic_whir_config<A: BinaryAir, H: HarnessHash>(
    air: &A,
    shape: TableShape,
    options: BinaryProofOptions,
    whir: WhirOptions,
) -> Result<CubicWhirStarkConfig<H>, BinaryProofError> {
    validate_options(air, shape, options)?;
    let (arity, _) = plan_stacked_layout(&[shape]);
    let absorbed = BitRingSwitch::<Val, Challenge>::ABSORBED;
    let packed_variables =
        arity
            .checked_sub(absorbed)
            .ok_or(BinaryProofError::WhirIncompatible(
                WhirIncompatibility::FoldingExceeds {
                    requested: absorbed,
                    committed: arity,
                },
            ))?;
    if options.folding == 0 || options.folding > packed_variables {
        return Err(BinaryProofError::WhirIncompatible(
            WhirIncompatibility::FoldingExceeds {
                requested: options.folding,
                committed: packed_variables,
            },
        ));
    }

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
    };
    let whir_config = profile
        .config::<Challenge, Val, ValChallenger<H>, _>(packed_variables, &domain)
        .map_err(BinaryProofError::WhirProfile)?;
    let first_fold = whir_config.folding_schedule()[0];
    let cap_height = recommended_cap_height(&whir_config);
    let merkle = ValMmcs::<H>::new(
        Hash::new(H::INSTANCE),
        Compress::<H, 2>::new(H::INSTANCE),
        cap_height,
    );
    let pcs = BooleanWhirPcs::new(BooleanWhirProver::new(whir_config, domain, merkle), arity)
        .map_err(|error| {
            BinaryProofError::WhirConfig(p3_binary_pcs::BooleanTraceCommitmentError::Boolean(error))
        })?;

    let (num_claims, successor_tensors) = claim_shape(air, shape, absorbed);
    let proof_shape = pcs.proof_shape(num_claims, successor_tensors);
    whir.budget
        .check_shape(
            &proof_shape,
            ENCODED_BASE_ELEMENT_BYTES,
            ENCODED_EXTENSION_ELEMENT_BYTES,
            DIGEST_BYTES,
        )
        .map_err(BinaryProofError::WhirBudget)?;
    let summary = WhirSummary {
        regime: whir.regime,
        term_security_bits: whir.term_security_bits,
        packed_variables,
        cap_height,
        total_opened_positions: proof_shape.stir_queries,
        pcs_payload_bytes: proof_shape.max_bytes(
            ENCODED_BASE_ELEMENT_BYTES,
            ENCODED_EXTENSION_ELEMENT_BYTES,
            DIGEST_BYTES,
        ),
        max_grinding_bits: proof_shape.grinding_bits,
    };
    Ok(CubicWhirStarkConfig {
        pcs: CubicPcs::from_commitment(pcs),
        leaf_elements: 1 << first_fold,
        summary,
        budget: whir.budget,
    })
}

/// Prove and verify a Boolean-valued `air` at this field pair, reporting size and timing.
///
/// The trace is committed as bits through the WHIR commitment `options.pcs` selects.
///
/// # Errors
///
/// - `options.pcs` is not WHIR, which is the only commitment built at this field pair.
/// - Any failure of the configuration, the security target, proving, or verification.
///
/// # Panics
///
/// - The trace height is not a power of two.
pub fn prove_boolean_air_cubic<A>(
    air: &A,
    trace: Table<Val>,
    options: BinaryProofOptions,
) -> Result<BinaryProofReport, BinaryProofError>
where
    A: BinaryAir + CubicAir,
{
    let BooleanPcsChoice::Whir(whir) = options.pcs else {
        return Err(BinaryProofError::WhirIncompatible(
            WhirIncompatibility::DenseField,
        ));
    };
    match options.hash {
        HashFamily::Keccak256 => prove_with::<A, Keccak256Hash>(air, trace, options, whir),
        HashFamily::Blake3 => prove_with::<A, Blake3>(air, trace, options, whir),
    }
}

/// [`prove_boolean_air_cubic`] over one hash.
fn prove_with<A, H>(
    air: &A,
    trace: Table<Val>,
    options: BinaryProofOptions,
    whir: WhirOptions,
) -> Result<BinaryProofReport, BinaryProofError>
where
    A: BinaryAir + CubicAir,
    H: CubicErrorProjection,
{
    let shape = trace.shape();
    let setup_start = Instant::now();
    let config = cubic_whir_config::<A, H>(air, shape, options, whir)?;

    // The statement is assessed once, before the timed phases.
    let (pk, vk) = setup(&config, &[air], &mut challenger::<H>()).map_err(H::prove_error)?;
    let instances = || {
        VerifierInstances::new(vec![VerifierInstance::new(
            air,
            &vk,
            shape.num_variables(),
            &[],
        )])
    };
    let report = security_report(&config, &instances()).map_err(BinaryProofError::Security)?;
    report
        .require_security(options.security_bits)
        .map_err(BinaryProofError::Security)?;
    let security_bits = report
        .security_bits()
        .expect("require_security succeeded, so every component is assessed");
    let setup_seconds = setup_start.elapsed().as_secs_f64();

    let prove_start = Instant::now();
    let proof = prove(
        &config,
        ProverInstances::new(vec![ProverInstance::new(air, trace, &pk, &[])]),
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
        .map_err(BinaryProofError::WhirBudget)?;

    let deserialize_start = Instant::now();
    let proof: MultiStarkProof<CubicWhirStarkConfig<H>> =
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
        rows: 1 << shape.num_variables(),
        width: shape.width(),
        stacked_variables: config.pcs().num_vars(),
        fields: BinaryFields::Gf64Gf192,
        hash: options.hash,
        leaf_elements: config.leaf_elements,
        requested_leaf_elements: options.leaf_elements,
        proof_bytes: bytes.len(),
        prove_seconds,
        verify_seconds,
        setup_seconds,
        security_bits,
        pcs: PcsIdentity::Whir,
        whir: Some(config.summary),
        serialize_seconds,
        deserialize_seconds,
        threads: current_num_threads(),
        witness_seconds: None,
    })
}

#[cfg(test)]
mod tests {
    use p3_binary_field::Gf2;
    use p3_blake3_air::Blake3BinaryAir;
    use p3_keccak_air::KeccakBinaryAir;

    use super::*;

    /// Options that reach 128 bits on a small trace, in the unique-decoding regime.
    fn options(hash: HashFamily) -> BinaryProofOptions {
        BinaryProofOptions {
            pcs: BooleanPcsChoice::Whir(WhirOptions {
                regime: WhirRegime::UniqueDecoding,
                term_security_bits: 140,
                budget: BinaryWhirBudget {
                    max_stir_queries: 4000,
                    max_proof_bytes: 1 << 24,
                    max_grinding_bits: 20,
                },
            }),
            log_inv_rate: 2,
            folding: 4,
            merkle_arity: 2,
            security_bits: 127,
            hash,
            ..BinaryProofOptions::default()
        }
    }

    #[test]
    fn a_packed_hash_trace_proves_at_the_hash_cap_under_both_hashes() {
        // Eight compressions, packed sixty-four rows to a word as the harness receives them.
        let air = Blake3BinaryAir::default();
        let words = air.generate_random_trace_packed::<Gf2>(8);
        for hash in [HashFamily::Keccak256, HashFamily::Blake3] {
            let report = prove_boolean_air_cubic(
                &air,
                Table::from_packed_bits(words.clone(), 3),
                options(hash),
            )
            .expect("the proof verifies");

            // Every challenge-field term clears 128, so only the hash's cap binds.
            assert_eq!(report.fields, BinaryFields::Gf64Gf192);
            assert!(report.security_bits > 127.99, "{}", report.security_bits);
            assert_eq!(
                report.whir.map(|whir| whir.regime),
                Some(WhirRegime::UniqueDecoding)
            );
        }
    }

    #[test]
    fn a_successor_view_proves() {
        // Keccak-f reads the next row, so the ring switch sends its successor elements.
        let air = KeccakBinaryAir::default();
        let words = air.generate_random_trace_packed::<Gf2>(1);
        let report = prove_boolean_air_cubic(
            &air,
            Table::from_packed_bits(words, 5),
            options(HashFamily::Keccak256),
        )
        .expect("the proof verifies");
        assert!(report.security_bits > 127.99, "{}", report.security_bits);
    }

    #[test]
    fn the_folding_commitment_is_refused() {
        // Only the WHIR commitment is built at this field pair.
        let air = Blake3BinaryAir::default();
        let words = air.generate_random_trace_packed::<Gf2>(8);
        let error = prove_boolean_air_cubic(
            &air,
            Table::from_packed_bits(words, 3),
            BinaryProofOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            BinaryProofError::WhirIncompatible(WhirIncompatibility::DenseField)
        ));
    }
}
