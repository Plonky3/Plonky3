use core::fmt::Debug;

use p3_air::Air;
use p3_air::symbolic::SymbolicAirBuilder;
use p3_challenger::{DuplexChallenger, SerializingChallenger32};
use p3_circle::CirclePcs;
use p3_commit::ExtensionMmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::extension::ComplexExtendable;
use p3_field::{
    ExtensionField, Field, PrimeField32, PrimeField64, TwoAdicField, UniformSamplingField,
};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_keccak::{Keccak256Hash, KeccakF};
use p3_mersenne_31::{Mersenne31, QM31};
use p3_stir::{SecurityAssumption, StirParameters, TwoAdicStirPcs};
use p3_symmetric::{CryptographicPermutation, PaddingFreeSponge, SerializingHasher};
use p3_uni_stark::{
    AirLayout, ConjecturedSecurity, OpeningShape, PcsError, Proof, StarkGenericConfig,
    StarkSecurityParams, VerificationError, prove, verify,
};
use rand::distr::StandardUniform;
use rand::prelude::Distribution;

use crate::airs::ExampleHashAir;
use crate::types::{
    KeccakCircleStarkConfig, KeccakCompressionFunction, KeccakMerkleMmcs, KeccakStarkConfig,
    Poseidon2CircleStarkConfig, Poseidon2Compression, Poseidon2MerkleMmcs, Poseidon2Sponge,
    Poseidon2StarkConfig, StirKeccakStarkConfig, StirPoseidon2StarkConfig,
};

/// Conjectured security target shared by the example PCS configurations.
const EXAMPLE_SECURITY_BITS: usize = 100;

/// PCS batching grind used by the STIR examples.
const STIR_BATCH_POW_BITS: usize = 16;

/// Circle's wide opening batches also need grinding to reach the shared target.
const CIRCLE_BATCH_POW_BITS: usize = 16;

/// Choose the fewest queries meeting the target, including the random-words correction.
fn example_fri_parameters<EF: Field, M>(mut params: FriParameters<M>) -> FriParameters<M> {
    // More queries cannot repair an algebraic bound below the target.
    assert!(
        ConjecturedSecurity::compute_ldt_only(
            params.log_blowup,
            usize::MAX,
            params.query_proof_of_work_bits,
            128,
            EF::bits(),
        )
        .security_bits
            >= EXAMPLE_SECURITY_BITS,
        "challenge field is too small for the example security target"
    );
    params.num_queries =
        (EXAMPLE_SECURITY_BITS - params.query_proof_of_work_bits).div_ceil(params.log_blowup);
    while ConjecturedSecurity::compute_ldt_only(
        params.log_blowup,
        params.num_queries,
        params.query_proof_of_work_bits,
        128,
        EF::bits(),
    )
    .security_bits
        < EXAMPLE_SECURITY_BITS
    {
        params.num_queries += 1;
    }
    params
}

fn example_circle_parameters<EF: Field, M>(mmcs: M) -> FriParameters<M> {
    example_fri_parameters::<EF, _>(FriParameters {
        batch_proof_of_work_bits: CIRCLE_BATCH_POW_BITS,
        ..FriParameters::new_benchmark(mmcs)
    })
}

/// Result type for Keccak-based two-adic proofs
type KeccakTwoAdicResult<F, EF, DFT> =
    Result<(), VerificationError<PcsError<KeccakStarkConfig<F, EF, DFT>>>>;

/// Result type for Poseidon2-based two-adic proofs
type Poseidon2TwoAdicResult<F, EF, DFT, Perm16, Perm24> =
    Result<(), VerificationError<PcsError<Poseidon2StarkConfig<F, EF, DFT, Perm16, Perm24>>>>;

/// Result type for Keccak-based, STIR-backed two-adic proofs
type StirKeccakTwoAdicResult<F, EF, DFT> =
    Result<(), VerificationError<PcsError<StirKeccakStarkConfig<F, EF, DFT>>>>;

/// Result type for Poseidon2-based, STIR-backed two-adic proofs
type StirPoseidon2TwoAdicResult<F, EF, DFT, Perm16, Perm24> =
    Result<(), VerificationError<PcsError<StirPoseidon2StarkConfig<F, EF, DFT, Perm16, Perm24>>>>;

/// Result type for Keccak-based circle proofs with Mersenne31
type KeccakCircleResult =
    Result<(), VerificationError<PcsError<KeccakCircleStarkConfig<Mersenne31, QM31>>>>;

/// Result type for Poseidon2-based circle proofs
type Poseidon2CircleResult<F, EF, Perm16, Perm24> =
    Result<(), VerificationError<PcsError<Poseidon2CircleStarkConfig<F, EF, Perm16, Perm24>>>>;

/// Produce a MerkleTreeMmcs which uses the KeccakF permutation.
const fn get_keccak_mmcs<F: Field>(cap_height: usize) -> KeccakMerkleMmcs<F> {
    let u64_hash = PaddingFreeSponge::<KeccakF, 25, 17, 4>::new(KeccakF {});

    let field_hash = SerializingHasher::new(u64_hash);

    let compress = KeccakCompressionFunction::new(u64_hash);

    KeccakMerkleMmcs::new(field_hash, compress, cap_height)
}

/// Produce a MerkleTreeMmcs from a pair of cryptographic field permutations.
///
/// The first permutation will be used for compression and the second for more sponge hashing.
/// Currently this is only intended to be used with a pair of Poseidon2 hashes of with 16 and 24
/// but this can easily be generalised in future if we desire.
const fn get_poseidon2_mmcs<
    F: Field,
    Perm16: CryptographicPermutation<[F; 16]> + CryptographicPermutation<[F::Packing; 16]>,
    Perm24: CryptographicPermutation<[F; 24]> + CryptographicPermutation<[F::Packing; 24]>,
>(
    perm16: Perm16,
    perm24: Perm24,
    cap_height: usize,
) -> Poseidon2MerkleMmcs<F, Perm16, Perm24> {
    let hash = Poseidon2Sponge::new(perm24);

    let compress = Poseidon2Compression::new(perm16);

    Poseidon2MerkleMmcs::<F, _, _>::new(hash, compress, cap_height)
}

/// Prove the given ProofGoal using the Keccak hash function to build the merkle tree.
///
/// This allows the user to choose:
/// - The Field
/// - The Proof Goal (Choice of both hash function and desired number of hashes to prove)
/// - The DFT
#[inline]
pub fn prove_monty31_keccak<
    F: PrimeField32 + TwoAdicField,
    EF: ExtensionField<F>,
    DFT: TwoAdicSubgroupDft<F>,
    PG: ExampleHashAir<F, KeccakStarkConfig<F, EF, DFT>> + Air<SymbolicAirBuilder<F, EF>>,
>(
    proof_goal: &PG,
    dft: DFT,
    num_hashes: usize,
) -> KeccakTwoAdicResult<F, EF, DFT>
where
    StandardUniform: Distribution<F>,
{
    let val_mmcs = get_keccak_mmcs(3);

    let challenge_mmcs = ExtensionMmcs::<F, EF, _>::new(val_mmcs.clone());
    let fri_params =
        example_fri_parameters::<EF, _>(FriParameters::new_benchmark_high_arity(challenge_mmcs));

    let security_params = StarkSecurityParams::from_air::<F, EF, _>(
        fri_params.security_regime(),
        proof_goal,
        AirLayout::from_air(proof_goal),
        EF::bits(),
        128,
        2,
        OpeningShape::new(),
        fri_params.grinding_sites(),
    );

    let trace = proof_goal.generate_trace_rows(num_hashes, fri_params.log_blowup);

    let pcs = TwoAdicFriPcs::new(dft, val_mmcs, fri_params);
    let challenger = SerializingChallenger32::from_hasher(vec![], Keccak256Hash {});

    let config = KeccakStarkConfig::new(pcs, challenger);

    let proof = prove(&config, proof_goal, trace, &[]);
    report_proof_size(&proof);

    let result = verify(&config, proof_goal, &proof, &[]);
    if result.is_ok() {
        report_parameter_security(&proof, &security_params);
    }
    result
}

/// Prove the given ProofGoal using the Poseidon2 hash function to build the merkle tree.
///
/// This allows the user to choose:
/// - The Field
/// - The Proof Goal (Choice of Hash function and number of hashes to prove)
/// - The DFT
#[inline]
pub fn prove_monty31_poseidon2<
    F: PrimeField32 + TwoAdicField,
    EF: ExtensionField<F>,
    DFT: TwoAdicSubgroupDft<F>,
    Perm16: CryptographicPermutation<[F; 16]> + CryptographicPermutation<[F::Packing; 16]>,
    Perm24: CryptographicPermutation<[F; 24]> + CryptographicPermutation<[F::Packing; 24]>,
    PG: ExampleHashAir<F, Poseidon2StarkConfig<F, EF, DFT, Perm16, Perm24>>
        + Air<SymbolicAirBuilder<F, EF>>,
>(
    proof_goal: &PG,
    dft: DFT,
    num_hashes: usize,
    perm16: Perm16,
    perm24: Perm24,
) -> Poseidon2TwoAdicResult<F, EF, DFT, Perm16, Perm24>
where
    StandardUniform: Distribution<F>,
{
    let val_mmcs = get_poseidon2_mmcs::<F, _, _>(perm16, perm24.clone(), 3);

    let challenge_mmcs = ExtensionMmcs::<F, EF, _>::new(val_mmcs.clone());
    let fri_params =
        example_fri_parameters::<EF, _>(FriParameters::new_benchmark_high_arity(challenge_mmcs));
    let security_params = StarkSecurityParams::from_air::<F, EF, _>(
        fri_params.security_regime(),
        proof_goal,
        AirLayout::from_air(proof_goal),
        EF::bits(),
        128,
        2,
        OpeningShape::new(),
        fri_params.grinding_sites(),
    );

    let trace = proof_goal.generate_trace_rows(num_hashes, fri_params.log_blowup);

    let pcs = TwoAdicFriPcs::new(dft, val_mmcs, fri_params);
    let challenger = DuplexChallenger::new(perm24);

    let config = Poseidon2StarkConfig::new(pcs, challenger);

    let proof = prove(&config, proof_goal, trace, &[]);
    report_proof_size(&proof);

    let result = verify(&config, proof_goal, &proof, &[]);
    if result.is_ok() {
        report_parameter_security(&proof, &security_params);
    }
    result
}

/// Prove the given ProofGoal using the Keccak hash function to build the merkle tree, with
/// STIR as the opening protocol.
///
/// This allows the user to choose:
/// - The Field
/// - The Proof Goal (Choice of both hash function and desired number of hashes to prove)
/// - The DFT
#[inline]
pub fn prove_monty31_keccak_stir<
    F: PrimeField32 + TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    DFT: TwoAdicSubgroupDft<F>,
    PG: ExampleHashAir<F, StirKeccakStarkConfig<F, EF, DFT>>,
>(
    proof_goal: &PG,
    dft: DFT,
    num_hashes: usize,
) -> StirKeccakTwoAdicResult<F, EF, DFT>
where
    StandardUniform: Distribution<F>,
{
    let val_mmcs = get_keccak_mmcs(3);
    let challenge_mmcs = ExtensionMmcs::<F, EF, _>::new(val_mmcs.clone());
    let stir_params = StirParameters {
        log_blowup: 1,
        log_folding_factor: 2,
        log_starting_folding_factor: 2,
        soundness_type: SecurityAssumption::CapacityBound,
        security_level: EXAMPLE_SECURITY_BITS,
        max_pow_bits: 20,
        mmcs: challenge_mmcs,
    };
    let (security_level, max_pow_bits) = (stir_params.security_level, stir_params.max_pow_bits);

    let trace = proof_goal.generate_trace_rows(num_hashes, stir_params.log_blowup);

    let pcs = TwoAdicStirPcs::new(dft, val_mmcs, stir_params)
        .with_batch_proof_of_work_bits(STIR_BATCH_POW_BITS);
    let challenger = SerializingChallenger32::from_hasher(vec![], Keccak256Hash {});

    let config = StirKeccakStarkConfig::new(pcs, challenger);

    let proof = prove(&config, proof_goal, trace, &[]);
    report_proof_size(&proof);

    let result = verify(&config, proof_goal, &proof, &[]);
    if result.is_ok() {
        report_stir_security_level(security_level, max_pow_bits);
    }
    result
}

/// Prove the given ProofGoal using the Poseidon2 hash function to build the merkle tree, with
/// STIR as the opening protocol.
///
/// This allows the user to choose:
/// - The Field
/// - The Proof Goal (Choice of Hash function and number of hashes to prove)
/// - The DFT
#[inline]
pub fn prove_monty31_poseidon2_stir<
    F: PrimeField32 + TwoAdicField + UniformSamplingField,
    EF: ExtensionField<F> + TwoAdicField,
    DFT: TwoAdicSubgroupDft<F>,
    Perm16: CryptographicPermutation<[F; 16]> + CryptographicPermutation<[F::Packing; 16]>,
    Perm24: CryptographicPermutation<[F; 24]> + CryptographicPermutation<[F::Packing; 24]>,
    PG: ExampleHashAir<F, StirPoseidon2StarkConfig<F, EF, DFT, Perm16, Perm24>>,
>(
    proof_goal: &PG,
    dft: DFT,
    num_hashes: usize,
    perm16: Perm16,
    perm24: Perm24,
) -> StirPoseidon2TwoAdicResult<F, EF, DFT, Perm16, Perm24>
where
    StandardUniform: Distribution<F>,
{
    let val_mmcs = get_poseidon2_mmcs::<F, _, _>(perm16, perm24.clone(), 3);
    let challenge_mmcs = ExtensionMmcs::<F, EF, _>::new(val_mmcs.clone());
    let stir_params = StirParameters {
        log_blowup: 1,
        log_folding_factor: 2,
        log_starting_folding_factor: 2,
        soundness_type: SecurityAssumption::CapacityBound,
        security_level: EXAMPLE_SECURITY_BITS,
        max_pow_bits: 20,
        mmcs: challenge_mmcs,
    };
    let (security_level, max_pow_bits) = (stir_params.security_level, stir_params.max_pow_bits);

    let trace = proof_goal.generate_trace_rows(num_hashes, stir_params.log_blowup);

    let pcs = TwoAdicStirPcs::new(dft, val_mmcs, stir_params)
        .with_batch_proof_of_work_bits(STIR_BATCH_POW_BITS);
    let challenger = DuplexChallenger::new(perm24);

    let config = StirPoseidon2StarkConfig::new(pcs, challenger);

    let proof = prove(&config, proof_goal, trace, &[]);
    report_proof_size(&proof);

    let result = verify(&config, proof_goal, &proof, &[]);
    if result.is_ok() {
        report_stir_security_level(security_level, max_pow_bits);
    }
    result
}

/// Prove the given ProofGoal using the Keccak hash function to build the merkle tree.
///
/// This fixes the field and Mersenne31 and makes use of the circle stark.
///
/// It currently allows the user to choose:
/// - The Proof Goal (Choice of Hash function and number of hashes to prove)
#[inline]
pub fn prove_m31_keccak<
    PG: ExampleHashAir<Mersenne31, KeccakCircleStarkConfig<Mersenne31, QM31>>
        + Air<SymbolicAirBuilder<Mersenne31, QM31>>,
>(
    proof_goal: &PG,
    num_hashes: usize,
) -> KeccakCircleResult {
    type F = Mersenne31;
    type EF = QM31;

    let val_mmcs = get_keccak_mmcs(0);
    let challenge_mmcs = ExtensionMmcs::<F, EF, _>::new(val_mmcs.clone());
    // Circle PCS only supports arity 2 (max_log_arity = 1)
    let fri_params = example_circle_parameters::<EF, _>(challenge_mmcs);
    let security_params = StarkSecurityParams::from_air::<F, EF, _>(
        fri_params.security_regime(),
        proof_goal,
        AirLayout::from_air(proof_goal),
        EF::bits(),
        128,
        2,
        OpeningShape::Circle,
        fri_params.grinding_sites(),
    );

    let trace = proof_goal.generate_trace_rows(num_hashes, fri_params.log_blowup);

    let pcs = CirclePcs::new(val_mmcs, fri_params);
    let challenger = SerializingChallenger32::from_hasher(vec![], Keccak256Hash {});

    let config = KeccakCircleStarkConfig::new(pcs, challenger);

    let proof = prove(&config, proof_goal, trace, &[]);
    report_proof_size(&proof);

    let result = verify(&config, proof_goal, &proof, &[]);
    if result.is_ok() {
        report_parameter_security(&proof, &security_params);
    }
    result
}

/// Prove the given ProofGoal using the Keccak hash function to build the merkle tree.
///
/// This fixes the field and Mersenne31 and makes use of the circle stark.
///
/// It currently allows the user to choose:
/// - The Proof Goal (Choice of Hash function and number of hashes to prove)
#[inline]
pub fn prove_m31_poseidon2<
    F: PrimeField64 + ComplexExtendable,
    EF: ExtensionField<F>,
    Perm16: CryptographicPermutation<[F; 16]> + CryptographicPermutation<[F::Packing; 16]>,
    Perm24: CryptographicPermutation<[F; 24]> + CryptographicPermutation<[F::Packing; 24]>,
    PG: ExampleHashAir<F, Poseidon2CircleStarkConfig<F, EF, Perm16, Perm24>>
        + Air<SymbolicAirBuilder<F, EF>>,
>(
    proof_goal: &PG,
    num_hashes: usize,
    perm16: Perm16,
    perm24: Perm24,
) -> Poseidon2CircleResult<F, EF, Perm16, Perm24>
where
    StandardUniform: Distribution<F>,
{
    let val_mmcs = get_poseidon2_mmcs::<F, _, _>(perm16, perm24.clone(), 0);

    let challenge_mmcs = ExtensionMmcs::<F, EF, _>::new(val_mmcs.clone());
    // Circle PCS only supports arity 2 (max_log_arity = 1)
    let fri_params = example_circle_parameters::<EF, _>(challenge_mmcs);
    let security_params = StarkSecurityParams::from_air::<F, EF, _>(
        fri_params.security_regime(),
        proof_goal,
        AirLayout::from_air(proof_goal),
        EF::bits(),
        128,
        2,
        OpeningShape::Circle,
        fri_params.grinding_sites(),
    );

    let trace = proof_goal.generate_trace_rows(num_hashes, fri_params.log_blowup);

    let pcs = CirclePcs::new(val_mmcs, fri_params);
    let challenger = DuplexChallenger::new(perm24);

    let config = Poseidon2CircleStarkConfig::new(pcs, challenger);

    let proof = prove(&config, proof_goal, trace, &[]);
    report_proof_size(&proof);

    let result = verify(&config, proof_goal, &proof, &[]);
    if result.is_ok() {
        report_parameter_security(&proof, &security_params);
    }
    result
}

/// Report the result of the proof.
///
/// Either print that the proof was successful or panic and return the error.
#[inline]
pub fn report_result(result: Result<(), impl Debug>) {
    if let Err(e) = result {
        panic!("{e:?}");
    } else {
        println!("Proof Verified Successfully");
    }
}

/// Report the size of the serialized proof.
///
/// Serializes the given proof instance using postcard and prints the size in bytes.
/// Panics if serialization fails.
#[inline]
pub fn report_proof_size<SC>(proof: &Proof<SC>)
where
    SC: StarkGenericConfig,
{
    let proof_bytes = postcard::to_allocvec(proof).expect("Failed to serialize proof");
    println!("Proof size: {} bytes", proof_bytes.len());
}

/// Report the security parameter of the proof.
///
/// Prints proven and conjectured security, followed by the historical legacy estimate.
#[inline]
pub fn report_parameter_security<SC>(proof: &Proof<SC>, security_params: &StarkSecurityParams)
where
    SC: StarkGenericConfig,
{
    let proven = proof.proven_security(security_params);
    println!(
        "Proven security: {} bits (UDR: {}, LDR: {})",
        proven.security_bits(),
        proven.unique_decoding_bits,
        proven.list_decoding_bits
    );
    println!(
        "Conjectured security: {} bits",
        proof.conjectured_security(security_params).security_bits
    );
    println!(
        "Legacy security (historical, not a bound): {} bits",
        proof.legacy_security(security_params).security_bits
    );
}

/// Report the security level the low-degree test was configured to target.
///
/// This is a commitment-scheme-level target, not the end-to-end figure the FRI path prints.
/// It excludes the DEEP-ALI, batching, and collision-resistance terms that figure folds in.
/// The two numbers are therefore not comparable.
///
/// The FRI path has a query count the caller picks, whose level is known after the fact.
/// Here every round's query count and grinding difficulty is derived from the target up front.
/// Nothing about the low-degree test is left to measure once the config exists.
///
/// What is missing is the whole-proof figure, which needs a proximity regime the STARK-level
/// accounting can consume.
#[inline]
pub fn report_stir_security_level(security_level: usize, max_pow_bits: usize) {
    println!(
        "STIR low-degree test configured to target {security_level} bits of conjectured \
         security ({max_pow_bits} bits of per-round grinding budget, \
         {STIR_BATCH_POW_BITS} batching grind bits); this excludes the STARK-level \
         (DEEP-ALI/batching) terms `--pcs fri` reports separately"
    );
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_koala_bear::KoalaBear;

    use super::*;

    fn check_fri_target<EF: Field>() {
        for params in [
            FriParameters::new_benchmark(()),
            FriParameters::new_benchmark_high_arity(()),
        ] {
            let params = example_fri_parameters::<EF, _>(params);
            let bits = |queries| {
                ConjecturedSecurity::compute_ldt_only(
                    params.log_blowup,
                    queries,
                    params.query_proof_of_work_bits,
                    128,
                    EF::bits(),
                )
                .security_bits
            };
            assert_eq!(bits(params.num_queries), 100);
            assert!(bits(params.num_queries - 1) < 100);
        }
    }

    #[test]
    fn fri_examples_use_the_fewest_queries_for_100_bits() {
        check_fri_target::<BinomialExtensionField<BabyBear, 4>>();
        check_fri_target::<BinomialExtensionField<KoalaBear, 4>>();
        check_fri_target::<QM31>();
    }

    #[test]
    fn circle_keccak_at_height_18_reaches_100_bits_including_batching() {
        let params = example_circle_parameters::<QM31, _>(());
        let air = p3_keccak_air::KeccakAir {};
        let security = StarkSecurityParams::from_air::<Mersenne31, QM31, _>(
            params.security_regime(),
            &air,
            AirLayout::from_air::<Mersenne31>(&air),
            QM31::bits(),
            128,
            2,
            OpeningShape::Circle,
            params.grinding_sites(),
        );
        assert_eq!(
            ConjecturedSecurity::compute_from_params(&security, 18).security_bits,
            100
        );
    }
}
