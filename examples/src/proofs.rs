use std::fmt::Debug;
use std::marker::PhantomData;

use p3_challenger::{DuplexChallenger, SerializingChallenger32};
use p3_circle::CirclePcs;
use p3_commit::ExtensionMmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::extension::BinomialExtensionField;
use p3_field::{ExtensionField, Field, PrimeField32, TwoAdicField};
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_keccak::{Keccak256Hash, KeccakF};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_mersenne_31::Mersenne31;
use p3_poseidon2::GenericPoseidon2LinearLayers;
use p3_symmetric::{
    CompressionFunctionFromHasher, CryptographicPermutation, PaddingFreeSponge,
    SerializingHasher32To64, TruncatedPermutation,
};
use p3_uni_stark::{prove, verify, StarkConfig, SymbolicExpression};
use rand::distributions::Standard;
use rand::prelude::Distribution;

use crate::airs::ProofObjective;

// Defining a bunch of types to keep clippy happy and avoid overly complex types.
const KECCAK_VECTOR_LEN: usize = p3_keccak::VECTOR_LEN;
type KeccakSerializingHasher32 = SerializingHasher32To64<PaddingFreeSponge<KeccakF, 25, 17, 4>>;
type KeccakCompressionFunction =
    CompressionFunctionFromHasher<PaddingFreeSponge<KeccakF, 25, 17, 4>, 2, 4>;
type KeccakSerializingChallenger32<F> =
    SerializingChallenger32<F, p3_challenger::HashChallenger<u8, Keccak256Hash, 32>>;
type KeccakMerkleMmcs<F> = MerkleTreeMmcs<
    [F; KECCAK_VECTOR_LEN],
    [u64; KECCAK_VECTOR_LEN],
    KeccakSerializingHasher32,
    KeccakCompressionFunction,
    4,
>;

/// Produce a MerkleTreeMmcs which uses the KeccakF permutation.
fn get_keccak_mmcs<F: Field>() -> KeccakMerkleMmcs<F> {
    let u64_hash = PaddingFreeSponge::<KeccakF, 25, 17, 4>::new(KeccakF {});

    let field_hash = SerializingHasher32To64::new(u64_hash);

    let compress = KeccakCompressionFunction::new(u64_hash);

    KeccakMerkleMmcs::new(field_hash, compress)
}

// Defining a bunch of types to keep clippy happy and avoid overly complex types.
type Poseidon2Sponge<Perm24> = PaddingFreeSponge<Perm24, 24, 16, 8>;
type Poseidon2Compression<Perm16> = TruncatedPermutation<Perm16, 2, 8, 16>;
type Poseidon2MerkleMmcs<F, Perm16, Perm24> = MerkleTreeMmcs<
    <F as Field>::Packing,
    <F as Field>::Packing,
    Poseidon2Sponge<Perm24>,
    Poseidon2Compression<Perm16>,
    8,
>;

/// Produce a MerkleTreeMmcs from a pair of cryptographic field permutations.
///
/// The first permutation will be used for compression and the second for more sponge hashing.
/// Currently this is only intended to be used with a pair of Poseidon2 hashes of with 16 and 24
/// but this can easily be generalised in future if we so desire.
fn get_poseidon2_mmcs<
    F: Field,
    Perm16: CryptographicPermutation<[F; 16]> + CryptographicPermutation<[F::Packing; 16]>,
    Perm24: CryptographicPermutation<[F; 24]> + CryptographicPermutation<[F::Packing; 24]>,
>(
    perm16: Perm16,
    perm24: Perm24,
) -> Poseidon2MerkleMmcs<F, Perm16, Perm24> {
    let hash = Poseidon2Sponge::new(perm24.clone());

    let compress = Poseidon2Compression::new(perm16.clone());

    Poseidon2MerkleMmcs::<F, _, _>::new(hash, compress)
}

type GenericExtensionFriConfig<F, EF, P, PW, H, C, const DIGEST_ELEMS: usize> =
    FriConfig<ExtensionMmcs<F, EF, MerkleTreeMmcs<P, PW, H, C, DIGEST_ELEMS>>>;

/// Produce a fri-config file from a MerkleTreeMmcs.
///
/// Currently all parameters in the config are fixed. These can easily be given by inputs if we want to customize them in some cases.
fn get_fri_config<F, EF, P: Clone, PW: Clone, H: Clone, C: Clone, const DIGEST_ELEMS: usize>(
    val_mmcs: MerkleTreeMmcs<P, PW, H, C, DIGEST_ELEMS>,
) -> GenericExtensionFriConfig<F, EF, P, PW, H, C, DIGEST_ELEMS> {
    let challenge_mmcs = ExtensionMmcs::<F, EF, _>::new(val_mmcs.clone());
    FriConfig {
        log_blowup: 1,
        num_queries: 100,
        proof_of_work_bits: 16,
        mmcs: challenge_mmcs,
    }
}

/// Make a pair of keccak based challengers using the SerializingChallenger32 construction.
fn get_keccak_challengers<F: PrimeField32>() -> (
    KeccakSerializingChallenger32<F>,
    KeccakSerializingChallenger32<F>,
) {
    let byte_hash = Keccak256Hash {};

    let proof_challenger = SerializingChallenger32::from_hasher(vec![], byte_hash);
    let verif_challenger = SerializingChallenger32::from_hasher(vec![], byte_hash);

    (proof_challenger, verif_challenger)
}

/// Make a pair of challengers from a given cryptographic permutation using the DuplexChallenger construction.
fn get_duplex_challengers<
    F: PrimeField32,
    Perm24: CryptographicPermutation<[F; 24]> + CryptographicPermutation<[F::Packing; 24]>,
>(
    perm24: Perm24,
) -> (
    DuplexChallenger<F, Perm24, 24, 16>,
    DuplexChallenger<F, Perm24, 24, 16>,
) {
    let proof_challenger = DuplexChallenger::new(perm24.clone());
    let verif_challenger = DuplexChallenger::new(perm24.clone());

    (proof_challenger, verif_challenger)
}

/// Prove the given ProofGoal using the Keccak hash function to build the merkle tree.
///
/// This allows the user to choose:
/// - The Field
/// - The Proof Goal (Choice of Hash function and number of hashes to prove)
/// - The DFT
#[inline]
pub fn prove_monty31_keccak<
    F: PrimeField32 + TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    DFT: TwoAdicSubgroupDft<F>,
    LinearLayers: GenericPoseidon2LinearLayers<F, P2_WIDTH>
        + GenericPoseidon2LinearLayers<SymbolicExpression<F>, P2_WIDTH>
        + GenericPoseidon2LinearLayers<F::Packing, P2_WIDTH>
        + GenericPoseidon2LinearLayers<EF, P2_WIDTH>,
    const P2_WIDTH: usize,
    const P2_SBOX_DEGREE: u64,
    const P2_SBOX_REGISTERS: usize,
    const P2_PARTIAL_ROUNDS: usize,
    const P2_HALF_FULL_ROUNDS: usize,
    const P2_VECTOR_LEN: usize,
>(
    proof_goal: ProofObjective<
        F,
        LinearLayers,
        P2_WIDTH,
        P2_SBOX_DEGREE,
        P2_SBOX_REGISTERS,
        P2_HALF_FULL_ROUNDS,
        P2_PARTIAL_ROUNDS,
        P2_VECTOR_LEN,
    >,
    dft: DFT,
    num_hashes: usize,
    _ef: PhantomData<EF>, // A simple workaround allowing the compiler to determine all generic parameters
) -> Result<(), impl Debug>
where
    Standard: Distribution<F>,
{
    let val_mmcs = get_keccak_mmcs();

    let fri_config = get_fri_config::<F, EF, _, _, _, _, 4>(val_mmcs.clone());

    let trace = proof_goal.generate_trace_rows(num_hashes);

    let pcs = TwoAdicFriPcs::new(dft, val_mmcs, fri_config);

    let config = StarkConfig::new(pcs);

    let (mut proof_challenger, mut verif_challenger) = get_keccak_challengers();

    let proof = prove(&config, &proof_goal, &mut proof_challenger, trace, &vec![]);
    verify(&config, &proof_goal, &mut verif_challenger, &proof, &vec![])
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
    EF: ExtensionField<F> + TwoAdicField,
    DFT: TwoAdicSubgroupDft<F>,
    LinearLayers: GenericPoseidon2LinearLayers<F, P2_WIDTH>
        + GenericPoseidon2LinearLayers<SymbolicExpression<F>, P2_WIDTH>
        + GenericPoseidon2LinearLayers<F::Packing, P2_WIDTH>
        + GenericPoseidon2LinearLayers<EF, P2_WIDTH>,
    Perm16: CryptographicPermutation<[F; 16]> + CryptographicPermutation<[F::Packing; 16]>,
    Perm24: CryptographicPermutation<[F; 24]> + CryptographicPermutation<[F::Packing; 24]>,
    const P2_WIDTH: usize,
    const P2_SBOX_DEGREE: u64,
    const P2_SBOX_REGISTERS: usize,
    const P2_PARTIAL_ROUNDS: usize,
    const P2_HALF_FULL_ROUNDS: usize,
    const P2_VECTOR_LEN: usize,
>(
    proof_goal: ProofObjective<
        F,
        LinearLayers,
        P2_WIDTH,
        P2_SBOX_DEGREE,
        P2_SBOX_REGISTERS,
        P2_HALF_FULL_ROUNDS,
        P2_PARTIAL_ROUNDS,
        P2_VECTOR_LEN,
    >,
    dft: DFT,
    num_hashes: usize,
    perm16: Perm16,
    perm24: Perm24,
    _ef: PhantomData<EF>, // A simple workaround allowing the compiler to determine all generic parameters
) -> Result<(), impl Debug>
where
    Standard: Distribution<F>,
{
    let val_mmcs = get_poseidon2_mmcs::<F, _, _>(perm16, perm24.clone());
    let fri_config = get_fri_config::<F, EF, _, _, _, _, 8>(val_mmcs.clone());

    let trace = proof_goal.generate_trace_rows(num_hashes);

    let pcs = TwoAdicFriPcs::new(dft, val_mmcs, fri_config);

    let config = StarkConfig::new(pcs);

    let (mut proof_challenger, mut verif_challenger) = get_duplex_challengers(perm24);

    let proof = prove(&config, &proof_goal, &mut proof_challenger, trace, &vec![]);
    verify(&config, &proof_goal, &mut verif_challenger, &proof, &vec![])
}

/// Prove the given ProofGoal using the Keccak hash function to build the merkle tree.
///
/// This allows the user to choose:
/// - The Proof Goal (Choice of Hash function and number of hashes to prove)
#[inline]
pub fn prove_m31_keccak<
    LinearLayers: GenericPoseidon2LinearLayers<Mersenne31, P2_WIDTH>
        + GenericPoseidon2LinearLayers<SymbolicExpression<Mersenne31>, P2_WIDTH>
        + GenericPoseidon2LinearLayers<<Mersenne31 as Field>::Packing, P2_WIDTH>
        + GenericPoseidon2LinearLayers<BinomialExtensionField<Mersenne31, 3>, P2_WIDTH>,
    const P2_WIDTH: usize,
    const P2_SBOX_DEGREE: u64,
    const P2_SBOX_REGISTERS: usize,
    const P2_PARTIAL_ROUNDS: usize,
    const P2_HALF_FULL_ROUNDS: usize,
    const P2_VECTOR_LEN: usize,
>(
    proof_goal: ProofObjective<
        Mersenne31,
        LinearLayers,
        P2_WIDTH,
        P2_SBOX_DEGREE,
        P2_SBOX_REGISTERS,
        P2_HALF_FULL_ROUNDS,
        P2_PARTIAL_ROUNDS,
        P2_VECTOR_LEN,
    >,
    num_hashes: usize,
) -> Result<(), impl Debug> {
    type F = Mersenne31;
    type EF = BinomialExtensionField<Mersenne31, 3>;

    let val_mmcs = get_keccak_mmcs();

    let fri_config = get_fri_config::<F, EF, _, _, _, _, 4>(val_mmcs.clone());

    let trace = proof_goal.generate_trace_rows(num_hashes);

    let pcs = CirclePcs::new(val_mmcs, fri_config);

    let config = StarkConfig::new(pcs);

    let (mut proof_challenger, mut verif_challenger) = get_keccak_challengers();

    let proof = prove(&config, &proof_goal, &mut proof_challenger, trace, &vec![]);
    verify(&config, &proof_goal, &mut verif_challenger, &proof, &vec![])
}

/// Prove the given ProofGoal using the Keccak hash function to build the merkle tree.
///
/// This allows the user to choose:
/// - The Proof Goal (Choice of Hash function and number of hashes to prove)
#[inline]
pub fn prove_m31_poseidon2<
    LinearLayers: GenericPoseidon2LinearLayers<Mersenne31, P2_WIDTH>
        + GenericPoseidon2LinearLayers<SymbolicExpression<Mersenne31>, P2_WIDTH>
        + GenericPoseidon2LinearLayers<<Mersenne31 as Field>::Packing, P2_WIDTH>
        + GenericPoseidon2LinearLayers<BinomialExtensionField<Mersenne31, 3>, P2_WIDTH>,
    Perm16: CryptographicPermutation<[Mersenne31; 16]>
        + CryptographicPermutation<[<Mersenne31 as Field>::Packing; 16]>,
    Perm24: CryptographicPermutation<[Mersenne31; 24]>
        + CryptographicPermutation<[<Mersenne31 as Field>::Packing; 24]>,
    const P2_WIDTH: usize,
    const P2_SBOX_DEGREE: u64,
    const P2_SBOX_REGISTERS: usize,
    const P2_PARTIAL_ROUNDS: usize,
    const P2_HALF_FULL_ROUNDS: usize,
    const P2_VECTOR_LEN: usize,
>(
    proof_goal: ProofObjective<
        Mersenne31,
        LinearLayers,
        P2_WIDTH,
        P2_SBOX_DEGREE,
        P2_SBOX_REGISTERS,
        P2_HALF_FULL_ROUNDS,
        P2_PARTIAL_ROUNDS,
        P2_VECTOR_LEN,
    >,
    num_hashes: usize,
    perm16: Perm16,
    perm24: Perm24,
) -> Result<(), impl Debug> {
    type F = Mersenne31;
    type EF = BinomialExtensionField<Mersenne31, 3>;

    let val_mmcs = get_poseidon2_mmcs::<F, _, _>(perm16, perm24.clone());
    let fri_config = get_fri_config::<F, EF, _, _, _, _, 8>(val_mmcs.clone());

    let trace = proof_goal.generate_trace_rows(num_hashes);

    let pcs = CirclePcs::new(val_mmcs, fri_config);

    let config = StarkConfig::new(pcs);

    let (mut proof_challenger, mut verif_challenger) = get_duplex_challengers(perm24);

    let proof = prove(&config, &proof_goal, &mut proof_challenger, trace, &vec![]);
    verify(&config, &proof_goal, &mut verif_challenger, &proof, &vec![])
}

#[inline]
pub fn report_result(result: Result<(), impl Debug>) {
    if result.is_ok() {
        println!("Proof Verified Successfully")
    } else {
        panic!("{:?}", result.unwrap_err())
    }
}
