use std::fmt::Debug;
use std::marker::PhantomData;

use p3_challenger::{DuplexChallenger, SerializingChallenger32};
use p3_commit::ExtensionMmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, PrimeField32, TwoAdicField};
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_keccak::{Keccak256Hash, KeccakF};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_poseidon2::GenericPoseidon2LinearLayers;
use p3_symmetric::{
    CompressionFunctionFromHasher, CryptographicPermutation, PaddingFreeSponge,
    SerializingHasher32To64, TruncatedPermutation,
};
use p3_uni_stark::{prove, verify, StarkConfig, SymbolicExpression};
use rand::distributions::Standard;
use rand::prelude::Distribution;

use crate::airs::ProofGoal;

/// Prove the given ProofGoal using the Keccak hash function to build the merkle tree.
///
/// This allows the user to choose:
/// - The Field
/// - The Proof Goal (Choice of Hash function and number of hashes to prove)
/// - The DFT
#[inline]
pub fn prove_hashes_keccak<
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
    proof_goal: ProofGoal<
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
    _phantom: PhantomData<EF>, // A simple workaround allowing the compiler to determine all generic parameters
) -> Result<(), impl Debug>
where
    Standard: Distribution<F>,
{
    let byte_hash = Keccak256Hash {};
    let u64_hash = PaddingFreeSponge::<KeccakF, 25, 17, 4>::new(KeccakF {});
    let field_hash = SerializingHasher32To64::new(u64_hash);

    let compress = CompressionFunctionFromHasher::<_, 2, 4>::new(u64_hash);

    let val_mmcs =
        MerkleTreeMmcs::<[F; p3_keccak::VECTOR_LEN], [u64; p3_keccak::VECTOR_LEN], _, _, 4>::new(
            field_hash, compress,
        );

    let challenge_mmcs = ExtensionMmcs::<F, EF, _>::new(val_mmcs.clone());

    let trace = proof_goal.generate_trace_rows(num_hashes);

    let fri_config = FriConfig {
        log_blowup: 1,
        num_queries: 100,
        proof_of_work_bits: 16,
        mmcs: challenge_mmcs,
    };

    let pcs = TwoAdicFriPcs::new(dft, val_mmcs, fri_config);

    let config = StarkConfig::new(pcs);

    let mut proof_challenger = SerializingChallenger32::from_hasher(vec![], byte_hash);
    let mut verif_challenger = SerializingChallenger32::from_hasher(vec![], byte_hash);

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
pub fn prove_hashes_poseidon2<
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
    proof_goal: ProofGoal<
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
    _phantom: PhantomData<EF>, // A simple workaround allowing the compiler to determine all generic parameters
) -> Result<(), impl Debug>
where
    Standard: Distribution<F>,
{
    let hash = PaddingFreeSponge::<_, 24, 16, 8>::new(perm24.clone());

    let compress = TruncatedPermutation::new(perm16.clone());

    let val_mmcs = MerkleTreeMmcs::<F::Packing, F::Packing, _, _, 8>::new(hash, compress);

    let challenge_mmcs = ExtensionMmcs::<F, EF, _>::new(val_mmcs.clone());

    let trace = proof_goal.generate_trace_rows(num_hashes);

    let fri_config = FriConfig {
        log_blowup: 1,
        num_queries: 100,
        proof_of_work_bits: 16,
        mmcs: challenge_mmcs,
    };

    let pcs = TwoAdicFriPcs::new(dft, val_mmcs, fri_config);

    let config = StarkConfig::new(pcs);

    let mut proof_challenger = DuplexChallenger::<_, _, 24, 16>::new(perm24.clone());
    let mut verif_challenger = DuplexChallenger::new(perm24.clone());

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
