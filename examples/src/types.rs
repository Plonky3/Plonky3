use p3_air::{Air, BaseAir};
use p3_challenger::{DuplexChallenger, SerializingChallenger32};
use p3_circle::CirclePcs;
use p3_commit::ExtensionMmcs;
use p3_field::{extension::ComplexExtendable, ExtensionField, Field, PrimeField64};
use p3_keccak::{Keccak256Hash, KeccakF};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{
    CompressionFunctionFromHasher, CryptographicPermutation, PaddingFreeSponge,
    SerializingHasher32To64, TruncatedPermutation,
};
use p3_uni_stark::{
    DebugConstraintBuilder, ProverConstraintFolder, StarkConfig, SymbolicAirBuilder,
    VerifierConstraintFolder,
};

use crate::airs::GenerableTrace;

// Defining a bunch of types to keep clippy happy and avoid overly complex types.
pub(crate) const KECCAK_VECTOR_LEN: usize = p3_keccak::VECTOR_LEN;
pub(crate) type KeccakSerializingHasher32 =
    SerializingHasher32To64<PaddingFreeSponge<KeccakF, 25, 17, 4>>;
pub(crate) type KeccakCompressionFunction =
    CompressionFunctionFromHasher<PaddingFreeSponge<KeccakF, 25, 17, 4>, 2, 4>;
pub(crate) type KeccakSerializingChallenger32<F> =
    SerializingChallenger32<F, p3_challenger::HashChallenger<u8, Keccak256Hash, 32>>;
pub(crate) type KeccakMerkleMmcs<F> = MerkleTreeMmcs<
    [F; KECCAK_VECTOR_LEN],
    [u64; KECCAK_VECTOR_LEN],
    KeccakSerializingHasher32,
    KeccakCompressionFunction,
    4,
>;

// Defining a bunch of types to keep clippy happy and avoid overly complex types.
pub(crate) type Poseidon2Sponge<Perm24> = PaddingFreeSponge<Perm24, 24, 16, 8>;
pub(crate) type Poseidon2Compression<Perm16> = TruncatedPermutation<Perm16, 2, 8, 16>;
pub(crate) type Poseidon2MerkleMmcs<F, Perm16, Perm24> = MerkleTreeMmcs<
    <F as Field>::Packing,
    <F as Field>::Packing,
    Poseidon2Sponge<Perm24>,
    Poseidon2Compression<Perm16>,
    8,
>;
pub(crate) type Poseidon2CircleStarkConfig<F, EF, Perm16, Perm24> = StarkConfig<
    CirclePcs<
        F,
        Poseidon2MerkleMmcs<F, Perm16, Perm24>,
        ExtensionMmcs<F, EF, Poseidon2MerkleMmcs<F, Perm16, Perm24>>,
    >,
    EF,
    DuplexChallenger<F, Perm24, 24, 16>,
>;

pub trait ExampleAirBasedCircleMerklePoseidon2<
    F: PrimeField64 + ComplexExtendable,
    EF: ExtensionField<F>,
    Perm16: CryptographicPermutation<[F; 16]> + CryptographicPermutation<[F::Packing; 16]>,
    Perm24: CryptographicPermutation<[F; 24]> + CryptographicPermutation<[F::Packing; 24]>,
>:
    GenerableTrace<F>
    + BaseAir<F>
    + for<'a> Air<DebugConstraintBuilder<'a, F>>
    + Air<SymbolicAirBuilder<F>>
    + for<'a> Air<ProverConstraintFolder<'a, Poseidon2CircleStarkConfig<F, EF, Perm16, Perm24>>>
    + for<'a> Air<VerifierConstraintFolder<'a, Poseidon2CircleStarkConfig<F, EF, Perm16, Perm24>>>
{
}
