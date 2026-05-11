//! STARK proving configurations.
//!
//! This module provides STARK configurations for different prime fields.
//!
//! # Quick Start
//!
//! ```ignore
//! use p3_circuit_prover::config;
//!
//! // Use a preconfigured setup
//! let config = config::baby_bear()
//!     .build();
//! ```

use core::marker::PhantomData;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear, default_babybear_poseidon2_16};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64, TwoAdicField};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear, default_koalabear_poseidon2_16};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicPermutation, PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::StarkConfig;

/// Compression function arity (number of inputs per compression).
const COMPRESS_ARITY: usize = 2;

/// A STARK configuration with all cryptographic primitives specified.
///
/// ### Type Parameters
/// - `F`: Base field.
/// - `PermHash`: Permutation function used for sponge hashing (leaves, transcript absorption).
/// - `PermCompress`: Permutation function used for Merkle tree compression.
/// - `HASH_PERM_WIDTH`: Width of the hash permutation state.
/// - `COMPRESS_PERM_WIDTH`: Width of the compression permutation state.
/// - `RATE`: Number of field elements absorbed per permutation in sponge mode.
/// - `OUT`: Number of output elements squeezed per permutation.
/// - `COMPRESS_CHUNK`: Number of elements per compression chunk in Merkle commitments.
/// - `CHALLENGE_DEGREE`: Extension field degree.
pub type Config<
    F,
    PermHash,
    PermCompress,
    const HASH_PERM_WIDTH: usize,
    const COMPRESS_PERM_WIDTH: usize,
    const RATE: usize,
    const OUT: usize,
    const COMPRESS_CHUNK: usize,
    const CHALLENGE_DEGREE: usize,
> = StarkConfig<
    TwoAdicFriPcs<
        F,
        Radix2DitParallel<F>,
        MerkleTreeMmcs<
            F,
            F,
            PaddingFreeSponge<PermHash, HASH_PERM_WIDTH, RATE, OUT>,
            TruncatedPermutation<PermCompress, COMPRESS_ARITY, COMPRESS_CHUNK, COMPRESS_PERM_WIDTH>,
            2,
            COMPRESS_CHUNK,
        >,
        ExtensionMmcs<
            F,
            BinomialExtensionField<F, CHALLENGE_DEGREE>,
            MerkleTreeMmcs<
                F,
                F,
                PaddingFreeSponge<PermHash, HASH_PERM_WIDTH, RATE, OUT>,
                TruncatedPermutation<
                    PermCompress,
                    COMPRESS_ARITY,
                    COMPRESS_CHUNK,
                    COMPRESS_PERM_WIDTH,
                >,
                2,
                COMPRESS_CHUNK,
            >,
        >,
    >,
    BinomialExtensionField<F, CHALLENGE_DEGREE>,
    DuplexChallenger<F, PermHash, HASH_PERM_WIDTH, RATE>,
>;

/// Configuration builder for STARK provers.
pub struct ConfigBuilder<
    F,
    PermHash,
    PermCompress,
    const HASH_PERM_WIDTH: usize,
    const COMPRESS_PERM_WIDTH: usize,
    const RATE: usize,
    const OUT: usize,
    const COMPRESS_CHUNK: usize,
    const CHALLENGE_DEGREE: usize,
> {
    perm_hash: PermHash,
    perm_compress: PermCompress,
    _phantom: PhantomData<F>,
}

impl<
    F,
    PermHash,
    PermCompress,
    const HASH_PERM_WIDTH: usize,
    const COMPRESS_PERM_WIDTH: usize,
    const RATE: usize,
    const OUT: usize,
    const COMPRESS_CHUNK: usize,
    const CHALLENGE_DEGREE: usize,
>
    ConfigBuilder<
        F,
        PermHash,
        PermCompress,
        HASH_PERM_WIDTH,
        COMPRESS_PERM_WIDTH,
        RATE,
        OUT,
        COMPRESS_CHUNK,
        CHALLENGE_DEGREE,
    >
where
    F: Field,
    PermHash: Clone + CryptographicPermutation<[F; HASH_PERM_WIDTH]>,
    PermCompress: Clone + CryptographicPermutation<[F; COMPRESS_PERM_WIDTH]>,
{
    const fn new(perm_hash: PermHash, perm_compress: PermCompress) -> Self {
        Self {
            perm_hash,
            perm_compress,
            _phantom: PhantomData,
        }
    }

    /// Builds the final STARK configuration.
    pub fn build(
        self,
    ) -> Config<
        F,
        PermHash,
        PermCompress,
        HASH_PERM_WIDTH,
        COMPRESS_PERM_WIDTH,
        RATE,
        OUT,
        COMPRESS_CHUNK,
        CHALLENGE_DEGREE,
    > {
        type Hash<Perm, const PERM_WIDTH: usize, const RATE: usize, const OUT: usize> =
            PaddingFreeSponge<Perm, PERM_WIDTH, RATE, OUT>;
        type Compress<Perm, const PERM_WIDTH: usize, const COMPRESS_CHUNK: usize> =
            TruncatedPermutation<Perm, COMPRESS_ARITY, COMPRESS_CHUNK, PERM_WIDTH>;

        let hash = Hash::<PermHash, HASH_PERM_WIDTH, RATE, OUT>::new(self.perm_hash.clone());
        let compress = Compress::<PermCompress, COMPRESS_PERM_WIDTH, COMPRESS_CHUNK>::new(
            self.perm_compress.clone(),
        );
        let val_mmcs = MerkleTreeMmcs::new(hash, compress, 3);
        let challenge_mmcs = ExtensionMmcs::new(val_mmcs.clone());
        let dft = Radix2DitParallel::default();
        let fri_params = FriParameters::new_benchmark_high_arity(challenge_mmcs);
        let pcs = TwoAdicFriPcs::new(dft, val_mmcs, fri_params);
        let challenger = DuplexChallenger::new(self.perm_hash);

        StarkConfig::new(pcs, challenger)
    }
}

/// Creates a standard BabyBear configuration.
///
/// BabyBear is a 31-bit prime field (2^31 - 2^27 + 1).
///
/// # Parameters
/// - **Hash permutation width**: 16 (appropriate for 32-bit fields)
/// - **Compression permutation width**: 16
/// - **Rate**: 8 (256 bits / 32 bits per element)
/// - **Output size**: 8 (256 bits / 32 bits per element)
/// - **Challenge degree**: 4
///
/// # Examples
///
/// ```ignore
/// let config = config::baby_bear().build();
/// let prover = BatchStarkProver::new(config);
/// ```
#[inline]
pub fn baby_bear()
-> ConfigBuilder<BabyBear, Poseidon2BabyBear<16>, Poseidon2BabyBear<16>, 16, 16, 8, 8, 8, 4> {
    let perm = default_babybear_poseidon2_16();
    ConfigBuilder::new(perm.clone(), perm)
}

/// Creates a standard KoalaBear configuration.
///
/// KoalaBear is a 31-bit prime field (2^31 - 2^24 + 1).
///
/// # Parameters
/// - **Hash permutation width**: 16 (appropriate for 32-bit fields)
/// - **Compression permutation width**: 16
/// - **Rate**: 8 (256 bits / 32 bits per element)
/// - **Output size**: 8 (256 bits / 32 bits per element)
/// - **Challenge degree**: 4
///
/// # Examples
///
/// ```ignore
/// let config = config::koala_bear().build();
/// let prover = BatchStarkProver::new(config);
/// ```
#[inline]
pub fn koala_bear()
-> ConfigBuilder<KoalaBear, Poseidon2KoalaBear<16>, Poseidon2KoalaBear<16>, 16, 16, 8, 8, 8, 4> {
    let perm = default_koalabear_poseidon2_16();
    ConfigBuilder::new(perm.clone(), perm)
}

/// Creates a standard Goldilocks configuration.
///
/// Goldilocks is a 64-bit prime field (2^64 - 2^32 + 1).
///
/// # Parameters
/// - **Hash permutation width**: 8 (appropriate for 64-bit fields)
/// - **Compression permutation width**: 8
/// - **Rate**: 4 (256 bits / 64 bits per element)
/// - **Output size**: 4 (256 bits / 64 bits per element)
/// - **Challenge degree**: 2
///
/// # Examples
///
/// ```ignore
/// let config = config::goldilocks().build();
/// let prover = BatchStarkProver::new(config);
/// ```
#[inline]
pub fn goldilocks()
-> ConfigBuilder<Goldilocks, Poseidon2Goldilocks<8>, Poseidon2Goldilocks<8>, 8, 8, 4, 4, 4, 2> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(1);
    let perm = p3_goldilocks::Poseidon2Goldilocks::<8>::new_from_rng_128(&mut rng);
    ConfigBuilder::new(perm.clone(), perm)
}

/// Type alias for BabyBear STARK configuration.
pub type BabyBearConfig =
    Config<BabyBear, Poseidon2BabyBear<16>, Poseidon2BabyBear<16>, 16, 16, 8, 8, 8, 4>;

/// Type alias for KoalaBear STARK configuration.
pub type KoalaBearConfig =
    Config<KoalaBear, Poseidon2KoalaBear<16>, Poseidon2KoalaBear<16>, 16, 16, 8, 8, 8, 4>;

/// Type alias for Goldilocks STARK configuration.
pub type GoldilocksConfig =
    Config<Goldilocks, Poseidon2Goldilocks<8>, Poseidon2Goldilocks<8>, 8, 8, 4, 4, 4, 2>;

/// Trait bounds for STARK-compatible fields.
pub trait StarkField: Field + PrimeCharacteristicRing + TwoAdicField + PrimeField64 {}

impl<F> StarkField for F where F: Field + PrimeCharacteristicRing + TwoAdicField + PrimeField64 {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_fields_configs_compile() {
        let _bb: BabyBearConfig = baby_bear().build();
        let _kb: KoalaBearConfig = koala_bear().build();
        let _gl: GoldilocksConfig = goldilocks().build();
    }
}
