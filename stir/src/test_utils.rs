use alloc::vec::Vec;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_field::extension::BinomialExtensionField;
use p3_field::Field;
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::{SecurityAssumption, StirConfig, StirParameters};

// Security levels used in the tests in bits.

/// The quintic extension of BabyBear is large enough that, using the
/// `CapacityBound` assumption, one gets manageable proof-of-work bit numbers
/// even when requiring 128 bits of security
pub const BB_EXT_SEC_LEVEL: usize = 128;

/// For unconditional (i. e. without relying on any assumptions) security,
/// setting the security level to 100 bits produces more palatable proof-of-work
/// challenges.
pub const BB_EXT_SEC_LEVEL_LOWER: usize = 100;

/// The quadratic extension of Goldilocks is smaller than the other two fields,
/// so we set the tests to 80 bits of security to get reasonable proof-of-work
/// challenges.
pub const GL_EXT_SEC_LEVEL: usize = 80;

/// The BabyBear field
pub type Bb = BabyBear;

/// A quintic extension of BabyBear
pub type BbExt = BinomialExtensionField<Bb, 5>;

type BbPerm = Poseidon2BabyBear<16>;
type BbHash = PaddingFreeSponge<BbPerm, 16, 8, 8>;
type BbCompress = TruncatedPermutation<BbPerm, 2, 8, 16>;
type BbPacking = <Bb as Field>::Packing;

type BbMmcs = MerkleTreeMmcs<BbPacking, BbPacking, BbHash, BbCompress, 8>;

/// A Mixed Matrix Commitment Scheme over the quintic extension of BabyBear
pub type BbExtMmcs = ExtensionMmcs<Bb, BbExt, BbMmcs>;

/// A challenger for the BabyBear field and its quintic extension
pub type BbChallenger = DuplexChallenger<Bb, BbPerm, 16, 8>;

/// The Goldilocks field
pub type Gl = Goldilocks;

/// A quadratic extension of Goldilocks
pub type GlExt = BinomialExtensionField<Gl, 2>;

type GlPerm = Poseidon2Goldilocks<8>;
type GlHash = PaddingFreeSponge<GlPerm, 8, 4, 4>;
type GlCompress = TruncatedPermutation<GlPerm, 2, 4, 8>;
type GlPacking = <Gl as Field>::Packing;

type GlMmcs = MerkleTreeMmcs<GlPacking, GlPacking, GlHash, GlCompress, 4>;

/// A Mixed Matrix Commitment Scheme over the quadratic extension of Goldilocks
pub type GlExtMmcs = ExtensionMmcs<Gl, GlExt, GlMmcs>;

/// A challenger for the Goldilocks field and its quadratic extension
pub type GlChallenger = DuplexChallenger<Gl, GlPerm, 8, 4>;

// This produces an MMCS for the chosen field. Computing it in a macro avoids
// some generic-related pains. We seed the generator in order to make the tests
// deterministic, but this is not necessary.
macro_rules! impl_test_mmcs_config {
    ($name:ident, $ext_mmcs:ty, $perm:ty, $hash:ty, $compress:ty, $mmcs:ty) => {
        pub fn $name() -> $ext_mmcs {
            let mut rng = ChaCha20Rng::seed_from_u64(0);
            let perm = <$perm>::new_from_rng_128(&mut rng);
            let hash = <$hash>::new(perm.clone());
            let compress = <$compress>::new(perm.clone());
            <$ext_mmcs>::new(<$mmcs>::new(hash, compress))
        }
    };
}

// This produces a challenger for the chosen field and its extension. We seed
// the generator in order to make the tests deterministic, but this is not
// necessary.
macro_rules! impl_test_challenger {
    ($name:ident, $challenger:ty, $perm:ty) => {
        pub fn $name() -> $challenger {
            let mut rng = ChaCha20Rng::seed_from_u64(0);
            let perm = <$perm>::new_from_rng_128(&mut rng);
            <$challenger>::new(perm)
        }
    };
}

// Simple wrapper around StirParameters::constant_folding_factor
macro_rules! impl_test_stir_config {
    ($name:ident, $ext:ty, $ext_mmcs:ty, $mmcs_config_fn:ident) => {
        pub fn $name(
            security_level: usize,
            security_assumption: SecurityAssumption,
            log_starting_degree: usize,
            log_starting_inv_rate: usize,
            log_folding_factor: usize,
            num_rounds: usize,
        ) -> StirConfig<$ext, $ext_mmcs> {
            let pow_bits = 20;

            let parameters = StirParameters::constant_folding_factor(
                (security_level, security_assumption),
                log_starting_degree,
                log_starting_inv_rate,
                log_folding_factor,
                num_rounds,
                pow_bits,
                $mmcs_config_fn(),
            );

            StirConfig::new(parameters)
        }
    };
}

// Simple wrapper around StirParameters::variable_folding_factor
macro_rules! impl_test_stir_config_folding_factors {
    ($name:ident, $ext:ty, $ext_mmcs:ty, $mmcs_config_fn:ident) => {
        pub fn $name(
            security_level: usize,
            security_assumption: SecurityAssumption,
            log_starting_degree: usize,
            log_starting_inv_rate: usize,
            log_folding_factors: Vec<usize>,
        ) -> StirConfig<$ext, $ext_mmcs> {
            let pow_bits = 20;

            let parameters = StirParameters::variable_folding_factor(
                (security_level, security_assumption),
                log_starting_degree,
                log_starting_inv_rate,
                log_folding_factors,
                pow_bits,
                $mmcs_config_fn(),
            );

            StirConfig::new(parameters)
        }
    };
}

impl_test_mmcs_config!(
    test_bb_mmcs_config,
    BbExtMmcs,
    BbPerm,
    BbHash,
    BbCompress,
    BbMmcs
);

impl_test_mmcs_config!(
    test_gl_mmcs_config,
    GlExtMmcs,
    GlPerm,
    GlHash,
    GlCompress,
    GlMmcs
);

impl_test_challenger!(test_bb_challenger, BbChallenger, BbPerm);
impl_test_challenger!(test_gl_challenger, GlChallenger, GlPerm);

impl_test_stir_config!(test_bb_stir_config, BbExt, BbExtMmcs, test_bb_mmcs_config);
impl_test_stir_config!(test_gl_stir_config, GlExt, GlExtMmcs, test_gl_mmcs_config);

impl_test_stir_config_folding_factors!(
    test_bb_stir_config_folding_factors,
    BbExt,
    BbExtMmcs,
    test_bb_mmcs_config
);

impl_test_stir_config_folding_factors!(
    test_gl_stir_config_folding_factors,
    GlExt,
    GlExtMmcs,
    test_gl_mmcs_config
);
