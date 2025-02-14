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
pub type BB = BabyBear;

/// A quintic extension of BabyBear
pub type BBExt = BinomialExtensionField<BB, 5>;

type BBPerm = Poseidon2BabyBear<16>;
type BBHash = PaddingFreeSponge<BBPerm, 16, 8, 8>;
type BBCompress = TruncatedPermutation<BBPerm, 2, 8, 16>;
type BBPacking = <BB as Field>::Packing;

type BBMMCS = MerkleTreeMmcs<BBPacking, BBPacking, BBHash, BBCompress, 8>;

/// A Mixed Matrix Commitment Scheme over the quintic extension of BabyBear
pub type BBExtMMCS = ExtensionMmcs<BB, BBExt, BBMMCS>;

/// A challenger for the BabyBear field and its quintic extension
pub type BBChallenger = DuplexChallenger<BB, BBPerm, 16, 8>;

/// The Goldilocks field
pub type GL = Goldilocks;

/// A quadratic extension of Goldilocks
pub type GLExt = BinomialExtensionField<GL, 2>;

type GLPerm = Poseidon2Goldilocks<8>;
type GLHash = PaddingFreeSponge<GLPerm, 8, 4, 4>;
type GLCompress = TruncatedPermutation<GLPerm, 2, 4, 8>;
type GLPacking = <GL as Field>::Packing;

type GLMMCS = MerkleTreeMmcs<GLPacking, GLPacking, GLHash, GLCompress, 4>;

/// A Mixed Matrix Commitment Scheme over the quadratic extension of Goldilocks
pub type GLExtMMCS = ExtensionMmcs<GL, GLExt, GLMMCS>;

/// A challenger for the Goldilocks field and its quadratic extension
pub type GLChallenger = DuplexChallenger<GL, GLPerm, 8, 4>;

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
        ) -> StirConfig<$ext_mmcs> {
            let pow_bits = 20;

            let parameters = StirParameters::constant_folding_factor(
                log_starting_degree,
                log_starting_inv_rate,
                log_folding_factor,
                num_rounds,
                security_assumption,
                security_level,
                pow_bits,
                $mmcs_config_fn(),
            );

            StirConfig::new::<$ext>(parameters)
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
        ) -> StirConfig<$ext_mmcs> {
            let pow_bits = 20;

            let parameters = StirParameters::variable_folding_factor(
                log_starting_degree,
                log_starting_inv_rate,
                log_folding_factors,
                security_assumption,
                security_level,
                pow_bits,
                $mmcs_config_fn(),
            );

            StirConfig::new::<$ext>(parameters)
        }
    };
}

impl_test_mmcs_config!(
    test_bb_mmcs_config,
    BBExtMMCS,
    BBPerm,
    BBHash,
    BBCompress,
    BBMMCS
);

impl_test_mmcs_config!(
    test_gl_mmcs_config,
    GLExtMMCS,
    GLPerm,
    GLHash,
    GLCompress,
    GLMMCS
);

impl_test_challenger!(test_bb_challenger, BBChallenger, BBPerm);
impl_test_challenger!(test_gl_challenger, GLChallenger, GLPerm);

impl_test_stir_config!(test_bb_stir_config, BBExt, BBExtMMCS, test_bb_mmcs_config);
impl_test_stir_config!(test_gl_stir_config, GLExt, GLExtMMCS, test_gl_mmcs_config);

impl_test_stir_config_folding_factors!(
    test_bb_stir_config_folding_factors,
    BBExt,
    BBExtMMCS,
    test_bb_mmcs_config
);

impl_test_stir_config_folding_factors!(
    test_gl_stir_config_folding_factors,
    GLExt,
    GLExtMMCS,
    test_gl_mmcs_config
);
