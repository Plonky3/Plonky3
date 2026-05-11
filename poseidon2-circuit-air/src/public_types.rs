//! Field-specific configurations and type aliases for the Poseidon2 circuit AIR.
//!
//! # Supported Configurations
//!
//! ```text
//!     Field       Extension degree   State width   Partial rounds
//!     ─────────   ────────────────   ───────────   ──────────────
//!     BabyBear    1                  16            13
//!     BabyBear    4                  16            13
//!     BabyBear    4                  24            21
//!     KoalaBear   1                  16            20
//!     KoalaBear   4                  16            20
//!     KoalaBear   4                  24            23
//!     Goldilocks  2                   8            22
//! ```

extern crate alloc;

use alloc::vec::Vec;

use p3_baby_bear::{BabyBear, GenericPoseidon2LinearLayersBabyBear};
use p3_circuit::ops::{GoldilocksD2Width8, Poseidon2Config, Poseidon2Params};
use p3_goldilocks::{GenericPoseidon2LinearLayersGoldilocks, Goldilocks};
use p3_koala_bear::{GenericPoseidon2LinearLayersKoalaBear, KoalaBear};
use p3_poseidon2_air::RoundConstants;
use rand::distr::StandardUniform;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::Poseidon2CircuitAir;

/// Configuration for BabyBear with base-field (`D=1`) challenges and a
/// 16-element Poseidon2 state (same permutation as quartic width-16).
///
/// Preprocessed CTL layout uses one column group per base element
/// (`WIDTH_EXT = 16`, `RATE_EXT = 8`).
pub struct BabyBearD1Width16;

impl Poseidon2Params for BabyBearD1Width16 {
    type BaseField = BabyBear;
    const CONFIG: Poseidon2Config = Poseidon2Config::BabyBearD1Width16;
}

impl BabyBearD1Width16 {
    pub const fn round_constants() -> RoundConstants<BabyBear, 16, 4, 13> {
        RoundConstants::new(
            p3_baby_bear::BABYBEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL,
            p3_baby_bear::BABYBEAR_POSEIDON2_RC_16_INTERNAL,
            p3_baby_bear::BABYBEAR_POSEIDON2_RC_16_EXTERNAL_FINAL,
        )
    }

    pub const fn default_air() -> Poseidon2CircuitAirBabyBearD1Width16 {
        Poseidon2CircuitAirBabyBearD1Width16::new(Self::round_constants())
    }

    pub fn default_air_with_preprocessed(
        preprocessed: Vec<BabyBear>,
        min_height: usize,
    ) -> Poseidon2CircuitAirBabyBearD1Width16 {
        Poseidon2CircuitAirBabyBearD1Width16::new_with_preprocessed(
            Self::round_constants(),
            preprocessed,
        )
        .with_min_height(min_height)
    }

    pub fn default_air_with_preprocessed_witness_bus5(
        preprocessed: Vec<BabyBear>,
        min_height: usize,
    ) -> Poseidon2CircuitAirBabyBearD1Width16WitnessBus5 {
        Poseidon2CircuitAirBabyBearD1Width16WitnessBus5::new_with_preprocessed(
            Self::round_constants(),
            preprocessed,
        )
        .with_min_height(min_height)
    }
}

/// Configuration for BabyBear with a quartic extension and a 16-element
/// Poseidon2 state.
///
/// Most common configuration for 31-bit recursive proofs.
///
/// S-box degree 7, 1 intermediate register, 4 half-full rounds, 13
/// partial rounds.
pub struct BabyBearD4Width16;

impl Poseidon2Params for BabyBearD4Width16 {
    type BaseField = BabyBear;
    const CONFIG: Poseidon2Config = Poseidon2Config::BabyBearD4Width16;
}

impl BabyBearD4Width16 {
    /// Return the canonical round constants for this configuration.
    pub const fn round_constants() -> RoundConstants<BabyBear, 16, 4, 13> {
        RoundConstants::new(
            p3_baby_bear::BABYBEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL,
            p3_baby_bear::BABYBEAR_POSEIDON2_RC_16_INTERNAL,
            p3_baby_bear::BABYBEAR_POSEIDON2_RC_16_EXTERNAL_FINAL,
        )
    }

    /// Create an AIR instance with canonical round constants and an empty
    /// preprocessed trace.
    pub const fn default_air() -> Poseidon2CircuitAirBabyBearD4Width16 {
        Poseidon2CircuitAirBabyBearD4Width16::new(Self::round_constants())
    }

    /// Create an AIR instance with canonical round constants, pre-populated
    /// preprocessed data, and a minimum trace height.
    pub fn default_air_with_preprocessed(
        preprocessed: Vec<BabyBear>,
        min_height: usize,
    ) -> Poseidon2CircuitAirBabyBearD4Width16 {
        Poseidon2CircuitAirBabyBearD4Width16::new_with_preprocessed(
            Self::round_constants(),
            preprocessed,
        )
        .with_min_height(min_height)
    }
}

/// Configuration for BabyBear with a quartic extension and a 24-element
/// Poseidon2 state.
///
/// The wider state provides higher throughput per permutation.
///
/// Costs more columns per row.
pub struct BabyBearD4Width24;

impl Poseidon2Params for BabyBearD4Width24 {
    type BaseField = BabyBear;
    const CONFIG: Poseidon2Config = Poseidon2Config::BabyBearD4Width24;
}

impl BabyBearD4Width24 {
    /// Return the canonical round constants for this configuration.
    pub const fn round_constants() -> RoundConstants<BabyBear, 24, 4, 21> {
        RoundConstants::new(
            p3_baby_bear::BABYBEAR_POSEIDON2_RC_24_EXTERNAL_INITIAL,
            p3_baby_bear::BABYBEAR_POSEIDON2_RC_24_INTERNAL,
            p3_baby_bear::BABYBEAR_POSEIDON2_RC_24_EXTERNAL_FINAL,
        )
    }

    /// Create an AIR instance with canonical round constants and an empty
    /// preprocessed trace.
    pub const fn default_air() -> Poseidon2CircuitAirBabyBearD4Width24 {
        Poseidon2CircuitAirBabyBearD4Width24::new(Self::round_constants())
    }

    /// Create an AIR instance with canonical round constants, pre-populated
    /// preprocessed data, and a minimum trace height.
    pub fn default_air_with_preprocessed(
        preprocessed: Vec<BabyBear>,
        min_height: usize,
    ) -> Poseidon2CircuitAirBabyBearD4Width24 {
        Poseidon2CircuitAirBabyBearD4Width24::new_with_preprocessed(
            Self::round_constants(),
            preprocessed,
        )
        .with_min_height(min_height)
    }
}

/// Configuration for KoalaBear with base-field (`D=1`) challenges and a
/// 16-element Poseidon2 state.
///
/// Preprocessed CTL layout uses one column group per base element
/// (`WIDTH_EXT = 16`, `RATE_EXT = 8`).
pub struct KoalaBearD1Width16;

impl Poseidon2Params for KoalaBearD1Width16 {
    type BaseField = KoalaBear;
    const CONFIG: Poseidon2Config = Poseidon2Config::KoalaBearD1Width16;
}

impl KoalaBearD1Width16 {
    pub const fn round_constants() -> RoundConstants<KoalaBear, 16, 4, 20> {
        RoundConstants::new(
            p3_koala_bear::KOALABEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL,
            p3_koala_bear::KOALABEAR_POSEIDON2_RC_16_INTERNAL,
            p3_koala_bear::KOALABEAR_POSEIDON2_RC_16_EXTERNAL_FINAL,
        )
    }

    pub const fn default_air() -> Poseidon2CircuitAirKoalaBearD1Width16 {
        Poseidon2CircuitAirKoalaBearD1Width16::new(Self::round_constants())
    }

    pub fn default_air_with_preprocessed(
        preprocessed: Vec<KoalaBear>,
        min_height: usize,
    ) -> Poseidon2CircuitAirKoalaBearD1Width16 {
        Poseidon2CircuitAirKoalaBearD1Width16::new_with_preprocessed(
            Self::round_constants(),
            preprocessed,
        )
        .with_min_height(min_height)
    }

    pub fn default_air_with_preprocessed_witness_bus5(
        preprocessed: Vec<KoalaBear>,
        min_height: usize,
    ) -> Poseidon2CircuitAirKoalaBearD1Width16WitnessBus5 {
        Poseidon2CircuitAirKoalaBearD1Width16WitnessBus5::new_with_preprocessed(
            Self::round_constants(),
            preprocessed,
        )
        .with_min_height(min_height)
    }
}

/// Configuration for KoalaBear with a quartic extension and a 16-element
/// Poseidon2 state.
///
/// KoalaBear is an alternative 31-bit field with different S-box
/// parameters.
///
/// S-box degree 3, 0 intermediate registers, 4 half-full rounds, 20
/// partial rounds.
pub struct KoalaBearD4Width16;

impl Poseidon2Params for KoalaBearD4Width16 {
    type BaseField = KoalaBear;
    const CONFIG: Poseidon2Config = Poseidon2Config::KoalaBearD4Width16;
}

impl KoalaBearD4Width16 {
    /// Return the canonical round constants for this configuration.
    pub const fn round_constants() -> RoundConstants<KoalaBear, 16, 4, 20> {
        RoundConstants::new(
            p3_koala_bear::KOALABEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL,
            p3_koala_bear::KOALABEAR_POSEIDON2_RC_16_INTERNAL,
            p3_koala_bear::KOALABEAR_POSEIDON2_RC_16_EXTERNAL_FINAL,
        )
    }

    /// Create an AIR instance with canonical round constants and an empty
    /// preprocessed trace.
    pub const fn default_air() -> Poseidon2CircuitAirKoalaBearD4Width16 {
        Poseidon2CircuitAirKoalaBearD4Width16::new(Self::round_constants())
    }

    /// Create an AIR instance with canonical round constants, pre-populated
    /// preprocessed data, and a minimum trace height.
    pub fn default_air_with_preprocessed(
        preprocessed: Vec<KoalaBear>,
        min_height: usize,
    ) -> Poseidon2CircuitAirKoalaBearD4Width16 {
        Poseidon2CircuitAirKoalaBearD4Width16::new_with_preprocessed(
            Self::round_constants(),
            preprocessed,
        )
        .with_min_height(min_height)
    }
}

/// Configuration for KoalaBear with a quartic extension and a 24-element
/// Poseidon2 state.
pub struct KoalaBearD4Width24;

impl Poseidon2Params for KoalaBearD4Width24 {
    type BaseField = KoalaBear;
    const CONFIG: Poseidon2Config = Poseidon2Config::KoalaBearD4Width24;
}

impl KoalaBearD4Width24 {
    /// Return the canonical round constants for this configuration.
    pub const fn round_constants() -> RoundConstants<KoalaBear, 24, 4, 23> {
        RoundConstants::new(
            p3_koala_bear::KOALABEAR_POSEIDON2_RC_24_EXTERNAL_INITIAL,
            p3_koala_bear::KOALABEAR_POSEIDON2_RC_24_INTERNAL,
            p3_koala_bear::KOALABEAR_POSEIDON2_RC_24_EXTERNAL_FINAL,
        )
    }

    /// Create an AIR instance with canonical round constants and an empty
    /// preprocessed trace.
    pub const fn default_air() -> Poseidon2CircuitAirKoalaBearD4Width24 {
        Poseidon2CircuitAirKoalaBearD4Width24::new(Self::round_constants())
    }

    /// Create an AIR instance with canonical round constants, pre-populated
    /// preprocessed data, and a minimum trace height.
    pub fn default_air_with_preprocessed(
        preprocessed: Vec<KoalaBear>,
        min_height: usize,
    ) -> Poseidon2CircuitAirKoalaBearD4Width24 {
        Poseidon2CircuitAirKoalaBearD4Width24::new_with_preprocessed(
            Self::round_constants(),
            preprocessed,
        )
        .with_min_height(min_height)
    }
}

/// BabyBear Poseidon2 circuit AIR with `D=1` and 16-element state (base-field CTL slots).
pub type Poseidon2CircuitAirBabyBearD1Width16 = Poseidon2CircuitAir<
    BabyBear,
    GenericPoseidon2LinearLayersBabyBear,
    { BabyBearD1Width16::D },
    { BabyBearD1Width16::WIDTH },
    { BabyBearD1Width16::WIDTH_EXT },
    { BabyBearD1Width16::RATE_EXT },
    { BabyBearD1Width16::CAPACITY_EXT },
    { BabyBearD1Width16::SBOX_DEGREE },
    { BabyBearD1Width16::SBOX_REGISTERS },
    { BabyBearD1Width16::HALF_FULL_ROUNDS },
    { BabyBearD1Width16::PARTIAL_ROUNDS },
    { BabyBearD1Width16::D },
>;

/// [`BabyBearD1Width16`] with witness-bus keys padded to quintic width (for EF5 + base Poseidon).
pub type Poseidon2CircuitAirBabyBearD1Width16WitnessBus5 = Poseidon2CircuitAir<
    BabyBear,
    GenericPoseidon2LinearLayersBabyBear,
    { BabyBearD1Width16::D },
    { BabyBearD1Width16::WIDTH },
    { BabyBearD1Width16::WIDTH_EXT },
    { BabyBearD1Width16::RATE_EXT },
    { BabyBearD1Width16::CAPACITY_EXT },
    { BabyBearD1Width16::SBOX_DEGREE },
    { BabyBearD1Width16::SBOX_REGISTERS },
    { BabyBearD1Width16::HALF_FULL_ROUNDS },
    { BabyBearD1Width16::PARTIAL_ROUNDS },
    5,
>;

/// BabyBear Poseidon2 circuit AIR with quartic extension and 16-element state.
pub type Poseidon2CircuitAirBabyBearD4Width16 = Poseidon2CircuitAir<
    BabyBear,
    GenericPoseidon2LinearLayersBabyBear,
    { BabyBearD4Width16::D },
    { BabyBearD4Width16::WIDTH },
    { BabyBearD4Width16::WIDTH_EXT },
    { BabyBearD4Width16::RATE_EXT },
    { BabyBearD4Width16::CAPACITY_EXT },
    { BabyBearD4Width16::SBOX_DEGREE },
    { BabyBearD4Width16::SBOX_REGISTERS },
    { BabyBearD4Width16::HALF_FULL_ROUNDS },
    { BabyBearD4Width16::PARTIAL_ROUNDS },
    { BabyBearD4Width16::D },
>;

/// BabyBear Poseidon2 circuit AIR with quartic extension and 24-element state.
pub type Poseidon2CircuitAirBabyBearD4Width24 = Poseidon2CircuitAir<
    BabyBear,
    GenericPoseidon2LinearLayersBabyBear,
    { BabyBearD4Width24::D },
    { BabyBearD4Width24::WIDTH },
    { BabyBearD4Width24::WIDTH_EXT },
    { BabyBearD4Width24::RATE_EXT },
    { BabyBearD4Width24::CAPACITY_EXT },
    { BabyBearD4Width24::SBOX_DEGREE },
    { BabyBearD4Width24::SBOX_REGISTERS },
    { BabyBearD4Width24::HALF_FULL_ROUNDS },
    { BabyBearD4Width24::PARTIAL_ROUNDS },
    { BabyBearD4Width24::D },
>;

/// KoalaBear Poseidon2 circuit AIR with `D=1` and 16-element state (base-field CTL slots).
pub type Poseidon2CircuitAirKoalaBearD1Width16 = Poseidon2CircuitAir<
    KoalaBear,
    GenericPoseidon2LinearLayersKoalaBear,
    { KoalaBearD1Width16::D },
    { KoalaBearD1Width16::WIDTH },
    { KoalaBearD1Width16::WIDTH_EXT },
    { KoalaBearD1Width16::RATE_EXT },
    { KoalaBearD1Width16::CAPACITY_EXT },
    { KoalaBearD1Width16::SBOX_DEGREE },
    { KoalaBearD1Width16::SBOX_REGISTERS },
    { KoalaBearD1Width16::HALF_FULL_ROUNDS },
    { KoalaBearD1Width16::PARTIAL_ROUNDS },
    { KoalaBearD1Width16::D },
>;

/// [`KoalaBearD1Width16`] with witness-bus keys padded to quintic width (for EF5 + base Poseidon).
pub type Poseidon2CircuitAirKoalaBearD1Width16WitnessBus5 = Poseidon2CircuitAir<
    KoalaBear,
    GenericPoseidon2LinearLayersKoalaBear,
    { KoalaBearD1Width16::D },
    { KoalaBearD1Width16::WIDTH },
    { KoalaBearD1Width16::WIDTH_EXT },
    { KoalaBearD1Width16::RATE_EXT },
    { KoalaBearD1Width16::CAPACITY_EXT },
    { KoalaBearD1Width16::SBOX_DEGREE },
    { KoalaBearD1Width16::SBOX_REGISTERS },
    { KoalaBearD1Width16::HALF_FULL_ROUNDS },
    { KoalaBearD1Width16::PARTIAL_ROUNDS },
    5,
>;

/// KoalaBear Poseidon2 circuit AIR with quartic extension and 16-element state.
pub type Poseidon2CircuitAirKoalaBearD4Width16 = Poseidon2CircuitAir<
    KoalaBear,
    GenericPoseidon2LinearLayersKoalaBear,
    { KoalaBearD4Width16::D },
    { KoalaBearD4Width16::WIDTH },
    { KoalaBearD4Width16::WIDTH_EXT },
    { KoalaBearD4Width16::RATE_EXT },
    { KoalaBearD4Width16::CAPACITY_EXT },
    { KoalaBearD4Width16::SBOX_DEGREE },
    { KoalaBearD4Width16::SBOX_REGISTERS },
    { KoalaBearD4Width16::HALF_FULL_ROUNDS },
    { KoalaBearD4Width16::PARTIAL_ROUNDS },
    { KoalaBearD4Width16::D },
>;

/// KoalaBear Poseidon2 circuit AIR with quartic extension and 24-element state.
pub type Poseidon2CircuitAirKoalaBearD4Width24 = Poseidon2CircuitAir<
    KoalaBear,
    GenericPoseidon2LinearLayersKoalaBear,
    { KoalaBearD4Width24::D },
    { KoalaBearD4Width24::WIDTH },
    { KoalaBearD4Width24::WIDTH_EXT },
    { KoalaBearD4Width24::RATE_EXT },
    { KoalaBearD4Width24::CAPACITY_EXT },
    { KoalaBearD4Width24::SBOX_DEGREE },
    { KoalaBearD4Width24::SBOX_REGISTERS },
    { KoalaBearD4Width24::HALF_FULL_ROUNDS },
    { KoalaBearD4Width24::PARTIAL_ROUNDS },
    { KoalaBearD4Width24::D },
>;

/// Goldilocks Poseidon2 circuit AIR with quadratic extension and 8-element state.
pub type Poseidon2CircuitAirGoldilocksD2Width8 = Poseidon2CircuitAir<
    Goldilocks,
    GenericPoseidon2LinearLayersGoldilocks,
    { GoldilocksD2Width8::D },
    { GoldilocksD2Width8::WIDTH },
    { GoldilocksD2Width8::WIDTH_EXT },
    { GoldilocksD2Width8::RATE_EXT },
    { GoldilocksD2Width8::CAPACITY_EXT },
    { GoldilocksD2Width8::SBOX_DEGREE },
    { GoldilocksD2Width8::SBOX_REGISTERS },
    { GoldilocksD2Width8::HALF_FULL_ROUNDS },
    { GoldilocksD2Width8::PARTIAL_ROUNDS },
    { GoldilocksD2Width8::D },
>;

/// Generate deterministic round constants for the Goldilocks width-8
/// configuration using a fixed seed.
pub fn goldilocks_d2_width8_round_constants() -> RoundConstants<Goldilocks, 8, 4, 22> {
    let mut rng = SmallRng::seed_from_u64(1);
    let beginning_full = rng.sample(StandardUniform);
    let ending_full = rng.sample(StandardUniform);
    let partial = rng.sample(StandardUniform);
    RoundConstants::new(beginning_full, partial, ending_full)
}

/// Create a Goldilocks width-8 AIR with deterministic round constants and
/// an empty preprocessed trace.
pub fn goldilocks_d2_width8_default_air() -> Poseidon2CircuitAirGoldilocksD2Width8 {
    Poseidon2CircuitAirGoldilocksD2Width8::new(goldilocks_d2_width8_round_constants())
}

/// Create a Goldilocks width-8 AIR with deterministic round constants,
/// pre-populated preprocessed data, and a minimum trace height.
pub fn goldilocks_d2_width8_default_air_with_preprocessed(
    preprocessed: Vec<Goldilocks>,
    min_height: usize,
) -> Poseidon2CircuitAirGoldilocksD2Width8 {
    Poseidon2CircuitAirGoldilocksD2Width8::new_with_preprocessed(
        goldilocks_d2_width8_round_constants(),
        preprocessed,
    )
    .with_min_height(min_height)
}
