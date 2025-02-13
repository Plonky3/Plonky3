use p3_poseidon2_air::RoundConstants;

use crate::{
    BabyBear, BABYBEAR_RC16_EXTERNAL_FINAL, BABYBEAR_RC16_EXTERNAL_INITIAL, BABYBEAR_RC16_INTERNAL,
    BABYBEAR_RC24_EXTERNAL_FINAL, BABYBEAR_RC24_EXTERNAL_INITIAL, BABYBEAR_RC24_INTERNAL,
};

/// Round constants for the 16-width Poseidon2 permutation on BabyBear.
///
/// See https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2_instance_babybear.rs
pub const BABYBEAR_RC16: RoundConstants<BabyBear, 16, 4, 13> = RoundConstants::new(
    BABYBEAR_RC16_EXTERNAL_INITIAL,
    BABYBEAR_RC16_INTERNAL,
    BABYBEAR_RC16_EXTERNAL_FINAL,
);

/// Round constants for the 24-width Poseidon2 permutation on BabyBear.
///
/// See https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2_instance_babybear.rs
pub const BABYBEAR_RC24: RoundConstants<BabyBear, 24, 4, 21> = RoundConstants::new(
    BABYBEAR_RC24_EXTERNAL_INITIAL,
    BABYBEAR_RC24_INTERNAL,
    BABYBEAR_RC24_EXTERNAL_FINAL,
);
