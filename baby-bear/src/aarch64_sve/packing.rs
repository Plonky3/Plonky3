use p3_monty_31::PackedMontyField31Sve;

use crate::BabyBearParameters;

pub type PackedBabyBearSve = PackedMontyField31Sve<BabyBearParameters>;

#[cfg(test)]
mod tests {
    use p3_field_testing::test_packed_field;

    use crate::BabyBear;

    #[cfg(not(target_feature = "sve2"))]
    const WIDTH: usize = 8;
    #[cfg(target_feature = "sve2")]
    const WIDTH: usize = 4;

    #[cfg(not(target_feature = "sve2"))]
    const SPECIAL_VALS: [BabyBear; WIDTH] = BabyBear::new_array([
        0x00000000, 0x00000001, 0x00000002, 0x78000000, 0x77ffffff, 0x00000003, 0x40000000,
        0x3fffffff,
    ]);
    #[cfg(target_feature = "sve2")]
    const SPECIAL_VALS: [BabyBear; WIDTH] =
        BabyBear::new_array([0x00000000, 0x00000001, 0x78000000, 0x77ffffff]);

    test_packed_field!(
        crate::PackedBabyBearSve,
        &[crate::PackedBabyBearSve::ZERO],
        &[crate::PackedBabyBearSve::ONE],
        p3_monty_31::PackedMontyField31Sve::<crate::BabyBearParameters>(super::SPECIAL_VALS)
    );
}
