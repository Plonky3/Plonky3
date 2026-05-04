use core::arch::aarch64::{int32x4_t, uint32x4_t};
use core::mem::transmute;

use p3_monty_31::{MontyParametersNeon, PackedMontyField31Neon};

use crate::BabyBearParameters;

const WIDTH: usize = 4;

impl MontyParametersNeon for BabyBearParameters {
    // SAFETY: This is a valid packed representation of P.
    const PACKED_P: uint32x4_t = unsafe { transmute::<[u32; WIDTH], _>([0x78000001; WIDTH]) };
    // This MU is the same 0x88000001 as elsewhere, just interpreted as an `i32`.
    // SAFETY: This is a valid packed representation of MU.
    const PACKED_MU: int32x4_t = unsafe { transmute::<[i32; WIDTH], _>([-0x77ffffff; WIDTH]) };
}

pub type PackedBabyBearNeon = PackedMontyField31Neon<BabyBearParameters>;

#[cfg(test)]
mod tests {
    use p3_field::PrimeField32;
    use p3_field_testing::{
        assert_packed_broadcast_dot_product_matches_scalar, test_packed_field,
        test_packed_field_dot_product_boundary,
    };

    use super::WIDTH;
    use crate::BabyBear;

    const SPECIAL_VALS: [BabyBear; WIDTH] =
        BabyBear::new_array([0x00000000, 0x00000001, 0x00000002, 0x78000000]);

    test_packed_field!(
        crate::PackedBabyBearNeon,
        &[crate::PackedBabyBearNeon::ZERO],
        &[crate::PackedBabyBearNeon::ONE],
        p3_monty_31::PackedMontyField31Neon::<crate::BabyBearParameters>(super::SPECIAL_VALS)
    );

    test_packed_field_dot_product_boundary!(crate::PackedBabyBearNeon);

    #[test]
    fn dot_product_5_carry_cascade_regression() {
        const P: u32 = BabyBear::ORDER_U32;
        // `HALF = P / 2`; `P` is odd so `2 * HALF == P - 1`.
        const HALF: u32 = P / 2;

        // Tuned to land `c_lo == 0`, `c_hi_red == P - 1`, low-half carry = 1.
        let lhs = [3, P - 5, P - 2, P - 1, P - 3];
        let rhs = [HALF, 0, HALF - 1, 6, HALF - 1];

        assert_packed_broadcast_dot_product_matches_scalar::<crate::PackedBabyBearNeon, 5>(
            lhs, rhs,
        );
    }

    #[test]
    fn dot_product_8_carry_cascade_regression() {
        const P: u32 = BabyBear::ORDER_U32;
        const HALF: u32 = P / 2;

        // Stresses the 4+4 split: low halves sum past `2^32`, highs stay in band.
        let lhs = [HALF, 9, P - 3, 9, HALF, 1, 2, 2];
        let rhs = [P - 3, 9, P - 2, 2, P - 3, 4, P - 3, 7];

        assert_packed_broadcast_dot_product_matches_scalar::<crate::PackedBabyBearNeon, 8>(
            lhs, rhs,
        );
    }
}
