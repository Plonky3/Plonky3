use core::arch::aarch64::{int32x4_t, uint32x4_t};
use core::mem::transmute;

use p3_monty_31::{MontyParametersNeon, PackedMontyField31Neon};

use crate::KoalaBearParameters;

const WIDTH: usize = 4;

impl MontyParametersNeon for KoalaBearParameters {
    const PACKED_P: uint32x4_t = unsafe { transmute::<[u32; WIDTH], _>([0x7f000001; WIDTH]) };
    // This MU is the same 0x88000001 as elsewhere, just interpreted as an `i32`.
    const PACKED_MU: int32x4_t = unsafe { transmute::<[i32; WIDTH], _>([-0x7effffff; WIDTH]) };
}

pub type PackedKoalaBearNeon = PackedMontyField31Neon<KoalaBearParameters>;

#[cfg(test)]
mod tests {
    use p3_field::PrimeField32;
    use p3_field_testing::{
        assert_packed_broadcast_dot_product_matches_scalar, test_packed_field,
        test_packed_field_dot_product_boundary,
    };

    use super::WIDTH;
    use crate::KoalaBear;

    const SPECIAL_VALS: [KoalaBear; WIDTH] =
        KoalaBear::new_array([0x00000000, 0x00000001, 0x00000002, 0x7f000000]);

    test_packed_field!(
        crate::PackedKoalaBearNeon,
        &[crate::PackedKoalaBearNeon::ZERO],
        &[crate::PackedKoalaBearNeon::ONE],
        p3_monty_31::PackedMontyField31Neon::<crate::KoalaBearParameters>(super::SPECIAL_VALS)
    );

    test_packed_field_dot_product_boundary!(crate::PackedKoalaBearNeon);

    #[test]
    fn dot_product_5_carry_cascade_regression() {
        const P: u32 = KoalaBear::ORDER_U32;

        // 5-tuple tuned for KoalaBear's prime; pushes the merge into the
        // reduce-after-carry boundary.
        let lhs = [P - 1, 1, 8, P - 3, P - 2];
        let rhs = [P - 4, 9, P - 2, P - 5, 6];

        assert_packed_broadcast_dot_product_matches_scalar::<crate::PackedKoalaBearNeon, 5>(
            lhs, rhs,
        );
    }

    #[test]
    fn dot_product_8_carry_cascade_regression() {
        const P: u32 = KoalaBear::ORDER_U32;
        // First lane sits just above the midpoint to shift where the low-half
        // overflow happens.
        const HALF_PLUS_ONE: u32 = (P / 2) + 1;

        let lhs = [HALF_PLUS_ONE, P - 2, 6, 0, P - 1, 5, 6, P - 1];
        let rhs = [9, 5, 2, 7, P - 3, P - 1, 9, P - 5];

        assert_packed_broadcast_dot_product_matches_scalar::<crate::PackedKoalaBearNeon, 8>(
            lhs, rhs,
        );
    }
}
