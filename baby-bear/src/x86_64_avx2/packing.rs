use core::arch::x86_64::__m256i;
use core::mem::transmute;

use p3_monty_31::{MontyParametersAVX2, PackedMontyField31AVX2};

use crate::BabyBearParameters;

pub type PackedBabyBearAVX2 = PackedMontyField31AVX2<BabyBearParameters>;

const WIDTH: usize = 8;

impl MontyParametersAVX2 for BabyBearParameters {
    const PACKED_P: __m256i = unsafe { transmute::<[u32; WIDTH], _>([0x78000001; WIDTH]) };
    const PACKED_MU: __m256i = unsafe { transmute::<[u32; WIDTH], _>([0x88000001; WIDTH]) };
}

#[cfg(test)]
mod tests {
    use p3_field_testing::{
        assert_packed_broadcast_dot_product_matches_scalar, test_packed_field,
        test_packed_field_dot_product_boundary,
    };

    use super::WIDTH;
    use crate::BabyBear;

    const SPECIAL_VALS: [BabyBear; WIDTH] = BabyBear::new_array([
        0x00000000, 0x00000001, 0x78000000, 0x77ffffff, 0x3c000000, 0x0ffffffe, 0x68000003,
        0x70000002,
    ]);

    test_packed_field!(
        crate::PackedBabyBearAVX2,
        &[crate::PackedBabyBearAVX2::ZERO],
        &[crate::PackedBabyBearAVX2::ONE],
        p3_monty_31::PackedMontyField31AVX2::<crate::BabyBearParameters>(super::SPECIAL_VALS)
    );

    test_packed_field_dot_product_boundary!(crate::PackedBabyBearAVX2);

    #[test]
    fn dot_product_5_carry_cascade_regression() {
        // Group A holds terms 0 to 3, group B holds term 4 alone.
        // Only group A can reach `2^{32} P`, so only group A is folded.
        //
        // Every step of the merge is load-bearing here:
        // - `hi_A = 1.748 P`, so the fold fires,
        // - unfolded, the merge reaches `1.001 * 2^{64}`, so dropping the fold wraps the lane,
        // - the low halves sum to `1.535 * 2^{32}`, so the merge carries into the high half,
        // - the merged high half is `1.135 P`, so the final conditional subtract fires.
        let lhs = [0x5290bab8, 0x0c0b1e10, 0x4bd2a313, 0x5f6ff18f, 0x55135da6];
        let rhs = [0x3d4989d9, 0x592aa670, 0x294dad70, 0x34bf6b81, 0x21f21c97];

        assert_packed_broadcast_dot_product_matches_scalar::<crate::PackedBabyBearAVX2, 5>(
            lhs, rhs,
        );
    }

    #[test]
    fn dot_product_6_carry_cascade_regression() {
        // Group A holds terms 0 to 3, group B holds terms 4 and 5.
        // Only group A can reach `2^{32} P`, so only group A is folded.
        //
        // Every step of the merge is load-bearing here:
        // - `hi_A = 1.545 P`, so the fold fires,
        // - unfolded, the merge reaches `1.047 * 2^{64}`, so dropping the fold wraps the lane,
        // - the low halves sum to `1.190 * 2^{32}`, so the merge carries into the high half,
        // - the merged high half is `1.234 P`, so the final conditional subtract fires.
        let lhs = [
            0x5eb4ed38, 0x68b818ff, 0x1837703b, 0x0b42801d, 0x01714971, 0x581065e2,
        ];
        let rhs = [
            0x08874da3, 0x18f1da78, 0x58caebf4, 0x2bbbc954, 0x76c55f53, 0x11521b47,
        ];

        assert_packed_broadcast_dot_product_matches_scalar::<crate::PackedBabyBearAVX2, 6>(
            lhs, rhs,
        );
    }

    #[test]
    fn dot_product_7_carry_cascade_regression() {
        // Group A holds terms 0 to 3, group B holds terms 4 to 6.
        // Both groups can reach `2^{32} P`, so both are folded.
        //
        // Every step of the merge is load-bearing here:
        // - `hi_A = 1.852 P` and `hi_B = 1.381 P`, so both folds fire,
        // - dropping either fold alone lets the merge reach `1.047 * 2^{64}`, which wraps,
        // - the low halves sum to `1.072 * 2^{32}`, so the merge carries into the high half,
        // - the merged high half is `1.233 P`, so the final conditional subtract fires.
        let lhs = [
            0x4f8b34e4, 0x12d0cf75, 0x4b52d2f3, 0x220c1a2d, 0x47e26545, 0x74b2a7b1, 0x5a3b4b76,
        ];
        let rhs = [
            0x0b4d55a6, 0x61f3d5c3, 0x0bb5b0f1, 0x61dc79be, 0x1ac4e4ec, 0x3c59bb5c, 0x625fd872,
        ];

        assert_packed_broadcast_dot_product_matches_scalar::<crate::PackedBabyBearAVX2, 7>(
            lhs, rhs,
        );
    }

    #[test]
    fn dot_product_8_carry_cascade_regression() {
        // Group A holds terms 0 to 3, group B holds terms 4 to 7.
        // Both groups can reach `2^{32} P`, so both are folded.
        //
        // Every step of the merge is load-bearing here:
        // - `hi_A = 1.509 P` and `hi_B = 1.741 P`, so both folds fire,
        // - dropping either fold alone lets the merge reach `1.055 * 2^{64}`, which wraps,
        // - the low halves sum to `1.621 * 2^{32}`, so the merge carries into the high half,
        // - the merged high half is `1.250 P`, so the final conditional subtract fires.
        let lhs = [
            0x6e0253e0, 0x1f5c1ba8, 0x2fc501e9, 0x124ece0f, 0x500cdbca, 0x744bbe5c, 0x4dddd284,
            0x09d3db39,
        ];
        let rhs = [
            0x1c33716f, 0x597a181f, 0x5123af6d, 0x70c41283, 0x0b58234d, 0x6dc86d54, 0x16afc27a,
            0x17803123,
        ];

        assert_packed_broadcast_dot_product_matches_scalar::<crate::PackedBabyBearAVX2, 8>(
            lhs, rhs,
        );
    }
}
