use core::arch::x86_64::__m512i;
use core::mem::transmute;

use p3_monty_31::{MontyParametersAVX512, PackedMontyField31AVX512};

use crate::KoalaBearParameters;

pub type PackedKoalaBearAVX512 = PackedMontyField31AVX512<KoalaBearParameters>;

const WIDTH: usize = 16;

impl MontyParametersAVX512 for KoalaBearParameters {
    const PACKED_P: __m512i = unsafe { transmute::<[u32; WIDTH], _>([0x7f000001; WIDTH]) };
    const PACKED_MU: __m512i = unsafe { transmute::<[u32; WIDTH], _>([0x81000001; WIDTH]) };
}

#[cfg(test)]
mod tests {
    use p3_field_testing::{
        assert_packed_broadcast_dot_product_matches_scalar, test_packed_field,
        test_packed_field_dot_product_boundary,
    };

    use super::WIDTH;
    use crate::KoalaBear;

    const SPECIAL_VALS: [KoalaBear; WIDTH] = KoalaBear::new_array([
        0x00000000, 0x00000001, 0x78000000, 0x77ffffff, 0x3c000000, 0x0ffffffe, 0x68000003,
        0x70000002, 0x00000000, 0x00000001, 0x78000000, 0x77ffffff, 0x3c000000, 0x0ffffffe,
        0x68000003, 0x70000002,
    ]);

    test_packed_field!(
        crate::PackedKoalaBearAVX512,
        &[crate::PackedKoalaBearAVX512::ZERO],
        &[crate::PackedKoalaBearAVX512::ONE],
        p3_monty_31::PackedMontyField31AVX512::<crate::KoalaBearParameters>(super::SPECIAL_VALS)
    );

    test_packed_field_dot_product_boundary!(crate::PackedKoalaBearAVX512);

    #[test]
    fn dot_product_5_carry_cascade_regression() {
        // Group A holds terms 0 to 3, group B holds term 4 alone.
        // Only group A can reach `2^{32} P`, so only group A is folded.
        //
        // Every step of the merge is load-bearing here:
        // - `hi_A = 1.839 P`, so the fold fires,
        // - unfolded, the merge reaches `1.137 * 2^{64}`, so dropping the fold wraps the lane,
        // - the low halves sum to `1.371 * 2^{32}`, so the merge carries into the high half,
        // - the merged high half is `1.291 P`, so the final conditional subtract fires.
        let lhs = [0x22d2c6b2, 0x7a259561, 0x55cf8e46, 0x4122e41e, 0x594d4489];
        let rhs = [0x46eae60f, 0x6f4e6f17, 0x53368b43, 0x1a46a028, 0x6301613c];

        assert_packed_broadcast_dot_product_matches_scalar::<crate::PackedKoalaBearAVX512, 5>(
            lhs, rhs,
        );
    }

    #[test]
    fn dot_product_6_carry_cascade_regression() {
        // Group A holds terms 0 to 3, group B holds terms 4 and 5.
        // Only group A can reach `2^{32} P`, so only group A is folded.
        //
        // Every step of the merge is load-bearing here:
        // - `hi_A = 1.467 P`, so the fold fires,
        // - unfolded, the merge reaches `1.182 * 2^{64}`, so dropping the fold wraps the lane,
        // - the low halves sum to `1.076 * 2^{32}`, so the merge carries into the high half,
        // - the merged high half is `1.383 P`, so the final conditional subtract fires.
        let lhs = [
            0x6c77e213, 0x3f84985f, 0x0e56970f, 0x1ed6d461, 0x4056625a, 0x6bb6a75c,
        ];
        let rhs = [
            0x2ff0ff6a, 0x32ad0a11, 0x445f736b, 0x43e155ae, 0x3981c6d3, 0x4e6c849b,
        ];

        assert_packed_broadcast_dot_product_matches_scalar::<crate::PackedKoalaBearAVX512, 6>(
            lhs, rhs,
        );
    }

    #[test]
    fn dot_product_7_carry_cascade_regression() {
        // Group A holds terms 0 to 3, group B holds terms 4 to 6.
        // Both groups can reach `2^{32} P`, so both are folded.
        //
        // Every step of the merge is load-bearing here:
        // - `hi_A = 1.836 P` and `hi_B = 1.386 P`, so both folds fire,
        // - dropping either fold alone lets the merge reach `1.102 * 2^{64}`, which wraps,
        // - the low halves sum to `1.551 * 2^{32}`, so the merge carries into the high half,
        // - the merged high half is `1.221 P`, so the final conditional subtract fires.
        let lhs = [
            0x70300e0c, 0x2f09af6c, 0x29349e8a, 0x228eb5e4, 0x24ff7987, 0x3d755ac3, 0x14e9a931,
        ];
        let rhs = [
            0x3d339a4b, 0x44f0858a, 0x5d643b26, 0x6bdb55ac, 0x6f768c37, 0x22c646a3, 0x0d20f92d,
        ];

        assert_packed_broadcast_dot_product_matches_scalar::<crate::PackedKoalaBearAVX512, 7>(
            lhs, rhs,
        );
    }

    #[test]
    fn dot_product_8_carry_cascade_regression() {
        // Group A holds terms 0 to 3, group B holds terms 4 to 7.
        // Both groups can reach `2^{32} P`, so both are folded.
        //
        // Every step of the merge is load-bearing here:
        // - `hi_A = 1.400 P` and `hi_B = 1.852 P`, so both folds fire,
        // - dropping either fold alone lets the merge reach `1.117 * 2^{64}`, which wraps,
        // - the low halves sum to `1.151 * 2^{32}`, so the merge carries into the high half,
        // - the merged high half is `1.252 P`, so the final conditional subtract fires.
        let lhs = [
            0x1e0bdfbc, 0x43f199e3, 0x6b4e9cd4, 0x380a48bd, 0x6d50f392, 0x2ae9e263, 0x36511dff,
            0x294fd76b,
        ];
        let rhs = [
            0x47bbd3af, 0x26c65b5b, 0x6c512731, 0x0841f6cd, 0x504c2c00, 0x2a2a8ecb, 0x37867b27,
            0x2f2a4f2a,
        ];

        assert_packed_broadcast_dot_product_matches_scalar::<crate::PackedKoalaBearAVX512, 8>(
            lhs, rhs,
        );
    }
}
