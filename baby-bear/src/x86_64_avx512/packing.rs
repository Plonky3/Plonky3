use core::arch::x86_64::__m512i;
use core::mem::transmute;

use p3_monty_31::{MontyParametersAVX512, PackedMontyField31AVX512};

use crate::BabyBearParameters;

pub type PackedBabyBearAVX512 = PackedMontyField31AVX512<BabyBearParameters>;

const WIDTH: usize = 16;

impl MontyParametersAVX512 for BabyBearParameters {
    const PACKED_P: __m512i = unsafe { transmute::<[u32; WIDTH], _>([0x78000001; WIDTH]) };
    const PACKED_MU: __m512i = unsafe { transmute::<[u32; WIDTH], _>([0x88000001; WIDTH]) };
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
        0x70000002, 0x00000000, 0x00000001, 0x78000000, 0x77ffffff, 0x3c000000, 0x0ffffffe,
        0x68000003, 0x70000002,
    ]);

    test_packed_field!(
        crate::PackedBabyBearAVX512,
        &[crate::PackedBabyBearAVX512::ZERO],
        &[crate::PackedBabyBearAVX512::ONE],
        p3_monty_31::PackedMontyField31AVX512::<crate::BabyBearParameters>(super::SPECIAL_VALS)
    );

    test_packed_field_dot_product_boundary!(crate::PackedBabyBearAVX512);

    #[test]
    fn dot_product_5_carry_cascade_regression() {
        // Group A holds terms 0 to 3, group B holds term 4 alone.
        // `hi_A` sits above `P`, so the `2^{32} P` fold has to fire before the merge.
        //
        // The low halves of the two folded groups sum past `2^{32}`.
        // Dropping that carry would leave the merged high half one short.
        //
        // The merged high half also exceeds `P`, so the final conditional subtract fires too.
        let lhs = [0x2e279ce6, 0x247900aa, 0x60ce65d9, 0x10da3c26, 0x29a91318];
        let rhs = [0x01859688, 0x61411115, 0x3a299723, 0x0395a3f0, 0x5deae270];

        assert_packed_broadcast_dot_product_matches_scalar::<crate::PackedBabyBearAVX512, 5>(
            lhs, rhs,
        );
    }

    #[test]
    fn dot_product_6_carry_cascade_regression() {
        // Group A holds terms 0 to 3, group B holds terms 4 and 5.
        // `hi_A` sits above `P`, so the `2^{32} P` fold has to fire before the merge.
        //
        // The low halves of the two folded groups sum past `2^{32}`.
        // Dropping that carry would leave the merged high half one short.
        //
        // The merged high half also exceeds `P`, so the final conditional subtract fires too.
        let lhs = [
            0x2acd899d, 0x031fd1a2, 0x564451e5, 0x08ed60e6, 0x66934215, 0x2fd4abce,
        ];
        let rhs = [
            0x2a37a9a7, 0x0deb7540, 0x40509b8e, 0x07fc1dfd, 0x50fbaf07, 0x335070bf,
        ];

        assert_packed_broadcast_dot_product_matches_scalar::<crate::PackedBabyBearAVX512, 6>(
            lhs, rhs,
        );
    }

    #[test]
    fn dot_product_7_carry_cascade_regression() {
        // Group A holds terms 0 to 3, group B holds terms 4 to 6.
        // Both `hi_A` and `hi_B` sit above `P`, so both groups have to be folded before the merge.
        //
        // The low halves of the two folded groups sum past `2^{32}`.
        // Dropping that carry would leave the merged high half one short.
        //
        // The merged high half also exceeds `P`, so the final conditional subtract fires too.
        let lhs = [
            0x53d4d828, 0x3b9701fe, 0x71060fb5, 0x6701b79a, 0x1e50de98, 0x4f1661d2, 0x28d351c6,
        ];
        let rhs = [
            0x44133c84, 0x38780a45, 0x5fd5ae2e, 0x4b5d10d8, 0x74d80051, 0x4ee1a8a2, 0x6df86de7,
        ];

        assert_packed_broadcast_dot_product_matches_scalar::<crate::PackedBabyBearAVX512, 7>(
            lhs, rhs,
        );
    }

    #[test]
    fn dot_product_8_carry_cascade_regression() {
        // Group A holds terms 0 to 3, group B holds terms 4 to 7.
        // Both `hi_A` and `hi_B` sit above `P`, so both groups have to be folded before the merge.
        //
        // The low halves of the two folded groups sum past `2^{32}`.
        // Dropping that carry would leave the merged high half one short.
        //
        // The merged high half also exceeds `P`, so the final conditional subtract fires too.
        let lhs = [
            0x5f5edeef, 0x01c04ca6, 0x716aabff, 0x05b01a30, 0x3632a95e, 0x1c8ad89a, 0x5aa6c176,
            0x00e1c21e,
        ];
        let rhs = [
            0x150daf83, 0x6d158d9c, 0x5dd86497, 0x00222f3d, 0x6529eaed, 0x26848a0c, 0x2da2806b,
            0x53e4d5cf,
        ];

        assert_packed_broadcast_dot_product_matches_scalar::<crate::PackedBabyBearAVX512, 8>(
            lhs, rhs,
        );
    }
}
