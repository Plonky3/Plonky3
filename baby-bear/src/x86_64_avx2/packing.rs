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
    use p3_field::AbstractField;
    use p3_field_testing::test_packed_field;
    use p3_monty_31::PackedMontyField31AVX2;

    use super::WIDTH;
    use crate::{BabyBear, BabyBearParameters};

    const SPECIAL_VALS: [BabyBear; WIDTH] = BabyBear::new_array([
        0x00000000, 0x00000001, 0x78000000, 0x77ffffff, 0x3c000000, 0x0ffffffe, 0x68000003,
        0x70000002,
    ]);

    test_packed_field!(
        crate::PackedBabyBearAVX2,
        crate::PackedBabyBearAVX2::zero(),
        p3_monty_31::PackedMontyField31AVX2::<crate::BabyBearParameters>(super::SPECIAL_VALS)
    );

    #[test]
    fn test_cube_vs_mul() {
        let vec = PackedMontyField31AVX2::<BabyBearParameters>(BabyBear::new_array([
            0x4efd5eaf, 0x311b8e0c, 0x74dd27c1, 0x449613f0, 0x017fa32b, 0xfedcba98, 0x12345678,
            0x55555555,
        ]));
        let res0 = vec * vec.square();
        let res1 = vec.cube();
        assert_eq!(res0, res1);
    }

    #[test]
    fn test_cube_vs_scalar() {
        let arr = BabyBear::new_array([
            0x57155037, 0x71bdcc8e, 0x301f94d, 0x435938a6, 0x017fa32b, 0xfedcba98, 0x12345678,
            0x55555555,
        ]);

        let vec = PackedMontyField31AVX2::<BabyBearParameters>(arr);
        let vec_res = vec.cube();

        #[allow(clippy::needless_range_loop)]
        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr[i].cube());
        }
    }

    #[test]
    fn test_cube_vs_scalar_special_vals() {
        let vec = PackedMontyField31AVX2::<BabyBearParameters>(SPECIAL_VALS);
        let vec_res = vec.cube();

        #[allow(clippy::needless_range_loop)]
        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], SPECIAL_VALS[i].cube());
        }
    }

    #[test]
    fn test_exp_5_vs_mul() {
        let vec = PackedMontyField31AVX2::<BabyBearParameters>(BabyBear::new_array([
            0x4efd5eaf, 0x311b8e0c, 0x74dd27c1, 0x449613f0, 0x017fa32b, 0xfedcba98, 0x12345678,
            0x55555555,
        ]));
        let res0 = vec * vec.square() * vec.square();
        let res1 = vec.exp_const_u64::<5>();
        assert_eq!(res0, res1);
    }

    #[test]
    fn test_exp_5_vs_scalar() {
        let arr = BabyBear::new_array([
            0x57155037, 0x71bdcc8e, 0x301f94d, 0x435938a6, 0x017fa32b, 0xfedcba98, 0x12345678,
            0x55555555,
        ]);

        let vec = PackedMontyField31AVX2::<BabyBearParameters>(arr);
        let vec_res = vec.exp_const_u64::<5>();

        #[allow(clippy::needless_range_loop)]
        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr[i].exp_const_u64::<5>());
        }
    }

    #[test]
    fn test_exp_5_vs_scalar_special_vals() {
        let vec = PackedMontyField31AVX2::<BabyBearParameters>(SPECIAL_VALS);
        let vec_res = vec.exp_const_u64::<5>();

        #[allow(clippy::needless_range_loop)]
        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], SPECIAL_VALS[i].exp_const_u64::<5>());
        }
    }

    #[test]
    fn test_exp_7_vs_mul() {
        let vec = PackedMontyField31AVX2::<BabyBearParameters>(BabyBear::new_array([
            0x4efd5eaf, 0x311b8e0c, 0x74dd27c1, 0x449613f0, 0x017fa32b, 0xfedcba98, 0x12345678,
            0x55555555,
        ]));
        let res0 = vec * vec.square() * vec.square() * vec.square();
        let res1 = vec.exp_const_u64::<7>();
        assert_eq!(res0, res1);
    }

    #[test]
    fn test_exp_7_vs_scalar() {
        let arr = BabyBear::new_array([
            0x57155037, 0x71bdcc8e, 0x301f94d, 0x435938a6, 0x017fa32b, 0xfedcba98, 0x12345678,
            0x55555555,
        ]);

        let vec = PackedMontyField31AVX2::<BabyBearParameters>(arr);
        let vec_res = vec.exp_const_u64::<7>();

        #[allow(clippy::needless_range_loop)]
        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr[i].exp_const_u64::<7>());
        }
    }

    #[test]
    fn test_exp_7_vs_scalar_special_vals() {
        let vec = PackedMontyField31AVX2::<BabyBearParameters>(SPECIAL_VALS);
        let vec_res = vec.exp_const_u64::<7>();

        #[allow(clippy::needless_range_loop)]
        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], SPECIAL_VALS[i].exp_const_u64::<7>());
        }
    }
}
