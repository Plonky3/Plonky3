use core::arch::aarch64::{int32x4_t, uint32x4_t};
use core::mem::transmute;

use p3_monty_31::{MontyParametersNeon, PackedMontyField31Neon};

use crate::BabyBearParameters;

const WIDTH: usize = 4;

impl MontyParametersNeon for BabyBearParameters {
    const PACKEDP: uint32x4_t = unsafe { transmute::<[u32; WIDTH], _>([0x78000001; WIDTH]) };
    // This MU is the same 0x88000001 as elsewhere, just interpreted as an `i32`.
    const PACKEDMU: int32x4_t = unsafe { transmute::<[i32; WIDTH], _>([-0x77ffffff; WIDTH]) };
}

pub type PackedBabyBearNeon = PackedMontyField31Neon<BabyBearParameters>;

#[cfg(test)]
mod tests {
    use p3_field_testing::test_packed_field;
    use p3_monty_31::PackedMontyField31Neon;

    use super::WIDTH;
    use crate::{BabyBear, BabyBearParameters};

    const SPECIAL_VALS: [BabyBear; WIDTH] =
        BabyBear::new_array([0x00000000, 0x00000001, 0x00000002, 0x78000000]);

    test_packed_field!(
        crate::PackedBabyBearNeon,
        crate::PackedBabyBearNeon::zero(),
        p3_monty_31::PackedMontyField31Neon::<crate::BabyBearParameters>(super::SPECIAL_VALS)
    );

    #[test]
    fn test_cube_vs_mul() {
        let vec = PackedMontyField31Neon::<BabyBearParameters>(BabyBear::new_array([
            0x4efd5eaf, 0x311b8e0c, 0x74dd27c1, 0x449613f0,
        ]));
        let res0 = vec * vec.square();
        let res1 = vec.cube();
        assert_eq!(res0, res1);
    }

    #[test]
    fn test_cube_vs_scalar() {
        let arr = BabyBear::new_array([0x57155037, 0x71bdcc8e, 0x301f94d, 0x435938a6]);

        let vec = PackedMontyField31Neon::<BabyBearParameters>(arr);
        let vec_res = vec.cube();

        #[allow(clippy::needless_range_loop)]
        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr[i].cube());
        }
    }

    #[test]
    fn test_cube_vs_scalar_special_vals() {
        let vec = PackedMontyField31Neon::<BabyBearParameters>(SPECIAL_VALS);
        let vec_res = vec.cube();

        #[allow(clippy::needless_range_loop)]
        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], SPECIAL_VALS[i].cube());
        }
    }
}
