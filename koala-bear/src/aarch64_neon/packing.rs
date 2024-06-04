use core::arch::aarch64::{int32x4_t, uint32x4_t};

use p3_monty_31::{FieldParametersNeon, PackedMontyField31Neon};

use crate::KoalaBearParameters;

const WIDTH: usize = 4;

impl FieldParametersNeon for KoalaBearParameters {
    const PACKEDP: uint32x4_t = unsafe { transmute::<[u32; WIDTH], _>([0x78000001; WIDTH]) };
    // This MU is the same 0x88000001 as elsewhere, just interpreted as an `i32`.
    const PACKEDMU: int32x4_t = unsafe { transmute::<[i32; WIDTH], _>([-0x77ffffff; WIDTH]) };
}

pub type PackedKoalaBearNeon = PackedMontyField31Neon<KoalaBearParameters>;

#[cfg(test)]
mod tests {
    use p3_field::{AbstractField, Field, PackedField};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    use super::*;
    use crate::KoalaBear;

    type F = KoalaBear;
    type P = PackedKoalaBearNeon;

    fn array_from_canonical(vals: [u32; WIDTH]) -> [F; WIDTH] {
        vals.map(F::from_canonical_u32)
    }

    fn packed_from_canonical(vals: [u32; WIDTH]) -> P {
        PackedMontyField31Neon::<KoalaBearParameters>(array_from_canonical(vals))
    }

    #[test]
    fn test_interleave_1() {
        let vec0 = packed_from_canonical([1, 2, 3, 4]);
        let vec1 = packed_from_canonical([5, 6, 7, 8]);

        let expected0 = packed_from_canonical([1, 5, 3, 7]);
        let expected1 = packed_from_canonical([2, 6, 4, 8]);

        let (res0, res1) = vec0.interleave(vec1, 1);
        assert_eq!(res0, expected0);
        assert_eq!(res1, expected1);
    }

    #[test]
    fn test_interleave_2() {
        let vec0 = packed_from_canonical([1, 2, 3, 4]);
        let vec1 = packed_from_canonical([5, 6, 7, 8]);

        let expected0 = packed_from_canonical([1, 2, 5, 6]);
        let expected1 = packed_from_canonical([3, 4, 7, 8]);

        let (res0, res1) = vec0.interleave(vec1, 2);
        assert_eq!(res0, expected0);
        assert_eq!(res1, expected1);
    }

    #[test]
    fn test_interleave_4() {
        let vec0 = packed_from_canonical([1, 2, 3, 4]);
        let vec1 = packed_from_canonical([5, 6, 7, 8]);

        let (res0, res1) = vec0.interleave(vec1, 4);
        assert_eq!(res0, vec0);
        assert_eq!(res1, vec1);
    }

    #[test]
    fn test_add_associative() {
        let vec0 = packed_from_canonical([0x5379f3d7, 0x702b9db2, 0x6f54190a, 0x0fd40697]);
        let vec1 = packed_from_canonical([0x4e1ce6a6, 0x07100ca0, 0x0f27d0e8, 0x6ab0f017]);
        let vec2 = packed_from_canonical([0x3767261e, 0x46966e27, 0x25690f5a, 0x2ba2b5fa]);

        let res0 = (vec0 + vec1) + vec2;
        let res1 = vec0 + (vec1 + vec2);

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_add_commutative() {
        let vec0 = packed_from_canonical([0x4431e0aa, 0x3f7cac53, 0x6c65b84f, 0x393370c6]);
        let vec1 = packed_from_canonical([0x13f3646a, 0x17bab2b2, 0x154d5424, 0x58a5a24c]);

        let res0 = vec0 + vec1;
        let res1 = vec1 + vec0;

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_additive_identity_right() {
        let vec = packed_from_canonical([0x37585a7d, 0x6f1de589, 0x41e1be7e, 0x712071b8]);
        let res = vec + P::zero();
        assert_eq!(res, vec);
    }

    #[test]
    fn test_additive_identity_left() {
        let vec = packed_from_canonical([0x2456f91e, 0x0783a205, 0x58826627, 0x1a5e3f16]);
        let res = P::zero() + vec;
        assert_eq!(res, vec);
    }

    #[test]
    fn test_additive_inverse_add_neg() {
        let vec = packed_from_canonical([0x28267ebf, 0x0b83d23e, 0x67a59e3d, 0x0ba2fb25]);
        let neg_vec = -vec;
        let res = vec + neg_vec;
        assert_eq!(res, P::zero());
    }

    #[test]
    fn test_additive_inverse_sub() {
        let vec = packed_from_canonical([0x2f0a7c0e, 0x50163480, 0x12eac826, 0x2e52b121]);
        let res = vec - vec;
        assert_eq!(res, P::zero());
    }

    #[test]
    fn test_sub_anticommutative() {
        let vec0 = packed_from_canonical([0x0a715ea4, 0x17877e5e, 0x1a67e27c, 0x29e13b42]);
        let vec1 = packed_from_canonical([0x4168263c, 0x3c9fc759, 0x435424e9, 0x5cac2afd]);

        let res0 = vec0 - vec1;
        let res1 = -(vec1 - vec0);

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_sub_zero() {
        let vec = packed_from_canonical([0x10df1248, 0x65050015, 0x73151d8d, 0x443341a8]);
        let res = vec - P::zero();
        assert_eq!(res, vec);
    }

    #[test]
    fn test_zero_sub() {
        let vec = packed_from_canonical([0x1af0d41c, 0x3c1795f4, 0x54da13f5, 0x43cd3f94]);
        let res0 = P::zero() - vec;
        let res1 = -vec;
        assert_eq!(res0, res1);
    }

    #[test]
    fn test_neg_own_inverse() {
        let vec = packed_from_canonical([0x25335335, 0x32d48910, 0x74468a5f, 0x61906a18]);
        let res = -(-vec);
        assert_eq!(res, vec);
    }

    #[test]
    fn test_sub_is_add_neg() {
        let vec0 = packed_from_canonical([0x2ab6719a, 0x0991137e, 0x0e5c6bea, 0x1dbbb162]);
        let vec1 = packed_from_canonical([0x26c7239d, 0x56a2318b, 0x1a839b59, 0x1ec6f977]);
        let res0 = vec0 - vec1;
        let res1 = vec0 + (-vec1);
        assert_eq!(res0, res1);
    }

    #[test]
    fn test_mul_associative() {
        let vec0 = packed_from_canonical([0x3b442fc7, 0x15b736fc, 0x5daa6c48, 0x4995dea0]);
        let vec1 = packed_from_canonical([0x582918b6, 0x55b89326, 0x3b579856, 0x10769872]);
        let vec2 = packed_from_canonical([0x6a7bbe26, 0x7139a20b, 0x280f42d5, 0x0efde6a8]);

        let res0 = (vec0 * vec1) * vec2;
        let res1 = vec0 * (vec1 * vec2);

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_mul_commutative() {
        let vec0 = packed_from_canonical([0x18e2fe1a, 0x54cb2eed, 0x35662447, 0x5be20656]);
        let vec1 = packed_from_canonical([0x7715ab49, 0x1937ec0d, 0x561c3def, 0x14f502f9]);

        let res0 = vec0 * vec1;
        let res1 = vec1 * vec0;

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_multiplicative_identity_right() {
        let vec = packed_from_canonical([0x64628378, 0x345e3dc8, 0x766770eb, 0x21e5ad7c]);
        let res = vec * P::one();
        assert_eq!(res, vec);
    }

    #[test]
    fn test_multiplicative_identity_left() {
        let vec = packed_from_canonical([0x48910ae4, 0x4dd95ad3, 0x334eaf5e, 0x44e5d03b]);
        let res = P::one() * vec;
        assert_eq!(res, vec);
    }

    #[test]
    fn test_multiplicative_inverse() {
        let vec = packed_from_canonical([0x1b288c21, 0x600c50af, 0x3ea44d7a, 0x62209fc9]);
        let inverses = packed_from_canonical([0x4c2a9dbd, 0x6a8fef49, 0x61816d95, 0x0dddd700]);
        let res = vec * inverses;
        assert_eq!(res, P::one());
    }

    #[test]
    fn test_mul_zero() {
        let vec = packed_from_canonical([0x675f87cd, 0x2bb57f1b, 0x1b636b90, 0x25fd5dbc]);
        let res = vec * P::zero();
        assert_eq!(res, P::zero());
    }

    #[test]
    fn test_zero_mul() {
        let vec = packed_from_canonical([0x76d898cd, 0x12fed26d, 0x385dd0ea, 0x0a6cfb68]);
        let res = P::zero() * vec;
        assert_eq!(res, P::zero());
    }

    #[test]
    fn test_mul_negone() {
        let vec = packed_from_canonical([0x3ac44c8d, 0x2690778c, 0x64c25465, 0x60c62b6d]);
        let res0 = vec * P::neg_one();
        let res1 = -vec;
        assert_eq!(res0, res1);
    }

    #[test]
    fn test_negone_mul() {
        let vec = packed_from_canonical([0x45fdb5d9, 0x3e2571d7, 0x1438d182, 0x6addc720]);
        let res0 = P::neg_one() * vec;
        let res1 = -vec;
        assert_eq!(res0, res1);
    }

    #[test]
    fn test_neg_distributivity_left() {
        let vec0 = packed_from_canonical([0x347079a0, 0x09f865aa, 0x3f469975, 0x48436fa4]);
        let vec1 = packed_from_canonical([0x354839ad, 0x6f464895, 0x2afb410c, 0x2918c070]);

        let res0 = vec0 * -vec1;
        let res1 = -(vec0 * vec1);

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_neg_distributivity_right() {
        let vec0 = packed_from_canonical([0x62fda8dc, 0x15a702d3, 0x4ee8e5a4, 0x2e8ea106]);
        let vec1 = packed_from_canonical([0x606f79ae, 0x3cc952a6, 0x43e31901, 0x34721ad8]);

        let res0 = -vec0 * vec1;
        let res1 = -(vec0 * vec1);

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_add_distributivity_left() {
        let vec0 = packed_from_canonical([0x46b0c8a7, 0x1f3058ee, 0x44451138, 0x3c97af99]);
        let vec1 = packed_from_canonical([0x6247b46a, 0x0614b336, 0x76730d3c, 0x15b1ab60]);
        let vec2 = packed_from_canonical([0x20619eaf, 0x628800a8, 0x672c9d96, 0x44de32c3]);

        let res0 = vec0 * (vec1 + vec2);
        let res1 = vec0 * vec1 + vec0 * vec2;

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_add_distributivity_right() {
        let vec0 = packed_from_canonical([0x0829c9c5, 0x6b66bdcb, 0x4e906be1, 0x16f11cfa]);
        let vec1 = packed_from_canonical([0x482922d7, 0x72816043, 0x5d63df54, 0x58ca0b7d]);
        let vec2 = packed_from_canonical([0x2127f6c0, 0x0814236c, 0x339d4b6f, 0x24d2b44d]);

        let res0 = (vec0 + vec1) * vec2;
        let res1 = vec0 * vec2 + vec1 * vec2;

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_sub_distributivity_left() {
        let vec0 = packed_from_canonical([0x1c123d16, 0x62d3de88, 0x64ff0336, 0x474de37c]);
        let vec1 = packed_from_canonical([0x06758404, 0x295c96ca, 0x6ffbc647, 0x3b111808]);
        let vec2 = packed_from_canonical([0x591a66de, 0x6b69fbb6, 0x2d206c14, 0x6e5f7d0d]);

        let res0 = vec0 * (vec1 - vec2);
        let res1 = vec0 * vec1 - vec0 * vec2;

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_sub_distributivity_right() {
        let vec0 = packed_from_canonical([0x00252ae1, 0x3e07c401, 0x6fd67c67, 0x767af10f]);
        let vec1 = packed_from_canonical([0x5c44c949, 0x180dc429, 0x0ccd2a7b, 0x51258be1]);
        let vec2 = packed_from_canonical([0x5126fb21, 0x58ed3919, 0x6a2f735d, 0x05ab2a69]);

        let res0 = (vec0 - vec1) * vec2;
        let res1 = vec0 * vec2 - vec1 * vec2;

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_one_plus_one() {
        assert_eq!(P::one() + P::one(), P::two());
    }

    #[test]
    fn test_negone_plus_two() {
        assert_eq!(P::neg_one() + P::two(), P::one());
    }

    #[test]
    fn test_double() {
        let vec = packed_from_canonical([0x6fc7aefd, 0x5166e726, 0x21e648d2, 0x1dd0790f]);
        let res0 = P::two() * vec;
        let res1 = vec + vec;
        assert_eq!(res0, res1);
    }

    #[test]
    fn test_add_vs_scalar() {
        let arr0 = array_from_canonical([0x496d8163, 0x68125590, 0x191cd03b, 0x65b9abef]);
        let arr1 = array_from_canonical([0x6db594e1, 0x5b1f6289, 0x74f15e13, 0x546936a8]);

        let vec0 = PackedMontyField31Neon::<KoalaBearParameters>(arr0);
        let vec1 = PackedMontyField31Neon::<KoalaBearParameters>(arr1);
        let vec_res = vec0 + vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] + arr1[i]);
        }
    }

    #[test]
    fn test_add_vs_scalar_special_vals_left() {
        let arr0 = [F::zero(), F::one(), F::two(), F::neg_one()];
        let arr1 = array_from_canonical([0x4205a2f6, 0x6f4715f1, 0x29ed7f70, 0x70915992]);

        let vec0 = PackedMontyField31Neon::<KoalaBearParameters>(arr0);
        let vec1 = PackedMontyField31Neon::<KoalaBearParameters>(arr1);
        let vec_res = vec0 + vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] + arr1[i]);
        }
    }

    #[test]
    fn test_add_vs_scalar_special_vals_right() {
        let arr0 = array_from_canonical([0x5f8329a7, 0x0f1166bb, 0x657bcb14, 0x0185c34a]);
        let arr1 = [F::zero(), F::one(), F::two(), F::neg_one()];

        let vec0 = PackedMontyField31Neon::<KoalaBearParameters>(arr0);
        let vec1 = PackedMontyField31Neon::<KoalaBearParameters>(arr1);
        let vec_res = vec0 + vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] + arr1[i]);
        }
    }

    #[test]
    fn test_sub_vs_scalar() {
        let arr0 = array_from_canonical([0x6daef778, 0x0e868440, 0x54e7ca64, 0x01a9acab]);
        let arr1 = array_from_canonical([0x45609584, 0x67b63536, 0x0f72a573, 0x234a312e]);

        let vec0 = PackedMontyField31Neon::<KoalaBearParameters>(arr0);
        let vec1 = PackedMontyField31Neon::<KoalaBearParameters>(arr1);
        let vec_res = vec0 - vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] - arr1[i]);
        }
    }

    #[test]
    fn test_sub_vs_scalar_special_vals_left() {
        let arr0 = [F::zero(), F::one(), F::two(), F::neg_one()];
        let arr1 = array_from_canonical([0x4205a2f6, 0x6f4715f1, 0x29ed7f70, 0x70915992]);

        let vec0 = PackedMontyField31Neon::<KoalaBearParameters>(arr0);
        let vec1 = PackedMontyField31Neon::<KoalaBearParameters>(arr1);
        let vec_res = vec0 - vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] - arr1[i]);
        }
    }

    #[test]
    fn test_sub_vs_scalar_special_vals_right() {
        let arr0 = array_from_canonical([0x5f8329a7, 0x0f1166bb, 0x657bcb14, 0x0185c34a]);
        let arr1 = [F::zero(), F::one(), F::two(), F::neg_one()];

        let vec0 = PackedMontyField31Neon::<KoalaBearParameters>(arr0);
        let vec1 = PackedMontyField31Neon::<KoalaBearParameters>(arr1);
        let vec_res = vec0 - vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] - arr1[i]);
        }
    }

    #[test]
    fn test_mul_vs_scalar() {
        let arr0 = array_from_canonical([0x13655880, 0x5223ea02, 0x5d7f4f90, 0x1494b624]);
        let arr1 = array_from_canonical([0x0ad5743c, 0x44956741, 0x533bc885, 0x7723a25b]);

        let vec0 = PackedMontyField31Neon::<KoalaBearParameters>(arr0);
        let vec1 = PackedMontyField31Neon::<KoalaBearParameters>(arr1);
        let vec_res = vec0 * vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] * arr1[i]);
        }
    }

    #[test]
    fn test_mul_vs_scalar_special_vals_left() {
        let arr0 = [F::zero(), F::one(), F::two(), F::neg_one()];
        let arr1 = array_from_canonical([0x4205a2f6, 0x6f4715f1, 0x29ed7f70, 0x70915992]);

        let vec0 = PackedMontyField31Neon::<KoalaBearParameters>(arr0);
        let vec1 = PackedMontyField31Neon::<KoalaBearParameters>(arr1);
        let vec_res = vec0 * vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] * arr1[i]);
        }
    }

    #[test]
    fn test_mul_vs_scalar_special_vals_right() {
        let arr0 = array_from_canonical([0x5f8329a7, 0x0f1166bb, 0x657bcb14, 0x0185c34a]);
        let arr1 = [F::zero(), F::one(), F::two(), F::neg_one()];

        let vec0 = PackedMontyField31Neon::<KoalaBearParameters>(arr0);
        let vec1 = PackedMontyField31Neon::<KoalaBearParameters>(arr1);
        let vec_res = vec0 * vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] * arr1[i]);
        }
    }

    #[test]
    fn test_neg_vs_scalar() {
        let arr = array_from_canonical([0x1971a7b5, 0x00305be1, 0x52c08410, 0x39cb2586]);

        let vec = PackedMontyField31Neon::<KoalaBearParameters>(arr);
        let vec_res = -vec;

        #[allow(clippy::needless_range_loop)]
        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], -arr[i]);
        }
    }

    #[test]
    fn test_neg_vs_scalar_special_vals() {
        let arr = [F::zero(), F::one(), F::two(), F::neg_one()];

        let vec = PackedMontyField31Neon::<KoalaBearParameters>(arr);
        let vec_res = -vec;

        #[allow(clippy::needless_range_loop)]
        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], -arr[i]);
        }
    }

    #[test]
    fn test_cube_vs_mul() {
        let vec = packed_from_canonical([0x4efd5eaf, 0x311b8e0c, 0x74dd27c1, 0x449613f0]);
        let res0 = vec * vec.square();
        let res1 = vec.cube();
        assert_eq!(res0, res1);
    }

    #[test]
    fn test_cube_vs_scalar() {
        let arr = array_from_canonical([0x57155037, 0x71bdcc8e, 0x301f94d, 0x435938a6]);

        let vec = PackedMontyField31Neon::<KoalaBearParameters>(arr);
        let vec_res = vec.cube();

        #[allow(clippy::needless_range_loop)]
        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr[i].cube());
        }
    }

    #[test]
    fn test_cube_vs_scalar_special_vals() {
        let arr = [F::zero(), F::one(), F::two(), F::neg_one()];

        let vec = PackedMontyField31Neon::<KoalaBearParameters>(arr);
        let vec_res = vec.cube();

        #[allow(clippy::needless_range_loop)]
        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr[i].cube());
        }
    }
}
