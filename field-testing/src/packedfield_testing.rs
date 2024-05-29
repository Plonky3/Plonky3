use p3_field::{Field, PackedField, PackedValue};
use core::{array, usize};


pub trait PackedTestingHelpers<const WIDTH: usize, F, PF>
where
    F: Field,
    PF: PackedField<Scalar = F> + PackedValue
{
    fn array_from_valid_reps(vals: [u32; WIDTH]) -> [F; WIDTH];
    fn packed_from_valid_reps(vals: [u32; WIDTH]) -> PF;
    fn array_from_random(seed: u64) -> [F; WIDTH];
    fn packed_from_random(seed: u64);
    const SPECIAL_VALS: [F; WIDTH];
}

/// Interleave arr1 and arr2 using chuncks of size i.
fn interleave<const WIDTH: usize>(arr1: [u32; WIDTH], arr2: [u32; WIDTH], i: usize) -> ([u32; WIDTH], [u32; WIDTH]) {
    assert!(WIDTH%i == 0);

    if i == WIDTH {
        return (arr1, arr2)
    }

    let mut outleft = [0_u32; WIDTH];
    let mut outright = [0_u32; WIDTH];

    let mut flag = false;

    for j in 0..WIDTH{
        if j%i == 0 {
            flag = !flag;
        }
        if flag {
            outleft[j] = arr1[j];
            outleft[j + i] = arr2[j];
        }
        else {
            outright[j - i] = arr1[j];
            outright[j] = arr2[j];
        }
    }

    (outleft, outright)
}

pub fn test_interleave<const WIDTH: usize, F, PF, PTH>(i: usize)
where
    F: Field,
    PF: PackedField<Scalar = F> + PackedValue + Eq,
    PTH: PackedTestingHelpers<WIDTH, F, PF>,
{
    assert!(WIDTH%i == 0);

    let arr1 = array::from_fn(|i| i as u32);
    let arr2 = array::from_fn(|i| (WIDTH + i) as u32);

    let vec0 = PTH::packed_from_valid_reps(arr1);
    let vec1 = PTH::packed_from_valid_reps(arr2);
    let (res0, res1) = vec0.interleave(vec1, 1);

    let (out1, out2) = interleave(arr1, arr2, i);

    let expected0 = PTH::packed_from_valid_reps(out1);
    let expected1 = PTH::packed_from_valid_reps(out2);
    
    assert_eq!(res0, expected0);
    assert_eq!(res1, expected1);
}

// #[test]
// fn test_add_associative() {
//     let vec0 = packed_from_random(0x8b078c2b693c893f);
//     let vec1 = packed_from_random(0x4ff5dec04791e481);
//     let vec2 = packed_from_random(0x5806c495e9451f8e);

//     let res0 = (vec0 + vec1) + vec2;
//     let res1 = vec0 + (vec1 + vec2);

//     assert_eq!(res0, res1);
// }

// #[test]
// fn test_add_commutative() {
//     let vec0 = packed_from_random(0xe1bf9cac02e9072a);
//     let vec1 = packed_from_random(0xb5061e7de6a6c677);

//     let res0 = vec0 + vec1;
//     let res1 = vec1 + vec0;

//     assert_eq!(res0, res1);
// }

// #[test]
// fn test_additive_identity_right() {
//     let vec = packed_from_random(0xbcd56facf6a714b5);
//     let res = vec + P::zero();
//     assert_eq!(res, vec);
// }

// #[test]
// fn test_additive_identity_left() {
//     let vec = packed_from_random(0xb614285cd641233c);
//     let res = P::zero() + vec;
//     assert_eq!(res, vec);
// }

// #[test]
// fn test_additive_inverse_add_neg() {
//     let vec = packed_from_random(0x4b89c8d023c9c62e);
//     let neg_vec = -vec;
//     let res = vec + neg_vec;
//     assert_eq!(res, P::zero());
// }

// #[test]
// fn test_additive_inverse_sub() {
//     let vec = packed_from_random(0x2c94652ee5561341);
//     let res = vec - vec;
//     assert_eq!(res, P::zero());
// }

// #[test]
// fn test_sub_anticommutative() {
//     let vec0 = packed_from_random(0xf3783730a14b460e);
//     let vec1 = packed_from_random(0x5b6f827a023525ee);

//     let res0 = vec0 - vec1;
//     let res1 = -(vec1 - vec0);

//     assert_eq!(res0, res1);
// }

// #[test]
// fn test_sub_zero() {
//     let vec = packed_from_random(0xc1a526f8226ec1e5);
//     let res = vec - P::zero();
//     assert_eq!(res, vec);
// }

// #[test]
// fn test_zero_sub() {
//     let vec = packed_from_random(0x4444b9c090519333);
//     let res0 = P::zero() - vec;
//     let res1 = -vec;
//     assert_eq!(res0, res1);
// }

// #[test]
// fn test_neg_own_inverse() {
//     let vec = packed_from_random(0xee4df174b850a35f);
//     let res = -(-vec);
//     assert_eq!(res, vec);
// }

// #[test]
// fn test_sub_is_add_neg() {
//     let vec0 = packed_from_random(0x18f4b5c3a08e49fe);
//     let vec1 = packed_from_random(0x39bd37a1dc24d492);
//     let res0 = vec0 - vec1;
//     let res1 = vec0 + (-vec1);
//     assert_eq!(res0, res1);
// }

// #[test]
// fn test_mul_associative() {
//     let vec0 = packed_from_random(0x0b1ee4d7c979d50c);
//     let vec1 = packed_from_random(0x39faa0844a36e45a);
//     let vec2 = packed_from_random(0x08fac4ee76260e44);

//     let res0 = (vec0 * vec1) * vec2;
//     let res1 = vec0 * (vec1 * vec2);

//     assert_eq!(res0, res1);
// }

// #[test]
// fn test_mul_commutative() {
//     let vec0 = packed_from_random(0x10debdcbd409270c);
//     let vec1 = packed_from_random(0x927bc073c1c92b2f);

//     let res0 = vec0 * vec1;
//     let res1 = vec1 * vec0;

//     assert_eq!(res0, res1);
// }

// #[test]
// fn test_multiplicative_identity_right() {
//     let vec = packed_from_random(0xdf0a646b6b0c2c36);
//     let res = vec * P::one();
//     assert_eq!(res, vec);
// }

// #[test]
// fn test_multiplicative_identity_left() {
//     let vec = packed_from_random(0x7b4d890bf7a38bd2);
//     let res = P::one() * vec;
//     assert_eq!(res, vec);
// }

// #[test]
// fn test_multiplicative_inverse() {
//     let arr = array_from_random(0xb0c7a5153103c5a8);
//     let arr_inv = arr.map(|x| x.inverse());

//     let vec = PackedKoalaBearAVX2(arr);
//     let vec_inv = PackedKoalaBearAVX2(arr_inv);

//     let res = vec * vec_inv;
//     assert_eq!(res, P::one());
// }

// #[test]
// fn test_mul_zero() {
//     let vec = packed_from_random(0x7f998daa72489bd7);
//     let res = vec * P::zero();
//     assert_eq!(res, P::zero());
// }

// #[test]
// fn test_zero_mul() {
//     let vec = packed_from_random(0x683bc2dd355b06e5);
//     let res = P::zero() * vec;
//     assert_eq!(res, P::zero());
// }

// #[test]
// fn test_mul_negone() {
//     let vec = packed_from_random(0x97cb9670a8251202);
//     let res0 = vec * P::neg_one();
//     let res1 = -vec;
//     assert_eq!(res0, res1);
// }

// #[test]
// fn test_negone_mul() {
//     let vec = packed_from_random(0xadae69873b5d3baf);
//     let res0 = P::neg_one() * vec;
//     let res1 = -vec;
//     assert_eq!(res0, res1);
// }

// #[test]
// fn test_neg_distributivity_left() {
//     let vec0 = packed_from_random(0xd0efd6f272c7de93);
//     let vec1 = packed_from_random(0xd5dd2cf5e76dd694);

//     let res0 = vec0 * -vec1;
//     let res1 = -(vec0 * vec1);

//     assert_eq!(res0, res1);
// }

// #[test]
// fn test_neg_distributivity_right() {
//     let vec0 = packed_from_random(0x0da9b03cd4b79b09);
//     let vec1 = packed_from_random(0x9964d3f4beaf1857);

//     let res0 = -vec0 * vec1;
//     let res1 = -(vec0 * vec1);

//     assert_eq!(res0, res1);
// }

// #[test]
// fn test_add_distributivity_left() {
//     let vec0 = packed_from_random(0x278d9e202925a1d1);
//     let vec1 = packed_from_random(0xf04cbac0cbad419f);
//     let vec2 = packed_from_random(0x76976e2abdc5a056);

//     let res0 = vec0 * (vec1 + vec2);
//     let res1 = vec0 * vec1 + vec0 * vec2;

//     assert_eq!(res0, res1);
// }

// #[test]
// fn test_add_distributivity_right() {
//     let vec0 = packed_from_random(0xbe1b606eafe2a2b8);
//     let vec1 = packed_from_random(0x552686a0978ab571);
//     let vec2 = packed_from_random(0x36f6eec4fd31a460);

//     let res0 = (vec0 + vec1) * vec2;
//     let res1 = vec0 * vec2 + vec1 * vec2;

//     assert_eq!(res0, res1);
// }

// #[test]
// fn test_sub_distributivity_left() {
//     let vec0 = packed_from_random(0x817d4a27febb0349);
//     let vec1 = packed_from_random(0x1eaf62a921d6519b);
//     let vec2 = packed_from_random(0xfec0fb8d3849465a);

//     let res0 = vec0 * (vec1 - vec2);
//     let res1 = vec0 * vec1 - vec0 * vec2;

//     assert_eq!(res0, res1);
// }

// #[test]
// fn test_sub_distributivity_right() {
//     let vec0 = packed_from_random(0x5a4a82e8e2394585);
//     let vec1 = packed_from_random(0x6006b1443a22b102);
//     let vec2 = packed_from_random(0x5a22deac65fcd454);

//     let res0 = (vec0 - vec1) * vec2;
//     let res1 = vec0 * vec2 - vec1 * vec2;

//     assert_eq!(res0, res1);
// }

// #[test]
// fn test_one_plus_one() {
//     assert_eq!(P::one() + P::one(), P::two());
// }

// #[test]
// fn test_negone_plus_two() {
//     assert_eq!(P::neg_one() + P::two(), P::one());
// }

// #[test]
// fn test_double() {
//     let vec = packed_from_random(0x2e61a907650881e9);
//     let res0 = P::two() * vec;
//     let res1 = vec + vec;
//     assert_eq!(res0, res1);
// }

// #[test]
// fn test_add_vs_scalar() {
//     let arr0 = array_from_random(0xac23b5a694dabf70);
//     let arr1 = array_from_random(0xd249ec90e8a6e733);

//     let vec0 = PackedKoalaBearAVX2(arr0);
//     let vec1 = PackedKoalaBearAVX2(arr1);
//     let vec_res = vec0 + vec1;

//     for i in 0..WIDTH {
//         assert_eq!(vec_res.0[i], arr0[i] + arr1[i]);
//     }
// }

// #[test]
// fn test_add_vs_scalar_special_vals_left() {
//     let arr0 = SPECIAL_VALS;
//     let arr1 = array_from_random(0x1e2b153f07b64cf3);

//     let vec0 = PackedKoalaBearAVX2(arr0);
//     let vec1 = PackedKoalaBearAVX2(arr1);
//     let vec_res = vec0 + vec1;

//     for i in 0..WIDTH {
//         assert_eq!(vec_res.0[i], arr0[i] + arr1[i]);
//     }
// }

// #[test]
// fn test_add_vs_scalar_special_vals_right() {
//     let arr0 = array_from_random(0xfcf974ac7625a260);
//     let arr1 = SPECIAL_VALS;

//     let vec0 = PackedKoalaBearAVX2(arr0);
//     let vec1 = PackedKoalaBearAVX2(arr1);
//     let vec_res = vec0 + vec1;

//     for i in 0..WIDTH {
//         assert_eq!(vec_res.0[i], arr0[i] + arr1[i]);
//     }
// }

// #[test]
// fn test_sub_vs_scalar() {
//     let arr0 = array_from_random(0x167ce9d8e920876e);
//     let arr1 = array_from_random(0x52ddcdd3461e046f);

//     let vec0 = PackedKoalaBearAVX2(arr0);
//     let vec1 = PackedKoalaBearAVX2(arr1);
//     let vec_res = vec0 - vec1;

//     for i in 0..WIDTH {
//         assert_eq!(vec_res.0[i], arr0[i] - arr1[i]);
//     }
// }

// #[test]
// fn test_sub_vs_scalar_special_vals_left() {
//     let arr0 = SPECIAL_VALS;
//     let arr1 = array_from_random(0x358498640bfe1375);

//     let vec0 = PackedKoalaBearAVX2(arr0);
//     let vec1 = PackedKoalaBearAVX2(arr1);
//     let vec_res = vec0 - vec1;

//     for i in 0..WIDTH {
//         assert_eq!(vec_res.0[i], arr0[i] - arr1[i]);
//     }
// }

// #[test]
// fn test_sub_vs_scalar_special_vals_right() {
//     let arr0 = array_from_random(0x05d81ebfb8f0005c);
//     let arr1 = SPECIAL_VALS;

//     let vec0 = PackedKoalaBearAVX2(arr0);
//     let vec1 = PackedKoalaBearAVX2(arr1);
//     let vec_res = vec0 - vec1;

//     for i in 0..WIDTH {
//         assert_eq!(vec_res.0[i], arr0[i] - arr1[i]);
//     }
// }

// #[test]
// fn test_mul_vs_scalar() {
//     let arr0 = array_from_random(0x4242ebdc09b74d77);
//     let arr1 = array_from_random(0x9937b275b3c056cd);

//     let vec0 = PackedKoalaBearAVX2(arr0);
//     let vec1 = PackedKoalaBearAVX2(arr1);
//     let vec_res = vec0 * vec1;

//     for i in 0..WIDTH {
//         assert_eq!(vec_res.0[i], arr0[i] * arr1[i]);
//     }
// }

// #[test]
// fn test_mul_vs_scalar_special_vals_left() {
//     let arr0 = SPECIAL_VALS;
//     let arr1 = array_from_random(0x5285448b835458a3);

//     let vec0 = PackedKoalaBearAVX2(arr0);
//     let vec1 = PackedKoalaBearAVX2(arr1);
//     let vec_res = vec0 * vec1;

//     for i in 0..WIDTH {
//         assert_eq!(vec_res.0[i], arr0[i] * arr1[i]);
//     }
// }

// #[test]
// fn test_mul_vs_scalar_special_vals_right() {
//     let arr0 = array_from_random(0x22508dc80001d865);
//     let arr1 = SPECIAL_VALS;

//     let vec0 = PackedKoalaBearAVX2(arr0);
//     let vec1 = PackedKoalaBearAVX2(arr1);
//     let vec_res = vec0 * vec1;

//     for i in 0..WIDTH {
//         assert_eq!(vec_res.0[i], arr0[i] * arr1[i]);
//     }
// }

// #[test]
// fn test_neg_vs_scalar() {
//     let arr = array_from_random(0xc3c273a9b334372f);

//     let vec = PackedKoalaBearAVX2(arr);
//     let vec_res = -vec;

//     #[allow(clippy::needless_range_loop)]
//     for i in 0..WIDTH {
//         assert_eq!(vec_res.0[i], -arr[i]);
//     }
// }

// #[test]
// fn test_neg_vs_scalar_special_vals() {
//     let arr = SPECIAL_VALS;

//     let vec = PackedKoalaBearAVX2(arr);
//     let vec_res = -vec;

//     #[allow(clippy::needless_range_loop)]
//     for i in 0..WIDTH {
//         assert_eq!(vec_res.0[i], -arr[i]);
//     }
// }
