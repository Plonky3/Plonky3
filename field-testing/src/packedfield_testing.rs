use alloc::vec;
use alloc::vec::Vec;

use p3_field::{Field, PackedField, PackedFieldPow2, PackedValue, PrimeCharacteristicRing};
use proptest::prelude::*;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

fn packed_from_random<PV>(seed: u64) -> PV
where
    PV: PackedValue,
    StandardUniform: Distribution<PV::Value>,
{
    let mut rng = SmallRng::seed_from_u64(seed);
    PV::from_fn(|_| rng.random())
}

/// Interleave arr1 and arr2 using chunks of size i.
fn interleave<T: Copy + Default>(arr1: &[T], arr2: &[T], i: usize) -> (Vec<T>, Vec<T>) {
    let width = arr1.len();
    assert_eq!(width, arr2.len());
    assert_eq!(width % i, 0);

    if i == width {
        return (arr1.to_vec(), arr2.to_vec());
    }

    let mut outleft = vec![T::default(); width];
    let mut outright = vec![T::default(); width];

    let mut flag = false;

    for j in 0..width {
        if j.is_multiple_of(i) {
            flag = !flag;
        }
        if flag {
            outleft[j] = arr1[j];
            outleft[j + i] = arr2[j];
        } else {
            outright[j - i] = arr1[j];
            outright[j] = arr2[j];
        }
    }

    (outleft, outright)
}

fn test_interleave<PF>(i: usize)
where
    PF: PackedFieldPow2 + Eq,
    StandardUniform: Distribution<PF::Scalar>,
{
    assert!(PF::WIDTH.is_multiple_of(i));

    let vec1 = packed_from_random::<PF>(0x4ff5dec04791e481);
    let vec2 = packed_from_random::<PF>(0x5806c495e9451f8e);

    let arr1 = vec1.as_slice();
    let arr2 = vec2.as_slice();

    let (res1, res2) = vec1.interleave(vec2, i);
    let (out1, out2) = interleave(arr1, arr2, i);

    assert_eq!(
        res1.as_slice(),
        &out1,
        "Error in left output when testing interleave {i}. Data is: \n {arr1:?} \n {arr2:?} \n {res1:?} \n {res2:?} \n {out1:?} \n {out2:?}.",
    );
    assert_eq!(
        res2.as_slice(),
        &out2,
        "Error in right output when testing interleave {i}.",
    );
}

pub fn test_interleaves<PF>()
where
    PF: PackedFieldPow2 + Eq,
    StandardUniform: Distribution<PF::Scalar>,
{
    let mut i = 1;
    while i <= PF::WIDTH {
        test_interleave::<PF>(i);
        i *= 2;
    }
}

pub fn test_packed_linear_combination<PF: PackedField + Eq>()
where
    StandardUniform: Distribution<PF> + Distribution<PF::Scalar>,
{
    let mut rng = SmallRng::seed_from_u64(1);
    let u: [PF::Scalar; 64] = rng.random();
    let v: [PF; 64] = rng.random();

    let mut dot = PF::ZERO;
    assert_eq!(dot, PF::packed_linear_combination::<0>(&u[..0], &v[..0]));
    dot += v[0] * u[0];
    assert_eq!(dot, PF::packed_linear_combination::<1>(&u[..1], &v[..1]));
    dot += v[1] * u[1];
    assert_eq!(dot, PF::packed_linear_combination::<2>(&u[..2], &v[..2]));
    dot += v[2] * u[2];
    assert_eq!(dot, PF::packed_linear_combination::<3>(&u[..3], &v[..3]));
    dot += v[3] * u[3];
    assert_eq!(dot, PF::packed_linear_combination::<4>(&u[..4], &v[..4]));
    dot += v[4] * u[4];
    assert_eq!(dot, PF::packed_linear_combination::<5>(&u[..5], &v[..5]));
    dot += v[5] * u[5];
    assert_eq!(dot, PF::packed_linear_combination::<6>(&u[..6], &v[..6]));
    dot += v[6] * u[6];
    assert_eq!(dot, PF::packed_linear_combination::<7>(&u[..7], &v[..7]));
    dot += v[7] * u[7];
    assert_eq!(dot, PF::packed_linear_combination::<8>(&u[..8], &v[..8]));
    dot += v[8] * u[8];
    assert_eq!(dot, PF::packed_linear_combination::<9>(&u[..9], &v[..9]));
    dot += v[9] * u[9];
    assert_eq!(dot, PF::packed_linear_combination::<10>(&u[..10], &v[..10]));
    dot += v[10] * u[10];
    assert_eq!(dot, PF::packed_linear_combination::<11>(&u[..11], &v[..11]));
    dot += v[11] * u[11];
    assert_eq!(dot, PF::packed_linear_combination::<12>(&u[..12], &v[..12]));
    dot += v[12] * u[12];
    assert_eq!(dot, PF::packed_linear_combination::<13>(&u[..13], &v[..13]));
    dot += v[13] * u[13];
    assert_eq!(dot, PF::packed_linear_combination::<14>(&u[..14], &v[..14]));
    dot += v[14] * u[14];
    assert_eq!(dot, PF::packed_linear_combination::<15>(&u[..15], &v[..15]));
    dot += v[15] * u[15];
    assert_eq!(dot, PF::packed_linear_combination::<16>(&u[..16], &v[..16]));

    let dot_64: PF = u
        .iter()
        .zip(v.iter())
        .fold(PF::ZERO, |acc, (&lhs, &rhs)| acc + (rhs * lhs));
    assert_eq!(dot_64, PF::packed_linear_combination::<64>(&u, &v));
}

pub fn test_packed_mixed_dot_product<PF: PackedField + Eq>()
where
    StandardUniform: Distribution<PF> + Distribution<PF::Scalar>,
{
    let mut rng = SmallRng::seed_from_u64(42);
    let a: [PF; 64] = rng.random();
    let f: [PF::Scalar; 64] = rng.random();

    let mut dot = PF::ZERO;
    assert_eq!(
        dot,
        PF::mixed_dot_product::<0>(a[..0].try_into().unwrap(), f[..0].try_into().unwrap())
    );
    dot += a[0] * f[0];
    assert_eq!(
        dot,
        PF::mixed_dot_product::<1>(a[..1].try_into().unwrap(), f[..1].try_into().unwrap())
    );
    dot += a[1] * f[1];
    assert_eq!(
        dot,
        PF::mixed_dot_product::<2>(a[..2].try_into().unwrap(), f[..2].try_into().unwrap())
    );
    dot += a[2] * f[2];
    assert_eq!(
        dot,
        PF::mixed_dot_product::<3>(a[..3].try_into().unwrap(), f[..3].try_into().unwrap())
    );
    dot += a[3] * f[3];
    assert_eq!(
        dot,
        PF::mixed_dot_product::<4>(a[..4].try_into().unwrap(), f[..4].try_into().unwrap())
    );
    dot += a[4] * f[4];
    assert_eq!(
        dot,
        PF::mixed_dot_product::<5>(a[..5].try_into().unwrap(), f[..5].try_into().unwrap())
    );
    dot += a[5] * f[5];
    assert_eq!(
        dot,
        PF::mixed_dot_product::<6>(a[..6].try_into().unwrap(), f[..6].try_into().unwrap())
    );
    dot += a[6] * f[6];
    assert_eq!(
        dot,
        PF::mixed_dot_product::<7>(a[..7].try_into().unwrap(), f[..7].try_into().unwrap())
    );
    dot += a[7] * f[7];
    assert_eq!(
        dot,
        PF::mixed_dot_product::<8>(a[..8].try_into().unwrap(), f[..8].try_into().unwrap())
    );
    dot += a[8] * f[8];
    assert_eq!(
        dot,
        PF::mixed_dot_product::<9>(a[..9].try_into().unwrap(), f[..9].try_into().unwrap())
    );
    dot += a[9] * f[9];
    assert_eq!(
        dot,
        PF::mixed_dot_product::<10>(a[..10].try_into().unwrap(), f[..10].try_into().unwrap())
    );
    dot += a[10] * f[10];
    assert_eq!(
        dot,
        PF::mixed_dot_product::<11>(a[..11].try_into().unwrap(), f[..11].try_into().unwrap())
    );
    dot += a[11] * f[11];
    assert_eq!(
        dot,
        PF::mixed_dot_product::<12>(a[..12].try_into().unwrap(), f[..12].try_into().unwrap())
    );
    dot += a[12] * f[12];
    assert_eq!(
        dot,
        PF::mixed_dot_product::<13>(a[..13].try_into().unwrap(), f[..13].try_into().unwrap())
    );
    dot += a[13] * f[13];
    assert_eq!(
        dot,
        PF::mixed_dot_product::<14>(a[..14].try_into().unwrap(), f[..14].try_into().unwrap())
    );
    dot += a[14] * f[14];
    assert_eq!(
        dot,
        PF::mixed_dot_product::<15>(a[..15].try_into().unwrap(), f[..15].try_into().unwrap())
    );
    dot += a[15] * f[15];
    assert_eq!(
        dot,
        PF::mixed_dot_product::<16>(a[..16].try_into().unwrap(), f[..16].try_into().unwrap())
    );

    let dot_64: PF = a
        .iter()
        .zip(f.iter())
        .fold(PF::ZERO, |acc, (&ai, &fi)| acc + (ai * fi));
    assert_eq!(dot_64, PF::mixed_dot_product::<64>(&a, &f));
}

pub fn test_vs_scalar<PF>(special_vals: PF)
where
    PF: PackedField + Eq,
    StandardUniform: Distribution<PF::Scalar>,
{
    let vec0: PF = packed_from_random(0x278d9e202925a1d1);
    let vec1: PF = packed_from_random(0xf04cbac0cbad419f);
    let vec_special = special_vals;

    let arr0 = vec0.as_slice();
    let arr1 = vec1.as_slice();

    let vec_sum = vec0 + vec1;
    let arr_sum = vec_sum.as_slice();
    let vec_special_sum_left = vec_special + vec0;
    let arr_special_sum_left = vec_special_sum_left.as_slice();
    let vec_special_sum_right = vec1 + vec_special;
    let arr_special_sum_right = vec_special_sum_right.as_slice();

    let vec_sub = vec0 - vec1;
    let arr_sub = vec_sub.as_slice();
    let vec_special_sub_left = vec_special - vec0;
    let arr_special_sub_left = vec_special_sub_left.as_slice();
    let vec_special_sub_right = vec1 - vec_special;
    let arr_special_sub_right = vec_special_sub_right.as_slice();

    let vec_mul = vec0 * vec1;
    let arr_mul = vec_mul.as_slice();
    let vec_special_mul_left = vec_special * vec0;
    let arr_special_mul_left = vec_special_mul_left.as_slice();
    let vec_special_mul_right = vec1 * vec_special;
    let arr_special_mul_right = vec_special_mul_right.as_slice();

    let vec_neg = -vec0;
    let arr_neg = vec_neg.as_slice();
    let vec_special_neg = -vec_special;
    let arr_special_neg = vec_special_neg.as_slice();

    let vec_exp_3 = vec0.exp_const_u64::<3>();
    let arr_exp_3 = vec_exp_3.as_slice();
    let vec_special_exp_3 = vec_special.exp_const_u64::<3>();
    let arr_special_exp_3 = vec_special_exp_3.as_slice();

    let vec_exp_5 = vec0.exp_const_u64::<5>();
    let arr_exp_5 = vec_exp_5.as_slice();
    let vec_special_exp_5 = vec_special.exp_const_u64::<5>();
    let arr_special_exp_5 = vec_special_exp_5.as_slice();

    let vec_exp_7 = vec0.exp_const_u64::<7>();
    let arr_exp_7 = vec_exp_7.as_slice();
    let vec_special_exp_7 = vec_special.exp_const_u64::<7>();
    let arr_special_exp_7 = vec_special_exp_7.as_slice();

    let special_vals = special_vals.as_slice();
    for i in 0..PF::WIDTH {
        assert_eq!(
            arr_sum[i],
            arr0[i] + arr1[i],
            "Error when testing add consistency of packed and scalar at location {i}.",
        );
        assert_eq!(
            arr_special_sum_left[i],
            special_vals[i] + arr0[i],
            "Error when testing consistency of left add for special values for packed and scalar at location {i}.",
        );
        assert_eq!(
            arr_special_sum_right[i],
            arr1[i] + special_vals[i],
            "Error when testing consistency of right add for special values for packed and scalar at location {i}.",
        );

        assert_eq!(
            arr_sub[i],
            arr0[i] - arr1[i],
            "Error when testing sub consistency of packed and scalar at location {i}.",
        );
        assert_eq!(
            arr_special_sub_left[i],
            special_vals[i] - arr0[i],
            "Error when testing consistency of left sub for special values for packed and scalar at location {i}.",
        );
        assert_eq!(
            arr_special_sub_right[i],
            arr1[i] - special_vals[i],
            "Error when testing consistency of right sub for special values for packed and scalar at location {i}.",
        );

        assert_eq!(
            arr_mul[i],
            arr0[i] * arr1[i],
            "Error when testing mul consistency of packed and scalar at location {i}.",
        );
        assert_eq!(
            arr_special_mul_left[i],
            special_vals[i] * arr0[i],
            "Error when testing consistency of left mul for special values for packed and scalar at location {i}.",
        );
        assert_eq!(
            arr_special_mul_right[i],
            arr1[i] * special_vals[i],
            "Error when testing consistency of right mul for special values for packed and scalar at location {i}.",
        );

        assert_eq!(
            arr_neg[i], -arr0[i],
            "Error when testing neg consistency of packed and scalar at location {i}.",
        );
        assert_eq!(
            arr_special_neg[i], -special_vals[i],
            "Error when testing consistency of neg for special values for packed and scalar at location {i}.",
        );
        assert_eq!(
            arr_exp_3[i],
            arr0[i].exp_const_u64::<3>(),
            "Error when testing exp_const_u64::<3> consistency of packed and scalar at location {i}.",
        );
        assert_eq!(
            arr_special_exp_3[i],
            special_vals[i].exp_const_u64::<3>(),
            "Error when testing consistency of exp_const_u64::<3> for special values for packed and scalar at location {i}.",
        );
        assert_eq!(
            arr_exp_5[i],
            arr0[i].exp_const_u64::<5>(),
            "Error when testing exp_const_u64::<5> consistency of packed and scalar at location {i}.",
        );
        assert_eq!(
            arr_special_exp_5[i],
            special_vals[i].exp_const_u64::<5>(),
            "Error when testing consistency of exp_const_u64::<5> for special values for packed and scalar at location {i}.",
        );
        assert_eq!(
            arr_exp_7[i],
            arr0[i].exp_const_u64::<7>(),
            "Error when testing exp_const_u64::<7> consistency of packed and scalar at location {i}.",
        );
        assert_eq!(
            arr_special_exp_7[i],
            special_vals[i].exp_const_u64::<7>(),
            "Error when testing consistency of exp_const_u64::<7> for special values for packed and scalar at location {i}.",
        );
    }
}

pub fn test_multiplicative_inverse<PF>()
where
    PF: PackedField + Eq,
    StandardUniform: Distribution<PF::Scalar>,
{
    let vec: PF = packed_from_random(0xb0c7a5153103c5a8);
    let arr = vec.as_slice();
    let vec_inv = PF::from_fn(|i| arr[i].inverse());
    let res = vec * vec_inv;
    assert_eq!(
        res,
        PF::ONE,
        "Error when testing multiplication by inverse."
    );
}

/// Test that [`PackedField::broadcast`] sets all lanes to the same scalar.
pub fn test_broadcast<PF>()
where
    PF: PackedField + Eq,
    StandardUniform: Distribution<PF::Scalar>,
{
    let mut rng = SmallRng::seed_from_u64(0xdeadbeef);
    let x: PF::Scalar = rng.random();
    let packed = PF::broadcast(x);
    for lane in 0..PF::WIDTH {
        assert_eq!(
            packed.as_slice()[lane],
            x,
            "broadcast mismatch at lane {lane}"
        );
    }

    let zero = PF::broadcast(PF::Scalar::default());
    assert_eq!(zero, PF::ZERO, "broadcast(default) should equal ZERO");
}

/// Test dot products with maximum field values (P-1) to catch overflow bugs.
///
/// This verifies that SIMD dot product implementations handle the edge case where
/// `N*(P-1)^2` can overflow `u64` (which happens for N >= 5 with 31-bit primes).
pub fn test_dot_product_boundary<PF>()
where
    PF: PackedField + Eq,
{
    let big = PF::from(PF::Scalar::NEG_ONE);
    let scalar_big = PF::Scalar::NEG_ONE;

    // Test dot_product for N = 1..=16 with all-maximum inputs.
    macro_rules! test_dot_n {
        ($n:literal) => {
            let packed_result = PF::dot_product::<$n>(&[big; $n], &[big; $n]);
            let scalar_result = PF::Scalar::dot_product::<$n>(&[scalar_big; $n], &[scalar_big; $n]);
            for lane in 0..PF::WIDTH {
                assert_eq!(
                    packed_result.as_slice()[lane],
                    scalar_result,
                    "dot_product::<{}> overflow mismatch at lane {}",
                    $n,
                    lane,
                );
            }
        };
    }
    test_dot_n!(1);
    test_dot_n!(2);
    test_dot_n!(3);
    test_dot_n!(4);
    test_dot_n!(5);
    test_dot_n!(6);
    test_dot_n!(7);
    test_dot_n!(8);
    test_dot_n!(9);
    test_dot_n!(10);
    test_dot_n!(11);
    test_dot_n!(12);
    test_dot_n!(13);
    test_dot_n!(14);
    test_dot_n!(15);
    test_dot_n!(16);
}

/// Test packed field operations vs scalar with 256 random packed vectors via proptest.
///
/// Verifies add, sub, mul, neg match lane-by-lane scalar results.
pub fn test_packed_vs_scalar_proptest<PF>()
where
    PF: PackedField + Eq + 'static,
    StandardUniform: Distribution<PF::Scalar>,
{
    let config = ProptestConfig::with_cases(256);
    proptest!(config, |(seed_a in any::<u64>(), seed_b in any::<u64>())| {
        let mut rng_a = SmallRng::seed_from_u64(seed_a);
        let mut rng_b = SmallRng::seed_from_u64(seed_b);
        let a = PF::from_fn(|_| rng_a.random());
        let b = PF::from_fn(|_| rng_b.random());

        let sum = a + b;
        let diff = a - b;
        let prod = a * b;
        let neg_a = -a;

        let sum = sum.as_slice();
        let diff = diff.as_slice();
        let prod = prod.as_slice();
        let neg_a = neg_a.as_slice();
        let a = a.as_slice();
        let b = b.as_slice();

        for i in 0..PF::WIDTH {
            prop_assert_eq!(sum[i], a[i] + b[i],
                "add mismatch at lane {}", i);
            prop_assert_eq!(diff[i], a[i] - b[i],
                "sub mismatch at lane {}", i);
            prop_assert_eq!(prod[i], a[i] * b[i],
                "mul mismatch at lane {}", i);
            prop_assert_eq!(neg_a[i], -a[i],
                "neg mismatch at lane {}", i);
        }
    });
}

#[macro_export]
macro_rules! test_packed_field {
    ($packedfield:ty, $zeros:expr, $ones:expr, $specials:expr) => {
        $crate::test_ring_with_eq!($packedfield, $zeros, $ones);

        mod packed_field_tests {
            use p3_field::PrimeCharacteristicRing;

            #[test]
            fn test_interleaves() {
                $crate::test_interleaves::<$packedfield>();
            }
            #[test]
            fn test_packed_linear_combination() {
                $crate::test_packed_linear_combination::<$packedfield>();
            }
            #[test]
            fn test_packed_mixed_dot_product() {
                $crate::test_packed_mixed_dot_product::<$packedfield>();
            }
            #[test]
            fn test_vs_scalar() {
                $crate::test_vs_scalar::<$packedfield>($specials);
            }
            #[test]
            fn test_multiplicative_inverse() {
                $crate::test_multiplicative_inverse::<$packedfield>();
            }
            #[test]
            fn test_dot_product_boundary() {
                $crate::test_dot_product_boundary::<$packedfield>();
            }
            #[test]
            fn test_broadcast() {
                $crate::test_broadcast::<$packedfield>();
            }
            #[test]
            fn test_packed_vs_scalar_proptest() {
                $crate::test_packed_vs_scalar_proptest::<$packedfield>();
            }
        }
    };
}

#[macro_export]
macro_rules! test_packed_extension_field {
    ($packedextfield:ty, $zeros:expr, $ones:expr) => {
        mod packed_field_tests {
            use p3_field::PrimeCharacteristicRing;

            #[test]
            fn test_ring_with_eq() {
                $crate::test_ring_with_eq::<$packedextfield>($zeros, $ones);
            }
            #[test]
            fn test_mul_2exp_u64() {
                $crate::test_mul_2exp_u64::<$packedextfield>();
            }
            #[test]
            fn test_div_2exp_u64() {
                $crate::test_div_2exp_u64::<$packedextfield>();
            }
            #[test]
            fn test_ring_axioms_proptest() {
                $crate::test_ring_axioms_proptest::<$packedextfield>();
            }
        }
    };
}
