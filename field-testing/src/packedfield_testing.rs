use alloc::vec;
use alloc::vec::Vec;

use p3_field::{AbstractField, Field, PackedField, PackedValue};
use rand::distributions::{Distribution, Standard};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn packed_from_random<PV>(seed: u64) -> PV
where
    PV: PackedValue,
    Standard: Distribution<PV::Value>,
{
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    PV::from_fn(|_| rng.gen())
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
        if j % i == 0 {
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
    PF: PackedField + Eq,
    Standard: Distribution<PF::Scalar>,
{
    assert!(PF::WIDTH % i == 0);

    let vec1 = packed_from_random::<PF>(0x4ff5dec04791e481);
    let vec2 = packed_from_random::<PF>(0x5806c495e9451f8e);

    let arr1 = vec1.as_slice();
    let arr2 = vec2.as_slice();

    let (res1, res2) = vec1.interleave(vec2, i);
    let (out1, out2) = interleave(arr1, arr2, i);

    assert_eq!(
        res1.as_slice(),
        &out1,
        "Error in left output when testing interleave {}.",
        i
    );
    assert_eq!(
        res2.as_slice(),
        &out2,
        "Error in right output when testing interleave {}.",
        i
    );
}

pub fn test_interleaves<PF>()
where
    PF: PackedField + Eq,
    Standard: Distribution<PF::Scalar>,
{
    let mut i = 1;
    while i <= PF::WIDTH {
        test_interleave::<PF>(i);
        i *= 2;
    }
}

#[allow(clippy::eq_op)]
pub fn test_add_neg<PF>(zeros: PF)
where
    PF: PackedField + Eq,
    Standard: Distribution<PF::Scalar>,
{
    let vec0 = packed_from_random::<PF>(0x8b078c2b693c893f);
    let vec1 = packed_from_random::<PF>(0x4ff5dec04791e481);
    let vec2 = packed_from_random::<PF>(0x5806c495e9451f8e);

    assert_eq!(
        (vec0 + vec1) + vec2,
        vec0 + (vec1 + vec2),
        "Error when testing associativity of add."
    );
    assert_eq!(
        vec0 + vec1,
        vec1 + vec0,
        "Error when testing commutativity of add."
    );
    assert_eq!(
        vec0,
        vec0 + zeros,
        "Error when testing additive identity right."
    );
    assert_eq!(
        vec0,
        zeros + vec0,
        "Error when testing additive identity left."
    );
    assert_eq!(
        vec0 + (-vec0),
        PF::zero(),
        "Error when testing additive inverse."
    );
    assert_eq!(
        vec0 - vec0,
        PF::zero(),
        "Error when testing subtracting of self."
    );
    assert_eq!(
        vec0 - vec1,
        -(vec1 - vec0),
        "Error when testing anticommutativity of sub."
    );
    assert_eq!(vec0, vec0 - zeros, "Error when testing subtracting zero.");
    assert_eq!(
        -vec0,
        zeros - vec0,
        "Error when testing subtracting from zero"
    );
    assert_eq!(vec0, -(-vec0), "Error when testing double negation");
    assert_eq!(
        vec0 - vec1,
        vec0 + (-vec1),
        "Error when testing addition of negation"
    );
    assert_eq!(PF::one() + PF::one(), PF::two(), "Error 1 + 1 =/= 2");
    assert_eq!(PF::neg_one() + PF::two(), PF::one(), "Error -1 + 2 =/= 1");
    assert_eq!(
        vec0.double(),
        vec0 + vec0,
        "Error when comparing x.double() to x + x"
    );
}

pub fn test_mul<PF>(zeros: PF)
where
    PF: PackedField + Eq,
    Standard: Distribution<PF::Scalar>,
{
    let vec0 = packed_from_random::<PF>(0x0b1ee4d7c979d50c);
    let vec1 = packed_from_random::<PF>(0x39faa0844a36e45a);
    let vec2 = packed_from_random::<PF>(0x08fac4ee76260e44);

    assert_eq!(
        (vec0 * vec1) * vec2,
        vec0 * (vec1 * vec2),
        "Error when testing associativity of mul."
    );
    assert_eq!(
        vec0 * vec1,
        vec1 * vec0,
        "Error when testing commutativity of mul."
    );
    assert_eq!(
        vec0,
        vec0 * PF::one(),
        "Error when testing multiplicative identity right."
    );
    assert_eq!(
        vec0,
        PF::one() * vec0,
        "Error when testing multiplicative identity left."
    );
    assert_eq!(
        vec0 * zeros,
        PF::zero(),
        "Error when testing right multiplication by 0."
    );
    assert_eq!(
        zeros * vec0,
        PF::zero(),
        "Error when testing left multiplication by 0."
    );
    assert_eq!(
        vec0 * PF::neg_one(),
        -(vec0),
        "Error when testing right multiplication by -1."
    );
    assert_eq!(
        PF::neg_one() * vec0,
        -(vec0),
        "Error when testing left multiplication by -1."
    );
    assert_eq!(
        vec0.double(),
        PF::two() * vec0,
        "Error when comparing x.double() to 2 * x."
    );
    assert_eq!(
        vec0.exp_const_u64::<3>(),
        vec0 * vec0 * vec0,
        "Error when comparing x.exp_const_u64::<3> to x*x*x."
    );
    assert_eq!(
        vec0.exp_const_u64::<5>(),
        vec0 * vec0 * vec0 * vec0 * vec0,
        "Error when comparing x.exp_const_u64::<5> to x*x*x*x*x."
    );
    assert_eq!(
        vec0.exp_const_u64::<7>(),
        vec0 * vec0 * vec0 * vec0 * vec0 * vec0 * vec0,
        "Error when comparing x.exp_const_u64::<7> to x*x*x*x*x*x*x."
    );
}

pub fn test_distributivity<PF>()
where
    PF: PackedField + Eq,
    Standard: Distribution<PF::Scalar>,
{
    let vec0 = packed_from_random::<PF>(0x278d9e202925a1d1);
    let vec1 = packed_from_random::<PF>(0xf04cbac0cbad419f);
    let vec2 = packed_from_random::<PF>(0x76976e2abdc5a056);

    assert_eq!(
        vec0 * (-vec1),
        -(vec0 * vec1),
        "Error when testing distributivity of mul and right neg."
    );
    assert_eq!(
        (-vec0) * vec1,
        -(vec0 * vec1),
        "Error when testing distributivity of mul and left neg."
    );

    assert_eq!(
        vec0 * (vec1 + vec2),
        vec0 * vec1 + vec0 * vec2,
        "Error when testing distributivity of add and left mul."
    );
    assert_eq!(
        (vec0 + vec1) * vec2,
        vec0 * vec2 + vec1 * vec2,
        "Error when testing distributivity of add and right mul."
    );
    assert_eq!(
        vec0 * (vec1 - vec2),
        vec0 * vec1 - vec0 * vec2,
        "Error when testing distributivity of sub and left mul."
    );
    assert_eq!(
        (vec0 - vec1) * vec2,
        vec0 * vec2 - vec1 * vec2,
        "Error when testing distributivity of sub and right mul."
    );
}

pub fn test_vs_scalar<PF>(special_vals: PF)
where
    PF: PackedField + Eq,
    Standard: Distribution<PF::Scalar>,
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
            "Error when testing add consistency of packed and scalar at location {}.",
            i
        );
        assert_eq!(
            arr_special_sum_left[i],
            special_vals[i] + arr0[i],
            "Error when testing consistency of left add for special values for packed and scalar at location {}.",
            i
        );
        assert_eq!(
            arr_special_sum_right[i],
            arr1[i] + special_vals[i],
            "Error when testing consistency of right add for special values for packed and scalar at location {}.",
            i
        );

        assert_eq!(
            arr_sub[i],
            arr0[i] - arr1[i],
            "Error when testing sub consistency of packed and scalar at location {}.",
            i
        );
        assert_eq!(arr_special_sub_left[i],
            special_vals[i] - arr0[i],
            "Error when testing consistency of left sub for special values for packed and scalar at location {}.",
            i
        );
        assert_eq!(arr_special_sub_right[i],
            arr1[i] - special_vals[i],
            "Error when testing consistency of right sub for special values for packed and scalar at location {}.",
            i
        );

        assert_eq!(
            arr_mul[i],
            arr0[i] * arr1[i],
            "Error when testing mul consistency of packed and scalar at location {}.",
            i
        );
        assert_eq!(arr_special_mul_left[i],
            special_vals[i] * arr0[i],
            "Error when testing consistency of left mul for special values for packed and scalar at location {}.",
            i
        );
        assert_eq!(arr_special_mul_right[i],
            arr1[i] * special_vals[i],
            "Error when testing consistency of right mul for special values for packed and scalar at location {}.",
            i
        );

        assert_eq!(
            arr_neg[i], -arr0[i],
            "Error when testing neg consistency of packed and scalar at location {}.",
            i
        );
        assert_eq!(arr_special_neg[i],
            -special_vals[i],
            "Error when testing consistency of neg for special values for packed and scalar at location {}.",
            i
        );
        assert_eq!(arr_exp_3[i],
            arr0[i].exp_const_u64::<3>(),
            "Error when testing exp_const_u64::<3> consistency of packed and scalar at location {}.",
            i
        );
        assert_eq!(arr_special_exp_3[i],
            special_vals[i].exp_const_u64::<3>(),
            "Error when testing consistency of exp_const_u64::<3> for special values for packed and scalar at location {}.",
            i
        );
        assert_eq!(arr_exp_5[i],
            arr0[i].exp_const_u64::<5>(),
            "Error when testing exp_const_u64::<5> consistency of packed and scalar at location {}.",
            i
        );
        assert_eq!(arr_special_exp_5[i],
            special_vals[i].exp_const_u64::<5>(),
            "Error when testing consistency of exp_const_u64::<5> for special values for packed and scalar at location {}.",
            i
        );
        assert_eq!(arr_exp_7[i],
            arr0[i].exp_const_u64::<7>(),
            "Error when testing exp_const_u64::<7> consistency of packed and scalar at location {}.",
            i
        );
        assert_eq!(arr_special_exp_7[i],
            special_vals[i].exp_const_u64::<7>(),
            "Error when testing consistency of exp_const_u64::<7> for special values for packed and scalar at location {}.",
            i
        );
    }
}

pub fn test_multiplicative_inverse<PF>()
where
    PF: PackedField + Eq,
    Standard: Distribution<PF::Scalar>,
{
    let vec: PF = packed_from_random(0xb0c7a5153103c5a8);
    let arr = vec.as_slice();
    let vec_inv = PF::from_fn(|i| arr[i].inverse());
    let res = vec * vec_inv;
    assert_eq!(
        res,
        PF::one(),
        "Error when testing multiplication by inverse."
    );
}

#[macro_export]
macro_rules! test_packed_field {
    ($packedfield:ty, $zeros:expr, $specials:expr) => {
        mod packed_field_tests {
            use p3_field::AbstractField;

            #[test]
            fn test_interleaves() {
                $crate::test_interleaves::<$packedfield>();
            }
            #[test]
            fn test_add_neg() {
                $crate::test_add_neg::<$packedfield>($zeros);
            }
            #[test]
            fn test_mul() {
                $crate::test_mul::<$packedfield>($zeros);
            }
            #[test]
            fn test_distributivity() {
                $crate::test_distributivity::<$packedfield>();
            }
            #[test]
            fn test_vs_scalar() {
                $crate::test_vs_scalar::<$packedfield>($specials);
            }
            #[test]
            fn test_multiplicative_inverse() {
                $crate::test_multiplicative_inverse::<$packedfield>();
            }
        }
    };
}
