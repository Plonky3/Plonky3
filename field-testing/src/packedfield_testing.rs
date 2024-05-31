use p3_field::{Field, PackedField, PackedValue};
use core::array;


pub trait PackedTestingHelpers<const WIDTH: usize, F, PF>
where
    F: Field,
    PF: PackedField<Scalar = F> + PackedValue
{
    fn packed_from_valid_reps(vals: [u32; WIDTH]) -> PF;
    fn array_from_random(seed: u64) -> [F; WIDTH];
    fn packed_from_random(seed: u64) -> PF;

    const ZEROS: [F; WIDTH]; // Some fields might have multiple representations for 0.
    const SPECIAL_VALS: [F; WIDTH]; // Let the user specify some particular values which could lead to errors.
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

fn test_interleave<const WIDTH: usize, F, PF, PTH>(i: usize)
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
    let (res0, res1) = vec0.interleave(vec1, i);

    let (out1, out2) = interleave(arr1, arr2, i);

    let expected0 = PTH::packed_from_valid_reps(out1);
    let expected1 = PTH::packed_from_valid_reps(out2);
    
    assert_eq!(res0, expected0, "Error in left output when testing interleave {}.", i);
    assert_eq!(res1, expected1, "Error in right output when testing interleave {}.", i);
}

pub fn test_interleaves<const WIDTH: usize, F, PF, PTH>()
where
    F: Field,
    PF: PackedField<Scalar = F> + PackedValue + Eq,
    PTH: PackedTestingHelpers<WIDTH, F, PF>,
{
    let mut i = 1;
    while i <= WIDTH {
        test_interleave::<WIDTH, F, PF, PTH>(i);
        i *= 2;
    }
}

#[allow(clippy::eq_op)]
pub fn test_add_neg<const WIDTH: usize, F, PF, PTH>()
where
    F: Field,
    PF: PackedField<Scalar = F> + PackedValue + Eq,
    PTH: PackedTestingHelpers<WIDTH, F, PF>,
{
    let vec0 = PTH::packed_from_random(0x8b078c2b693c893f);
    let vec1 = PTH::packed_from_random(0x4ff5dec04791e481);
    let vec2 = PTH::packed_from_random(0x5806c495e9451f8e);
    let zeros = *(PF::from_slice(&PTH::ZEROS));

    assert_eq!((vec0 + vec1) + vec2, vec0 + (vec1 + vec2), "Error when testing associativity of add.");
    assert_eq!(vec0 + vec1, vec1 + vec0, "Error when testing commutativity of add.");
    assert_eq!(vec0, vec0 + zeros, "Error when testing additive identity right.");
    assert_eq!(vec0, zeros + vec0, "Error when testing additive identity left.");
    assert_eq!(vec0 + (-vec0), PF::zero(), "Error when testing additive inverse.");
    assert_eq!(vec0 - vec0, PF::zero(), "Error when testing subtracting of self.");
    assert_eq!(vec0 - vec1, -(vec1 - vec0), "Error when testing anticommutativity of sub.");
    assert_eq!(vec0, vec0 - zeros, "Error when testing subtracting zero.");
    assert_eq!(-vec0, zeros - vec0, "Error when testing subtracting from zero");
    assert_eq!(vec0, -(-vec0), "Error when testing double negation");
    assert_eq!(vec0 - vec1, vec0 + (-vec1), "Error when testing addition of negation");
    assert_eq!(PF::one() + PF::one(), PF::two(), "Error 1 + 1 =/= 2");
    assert_eq!(PF::neg_one() + PF::two(), PF::one(), "Error -1 + 2 =/= 1");
    assert_eq!(vec0.double(), vec0 + vec0, "Error when comparing x.double() to x + x");
}


#[allow(clippy::eq_op)]
pub fn test_mul<const WIDTH: usize, F, PF, PTH>()
where
    F: Field,
    PF: PackedField<Scalar = F> + PackedValue + Eq,
    PTH: PackedTestingHelpers<WIDTH, F, PF>,
{
    let vec0 = PTH::packed_from_random(0x0b1ee4d7c979d50c);
    let vec1 = PTH::packed_from_random(0x39faa0844a36e45a);
    let vec2 = PTH::packed_from_random(0x08fac4ee76260e44);
    let zeros = *(PF::from_slice(&PTH::ZEROS));

    assert_eq!((vec0 * vec1) * vec2, vec0 * (vec1 * vec2), "Error when testing associativity of mul.");
    assert_eq!(vec0 * vec1, vec1 * vec0, "Error when testing commutativity of mul.");
    assert_eq!(vec0, vec0 * PF::one(), "Error when testing multiplicative identity right.");
    assert_eq!(vec0, PF::one() *  vec0, "Error when testing multiplicative identity left.");
    assert_eq!(vec0 * zeros, PF::zero(), "Error when testing right multiplication by 0.");
    assert_eq!(zeros * vec0, PF::zero(), "Error when testing left multiplication by 0.");
    assert_eq!(vec0 * PF::neg_one(), -(vec0), "Error when testing right multiplication by -1.");
    assert_eq!(PF::neg_one() * vec0, -(vec0), "Error when testing left multiplication by -1.");
    assert_eq!(vec0.double(), PF::two() * vec0, "Error when comparing x.double() to 2 * x");
}

#[allow(clippy::eq_op)]
pub fn test_distributivity<const WIDTH: usize, F, PF, PTH>()
where
    F: Field,
    PF: PackedField<Scalar = F> + PackedValue + Eq,
    PTH: PackedTestingHelpers<WIDTH, F, PF>,
{
    let vec0 = PTH::packed_from_random(0x278d9e202925a1d1);
    let vec1 = PTH::packed_from_random(0xf04cbac0cbad419f);
    let vec2 = PTH::packed_from_random(0x76976e2abdc5a056);

    assert_eq!(vec0 * (-vec1), -(vec0 * vec1), "Error when testing distributivity of mul and right neg.");
    assert_eq!((-vec0) * vec1, -(vec0 * vec1), "Error when testing distributivity of mul and left neg.");

    assert_eq!(vec0 * (vec1 + vec2), vec0 * vec1 + vec0 * vec2, "Error when testing distributivity of add and left mul.");
    assert_eq!((vec0 + vec1) * vec2, vec0 * vec2 + vec1 * vec2, "Error when testing distributivity of add and right mul.");
    assert_eq!(vec0 * (vec1 - vec2), vec0 * vec1 - vec0 * vec2, "Error when testing distributivity of sub and left mul.");
    assert_eq!((vec0 - vec1) * vec2, vec0 * vec2 - vec1 * vec2, "Error when testing distributivity of sub and right mul.");
}

#[allow(clippy::eq_op)]
pub fn test_vs_scalar<const WIDTH: usize, F, PF, PTH>()
where
    F: Field,
    PF: PackedField<Scalar = F> + PackedValue + Eq,
    PTH: PackedTestingHelpers<WIDTH, F, PF>,
{
    let arr0 = PTH::array_from_random(0x278d9e202925a1d1);
    let arr1 = PTH::array_from_random(0xf04cbac0cbad419f);

    let vec0 = *(PF::from_slice(&arr0));
    let vec1 = *(PF::from_slice(&arr1));
    let vec_special = *(PF::from_slice(&PTH::SPECIAL_VALS));

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

    for i in 0..WIDTH {
        assert_eq!(arr_sum[i], arr0[i] + arr1[i]);
        assert_eq!(arr_special_sum_left[i], PTH::SPECIAL_VALS[i] + arr0[i]);
        assert_eq!(arr_special_sum_right[i], arr1[i] + PTH::SPECIAL_VALS[i]);

        assert_eq!(arr_sub[i], arr0[i] - arr1[i]);
        assert_eq!(arr_special_sub_left[i], PTH::SPECIAL_VALS[i] - arr0[i]);
        assert_eq!(arr_special_sub_right[i], arr1[i] - PTH::SPECIAL_VALS[i]);

        assert_eq!(arr_mul[i], arr0[i] * arr1[i]);
        assert_eq!(arr_special_mul_left[i], PTH::SPECIAL_VALS[i] * arr0[i]);
        assert_eq!(arr_special_mul_right[i], arr1[i] * PTH::SPECIAL_VALS[i]);

        assert_eq!(arr_neg[i], -arr0[i]);
        assert_eq!(arr_special_neg[i], -PTH::SPECIAL_VALS[i]);
    }
}

pub fn test_multiplicative_inverse<const WIDTH: usize, F, PF, PTH>()
    where
        F: Field,
        PF: PackedField<Scalar = F> + PackedValue + Eq,
        PTH: PackedTestingHelpers<WIDTH, F, PF>,
    {

    let arr = PTH::array_from_random(0xb0c7a5153103c5a8);
    let arr_inv = arr.map(|x| x.inverse());

    let vec = *(PF::from_slice(&arr));
    let vec_inv = *(PF::from_slice(&arr_inv));

    let res = vec * vec_inv;
    assert_eq!(res, PF::one());
}

#[macro_export]
macro_rules! test_packed_field {
    ($width:expr, $field:ty, $packedfield:ty, $helper:ty) => {
        mod packed_field_tests {
            #[test]
            fn test_interleaves() {
                $crate::test_interleaves::<$width, $field, $packedfield, $helper>();
            }
            #[test]
            fn test_add_neg() {
                $crate::test_add_neg::<$width, $field, $packedfield, $helper>();
            }
            #[test]
            fn test_mul() {
                $crate::test_mul::<$width, $field, $packedfield, $helper>();
            }
            #[test]
            fn test_distributivity() {
                $crate::test_distributivity::<$width, $field, $packedfield, $helper>();
            }
            #[test]
            fn test_vs_scalar() {
                $crate::test_vs_scalar::<$width, $field, $packedfield, $helper>();
            }
            #[test]
            fn test_multiplicative_inverse() {
                $crate::test_multiplicative_inverse::<$width, $field, $packedfield, $helper>();
            }
        }
    };
}