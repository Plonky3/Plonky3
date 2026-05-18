use alloc::vec;
use alloc::vec::Vec;
use core::ops::Div;

use p3_field::extension::{
    BinomialExtensionField, BinomiallyExtendable, PackedBinomialExtensionField,
};
use p3_field::integers::QuotientMap;
use p3_field::{
    Algebra, BasedVectorSpace, ExtensionField, Field, PackedField, PackedFieldExtension,
    PackedFieldPow2, PackedValue, PrimeCharacteristicRing, PrimeField32,
};
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

pub fn test_batched_linear_combination<PF: PackedField + Eq>()
where
    StandardUniform: Distribution<PF> + Distribution<PF::Scalar>,
{
    let mut rng = SmallRng::seed_from_u64(99);
    let values: [PF; 64] = rng.random();
    let coeffs: [PF::Scalar; 64] = rng.random();

    for len in [0, 1, 3, 7, 8, 9, 15, 16, 17, 32, 64] {
        let expected: PF = values[..len]
            .iter()
            .zip(&coeffs[..len])
            .fold(PF::ZERO, |acc, (&v, &c)| acc + v * c);
        let got = PF::batched_linear_combination(&values[..len], &coeffs[..len]);
        assert_eq!(expected, got, "failed for len={len}");
    }
}

pub fn test_batched_linear_combination_ext<BF, EF, PE>()
where
    BF: Field,
    EF: ExtensionField<BF, ExtensionPacking = PE>,
    PE: PackedFieldExtension<BF, EF> + Algebra<EF> + Copy + Eq,
    StandardUniform: Distribution<PE> + Distribution<EF>,
{
    let mut rng = SmallRng::seed_from_u64(99);
    let values: [PE; 64] = rng.random();
    let coeffs: [EF; 64] = rng.random();

    for len in [0, 1, 3, 7, 8, 9, 15, 16, 17, 32, 64] {
        let expected: PE = values[..len]
            .iter()
            .zip(&coeffs[..len])
            .fold(PE::ZERO, |acc, (&v, &c)| acc + v * c);
        let got = PE::batched_linear_combination(&values[..len], &coeffs[..len]);
        assert_eq!(expected, got, "failed for len={len}");
    }
}

/// Verify packed binomial extension division against scalar reference results.
///
/// Tests four operations:
/// - Packed / packed (lane-wise division via Montgomery's trick)
/// - Packed /= packed (in-place variant)
/// - Packed / scalar (broadcast scalar inverse, then multiply)
/// - Packed /= scalar (in-place variant)
///
/// Each is checked by extracting individual SIMD lanes and comparing
/// against the equivalent scalar division.
pub fn test_packed_binomial_extension_division<F, const D: usize>()
where
    F: BinomiallyExtendable<D>,
    StandardUniform: Distribution<BinomialExtensionField<F, D>>,
{
    // Deterministic RNG for reproducible test failures.
    let mut rng = SmallRng::seed_from_u64(0x04dd6059d9d02758);
    // Number of SIMD lanes in the packed representation.
    let width = F::Packing::WIDTH;

    // Generate one random extension field element per lane for the numerator.
    let numerators: Vec<BinomialExtensionField<F, D>> = (0..width).map(|_| rng.random()).collect();
    // Rejection-sample non-zero elements for the denominator (zero is not invertible).
    let mut sample_nonzero = || loop {
        let x: BinomialExtensionField<F, D> = rng.random();
        if !x.is_zero() {
            break x;
        }
    };
    let denominators: Vec<BinomialExtensionField<F, D>> =
        (0..width).map(|_| sample_nonzero()).collect();

    // Pack the scalar vectors into the SoA packed representation.
    let packed_num = PackedBinomialExtensionField::<F, F::Packing, D>::from_ext_slice(&numerators);
    let packed_den =
        PackedBinomialExtensionField::<F, F::Packing, D>::from_ext_slice(&denominators);

    // Helper closure: extract a single extension field element from a given SIMD lane.
    // Reads the lane-th scalar from each of the D packed coefficient arrays,
    // then reassembles them into a scalar extension field element.
    let extract_lane = |x: &PackedBinomialExtensionField<F, F::Packing, D>, lane: usize| {
        BinomialExtensionField::<F, D>::new(core::array::from_fn(|i| {
            <PackedBinomialExtensionField<F, F::Packing, D> as BasedVectorSpace<F::Packing>>::as_basis_coefficients_slice(x)[i]
                .as_slice()[lane]
        }))
    };

    // Test 1: packed / packed division.
    // Each lane of the result must equal the scalar quotient of the corresponding lane inputs.
    let quot = packed_num / packed_den;
    for lane in 0..width {
        assert_eq!(
            extract_lane(&quot, lane),
            numerators[lane] / denominators[lane],
            "packed/packed division mismatch at lane {lane}"
        );
    }

    // Test 2: packed /= packed (in-place assignment).
    // Must produce the same result as the non-assignment variant above.
    let mut quot_assign = packed_num;
    quot_assign /= packed_den;
    for lane in 0..width {
        assert_eq!(
            extract_lane(&quot_assign, lane),
            extract_lane(&quot, lane),
            "packed/packed div_assign mismatch at lane {lane}"
        );
    }

    // Test 3: packed / scalar (broadcast division).
    // Divides every lane by the same scalar extension field element.
    let scalar_den = sample_nonzero();
    let quot_scalar = packed_num / scalar_den;
    for (lane, numerator) in numerators.iter().enumerate() {
        assert_eq!(
            extract_lane(&quot_scalar, lane),
            *numerator / scalar_den,
            "packed/scalar division mismatch at lane {lane}"
        );
    }

    // Test 4: packed /= scalar (in-place broadcast division).
    // Must produce the same result as the non-assignment variant above.
    let mut quot_scalar_assign = packed_num;
    quot_scalar_assign /= scalar_den;
    for lane in 0..width {
        assert_eq!(
            extract_lane(&quot_scalar_assign, lane),
            extract_lane(&quot_scalar, lane),
            "packed/scalar div_assign mismatch at lane {lane}"
        );
    }
}

/// Verify that packed divide agrees with scalar divide on every SIMD lane.
///
/// # Invariant
///
/// For every lane `i`:
///
/// ```text
/// (a / b).lane(i) == a.lane(i) / b.lane(i)
/// ```
pub fn test_packed_extension_div_consistency<F, EF, PEF>()
where
    F: Field,
    EF: ExtensionField<F, ExtensionPacking = PEF>,
    PEF: PackedFieldExtension<F, EF> + Div<Output = PEF> + Copy,
    StandardUniform: Distribution<EF>,
{
    // SIMD lane count.
    // - Goldilocks NEON → 2.
    // - KoalaBear NEON → 4.
    // - Scalar → 1.
    let width = F::Packing::WIDTH;

    // Fixed seed → reproducible bytes on any failure.
    let mut rng = SmallRng::seed_from_u64(0x_d1ef_d1ef_d1ef_d1ef);

    // Numerator lane fixture: any value, zeros are fine.
    //
    //     lane:   0    1    …    W-1
    //     nums:  [n0,  n1,  …,   n_{W-1}]
    let nums: Vec<EF> = (0..width).map(|_| rng.random()).collect();

    // Denominator lane fixture: reject-sample so every lane is invertible.
    //
    //     lane:   0       1       …    W-1
    //     dens:  [d0 ≠ 0, d1 ≠ 0, …,   d_{W-1} ≠ 0]
    let dens: Vec<EF> = (0..width)
        .map(|_| {
            loop {
                let x: EF = rng.random();
                if !x.is_zero() {
                    break x;
                }
            }
        })
        .collect();

    // Pack the per-lane scalars into one SIMD value each.
    //
    //     lanes [n0, n1, …, n_{W-1}]  →  one packed extension value
    //     lanes [d0, d1, …, d_{W-1}]  →  one packed extension value
    let pef_n: PEF = PEF::from_ext_slice(&nums);
    let pef_d: PEF = PEF::from_ext_slice(&dens);

    // Run the packed divide. Each lane must independently compute n_i / d_i.
    let pef_q = pef_n / pef_d;

    // Per-lane invariant:
    //
    //     packed quotient at lane i  ==  nums[i] / dens[i]
    for lane in 0..width {
        let expected = nums[lane] / dens[lane];
        let got = pef_q.extract(lane);
        assert_eq!(
            got, expected,
            "lane {lane}: packed Div disagrees with scalar Div (W = {width})"
        );
    }
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

    let vec_div = vec0 / vec1;
    let arr_div = vec_div.as_slice();

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
            arr_div[i],
            arr0[i] / arr1[i],
            "Error when testing div consistency of packed and scalar at location {i}.",
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

pub fn test_pack_columns<PF>()
where
    PF: PackedField + Eq,
    StandardUniform: Distribution<PF::Scalar>,
{
    let mut rng = SmallRng::seed_from_u64(0xc0ffee42);

    // Test round-trip: pack_columns then unpack_into
    let rows: Vec<[PF::Scalar; 4]> = (0..PF::WIDTH)
        .map(|_| [rng.random(), rng.random(), rng.random(), rng.random()])
        .collect();
    let packed = PF::pack_columns::<4>(&rows);
    let mut unpacked = vec![[PF::Scalar::default(); 4]; PF::WIDTH];
    PF::unpack_into(&packed, &mut unpacked);
    assert_eq!(
        rows, unpacked,
        "pack_columns -> unpack_into round-trip failed"
    );

    // Test round-trip: unpack_into then pack_columns
    let original: [PF; 4] = [
        packed_from_random(0x1111),
        packed_from_random(0x2222),
        packed_from_random(0x3333),
        packed_from_random(0x4444),
    ];
    let mut rows2 = vec![[PF::Scalar::default(); 4]; PF::WIDTH];
    PF::unpack_into(&original, &mut rows2);
    let repacked = PF::pack_columns::<4>(&rows2);
    assert_eq!(
        original, repacked,
        "unpack_into -> pack_columns round-trip failed"
    );
}

pub fn test_pack_columns_fn<PF>()
where
    PF: PackedField + Eq,
    StandardUniform: Distribution<PF::Scalar>,
{
    let mut rng = SmallRng::seed_from_u64(0xbaadf00d);
    let rows: Vec<[PF::Scalar; 4]> = (0..PF::WIDTH)
        .map(|_| [rng.random(), rng.random(), rng.random(), rng.random()])
        .collect();

    let from_slice = PF::pack_columns::<4>(&rows);
    let from_fn = PF::pack_columns_fn(|lane| rows[lane]);
    assert_eq!(
        from_slice, from_fn,
        "pack_columns_fn should match pack_columns"
    );
}

pub fn test_unpack_iter<PF>()
where
    PF: PackedField + Eq,
    StandardUniform: Distribution<PF::Scalar>,
{
    let packed: [PF; 4] = [
        packed_from_random(0xaaaa),
        packed_from_random(0xbbbb),
        packed_from_random(0xcccc),
        packed_from_random(0xdddd),
    ];

    // Compare with unpack_into
    let mut rows_via_into = vec![[PF::Scalar::default(); 4]; PF::WIDTH];
    PF::unpack_into(&packed, &mut rows_via_into);
    let rows_via_iter: Vec<[PF::Scalar; 4]> = PF::unpack_iter(packed).collect();
    assert_eq!(
        rows_via_into, rows_via_iter,
        "unpack_iter should match unpack_into"
    );

    // Round-trip with pack_columns
    let repacked = PF::pack_columns::<4>(&rows_via_iter);
    assert_eq!(
        packed, repacked,
        "unpack_iter -> pack_columns round-trip failed"
    );
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

/// Eight canonical edge values that probe carry propagation in dot products.
///
/// - `0`: additive identity.
/// - `1`, `2`: small coefficients.
/// - `p / 4`, `p / 2`, `p / 2 + 1`: mid-range, straddle `2^32` after squaring.
/// - `p - 2`, `p - 1`: maximum canonical band, products near `(p - 1)^2`.
const fn boundary_u32_values<F: PrimeField32>() -> [u32; 8] {
    // `ORDER_U32 > 4`, so every entry is canonical.
    let p = F::ORDER_U32;
    [0, 1, 2, p / 4, p / 2, p / 2 + 1, p - 2, p - 1]
}

/// Map a length-`N` array of canonical `u32`s through the field's quotient map.
fn u32_array_to_field<F, const N: usize>(values: [u32; N]) -> [F; N]
where
    F: PrimeField32 + QuotientMap<u32>,
{
    values.map(F::from_int)
}

/// Compare a broadcast packed dot product against the scalar reference.
///
/// Each input is replicated across every SIMD lane, so all lanes must agree
/// with the scalar result.
///
/// # Panics
///
/// On lane mismatch. The message echoes the raw inputs for replay.
pub fn assert_packed_broadcast_dot_product_matches_scalar<PF, const N: usize>(
    lhs_raw: [u32; N],
    rhs_raw: [u32; N],
) where
    PF: PackedField + Eq,
    PF::Scalar: PrimeField32 + QuotientMap<u32>,
{
    // Lift raw u32 inputs into the field.
    let lhs = u32_array_to_field::<PF::Scalar, N>(lhs_raw);
    let rhs = u32_array_to_field::<PF::Scalar, N>(rhs_raw);

    // Scalar reference.
    let scalar_ref = PF::Scalar::dot_product::<N>(&lhs, &rhs);

    // Packed path with each scalar broadcast over all lanes.
    let packed_lhs: [PF; N] = lhs.map(PF::broadcast);
    let packed_rhs: [PF; N] = rhs.map(PF::broadcast);
    let packed = PF::dot_product::<N>(&packed_lhs, &packed_rhs);

    // Every lane must equal the scalar reference.
    for (lane, got) in packed.as_slice().iter().enumerate() {
        assert_eq!(
            *got, scalar_ref,
            "lane {lane}: packed dot_product::<{N}> mismatch — lhs={lhs_raw:?} rhs={rhs_raw:?}",
        );
    }
}

/// Run `dot_product::<N>` against every (left, right) pair from the edge table.
///
/// Eight edges → 64 invocations per `N`. Deterministic.
pub fn test_packed_dot_product_broadcast_boundary_sweep<PF, const N: usize>()
where
    PF: PackedField + Eq,
    PF::Scalar: PrimeField32 + QuotientMap<u32>,
{
    let edges = boundary_u32_values::<PF::Scalar>();

    for &v in &edges {
        for &w in &edges {
            // Broadcast (v, w) into both length-N input arrays.
            assert_packed_broadcast_dot_product_matches_scalar::<PF, N>([v; N], [w; N]);
        }
    }
}

/// Per-lane edge-biased random stress for `dot_product::<N>`.
///
/// Each SIMD lane gets independently sampled inputs to expose lane-cross bugs
/// that broadcast tests miss.
pub fn test_packed_dot_product_lanes_random<PF, const N: usize>()
where
    PF: PackedField + Eq,
    PF::Scalar: PrimeField32 + QuotientMap<u32>,
{
    // 1024 iterations × WIDTH lanes × N elements stays under a second per N.
    const ITERS: usize = 1024;

    let p = PF::Scalar::ORDER_U32;
    let edges = boundary_u32_values::<PF::Scalar>();

    // Fixed seed so failures replay verbatim.
    let mut rng = SmallRng::seed_from_u64(0xD07B_0571_0DDE_DEAD);

    for _ in 0..ITERS {
        // One length-N array per lane, sampled independently per side.
        let lhs_per_lane: Vec<[PF::Scalar; N]> = (0..PF::WIDTH)
            .map(|_| {
                u32_array_to_field::<PF::Scalar, N>(core::array::from_fn(|_| {
                    sample_edge_or_uniform(&mut rng, &edges, p)
                }))
            })
            .collect();
        let rhs_per_lane: Vec<[PF::Scalar; N]> = (0..PF::WIDTH)
            .map(|_| {
                u32_array_to_field::<PF::Scalar, N>(core::array::from_fn(|_| {
                    sample_edge_or_uniform(&mut rng, &edges, p)
                }))
            })
            .collect();

        // Transpose lane-major rows to N packed columns.
        let packed_lhs = PF::pack_columns_fn::<N>(|lane| lhs_per_lane[lane]);
        let packed_rhs = PF::pack_columns_fn::<N>(|lane| rhs_per_lane[lane]);

        let result = PF::dot_product::<N>(&packed_lhs, &packed_rhs);

        // Per-lane scalar reference.
        for lane in 0..PF::WIDTH {
            let expected = PF::Scalar::dot_product::<N>(&lhs_per_lane[lane], &rhs_per_lane[lane]);
            assert_eq!(
                result.as_slice()[lane],
                expected,
                "lane {lane}: packed dot_product::<{N}> mismatch (random seed)",
            );
        }
    }
}

/// 50/50 draw between the eight-element edge table and uniform `[0, p)`.
fn sample_edge_or_uniform(rng: &mut SmallRng, edges: &[u32; 8], prime: u32) -> u32 {
    if rng.random::<bool>() {
        edges[(rng.random::<u32>() as usize) % edges.len()]
    } else {
        rng.random::<u32>() % prime
    }
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
        let quot = a / b;
        let neg_a = -a;

        let sum = sum.as_slice();
        let diff = diff.as_slice();
        let prod = prod.as_slice();
        let quot = quot.as_slice();
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
            prop_assert_eq!(quot[i], a[i] / b[i],
                "div mismatch at lane {}", i);
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
            fn test_packed_mixed_dot_product() {
                $crate::test_packed_mixed_dot_product::<$packedfield>();
            }
            #[test]
            fn test_batched_linear_combination() {
                $crate::test_batched_linear_combination::<$packedfield>();
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
            fn test_pack_columns() {
                $crate::test_pack_columns::<$packedfield>();
            }
            #[test]
            fn test_pack_columns_fn() {
                $crate::test_pack_columns_fn::<$packedfield>();
            }
            #[test]
            fn test_unpack_iter() {
                $crate::test_unpack_iter::<$packedfield>();
            }
            #[test]
            fn test_packed_vs_scalar_proptest() {
                $crate::test_packed_vs_scalar_proptest::<$packedfield>();
            }
        }
    };
}

/// Run the dot-product carry / boundary stress suite on a packed `PrimeField32`.
///
/// Two test functions per `N`:
/// - boundary-pair sweep over an 8-value edge table,
/// - per-lane edge-biased random.
///
/// `N` covers every SIMD dispatch arm:
/// - tiny (`1`),
/// - specialized (`2`, `4`, `5`, `8`),
/// - chunk-of-4 fallbacks (`3`, `6`, `7`),
/// - longer loops (`9`, `16`).
#[macro_export]
macro_rules! test_packed_field_dot_product_boundary {
    ($packedfield:ty) => {
        mod packed_dot_product_boundary_tests {
            // Exhaustive sweep of (v, w) edge pairs.

            #[test]
            fn boundary_sweep_n1() {
                $crate::test_packed_dot_product_broadcast_boundary_sweep::<$packedfield, 1>();
            }
            #[test]
            fn boundary_sweep_n2() {
                $crate::test_packed_dot_product_broadcast_boundary_sweep::<$packedfield, 2>();
            }
            #[test]
            fn boundary_sweep_n3() {
                $crate::test_packed_dot_product_broadcast_boundary_sweep::<$packedfield, 3>();
            }
            #[test]
            fn boundary_sweep_n4() {
                $crate::test_packed_dot_product_broadcast_boundary_sweep::<$packedfield, 4>();
            }
            #[test]
            fn boundary_sweep_n5() {
                $crate::test_packed_dot_product_broadcast_boundary_sweep::<$packedfield, 5>();
            }
            #[test]
            fn boundary_sweep_n6() {
                $crate::test_packed_dot_product_broadcast_boundary_sweep::<$packedfield, 6>();
            }
            #[test]
            fn boundary_sweep_n7() {
                $crate::test_packed_dot_product_broadcast_boundary_sweep::<$packedfield, 7>();
            }
            #[test]
            fn boundary_sweep_n8() {
                $crate::test_packed_dot_product_broadcast_boundary_sweep::<$packedfield, 8>();
            }
            #[test]
            fn boundary_sweep_n9() {
                $crate::test_packed_dot_product_broadcast_boundary_sweep::<$packedfield, 9>();
            }
            #[test]
            fn boundary_sweep_n16() {
                $crate::test_packed_dot_product_broadcast_boundary_sweep::<$packedfield, 16>();
            }

            // Independent per-lane inputs catch shuffle / lane-cross bugs.

            #[test]
            fn lanes_random_n2() {
                $crate::test_packed_dot_product_lanes_random::<$packedfield, 2>();
            }
            #[test]
            fn lanes_random_n3() {
                $crate::test_packed_dot_product_lanes_random::<$packedfield, 3>();
            }
            #[test]
            fn lanes_random_n4() {
                $crate::test_packed_dot_product_lanes_random::<$packedfield, 4>();
            }
            #[test]
            fn lanes_random_n5() {
                $crate::test_packed_dot_product_lanes_random::<$packedfield, 5>();
            }
            #[test]
            fn lanes_random_n6() {
                $crate::test_packed_dot_product_lanes_random::<$packedfield, 6>();
            }
            #[test]
            fn lanes_random_n7() {
                $crate::test_packed_dot_product_lanes_random::<$packedfield, 7>();
            }
            #[test]
            fn lanes_random_n8() {
                $crate::test_packed_dot_product_lanes_random::<$packedfield, 8>();
            }
            #[test]
            fn lanes_random_n9() {
                $crate::test_packed_dot_product_lanes_random::<$packedfield, 9>();
            }
            #[test]
            fn lanes_random_n16() {
                $crate::test_packed_dot_product_lanes_random::<$packedfield, 16>();
            }
        }
    };
}

#[macro_export]
macro_rules! test_packed_extension_field {
    ($basefield:ty, $extfield:ty, $packedextfield:ty, $zeros:expr, $ones:expr) => {
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
            #[test]
            fn test_batched_linear_combination_ext() {
                $crate::test_batched_linear_combination_ext::<
                    $basefield,
                    $extfield,
                    $packedextfield,
                >();
            }
            #[test]
            fn test_packed_extension_div_consistency() {
                $crate::test_packed_extension_div_consistency::<
                    $basefield,
                    $extfield,
                    $packedextfield,
                >();
            }
        }
    };
}

#[macro_export]
macro_rules! test_packed_binomial_extension_division {
    ($basefield:ty, $degree:expr) => {
        #[test]
        fn test_packed_binomial_extension_division() {
            $crate::test_packed_binomial_extension_division::<$basefield, $degree>();
        }
    };
}
