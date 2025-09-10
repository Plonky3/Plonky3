use crate::KoalaBear;


/// If no packings are available, we use the generic binomial extension multiplication functions.
/// 
#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "x86_64", target_feature = "avx2",)
)))]
#[inline]
pub(crate) fn kb_quintic_mul_packed(
    a: &[KoalaBear; 5],
    b: &[KoalaBear; 5],
    res: &mut [KoalaBear; 5],
) {
    super::quintic_extension::kb_quintic_mul(a, b, res);
}


#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
/// Multiplication in a quintic binomial extension field.
#[inline]
pub(crate) fn kb_quintic_mul_packed(
    a: &[KoalaBear; 5],
    b: &[KoalaBear; 5],
    res: &mut [KoalaBear; 5],
) {
    use p3_monty_31::PackedMontyField31AVX2;
    // TODO: This could likely be optimised further with more effort.
    // in particular it would benefit from a custom AVX2 implementation.

    let zero = KoalaBear::ZERO;
    let b_0_minus_3 = b[0] - b[3];
    let b_1_minus_4 = b[1] - b[4];
    let b_4_minus_2 = b[4] - b[2];
    let b_3_minus_b_1_minus_4 = b[3] - b_1_minus_4;

    let lhs = [
        PackedMontyField31AVX2([a[0], a[0], a[0], a[0], a[0], a[4], a[4], a[4]]),
        PackedMontyField31AVX2([a[1], a[1], a[1], a[1], a[1], zero, zero, zero]),
        PackedMontyField31AVX2([a[2], a[2], a[2], a[2], a[2], zero, zero, zero]),
        PackedMontyField31AVX2([a[3], a[3], a[3], a[3], a[3], zero, zero, zero]),
    ];
    let rhs = [
        PackedMontyField31AVX2([
            b[0],
            b[1],
            b[2],
            b[3],
            b[4],
            b_1_minus_4,
            b[2],
            b_3_minus_b_1_minus_4,
        ]),
        PackedMontyField31AVX2([b[4], b[0], b_1_minus_4, b[2], b[3], zero, zero, zero]),
        PackedMontyField31AVX2([b[3], b[4], b_0_minus_3, b_1_minus_4, b[2], zero, zero, zero]),
        PackedMontyField31AVX2([
            b[2],
            b[3],
            b_4_minus_2,
            b_0_minus_3,
            b_1_minus_4,
            zero,
            zero,
            zero,
        ]),
    ];

    let dot_res =
        unsafe { PackedMontyField31AVX2::from_vector(p3_monty_31::dot_product_4(lhs, rhs)) };

    // We managed to compute 3 of the extra terms in the last 3 places of the dot product.
    // This leaves us with 2 terms remaining we need to compute manually.
    let extra1 = b_4_minus_2 * a[4];
    let extra2 = b_0_minus_3 * a[4];

    let extra_addition = PackedMontyField31AVX2([
        dot_res.0[5],
        dot_res.0[6],
        dot_res.0[7],
        extra1,
        extra2,
        zero,
        zero,
        zero,
    ]);
    let total = dot_res + extra_addition;

    res.copy_from_slice(&total.0[..5]);
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
/// Multiplication in a quintic binomial extension field.
#[inline]
pub(crate) fn kb_quintic_mul_packed<FP>(
    a: &[KoalaBear; 5],
    b: &[KoalaBear; 5],
    res: &mut [KoalaBear; 5],
) {
    use p3_field::PrimeCharacteristicRing;
    use p3_monty_31::PackedMontyField31AVX512;
    use p3_monty_31::dot_product_2;

    // TODO: It's plausible that this could be improved by folding the computation of packed_b into
    // the custom AVX512 implementation. Moreover, AVX512 is really a bit to large so we are wasting a lot
    // of space. A custom implementation which mixes AVX512 and AVX2 code might well be able to
    // improve one that is here.
    let zero = KoalaBear::ZERO;
    let b_0_minus_3 = b[0] - b[3];
    let b_1_minus_4 = b[1] - b[4];
    let b_4_minus_2 = b[4] - b[2];
    let b_3_minus_b_1_minus_4 = b[3] - b_1_minus_4;

    // Constant term = a0*b0 + w(a1*b4 + a2*b3 + a3*b2 + a4*b1)
    // Linear term = a0*b1 + a1*b0 + w(a2*b4 + a3*b3 + a4*b2)
    // Square term = a0*b2 + a1*b1 + a2*b0 + w(a3*b4 + a4*b3)
    // Cubic term = a0*b3 + a1*b2 + a2*b1 + a3*b0 + w*a4*b4
    // Quartic term = a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0

    // Each packed vector can do 8 multiplications at once. As we have
    // 25 multiplications to do we will need to use at least 3 packed vectors
    // but we might as well use 4 so we can make use of dot_product_2.
    // TODO: This can probably be improved by using a custom function.
    let lhs = [
        PackedMontyField31AVX512([
            a[0], a[2], a[0], a[2], a[0], a[2], a[0], a[2], a[2], a[2], a[4], a[4], a[4], a[4],
            a[4], zero,
        ]),
        PackedMontyField31AVX512([
            a[1], a[3], a[1], a[3], a[1], a[3], a[1], a[3], a[1], a[3], zero, zero, zero, zero,
            zero, zero,
        ]),
    ];
    let rhs = [
        PackedMontyField31AVX512([
            b[0],
            b[3],
            b[1],
            b[4],
            b[2],
            b_0_minus_3,
            b[3],
            b_1_minus_4,
            b[4],
            b[2],
            b_1_minus_4,
            b[2],
            b_3_minus_b_1_minus_4,
            b_4_minus_2,
            b_0_minus_3,
            zero,
        ]),
        PackedMontyField31AVX512([
            b[4],
            b[2],
            b[0],
            b[3],
            b_1_minus_4,
            b_4_minus_2,
            b[2],
            b_0_minus_3,
            b[3],
            b_1_minus_4,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
        ]),
    ];

    let dot = unsafe { PackedMontyField31AVX512::from_vector(dot_product_2(lhs, rhs)).0 };

    let sumand1 =
        PackedMontyField31AVX512::from_monty_array([dot[0], dot[2], dot[4], dot[6], dot[8]]);
    let sumand2 =
        PackedMontyField31AVX512::from_monty_array([dot[1], dot[3], dot[5], dot[7], dot[9]]);
    let sumand3 =
        PackedMontyField31AVX512::from_monty_array([dot[10], dot[11], dot[12], dot[13], dot[14]]);
    let sum = sumand1 + sumand2 + sumand3;

    res.copy_from_slice(&sum.0[..5]);
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
/// Multiplication in a quintic binomial extension field.
#[inline]
pub(crate) fn kb_quintic_mul_packed(
    a: &[KoalaBear; 5],
    b: &[KoalaBear; 5],
    res: &mut [KoalaBear; 5],
) {
    // TODO: This could be optimised further with a custom NEON implementation.

    use p3_field::PrimeCharacteristicRing;
    use p3_monty_31::PackedMontyField31Neon;

    let b_0_minus_3 = b[0] - b[3];
    let b_1_minus_4 = b[1] - b[4];
    let b_4_minus_2 = b[4] - b[2];
    let b_3_minus_b_1_minus_4 = b[3] - b_1_minus_4;

    // Constant term = a0*b0 + w(a1*b4 + a2*b3 + a3*b2 + a4*b1)
    // Linear term = a0*b1 + a1*b0 + w(a2*b4 + a3*b3 + a4*b2)
    // Square term = a0*b2 + a1*b1 + a2*b0 + w(a3*b4 + a4*b3)
    // Cubic term = a0*b3 + a1*b2 + a2*b1 + a3*b0 + w*a4*b4
    // Quartic term = a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0
    let lhs: [PackedMontyField31Neon<crate::KoalaBearParameters>; 5] = [
        a[0].into(),
        a[1].into(),
        a[2].into(),
        a[3].into(),
        a[4].into(),
    ];
    let rhs = [
        PackedMontyField31Neon([b[0], b[1], b[2], b[4]]),
        PackedMontyField31Neon([b[4], b[0], b_1_minus_4, b[3]]),
        PackedMontyField31Neon([b[3], b[4], b_0_minus_3, b[2]]),
        PackedMontyField31Neon([b[2], b[3], b_4_minus_2, b_1_minus_4]),
        PackedMontyField31Neon([b_1_minus_4, b[2], b_3_minus_b_1_minus_4, b_0_minus_3]),
    ];

    let dot = PackedMontyField31Neon::dot_product(&lhs, &rhs).0;

    res[..4].copy_from_slice(&dot);
    res[4] = KoalaBear::dot_product::<5>(
        &[a[0], a[1], a[2], a[3], a[4]],
        &[b[4], b[3], b[2], b_1_minus_4, b_0_minus_3],
    );
}
