use alloc::vec::Vec;

use p3_field::extension::HasFrobenius;
use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue};
use proptest::prelude::*;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::exp_biguint;

/// Ensure that the methods `is_in_basefield` and `as_base` work as expected.
pub fn test_to_from_extension_field<F, EF>()
where
    F: Field,
    EF: ExtensionField<F>,
    StandardUniform: Distribution<F> + Distribution<EF>,
{
    let mut rng = SmallRng::seed_from_u64(1);

    let base_elem: F = rng.random();
    let base_elem_in_ext: EF = base_elem.into();
    assert!(base_elem_in_ext.is_in_basefield());
    assert_eq!(base_elem_in_ext.as_base(), Some(base_elem));

    let extension_elem: EF = rng.random();
    let ext_degree = EF::DIMENSION;

    if ext_degree == 1 {
        assert!(
            extension_elem.is_in_basefield(),
            "The element {extension_elem} does not lie in the base field, but it should.",
        );
    } else {
        // In principle it's possible that a randomly chosen element does lie in the base field.
        // But this is very unlikely. If this comes up regularly, we can change the test.
        assert!(
            !extension_elem.is_in_basefield(),
            "The randomly chosen element {extension_elem} lies in the base field, but it (likely) should not.",
        );
        assert!(extension_elem.as_base().is_none());
    }
}

/// Test that products and sums of galois conjugates lie in the base field.
///
/// This test can be skipped for extension fields of degree 1 as it is trivial.
pub fn test_galois_extension<F, EF>()
where
    F: Field,
    EF: ExtensionField<F>,
    StandardUniform: Distribution<EF>,
{
    let mut rng = SmallRng::seed_from_u64(1);

    let extension_elem = rng.random();
    let ext_degree = EF::DIMENSION;

    let field_order = F::order();

    // Let |F| = p and |EF| = p^d.
    // Then `(|EF| - 1)/(|F| - 1) = 1 + p + ... + p^(d-1)`.
    // Given any element `x`, any symmetric function of `x, x^p, ... x^{p^(d-1)}` must lie in the base field.
    // In particular:
    //
    // `Norm(x)  = x^{(|EF| - 1)/(|F| - 1)} = x^{1 + p + ... + p^(d-1)}`
    // `Trace(x) = x + x^p + ... + x^{p^{d - 1}}`
    // `x^{p^d}  = x`
    // We could test other symmetric functions but that seems unnecessary for now.
    let (trace, norm, power) = (1..ext_degree).fold(
        (extension_elem, extension_elem, extension_elem),
        |(acc, prod, power), _| {
            let next_power = exp_biguint(power, &field_order);
            (acc + next_power, prod * next_power, next_power)
        },
    );

    let ext_power_p_d = exp_biguint(power, &field_order);

    assert!(
        norm.is_in_basefield(),
        "The product of Galois conjugates {norm} of the element {extension_elem} does not lie in the base field.",
    );
    assert!(
        trace.is_in_basefield(),
        "The sum of Galois conjugates {trace} of the element {extension_elem} does not lie in the base field.",
    );
    assert_eq!(
        extension_elem, ext_power_p_d,
        "The element {extension_elem} raised to the power of p^d does not equal itself.",
    );
}

/// Ensure that the methods `from_ext_slice`, `to_ext_iter`, `unpack_transpose_into`,
/// `packed_ext_powers` and `packed_ext_powers_capped` all work as expected.
pub fn test_packed_extension<F, EF>()
where
    F: Field,
    EF: ExtensionField<F>,
    StandardUniform: Distribution<EF>,
{
    let mut rng = SmallRng::seed_from_u64(1);
    let width = F::Packing::WIDTH;
    let extension_elements: Vec<EF> = (0..width).map(|_| rng.random()).collect();

    let packed_extension = EF::ExtensionPacking::from_ext_slice(&extension_elements);
    let unpacked_extension: Vec<EF> =
        EF::ExtensionPacking::to_ext_iter([packed_extension]).collect();

    assert_eq!(extension_elements, unpacked_extension);

    // The fused unpack-and-transpose must equal a flat unpack followed by a transpose.
    //
    // `check` pins one shape against that naive reference.
    //   - The logical matrix is row-major, `src_height` rows by `src_width` columns.
    //   - The total scalar count is `len = src_width * src_height`.
    let mut check = |src_width: usize, len: usize| {
        // Draw `len` random scalars, then pack `width` consecutive ones per packed element.
        let scalars: Vec<EF> = (0..len).map(|_| rng.random()).collect();
        let packed: Vec<_> = scalars
            .chunks(width)
            .map(EF::ExtensionPacking::from_ext_slice)
            .collect();

        // Naive reference: column `c`, row `r` of the transpose is row `r`, column `c` of the input.
        let src_height = len / src_width;
        let mut expected = EF::zero_vec(len);
        for r in 0..src_height {
            for c in 0..src_width {
                expected[c * src_height + r] = scalars[r * src_width + c];
            }
        }

        // Fused path under test: must reproduce the naive transpose exactly.
        let mut transposed = EF::zero_vec(len);
        EF::ExtensionPacking::unpack_transpose_into(&packed, &mut transposed, src_width);
        assert_eq!(
            transposed, expected,
            "transpose mismatch at src_width {src_width}, len {len}",
        );
    };

    // Power-of-two shapes hit both the blocked fast path and the scalar fallback.
    for log_width in 0..6 {
        for log_height in 0..6 {
            let src_width = 1 << log_width;
            let len = src_width << log_height;
            // Skip shapes too small to fill even one packed element.
            if len < width {
                continue;
            }
            check(src_width, len);
        }
    }

    // Non-power-of-two shapes drive the scalar fallback on odd dimensions.
    //
    // Invariant: the `* width` factor in each pair keeps the scalar count a whole number of packed elements.
    // The fallback fires whenever a dimension is not a multiple of the packing width:
    //   - odd width  -> rows do not start on a packed boundary
    //   - odd height -> output runs are shorter than one packed element
    let odd_width_shapes = [(3, width), (5, width), (6, width), (7, 2 * width)];
    let odd_height_shapes = [(width, 3), (width, 5), (2 * width, 3), (3 * width, 7)];
    // Each pair is (columns, rows); the scalar count is their product.
    for (src_width, src_height) in odd_width_shapes.into_iter().chain(odd_height_shapes) {
        check(src_width, src_width * src_height);
    }

    let base_powers = extension_elements[0].powers().collect_n(10 * width);

    let packed_powers = EF::ExtensionPacking::packed_ext_powers(extension_elements[0]);
    let unpacked_powers: Vec<EF> = EF::ExtensionPacking::to_ext_iter(packed_powers)
        .take(10 * width)
        .collect();
    assert_eq!(base_powers, unpacked_powers);

    let packed_powers_capped =
        EF::ExtensionPacking::packed_ext_powers_capped(extension_elements[0], 10 * width);

    let unpacked_powers: Vec<EF> =
        EF::ExtensionPacking::to_ext_iter(packed_powers_capped).collect();
    assert_eq!(base_powers, unpacked_powers);
}

/// Test that Frobenius fixes base field elements: φ(a) = a for a ∈ F.
pub fn test_frobenius_fixes_base_field<F, EF>()
where
    F: Field,
    EF: ExtensionField<F> + HasFrobenius<F>,
    StandardUniform: Distribution<F>,
{
    let mut rng = SmallRng::seed_from_u64(1);
    let base_elem: F = rng.random();
    let base_elem_in_ext: EF = base_elem.into();
    assert_eq!(
        base_elem_in_ext.frobenius(),
        base_elem_in_ext,
        "Frobenius should fix base field elements"
    );

    // Test with special base field elements
    assert_eq!(EF::ZERO.frobenius(), EF::ZERO);
    assert_eq!(EF::ONE.frobenius(), EF::ONE);
    assert_eq!(EF::TWO.frobenius(), EF::TWO);
    assert_eq!(EF::NEG_ONE.frobenius(), EF::NEG_ONE);
}

/// Test Frobenius automorphism properties with 256 random pairs via proptest.
///
/// Verifies additivity: φ(a+b) = φ(a)+φ(b)
/// and multiplicativity: φ(a·b) = φ(a)·φ(b).
pub fn test_frobenius_proptest<F, EF>()
where
    F: Field,
    EF: ExtensionField<F> + HasFrobenius<F> + core::fmt::Debug + 'static,
    StandardUniform: Distribution<EF>,
{
    let config = ProptestConfig::with_cases(256);
    let arb_ef = || {
        any::<u64>().prop_map(|seed| {
            let mut rng = SmallRng::seed_from_u64(seed);
            rng.random::<EF>()
        })
    };
    proptest!(config, |(a in arb_ef(), b in arb_ef())| {
        // Additivity
        prop_assert_eq!(
            (a + b).frobenius(),
            a.frobenius() + b.frobenius(),
            "Frobenius additivity"
        );
        // Multiplicativity
        prop_assert_eq!(
            (a * b).frobenius(),
            a.frobenius() * b.frobenius(),
            "Frobenius multiplicativity"
        );
    });
}
