use alloc::vec::Vec;

use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue};
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

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

/// Ensure that the methods `from_ext_slice`, `to_ext_iter`, `packed_ext_powers` and `packed_ext_powers_capped`
/// all work as expected.
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

    let base_powers: Vec<EF> = extension_elements[0].powers().take(10 * width).collect();

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
