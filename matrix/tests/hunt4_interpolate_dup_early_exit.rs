//! hunt-4: `interpolate_arbitrary_point` early-exit can violate the
//! "duplicate domain points → None" contract.
//!
//! From the docstring (matrix/src/interpolation.rs):
//!   # Returns
//!   - `None` if any domain points coincide.
//!   - The matching row directly when the target equals a domain point.
//!
//! However the implementation checks "target equals a domain point"
//! BEFORE the duplicate-domain check inside `barycentric_weights`. So if
//! the caller passes `x_coords` with a duplicate, AND the target happens
//! to equal that duplicate value, the function returns `Some(row)` instead
//! of `None` — silently masking the invalid input.
//!
//! This violates the documented contract and yields ambiguous output:
//! when `x_coords[i] == x_coords[j]` and the matrix's row i differs from
//! row j, there is no well-defined polynomial through the points, but
//! the function commits to row i without warning.

use p3_baby_bear::BabyBear;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::interpolation::InterpolateArbitrary;

type F = BabyBear;

#[test]
fn duplicate_domain_with_matching_target_should_return_none() {
    // Three rows, with x_coords[0] == x_coords[2] but different values
    // in those rows — the interpolation problem is ill-posed.
    let xs = [F::ONE, F::TWO, F::ONE];
    let evals = RowMajorMatrix::new(
        vec![F::from_u32(10), F::from_u32(20), F::from_u32(30)],
        1,
    );

    // Per the docstring, duplicate domain → None.
    let result = evals.interpolate_arbitrary_point(&xs, F::ONE);
    assert_eq!(
        result, None,
        "interpolate_arbitrary_point must return None for duplicate domain, got {result:?}"
    );
}

#[test]
fn duplicate_domain_with_non_matching_target_returns_none() {
    // Same duplicate setup, but the target does NOT match any domain
    // point. The early-exit doesn't fire, and `barycentric_weights`
    // catches the duplicate. This is the working path; included for
    // contrast.
    let xs = [F::ONE, F::TWO, F::ONE];
    let evals = RowMajorMatrix::new(
        vec![F::from_u32(10), F::from_u32(20), F::from_u32(30)],
        1,
    );

    let result = evals.interpolate_arbitrary_point(&xs, F::from_u32(99));
    assert_eq!(result, None, "non-matching target on duplicate domain");
}
