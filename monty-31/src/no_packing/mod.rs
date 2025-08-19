//! A couple of simple functions needed in the case that this is compiled without architecture optimizations available.

mod poseidon2;

use p3_field::extension::{octic_mul, quartic_mul, quintic_mul};
pub use poseidon2::*;

use crate::{BinomialExtensionData, FieldParameters, MontyField31};

/// If no packings are available, we use the generic binomial extension multiplication functions.
#[inline]
pub(crate) fn quartic_mul_packed<FP, const WIDTH: usize>(
    a: &[MontyField31<FP>; WIDTH],
    b: &[MontyField31<FP>; WIDTH],
    res: &mut [MontyField31<FP>; WIDTH],
) where
    FP: FieldParameters + BinomialExtensionData<WIDTH>,
{
    quartic_mul(a, b, res, FP::W);
}

/// If no packings are available, we use the generic binomial extension multiplication functions.
#[inline]
pub(crate) fn quintic_mul_packed<FP, const WIDTH: usize>(
    a: &[MontyField31<FP>; WIDTH],
    b: &[MontyField31<FP>; WIDTH],
    res: &mut [MontyField31<FP>; WIDTH],
) where
    FP: FieldParameters + BinomialExtensionData<WIDTH>,
{
    quintic_mul(a, b, res, FP::W);
}

/// If no packings are available, we use the generic binomial extension multiplication functions.
#[inline]
pub(crate) fn octic_mul_packed<FP, const WIDTH: usize>(
    a: &[MontyField31<FP>; WIDTH],
    b: &[MontyField31<FP>; WIDTH],
    res: &mut [MontyField31<FP>; WIDTH],
) where
    FP: FieldParameters + BinomialExtensionData<WIDTH>,
{
    octic_mul(a, b, res, FP::W);
}

/// Multiplication by a base field element in a binomial extension field.
#[inline]
pub(crate) fn base_mul_packed<FP, const WIDTH: usize>(
    a: [MontyField31<FP>; WIDTH],
    b: MontyField31<FP>,
    res: &mut [MontyField31<FP>; WIDTH],
) where
    FP: FieldParameters + BinomialExtensionData<WIDTH>,
{
    res.iter_mut().zip(a.iter()).for_each(|(r, a)| *r = *a * b);
}
