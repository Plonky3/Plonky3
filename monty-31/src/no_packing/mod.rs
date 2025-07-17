//! A couple of simple functions needed in the case that this is compiled without architecture optimizations available.

mod poseidon2;

pub use poseidon2::*;

use crate::{FieldParameters, BinomialExtensionData, MontyField31};
use p3_field::extension::{quartic_mul, quintic_mul, octic_mul};

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
