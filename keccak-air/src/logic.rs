use p3_field::{AbstractField, PrimeField64};

#[inline(always)]
pub(crate) fn xor<F: PrimeField64, const N: usize>(xs: [F; N]) -> F {
    let mut acc: u64 = 0;
    for x in xs {
        debug_assert!(x.is_zero() || x.is_one());
        acc ^= x.is_one() as u64;
    }
    if acc == 0 {
        F::zero()
    } else {
        F::one()
    }
}

/// Computes the arithmetic generalization of `xor(x, y)`, i.e. `x + y - 2 x y`.
#[inline(always)]
pub(crate) fn xor_gen<AF: AbstractField>(x: AF, y: AF) -> AF {
    x.clone() + y.clone() - x * y.double()
}

/// Computes the arithmetic generalization of `xor3(x, y, z)`.
#[inline(always)]
pub(crate) fn xor3_gen<AF: AbstractField>(x: AF, y: AF, z: AF) -> AF {
    xor_gen(x, xor_gen(y, z))
}

#[inline(always)]
pub(crate) fn andn<F: PrimeField64>(x: F, y: F) -> F {
    debug_assert!(x.is_zero() || x.is_one());
    debug_assert!(y.is_zero() || y.is_one());
    let x = x.is_one();
    let y = y.is_one();
    if !x && y {
        F::one()
    } else {
        F::zero()
    }
}

#[inline(always)]
pub(crate) fn andn_gen<AF: AbstractField>(x: AF, y: AF) -> AF {
    (AF::one() - x) * y
}
