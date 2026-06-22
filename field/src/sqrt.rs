//! Square roots in finite fields via the Tonelli–Shanks algorithm.

use num_bigint::BigUint;

use crate::{Field, TwoAdicField};

/// Compute `base^exponent` for an arbitrarily large exponent.
///
/// Uses the standard square-and-multiply approach over the bits of `exponent`.
/// This accepts exponents which do not fit in a `u64`, as required by fields
/// whose order exceeds `2^64`.
fn exp_biguint<F: Field>(base: F, exponent: &BigUint) -> F {
    let mut product = F::ONE;
    let mut current = base;
    for j in 0..exponent.bits() {
        if exponent.bit(j) {
            product *= current;
        }
        current = current.square();
    }
    product
}

/// The core of the Tonelli–Shanks algorithm.
///
/// Given a nonzero `a`, the decomposition `|F| - 1 = q * 2^s` with `q` odd, and a
/// primitive `2^s`-th root of unity `c` (the `2`-Sylow generator), return a square
/// root of `a` if one exists.
///
/// `c` must be a quadratic non-residue raised to the `q`-th power; both
/// [`tonelli_shanks`] and [`tonelli_shanks_two_adic`] provide such a value.
fn tonelli_shanks_inner<F: Field>(a: F, s: usize, q: &BigUint, mut c: F) -> Option<F> {
    // A single exponentiation yields both candidates: with `u = a^((q-1)/2)`,
    // `r = u * a = a^((q+1)/2)` is the prospective root and `t = r * u = a^q`
    // tracks the residue's `2`-power component.
    let u = exp_biguint(a, &((q - 1u32) >> 1));
    let mut r = u * a;
    let mut t = r * u;
    let mut m = s;

    while !t.is_one() {
        // Find the least `i`, with `0 < i < m`, such that `t^(2^i) == 1`.
        let mut i = 0;
        let mut t2i = t;
        while !t2i.is_one() {
            t2i = t2i.square();
            i += 1;
            if i == m {
                // `t` has order `2^m`, which only happens when `a` is a
                // quadratic non-residue, so no square root exists.
                return None;
            }
        }

        let b = c.exp_power_of_2(m - i - 1);
        m = i;
        c = b.square();
        t *= c;
        r *= b;
    }

    Some(r)
}

/// Return a square root of `a` if one exists, otherwise `None`.
///
/// This is the generic Tonelli–Shanks algorithm. Writing the multiplicative
/// group order as `|F| - 1 = q * 2^s` with `q` odd, it uses [`Field::GENERATOR`]
/// (a generator of `F^*`, hence a quadratic non-residue) to seed the `2`-Sylow
/// subgroup via `c = GENERATOR^q`.
///
/// For a quadratic residue this returns one of its two square roots; which one
/// is unspecified. `ZERO` maps to `ZERO`.
pub fn tonelli_shanks<F: Field>(a: F) -> Option<F> {
    // Zero is its own square root and would otherwise break the logic below.
    if a.is_zero() {
        return Some(F::ZERO);
    }

    // Write `|F| - 1 = q * 2^s` with `q` odd.
    let order_minus_one = F::order() - BigUint::from(1u8);
    // `|F| >= 2`, so `order_minus_one >= 1` and `trailing_zeros` is never `None`.
    let s = order_minus_one
        .trailing_zeros()
        .expect("field order must be at least two") as usize;
    let q = &order_minus_one >> s;

    let c = exp_biguint(F::GENERATOR, &q);
    tonelli_shanks_inner(a, s, &q, c)
}

/// Return a square root of `a` if one exists, otherwise `None`.
///
/// A variant of [`tonelli_shanks`] for two-adic fields. It seeds the `2`-Sylow
/// subgroup directly from [`TwoAdicField::two_adic_generator`] (a primitive
/// `2^TWO_ADICITY`-th root of unity), avoiding the `GENERATOR^q` exponentiation
/// that the generic version performs on every call.
///
/// For a quadratic residue this returns one of its two square roots; which one
/// is unspecified. `ZERO` maps to `ZERO`.
pub fn tonelli_shanks_two_adic<F: TwoAdicField>(a: F) -> Option<F> {
    // Zero is its own square root and would otherwise break the logic below.
    if a.is_zero() {
        return Some(F::ZERO);
    }

    // For a two-adic field, `s` is exactly `TWO_ADICITY` and the `2`-Sylow
    // generator is available as a constant.
    let s = F::TWO_ADICITY;
    let q = (F::order() - BigUint::from(1u8)) >> s;
    let c = F::two_adic_generator(s);
    tonelli_shanks_inner(a, s, &q, c)
}
