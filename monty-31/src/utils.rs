use crate::{FieldParameters, MontyParameters};

/// Convert a u32 into MONTY form.
/// There are no constraints on the input.
/// The output will be a u32 in range `[0, P)`.
#[inline]
pub(crate) const fn to_monty<MP: MontyParameters>(x: u32) -> u32 {
    (((x as u64) << MP::MONTY_BITS) % MP::PRIME as u64) as u32
}

/// Convert an i32 into MONTY form.
/// There are no constraints on the input.
/// The output will be a u32 in range `[0, P)`.
#[inline]
pub(crate) const fn to_monty_signed<MP: MontyParameters>(x: i32) -> u32 {
    let red = (((x as i64) << MP::MONTY_BITS) % MP::PRIME as i64) as i32;
    if red >= 0 {
        red as u32
    } else {
        MP::PRIME.wrapping_add_signed(red)
    }
}

/// Convert a u64 into MONTY form.
/// There are no constraints on the input.
/// The output will be a u32 in range `[0, P)`.
#[inline]
pub(crate) const fn to_monty_64<MP: MontyParameters>(x: u64) -> u32 {
    (((x as u128) << MP::MONTY_BITS) % MP::PRIME as u128) as u32
}

/// Convert an i64 into MONTY form.
/// There are no constraints on the input.
/// The output will be a u32 in range `[0, P)`.
#[inline]
pub(crate) const fn to_monty_64_signed<MP: MontyParameters>(x: i64) -> u32 {
    let red = (((x as i128) << MP::MONTY_BITS) % MP::PRIME as i128) as i32;
    if red >= 0 {
        red as u32
    } else {
        MP::PRIME.wrapping_add_signed(red)
    }
}

/// Convert a u32 out of MONTY form.
/// There are no constraints on the input.
/// The output will be a u32 in range `[0, P)`.
#[inline]
#[must_use]
pub(crate) const fn from_monty<MP: MontyParameters>(x: u32) -> u32 {
    monty_reduce::<MP>(x as u64)
}

/// Given an element `x` from a 31 bit field `F` compute `x/2`.
/// The input must be in `[0, P)`.
/// The output will also be in `[0, P)`.
#[inline]
pub(crate) const fn halve_u32<FP: FieldParameters>(input: u32) -> u32 {
    let shr = input >> 1;
    let lo_bit = input & 1;
    let shr_corr = shr + FP::HALF_P_PLUS_1;
    if lo_bit == 0 { shr } else { shr_corr }
}

/// Montgomery reduction of a value in `0..P << MONTY_BITS`.
/// the input must be in `[0, MONTY * P)`.
/// the output will be in `[0, P)`.
#[inline]
#[must_use]
pub(crate) const fn monty_reduce<MP: MontyParameters>(x: u64) -> u32 {
    // t = x * MONTY_MU mod MONTY
    let t = x.wrapping_mul(MP::MONTY_MU as u64) & (MP::MONTY_MASK as u64);

    // u = t * P
    let u = t * (MP::PRIME as u64);

    // Thus:
    // 1. x - u = x - t * P = x mod P
    // 2. x - u = x - x * MONTY_MU * P mod MONTY = 0 mod MONTY
    // For the second point note that MONTY_MU = P^{-1} mod MONTY.

    // Additionally, u < MONTY * P so: - MONTY * P < x - u < MONTY * P
    // Thus after dividing by MONTY, -P < (x - u)/MONTY < P.
    // So we can just add P to the result if it is negative.

    let (x_sub_u, over) = x.overflowing_sub(u);
    let x_sub_u_hi = (x_sub_u >> MP::MONTY_BITS) as u32;
    let corr = if over { MP::PRIME } else { 0 };
    x_sub_u_hi.wrapping_add(corr)
}

/// Montgomery reduction of a value in `0..P << MONTY_BITS`.
/// The input must be in [0, 2 * MONTY * P).
/// The output will be in [0, P).
///
/// This is slower than `monty_reduce` but has a larger input range.
#[inline]
#[must_use]
pub(crate) const fn large_monty_reduce<MP: MontyParameters>(x: u64) -> u32 {
    // t = x * MONTY_MU mod MONTY
    let t = x.wrapping_mul(MP::MONTY_MU as u64) & (MP::MONTY_MASK as u64);

    // u = t * P
    let u = t * (MP::PRIME as u64);

    // Thus:
    // 1. x - u = x - t * P = x mod P
    // 2. x - u = x - x * MONTY_MU * P mod MONTY = 0 mod MONTY
    // For the second point note that MONTY_MU = P^{-1} mod MONTY.

    // This time, - MONTY * P < x - u < 2 * MONTY * P so we need to be
    // more careful with our reduction.
    // The trick is just to first reduce x to lie in [0, MONTY * P).
    let (x_prime, over) = x.overflowing_sub((MP::PRIME as u64) << MP::MONTY_BITS);
    let x_corr = if over { x } else { x_prime };

    // Now we can do the same as before.

    let (x_sub_u, over) = x_corr.overflowing_sub(u);
    let x_sub_u_hi = (x_sub_u >> MP::MONTY_BITS) as u32;
    let corr = if over { MP::PRIME } else { 0 };
    x_sub_u_hi.wrapping_add(corr)
}

/// Perform a monty reduction on a u128 in the range `[0, 2^96)`
///
/// The input will be in `[0, P)` and be equal to `x * MONTY^{-1} mod P`.
pub(crate) fn monty_reduce_u128<MP: MontyParameters>(x: u128) -> u32 {
    // TODO: There is probably a way to do this faster than using %.

    // Need to find MONTY^{-1} mod P.
    // As P * MONTY_MU = 1 mod MONTY, we know that P * MONTY_MU = 1 + k * MONTY for some k.
    // Thus k * MONTY = -1 mod P.
    // Rearranging, we get k = (P * MONTY_MU - 1) / MONTY.
    // Thus we want -k = P - k = P - (P * MONTY_MU - 1) / MONTY.

    // Compiler should realize that this is a constant.
    let monty_inv_mod_p =
        MP::PRIME - ((((MP::PRIME as u64) * (MP::MONTY_MU as u64)) - 1) >> MP::MONTY_BITS) as u32;

    // As monty_inv_mod_p < 2^32, x * monty_inv_mod_p < 2^128 so the product below will not overflow.
    ((x * (monty_inv_mod_p as u128)) % (MP::PRIME as u128)) as u32
}
