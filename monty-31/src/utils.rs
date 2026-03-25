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

/// Add two integers modulo `P = MP::PRIME`.
///
/// Assumes that `P` is less than `2^31` and `a + b <= 2P` for all array pairs `a, b`.
/// If the inputs are not in this range, the result may be incorrect.
/// The result will be in the range `[0, P]` and equal to `(a + b) mod P`.
/// It will be equal to `P` if and only if `a + b = 2P` so provided `a + b < 2P`
/// the result is guaranteed to be less than `P`.
#[inline]
#[must_use]
pub(crate) const fn add<MP: MontyParameters>(lhs: u32, rhs: u32) -> u32 {
    let mut sum = lhs + rhs;
    let (corr_sum, over) = sum.overflowing_sub(MP::PRIME);
    if !over {
        sum = corr_sum;
    }
    sum
}

/// Subtract two integers modulo `P = MP::PRIME`.
///
/// Assumes that `P` is less than `2^31` and `|a - b| <= P` for all array pairs `a, b`.
/// If the inputs are not in this range, the result may be incorrect.
/// The result will be in the range `[0, P]` and equal to `(a - b) mod P`.
/// It will be equal to `P` if and only if `a - b = P` so provided `a - b < P`
/// the result is guaranteed to be less than `P`.
#[inline]
#[must_use]
pub(crate) const fn sub<MP: MontyParameters>(lhs: u32, rhs: u32) -> u32 {
    let (mut diff, over) = lhs.overflowing_sub(rhs);
    let corr = if over { MP::PRIME } else { 0 };
    diff = diff.wrapping_add(corr);
    diff
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
///
/// The input must be in `[0, MONTY * P)`.
/// The output will be in `[0, P)`.
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

/// Montgomery reduction of a 128-bit value, returning `x * R^{-1} mod P`.
///
/// where `R = 2^MONTY_BITS = 2^32`.
///
/// # Safety
///
/// - Input must satisfy `x < 2^96`.
/// - Output is in `[0, P)`.
///
/// # Algorithm
///
/// Split `x` into two limbs:
///
/// ```text
///     x = hi * 2^64 + lo
///         ──          ──
///         u32         u64
/// ```
///
/// Since `R = 2^32`, multiplying by `R^{-1}` gives:
///
/// ```text
///     x * R^{-1}  =  hi * 2^64 * 2^{-32}  +  lo * 2^{-32}   (mod P)
///                 =  hi * 2^32            +  lo * R^{-1}    (mod P)
/// ```
///
/// Each piece is reduced independently:
/// - The low limb `lo * R^{-1} mod P` is a standard Montgomery reduction on a `u64`.
/// - The high limb `hi * 2^32 mod P` is a conversion into Montgomery form.
/// - The two residues are combined with a single modular addition.
///
/// # Performance
///
/// All arithmetic stays at 64 bits or below:
/// - One Montgomery reduction on a `u64` (multiplies and shifts, no division).
/// - One `u64 % P` where `P` is a compile-time constant (LLVM emits a
///   multiply-and-shift sequence, ~3-5 cycles).
/// - One conditional subtraction for the final modular addition.
///
/// No 128-bit division or modulo is ever performed.
pub(crate) const fn monty_reduce_u128<MP: MontyParameters>(x: u128) -> u32 {
    // Split the 128-bit input into its two limbs.
    //
    // ```text
    //     x  (up to 96 bits)
    //     |-- hi: bits [64..96)  ->  u32   (fits because x < 2^96)
    //     |-- lo: bits [ 0..64)  ->  u64
    // ```
    let lo = x as u64;
    let hi = (x >> 64) as u32;

    // Bring the low limb into the valid input range [0, 2*R*P) for
    // the Montgomery reduction helper.
    //
    // Range analysis:
    //   R*P  = 2^32 * P  <  2^63       (P is a 31-bit prime)
    //   2*R*P            <  2^64
    //   lo               <= 2^64 - 1   (arbitrary u64)
    //
    // So the low limb can exceed the accepted range by at most one copy of 2*R*P.
    // Subtracting it once is always enough because 2^64 < 4*R*P for any 31-bit prime.
    //
    // Correctness: 2*R*P is a multiple of P, so this subtraction
    // does not change the residue modulo P.
    let two_rp = ((MP::PRIME as u64) << MP::MONTY_BITS) << 1;
    let lo_reduced = if lo >= two_rp { lo - two_rp } else { lo };

    // Reduce the low limb: lo * R^{-1} mod P.
    //
    // This is a standard Montgomery reduction on a u64 in [0, 2*R*P).
    // It uses only multiplies, bitwise ops, and conditional subtraction
    // -- no division instruction at all.
    let r = large_monty_reduce::<MP>(lo_reduced);

    // Reduce the high limb: hi * 2^32 mod P.
    //
    // Converting into Montgomery form computes exactly
    //     ((hi as u64) << 32) % P
    // Because P is a compile-time constant, LLVM replaces this
    // single u64-by-constant modulo with a multiply-and-shift
    // sequence (~3-5 cycles, no `div` instruction).
    let hi_r_mod_p = to_monty::<MP>(hi);

    // Combine the two reduced halves with a modular addition.
    //
    // Both operands are in [0, P), so their sum is in [0, 2*P).
    // A single conditional subtraction of P yields a result in [0, P).
    add::<MP>(hi_r_mod_p, r)
}
