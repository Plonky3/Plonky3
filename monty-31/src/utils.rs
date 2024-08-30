use crate::{FieldParameters, MontyParameters};

/// Convert a u32 into MONTY form.
/// There are no constraints on the input.
/// The output will be a u32 in range [0, P).
#[inline]
pub(crate) const fn to_monty<MP: MontyParameters>(x: u32) -> u32 {
    (((x as u64) << MP::MONTY_BITS) % MP::PRIME as u64) as u32
}

/// Convert a u64 into MONTY form.
/// There are no constraints on the input.
/// The output will be a u32 in range [0, P).
#[inline]
pub(crate) const fn to_monty_64<MP: MontyParameters>(x: u64) -> u32 {
    (((x as u128) << MP::MONTY_BITS) % MP::PRIME as u128) as u32
}

/// Convert a u32 out of MONTY form.
/// There are no constraints on the input.
/// The output will be a u32 in range [0, P).
#[inline]
#[must_use]
pub(crate) const fn from_monty<MP: MontyParameters>(x: u32) -> u32 {
    monty_reduce::<MP>(x as u64)
}

/// Given an element x from a 31 bit field F_P compute x/2.
/// The input must be in [0, P).
/// The output will also be in [0, P).
#[inline]
pub(crate) const fn halve_u32<FP: FieldParameters>(input: u32) -> u32 {
    let shr = input >> 1;
    let lo_bit = input & 1;
    let shr_corr = shr + FP::HALF_P_PLUS_1;
    if lo_bit == 0 {
        shr
    } else {
        shr_corr
    }
}

/// Montgomery reduction of a value in `0..P << MONTY_BITS`.
/// the input must be in [0, MONTY * P).
/// the output will be in [0, P).
#[inline]
#[must_use]
pub(crate) const fn monty_reduce<MP: MontyParameters>(x: u64) -> u32 {
    let t = x.wrapping_mul(MP::MONTY_MU as u64) & (MP::MONTY_MASK as u64);
    let u = t * (MP::PRIME as u64);

    let (x_sub_u, over) = x.overflowing_sub(u);
    let x_sub_u_hi = (x_sub_u >> MP::MONTY_BITS) as u32;
    let corr = if over { MP::PRIME } else { 0 };
    x_sub_u_hi.wrapping_add(corr)
}

/// Given x in `0..P << MONTY_BITS`, return x mod P in [0, 2p).
#[inline(always)]
#[must_use]
pub(crate) fn partial_monty_reduce<MP: MontyParameters>(x: u64) -> u32 {
    let q = MP::MONTY_MU.wrapping_mul(x as u32);
    let h = ((q as u64 * MP::PRIME as u64) >> 32) as u32;
    MP::PRIME - h + (x >> 32) as u32
}

/// Given x in [0, 2p), return the representative of x mod p in [0, p)
#[inline(always)]
#[must_use]
pub(crate) fn reduce_2p<MP: MontyParameters>(x: u32) -> u32 {
    debug_assert!(x < 2 * MP::PRIME);

    if x < MP::PRIME {
        x
    } else {
        x - MP::PRIME
    }
}

/// Given x in [0, 4p), return the representative of x mod p in [0, p)
#[inline(always)]
#[must_use]
pub(crate) fn reduce_4p<MP: MontyParameters>(mut x: u64) -> u32 {
    debug_assert!(x < 4 * (MP::PRIME as u64));

    if x > (MP::PRIME as u64) {
        x -= MP::PRIME as u64;
    }
    if x > (MP::PRIME as u64) {
        x -= MP::PRIME as u64;
    }
    if x > (MP::PRIME as u64) {
        x -= MP::PRIME as u64;
    }
    x as u32
}
