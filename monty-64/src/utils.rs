//! Utility functions for 64-bit Montgomery arithmetic.

/// Trait for Montgomery parameters for 64-bit fields.
pub trait MontyParameters64:
    Copy + Clone + Default + Eq + core::hash::Hash + Send + Sync + 'static
{
    /// The prime modulus P.
    const PRIME: u64;

    /// Montgomery parameter R = 2^64 mod P.
    const MONTY_R: u64;

    /// R^2 mod P, used for conversion to Montgomery form.
    const MONTY_R2: u64;

    /// -P^{-1} mod 2^64, used in Montgomery reduction.
    const MONTY_INV: u64;

    /// The Montgomery form of 0.
    const MONTY_ZERO: MontyField64<Self>;

    /// The Montgomery form of 1.
    const MONTY_ONE: MontyField64<Self>;

    /// The Montgomery form of 2.
    const MONTY_TWO: MontyField64<Self>;

    /// The Montgomery form of -1.
    const MONTY_NEG_ONE: MontyField64<Self>;
}

use crate::MontyField64;

/// Convert a u64 to Montgomery form.
#[inline(always)]
pub const fn to_monty<MP: MontyParameters64>(value: u64) -> u64 {
    mont_red_const::<MP>((value as u128) * (MP::MONTY_R2 as u128))
}

/// Convert from Montgomery form to standard form.
#[inline(always)]
pub const fn from_monty<MP: MontyParameters64>(value: u64) -> u64 {
    let (a, e) = value.overflowing_add(value << 32);
    let b = a.wrapping_sub(a >> 32).wrapping_sub(e as u64);

    let (r, c) = 0u64.overflowing_sub(b);
    r.wrapping_sub(0u32.wrapping_sub(c as u32) as u64)
}

/// Montgomery reduction: compute (a * R^{-1}) mod P.
#[inline(always)]
pub const fn mont_red_const<MP: MontyParameters64>(a: u128) -> u64 {
    // See reference above for a description of the following implementation.
    let xl = a as u64;
    let xh = (a >> 64) as u64;
    let (a, e) = xl.overflowing_add(xl << 32);

    let b = a.wrapping_sub(a >> 32).wrapping_sub(e as u64);

    let (r, c) = xh.overflowing_sub(b);
    r.wrapping_sub(0u32.wrapping_sub(c as u32) as u64)
}

/// Addition in Montgomery form.
#[inline(always)]
pub const fn add<MP: MontyParameters64>(a: u64, b: u64) -> u64 {
    // We compute a + b = a - (p - b).
    let (x1, c1) = a.overflowing_sub(MP::PRIME - b);
    let adj = 0u32.wrapping_sub(c1 as u32);
    x1.wrapping_sub(adj as u64)
}

/// Subtraction in Montgomery form.
#[inline(always)]
pub const fn sub<MP: MontyParameters64>(a: u64, b: u64) -> u64 {
    let (x1, c1) = a.overflowing_sub(b);
    let adj = 0u32.wrapping_sub(c1 as u32);
    x1.wrapping_sub(adj as u64)
}

/// Multiplication in Montgomery form.
#[inline(always)]
pub const fn mul<MP: MontyParameters64>(a: u64, b: u64) -> u64 {
    mont_red_const::<MP>((a as u128) * (b as u128))
}
