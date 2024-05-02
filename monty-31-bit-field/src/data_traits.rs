use core::hash::Hash;

use p3_field::{AbstractField, Field};

pub trait FieldParameters:
    Copy
    + Clone
    + Default
    + Eq
    + PartialEq
    + Sync
    + Send
    + Hash
    + 'static
    + MontyParameters
    + FieldConstants
    + TwoAdicData
    + BinomialExtensionData<4>
    + BinomialExtensionData<5>
{
}

pub trait MontyParameters {
    // A 31-bit prime.
    const PRIME: u32;

    // The log_2 of the MONTY constant we use for faster multiplication.
    const MONTY_BITS: u32;

    // We define MU = P^-1 (mod 2^MONTY_BITS). This is different from the usual convention
    // (MU = -P^-1 (mod 2^MONTY_BITS)) but it avoids a carry.
    const MONTY_MU: u32;
    const MONTY_MASK: u32 = ((1u64 << Self::MONTY_BITS) - 1) as u32;
}

pub trait FieldConstants: MontyParameters + Sized {
    // Simple Field Values.
    const MONTY_ZERO: u32 = 0; // The monty form of 0 is always 0.
    const MONTY_ONE: u32 = to_monty::<Self>(1);
    const MONTY_TWO: u32 = to_monty::<Self>(2);
    const MONTY_NEG_ONE: u32 = Self::PRIME - Self::MONTY_ONE; // As MONTY_ONE =/= 0, MONTY_NEG_ONE = P - MONTY_ONE.

    const GEN: u32; // A generator of the fields multiplicative group.
    const MONTY_GEN: u32 = to_monty::<Self>(Self::GEN); // Generator saved in MONTY form

    fn exp_u64_generic<AF: AbstractField>(val: AF, power: u64) -> AF;
    fn try_inverse<F: Field>(p1: F) -> Option<F>;
}

pub trait TwoAdicData {
    const TWO_ADICITY: usize;

    // TODO:
    // Ideally GENERATORS: [u32; usize] and we would have a MONTY_GENERATORS
    // which would be the monty form of each element.
    // Can't seem to do it without passing in TWO_ADICITY as TwoAdicData<TWO_ADICITY> which I'd prefer to avoid for now.
    // Also can's use Vec<u32> as allocators are not allowed in constants
    fn u32_two_adic_generator(bits: usize) -> u32;
}

pub trait BinomialExtensionData<const DEG: usize>: MontyParameters + Sized {
    // W is a value such that (x^DEG - WN) is irreducible.
    const W: u32;
    const MONTY_W: u32 = to_monty::<Self>(Self::W); // The MONTY form of W4

    // DTH_ROOTN = W^((p - 1)/DEG)
    const DTH_ROOT: u32;
    const MONTY_DTH_ROOT: u32 = to_monty::<Self>(Self::DTH_ROOT);

    const EXT_GENERATOR: [u32; DEG];
    const MONTY_EXT_GENERATOR: [u32; DEG] = to_monty_form_array::<DEG, Self>(Self::EXT_GENERATOR);

    const EXT_TWO_ADICITY: usize;

    fn u32_ext_two_adic_generator(bits: usize) -> [u32; DEG];
}

/// Given an element x from a 31 bit field F_P compute x/2.
#[inline]
pub(crate) const fn halve_u32<MP: MontyParameters>(input: u32) -> u32 {
    let shift = (MP::PRIME + 1) >> 1;
    let shr = input >> 1;
    let lo_bit = input & 1;
    let shr_corr = shr + shift;
    if lo_bit == 0 {
        shr
    } else {
        shr_corr
    }
}

#[inline]
pub const fn to_monty<MP: MontyParameters>(x: u32) -> u32 {
    (((x as u64) << MP::MONTY_BITS) % MP::PRIME as u64) as u32
}

#[inline]
pub(crate) const fn to_monty_64<MP: MontyParameters>(x: u64) -> u32 {
    (((x as u128) << MP::MONTY_BITS) % MP::PRIME as u128) as u32
}

#[inline]
#[must_use]
pub(crate) const fn from_monty<MP: MontyParameters>(x: u32) -> u32 {
    monty_reduce::<MP>(x as u64)
}

/// Montgomery reduction of a value in `0..P << MONTY_BITS`.
#[inline]
#[must_use]
pub const fn monty_reduce<MP: MontyParameters>(x: u64) -> u32 {
    let t = x.wrapping_mul(MP::MONTY_MU as u64) & (MP::MONTY_MASK as u64);
    let u = t * (MP::PRIME as u64);

    let (x_sub_u, over) = x.overflowing_sub(u);
    let x_sub_u_hi = (x_sub_u >> MP::MONTY_BITS) as u32;
    let corr = if over { MP::PRIME } else { 0 };
    x_sub_u_hi.wrapping_add(corr)
}

/// Convert a constant u32 array into a constant u32 array with each u32 converted to monty form.
#[inline]
#[must_use]
pub const fn to_monty_form_array<const N: usize, MP: MontyParameters>(input: [u32; N]) -> [u32; N] {
    let mut output = [0; N];
    let mut i = 0;
    loop {
        if i == N {
            break;
        }
        output[i] = to_monty::<MP>(input[i]);
        i += 1;
    }
    output
}
