use crate::{FieldParameters, MontyField31, MontyParameters};

#[inline]
pub const fn to_monty<MP: MontyParameters>(x: u32) -> u32 {
    (((x as u64) << MP::MONTY_BITS) % MP::PRIME as u64) as u32
}

#[inline]
pub const fn to_monty_elem<MP: MontyParameters>(x: u32) -> MontyField31<MP> {
    let val = (((x as u64) << MP::MONTY_BITS) % MP::PRIME as u64) as u32;
    MontyField31::new_monty(val)
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

/// Given an element x from a 31 bit field F_P compute x/2.
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

/// Convert a constant u32 array into a constant u32 array with each u32 converted to monty form.
#[inline]
#[must_use]
pub const fn to_monty_from_array<const N: usize, MP: MontyParameters>(input: [u32; N]) -> [u32; N] {
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

/// Produce a canonical u32 from an element saved in monty form.
pub(crate) fn as_canonical_u32<MP: MontyParameters>(elem: &MontyField31<MP>) -> u32 {
    from_monty::<MP>(elem.value)
}
