use crate::MontyParameters;

#[inline]
pub(crate) const fn to_monty<MP: MontyParameters>(x: u32) -> u32 {
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
