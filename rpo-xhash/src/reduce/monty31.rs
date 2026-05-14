//! Reduction helpers for MontyField31 fields (BabyBear, KoalaBear).
//!
//! These just compute `x mod p` for the field prime.
//!
//! For the BB/KB MDS layer, inputs stay in raw Montgomery form, and a linear
//! combination of raw Montgomery residues is still the correct raw Montgomery
//! residue after reduction mod `p`, so the result can be rewrapped directly.
//! For canonical raw-u32 arithmetic, the same helpers also yield the usual
//! representative in `[0, p)`.

use p3_field::PrimeField32;

/// BabyBear: p = 15 * 2^27 + 1 = 2013265921
pub const BB_P: u64 = 15 * (1u64 << 27) + 1;

/// KoalaBear: p = 2^31 - 2^24 + 1 = 2130706433
pub const KB_P: u64 = (1u64 << 31) - (1u64 << 24) + 1;

const MONTY_BITS: u32 = 32;
const MONTY_MASK: u64 = u32::MAX as u64;
const BB_MONTY_MU: u32 = 0x8800_0001;
const KB_MONTY_MU: u32 = 0x8100_0001;

/// Rewrap a raw Montgomery residue as a field element.
///
/// This is valid for Plonky3's `MontyField31`-based fields, which are
/// `repr(transparent)` over their internal `u32` Montgomery residue.
#[inline(always)]
pub fn from_raw_monty_u32<F: PrimeField32>(x: u32) -> F {
    debug_assert!(x < F::ORDER_U32);
    const {
        assert!(core::mem::size_of::<F>() == core::mem::size_of::<u32>());
        assert!(core::mem::align_of::<F>() == core::mem::align_of::<u32>());
    }
    // SAFETY:
    // - `MontyField31` fields are transparent wrappers over `u32`
    // - the value is already a reduced Montgomery residue in `[0, p)`
    unsafe { core::mem::transmute_copy::<u32, F>(&x) }
}

/// Reduce u64 to BabyBear canonical [0, p).
/// The compiler generates an efficient reciprocal multiply for the constant p.
#[inline(always)]
pub fn reduce_bb(x: u64) -> u32 {
    (x % BB_P) as u32
}

/// Montgomery reduction for BabyBear raw Montgomery residues.
/// Input must be in `[0, 2^32 * p)`, which holds for single base-field products.
#[inline(always)]
pub fn monty_reduce_bb(x: u64) -> u32 {
    let t = x.wrapping_mul(BB_MONTY_MU as u64) & MONTY_MASK;
    let u = t * BB_P;
    let (x_sub_u, over) = x.overflowing_sub(u);
    let hi = (x_sub_u >> MONTY_BITS) as u32;
    hi.wrapping_add(if over { BB_P as u32 } else { 0 })
}

/// Montgomery reduction for KoalaBear raw Montgomery residues.
/// Input must be in `[0, 2^32 * p)`, which holds for single base-field products.
#[inline(always)]
pub fn monty_reduce_kb(x: u64) -> u32 {
    let t = x.wrapping_mul(KB_MONTY_MU as u64) & MONTY_MASK;
    let u = t * KB_P;
    let (x_sub_u, over) = x.overflowing_sub(u);
    let hi = (x_sub_u >> MONTY_BITS) as u32;
    hi.wrapping_add(if over { KB_P as u32 } else { 0 })
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeField32;
    use p3_koala_bear::KoalaBear;

    use super::from_raw_monty_u32;

    #[test]
    fn raw_monty_wrap_round_trips_babybear() {
        let x = BabyBear::new(123456789);
        let raw = x.to_unique_u32();
        let rebuilt = from_raw_monty_u32::<BabyBear>(raw);
        assert_eq!(rebuilt, x);
        assert_eq!(rebuilt.to_unique_u32(), raw);
    }

    #[test]
    fn raw_monty_wrap_round_trips_koalabear() {
        let x = KoalaBear::new(987654321);
        let raw = x.to_unique_u32();
        let rebuilt = from_raw_monty_u32::<KoalaBear>(raw);
        assert_eq!(rebuilt, x);
        assert_eq!(rebuilt.to_unique_u32(), raw);
    }
}
