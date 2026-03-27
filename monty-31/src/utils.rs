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

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[derive(Copy, Clone, Default, Debug, Eq, PartialEq, Hash)]
    struct KB;
    impl MontyParameters for KB {
        const PRIME: u32 = 0x7f000001; // 2^31 - 2^24 + 1
        const MONTY_BITS: u32 = 32;
        const MONTY_MU: u32 = 0x81000001;
    }

    #[derive(Copy, Clone, Default, Debug, Eq, PartialEq, Hash)]
    struct BB;
    impl MontyParameters for BB {
        const PRIME: u32 = 0x78000001; // 2^31 - 2^27 + 1
        const MONTY_BITS: u32 = 32;
        const MONTY_MU: u32 = 0x88000001;
    }

    const KB_P: u32 = KB::PRIME;
    const BB_P: u32 = BB::PRIME;

    // R = 2^32, the Montgomery radix for both fields.
    const R: u64 = 1u64 << 32;

    // Input range upper bounds for the two reduction variants.
    const KB_RP: u64 = KB_P as u64 * R;
    const KB_TWO_RP: u64 = KB_RP * 2;
    const BB_RP: u64 = BB_P as u64 * R;
    const BB_TWO_RP: u64 = BB_RP * 2;

    proptest! {
        // ---------------------------------------------------------------
        // to_monty / from_monty roundtrip
        //
        // Encoding then decoding must recover the original element:
        //   from_monty(to_monty(x))  =  (x * R) * R^{-1}  =  x   (mod P)
        // ---------------------------------------------------------------

        #[test]
        fn kb_to_from_monty_roundtrip(x in 0..KB_P) {
            prop_assert_eq!(from_monty::<KB>(to_monty::<KB>(x)), x);
        }

        #[test]
        fn bb_to_from_monty_roundtrip(x in 0..BB_P) {
            prop_assert_eq!(from_monty::<BB>(to_monty::<BB>(x)), x);
        }

        // Reverse direction — decoding then re-encoding:
        //   to_monty(from_monty(m))  =  (m * R^{-1}) * R  =  m   (mod P)

        #[test]
        fn kb_from_to_monty_roundtrip(m in 0..KB_P) {
            prop_assert_eq!(to_monty::<KB>(from_monty::<KB>(m)), m);
        }

        #[test]
        fn bb_from_to_monty_roundtrip(m in 0..BB_P) {
            prop_assert_eq!(to_monty::<BB>(from_monty::<BB>(m)), m);
        }

        // ---------------------------------------------------------------
        // Output range: every conversion must land in [0, P).
        // ---------------------------------------------------------------

        #[test]
        fn kb_to_monty_output_range(x: u32) {
            prop_assert!(to_monty::<KB>(x) < KB_P);
        }

        #[test]
        fn bb_to_monty_output_range(x: u32) {
            prop_assert!(to_monty::<BB>(x) < BB_P);
        }

        #[test]
        fn kb_to_monty_signed_output_range(x: i32) {
            prop_assert!(to_monty_signed::<KB>(x) < KB_P);
        }

        #[test]
        fn bb_to_monty_signed_output_range(x: i32) {
            prop_assert!(to_monty_signed::<BB>(x) < BB_P);
        }

        // ---------------------------------------------------------------
        // Signed / unsigned consistency
        //
        // For non-negative inputs below P the signed and unsigned
        // conversions must agree.
        // ---------------------------------------------------------------

        #[test]
        fn kb_to_monty_signed_agrees_for_positive(x in 0..KB_P) {
            prop_assert_eq!(to_monty_signed::<KB>(x as i32), to_monty::<KB>(x));
        }

        #[test]
        fn bb_to_monty_signed_agrees_for_positive(x in 0..BB_P) {
            prop_assert_eq!(to_monty_signed::<BB>(x as i32), to_monty::<BB>(x));
        }

        // ---------------------------------------------------------------
        // Negation identity
        //
        // The Montgomery forms of x and -x must sum to zero modulo P:
        //   to_monty(x) + to_monty_signed(-x)  =  x*R + (-x)*R  =  0
        // ---------------------------------------------------------------

        #[test]
        fn kb_to_monty_signed_negation(x in 1..KB_P) {
            let pos = to_monty::<KB>(x);
            let neg = to_monty_signed::<KB>(-(x as i32));
            prop_assert_eq!(add::<KB>(pos, neg), 0);
        }

        #[test]
        fn bb_to_monty_signed_negation(x in 1..BB_P) {
            let pos = to_monty::<BB>(x);
            let neg = to_monty_signed::<BB>(-(x as i32));
            prop_assert_eq!(add::<BB>(pos, neg), 0);
        }

        // ---------------------------------------------------------------
        // 64-bit path must extend the 32-bit path
        //
        // Converting a u32 via the 64-bit function must give the same
        // Montgomery form as the 32-bit function.
        // ---------------------------------------------------------------

        #[test]
        fn kb_to_monty_64_extends_32(x: u32) {
            prop_assert_eq!(to_monty_64::<KB>(x as u64), to_monty::<KB>(x));
        }

        #[test]
        fn bb_to_monty_64_extends_32(x: u32) {
            prop_assert_eq!(to_monty_64::<BB>(x as u64), to_monty::<BB>(x));
        }

        // Signed 64-bit must agree with unsigned 64-bit for non-negative
        // inputs below P.

        #[test]
        fn kb_to_monty_64_signed_agrees_for_positive(x in 0..KB_P) {
            prop_assert_eq!(
                to_monty_64_signed::<KB>(x as i64),
                to_monty_64::<KB>(x as u64),
            );
        }

        #[test]
        fn bb_to_monty_64_signed_agrees_for_positive(x in 0..BB_P) {
            prop_assert_eq!(
                to_monty_64_signed::<BB>(x as i64),
                to_monty_64::<BB>(x as u64),
            );
        }

        // ---------------------------------------------------------------
        // Modular addition: commutativity
        // ---------------------------------------------------------------

        #[test]
        fn kb_add_commutative(a in 0..KB_P, b in 0..KB_P) {
            prop_assert_eq!(add::<KB>(a, b), add::<KB>(b, a));
        }

        #[test]
        fn bb_add_commutative(a in 0..BB_P, b in 0..BB_P) {
            prop_assert_eq!(add::<BB>(a, b), add::<BB>(b, a));
        }

        // ---------------------------------------------------------------
        // Modular add / sub roundtrip
        //
        // Subtraction must undo addition and vice versa.
        // ---------------------------------------------------------------

        #[test]
        fn kb_add_sub_roundtrip(a in 0..KB_P, b in 0..KB_P) {
            prop_assert_eq!(sub::<KB>(add::<KB>(a, b), b), a);
        }

        #[test]
        fn bb_add_sub_roundtrip(a in 0..BB_P, b in 0..BB_P) {
            prop_assert_eq!(sub::<BB>(add::<BB>(a, b), b), a);
        }

        #[test]
        fn kb_sub_add_roundtrip(a in 0..KB_P, b in 0..KB_P) {
            prop_assert_eq!(add::<KB>(sub::<KB>(a, b), b), a);
        }

        #[test]
        fn bb_sub_add_roundtrip(a in 0..BB_P, b in 0..BB_P) {
            prop_assert_eq!(add::<BB>(sub::<BB>(a, b), b), a);
        }

        // ---------------------------------------------------------------
        // monty_reduce: output range and core Montgomery identity
        //
        // The core identity that defines Montgomery reduction:
        //   result * R  ≡  x   (mod P)
        //
        // This verifies correctness without computing R^{-1} explicitly.
        // ---------------------------------------------------------------

        #[test]
        fn kb_monty_reduce_output_range(x in 0..KB_RP) {
            prop_assert!(monty_reduce::<KB>(x) < KB_P);
        }

        #[test]
        fn bb_monty_reduce_output_range(x in 0..BB_RP) {
            prop_assert!(monty_reduce::<BB>(x) < BB_P);
        }

        #[test]
        fn kb_monty_reduce_identity(x in 0..KB_RP) {
            let result = monty_reduce::<KB>(x) as u64;
            prop_assert_eq!((result * R) % KB_P as u64, x % KB_P as u64);
        }

        #[test]
        fn bb_monty_reduce_identity(x in 0..BB_RP) {
            let result = monty_reduce::<BB>(x) as u64;
            prop_assert_eq!((result * R) % BB_P as u64, x % BB_P as u64);
        }

        // ---------------------------------------------------------------
        // large_monty_reduce: must agree with monty_reduce on the
        // overlapping range [0, R*P), and satisfy the same identity
        // on its full domain [0, 2*R*P).
        // ---------------------------------------------------------------

        #[test]
        fn kb_large_agrees_with_standard(x in 0..KB_RP) {
            prop_assert_eq!(large_monty_reduce::<KB>(x), monty_reduce::<KB>(x));
        }

        #[test]
        fn bb_large_agrees_with_standard(x in 0..BB_RP) {
            prop_assert_eq!(large_monty_reduce::<BB>(x), monty_reduce::<BB>(x));
        }

        #[test]
        fn kb_large_monty_reduce_output_range(x in 0..KB_TWO_RP) {
            prop_assert!(large_monty_reduce::<KB>(x) < KB_P);
        }

        #[test]
        fn bb_large_monty_reduce_output_range(x in 0..BB_TWO_RP) {
            prop_assert!(large_monty_reduce::<BB>(x) < BB_P);
        }

        #[test]
        fn kb_large_monty_reduce_identity(x in 0..KB_TWO_RP) {
            let result = large_monty_reduce::<KB>(x) as u64;
            prop_assert_eq!((result * R) % KB_P as u64, x % KB_P as u64);
        }

        #[test]
        fn bb_large_monty_reduce_identity(x in 0..BB_TWO_RP) {
            let result = large_monty_reduce::<BB>(x) as u64;
            prop_assert_eq!((result * R) % BB_P as u64, x % BB_P as u64);
        }

        // ---------------------------------------------------------------
        // monty_reduce_u128: output range, core identity, and
        // agreement with the u64 version on the overlapping range.
        // ---------------------------------------------------------------

        #[test]
        fn kb_monty_reduce_u128_output_range(lo: u64, hi: u32) {
            let x = (hi as u128) << 64 | lo as u128;
            prop_assert!(monty_reduce_u128::<KB>(x) < KB_P);
        }

        #[test]
        fn bb_monty_reduce_u128_output_range(lo: u64, hi: u32) {
            let x = (hi as u128) << 64 | lo as u128;
            prop_assert!(monty_reduce_u128::<BB>(x) < BB_P);
        }

        #[test]
        fn kb_monty_reduce_u128_identity(lo: u64, hi: u32) {
            let x = (hi as u128) << 64 | lo as u128;
            let result = monty_reduce_u128::<KB>(x) as u128;
            prop_assert_eq!((result * R as u128) % KB_P as u128, x % KB_P as u128);
        }

        #[test]
        fn bb_monty_reduce_u128_identity(lo: u64, hi: u32) {
            let x = (hi as u128) << 64 | lo as u128;
            let result = monty_reduce_u128::<BB>(x) as u128;
            prop_assert_eq!((result * R as u128) % BB_P as u128, x % BB_P as u128);
        }

        #[test]
        fn kb_monty_reduce_u128_agrees_with_u64(x in 0..KB_RP) {
            prop_assert_eq!(
                monty_reduce_u128::<KB>(x as u128),
                monty_reduce::<KB>(x),
            );
        }

        #[test]
        fn bb_monty_reduce_u128_agrees_with_u64(x in 0..BB_RP) {
            prop_assert_eq!(
                monty_reduce_u128::<BB>(x as u128),
                monty_reduce::<BB>(x),
            );
        }

        // ---------------------------------------------------------------
        // Montgomery multiplication
        //
        // Reducing the product of two Montgomery-form values must yield
        // the Montgomery form of their field product:
        //   reduce(to_monty(a) * to_monty(b))
        //     = (a*R * b*R) * R^{-1}
        //     = a*b*R
        //     = to_monty(a*b mod P)
        // ---------------------------------------------------------------

        #[test]
        fn kb_monty_mul_correct(a in 0..KB_P, b in 0..KB_P) {
            let product = to_monty::<KB>(a) as u64 * to_monty::<KB>(b) as u64;
            let expected = to_monty::<KB>(((a as u64 * b as u64) % KB_P as u64) as u32);
            prop_assert_eq!(monty_reduce::<KB>(product), expected);
        }

        #[test]
        fn bb_monty_mul_correct(a in 0..BB_P, b in 0..BB_P) {
            let product = to_monty::<BB>(a) as u64 * to_monty::<BB>(b) as u64;
            let expected = to_monty::<BB>(((a as u64 * b as u64) % BB_P as u64) as u32);
            prop_assert_eq!(monty_reduce::<BB>(product), expected);
        }
    }
}
