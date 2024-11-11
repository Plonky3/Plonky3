//! A collection of utility functions helpful in defining AIR's.

use core::array;

use p3_field::{AbstractField, Field};

use crate::AirBuilder;

/// Pack a collection of bits into a number.
///
/// Given vec = [v0, v1, ..., v_n] returns v0 + 2v_1 + ... + 2^n v_n
#[inline]
pub fn pack_bits_le<AF, Var, I>(iter: I) -> AF
where
    AF: AbstractField,
    Var: Into<AF> + Clone,
    I: DoubleEndedIterator<Item = Var>,
{
    let mut output = AF::ZERO;
    for elem in iter.rev() {
        output = output.double();
        output += elem.clone().into();
    }
    output
}

/// Computes the arithmetic generalization of boolean `xor`.
///
/// For boolean inputs, `x ^ y = x + y - 2 xy`.
#[inline(always)]
pub fn xor<AF: AbstractField>(x: AF, y: AF) -> AF {
    x.clone() + y.clone() - x * y.double()
}

/// Computes the arithmetic generalization of a triple `xor`.
///
/// For boolean inputs `x ^ y ^ z = x + y + z - 2(xy + xz + yz) + 4xyz`.
#[inline(always)]
pub fn xor3<AF: AbstractField>(x: AF, y: AF, z: AF) -> AF {
    // The cheapest way to implement this polynomial is to simply apply xor twice.
    // This costs 2 adds, 2 subs, 2 muls and 2 doubles.
    xor(x, xor(y, z))
}

/// Computes the arithmetic generalization of `andnot`.
///
/// For boolean inputs `(!x) & y = (1 - x)y`
#[inline(always)]
pub fn andn<AF: AbstractField>(x: AF, y: AF) -> AF {
    (AF::ONE - x) * y
}

/// Compute `xor` on a list of field elements using the arithmetic generalization.
///
/// Additionally verifies at debug time that all inputs are boolean.
#[inline(always)]
pub fn checked_xor<F: Field, const N: usize>(xs: [F; N]) -> F {
    xs.into_iter().fold(F::ZERO, |acc, x| {
        debug_assert!(x.is_zero() || x.is_one());
        xor(acc, x)
    })
}

/// Compute the `andnot` on a pair of field elements using the arithmetic generalization.
///
/// Additionally verifies at debug time that both inputs are boolean.
#[inline(always)]
pub fn checked_andn<F: Field>(x: F, y: F) -> F {
    debug_assert!(x.is_zero() || x.is_one());
    debug_assert!(y.is_zero() || y.is_one());
    andn(x, y)
}

/// Convert a 32-bit integer into an array of 32 0 or 1 field elements.
///
/// The output array is in little-endian order.
#[inline]
pub fn u32_to_bits_le<AF: AbstractField>(val: u32) -> [AF; 32] {
    // We do this over F::from_canonical_u32 as from_canonical_u32 can be slow
    // like in the case of monty field.
    array::from_fn(|i| {
        if val & (1 << i) != 0 {
            AF::ONE
        } else {
            AF::ZERO
        }
    })
}

/// Compute a + b mod 2^32 returning the summation and data allowing us to verify the summation.
///
/// The data we need is whether a + b > 2^32 and similarly
/// whether (a mod 2^16) + (b mod 2^16) > 2^16;
#[inline]
pub fn verifiable_add(a: u32, b: u32) -> (u32, [bool; 2]) {
    let a_16 = a as u16;
    let b_16 = b as u16;

    let (_, overflow_16) = a_16.overflowing_add(b_16);
    let (sum, overflow) = a.overflowing_add(b);

    (sum, [overflow, overflow_16])
}

/// Verify that `a = b + c + d mod 2^32`
///
/// We assume that a, b, c, d are all given as `2 16` bit limbs (e.g. `a = a[0] + 2^16a[1]` which
/// have been range checked separately. This verifies that the auxiliary variables lie in {0, 1, 2} and satisfy
///
/// `2^32 aux[0] + a = b + c + d mod P`
///
/// `2^16 aux[1] + a[0] = b[0] + c[0] + d[0] mod P`
#[inline]
pub fn triple_add<AB: AirBuilder>(
    builder: &mut AB,
    a: &[<AB as AirBuilder>::Var; 2],
    b: &[<AB as AirBuilder>::Var; 2],
    c: &[<AB as AirBuilder>::Expr; 2],
    d: &[<AB as AirBuilder>::Expr; 2],
    aux: &[<AB as AirBuilder>::Var; 2],
) {
    // We add four constraints:
    //
    // aux[0] * (aux[0] - 1) * (aux[0] - 2) = 0
    // aux[1] * (aux[1] - 1) * (aux[1] - 2) = 0
    // 2^32 aux[0] + a - b - c - d = 0
    // 2^16 aux[1] + a[0] - b[0] - c[0] - d[0] = 0
    //
    // The first two check that both auxiliary variables lie in {0, 1, 2}.
    // The third equation checks that the sum `2^32 aux[0] + a = b + c + d` holds mod P.
    // For the final equation, note that overflow is impossible and `x[0] = x mod 2^16`.
    // Thus if the final equation holds then `2^32 aux[0] + a = b + c + d mod 2^{16}`
    //
    // Using the chinese remainder theorem we conclude `a = b + c + d mod 2^32`

    builder.assert_tern(aux[0]);
    builder.assert_tern(aux[1]);

    // TODO: Ideally these should eventually be saved as constants. Or at the very least
    // there should be a quicker method of generating them than exp_u64.
    let two_16 = <AB as AirBuilder>::Expr::TWO.exp_u64(16);
    let two_32 = <AB as AirBuilder>::Expr::TWO.exp_u64(32);

    let sum_16 = a[0] - b[0] - c[0].clone() - d[0].clone();
    let sum_32 = a[1] - b[1] - c[1].clone() - d[1].clone();

    builder.assert_zero(two_32 * aux[0] + sum_16.clone() + two_16.clone() * sum_32);
    builder.assert_zero(two_16 * aux[1] + sum_16);
}

/// Verify that `a = b + c mod 2^32`
///
/// We assume that a, b, c are all given as `2 16` bit limbs (e.g. `a = a[0] + 2^16a[1]` which
/// have been range checked separately. This verifies that the auxiliary variables lie in {0, 1} and satisfy
///
/// `2^32 aux[0] + a = b + c mod P`
///
/// `2^16 aux[1] + a[0] = b[0] + c[0] mod P`
#[inline]
pub fn double_add<AB: AirBuilder>(
    builder: &mut AB,
    a: &[<AB as AirBuilder>::Var; 2],
    b: &[<AB as AirBuilder>::Var; 2],
    c: &[<AB as AirBuilder>::Expr; 2],
    aux: &[<AB as AirBuilder>::Var; 2],
) {
    // We add four constraints:
    //
    // aux[0] * (aux[0] - 1) = 0
    // aux[1] * (aux[1] - 1) = 0
    // 2^32 aux[0] + a - b - c = 0
    // 2^16 aux[1] + a[0] - b[0] - c[0] = 0
    //
    // The first two check that both auxiliary variables lie in {0, 1}.
    // The third equation checks that the sum `2^32 aux[0] + a = b + c` holds mod P.
    // For the final equation, note that overflow is impossible and `x[0] = x mod 2^16`.
    // Thus if the final equation holds then `2^32 aux[0] + a = b + c mod 2^{16}`
    //
    // Using the chinese remainder theorem we conclude `a = b + c mod 2^32`

    builder.assert_bool(aux[0]);
    builder.assert_bool(aux[1]);

    // TODO: Ideally these should eventually be saved as constants. Or at the very least
    // there should be a quicker method of generating them than exp_u64.
    let two_16 = <AB as AirBuilder>::Expr::TWO.exp_u64(16);
    let two_32 = <AB as AirBuilder>::Expr::TWO.exp_u64(32);

    let sum_16 = a[0] - b[0] - c[0].clone();
    let sum_32 = a[1] - b[1] - c[1].clone();

    builder.assert_zero(two_32 * aux[0] + sum_16.clone() + two_16.clone() * sum_32);
    builder.assert_zero(two_16 * aux[1] + sum_16);
}

/// Verify that `a = (b ^ (c << shift))`
///
/// We assume that a is given as `2 16` bit limbs and both b and c are unpacked into 32 individual bits.
/// We assume that the bits of b have been range checked but not the inputs in c or a. Both of these are
/// range checked as part of this function.
#[inline]
pub fn xor_32_shift<AB: AirBuilder>(
    builder: &mut AB,
    a: &[<AB as AirBuilder>::Var; 2],
    b: &[<AB as AirBuilder>::Var; 32],
    c: &[<AB as AirBuilder>::Var; 32],
    shift: usize,
) {
    // First we range check all elements of c.
    c.iter().for_each(|&elem| builder.assert_bool(elem));

    // Next we compute (b ^ (c << shift)) and pack the result into two 16-bit integers.
    let xor_shift_c_0_16 = b[..16]
        .iter()
        .enumerate()
        .map(|(i, elem)| xor((*elem).into(), c[(32 + i - shift) % 32].into()));
    let sum_0_16: <AB as AirBuilder>::Expr = pack_bits_le(xor_shift_c_0_16);

    let xor_shift_c_16_32 = b[16..]
        .iter()
        .enumerate()
        .map(|(i, elem)| xor((*elem).into(), c[(32 + (i + 16) - shift) % 32].into()));
    let sum_16_32: <AB as AirBuilder>::Expr = pack_bits_le(xor_shift_c_16_32);

    // As both b and c have been range checked to be boolean, all the (b ^ (c << shift))
    // are also boolean and so this final check additionally has the effect of range checking a[0], a[1].
    builder.assert_eq(a[0], sum_0_16);
    builder.assert_eq(a[1], sum_16_32);
}
