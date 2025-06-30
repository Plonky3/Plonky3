use alloc::vec::Vec;

use num_bigint::BigUint;
use p3_field::Field;

use crate::{BN254_MONTY_MU_64, BN254_PRIME, BN254_PRIME_U128};

/// Convert a fixed-size array of u64s to a BigUint.
#[inline]
pub(crate) fn to_biguint<const N: usize>(value: [u64; N]) -> BigUint {
    let bytes: Vec<u8> = value.iter().flat_map(|x| x.to_le_bytes()).collect();
    BigUint::from_bytes_le(&bytes)
}

/// Basically copied the implementation here: https://doc.rust-lang.org/std/primitive.u32.html#method.carrying_add
///
/// Once this moves to standard rust (currently nightly) we can use that directly.
/// Tracking Issue is here: https://github.com/rust-lang/rust/issues/85532
#[inline]
const fn carrying_add(lhs: u64, rhs: u64, carry: bool) -> (u64, bool) {
    let (a, c1) = lhs.overflowing_add(rhs);
    let (b, c2) = a.overflowing_add(carry as u64);

    // Ideally LLVM would know this is disjoint without us telling them,
    // but it doesn't <https://github.com/llvm/llvm-project/issues/118162>
    // Just doing a standard or for now.
    (b, c1 | c2)
}

/// Compute `lhs + rhs`, returning a bool if overflow occurred.
#[inline]
pub(crate) fn wrapping_add<const N: usize>(lhs: [u64; N], rhs: [u64; N]) -> ([u64; N], bool) {
    let mut carry = false;
    let mut output = [0; N];

    for i in 0..N {
        (output[i], carry) = carrying_add(lhs[i], rhs[i], carry);
    }

    (output, carry)
}

/// Basically copied the implementation here: https://doc.rust-lang.org/std/primitive.u32.html#method.borrowing_sub
///
/// Once this moves to standard rust (currently nightly) we can use that directly.
/// Tracking Issue is here: https://github.com/rust-lang/rust/issues/85532
#[inline]
const fn borrowing_sub(lhs: u64, rhs: u64, borrow: bool) -> (u64, bool) {
    let (a, c1) = lhs.overflowing_sub(rhs);
    let (b, c2) = a.overflowing_sub(borrow as u64);

    // Ideally LLVM would know this is disjoint without us telling them,
    // but it doesn't <https://github.com/llvm/llvm-project/issues/118162>
    // Just doing a standard or for now.
    (b, c1 | c2)
}

/// Compute `lhs - rhs`, returning a bool if underflow occurred.
#[inline]
pub(crate) fn wrapping_sub<const N: usize>(lhs: [u64; N], rhs: [u64; N]) -> ([u64; N], bool) {
    let mut borrow = false;
    let mut output = [0; N];

    for i in 0..N {
        (output[i], borrow) = borrowing_sub(lhs[i], rhs[i], borrow);
    }

    (output, borrow)
}

/// Compute a * b with a in the range 0..2^256 and b in the range 0..2^64.
///
/// Returns the lowest output limb and the remaining limbs in a 4-limb array.
#[inline]
pub(crate) fn mul_small(lhs: [u64; 4], rhs: u64) -> (u64, [u64; 4]) {
    let mut output = [0u64; 4];
    let mut acc;

    // Process the first limb separately to get the lowest output limb.
    acc = (lhs[0] as u128) * (rhs as u128);
    let out_0 = acc as u64;

    // acc < 2^64
    acc >>= 64;

    // Process the remaining limbs.
    for i in 1..4 {
        // Product of u64's < 2^128 - 2^64 so this addition will not overflow.
        acc += (lhs[i] as u128) * (rhs as u128);
        output[i - 1] = acc as u64;

        // acc < 2^64
        acc >>= 64;
    }
    output[3] = acc as u64;

    (out_0, output)
}

/// Compute a * b + c with a, c in the range 0..2^256 and b in the range 0..2^64.
///
/// Returns the lowest output limb and the remaining limbs in a 4-limb array.
#[inline]
pub(crate) fn mul_small_and_acc(lhs: [u64; 4], rhs: u64, add: [u64; 4]) -> (u64, [u64; 4]) {
    let mut output = [0u64; 4];
    let mut acc;

    // Process the first limb separately to get the lowest output limb.
    acc = (lhs[0] as u128) * (rhs as u128) + (add[0] as u128);
    let out_0 = acc as u64;

    // acc < 2^64
    acc >>= 64;

    // Process the remaining limbs.
    for i in 1..4 {
        // Product of u64's < 2^128 - 2^64 so this addition will not overflow.
        acc += (lhs[i] as u128) * (rhs as u128) + (add[i] as u128);
        output[i - 1] = acc as u64;

        // acc < 2^64
        acc >>= 64;
    }
    output[3] = acc as u64;

    (out_0, output)
}

// Interleaved Montgomery multiplication:
//
// When working with Big-Nums where the base multiplication is expensive, we
// use a variant of montgomery multiplication which is more efficient. The idea
// is to interleave the multiplication and reduction steps which lets us
// avoid the need for Big-Num x Big-Num multiplications.
//
// Let P be our prime and `mu = P^{-1} mod 2^64`.
// The Interleaved Montgomery reduction (IMR) algorithm takes as input 320-bit number `x`
// and returns a 256-bit number equal to `2^{-64}x mod P`.
//
// 1. Define `t = x * mu mod 2^64`.
// 2. Define `u = t * P`.
// 3. Define `sub = (x - u) / 2^{64}`.
// 4. If `sub < 0`, return `sub + P` else return `sub`.
//
// The division in step 3 is exact as `u mod 2^64 = t * P mod 2^64 = x * mu * P mod 2^64 = t mod 2^64`.
// Additionally note that the output is `< P` if `sub < 0` and otherwise is `< x/2^{64}`. Hence if we assume
// that `x < 2^{64}P`, then the output is always `< P`.
//
// We will apply this algorithm 4 times, once for each limb of the input number.
// Given two inputs x, y, we compute `x * y * 2^{-256} mod P` as follows:
// 1. Compute `acc0 = x * y[0]` and `res0 = IMR(acc0)`.
// 2. Compute `acc1 = x * y[1] + res0` and `res1 = IMR(acc1)`.
// 3. Compute `acc2 = x * y[2] + res1` and `res2 = IMR(acc2)`.
// 4. Compute `acc3 = x * y[3] + res2` and `res3 = IMR(acc3)`.
// 5. Return `res3`.
//
// Assume that x < P. (We make no assumptions about y). Then `res0 < P` as either `sub < 0` or
// `res0 < x * y[0]/(2^{64})` and `y[0] < 2^64`.
//
// Now assume that `resi < P`. Then `res{i+1} < P` as either `sub < 0` or
// `res{i+1} < (x * y[i+1] + resi)/(2^{64}) < P * (y[i+1] + 1)/(2^{64}) < P`.
//
// Hence by induction we have `res3 < P`.

///
/// The incoming number is split into 5 64-bit limbs with the
/// first limb separated out as it will be treated differently.
#[inline]
fn interleaved_monty_reduction(acc0: u64, acc: [u64; 4]) -> [u64; 4] {
    let t = acc0.wrapping_mul(BN254_MONTY_MU_64);
    let (_, u) = mul_small(BN254_PRIME, t);

    let (sub, under) = wrapping_sub::<4>(acc, u);
    if under {
        let (sub_corr, _) = wrapping_add(sub, BN254_PRIME);
        sub_corr
    } else {
        sub
    }
}

/// Montgomery multiplication and reduction algorithm for BN254.
///
/// The algorithm assumes that `lhs < P` but puts no constraints on `rhs`.
///
/// The output is a 4-limb array representing the result of `lhs * rhs * 2^{-256} mod P`
/// guaranteed to be in the range `[0, P)`.
#[inline]
pub(crate) fn monty_mul(lhs: [u64; 4], rhs: [u64; 4]) -> [u64; 4] {
    // We need to ensure that `lhs < P` otherwise it's possible for the
    // algorithm to fail and produce a value which is too large.
    debug_assert!(lhs.iter().rev().cmp(BN254_PRIME.iter().rev()) == core::cmp::Ordering::Less);

    // Our accumulator starts at 0 so we start with mul_small
    let (acc0, acc) = mul_small(lhs, rhs[0]);
    let res0 = interleaved_monty_reduction(acc0, acc);

    // Then we repeat the above process for the remaining rhs limbs
    // including the previous result as an accumulator.
    let (acc0, acc) = mul_small_and_acc(lhs, rhs[1], res0);
    let res1 = interleaved_monty_reduction(acc0, acc);
    let (acc0, acc) = mul_small_and_acc(lhs, rhs[2], res1);
    let res2 = interleaved_monty_reduction(acc0, acc);
    let (acc0, acc) = mul_small_and_acc(lhs, rhs[3], res2);
    interleaved_monty_reduction(acc0, acc)
}

#[inline(always)]
const fn carrying_add_128(x: u128, y: u128, carry: bool) -> (u128, bool) {
    let (a, b) = x.overflowing_add(y);
    let (c, d) = a.overflowing_add(carry as u128);
    (c, b || d)
}

#[inline(always)]
const fn wrapping_add_128_no_overflow(x: u128, y: u128, carry: bool) -> u128 {
    let a = x.wrapping_add(y);
    a.wrapping_add(carry as u128)
}

/// Compute `lhs + rhs`, returning a bool if overflow occurred.
#[inline(always)]
pub(crate) fn wrapping_add_u128(lhs: [u128; 2], rhs: [u128; 2]) -> ([u128; 2], bool) {
    let mut carry = false;
    let mut output = [0; 2];

    (output[0], carry) = carrying_add_128(lhs[0], rhs[0], carry);
    output[1] = wrapping_add_128_no_overflow(lhs[1], rhs[1], carry);

    (output, carry)
}

#[inline]
pub(crate) fn halve_bn254(mut input: [u64; 4]) -> [u64; 4] {
    // if input[0] & 1 == 1 {
    //     (input, _) = wrapping_add(input, BN254_PRIME);
    // }
    // let bot_bit_1 = input[1] << 63;
    // let bot_bit_2 = input[2] << 63;
    // let bot_bit_3 = input[3] << 63;

    // input[0] = (input[0] >> 1) | bot_bit_1;
    // input[1] = (input[1] >> 1) | bot_bit_2;
    // input[2] = (input[2] >> 1) | bot_bit_3;
    // input[3] >>= 1;
    // input

    let mut input0_u128 = (input[1] as u128) << 64 | (input[0] as u128);
    let mut input1_u128 = (input[3] as u128) << 64 | (input[2] as u128);
    if input0_u128 & 1 == 1 {
        ([input0_u128, input1_u128], _) =
            wrapping_add_u128([input0_u128, input1_u128], BN254_PRIME_U128);
    }
    let bot_bit_1 = (input1_u128 << 63) as u64;

    input[0] = (input0_u128 >> 1) as u64;
    input[1] = (input0_u128 >> 65) as u64 | bot_bit_1;
    input[2] = (input1_u128 >> 1) as u64;
    input[3] = (input1_u128 >> 65) as u64;
    input
}

/// Compute `base^{2^num_sq} * mul`
#[inline]
fn sq_and_mul<F: Field>(base: F, num_sq: usize, mul: F) -> F {
    base.exp_power_of_2(num_sq) * mul
}

/// Invert an element in the BN254 field using addition chain exponentiation.
///
/// Explicitly this function computes the exponential map:
/// `x -> x^21888242871839275222246405745257275088548364400416034343698204186575808495615`.
#[inline]
pub(crate) fn exp_bn_inv<F: Field>(val: F) -> F {
    // Note the binary expansion: 21888242871839275222246405745257275088548364400416034343698204186575808495615
    //  = 1100000110010001001110011100101110000100110001101000000010100110111000010100000100010110110110100000
    //       0110000001010110000101110100101000001100111110100001001000011110011011100101110000100100010100001
    //       111100001111101011001001111101111111111111111111111111111.
    // This uses 251 Squares + 56 Multiplications => 307 Operations total.
    // It is likely that this could be improved through further effort.

    // The basic idea we follow here is to create some simple binary building blocks to save on multiplications.
    let p1 = val;
    let p10 = p1.square();
    let p11 = p10 * p1;
    let p101 = p11 * p10;
    let p111 = p101 * p10;
    let p1111 = sq_and_mul(p111, 1, p1);
    let p11111 = sq_and_mul(p1111, 1, p1);
    let p1111100 = p11111.exp_power_of_2(2);
    let p1111101 = p1111100 * p1;
    let p1111111 = p1111101 * p10;
    let p10000001 = p1111111 * p10;
    let mut output = sq_and_mul(p10000001, 1, p10000001);

    // output now agrees with the first 9 digits of the binary expansion. We just have to do the
    // remaining 245...

    output = sq_and_mul(output, 3, p1); // 001
    output = sq_and_mul(output, 4, p1); // 0001
    output = sq_and_mul(output, 5, p111); // 00111
    output = sq_and_mul(output, 5, p111); // 00111
    output = sq_and_mul(output, 3, p1); // 001
    output = sq_and_mul(output, 4, p111); // 0111
    output = sq_and_mul(output, 5, p1); // 00001
    output = sq_and_mul(output, 4, p11); // 0011
    output = sq_and_mul(output, 5, p11); // 00011
    output = sq_and_mul(output, 2, p1); // 01
    output = sq_and_mul(output, 10, p101); // 0000000101
    output = sq_and_mul(output, 4, p11); // 0011
    output = sq_and_mul(output, 4, p111); // 0111
    output = sq_and_mul(output, 7, p101); // 0000101
    output = sq_and_mul(output, 6, p1); // 000001
    output = sq_and_mul(output, 6, p101); // 000101
    output = sq_and_mul(output, 3, p101); // 101
    output = sq_and_mul(output, 3, p101); // 101
    output = sq_and_mul(output, 3, p101); // 101
    output = sq_and_mul(output, 8, p11); // 00000011
    output = sq_and_mul(output, 9, p101); // 000000101
    output = sq_and_mul(output, 3, p11); // 011
    output = sq_and_mul(output, 5, p1); // 00001
    output = sq_and_mul(output, 4, p111); // 0111
    output = sq_and_mul(output, 2, p1); // 01
    output = sq_and_mul(output, 5, p101); // 00101
    output = sq_and_mul(output, 7, p11); // 0000011
    output = sq_and_mul(output, 9, p1111101); // 001111101
    output = sq_and_mul(output, 5, p1); // 00001
    output = sq_and_mul(output, 3, p1); // 001
    output = sq_and_mul(output, 8, p1111); // 00001111
    output = sq_and_mul(output, 4, p11); // 0011
    output = sq_and_mul(output, 4, p111); // 0111
    output = sq_and_mul(output, 3, p1); // 001
    output = sq_and_mul(output, 4, p111); // 0111
    output = sq_and_mul(output, 5, p1); // 00001
    output = sq_and_mul(output, 3, p1); // 001
    output = sq_and_mul(output, 6, p101); // 000101
    output = sq_and_mul(output, 9, p11111); // 000011111
    output = sq_and_mul(output, 11, p1111101); // 00001111101
    output = sq_and_mul(output, 3, p11); // 011
    output = sq_and_mul(output, 3, p1); // 001
    output = sq_and_mul(output, 7, p11111); // 0011111
    output = sq_and_mul(output, 8, p1111111); // 01111111
    output = sq_and_mul(output, 7, p1111111); // 1111111
    output = sq_and_mul(output, 7, p1111111); // 1111111
    output = sq_and_mul(output, 7, p1111111); // 1111111
    output
}
