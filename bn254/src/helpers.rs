use alloc::vec::Vec;
use core::cmp::Ordering::{Equal, Greater, Less};

use num_bigint::BigUint;

use crate::{BN254_MONTY_MU_64, BN254_PRIME};

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

/// Halve
#[inline]
pub(crate) fn halve_bn254(mut input: [u64; 4]) -> [u64; 4] {
    if input[0] & 1 == 1 {
        (input, _) = wrapping_add(input, BN254_PRIME);
    }
    halve_even(input)
}

#[inline]
pub(crate) fn halve_even(mut input: [u64; 4]) -> [u64; 4] {
    let bot_bit_1 = input[1] << 63;
    let bot_bit_2 = input[2] << 63;
    let bot_bit_3 = input[3] << 63;

    input[0] = (input[0] >> 1) | bot_bit_1;
    input[1] = (input[1] >> 1) | bot_bit_2;
    input[2] = (input[2] >> 1) | bot_bit_3;
    input[3] >>= 1;
    input
}

// The following approach to a GCD based inversion algorithm is taken from here: https://eprint.iacr.org/2020/972.pdf
// Explicitly, we implement Algorithm 2, roughly following the C implementation linked in that paper (in section 3).
// The algorithm is a variant of the Binary Extended Euclidean Algorithm which, allows for most of the iterations to
// be performed using only u64's even in cases where the inputs are much larger.

#[inline]
fn num_bits(val: [u64; 4]) -> usize {
    for i in (0..4).rev() {
        if val[i] != 0 {
            return 64 * (i + 1) - val[i].leading_zeros() as usize;
        }
    }
    // If we have gotten to this point, the value is 0.
    0
}

#[inline]
fn rm_middle<const K: usize>(val: [u64; 4], n: usize) -> u64 {
    // Get the bottom K bits.
    let last_k = val[0] & ((1 << K) - 1);

    // Get the k+2 bits n to n - k - 1 inclusive where the bits are numbered with the least significant bit being 1.
    let n_limb = (n - 1) / 64; // Which limb is the n-th bit in.
    let n_remainder = (n - 1) % 64; // How far into the limb is the n-th bit.

    // Get the top k + 2 bits starting from the n-th bit shifted into bits k -> 2k + 2.
    let first_k_plus_2 = match core::cmp::Ord::cmp(&n_remainder, &(K + 1)) {
        Equal | Greater => {
            // This is the easiest case as all K + 2 bits are in the n'th_limb
            // We also already know that all bits above the n-th bit are 0.
            let shift = n_remainder - (K + 1);
            (val[n_limb] >> shift) << K
        }
        Less => {
            // In this case we need to get some bits from the next limb as well.
            let num_extra_bits = (K + 1) - n_remainder;
            let next_limb_bits = val[n_limb - 1] >> (64 - num_extra_bits);
            ((val[n_limb] << num_extra_bits) | next_limb_bits) << K
        }
    };
    first_k_plus_2 | last_k
}

/// Negate a (in the 2's complement sense) if sign is `-1 = 2^64 - 1`
/// Leave a unchanged if sign is `0`.
///
/// Sign is assumed ot be either `0` or `-1`.
#[inline]
fn conditional_neg(a: &mut [u64; 4], sign: u64) {
    let mut carry;
    (a[0], carry) = (a[0] ^ sign).overflowing_add((-(sign as i64)) as u64);
    (a[1], carry) = (a[1] ^ sign).overflowing_add(carry as u64);
    (a[2], carry) = (a[2] ^ sign).overflowing_add(carry as u64);
    (a[3], _) = (a[3] ^ sign).overflowing_add(carry as u64);
}

// Want to map x -> -x = 1 + !x

#[inline]
fn linear_comb(a: [u64; 4], b: [u64; 4], f: u64, g: u64) -> [u64; 5] {
    let mut output = [0_u64; 5];
    let mut carry = (a[0] as u128) * (f as u128) + (b[0] as u128) * (g as u128);
    output[0] = carry as u64;
    carry >>= 64;
    for i in 1..4 {
        carry += (a[i] as u128) * (f as u128) + (b[i] as u128) * (g as u128);
        output[i] = carry as u64;
        carry >>= 64;
    }
    output[4] = carry as u64;

    output
}

#[inline]
fn linear_comb_div(mut a: [u64; 4], mut b: [u64; 4], f: i64, g: i64, k: usize) -> ([u64; 4], i64) {
    // Get the signs and absolute values of f and g
    let s_f = (f >> 63) as u64;
    let s_g = (g >> 63) as u64;
    let abs_f = f.unsigned_abs();
    let abs_g = g.unsigned_abs();

    // Apply the signs to a and b using 2's complement.
    // `-a = 2^{256} - a = (2^{256} - 1) - (a - 1)`
    // The larger subtraction is simply a NOT.
    conditional_neg(&mut a, s_f);
    conditional_neg(&mut b, s_g);

    let mut product = linear_comb(a, b, abs_f, abs_g);

    // Now we need to correct for the signs. If f < 0 then we computed -f * (2^{256} - a) instead of af.
    // Hence in this case, we need to add f * 2^{256}.
    let s_a = a[3] >> 63;
    let s_b = b[3] >> 63;
    // Similarly if g < 0 we need to add g * 2^{256}.
    // The good news is that this only effects output 4.
    product[4] = product[4].wrapping_add(((s_a as i64) * f + (s_b as i64) * g) as u64);

    let mut output = [0_u64; 4];

    assert_eq!(product[0] & ((1 << k) - 1), 0);
    // Next we need to apply the k shift:
    output[0] = (product[0] >> k) | (product[1] << (64 - k));
    output[1] = (product[1] >> k) | (product[2] << (64 - k));
    output[2] = (product[2] >> k) | (product[3] << (64 - k));
    output[3] = (product[3] >> k) | (product[4] << (64 - k));

    // Finally, if the result is negative, we negate it again.
    let sign = (product[4] as i64) >> 63;
    conditional_neg(&mut output, sign as u64);
    (output, sign)
}

#[inline]
fn linear_comb_monty_red(a: [u64; 4], b: [u64; 4], f: i64, g: i64) -> [u64; 4] {
    // Get the signs and absolute values of f and g
    let s_f = f >> 63;
    let s_g = g >> 63;
    let abs_f = f.unsigned_abs();
    let abs_g = g.unsigned_abs();

    // If we need to invert, do that by subtracting from P.
    let a_signed = if s_f == -1 {
        let (sub, _) = wrapping_sub(BN254_PRIME, a);
        sub
    } else {
        a
    };
    let b_signed = if s_g == -1 {
        let (sub, _) = wrapping_sub(BN254_PRIME, b);
        sub
    } else {
        b
    };

    let product = linear_comb(a_signed, b_signed, abs_f, abs_g);
    interleaved_monty_reduction(product[0], product[1..].try_into().unwrap())
}

/// TODO
pub(crate) const BN254_2_POW_1030: [u64; 4] = [
    0x1f7ca21e7fcb111b,
    0x61a09399fcfe8a6c,
    0x1438cc5aab55aedb,
    0x020c9ba0aeb6b6c7,
];

//

pub(crate) fn gcd_inversion(val: [u64; 4]) -> [u64; 4] {
    // When u = 1 and v 0, this outputs a^{-1} mod P.
    // More generally, when u = K, this output a^{-1}K mod P. Hence if we can choose the right
    // initial value for u, we can save some ourselves some work.
    // We start with aR so this will output (aR)^{-1} whereas we want a^{-1}R. Hence we need to multiply
    // by R^2 = 2^{512}
    // We do 31 * 15 + 41 = 506 iterations. Each iteration injects a power of 2 so we need to divide by 2^{506}.
    // Each pair of calls to linear_comb_monty_red, involves dividing by `2^64`. Hence to correct for all these
    // we need to multiply by `2^{16 * 64}`.
    // Overall it seems like we want `u` to be `2^{16 * 64 + 6} = 2^{1030} mod P`
    let (mut a, mut u, mut b, mut v) = (val, BN254_2_POW_1030, BN254_PRIME, [0, 0, 0, 0]);

    const ROUND_SIZE: usize = 31;
    const FINAL_ROUND_SIZE: usize = 41;
    for _ in 0..15 {
        let n = num_bits(a).max(num_bits(b)).max(2 * ROUND_SIZE + 2);
        let a_tilde = rm_middle::<ROUND_SIZE>(a, n);
        let b_tilde = rm_middle::<ROUND_SIZE>(b, n);

        // println!("a_3: {:0b}, b_3: {:0b}", a[3], b[3]);
        // println!("a_2: {:0b}, b_2: {:0b}", a[2], b[2]);
        // println!("a_1: {:0b}, b_1: {:0b}", a[1], b[1]);
        // println!("a_0: {:0b}, b_0: {:0b}", a[0], b[0]);
        // println!("a_tidle: {a_tilde:0b}, b_tidle: {b_tilde:0b}");

        let (mut f0, mut g0, mut f1, mut g1) = gcd_inner::<ROUND_SIZE>(a_tilde, b_tilde);

        // Update a and b
        let (new_a, sign) = linear_comb_div(a, b, f0, g0, ROUND_SIZE);
        // If sign = -1, need to flip f0 and g0, if sign = 0, do nothing.
        f0 = (f0 ^ sign).wrapping_sub(sign);
        g0 = (g0 ^ sign).wrapping_sub(sign);

        let (new_b, sign) = linear_comb_div(a, b, f1, g1, ROUND_SIZE);
        // If sign = -1, need to flip f1 and g1, if sign = 0, do nothing.
        f1 = (f1 ^ sign).wrapping_sub(sign);
        g1 = (g1 ^ sign).wrapping_sub(sign);

        // Update u and v
        let new_u = linear_comb_monty_red(u, v, f0, g0);
        let new_v = linear_comb_monty_red(u, v, f1, g1);

        a = new_a;
        b = new_b;
        u = new_u;
        v = new_v;
    }

    let (_, _, f1, g1) = gcd_inner::<FINAL_ROUND_SIZE>(a[0], b[0]);

    // let (new_a, _) = linear_comb_div(a, b, f0, g0, FINAL_ROUND_SIZE);
    // let (new_b, _) = linear_comb_div(a, b, f1, g1, FINAL_ROUND_SIZE);
    // assert_eq!(new_a, [1, 0, 0, 0]);
    // assert_eq!(new_b, [1, 0, 0, 0]);

    // let out1 = linear_comb_monty_red(u, v, f0, g0);
    linear_comb_monty_red(u, v, f1, g1)
}

/// Inner loop of the GCD algorithm.
#[inline]
fn gcd_inner<const NUM_ROUNDS: usize>(mut a: u64, mut b: u64) -> (i64, i64, i64, i64) {
    // Initialise update factors.
    // At the start of round 0: -1 <= f0, g0, f1, g1 <= 1
    let (mut f0, mut g0, mut f1, mut g1) = (1, 0, 0, 1);

    // If at the start of a round: -2^i <= f0, g0, f1, g1 <= 2^i
    // Then, at the end of the round: -2^{i + 1} <= f0, g0, f1, g1 <= 2^{i + 1}
    for _ in 0..NUM_ROUNDS {
        if a & 1 == 0 {
            a >>= 1;
        } else {
            if a < b {
                (a, b) = (b, a);
                (f0, f1) = (f1, f0);
                (g0, g1) = (g1, g0);
            }
            a -= b;
            a >>= 1;
            f0 -= f1;
            g0 -= g1;
        }
        f1 <<= 1;
        g1 <<= 1;
    }

    // -2^NUM_ROUNDS <= f0, g0, f1, g1 <= 2^NUM_ROUNDS
    // Hence provided NUM_ROUNDS <= 30, we will not get any overflow.
    (f0, g0, f1, g1)
}
