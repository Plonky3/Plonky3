use alloc::vec::Vec;

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

/// The BN254 prime represented as a little-endian array of 2-u128s.
///
/// Equal to: `21888242871839275222246405745257275088548364400416034343698204186575808495617`
const BN254_PRIME_U128: [u128; 2] = [
    0x2833e84879b9709143e1f593f0000001,
    0x30644e72e131a029b85045b68181585d,
];

/// Efficiently halve a Bn254 element.
#[inline]
pub(crate) fn halve_bn254(mut input: [u64; 4]) -> [u64; 4] {
    // Seems to be a little faster to convert into u128s.
    // It's essentially identical under the hood so this is
    // likely just helping the compiler generate simpler assembly somehow.
    let mut input0_u128 = (input[1] as u128) << 64 | (input[0] as u128);
    let mut input1_u128 = (input[3] as u128) << 64 | (input[2] as u128);
    let carry;

    if input0_u128 & 1 == 1 {
        // If the element is odd, we add P.
        (input0_u128, carry) = input0_u128.overflowing_add(BN254_PRIME_U128[0]);
        // Can ignore overflow here as the sum is < 2^256.
        input1_u128 = input1_u128.wrapping_add(BN254_PRIME_U128[1]);
        input1_u128 = input1_u128.wrapping_add(carry as u128);
    }

    // As this point the element is guaranteed to be even so we just
    // need to shift down by 1.
    let carry_bit = (input1_u128 << 63) as u64;

    input[0] = (input0_u128 >> 1) as u64;
    input[1] = (input0_u128 >> 65) as u64 | carry_bit;
    input[2] = (input1_u128 >> 1) as u64;
    input[3] = (input1_u128 >> 65) as u64;
    input
}

/// Find the size (number of bits) of val.
///
/// Return `num_bits / 64` and `num_bits % 64`
///
/// This assumes that val is actually a 256 bit number and we simply have the top 192 bits.
/// The bottom 64 bits are assumed to all be non zero so the "smallest" `num_bits` output this
/// will give is `num_bits = 64` or `(1, 0)`.
#[inline]
fn num_bits(val: [u64; 3]) -> (usize, usize) {
    let v3 = 64 - (val[2].leading_zeros() as i64);
    let v2 = 64 - (val[1].leading_zeros() as i64);
    let v1 = 64 - (val[0].leading_zeros() as i64);

    let v3_non_0 = (-v3) >> 63;
    let v2_non_0 = ((-v2) >> 63) & (!v3_non_0);
    let v1_non_0 = ((-v1) >> 63) & (!v3_non_0) & (!v2_non_0);

    let limb = (1 - v2_non_0 - (v3_non_0 << 1)) as usize;
    let bits_mod_64 = ((v3_non_0 & v3) | (v2_non_0 & v2) | (v1_non_0 & v1)) as usize;

    (limb, bits_mod_64)
}

/// Get the bottom `31` bits of val along with the top `33` bits starting from the `n`-th bit.
///
/// `n` is given by `limb = n / 64` and `bits_mod_64 = n % 64` for convenience.
#[inline]
fn get_approximation(val: [u64; 4], limb: usize, bits_mod_64: usize) -> u64 {
    const K: usize = 32;

    // Get the bottom K - 1 bits.
    let last_k_min_1 = val[0] & ((1 << (K - 1)) - 1);

    let joined_limb = ((val[limb] as u128) << 64) | (val[limb - 1] as u128);

    // 64 - bits_mod_64 = number of leading 0's. => Number of numbers is 64 + bits_mod_64
    // a >> b kills b numbers at the base. 64 + bits_mod_64 - (bits_mod_64 + 64 - (K + 1)) = K + 1
    let top_k_plus_1 = (joined_limb >> (bits_mod_64 + 64 - (K + 1))) as u64;

    (top_k_plus_1 << (K - 1)) | last_k_min_1
}

/// Negate a (in the 2's complement sense) if sign is `-1 = 2^64 - 1`
/// Leave a unchanged if sign is `0`.
///
/// Sign is assumed to be either `0` or `-1`.
#[inline]
fn conditional_neg(a: &mut [u64; 4], sign: u64) {
    let mut carry;
    (a[0], carry) = (a[0] ^ sign).overflowing_add((-(sign as i64)) as u64);
    (a[1], carry) = (a[1] ^ sign).overflowing_add(carry as u64);
    (a[2], carry) = (a[2] ^ sign).overflowing_add(carry as u64);
    (a[3], _) = (a[3] ^ sign).overflowing_add(carry as u64);
}

/// Compute the linear combination `af + bg` where `a, b` are `256-bit` positive integers
/// and `f, g` are 64-bit signed integers.
///
/// The result is a 320-bit signed integer represented as 4 64-bit limbs of positive integers
/// and an i64 for the highest limb.
#[inline]
fn linear_comb_signed(a: [u64; 4], b: [u64; 4], f: i64, g: i64) -> ([u64; 4], i64) {
    let mut output = [0_u64; 4];
    let mut carry = (a[0] as i128) * (f as i128) + (b[0] as i128) * (g as i128);
    output[0] = carry as u64;
    carry >>= 64;
    for i in 1..4 {
        carry += (a[i] as i128) * (f as i128) + (b[i] as i128) * (g as i128);
        output[i] = carry as u64;
        carry >>= 64;
    }

    (output, carry as i64)
}

/// Compute the linear combination `af + bg` where `a, b` are `256-bit` positive integers
/// and `f, g` are `64-bit` positive integers.
#[inline]
fn linear_comb_unsigned(a: [u64; 4], b: [u64; 4], f: u64, g: u64) -> [u64; 5] {
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

/// Compute the linear combination `(af + bg)/2^k` where `a, b` are `256-bit` integers and `f, g` are `64-bit` integers.
/// The division is assumed to be exact and the result is assumed to fit in a `256-bit` integer.
///
/// If the output would be negative, it is negated using 2's complement. A i64 is returned indicating
/// if the negation was applied. The i64 is `-1` if the output was negated `0` otherwise.
#[inline]
fn linear_comb_div(a: [u64; 4], b: [u64; 4], f: i64, g: i64, k: usize) -> ([u64; 4], i64) {
    let (product, hi_limb) = linear_comb_signed(a, b, f, g);

    let mut output = [0_u64; 4];

    // Now we apply the division by 2^k.
    output[0] = (product[0] >> k) | (product[1] << (64 - k));
    output[1] = (product[1] >> k) | (product[2] << (64 - k));
    output[2] = (product[2] >> k) | (product[3] << (64 - k));
    output[3] = (product[3] >> k) | ((hi_limb << (64 - k)) as u64);

    // Finally, if the result is negative, we negate it.
    let sign = hi_limb >> 63;
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

    // If we need to invert, we can easily invert a, b by subtracting from P.
    let (a_sub, _) = wrapping_sub(BN254_PRIME, a);
    let a_signed = if s_f == -1 { a_sub } else { a };
    let (b_sub, _) = wrapping_sub(BN254_PRIME, b);
    let b_signed = if s_g == -1 { b_sub } else { b };

    let product = linear_comb_unsigned(a_signed, b_signed, abs_f, abs_g);
    interleaved_monty_reduction(product[0], product[1..].try_into().unwrap())
}

/// An adjustment factor equal to `2^{1030} mod P`
pub(crate) const BN254_2_POW_1030: [u64; 4] = [
    0x1f7ca21e7fcb111b,
    0x61a09399fcfe8a6c,
    0x1438cc5aab55aedb,
    0x020c9ba0aeb6b6c7,
];

/// Invert a value in the BN254 field using a GCD based inversion algorithm.
///
/// The following approach to a GCD based inversion algorithm is taken from the paper:
/// Optimized Binary GCD for Modular Inversion by Thomas Pornin.
/// It can be found here: https://eprint.iacr.org/2020/972.pdf
//
/// Explicitly, we implement Algorithm 2, roughly following the C implementation linked in that paper (in section 3).
/// The algorithm is a variant of the Binary Extended Euclidean Algorithm which, allows for most of the iterations to
/// be performed using only u64's even in cases where the inputs are much larger.
//
/// This implementation is also impervious to side-channel attacks as an added bonus. In principal we could make
/// the average case a little faster if we didn't care about this property but the worst case would be unchanged and
/// potentially even slightly worse.
pub(crate) fn gcd_inversion(val: [u64; 4]) -> [u64; 4] {
    // When u = 1 and v 0, this outputs a^{-1} mod P.
    // More generally, when u = K, this output a^{-1}K mod P. Hence if we can choose the right
    // initial value for u, we can save some ourselves some work.
    // We start with aR so this will output (aR)^{-1} whereas we want a^{-1}R. Hence we need to multiply
    // by R^2 = 2^{512}
    // We do 31 * 15 + 41 = 506 iterations. Each iteration injects a power of 2 so we need to divide by 2^{506}.
    // Each pair of calls to linear_comb_monty_red, involves dividing by `2^64`. Hence to correct for all these
    // we need to multiply by `2^{16 * 64}`.
    // Overall we want `u` to equal `2^{16 * 64 + 6} = 2^{1030} mod P`
    let (mut a, mut u, mut b, mut v) = (val, BN254_2_POW_1030, BN254_PRIME, [0, 0, 0, 0]);

    // We need 506 iterations as in each iteration, all we can guarantee is that len(a) + len(b) will decrease by 1.
    // Initially, `len(a) + len(b) <= 2 * 254 = 508` so we need 506 iterations to get to the point that the sum of the
    // lengths is 2. Once it is 2, we know both `a` and `b` must be `1` or `0` as neither can be `0` without the other being
    // `1` due to the fact that the GCD is `1`.
    // We split the iterations into 15 initial rounds of size 31 and a final round of size 41.
    const ROUND_SIZE: usize = 31; // If you want to change round size, you will also need to modify the constant in rm_middle.
    const FINAL_ROUND_SIZE: usize = 41;
    for _ in 0..15 {
        // Find the a and b approximations for this set of inner rounds.
        // If both a and b now fit in a u64, return those. Otherwise take the bottom
        // 31 bits and the top 33 bits and assemble into a u64.
        let (limb, bits_mod_64) = num_bits([a[1] | b[1], a[2] | b[2], a[3] | b[3]]);
        let a_tilde = get_approximation(a, limb, bits_mod_64);
        let b_tilde = get_approximation(b, limb, bits_mod_64);

        // The key idea for the algorithm is that, instead of doing the standard
        // binary GCD algorithm on the big-ints a, b we can do the GCD algorithm
        // on a_tilde and b_tilde and then apply the updates all at once to a, b
        // and u, v.
        // By construction, a_tilde, b_tilde will function as good enough approximations
        // of a and b for at least 31 inner rounds.

        // Do the inner GCD loop on a_tilde and b_tilde to get the adjustment
        // factors for this round.
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

    // a and b are now guaranteed to fit in a u64 so we can just use the inner loop
    // for the remaining layers.
    let (_, _, f1, g1) = gcd_inner::<FINAL_ROUND_SIZE>(a[0], b[0]);

    // We can now compute the final result:
    linear_comb_monty_red(u, v, f1, g1)
}

/// Inner loop of the GCD algorithm.
///
/// This is basically a mini GCD which builds up a transformation to apply to the larger
/// numbers in the main loop. The key point is that this small loop only uses u64s and
/// does not require any BigNum multiplications.
///
/// The bottom `NUM_ROUNDS` bits of `a` and `b` should match the bottom `NUM_ROUNDS` bits of
/// the corresponding big-ints and the top `NUM_ROUNDS + 2` should match the top bits including
/// zeroes if the original numbers have different sizes.
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
    // Hence provided NUM_ROUNDS <= 62, we will not get any overflow.
    (f0, g0, f1, g1)
}
