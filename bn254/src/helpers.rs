use alloc::vec::Vec;
// use core::cmp::Ordering::{Equal, Greater, Less};

use num_bigint::BigUint;
use p3_field::Field;

use crate::{BN254_MONTY_MU, BN254_MONTY_R_SQ, BN254_PRIME, Bn254};

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

/// Simple big-num widening multiplication.
#[inline]
fn widening_mul(lhs: [u64; 4], rhs: [u64; 4]) -> [u64; 8] {
    // TODO: This is the key component of the Montgomery multiplication algorithm so we should look into it
    // for optimizations in the future.
    // Could try a Karatsuba approach?
    let mut output = [0_u64; 8];
    let mut overflow;

    for i in 0..4 {
        let mut carry = 0_u128;
        for j in 0..4 {
            // prod_u128 <= (2^64 - 1)^2 <= 2^128 - 2^65 + 1
            let prod_u128 = lhs[i] as u128 * rhs[j] as u128;

            // carry < 2^64 so this sum is < 2^128 - 1.
            carry += prod_u128;

            // Get bottom 64 bits of carry and add into output accumulator.
            let lo = carry as u64;
            (output[i + j], overflow) = output[i + j].overflowing_add(lo);

            // Move top bits down. As carry < 2^128 - 1, after this reduction and
            // addition it is < 2^64 - 1.
            carry >>= 64;
            carry += overflow as u128;
        }
        // As i is increasing, `output[i + 4]` currently stores a 0.
        output[i + 4] = carry as u64;
    }
    output
}

/// Multiplication of big-nums mod `2^256`.
///
/// Lets us avoid multiplication we know will result in a multiple of `2^256`.
#[inline]
fn mul_mod_2_exp_256(lhs: [u64; 4], rhs: [u64; 4]) -> [u64; 4] {
    let mut output = [0_u64; 4];

    // As we are working mod `2^256`, we can simplify some of our computations and ignore some carries.
    let limb0 = (lhs[0] as u128) * (rhs[0] as u128);
    output[0] = limb0 as u64;

    // Note that the first add cannot overflow as (limb0 >> 64) < 2^64 and any product of u64's is
    // less than or equal to 2^128 - 2^65 + 1.
    let (limb1, carry) = (limb0 >> 64)
        .wrapping_add((lhs[0] as u128) * (rhs[1] as u128))
        .overflowing_add((lhs[1] as u128) * (rhs[0] as u128));
    output[1] = limb1 as u64;

    // Overflow does not matter for limb2 as the overflow is > 2^256.
    let limb2 = ((limb1 >> 64) + ((carry as u128) << 64))
        .wrapping_add((lhs[0] as u128) * (rhs[2] as u128))
        .wrapping_add((lhs[1] as u128) * (rhs[1] as u128))
        .wrapping_add((lhs[2] as u128) * (rhs[0] as u128));
    output[2] = limb2 as u64;

    // For limb3 we can work with everything as u64s.
    output[3] = ((limb2 >> 64) as u64)
        .wrapping_add(lhs[0].wrapping_mul(rhs[3]))
        .wrapping_add(lhs[1].wrapping_mul(rhs[2]))
        .wrapping_add(lhs[2].wrapping_mul(rhs[1]))
        .wrapping_add(lhs[3].wrapping_mul(rhs[0]));

    output
}

/// Montgomery multiplication and reduction algorithm for BN254.
///
/// Uses the montgomery constant `2^256` making division free as we can
/// simply ignore the bottom 4 u64s.
#[inline]
pub(crate) fn monty_mul(lhs: [u64; 4], rhs: [u64; 4]) -> [u64; 4] {
    // TODO: It's likely worth it to remove the 'prod' variable here
    // and instead have this function simply do the monty reduction.
    // This allows us to compute the product elsewhere which will be
    // cheaper in some cases.
    // There may also be a cleverer algorithm (interleaved Montgomery multiplication)
    // which lets us do four smaller monty reductions instead of one big one
    // and avoids all the widening multiplications.
    let prod = widening_mul(lhs, rhs);

    let prod_lo: [u64; 4] = prod[..4].try_into().unwrap();
    let prod_hi: [u64; 4] = prod[4..].try_into().unwrap();

    let t_lo = mul_mod_2_exp_256(prod_lo, BN254_MONTY_MU);

    // TODO: For u, we only actually need the top 4 u64s.
    // It may be possible to use a simpler multiplication
    // algorithm.
    let u = widening_mul(t_lo, BN254_PRIME);
    let u_hi: [u64; 4] = u[4..].try_into().unwrap();

    let (sub, over) = wrapping_sub(prod_hi, u_hi);
    if over {
        let (sub_corr, _) = wrapping_add(sub, BN254_PRIME);
        sub_corr
    } else {
        sub
    }
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

pub(crate) fn gcd_inversion_simple(val: Bn254) -> Bn254 {
    let mut a = val.value;
    let mut u = BN254_MONTY_R_SQ;
    let mut v = Bn254::new_monty([0, 0, 0, 0]);
    let mut b = BN254_PRIME;

    while !a.iter().all(|&x| x == 0) {
        // println!("{a}, {b}");
        if a[0] & 1 == 0 {
            a = halve_even(a);
            u = u.halve();
        } else {
            if a.iter().rev().cmp(b.iter().rev()) == core::cmp::Ordering::Less {
                (a, u, b, v) = (b, v, a, u)
            }
            let (sub, _) = wrapping_sub(a, b);
            a = halve_even(sub);
            u = (u - v).halve();
        }
    }
    v
}

// // The following approach to a GCD based inversion algorithm is taken from here: https://eprint.iacr.org/2020/972.pdf
// // 254 + 254 - 1 = 507 = 16 * 30 + 27 (Could also do 507 = 16 * 31 + 11)

// fn num_bits(val: [u64; 4]) -> usize {
//     for i in (0..4).rev() {
//         if val[i] != 0 {
//             return 64 * (i + 1) - val[i].leading_zeros() as usize;
//         }
//     }
//     // If we have gotten to this point, the value is 0.
//     0
// }

// fn rm_middle<const K: usize>(val: [u64; 4], n: usize) -> u64 {
//     // Get the bottom k-1 bits.
//     let last_k_min_1 = val[0] & ((1 << (K - 1)) - 1);

//     // Get the k+1 bits n to n - k inclusive where the bits are numbered with the least significant bit being 1.
//     let n_limb = (n - 1) / 64; // Which limb is the n-th bit in.
//     let n_remainder = (n - 1) % 64; // How far into the limb is the n-th bit.

//     // Get the top k + 1 bits starting from the n-th bit shifted into bits k - 1 -> 2k - 1.
//     let first_k_plus_1 = match core::cmp::Ord::cmp(&n_remainder, &K) {
//         Equal | Greater => {
//             // This is the easiest case as all K + 1 bits are in the n'th_limb
//             // We also already know that all bits above the n-th bit are 0.
//             let shift = n_remainder - K;
//             (val[n_limb] >> shift) << (K - 1)
//         }
//         Less => {
//             // In this case we need to get some bits from the next limb as well.
//             let num_extra_bits = K - n_remainder;
//             let next_limb_bits = val[n_limb - 1] >> (64 - num_extra_bits);
//             ((val[n_limb] << num_extra_bits) | next_limb_bits) << (K - 1)
//         }
//     };

//     first_k_plus_1 | last_k_min_1
// }

// fn gcd_inversion(val: [u64; 4]) -> [u64; 4] {
//     let (mut a, mut u, mut b, mut v) = (val, [1, 0, 0, 0], BN254_PRIME, [0, 0, 0, 0]);

//     const ROUND_SIZE: usize = 30;
//     const FINAL_ROUND_SIZE: usize = 27;
//     for _ in 0..16 {
//         let n = num_bits(a).max(num_bits(b)).max(2 * ROUND_SIZE);
//         let a_tilde = rm_middle::<ROUND_SIZE>(a, n);
//         let b_tilde = rm_middle::<ROUND_SIZE>(b, n);

//         let (f0, g0, f1, g1) = gcd_inner::<ROUND_SIZE>(a_tilde, b_tilde);
//         todo!()
//     }
//     v
// }

// /// Inner loop of the GCD algorithm.
// fn gcd_inner<const NUM_ROUNDS: usize>(mut a: u64, mut b: u64) -> (i32, i32, i32, i32) {
//     // Initialise update factors.
//     // At the start of round 0: -1 <= f0, g0, f1, g1 <= 1
//     let (mut f0, mut g0, mut f1, mut g1) = (1, 0, 0, 1);

//     // If at the start of a round: -2^i <= f0, g0, f1, g1 <= 2^i
//     // Then, at the end of the round: -2^{i + 1} <= f0, g0, f1, g1 <= 2^{i + 1}
//     for _ in 0..NUM_ROUNDS {
//         if a & 1 == 0 {
//             a >>= 1;
//         } else {
//             if a < b {
//                 core::mem::swap(&mut a, &mut b);
//                 core::mem::swap(&mut f0, &mut f1);
//                 core::mem::swap(&mut g0, &mut g1);
//             }
//             a -= b;
//             a <<= 1;
//             f0 -= f1;
//             g0 -= g1;
//         }
//         f1 <<= 1;
//         g1 <<= 1;
//     }

//     // -2^NUM_ROUNDS <= f0, g0, f1, g1 <= 2^NUM_ROUNDS
//     // Hence provided NUM_ROUNDS <= 30, we will not get any overflow.
//     (f0, g0, f1, g1)
// }
