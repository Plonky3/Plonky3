use alloc::vec::Vec;
use num_bigint::BigUint;
use p3_field::Field;

use crate::{BN254_MONTY_MU, BN254_PRIME};

/// Convert a fixed-size array of u64s to a BigUint.
pub(crate) fn to_biguint<const N: usize>(value: [u64; N]) -> BigUint {
    let bytes: Vec<u8> = value.iter().flat_map(|x| x.to_le_bytes()).collect();
    BigUint::from_bytes_le(&bytes)
}

/// Basically copied the implementation here: https://doc.rust-lang.org/std/primitive.u32.html#method.carrying_add
///
/// Once this moves to standard rust (currently nightly) we can use that directly.
/// Tracking Issue is here: https://github.com/rust-lang/rust/issues/85532
const fn carrying_add(lhs: u64, rhs: u64, carry: bool) -> (u64, bool) {
    let (a, c1) = lhs.overflowing_add(rhs);
    let (b, c2) = a.overflowing_add(carry as u64);

    // Ideally LLVM would know this is disjoint without us telling them,
    // but it doesn't <https://github.com/llvm/llvm-project/issues/118162>
    // Just doing a standard or for now.
    (b, c1 | c2)
}

// Compute `lhs + rhs`, returning a bool if overflow occurred.
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
const fn borrowing_sub(lhs: u64, rhs: u64, borrow: bool) -> (u64, bool) {
    let (a, c1) = lhs.overflowing_sub(rhs);
    let (b, c2) = a.overflowing_sub(borrow as u64);

    // Ideally LLVM would know this is disjoint without us telling them,
    // but it doesn't <https://github.com/llvm/llvm-project/issues/118162>
    // Just doing a standard or for now.
    (b, c1 | c2)
}

// Compute `lhs - rhs`, returning a bool if underflow occurred.
pub(crate) fn wrapping_sub<const N: usize>(lhs: [u64; N], rhs: [u64; N]) -> ([u64; N], bool) {
    let mut borrow = false;
    let mut output = [0; N];

    for i in 0..N {
        (output[i], borrow) = borrowing_sub(lhs[i], rhs[i], borrow);
    }

    (output, borrow)
}

/// Simple big-num widening multiplication.
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

/// Montgomery multiplication and reduction algorithm for BN254.
///
/// Uses the montgomery constant `2^256` making division free as we can
/// simply ignore the bottom 4 u64s.
pub(crate) fn monty_mul(lhs: [u64; 4], rhs: [u64; 4]) -> [u64; 4] {
    // TODO: It's likely worth it to remove the 'prod' variable here
    // and instead have this function simple do the monty reduction.
    // This allows us to compute the product elsewhere which will be
    // cheaper in some cases.
    let prod = widening_mul(lhs, rhs);

    let prod_lo: [u64; 4] = prod[..4].try_into().unwrap();
    let prod_hi: [u64; 4] = prod[4..].try_into().unwrap();
    let t = widening_mul(prod_lo, BN254_MONTY_MU);
    let t_lo: [u64; 4] = t[..4].try_into().unwrap();

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

/// Compute `base^{2^num_sq} * mul`
fn sq_and_mul<F: Field>(base: F, num_sq: usize, mul: F) -> F {
    base.exp_power_of_2(num_sq) * mul
}

/// Invert and element in the BN254 field using addition chain exponentiation.
///
/// Explicitly this function computes the exponential map:
/// `x -> x^21888242871839275222246405745257275088548364400416034343698204186575808495615`.
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
