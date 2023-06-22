use gcd::Gcd;
use modinverse::modinverse;
use num::{BigUint, One};
use p3_field::PrimeField64;
use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256,
};

/// The binomial function from combinatorics, used to compute the number of rounds needed for soundness.
pub(crate) fn binomial(n: usize, k: usize) -> BigUint {
    let mut result = BigUint::one();
    for i in 0..k {
        result *= n - i;
        result /= i + 1;
    }

    result
}

/// Generate alpha, the smallest integer relatively prime to p − 1.
pub(crate) fn get_alpha<F: PrimeField64>() -> u64 {
    let p = F::ORDER_U64;

    (3..p).find(|&a| a.gcd(p - 1) == 1).unwrap()
}

/// Given alpha, find its multiplicative inverse in Z/⟨p − 1⟩.
pub(crate) fn get_inverse<F: PrimeField64>(alpha: u64) -> u64 {
    let p = F::ORDER_U64 as i64;
    modinverse(alpha as i64, p - 1).expect("x^alpha not a permutation").abs() as u64
}

/// Compute the SHAKE256 variant of SHA-3.
/// This is used to generate the round constants from a seed string.
pub(crate) fn shake256_hash(seed_bytes: &[u8], num_bytes: usize) -> Vec<u8> {
    let mut hasher = Shake256::default();
    hasher.update(seed_bytes);
    let mut reader = hasher.finalize_xof();
    let mut result = vec![0u8; num_bytes];
    reader.read(&mut result);
    result
}
