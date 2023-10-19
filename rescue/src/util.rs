use alloc::vec;
use alloc::vec::Vec;

use gcd::Gcd;
use modinverse::modinverse;
use p3_field::PrimeField64;
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake256;

/// Generate alpha, the smallest integer relatively prime to `p − 1`.
pub(crate) fn get_alpha<F: PrimeField64>() -> u64 {
    let p = F::ORDER_U64;

    (3..p).find(|&a| a.gcd(p - 1) == 1).unwrap()
}

/// Given alpha, find its multiplicative inverse in `Z/⟨p − 1⟩`.
pub(crate) fn get_inverse<F: PrimeField64>(alpha: u64) -> u64 {
    let p = F::ORDER_U64 as i128;
    modinverse(alpha as i128, p - 1)
        .expect("x^alpha not a permutation")
        .unsigned_abs()
        .try_into()
        .unwrap()
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
