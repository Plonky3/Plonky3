use alloc::vec;
use alloc::vec::Vec;

use sha3::Shake256;
use sha3::digest::{ExtendableOutput, Update, XofReader};

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
