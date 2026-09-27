//! Hashing many messages one at a time, on targets without a vector backend.

use crate::DIGEST_BYTES;

/// One message per call, since there is no vector unit to batch on.
pub(crate) const LANES: usize = 1;

/// Hash `out.len()` equal-length messages laid end to end in `input`.
///
/// # Panics
///
/// Panics if the input length is not a whole multiple of the digest count.
pub(crate) fn hash_many(input: &[u8], out: &mut [[u8; DIGEST_BYTES]]) {
    // No digests requested means there is nothing to read from the input.
    if out.is_empty() {
        return;
    }

    // Every message has the same length, so the split is exact by contract.
    assert!(
        input.len().is_multiple_of(out.len()),
        "input length ({}) must be a whole multiple of the digest count ({})",
        input.len(),
        out.len()
    );
    let len = input.len() / out.len();

    // Message i is the i-th run of that length.
    for (i, digest) in out.iter_mut().enumerate() {
        *digest = crate::Blake2s256::hash(&input[i * len..][..len]);
    }
}
