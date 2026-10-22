//! The SHA-2 hash functions of security category 1 (FIPS 205, Section 11.2.1).

use sha2::{Digest, Sha256};

use crate::address::Address;
use crate::params::{M, N};

/// Bytes of a SHA-256 block.
pub(crate) const BLOCK_BYTES: usize = 64;

/// Bytes of a SHA-256 digest.
pub(crate) const DIGEST_BYTES: usize = 32;

/// The tweakable hashes `F`, `H` and `T_l` of one public seed.
///
/// Each hashes `PK.seed || toByte(0, 64 - n)` first, a whole block whose compression is shared.
#[derive(Clone, Debug)]
pub(crate) struct Hashes {
    seeded: Sha256,
}

impl Hashes {
    /// The hashes keyed by `seed`, with the seed block absorbed once.
    pub(crate) fn new(seed: &[u8; N]) -> Self {
        let mut seeded = Sha256::new();
        seeded.update(seed);
        seeded.update([0; BLOCK_BYTES - N]);
        Self { seeded }
    }

    /// `Trunc_n(SHA-256(PK.seed || toByte(0, 64 - n) || ADRSc || M))`, with `M` the concatenated parts.
    pub(crate) fn thash(&self, address: &Address, parts: &[&[u8]]) -> [u8; N] {
        let mut hasher = self.seeded.clone();
        hasher.update(address.compressed());
        for part in parts {
            hasher.update(part);
        }
        hasher.finalize()[..N]
            .try_into()
            .expect("a SHA-256 digest is longer than n")
    }
}

/// `H_msg = MGF1-SHA-256(R || PK.seed || SHA-256(R || PK.seed || PK.root || M), m)`, with `M` the concatenated parts.
pub(crate) fn h_msg(r: &[u8; N], seed: &[u8; N], root: &[u8; N], parts: &[&[u8]]) -> [u8; M] {
    let mut inner = Sha256::new();
    inner.update(r);
    inner.update(seed);
    inner.update(root);
    for part in parts {
        inner.update(part);
    }
    let inner = inner.finalize();

    // MGF1 (RFC 8017, Appendix B.2.1): one block per 32-bit big-endian counter.
    let mut out = [0; M];
    for (counter, chunk) in out.chunks_mut(DIGEST_BYTES).enumerate() {
        let mut block = Sha256::new();
        block.update(r);
        block.update(seed);
        block.update(inner);
        block.update((counter as u32).to_be_bytes());
        chunk.copy_from_slice(&block.finalize()[..chunk.len()]);
    }
    out
}
