//! The parameters of one instance of the scheme.

use p3_field::PrimeField32;

use crate::Digest;
use crate::hash::DIGEST_LEN;

/// Digit bits taken from each digest element: its low 24 bits, nearly uniform since `2^24` divides `p - 1`.
pub(crate) const BITS_PER_ELEMENT: usize = 24;

/// Lifetime, Winternitz chunking and target sum of an instance.
///
/// [`Params::with_lifetime`] picks `w = 4`, 64 chains and target sum 96: about 123 bits of conjectured classical security.
///
/// That bound is the logarithm of the number of codewords, `4^64 / 22.4`, and every hash keeps 248-bit outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Params {
    log_lifetime: usize,
    chunk_bits: usize,
    num_chains: usize,
    target_sum: usize,
}

impl Params {
    /// Parameters for `2^log_lifetime` epochs, `num_chains` chains of base `2^chunk_bits`, and digits summing to `target_sum`.
    ///
    /// # Panics
    ///
    /// When `log_lifetime` is not in `1..=30`, `chunk_bits` not in `1..=4`, the digits overflow one digest, or the target is unreachable.
    pub const fn new(
        log_lifetime: usize,
        chunk_bits: usize,
        num_chains: usize,
        target_sum: usize,
    ) -> Self {
        assert!(
            1 <= log_lifetime && log_lifetime <= 30,
            "epochs fit in 30 bits"
        );
        assert!(
            1 <= chunk_bits && chunk_bits <= 4,
            "a chunk has 1 to 4 bits"
        );
        assert!(
            num_chains >= 1 && num_chains * chunk_bits <= DIGEST_LEN * BITS_PER_ELEMENT,
            "the digits fit in one digest"
        );
        assert!(
            target_sum <= num_chains * ((1 << chunk_bits) - 1),
            "the target sum is reachable"
        );
        Self {
            log_lifetime,
            chunk_bits,
            num_chains,
            target_sum,
        }
    }

    /// The recommended parameters for `2^log_lifetime` epochs: `w = 4`, 64 chains, target sum 96.
    ///
    /// # Panics
    ///
    /// When `log_lifetime` is not in `1..=30`.
    pub const fn with_lifetime(log_lifetime: usize) -> Self {
        Self::new(log_lifetime, 2, 64, 96)
    }

    /// The base-two logarithm of the number of epochs, which is the Merkle tree height.
    pub const fn log_lifetime(&self) -> usize {
        self.log_lifetime
    }

    /// The bits per digit.
    pub const fn chunk_bits(&self) -> usize {
        self.chunk_bits
    }

    /// The number of Winternitz chains, which is the number of digits.
    pub const fn num_chains(&self) -> usize {
        self.num_chains
    }

    /// The sum every encoded message's digits must reach.
    pub const fn target_sum(&self) -> usize {
        self.target_sum
    }

    /// The number of hash steps from a chain's start to its end, `2^chunk_bits - 1`.
    pub const fn chain_len(&self) -> usize {
        (1 << self.chunk_bits) - 1
    }

    /// The number of digest elements the digits come from.
    pub(crate) const fn digit_elements(&self) -> usize {
        (self.num_chains * self.chunk_bits).div_ceil(BITS_PER_ELEMENT)
    }

    /// The digits of a message digest, read from the low bits of each element, least significant first.
    pub(crate) fn digits(&self, digest: &Digest) -> impl Iterator<Item = usize> {
        let per_element = BITS_PER_ELEMENT / self.chunk_bits;
        let (bits, mask) = (self.chunk_bits, self.chain_len() as u32);
        let digest = *digest;
        (0..self.num_chains).map(move |i| {
            let element = digest[i / per_element].as_canonical_u32();
            ((element >> ((i % per_element) * bits)) & mask) as usize
        })
    }
}
