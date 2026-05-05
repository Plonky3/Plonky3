//! Fiat-Shamir transcript helpers.
//!
//! WARP follows the BCS [BMNW25b] compilation pattern: a single duplex-sponge
//! [`FieldChallenger`] state that the protocol "observes" prover messages and
//! "samples" verifier challenges from. Cross-protocol soundness is provided
//! by binding the protocol description into the challenger before any
//! interaction.
//!
//! This module is intentionally minimal:
//!
//! - [`bind_protocol`] absorbs the protocol description and config bytes
//!   (encoded as base-field elements) into the challenger. Both prover and
//!   verifier call it before any other transcript operation. Parameter
//!   mismatch between the two sides causes immediate challenge divergence
//!   and verification failure.
//! - [`sample_indices`] draws `t` shift-query indices in `[0, n)` from the
//!   challenger using `log_2 n` bits per draw.

use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger};
use p3_field::Field;

/// Absorb the protocol description bytes into the Fiat-Shamir challenger.
///
/// Each input byte is lifted to a base-field element via
/// [`Field::from_u8`](p3_field::PrimeCharacteristicRing::from_u8) and observed
/// individually. Both prover and verifier must call this with identical
/// inputs before any other transcript operation.
///
/// The fields absorbed are:
///
/// - `description`: the [`BundledPesat::description`](crate::relation::BundledPesat::description)
///   bytes for this PESAT instance.
/// - `(num_fresh, num_acc)`: the count of fresh instances `ℓ_1` and prior
///   accumulators `ℓ_2`.
/// - `(s, t)`: §7.2 OOD-sample count and shift-query count.
/// - `(log_n, log_h)`: codeword-length and trace-height logs.
pub fn bind_protocol<F, Ch>(
    challenger: &mut Ch,
    description: &[u8],
    num_fresh: usize,
    num_acc: usize,
    s: usize,
    t: usize,
    log_n: usize,
    log_h: usize,
) where
    F: Field,
    Ch: CanObserve<F>,
{
    for &b in b"p3-warp-v1" {
        challenger.observe(F::from_u8(b));
    }
    for b in (description.len() as u64).to_le_bytes() {
        challenger.observe(F::from_u8(b));
    }
    for &b in description {
        challenger.observe(F::from_u8(b));
    }
    for &v in &[num_fresh, num_acc, s, t, log_n, log_h] {
        let bytes = (v as u64).to_le_bytes();
        for b in bytes {
            challenger.observe(F::from_u8(b));
        }
    }
}

/// Sample `count` shift-query indices in `[0, 1 << log_domain_size)`.
///
/// Each index consumes `log_domain_size` bits from the challenger. Returns
/// the indices in the order sampled (no deduplication).
///
/// # Panics
///
/// - `log_domain_size` must be ≤ `usize::BITS`.
pub fn sample_indices<F, Ch>(
    challenger: &mut Ch,
    log_domain_size: usize,
    count: usize,
) -> Vec<usize>
where
    F: Field,
    Ch: FieldChallenger<F>,
{
    assert!(log_domain_size < usize::BITS as usize);
    (0..count)
        .map(|_| challenger.sample_bits(log_domain_size))
        .collect()
}
