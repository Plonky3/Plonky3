//! IETF `Ns + 16` byte-to-field decoding helpers.
//!
//! Decodes a near-uniform field element from `byte_size + 16` bytes.
//! The 16 spares give `2^-128` statistical distance from uniform
//! independently of the prime (DSFS Lemma C.1, IETF §6).

use alloc::vec::Vec;

use num_bigint::BigUint;
use p3_field::PrimeField;

use crate::fs::error::TranscriptError;

/// Spare bytes beyond the minimal field encoding length.
///
/// 16 bytes (128 bits) is the IETF / DSFS budget for the bias bound.
pub const EXTRA_BYTES: usize = 16;

/// Bytes required for the IETF decoding pattern at field `F`:
///
/// `ceil(log2(p) / 8) + 16`.
pub(crate) fn required_bytes<F: PrimeField>() -> usize {
    field_byte_size::<F>() + EXTRA_BYTES
}

/// Bytes occupied by the canonical big-endian encoding of `F`:
///
/// `ceil(log2(p) / 8)`.
pub(crate) fn field_byte_size<F: PrimeField>() -> usize {
    F::bits().div_ceil(8)
}

/// Decode `bytes` into a near-uniform element of `F`.
///
/// Folds the big-endian prefix directly in `F` by Horner's rule:
///
/// ```text
///     acc = sum_i bytes[i] * 256^(need-1-i)   (mod p)
/// ```
///
/// Field arithmetic reduces mod `p` at every step.
/// No big-integer allocation or division is needed.
///
/// # Panics
///
/// When `bytes.len() < required_bytes::<F>()`.
pub(crate) fn decode_field_via_extra_bytes<F: PrimeField>(bytes: &[u8]) -> F {
    let need = required_bytes::<F>();
    assert!(
        bytes.len() >= need,
        "decode_field_via_extra_bytes needs at least {need} bytes for this field, got {}",
        bytes.len(),
    );
    // radix = 256: one byte is a single base-256 digit of a big-endian integer.
    let radix = F::from_u16(256);
    // Horner over the bytes, evaluated in F, most-significant byte first:
    //     acc <- acc * 256 + byte
    // Working in F means each step reduces mod p, so the fold yields `big mod p`.
    bytes[..need]
        .iter()
        .fold(F::ZERO, |acc, &b| acc * radix + F::from_u8(b))
}

/// Big-endian canonical encoding of `value`, zero-padded to `field_byte_size::<F>()`.
#[must_use]
pub(crate) fn encode_field_be<F: PrimeField>(value: &F) -> Vec<u8> {
    let target = field_byte_size::<F>();
    let raw = value.as_canonical_biguint().to_bytes_be();
    let mut out: Vec<u8> = Vec::with_capacity(target);
    if raw.len() < target {
        out.resize(target - raw.len(), 0);
    }
    out.extend_from_slice(&raw);
    out
}

/// Decode a fixed-length canonical big-endian byte slice into `F`.
///
/// # Errors
///
/// When the slice is too short or encodes an integer outside `[0, p)`.
pub(crate) fn decode_field_be_canonical<F: PrimeField>(bytes: &[u8]) -> Result<F, TranscriptError> {
    let need = field_byte_size::<F>();
    if bytes.len() < need {
        return Err(TranscriptError::BadProofShape {
            reason: "not enough bytes for a canonical field encoding",
        });
    }
    let big = BigUint::from_bytes_be(&bytes[..need]);
    // Reject non-canonical encodings: a malformed proof, not a near-uniform sample.
    if big >= F::order() {
        return Err(TranscriptError::BadProofShape {
            reason: "field encoding outside canonical range",
        });
    }
    Ok(biguint_to_field::<F>(&big))
}

/// Convert a `BigUint` known to lie in `[0, F::order())` into an `F`.
///
/// - Single-limb fields take the fast path;
/// - Bigger fields fold limbs via base-`2^64` Horner accumulation.
fn biguint_to_field<F: PrimeField>(value: &BigUint) -> F {
    let limbs: Vec<u64> = value.iter_u64_digits().collect();

    match limbs.as_slice() {
        [] => F::ZERO,
        // Fast path: any field with `order < 2^64`.
        &[lo] => F::from_u64(lo),
        more => {
            // radix = 2^64 in F, built as `(2^63) * 2` so the literal fits in `u64`.
            let radix = F::from_u64(1u64 << 63).double();
            more.iter()
                .rev()
                .fold(F::ZERO, |acc, &limb| acc * radix + F::from_u64(limb))
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::{Field, PrimeCharacteristicRing, PrimeField32};
    use p3_goldilocks::Goldilocks;
    use proptest::prelude::*;

    use super::*;

    /// Concrete field exercised in this module's tests.
    type F = BabyBear;

    #[test]
    fn required_bytes_matches_field_size_plus_sixteen() {
        // 31-bit prime -> 4 bytes. 64-bit prime -> 8 bytes. Plus 16 spares.
        assert_eq!(required_bytes::<F>(), 4 + 16);
        assert_eq!(required_bytes::<Goldilocks>(), 8 + 16);
    }

    #[test]
    fn decode_zero_input_yields_zero() {
        // Zero in, zero out.
        let bytes = vec![0u8; required_bytes::<F>()];
        let f: F = decode_field_via_extra_bytes(&bytes);
        assert_eq!(f.as_canonical_u32(), 0);
    }

    #[test]
    fn decode_one_input_yields_one() {
        // Single 1 in the low byte: 1 < p for every supported field.
        let mut bytes = vec![0u8; required_bytes::<F>()];
        let last = bytes.len() - 1;
        bytes[last] = 1;
        let f: F = decode_field_via_extra_bytes(&bytes);
        assert_eq!(f.as_canonical_u32(), 1);
    }

    #[test]
    fn decode_matches_manual_reduction_baby_bear() {
        // Decoding equals `from_int(bytes_be mod p)` by construction.
        let mut bytes = vec![0u8; 20];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i as u8 ^ 0xa5;
        }
        let from_codec: F = decode_field_via_extra_bytes(&bytes);

        let manual = {
            let big = BigUint::from_bytes_be(&bytes);
            let reduced = big % F::order();
            let limb = reduced.iter_u32_digits().next().unwrap_or(0);
            F::from_u32(limb)
        };
        assert_eq!(from_codec, manual);
    }

    /// BigUint reference for the Horner decode: read big-endian, reduce mod p, lift to F.
    fn decode_reference<G: PrimeField>(bytes: &[u8]) -> G {
        let need = required_bytes::<G>();
        let reduced = BigUint::from_bytes_be(&bytes[..need]) % G::order();
        biguint_to_field::<G>(&reduced)
    }

    proptest! {
        #[test]
        fn random_bytes_decode_to_canonical_baby_bear(seed in any::<[u8; 32]>()) {
            // The bias bound is proven analytically; this is a smoke check.
            let mut bytes = vec![0u8; required_bytes::<F>()];
            for (i, b) in bytes.iter_mut().enumerate() {
                *b = seed[i % seed.len()];
            }
            let f: F = decode_field_via_extra_bytes(&bytes);
            prop_assert!(f.as_canonical_u32() < F::ORDER_U32);
        }

        #[test]
        fn horner_decode_matches_biguint_reference(raw in any::<[u8; 36]>()) {
            // The field-Horner fast path must equal the BigUint reduce-and-lift reference.
            //
            // Cover a single-limb field (BabyBear) and a 64-bit field (Goldilocks).
            prop_assert_eq!(
                decode_field_via_extra_bytes::<F>(&raw[..required_bytes::<F>()]),
                decode_reference::<F>(&raw),
            );
            prop_assert_eq!(
                decode_field_via_extra_bytes::<Goldilocks>(&raw[..required_bytes::<Goldilocks>()]),
                decode_reference::<Goldilocks>(&raw),
            );
        }
    }
}
