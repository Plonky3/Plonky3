//! Field-element encoding shared by every codec in this module.
//!
//! # Overview
//!
//! Two directions, two rules.
//!
//! Sampling a challenge follows the IETF `Ns + 16` rule: read `byte_size + 16`
//! bytes and reduce them mod `p`.
//!
//! The 16 spare bytes bound the statistical distance from uniform by `2^-128`,
//! independently of the prime (DSFS Lemma C.1, IETF §6).
//!
//! Reading a prover message follows the canonical rule instead: exactly
//! `byte_size` big-endian bytes, and any integer `>= p` is rejected as
//! malformed input rather than reduced.
//!
//! Both directions stay allocation-free: every supported field has a modulus
//! below `2^64`, so the canonical integer fits in a `u64` and never needs a
//! big-integer detour.

use alloc::vec::Vec;

use p3_field::PrimeField64;

use crate::fs::error::TranscriptError;

/// Spare bytes beyond the minimal field encoding length.
///
/// 16 bytes (128 bits) is the IETF / DSFS budget for the bias bound.
pub const EXTRA_BYTES: usize = 16;

/// Bytes required for the IETF decoding pattern at field `F`:
///
/// `ceil(log2(p) / 8) + 16`.
pub(crate) fn required_bytes<F: PrimeField64>() -> usize {
    field_byte_size::<F>() + EXTRA_BYTES
}

/// Bytes occupied by the canonical big-endian encoding of `F`:
///
/// `ceil(log2(p) / 8)`.
pub(crate) fn field_byte_size<F: PrimeField64>() -> usize {
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
pub(crate) fn decode_field_via_extra_bytes<F: PrimeField64>(bytes: &[u8]) -> F {
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

/// Append the canonical big-endian encoding of `value` to `out`.
///
/// Writes exactly `field_byte_size::<F>()` bytes, left-padded with zeros.
pub(crate) fn encode_field_be<F: PrimeField64>(value: &F, out: &mut Vec<u8>) {
    let need = field_byte_size::<F>();
    // Canonical representative in `[0, p)` as a plain integer: no allocation.
    let canonical = value.as_canonical_u64().to_be_bytes();
    // `need <= 8` for every prime below `2^64`, so the tail holds the whole value.
    out.extend_from_slice(&canonical[8 - need..]);
}

/// Decode a canonical big-endian field encoding from the head of `bytes`.
///
/// # Errors
///
/// When the slice is too short or encodes an integer outside `[0, p)`.
pub(crate) fn decode_field_be_canonical<F: PrimeField64>(
    bytes: &[u8],
) -> Result<F, TranscriptError> {
    let need = field_byte_size::<F>();
    if bytes.len() < need {
        return Err(TranscriptError::BadProofShape {
            reason: "not enough bytes for a canonical field encoding",
        });
    }
    // Right-align the fixed-width prefix inside a full 8-byte big-endian buffer.
    let mut padded = [0u8; 8];
    padded[8 - need..].copy_from_slice(&bytes[..need]);
    let raw = u64::from_be_bytes(padded);
    // `from_canonical_checked` returns `None` on exactly the non-canonical inputs,
    // so the range check and the lift are one operation.
    F::from_canonical_checked(raw).ok_or(TranscriptError::BadProofShape {
        reason: "field encoding outside canonical range",
    })
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use num_bigint::BigUint;
    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, PrimeField32};
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
    fn ietf_extra_byte_decoding_matches_independent_vectors() {
        // Known-answer vectors for the `Ns + 16` rule of IETF §6.
        //
        // Each expected value is `int.from_bytes(input, "big") % p`, computed
        // outside this crate with arbitrary-precision integer arithmetic.
        //
        // The vectors therefore pin the in-field Horner fold against a
        // reference it did not produce.
        //
        // ```text
        //     BabyBear   consumes the first 4 + 16 = 20 bytes
        //     Goldilocks consumes the first 8 + 16 = 24 bytes
        // ```
        const VECTORS: [([u8; 24], u32, u64); 3] = [
            (
                [
                    0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98, 0x76,
                    0x54, 0x32, 0x10, 0x0f, 0x1e, 0x2d, 0x3c, 0x4b, 0x5a, 0x69, 0x78,
                ],
                497_719_269,
                18_060_362_441_347_088_952,
            ),
            (
                [
                    0x9e, 0x37, 0x79, 0xb9, 0x7f, 0x4a, 0x7c, 0x15, 0xf3, 0x9c, 0xc0, 0x60, 0x5c,
                    0xed, 0xc8, 0x34, 0xd1, 0x31, 0x0b, 0xa6, 0x98, 0xdf, 0xb5, 0xac,
                ],
                793_107_742,
                1_197_075_743_667_693_265,
            ),
            (
                [
                    0x25, 0x45, 0xf4, 0x91, 0x4f, 0x6c, 0xdd, 0x1d, 0xbf, 0x58, 0x47, 0x6d, 0x1c,
                    0xe4, 0xe5, 0xb9, 0xa4, 0x09, 0x38, 0x22, 0x29, 0x9f, 0x31, 0xd0,
                ],
                1_866_441_294,
                5_493_067_924_294_334_779,
            ),
        ];

        for (input, want_baby_bear, want_goldilocks) in VECTORS {
            // A 31-bit prime reads the 20-byte prefix.
            assert_eq!(
                decode_field_via_extra_bytes::<F>(&input[..required_bytes::<F>()])
                    .as_canonical_u32(),
                want_baby_bear,
            );
            // A 64-bit prime reads all 24 bytes.
            assert_eq!(
                decode_field_via_extra_bytes::<Goldilocks>(&input).as_canonical_u64(),
                want_goldilocks,
            );
        }
    }

    #[test]
    fn canonical_encoding_round_trips_at_the_boundaries() {
        // Invariant: encode then decode is the identity on `[0, p)`.
        //
        // Fixture state: the three interesting representatives 0, 1, p-1.
        for value in [F::ZERO, F::ONE, F::NEG_ONE] {
            let mut buf = Vec::new();
            encode_field_be(&value, &mut buf);
            assert_eq!(buf.len(), field_byte_size::<F>());
            assert_eq!(decode_field_be_canonical::<F>(&buf).unwrap(), value);
        }
    }

    #[test]
    fn canonical_decoding_rejects_values_at_or_above_the_modulus() {
        // Boundary: `p` itself is the smallest rejected encoding.
        let mut at_modulus = Vec::new();
        at_modulus.extend_from_slice(&F::ORDER_U32.to_be_bytes());
        assert_eq!(
            decode_field_be_canonical::<F>(&at_modulus),
            Err(TranscriptError::BadProofShape {
                reason: "field encoding outside canonical range",
            })
        );

        // The largest 4-byte integer is rejected for the same reason.
        assert!(decode_field_be_canonical::<F>(&[0xff; 4]).is_err());
    }

    #[test]
    fn canonical_decoding_rejects_a_short_slice() {
        // Boundary: one byte short of the field width.
        assert_eq!(
            decode_field_be_canonical::<F>(&[0u8; 3]),
            Err(TranscriptError::BadProofShape {
                reason: "not enough bytes for a canonical field encoding",
            })
        );
    }

    /// BigUint reference for the Horner decode: read big-endian, reduce mod p, lift to F.
    fn decode_reference<G: PrimeField64>(bytes: &[u8]) -> G {
        let need = required_bytes::<G>();
        let reduced = BigUint::from_bytes_be(&bytes[..need]) % G::order();
        // Every supported modulus is below 2^64, so the residue is a single limb.
        G::from_u64(reduced.iter_u64_digits().next().unwrap_or(0))
    }

    proptest! {
        #[test]
        fn horner_decode_matches_biguint_reference(raw in any::<[u8; 36]>()) {
            // The field-Horner fast path must equal the BigUint reduce-and-lift reference.
            //
            // Cover a 31-bit field (BabyBear) and a 64-bit field (Goldilocks).
            prop_assert_eq!(
                decode_field_via_extra_bytes::<F>(&raw[..required_bytes::<F>()]),
                decode_reference::<F>(&raw),
            );
            prop_assert_eq!(
                decode_field_via_extra_bytes::<Goldilocks>(&raw[..required_bytes::<Goldilocks>()]),
                decode_reference::<Goldilocks>(&raw),
            );
        }

        #[test]
        fn canonical_wire_encoding_round_trips(value in any::<u32>()) {
            // Every field element survives encode -> decode unchanged.
            let f = F::from_u32(value % F::ORDER_U32);
            let mut buf = Vec::new();
            encode_field_be(&f, &mut buf);
            prop_assert_eq!(decode_field_be_canonical::<F>(&buf).unwrap(), f);
        }
    }
}
