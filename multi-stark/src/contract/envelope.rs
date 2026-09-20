//! The framing a proof travels in.
//!
//! This module owns the wire layout and nothing else.
//!
//! It decides whether a byte string is framed at all, not whether the framing suits a statement.

use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::config::MultiStarkConfig;
use crate::contract::error::EnvelopeError;
use crate::contract::secrecy::{Secrecy, SecrecyLevel};
use crate::proof::MultiStarkProof;

/// Bytes that open every sealed proof.
pub const MAGIC: [u8; 8] = *b"P3PROOF\x1a";

/// Revision of the framing itself.
///
/// Bump it whenever the header layout changes.
pub const ENVELOPE_VERSION: u16 = 1;

/// Revision of the encoded body.
///
/// Bump it whenever the serialized form of a proof changes in any way.
pub const BODY_REVISION: u16 = 1;

/// Size of the fixed header, in bytes.
pub const HEADER_LEN: usize = 48;

const MAGIC_RANGE: core::ops::Range<usize> = 0..8;
const ENVELOPE_VERSION_RANGE: core::ops::Range<usize> = 8..10;
const BODY_REVISION_RANGE: core::ops::Range<usize> = 10..12;
const RUN_RANGE: core::ops::Range<usize> = 12..44;
const LENGTH_RANGE: core::ops::Range<usize> = 44..48;

/// What the fixed header carries.
///
/// A parsed one has already passed the checks that do not depend on any statement.
#[derive(Debug)]
pub(super) struct Header {
    /// Fingerprint the sender claims to have sealed against.
    pub(super) fingerprint: [u8; 32],
    /// Length the sender claims the body has.
    pub(super) body_len: usize,
}

impl Header {
    /// Lay out a header in front of a body of the given length.
    pub(super) fn write(fingerprint: &[u8; 32], body_len: u32) -> [u8; HEADER_LEN] {
        let mut header = [0u8; HEADER_LEN];
        header[MAGIC_RANGE].copy_from_slice(&MAGIC);
        header[ENVELOPE_VERSION_RANGE].copy_from_slice(&ENVELOPE_VERSION.to_le_bytes());
        header[BODY_REVISION_RANGE].copy_from_slice(&BODY_REVISION.to_le_bytes());
        header[RUN_RANGE].copy_from_slice(fingerprint);
        header[LENGTH_RANGE].copy_from_slice(&body_len.to_le_bytes());
        header
    }

    /// Read the header off an input, rejecting anything this build does not speak.
    ///
    /// # Errors
    ///
    /// Returns an error when the input is too short, mislabelled, or of another revision.
    pub(super) fn parse(bytes: &[u8]) -> Result<Self, EnvelopeError> {
        if bytes.len() < HEADER_LEN {
            return Err(EnvelopeError::HeaderTooShort { found: bytes.len() });
        }
        if bytes[MAGIC_RANGE] != MAGIC {
            return Err(EnvelopeError::BadMagic);
        }

        let envelope_version = read_u16(&bytes[ENVELOPE_VERSION_RANGE]);
        if envelope_version != ENVELOPE_VERSION {
            return Err(EnvelopeError::EnvelopeVersion {
                found: envelope_version,
                expected: ENVELOPE_VERSION,
            });
        }

        let body_revision = read_u16(&bytes[BODY_REVISION_RANGE]);
        if body_revision != BODY_REVISION {
            return Err(EnvelopeError::BodyRevision {
                found: body_revision,
                expected: BODY_REVISION,
            });
        }

        let mut fingerprint = [0u8; 32];
        fingerprint.copy_from_slice(&bytes[RUN_RANGE]);

        Ok(Self {
            fingerprint,
            body_len: read_u32(&bytes[LENGTH_RANGE]) as usize,
        })
    }
}

/// A proof framed for transport under one statement and one run of it.
///
/// The commitment promise is part of the type.
///
/// A proof that only binds its trace cannot stand in for one that also hides it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SealedProof<S> {
    bytes: Vec<u8>,
    secrecy: PhantomData<fn() -> S>,
}

impl<S: SecrecyLevel> SealedProof<S> {
    /// Wrap bytes that a header has just been written in front of.
    pub(super) const fn new(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            secrecy: PhantomData,
        }
    }

    /// What the commitments behind this proof promise.
    #[must_use]
    pub const fn secrecy(&self) -> Secrecy {
        S::SECRECY
    }

    /// The bytes to transmit.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Take the bytes to transmit.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// A proof whose framing and shape already agree with a statement.
///
/// Holding one is the evidence that every check preceding transcript replay has passed.
///
/// It also carries the grinding difficulty the run fixed.
///
/// Verification cannot then run at a difficulty the proof was not sealed under.
pub struct AcceptedProof<S, C: MultiStarkConfig> {
    proof: MultiStarkProof<C>,
    pow_bits: usize,
    secrecy: PhantomData<fn() -> S>,
}

impl<S: SecrecyLevel, C: MultiStarkConfig> core::fmt::Debug for AcceptedProof<S, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AcceptedProof")
            .field("secrecy", &S::SECRECY)
            .field("pow_bits", &self.pow_bits)
            .finish_non_exhaustive()
    }
}

impl<S: SecrecyLevel, C: MultiStarkConfig> AcceptedProof<S, C> {
    /// Record a proof that has passed every pre-replay check.
    pub(super) const fn new(proof: MultiStarkProof<C>, pow_bits: usize) -> Self {
        Self {
            proof,
            pow_bits,
            secrecy: PhantomData,
        }
    }

    /// The decoded proof, for a caller that only wants to look at it.
    #[must_use]
    pub const fn proof(&self) -> &MultiStarkProof<C> {
        &self.proof
    }

    /// Grinding difficulty the run fixed.
    #[must_use]
    pub const fn pow_bits(&self) -> usize {
        self.pow_bits
    }
}

const fn read_u16(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}

const fn read_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    const FINGERPRINT: [u8; 32] = [7u8; 32];

    #[test]
    fn a_written_header_parses_back() {
        let header = Header::write(&FINGERPRINT, 1234);
        let parsed = Header::parse(&header).unwrap();
        assert_eq!(parsed.fingerprint, FINGERPRINT);
        assert_eq!(parsed.body_len, 1234);
    }

    #[test]
    fn an_input_shorter_than_the_header_is_refused() {
        let header = Header::write(&FINGERPRINT, 0);
        for length in [0, 1, HEADER_LEN - 1] {
            assert_eq!(
                Header::parse(&header[..length]).unwrap_err(),
                EnvelopeError::HeaderTooShort { found: length }
            );
        }
    }

    #[test]
    fn the_wrong_opening_bytes_are_refused() {
        let mut header = Header::write(&FINGERPRINT, 0);
        header[0] ^= 1;
        assert_eq!(Header::parse(&header).unwrap_err(), EnvelopeError::BadMagic);
    }

    #[test]
    fn a_revision_this_build_does_not_speak_is_refused() {
        let mut framing = Header::write(&FINGERPRINT, 0);
        framing[8..10].copy_from_slice(&(ENVELOPE_VERSION + 1).to_le_bytes());
        assert!(matches!(
            Header::parse(&framing).unwrap_err(),
            EnvelopeError::EnvelopeVersion { .. }
        ));

        let mut body = Header::write(&FINGERPRINT, 0);
        body[10..12].copy_from_slice(&(BODY_REVISION + 1).to_le_bytes());
        assert!(matches!(
            Header::parse(&body).unwrap_err(),
            EnvelopeError::BodyRevision { .. }
        ));
    }
}
