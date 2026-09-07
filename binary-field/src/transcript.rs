//! Canonical typed-transcript encoding for the 128-bit tower field.

use alloc::vec::Vec;

use p3_challenger::CanObserve;
use p3_challenger::fs::{TranscriptError, TranscriptField, TypeTag};

use crate::{BinaryField128, TowerLevel};

impl TranscriptField for BinaryField128 {
    fn algebra_tag(degree: usize) -> TypeTag {
        TypeTag::BinaryTower { bits: 128, degree }
    }

    fn observe_seed<C: CanObserve<Self>>(challenger: &mut C, bytes: &[u8]) {
        // Embed the length and chunks in the tower basis, not via `from_u128`,
        // which embeds integers in GF(2) and would retain only their parity.
        challenger.observe(Self::from_repr(bytes.len() as u128));
        for chunk in bytes.chunks(16) {
            let mut padded = [0; 16];
            padded[..chunk.len()].copy_from_slice(chunk);
            challenger.observe(Self::from_repr(u128::from_le_bytes(padded)));
        }
    }

    fn wire_len() -> usize {
        16
    }

    fn encode(value: &Self, out: &mut Vec<u8>) {
        out.extend_from_slice(&value.to_repr().to_be_bytes());
    }

    fn decode(bytes: &[u8]) -> Result<Self, TranscriptError> {
        let prefix = bytes.get(..16).ok_or(TranscriptError::BadProofShape {
            reason: "not enough bytes for a canonical binary field encoding",
        })?;
        // Every 128-bit representation is canonical in this field.
        Ok(Self::from_repr(u128::from_be_bytes(
            prefix.try_into().unwrap(),
        )))
    }
}
