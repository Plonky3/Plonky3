//! Fixed-width canonical proof transport, sized entirely by the compiled AIR.

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::TowerLevel;
use p3_symmetric::MerkleCap;

use super::{
    ReferencePcsConfig, ReferenceZkAir, Val, ZkAirProof, ZkCommitment, ZkError, ZkOpeningProof,
    ZkQueryOpening, ZkRowOpening,
};

// This transport version is independent of the protocol transcript version.
const MAGIC: &[u8; 8] = b"P3ZKAIR2";

/// Compile-time checked transport dimensions.
/// No wire value controls allocation.
#[derive(Debug, Clone)]
pub(super) struct WireGeometry {
    header: [u64; 3],
    width: usize,
    coefficient_len: usize,
    num_queries: usize,
    path_len: usize,
    encoded_len: usize,
}

impl WireGeometry {
    pub(super) fn new(width: usize, pcs: &ReferencePcsConfig) -> Result<Self, ZkError> {
        let overflow = ZkError::InvalidConfig("proof wire geometry overflow");
        let header = [
            u64::try_from(width).map_err(|_| overflow.clone())?,
            u64::try_from(pcs.message_len()).map_err(|_| overflow.clone())?,
            u64::try_from(pcs.num_queries()).map_err(|_| overflow.clone())?,
        ];
        let path_len = pcs.codeword_len().ilog2() as usize;
        let field_bytes = width.checked_mul(16).ok_or_else(|| overflow.clone())?;
        let row_bytes = path_len
            .checked_mul(32)
            .and_then(|v| v.checked_add(32))
            .and_then(|v| v.checked_add(field_bytes))
            .ok_or_else(|| overflow.clone())?;
        // Header + roots, claims + sigma, column-major blends, h/u queries.
        let encoded_len = field_bytes
            .checked_mul(pcs.coefficient_len())
            .and_then(|v| v.checked_add(96))
            .and_then(|v| v.checked_add(field_bytes.checked_mul(2)?))
            .and_then(|v| v.checked_add(row_bytes.checked_mul(pcs.num_queries())?.checked_mul(2)?))
            .ok_or(overflow)?;
        Ok(Self {
            header,
            width,
            coefficient_len: pcs.coefficient_len(),
            num_queries: pcs.num_queries(),
            path_len,
            encoded_len,
        })
    }
}

impl ReferenceZkAir {
    /// Exact canonical v2 proof size for this compiled AIR, in bytes.
    pub const fn encoded_proof_len(&self) -> usize {
        self.wire.encoded_len
    }

    /// Encode a shape-correct proof with fixed-width little-endian geometry and tower-field representations.
    /// This validates shapes, not proof validity.
    pub fn encode_proof(&self, proof: &ZkAirProof) -> Result<Vec<u8>, ZkError> {
        self.pcs.validate_commitment(&proof.commitment)?;
        if proof.commitment.num_polynomials != self.wire.width {
            return Err(ZkError::InvalidShape(
                "commitment batch does not match compiled AIR",
            ));
        }
        self.pcs.validate_proof(self.wire.width, &proof.opening)?;
        let mut bytes = Vec::with_capacity(self.encoded_proof_len());
        bytes.extend_from_slice(MAGIC);
        for value in self.wire.header {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&proof.commitment.root.roots()[0]);
        write_fields(&mut bytes, &proof.opening.claimed_values);
        bytes.extend_from_slice(&proof.opening.mask_root.roots()[0]);
        write_fields(&mut bytes, &proof.opening.sigma);
        for column in &proof.opening.blended_coefficients {
            write_fields(&mut bytes, column);
        }
        for query in &proof.opening.queries {
            for row in [&query.h, &query.u] {
                write_fields(&mut bytes, &row.values);
                write_fields(&mut bytes, &row.salts);
                for hash in &row.authentication_path {
                    bytes.extend_from_slice(hash);
                }
            }
        }
        Ok(bytes)
    }

    /// Decode bounded canonical v2 transport.
    /// Exact size, magic, and geometry are checked before allocating any proof vectors; all allocation counts come from this compiled AIR.
    /// Decoding alone does not verify a proof.
    ///
    /// Use this codec for untrusted network input.
    /// Generic serde/postcard decoding remains available, but does not provide these transport bounds.
    pub fn decode_proof(&self, bytes: &[u8]) -> Result<ZkAirProof, ZkError> {
        if bytes.len() != self.encoded_proof_len() {
            return Err(ZkError::InvalidShape("proof wire length mismatch"));
        }
        let mut reader = Reader { remaining: bytes };
        if &reader.take::<8>()? != MAGIC {
            return Err(ZkError::InvalidShape("proof wire version mismatch"));
        }
        for expected in self.wire.header {
            if u64::from_le_bytes(reader.take()?) != expected {
                return Err(ZkError::InvalidShape("proof wire geometry mismatch"));
            }
        }
        let root = MerkleCap::new(vec![reader.take()?]);
        let claimed_values = reader.fields(self.wire.width)?;
        let mask_root = MerkleCap::new(vec![reader.take()?]);
        let sigma = reader.fields(self.wire.width)?;
        let blended_coefficients = (0..self.wire.width)
            .map(|_| reader.fields(self.wire.coefficient_len))
            .collect::<Result<_, _>>()?;
        let queries = (0..self.wire.num_queries)
            .map(|_| {
                Ok(ZkQueryOpening {
                    h: reader.row(&self.wire)?,
                    u: reader.row(&self.wire)?,
                })
            })
            .collect::<Result<_, ZkError>>()?;
        if !reader.remaining.is_empty() {
            return Err(ZkError::InvalidShape("trailing proof wire bytes"));
        }
        Ok(ZkAirProof {
            commitment: ZkCommitment {
                root,
                num_polynomials: self.wire.width,
                message_len: self.pcs.config().message_len(),
                num_queries: self.wire.num_queries,
            },
            opening: ZkOpeningProof {
                claimed_values,
                mask_root,
                sigma,
                blended_coefficients,
                queries,
            },
        })
    }

    /// Decode bounded transport, then verify the complete proof and statement.
    pub fn verify_bytes(&self, bytes: &[u8], public_values: &[Val]) -> Result<(), ZkError> {
        self.verify(&self.decode_proof(bytes)?, public_values)
    }
}

fn write_fields(bytes: &mut Vec<u8>, values: &[Val]) {
    for value in values {
        bytes.extend_from_slice(&value.to_repr().to_le_bytes());
    }
}

struct Reader<'a> {
    remaining: &'a [u8],
}

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N], ZkError> {
        let (value, remaining) = self
            .remaining
            .split_first_chunk::<N>()
            .ok_or(ZkError::InvalidShape("truncated proof wire bytes"))?;
        self.remaining = remaining;
        Ok(*value)
    }

    fn field(&mut self) -> Result<Val, ZkError> {
        Ok(Val::from_repr(u128::from_le_bytes(self.take()?)))
    }

    fn fields(&mut self, len: usize) -> Result<Vec<Val>, ZkError> {
        (0..len).map(|_| self.field()).collect()
    }

    fn row(&mut self, geometry: &WireGeometry) -> Result<ZkRowOpening, ZkError> {
        Ok(ZkRowOpening {
            values: self.fields(geometry.width)?,
            salts: [self.field()?, self.field()?],
            authentication_path: (0..geometry.path_len)
                .map(|_| self.take())
                .collect::<Result<_, _>>()?,
        })
    }
}
