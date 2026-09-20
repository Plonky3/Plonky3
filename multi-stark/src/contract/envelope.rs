//! The framing a proof travels in, and the checks it passes before anything reads it.
//!
//! An encoded proof is attacker-controlled from the first byte.
//!
//! Nothing inside it is allowed to decide how much work reading it costs.
//!
//! The reader therefore settles the framing first, against the declaration's own numbers.
//!
//! Only then does it hand the body to the decoder.

use alloc::vec::Vec;
use core::fmt::Debug;
use core::marker::PhantomData;

use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_sumcheck::PrescribedPointPcs;
use thiserror::Error;

use crate::config::{Commitment, MultiStarkConfig, PcsError};
use crate::contract::declaration::{DeclarationError, MachineDeclaration, Run};
use crate::contract::secrecy::{Secrecy, SecrecyLevel};
use crate::folder::VerifierAir;
use crate::instance::VerifierInstances;
use crate::proof::MultiStarkProof;
use crate::verifier::{VerificationError, verify};

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

/// Why a byte string is not an acceptable proof for a statement.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum EnvelopeError {
    /// The run does not belong to the statement it was used with.
    #[error("declaration: {0}")]
    Declaration(DeclarationError),
    /// The input is shorter than the fixed header.
    #[error("input is {found} bytes, shorter than the {HEADER_LEN}-byte header")]
    HeaderTooShort {
        /// Length of the input.
        found: usize,
    },
    /// The input does not start with the bytes every sealed proof starts with.
    #[error("input does not carry the expected opening bytes")]
    BadMagic,
    /// The framing revision is not the one this build speaks.
    #[error("framing revision {found} is not the supported {expected}")]
    EnvelopeVersion {
        /// Revision the input declares.
        found: u16,
        /// Revision this build speaks.
        expected: u16,
    },
    /// The body revision is not the one this build speaks.
    #[error("body revision {found} is not the supported {expected}")]
    BodyRevision {
        /// Revision the input declares.
        found: u16,
        /// Revision this build speaks.
        expected: u16,
    },
    /// The input was sealed against a different statement or a different run of it.
    #[error("the input was sealed against a different statement")]
    RunMismatch,
    /// The declared body length is above the statement's budget.
    #[error("declared body length {found} is above the budget of {budget} bytes")]
    BodyAboveBudget {
        /// Length the header declares.
        found: usize,
        /// Largest length the statement accepts.
        budget: usize,
    },
    /// The input stops before the body the header declares.
    #[error("header declares {declared} body bytes but only {available} follow")]
    Truncated {
        /// Length the header declares.
        declared: usize,
        /// Length actually present.
        available: usize,
    },
    /// The input continues past the body the header declares.
    #[error("{extra} bytes follow the declared body")]
    TrailingBytes {
        /// Number of bytes past the declared body.
        extra: usize,
    },
    /// The body is not a well-formed encoding.
    #[error("the body is not a well-formed encoding")]
    Malformed,
    /// The body decodes but leaves bytes behind.
    #[error("{remaining} body bytes were not consumed by the decoder")]
    UnreadBodyBytes {
        /// Number of bytes the decoder did not consume.
        remaining: usize,
    },
    /// A part of the proof is present when the statement declares none, or the reverse.
    #[error("the {section} part is present ({present}) against what the statement declares")]
    SectionMismatch {
        /// Which part disagrees.
        section: &'static str,
        /// Whether the proof carries it.
        present: bool,
    },
    /// A count inside the proof disagrees with the statement.
    #[error("the {section} count is {found} but the statement declares {expected}")]
    CountMismatch {
        /// Which count disagrees.
        section: &'static str,
        /// Count the statement declares.
        expected: usize,
        /// Count the proof carries.
        found: usize,
    },
    /// The encoded proof is longer than the statement's budget.
    #[error("the encoded proof is {found} bytes, above the budget of {budget}")]
    ProofAboveBudget {
        /// Length of the encoding.
        found: usize,
        /// Largest length the statement accepts.
        budget: usize,
    },
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

impl<S: SecrecyLevel> MachineDeclaration<S> {
    /// Frame a proof for transport under one run of this statement.
    ///
    /// # Errors
    ///
    /// Returns an error when the encoding is longer than the declared budget.
    ///
    /// Returns an error when the run belongs to a different statement.
    pub fn seal<C: MultiStarkConfig>(
        &self,
        run: &Run,
        proof: &MultiStarkProof<C>,
    ) -> Result<SealedProof<S>, EnvelopeError> {
        let fingerprint = self.run_digest(run).map_err(EnvelopeError::Declaration)?;
        let body = postcard::to_allocvec(proof).map_err(|_| EnvelopeError::Malformed)?;

        if body.len() > self.max_proof_bytes() {
            return Err(EnvelopeError::ProofAboveBudget {
                found: body.len(),
                budget: self.max_proof_bytes(),
            });
        }
        let length = u32::try_from(body.len()).map_err(|_| EnvelopeError::ProofAboveBudget {
            found: body.len(),
            budget: self.max_proof_bytes(),
        })?;

        let mut bytes = Vec::with_capacity(HEADER_LEN + body.len());
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&ENVELOPE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&BODY_REVISION.to_le_bytes());
        bytes.extend_from_slice(&fingerprint);
        bytes.extend_from_slice(&length.to_le_bytes());
        bytes.extend_from_slice(&body);

        Ok(SealedProof {
            bytes,
            secrecy: PhantomData,
        })
    }

    /// Check a byte string against this statement and decode it.
    ///
    /// Every check runs before the transcript is touched, in this order:
    ///
    /// - the input is long enough to hold a header;
    /// - the opening bytes, the framing revision, and the body revision are the expected ones;
    /// - the fingerprint matches the one this statement and run produce;
    /// - the declared body length is within the declared budget;
    /// - the input holds exactly that many further bytes, with none left over;
    /// - the decoder consumes the whole body;
    /// - every optional part is present exactly when the statement says so;
    /// - every count the statement fixes agrees with the proof.
    ///
    /// # Errors
    ///
    /// Returns an error at the first of those checks that fails.
    pub fn open<C: MultiStarkConfig>(
        &self,
        run: &Run,
        bytes: &[u8],
    ) -> Result<AcceptedProof<S, C>, EnvelopeError> {
        let fingerprint = self.run_digest(run).map_err(EnvelopeError::Declaration)?;

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

        if bytes[RUN_RANGE] != fingerprint {
            return Err(EnvelopeError::RunMismatch);
        }

        // This is the only length the reader takes from the input.
        //
        // It is checked against the budget before it is used to slice anything.
        let declared = read_u32(&bytes[LENGTH_RANGE]) as usize;
        if declared > self.max_proof_bytes() {
            return Err(EnvelopeError::BodyAboveBudget {
                found: declared,
                budget: self.max_proof_bytes(),
            });
        }

        let available = bytes.len() - HEADER_LEN;
        if available < declared {
            return Err(EnvelopeError::Truncated {
                declared,
                available,
            });
        }
        if available > declared {
            return Err(EnvelopeError::TrailingBytes {
                extra: available - declared,
            });
        }

        // The decoder allocates in proportion to what it reads, and it reads only this slice.
        //
        // The budget checked above is therefore what bounds its memory.
        let body = &bytes[HEADER_LEN..];
        let (proof, rest) = postcard::take_from_bytes::<MultiStarkProof<C>>(body)
            .map_err(|_| EnvelopeError::Malformed)?;
        if !rest.is_empty() {
            return Err(EnvelopeError::UnreadBodyBytes {
                remaining: rest.len(),
            });
        }

        self.check_shape(&proof)?;

        Ok(AcceptedProof {
            proof,
            pow_bits: run.pow_bits(),
            secrecy: PhantomData,
        })
    }

    /// Reject a decoded proof whose parts disagree with what the statement declares.
    fn check_shape<C: MultiStarkConfig>(
        &self,
        proof: &MultiStarkProof<C>,
    ) -> Result<(), EnvelopeError> {
        let section = |section, present: bool, declared: bool| {
            (present == declared)
                .then_some(())
                .ok_or(EnvelopeError::SectionMismatch { section, present })
        };

        section("lookup", proof.lookup.is_some(), self.has_lookups())?;
        section(
            "preprocessed opening",
            proof.preprocessed_opening.is_some(),
            self.has_preprocessed(),
        )?;

        let reads = self.num_indexed_reads();
        section("indexed", proof.indexed.is_some(), reads > 0)?;

        if let Some(indexed) = proof.indexed.as_ref()
            && indexed.reader_claims.len() != reads
        {
            return Err(EnvelopeError::CountMismatch {
                section: "indexed reader",
                expected: reads,
                found: indexed.reader_claims.len(),
            });
        }

        Ok(())
    }
}

/// Why a sealed proof was not accepted.
#[derive(Debug, Error)]
pub enum SealedVerificationError<E: Debug> {
    /// The byte string never became a proof.
    #[error("envelope: {0}")]
    Envelope(EnvelopeError),
    /// The run and the instances describe different statements.
    #[error("the run and the instances disagree on {what}")]
    RunDisagreement {
        /// Which part disagrees.
        what: &'static str,
    },
    /// The proof was well framed but did not verify.
    #[error("verification: {0}")]
    Verification(VerificationError<E>),
}

/// Check a byte string against a statement and verify what comes out of it.
///
/// The grinding difficulty comes from the run rather than from the caller.
///
/// The heights the instances carry are compared against the run rather than trusted.
///
/// # Errors
///
/// Returns an error when the framing, the shape, the heights, or the proof itself fails.
pub fn verify_sealed<'a, S, C, A>(
    declaration: &MachineDeclaration<S>,
    run: &Run,
    bytes: &[u8],
    config: &C,
    instances: VerifierInstances<'a, C, A>,
    challenger: &mut C::Challenger,
) -> Result<(), SealedVerificationError<PcsError<C>>>
where
    S: SecrecyLevel,
    C: MultiStarkConfig,
    C::Pcs: PrescribedPointPcs<C::Challenge, C::Challenger>,
    C::Challenger: FieldChallenger<C::Val>
        + GrindingChallenger<Witness = C::Val>
        + CanSampleUniformBits<C::Val>
        + CanObserve<Commitment<C>>,
    Commitment<C>: Clone,
    A: VerifierAir<C::Val, C::Challenge>,
{
    let accepted = declaration
        .open::<C>(run, bytes)
        .map_err(SealedVerificationError::Envelope)?;

    if instances.len() != run.log_heights().len() {
        return Err(SealedVerificationError::RunDisagreement {
            what: "the number of tables",
        });
    }
    let heights_agree = instances
        .iter()
        .zip(run.log_heights())
        .all(|(instance, &declared)| instance.num_variables() as u64 == u64::from(declared));
    if !heights_agree {
        return Err(SealedVerificationError::RunDisagreement {
            what: "a table height",
        });
    }

    verify(
        config,
        instances,
        accepted.proof(),
        accepted.pow_bits(),
        challenger,
    )
    .map_err(SealedVerificationError::Verification)
}

const fn read_u16(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}

const fn read_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}
