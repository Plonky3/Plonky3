//! The statement a machine publishes, and the operations that ride on it.
//!
//! Sealing frames a proof against the statement.
//!
//! Opening refuses a byte string the statement does not account for, before any replay.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_sumcheck::PrescribedPointPcs;
use p3_symmetric::CryptographicHasher;

use crate::config::{Commitment, MultiStarkConfig, PcsError};
use crate::contract::digest::Preimage;
use crate::contract::envelope::{AcceptedProof, HEADER_LEN, Header, SealedProof};
use crate::contract::error::{DeclarationError, EnvelopeError, SealedVerificationError};
use crate::contract::run::Run;
use crate::contract::secrecy::SecrecyLevel;
use crate::contract::table::TableDeclaration;
use crate::folder::VerifierAir;
use crate::instance::VerifierInstances;
use crate::proof::MultiStarkProof;
use crate::verifier::verify;

/// Largest number of tables one statement may declare.
pub const MAX_TABLES: usize = 1 << 12;

/// Largest grinding difficulty a run may request.
pub const MAX_POW_BITS: u32 = 64;

/// Hard ceiling on any declared proof-size budget, in bytes.
pub const MAX_PROOF_BYTES: usize = 1 << 30;

/// Every table of one statement, the commitment promise, the hash, and the size budget.
///
/// The promise and the hash are type parameters.
///
/// A statement proved under one pair is not the statement proved under another.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MachineDeclaration<S, H> {
    hasher: H,
    tables: Vec<TableDeclaration>,
    max_proof_bytes: usize,
    secrecy: PhantomData<fn() -> S>,
}

impl<S, H> MachineDeclaration<S, H>
where
    S: SecrecyLevel,
    H: CryptographicHasher<u8, [u8; 32]>,
{
    /// Fix the tables of a statement, the hash that fingerprints it, and the byte budget.
    ///
    /// The budget is a public parameter, and the proof reader rejects anything longer.
    ///
    /// # Errors
    ///
    /// Returns an error when a declared count is outside the accepted range.
    pub fn new(
        hasher: H,
        tables: Vec<TableDeclaration>,
        max_proof_bytes: usize,
    ) -> Result<Self, DeclarationError> {
        if tables.is_empty() {
            return Err(DeclarationError::NoTables);
        }
        if tables.len() > MAX_TABLES {
            return Err(DeclarationError::AboveLimit {
                table: 0,
                what: "table count",
                found: tables.len(),
                limit: MAX_TABLES,
            });
        }
        if max_proof_bytes == 0 || max_proof_bytes > MAX_PROOF_BYTES {
            return Err(DeclarationError::BudgetOutOfRange {
                found: max_proof_bytes,
                limit: MAX_PROOF_BYTES,
            });
        }
        for (index, table) in tables.iter().enumerate() {
            table.validate(index)?;
        }

        Ok(Self {
            hasher,
            tables,
            max_proof_bytes,
            secrecy: PhantomData,
        })
    }

    /// The tables of this statement, in the order proofs list them.
    #[must_use]
    pub fn tables(&self) -> &[TableDeclaration] {
        &self.tables
    }

    /// The largest encoded proof this statement accepts, in bytes.
    #[must_use]
    pub const fn max_proof_bytes(&self) -> usize {
        self.max_proof_bytes
    }

    /// Whether any table commits columns fixed at setup.
    #[must_use]
    pub fn has_preprocessed(&self) -> bool {
        self.tables
            .iter()
            .any(|table| table.columns().preprocessed > 0)
    }

    /// Whether any table takes part in a lookup argument.
    #[must_use]
    pub fn has_lookups(&self) -> bool {
        self.tables.iter().any(TableDeclaration::has_lookups)
    }

    /// Total number of indexed reads across every table.
    #[must_use]
    pub fn num_indexed_reads(&self) -> usize {
        self.tables
            .iter()
            .map(TableDeclaration::indexed_reads)
            .sum()
    }

    /// Pick the height of every table and the grinding difficulty for one proof.
    ///
    /// # Errors
    ///
    /// Returns an error when a height is outside its table's declared range.
    ///
    /// Returns an error when the heights and the tables disagree in number.
    pub fn run(&self, log_heights: &[usize], pow_bits: usize) -> Result<Run, DeclarationError> {
        if log_heights.len() != self.tables.len() {
            return Err(DeclarationError::HeightCountMismatch {
                expected: self.tables.len(),
                found: log_heights.len(),
            });
        }
        if pow_bits > MAX_POW_BITS as usize {
            return Err(DeclarationError::PowBitsAboveLimit {
                found: u32::try_from(pow_bits).unwrap_or(u32::MAX),
                limit: MAX_POW_BITS,
            });
        }

        let mut heights = Vec::with_capacity(log_heights.len());
        for (index, (&log_height, table)) in log_heights.iter().zip(&self.tables).enumerate() {
            let range = table.heights();
            let found = u32::try_from(log_height).unwrap_or(u32::MAX);
            if found < range.min || found > range.max {
                return Err(DeclarationError::HeightNotDeclared {
                    table: index,
                    found,
                    min: range.min,
                    max: range.max,
                });
            }
            heights.push(found);
        }

        Ok(Run::new(self.statement_digest(), heights, pow_bits as u32))
    }

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

        let too_long = || EnvelopeError::ProofAboveBudget {
            found: body.len(),
            budget: self.max_proof_bytes,
        };
        if body.len() > self.max_proof_bytes {
            return Err(too_long());
        }
        let length = u32::try_from(body.len()).map_err(|_| too_long())?;

        let mut bytes = Vec::with_capacity(HEADER_LEN + body.len());
        bytes.extend_from_slice(&Header::write(&fingerprint, length));
        bytes.extend_from_slice(&body);
        Ok(SealedProof::new(bytes))
    }

    /// Check a byte string against this statement and decode it.
    ///
    /// Every check runs before the transcript is touched, in this order:
    ///
    /// - the framing is one this build speaks;
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
        let header = Header::parse(bytes)?;

        if header.fingerprint != fingerprint {
            return Err(EnvelopeError::RunMismatch);
        }

        // This is the only length the reader takes from the input.
        //
        // It is checked against the budget before it is used to slice anything.
        let declared = header.body_len;
        if declared > self.max_proof_bytes {
            return Err(EnvelopeError::BodyAboveBudget {
                found: declared,
                budget: self.max_proof_bytes,
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
        Ok(AcceptedProof::new(proof, run.pow_bits()))
    }

    /// Check a byte string against this statement and verify what comes out of it.
    ///
    /// The grinding difficulty comes from the run rather than from the caller.
    ///
    /// The heights the instances carry are compared against the run rather than trusted.
    ///
    /// # Errors
    ///
    /// Returns an error when the framing, the shape, the heights, or the proof itself fails.
    pub fn verify<'a, C, A>(
        &self,
        run: &Run,
        bytes: &[u8],
        config: &C,
        instances: VerifierInstances<'a, C, A>,
        challenger: &mut C::Challenger,
    ) -> Result<(), SealedVerificationError<PcsError<C>>>
    where
        C: MultiStarkConfig,
        C::Pcs: PrescribedPointPcs<C::Challenge, C::Challenger>,
        C::Challenger: FieldChallenger<C::Val>
            + GrindingChallenger<Witness = C::Val>
            + CanSampleUniformBits<C::Val>
            + CanObserve<Commitment<C>>,
        Commitment<C>: Clone,
        A: VerifierAir<C::Val, C::Challenge>,
    {
        let accepted = self
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

    /// Fingerprint of everything fixed before a proof exists.
    fn statement_digest(&self) -> [u8; 32] {
        let mut preimage = Preimage::new(b"p3-backend-contract/statement/v1");
        preimage.byte(S::SECRECY.tag());
        preimage.usize(self.max_proof_bytes);
        preimage.usize(self.tables.len());
        for table in &self.tables {
            table.absorb(&mut preimage);
        }
        preimage.finish(&self.hasher)
    }

    /// Fingerprint of the statement together with the choices one proof makes.
    pub(crate) fn run_digest(&self, run: &Run) -> Result<[u8; 32], DeclarationError> {
        let statement = self.statement_digest();
        if run.statement() != &statement {
            return Err(DeclarationError::ForeignRun);
        }

        let mut preimage = Preimage::new(b"p3-backend-contract/run/v1");
        preimage.bytes(&statement);
        preimage.u32(run.pow_bits_raw());
        preimage.usize(run.log_heights().len());
        for &log_height in run.log_heights() {
            preimage.u32(log_height);
        }
        Ok(preimage.finish(&self.hasher))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_keccak::Keccak256Hash;

    use super::*;
    use crate::contract::secrecy::BindingOnly;
    use crate::contract::table::{
        ColumnCounts, FlushDeclaration, FlushDirection, HeightRange, LocalConstraints,
    };

    type Declaration = MachineDeclaration<BindingOnly, Keccak256Hash>;

    fn table() -> TableDeclaration {
        TableDeclaration::new(
            ColumnCounts {
                committed: 4,
                preprocessed: 0,
                public: 1,
            },
            LocalConstraints {
                count: 3,
                degree: 2,
            },
            HeightRange::new(2, 16),
        )
    }

    #[test]
    fn a_statement_needs_at_least_one_table() {
        assert_eq!(
            Declaration::new(Keccak256Hash, vec![], 1024).unwrap_err(),
            DeclarationError::NoTables
        );
    }

    #[test]
    fn a_budget_outside_the_ceiling_is_refused() {
        for budget in [0, MAX_PROOF_BYTES + 1] {
            assert_eq!(
                Declaration::new(Keccak256Hash, vec![table()], budget).unwrap_err(),
                DeclarationError::BudgetOutOfRange {
                    found: budget,
                    limit: MAX_PROOF_BYTES,
                }
            );
        }
    }

    #[test]
    fn a_table_that_fails_its_own_checks_is_refused() {
        let inverted = TableDeclaration::new(
            ColumnCounts::default(),
            LocalConstraints::default(),
            HeightRange::new(9, 8),
        );
        assert_eq!(
            Declaration::new(Keccak256Hash, vec![table(), inverted], 1024).unwrap_err(),
            DeclarationError::EmptyHeightRange {
                table: 1,
                min: 9,
                max: 8,
            }
        );
    }

    #[test]
    fn a_height_outside_the_declared_range_is_refused() {
        let declaration = Declaration::new(Keccak256Hash, vec![table()], 1024).unwrap();
        // The one table declares 2..=16, so one below the floor names the whole range back.
        assert_eq!(
            declaration.run(&[1], 0).unwrap_err(),
            DeclarationError::HeightNotDeclared {
                table: 0,
                found: 1,
                min: 2,
                max: 16,
            }
        );
        assert_eq!(
            declaration.run(&[8, 8], 0).unwrap_err(),
            DeclarationError::HeightCountMismatch {
                expected: 1,
                found: 2,
            }
        );
    }

    #[test]
    fn grinding_above_the_ceiling_is_refused() {
        let declaration = Declaration::new(Keccak256Hash, vec![table()], 1024).unwrap();
        assert_eq!(
            declaration.run(&[8], 1000).unwrap_err(),
            DeclarationError::PowBitsAboveLimit {
                found: 1000,
                limit: 64,
            }
        );
    }

    #[test]
    fn a_run_of_one_statement_is_refused_by_another() {
        // Two statements differing only in the channel a table flushes on.
        let plain = Declaration::new(Keccak256Hash, vec![table()], 1024).unwrap();
        let flushing = Declaration::new(
            Keccak256Hash,
            vec![table().with_flushes(vec![FlushDeclaration {
                channel: "shared".into(),
                direction: FlushDirection::Push,
                tuple_width: 3,
                max_multiplicity: 1,
            }])],
            1024,
        )
        .unwrap();

        let run = plain.run(&[8], 0).unwrap();
        assert_eq!(
            flushing.run_digest(&run).unwrap_err(),
            DeclarationError::ForeignRun
        );
        assert!(plain.run_digest(&run).is_ok());
    }

    #[test]
    fn the_choices_a_run_makes_change_the_fingerprint() {
        let declaration = Declaration::new(Keccak256Hash, vec![table()], 1024).unwrap();
        let low = declaration.run(&[8], 0).unwrap();
        let high = declaration.run(&[9], 0).unwrap();
        let ground = declaration.run(&[8], 1).unwrap();

        let digest = |run| declaration.run_digest(run).unwrap();
        assert_ne!(digest(&low), digest(&high));
        assert_ne!(digest(&low), digest(&ground));
    }
}
