//! The statement a machine publishes, and the operations that ride on it.
//!
//! Sealing frames a proof against the statement.
//!
//! Opening refuses a byte string the statement does not account for, before any replay.

use alloc::vec::Vec;

use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_sumcheck::PrescribedPointPcs;
use p3_symmetric::CryptographicHasher;

use crate::config::{Commitment, MultiStarkConfig, PcsError};
use crate::contract::digest::Preimage;
use crate::contract::envelope::{AcceptedProof, HEADER_LEN, Header, SealedProof};
use crate::contract::error::{DeclarationError, EnvelopeError, SealedVerificationError};
use crate::contract::run::Run;
use crate::contract::table::TableDeclaration;
use crate::folder::VerifierAir;
use crate::instance::VerifierInstances;
use crate::proof::MultiStarkProof;
use crate::verifier::verify_with_security;

/// Largest number of tables one statement may declare.
pub const MAX_TABLES: usize = 1 << 12;

/// Largest grinding difficulty a run may request.
///
/// Every field this backend proves over has more than two to this power of elements.
///
/// That is what a transcript needs in order to sample that many bits at all.
pub const MAX_POW_BITS: u32 = 30;

/// Largest security target a statement may ask for, in bits.
pub const MAX_SECURITY_BITS: usize = 1 << 10;

/// Hard ceiling on any declared proof-size budget, in bytes.
///
/// A decoder can hold about twenty-four bytes of memory per byte it reads.
///
/// An empty inner list costs one input byte and a pointer triple to keep.
pub const MAX_PROOF_BYTES: usize = 1 << 26;

/// Every table of one statement, the hash that names it, the size budget, and the target.
///
/// The hash is a type parameter.
///
/// A statement named by one hash is not the statement named by another.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MachineDeclaration<H> {
    hasher: H,
    tables: Vec<TableDeclaration>,
    max_proof_bytes: usize,
    security_bits: usize,
    statement: [u8; 32],
}

impl<H> MachineDeclaration<H>
where
    H: CryptographicHasher<u8, [u8; 32]>,
{
    /// Fix the tables of a statement, the hash that names it, the byte budget, and the target.
    ///
    /// The budget is a public parameter, and the proof reader rejects anything longer.
    ///
    /// The target is the security level every verification of this statement must reach.
    ///
    /// # Errors
    ///
    /// Returns an error when a declared count is outside the accepted range.
    pub fn new(
        hasher: H,
        tables: Vec<TableDeclaration>,
        max_proof_bytes: usize,
        security_bits: usize,
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
        if security_bits == 0 || security_bits > MAX_SECURITY_BITS {
            return Err(DeclarationError::SecurityOutOfRange {
                found: security_bits,
                limit: MAX_SECURITY_BITS,
            });
        }
        for (index, table) in tables.iter().enumerate() {
            table.validate(index)?;
        }

        // Absorbing the tables walks every constraint, so it happens here and nowhere else.
        let mut preimage = Preimage::new(b"p3-backend-contract/statement/v1");
        preimage.usize(max_proof_bytes);
        preimage.usize(security_bits);
        preimage.usize(tables.len());
        for table in &tables {
            table.absorb(&mut preimage);
        }
        let statement = preimage.finish(&hasher);

        Ok(Self {
            hasher,
            tables,
            max_proof_bytes,
            security_bits,
            statement,
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

    /// The security level every verification of this statement must reach.
    #[must_use]
    pub const fn security_bits(&self) -> usize {
        self.security_bits
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

    /// Whether any table moves a tuple across a bus.
    #[must_use]
    pub fn has_buses(&self) -> bool {
        self.tables.iter().any(TableDeclaration::has_buses)
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

        Ok(Run::new(self.statement, heights, pow_bits as u32))
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
    ) -> Result<SealedProof, EnvelopeError> {
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
    ) -> Result<AcceptedProof<C>, EnvelopeError> {
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
    /// Each constraint system is read back and compared against the table that claims it.
    ///
    /// The declared security target is then enforced before the transcript is replayed.
    ///
    /// # Errors
    ///
    /// Returns an error when the framing, the shape, the heights, or the proof itself fails.
    ///
    /// Returns an error when a table and its constraint system describe different statements.
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

        // Nothing above reads a constraint system, so this is where the two sides meet.
        for (table, (declared, instance)) in self.tables.iter().zip(instances.iter()).enumerate() {
            if let Some(what) = declared.disagreement::<C::Val, C::Challenge, A>(instance.air()) {
                return Err(SealedVerificationError::AirDisagreement { table, what });
            }
        }

        verify_with_security(
            config,
            instances,
            accepted.proof(),
            accepted.pow_bits(),
            self.security_bits,
            challenger,
        )
        .map_err(SealedVerificationError::Verification)
    }

    /// Reject a decoded proof whose parts disagree with what the statement declares.
    ///
    /// The verifier reaches the same verdict from the constraint systems.
    ///
    /// It holds those and this reader does not.
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
        section("bus", proof.bus.is_some(), self.has_buses())?;
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
    #[must_use]
    pub const fn statement_digest(&self) -> [u8; 32] {
        self.statement
    }

    /// Fingerprint of the statement together with the choices one proof makes.
    pub(crate) fn run_digest(&self, run: &Run) -> Result<[u8; 32], DeclarationError> {
        if run.statement() != &self.statement {
            return Err(DeclarationError::ForeignRun);
        }

        let mut preimage = Preimage::new(b"p3-backend-contract/run/v1");
        preimage.bytes(&self.statement);
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

    use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
    use p3_baby_bear::BabyBear;
    use p3_keccak::Keccak256Hash;

    use super::*;
    use crate::contract::table::HeightRange;

    type F = BabyBear;
    type Declaration = MachineDeclaration<Keccak256Hash>;

    const TARGET: usize = 80;

    // Two tables alike in every count, telling the trace to do different things.
    struct Toy {
        doubling: bool,
    }

    impl<X> BaseAir<X> for Toy {
        fn width(&self) -> usize {
            4
        }
    }

    impl<AB: AirBuilder> Air<AB> for Toy {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let current = main.current(0).expect("the table has four columns");
            let next = main.next(0).expect("the table has four columns");
            let mut transition = builder.when_transition();
            if self.doubling {
                transition.assert_eq(current + current, next);
            } else {
                transition.assert_eq(current, next);
            }
        }
    }

    fn table(heights: HeightRange) -> TableDeclaration {
        TableDeclaration::from_constraints::<F, F, Toy>(&Toy { doubling: false }, heights)
    }

    fn declaration() -> Declaration {
        Declaration::new(
            Keccak256Hash,
            vec![table(HeightRange::new(2, 16))],
            1024,
            TARGET,
        )
        .unwrap()
    }

    #[test]
    fn a_statement_needs_at_least_one_table() {
        assert_eq!(
            Declaration::new(Keccak256Hash, vec![], 1024, TARGET).unwrap_err(),
            DeclarationError::NoTables
        );
    }

    #[test]
    fn a_budget_outside_the_ceiling_is_refused() {
        for budget in [0, MAX_PROOF_BYTES + 1] {
            assert_eq!(
                Declaration::new(
                    Keccak256Hash,
                    vec![table(HeightRange::new(2, 16))],
                    budget,
                    TARGET
                )
                .unwrap_err(),
                DeclarationError::BudgetOutOfRange {
                    found: budget,
                    limit: MAX_PROOF_BYTES,
                }
            );
        }
    }

    #[test]
    fn a_security_target_outside_the_ceiling_is_refused() {
        // Zero is refused because a statement that asks for nothing is the defect.
        for target in [0, MAX_SECURITY_BITS + 1] {
            assert_eq!(
                Declaration::new(
                    Keccak256Hash,
                    vec![table(HeightRange::new(2, 16))],
                    1024,
                    target
                )
                .unwrap_err(),
                DeclarationError::SecurityOutOfRange {
                    found: target,
                    limit: MAX_SECURITY_BITS,
                }
            );
        }
    }

    #[test]
    fn the_security_target_reaches_the_fingerprint() {
        // Two statements alike but for the target are not the same statement.
        let weak = declaration();
        let strong = Declaration::new(
            Keccak256Hash,
            vec![table(HeightRange::new(2, 16))],
            1024,
            TARGET + 1,
        )
        .unwrap();
        let run = weak.run(&[8], 0).unwrap();
        assert_eq!(
            strong.run_digest(&run).unwrap_err(),
            DeclarationError::ForeignRun
        );
    }

    #[test]
    fn a_table_that_fails_its_own_checks_is_refused() {
        let inverted = table(HeightRange::new(9, 8));
        assert_eq!(
            Declaration::new(
                Keccak256Hash,
                vec![table(HeightRange::new(2, 16)), inverted],
                1024,
                TARGET
            )
            .unwrap_err(),
            DeclarationError::EmptyHeightRange {
                table: 1,
                min: 9,
                max: 8,
            }
        );
    }

    #[test]
    fn a_height_outside_the_declared_range_is_refused() {
        let declaration = declaration();
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
        assert_eq!(
            declaration().run(&[8], 1000).unwrap_err(),
            DeclarationError::PowBitsAboveLimit {
                found: 1000,
                limit: MAX_POW_BITS,
            }
        );
    }

    #[test]
    fn a_run_of_one_statement_is_refused_by_another() {
        // The two tables agree on every count and differ only in what they assert.
        let doubling = TableDeclaration::from_constraints::<F, F, Toy>(
            &Toy { doubling: true },
            HeightRange::new(2, 16),
        );
        let plain = table(HeightRange::new(2, 16));
        assert_eq!(plain.columns(), doubling.columns());
        assert_eq!(plain.constraints(), doubling.constraints());

        let one = declaration();
        let other = Declaration::new(Keccak256Hash, vec![doubling], 1024, TARGET).unwrap();

        let run = one.run(&[8], 0).unwrap();
        assert_eq!(
            other.run_digest(&run).unwrap_err(),
            DeclarationError::ForeignRun
        );
        assert!(one.run_digest(&run).is_ok());
    }

    #[test]
    fn the_choices_a_run_makes_change_the_fingerprint() {
        let declaration = declaration();
        let low = declaration.run(&[8], 0).unwrap();
        let high = declaration.run(&[9], 0).unwrap();
        let ground = declaration.run(&[8], 1).unwrap();

        let digest = |run| declaration.run_digest(run).unwrap();
        assert_ne!(digest(&low), digest(&high));
        assert_ne!(digest(&low), digest(&ground));
    }

    #[test]
    fn a_table_that_disagrees_with_its_constraint_system_is_named() {
        // The declaration is read off one system and then checked against another.
        let plain = table(HeightRange::new(2, 16));
        assert_eq!(
            plain.disagreement::<F, F, Toy>(&Toy { doubling: false }),
            None
        );
        assert_eq!(
            plain.disagreement::<F, F, Toy>(&Toy { doubling: true }),
            Some("the constraints themselves")
        );
    }
}
