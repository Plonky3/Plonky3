//! A bit witness committed narrow, and opened through WHIR over the additive binary domain.
//!
//! ```text
//!     packing    bits  ->  the elements a commitment holds, by reinterpretation
//!     switch     a claim about the bits  ->  a claim about the packing
//!     opening    that claim  ->  discharged by WHIR
//! ```
//!
//! The packing is a bijection between bit strings and elements of the alphabet.
//!
//! One committed byte therefore holds eight trace bits, whatever the alphabet's width.
//!
//! No step of the commit path widens a bit into an element of its own.
//!
//! # Soundness
//!
//! Two errors compose by a union bound.
//!
//! ```text
//!     commitment   the proximity argument's own budget, its claim batching included
//!     reduction    one bit ring switch per claim
//! ```
//!
//! Both come back labelled, so a report says which one is short.
//!
//! The reductions run before the proximity argument names one codeword.
//!
//! So each is charged over every candidate the commitment still leaves open.

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::{PackedGf2, TowerLevel, Underlier};
use p3_challenger::fs::TranscriptField;
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::Field;
use p3_multilinear_util::point::Point;
use p3_security::multilinear::bit_ring_switch_tensors_term;
use p3_sumcheck::layout::{Layout, SuffixProver};
use p3_sumcheck::ring_switch::bits::{BitPacking, BitPackingView, BitRingSwitch};
use p3_sumcheck::{
    OpeningBatch, OpeningProtocol, PrescribedOpeningSecurity, PrescribedPointPcs, TableShape,
    TableSpec,
};
use p3_whir::{WhirDomain, WhirProver, WhirProverData};

use crate::boolean::{BitOpening, BitReadings, BooleanBackend, BooleanMultilinearPcs};
use crate::boolean_trace::BooleanTraceCommitment;
use crate::fold::ChallengeField;
use crate::packing::{Coordinates, PackedStack};
use crate::whir::error::BooleanWhirError;
use crate::whir::proof::BooleanWhirProof;
use crate::whir::shape::ProofShape;

/// The binding mode the committed layout uses.
///
/// One mode is fixed rather than chosen, so a commitment and its replay never disagree.
type Binding<EF> = SuffixProver<EF, EF>;

/// The proximity argument every packed claim is discharged against.
pub type BooleanWhirProver<EF, Dft, MT, Challenger> =
    WhirProver<EF, EF, Dft, MT, Challenger, Binding<EF>>;

/// Prover-side data retained between committing and opening.
pub type BooleanWhirData<EF, MT> = WhirProverData<EF, EF, MT, SuffixProver<EF, EF>>;

/// A batched trace of Boolean columns, discharged through the additive-domain proximity argument.
pub type BooleanWhirTracePcs<EF, Dft, MT, Challenger> =
    BooleanTraceCommitment<EF, BooleanWhirPcs<EF, Dft, MT, Challenger>>;

/// A commitment to a function from the hypercube to `{0, 1}`, opened through WHIR.
///
/// The committed object is a bit witness.
///
/// An opening answers for its multilinear extension at a point of the challenge field.
pub struct BooleanWhirPcs<EF: Field, Dft, MT, Challenger> {
    /// The proximity argument the packed multilinear is discharged against.
    inner: BooleanWhirProver<EF, Dft, MT, Challenger>,
    /// Variables the bit witness has, which is the packing's plus the absorbed ones.
    num_variables: usize,
}

impl<EF, Dft, MT, Challenger> BooleanWhirPcs<EF, Dft, MT, Challenger>
where
    EF: Field + TranscriptField + TowerLevel + Coordinates + Ord,
    Dft: WhirDomain<EF, EF>,
    MT: Mmcs<EF>,
{
    /// Wrap a proximity argument as a commitment to a bit witness of this many variables.
    ///
    /// # Errors
    ///
    /// Returns an error unless one element can absorb the witness's low coordinates.
    ///
    /// Returns an error unless the schedule commits exactly the elements the packing holds.
    pub fn new(
        inner: BooleanWhirProver<EF, Dft, MT, Challenger>,
        num_variables: usize,
    ) -> Result<Self, BooleanWhirError> {
        let packed = Self::packed_variables(num_variables)?;
        if inner.num_variables() != packed {
            return Err(BooleanWhirError::ConfigArity {
                expected: packed,
                actual: inner.num_variables(),
            });
        }
        Ok(Self {
            inner,
            num_variables,
        })
    }

    /// Variables the committed bit witness has, so `2^n` bits in all.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Bytes the commitment holds, which is one per eight bits of the witness.
    #[must_use]
    pub const fn committed_bytes(&self) -> usize {
        1 << (self.num_variables - 3)
    }

    /// Variables the packing keeps, the witness's less the ones one element absorbs.
    fn packed_variables(num_variables: usize) -> Result<usize, BooleanWhirError> {
        let absorbed = BitRingSwitch::<EF>::ABSORBED;
        num_variables
            .checked_sub(absorbed)
            .ok_or(BooleanWhirError::WitnessTooNarrow {
                needed: absorbed,
                actual: num_variables,
            })
    }

    /// The opening schedule this many surviving claims are discharged through.
    ///
    /// One table of one column, opened directly at one point per claim.
    fn protocol(&self, num_claims: usize) -> OpeningProtocol {
        OpeningProtocol::new(vec![TableSpec::new(
            TableShape::new(self.inner.num_variables(), 1),
            (0..num_claims)
                .map(|_| OpeningBatch::new(vec![0], Vec::new()))
                .collect(),
        )])
    }

    /// The reduction answering every reading one opening asks for.
    fn reduction(opening: &BitOpening<EF>) -> Result<BitRingSwitch<EF>, BooleanWhirError> {
        if opening.next {
            BitRingSwitch::with_successor(&opening.point, opening.row_variables)
        } else {
            BitRingSwitch::new(&opening.point)
        }
        .map_err(BooleanWhirError::Reduction)
    }

    /// Each point as an opening asking for the current reading alone.
    fn current_openings(points: &[Point<EF>]) -> Vec<BitOpening<EF>> {
        points
            .iter()
            .map(|point| BitOpening {
                point: point.clone(),
                row_variables: point.num_variables(),
                current: true,
                next: false,
            })
            .collect()
    }

    /// Check that every opening names the witness's variables and asks for a reading it has.
    fn check_openings(&self, openings: &[BitOpening<EF>]) -> Result<(), BooleanWhirError> {
        if openings.is_empty() {
            return Err(BooleanWhirError::NoPoints);
        }
        for (index, opening) in openings.iter().enumerate() {
            if opening.point.num_variables() != self.num_variables {
                return Err(BooleanWhirError::PointArity {
                    expected: self.num_variables,
                    actual: opening.point.num_variables(),
                });
            }
            if !opening.current && !opening.next {
                return Err(BooleanWhirError::NoReading { index });
            }
            if opening.row_variables > self.num_variables {
                return Err(BooleanWhirError::RowVariables {
                    index,
                    row_variables: opening.row_variables,
                    num_variables: self.num_variables,
                });
            }
        }
        Ok(())
    }

    /// The packed multilinear the commitment holds, borrowed from the retained table.
    ///
    /// # Panics
    ///
    /// Never for a table this scheme committed.
    ///
    /// Its alphabet is byte aligned and its height is a power of two.
    fn packing(prover_data: &BooleanWhirData<EF, MT>) -> BitPackingView<'_, EF> {
        BitPacking::from_packed(prover_data.table(0).poly(0))
            .expect("a committed table is a hypercube over a byte-aligned level")
    }
}

impl<EF, Dft, MT, Challenger> BooleanWhirPcs<EF, Dft, MT, Challenger>
where
    EF: Field + TranscriptField + TowerLevel + Coordinates + Ord + Send + Sync + ChallengeField<EF>,
    Dft: WhirDomain<EF, EF>,
    MT: Mmcs<EF>,
    Challenger: FieldChallenger<EF>
        + GrindingChallenger<Witness = EF>
        + CanSampleUniformBits<EF>
        + CanObserve<MT::Commitment>,
{
    /// Every labelled algebraic error one opening of this many claims charges.
    ///
    /// ```text
    ///     commitment   the proximity argument's own budget, its claim batching included
    ///     reduction    one bit ring switch per claim, batching its elements under one draw
    /// ```
    ///
    /// The two are independent draws, so they compose by a union bound.
    ///
    /// The reduction is charged over every candidate the commitment leaves open.
    ///
    /// Its challenges are drawn before the proximity argument names one of them.
    ///
    /// Nothing here covers hash or transcript collisions, which the caller supplies.
    ///
    /// # Returns
    ///
    /// Nothing when the proximity argument declines to price the schedule.
    #[must_use]
    pub fn readings_security(
        &self,
        num_claims: usize,
        successor_tensors: bool,
    ) -> Option<PrescribedOpeningSecurity> {
        let protocol = self.protocol(num_claims);
        let mut security = self.inner.prescribed_security(&protocol)?;
        // The tensor alone, or the tensor with carry and last.
        let num_tensors = if successor_tensors { 3 } else { 1 };
        // The reductions run before one candidate is named, so they pay for every one left open.
        security.charge_reduction(bit_ring_switch_tensors_term(
            num_claims,
            num_tensors,
            BitRingSwitch::<EF>::ABSORBED,
            self.inner.num_variables(),
            EF::bits(),
        ));
        Some(security)
    }

    /// The shape of every proof an opening of this many claims can produce.
    ///
    /// It covers the reductions as well as the opening, so a ceiling graded against it binds both.
    #[must_use]
    pub fn proof_shape(&self, num_claims: usize, successor_tensors: bool) -> ProofShape {
        ProofShape::of_bit_readings(&self.inner, num_claims, successor_tensors)
    }

    /// Open the bit witness with the readings every opening asks for, in one proof.
    ///
    /// No point needs prior transcript binding: each reduction binds its own.
    ///
    /// # Returns
    ///
    /// One set of readings per opening, in the order the openings were supplied.
    ///
    /// # Errors
    ///
    /// Before the transcript moves, an opening that names the wrong variables.
    ///
    /// An opening asking for no reading, or stepping within more rows than the witness has.
    #[allow(clippy::type_complexity)]
    pub fn open_readings(
        &self,
        prover_data: BooleanWhirData<EF, MT>,
        openings: &[BitOpening<EF>],
        challenger: &mut Challenger,
    ) -> Result<(Vec<BitReadings<EF>>, BooleanWhirProof<EF, MT>), BooleanWhirError> {
        self.check_openings(openings)?;
        // Every reduction is set up before any runs, so a refused one leaves the transcript alone.
        let reductions = openings
            .iter()
            .map(Self::reduction)
            .collect::<Result<Vec<_>, _>>()?;
        let packing = Self::packing(&prover_data);

        let mut readings = Vec::with_capacity(openings.len());
        let mut sent = Vec::with_capacity(openings.len());
        let mut surviving_points = Vec::with_capacity(openings.len());

        for (opening, reduction) in openings.iter().zip(&reductions) {
            let (proof, surviving_point, _) =
                tracing::info_span!("bit ring switch").in_scope(|| {
                    reduction.prove::<<EF as ChallengeField<EF>>::SumcheckRepr, _, _>(
                        &packing, challenger,
                    )
                });

            // The elements the reduction sends already hold the witness's readings.
            let current = opening
                .current
                .then(|| reduction.incoming_claim(&proof.tensor));
            let next = opening
                .next
                .then(|| reduction.successor_claim(&proof.tensor, proof.successor.as_ref()))
                .transpose()
                .map_err(BooleanWhirError::Reduction)?;

            readings.push(BitReadings { current, next });
            surviving_points.push(surviving_point);
            sent.push(proof);
        }

        // Each surviving value crosses the wire twice, and the closing check is that the two agree.
        // Every surviving point came out of a reduction's rounds, so all are bound already.
        let opening = self
            .inner
            .open_at(
                prover_data,
                &self.protocol(openings.len()),
                &surviving_points,
                challenger,
            )
            .map_err(BooleanWhirError::Commit)?;

        Ok((
            readings,
            BooleanWhirProof {
                reductions: sent,
                opening,
            },
        ))
    }

    /// Check one proof against the readings it claims at every opening.
    ///
    /// The commitment's binding is the caller's, replayed before this is reached.
    ///
    /// # Errors
    ///
    /// Before the transcript moves, a refused opening or a reading count that disagrees.
    ///
    /// After it, a false reading, a failed reduction, the commitment, or an unclosed claim.
    pub fn verify_readings(
        &self,
        commitment: &MT::Commitment,
        openings: &[BitOpening<EF>],
        readings: &[BitReadings<EF>],
        proof: &BooleanWhirProof<EF, MT>,
        challenger: &mut Challenger,
    ) -> Result<(), BooleanWhirError> {
        self.check_openings(openings)?;
        if readings.len() != openings.len() || proof.reductions.len() != openings.len() {
            return Err(BooleanWhirError::ClaimCount {
                expected: openings.len(),
                values: readings.len(),
                reductions: proof.reductions.len(),
            });
        }
        // The openings fix which readings are checked, so a missing one is never skipped.
        if let Some(index) = openings
            .iter()
            .zip(readings)
            .position(|(opening, reading)| {
                reading.current.is_some() != opening.current
                    || reading.next.is_some() != opening.next
            })
        {
            return Err(BooleanWhirError::ReadingShape { index });
        }
        let reductions = openings
            .iter()
            .map(Self::reduction)
            .collect::<Result<Vec<_>, _>>()?;

        let mut surviving_points = Vec::with_capacity(openings.len());
        let mut surviving_values = Vec::with_capacity(openings.len());
        for ((reduction, reading), sent) in reductions.iter().zip(readings).zip(&proof.reductions) {
            let (surviving_point, surviving_value) = reduction
                .verify_readings(sent, reading.current, reading.next, challenger)
                .map_err(BooleanWhirError::ReductionProof)?;
            surviving_points.push(surviving_point);
            surviving_values.push(surviving_value);
        }

        // One commitment opening answers for every surviving point at once.
        let evals = self
            .inner
            .verify_at(
                commitment,
                &proof.opening,
                &self.protocol(openings.len()),
                &surviving_points,
                challenger,
            )
            .map_err(BooleanWhirError::Opening)?;

        // Each reduction closes against its own opened value.
        if evals.len() != surviving_values.len() {
            return Err(BooleanWhirError::SurvivingClaim);
        }
        for (batch, &surviving) in evals.iter().zip(&surviving_values) {
            if batch.current().first() != Some(&surviving) {
                return Err(BooleanWhirError::SurvivingClaim);
            }
        }

        Ok(())
    }
}

impl<EF, Dft, MT, Challenger> BooleanMultilinearPcs<EF, Challenger>
    for BooleanWhirPcs<EF, Dft, MT, Challenger>
where
    EF: Field + TranscriptField + TowerLevel + Coordinates + Ord + Send + Sync + ChallengeField<EF>,
    Dft: WhirDomain<EF, EF>,
    MT: Mmcs<EF>,
    Challenger: FieldChallenger<EF>
        + GrindingChallenger<Witness = EF>
        + CanSampleUniformBits<EF>
        + CanObserve<MT::Commitment>,
{
    fn observe_commitment(&self, commitment: &Self::Commitment, challenger: &mut Challenger) {
        p3_commit::MultilinearPcs::<EF, Challenger>::observe_commitment(
            &self.inner,
            commitment,
            challenger,
        );
    }

    fn commit_bits<U: Underlier>(
        &self,
        bits: &[PackedGf2<U>],
        challenger: &mut Challenger,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::Error> {
        // The packing is one copy of the bits, so the witness is never swept for arithmetic.
        let stack = PackedStack::<PackedGf2<U>, EF>::from_columns(&[bits])
            .map_err(BooleanWhirError::Packing)?;
        if stack.column_num_variables() != self.inner.num_variables() {
            return Err(BooleanWhirError::WitnessArity {
                expected: self.inner.num_variables(),
                actual: stack.column_num_variables(),
            });
        }

        let folding = self.inner.round_folding_factor(0);
        let witness = Binding::<EF>::new_witness(vec![stack.into_table()], folding);
        p3_commit::MultilinearPcs::<EF, Challenger>::commit(&self.inner, witness, challenger)
            .map_err(BooleanWhirError::Commit)
    }

    fn open_readings(
        &self,
        prover_data: Self::ProverData,
        openings: &[BitOpening<EF>],
        challenger: &mut Challenger,
    ) -> Result<(Vec<BitReadings<EF>>, Self::Proof), Self::Error> {
        Self::open_readings(self, prover_data, openings, challenger)
    }

    fn verify_readings(
        &self,
        commitment: &Self::Commitment,
        openings: &[BitOpening<EF>],
        readings: &[BitReadings<EF>],
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        Self::verify_readings(self, commitment, openings, readings, proof, challenger)
    }

    fn readings_security(
        &self,
        num_claims: usize,
        successor_tensors: bool,
    ) -> Option<PrescribedOpeningSecurity> {
        Self::readings_security(self, num_claims, successor_tensors)
    }

    fn open_at_points(
        &self,
        prover_data: Self::ProverData,
        points: &[Point<EF>],
        challenger: &mut Challenger,
    ) -> Result<(Vec<EF>, Self::Proof), Self::Error> {
        let (readings, proof) =
            self.open_readings(prover_data, &Self::current_openings(points), challenger)?;
        let values = readings
            .into_iter()
            .map(|reading| {
                reading
                    .current
                    .expect("an opening asking for the current reading carries it")
            })
            .collect();
        Ok((values, proof))
    }

    fn verify_at_points(
        &self,
        commitment: &Self::Commitment,
        points: &[Point<EF>],
        values: &[EF],
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        let readings: Vec<BitReadings<EF>> = values
            .iter()
            .map(|&value| BitReadings {
                current: Some(value),
                next: None,
            })
            .collect();
        self.verify_readings(
            commitment,
            &Self::current_openings(points),
            &readings,
            proof,
            challenger,
        )
    }
}

impl<EF, Dft, MT, Challenger> BooleanBackend<EF> for BooleanWhirPcs<EF, Dft, MT, Challenger>
where
    EF: Field + TranscriptField + TowerLevel + Coordinates + Ord + Send + Sync,
    Dft: WhirDomain<EF, EF>,
    MT: Mmcs<EF>,
{
    type Commitment = MT::Commitment;
    type ProverData = BooleanWhirData<EF, MT>;
    type Proof = BooleanWhirProof<EF, MT>;
    type Error = BooleanWhirError;

    fn num_variables(&self) -> usize {
        self.num_variables
    }
}
