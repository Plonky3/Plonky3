//! A commitment to a function from the hypercube to `{0, 1}`, opened at any point.
//!
//! ```text
//!     commit   a bit witness, bit-sliced
//!     open     the multilinear extension at a point of the challenge field
//! ```
//!
//! # Three pieces, one interface
//!
//! ```text
//!     packing    bits  ->  the elements a commitment holds, by reinterpretation
//!     switch     a claim about the bits  ->  a claim about the packing
//!     opening    that claim  ->  discharged against the commitment
//! ```
//!
//! # Booleanity is free
//!
//! The packing is a bijection between bit strings and elements of the level.
//! Every bit pattern is an element, and every element is some bit pattern.
//!
//! A commitment to a packed multilinear is therefore a commitment to a bit witness.
//! No commitment exists to a function the hypercube sends outside `{0, 1}`.
//!
//! There is no range check here because there is nothing one could rule out.
//!
//! # One opening, not one per claim
//!
//! A claim names a point and the readings asked for there:
//!
//! ```text
//!     current   the witness at the point
//!     next      the witness one row further on, the last row repeating
//! ```
//!
//! One reduction answers every reading of its claim, and leaves one claim about the packing.
//! A successor view whose rows outrun one element sends two more elements, not a second sumcheck.
//!
//! Every claim so reduces to one about the same packing, at a point of its own.
//! The commitment answers for all of them in one call, one batch per claim.
//!
//! Batching those `k` claims under powers of one challenge is the commitment's own step.
//! It draws that challenge after all `k` values are bound, and charges it as its own term.
//!
//! Each reduction then closes by comparing its surviving value to the one opened.
//!
//! # What binds what
//!
//! Committing binds the root, and the verifying side replays that binding itself.
//! The two sides therefore start from one sponge state.
//!
//! A protocol holding several commitments binds each of them once, in its own order.
//! So the binding is a call of its own rather than a step inside verification.
//!
//! The reduction then binds the opening point itself, before it sends anything.
//! A caller therefore owes no binding of its own, unlike a bare prescribed-point opening.
//!
//! The point the commitment is opened at comes out of the reduction's own rounds.
//!
//! # Soundness
//!
//! Two errors compose by a union bound.
//!
//! ```text
//!     commitment   the packed multilinear's own budget, its k-claim batching included
//!     reduction    (d_log + K - 1 + 2 l') / |EF| per claim
//! ```
//!
//! `K` counts the elements a reduction sends, batched under powers of one challenge.
//! It is three when a successor view's rows outrun one element, and one otherwise.
//!
//! Neither is subtracted from the other.
//!
//! Both come back labelled, so a report says which one is short.

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_dft::EncodableLevel;
use p3_binary_field::{PackedGf2, TowerLevel, Underlier};
use p3_challenger::fs::TranscriptField;
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, MultilinearPcs};
use p3_field::Field;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_security::SecurityTerm;
use p3_security::multilinear::{bit_ring_switch_tensors_term, bit_ring_switch_term};
use p3_sumcheck::layout::{Layout, SuffixProver};
use p3_sumcheck::ring_switch::bits::{
    BitPacking, BitRingSwitch, BitRingSwitchProof, BitRingSwitchProofError,
};
use p3_sumcheck::{
    OpeningBatch, OpeningProtocol, PrescribedOpeningSecurity, PrescribedPointPcs, TableShape,
    TableSpec,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::error::BinaryPcsError;
use crate::fold::{ChallengeField, FoldAlphabet};
use crate::packing::{Coordinates, PackError, PackedStack};
use crate::params::{BinaryPcsConfig, BinaryPcsConfigError};
use crate::pcs::BinaryPcs;
use crate::proof::BinaryPcsProof;
use crate::prover::BinaryPcsProverData;

/// A commitment to a function from the hypercube to `{0, 1}`.
///
/// The committed object is a bit witness.
/// An opening answers for its multilinear extension at a point of the challenge field.
///
/// # What a proving system pays on top
///
/// A batched prover lends its trace back as a borrowed table of base-field cells.
///
/// A bit witness has none to lend, so an integration holds the table and packs at commit time.
///
/// The base field then has to be one the challenge field extends.
///
/// The narrowest such level of the tower is `GF(2^8)`.
///
/// This scheme also draws its challenges from the field its own elements live in.
///
/// So one cell per bit at the full challenge width is what the borrow costs today.
///
/// Constraints would also run wider if the base field packed bit-sliced.
///
/// The packing trait forbids it, casting a packed value to an array of scalars unchanged.
///
/// Bit-slicing is a compression, not a reinterpretation.
pub trait BooleanMultilinearPcs<EF, Challenger> {
    /// Succinct binding commitment sent to the verifier.
    type Commitment;
    /// Prover-side data retained between commitment and opening.
    type ProverData;
    /// Opening proof checked by the verifier.
    type Proof;
    /// Why a commitment or an opening was refused.
    type Error;

    /// Variables the committed function has, so `2^n` bits in all.
    fn num_variables(&self) -> usize;

    /// Bind a commitment into the transcript, as committing to one does.
    ///
    /// A prover binds while producing its commitment, and a verifier never produces one.
    ///
    /// Routing both sides through this call is what keeps the two sponges in step.
    fn observe_commitment(&self, commitment: &Self::Commitment, challenger: &mut Challenger);

    /// Commit to a bit witness supplied bit-sliced, lane `j` of block `b` being bit `d*b + j`.
    ///
    /// # Errors
    ///
    /// Returns an error unless the witness covers exactly the committed hypercube.
    fn commit_bits<U: Underlier>(
        &self,
        bits: &[PackedGf2<U>],
        challenger: &mut Challenger,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::Error>;

    /// Open the multilinear extension at every point, in one proof.
    ///
    /// No point needs prior transcript binding: each reduction binds its own.
    ///
    /// # Returns
    ///
    /// One value per point, in the order the points were supplied.
    ///
    /// # Errors
    ///
    /// Returns an error unless every point names the committed function's variables.
    fn open_at_points(
        &self,
        prover_data: Self::ProverData,
        points: &[Point<EF>],
        challenger: &mut Challenger,
    ) -> Result<(Vec<EF>, Self::Proof), Self::Error>;

    /// Check one proof against the values it claims at every point.
    ///
    /// The commitment's binding is the caller's, replayed before this is reached.
    ///
    /// # Errors
    ///
    /// Returns an error if a claim, a reduction, the commitment or the fold fails.
    fn verify_at_points(
        &self,
        commitment: &Self::Commitment,
        points: &[Point<EF>],
        values: &[EF],
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error>;
}

/// One claim about the bit witness: a point, and the readings asked for there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitOpening<EF> {
    /// Point over every variable of the bit witness, its row coordinates trailing.
    pub point: Point<EF>,
    /// Trailing coordinates addressing rows; the successor view steps within them.
    pub row_variables: usize,
    /// Whether the witness at the point is asked for.
    pub current: bool,
    /// Whether the witness one row further on, the last row repeating, is asked for.
    pub next: bool,
}

/// The readings one [`BitOpening`] asked for, and nothing where it asked for nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitReadings<EF> {
    /// The witness at the point.
    pub current: Option<EF>,
    /// The witness one row further on, the last row repeating.
    pub next: Option<EF>,
}

/// A Boolean commitment built from the packing, the bit ring switch and the binary PCS.
///
/// The committed elements hold `d` bits each.
/// The codeword is therefore `d` times shorter than an element-per-bit one.
pub struct BooleanPcs<EF: EncodableLevel, MT, MX> {
    /// The commitment the packed multilinear is discharged against.
    inner: BinaryPcs<EF, EF, MT, MX>,
    /// Schedule the commitment was built from, which prices its own openings.
    config: BinaryPcsConfig,
    /// Variables the bit witness has, which is the packing's plus the absorbed ones.
    num_variables: usize,
}

impl<EF, MT, MX> BooleanPcs<EF, MT, MX>
where
    EF: ChallengeField<EF>
        + EncodableLevel
        + TranscriptField
        + TowerLevel
        + FoldAlphabet<EF>
        + Coordinates,
    MT: Mmcs<EF>,
    MX: Mmcs<EF, Error = MT::Error>,
{
    /// Build a Boolean commitment over a bit witness of `num_variables` variables.
    ///
    /// # Errors
    ///
    /// Returns an error unless the schedule was derived for `(EF, EF)`.
    /// Returns an error unless the schedule commits exactly the elements the packing holds.
    pub fn new(
        config: BinaryPcsConfig,
        mmcs: MT,
        round_mmcs: MX,
        num_variables: usize,
    ) -> Result<Self, BooleanPcsError<EF, MT::Error>> {
        let packed = Self::packed_variables(num_variables)?;
        if config.num_variables() != packed {
            return Err(BooleanPcsError::ConfigArity {
                expected: packed,
                actual: config.num_variables(),
            });
        }
        Ok(Self {
            inner: BinaryPcs::new(config, mmcs, round_mmcs).map_err(BooleanPcsError::Config)?,
            config,
            num_variables,
        })
    }

    /// Variables the packing keeps, the witness's less the ones one element absorbs.
    fn packed_variables(num_variables: usize) -> Result<usize, BooleanPcsError<EF, MT::Error>> {
        let absorbed = BitRingSwitch::<EF>::ABSORBED;
        num_variables
            .checked_sub(absorbed)
            .ok_or(BooleanPcsError::WitnessTooNarrow {
                needed: absorbed,
                actual: num_variables,
            })
    }

    /// Variables the committed bit witness has, so `2^n` bits in all.
    ///
    /// The same number the trait reports, reachable without naming a challenger.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.num_variables
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

    /// Soundness cost of reducing this many bit claims to claims about the packing.
    ///
    /// One reduction per claim, none of them sharing a challenge with another.
    #[must_use]
    pub fn reduction_security(&self, num_claims: usize) -> SecurityTerm {
        bit_ring_switch_term(
            num_claims,
            BitRingSwitch::<EF>::ABSORBED,
            self.inner.num_variables(),
            EF::bits(),
        )
    }

    /// Every labelled algebraic error one opening of this many points charges.
    ///
    /// A point asks for the current reading alone, so no reduction sends successor elements.
    #[must_use]
    pub fn opening_security(&self, num_claims: usize) -> PrescribedOpeningSecurity {
        self.readings_security(num_claims, false)
    }

    /// Every labelled algebraic error one opening of this many claims charges.
    ///
    /// ```text
    ///     commitment   the packed multilinear's own budget, its k-claim batching included
    ///     reduction    one bit ring switch per claim, batching K elements under alpha
    /// ```
    ///
    /// `K` is three when `successor_tensors` holds, one otherwise.
    /// A reduction sends carry and last when its successor view's rows outrun one element.
    /// Charging three to every claim once any one of them sends them is an upper bound.
    ///
    /// The commitment prices batching its own `k` claims under powers of one challenge.
    /// Each surviving claim is then closed by an equality, which carries no error.
    ///
    /// The two are independent draws, so a caller composes them by a union bound.
    /// Nothing here covers hash or transcript collisions, which the caller supplies.
    #[must_use]
    pub fn readings_security(
        &self,
        num_claims: usize,
        successor_tensors: bool,
    ) -> PrescribedOpeningSecurity {
        // The tensor alone, or the tensor with carry and last.
        let num_tensors = if successor_tensors { 3 } else { 1 };
        PrescribedOpeningSecurity {
            terms: vec![
                self.config.security_regime().opening_term(num_claims),
                bit_ring_switch_tensors_term(
                    num_claims,
                    num_tensors,
                    BitRingSwitch::<EF>::ABSORBED,
                    self.inner.num_variables(),
                    EF::bits(),
                ),
            ],
            // Unique decoding fixes one candidate polynomial at commitment time.
            log2_max_candidates: 0.0,
        }
    }

    /// The packed multilinear the commitment holds, read back out of the retained table.
    ///
    /// # Panics
    ///
    /// Never for a table this scheme committed, since the read-back refuses only these:
    ///
    /// ```text
    ///     sub-byte level   excluded by the encodable bound, which starts at a byte
    ///     no hypercube     excluded by the layout, whose tables are power-of-two
    /// ```
    fn packing(prover_data: &BinaryPcsProverData<EF, EF, MT>) -> BitPacking<EF> {
        let packed = Poly::new(prover_data.table(0).poly(0).as_slice().to_vec());
        BitPacking::from_packed(packed)
            .expect("a committed table is a hypercube over a byte-aligned level")
    }

    /// Check that every opening names the witness's variables and asks for a reading it has.
    fn check_openings(
        &self,
        openings: &[BitOpening<EF>],
    ) -> Result<(), BooleanPcsError<EF, MT::Error>> {
        if openings.is_empty() {
            return Err(BooleanPcsError::NoPoints);
        }
        for (index, opening) in openings.iter().enumerate() {
            if opening.point.num_variables() != self.num_variables {
                return Err(BooleanPcsError::PointArity {
                    expected: self.num_variables,
                    actual: opening.point.num_variables(),
                });
            }
            if !opening.current && !opening.next {
                return Err(BooleanPcsError::NoReading { index });
            }
            if opening.row_variables > self.num_variables {
                return Err(BooleanPcsError::RowVariables {
                    index,
                    row_variables: opening.row_variables,
                    num_variables: self.num_variables,
                });
            }
        }
        Ok(())
    }

    /// The reduction answering every reading one opening asks for.
    ///
    /// Only the successor reading needs the successor view.
    /// An opening asking for the current reading alone plays the plain reduction.
    fn reduction(
        opening: &BitOpening<EF>,
    ) -> Result<BitRingSwitch<EF>, BooleanPcsError<EF, MT::Error>> {
        if opening.next {
            BitRingSwitch::with_successor(&opening.point, opening.row_variables)
        } else {
            BitRingSwitch::new(&opening.point)
        }
        .map_err(BooleanPcsError::Reduction)
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

    /// Open the bit witness with the readings every opening asks for, in one proof.
    ///
    /// No point needs prior transcript binding: each reduction binds its own.
    ///
    /// # Returns
    ///
    /// One set of readings per opening, in the order the openings were supplied.
    /// Each holds a value exactly where its opening asks for one.
    ///
    /// # Errors
    ///
    /// Before the transcript moves, an opening that:
    ///
    /// - does not name the committed function's variables,
    /// - asks for no reading,
    /// - or steps within more rows than the witness has.
    #[allow(clippy::type_complexity)]
    pub fn open_readings<Challenger>(
        &self,
        prover_data: BinaryPcsProverData<EF, EF, MT>,
        openings: &[BitOpening<EF>],
        challenger: &mut Challenger,
    ) -> Result<(Vec<BitReadings<EF>>, BooleanProof<EF, MT, MX>), BooleanPcsError<EF, MT::Error>>
    where
        Challenger: FieldChallenger<EF>
            + GrindingChallenger<Witness = EF>
            + CanSampleUniformBits<EF>
            + CanObserve<MT::Commitment>
            + CanObserve<MX::Commitment>,
    {
        self.check_openings(openings)?;
        // Every reduction is set up before any runs, so a refused one leaves the transcript alone.
        let reductions = openings
            .iter()
            .map(Self::reduction)
            .collect::<Result<Vec<_>, _>>()?;
        let packing = Self::packing(&prover_data);

        // One reduction per opening, each leaving one claim about the same packing.
        let mut readings = Vec::with_capacity(openings.len());
        let mut sent = Vec::with_capacity(openings.len());
        let mut surviving_points = Vec::with_capacity(openings.len());

        for (opening, reduction) in openings.iter().zip(&reductions) {
            let (proof, surviving_point, _) = reduction.prove(&packing, challenger);

            // The elements the reduction sends already hold the witness's readings.
            // Read by columns they are the claimed values, so neither costs a pass of its own.
            let current = opening
                .current
                .then(|| reduction.incoming_claim(&proof.tensor));
            let next = opening
                .next
                .then(|| reduction.successor_claim(&proof.tensor, proof.successor.as_ref()))
                .transpose()
                .map_err(BooleanPcsError::Reduction)?;

            readings.push(BitReadings { current, next });
            surviving_points.push(surviving_point);
            sent.push(proof);
        }

        // The surviving values never cross the wire: a verifier recomputes its own.
        // Every surviving point came out of a reduction's rounds, so all are bound already.
        let opening = self
            .inner
            .try_open_at(
                prover_data,
                &self.protocol(openings.len()),
                &surviving_points,
                challenger,
            )
            .map_err(BooleanPcsError::Commitment)?;

        Ok((
            readings,
            BooleanProof {
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
    /// Before the transcript moves:
    ///
    /// - An opening the proving side refuses.
    /// - Reading or reduction counts that differ from the opening count.
    /// - A reading present where its opening asks for none, or missing where it asks for one.
    ///
    /// After it: a false reading, a failed reduction, the commitment, or an unclosed claim.
    pub fn verify_readings<Challenger>(
        &self,
        commitment: &MT::Commitment,
        openings: &[BitOpening<EF>],
        readings: &[BitReadings<EF>],
        proof: &BooleanProof<EF, MT, MX>,
        challenger: &mut Challenger,
    ) -> Result<(), BooleanPcsError<EF, MT::Error>>
    where
        Challenger: FieldChallenger<EF>
            + GrindingChallenger<Witness = EF>
            + CanSampleUniformBits<EF>
            + CanObserve<MT::Commitment>
            + CanObserve<MX::Commitment>,
    {
        self.check_openings(openings)?;
        if readings.len() != openings.len() || proof.reductions.len() != openings.len() {
            return Err(BooleanPcsError::ClaimCount {
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
            return Err(BooleanPcsError::ReadingShape { index });
        }
        let reductions = openings
            .iter()
            .map(Self::reduction)
            .collect::<Result<Vec<_>, _>>()?;

        // Each reduction turns the readings about the bits into one claim about the packing.
        let mut surviving_points = Vec::with_capacity(openings.len());
        let mut surviving_values = Vec::with_capacity(openings.len());
        for ((reduction, reading), sent) in reductions.iter().zip(readings).zip(&proof.reductions) {
            let (surviving_point, surviving_value) = reduction
                .verify_readings(sent, reading.current, reading.next, challenger)
                .map_err(BooleanPcsError::ReductionProof)?;
            surviving_points.push(surviving_point);
            surviving_values.push(surviving_value);
        }

        // One commitment opening answers for every surviving point at once.
        // It pins each opened value to the committed polynomial at that point.
        let evals = self
            .inner
            .verify_at(
                commitment,
                &proof.opening,
                &self.protocol(openings.len()),
                &surviving_points,
                challenger,
            )
            .map_err(BooleanPcsError::Commitment)?;

        // Each reduction closes against its own opened value.
        //
        //     reduction i  ->  t'(r'_i) = s'_i, the claim it left behind
        //     opening      ->  the value the committed polynomial takes at r'_i
        //
        // Folding them under a second challenge would add (k - 1) / |EF| for the same k.
        if evals.len() != surviving_values.len() {
            return Err(BooleanPcsError::SurvivingClaim);
        }
        for (batch, &surviving) in evals.iter().zip(&surviving_values) {
            if batch.current().first() != Some(&surviving) {
                return Err(BooleanPcsError::SurvivingClaim);
            }
        }

        Ok(())
    }
}

impl<EF, MT, MX, Challenger> BooleanMultilinearPcs<EF, Challenger> for BooleanPcs<EF, MT, MX>
where
    EF: ChallengeField<EF>
        + EncodableLevel
        + TranscriptField
        + TowerLevel
        + FoldAlphabet<EF>
        + Coordinates,
    MT: Mmcs<EF>,
    MX: Mmcs<EF, Error = MT::Error>,
    Challenger: FieldChallenger<EF>
        + GrindingChallenger<Witness = EF>
        + CanSampleUniformBits<EF>
        + CanObserve<MT::Commitment>
        + CanObserve<MX::Commitment>,
{
    type Commitment = MT::Commitment;
    type ProverData = BinaryPcsProverData<EF, EF, MT>;
    type Proof = BooleanProof<EF, MT, MX>;
    type Error = BooleanPcsError<EF, MT::Error>;

    fn num_variables(&self) -> usize {
        Self::num_variables(self)
    }

    fn observe_commitment(&self, commitment: &Self::Commitment, challenger: &mut Challenger) {
        self.inner.observe_commitment(commitment, challenger);
    }

    fn commit_bits<U: Underlier>(
        &self,
        bits: &[PackedGf2<U>],
        challenger: &mut Challenger,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::Error> {
        // The packing is one copy of the bits, so the witness is never swept for arithmetic.
        let stack = PackedStack::<PackedGf2<U>, EF>::from_columns(&[bits])?;
        if stack.column_num_variables() != self.inner.num_variables() {
            return Err(BooleanPcsError::WitnessArity {
                expected: self.inner.num_variables(),
                actual: stack.column_num_variables(),
            });
        }

        let witness = SuffixProver::<EF, EF>::new_witness(vec![stack.into_table()], 0);
        self.inner
            .commit(witness, challenger)
            .map_err(BooleanPcsError::Commitment)
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

/// One Boolean opening: the reductions, and the commitment opening that discharges them.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "EF: TowerLevel, MT::Commitment: Serialize, MT::MultiProof: Serialize, MX::Commitment: Serialize, MX::MultiProof: Serialize",
    deserialize = "EF: TowerLevel, MT::Commitment: Deserialize<'de>, MT::MultiProof: Deserialize<'de>, MX::Commitment: Deserialize<'de>, MX::MultiProof: Deserialize<'de>"
))]
pub struct BooleanProof<EF: Field, MT: Mmcs<EF>, MX: Mmcs<EF>> {
    /// One bit-alphabet ring switch per claim, in the order the claims came in.
    pub reductions: Vec<BitRingSwitchProof<EF>>,
    /// The single commitment opening that discharges every packed claim.
    pub opening: BinaryPcsProof<EF, EF, MT, MX>,
}

/// Why a Boolean commitment or opening was refused.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum BooleanPcsError<EF, MmcsError> {
    /// The witness is narrower than the coordinates one element absorbs.
    #[error("a bit witness of {actual} variables cannot absorb {needed} into one element")]
    WitnessTooNarrow {
        /// Coordinates one element absorbs.
        needed: usize,
        /// Variables the witness has.
        actual: usize,
    },

    /// The schedule was not derived for the level this commitment runs over.
    #[error(transparent)]
    Config(BinaryPcsConfigError),

    /// The commitment schedule does not commit the elements the packing holds.
    #[error("the schedule commits {actual} variables, the packing holds {expected}")]
    ConfigArity {
        /// Variables the packing holds.
        expected: usize,
        /// Variables the schedule commits.
        actual: usize,
    },

    /// The witness does not cover the committed hypercube.
    #[error("the packed witness has {actual} variables, expected {expected}")]
    WitnessArity {
        /// Variables the commitment holds.
        expected: usize,
        /// Variables the witness packs to.
        actual: usize,
    },

    /// No opening point was supplied, so there is nothing to prove.
    #[error("an opening needs at least one point")]
    NoPoints,

    /// The claim counts on the two sides of an opening disagree.
    #[error("{expected} points against {values} values and {reductions} reductions")]
    ClaimCount {
        /// Points supplied.
        expected: usize,
        /// Values supplied.
        values: usize,
        /// Reductions the proof carries.
        reductions: usize,
    },

    /// The opening point does not name the committed function's variables.
    #[error("the opening point names {actual} variables, expected {expected}")]
    PointArity {
        /// Variables the committed function has.
        expected: usize,
        /// Variables the point names.
        actual: usize,
    },

    /// An opening asks for neither reading, so nothing at its point would be checked.
    #[error("opening {index} asks for neither the current nor the successor reading")]
    NoReading {
        /// Position of the opening among those supplied.
        index: usize,
    },

    /// The successor view steps within more coordinates than the witness has.
    #[error("opening {index} names {row_variables} row variables, the witness {num_variables}")]
    RowVariables {
        /// Position of the opening among those supplied.
        index: usize,
        /// Row coordinates the opening names.
        row_variables: usize,
        /// Variables the committed function has.
        num_variables: usize,
    },

    /// A reading is present where its opening asks for none, or missing where it asks for one.
    #[error("the readings at opening {index} are not the ones it asks for")]
    ReadingShape {
        /// Position of the opening among those supplied.
        index: usize,
    },

    /// The bits could not be packed into the elements a commitment holds.
    #[error(transparent)]
    Packing(#[from] PackError),

    /// The reduction could not be set up over the point supplied.
    #[error(transparent)]
    Reduction(p3_sumcheck::ring_switch::bits::BitRingSwitchError),

    /// The reduction's own proof was rejected.
    #[error(transparent)]
    ReductionProof(BitRingSwitchProofError),

    /// The commitment refused the opening.
    #[error(transparent)]
    Commitment(BinaryPcsError<EF, MmcsError>),

    /// The commitment opened a value the reduction did not leave behind.
    ///
    /// Without this the reduction would prove a claim about some other polynomial.
    #[error("the opened value is not the claim the reduction left behind")]
    SurvivingClaim,
}

#[cfg(test)]
mod tests {
    use p3_binary_field::{BinaryField64, BinaryField128, Gf2, PackedGf2x64};
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::params::BinaryPcsParams;
    use crate::test_util::{MyChallenger, MyMmcs, challenger, mmcs};

    /// A verifying transcript with the commitment's binding already replayed.
    fn replayed(
        pcs: &BooleanPcs<EF, MyMmcs, MyMmcs>,
        commitment: &<MyMmcs as Mmcs<EF>>::Commitment,
    ) -> MyChallenger {
        // The surrounding protocol owns this binding, so a test plays its part.
        let mut challenger = challenger();
        pcs.observe_commitment(commitment, &mut challenger);
        challenger
    }

    type EF = BinaryField128;

    /// Blocks a fixture of the given log bit count holds.
    const fn blocks(log_bits: usize) -> usize {
        1 << (log_bits - 6)
    }

    /// A Boolean commitment over a witness of `log_bits` variables.
    fn boolean_pcs(log_bits: usize) -> BooleanPcs<EF, MyMmcs, MyMmcs> {
        let params = BinaryPcsParams {
            log_inv_rate: 2,
            pow_bits: 0,
            security_level: 40,
        };
        // The packing absorbs seven variables, so the commitment holds the rest.
        let config = BinaryPcsConfig::try_new::<EF, EF>(log_bits - 7, params).unwrap();
        BooleanPcs::new(config, mmcs(), mmcs(), log_bits).unwrap()
    }

    /// A random bit-sliced witness of the given log bit count.
    fn witness(seed: u64, log_bits: usize) -> Vec<PackedGf2x64> {
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..blocks(log_bits))
            .map(|_| PackedGf2x64::new(rng.random::<u64>()))
            .collect()
    }

    /// The witness as a multilinear over every variable, one element per bit.
    ///
    /// The definition of what an opening answers for, with no packing in it.
    fn embedded(bits: &[PackedGf2x64]) -> Poly<EF> {
        Poly::new(
            bits.iter()
                .flat_map(|block| {
                    (0..PackedGf2x64::WIDTH).map(move |lane| {
                        if block.get(lane) == Gf2::ONE {
                            EF::ONE
                        } else {
                            EF::ZERO
                        }
                    })
                })
                .collect::<Vec<EF>>(),
        )
    }

    #[test]
    fn a_boolean_opening_round_trips_at_the_value_the_bits_define() {
        // Invariant: an opening answers for the bit witness, not for its packing.
        //
        // Fixture state: 2^13 bits, of which one element absorbs 2^7.
        //
        //     witness     128 blocks of 64 lanes
        //     commitment   64 elements, so six variables
        //
        // The value is compared against the one-element-per-bit multilinear.
        // That reference shares nothing with the packing or with the reduction.
        const LOG_BITS: usize = 13;

        let bits = witness(0xB001, LOG_BITS);
        let pcs = boolean_pcs(LOG_BITS);
        assert_eq!(pcs.num_variables(), LOG_BITS);

        let mut prover_chal = challenger();
        let (commitment, data) = pcs.commit_bits(&bits, &mut prover_chal).unwrap();

        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB002), LOG_BITS);
        let points = alloc::vec![point.clone()];
        let (values, proof) = pcs.open_at_points(data, &points, &mut prover_chal).unwrap();

        assert_eq!(values, alloc::vec![embedded(&bits).eval_base(&point)]);

        pcs.verify_at_points(
            &commitment,
            &points,
            &values,
            &proof,
            &mut replayed(&pcs, &commitment),
        )
        .unwrap();
    }

    #[test]
    fn several_points_cost_one_commitment_opening() {
        // Invariant: every surviving claim is discharged by one opening.
        //
        //     - reductions   one per point, each binding its own point
        //     - opening      one, carrying one opened value per surviving point
        //     - closing      one equality per reduction, against its own opened value
        //
        // Batching the four claims is the commitment's own step, charged in its own term.
        const LOG_BITS: usize = 13;
        const NUM_POINTS: usize = 4;

        let bits = witness(0xB00F, LOG_BITS);
        let pcs = boolean_pcs(LOG_BITS);
        let mut rng = SmallRng::seed_from_u64(0xB010);
        let points: Vec<Point<EF>> = (0..NUM_POINTS)
            .map(|_| Point::<EF>::rand(&mut rng, LOG_BITS))
            .collect();

        let mut prover_chal = challenger();
        let (commitment, data) = pcs.commit_bits(&bits, &mut prover_chal).unwrap();
        let (values, proof) = pcs.open_at_points(data, &points, &mut prover_chal).unwrap();

        // One reduction per point, and exactly one commitment opening for all of them.
        assert_eq!(proof.reductions.len(), NUM_POINTS);

        // Each value is the bit witness at its own point.
        let reference = embedded(&bits);
        for (point, &value) in points.iter().zip(&values) {
            assert_eq!(value, reference.eval_base(point));
        }

        pcs.verify_at_points(
            &commitment,
            &points,
            &values,
            &proof,
            &mut replayed(&pcs, &commitment),
        )
        .unwrap();

        // One wrong value among four is rejected, wherever it sits.
        for index in 0..NUM_POINTS {
            let mut tampered = values.clone();
            tampered[index] += EF::ONE;
            assert!(
                pcs.verify_at_points(
                    &commitment,
                    &points,
                    &tampered,
                    &proof,
                    &mut replayed(&pcs, &commitment)
                )
                .is_err(),
                "value {index}"
            );
        }

        // Two labelled terms, and no third for a second round of batching.
        let evidence = pcs.opening_security(NUM_POINTS);
        let labels: Vec<&str> = evidence.terms.iter().map(|term| term.label).collect();
        assert_eq!(
            labels,
            alloc::vec![
                p3_security::binary::BINARY_PCS_OPENING_LABEL,
                p3_security::BIT_RING_SWITCH_LABEL
            ]
        );

        // Four reductions of seven draws and six rounds each.
        assert_eq!(
            evidence.terms[1].bits.bits(),
            p3_security::bit_ring_switch_error(NUM_POINTS, 7, 6, 128).bits()
        );
    }

    #[test]
    fn a_false_value_is_rejected() {
        // The element's column reading is what ties the claim to the committed bits.
        const LOG_BITS: usize = 13;

        let bits = witness(0xFA15, LOG_BITS);
        let pcs = boolean_pcs(LOG_BITS);
        let mut prover_chal = challenger();
        let (commitment, data) = pcs.commit_bits(&bits, &mut prover_chal).unwrap();

        let points = alloc::vec![Point::<EF>::rand(
            &mut SmallRng::seed_from_u64(0xFA16),
            LOG_BITS
        )];
        let (values, proof) = pcs.open_at_points(data, &points, &mut prover_chal).unwrap();

        let false_values = alloc::vec![values[0] + EF::ONE];
        let err = pcs
            .verify_at_points(
                &commitment,
                &points,
                &false_values,
                &proof,
                &mut replayed(&pcs, &commitment),
            )
            .unwrap_err();
        assert!(matches!(err, BooleanPcsError::ReductionProof(_)), "{err:?}");
    }

    #[test]
    fn a_non_first_reduction_over_another_witness_leaves_a_claim_the_commitment_does_not_open() {
        // Invariant: the closing comparison is what ties the reductions to the commitment.
        //
        // Mutation: reduce the middle claim over witness A and open witness B.
        //
        //     - transcript  A's reduction runs on the sponge that absorbed B's root
        //     - reduction   replays and accepts, being a true proof about A
        //     - commitment  accepts, being a true opening of B
        //     - closing     t'_A(r') != t'_B(r')  ->  the claim does not survive
        //
        // A fresh sponge would split the transcripts at the first draw and pin nothing.
        const LOG_BITS: usize = 13;

        let pcs = boolean_pcs(LOG_BITS);
        const FORGED: usize = 1;
        let mut rng = SmallRng::seed_from_u64(0x0A11);
        let points = (0..3)
            .map(|_| Point::<EF>::rand(&mut rng, LOG_BITS))
            .collect::<Vec<_>>();

        // Witness A supplies the packing the reduction runs over, B the root and the opening.
        let (_, data_a) = pcs
            .commit_bits(&witness(0xAAAA, LOG_BITS), &mut challenger())
            .unwrap();

        let mut chal = challenger();
        let (root_b, data_b) = pcs
            .commit_bits(&witness(0xBBBB, LOG_BITS), &mut chal)
            .unwrap();

        // Reductions zero and two honestly use B.
        // The middle reduction is a true proof about A, played on B's sponge.
        let packing_a = BooleanPcs::<EF, MyMmcs, MyMmcs>::packing(&data_a);
        let packing_b = BooleanPcs::<EF, MyMmcs, MyMmcs>::packing(&data_b);
        let mut reductions = Vec::with_capacity(points.len());
        let mut values = Vec::with_capacity(points.len());
        let mut surviving_points = Vec::with_capacity(points.len());
        let mut surviving_values = Vec::with_capacity(points.len());
        for (index, point) in points.iter().enumerate() {
            let packing = if index == FORGED {
                &packing_a
            } else {
                &packing_b
            };
            let reduction = BitRingSwitch::new(point).unwrap();
            let (sent, surviving_point, surviving_value) = reduction.prove(packing, &mut chal);
            values.push(reduction.incoming_claim(&sent.tensor));
            reductions.push(sent);
            surviving_points.push(surviving_point);
            surviving_values.push(surviving_value);
        }

        // The commitment then opens B at all three surviving points.
        let opening = pcs
            .inner
            .try_open_at(
                data_b,
                &pcs.protocol(points.len()),
                &surviving_points,
                &mut chal,
            )
            .unwrap();

        // The first claim agrees, so checking only the first pair would accept this forgery.
        assert_eq!(opening.evals[0].current()[0], surviving_values[0]);
        // The middle claims about the same point disagree, which is what is caught below.
        assert_ne!(opening.evals[FORGED].current()[0], surviving_values[FORGED]);
        // The final claim also agrees, isolating the forged non-first index.
        assert_eq!(opening.evals[2].current()[0], surviving_values[2]);

        let proof = BooleanProof {
            reductions,
            opening,
        };
        let err = pcs
            .verify_at_points(
                &root_b,
                &points,
                &values,
                &proof,
                &mut replayed(&pcs, &root_b),
            )
            .unwrap_err();
        assert!(matches!(err, BooleanPcsError::SurvivingClaim), "{err:?}");

        // B's own reductions at the same points are accepted, so the rejection is the mismatch.
        let mut honest = challenger();
        let (root, data) = pcs
            .commit_bits(&witness(0xBBBB, LOG_BITS), &mut honest)
            .unwrap();
        let (honest_values, honest_proof) = pcs.open_at_points(data, &points, &mut honest).unwrap();
        pcs.verify_at_points(
            &root,
            &points,
            &honest_values,
            &honest_proof,
            &mut replayed(&pcs, &root),
        )
        .unwrap();
    }

    #[test]
    fn a_shape_the_commitment_does_not_hold_is_refused() {
        // Every mismatch is reported rather than asserted, because the shapes are inputs.
        let params = BinaryPcsParams {
            log_inv_rate: 2,
            pow_bits: 0,
            security_level: 40,
        };

        // A witness narrower than one element absorbs has no packing at all.
        let config = BinaryPcsConfig::try_new::<EF, EF>(6, params).unwrap();
        let narrow_witness = BooleanPcs::<EF, MyMmcs, MyMmcs>::new(config, mmcs(), mmcs(), 4).err();
        assert!(matches!(
            narrow_witness,
            Some(BooleanPcsError::WitnessTooNarrow {
                needed: 7,
                actual: 4
            })
        ));

        // A schedule for another committed field reaches the nested configuration error.
        let config = BinaryPcsConfig::try_new::<BinaryField64, EF>(6, params).unwrap();
        let wrong_field = BooleanPcs::<EF, MyMmcs, MyMmcs>::new(config, mmcs(), mmcs(), 13).err();
        assert!(matches!(
            wrong_field,
            Some(BooleanPcsError::Config(
                BinaryPcsConfigError::CommittedFieldMismatch {
                    derived: 64,
                    actual: 128,
                }
            ))
        ));

        // A schedule committing a different arity than the packing holds is refused.
        let config = BinaryPcsConfig::try_new::<EF, EF>(6, params).unwrap();
        let wrong_arity = BooleanPcs::<EF, MyMmcs, MyMmcs>::new(config, mmcs(), mmcs(), 14).err();
        assert!(matches!(
            wrong_arity,
            Some(BooleanPcsError::ConfigArity {
                expected: 7,
                actual: 6
            })
        ));

        // A point of the wrong width reaches neither the reduction nor the commitment.
        let pcs = boolean_pcs(13);
        let mut chal = challenger();
        let (commitment, data) = pcs.commit_bits(&witness(0x5417, 13), &mut chal).unwrap();
        let narrow = alloc::vec![Point::<EF>::rand(&mut SmallRng::seed_from_u64(0x5418), 12)];
        assert!(matches!(
            pcs.open_at_points(data, &narrow, &mut chal).err(),
            Some(BooleanPcsError::PointArity {
                expected: 13,
                actual: 12
            })
        ));
        assert!(matches!(
            pcs.verify_at_points(&commitment, &narrow, &[EF::ZERO], &proof_stub(), &mut chal),
            Err(BooleanPcsError::PointArity { .. })
        ));

        // No point at all describes nothing to prove.
        let (_, data) = pcs.commit_bits(&witness(0x5419, 13), &mut chal).unwrap();
        assert!(matches!(
            pcs.open_at_points(data, &[], &mut chal).err(),
            Some(BooleanPcsError::NoPoints)
        ));
    }

    /// A proof the width check never looks past.
    fn proof_stub() -> BooleanProof<EF, MyMmcs, MyMmcs> {
        let bits = witness(0x57AB, 13);
        let pcs = boolean_pcs(13);
        let mut chal = challenger();
        let (_, data) = pcs.commit_bits(&bits, &mut chal).unwrap();
        let points = alloc::vec![Point::<EF>::rand(&mut SmallRng::seed_from_u64(0x57AC), 13)];
        pcs.open_at_points(data, &points, &mut chal).unwrap().1
    }

    /// The witness one row further on, the last row repeating, batched over the selector.
    ///
    /// The definition of the successor reading, with no packing in it:
    ///
    /// ```text
    ///     sum_{c, z} eq(selector, c) * eq(rho, z) * t(c, min(z + 1, max))
    /// ```
    ///
    /// The selector leads the point and the rows trail it, `row_variables` of them.
    fn dense_successor_reading(cells: &Poly<EF>, point: &Point<EF>, row_variables: usize) -> EF {
        let (selector, rho) = point.split_at(point.num_variables() - row_variables);
        let eq_selector = Poly::<EF>::new_from_point(selector.as_slice(), EF::ONE);
        let eq_rho = Poly::<EF>::new_from_point(rho.as_slice(), EF::ONE);
        let rows = 1usize << row_variables;
        let mut reading = EF::ZERO;
        for (c, &gate) in eq_selector.as_slice().iter().enumerate() {
            for (z, &weight) in eq_rho.as_slice().iter().enumerate() {
                let x = (z + 1).min(rows - 1);
                reading += gate * weight * cells.as_slice()[c * rows + x];
            }
        }
        reading
    }

    /// Three openings, one per view: current alone, next alone, and both.
    ///
    /// ```text
    ///     opening 0   current,         no successor view
    ///     opening 1   next,            10 row variables, so carry and last travel
    ///     opening 2   current + next,   3 row variables, so the view is a column reading
    /// ```
    fn mixed_openings(seed: u64, log_bits: usize) -> Vec<BitOpening<EF>> {
        let mut rng = SmallRng::seed_from_u64(seed);
        [(log_bits, true, false), (10, false, true), (3, true, true)]
            .into_iter()
            .map(|(row_variables, current, next)| BitOpening {
                point: Point::<EF>::rand(&mut rng, log_bits),
                row_variables,
                current,
                next,
            })
            .collect()
    }

    #[test]
    fn a_successor_reading_round_trips_at_the_value_the_bits_define() {
        // Invariant: both readings answer for the bit witness, whatever the row count.
        //
        // Fixture state: 2^13 bits, of which one element absorbs 2^7.
        //
        //     rows  0        the successor view reads the point itself
        //     rows  3, 7     the rows fit in one element, so no successor elements travel
        //     rows 10, 13    the rows outrun one element, so carry and last travel
        //
        // Both values are compared against the one-element-per-bit multilinear.
        const LOG_BITS: usize = 13;

        let bits = witness(0x5ACC, LOG_BITS);
        let reference = embedded(&bits);
        let pcs = boolean_pcs(LOG_BITS);
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0x5ACD), LOG_BITS);

        let mut committed = challenger();
        let (commitment, data) = pcs.commit_bits(&bits, &mut committed).unwrap();

        for row_variables in [0, 3, 7, 10, 13] {
            let openings = alloc::vec![BitOpening {
                point: point.clone(),
                row_variables,
                current: true,
                next: true,
            }];
            let mut prover_chal = committed.clone();
            let (readings, proof) = pcs
                .open_readings(data.clone(), &openings, &mut prover_chal)
                .unwrap();

            assert_eq!(
                readings,
                alloc::vec![BitReadings {
                    current: Some(reference.eval_base(&point)),
                    next: Some(dense_successor_reading(&reference, &point, row_variables)),
                }],
                "rows {row_variables}"
            );
            // The successor elements travel exactly when the rows outrun one element.
            assert_eq!(
                proof.reductions[0].successor.is_some(),
                row_variables > BitRingSwitch::<EF>::ABSORBED,
                "rows {row_variables}"
            );

            pcs.verify_readings(
                &commitment,
                &openings,
                &readings,
                &proof,
                &mut replayed(&pcs, &commitment),
            )
            .unwrap();
        }
    }

    #[test]
    fn several_openings_with_mixed_views_cost_one_commitment_opening() {
        // Invariant: each opening leaves one surviving claim, whichever readings it asks for.
        //
        //     - reductions   one per opening, each answering every reading it asks for
        //     - opening      one, carrying one opened value per surviving claim
        //     - readings     present exactly where asked, at the values the bits define
        const LOG_BITS: usize = 13;

        let bits = witness(0x3A1D, LOG_BITS);
        let reference = embedded(&bits);
        let pcs = boolean_pcs(LOG_BITS);
        let openings = mixed_openings(0x3A1E, LOG_BITS);

        let mut prover_chal = challenger();
        let (commitment, data) = pcs.commit_bits(&bits, &mut prover_chal).unwrap();
        let (readings, proof) = pcs
            .open_readings(data, &openings, &mut prover_chal)
            .unwrap();

        // One reduction per opening, and one commitment opening for all of them.
        assert_eq!(proof.reductions.len(), 3);
        assert_eq!(proof.opening.evals.len(), 3);
        for batch in &proof.opening.evals {
            assert_eq!(batch.current().len(), 1);
            assert!(batch.next().is_empty());
        }

        for (index, (opening, reading)) in openings.iter().zip(&readings).enumerate() {
            let current = opening.current.then(|| reference.eval_base(&opening.point));
            let next = opening.next.then(|| {
                dense_successor_reading(&reference, &opening.point, opening.row_variables)
            });
            assert_eq!(*reading, BitReadings { current, next }, "opening {index}");
        }

        pcs.verify_readings(
            &commitment,
            &openings,
            &readings,
            &proof,
            &mut replayed(&pcs, &commitment),
        )
        .unwrap();
    }

    #[test]
    fn a_false_successor_reading_is_rejected() {
        // The successor reading is tied to the elements' columns, as the current one is.
        //
        //     rows  3    read off the tensor alone
        //     rows 10    read off the tensor, carry and last
        const LOG_BITS: usize = 13;

        let bits = witness(0xFA17, LOG_BITS);
        let pcs = boolean_pcs(LOG_BITS);
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xFA18), LOG_BITS);

        for row_variables in [3, 10] {
            let mut prover_chal = challenger();
            let (commitment, data) = pcs.commit_bits(&bits, &mut prover_chal).unwrap();
            let openings = alloc::vec![BitOpening {
                point: point.clone(),
                row_variables,
                current: true,
                next: true,
            }];
            let (readings, proof) = pcs
                .open_readings(data, &openings, &mut prover_chal)
                .unwrap();

            let false_readings = alloc::vec![BitReadings {
                next: readings[0].next.map(|next| next + EF::ONE),
                ..readings[0]
            }];
            let err = pcs
                .verify_readings(
                    &commitment,
                    &openings,
                    &false_readings,
                    &proof,
                    &mut replayed(&pcs, &commitment),
                )
                .unwrap_err();
            assert!(
                matches!(
                    err,
                    BooleanPcsError::ReductionProof(
                        BitRingSwitchProofError::SuccessorClaimMismatch
                    )
                ),
                "rows {row_variables}: {err:?}"
            );
        }
    }

    #[test]
    fn the_point_api_is_the_current_reading_api() {
        // Invariant: a point is an opening asking for the current reading alone.
        //
        // Fixture state: one committed witness, and two copies of the sponge that committed it.
        //
        // Equal proof bytes do not show the transcripts stayed in step on their own.
        // Two sponges that diverged could still have produced the same bytes.
        // So the next draw of each is compared as well, on both sides.
        const LOG_BITS: usize = 13;
        const NUM_POINTS: usize = 3;

        let bits = witness(0x9017, LOG_BITS);
        let pcs = boolean_pcs(LOG_BITS);
        let mut rng = SmallRng::seed_from_u64(0x9018);
        let points: Vec<Point<EF>> = (0..NUM_POINTS)
            .map(|_| Point::<EF>::rand(&mut rng, LOG_BITS))
            .collect();
        let openings: Vec<BitOpening<EF>> = points
            .iter()
            .map(|point| BitOpening {
                point: point.clone(),
                row_variables: point.num_variables(),
                current: true,
                next: false,
            })
            .collect();

        let mut committed = challenger();
        let (commitment, data) = pcs.commit_bits(&bits, &mut committed).unwrap();

        let mut point_chal = committed.clone();
        let (values, point_proof) = pcs
            .open_at_points(data.clone(), &points, &mut point_chal)
            .unwrap();
        let mut reading_chal = committed;
        let (readings, reading_proof) = pcs
            .open_readings(data, &openings, &mut reading_chal)
            .unwrap();

        assert_eq!(
            postcard::to_allocvec(&point_proof).unwrap(),
            postcard::to_allocvec(&reading_proof).unwrap()
        );
        let expected: Vec<BitReadings<EF>> = values
            .iter()
            .map(|&value| BitReadings {
                current: Some(value),
                next: None,
            })
            .collect();
        assert_eq!(readings, expected);
        assert_eq!(
            point_chal.sample_algebra_element::<EF>(),
            reading_chal.sample_algebra_element::<EF>()
        );

        // The two verifying routes leave their transcripts in step as well.
        let mut point_verifier = replayed(&pcs, &commitment);
        pcs.verify_at_points(
            &commitment,
            &points,
            &values,
            &point_proof,
            &mut point_verifier,
        )
        .unwrap();
        let mut reading_verifier = replayed(&pcs, &commitment);
        pcs.verify_readings(
            &commitment,
            &openings,
            &readings,
            &reading_proof,
            &mut reading_verifier,
        )
        .unwrap();
        assert_eq!(
            point_verifier.sample_algebra_element::<EF>(),
            reading_verifier.sample_algebra_element::<EF>()
        );
    }

    #[test]
    fn successor_tensors_are_priced() {
        // Invariant: a reduction sending carry and last pays for batching three elements.
        //
        //     one element      (d_log + 2 l') / |EF|
        //     three elements   (d_log + 2 + 2 l') / |EF|
        //
        // The point API asks for no successor view, so it pays the one-element price.
        let pcs = boolean_pcs(13);
        let with = pcs.readings_security(1, true);
        let without = pcs.readings_security(1, false);

        // The commitment's own term does not depend on the elements a reduction sends.
        assert_eq!(with.terms[0], without.terms[0]);
        assert_eq!(with.terms[1].label, p3_security::BIT_RING_SWITCH_LABEL);
        assert!(with.terms[1].bits.bits() < without.terms[1].bits.bits());

        // One reduction of seven draws, two powers of alpha and six rounds.
        assert_eq!(
            with.terms[1].bits.bits(),
            p3_security::bit_ring_switch_tensors_error(1, 3, 7, 6, 128).bits()
        );

        let points = pcs.opening_security(1);
        assert_eq!(points.terms, without.terms);
        assert_eq!(points.log2_max_candidates, without.log2_max_candidates);
    }

    #[test]
    fn a_reading_shape_the_opening_did_not_ask_for_is_refused() {
        // Invariant: a reading is present exactly where its opening asks for one.
        //
        // Mutation: add or drop one reading, at each opening in turn.
        //
        // The refusal comes before the transcript moves.
        // So the sponge still draws what one that only replayed the binding draws.
        const LOG_BITS: usize = 13;

        let pcs = boolean_pcs(LOG_BITS);
        let openings = mixed_openings(0x54A9, LOG_BITS);
        let mut prover_chal = challenger();
        let (commitment, data) = pcs
            .commit_bits(&witness(0x5AA0, LOG_BITS), &mut prover_chal)
            .unwrap();
        let (readings, proof) = pcs
            .open_readings(data, &openings, &mut prover_chal)
            .unwrap();

        let stray = EF::ONE;
        let mutations = [
            (
                0,
                BitReadings {
                    next: Some(stray),
                    ..readings[0]
                },
            ),
            (
                0,
                BitReadings {
                    current: None,
                    ..readings[0]
                },
            ),
            (
                1,
                BitReadings {
                    current: Some(stray),
                    ..readings[1]
                },
            ),
            (
                2,
                BitReadings {
                    next: None,
                    ..readings[2]
                },
            ),
        ];
        for (index, reading) in mutations {
            let mut tampered = readings.clone();
            tampered[index] = reading;
            let mut verifier = replayed(&pcs, &commitment);
            let err = pcs
                .verify_readings(&commitment, &openings, &tampered, &proof, &mut verifier)
                .unwrap_err();
            assert!(
                matches!(err, BooleanPcsError::ReadingShape { index: refused } if refused == index),
                "{err:?}"
            );
            assert_eq!(
                verifier.sample_algebra_element::<EF>(),
                replayed(&pcs, &commitment).sample_algebra_element::<EF>(),
                "opening {index}"
            );
        }
    }

    #[test]
    fn an_opening_the_witness_cannot_answer_is_refused() {
        // Invariant: an opening asks for some reading, over rows the witness has.
        //
        //     no reading       nothing would be checked
        //     too many rows    the successor view would step past the witness
        //
        // Each is refused at its own index, on both sides, before the transcript moves.
        const LOG_BITS: usize = 13;

        let pcs = boolean_pcs(LOG_BITS);
        let mut chal = challenger();
        let (commitment, data) = pcs
            .commit_bits(&witness(0xA5C0, LOG_BITS), &mut chal)
            .unwrap();
        let mut rng = SmallRng::seed_from_u64(0xA5C1);
        let answered = BitOpening {
            point: Point::<EF>::rand(&mut rng, LOG_BITS),
            row_variables: LOG_BITS,
            current: true,
            next: false,
        };
        let silent = BitOpening {
            point: Point::<EF>::rand(&mut rng, LOG_BITS),
            row_variables: LOG_BITS,
            current: false,
            next: false,
        };
        let past_the_witness = BitOpening {
            point: Point::<EF>::rand(&mut rng, LOG_BITS),
            row_variables: LOG_BITS + 1,
            current: false,
            next: true,
        };
        let readings = [
            BitReadings {
                current: Some(EF::ZERO),
                next: None,
            },
            BitReadings {
                current: None,
                next: Some(EF::ZERO),
            },
        ];

        let silent_openings = [answered.clone(), silent];
        let wide_openings = [answered, past_the_witness];
        let mut before = chal.clone();

        let err = pcs
            .open_readings(data.clone(), &silent_openings, &mut chal)
            .err();
        assert!(
            matches!(err, Some(BooleanPcsError::NoReading { index: 1 })),
            "{err:?}"
        );
        let err = pcs.open_readings(data, &wide_openings, &mut chal).err();
        assert!(
            matches!(
                err,
                Some(BooleanPcsError::RowVariables {
                    index: 1,
                    row_variables: 14,
                    num_variables: 13,
                })
            ),
            "{err:?}"
        );
        assert_eq!(
            chal.sample_algebra_element::<EF>(),
            before.sample_algebra_element::<EF>()
        );

        let proof = proof_stub();
        let mut verifier = replayed(&pcs, &commitment);
        let err = pcs
            .verify_readings(
                &commitment,
                &silent_openings,
                &readings,
                &proof,
                &mut verifier,
            )
            .unwrap_err();
        assert!(
            matches!(err, BooleanPcsError::NoReading { index: 1 }),
            "{err:?}"
        );
        let err = pcs
            .verify_readings(
                &commitment,
                &wide_openings,
                &readings,
                &proof,
                &mut verifier,
            )
            .unwrap_err();
        assert!(
            matches!(err, BooleanPcsError::RowVariables { index: 1, .. }),
            "{err:?}"
        );
        assert_eq!(
            verifier.sample_algebra_element::<EF>(),
            replayed(&pcs, &commitment).sample_algebra_element::<EF>()
        );
    }

    proptest! {
        // Each case runs a whole commitment and opening on both sides.
        #![proptest_config(ProptestConfig { cases: 8, ..ProptestConfig::default() })]

        /// Every witness height the fixture admits, at random points each time.
        #[test]
        fn a_boolean_opening_round_trips_over_random_inputs(
            log_bits in 13usize..=15,
            num_points in 1usize..=3,
            witness_seed: u64,
            point_seed: u64,
        ) {
            let bits = witness(witness_seed, log_bits);
            let pcs = boolean_pcs(log_bits);

            let mut prover_chal = challenger();
            let (commitment, data) = pcs.commit_bits(&bits, &mut prover_chal).unwrap();
            let mut rng = SmallRng::seed_from_u64(point_seed);
            let points: Vec<Point<EF>> = (0..num_points)
                .map(|_| Point::<EF>::rand(&mut rng, log_bits))
                .collect();
            let (values, proof) = pcs.open_at_points(data, &points, &mut prover_chal).unwrap();

            let reference = embedded(&bits);
            for (point, &value) in points.iter().zip(&values) {
                prop_assert_eq!(value, reference.eval_base(point));
            }
            prop_assert!(
                pcs.verify_at_points(
                    &commitment,
                    &points,
                    &values,
                    &proof,
                    &mut replayed(&pcs, &commitment)
                )
                .is_ok()
            );
        }
    }
}
