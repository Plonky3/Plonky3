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
//! Three errors compose by a union bound.
//!
//! ```text
//!     commitment       the proximity argument's own budget, at one opened point
//!     reduction        one bit ring switch for every claim, its rounds run once
//!     claim batching   the claims folded under powers of lambda
//! ```
//!
//! All come back labelled, so a report says which one is short.
//!
//! The batch runs before the proximity argument names one codeword.
//!
//! So it is charged over every candidate the commitment still leaves open.

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::{BitCoordinates, PackedGf2, TowerLevel, Underlier};
use p3_challenger::fs::TranscriptField;
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_security::multilinear::{bit_ring_switch_claim_batching_term, bit_ring_switch_tensors_term};
use p3_sumcheck::layout::{Layout, SuffixProver, Table};
use p3_sumcheck::ring_switch::bits::{
    BitPacking, BitPackingView, BitRingSwitch, BitRingSwitchClaims, BitTensor, ClaimElements,
    SuffixTensors,
};
use p3_sumcheck::{
    FoldedColumn, OpeningBatch, OpeningProtocol, PrescribedOpeningSecurity, PrescribedPointPcs,
    TableShape, TableSpec,
};
use p3_whir::{InitialWeightCheck, WhirDomain, WhirProver, WhirProverData};

use crate::boolean::{
    BitOpening, BitReadings, BooleanBackend, BooleanMultilinearPcs, WeightedClaim, WeightedOpening,
};
use crate::boolean_trace::BooleanTraceCommitment;
use crate::fold::BitChallengeField;
use crate::packing::{Coordinates, PackedStack, PackedWords, hypercube_variables};
use crate::whir::error::BooleanWhirError;
use crate::whir::proof::{BooleanWhirDeferredProof, BooleanWhirOpening, BooleanWhirProof};
use crate::whir::shape::{ProofShape, ProofShapeError};

/// The value of every weighted opening, in order, and the one proof that opens them all.
type WeightedOpened<F, EF, MT> = (Vec<EF>, BooleanWhirProof<F, EF, MT>);

/// The binding mode the committed layout uses.
///
/// One mode is fixed rather than chosen, so a commitment and its replay never disagree.
type Binding<F, EF> = SuffixProver<F, EF>;

/// The proximity argument every packed claim is discharged against.
///
/// It commits the packing level `F` and draws its challenges from `EF`.
pub type BooleanWhirProver<F, EF, Dft, MT, Challenger> =
    WhirProver<EF, F, Dft, MT, Challenger, Binding<F, EF>>;

/// Prover-side data retained between committing and opening.
pub type BooleanWhirData<F, EF, MT> = WhirProverData<F, EF, MT, Binding<F, EF>>;

/// A batched trace of Boolean columns, discharged through the additive-domain proximity argument.
pub type BooleanWhirTracePcs<F, EF, Dft, MT, Challenger> =
    BooleanTraceCommitment<EF, BooleanWhirPcs<F, EF, Dft, MT, Challenger>>;

/// A commitment to a function from the hypercube to `{0, 1}`, opened through WHIR.
///
/// The committed object is a bit witness, packed into the tower level `F`.
///
/// An opening answers for its multilinear extension at a point of the challenge field `EF`.
///
/// ```text
///     F = EF = GF(2^128)                 128 bits per committed element
///     F = GF(2^64),  EF = GF(2^192)      64 bits per committed element, wider challenges
/// ```
pub struct BooleanWhirPcs<F: Field, EF: ExtensionField<F>, Dft, MT, Challenger> {
    /// The proximity argument the packed multilinear is discharged against.
    inner: BooleanWhirProver<F, EF, Dft, MT, Challenger>,
    /// Variables the bit witness has, which is the packing's plus the absorbed ones.
    num_variables: usize,
}

impl<F, EF, Dft, MT, Challenger> BooleanWhirPcs<F, EF, Dft, MT, Challenger>
where
    F: Field + TranscriptField + TowerLevel + Coordinates + Ord,
    EF: BitCoordinates + ExtensionField<F>,
    Dft: WhirDomain<F, EF>,
    MT: Mmcs<F>,
{
    /// Wrap a proximity argument as a commitment to a bit witness of this many variables.
    ///
    /// # Errors
    ///
    /// Returns an error unless one element can absorb the witness's low coordinates.
    ///
    /// Returns an error unless the schedule commits exactly the elements the packing holds.
    pub fn new(
        inner: BooleanWhirProver<F, EF, Dft, MT, Challenger>,
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
        let absorbed = BitRingSwitch::<F, EF>::ABSORBED;
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
    /// A batch of claims leaves one surviving claim, whatever its size.
    fn protocol(&self, num_claims: usize) -> OpeningProtocol {
        OpeningProtocol::new(vec![TableSpec::new(
            TableShape::new(self.inner.num_variables(), 1),
            (0..num_claims)
                .map(|_| OpeningBatch::new(vec![0], Vec::new()))
                .collect(),
        )])
    }

    /// Every opening's reduction, then every weighted claim's, gathered into the one batch that reduces them all.
    fn claims<'a>(
        &self,
        openings: &[BitOpening<EF>],
        weighted: impl Iterator<Item = &'a WeightedOpening<EF>>,
    ) -> Result<BitRingSwitchClaims<F, EF>, BooleanWhirError>
    where
        EF: 'a,
    {
        let kept = self.inner.num_variables();
        let mut reductions = openings
            .iter()
            .map(Self::reduction)
            .collect::<Result<Vec<_>, _>>()?;
        for opening in weighted {
            // The kept coordinates are the witness's less the absorbed ones.
            if opening.high.num_variables() != kept {
                return Err(BooleanWhirError::PointArity {
                    expected: kept,
                    actual: opening.high.num_variables(),
                });
            }
            reductions.push(
                BitRingSwitch::with_low_weights(&opening.high, &opening.low_weights)
                    .map_err(BooleanWhirError::Reduction)?,
            );
        }
        if reductions.is_empty() {
            return Err(BooleanWhirError::NoPoints);
        }
        BitRingSwitchClaims::new(reductions).map_err(BooleanWhirError::Reduction)
    }

    /// The reduction answering every reading one opening asks for.
    fn reduction(opening: &BitOpening<EF>) -> Result<BitRingSwitch<F, EF>, BooleanWhirError> {
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
    fn packing(prover_data: &BooleanWhirData<F, EF, MT>) -> BitPackingView<'_, F> {
        BitPacking::from_packed(prover_data.table(0).poly(0))
            .expect("a committed table is a hypercube over a byte-aligned level")
    }
}

impl<F, EF, Dft, MT, Challenger> BooleanWhirPcs<F, EF, Dft, MT, Challenger>
where
    F: Field + TranscriptField + TowerLevel + Coordinates + Ord + Send + Sync,
    EF: BitChallengeField<F>,
    Dft: WhirDomain<F, EF>,
    MT: Mmcs<F>,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>
        + CanObserve<MT::Commitment>,
{
    /// Variables the proximity argument's first fold binds, the packing's trailing ones.
    #[must_use]
    pub fn first_folding(&self) -> usize {
        self.inner.round_folding_factor(0)
    }

    /// Every labelled algebraic error one opening of this many claims charges.
    ///
    /// ```text
    ///     commitment       the proximity argument's own budget, at the one surviving point
    ///     reduction        one batched bit ring switch, batching its elements under one draw
    ///     claim batching   the claims folded under powers of lambda
    /// ```
    ///
    /// The terms are independent draws, so they compose by a union bound.
    ///
    /// The reduction is charged over every candidate the commitment leaves open.
    ///
    /// Its challenges are drawn before the proximity argument names one of them.
    ///
    /// The count itself is forwarded untouched, because this is a link and not the end.
    ///
    /// A caller stacking its own draw faces the same list the proximity argument left.
    ///
    /// So it charges that draw over the same count.
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
        // Every claim of a batch survives as one claim, at one point.
        let protocol = self.protocol(num_claims.min(1));
        let mut security = self.inner.prescribed_security(&protocol)?;
        // The tensor alone, or the tensor with carry and last.
        let num_tensors = if successor_tensors { 3 } else { 1 };
        // The batch runs before one candidate is named, so it pays for all of them.
        //
        // The charge leaves the count alone, so the caller above still sees the same list.
        //
        // It then charges its own draws over that list.
        security.charge_reduction(bit_ring_switch_tensors_term(
            num_claims.min(1),
            num_tensors,
            BitRingSwitch::<F, EF>::BATCHED,
            self.inner.num_variables(),
            EF::bits(),
        ));
        // A single claim draws no lambda, so its report carries no batching term.
        if num_claims > 1 {
            security.charge_reduction(bit_ring_switch_claim_batching_term(num_claims, EF::bits()));
        }
        Some(security)
    }

    /// The shape of every proof an opening of this many claims can produce.
    ///
    /// It covers the reductions as well as the opening, so a ceiling graded against it binds both.
    /// Serializer framing is excluded.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported MMCS geometry or unrepresentable counts.
    pub fn proof_shape(
        &self,
        num_claims: usize,
        successor_tensors: bool,
    ) -> Result<ProofShape, ProofShapeError> {
        ProofShape::of_bit_readings(
            &self.inner,
            self.inner.mmcs(),
            num_claims,
            successor_tensors,
        )
    }

    /// Commit to packed elements as the one column of the one table the layout stacks.
    #[allow(clippy::type_complexity)]
    fn commit_column(
        &self,
        elements: Vec<F>,
        challenger: &mut Challenger,
    ) -> Result<(MT::Commitment, BooleanWhirData<F, EF, MT>), BooleanWhirError> {
        let actual = hypercube_variables(elements.len()).map_err(BooleanWhirError::Packing)?;
        if actual != self.inner.num_variables() {
            return Err(BooleanWhirError::WitnessArity {
                expected: self.inner.num_variables(),
                actual,
            });
        }

        let len = elements.len();
        let table = Table::new(RowMajorMatrix::new(elements, len));
        let folding = self.inner.round_folding_factor(0);
        let witness = Binding::<F, EF>::new_witness(vec![table], folding);
        p3_commit::MultilinearPcs::<EF, Challenger>::commit(&self.inner, witness, challenger)
            .map_err(BooleanWhirError::Commit)
    }

    /// A claim's element split over the trailing coordinates the first fold binds, for the deferred switch.
    ///
    /// `folded` holds one value per bit of the inner coordinates, the leading `log_outer` of `high` folded away.
    ///
    /// # Returns
    ///
    /// Nothing when the inner coordinates are fewer than the first fold binds, or `folded` has another length.
    #[must_use]
    pub fn suffix_tensors(
        &self,
        folded: &[EF],
        high: &[EF],
        log_outer: usize,
    ) -> Option<SuffixTensors<EF, F>> {
        let width = 1 << BitRingSwitch::<F, EF>::ABSORBED;
        let depth = self.inner.round_folding_factor(0);
        let inner = high.get(log_outer..)?;
        let head = inner.len().checked_sub(depth)?;
        // The fold holds one row of slices per point of the inner coordinates.
        //
        //     expected length = width * 2^(inner coordinates)
        //
        // A length past a word is refused rather than wrapped.
        let expected = u32::try_from(inner.len())
            .ok()
            .and_then(|log_rows| 1usize.checked_shl(log_rows))
            .and_then(|rows| rows.checked_mul(width))?;
        if folded.len() != expected {
            return None;
        }
        let weights = Poly::new_from_point(&inner[..head], EF::ONE);
        let banks = (0..1usize << depth)
            .map(|bank| {
                let mut slices = vec![EF::ZERO; width];
                for (high, &weight) in weights.as_slice().iter().enumerate() {
                    let row = &folded[((high << depth) | bank) * width..][..width];
                    for (slice, &value) in slices.iter_mut().zip(row) {
                        *slice += weight * value;
                    }
                }
                BitTensor::from_columns(&slices)
            })
            .collect();
        SuffixTensors::new(banks)
    }

    /// Open the bit witness with the readings every opening asks for and every weighted claim, in one proof.
    ///
    /// No point needs prior transcript binding: each reduction binds its own.
    ///
    /// # Returns
    ///
    /// One set of readings per opening, in the order the openings were supplied.
    ///
    /// Then one value per weighted claim, `sum_v low_weights[v] * t(high, v)`, in the order supplied.
    ///
    /// # Errors
    ///
    /// Before the transcript moves, no claim at all, or a claim that names the wrong variables.
    ///
    /// An opening asking for no reading, or stepping within more rows than the witness has.
    ///
    /// A batch reading no successor row defers its ring switch to WHIR's first fold.
    ///
    /// Entry `i` of `folded`, when present, gives weighted claim `i`'s split without a sweep.
    #[allow(clippy::type_complexity)]
    pub fn open_weighted_readings(
        &self,
        prover_data: BooleanWhirData<F, EF, MT>,
        openings: &[BitOpening<EF>],
        weighted: &[WeightedOpening<EF>],
        folded: &[Option<&FoldedColumn<EF>>],
        challenger: &mut Challenger,
    ) -> Result<(Vec<BitReadings<EF>>, Vec<EF>, BooleanWhirOpening<F, EF, MT>), BooleanWhirError>
    where
        Challenger: CanObserve<MT::Commitment>,
    {
        self.check_openings(openings)?;
        // Every reduction is set up before any runs, so a refused one leaves the transcript alone.
        let claims = self.claims(openings, weighted.iter())?;
        let proof = if defers(openings) {
            // A folded column splits its claim directly.
            // Every other claim sweeps the packing for its split.
            let splits = openings
                .iter()
                .map(|_| None)
                .chain(weighted.iter().enumerate().map(|(index, opening)| {
                    folded.get(index).copied().flatten().and_then(|folded| {
                        self.suffix_tensors(
                            &folded.values,
                            opening.high.as_slice(),
                            folded.log_outer,
                        )
                    })
                }))
                .collect::<Vec<_>>();
            let splits = splits.iter().map(Option::as_ref).collect::<Vec<_>>();
            BooleanWhirOpening::Deferred(self.open_claims_deferred(
                prover_data,
                &claims,
                &splits,
                challenger,
            )?)
        } else {
            BooleanWhirOpening::Batched(self.open_claims(prover_data, &claims, challenger)?)
        };
        let (readings, values) = Self::readings(openings, &claims, proof.elements())?;
        Ok((readings, values, proof))
    }

    /// The readings each opening asked for, then each weighted claim's value, off the elements sent.
    #[allow(clippy::type_complexity)]
    fn readings(
        openings: &[BitOpening<EF>],
        claims: &BitRingSwitchClaims<F, EF>,
        elements: &[ClaimElements<F, EF>],
    ) -> Result<(Vec<BitReadings<EF>>, Vec<EF>), BooleanWhirError> {
        // The elements each claim sends already hold its readings.
        let readings = openings
            .iter()
            .zip(claims.reductions())
            .zip(elements)
            .map(|((opening, switch), elements)| {
                let current = opening
                    .current
                    .then(|| switch.incoming_claim(&elements.tensor));
                let next = opening
                    .next
                    .then(|| switch.successor_claim(&elements.tensor, elements.successor.as_ref()))
                    .transpose()
                    .map_err(BooleanWhirError::Reduction)?;
                Ok(BitReadings { current, next })
            })
            .collect::<Result<Vec<_>, BooleanWhirError>>()?;
        // A weighted claim's value is its incoming claim, read off the same elements.
        let values = claims.reductions()[openings.len()..]
            .iter()
            .zip(&elements[openings.len()..])
            .map(|(switch, elements)| switch.incoming_claim(&elements.tensor))
            .collect();
        Ok((readings, values))
    }

    /// Reduce a batch of claims to one about the packing, then open WHIR at its point.
    fn open_claims(
        &self,
        prover_data: BooleanWhirData<F, EF, MT>,
        claims: &BitRingSwitchClaims<F, EF>,
        challenger: &mut Challenger,
    ) -> Result<BooleanWhirProof<F, EF, MT>, BooleanWhirError> {
        let packing = Self::packing(&prover_data);

        // One batch for every opening, leaving one claim about the packing.
        let (reduction, surviving_point, _) =
            tracing::info_span!("bit ring switch").in_scope(|| {
                claims
                    .prove::<<EF as BitChallengeField<F>>::SumcheckRepr, _, _>(&packing, challenger)
            });

        // The surviving value crosses the wire twice, and the closing check is that the two agree.
        // The surviving point came out of the batch's rounds, so it is bound already.
        let opening = self
            .inner
            .open_at(
                prover_data,
                &self.protocol(1),
                &[surviving_point],
                challenger,
            )
            .map_err(BooleanWhirError::Commit)?;
        Ok(BooleanWhirProof { reduction, opening })
    }

    /// Open the bit witness at weighted claims, in one proof.
    ///
    /// # Returns
    ///
    /// One value per claim, `sum_v low_weights[v] * t(high, v)`, in the order supplied.
    ///
    /// # Errors
    ///
    /// Before the transcript moves, no claim, or a claim of the wrong width.
    pub fn open_weighted(
        &self,
        prover_data: BooleanWhirData<F, EF, MT>,
        openings: &[WeightedOpening<EF>],
        challenger: &mut Challenger,
    ) -> Result<WeightedOpened<F, EF, MT>, BooleanWhirError> {
        let claims = self.claims(&[], openings.iter())?;
        let proof = self.open_claims(prover_data, &claims, challenger)?;
        let (_, values) = Self::readings(&[], &claims, &proof.reduction.claims)?;
        Ok((values, proof))
    }

    /// Open the bit witness at weighted claims, the ring switch's sum proven by WHIR's first fold.
    ///
    /// No separate ring-switch sumcheck runs, and no opening point is evaluated.
    ///
    /// Entry `i` of `suffix_tensors`, when present, splits claim `i`'s element over the trailing
    /// coordinates the first fold binds.
    ///
    /// # Returns
    ///
    /// One value per claim, `sum_v low_weights[v] * t(high, v)`, in the order supplied.
    ///
    /// # Errors
    ///
    /// Before the transcript moves, no claim, or a claim of the wrong width.
    #[allow(clippy::type_complexity)]
    pub fn open_weighted_deferred(
        &self,
        prover_data: BooleanWhirData<F, EF, MT>,
        openings: &[WeightedOpening<EF>],
        suffix_tensors: &[Option<&SuffixTensors<EF, F>>],
        challenger: &mut Challenger,
    ) -> Result<(Vec<EF>, BooleanWhirDeferredProof<F, EF, MT>), BooleanWhirError>
    where
        Challenger: CanObserve<MT::Commitment>,
    {
        let claims = self.claims(&[], openings.iter())?;
        let proof = self.open_claims_deferred(prover_data, &claims, suffix_tensors, challenger)?;
        let (_, values) = Self::readings(&[], &claims, &proof.reduction.claims)?;
        Ok((values, proof))
    }

    /// Bind a batch of claims and prove its sum through WHIR's first fold.
    ///
    /// Entry `i` of `suffix_tensors`, when present, is claim `i`'s split.
    ///
    /// A missing one costs one sweep of the packing.
    fn open_claims_deferred(
        &self,
        prover_data: BooleanWhirData<F, EF, MT>,
        claims: &BitRingSwitchClaims<F, EF>,
        suffix_tensors: &[Option<&SuffixTensors<EF, F>>],
        challenger: &mut Challenger,
    ) -> Result<BooleanWhirDeferredProof<F, EF, MT>, BooleanWhirError>
    where
        Challenger: CanObserve<MT::Commitment>,
    {
        let depth = self.inner.round_folding_factor(0);
        let (reduction, weight) = tracing::info_span!("bit ring switch")
            .in_scope(|| {
                claims.prove_deferred(
                    &Self::packing(&prover_data),
                    suffix_tensors,
                    depth,
                    challenger,
                )
            })
            .map_err(BooleanWhirError::Reduction)?;
        let opening = self
            .inner
            .open_with_initial_weight(prover_data, weight, challenger)
            .map_err(BooleanWhirError::Commit)?;
        Ok(BooleanWhirDeferredProof { reduction, opening })
    }

    /// Check a deferred proof against every weighted claim.
    ///
    /// # Errors
    ///
    /// Before the transcript moves, a refused claim or a claim count that disagrees.
    ///
    /// After it, a false value, the commitment, or a first fold that does not close.
    pub fn verify_weighted_deferred(
        &self,
        commitment: &MT::Commitment,
        claims: &[WeightedClaim<EF>],
        proof: &BooleanWhirDeferredProof<F, EF, MT>,
        challenger: &mut Challenger,
    ) -> Result<(), BooleanWhirError>
    where
        Challenger: CanObserve<MT::Commitment>,
    {
        let batch = self.claims(&[], claims.iter().map(|claim| &claim.opening))?;
        let values = claims.iter().map(|claim| claim.value).collect::<Vec<_>>();
        self.verify_claims_deferred(commitment, &batch, &values, proof, challenger)
    }

    /// Replay a deferred batch of claims, then the WHIR opening whose first fold closes it.
    fn verify_claims_deferred(
        &self,
        commitment: &MT::Commitment,
        batch: &BitRingSwitchClaims<F, EF>,
        values: &[EF],
        proof: &BooleanWhirDeferredProof<F, EF, MT>,
        challenger: &mut Challenger,
    ) -> Result<(), BooleanWhirError>
    where
        Challenger: CanObserve<MT::Commitment>,
    {
        let closing = batch
            .verify_deferred(&proof.reduction, values, challenger)
            .map_err(BooleanWhirError::ReductionProof)?;
        let shapes = [TableShape::new(self.inner.num_variables(), 1)];
        // The point always names the packing's variables, so a refusal only fails the check.
        let weight_at = |point: &Point<EF>| closing.weight_at(point).unwrap_or(EF::ZERO);
        self.inner
            .verify_with_initial_weight(
                commitment,
                &proof.opening,
                &shapes,
                InitialWeightCheck {
                    sum: closing.claimed_sum(),
                    eval: &weight_at,
                },
                challenger,
            )
            .map_err(BooleanWhirError::Opening)
    }

    /// Check one proof against every weighted claim.
    ///
    /// # Errors
    ///
    /// Before the transcript moves, a refused claim or a reduction count that disagrees.
    ///
    /// After it, a false value, a failed reduction, the commitment, or an unclosed claim.
    pub fn verify_weighted(
        &self,
        commitment: &MT::Commitment,
        claims: &[WeightedClaim<EF>],
        proof: &BooleanWhirProof<F, EF, MT>,
        challenger: &mut Challenger,
    ) -> Result<(), BooleanWhirError> {
        let batch = self.claims(&[], claims.iter().map(|claim| &claim.opening))?;
        if proof.reduction.claims.len() != claims.len() {
            return Err(BooleanWhirError::ClaimCount {
                expected: claims.len(),
                values: claims.len(),
                reductions: proof.reduction.claims.len(),
            });
        }
        let readings = claims
            .iter()
            .map(|claim| (Some(claim.value), None))
            .collect::<Vec<_>>();
        self.verify_claims(commitment, &batch, &readings, proof, challenger)
    }

    /// Check one proof against the readings it claims at every opening and the value of every weighted claim.
    ///
    /// The commitment's binding is the caller's, replayed before this is reached.
    ///
    /// # Errors
    ///
    /// Before the transcript moves, a refused claim or a reading count that disagrees.
    ///
    /// After it, a false reading or value, a failed reduction, the commitment, or an unclosed claim.
    ///
    /// A batch reading no successor row must come deferred, and any other batched.
    pub fn verify_weighted_readings(
        &self,
        commitment: &MT::Commitment,
        openings: &[BitOpening<EF>],
        readings: &[BitReadings<EF>],
        weighted: &[WeightedClaim<EF>],
        proof: &BooleanWhirOpening<F, EF, MT>,
        challenger: &mut Challenger,
    ) -> Result<(), BooleanWhirError>
    where
        Challenger: CanObserve<MT::Commitment>,
    {
        self.check_openings(openings)?;
        let expected = openings.len() + weighted.len();
        if readings.len() != openings.len() || proof.elements().len() != expected {
            return Err(BooleanWhirError::ClaimCount {
                expected,
                values: readings.len() + weighted.len(),
                reductions: proof.elements().len(),
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
        let claims = self.claims(openings, weighted.iter().map(|claim| &claim.opening))?;
        let readings = readings
            .iter()
            .map(|reading| (reading.current, reading.next))
            .chain(weighted.iter().map(|claim| (Some(claim.value), None)))
            .collect::<Vec<_>>();
        // The route is a function of the openings, so a proof taking the other one is refused.
        match (defers(openings), proof) {
            (true, BooleanWhirOpening::Deferred(proof)) => {
                // A deferred batch reads every opening at its current row only.
                let values = readings
                    .iter()
                    .map(|&(current, _)| current.expect("a deferred opening asks the current row"))
                    .collect::<Vec<_>>();
                self.verify_claims_deferred(commitment, &claims, &values, proof, challenger)
            }
            (false, BooleanWhirOpening::Batched(proof)) => {
                self.verify_claims(commitment, &claims, &readings, proof, challenger)
            }
            _ => Err(BooleanWhirError::OpeningRoute),
        }
    }

    /// Replay a batch of claims, then the WHIR opening at the point it leaves.
    fn verify_claims(
        &self,
        commitment: &MT::Commitment,
        claims: &BitRingSwitchClaims<F, EF>,
        readings: &[(Option<EF>, Option<EF>)],
        proof: &BooleanWhirProof<F, EF, MT>,
        challenger: &mut Challenger,
    ) -> Result<(), BooleanWhirError> {
        // The batch turns every reading about the bits into one claim about the packing.
        let (surviving_point, surviving_value) = claims
            .verify_readings(&proof.reduction, readings, challenger)
            .map_err(BooleanWhirError::ReductionProof)?;

        // One proximity opening pins the one surviving point to the committed polynomial.
        let evals = self
            .inner
            .verify_at(
                commitment,
                &proof.opening,
                &self.protocol(1),
                &[surviving_point],
                challenger,
            )
            .map_err(BooleanWhirError::Opening)?;

        // The batch closes against the value opened at its surviving point.
        match evals.as_slice() {
            [batch] if batch.current().first() == Some(&surviving_value) => Ok(()),
            _ => Err(BooleanWhirError::SurvivingClaim),
        }
    }
}

/// Whether a batch defers its ring switch: none of its openings reads a successor row.
fn defers<EF>(openings: &[BitOpening<EF>]) -> bool {
    openings.iter().all(|opening| !opening.next)
}

impl<F, EF, Dft, MT, Challenger> BooleanMultilinearPcs<EF, Challenger>
    for BooleanWhirPcs<F, EF, Dft, MT, Challenger>
where
    F: Field + TranscriptField + TowerLevel + Coordinates + Ord + Send + Sync,
    EF: BitChallengeField<F>,
    Dft: WhirDomain<F, EF>,
    MT: Mmcs<F>,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>
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
        let stack = PackedStack::<PackedGf2<U>, F>::from_columns(&[bits])
            .map_err(BooleanWhirError::Packing)?;
        self.commit_column(stack.into_poly().into_evals(), challenger)
    }

    fn commit_packed(
        &self,
        packed: PackedWords<F>,
        challenger: &mut Challenger,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::Error> {
        self.commit_column(packed.into_elements(), challenger)
    }

    fn packed_words(words: usize) -> Option<PackedWords<F>> {
        PackedWords::zeroed(words)
    }

    fn open_weighted_readings(
        &self,
        prover_data: Self::ProverData,
        openings: &[BitOpening<EF>],
        weighted: &[WeightedOpening<EF>],
        folded: &[Option<&FoldedColumn<EF>>],
        challenger: &mut Challenger,
    ) -> Result<(Vec<BitReadings<EF>>, Vec<EF>, Self::Proof), Self::Error> {
        Self::open_weighted_readings(self, prover_data, openings, weighted, folded, challenger)
    }

    fn verify_weighted_readings(
        &self,
        commitment: &Self::Commitment,
        openings: &[BitOpening<EF>],
        readings: &[BitReadings<EF>],
        weighted: &[WeightedClaim<EF>],
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        Self::verify_weighted_readings(
            self, commitment, openings, readings, weighted, proof, challenger,
        )
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

impl<F, EF, Dft, MT, Challenger> BooleanBackend<EF> for BooleanWhirPcs<F, EF, Dft, MT, Challenger>
where
    F: Field + TranscriptField + TowerLevel + Coordinates + Ord + Send + Sync,
    EF: BitCoordinates + ExtensionField<F>,
    Dft: WhirDomain<F, EF>,
    MT: Mmcs<F>,
{
    type Val = F;
    type Commitment = MT::Commitment;
    type ProverData = BooleanWhirData<F, EF, MT>;
    type Proof = BooleanWhirOpening<F, EF, MT>;
    type Error = BooleanWhirError;

    fn num_variables(&self) -> usize {
        self.num_variables
    }
}
