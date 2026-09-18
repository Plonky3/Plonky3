//! One ring-switching reduction at a bit alphabet, end to end.
//!
//! The five values the reduction is built from, the messages it sends, and the two sides that run it.
//!
//! Both sides are methods, because here the reduction has a type to hang them on.
//! The sibling module's are free functions because there the reduction has none.

use p3_binary_field::TowerLevel;
use p3_challenger::fs::TranscriptField;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use serde::{Deserialize, Serialize};

use super::basis::Coefficients;
use super::packing::BitPacking;
use super::tensor::BitTensor;
use super::transcript::{
    BitRingSwitchProverTranscript, BitRingSwitchShape, BitRingSwitchVerifierTranscript,
    TranscriptWidth,
};
use crate::data::SumcheckData;
use crate::error::SumcheckError;
use crate::product_polynomial::ProductPolynomial;
use crate::strategy::{Basis, SumcheckProver, VariableOrder};

/// The hypercube points one task accumulates before its partial combines.
///
/// Large enough that the `d`-element accumulator is amortised.
/// Small enough to keep every core fed at the heights this reduction runs at.
#[cfg(not(test))]
const CHUNK: usize = 1 << 12;

/// Small enough under test that a case above one variable splits in chunks.
///
/// The partial-combination fold is then on the path the tests take.
#[cfg(test)]
const CHUNK: usize = 1 << 1;

/// One ring-switching reduction at a bit alphabet, before the batching draw.
///
/// # Overview
///
/// The reduction turns a claim about a bit witness into one about its packing:
///
/// ```text
///     t(r) = s        a claim about the bits
///       ->  one degree-two sumcheck of l' rounds
///     t'(r') = s'     a claim about the packing, which a commitment answers
/// ```
///
/// # The order the protocol fixes
///
/// The batching challenge is drawn after the tensor element is bound.
/// So it cannot be a constructor argument.
///
/// ```text
///     new(r)          tensor, incoming_claim
///     bind            r, then the element, then draw r''
///     batch(r'')      initial_sum, closing_weight, weights
/// ```
///
/// Drawing `r''` first is unsound.
///
/// A bit matrix solving two `F_2`-linear systems moves the claim, sum held.
/// The rounds and the closing check then accept a true surviving claim.
///
/// # What the split settles, and what it leaves
///
/// It settles that `tensor` never needs the challenge.
/// Nothing is pushed into drawing one early just to obtain an element.
///
/// It does not settle the order.
///
/// The second stage borrows this one alone, so a caller can reach it first.
/// The forgery goes through with the element formed last.
///
/// Only a transcript binding `r` and the element before `r''` fixes that.
/// The two sides below own that transcript, so the ordering test is theirs.
///
/// # What it costs a verifier
///
/// Nothing witness-sized.
/// This stage holds the kept coordinates and one `d`-entry table.
///
/// Only `tensor` and `weights` touch the `2^l'` equality table.
/// Both are prover-side, so each builds it rather than every caller paying.
///
/// # Why a bit alphabet is different
///
/// A coordinate is one bit, so a product by a coordinate is a conditional add.
///
/// Both the tensor accumulation and the weight multilinear are subset sums.
/// A general alphabet needs `d` multiplications per hypercube point instead.
#[derive(Clone, Debug)]
pub struct BitRingSwitch<EF> {
    /// The evaluation point of the claim, over every variable of the witness.
    ///
    /// Held whole because the transcript binds it whole.
    point: Point<EF>,
    /// The equality table of the coordinates one packed element absorbs.
    eq_low: Poly<EF>,
}

impl<EF: TowerLevel> BitRingSwitch<EF> {
    /// Number of coordinates one packed element absorbs.
    pub const ABSORBED: usize = Coefficients::<EF>::LOG_DIMENSION;

    /// Set up the reduction of a claim at one point.
    ///
    /// The batching challenge is not an argument.
    /// It belongs after the element is bound, which the second stage is for.
    ///
    /// # Arguments
    ///
    /// The evaluation point of the claim, over every variable of the witness.
    ///
    /// # Errors
    ///
    /// Returns an error when the point is narrower than one element absorbs.
    pub fn new(r: &Point<EF>) -> Result<Self, BitRingSwitchError> {
        if r.num_variables() < Self::ABSORBED {
            return Err(BitRingSwitchError::PointTooNarrow {
                needed: Self::ABSORBED,
                actual: r.num_variables(),
            });
        }

        // The packing keeps the leading coordinates and absorbs the rest.
        let (_, low) = r.split_at(r.num_variables() - Self::ABSORBED);

        Ok(Self {
            eq_low: Poly::new_from_point(low.as_slice(), EF::ONE),
            point: r.clone(),
        })
    }

    /// Variables the packed polynomial this reduction runs against must have.
    ///
    /// This is also the number of sumcheck rounds the reduction takes.
    pub const fn num_variables(&self) -> usize {
        self.point.num_variables() - Self::ABSORBED
    }

    /// The coordinates of the evaluation point the packing keeps.
    ///
    /// They lead the point, so this is a borrow rather than a split.
    fn high(&self) -> &[EF] {
        &self.point.as_slice()[..self.num_variables()]
    }

    /// Move on to the stage the batching challenge opens.
    ///
    /// # Errors
    ///
    /// Returns an error unless the challenge names the absorbed coordinates.
    pub fn batch<'a>(
        &'a self,
        r_batch: &Point<EF>,
    ) -> Result<BitRingSwitchBatch<'a, EF>, BitRingSwitchError> {
        if r_batch.num_variables() != Self::ABSORBED {
            return Err(BitRingSwitchError::BatchWidthMismatch {
                expected: Self::ABSORBED,
                actual: r_batch.num_variables(),
            });
        }
        Ok(BitRingSwitchBatch {
            reduction: self,
            eq_batch: Poly::new_from_point(r_batch.as_slice(), EF::ONE),
        })
    }

    /// Where the equality table over the kept coordinates is nonzero, and its values there.
    ///
    /// # Algorithm
    ///
    /// The equality table factors over the variables.
    /// A coordinate that is already zero or one turns its factor into an indicator.
    ///
    /// ```text
    ///     high = (b_0 .. b_{p-1}, z_p .. z_{l'-1})   every b in {0, 1}
    ///     eq(high, w) = 0  unless the top p bits of w spell b
    /// ```
    ///
    /// The table is therefore supported on one run of `2^(l' - p)` consecutive elements,
    /// and on that run it is the equality table of the remaining coordinates alone.
    ///
    /// A claim about one column of a stacked trace arrives exactly this way, its slot
    /// address leading the row point.
    ///
    /// A point drawn from a transcript has no Boolean coordinate, so the run is the whole
    /// hypercube and this is the dense table.
    ///
    /// # Returns
    ///
    /// The first element of the run, and the table over it.
    fn support(&self) -> (usize, Poly<EF>) {
        // Read the leading Boolean coordinates as the address of the run, highest bit first.
        //
        // The equality table is indexed with the first coordinate as the highest bit, so
        // those coordinates select a run rather than a scattered set.
        let mut prefix = 0;
        let mut address = 0usize;
        for &coordinate in self.high() {
            if coordinate == EF::ZERO {
                address <<= 1;
            } else if coordinate == EF::ONE {
                address = (address << 1) | 1;
            } else {
                break;
            }
            prefix += 1;
        }

        // One element of the run per assignment of the coordinates that are left.
        let table = Poly::new_from_point(&self.high()[prefix..], EF::ONE);
        (address * table.num_evals(), table)
    }

    /// `sum_w eq(r_high, w) ⊗ t'(w)`, the element the prover sends.
    ///
    /// # Overview
    ///
    /// Read by columns it gives the bit planes at the kept coordinates.
    /// That is the reading which tests the incoming claim.
    ///
    /// Read by rows it gives the sumcheck its starting sum.
    ///
    /// The two readings are of the same coefficients.
    /// That is what makes a dishonest element catchable at all.
    ///
    /// # Performance
    ///
    /// Accumulated in place, one partial element per task.
    /// Forming each term separately would allocate `d` elements per point.
    ///
    /// The equality table is built here rather than held.
    /// That is strictly below the accumulation it feeds.
    ///
    /// It also keeps a verifier from paying for a table it never reads.
    ///
    /// Only the run the table is nonzero on is swept.
    /// A claim about one column of a stacked trace reaches `2^(l' - p)` of its `2^l'` elements.
    ///
    /// # Errors
    ///
    /// Returns an error unless the packing has the reduction's variables.
    pub fn tensor(&self, packing: &BitPacking<EF>) -> Result<BitTensor<EF>, BitRingSwitchError>
    where
        EF: Send + Sync,
    {
        self.check_width(packing.num_variables())?;

        // Elements outside the run weigh zero, so leaving them out changes no sum.
        let (offset, weights) = self.support();
        let values = &packing.poly().as_slice()[offset..offset + weights.num_evals()];

        Ok(weights
            .as_slice()
            .par_chunks(CHUNK)
            .zip(values.par_chunks(CHUNK))
            .par_fold_reduce(
                BitTensor::zero,
                |mut accumulator, (weights, values)| {
                    for (&weight, &value) in weights.iter().zip(values) {
                        accumulator.add_exterior_product(weight, value);
                    }
                    accumulator
                },
                |mut accumulator, partial| {
                    // Addition is associative, so regrouping cannot change it.
                    accumulator += partial;
                    accumulator
                },
            ))
    }

    /// What the claim being reduced must equal, given the element sent.
    ///
    /// # Algorithm
    ///
    /// Column `v` is the witness's bit plane `v` read at the kept coordinates.
    /// The absorbed coordinates weigh those planes back together:
    ///
    /// ```text
    ///     t(r) = sum_v eq(r_low, v) * column v
    /// ```
    ///
    /// This is the reduction's only use of the absorbed coordinates.
    #[must_use]
    pub fn incoming_claim(&self, tensor: &BitTensor<EF>) -> EF {
        tensor
            .columns()
            .iter()
            .zip(self.eq_low.as_slice())
            .map(|(&column, &weight)| column * weight)
            .sum()
    }

    /// Check that something names the variables this reduction runs over.
    const fn check_width(&self, actual: usize) -> Result<(), BitRingSwitchError> {
        if actual == self.num_variables() {
            Ok(())
        } else {
            Err(BitRingSwitchError::WidthMismatch {
                expected: self.num_variables(),
                actual,
            })
        }
    }

    /// The equality element at the kept coordinates and one other point.
    ///
    /// # Algorithm
    ///
    /// The equality polynomial factors over the variables:
    ///
    /// ```text
    ///     eq(X, Y) = prod_i ( X_i*Y_i + (1 - X_i)(1 - Y_i) )
    /// ```
    ///
    /// Each factor lands on a different tensor leg.
    /// From the identity, every variable adds an agree and a disagree term:
    ///
    /// ```text
    ///     agree     scale a copy's columns by a, its rows by b
    ///     disagree  scale the element's columns by 1 - a, its rows by 1 - b
    /// ```
    ///
    /// The two scalings commute, so either order gives the term.
    /// The alternative sums the element over the hypercube, exponentially.
    fn equality_element(&self, r_prime: &Point<EF>) -> BitTensor<EF> {
        let mut element = BitTensor::one();
        for i in 0..self.num_variables() {
            let (a, b) = (self.high()[i], r_prime[i]);

            let mut agree = element.clone();
            agree.scale_columns(a);
            agree.scale_rows(b);

            element.scale_columns(EF::ONE - a);
            element.scale_rows(EF::ONE - b);
            element += agree;
        }
        element
    }
}

/// The same reduction, once the batching challenge has been drawn.
///
/// Holding one of these is not evidence that the element was bound first.
/// Enforcing that is the transcript's job, as the stage above records.
#[derive(Clone, Debug)]
pub struct BitRingSwitchBatch<'a, EF> {
    /// The stage the evaluation point alone fixes.
    reduction: &'a BitRingSwitch<EF>,
    /// The equality table of the batching challenge.
    eq_batch: Poly<EF>,
}

impl<EF: TowerLevel> BitRingSwitchBatch<'_, EF> {
    /// The weight multilinear the sumcheck runs against the packing.
    ///
    /// # Algorithm
    ///
    /// The equality table decomposes over `F_2`.
    /// The challenge weighs those coordinates:
    ///
    /// ```text
    ///     A(w) = sum_u eq(u, r_batch) * coordinate u of eq(r_high, w)
    /// ```
    ///
    /// The coordinates are bits, so each entry is a subset sum.
    ///
    /// # Performance
    ///
    /// Prover-side, and the only other place the equality table is built.
    ///
    /// The sumcheck reads this dense, so every slot exists.
    /// Only the run the equality table is nonzero on is written, the rest staying zero.
    pub fn weights(&self) -> Poly<EF>
    where
        EF: Send + Sync,
    {
        let (offset, equality) = self.reduction.support();
        let mut table = Poly::zero(self.reduction.num_variables());

        // A zero weight has no set coordinate, so its slot would be written zero anyway.
        table.as_mut_slice()[offset..offset + equality.num_evals()]
            .par_iter_mut()
            .zip(equality.as_slice().par_iter())
            .for_each(|(slot, &value)| {
                *slot = Coefficients::of(value)
                    .iter_set()
                    .map(|u| self.eq_batch.as_slice()[u])
                    .sum();
            });
        table
    }

    /// The sum the reduction's sumcheck starts from.
    ///
    /// Derived from the element's rows rather than taken from the prover.
    /// That ties the sumcheck to the coefficients the claim was checked on.
    #[must_use]
    pub fn initial_sum(&self, tensor: &BitTensor<EF>) -> EF {
        self.batch_rows(tensor)
    }

    /// The weight the surviving claim is scaled by, where the rounds ended.
    ///
    /// # Overview
    ///
    /// This is the weight multilinear read at that point.
    /// Taken through the equality element, not by a pass over the weights.
    ///
    /// # Errors
    ///
    /// Returns an error unless the point names the reduction's variables.
    pub fn closing_weight(&self, r_prime: &Point<EF>) -> Result<EF, BitRingSwitchError> {
        self.reduction.check_width(r_prime.num_variables())?;
        Ok(self.batch_rows(&self.reduction.equality_element(r_prime)))
    }

    /// The rows of an element, batched against the batching challenge.
    ///
    /// Both ends of the sumcheck are this operation on a different element.
    fn batch_rows(&self, tensor: &BitTensor<EF>) -> EF {
        tensor
            .rows()
            .iter()
            .zip(self.eq_batch.as_slice())
            .map(|(&row, &weight)| row * weight)
            .sum()
    }
}

/// Reasons a bit-alphabet reduction cannot be set up or run.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum BitRingSwitchError {
    /// The evaluation point does not cover the coordinates one element absorbs.
    #[error("the evaluation point names {actual} variables, at least {needed} are needed")]
    PointTooNarrow {
        /// Coordinates one packed element absorbs.
        needed: usize,
        /// Coordinates the point names.
        actual: usize,
    },
    /// The batching challenge does not name the absorbed coordinates exactly.
    ///
    /// A shorter one would zip-truncate, dropping identity rows.
    /// Both sides move the same way, so no later check would catch it.
    #[error("the batching challenge names {actual} variables, expected {expected}")]
    BatchWidthMismatch {
        /// Coordinates the reduction absorbs.
        expected: usize,
        /// Coordinates the challenge names.
        actual: usize,
    },
    /// A polynomial or point does not name the variables the reduction uses.
    #[error("the reduction runs over {expected} variables, got {actual}")]
    WidthMismatch {
        /// Variables the reduction runs over.
        expected: usize,
        /// Variables the argument names.
        actual: usize,
    },
}

/// The messages one bit-alphabet reduction puts on the wire.
///
/// The element travels by rows, one bit per matrix entry:
///
/// ```text
///     by rows          d elements
///     byte per entry   d^2 elements
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "EF: TowerLevel", deserialize = "EF: TowerLevel"))]
pub struct BitRingSwitchProof<EF> {
    /// The tensor element both checks read, by rows and by columns.
    pub tensor: BitTensor<EF>,
    /// The rounds of the batched degree-two sumcheck.
    pub sumcheck: SumcheckData<EF, EF>,
    /// The value of the surviving claim.
    pub final_eval: EF,
}

/// Why a bit-alphabet reduction was rejected.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum BitRingSwitchProofError {
    /// A list crossing the wire is not the width the description fixes.
    #[error(transparent)]
    Width(#[from] TranscriptWidth),

    /// The reduction could not be run over the point it was set up at.
    #[error(transparent)]
    Reduction(#[from] BitRingSwitchError),

    /// The claimed evaluation is not the column reading of the element.
    #[error("the claimed evaluation is not the column reading of the tensor element")]
    ClaimMismatch,

    /// A round of the batched sumcheck failed.
    #[error(transparent)]
    Sumcheck(#[from] SumcheckError),

    /// The surviving claim does not close the sumcheck against the closing weight.
    #[error("the surviving claim does not close the sumcheck")]
    FinalCheck,

    /// The sumcheck carries grinding witnesses this reduction never searches for.
    #[error("the sumcheck carries {actual} proof-of-work witnesses, expected none")]
    NonEmptyPowWitnesses {
        /// Witnesses the proof supplied.
        actual: usize,
    },
}

/// The two sides of the reduction, each over the transcript the other replays.
///
/// # A reduction, not a filter
///
/// A false input claim is not rejected outright.
/// It survives as a false surviving claim, except with the probability below.
///
/// Two cases the rounds cannot catch are left to whatever discharges the claim:
///
/// - a closing weight of zero, which constrains the surviving value not at all
/// - a tampered element, with the rest of the proof adapted to the sum it implies
///
/// # Booleanity is free
///
/// The packing is a bijection between bit strings and elements of the level.
/// Every bit pattern is an element, and every element is some bit pattern.
///
/// A commitment to a packed multilinear is therefore a commitment to a bit witness.
/// No range check, no auxiliary constraint, and nothing here to verify.
///
/// # Soundness
///
/// The reduction's own error is `(d_log + 2 l') / |EF|` (eprint 2024/504, Theorem 3.5).
/// It runs over `d_log` absorbed coordinates and `l'` rounds:
///
/// - `d_log / |EF|` from the batching draw that collapses the row claims into one.
/// - `2 l' / |EF|` for the rounds of degree-two sumcheck.
///
/// Both terms are per-attempt, because the description holds no grinding step.
/// A protocol needing a total bound supplies the grinding outside this run.
impl<EF: TranscriptField + TowerLevel> BitRingSwitch<EF> {
    /// Reduce the claim this reduction was set up over to one about the packing.
    ///
    /// # Returns
    ///
    /// The proof, the point the rounds ended at, and the surviving claim's value.
    ///
    /// # Panics
    ///
    /// Panics unless the packing has the variables the evaluation point leaves.
    pub fn prove<Challenger>(
        &self,
        packing: &BitPacking<EF>,
        challenger: &mut Challenger,
    ) -> (BitRingSwitchProof<EF>, Point<EF>, EF)
    where
        EF: Send + Sync,
        Challenger: FieldChallenger<EF> + GrindingChallenger<Witness = EF>,
    {
        let rounds = self.num_variables();
        assert_eq!(
            packing.num_variables(),
            rounds,
            "the packing must have the {rounds} variables the evaluation point leaves"
        );

        let tensor = self
            .tensor(packing)
            .expect("the packing was just checked against the reduction");

        // The element is a function of the kept coordinates alone.
        // It is therefore ready before the transcript needs it.
        let shape = BitRingSwitchShape::new(self.point.num_variables());
        let mut transcript =
            BitRingSwitchProverTranscript::<Challenger, EF>::new(challenger, shape);
        let r_batch = transcript.statement(&self.point, tensor.rows());

        // The batching challenge arrived after the element, which is the order soundness needs.
        let batch = self
            .batch(&r_batch)
            .expect("the draw names the absorbed coordinates by construction");
        let poly = ProductPolynomial::new_unpacked(
            VariableOrder::Prefix,
            packing.poly().clone(),
            batch.weights(),
        );
        let mut prover = SumcheckProver::new(poly, batch.initial_sum(&tensor));
        let mut sumcheck = SumcheckData::default();

        let r_prime = transcript.batched_sumcheck(|challenger| {
            prover.compute_sumcheck_polynomials(&mut sumcheck, challenger, rounds, 0, None)
        });

        // After the last round the evaluation side has folded to the packing at that point.
        // No second pass over the packing is needed to find it.
        let final_eval = prover.evals().as_slice()[0];
        transcript.surviving_claim(final_eval);
        transcript.finish();

        (
            BitRingSwitchProof {
                tensor,
                sumcheck,
                final_eval,
            },
            r_prime,
            final_eval,
        )
    }

    /// Replay the reduction and return the claim it leaves behind, as a point and a value.
    ///
    /// Discharging that pair against a commitment to the packing is the caller's business.
    ///
    /// # Errors
    ///
    /// - A malformed element.
    /// - A non-empty grinding witness list, since this reduction never grinds.
    /// - A claimed evaluation disagreeing with the element's columns.
    /// - A failed sumcheck round, or a final claim that does not close it.
    pub fn verify<Challenger>(
        &self,
        proof: &BitRingSwitchProof<EF>,
        claimed_sum: EF,
        challenger: &mut Challenger,
    ) -> Result<(Point<EF>, EF), BitRingSwitchProofError>
    where
        Challenger: FieldChallenger<EF> + GrindingChallenger<Witness = EF>,
    {
        // Both structural rejections below run before the challenger is touched.
        // A malformed proof therefore never leaves a half-advanced transcript.
        if !proof.tensor.is_well_formed() {
            return Err(TranscriptWidth::TensorRows {
                expected: BitTensor::<EF>::DIMENSION,
                actual: proof.tensor.rows().len(),
            }
            .into());
        }
        if !proof.sumcheck.pow_witnesses.is_empty() {
            return Err(BitRingSwitchProofError::NonEmptyPowWitnesses {
                actual: proof.sumcheck.pow_witnesses.len(),
            });
        }

        let rounds = self.num_variables();
        let shape = BitRingSwitchShape::new(self.point.num_variables());
        let mut transcript =
            BitRingSwitchVerifierTranscript::<Challenger, EF>::new(challenger, shape);
        let r_batch = transcript.statement(&self.point, proof.tensor.rows())?;

        // The columns are the witness's bit planes at the kept coordinates.
        // The absorbed coordinates weigh them back together, their only use here.
        if self.incoming_claim(&proof.tensor) != claimed_sum {
            transcript.abort();
            return Err(BitRingSwitchProofError::ClaimMismatch);
        }

        let batch = self
            .batch(&r_batch)
            .expect("the draw names the absorbed coordinates by construction");

        // The initial sum is derived from the element's rows, never taken from the prover.
        // That is what makes a dishonest element catchable at all.
        let mut sum = batch.initial_sum(&proof.tensor);
        let replay = transcript.batched_sumcheck(|challenger| {
            proof
                .sumcheck
                .verify_rounds(challenger, &mut sum, rounds, 0, Basis::Evaluation)
        });
        let r_prime = match replay {
            Ok(point) => point,
            Err(error) => {
                transcript.abort();
                return Err(error.into());
            }
        };

        transcript.surviving_claim(proof.final_eval);
        transcript.finish();

        // The rounds close on the weight multilinear at their end point, times the value.
        // The weight comes through the equality element rather than another pass.
        let closing = batch.closing_weight(&r_prime)?;
        if sum != closing * proof.final_eval {
            return Err(BitRingSwitchProofError::FinalCheck);
        }

        Ok((r_prime, proof.final_eval))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_binary_field::{BinaryChallenger, BinaryField16};
    use p3_challenger::HashChallenger;
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_keccak::Keccak256Hash;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type EF = BinaryField16;
    type Chal = BinaryChallenger<EF, HashChallenger<u8, Keccak256Hash, 32>>;

    /// A fresh sponge, so the two sides start from the same state.
    fn challenger() -> Chal {
        Chal::from_hasher(Vec::new(), Keccak256Hash)
    }

    /// A random bit witness of the given byte length.
    fn bits(seed: u64, bytes: usize) -> Vec<u8> {
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..bytes).map(|_| rng.random::<u8>()).collect()
    }

    /// One reduction over a witness of the given byte length.
    ///
    /// The batching challenge comes back beside it rather than folded in.
    /// A test then reaches the second stage the way the protocol does.
    fn fixture(
        seed: u64,
        bytes: usize,
    ) -> (BitRingSwitch<EF>, BitPacking<EF>, Point<EF>, Point<EF>) {
        let packing = BitPacking::<EF>::new(&bits(seed, bytes)).unwrap();
        let mut rng = SmallRng::seed_from_u64(seed ^ 0xFFFF);

        // The whole point, of which the packing keeps the leading coordinates.
        let r = Point::<EF>::rand(
            &mut rng,
            packing.num_variables() + BitRingSwitch::<EF>::ABSORBED,
        );
        let r_batch = Point::<EF>::rand(&mut rng, BitRingSwitch::<EF>::ABSORBED);
        let reduction = BitRingSwitch::new(&r).unwrap();

        (reduction, packing, r, r_batch)
    }

    /// The witness as a multilinear over every variable, one element per bit.
    fn embedded(witness: &[u8]) -> Poly<EF> {
        Poly::new(
            (0..witness.len() * 8)
                .map(|cell| {
                    if (witness[cell / 8] >> (cell % 8)) & 1 == 1 {
                        EF::ONE
                    } else {
                        EF::ZERO
                    }
                })
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn the_reduction_runs_over_the_variables_the_packing_keeps() {
        // Fixture state: 128 cells is 7 variables, of which 16 bits absorb 4.
        let (reduction, packing, _, _) = fixture(0x5AE, 16);

        assert_eq!(BitRingSwitch::<EF>::ABSORBED, 4);
        assert_eq!(reduction.num_variables(), 3);
        assert_eq!(packing.num_variables(), 3);
    }

    /// The equality table over the kept coordinates, dense over the whole hypercube.
    ///
    /// The production path computes the same table without the run of zeros a Boolean
    /// coordinate puts in it, so this is what every case below checks it against.
    fn dense_eq_high(reduction: &BitRingSwitch<EF>) -> Poly<EF> {
        Poly::new_from_point(
            &reduction.point.as_slice()[..reduction.num_variables()],
            EF::ONE,
        )
    }

    #[test]
    fn the_incoming_claim_is_the_witness_at_the_point() {
        // Invariant: what the reduction demands of the claim is the witness's.
        //
        //     sum_v eq(r_low, v) * column v  ==  t(r)
        //
        // That identity is what the reduction's first check rests on.
        let witness = bits(0xC1A1, 16);
        let packing = BitPacking::<EF>::new(&witness).unwrap();
        let mut rng = SmallRng::seed_from_u64(0xF001);
        let r = Point::<EF>::rand(&mut rng, 7);
        let reduction = BitRingSwitch::new(&r).unwrap();

        let tensor = reduction.tensor(&packing).unwrap();

        assert_eq!(
            reduction.incoming_claim(&tensor),
            embedded(&witness).eval_base(&r)
        );
    }

    #[test]
    fn the_columns_are_the_bit_planes_at_the_kept_coordinates() {
        // Invariant: column `v` is cells `d*w + v` read at `r_high`.
        // Checked column by column, so a failure localises.
        let (reduction, packing, _, _) = fixture(0xB1A, 16);
        let eq = dense_eq_high(&reduction);

        for (v, &column) in reduction
            .tensor(&packing)
            .unwrap()
            .columns()
            .iter()
            .enumerate()
        {
            let expected: EF = (0..packing.len())
                .filter(|&w| packing.coefficients(w).get(v))
                .map(|w| eq.as_slice()[w])
                .sum();
            assert_eq!(column, expected, "column {v}");
        }
    }

    #[test]
    fn each_row_is_its_own_weighted_sum() {
        // Invariant: row `u` is the packing weighted by coordinate `u` of eq.
        //
        //     row u  =  sum_w A_{w,u} * t'(w)
        let (reduction, packing, _, _) = fixture(0x0A5, 16);
        let eq = dense_eq_high(&reduction);

        for (u, &row) in reduction
            .tensor(&packing)
            .unwrap()
            .rows()
            .iter()
            .enumerate()
        {
            let expected: EF = (0..packing.len())
                .filter(|&w| Coefficients::of(eq.as_slice()[w]).get(u))
                .map(|w| packing.poly().as_slice()[w])
                .sum();
            assert_eq!(row, expected, "row {u}");
        }
    }

    #[test]
    fn a_weight_is_the_subset_sum_the_coordinates_select() {
        // Invariant: the weight at `w` adds the table where bits are set.
        let (reduction, _, _, r_batch) = fixture(0x5E7, 16);
        let batch = reduction.batch(&r_batch).unwrap();
        let weights = batch.weights();

        // Built from the challenge here, not read off the stage under test.
        let eq_batch = Poly::<EF>::new_from_point(r_batch.as_slice(), EF::ONE);

        for (w, &value) in dense_eq_high(&reduction).as_slice().iter().enumerate() {
            let expected: EF = Coefficients::of(value)
                .iter()
                .zip(eq_batch.as_slice())
                .filter(|&(bit, _)| bit)
                .map(|(_, &weight)| weight)
                .sum();
            assert_eq!(weights.as_slice()[w], expected, "point {w}");
        }
        assert_eq!(weights.num_variables(), reduction.num_variables());
    }

    #[test]
    fn the_initial_sum_is_the_sum_the_sumcheck_proves() {
        // Invariant: the weighted sum over the hypercube is the batched rows.
        //
        //     sum_w A(w) * t'(w)  ==  sum_u eq(u, r_batch) * row u
        //
        // Both sides are derived, never sent, which a dishonest element hits.
        let (reduction, packing, _, r_batch) = fixture(0xBA7, 32);
        let batch = reduction.batch(&r_batch).unwrap();
        let tensor = reduction.tensor(&packing).unwrap();

        let dot: EF = batch
            .weights()
            .as_slice()
            .iter()
            .zip(packing.poly().as_slice())
            .map(|(&weight, &value)| weight * value)
            .sum();

        assert_eq!(dot, batch.initial_sum(&tensor));
    }

    #[test]
    fn the_batched_rows_weigh_by_the_challenge_itself() {
        // Invariant: row `u` carries the equality weight at point `u`.
        //
        //     initial_sum(s_hat) == sum_u eq(hypercube(u, k), r'') * row u
        //
        // Every other batch-stage test reads the table this stage holds.
        // A stage ignoring the challenge would pass all of them.
        //
        // That is the forgery with a challenge every prover knows.
        //
        // So the reference here is built from the challenge itself.
        let (reduction, packing, _, r_batch) = fixture(0xC4A, 16);
        let batch = reduction.batch(&r_batch).unwrap();
        let tensor = reduction.tensor(&packing).unwrap();
        let absorbed = BitRingSwitch::<EF>::ABSORBED;

        let expected: EF = tensor
            .rows()
            .iter()
            .enumerate()
            .map(|(u, &row)| {
                let point = Point::<EF>::hypercube(u, absorbed);
                row * Point::eval_eq(point.as_slice(), r_batch.as_slice())
            })
            .sum();

        assert_eq!(batch.initial_sum(&tensor), expected);
    }

    #[test]
    fn the_closing_weight_is_the_weight_multilinear_at_the_point() {
        // Invariant: the element closes the sumcheck with no weights pass.
        //
        //     closing_weight(r')  ==  A(r')
        let (reduction, _, _, r_batch) = fixture(0xA7, 16);
        let batch = reduction.batch(&r_batch).unwrap();
        let r_prime = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xA8), 3);

        assert_eq!(
            batch.closing_weight(&r_prime).unwrap(),
            batch.weights().eval_base(&r_prime)
        );
    }

    #[test]
    fn the_equality_element_matches_its_hypercube_definition() {
        // The recurrence is linear in the variables.
        // The definition below is exponential.
        //
        //     e = sum_w eq(r_high, w) ⊗ eq(w, r')
        let mut rng = SmallRng::seed_from_u64(0xE9);
        for num_variables in 0..5 {
            let r = Point::<EF>::rand(&mut rng, num_variables + BitRingSwitch::<EF>::ABSORBED);
            let reduction = BitRingSwitch::new(&r).unwrap();
            let r_prime = Point::<EF>::rand(&mut rng, num_variables);

            let eq_prime = Poly::<EF>::new_from_point(r_prime.as_slice(), EF::ONE);
            let mut expected = BitTensor::zero();
            let eq_high = dense_eq_high(&reduction);
            for (&a, &b) in eq_high.as_slice().iter().zip(eq_prime.as_slice()) {
                expected.add_exterior_product(a, b);
            }

            assert_eq!(
                reduction.equality_element(&r_prime),
                expected,
                "{num_variables} variables"
            );
        }
    }

    #[test]
    fn a_point_narrower_than_one_element_is_refused() {
        // Fewer coordinates than one element absorbs leaves nothing to keep.
        let mut rng = SmallRng::seed_from_u64(0xBAD);
        let r = Point::<EF>::rand(&mut rng, 3);

        assert_eq!(
            BitRingSwitch::new(&r).unwrap_err(),
            BitRingSwitchError::PointTooNarrow {
                needed: 4,
                actual: 3
            }
        );
    }

    #[test]
    fn a_batching_challenge_of_the_wrong_width_is_refused() {
        // A short challenge would zip-truncate, dropping identity rows.
        // Both sides move the same way, so no later check would catch it.
        let mut rng = SmallRng::seed_from_u64(0xBAD2);
        let r = Point::<EF>::rand(&mut rng, 7);
        let reduction = BitRingSwitch::new(&r).unwrap();

        for width in [3usize, 5] {
            let r_batch = Point::<EF>::rand(&mut rng, width);
            assert_eq!(
                reduction.batch(&r_batch).unwrap_err(),
                BitRingSwitchError::BatchWidthMismatch {
                    expected: 4,
                    actual: width,
                }
            );
        }
    }

    #[test]
    fn a_packing_of_the_wrong_width_is_refused() {
        // A mismatched packing would zip-truncate the accumulation.
        let (reduction, _, _, r_batch) = fixture(0xBAD3, 16);
        let batch = reduction.batch(&r_batch).unwrap();
        let wider = BitPacking::<EF>::new(&bits(0xBAD4, 32)).unwrap();

        assert_eq!(
            reduction.tensor(&wider).unwrap_err(),
            BitRingSwitchError::WidthMismatch {
                expected: 3,
                actual: 4
            }
        );
        assert_eq!(
            batch
                .closing_weight(&Point::<EF>::new(alloc::vec![EF::ONE; 4]))
                .unwrap_err(),
            BitRingSwitchError::WidthMismatch {
                expected: 3,
                actual: 4
            }
        );
    }

    #[test]
    fn a_boolean_prefix_leaves_the_element_and_the_weights_unchanged() {
        // Invariant: skipping the zeros of the equality table changes no sum.
        //
        // A claim about one column of a stacked trace arrives with a Boolean slot address
        // leading its row point, and that prefix makes the table an indicator:
        //
        //     high = (1, 0, z)   ->  eq(high, w) = 0 unless w is 100 or 101
        //
        // Fixture state: 2^7 cells, 4 absorbed, so 3 kept variables and 8 elements.
        //
        //     prefix 0   ->  the run is all 8, which is the dense case
        //     prefix 1   ->  4 of 8
        //     prefix 2   ->  2 of 8
        //     prefix 3   ->  1 of 8, the whole table an indicator
        //
        // The dense reference forms every term, zeros included, and must agree with each.
        let witness = bits(0xB0015, 16);
        let packing = BitPacking::<EF>::new(&witness).unwrap();
        let mut rng = SmallRng::seed_from_u64(0xB0016);

        for prefix in 0..=3usize {
            for address in 0..1usize << prefix {
                // Leading coordinates spell the address, highest bit first.
                let mut coordinates: Vec<EF> = (0..prefix)
                    .map(|bit| {
                        if (address >> (prefix - 1 - bit)) & 1 == 1 {
                            EF::ONE
                        } else {
                            EF::ZERO
                        }
                    })
                    .collect();
                // The rest of the point is ordinary field randomness.
                coordinates.extend((prefix..7).map(|_| rng.random::<EF>()));
                let r = Point::new(coordinates);

                let reduction = BitRingSwitch::new(&r).unwrap();
                let r_batch = Point::<EF>::rand(&mut rng, BitRingSwitch::<EF>::ABSORBED);
                let batch = reduction.batch(&r_batch).unwrap();

                // The element, against the term-by-term sum over the whole hypercube.
                let mut expected = BitTensor::zero();
                for (&weight, &value) in dense_eq_high(&reduction)
                    .as_slice()
                    .iter()
                    .zip(packing.poly().as_slice())
                {
                    expected.add_exterior_product(weight, value);
                }
                assert_eq!(
                    reduction.tensor(&packing).unwrap(),
                    expected,
                    "prefix {prefix} address {address}"
                );

                // The weight multilinear, against the same table read subset sum by subset sum.
                let eq_batch = Poly::<EF>::new_from_point(r_batch.as_slice(), EF::ONE);
                let weights = batch.weights();
                assert_eq!(weights.num_variables(), reduction.num_variables());
                for (w, &value) in dense_eq_high(&reduction).as_slice().iter().enumerate() {
                    let want: EF = Coefficients::of(value)
                        .iter()
                        .zip(eq_batch.as_slice())
                        .filter(|&(bit, _)| bit)
                        .map(|(_, &weight)| weight)
                        .sum();
                    assert_eq!(weights.as_slice()[w], want, "prefix {prefix} point {w}");
                }
            }
        }
    }

    proptest! {
        #[test]
        fn the_chunked_accumulation_matches_the_term_by_term_sum(seed: u64, log_n in 1usize..6) {
            // The parallel path accumulates in chunks, combining partials.
            // The reference forms every term as its own element and adds them.
            let (reduction, packing, _, _) = fixture(seed, (1 << log_n) * 2);

            let mut expected = BitTensor::zero();
            for (&weight, &value) in dense_eq_high(&reduction)
                .as_slice()
                .iter()
                .zip(packing.poly().as_slice())
            {
                expected += BitTensor::exterior_product(weight, value);
            }

            prop_assert_eq!(reduction.tensor(&packing).unwrap(), expected);
        }
    }

    #[test]
    fn the_reduction_round_trips_and_leaves_a_true_claim() {
        // Invariant: the surviving claim is the truth about the packed polynomial.
        //
        //     in    t(r) = s   at a random point over all 8 variables
        //     out   t'(r')     over the 4 the packing keeps, 4 being absorbed
        let witness = bits(0x81A5, 32);
        let packing = BitPacking::<EF>::new(&witness).unwrap();
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0x81A6), 8);
        let reduction = BitRingSwitch::new(&r).unwrap();
        let claim = embedded(&witness).eval_base(&r);

        let mut prover_chal = challenger();
        let (proof, r_prime_p, s_prime_p) = reduction.prove(&packing, &mut prover_chal);

        let mut verifier_chal = challenger();
        let (r_prime_v, s_prime_v) = reduction.verify(&proof, claim, &mut verifier_chal).unwrap();

        assert_eq!(r_prime_p, r_prime_v);
        assert_eq!(s_prime_p, s_prime_v);
        assert_eq!(s_prime_v, packing.poly().eval_base(&r_prime_v));
    }

    #[test]
    fn a_claim_the_element_does_not_support_is_rejected() {
        // The column reading is what ties the element to the incoming claim.
        let witness = bits(0x0AD, 32);
        let packing = BitPacking::<EF>::new(&witness).unwrap();
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0x0AE), 8);
        let reduction = BitRingSwitch::new(&r).unwrap();
        let claim = embedded(&witness).eval_base(&r);

        let (proof, _, _) = reduction.prove(&packing, &mut challenger());

        let err = reduction
            .verify(&proof, claim + EF::ONE, &mut challenger())
            .unwrap_err();
        assert_eq!(err, BitRingSwitchProofError::ClaimMismatch);
    }

    #[test]
    fn a_tampered_element_is_caught_by_whichever_reading_it_moves() {
        // Invariant: both readings are of the same coefficients.
        //
        //     rows    -> the sum the rounds start from
        //     columns -> the claim they are checked against
        //
        // A tamper has to survive the column reading before the rounds ever run.
        // The two cases below are therefore checked separately.
        let witness = bits(0x7A17, 32);
        let packing = BitPacking::<EF>::new(&witness).unwrap();
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0x7A18), 8);
        let reduction = BitRingSwitch::new(&r).unwrap();
        let claim = embedded(&witness).eval_base(&r);
        let (honest, _, _) = reduction.prove(&packing, &mut challenger());

        // Case 1: add one to a row.
        //
        // One is the first basis coordinate, so this flips entry (3, 0) of the matrix.
        // That moves column 0 as well, so the column reading moves with it.
        //
        // The run stops at the claim check without reaching a single round.
        let mut flipped = honest.clone();
        let mut rows = flipped.tensor.rows().to_vec();
        rows[3] += EF::ONE;
        flipped.tensor = BitTensor::try_from(rows).unwrap();

        assert_eq!(
            reduction
                .verify(&flipped, claim, &mut challenger())
                .unwrap_err(),
            BitRingSwitchProofError::ClaimMismatch,
        );

        // Case 2: a delta the column reading cannot see.
        //
        // The claim check weighs the columns by the absorbed coordinates:
        //
        //     claim   = sum_v column_v * eq_low[v]
        //     hidden  when  d_0 * eq_low[0] + d_1 * eq_low[1] = 0
        //
        // Over characteristic two that is d_1 = d_0 * eq_low[0] / eq_low[1].
        // Nothing constrains d_0, so any nonzero value does.
        //
        // It comes from a byte pattern, since an integer would reduce modulo two.
        let eq_low = reduction.eq_low.as_slice();
        assert_ne!(
            eq_low[1],
            EF::ZERO,
            "the construction divides by this entry"
        );

        let d_0 = EF::from_le_byte_iter([0x2C, 0x9B].into_iter());
        let d_1 = d_0 * eq_low[0] * eq_low[1].inverse();
        assert_ne!(d_0, EF::ZERO);

        // A row list read as columns is the transpose, which is its own inverse.
        let mut delta_columns = alloc::vec![EF::ZERO; BitTensor::<EF>::DIMENSION];
        delta_columns[0] = d_0;
        delta_columns[1] = d_1;
        let delta = BitTensor::try_from(delta_columns).unwrap().columns();

        let mut hidden = honest.clone();
        let rows = hidden
            .tensor
            .rows()
            .iter()
            .zip(&delta)
            .map(|(&row, &shift)| row + shift)
            .collect::<Vec<_>>();
        hidden.tensor = BitTensor::try_from(rows).unwrap();

        // The element genuinely changed, yet its column reading did not.
        assert_ne!(hidden.tensor, honest.tensor);
        assert_eq!(
            reduction.incoming_claim(&hidden.tensor),
            reduction.incoming_claim(&honest.tensor),
        );

        // The row reading is what the rounds start from, and that one moved.
        //
        // No round rejects on its own.
        //
        //     sent      one value of the round polynomial, and its leading coefficient
        //     derived   the other value, from the running claim, never checked
        //
        //     wrong starting sum  ->  survives every round
        //     closing check       ->  the first place the two sides disagree
        assert_eq!(
            reduction
                .verify(&hidden, claim, &mut challenger())
                .unwrap_err(),
            BitRingSwitchProofError::FinalCheck,
        );

        // The honest element passes the same path, so the rejection is the delta.
        assert!(reduction.verify(&honest, claim, &mut challenger()).is_ok());
    }

    #[test]
    fn a_stray_grinding_witness_is_refused_before_the_transcript() {
        // Invariant: this reduction never grinds, so a witness rides along unbound.
        //
        // The rejection is structural, so it may not advance the sponge.
        //
        // A short element cannot be built at all.
        // Every route into one checks the row count, deserialization included.
        //
        // The verifier's shape check is therefore defence in depth, not a reachable path.
        let witness = bits(0x5407, 32);
        let packing = BitPacking::<EF>::new(&witness).unwrap();
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0x5408), 8);
        let reduction = BitRingSwitch::new(&r).unwrap();
        let claim = embedded(&witness).eval_base(&r);

        let (mut proof, _, _) = reduction.prove(&packing, &mut challenger());
        proof.sumcheck.pow_witnesses.push(EF::ONE);

        assert_eq!(
            reduction
                .verify(&proof, claim, &mut challenger())
                .unwrap_err(),
            BitRingSwitchProofError::NonEmptyPowWitnesses { actual: 1 }
        );

        // A row count the level does not admit has no constructor.
        assert!(BitTensor::<EF>::try_from(alloc::vec![EF::ONE; 4]).is_err());
    }

    #[test]
    fn a_point_the_prover_did_not_run_over_is_rejected() {
        // Invariant: the two sides must run over the same point, not merely the same width.
        //
        // The other point keeps the absorbed coordinates and moves one kept coordinate.
        // The column reading is then unchanged, so the claim check passes and the rest must catch it.
        //
        // That is the discriminating case.
        // Moving an absorbed coordinate would stop at the claim check instead.
        let witness = bits(0x9107, 32);
        let packing = BitPacking::<EF>::new(&witness).unwrap();
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0x9108), 8);
        let claim = embedded(&witness).eval_base(&r);
        let reduction = BitRingSwitch::new(&r).unwrap();
        let (proof, _, _) = reduction.prove(&packing, &mut challenger());

        let mut moved = r.as_slice().to_vec();
        moved[0] += EF::ONE;
        let elsewhere = BitRingSwitch::new(&Point::new(moved)).unwrap();

        let err = elsewhere
            .verify(&proof, claim, &mut challenger())
            .unwrap_err();
        assert_ne!(err, BitRingSwitchProofError::ClaimMismatch, "{err:?}");
    }

    proptest! {
        // Each case runs a full reduction on both sides, so a few dozen keep the suite fast.
        #![proptest_config(ProptestConfig { cases: 24, ..ProptestConfig::default() })]

        /// Every witness length the level admits, at a random point each time.
        #[test]
        fn the_reduction_round_trips_over_random_inputs(
            log_bytes in 1usize..=6,
            witness_seed: u64,
            point_seed: u64,
        ) {
            let witness = bits(witness_seed, 1 << log_bytes);
            let packing = BitPacking::<EF>::new(&witness).unwrap();
            let variables = log_bytes + 3;
            let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(point_seed), variables);
            let reduction = BitRingSwitch::new(&r).unwrap();
            let claim = embedded(&witness).eval_base(&r);

            let (proof, _, s_prime_p) = reduction.prove(&packing, &mut challenger());
            let (r_prime, s_prime) =
                reduction.verify(&proof, claim, &mut challenger()).unwrap();

            prop_assert_eq!(s_prime, s_prime_p);
            prop_assert_eq!(s_prime, packing.poly().eval_base(&r_prime));
        }
    }
}
