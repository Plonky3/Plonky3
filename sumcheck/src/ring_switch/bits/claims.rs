//! Several bit-alphabet claims about one packing, reduced by one sumcheck.
//!
//! One reduction per claim leaves one surviving claim per point:
//!
//! ```text
//!     k claims  ->  k sumchecks of l' rounds  ->  k points the commitment opens
//! ```
//!
//! Every claim's rounds prove a sum of the same shape against the same packing:
//!
//! ```text
//!     sum_w t'(w) * A_i(w) = sigma_i
//! ```
//!
//! So a random combination of the sums is one sum of the same shape:
//!
//! ```text
//!     sum_w t'(w) * ( sum_i lambda^i * A_i(w) ) = sum_i lambda^i * sigma_i
//! ```
//!
//! One sumcheck proves it, and leaves one point the commitment opens:
//!
//! ```text
//!     k claims  ->  one sumcheck of l' rounds  ->  one point
//! ```
//!
//! # What each claim still sends
//!
//! Its tensor element, and its successor elements when it has them.
//! Each claim's readings are still checked against its own element's columns.
//!
//! Only the rounds and the surviving claim are shared.
//!
//! # The order the protocol fixes
//!
//! ```text
//!     bind      every claim's point, then every claim's elements
//!     draw      r'' once, [alpha once,] lambda
//!     prove     one sumcheck against sum_i lambda^i * A_i
//! ```
//!
//! Every draw comes after every element is bound.
//! A draw taken between two claims would let the later element be chosen to suit it.
//!
//! # Soundness
//!
//! Write `D_i(u)` for the row error of claim `i`'s elements, already folded under `alpha`.
//! A wrong element leaves the batched starting sum off by
//!
//! ```text
//!     sum_i lambda^i * sum_u eq(r'', u) * D_i(u)
//! ```
//!
//! That is a nonzero polynomial in `(r'', alpha, lambda)` of total degree at most
//!
//! ```text
//!     d_log  +  (K - 1)  +  (k - 1)
//! ```
//!
//! The rounds then add `2 l'` once, not once per claim:
//!
//! ```text
//!     batched      (d_log + K - 1 + (k - 1) + 2 l') / |EF|
//!     per claim    k * (d_log + K - 1 + 2 l') / |EF|
//! ```
//!
//! The added source is the `(k - 1) / |EF|` of `lambda` (Schwartz-Zippel).
//! The commitment then pays for one opened point rather than `k`.
//!
//! Every term is per-attempt, as for the one-claim reduction: nothing here grinds.
//!
//! # A batch of one
//!
//! A single claim plays the one-claim reduction, transcript and all.
//! Its batch draws no `lambda` and its proof carries the same messages.

use alloc::vec::Vec;
use core::borrow::Borrow;

use p3_binary_field::{BitCoordinates, TowerLevel};
use p3_challenger::fs::TranscriptField;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use serde::{Deserialize, Serialize};

use super::packing::BitPacking;
use super::reduction::{
    BitRingSwitch, BitRingSwitchBatch, BitRingSwitchError, BitRingSwitchProof,
    BitRingSwitchProofError, SuccessorTensors, check_no_grinding, slot_packing,
};
use super::tensor::BitTensor;
use super::transcript::{
    BitRingSwitchClaimsProverTranscript, BitRingSwitchClaimsShape,
    BitRingSwitchClaimsVerifierTranscript, ClaimStatement, ClaimsDraws,
};
use crate::data::SumcheckData;
use crate::strategy::{Basis, FromTable, IntoTranscriptField, ReprSumcheckProver, VariableOrder};

/// Several claims about one packing, each at a point of its own, reduced together.
///
/// Every claim is a [`BitRingSwitch`] set up at its own point.
/// All of them name the same number of variables, since they read the same packing.
#[derive(Clone, Debug)]
pub struct BitRingSwitchClaims<F, EF = F> {
    /// One reduction per claim, in the order the claims are bound.
    reductions: Vec<BitRingSwitch<F, EF>>,
}

/// What one claim of a batch sends: its tensor element, and its successor elements.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: BitCoordinates, EF: BitCoordinates",
    deserialize = "F: BitCoordinates, EF: BitCoordinates"
))]
pub struct ClaimElements<F, EF = F> {
    /// The tensor element both checks read, by rows and by columns.
    pub tensor: BitTensor<EF, F>,
    /// The carry and last elements, present exactly when the claim's setup sends them.
    pub successor: Option<SuccessorTensors<F, EF>>,
}

/// The messages a batch of claims puts on the wire.
///
/// ```text
///     per claim   its elements
///     once        the rounds, and the surviving value
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: BitCoordinates, EF: BitCoordinates",
    deserialize = "F: BitCoordinates, EF: BitCoordinates"
))]
pub struct BitRingSwitchClaimsProof<F, EF = F> {
    /// Every claim's elements, in the order the claims are bound.
    pub claims: Vec<ClaimElements<F, EF>>,
    /// The one degree-two sumcheck every claim shares.
    pub sumcheck: SumcheckData<F, EF>,
    /// The value of the one surviving claim.
    pub final_eval: EF,
}

impl<F, EF> BitRingSwitchClaimsProof<F, EF> {
    /// Repackage a one-claim proof as a batch of one.
    fn of_single(proof: BitRingSwitchProof<F, EF>) -> Self {
        Self {
            claims: alloc::vec![ClaimElements {
                tensor: proof.tensor,
                successor: proof.successor,
            }],
            sumcheck: proof.sumcheck,
            final_eval: proof.final_eval,
        }
    }
}

impl<F: TowerLevel, EF: BitCoordinates + ExtensionField<F>> BitRingSwitchClaims<F, EF> {
    /// Gather claims to be reduced together.
    ///
    /// # Errors
    ///
    /// - No claim is given.
    /// - Two claims name different numbers of variables.
    pub fn new(reductions: Vec<BitRingSwitch<F, EF>>) -> Result<Self, BitRingSwitchError> {
        let first = reductions.first().ok_or(BitRingSwitchError::NoClaims)?;
        let expected = first.num_variables();
        if let Some(other) = reductions
            .iter()
            .find(|reduction| reduction.num_variables() != expected)
        {
            return Err(BitRingSwitchError::WidthMismatch {
                expected,
                actual: other.num_variables(),
            });
        }
        Ok(Self { reductions })
    }

    /// The claims, in the order they are bound.
    #[must_use]
    pub fn reductions(&self) -> &[BitRingSwitch<F, EF>] {
        &self.reductions
    }

    /// Variables the packed polynomial every claim reads must have.
    #[must_use]
    pub fn num_variables(&self) -> usize {
        self.reductions[0].num_variables()
    }

    /// The transcript description a batch of at least two claims plays.
    fn shape(&self) -> BitRingSwitchClaimsShape {
        BitRingSwitchClaimsShape {
            claims: self.reductions.iter().map(BitRingSwitch::shape).collect(),
        }
    }

    /// The leading Boolean coordinates every claim shares, and the slot address they spell.
    ///
    /// # Algorithm
    ///
    /// Each claim fixes its own Boolean prefix, which selects one slot of the packing.
    /// The shared sumcheck can skip only the coordinates every claim fixes the same way:
    ///
    /// ```text
    ///     claim 0   1 0 1 z ..
    ///     claim 1   1 0 0 z ..
    ///     shared    1 0          ->  two rounds skipped, slot address 0b10
    /// ```
    ///
    /// Inside that slot each claim's weight lives on its own sub-slot.
    fn common_prefix(&self) -> (usize, usize) {
        // No claim may lose a coordinate its own prefix does not fix.
        let limit = self
            .reductions
            .iter()
            .map(|reduction| reduction.fixed_prefix().0)
            .min()
            .expect("a batch holds at least one claim");
        let lead = &self.reductions[0].high()[..limit];
        let prefix = (0..limit)
            .find(|&j| {
                self.reductions[1..]
                    .iter()
                    .any(|other| other.high()[j] != lead[j])
            })
            .unwrap_or(limit);
        // Read the address from most significant bit to least significant bit.
        let address = lead[..prefix].iter().fold(0usize, |address, &bit| {
            (address << 1) | usize::from(bit == EF::ONE)
        });
        (prefix, address)
    }
}

/// `lambda^0, lambda^1, ..`, one weight per claim.
fn claim_weights<EF: Field>(lambda: EF, num_claims: usize) -> Vec<EF> {
    lambda.powers().take(num_claims).collect()
}

/// `eq(bits, r)` for Boolean `bits`: one factor `r_j` or `1 - r_j` per coordinate.
fn boolean_equality<EF: Field>(bits: &[EF], r: &[EF]) -> EF {
    bits.iter()
        .zip(r)
        .map(|(&bit, &r)| if bit == EF::ONE { r } else { EF::ONE - r })
        .product()
}

impl<F, EF> BitRingSwitchClaims<F, EF>
where
    F: TranscriptField + TowerLevel,
    EF: BitCoordinates + ExtensionField<F>,
{
    /// Each claim's second stage, over the challenges the batch drew.
    fn batches(&self, draws: &ClaimsDraws<EF>) -> Vec<BitRingSwitchBatch<'_, F, EF>> {
        self.reductions
            .iter()
            .map(|reduction| {
                // Only a claim sending successor elements weighs them under the shared alpha.
                let alpha = draws.alpha.filter(|_| reduction.sends_successor_tensors());
                reduction
                    .batch_drawn(&draws.batching_point, alpha)
                    .expect("the transcript draws what every claim's kind batches with")
            })
            .collect()
    }

    /// Reduce every claim to one claim about the packing, at one point.
    ///
    /// # The field the rounds run in
    ///
    /// As for [`BitRingSwitch::prove`]: `R` holds the packing, the weights and the rounds.
    ///
    /// # Returns
    ///
    /// The proof, the point the rounds ended at, and the surviving claim's value.
    /// A Boolean slot prefix every claim shares is restored in front of the random coordinates.
    ///
    /// # Panics
    ///
    /// Panics unless the packing has the variables the evaluation points leave.
    pub fn prove<R, Challenger, S>(
        &self,
        packing: &BitPacking<F, S>,
        challenger: &mut Challenger,
    ) -> (BitRingSwitchClaimsProof<F, EF>, Point<EF>, EF)
    where
        F: Send + Sync,
        EF: Send + Sync,
        R: IntoTranscriptField<EF> + FromTable<F> + Sync,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
        S: Borrow<[F]>,
    {
        // A batch of one is the one-claim reduction, its compact rounds included.
        if let [reduction] = self.reductions.as_slice() {
            let (proof, point, value) = reduction.prove::<R, _, _>(packing, challenger);
            return (BitRingSwitchClaimsProof::of_single(proof), point, value);
        }

        let num_variables = self.num_variables();
        assert_eq!(
            packing.num_variables(),
            num_variables,
            "the packing must have the {num_variables} variables the evaluation points leave"
        );

        // Every claim's elements are functions of its own point alone.
        // They are all ready before the transcript binds the first of them.
        let supports = self
            .reductions
            .iter()
            .map(BitRingSwitch::support)
            .collect::<Vec<_>>();
        let claims = self
            .reductions
            .iter()
            .zip(&supports)
            .map(|(reduction, (_, offset, equality))| ClaimElements {
                tensor: BitRingSwitch::tensor_over(packing, *offset, equality),
                successor: reduction.successor_tensors_over(packing, *offset, equality),
            })
            .collect::<Vec<_>>();

        let mut transcript =
            BitRingSwitchClaimsProverTranscript::<Challenger, F, EF>::new(challenger, self.shape());
        let statements = self.statements(&claims);
        let draws = transcript.statement(&statements);
        let batches = self.batches(&draws);
        let weights = claim_weights(draws.lambda, self.reductions.len());

        // The shared rounds run over the slot every claim's prefix agrees on.
        let (prefix, address) = self.common_prefix();
        let rounds = num_variables - prefix;
        let slot = 1usize << rounds;
        let slot_offset = address * slot;

        // sum_i lambda^i * A_i, each A_i written into the sub-slot its own prefix selects.
        let mut table = Poly::<R>::zero(rounds);
        for ((batch, (_, offset, equality)), &weight) in batches.iter().zip(&supports).zip(&weights)
        {
            let claim_weights = batch.weights_over::<R>(equality);
            let start = offset - slot_offset;
            let weight = <R as From<EF>>::from(weight);
            table.as_mut_slice()[start..start + claim_weights.num_evals()]
                .par_iter_mut()
                .zip(claim_weights.as_slice().par_iter())
                .for_each(|(entry, &value)| *entry += weight * value);
        }
        let initial_sum = Self::batched_sum(&batches, &claims, &weights);

        let mut prover = ReprSumcheckProver::<F, EF, R>::from_repr_tables(
            VariableOrder::Prefix,
            slot_packing(packing, slot_offset, slot),
            table,
            initial_sum,
        );
        let mut sumcheck = SumcheckData::default();
        let r_prime = transcript.batched_sumcheck(|challenger| {
            prover.compute_sumcheck_polynomials(&mut sumcheck, challenger, rounds, 0)
        });

        // The rounds folded the packing down to its value at the point they ended on.
        let final_eval = prover.evals().as_slice()[0].into_transcript();
        transcript.surviving_claim(final_eval);
        transcript.finish();

        (
            BitRingSwitchClaimsProof {
                claims,
                sumcheck,
                final_eval,
            },
            self.restore_prefix(prefix, &r_prime),
            final_eval,
        )
    }

    /// Replay the batch and return the one claim it leaves behind, as a point and a value.
    ///
    /// Discharging that pair against a commitment to the packing is the caller's business.
    ///
    /// # Arguments
    ///
    /// - The proof.
    /// - Per claim, its current and successor readings, each when it is to be checked.
    /// - The challenger, advanced only once every structural check passes.
    ///
    /// # Errors
    ///
    /// Before the challenger is touched:
    ///
    /// - Elements or readings for a number of claims other than the setup's.
    /// - Any claim's structural rejection, as in [`BitRingSwitch::verify_readings`].
    ///
    /// After it:
    ///
    /// - A reading disagreeing with its claim's elements.
    /// - A failed round, or a surviving value that does not close the batched sum.
    pub fn verify_readings<Challenger>(
        &self,
        proof: &BitRingSwitchClaimsProof<F, EF>,
        readings: &[(Option<EF>, Option<EF>)],
        challenger: &mut Challenger,
    ) -> Result<(Point<EF>, EF), BitRingSwitchProofError>
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let expected = self.reductions.len();
        if let Some(actual) = [proof.claims.len(), readings.len()]
            .into_iter()
            .find(|&count| count != expected)
        {
            return Err(BitRingSwitchProofError::ClaimCount { expected, actual });
        }

        // A batch of one replays the one-claim reduction it was proved with.
        if let ([reduction], [claim], [(current, next)]) = (
            self.reductions.as_slice(),
            proof.claims.as_slice(),
            readings,
        ) {
            let single = BitRingSwitchProof {
                tensor: claim.tensor.clone(),
                successor: claim.successor.clone(),
                sumcheck: proof.sumcheck.clone(),
                final_eval: proof.final_eval,
            };
            return reduction.verify_readings(&single, *current, *next, challenger);
        }

        // Every structural rejection runs before the challenger is touched.
        for ((reduction, claim), &(current, next)) in
            self.reductions.iter().zip(&proof.claims).zip(readings)
        {
            reduction.check_elements(&claim.tensor, claim.successor.as_ref(), current, next)?;
        }
        check_no_grinding(&proof.sumcheck)?;

        // The verifier derives the shared slot from the public points alone.
        let (prefix, _) = self.common_prefix();
        let rounds = self.num_variables() - prefix;
        let mut transcript = BitRingSwitchClaimsVerifierTranscript::<Challenger, F, EF>::new(
            challenger,
            self.shape(),
        );
        let draws = transcript.statement(&self.statements(&proof.claims))?;

        // Each reading is checked against its own claim's columns, before any round.
        for ((reduction, claim), &(current, next)) in
            self.reductions.iter().zip(&proof.claims).zip(readings)
        {
            if let Err(error) = reduction.check_element_readings(
                &claim.tensor,
                claim.successor.as_ref(),
                current,
                next,
            ) {
                transcript.abort();
                return Err(error);
            }
        }

        let batches = self.batches(&draws);
        let weights = claim_weights(draws.lambda, self.reductions.len());

        // The starting sum is derived from the elements' rows, never taken from the prover.
        let mut sum = Self::batched_sum(&batches, &proof.claims, &weights);
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

        // The rounds close on sum_i lambda^i * A_i(r'), times the surviving value.
        let closing = self.closing_weight(&batches, &weights, prefix, &r_prime)?;
        if sum != closing * proof.final_eval {
            return Err(BitRingSwitchProofError::FinalCheck);
        }

        Ok((self.restore_prefix(prefix, &r_prime), proof.final_eval))
    }

    /// What every claim binds, borrowed from the claims and their elements.
    fn statements<'a>(
        &'a self,
        claims: &'a [ClaimElements<F, EF>],
    ) -> Vec<ClaimStatement<'a, F, EF>> {
        self.reductions
            .iter()
            .zip(claims)
            .map(|(reduction, claim)| ClaimStatement {
                point: reduction.point(),
                rows: claim.tensor.rows(),
                successor: claim
                    .successor
                    .as_ref()
                    .map(|elements| (elements.carry.rows(), elements.last.rows())),
            })
            .collect()
    }

    /// `sum_i lambda^i * sigma_i`, the sum the shared rounds start from.
    fn batched_sum(
        batches: &[BitRingSwitchBatch<'_, F, EF>],
        claims: &[ClaimElements<F, EF>],
        weights: &[EF],
    ) -> EF {
        batches
            .iter()
            .zip(claims)
            .zip(weights)
            .map(|((batch, claim), &weight)| {
                weight * batch.initial_sum(&claim.tensor, claim.successor.as_ref())
            })
            .sum()
    }

    /// `sum_i lambda^i * A_i(r')`, the combined weight where the shared rounds ended.
    ///
    /// # Algorithm
    ///
    /// Claim `i` fixes `p_i >= p` leading bits, of which the batch skips the shared `p`.
    /// Its weight vanishes off its sub-slot and factors on the rest:
    ///
    /// ```text
    ///     A_i(r') = eq(bits_i[p..p_i], r'[..p_i - p]) * A_i|slot(r'[p_i - p..])
    /// ```
    fn closing_weight(
        &self,
        batches: &[BitRingSwitchBatch<'_, F, EF>],
        weights: &[EF],
        prefix: usize,
        r_prime: &Point<EF>,
    ) -> Result<EF, BitRingSwitchError> {
        let mut closing = EF::ZERO;
        for ((reduction, batch), &weight) in self.reductions.iter().zip(batches).zip(weights) {
            let own = reduction.fixed_prefix().0 - prefix;
            let (selected, rest) = r_prime.as_slice().split_at(own);
            let gate = boolean_equality(&reduction.high()[prefix..prefix + own], selected);
            let inside = batch.closing_weight(&Point::new(rest.to_vec()))?;
            closing += weight * gate * inside;
        }
        Ok(closing)
    }

    /// Restore the shared Boolean slot address in front of a point inside that slot.
    fn restore_prefix(&self, prefix: usize, point: &Point<EF>) -> Point<EF> {
        let mut coordinates = self.reductions[0].high()[..prefix].to_vec();
        coordinates.extend_from_slice(point.as_slice());
        Point::new(coordinates)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use alloc::{format, vec};

    use p3_binary_field::{BinaryChallenger, BinaryField16};
    use p3_challenger::{CanSample, HashChallenger};
    use p3_field::PrimeCharacteristicRing;
    use p3_keccak::Keccak256Hash;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type EF = BinaryField16;
    type Chal = BinaryChallenger<EF, HashChallenger<u8, Keccak256Hash, 32>>;

    /// Coordinates a 16-bit element absorbs.
    const ABSORBED: usize = BitRingSwitch::<EF>::ABSORBED;

    fn challenger() -> Chal {
        Chal::from_hasher(Vec::new(), Keccak256Hash)
    }

    fn bits(seed: u64, bytes: usize) -> Vec<u8> {
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..bytes).map(|_| rng.random::<u8>()).collect()
    }

    /// The bit witness, one field element per bit.
    fn embedded(witness: &[u8]) -> Poly<EF> {
        Poly::new(
            (0..witness.len() * 8)
                .map(|cell| EF::from_bool((witness[cell / 8] >> (cell % 8)) & 1 == 1))
                .collect(),
        )
    }

    /// The witness one row further on, the last row repeating, straight from the definition.
    fn dense_successor(witness: &[u8], point: &Point<EF>, row_variables: usize) -> EF {
        let cells = embedded(witness);
        let (selector, rho) = point.split_at(point.num_variables() - row_variables);
        let eq_selector = Poly::<EF>::new_from_point(selector.as_slice(), EF::ONE);
        let eq_rho = Poly::<EF>::new_from_point(rho.as_slice(), EF::ONE);
        let rows = 1usize << row_variables;
        let mut claim = EF::ZERO;
        for (c, &gate) in eq_selector.as_slice().iter().enumerate() {
            for (z, &weight) in eq_rho.as_slice().iter().enumerate() {
                claim += gate * weight * cells.as_slice()[c * rows + (z + 1).min(rows - 1)];
            }
        }
        claim
    }

    /// One claim's setup: its point, and the successor rows when it reads the next row too.
    #[derive(Clone, Debug)]
    struct Claim {
        point: Point<EF>,
        successor: Option<usize>,
    }

    impl Claim {
        fn reduction(&self) -> BitRingSwitch<EF> {
            self.successor
                .map_or_else(
                    || BitRingSwitch::<EF>::new(&self.point),
                    |rows| BitRingSwitch::<EF>::with_successor(&self.point, rows),
                )
                .unwrap()
        }

        /// The honest readings, straight from the witness.
        fn readings(&self, witness: &[u8]) -> (Option<EF>, Option<EF>) {
            let current = embedded(witness).eval_base(&self.point);
            let next = self
                .successor
                .map(|rows| dense_successor(witness, &self.point, rows));
            (Some(current), next)
        }
    }

    fn batch(claims: &[Claim]) -> BitRingSwitchClaims<EF> {
        BitRingSwitchClaims::new(claims.iter().map(Claim::reduction).collect()).unwrap()
    }

    fn honest_readings(claims: &[Claim], witness: &[u8]) -> Vec<(Option<EF>, Option<EF>)> {
        claims.iter().map(|claim| claim.readings(witness)).collect()
    }

    /// A point whose leading coordinates spell `prefix`, the rest drawn at random.
    fn point_with_prefix(rng: &mut SmallRng, prefix: &[bool], num_variables: usize) -> Point<EF> {
        Point::new(
            (0..num_variables)
                .map(|i| match prefix.get(i) {
                    Some(&bit) => EF::from_bool(bit),
                    None => rng.random(),
                })
                .collect(),
        )
    }

    /// Prove and verify one batch, then check the surviving claim against the packing.
    fn round_trip(witness: &[u8], claims: &[Claim]) -> BitRingSwitchClaimsProof<EF> {
        let packing = BitPacking::<EF>::new(witness).unwrap();
        let setup = batch(claims);
        let mut prover = challenger();
        let (proof, point_p, value_p) = setup.prove::<EF, _, _>(&packing, &mut prover);
        let mut verifier = challenger();
        let (point, value) = setup
            .verify_readings(&proof, &honest_readings(claims, witness), &mut verifier)
            .unwrap();

        // Both sides agree on the one surviving claim, and it is true of the packing.
        assert_eq!((&point, value), (&point_p, value_p));
        assert_eq!(value, packing.poly().eval_base(&point));
        // The two sponges end in the same state.
        assert_eq!(
            CanSample::<EF>::sample(&mut prover),
            CanSample::<EF>::sample(&mut verifier)
        );
        proof
    }

    #[test]
    fn an_empty_or_ragged_batch_is_refused() {
        // No claim at all.
        assert_eq!(
            BitRingSwitchClaims::<EF>::new(Vec::new()).unwrap_err(),
            BitRingSwitchError::NoClaims
        );

        // Two claims over packings of different widths.
        let mut rng = SmallRng::seed_from_u64(1);
        let narrow = BitRingSwitch::<EF>::new(&Point::<EF>::rand(&mut rng, 8)).unwrap();
        let wide = BitRingSwitch::<EF>::new(&Point::<EF>::rand(&mut rng, 9)).unwrap();
        assert_eq!(
            BitRingSwitchClaims::new(vec![narrow, wide]).unwrap_err(),
            BitRingSwitchError::WidthMismatch {
                expected: 4,
                actual: 5
            }
        );
    }

    #[test]
    fn a_batch_of_one_is_the_one_claim_reduction() {
        // One claim, proved alone and as a batch of one.
        let witness = bits(0xB1, 64);
        let packing = BitPacking::<EF>::new(&witness).unwrap();
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB2), 9);
        let reduction = BitRingSwitch::<EF>::new(&point).unwrap();
        let (single, single_point, _) = reduction.prove::<EF, _, _>(&packing, &mut challenger());
        let setup = BitRingSwitchClaims::new(vec![reduction]).unwrap();
        let (batched, batched_point, _) = setup.prove::<EF, _, _>(&packing, &mut challenger());

        // The same messages, so the same transcript and the same surviving point.
        assert_eq!(batched.claims[0].tensor, single.tensor);
        assert_eq!(
            format!("{:?}", batched.sumcheck),
            format!("{:?}", single.sumcheck)
        );
        assert_eq!(batched_point, single_point);
    }

    #[test]
    fn several_random_points_leave_one_true_claim() {
        // Four claims at unrelated points: no shared prefix, every round runs.
        let witness = bits(0xC1, 64);
        let mut rng = SmallRng::seed_from_u64(0xC2);
        let claims = (0..4)
            .map(|_| Claim {
                point: Point::rand(&mut rng, 9),
                successor: None,
            })
            .collect::<Vec<_>>();
        let proof = round_trip(&witness, &claims);

        // One element per claim, and one sumcheck of l' = 9 - 4 rounds for all of them.
        assert_eq!(proof.claims.len(), 4);
        assert_eq!(proof.sumcheck.num_rounds(), 9 - ABSORBED);
    }

    #[test]
    fn claims_mixing_successor_views_share_one_alpha() {
        // Invariant: alpha is drawn once, and weighs only the claims that send successor elements.
        //
        //     claim 0   current reading only                 K = 1
        //     claim 1   successor rows outrun one element    K = 3
        //     claim 2   successor rows inside one element    K = 1
        let witness = bits(0xD1, 64);
        let mut rng = SmallRng::seed_from_u64(0xD2);
        let claims = [None, Some(7), Some(2)].map(|successor| Claim {
            point: Point::rand(&mut rng, 9),
            successor,
        });
        let proof = round_trip(&witness, &claims);
        assert!(proof.claims[1].successor.is_some());
        assert!(proof.claims[0].successor.is_none() && proof.claims[2].successor.is_none());
    }

    #[test]
    fn a_shared_boolean_prefix_skips_its_rounds() {
        // Both claims read column 0b10 of a stacked trace, at different rows.
        let witness = bits(0xE1, 128);
        let mut rng = SmallRng::seed_from_u64(0xE2);
        let claims = [0, 1].map(|_| Claim {
            point: point_with_prefix(&mut rng, &[true, false], 10),
            successor: None,
        });
        let proof = round_trip(&witness, &claims);

        // The two shared bits select the slot, so two of the six rounds never run.
        assert_eq!(proof.sumcheck.num_rounds(), 10 - ABSORBED - 2);
    }

    #[test]
    fn prefixes_that_part_keep_each_claim_in_its_own_sub_slot() {
        // Invariant: only the leading bits every claim agrees on are skipped.
        //
        //     claim 0   1 0 1 ..    its weight on sub-slot 0b01 of slot 0b1
        //     claim 1   1 1 ..      its weight on sub-slot 0b1  of slot 0b1
        //     claim 2   1 z ..      spread over all of slot 0b1
        //     claim 3   1 0 1 ..    the same sub-slot as claim 0, another row
        let witness = bits(0xF1, 128);
        let mut rng = SmallRng::seed_from_u64(0xF2);
        let claims = [
            vec![true, false, true],
            vec![true, true],
            vec![true],
            vec![true, false, true],
        ]
        .map(|prefix| Claim {
            point: point_with_prefix(&mut rng, &prefix, 10),
            successor: None,
        });
        let proof = round_trip(&witness, &claims);
        assert_eq!(proof.sumcheck.num_rounds(), 10 - ABSORBED - 1);

        // Claims in disjoint slots share nothing, and every round runs.
        let apart = [vec![false], vec![true]].map(|prefix| Claim {
            point: point_with_prefix(&mut rng, &prefix, 10),
            successor: Some(3),
        });
        let proof = round_trip(&witness, &apart);
        assert_eq!(proof.sumcheck.num_rounds(), 10 - ABSORBED);
    }

    /// Three honest claims, their batch, and the proof it produced.
    fn honest_batch() -> (Vec<u8>, Vec<Claim>, BitRingSwitchClaimsProof<EF>) {
        let witness = bits(0xA1, 64);
        let mut rng = SmallRng::seed_from_u64(0xA2);
        let claims = [None, Some(6), None]
            .map(|successor| Claim {
                point: Point::rand(&mut rng, 9),
                successor,
            })
            .to_vec();
        let packing = BitPacking::<EF>::new(&witness).unwrap();
        let (proof, _, _) = batch(&claims).prove::<EF, _, _>(&packing, &mut challenger());
        (witness, claims, proof)
    }

    #[test]
    fn a_false_reading_at_one_point_is_rejected() {
        let (witness, claims, proof) = honest_batch();
        let setup = batch(&claims);

        // Move the current reading of the middle claim alone.
        let mut readings = honest_readings(&claims, &witness);
        readings[1].0 = readings[1].0.map(|value| value + EF::ONE);
        assert_eq!(
            setup
                .verify_readings(&proof, &readings, &mut challenger())
                .unwrap_err(),
            BitRingSwitchProofError::ClaimMismatch
        );

        // Move its successor reading instead.
        let mut readings = honest_readings(&claims, &witness);
        readings[1].1 = readings[1].1.map(|value| value + EF::ONE);
        assert_eq!(
            setup
                .verify_readings(&proof, &readings, &mut challenger())
                .unwrap_err(),
            BitRingSwitchProofError::SuccessorClaimMismatch
        );
    }

    /// A prover that tampers with the elements before binding them, then plays the rounds honestly.
    ///
    /// ```text
    ///     honest elements  ->  tamper  ->  bind  ->  r'', alpha, lambda
    ///     rounds           ->  measured on the honest tables, under those draws
    /// ```
    ///
    /// Each round message is read off the tables alone.
    /// The verifier derives the starting sum from the forged rows.
    /// So the rounds prove the forged sum exactly when it equals the honest one.
    fn forge(
        setup: &BitRingSwitchClaims<EF>,
        packing: &BitPacking<EF>,
        tamper: impl FnOnce(&mut [ClaimElements<EF>]),
    ) -> BitRingSwitchClaimsProof<EF> {
        let supports = setup
            .reductions()
            .iter()
            .map(BitRingSwitch::support)
            .collect::<Vec<_>>();
        let honest = setup
            .reductions()
            .iter()
            .zip(&supports)
            .map(|(reduction, (_, offset, equality))| ClaimElements {
                tensor: BitRingSwitch::tensor_over(packing, *offset, equality),
                successor: reduction.successor_tensors_over(packing, *offset, equality),
            })
            .collect::<Vec<_>>();
        let mut claims = honest.clone();
        tamper(&mut claims);

        // The forged elements are what the transcript binds.
        let mut challenger = challenger();
        let mut transcript =
            BitRingSwitchClaimsProverTranscript::<Chal, EF>::new(&mut challenger, setup.shape());
        let draws = transcript.statement(&setup.statements(&claims));
        let batches = setup.batches(&draws);
        let weights = claim_weights(draws.lambda, claims.len());

        // The combined weight table, exactly as the honest prover builds it.
        let (prefix, address) = setup.common_prefix();
        let rounds = setup.num_variables() - prefix;
        let slot = 1usize << rounds;
        let mut table = Poly::<EF>::zero(rounds);
        for ((batch, (_, offset, equality)), &weight) in batches.iter().zip(&supports).zip(&weights)
        {
            let start = offset - address * slot;
            let claim_weights = batch.weights_over::<EF>(equality);
            for (entry, &value) in table.as_mut_slice()[start..]
                .iter_mut()
                .zip(claim_weights.as_slice())
            {
                *entry += weight * value;
            }
        }

        // The prover is seeded with the sum the tables really have, which only its debug checks read.
        let mut prover = ReprSumcheckProver::<EF, EF, EF>::from_repr_tables(
            VariableOrder::Prefix,
            slot_packing(packing, address * slot, slot),
            table,
            BitRingSwitchClaims::batched_sum(&batches, &honest, &weights),
        );
        let mut sumcheck = SumcheckData::default();
        let _ = transcript.batched_sumcheck(|challenger| {
            prover.compute_sumcheck_polynomials(&mut sumcheck, challenger, rounds, 0)
        });
        let final_eval = prover.evals().as_slice()[0];
        transcript.surviving_claim(final_eval);
        transcript.finish();

        BitRingSwitchClaimsProof {
            claims,
            sumcheck,
            final_eval,
        }
    }

    /// The readings a proof's elements imply, so no column check can object to them.
    fn implied_readings(
        setup: &BitRingSwitchClaims<EF>,
        proof: &BitRingSwitchClaimsProof<EF>,
        claims: &[Claim],
    ) -> Vec<(Option<EF>, Option<EF>)> {
        setup
            .reductions()
            .iter()
            .zip(&proof.claims)
            .zip(claims)
            .map(|((reduction, elements), claim)| {
                let current = reduction.incoming_claim(&elements.tensor);
                let next = claim.successor.map(|_| {
                    reduction
                        .successor_claim(&elements.tensor, elements.successor.as_ref())
                        .unwrap()
                });
                (Some(current), next)
            })
            .collect()
    }

    /// Add one to row 5 of the tensor element of every listed claim.
    fn shift_row(claims: &mut [ClaimElements<EF>], indices: &[usize]) {
        for &index in indices {
            let mut rows = claims[index].tensor.rows().to_vec();
            rows[5] += EF::ONE;
            claims[index].tensor = BitTensor::try_from(rows).unwrap();
        }
    }

    #[test]
    fn a_forged_element_at_one_point_moves_the_starting_sum() {
        // Invariant: the rows of every claim's element reach the batched starting sum.
        //
        //     columns  ->  the forged readings, which the column check accepts
        //     rows     ->  sum_i lambda^i * sigma_i, off by lambda^index * D(r'')
        let (witness, claims, _) = honest_batch();
        let packing = BitPacking::<EF>::new(&witness).unwrap();
        let setup = batch(&claims);
        for index in 0..claims.len() {
            // Forge one claim's element, current-only or successor alike.
            let forged = forge(&setup, &packing, |elements| shift_row(elements, &[index]));
            let readings = implied_readings(&setup, &forged, &claims);

            // The rounds were played for the forged transcript, yet the sum they close on is wrong.
            assert_eq!(
                setup
                    .verify_readings(&forged, &readings, &mut challenger())
                    .unwrap_err(),
                BitRingSwitchProofError::FinalCheck,
                "claim {index}"
            );
        }
    }

    #[test]
    fn one_row_error_in_two_claims_does_not_cancel() {
        // Invariant: the powers of lambda keep two equal row errors apart.
        //
        //     weights 1, 1          D(r'') + D(r'') = 0 in characteristic 2  ->  accepted
        //     weights 1, lambda     (1 + lambda) * D(r'') != 0               ->  refused
        let witness = bits(0xA1, 64);
        let packing = BitPacking::<EF>::new(&witness).unwrap();
        let mut rng = SmallRng::seed_from_u64(0xA3);
        let point = Point::<EF>::rand(&mut rng, 9);
        let distinct = [point.clone(), Point::rand(&mut rng, 9)];
        let repeated = [point.clone(), point];

        // Two plain claims at distinct points, then at one point used twice.
        for points in [distinct, repeated] {
            let claims = points.map(|point| Claim {
                point,
                successor: None,
            });
            let setup = batch(&claims);

            // The same row error in both elements.
            let forged = forge(&setup, &packing, |elements| shift_row(elements, &[0, 1]));
            let readings = implied_readings(&setup, &forged, &claims);

            // Both readings moved, and each still agrees with its own element's columns.
            assert_ne!(readings, honest_readings(&claims, &witness));
            assert_eq!(
                setup
                    .verify_readings(&forged, &readings, &mut challenger())
                    .unwrap_err(),
                BitRingSwitchProofError::FinalCheck
            );
        }
    }

    #[test]
    fn a_tampered_surviving_value_does_not_close_the_rounds() {
        let (witness, claims, mut proof) = honest_batch();
        proof.final_eval += EF::ONE;
        assert_eq!(
            batch(&claims)
                .verify_readings(
                    &proof,
                    &honest_readings(&claims, &witness),
                    &mut challenger()
                )
                .unwrap_err(),
            BitRingSwitchProofError::FinalCheck
        );
    }

    #[test]
    fn a_proof_does_not_verify_with_its_claims_reordered() {
        // Swap the first and last claims, and their elements with them.
        let (witness, mut claims, mut proof) = honest_batch();
        claims.swap(0, 2);
        proof.claims.swap(0, 2);
        let setup = batch(&claims);
        let readings = honest_readings(&claims, &witness);

        // Every element still sits beside its own claim, so every column check passes.
        for ((reduction, elements), &(current, next)) in
            setup.reductions().iter().zip(&proof.claims).zip(&readings)
        {
            reduction
                .check_element_readings(
                    &elements.tensor,
                    elements.successor.as_ref(),
                    current,
                    next,
                )
                .unwrap();
        }

        // The swapped statement draws other challenges, which the recorded rounds do not answer.
        let error = setup
            .verify_readings(&proof, &readings, &mut challenger())
            .unwrap_err();
        assert!(
            matches!(
                error,
                BitRingSwitchProofError::Sumcheck(_) | BitRingSwitchProofError::FinalCheck
            ),
            "{error:?}"
        );
    }

    #[test]
    fn a_claim_count_other_than_the_setup_is_refused() {
        let (witness, claims, proof) = honest_batch();
        let setup = batch(&claims);
        let readings = honest_readings(&claims, &witness);

        // One element short.
        let mut short = proof.clone();
        short.claims.pop();
        assert_eq!(
            setup
                .verify_readings(&short, &readings, &mut challenger())
                .unwrap_err(),
            BitRingSwitchProofError::ClaimCount {
                expected: 3,
                actual: 2
            }
        );

        // One reading short.
        assert_eq!(
            setup
                .verify_readings(&proof, &readings[..2], &mut challenger())
                .unwrap_err(),
            BitRingSwitchProofError::ClaimCount {
                expected: 3,
                actual: 2
            }
        );
    }

    #[test]
    fn a_stray_grinding_witness_is_refused() {
        let (witness, claims, mut proof) = honest_batch();
        proof.sumcheck.pow_witnesses.push(EF::ZERO);
        assert_eq!(
            batch(&claims)
                .verify_readings(
                    &proof,
                    &honest_readings(&claims, &witness),
                    &mut challenger()
                )
                .unwrap_err(),
            BitRingSwitchProofError::NonEmptyPowWitnesses { actual: 1 }
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 24, ..ProptestConfig::default() })]

        /// The batch against one reduction per point, over random witnesses, points and views.
        #[test]
        fn the_batch_agrees_with_one_reduction_per_point(
            log_bytes in 1usize..=5,
            num_claims in 1usize..=4,
            seed: u64,
            successor_mask in 0u8..16,
            prefix_bits in 0usize..=2,
        ) {
            let witness = bits(seed, 1 << log_bytes);
            let packing = BitPacking::<EF>::new(&witness).unwrap();
            let num_variables = log_bytes + 3;
            let mut rng = SmallRng::seed_from_u64(seed ^ 0x5EED);
            // A shared Boolean prefix, never longer than the packing's own coordinates.
            let prefix = (0..prefix_bits.min(num_variables - ABSORBED))
                .map(|_| rng.random::<bool>())
                .collect::<Vec<_>>();
            let claims = (0..num_claims)
                .map(|i| Claim {
                    point: point_with_prefix(&mut rng, &prefix, num_variables),
                    successor: ((successor_mask >> i) & 1 == 1)
                        .then(|| rng.random_range(0..=num_variables)),
                })
                .collect::<Vec<_>>();

            // Reference: one reduction per point, each on its own sponge.
            let reference = claims
                .iter()
                .map(|claim| {
                    let (proof, _, _) =
                        claim.reduction().prove::<EF, _, _>(&packing, &mut challenger());
                    ClaimElements { tensor: proof.tensor, successor: proof.successor }
                })
                .collect::<Vec<_>>();

            let proof = round_trip(&witness, &claims);

            // Every claim sends the elements its own reduction would.
            prop_assert_eq!(&proof.claims, &reference);
            // And those elements carry the readings the witness defines.
            let setup = batch(&claims);
            for ((reduction, elements), claim) in
                setup.reductions().iter().zip(&proof.claims).zip(&claims)
            {
                let (current, next) = claim.readings(&witness);
                prop_assert_eq!(Some(reduction.incoming_claim(&elements.tensor)), current);
                if next.is_some() {
                    let successor = reduction
                        .successor_claim(&elements.tensor, elements.successor.as_ref())
                        .unwrap();
                    prop_assert_eq!(Some(successor), next);
                }
            }
        }
    }

    /// Several claims at a cubic challenge field, packed at sixty-four bits.
    mod wide_challenge {
        use p3_binary_field::{Poly64, Poly192};

        use super::*;

        type F = Poly64;
        type EF = Poly192;
        type NarrowChal = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;

        fn challenger() -> NarrowChal {
            NarrowChal::from_hasher(Vec::new(), Keccak256Hash)
        }

        /// The bit witness, one challenge-field element per bit.
        fn embedded(witness: &[u8]) -> Poly<EF> {
            Poly::new(
                (0..witness.len() * 8)
                    .map(|cell| EF::from_bool((witness[cell / 8] >> (cell % 8)) & 1 == 1))
                    .collect(),
            )
        }

        #[test]
        fn a_mixed_batch_round_trips_and_lands_on_the_packing() {
            // Three claims share one sumcheck: a plain one, a short and a long successor view.
            let witness = bits(0xC1A1, 1 << 7);
            let packing = BitPacking::<F>::new(&witness).unwrap();
            let mut rng = SmallRng::seed_from_u64(0xC1A2);
            let points = (0..3)
                .map(|_| Point::<EF>::rand(&mut rng, 10))
                .collect::<Vec<_>>();
            let setup = BitRingSwitchClaims::new(vec![
                BitRingSwitch::<F, EF>::new(&points[0]).unwrap(),
                BitRingSwitch::<F, EF>::with_successor(&points[1], 4).unwrap(),
                BitRingSwitch::<F, EF>::with_successor(&points[2], 9).unwrap(),
            ])
            .unwrap();

            // The successor readings come from the square-case definition, lifted.
            let cells = embedded(&witness);
            let successor = |point: &Point<EF>, rows: usize| {
                let (selector, rho) = point.split_at(point.num_variables() - rows);
                let eq_selector = Poly::<EF>::new_from_point(selector.as_slice(), EF::ONE);
                let eq_rho = Poly::<EF>::new_from_point(rho.as_slice(), EF::ONE);
                let height = 1usize << rows;
                let mut claim = EF::ZERO;
                for (c, &gate) in eq_selector.as_slice().iter().enumerate() {
                    for (z, &weight) in eq_rho.as_slice().iter().enumerate() {
                        let x = (z + 1).min(height - 1);
                        claim += gate * weight * cells.as_slice()[c * height + x];
                    }
                }
                claim
            };
            let readings = vec![
                (Some(cells.eval_base(&points[0])), None),
                (
                    Some(cells.eval_base(&points[1])),
                    Some(successor(&points[1], 4)),
                ),
                (
                    Some(cells.eval_base(&points[2])),
                    Some(successor(&points[2], 9)),
                ),
            ];

            let (proof, point_p, value_p) = setup.prove::<EF, _, _>(&packing, &mut challenger());
            let (point, value) = setup
                .verify_readings(&proof, &readings, &mut challenger())
                .unwrap();
            assert_eq!((&point, value), (&point_p, value_p));
            assert_eq!(value, packing.poly().eval_base(&point));

            // A reading one off is refused, whichever claim carries it.
            for index in 0..readings.len() {
                let mut wrong = readings.clone();
                wrong[index].0 = wrong[index].0.map(|value| value + EF::ONE);
                assert!(
                    setup
                        .verify_readings(&proof, &wrong, &mut challenger())
                        .is_err(),
                    "claim {index}"
                );
            }
        }
    }
}
