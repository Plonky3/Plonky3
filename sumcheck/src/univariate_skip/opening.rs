//! Turning a skip round's Lagrange-weighted claim into one evaluation point.

use alloc::vec::Vec;

use p3_challenger::fs::TranscriptField;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::Field;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;

use super::lde::CHUNK_BITS;
use super::transcript::{
    SkipOpeningShape, SkipOpeningTranscriptError, SkipOpeningVerifierTranscript,
};
use crate::generic_degree::{GenericDegreeError, GenericDegreeProof, RoundProver};

/// Per-variable degree of the summand this reduction proves.
///
/// The Lagrange vector and the witness fold contribute one each.
pub const OPENING_DEGREE: usize = 2;

/// The claim a skip round leaves, and the reduction that makes it openable.
///
/// # Overview
///
/// A skip round binds its variables with one challenge, not one per variable.
/// What comes back is not an evaluation of the committed polynomial.
/// Writing `rho` for the residual point and `lam` for the skip challenge:
///
/// ```text
///     f(rho, lam) = sum_x eq(rho, x) * sum_c L_c(lam) * f~(x, c)
///                 = sum_c L_c(lam) * f~(rho, c)
/// ```
///
/// That is a weighted blend of `2^k` positions of the committed multilinear.
/// A commitment answers one question, the value at a point, so the blend goes.
///
/// # The reduction
///
/// The blend is itself a sum over the skipped cube, so a sumcheck collapses it:
///
/// ```text
///     sum_c L(c) * g(c)          L(c) = L_c(lam),  g(c) = f~(rho, c)
/// ```
///
/// - `k` rounds, degree two, through the ordinary sumcheck driver.
/// - Both factors are already to hand.
/// - One is the round's Lagrange vector, the other the witness folded at `rho`.
/// - The output is `f~(rho, tau)`, the committed multilinear at one point.
///
/// Several committed polynomials share the same Lagrange factor.
/// One challenge therefore batches them into a single run.
///
/// # Verifier cost
///
/// The verifier holds the same vector and reads it at the output point.
/// That is `2^k` terms.
/// Building the vector costs `2^k` terms and one inversion, off the subspace.
///
/// A challenge on a subspace point makes one entry one and the rest zero.
/// The barycentric form cannot express that, so it is handled separately.
/// Neither is a new asymptotic, since the round message already costs `2^k`.
///
/// The vector is all the verifier needs.
/// The round hands it over without the byte table only its rows use.
#[derive(Debug, Clone)]
pub struct SkipOpening<EF> {
    /// The Lagrange vector of the skipped subspace, at the round's challenge.
    lagrange: Poly<EF>,
}

impl<EF: Field> SkipOpening<EF> {
    /// Take the Lagrange vector a skip round produced.
    ///
    /// # Panics
    ///
    /// Panics if the vector does not cover a whole number of byte-sized chunks.
    #[must_use]
    pub fn new(lagrange: Poly<EF>) -> Self {
        assert_eq!(
            lagrange.num_evals() % CHUNK_BITS,
            0,
            "the skipped subspace covers whole byte chunks"
        );
        Self { lagrange }
    }

    /// Number of variables this reduction binds, which the round skipped.
    #[must_use]
    pub fn num_variables(&self) -> usize {
        self.lagrange.num_variables()
    }

    /// Number of points the skipped subspace holds.
    #[must_use]
    pub fn size(&self) -> usize {
        self.lagrange.num_evals()
    }

    /// The Lagrange vector this reduction weighs by.
    pub const fn lagrange(&self) -> &Poly<EF> {
        &self.lagrange
    }

    /// Read the Lagrange vector at the point the reduction ends on.
    ///
    /// The verifier needs this to strip the weight off the final value.
    #[must_use]
    pub fn lagrange_at(&self, tau: &Point<EF>) -> EF {
        self.lagrange.eval_base(tau)
    }

    /// Fold packed rows over the row variables, giving the skipped cube.
    ///
    /// # Arguments
    ///
    /// - `packed`: one polynomial's rows back to back, lowest bit first.
    /// - `eq_rows`: the equality table of the residual point over the rows.
    ///
    /// # Returns
    ///
    /// The values `f~(rho, c)` for every skipped point `c`.
    ///
    /// # Panics
    ///
    /// Panics if the packed rows do not match the row count and subspace size.
    pub fn partial_evaluation(&self, packed: &[u8], eq_rows: &Poly<EF>) -> Poly<EF>
    where
        EF: Send + Sync,
    {
        let row_bytes = self.size() / CHUNK_BITS;
        assert_eq!(
            packed.len(),
            eq_rows.num_evals() * row_bytes,
            "one packed row per equality weight"
        );

        // A row weighs exactly the skipped points whose bit is set.
        //
        //     f~(rho, c) = sum over rows with bit c set of eq(rho, row)
        //
        // The witness is bits, so this counts with no multiplications:
        // one addition per set bit, and the zero bits cost nothing at all.
        let values = packed
            .par_chunks_exact(row_bytes)
            .zip(eq_rows.as_slice().par_iter())
            .par_fold_reduce(
                || EF::zero_vec(self.size()),
                |mut accumulator, (row, &weight)| {
                    for (chunk, &byte) in row.iter().enumerate() {
                        // Peel the set bits one at a time, skipping zeros.
                        let mut bits = byte;
                        while bits != 0 {
                            let bit = bits.trailing_zeros() as usize;
                            accumulator[chunk * CHUNK_BITS + bit] += weight;
                            bits &= bits - 1;
                        }
                    }
                    accumulator
                },
                |mut left, right| {
                    // Addition is associative, so regrouping cannot change it.
                    for (entry, value) in left.iter_mut().zip(right) {
                        *entry += value;
                    }
                    left
                },
            );

        Poly::new(values)
    }

    /// Fold several committed polynomials at once, sharing the equality table.
    ///
    /// # Panics
    ///
    /// Panics if any polynomial's packed rows do not match the row count.
    #[must_use]
    pub fn partial_evaluations<'a>(
        &self,
        packed: impl IntoIterator<Item = &'a [u8]>,
        rho: &Point<EF>,
    ) -> Vec<Poly<EF>>
    where
        EF: Send + Sync,
    {
        // The equality table depends only on rho, so it is built once.
        let eq_rows = Poly::new_from_point(rho.as_slice(), EF::ONE);
        packed
            .into_iter()
            .map(|rows| self.partial_evaluation(rows, &eq_rows))
            .collect()
    }

    /// Batch several folded witnesses into the one the reduction proves.
    ///
    /// # Arguments
    ///
    /// - `folded`: one folded witness per committed polynomial.
    /// - `gamma`: the challenge whose powers separate them.
    ///
    /// # Panics
    ///
    /// Panics if the folded witnesses disagree on size, or if there are none.
    pub fn batch(folded: &[Poly<EF>], gamma: EF) -> Poly<EF> {
        assert!(!folded.is_empty(), "at least one committed polynomial");
        let size = folded[0].num_evals();
        assert!(
            folded.iter().all(|poly| poly.num_evals() == size),
            "every folded witness covers the same skipped cube"
        );

        // Horner keeps this to one multiplication per entry per operand.
        let mut batched = EF::zero_vec(size);
        for poly in folded.iter().rev() {
            for (entry, &value) in batched.iter_mut().zip(poly.as_slice()) {
                *entry = *entry * gamma + value;
            }
        }
        Poly::new(batched)
    }

    /// The claim the reduction starts from, given each blended value.
    ///
    /// # Panics
    ///
    /// Panics if there are no claims.
    #[must_use]
    pub fn batch_claims(claims: &[EF], gamma: EF) -> EF {
        assert!(!claims.is_empty(), "at least one claim");

        // The same Horner order the witnesses were batched in.
        claims
            .iter()
            .rev()
            .fold(EF::ZERO, |accumulator, &claim| accumulator * gamma + claim)
    }

    /// Replay one reduction against its own transcript.
    ///
    /// # Overview
    ///
    /// The whole sequence lives here because its order is what makes it sound:
    ///
    /// ```text
    ///     bind the claims  ->  draw gamma  ->  replay the rounds  ->  finish
    ///     then             ->  starting sum == sum_i gamma^i * blend_i
    /// ```
    ///
    /// The starting sum is read from the proof, not taken as an argument.
    /// A caller passing its own batch would make that check vacuous.
    /// It runs after the driver is finished, so a rejection is never a panic.
    ///
    /// # Arguments
    ///
    /// - The shape both sides derive from configuration, never from a proof.
    /// - The delegated sumcheck's record.
    /// - The per-polynomial blends the skip round left behind.
    /// - The transcript, in the state the prover left it.
    ///
    /// # Returns
    ///
    /// The point the rounds ended on, and the value the batched openings owe.
    ///
    /// # Errors
    ///
    /// - The shape disagrees with this reduction's width.
    /// - The proof carries a claim count the shape forbids.
    /// - A delegated sumcheck round fails.
    /// - The starting sum is not the batch of the blends.
    /// - The Lagrange weight vanished, leaving the openings unconstrained.
    pub fn verify<C>(
        &self,
        shape: SkipOpeningShape,
        proof: &GenericDegreeProof<EF, EF>,
        blends: &[EF],
        challenger: &mut C,
    ) -> Result<SkipOpeningClaim<EF>, SkipOpeningError>
    where
        EF: TranscriptField,
        C: FieldChallenger<EF> + GrindingChallenger<Witness = EF>,
    {
        // The shape is the caller's configuration.
        // It has to describe this very reduction.
        if shape.num_variables != self.num_variables() {
            return Err(SkipOpeningError::WidthMismatch {
                expected: self.num_variables(),
                actual: shape.num_variables,
            });
        }

        let mut transcript = SkipOpeningVerifierTranscript::<C, EF, EF>::new(challenger, shape);

        // The claims are bound before the challenge that separates them.
        let gamma = transcript.batching_challenge(blends)?;

        // The rounds are a sub-protocol.
        // A rejection there releases the driver rather than panicking on drop.
        let (tau, closing_value) = transcript.sumcheck(|challenger| {
            proof.verify(
                challenger,
                shape.num_variables,
                OPENING_DEGREE,
                shape.pow_bits,
            )
        })?;
        transcript.finish();

        // Nothing inside the sumcheck ties its starting sum to the blends.
        if proof.claimed_sum != Self::batch_claims(blends, gamma) {
            return Err(SkipOpeningError::StartingSumMismatch);
        }

        Ok(SkipOpeningClaim {
            value: self.closing_target(&tau, closing_value)?,
            point: tau,
            gamma,
            num_polynomials: shape.num_polynomials,
        })
    }

    /// Strip the Lagrange weight off the value the rounds closed on.
    ///
    /// The rounds close on `L^(tau) * sum_i gamma^i * opening_i`.
    /// Dividing leaves what the openings themselves owe.
    ///
    /// # Errors
    ///
    /// Returns an error when the weight vanished where the rounds reached.
    /// The closing check would then read `0 == 0` and constrain nothing.
    pub fn closing_target(
        &self,
        tau: &Point<EF>,
        closing_value: EF,
    ) -> Result<EF, SkipOpeningError> {
        let weight = self.lagrange_at(tau);
        if weight.is_zero() {
            return Err(SkipOpeningError::DegenerateWeight);
        }
        Ok(closing_value * weight.inverse())
    }

    /// Build the prover state for the reduction's sumcheck.
    #[must_use]
    pub fn prover(&self, batched: Poly<EF>) -> SkipOpeningProver<EF> {
        SkipOpeningProver {
            lagrange: self.lagrange.clone(),
            batched,
        }
    }
}

/// Reasons a proof of the opening reduction is rejected.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SkipOpeningError {
    /// The blends and the openings disagree on how many were batched.
    #[error("the proof carries {blends} blends and {openings} openings")]
    ClaimCountMismatch {
        /// Number of blended values the proof carries.
        blends: usize,
        /// Number of committed evaluations it carries.
        openings: usize,
    },
    /// The delegated sumcheck ran on a sum that is not the blends' batch.
    #[error("the sumcheck's starting sum is not the batch of the claims")]
    StartingSumMismatch,
    /// The closing value is not the weighted batch of the openings.
    #[error("the sumcheck's closing value is not the weighted batch of the openings")]
    ClosingValueMismatch,
    /// The shape describes a reduction of a different width than this one.
    #[error("the reduction binds {expected} variables, the shape says {actual}")]
    WidthMismatch {
        /// Variables this reduction binds.
        expected: usize,
        /// Variables the shape declares.
        actual: usize,
    },
    /// The Lagrange weight vanished where the rounds ended.
    ///
    /// The closing check would read `0 == 0` and constrain no opening.
    #[error("the Lagrange weight vanished at the point the rounds reached")]
    DegenerateWeight,
    /// The transcript replay refused the proof.
    #[error(transparent)]
    Transcript(#[from] SkipOpeningTranscriptError),
    /// A delegated sumcheck round failed.
    #[error("opening sumcheck: {0}")]
    Sumcheck(#[from] GenericDegreeError),
}
/// What a replayed opening reduction leaves for a commitment to answer.
///
/// Nothing before this claim ties the replay to a commitment.
///
/// Dropping it without checking committed openings therefore accepts everything.
#[must_use]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkipOpeningClaim<EF> {
    /// The point the rounds ended on.
    pub point: Point<EF>,
    /// The challenge the openings are batched under.
    pub gamma: EF,
    /// The value the batched openings must equal.
    ///
    /// The Lagrange weight is already divided out.
    /// So this is what a commitment answers for directly.
    pub value: EF,
    /// Number of polynomials the batch covers.
    pub num_polynomials: usize,
}

impl<EF: Field> SkipOpeningClaim<EF> {
    /// Check committed openings against this claim.
    ///
    /// Nothing before this ties the replay to a commitment.
    /// A caller that skips it has verified a reduction over no witness.
    ///
    /// # Arguments
    ///
    /// The openings in the order the batch was taken over.
    ///
    /// # Errors
    ///
    /// - The opening count is not the width the batch covers.
    /// - The openings do not recombine to the claimed value.
    pub fn discharge(&self, openings: &[EF]) -> Result<(), SkipOpeningError> {
        if openings.len() != self.num_polynomials {
            return Err(SkipOpeningError::ClaimCountMismatch {
                blends: self.num_polynomials,
                openings: openings.len(),
            });
        }
        if SkipOpening::batch_claims(openings, self.gamma) != self.value {
            return Err(SkipOpeningError::ClosingValueMismatch);
        }
        Ok(())
    }
}

/// Prover state of the reduction's sumcheck.
///
/// The summand is the Lagrange vector times the batched witness, degree two.
#[derive(Debug, Clone)]
pub struct SkipOpeningProver<EF> {
    /// The Lagrange vector, folded alongside the witness.
    lagrange: Poly<EF>,
    /// The batched witness over the skipped cube.
    batched: Poly<EF>,
}

impl<EF: Field> SkipOpeningProver<EF> {
    /// The value the reduction leaves on the batched witness.
    ///
    /// Reading it after the rounds gives the evaluation a commitment opens.
    ///
    /// # Panics
    ///
    /// Panics if the rounds have not bound every variable.
    #[must_use]
    pub fn surviving_claim(&self) -> EF {
        assert_eq!(self.batched.num_evals(), 1, "every variable is bound");
        self.batched.as_slice()[0]
    }
}

impl<EF: Field> RoundProver<EF> for SkipOpeningProver<EF> {
    fn fold(&mut self, r: EF) {
        // Binding a variable halves both factors in step.
        self.lagrange.fix_prefix_var_mut(r);
        self.batched.fix_prefix_var_mut(r);
    }

    fn round_poly(&self) -> Vec<EF> {
        // A degree-two round polynomial is reported at two nodes.
        // The value at the third is recoverable from the running claim.
        let half = self.lagrange.num_evals() / 2;
        [0, 2]
            .into_iter()
            .map(|node| {
                let node = EF::interpolation_node(node);

                // Each factor is read between its halves, then multiplied.
                (0..half)
                    .map(|index| {
                        let at = |poly: &Poly<EF>| {
                            let values = poly.as_slice();
                            values[index] + (values[index + half] - values[index]) * node
                        };
                        at(&self.lagrange) * at(&self.batched)
                    })
                    .sum()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use p3_binary_field::{BinaryChallenger, BinaryField8, BinaryField128};
    use p3_challenger::HashChallenger;
    use p3_field::PrimeCharacteristicRing;
    use p3_keccak::Keccak256Hash;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::generic_degree::{GenericDegreeError, GenericDegreeProof};
    use crate::univariate_skip::{
        SkipOpeningProverTranscript, SkipOpeningShape, SkipOpeningTranscriptError, SkipRound,
    };

    /// The subspace the skip round runs over lives in a byte field.
    type F = BinaryField8;

    /// Challenges and folded values live in the 128-bit field above it.
    type EF = BinaryField128;

    /// The transcript both sides are driven with.
    type Challenger = BinaryChallenger<EF, HashChallenger<u8, Keccak256Hash, 32>>;

    /// Skipped variables in every round-trip fixture below.
    const LOG_SKIP: usize = 3;

    /// Row variables in every round-trip fixture below.
    const LOG_ROWS: usize = 2;

    /// Committed polynomials the round trip batches.
    const NUM_POLYS: usize = 3;

    const fn fresh_challenger() -> Challenger {
        Challenger::from_hasher(Vec::new(), Keccak256Hash)
    }

    /// Everything one run of the reduction puts on the wire.
    #[derive(Clone, Debug)]
    struct Run {
        /// The blend the skip round left behind, one per polynomial.
        blends: Vec<EF>,
        /// The delegated sumcheck's record.
        sumcheck: GenericDegreeProof<EF, EF>,
        /// The committed evaluations at the point the rounds ended on.
        ///
        /// A real verifier reads these from a commitment.
        /// Here the run carries them.
        openings: Vec<EF>,
    }

    /// The shape both sides of the round trip are described with.
    const fn shape_at(pow_bits: usize) -> SkipOpeningShape {
        SkipOpeningShape::new(LOG_SKIP, NUM_POLYS, pow_bits)
    }

    /// The shape every test without grinding uses.
    const fn shape() -> SkipOpeningShape {
        shape_at(0)
    }

    /// Prove one reduction over several committed polynomials.
    fn prove(seed: u64) -> (SkipOpening<EF>, Point<EF>, Run) {
        prove_at(seed, 0)
    }

    /// The same, with the grinding difficulty spelled out.
    fn prove_at(seed: u64, pow_bits: usize) -> (SkipOpening<EF>, Point<EF>, Run) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let round = SkipRound::<F>::new(LOG_SKIP, 2).unwrap();
        let num_rows = 1 << LOG_ROWS;

        // One packed witness per polynomial, and the point rows are read at.
        let witnesses = (0..NUM_POLYS)
            .map(|_| packed(&mut rng, num_rows, round.row_bytes()))
            .collect::<Vec<_>>();
        let lambda = rng.random::<EF>();
        let opening = SkipOpening::new(round.lagrange::<EF>(lambda));
        let rho = Point::new((0..LOG_ROWS).map(|_| rng.random::<EF>()).collect());

        // The fold each polynomial contributes, and the blend left on it.
        let folded = opening.partial_evaluations(witnesses.iter().map(Vec::as_slice), &rho);
        let blends = folded
            .iter()
            .map(|poly| {
                poly.as_slice()
                    .iter()
                    .zip(opening.lagrange().as_slice())
                    .map(|(&value, &weight)| value * weight)
                    .sum::<EF>()
            })
            .collect::<Vec<_>>();

        let mut challenger = fresh_challenger();
        let mut transcript = SkipOpeningProverTranscript::<Challenger, EF, EF>::new(
            &mut challenger,
            shape_at(pow_bits),
        );

        // The claims are bound here, before the challenge that separates them.
        let gamma = transcript.batching_challenge(&blends);
        let starting_sum = SkipOpening::batch_claims(&blends, gamma);

        let mut prover = opening.prover(SkipOpening::batch(&folded, gamma));
        let (sumcheck, tau) = transcript.sumcheck(|challenger| {
            prover.prove::<EF, _>(challenger, LOG_SKIP, OPENING_DEGREE, pow_bits, starting_sum)
        });
        transcript.finish();

        // Each polynomial at the point the rounds ended on, residual first.
        let whole = Point::new(rho.as_slice().iter().copied().chain(tau).collect());
        let openings = witnesses
            .iter()
            .map(|rows| Poly::new(unpack(rows)).eval_base(&whole))
            .collect::<Vec<_>>();

        (
            opening,
            rho,
            Run {
                blends,
                sumcheck,
                openings,
            },
        )
    }

    /// Replay one run the way a caller would, with only the run and the shape.
    ///
    /// The library owns the whole replay, so this holds only the discharge.
    fn verify(opening: &SkipOpening<EF>, run: &Run) -> Result<(), SkipOpeningError> {
        verify_at(opening, run, 0)
    }

    /// The same, replaying under the difficulty the run was proved with.
    fn verify_at(
        opening: &SkipOpening<EF>,
        run: &Run,
        pow_bits: usize,
    ) -> Result<(), SkipOpeningError> {
        let mut challenger = fresh_challenger();
        let claim = opening.verify(
            shape_at(pow_bits),
            &run.sumcheck,
            &run.blends,
            &mut challenger,
        )?;
        claim.discharge(&run.openings)
    }

    /// The batching challenge one set of claims produces.
    fn challenge_for(claims: &[EF]) -> EF {
        let mut challenger = fresh_challenger();
        let mut transcript =
            SkipOpeningProverTranscript::<Challenger, EF, EF>::new(&mut challenger, shape());
        let gamma = transcript.batching_challenge(claims);
        transcript.sumcheck(|_| ());
        transcript.finish();
        gamma
    }

    /// Read one packed bit-valued polynomial as its values over the hypercube.
    fn unpack(packed: &[u8]) -> Vec<EF> {
        // Bit `i` of byte `i / 8` is the value at hypercube index `i`.
        (0..packed.len() * CHUNK_BITS)
            .map(|index| {
                let bit = (packed[index / CHUNK_BITS] >> (index % CHUNK_BITS)) & 1;
                if bit == 1 { EF::ONE } else { EF::ZERO }
            })
            .collect()
    }

    /// Draw a packed witness of the given shape.
    fn packed(rng: &mut SmallRng, num_rows: usize, row_bytes: usize) -> Vec<u8> {
        (0..num_rows * row_bytes)
            .map(|_| rng.random::<u8>())
            .collect()
    }

    #[test]
    fn the_fold_is_the_witness_read_at_the_residual_point() {
        // Invariant: folding the rows at rho gives the witness over the cube.
        //
        //     f~(rho, c) = sum_x eq(rho, x) * bit(x, c)
        //
        // Fixture state: 2^4 rows of 2^6 bits.
        let mut rng = SmallRng::seed_from_u64(0x09E7);
        let round = SkipRound::<F>::new(6, 2).unwrap();
        let num_rows = 1 << 4;
        let rows = packed(&mut rng, num_rows, round.row_bytes());
        let bits = unpack(&rows);

        let lambda = rng.random::<EF>();
        let opening = SkipOpening::new(round.selector::<EF>(lambda).lagrange().clone());
        let rho = Point::<EF>::rand(&mut rng, 4);

        let folded =
            opening.partial_evaluation(&rows, &Poly::new_from_point(rho.as_slice(), EF::ONE));

        // The reference weighs every cell of the hypercube directly.
        let eq = Poly::new_from_point(rho.as_slice(), EF::ONE);
        let size = opening.size();
        for column in 0..size {
            let expected = (0..num_rows)
                .map(|row| eq.as_slice()[row] * bits[row * size + column])
                .sum::<EF>();
            assert_eq!(folded.as_slice()[column], expected, "c={column}");
        }
    }

    #[test]
    fn the_blend_the_round_leaves_is_the_weighted_fold() {
        // Invariant: the claim the round hands over is the Lagrange blend.
        //
        //     bound row at rho  ==  sum_c L_c(lam) * f~(rho, c)
        //
        // This is the identity the whole reduction exists to discharge.
        // It is pinned against the round's own binding rather than assumed.
        let mut rng = SmallRng::seed_from_u64(0xB1E0);
        let round = SkipRound::<F>::new(6, 2).unwrap();
        let num_rows = 1 << 4;
        let rows = packed(&mut rng, num_rows, round.row_bytes());

        let lambda = rng.random::<EF>();
        let selector = round.selector::<EF>(lambda);
        let rho = Point::new((0..4).map(|_| rng.random::<EF>()).collect());

        // What the round leaves: the bound rows, read at the residual point.
        let blended = selector.bind(&rows).eval_base(&rho);

        // What the reduction starts from: the Lagrange vector against the fold.
        let opening = SkipOpening::new(selector.lagrange().clone());
        let folded =
            opening.partial_evaluation(&rows, &Poly::new_from_point(rho.as_slice(), EF::ONE));
        let weighted = folded
            .as_slice()
            .iter()
            .zip(opening.lagrange().as_slice())
            .map(|(&value, &weight)| value * weight)
            .sum::<EF>();

        assert_eq!(blended, weighted);
    }

    #[test]
    fn batching_agrees_with_weighing_the_claims_by_hand() {
        // Invariant: batching witnesses and batching claims use one order.
        // A mismatch here would prove the right sum against the wrong claim.
        let mut rng = SmallRng::seed_from_u64(0xBA7C);
        let size = 64;
        let polys = (0..3)
            .map(|_| Poly::new((0..size).map(|_| rng.random::<EF>()).collect::<Vec<_>>()))
            .collect::<Vec<_>>();
        let weights = (0..size).map(|_| rng.random::<EF>()).collect::<Vec<_>>();
        let gamma = rng.random::<EF>();

        // The batched witness weighed once.
        let batched = SkipOpening::batch(&polys, gamma);
        let combined = batched
            .as_slice()
            .iter()
            .zip(&weights)
            .map(|(&value, &weight)| value * weight)
            .sum::<EF>();

        // Each witness weighed, then the claims batched.
        let claims = polys
            .iter()
            .map(|poly| {
                poly.as_slice()
                    .iter()
                    .zip(&weights)
                    .map(|(&value, &weight)| value * weight)
                    .sum::<EF>()
            })
            .collect::<Vec<_>>();

        assert_eq!(combined, SkipOpening::batch_claims(&claims, gamma));
    }

    #[test]
    fn an_honest_run_verifies_through_both_drivers() {
        // Fixture state: 3 skipped variables, 2 row variables, 3 polynomials.
        //
        //     blends -> gamma -> 3 sumcheck rounds -> tau -> openings
        //
        // Invariant: the run replays, and both equalities hold on it.
        let (opening, _, run) = prove(0xE2E);

        assert_eq!(verify(&opening, &run), Ok(()));
    }

    #[test]
    fn the_batching_challenge_follows_the_claims() {
        // Invariant: the challenge is a function of the claims it separates.
        // Were it drawn first, a prover seeing it could move value between:
        //
        //     v_0 += gamma * d,  v_1 -= d      leaves the batch unchanged
        //
        // Binding the claims ahead of the draw is what moves gamma with them.
        let mut rng = SmallRng::seed_from_u64(0x6A3);
        let claims = (0..NUM_POLYS)
            .map(|_| rng.random::<EF>())
            .collect::<Vec<_>>();
        let gamma = challenge_for(&claims);

        // The shift that leaves the batch fixed, applied to the claims.
        let delta = rng.random::<EF>();
        let mut shifted = claims.clone();
        shifted[0] += gamma * delta;
        shifted[1] -= delta;
        assert_eq!(
            SkipOpening::batch_claims(&shifted, gamma),
            SkipOpening::batch_claims(&claims, gamma),
            "the shift is chosen to leave the batch alone"
        );

        // The challenge nonetheless moves, so the shifted claims batch anew.
        assert_ne!(challenge_for(&shifted), gamma);
    }

    #[test]
    fn a_shift_that_preserves_the_batch_is_still_rejected() {
        // Mutation: move value between two blends, keeping their honest batch.
        // The claims are bound before the draw, so the replay draws another.
        // The starting sum the proof carries then no longer matches.
        let (opening, _, honest) = prove(0x5417);
        let gamma = challenge_for(&honest.blends);

        // A nonzero shift, drawn rather than built from an integer.
        // In characteristic two an integer is its parity, so an even one is 0.
        let delta = SmallRng::seed_from_u64(0x5418).random::<EF>();
        let mut run = honest;
        run.blends[0] += gamma * delta;
        run.blends[1] -= delta;

        // Under the honest challenge the batch is untouched: the whole attack.
        assert_eq!(
            SkipOpening::batch_claims(&run.blends, gamma),
            run.sumcheck.claimed_sum
        );

        assert_eq!(
            verify(&opening, &run),
            Err(SkipOpeningError::StartingSumMismatch)
        );
    }

    #[test]
    fn a_tampered_starting_sum_is_rejected() {
        // Mutation: the starting sum, which the driver reads from the proof.
        // Nothing inside the sumcheck ties it to the claims, so this must.
        let (opening, _, honest) = prove(0x57A47);

        let mut run = honest;
        run.sumcheck.claimed_sum += EF::ONE;

        // The replay rejects: the altered sum is bound, so nothing closes.
        assert!(verify(&opening, &run).is_err());
    }

    #[test]
    fn a_tampered_opening_is_rejected() {
        // Mutation: one committed evaluation the reduction closes against.
        // This is the check that strips the Lagrange weight.
        // It is all that stands between the rounds and a wrong answer.
        let (opening, _, honest) = prove(0x09E1);

        let mut run = honest;
        run.openings[2] += EF::ONE;

        assert_eq!(
            verify(&opening, &run),
            Err(SkipOpeningError::ClosingValueMismatch)
        );
    }

    #[test]
    fn a_tampered_round_polynomial_is_rejected() {
        // Mutation: zero one round polynomial, as a skipping prover would.
        // A degree-two round sends two nodes, taking the third from the claim.
        // No round is self-contradicting, and the sumcheck alone accepts.
        //
        // The rounds reduce, they do not filter, so the tamper reaches the end.
        // The closing check against the committed openings is what refuses it.
        let (opening, _, honest) = prove(0x20D);

        let mut run = honest;
        run.sumcheck.round_polys[1] = alloc::vec![EF::ZERO; OPENING_DEGREE];

        assert_eq!(
            verify(&opening, &run),
            Err(SkipOpeningError::ClosingValueMismatch)
        );
    }

    #[test]
    fn a_round_the_nested_sumcheck_refuses_is_an_error_not_a_panic() {
        // Mutation: drop one round, so the nested replay refuses the shape.
        //
        // The driver's bracket is half played when that happens.
        // The transcript panics on drop unless the rejection releases it.
        // So this pins the release, which a tamper caught later cannot.
        let (opening, _, honest) = prove(0xAB07);

        let mut run = honest;
        run.sumcheck.round_polys.pop();

        assert_eq!(
            verify(&opening, &run),
            Err(SkipOpeningError::Sumcheck(
                GenericDegreeError::RoundCountMismatch {
                    expected: LOG_SKIP,
                    actual: LOG_SKIP - 1,
                }
            ))
        );
    }

    #[test]
    fn a_grinding_witness_the_nested_sumcheck_refuses_is_an_error() {
        // The same release, through the other rejection the replay can raise.
        //
        // Fixture state: 4 bits of difficulty, one witness per round.
        let (opening, _, honest) = prove_at(0xA607, 4);

        let mut run = honest;
        run.sumcheck.pow_witnesses.pop();

        assert!(matches!(
            verify_at(&opening, &run, 4),
            Err(SkipOpeningError::Sumcheck(
                GenericDegreeError::PowWitnessCountMismatch { .. }
            ))
        ));
    }

    #[test]
    fn a_wrong_claim_count_is_rejected() {
        // Mutation: drop one blend, so the batch is narrower than described.
        // The count is bound as the claims step's width, so the replay refuses.
        let (opening, _, honest) = prove(0xC07);

        let mut run = honest;
        run.blends.pop();

        assert_eq!(
            verify(&opening, &run),
            Err(SkipOpeningError::Transcript(
                SkipOpeningTranscriptError::ClaimCountMismatch {
                    expected: NUM_POLYS,
                    actual: NUM_POLYS - 1,
                }
            ))
        );
    }

    #[test]
    fn blends_and_openings_must_agree_on_count() {
        // The two batches weigh the same polynomials, so a mismatch is void.
        let (opening, _, honest) = prove(0xC08);

        let mut run = honest;
        run.openings.pop();

        assert_eq!(
            verify(&opening, &run),
            Err(SkipOpeningError::ClaimCountMismatch {
                blends: NUM_POLYS,
                openings: NUM_POLYS - 1,
            })
        );
    }

    #[test]
    fn the_verifier_builds_the_same_lagrange_vector_as_the_prover() {
        // The verifier needs the vector alone, not the 32 KiB table beside it.
        // Both come from one computation, so the two cannot drift apart.
        let mut rng = SmallRng::seed_from_u64(0x1A6);
        let round = SkipRound::<F>::new(LOG_SKIP, 2).unwrap();

        for lambda in (0..8)
            .map(|_| rng.random::<EF>())
            .chain(round.domain().subspace().iter().map(|&s| EF::from(s)))
        {
            assert_eq!(
                round.lagrange::<EF>(lambda),
                *round.selector::<EF>(lambda).lagrange()
            );
        }
    }

    #[test]
    #[should_panic(expected = "batches at least one polynomial")]
    fn an_empty_batch_is_refused() {
        // Nothing to reduce, and every later step assumes one claim at least.
        let _ = SkipOpeningShape::new(LOG_SKIP, 0, 0);
    }

    proptest! {
        #[test]
        fn the_reduction_ends_on_the_committed_evaluation(seed: u64, log_rows in 1usize..=4) {
            // Invariant: the rounds leave the value a commitment opens.
            //
            //     surviving claim  ==  f~(rho, tau)
            //
            // The point is (rho, tau): residual variables, then skipped.
            let mut rng = SmallRng::seed_from_u64(seed);
            let round = SkipRound::<F>::new(3, 2).unwrap();
            let num_rows = 1 << log_rows;
            let rows = packed(&mut rng, num_rows, round.row_bytes());
            let bits = Poly::new(unpack(&rows));

            let lambda = rng.random::<EF>();
            let opening = SkipOpening::new(round.selector::<EF>(lambda).lagrange().clone());
            let rho = Point::new((0..log_rows).map(|_| rng.random::<EF>()).collect());

            // Run the rounds by hand, folding at fresh challenges.
            let folded = opening.partial_evaluations([rows.as_slice()], &rho);
            let mut prover = opening.prover(SkipOpening::batch(&folded, EF::ONE));
            let mut tau = Vec::new();
            for _ in 0..opening.num_variables() {
                let _ = prover.round_poly();
                let challenge = rng.random::<EF>();
                prover.fold(challenge);
                tau.push(challenge);
            }

            // The full point puts the residual variables first.
            let whole = Point::new(rho.as_slice().iter().copied().chain(tau).collect());
            prop_assert_eq!(prover.surviving_claim(), bits.eval_base(&whole));
        }
    }
}
