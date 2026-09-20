//! Sumcheck rounds over tables held in a field isomorphic to the challenge field.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_binary_field::{BinaryField64, BinaryField128, Ghash128, Poly64};
use p3_challenger::fs::TranscriptField;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;

use super::{Basis, SumcheckProver, VariableOrder};
use crate::SumcheckData;
use crate::product_polynomial::ProductPolynomial;
use crate::transcript::{ProverTranscript, SumcheckShape};

/// A [`SumcheckProver`] whose tables live in a field `R` isomorphic to its challenge field `EF`.
///
/// # Overview
///
/// Both tables, the running claim and a held challenge are all elements of `R`.
/// Only the transcript sees `EF`:
///
/// ```text
///     round message   measured in R, crosses into EF, then observed
///     challenge       sampled in EF, crosses into R, then held
/// ```
///
/// The transcript is therefore exactly the one the `EF` prover writes.
/// What changes is the cost of each multiply inside a binding or measuring pass.
///
/// # Contract
///
/// `R::from` and [`IntoTranscriptField::into_transcript`] must be mutually inverse field
/// isomorphisms.
/// A map that is not a ring homomorphism measures a different round message.
///
/// # Storage
///
/// Tables are held as scalars. [`Self::from_tables`] takes scalar tables to begin with;
/// [`Self::new`] unpacks whatever storage the source prover used.
///
/// That costs nothing where the tables were scalar already: suffix binding always is, and
/// so is a field whose packing is itself, like `BinaryField128`. A packed prefix pair over
/// any other field loses its SIMD lanes for every remaining round.
#[derive(Debug, Clone)]
pub struct ReprSumcheckProver<F, EF, R: Field> {
    /// The rounds, run entirely in `R`.
    inner: SumcheckProver<R, R>,
    /// The base and challenge fields the transcript is written over.
    _transcript: PhantomData<(F, EF)>,
}

impl<F, EF, R> ReprSumcheckProver<F, EF, R>
where
    F: Field,
    EF: ExtensionField<F>,
    R: IntoTranscriptField<EF>,
{
    /// Moves a prover's tables, claim and held challenge into `R`.
    ///
    /// A held challenge crosses as it is, so the next measuring pass still absorbs it.
    ///
    /// This is the only entry that takes a prover mid-fold: it carries an outstanding challenge
    /// across, and it accepts a packed prefix pair, which it unpacks.
    #[tracing::instrument(skip_all)]
    pub fn new(prover: SumcheckProver<F, EF>) -> Self {
        let SumcheckProver {
            poly,
            sum,
            outstanding,
        } = prover;
        let (order, evals, weights) = poly.into_scalar_tables();

        // One table at a time, so only one source and its image are resident together.
        let evals = Poly::new(R::from_table(evals.into_evals()));
        let weights = Poly::new(R::from_table(weights.into_evals()));

        Self {
            inner: SumcheckProver {
                poly: ProductPolynomial::new_unpacked(order, evals, weights),
                sum: R::from(sum),
                outstanding: outstanding.map(R::from),
            },
            _transcript: PhantomData,
        }
    }

    /// Builds a prover from an evaluation table in `EF` and a weight table already in `R`.
    ///
    /// Only the evaluations cross into `R` here; a caller that can accumulate its weights in
    /// `R` directly saves the second crossing.
    #[tracing::instrument(skip_all)]
    pub fn from_tables(order: VariableOrder, evals: Poly<EF>, weights: Poly<R>, sum: EF) -> Self {
        let evals = Poly::new(R::from_table(evals.into_evals()));
        Self::from_repr_tables(order, evals, weights, sum)
    }

    /// Builds a prover from an evaluation table and a weight table both already in `R`.
    pub(crate) fn from_repr_tables(
        order: VariableOrder,
        evals: Poly<R>,
        weights: Poly<R>,
        sum: EF,
    ) -> Self {
        let poly = ProductPolynomial::new_unpacked(order, evals, weights);

        // `SumcheckProver::new` checks, in debug builds, that the claim and this pair agree in `R`.
        Self {
            inner: SumcheckProver::new(poly, R::from(sum)),
            _transcript: PhantomData,
        }
    }

    /// Returns the current claimed sum over the remaining unbound variables.
    pub fn claimed_sum(&self) -> EF {
        self.inner.claimed_sum().into_transcript()
    }

    /// Returns the number of remaining (unbound) variables.
    pub fn num_variables(&self) -> usize {
        self.inner.num_variables()
    }

    /// Applies an outstanding binding, so the tables are current with the claim.
    ///
    /// See [`SumcheckProver::settle`].
    pub fn settle(&mut self) {
        self.inner.settle();
    }

    /// Runs `folding_factor` sumcheck rounds.
    ///
    /// Plays the same transcript as [`SumcheckProver::compute_sumcheck_polynomials`] without a
    /// constraint, including the challenge left outstanding on return.
    ///
    /// # Returns
    ///
    /// The verifier challenges sampled during this batch.
    ///
    /// # Panics
    ///
    /// - Folding factor must not exceed the current number of remaining variables.
    #[tracing::instrument(skip_all, level = "debug")]
    pub fn compute_sumcheck_polynomials<Challenger>(
        &mut self,
        sumcheck_data: &mut SumcheckData<F, EF>,
        challenger: &mut Challenger,
        folding_factor: usize,
        pow_bits: usize,
    ) -> Point<EF>
    where
        F: TranscriptField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let shape = SumcheckShape::new(folding_factor, pow_bits, Basis::Evaluation);
        let mut transcript = ProverTranscript::<Challenger, F, EF>::new(challenger, shape);

        let challenges = (0..folding_factor)
            .map(|_| self.round(sumcheck_data, &mut transcript))
            .collect();

        // Require that every described step was played.
        transcript.finish();

        Point::new(challenges)
    }

    /// Plays one round inside a transcript the caller owns, holding its challenge back.
    ///
    /// # Returns
    ///
    /// The verifier challenge sampled for this round.
    pub(crate) fn round<Challenger>(
        &mut self,
        sumcheck_data: &mut SumcheckData<F, EF>,
        transcript: &mut ProverTranscript<'_, Challenger, F, EF>,
    ) -> EF
    where
        F: TranscriptField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Measure in R, absorbing whatever binding the last round left behind.
        let (c_a, c_inf) = self.inner.measure_round();

        // The transcript only ever sees the challenge field.
        let r = sumcheck_data.observe_and_sample(
            transcript,
            c_a.into_transcript(),
            c_inf.into_transcript(),
        );
        let r_repr = R::from(r);
        debug_assert_eq!(r_repr.into_transcript(), r);

        // The round identity is a polynomial in its inputs, so it commutes with the isomorphism.
        self.inner.sum = Basis::Evaluation.reduce_claim(c_a, c_inf, r_repr, self.inner.sum);

        self.inner.hold(r_repr);
        r
    }

    /// Returns the current weight table, applying any outstanding binding first.
    pub(crate) fn weights(&mut self) -> Poly<R> {
        self.inner.weights()
    }

    /// Returns the current evaluation table, applying any outstanding binding first.
    pub(crate) fn evals(&mut self) -> Poly<R> {
        self.inner.evals()
    }
}

/// A field that takes a whole table of another field's elements at once.
///
/// The table's image is [`From`] applied to every entry.
///
/// The default builds it entry by entry into a new buffer and then releases the source.
/// A field with a bulk kernel for the same map overrides it.
pub trait FromTable<EF: Copy + Send + Sync>: From<EF> + Send {
    /// Every entry of `table`, mapped into this field.
    fn from_table(table: Vec<EF>) -> Vec<Self> {
        let image = table.par_iter().map(|&x| Self::from(x)).collect();
        drop(table);
        image
    }
}

/// An arithmetic representation that maps back into the transcript field.
///
/// Keeping this conversion on the representation lets generic callers select `R` without
/// separately repeating the reverse-conversion bound at every layer of their API.
pub trait IntoTranscriptField<EF: Copy + Send + Sync>: Field + FromTable<EF> {
    /// Maps one arithmetic value back into the field observed by the transcript.
    fn into_transcript(self) -> EF;
}

impl<EF, R> IntoTranscriptField<EF> for R
where
    EF: Copy + Send + Sync + From<R>,
    R: Field + FromTable<EF>,
{
    fn into_transcript(self) -> EF {
        EF::from(self)
    }
}

/// The identity map, which hands back the table it was given.
impl<T: Field> FromTable<T> for T {
    fn from_table(table: Vec<Self>) -> Vec<Self> {
        table
    }
}

impl FromTable<BinaryField64> for Poly64 {
    /// Converts the table in its existing allocation.
    fn from_table(table: Vec<BinaryField64>) -> Vec<Self> {
        Self::from_tower_vec(table)
    }
}

impl FromTable<BinaryField128> for Ghash128 {
    /// Converts in the table's own buffer, a block at a time where the build has the kernel.
    fn from_table(table: Vec<BinaryField128>) -> Vec<Self> {
        Self::from_tower_vec(table)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_binary_field::{BinaryChallenger, BinaryField128, Ghash128};
    use p3_challenger::{CanSample, HashChallenger};
    use p3_field::Field;
    use p3_keccak::Keccak256Hash;
    use p3_multilinear_util::poly::Poly;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::{FromTable, ReprSumcheckProver};
    use crate::SumcheckData;
    use crate::product_polynomial::ProductPolynomial;
    use crate::strategy::{SumcheckProver, VariableOrder};

    type F = BinaryField128;
    type Ch = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;

    fn fresh_challenger() -> Ch {
        Ch::from_hasher(Vec::new(), Keccak256Hash)
    }

    /// A prover over random tables, stored packed for prefix binding and scalar for suffix.
    fn random_prover(
        order: VariableOrder,
        num_variables: usize,
        seed: u64,
    ) -> SumcheckProver<F, F> {
        let mut rng = SmallRng::seed_from_u64(seed);
        let evals: Vec<F> = (0..1 << num_variables).map(|_| rng.random()).collect();
        let weights: Vec<F> = (0..1 << num_variables).map(|_| rng.random()).collect();
        let poly = match order {
            VariableOrder::Prefix => {
                ProductPolynomial::new_packed(order, Poly::new(evals), Poly::new(weights))
            }
            VariableOrder::Suffix => {
                ProductPolynomial::new_unpacked(order, Poly::new(evals), Poly::new(weights))
            }
        };
        let sum = poly.dot_product();
        SumcheckProver::new(poly, sum)
    }

    /// Plays the challenge-field transcript through a prover represented in `R`.
    fn assert_repr_rounds_play_the_challenge_field_transcript<R>()
    where
        R: Field + FromTable<F>,
        F: From<R>,
    {
        // Batches mix single rounds, which fuse across calls, with a settle and a longer batch.
        let batches = [1, 2, 1, 3, 1, 1];

        // Grinding stays off: a parallel grinder may return any valid witness,
        // so two provers could part ways at the first grinding step.
        let pow_bits = 0;
        let num_variables: usize = batches.iter().sum();

        for order in [VariableOrder::Prefix, VariableOrder::Suffix] {
            // Rounds the challenge-field prover plays before handing over.
            for split in 0..3 {
                let mut reference = random_prover(order, num_variables, 7);
                let mut reference_data = SumcheckData::default();
                let mut reference_challenger = fresh_challenger();

                let mut handoff = random_prover(order, num_variables, 7);
                let mut handoff_data = SumcheckData::default();
                let mut handoff_challenger = fresh_challenger();

                // The batch sizes are part of the transcript, so the head replays them.
                //
                // A handoff after any batch carries that batch's last challenge across.
                for &rounds in &batches[..split] {
                    handoff.compute_sumcheck_polynomials(
                        &mut handoff_data,
                        &mut handoff_challenger,
                        rounds,
                        pow_bits,
                        None,
                    );
                }
                let head = batches[..split].iter().sum::<usize>();
                let mut repr = ReprSumcheckProver::<F, F, R>::new(handoff);

                let mut played = 0;
                for (batch, &rounds) in batches.iter().enumerate() {
                    let expected = reference.compute_sumcheck_polynomials(
                        &mut reference_data,
                        &mut reference_challenger,
                        rounds,
                        pow_bits,
                        None,
                    );
                    played += rounds;
                    if batch == 3 {
                        reference.settle();
                    }
                    if played <= head {
                        continue;
                    }

                    let got = repr.compute_sumcheck_polynomials(
                        &mut handoff_data,
                        &mut handoff_challenger,
                        rounds,
                        pow_bits,
                    );
                    if batch == 3 {
                        repr.settle();
                    }

                    assert_eq!(got, expected, "{order:?} split {split}");
                    assert_eq!(repr.claimed_sum(), reference.claimed_sum());
                    assert_eq!(repr.num_variables(), reference.num_variables());
                }

                assert_eq!(
                    handoff_data.polynomial_evaluations,
                    reference_data.polynomial_evaluations
                );
                assert_eq!(handoff_data.pow_witnesses, reference_data.pow_witnesses);
                assert_eq!(
                    CanSample::<F>::sample(&mut handoff_challenger),
                    CanSample::<F>::sample(&mut reference_challenger)
                );

                // The final binding lands on the claim in both fields.
                reference.settle();
                repr.settle();
                assert_eq!(repr.claimed_sum(), reference.claimed_sum());
            }
        }
    }

    #[test]
    fn repr_rounds_play_the_challenge_field_transcript() {
        // The polynomial basis takes tables through its own kernel.
        assert_repr_rounds_play_the_challenge_field_transcript::<Ghash128>();

        // The field itself takes them through the default map.
        assert_repr_rounds_play_the_challenge_field_transcript::<F>();
    }
}
