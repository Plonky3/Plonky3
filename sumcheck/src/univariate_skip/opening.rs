//! Turning a skip round's Lagrange-weighted claim into one evaluation point.

use alloc::vec::Vec;

use p3_field::Field;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;

use super::lde::CHUNK_BITS;
use crate::generic_degree::RoundProver;

/// Per-variable degree of the summand this reduction proves.
///
/// The Lagrange vector and the witness fold contribute one each.
pub const OPENING_DEGREE: usize = 2;

/// The claim a skip round leaves behind, and the reduction that makes it openable.
///
/// # Overview
///
/// A skip round binds its variables with one challenge rather than one per variable.
///
/// What comes back is not an evaluation of the committed polynomial.
///
/// Writing `rho` for the residual point and `lam` for the skip challenge:
///
/// ```text
///     f(rho, lam) = sum_x eq(rho, x) * sum_c L_c(lam) * f~(x, c)
///                 = sum_c L_c(lam) * f~(rho, c)
/// ```
///
/// That is a weighted blend of `2^k` positions of the committed multilinear.
///
/// A commitment answers one question — the value at a point — so the blend has to go.
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
/// - The output is `f~(rho, tau)`, an evaluation of the committed multilinear at one point.
///
/// Several committed polynomials share the same Lagrange factor.
///
/// One challenge therefore batches them into a single run.
///
/// # Verifier cost
///
/// The verifier reads the Lagrange vector at the output point:
///
/// ```text
///     L^(tau) = Z_S(lam) / Z_S'(S) * sum_c eq(tau, c) / (lam + s_c)
/// ```
///
/// That is `2^k` terms and one batch inversion.
///
/// It adds no new asymptotic, since reading the round message already costs `2^k`.
#[derive(Debug, Clone)]
pub struct SkipOpening<EF> {
    /// The Lagrange vector of the skipped subspace, read at the round's challenge.
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

    /// Number of variables this reduction binds, which is the number the round skipped.
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

    /// Fold packed rows over the row variables, giving the witness over the skipped cube.
    ///
    /// # Arguments
    ///
    /// - `packed`: one polynomial's rows back to back, least significant bit first per byte.
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

        // A row contributes its weight to exactly the skipped points whose bit is set.
        //
        //     f~(rho, c) = sum over rows with bit c set of eq(rho, row)
        //
        // The witness is bits, so this is an equality-weighted count with no multiplications:
        // one addition per set bit, and the zero bits cost nothing at all.
        let values = packed
            .par_chunks_exact(row_bytes)
            .zip(eq_rows.as_slice().par_iter())
            .par_fold_reduce(
                || EF::zero_vec(self.size()),
                |mut accumulator, (row, &weight)| {
                    for (chunk, &byte) in row.iter().enumerate() {
                        // Peel the set bits one at a time, skipping the runs of zeros.
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
                    // Addition is associative, so regrouping the splits cannot change the sum.
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
        // The equality table depends only on the residual point, so it is built once.
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

        // Horner over the polynomials keeps this to one multiplication per entry per operand.
        let mut batched = EF::zero_vec(size);
        for poly in folded.iter().rev() {
            for (entry, &value) in batched.iter_mut().zip(poly.as_slice()) {
                *entry = *entry * gamma + value;
            }
        }
        Poly::new(batched)
    }

    /// The claim the reduction starts from, given each polynomial's blended value.
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

    /// Build the prover state for the reduction's sumcheck.
    #[must_use]
    pub fn prover(&self, batched: Poly<EF>) -> SkipOpeningProver<EF> {
        SkipOpeningProver {
            lagrange: self.lagrange.clone(),
            batched,
        }
    }
}

/// Prover state of the reduction's sumcheck.
///
/// The summand is the Lagrange vector times the batched witness, degree two per variable.
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
    /// Reading it after the rounds have run gives the evaluation a commitment can open.
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
        //
        // The value at the third is recoverable from the running claim.
        let half = self.lagrange.num_evals() / 2;
        [0, 2]
            .into_iter()
            .map(|node| {
                let node = EF::interpolation_node(node);

                // Each factor is read between its two halves at the node, then multiplied.
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
    use p3_binary_field::{BinaryField8, BinaryField128};
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::univariate_skip::SkipRound;

    /// The subspace the skip round runs over lives in a byte field.
    type F = BinaryField8;

    /// Challenges and folded values live in the 128-bit field above it.
    type EF = BinaryField128;

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
        // Invariant: folding the rows at rho gives the witness over the skipped cube.
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
        let rho = Point::new((0..4).map(|_| rng.random::<EF>()).collect());

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
        // Invariant: the claim the skip round hands over really is the Lagrange blend.
        //
        //     bound row at rho  ==  sum_c L_c(lam) * f~(rho, c)
        //
        // This is the identity the whole reduction exists to discharge.
        //
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
        // Invariant: batching the witnesses and batching their claims use one Horner order.
        //
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

    proptest! {
        #[test]
        fn the_reduction_ends_on_the_committed_evaluation(seed: u64, log_rows in 1usize..=4) {
            // Invariant: running the rounds to the end leaves the value a commitment opens.
            //
            //     surviving claim  ==  f~(rho, tau)
            //
            // The point is (rho, tau): the residual variables, then the skipped ones.
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
