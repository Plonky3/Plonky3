//! One ring-switching reduction at a bit alphabet, and its five values.

use alloc::vec::Vec;

use p3_binary_field::TowerLevel;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;

use super::basis::Coefficients;
use super::packing::BitPacking;
use super::tensor::BitTensor;

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
///     bind r, bind s_hat, draw r''
///     batch(r'')      initial_sum, closing_weight, weights
/// ```
///
/// Drawing `r''` first is unsound.
/// A bit matrix solving two `F_2`-linear systems moves the claim, sum held.
/// The rounds and the closing check then accept a true surviving claim.
///
/// The split is what stops a driver reaching for that order.
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
/// Both the tensor accumulation and the weight multilinear are subset sums.
/// A general alphabet needs `d` multiplications per hypercube point instead.
#[derive(Clone, Debug)]
pub struct BitRingSwitch<EF> {
    /// The coordinates of the evaluation point the packing keeps.
    high: Point<EF>,
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
        let (high, low) = r.split_at(r.num_variables() - Self::ABSORBED);

        Ok(Self {
            eq_low: Poly::new_from_point(low.as_slice(), EF::ONE),
            high,
        })
    }

    /// Variables the packed polynomial this reduction runs against must have.
    ///
    /// This is also the number of sumcheck rounds the reduction takes.
    pub const fn num_variables(&self) -> usize {
        self.high.num_variables()
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

    /// The equality table over the kept coordinates.
    ///
    /// `2^l'` entries, so only the prover's two operations build it.
    fn eq_high(&self) -> Poly<EF> {
        Poly::new_from_point(self.high.as_slice(), EF::ONE)
    }

    /// `sum_w eq(r_high, w) ⊗ t'(w)`, the element the prover sends.
    ///
    /// # Overview
    ///
    /// Read by columns it gives the bit planes at the kept coordinates.
    /// That is the reading which tests the incoming claim.
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
    /// The `2^l'` equality table is built here rather than held.
    /// That is strictly below the accumulation it feeds.
    /// It also keeps a verifier from paying for a table it never reads.
    ///
    /// # Errors
    ///
    /// Returns an error unless the packing has the reduction's variables.
    pub fn tensor(&self, packing: &BitPacking<EF>) -> Result<BitTensor<EF>, BitRingSwitchError>
    where
        EF: Send + Sync,
    {
        self.check_width(packing.num_variables())?;

        Ok(self
            .eq_high()
            .as_slice()
            .par_chunks(CHUNK)
            .zip(packing.poly().as_slice().par_chunks(CHUNK))
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
            let (a, b) = (self.high[i], r_prime[i]);

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
/// Reaching this stage is the proof that the element was bound first.
/// That is the order the construction's soundness rests on.
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
    /// Prover-side, and the only other place the `2^l'` table is built.
    pub fn weights(&self) -> Poly<EF>
    where
        EF: Send + Sync,
    {
        Poly::new(
            self.reduction
                .eq_high()
                .as_slice()
                .par_iter()
                .map(|&value| {
                    Coefficients::of(value)
                        .iter_set()
                        .map(|u| self.eq_batch.as_slice()[u])
                        .sum()
                })
                .collect::<Vec<_>>(),
        )
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

#[cfg(test)]
mod tests {
    use p3_binary_field::BinaryField16;
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type EF = BinaryField16;

    /// A random bit witness of the given byte length.
    fn bits(seed: u64, bytes: usize) -> Vec<u8> {
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..bytes).map(|_| rng.random::<u8>()).collect()
    }

    /// One reduction over a witness of the given byte length.
    ///
    /// The batching challenge comes back beside it rather than folded in.
    /// A test then reaches the second stage the way a driver does.
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
        let eq = reduction.eq_high();

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
        let eq = reduction.eq_high();

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

        for (w, &value) in reduction.eq_high().as_slice().iter().enumerate() {
            let expected: EF = Coefficients::of(value)
                .iter()
                .zip(batch.eq_batch.as_slice())
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
            let eq_high = reduction.eq_high();
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

    proptest! {
        #[test]
        fn the_chunked_accumulation_matches_the_term_by_term_sum(seed: u64, log_n in 1usize..6) {
            // The parallel path accumulates in chunks, combining partials.
            // The reference forms every term as its own element and adds them.
            let (reduction, packing, _, _) = fixture(seed, (1 << log_n) * 2);

            let mut expected = BitTensor::zero();
            for (&weight, &value) in reduction
                .eq_high()
                .as_slice()
                .iter()
                .zip(packing.poly().as_slice())
            {
                expected += BitTensor::exterior_product(weight, value);
            }

            prop_assert_eq!(reduction.tensor(&packing).unwrap(), expected);
        }
    }
}
