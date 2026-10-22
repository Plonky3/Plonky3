//! A batch of bit-alphabet claims whose sum the commitment's own initial sumcheck proves.
//!
//! The batch binds every claim's element and draws its challenges, as the reduced batch does.
//!
//! It then runs no rounds: the combined weight `sum_i lambda^i * A_i` joins the commitment's first fold.
//!
//! That fold binds the trailing kept coordinates first, so each claim splits its element over them.
//!
//! Bank `b` of a claim pairs its rows with the coordinate images of `eq(tail, b)`, as the compact head does over leading ones.

use alloc::vec::Vec;

use p3_binary_field::{BitCoordinates, TowerLevel};
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use serde::{Deserialize, Serialize};

use super::basis::{Coefficients, CoordinateSums};
use super::claims::{BitRingSwitchClaims, ClaimElements};
use super::equality::FactoredEquality;
use super::reduction::{BitRingSwitch, BitRingSwitchBatch, BitRingSwitchError};
use super::tensor::BitTensor;
use crate::layout::InitialWeight;
use crate::product_polynomial::ProductPolynomial;
use crate::strategy::{SumcheckProver, VariableOrder};

/// The messages a deferred batch puts on the wire: every claim's elements, and nothing else.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: BitCoordinates, EF: BitCoordinates",
    deserialize = "F: BitCoordinates, EF: BitCoordinates"
))]
pub struct BitRingSwitchDeferredProof<F, EF = F> {
    /// Every claim's elements, in the order the claims are bound.
    pub claims: Vec<ClaimElements<F, EF>>,
}

/// The prover's share of a deferred batch: the combined weight, measured trailing coordinates first.
#[derive(Debug)]
pub struct DeferredWeight<'a, F, EF: Field = F> {
    /// One measuring prover per claim.
    heads: Vec<ClaimHead<'a, F, EF>>,
    /// `sum_i lambda^i * sigma_i`, the sum the weight claims.
    sum: EF,
}

/// One claim's banks against its coordinate images, over the trailing coordinates.
#[derive(Debug)]
struct ClaimHead<'a, F, EF: Field> {
    /// The claim's second stage.
    batch: BitRingSwitchBatch<'a, F, EF>,
    /// Bank rows times scaled coordinate images, the bank index bit-reversed.
    prover: SumcheckProver<EF, EF>,
    /// The last message measured, which the next challenge reduces the claim by.
    message: (EF, EF),
    /// Challenges applied so far.
    bound: usize,
    /// The equality table of the leading coordinates.
    head: FactoredEquality<EF>,
}

impl<'a, F, EF> ClaimHead<'a, F, EF>
where
    F: TowerLevel,
    EF: BitCoordinates + ExtensionField<F>,
{
    /// Pair the banks' rows with the images of `eq(tail, b)`, every image scaled by `scale`.
    fn new(batch: BitRingSwitchBatch<'a, F, EF>, banks: &[BitTensor<EF, F>], scale: EF) -> Self {
        let reduction = batch.reduction();
        let n = reduction.num_variables();
        let depth = banks.len().trailing_zeros() as usize;
        let (head, tail) = reduction.high().split_at(n - depth);
        let tail = Poly::new_from_point(tail, EF::ONE);
        let dimension = Coefficients::<EF>::DIMENSION;
        let slots = 1usize << BitRingSwitch::<F, EF>::BATCHED;
        let sums = CoordinateSums::<EF, EF>::new(batch.coordinate_weights());

        // Bank `b` sits at `reverse(b)`, so binding the index from its top binds the last coordinate first.
        let mut factors = EF::zero_vec(banks.len() * slots);
        let mut generators = EF::zero_vec(banks.len() * slots);
        for (b, (bank, &weight)) in banks.iter().zip(tail.as_slice()).enumerate() {
            let position = reverse_bits(b, depth) * slots;
            for coordinate in 0..dimension {
                let mut basis = Coefficients::<EF>::zero();
                basis.set(coordinate);
                factors[position + coordinate] = EF::from(bank.rows()[coordinate]);
                generators[position + coordinate] = scale * sums.sum(weight * basis.element());
            }
        }
        let sum = factors.iter().zip(&generators).map(|(&f, &g)| f * g).sum();
        let poly = ProductPolynomial::<EF, EF>::new_unpacked(
            VariableOrder::Prefix,
            Poly::new(factors),
            Poly::new(generators),
        );
        Self {
            batch,
            prover: SumcheckProver::new(poly, sum),
            message: (EF::ZERO, EF::ZERO),
            bound: 0,
            head: BitRingSwitch::<F, EF>::factored_equality(head),
        }
    }

    /// Apply the challenges not applied yet, at most the one the last measure left.
    fn catch_up(&mut self, challenges: &[EF]) {
        debug_assert!(
            challenges.len() <= self.bound + 1,
            "one challenge per round"
        );
        if let Some(&r) = challenges.get(self.bound) {
            let (c0, c_inf) = self.message;
            self.prover.reduce_claim_with_coefficients(c0, c_inf, r);
            self.prover.hold(r);
            self.bound += 1;
        }
    }
}

/// `b` with its `bits` lowest bits in reverse order.
const fn reverse_bits(b: usize, bits: usize) -> usize {
    if bits == 0 {
        0
    } else {
        b.reverse_bits() >> (usize::BITS as usize - bits)
    }
}

impl<'a, F, EF> DeferredWeight<'a, F, EF>
where
    F: TowerLevel,
    EF: BitCoordinates + ExtensionField<F>,
{
    /// Every claim's head, each scaled by its batching weight, and the sum they claim.
    pub(super) fn new(
        batches: Vec<BitRingSwitchBatch<'a, F, EF>>,
        banks: &[Vec<BitTensor<EF, F>>],
        weights: &[EF],
        sum: EF,
    ) -> Self {
        let heads = batches
            .into_iter()
            .zip(banks)
            .zip(weights)
            .map(|((batch, banks), &weight)| ClaimHead::new(batch, banks, weight))
            .collect();
        Self { heads, sum }
    }
}

impl<F, EF> InitialWeight<EF> for DeferredWeight<'_, F, EF>
where
    F: TowerLevel,
    EF: BitCoordinates + ExtensionField<F>,
{
    fn claimed_sum(&self) -> EF {
        self.sum
    }

    fn round(&mut self, challenges: &[EF]) -> (EF, EF) {
        let mut total = (EF::ZERO, EF::ZERO);
        for head in &mut self.heads {
            head.catch_up(challenges);
            head.message = head.prover.measure_round();
            total.0 += head.message.0;
            total.1 += head.message.1;
        }
        total
    }

    fn into_folded(self, challenges: &[EF]) -> Poly<EF> {
        let mut folded: Option<Poly<EF>> = None;
        for mut head in self.heads {
            head.catch_up(challenges);
            // Every bank coordinate is bound, so one image per coordinate is left.
            let images = head.prover.weights();
            let weights = head
                .batch
                .weights_from_generators::<EF>(&head.head, images.as_slice());
            folded = Some(match folded {
                None => weights,
                Some(mut sum) => {
                    for (sum, &weight) in sum.as_mut_slice().iter_mut().zip(weights.as_slice()) {
                        *sum += weight;
                    }
                    sum
                }
            });
        }
        folded.expect("a batch holds at least one claim")
    }
}

/// What a verifier keeps of a deferred batch: the sum it claims, and its combined weight.
#[derive(Debug)]
pub struct DeferredClosing<'a, F, EF = F> {
    /// The claims the batch reduced.
    claims: &'a BitRingSwitchClaims<F, EF>,
    /// Every claim's second stage.
    batches: Vec<BitRingSwitchBatch<'a, F, EF>>,
    /// `lambda^i`, one per claim.
    weights: Vec<EF>,
    /// `sum_i lambda^i * sigma_i`, derived from the elements' rows.
    sum: EF,
}

impl<'a, F, EF> DeferredClosing<'a, F, EF>
where
    F: TowerLevel + p3_challenger::fs::TranscriptField,
    EF: BitCoordinates + ExtensionField<F>,
{
    /// Keep what the batch fixed for the commitment's run to close.
    pub(super) const fn new(
        claims: &'a BitRingSwitchClaims<F, EF>,
        batches: Vec<BitRingSwitchBatch<'a, F, EF>>,
        weights: Vec<EF>,
        sum: EF,
    ) -> Self {
        Self {
            claims,
            batches,
            weights,
            sum,
        }
    }

    /// The sum the combined weight claims against the packing.
    #[must_use]
    pub const fn claimed_sum(&self) -> EF {
        self.sum
    }

    /// The combined weight's multilinear extension at a point naming every kept coordinate.
    ///
    /// # Errors
    ///
    /// Returns an error unless the point names the packing's variables.
    pub fn weight_at(&self, point: &Point<EF>) -> Result<EF, BitRingSwitchError> {
        self.claims
            .closing_weight(&self.batches, &self.weights, 0, point)
    }
}

/// The trailing coordinates of every claim's element must fit the packing.
pub(super) const fn check_depth(
    depth: usize,
    num_variables: usize,
) -> Result<(), BitRingSwitchError> {
    if depth > num_variables {
        return Err(BitRingSwitchError::DeferredDepth {
            depth,
            num_variables,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_binary_field::{BinaryChallenger, BinaryField16};
    use p3_challenger::{CanSample, HashChallenger};
    use p3_field::PrimeCharacteristicRing;
    use p3_keccak::Keccak256Hash;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::ring_switch::bits::{BitPacking, SuffixTensors};

    type EF = BinaryField16;
    type Chal = BinaryChallenger<EF, HashChallenger<u8, Keccak256Hash, 32>>;

    /// Trailing coordinates every claim splits over.
    const DEPTH: usize = 3;

    fn challenger() -> Chal {
        Chal::from_hasher(Vec::new(), Keccak256Hash)
    }

    /// A random packing, a plain claim and a weighted one over it.
    fn fixture(seed: u64) -> (BitPacking<EF>, BitRingSwitchClaims<EF>) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let witness: Vec<u8> = (0..1 << 10).map(|_| rng.random()).collect();
        let packing = BitPacking::<EF>::new(&witness).unwrap();
        let n = packing.num_variables();
        let absorbed = BitRingSwitch::<EF>::ABSORBED;
        let plain = BitRingSwitch::new(&Point::rand(&mut rng, n + absorbed)).unwrap();
        let weighted = BitRingSwitch::with_low_weights(
            &Point::rand(&mut rng, n),
            &(0..1 << absorbed).map(|_| rng.random()).collect::<Vec<_>>(),
        )
        .unwrap();
        (
            packing,
            BitRingSwitchClaims::new(vec![plain, weighted]).unwrap(),
        )
    }

    /// What each claim reads, from the elements the prover sent.
    fn readings(
        claims: &BitRingSwitchClaims<EF>,
        proof: &BitRingSwitchDeferredProof<EF>,
    ) -> Vec<EF> {
        claims
            .reductions()
            .iter()
            .zip(&proof.claims)
            .map(|(reduction, claim)| reduction.incoming_claim(&claim.tensor))
            .collect()
    }

    #[test]
    fn the_deferred_weight_measures_the_dense_combined_weight() {
        let (packing, claims) = fixture(1);
        let (proof, mut weight) = claims
            .prove_deferred(&packing, &[], DEPTH, &mut challenger())
            .unwrap();
        let mut verifier = challenger();
        let closing = claims
            .verify_deferred(&proof, &readings(&claims, &proof), &mut verifier)
            .unwrap();

        // The dense combined weight, against which every message is a plain sumcheck.
        let mut dense = Poly::<EF>::zero(packing.num_variables());
        for (batch, &scale) in closing.batches.iter().zip(&closing.weights) {
            for (entry, &value) in dense
                .as_mut_slice()
                .iter_mut()
                .zip(batch.weights().as_slice())
            {
                *entry += scale * value;
            }
        }
        let mut evals = Poly::new(packing.poly().as_slice().to_vec());
        let inner = |e: &Poly<EF>, w: &Poly<EF>| -> EF {
            e.as_slice()
                .iter()
                .zip(w.as_slice())
                .map(|(&a, &b)| a * b)
                .sum()
        };
        assert_eq!(closing.claimed_sum(), inner(&evals, &dense));
        assert_eq!(weight.claimed_sum(), closing.claimed_sum());

        // Each round is the dense round, the last variable first.
        let mut rng = SmallRng::seed_from_u64(2);
        let mut challenges = Vec::new();
        let mut folded = dense.clone();
        for _ in 0..DEPTH {
            let message =
                VariableOrder::Suffix.sumcheck_coefficients(evals.as_slice(), folded.as_slice());
            assert_eq!(weight.round(&challenges), (message.c_a, message.c_inf));
            let r: EF = rng.random();
            VariableOrder::Suffix.fix_var(&mut evals, r);
            VariableOrder::Suffix.fix_var(&mut folded, r);
            challenges.push(r);
        }
        assert_eq!(weight.into_folded(&challenges), folded);

        // The closing weight is the dense weight's extension at any point.
        let point = Point::<EF>::rand(&mut rng, packing.num_variables());
        assert_eq!(closing.weight_at(&point).unwrap(), dense.eval_base(&point));
    }

    #[test]
    fn supplied_splits_reproduce_the_computed_batch() {
        let (packing, claims) = fixture(3);
        let splits: Vec<_> = claims
            .reductions()
            .iter()
            .map(|reduction| {
                SuffixTensors::new(reduction.suffix_bank_tensors(&packing, DEPTH)).unwrap()
            })
            .collect();
        let supplied: Vec<_> = splits.iter().map(Some).collect();
        let (mut computed_challenger, mut supplied_challenger) = (challenger(), challenger());
        let (computed, _) = claims
            .prove_deferred(&packing, &[], DEPTH, &mut computed_challenger)
            .unwrap();
        let (read, _) = claims
            .prove_deferred(&packing, &supplied, DEPTH, &mut supplied_challenger)
            .unwrap();
        assert_eq!(read.claims, computed.claims);
        let (a, b): (EF, EF) = (computed_challenger.sample(), supplied_challenger.sample());
        assert_eq!(a, b);

        // Mutation: one bank off by one moves the plain claim's reading, which is refused.
        let mut banks = splits[0].banks().to_vec();
        banks[5] += BitTensor::exterior_product(EF::ONE, EF::ONE);
        let wrong = SuffixTensors::new(banks).unwrap();
        let (proof, _) = claims
            .prove_deferred(&packing, &[Some(&wrong), None], DEPTH, &mut challenger())
            .unwrap();
        let honest = readings(&claims, &computed);
        assert!(
            claims
                .verify_deferred(&proof, &honest, &mut challenger())
                .is_err()
        );

        // A false reading against honest elements is refused too.
        let mut false_readings = honest;
        false_readings[1] += EF::ONE;
        assert!(
            claims
                .verify_deferred(&computed, &false_readings, &mut challenger())
                .is_err()
        );
    }
}
