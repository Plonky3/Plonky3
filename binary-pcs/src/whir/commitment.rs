//! A bit commitment: packed words under WHIR, opened at weighted claims through the deferred ring switch.

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::TowerLevel;
use p3_challenger::fs::TranscriptField;
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs, MmcsArith};
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::ring_switch::bits::{BitRingSwitch, BitTensor, BitTensorArith, SuffixTensors};
use p3_verifier_arith::Lift;
use p3_whir::WhirDomain;
use tracing::instrument;

use crate::BooleanMultilinearPcs;
use crate::fold::BitChallengeField;
use crate::packing::{Coordinates, PackedWords};
use crate::whir::{
    BinaryWhirDomain, BinaryWhirProfile, BooleanWhirData, BooleanWhirDeferredProof, BooleanWhirPcs,
    BooleanWhirProver, CommitmentError, ProfileError, WeightedClaim, recommended_cap_height,
};

/// Bits of one packed word, as a base-two logarithm.
const LOG_WORD_BITS: usize = 6;

/// Bits past the target a single error term may be asked for, before the search gives up.
const MAX_TERM_SLACK: usize = 16;

/// WHIR parameters of the bit commitment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitmentParams {
    /// Bits the composed opening error must reach.
    pub security_bits: usize,
    /// Log inverse rate of the first codeword.
    pub log_inv_rate: usize,
    /// Variables the first WHIR round folds, one Merkle leaf holding `2^first_folding` elements.
    pub first_folding: usize,
    /// Bits the codeword domain loses after the first round.
    pub first_reduction: usize,
    /// Variables each later WHIR round folds.
    pub folding: usize,
    /// Grinding before every query phase.
    pub query_grinding: usize,
    /// Whether the initial commitment takes its own out-of-domain samples.
    ///
    /// Without them the outer protocol's soundness error grows by the list size (ePrint 2026/1329, Appendix C, Remark 11).
    pub commitment_ood: bool,
}

impl Default for CommitmentParams {
    /// 128 composed bits at rate 1/2, list decoding up to the Johnson bound.
    fn default() -> Self {
        Self {
            security_bits: 128,
            log_inv_rate: 1,
            first_folding: 6,
            first_reduction: 4,
            folding: 4,
            query_grinding: 16,
            commitment_ood: false,
        }
    }
}

/// A commitment to `2^n` bits packed into elements of `F`, opened at weighted claims in `EF`.
pub struct BitCommitment<F: Field, EF: ExtensionField<F>, MT, Ch> {
    /// The WHIR commitment to the packed bits.
    inner: BooleanWhirPcs<F, EF, BinaryWhirDomain<F>, MT, Ch>,
    /// The most claims one opening batches, the count the security bound was certified for.
    max_claims: usize,
}

impl<F, EF, MT, Ch> BitCommitment<F, EF, MT, Ch>
where
    F: Field + TranscriptField + TowerLevel + Coordinates + Ord + Send + Sync,
    EF: BitChallengeField<F>,
    BinaryWhirDomain<F>: WhirDomain<F, EF>,
    MT: Mmcs<F>,
    Ch: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>
        + CanObserve<MT::Commitment>,
{
    /// The bit commitment for `2^num_variables` bits, one opening batching `claims` weighted claims.
    ///
    /// Every error term is targeted a little past `security_bits`, so that their union bound reaches it.
    ///
    /// `mmcs` builds the Merkle commitment for a cap height.
    ///
    /// # Errors
    ///
    /// The witness is narrower than one element, the parameters admit no schedule, or no target reaches the bound.
    pub fn new(
        num_variables: usize,
        params: &CommitmentParams,
        claims: usize,
        mmcs: impl Fn(usize) -> MT,
    ) -> Result<Self, CommitmentError> {
        const {
            assert!(
                size_of::<F>().is_multiple_of(size_of::<u64>())
                    && align_of::<F>() >= align_of::<u64>(),
                "the packing level holds whole words"
            );
        };
        for term_bits in params.security_bits..=params.security_bits + MAX_TERM_SLACK {
            // Grinding can cover a whole term, leaving no query; a higher target may still need some.
            let mut commitment = match Self::with_term_bits(num_variables, params, term_bits, &mmcs)
            {
                Err(CommitmentError::Profile(ProfileError::ZeroQueries)) => continue,
                commitment => commitment?,
            };
            commitment.max_claims = claims;
            let composed = commitment
                .inner
                .readings_security(claims, false)
                .map(|security| security.error().bits());
            if composed.is_some_and(|bits| bits >= params.security_bits as f64) {
                return Ok(commitment);
            }
        }
        Err(CommitmentError::Security {
            target: params.security_bits,
        })
    }

    /// The bit commitment with every error term targeted at `term_bits`.
    fn with_term_bits(
        num_variables: usize,
        params: &CommitmentParams,
        term_bits: usize,
        mmcs: impl Fn(usize) -> MT,
    ) -> Result<Self, CommitmentError> {
        let packed = num_variables
            .checked_sub(BitRingSwitch::<F, EF>::ABSORBED)
            .ok_or(CommitmentError::TooFewVariables { num_variables })?;
        let domain = BinaryWhirDomain::<F>::default();
        let profile = BinaryWhirProfile::proven_list_decoding(
            term_bits,
            params.log_inv_rate,
            params.folding.min(packed),
        )
        .with_first_round(params.first_folding.min(packed), params.first_reduction)
        .with_query_grinding(params.query_grinding);
        let config = profile.config::<EF, F, Ch, _>(packed, &domain)?;
        let config = if params.commitment_ood {
            config
        } else {
            config.without_commitment_ood()
        };
        let mmcs = mmcs(recommended_cap_height(&config));
        Ok(Self {
            inner: BooleanWhirPcs::new(
                BooleanWhirProver::new(config, domain, mmcs),
                num_variables,
            )?,
            max_claims: 0,
        })
    }

    /// Variables the committed bits have, so `2^n` bits in all.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.inner.num_variables()
    }

    /// The most claims one opening batches.
    ///
    /// A larger batch would fall below the certified security bound, so it is refused.
    #[must_use]
    pub const fn max_claims(&self) -> usize {
        self.max_claims
    }

    /// Variables the first WHIR fold binds, the packing's trailing ones.
    #[must_use]
    pub fn first_folding(&self) -> usize {
        self.inner.first_folding()
    }

    /// The WHIR commitment the bits are packed under.
    #[must_use]
    pub const fn pcs(&self) -> &BooleanWhirPcs<F, EF, BinaryWhirDomain<F>, MT, Ch> {
        &self.inner
    }

    /// Commit to packed words, taking them over, and observe the commitment.
    ///
    /// Bit `j` of word `w` is bit `64 w + j`.
    ///
    /// # Errors
    ///
    /// The words do not have the committed variables.
    #[allow(clippy::type_complexity)]
    #[instrument(skip_all, name = "bit commit")]
    pub fn commit(
        &self,
        words: Vec<u64>,
        challenger: &mut Ch,
    ) -> Result<(MT::Commitment, BooleanWhirData<F, EF, MT>), CommitmentError> {
        let expected = 1usize << (self.num_variables() - LOG_WORD_BITS);
        if words.len() != expected {
            return Err(CommitmentError::WitnessLength {
                expected,
                actual: words.len(),
            });
        }
        let packed = PackedWords::<F>::from_word_vec(words).expect("the level holds whole words");
        Ok(self.inner.commit_packed(packed, challenger)?)
    }

    /// The words a commitment holds.
    ///
    /// # Panics
    ///
    /// Never for data this commitment made.
    #[must_use]
    pub fn committed_words(data: &BooleanWhirData<F, EF, MT>) -> &[u64] {
        PackedWords::<F>::words_of(data.table(0).poly(0).into_slice())
            .expect("the level holds whole words")
    }

    /// Observe a commitment on the verifier side, as committing does on the prover side.
    pub fn observe(&self, commitment: &MT::Commitment, challenger: &mut Ch) {
        self.inner.observe_commitment(commitment, challenger);
    }

    /// As [`Self::observe`], within an arithmetic, `target` being the commitment there.
    pub fn observe_in<A, C>(
        &self,
        arith: &mut A,
        commitment: &<MT as Mmcs<F>>::Commitment,
        target: &<MT as MmcsArith<A, A::F>>::Target,
        challenger: &mut C,
    ) where
        A: Lift<Val = F, Challenge = EF>,
        C: CanObserve<F> + CanObserve<<MT as Mmcs<F>>::Commitment>,
        MT: MmcsArith<A, A::F>,
    {
        self.inner
            .observe_commitment_in(arith, commitment, target, challenger);
    }

    /// The Merkle scheme every round commits with.
    #[must_use]
    pub const fn mmcs(&self) -> &MT {
        self.inner.mmcs()
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
        let depth = self.first_folding();
        let inner = high.get(log_outer..)?;
        let head = inner.len().checked_sub(depth)?;
        if folded.len() != width << inner.len() {
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

    /// Open the committed bits at every claim, in one proof.
    ///
    /// The ring switch's sum is proven by WHIR's first fold, which binds no separate point.
    ///
    /// Entry `i` of `splits`, when present, is claim `i`'s element split by `suffix_tensors`.
    ///
    /// # Errors
    ///
    /// A claim of the wrong width, a claim the bits do not satisfy, or a refused opening.
    #[instrument(skip_all, name = "bit open")]
    pub fn open(
        &self,
        data: BooleanWhirData<F, EF, MT>,
        claims: &[WeightedClaim<EF>],
        splits: &[Option<&SuffixTensors<EF, F>>],
        challenger: &mut Ch,
    ) -> Result<BooleanWhirDeferredProof<F, EF, MT>, CommitmentError> {
        self.check_arity(claims)?;
        let openings = claims
            .iter()
            .map(|claim| claim.opening.clone())
            .collect::<Vec<_>>();
        let (values, proof) = self
            .inner
            .open_weighted_deferred(data, &openings, splits, challenger)?;
        if let Some(index) = values
            .iter()
            .zip(claims)
            .position(|(&value, claim)| value != claim.value)
        {
            return Err(CommitmentError::FalseClaim { index });
        }
        Ok(proof)
    }

    /// Check one opening proof against every claim.
    ///
    /// # Errors
    ///
    /// A claim of the wrong width, or any refusal of the ring switch or WHIR.
    #[instrument(skip_all, name = "bit verify")]
    pub fn verify(
        &self,
        commitment: &MT::Commitment,
        claims: &[WeightedClaim<EF>],
        proof: &BooleanWhirDeferredProof<F, EF, MT>,
        challenger: &mut Ch,
    ) -> Result<(), CommitmentError> {
        self.check_arity(claims)?;
        Ok(self
            .inner
            .verify_weighted_deferred(commitment, claims, proof, challenger)?)
    }

    /// As [`Self::verify`], within an arithmetic.
    ///
    /// # Errors
    ///
    /// As [`Self::verify`].
    #[instrument(skip_all, name = "bit verify")]
    pub fn verify_in<'s, A, C>(
        &'s self,
        arith: &mut A,
        commitment: &<MT as MmcsArith<A, A::F>>::Target,
        claims: &[WeightedClaim<A::EF>],
        proof: &BooleanWhirDeferredProof<F, EF, MT>,
        challenger: &mut C,
    ) -> Result<(), CommitmentError>
    where
        A: BitTensorArith + Lift<Val = F, Challenge = EF>,
        C: FieldChallenger<F>
            + GrindingChallenger<Witness = F>
            + CanSampleUniformBits<F>
            + CanObserve<<MT as Mmcs<F>>::Commitment>,
        MT: MmcsArith<
                A,
                A::F,
                Commitment = <MT as Mmcs<F>>::Commitment,
                MultiProof = <MT as Mmcs<F>>::MultiProof,
            >,
        ExtensionMmcs<F, EF, &'s MT>: MmcsArith<
                A,
                A::EF,
                MultiProof = <MT as Mmcs<F>>::MultiProof,
                Target = <MT as MmcsArith<A, A::F>>::Target,
            >,
    {
        self.check_arity(claims)?;
        Ok(self
            .inner
            .verify_weighted_deferred_in(arith, commitment, claims, proof, challenger)?)
    }

    /// Refuse more claims than were certified, or a point that does not name the kept coordinates, before the transcript moves.
    fn check_arity<E>(&self, claims: &[WeightedClaim<E>]) -> Result<(), CommitmentError> {
        if claims.len() > self.max_claims {
            return Err(CommitmentError::TooManyClaims {
                maximum: self.max_claims,
                actual: claims.len(),
            });
        }
        let expected = self.num_variables() - BitRingSwitch::<F, EF>::ABSORBED;
        claims
            .iter()
            .map(|claim| claim.opening.high.num_variables())
            .find(|&actual| actual != expected)
            .map_or(Ok(()), |actual| {
                Err(CommitmentError::PointArity { expected, actual })
            })
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_binary_field::{Poly64, Poly192, fold_rows};
    use p3_field::PrimeCharacteristicRing;
    use p3_keccak::Keccak256Hash;
    use p3_multilinear_util::point::Point;
    use p3_sumcheck::ring_switch::bits::BitPacking;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, RngExt, SeedableRng};

    use super::*;
    use crate::whir::preset::keccak64::{self, Challenger, OpeningProof, WitnessCommitment};
    use crate::whir::{Region, RegionLayout, WeightedOpening};

    /// Variables of the test commitment.
    const N: usize = 14;

    /// A fresh sponge.
    fn sponge() -> Challenger {
        Challenger::from_hasher(Vec::new(), Keccak256Hash)
    }

    /// `sum_v low_weights[v] * t(high, v)`, straight from the words.
    fn reference(words: &[u64], opening: &WeightedOpening<Poly192>) -> Poly192 {
        let eq = Poly::new_from_point(opening.high.as_slice(), Poly192::ONE);
        words
            .iter()
            .zip(eq.as_slice())
            .map(|(&word, &e)| {
                let bits: Poly192 = (0..64)
                    .filter(|v| word >> v & 1 == 1)
                    .map(|v| opening.low_weights[v])
                    .sum();
                e * bits
            })
            .sum()
    }

    /// A random point of `m` coordinates.
    fn point(rng: &mut SmallRng, m: usize) -> Point<Poly192> {
        Point::new((0..m).map(|_| rng.random()).collect())
    }

    /// The claim kinds a shared commitment batches.
    #[derive(Clone, Copy, Debug)]
    enum Kind {
        /// Free weights on the absorbed bits, as a univariate-skip claim has.
        Quirky,
        /// A multilinear evaluation over every variable.
        Mle,
        /// A multilinear evaluation on one region.
        Region(usize),
    }

    /// The opening of one claim of `kind`.
    fn opening(rng: &mut SmallRng, kind: Kind, layout: &RegionLayout) -> WeightedOpening<Poly192> {
        match kind {
            Kind::Quirky => WeightedOpening {
                high: point(rng, N - 6),
                low_weights: (0..64).map(|_| rng.random()).collect(),
            },
            Kind::Mle => WeightedOpening::at_point(&point(rng, N), 6),
            Kind::Region(i) => {
                let region = layout.regions()[i];
                WeightedOpening::at_point(&point(rng, region.log_bits), 6)
                    .within(&region.selector(layout.num_variables))
            }
        }
    }

    /// Commit to random words and open them at one claim per kind.
    struct Run {
        commitment: WitnessCommitment,
        root: keccak64::Commitment,
        claims: Vec<WeightedClaim<Poly192>>,
        proof: OpeningProof,
    }

    fn run(seed: u64, kinds: &[Kind]) -> Run {
        let mut rng = SmallRng::seed_from_u64(seed);
        let layout = RegionLayout::allocate(&[N - 1, N - 2, N - 2]);
        assert_eq!(layout.num_variables, N);
        let words: Vec<u64> = (0..1 << (N - 6)).map(|_| rng.next_u64()).collect();
        let commitment =
            keccak64::witness_commitment(N, &CommitmentParams::default(), kinds.len()).unwrap();
        let claims: Vec<_> = kinds
            .iter()
            .map(|&kind| {
                let opening = opening(&mut rng, kind, &layout);
                let value = reference(&words, &opening);
                WeightedClaim { opening, value }
            })
            .collect();
        let mut challenger = sponge();
        let (root, data) = commitment.commit(words, &mut challenger).unwrap();
        let proof = commitment
            .open(data, &claims, &[], &mut challenger)
            .unwrap();
        Run {
            commitment,
            root,
            claims,
            proof,
        }
    }

    impl Run {
        fn verify(
            &self,
            claims: &[WeightedClaim<Poly192>],
            proof: &OpeningProof,
        ) -> Result<(), CommitmentError> {
            let mut challenger = sponge();
            self.commitment.observe(&self.root, &mut challenger);
            self.commitment
                .verify(&self.root, claims, proof, &mut challenger)
        }
    }

    /// Up to four claims, every mix of kinds.
    const BATCHES: [&[Kind]; 6] = [
        &[Kind::Mle],
        &[Kind::Quirky],
        &[Kind::Region(1)],
        &[Kind::Quirky, Kind::Region(0)],
        &[Kind::Mle, Kind::Quirky, Kind::Region(2)],
        &[Kind::Region(0), Kind::Mle, Kind::Region(1), Kind::Quirky],
    ];

    #[test]
    fn mixed_claims_on_one_commitment_verify() {
        for (seed, kinds) in BATCHES.iter().enumerate() {
            let r = run(seed as u64, kinds);
            r.verify(&r.claims, &r.proof)
                .unwrap_or_else(|e| panic!("{kinds:?}: {e:?}"));
        }
    }

    #[test]
    fn a_region_claim_is_the_claim_on_the_region_words() {
        let mut rng = SmallRng::seed_from_u64(7);
        let layout = RegionLayout::allocate(&[8, 10, 9]);
        let words: Vec<u64> = (0..1 << (layout.num_variables - 6))
            .map(|_| rng.next_u64())
            .collect();
        for region in layout.regions() {
            let local = WeightedOpening::at_point(&point(&mut rng, region.log_bits), 6);
            let lifted = local.clone().within(&region.selector(layout.num_variables));
            assert_eq!(
                reference(&words, &lifted),
                reference(region.words(&words), &local)
            );
        }
    }

    #[test]
    fn a_false_claim_or_a_tampered_proof_is_refused() {
        let r = run(9, BATCHES[5]);
        r.verify(&r.claims, &r.proof).unwrap();

        // Mutation: each claim's value, then one of its kept coordinates.
        for index in 0..r.claims.len() {
            let mut claims = r.claims.clone();
            claims[index].value += Poly192::ONE;
            assert!(r.verify(&claims, &r.proof).is_err(), "value {index}");
            let mut claims = r.claims.clone();
            let mut high = claims[index].opening.high.as_slice().to_vec();
            high[0] += Poly192::ONE;
            claims[index].opening.high = Point::new(high);
            assert!(r.verify(&claims, &r.proof).is_err(), "point {index}");
        }

        // Mutation: a region claim checked against another region's selector.
        let mut claims = r.claims.clone();
        let mut high = claims[0].opening.high.as_slice().to_vec();
        high[0] = Poly192::ONE - high[0];
        claims[0].opening.high = Point::new(high);
        assert!(r.verify(&claims, &r.proof).is_err(), "selector");

        // Mutation: a claim dropped, or claims swapped.
        assert!(r.verify(&r.claims[1..], &r.proof).is_err(), "dropped");
        let mut claims = r.claims.clone();
        claims.swap(0, 1);
        assert!(r.verify(&claims, &r.proof).is_err(), "swapped");

        // Mutation: a tensor row, an out-of-domain answer, the first fold, the final polynomial.
        let mut proof = r.proof.clone();
        let mut rows: Vec<Poly64> = proof.reduction.claims[0].tensor.rows().to_vec();
        rows[7] += Poly64::ONE;
        proof.reduction.claims[0].tensor = rows.try_into().unwrap();
        assert!(r.verify(&r.claims, &proof).is_err(), "tensor row");
        assert!(r.proof.opening.initial_ood_answers.is_empty());
        let mut proof = r.proof.clone();
        proof.opening.initial_ood_answers.push(Poly192::ONE);
        assert!(r.verify(&r.claims, &proof).is_err(), "out-of-domain answer");
        let mut proof = r.proof.clone();
        proof.opening.initial_sumcheck.polynomial_evaluations[0][0] += Poly192::ONE;
        assert!(r.verify(&r.claims, &proof).is_err(), "first fold");
        let mut proof = r.proof.clone();
        let mut last = proof.opening.final_poly.clone().unwrap().into_evals();
        last[0] += Poly192::ONE;
        proof.opening.final_poly = Some(Poly::new(last));
        assert!(r.verify(&r.claims, &proof).is_err(), "final polynomial");

        // Mutation: another commitment.
        let other = run(10, BATCHES[5]);
        let mut challenger = sponge();
        r.commitment.observe(&other.root, &mut challenger);
        assert!(
            r.commitment
                .verify(&other.root, &r.claims, &r.proof, &mut challenger)
                .is_err(),
            "commitment"
        );

        // A claim of the wrong width is refused before the transcript moves.
        let mut claims = r.claims.clone();
        let mut high = claims[1].opening.high.as_slice().to_vec();
        high.pop();
        claims[1].opening.high = Point::new(high);
        assert!(matches!(
            r.verify(&claims, &r.proof),
            Err(CommitmentError::PointArity {
                expected: 8,
                actual: 7
            })
        ));
    }

    #[test]
    fn an_opening_past_the_certified_claim_count_is_refused() {
        // Certified for 2 claims: a batch of 3 would compose below the bound, so neither side accepts it.
        let r = run(11, &[Kind::Mle, Kind::Mle]);
        assert_eq!(r.commitment.max_claims(), 2);
        let mut claims = r.claims.clone();
        claims.push(claims[0].clone());
        assert!(matches!(
            r.verify(&claims, &r.proof),
            Err(CommitmentError::TooManyClaims {
                maximum: 2,
                actual: 3
            })
        ));
        let mut rng = SmallRng::seed_from_u64(11);
        let words: Vec<u64> = (0..1 << (N - 6)).map(|_| rng.next_u64()).collect();
        let mut challenger = sponge();
        let (_, data) = r.commitment.commit(words, &mut challenger).unwrap();
        assert!(matches!(
            r.commitment.open(data, &claims, &[], &mut challenger),
            Err(CommitmentError::TooManyClaims {
                maximum: 2,
                actual: 3
            })
        ));
    }

    #[test]
    fn the_prover_refuses_a_false_claim_or_a_witness_of_another_width() {
        let mut rng = SmallRng::seed_from_u64(4);
        let words: Vec<u64> = (0..1 << (N - 6)).map(|_| rng.next_u64()).collect();
        let commitment = keccak64::witness_commitment(N, &CommitmentParams::default(), 1).unwrap();
        let mut challenger = sponge();
        let (_, data) = commitment.commit(words.clone(), &mut challenger).unwrap();
        assert_eq!(WitnessCommitment::committed_words(&data), words.as_slice());
        let opening = WeightedOpening::at_point(&point(&mut rng, N), 6);
        let value = reference(&words, &opening) + Poly192::ONE;
        assert!(matches!(
            commitment.open(
                data.clone(),
                &[WeightedClaim { opening, value }],
                &[],
                &mut challenger.clone()
            ),
            Err(CommitmentError::FalseClaim { index: 0 })
        ));
        let narrow = WeightedOpening::at_point(&point(&mut rng, N - 1), 6);
        assert!(matches!(
            commitment.open(
                data,
                &[WeightedClaim {
                    opening: narrow,
                    value: Poly192::ZERO
                }],
                &[],
                &mut challenger
            ),
            Err(CommitmentError::PointArity { .. })
        ));
        assert!(matches!(
            commitment.commit(words[..128].to_vec(), &mut sponge()),
            Err(CommitmentError::WitnessLength {
                expected: 256,
                actual: 128
            })
        ));
        assert!(matches!(
            keccak64::witness_commitment(5, &CommitmentParams::default(), 1),
            Err(CommitmentError::TooFewVariables { num_variables: 5 })
        ));
    }

    #[test]
    fn a_split_from_the_folded_rows_reproduces_the_computed_opening() {
        // Invariant: the split a row fold yields is the one the commitment would compute.
        let mut rng = SmallRng::seed_from_u64(5);
        let (log_outer, n) = (4, 18);
        let words: Vec<u64> = (0..1 << (n - 6)).map(|_| rng.next_u64()).collect();
        let commitment = keccak64::witness_commitment(n, &CommitmentParams::default(), 1).unwrap();
        let mut challenger = sponge();
        let (_, data) = commitment.commit(words.clone(), &mut challenger).unwrap();
        let at = point(&mut rng, n);
        let opening = WeightedOpening::at_point(&at, 6);
        let weights = Poly::new_from_point(&at.as_slice()[..log_outer], Poly192::ONE);
        let folded = fold_rows(
            &words,
            n - log_outer,
            1 << (n - log_outer),
            weights.as_slice(),
        );
        let split = commitment.suffix_tensors(&folded, opening.high.as_slice(), log_outer);
        assert!(split.is_some());
        // A fold of another width, or fewer inner coordinates than the first fold binds, has no split.
        assert!(
            commitment
                .suffix_tensors(&folded[1..], opening.high.as_slice(), log_outer)
                .is_none()
        );
        assert!(
            commitment
                .suffix_tensors(&folded, opening.high.as_slice(), n)
                .is_none()
        );

        let value = reference(&words, &opening);
        let claims = [WeightedClaim { opening, value }];
        let computed = commitment
            .open(data.clone(), &claims, &[], &mut challenger.clone())
            .unwrap();
        let read = commitment
            .open(data, &claims, &[split.as_ref()], &mut challenger)
            .unwrap();
        // The elements and the first fold come before any grinding, whose witness a parallel search may pick freely.
        assert_eq!(read.reduction.claims, computed.reduction.claims);
        assert_eq!(
            read.opening.initial_sumcheck.polynomial_evaluations,
            computed.opening.initial_sumcheck.polynomial_evaluations
        );
    }

    #[test]
    fn the_default_commitment_composes_to_its_target() {
        // Invariant: the union of every opening error term reaches the requested bits at every size and batch.
        for num_variables in [12, 14, 20, 26, 32] {
            for claims in 1..=4 {
                let commitment = keccak64::witness_commitment(
                    num_variables,
                    &CommitmentParams::default(),
                    claims,
                )
                .unwrap();
                let bits = commitment
                    .pcs()
                    .readings_security(claims, false)
                    .unwrap()
                    .error()
                    .bits();
                assert!(
                    bits >= 128.0,
                    "{num_variables} variables, {claims} claims compose to {bits} bits"
                );
            }
        }
    }

    #[test]
    fn a_target_grinding_covers_whole_moves_on_to_the_next() {
        // Grinding 16 bits leaves the 16-bit target no query, while 17 bits composes to the bound.
        let params = CommitmentParams {
            security_bits: 16,
            query_grinding: 16,
            ..CommitmentParams::default()
        };
        let mmcs = |cap_height| {
            keccak64::Mmcs::new(
                keccak64::Hash::new(Keccak256Hash),
                keccak64::Compress::new(Keccak256Hash),
                cap_height,
            )
        };
        assert!(matches!(
            WitnessCommitment::with_term_bits(12, &params, 16, mmcs),
            Err(CommitmentError::Profile(ProfileError::ZeroQueries))
        ));
        let commitment = keccak64::witness_commitment(12, &params, 1).unwrap();
        let bits = commitment
            .pcs()
            .readings_security(1, false)
            .unwrap()
            .error()
            .bits();
        assert!(bits >= 16.0, "composes to {bits} bits");
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(16))]

        #[test]
        fn at_point_is_the_ring_switch_of_the_point(seed: u64, m in 6usize..12) {
            // Invariant: the weighted claim at a point is the point's own reduction, element and value.
            let mut rng = SmallRng::seed_from_u64(seed);
            let elements: Vec<Poly64> = (0..1 << (m - 6)).map(|_| Poly64::new(rng.next_u64())).collect();
            let packing = BitPacking::from_packed(Poly::new(elements)).unwrap();
            let at = point(&mut rng, m);
            let opening = WeightedOpening::at_point(&at, 6);
            let weighted = BitRingSwitch::<Poly64, Poly192>::with_low_weights(&opening.high, &opening.low_weights).unwrap();
            let direct = BitRingSwitch::<Poly64, Poly192>::new(&at).unwrap();
            let tensor = weighted.tensor(&packing).unwrap();
            prop_assert_eq!(&tensor, &direct.tensor(&packing).unwrap());
            prop_assert_eq!(weighted.incoming_claim(&tensor), direct.incoming_claim(&tensor));
        }

        #[test]
        fn a_region_lift_selects_only_the_region(seed: u64, slot in 0usize..4) {
            // Invariant: a claim lifted into region `slot` reads nothing outside it.
            let mut rng = SmallRng::seed_from_u64(seed);
            let region = Region { log_bits: 8, offset_bits: slot << 8 };
            let local = WeightedOpening::at_point(&point(&mut rng, 8), 6);
            let lifted = local.clone().within(&region.selector(10));
            let mut words: Vec<u64> = (0..16).map(|_| rng.next_u64()).collect();
            let value = reference(&words, &lifted);
            // Every word outside the region flips, and the value stays.
            for (w, word) in words.iter_mut().enumerate() {
                if w / 4 != slot {
                    *word = !*word;
                }
            }
            prop_assert_eq!(reference(&words, &lifted), value);
            prop_assert_eq!(reference(region.words(&words), &local), value);
        }
    }
}
