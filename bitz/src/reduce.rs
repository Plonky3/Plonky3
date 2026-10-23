//! The reduction of linear claims on committed bits to one claim on the commitment (Construction 4.4, Round 2 and Phase 3).
//!
//! Round 2 sends the integer folds, Phase 3 proves each fold in the exponent of `g` through a batched grand product, and `Pi_ip` ties its leaves to the bits.

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_binary_pcs::whir::preset::keccak64::{ProverData, WitnessCommitment};
use p3_binary_pcs::whir::{Region, RegionClaim, WeightedClaim, WeightedOpening};
use p3_field::{Field, FixedBase, PrimeCharacteristicRing, PrimeField64};
use p3_gkr::{ProductGkrProof, ProductGkrShape};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_security::SecurityTerm;
use p3_security::bitz::BitzSecurityModel;
use p3_sumcheck::ring_switch::bits::BitRingSwitch;
use p3_sumcheck::strategy::Basis;
use tracing::{info_span, instrument};

use crate::claim::{ClaimBinding, TensorClaim, digest};
use crate::error::{ClaimError, ParamsError, ProvingError, RegionError, VerificationError};
use crate::fold::{FoldBound, fold, project};
use crate::leaves::BitLeaves;
use crate::params::BitShape;
use crate::proof::BitzReduction;
use crate::transcript::ReductionShape;
use crate::{Challenge, Challenger, Val, inner};

/// Bits of the exponents the generator table covers, the width of `ord(g) = 2^192 - 1`.
const EXPONENT_BITS: u32 = 192;

/// The reduction for one shape and one target field `F`.
#[derive(Clone, Debug)]
pub struct Bitz<F> {
    /// The split of the bits.
    shape: BitShape,
    /// Grinding before every inner-product challenge.
    inner_pow_bits: usize,
    /// The range and encoding of the folds.
    pub(crate) bound: FoldBound,
    /// Powers of `g = Poly192::GENERATOR`.
    generator: FixedBase<Challenge>,
    /// The target field.
    _field: PhantomData<F>,
}

impl<F: PrimeField64> Bitz<F> {
    /// The reduction for a shape over `F`.
    ///
    /// # Errors
    ///
    /// A fold could reach `ord(g)` (Section 4.4).
    pub fn new(shape: BitShape, inner_pow_bits: usize) -> Result<Self, ParamsError> {
        Ok(Self {
            shape,
            inner_pow_bits,
            bound: FoldBound::new::<F>(shape)?,
            generator: FixedBase::new(Challenge::GENERATOR, EXPONENT_BITS),
            _field: PhantomData,
        })
    }

    /// The split of the bits.
    #[must_use]
    pub const fn shape(&self) -> BitShape {
        self.shape
    }

    /// Every algebraic error term of reducing `claims` claims, before the opening's.
    #[must_use]
    pub fn security_terms(&self, claims: usize) -> Vec<SecurityTerm> {
        BitzSecurityModel::new(
            Challenge::bits(),
            self.shape.log_rows(),
            self.inner_pow_bits,
            self.gkr_shape(claims).security_profile(),
        )
        .components()
    }

    /// Prove the claims on a region of a shared commitment, leaving one claim on the commitment to open.
    ///
    /// The commitment is observed before; the returned claim is opened after, batched with any other.
    ///
    /// # Errors
    ///
    /// A claim or the region does not fit the shape, or the committed bits do not satisfy a claim.
    #[instrument(skip_all, name = "bitz reduce", fields(n = self.shape.num_variables(), claims = claims.len()))]
    pub fn reduce(
        &self,
        claims: &[TensorClaim<F>],
        binding: ClaimBinding,
        commitment: &WitnessCommitment,
        region: Region,
        data: &ProverData,
        challenger: &mut Challenger,
    ) -> Result<(BitzReduction, RegionClaim<Val, Challenge>), ProvingError> {
        self.check(claims, commitment.num_variables(), region)?;
        let words = region.words(WitnessCommitment::committed_words(data));
        let (r1, r2) = (self.shape.log_rows(), self.shape.log_columns());

        // Round 2: the folds over the integers, checked against every claim before they are sent.
        let gammas: Vec<Vec<u64>> = claims.iter().map(TensorClaim::lifted_rows).collect();
        let folds: Vec<u128> = info_span!("bitz folds").in_scope(|| {
            gammas
                .iter()
                .flat_map(|gamma| fold(words, r1, gamma, self.bound.pieces()))
                .collect()
        });
        for (index, (claim, folds)) in claims
            .iter()
            .zip(folds.chunks_exact(self.shape.columns()))
            .enumerate()
        {
            if column_sum(claim, folds) != claim.value {
                return Err(ProvingError::FalseClaim { index });
            }
        }
        let encoded = self.bound.encode(&folds);
        let transcript = self.transcript(claims, binding, commitment.num_variables(), region);
        transcript.observe_folds(&encoded, challenger, true);

        // Phase 3: every product of leaves is g to its fold.
        let images: Vec<Vec<Challenge>> = info_span!("bitz row images")
            .in_scope(|| gammas.iter().map(|gamma| self.images(gamma)).collect());
        let shape = self.gkr_shape(claims.len());
        let leaves = BitLeaves {
            words,
            shape: self.shape,
            images: &images,
        };
        let (gkr, out) = info_span!("bitz gkr")
            .in_scope(|| ProductGkrProof::prove_batched::<Val, _, _>(&leaves, shape, challenger));

        // Pi_ip: the leaf claim at (rho_t, rho_c, rho_b) on the bits at (rho_c, rho').
        let log_trees = shape.batch_variables() - r2;
        let (tree_point, rest) = out.point.split_at(log_trees);
        let (column_point, row_point) = rest.split_at(r2);
        let (inner, rho, folded) = info_span!("bitz inner").in_scope(|| {
            let weights = inner::row_weights(&images, tree_point, row_point);
            inner::prove(
                self.shape,
                words,
                column_point,
                weights,
                out.values[0] - Challenge::ONE,
                self.inner_pow_bits,
                challenger,
            )
        });
        transcript.observe_eval(inner.eval, challenger, true);

        let mut point = column_point.to_vec();
        point.extend_from_slice(rho.as_slice());
        let opening = WeightedOpening::at_point(
            &Point::new(point),
            BitRingSwitch::<Val, Challenge>::ABSORBED,
        );
        // The selector's eq is one on the region, so the region's fold is the commitment's.
        let split = commitment.suffix_tensors(&folded, opening.high.as_slice(), r2);
        let claim = WeightedClaim {
            opening: opening.within(&region.selector(commitment.num_variables())),
            value: inner.eval,
        };
        Ok((
            BitzReduction {
                folds: encoded,
                gkr,
                inner,
            },
            RegionClaim { claim, split },
        ))
    }

    /// Replay the reduction of the claims on a region, returning the claim the caller must still open.
    ///
    /// # Errors
    ///
    /// A claim or the region does not fit the shape, or any check of Round 2, the grand product or `Pi_ip` fails.
    #[instrument(skip_all, name = "bitz verify reduction", fields(n = self.shape.num_variables(), claims = claims.len()))]
    pub fn verify_reduction(
        &self,
        claims: &[TensorClaim<F>],
        binding: ClaimBinding,
        commitment: &WitnessCommitment,
        region: Region,
        proof: &BitzReduction,
        challenger: &mut Challenger,
    ) -> Result<WeightedClaim<Challenge>, VerificationError> {
        self.check(claims, commitment.num_variables(), region)?;
        let (r1, r2) = (self.shape.log_rows(), self.shape.log_columns());

        // Round 2: every fold in range, and the folds of each claim sum to its value (eq. 38).
        let folds = self
            .bound
            .decode(&proof.folds, claims.len() * self.shape.columns())?;
        for (index, (claim, folds)) in claims
            .iter()
            .zip(folds.chunks_exact(self.shape.columns()))
            .enumerate()
        {
            if column_sum(claim, folds) != claim.value {
                return Err(VerificationError::ClaimSum { index });
            }
        }
        let transcript = self.transcript(claims, binding, commitment.num_variables(), region);
        transcript.observe_folds(&proof.folds, challenger, false);

        // Phase 3: the roots are g to the folds, and the padding trees' roots are one.
        let shape = self.gkr_shape(claims.len());
        let out = proof.gkr.verify::<Val, _>(shape, challenger)?;
        let exponents: Vec<[u64; 2]> = folds
            .iter()
            .map(|&fold| [fold as u64, (fold >> 64) as u64])
            .collect();
        let mut roots = vec![Challenge::ONE; out.roots.len()];
        self.generator
            .pow_into::<Val, 2>(&exponents, &mut roots[..exponents.len()]);
        if let Some(index) = roots.iter().zip(&out.roots).position(|(a, b)| a != b) {
            return Err(VerificationError::RootMismatch { index });
        }

        // Pi_ip: replay the rounds, then close them on W0~(rho') e.
        let images: Vec<Vec<Challenge>> = claims
            .iter()
            .map(|claim| self.images(&claim.lifted_rows()))
            .collect();
        let log_trees = shape.batch_variables() - r2;
        let (tree_point, rest) = out.point.split_at(log_trees);
        let (column_point, row_point) = rest.split_at(r2);
        let mut sum = out.values[0] - Challenge::ONE;
        let rho = proof.inner.sumcheck.verify_rounds(
            challenger,
            &mut sum,
            r1,
            self.inner_pow_bits,
            Basis::Evaluation,
        )?;
        let weights = inner::row_weights(&images, tree_point, row_point);
        let at_rho = Poly::new(weights).eval_base(&rho);
        if sum != at_rho * proof.inner.eval {
            return Err(VerificationError::InnerEval);
        }
        transcript.observe_eval(proof.inner.eval, challenger, false);

        let mut point = column_point.to_vec();
        point.extend_from_slice(rho.as_slice());
        let opening = WeightedOpening::at_point(
            &Point::new(point),
            BitRingSwitch::<Val, Challenge>::ABSORBED,
        );
        Ok(WeightedClaim {
            opening: opening.within(&region.selector(commitment.num_variables())),
            value: proof.inner.eval,
        })
    }

    /// Refuse claims or a region that do not fit the shape, before the transcript moves.
    fn check(
        &self,
        claims: &[TensorClaim<F>],
        num_variables: usize,
        region: Region,
    ) -> Result<(), ClaimError> {
        if claims.is_empty() {
            return Err(ClaimError::NoClaims);
        }
        for (index, claim) in claims.iter().enumerate() {
            claim.check(index, self.shape)?;
        }
        if region.log_bits != self.shape.num_variables() || !region.lies_within(num_variables) {
            return Err(ClaimError::Region(RegionError {
                region,
                log_bits: self.shape.num_variables(),
                num_variables,
            }));
        }
        Ok(())
    }

    /// The batched product of `claims` claims: trees `(t, c)`, at least one batch coordinate.
    fn gkr_shape(&self, claims: usize) -> ProductGkrShape {
        let log_trees = (claims.next_power_of_two().trailing_zeros() as usize)
            .max(usize::from(self.shape.log_columns() == 0));
        ProductGkrShape::batched(self.shape.log_rows(), log_trees + self.shape.log_columns())
            .expect("a checked shape is addressable")
    }

    /// `g^gamma` for every lifted weight.
    fn images(&self, gamma: &[u64]) -> Vec<Challenge> {
        let exponents: Vec<[u64; 1]> = gamma.iter().map(|&g| [g]).collect();
        let mut out = Challenge::zero_vec(gamma.len());
        self.generator.pow_into::<Val, 1>(&exponents, &mut out);
        out
    }

    /// The transcript shape of these claims on this region.
    fn transcript(
        &self,
        claims: &[TensorClaim<F>],
        binding: ClaimBinding,
        num_variables: usize,
        region: Region,
    ) -> ReductionShape {
        ReductionShape {
            log_rows: self.shape.log_rows(),
            log_columns: self.shape.log_columns(),
            claims: claims.len(),
            inner_pow_bits: self.inner_pow_bits,
            order: F::ORDER_U64,
            binding: binding.tag(),
            digest: (binding == ClaimBinding::Public).then(|| digest(claims)),
            num_variables,
            offset_bits: region.offset_bits,
            fold_words: self.bound.words(claims.len() * self.shape.columns()),
        }
    }
}

/// `sum_c pi(mu_c) u2[c]`, a claim's value as its folds give it (Construction 4.4, eq. 38).
fn column_sum<F: PrimeField64>(claim: &TensorClaim<F>, folds: &[u128]) -> F {
    folds
        .iter()
        .zip(claim.column_weights())
        .map(|(&fold, weight)| project::<F>(fold) * weight)
        .sum()
}
