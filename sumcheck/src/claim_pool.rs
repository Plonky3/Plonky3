//! One pool of claims about one committed polynomial, and the single draw that batches them.
//!
//! ```text
//!     deposit   every reduction leaves its surviving claim here
//!     seal      bind all of them, then draw one challenge
//!     fold      weights are that challenge's distinct powers
//! ```
//!
//! # Why the order is a type and not a comment
//!
//! The challenge must be drawn after every claim value is bound.
//! Drawing it first lets a prover pick its claims knowing the weights.
//!
//! One linear equation in the claim values is then always solvable.
//!
//! Three things make that order unforgeable here:
//!
//! - the challenge lives only on the sealed pool, which sealing alone produces
//! - sealing consumes the open pool, so no claim can be added once it exists
//! - an open pool cannot be duplicated, so sealing a copy cannot reveal the draw early
//!
//! The described transcript says the same thing a second way.
//! The draw is the last step, after both message steps.
//!
//! A driver that reordered them would fail the description rather than prove anything.
//!
//! # One opening, not one per claim
//!
//! A commitment that takes several evaluation claims discharges the whole pool in one call.
//! The fold then closes it: one equality over the opened values, instead of `k`.
//!
//! No sumcheck runs for the batching itself.
//!
//! # Soundness
//!
//! The fold costs `(k - 1) / |EF|`, priced under its own label in `p3-security`.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptField,
};
use p3_challenger::{CanObserve, CanSample};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use thiserror::Error;

/// Version byte bound into the seed, separating two revisions of this protocol.
const VERSION: u8 = 1;

/// Protocol name bound into the seed.
const NAME: &[u8] = b"p3-sumcheck-claim-pool";

/// Step label of the points the claims are stated at.
const CLAIM_POINTS: &str = "claim_points";

/// Step label of the values the claims carry.
const CLAIM_VALUES: &str = "claim_values";

/// Step label of the one challenge that batches the pool.
const BATCHING_CHALLENGE: &str = "batching_challenge";

/// Sponge alphabet of a challenger that speaks the claim field natively.
type Alphabet<EF> = FieldUnit<EF>;

/// Numbers that fix the transcript of one pool.
///
/// Both sides build this from the public schedule, never from a proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClaimPoolShape {
    /// Variables the committed polynomial has, so the width of one claim's point.
    pub num_variables: usize,
    /// Claims the pool holds.
    pub num_claims: usize,
}

impl ClaimPoolShape {
    /// Describe a pool of this many claims about a polynomial of this arity.
    #[must_use]
    pub const fn new(num_variables: usize, num_claims: usize) -> Self {
        Self {
            num_variables,
            num_claims,
        }
    }

    /// Describe every step of one pool.
    ///
    /// The draw is last, so no description admits a challenge before the claims.
    ///
    /// # Panics
    ///
    /// Never: a flat sequence of leaf steps always validates.
    #[must_use]
    pub fn pattern<EF: TranscriptField>(&self) -> InteractionPattern {
        let steps = alloc::vec![
            Interaction::algebra::<EF, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                CLAIM_POINTS,
                Length::Fixed(self.num_claims * self.num_variables),
            ),
            Interaction::algebra::<EF, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                CLAIM_VALUES,
                Length::Fixed(self.num_claims),
            ),
            Interaction::algebra::<EF, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                BATCHING_CHALLENGE,
                Length::Scalar,
            ),
        ];

        InteractionPattern::new(steps).expect("a flat sequence of leaf steps is well formed")
    }

    /// Bind the protocol identity and this shape into a seed.
    #[must_use]
    pub fn domain_separator<EF: TranscriptField>(&self) -> DomainSeparator<Alphabet<EF>> {
        DomainSeparator::new(VERSION, NAME, self.pattern::<EF>())
    }
}

/// Claims about one committed polynomial, waiting for the draw that batches them.
///
/// Deliberately not duplicable: a copy could be sealed to learn the draw early.
#[derive(Debug, PartialEq, Eq)]
pub struct ClaimPool<EF> {
    /// Variables the committed polynomial has.
    num_variables: usize,
    /// Every deposited claim, in deposit order.
    claims: Vec<(Point<EF>, EF)>,
}

impl<EF: TranscriptField> ClaimPool<EF> {
    /// An empty pool over a polynomial of this arity.
    #[must_use]
    pub const fn new(num_variables: usize) -> Self {
        Self {
            num_variables,
            claims: Vec::new(),
        }
    }

    /// Deposit one claim, returning the index the fold will weigh it at.
    ///
    /// # Errors
    ///
    /// Returns an error unless the point names the committed polynomial's variables.
    pub fn deposit(&mut self, point: Point<EF>, value: EF) -> Result<usize, ClaimPoolError> {
        if point.num_variables() != self.num_variables {
            return Err(ClaimPoolError::PointWidth {
                expected: self.num_variables,
                actual: point.num_variables(),
            });
        }
        self.claims.push((point, value));
        Ok(self.claims.len() - 1)
    }

    /// Claims the pool holds.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.claims.len()
    }

    /// Whether the pool holds no claim.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.claims.is_empty()
    }

    /// Variables the committed polynomial has.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Every deposited claim, in deposit order.
    #[must_use]
    pub fn claims(&self) -> &[(Point<EF>, EF)] {
        &self.claims
    }

    /// The shape this pool's transcript is described with.
    #[must_use]
    pub const fn shape(&self) -> ClaimPoolShape {
        ClaimPoolShape::new(self.num_variables, self.claims.len())
    }

    /// Bind every claim, then draw the one challenge that batches them.
    ///
    /// The pool is consumed, so nothing can be deposited once the challenge exists.
    ///
    /// # Panics
    ///
    /// Never: the described widths are computed from this pool's own contents.
    #[must_use]
    pub fn seal<C>(self, challenger: &mut C) -> SealedClaims<EF>
    where
        C: CanObserve<EF> + CanSample<EF>,
    {
        let shape = self.shape();
        let mut state = ProverState::new(challenger, &shape.domain_separator::<EF>());

        // Every coordinate of every point, in deposit order, then every value.
        let points: Vec<EF> = self
            .claims
            .iter()
            .flat_map(|(point, _)| point.as_slice().iter().copied())
            .collect();
        let values: Vec<EF> = self.claims.iter().map(|&(_, value)| value).collect();
        let _ = state.observe_extensions::<EF, EF, FieldToFieldCodec<EF>>(CLAIM_POINTS, &points);
        let _ = state.observe_extensions::<EF, EF, FieldToFieldCodec<EF>>(CLAIM_VALUES, &values);

        // Only now is the challenge drawn, so no claim above it knew it.
        let alpha = state
            .challenge_extension::<EF, EF, FieldToFieldCodec<EF>>(BATCHING_CHALLENGE)
            .into_inner();
        assert!(
            state.finalize().is_empty(),
            "a claim pool carries every value in its caller's own proof",
        );

        SealedClaims {
            claims: self.claims,
            num_variables: self.num_variables,
            alpha,
            _field: PhantomData,
        }
    }
}

/// A pool whose batching challenge has been drawn.
///
/// The only way to reach this type is to seal an open pool, which consumes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SealedClaims<EF> {
    /// Every claim, in deposit order, which is the order the powers weigh them in.
    claims: Vec<(Point<EF>, EF)>,
    /// Variables the committed polynomial has.
    num_variables: usize,
    /// The one challenge whose powers are the weights.
    alpha: EF,
    /// Marker keeping the field on the type even when no claim was deposited.
    _field: PhantomData<EF>,
}

impl<EF: TranscriptField> SealedClaims<EF> {
    /// The challenge whose distinct powers weigh the claims.
    #[must_use]
    pub const fn challenge(&self) -> EF {
        self.alpha
    }

    /// Claims the pool held.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.claims.len()
    }

    /// Whether the pool held no claim.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.claims.is_empty()
    }

    /// Every claim, in the order the powers weigh them.
    #[must_use]
    pub fn claims(&self) -> &[(Point<EF>, EF)] {
        &self.claims
    }

    /// The weights, which are the challenge's distinct powers in deposit order.
    #[must_use]
    pub fn weights(&self) -> Vec<EF> {
        self.alpha.powers().take(self.claims.len()).collect()
    }

    /// The batched claim value, `sum_i alpha^i * value_i`.
    #[must_use]
    pub fn folded_value(&self) -> EF {
        self.fold(self.claims.iter().map(|&(_, value)| value))
    }

    /// The same fold applied to values a caller holds, in the pool's own order.
    ///
    /// A verifier closes the pool by folding the values a commitment opened and comparing.
    ///
    /// # Panics
    ///
    /// Panics if the iterator does not yield one value per claim.
    #[must_use]
    pub fn fold(&self, values: impl IntoIterator<Item = EF>) -> EF {
        let mut folded = EF::ZERO;
        let mut weights = self.alpha.powers();
        let mut count = 0;
        for value in values {
            folded += weights.next().expect("powers never run out") * value;
            count += 1;
        }
        assert_eq!(count, self.claims.len(), "one value per claim");
        folded
    }

    /// The batched weight multilinear, `sum_i alpha^i * eq(point_i, .)`.
    ///
    /// This is what a sumcheck over the committed polynomial runs against.
    /// A caller handing the pool to a commitment instead never builds it.
    pub fn weight_poly(&self) -> Poly<EF> {
        let mut table = Poly::zero(self.num_variables);
        for (&weight, (point, _)) in self.weights().iter().zip(&self.claims) {
            let term = Poly::new_from_point(point.as_slice(), weight);
            for (slot, value) in table.as_mut_slice().iter_mut().zip(term.as_slice()) {
                *slot += *value;
            }
        }
        table
    }

    /// The batched weight at one point, without building the table.
    #[must_use]
    pub fn weight_at(&self, x: &Point<EF>) -> EF {
        self.weights()
            .iter()
            .zip(&self.claims)
            .map(|(&weight, (point, _))| weight * Point::eval_eq(point.as_slice(), x.as_slice()))
            .sum()
    }
}

/// Why a claim could not join a pool.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum ClaimPoolError {
    /// The claim's point does not name the committed polynomial's variables.
    #[error("a claim's point names {actual} variables, expected {expected}")]
    PointWidth {
        /// Variables the committed polynomial has.
        expected: usize,
        /// Variables the point names.
        actual: usize,
    },
}

#[cfg(test)]
mod tests {
    use p3_binary_field::{BinaryChallenger, BinaryField128};
    use p3_challenger::HashChallenger;
    use p3_field::PrimeCharacteristicRing;
    use p3_keccak::Keccak256Hash;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type EF = BinaryField128;
    type Chal = BinaryChallenger<EF, HashChallenger<u8, Keccak256Hash, 32>>;

    fn challenger() -> Chal {
        Chal::from_hasher(Vec::new(), Keccak256Hash)
    }

    /// A pool of random claims about a polynomial of the given arity.
    fn pool(seed: u64, num_variables: usize, num_claims: usize) -> ClaimPool<EF> {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut pool = ClaimPool::new(num_variables);
        for _ in 0..num_claims {
            let point = Point::<EF>::rand(&mut rng, num_variables);
            pool.deposit(point, rng.random()).unwrap();
        }
        pool
    }

    /// The same claims in the same order, in a pool of their own.
    fn rebuilt(claims: &[(Point<EF>, EF)], num_variables: usize) -> ClaimPool<EF> {
        let mut pool = ClaimPool::new(num_variables);
        for (point, value) in claims {
            pool.deposit(point.clone(), *value).unwrap();
        }
        pool
    }

    #[test]
    fn the_weights_are_the_distinct_powers_of_one_challenge() {
        // Invariant: weight `i` is `alpha^i`, and no two of them coincide.
        //
        //     - claim 0  ->  1
        //     - claim 1  ->  alpha
        //     - claim 2  ->  alpha^2
        //
        // Repeating a weight would let two claims trade values undetected.
        let sealed = pool(0xA1FA, 4, 5).seal(&mut challenger());
        let alpha = sealed.challenge();
        let weights = sealed.weights();

        assert_eq!(weights.len(), 5);
        let mut expected = EF::ONE;
        for (index, &weight) in weights.iter().enumerate() {
            assert_eq!(weight, expected, "weight {index}");
            expected *= alpha;
        }
        for (i, &a) in weights.iter().enumerate() {
            for &b in &weights[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn the_fold_is_the_weighted_sum_of_the_claim_values() {
        // The folded value is what a verifier compares one opened fold against.
        let sealed = pool(0xF01D, 3, 4).seal(&mut challenger());

        let expected: EF = sealed
            .weights()
            .iter()
            .zip(sealed.claims())
            .map(|(&weight, &(_, value))| weight * value)
            .sum();
        assert_eq!(sealed.folded_value(), expected);

        // Folding the same values through the public entry agrees.
        let values: Vec<EF> = sealed.claims().iter().map(|&(_, value)| value).collect();
        assert_eq!(sealed.fold(values), expected);
    }

    #[test]
    fn the_challenge_moves_with_every_claim_the_pool_holds() {
        // Invariant: the draw is a function of every deposited point and value.
        //
        // Mutation: move one value, leaving the points alone.
        //
        //     baseline  -> one challenge
        //     perturbed -> another
        //
        // A draw that ignored the values would let a prover choose them afterwards.
        let claims = pool(0xD1FF, 3, 4).claims().to_vec();
        let baseline = rebuilt(&claims, 3).seal(&mut challenger()).challenge();

        let mut moved = claims.clone();
        moved[2].1 += EF::ONE;
        assert_ne!(
            rebuilt(&moved, 3).seal(&mut challenger()).challenge(),
            baseline
        );

        // A moved point moves it too.
        let mut shifted = claims;
        let mut coords = shifted[1].0.as_slice().to_vec();
        coords[0] += EF::ONE;
        shifted[1].0 = Point::new(coords);
        assert_ne!(
            rebuilt(&shifted, 3).seal(&mut challenger()).challenge(),
            baseline
        );
    }

    #[test]
    fn the_weight_multilinear_agrees_with_its_pointwise_reading() {
        // Invariant: the table and the closed form are the same polynomial.
        //
        //     table  -> sum_i alpha^i * eq(point_i, .), built over the hypercube
        //     point  -> the same sum, evaluated at one place
        //
        // A sumcheck runs against the table and a verifier reads the closed form.
        // A disagreement between them would split the two sides.
        let mut rng = SmallRng::seed_from_u64(0x3A81);
        let sealed = pool(0x3A80, 5, 3).seal(&mut challenger());

        let table = sealed.weight_poly();
        assert_eq!(table.num_variables(), 5);

        for _ in 0..8 {
            let x = Point::<EF>::rand(&mut rng, 5);
            assert_eq!(table.eval_base(&x), sealed.weight_at(&x));
        }

        // A pool of one claim weighs nothing, so its table is that claim.s equality alone.
        let single = pool(0x3A82, 5, 1).seal(&mut challenger());
        let (point, value) = &single.claims()[0];
        assert_eq!(single.folded_value(), *value);
        let x = Point::<EF>::rand(&mut rng, 5);
        assert_eq!(
            single.weight_at(&x),
            Point::eval_eq(point.as_slice(), x.as_slice())
        );
    }

    #[test]
    fn a_claim_of_the_wrong_width_never_joins_the_pool() {
        // The pool's arity is the committed polynomial's, so a mismatch is an input error.
        let mut pool = ClaimPool::<EF>::new(4);
        let narrow = Point::<EF>::rand(&mut SmallRng::seed_from_u64(1), 3);

        assert_eq!(
            pool.deposit(narrow, EF::ONE).unwrap_err(),
            ClaimPoolError::PointWidth {
                expected: 4,
                actual: 3
            }
        );
        assert!(pool.is_empty());
    }

    proptest! {
        /// The fold closes on every pool size, and the weight readings stay in step.
        #[test]
        fn the_fold_and_the_weights_agree_over_random_pools(
            num_variables in 1usize..=6,
            num_claims in 1usize..=8,
            seed: u64,
        ) {
            let sealed = pool(seed, num_variables, num_claims).seal(&mut challenger());

            let values: Vec<EF> = sealed.claims().iter().map(|&(_, v)| v).collect();
            prop_assert_eq!(sealed.fold(values), sealed.folded_value());

            let table = sealed.weight_poly();
            let x = Point::<EF>::rand(&mut SmallRng::seed_from_u64(!seed), num_variables);
            prop_assert_eq!(table.eval_base(&x), sealed.weight_at(&x));
        }
    }
}
