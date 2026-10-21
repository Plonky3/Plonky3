use alloc::vec;
use alloc::vec::Vec;

use p3_baby_bear::BabyBear;
use p3_binary_pcs::whir::preset::keccak64;
use p3_binary_pcs::whir::{CommitmentParams, Region, RegionLayout, WeightedClaim};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_gkr::ProductGkrLayerProof;
use p3_goldilocks::Goldilocks;
use p3_keccak::Keccak256Hash;
use p3_koala_bear::KoalaBear;
use p3_mersenne_31::Mersenne31;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::*;

/// A fresh sponge.
fn sponge() -> Challenger {
    Challenger::from_hasher(Vec::new(), Keccak256Hash)
}

/// A uniform-enough element.
fn element<F: PrimeField64>(rng: &mut SmallRng) -> F {
    F::from_u64(rng.next_u64())
}

/// Random weights of the given kinds, with the value the words give them.
fn claim<F: PrimeField64>(
    rng: &mut SmallRng,
    shape: BitShape,
    words: &[u64],
    tensor: bool,
    equality: bool,
) -> TensorClaim<F> {
    let rows = if tensor {
        let slots = 1 << (shape.log_rows() / 2);
        RowWeights::Tensor {
            outer: (0..shape.rows() / slots).map(|_| element(rng)).collect(),
            slots: (0..slots).map(|_| element(rng)).collect(),
        }
    } else {
        RowWeights::Dense((0..shape.rows()).map(|_| element(rng)).collect())
    };
    let columns = if equality {
        ColumnWeights::Equality((0..shape.log_columns()).map(|_| element(rng)).collect())
    } else {
        ColumnWeights::Dense((0..shape.columns()).map(|_| element(rng)).collect())
    };
    let mut claim = TensorClaim {
        rows,
        columns,
        value: F::ZERO,
    };
    // The value straight from the bits.
    let u1: Vec<F> = claim.lifted_rows().into_iter().map(F::from_u64).collect();
    let u2 = claim.column_weights();
    claim.value = (0..shape.rows() * shape.columns())
        .filter(|&i| words[i / 64] >> (i % 64) & 1 == 1)
        .map(|i| u1[i % shape.rows()] * u2[i / shape.rows()])
        .sum();
    claim
}

/// A committed shape, its claims and a proof of them.
struct Run<F> {
    pcs: BitzPcs<F>,
    root: Commitment,
    claims: Vec<TensorClaim<F>>,
    proof: BitzProof,
}

impl<F: PrimeField64> Run<F> {
    fn new(log_rows: usize, log_columns: usize, count: usize, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let shape = BitShape::new(log_rows, log_columns).unwrap();
        let words: Vec<u64> = (0..shape.words()).map(|_| rng.next_u64()).collect();
        let claims: Vec<_> = (0..count)
            .map(|i| claim(&mut rng, shape, &words, i % 2 == 1, i % 3 != 2))
            .collect();
        let pcs = BitzPcs::<F>::new(shape, &BitzParams::default()).unwrap();
        let mut challenger = sponge();
        let (root, data) = pcs.commit(words, &mut challenger).unwrap();
        let proof = pcs.prove(data, &claims, &mut challenger).unwrap();
        Self {
            pcs,
            root,
            claims,
            proof,
        }
    }

    fn verify(
        &self,
        claims: &[TensorClaim<F>],
        proof: &BitzProof,
    ) -> Result<(), VerificationError> {
        self.pcs.verify(&self.root, claims, proof, &mut sponge())
    }

    fn verify_reduction(&self, reduction: BitzReduction) -> Result<(), VerificationError> {
        let proof = BitzProof {
            reduction,
            opening: self.proof.opening.clone(),
        };
        self.verify(&self.claims, &proof)
    }
}

/// Every field, a shape whose first fold binds only rows, one with too few rows for the split, and no columns.
fn completeness<F: PrimeField64>() {
    for (seed, (log_rows, log_columns, count)) in [(12, 3, 1), (8, 4, 3), (9, 0, 1), (7, 5, 2)]
        .into_iter()
        .enumerate()
    {
        let run = Run::<F>::new(log_rows, log_columns, count, seed as u64);
        run.verify(&run.claims, &run.proof)
            .unwrap_or_else(|e| panic!("({log_rows}, {log_columns}, {count}): {e:?}"));
    }
}

#[test]
fn claims_over_babybear_verify() {
    completeness::<BabyBear>();
}

#[test]
fn claims_over_koalabear_verify() {
    completeness::<KoalaBear>();
}

#[test]
fn claims_over_goldilocks_verify() {
    completeness::<Goldilocks>();
}

#[test]
fn claims_over_mersenne31_verify() {
    completeness::<Mersenne31>();
}

#[test]
fn the_default_parameters_reach_128_bits() {
    for n in [12, 20, 28] {
        let pcs =
            BitzPcs::<Goldilocks>::new(BitShape::balanced(n).unwrap(), &BitzParams::default())
                .unwrap();
        assert!(pcs.security_bits(1) >= 128.0);
    }
    // A target no schedule reaches is refused.
    let params = BitzParams {
        commitment: CommitmentParams {
            security_bits: 400,
            ..CommitmentParams::default()
        },
        inner_pow_bits: 0,
    };
    assert!(BitzPcs::<Goldilocks>::new(BitShape::new(12, 2).unwrap(), &params).is_err());
}

#[test]
fn the_balanced_shape_follows_the_paper_split() {
    for (n, r1) in [(20, 11), (24, 14), (26, 15), (28, 16), (6, 6), (8, 6)] {
        let shape = BitShape::balanced(n).unwrap();
        assert_eq!((shape.log_rows(), shape.num_variables()), (r1, n));
    }
    assert!(BitShape::balanced(5).is_err());
    assert!(BitShape::new(5, 3).is_err());
    assert!(BitShape::new(32, 0).is_err());
}

#[test]
fn every_gate_rejects_its_tampering() {
    let run = Run::<KoalaBear>::new(12, 3, 2, 7);
    run.verify(&run.claims, &run.proof).unwrap();
    let bound = run.pcs.bitz().bound;
    let count = run.claims.len() * run.pcs.bitz().shape().columns();
    let folds = bound.decode(&run.proof.reduction.folds, count).unwrap();

    // A fold out of range: its first fold's bits all set.
    let mut reduction = run.proof.reduction.clone();
    reduction.folds[0] |= (1 << bound.width) - 1;
    assert!(matches!(
        run.verify_reduction(reduction),
        Err(VerificationError::FoldRange { index: 0 })
    ));

    // A wrong fold with the right projection passes the sum but not the exponent: the replayed product no longer closes.
    let index = folds
        .iter()
        .position(|&f| f + u128::from(KoalaBear::ORDER_U64) <= bound.max)
        .unwrap();
    let mut shifted = folds.clone();
    shifted[index] += u128::from(KoalaBear::ORDER_U64);
    let mut reduction = run.proof.reduction.clone();
    reduction.folds = bound.encode(&shifted);
    assert!(matches!(
        run.verify_reduction(reduction),
        Err(VerificationError::Gkr(_) | VerificationError::RootMismatch { .. })
    ));

    // A wrong fold that changes the projection fails the sum.
    let mut changed = folds;
    changed[0] ^= 1;
    let mut reduction = run.proof.reduction.clone();
    reduction.folds = bound.encode(&changed);
    assert!(matches!(
        run.verify_reduction(reduction),
        Err(VerificationError::ClaimSum { index: 0 })
    ));

    // A false claim value.
    let mut claims = run.claims.clone();
    claims[1].value += KoalaBear::ONE;
    assert!(matches!(
        run.verify(&claims, &run.proof),
        Err(VerificationError::ClaimSum { index: 1 })
    ));

    // A grand-product round, child, or root.
    let mut reduction = run.proof.reduction.clone();
    if let ProductGkrLayerProof::RadixFour { round_polys, .. } = &mut reduction.gkr.layers[1] {
        round_polys[0][0] += Challenge::ONE;
    }
    assert!(matches!(
        run.verify_reduction(reduction),
        Err(VerificationError::Gkr(_))
    ));
    let mut reduction = run.proof.reduction.clone();
    if let ProductGkrLayerProof::RadixFour { children, .. } =
        reduction.gkr.layers.last_mut().unwrap()
    {
        children[0][1] += Challenge::ONE;
    }
    assert!(matches!(
        run.verify_reduction(reduction),
        Err(VerificationError::Gkr(_) | VerificationError::InnerEval)
    ));
    let mut reduction = run.proof.reduction.clone();
    reduction.gkr.roots[3] += Challenge::ONE;
    assert!(run.verify_reduction(reduction).is_err());

    // An inner-product round, or the bit evaluation.
    let mut reduction = run.proof.reduction.clone();
    reduction.inner.sumcheck.polynomial_evaluations[2][1] += Challenge::ONE;
    assert!(matches!(
        run.verify_reduction(reduction),
        Err(VerificationError::InnerEval)
    ));
    let mut reduction = run.proof.reduction.clone();
    reduction.inner.eval += Challenge::ONE;
    assert!(matches!(
        run.verify_reduction(reduction),
        Err(VerificationError::InnerEval)
    ));

    // The opening, and another commitment.
    let mut proof = run.proof.clone();
    proof
        .opening
        .opening
        .initial_sumcheck
        .polynomial_evaluations[0][0] += Challenge::ONE;
    assert!(matches!(
        run.verify(&run.claims, &proof),
        Err(VerificationError::Commitment(_))
    ));
    let other = Run::<KoalaBear>::new(12, 3, 2, 8);
    assert!(
        run.pcs
            .verify(&other.root, &run.claims, &run.proof, &mut sponge())
            .is_err()
    );

    // Claims of the wrong shape are refused before the transcript moves.
    assert!(matches!(
        run.verify(&[], &run.proof),
        Err(VerificationError::Claim(ClaimError::NoClaims))
    ));
    let mut claims = run.claims.clone();
    claims[0].columns = ColumnWeights::Dense(vec![KoalaBear::ONE; 3]);
    assert!(matches!(
        run.verify(&claims, &run.proof),
        Err(VerificationError::Claim(ClaimError::ColumnLength {
            index: 0,
            ..
        }))
    ));
}

#[test]
fn the_prover_refuses_a_false_claim() {
    let mut rng = SmallRng::seed_from_u64(3);
    let shape = BitShape::new(8, 2).unwrap();
    let words: Vec<u64> = (0..shape.words()).map(|_| rng.next_u64()).collect();
    let mut claims = vec![claim::<BabyBear>(&mut rng, shape, &words, false, true)];
    claims[0].value += BabyBear::ONE;
    let pcs = BitzPcs::<BabyBear>::new(shape, &BitzParams::default()).unwrap();
    let mut challenger = sponge();
    let (_, data) = pcs.commit(words, &mut challenger).unwrap();
    assert!(matches!(
        pcs.prove(data, &claims, &mut challenger),
        Err(ProvingError::FalseClaim { index: 0 })
    ));
}

#[test]
fn a_claim_on_one_region_of_a_shared_commitment_verifies_there_only() {
    let mut rng = SmallRng::seed_from_u64(11);
    let layout = RegionLayout::allocate(&[12, 12]);
    let region = layout.regions()[1];
    let shape = BitShape::new(9, 3).unwrap();
    let words: Vec<u64> = (0..1 << (layout.num_variables - 6))
        .map(|_| rng.next_u64())
        .collect();
    let claims = vec![claim::<Goldilocks>(
        &mut rng,
        shape,
        region.words(&words),
        true,
        true,
    )];
    let bitz = Bitz::<Goldilocks>::new(shape, 0).unwrap();
    let commitment =
        keccak64::witness_commitment(layout.num_variables, &CommitmentParams::default(), 1)
            .unwrap();
    let mut prover = sponge();
    let (root, data) = commitment.commit(words.clone(), &mut prover).unwrap();
    let (reduction, lifted) = bitz
        .reduce(
            &claims,
            ClaimBinding::Derived,
            &commitment,
            region,
            &data,
            &mut prover,
        )
        .unwrap();
    let opening = commitment
        .open(
            data,
            core::slice::from_ref(&lifted.claim),
            &[lifted.split.as_ref()],
            &mut prover,
        )
        .unwrap();

    let verify = |region: Region, binding: ClaimBinding| -> bool {
        let mut verifier = sponge();
        commitment.observe(&root, &mut verifier);
        bitz.verify_reduction(
            &claims,
            binding,
            &commitment,
            region,
            &reduction,
            &mut verifier,
        )
        .is_ok_and(|claim: WeightedClaim<Challenge>| {
            commitment
                .verify(&root, &[claim], &opening, &mut verifier)
                .is_ok()
        })
    };
    assert!(verify(region, ClaimBinding::Derived));
    // The other region's prefix, or another binding, is refused.
    assert!(!verify(layout.regions()[0], ClaimBinding::Derived));
    assert!(!verify(region, ClaimBinding::Public));
    // A region of another size does not hold the shape.
    let mut verifier = sponge();
    let wide = Region {
        log_bits: 13,
        offset_bits: 0,
    };
    assert!(matches!(
        bitz.verify_reduction(
            &claims,
            ClaimBinding::Derived,
            &commitment,
            wide,
            &reduction,
            &mut verifier
        ),
        Err(VerificationError::Claim(ClaimError::Region(_)))
    ));
}

#[test]
fn proofs_round_trip_through_postcard() {
    let run = Run::<Mersenne31>::new(12, 2, 2, 5);
    let bytes = postcard::to_allocvec(&run.proof).unwrap();
    let back: BitzProof = postcard::from_bytes(&bytes).unwrap();
    run.verify(&run.claims, &back).unwrap();
    assert_eq!(postcard::to_allocvec(&back).unwrap(), bytes);
}
