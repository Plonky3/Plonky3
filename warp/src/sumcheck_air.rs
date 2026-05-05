//! AIR gadgets for WARP's coefficient-form sumcheck verifier.
//!
//! This module arithmetizes the pure recurrence used by
//! [`verify_sumcheck`](crate::sumcheck::verify_sumcheck):
//!
//! ```text
//! h_j(0) + h_j(1) = claim_j
//! claim_{j+1} = h_j(r_j)
//! ```
//!
//! The Fiat-Shamir derivation of `r_j` is intentionally not encoded here.
//! This AIR is the arithmetic core that a transcript/verifier AIR should feed.

use alloc::vec::Vec;
use core::array;

#[cfg(feature = "stark-backend")]
use openvm_stark_backend::{
    AirRef, PartitionedBaseAir, StarkProtocolConfig,
    prover::{AirProvingContext, ColMajorMatrix, CpuColMajorBackend, ProvingContext},
};
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::extension::{BinomialExtensionField, BinomiallyExtendable};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;

use crate::sumcheck::SumcheckProof;

/// AIR for a fixed-round, fixed-degree coefficient-form sumcheck verifier.
#[derive(Clone, Debug)]
pub struct BinomialSumcheckAir<F, const D: usize> {
    num_rounds: usize,
    degree: usize,
    extension_w: F,
}

impl<F, const D: usize> BinomialSumcheckAir<F, D>
where
    F: Copy,
{
    /// Create an AIR using the binomial extension relation `X^D = extension_w`.
    pub fn new(num_rounds: usize, degree: usize, extension_w: F) -> Self {
        assert!(D > 1, "extension degree must be greater than one");
        assert!(num_rounds > 0, "sumcheck AIR needs at least one round");
        Self {
            num_rounds,
            degree,
            extension_w,
        }
    }

    /// Create an AIR for Plonky3's canonical binomial extension of `F`.
    pub fn for_binomial_extension(num_rounds: usize, degree: usize) -> Self
    where
        F: BinomiallyExtendable<D>,
    {
        Self::new(num_rounds, degree, F::W)
    }

    pub const fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    pub const fn degree(&self) -> usize {
        self.degree
    }

    const fn active_offset(&self) -> usize {
        0
    }

    const fn selectors_offset(&self) -> usize {
        1
    }

    const fn claim_in_offset(&self) -> usize {
        1 + self.num_rounds
    }

    const fn claim_out_offset(&self) -> usize {
        self.claim_in_offset() + D
    }

    const fn challenge_offset(&self) -> usize {
        self.claim_out_offset() + D
    }

    const fn coeffs_offset(&self) -> usize {
        self.challenge_offset() + D
    }

    fn row_width(&self) -> usize {
        1 + self.num_rounds + D * (self.degree + 4)
    }
}

impl<F, const D: usize> BaseAir<F> for BinomialSumcheckAir<F, D>
where
    F: Copy + Sync,
{
    fn width(&self) -> usize {
        self.row_width()
    }

    fn num_public_values(&self) -> usize {
        D * (self.num_rounds + 2)
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        Some((self.degree + 2).max(2))
    }
}

#[cfg(feature = "stark-backend")]
impl<F, const D: usize> PartitionedBaseAir<F> for BinomialSumcheckAir<F, D> where F: Copy + Sync {}

impl<AB, F, const D: usize> Air<AB> for BinomialSumcheckAir<F, D>
where
    AB: AirBuilder<F = F>,
    F: Field,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice().to_vec();
        let next = main.next_slice().to_vec();
        let public_values = builder.public_values().to_vec();

        let active: AB::Expr = local[self.active_offset()].into();
        let next_active: AB::Expr = next[self.active_offset()].into();
        builder.assert_bool(active.clone());

        let selectors = local[self.selectors_offset()..self.selectors_offset() + self.num_rounds]
            .iter()
            .copied()
            .map(Into::into)
            .collect::<Vec<AB::Expr>>();
        let next_selectors = next
            [self.selectors_offset()..self.selectors_offset() + self.num_rounds]
            .iter()
            .copied()
            .map(Into::into)
            .collect::<Vec<AB::Expr>>();
        for selector in selectors.iter().cloned() {
            builder.assert_bool(selector);
        }
        let selector_sum = selectors
            .iter()
            .cloned()
            .fold(AB::Expr::ZERO, |acc, selector| acc + selector);
        builder.assert_eq(selector_sum, active.clone());
        builder.when_first_row().assert_one(selectors[0].clone());
        builder.when_first_row().assert_one(active.clone());
        if self.num_rounds.is_power_of_two() {
            builder
                .when_last_row()
                .assert_one(selectors[self.num_rounds - 1].clone());
        } else {
            builder.when_last_row().assert_zero(active.clone());
        }
        let transition_active = active.clone() - selectors[self.num_rounds - 1].clone();
        builder
            .when_transition()
            .assert_eq(next_active, transition_active.clone());
        builder
            .when_transition()
            .assert_zero(next_selectors[0].clone());
        for i in 1..self.num_rounds {
            builder
                .when_transition()
                .assert_eq(next_selectors[i].clone(), selectors[i - 1].clone());
        }

        let claim_in = read_ext::<AB, D>(&local, self.claim_in_offset());
        let claim_out = read_ext::<AB, D>(&local, self.claim_out_offset());
        let challenge = read_ext::<AB, D>(&local, self.challenge_offset());
        let coeffs = (0..=self.degree)
            .map(|i| read_ext::<AB, D>(&local, self.coeffs_offset() + i * D))
            .collect::<Vec<_>>();
        let expected_challenge =
            select_public_challenge::<AB, D>(&selectors, &public_values, 2 * D);
        assert_ext_eq_when::<AB, D>(
            builder,
            active.clone(),
            challenge.clone(),
            expected_challenge,
        );

        let h_0 = coeffs[0].clone();
        let h_1 = coeffs
            .iter()
            .cloned()
            .fold(ext_zero::<AB, D>(), ext_add::<AB, D>);
        assert_ext_eq_when::<AB, D>(
            builder,
            active.clone(),
            ext_add::<AB, D>(h_0, h_1),
            claim_in.clone(),
        );

        let h_r = coeffs
            .iter()
            .rev()
            .cloned()
            .fold(ext_zero::<AB, D>(), |acc, coeff| {
                ext_add::<AB, D>(
                    ext_mul::<AB, F, D>(acc, challenge.clone(), self.extension_w),
                    coeff,
                )
            });
        assert_ext_eq_when::<AB, D>(builder, active.clone(), h_r, claim_out.clone());

        let public_initial = read_public_ext::<AB, D>(&public_values, 0);
        let public_final = read_public_ext::<AB, D>(&public_values, D);
        let next_claim_in = read_ext::<AB, D>(&next, self.claim_in_offset());
        builder
            .when_first_row()
            .assert_zeros(ext_sub::<AB, D>(claim_in.clone(), public_initial));
        builder
            .when_transition()
            .when(transition_active)
            .assert_zeros(ext_sub::<AB, D>(next_claim_in, claim_out.clone()));
        builder
            .when(selectors[self.num_rounds - 1].clone())
            .assert_zeros(ext_sub::<AB, D>(claim_out, public_final));
    }
}

fn select_public_challenge<AB, const D: usize>(
    selectors: &[AB::Expr],
    public_values: &[AB::PublicVar],
    offset: usize,
) -> [AB::Expr; D]
where
    AB: AirBuilder,
{
    array::from_fn(|limb| {
        selectors
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (round, selector)| {
                acc + selector.clone() * public_values[offset + round * D + limb].into()
            })
    })
}

fn read_ext<AB, const D: usize>(row: &[AB::Var], offset: usize) -> [AB::Expr; D]
where
    AB: AirBuilder,
{
    array::from_fn(|i| row[offset + i].into())
}

fn read_public_ext<AB, const D: usize>(
    public_values: &[AB::PublicVar],
    offset: usize,
) -> [AB::Expr; D]
where
    AB: AirBuilder,
{
    array::from_fn(|i| public_values[offset + i].into())
}

fn ext_zero<AB, const D: usize>() -> [AB::Expr; D]
where
    AB: AirBuilder,
{
    array::from_fn(|_| AB::Expr::ZERO)
}

fn ext_add<AB, const D: usize>(lhs: [AB::Expr; D], rhs: [AB::Expr; D]) -> [AB::Expr; D]
where
    AB: AirBuilder,
{
    array::from_fn(|i| lhs[i].clone() + rhs[i].clone())
}

fn ext_sub<AB, const D: usize>(lhs: [AB::Expr; D], rhs: [AB::Expr; D]) -> [AB::Expr; D]
where
    AB: AirBuilder,
{
    array::from_fn(|i| lhs[i].clone() - rhs[i].clone())
}

fn ext_mul<AB, F, const D: usize>(
    lhs: [AB::Expr; D],
    rhs: [AB::Expr; D],
    extension_w: F,
) -> [AB::Expr; D]
where
    AB: AirBuilder<F = F>,
    F: Field,
{
    let mut out = ext_zero::<AB, D>();
    for i in 0..D {
        for j in 0..D {
            let target = (i + j) % D;
            let mut term = lhs[i].clone() * rhs[j].clone();
            if i + j >= D {
                term *= extension_w;
            }
            out[target] += term;
        }
    }
    out
}

fn assert_ext_eq_when<AB, const D: usize>(
    builder: &mut AB,
    condition: AB::Expr,
    lhs: [AB::Expr; D],
    rhs: [AB::Expr; D],
) where
    AB: AirBuilder,
{
    builder
        .when(condition)
        .assert_zeros(ext_sub::<AB, D>(lhs, rhs));
}

/// Build a trace and public values for [`BinomialSumcheckAir`].
pub fn binomial_sumcheck_air_trace<F, const D: usize>(
    degree: usize,
    initial_claim: BinomialExtensionField<F, D>,
    proof: &SumcheckProof<BinomialExtensionField<F, D>>,
    round_challenges: &[BinomialExtensionField<F, D>],
) -> (RowMajorMatrix<F>, Vec<F>, BinomialExtensionField<F, D>)
where
    F: Field + BinomiallyExtendable<D>,
{
    assert_eq!(
        proof.round_polys.len(),
        round_challenges.len(),
        "sumcheck trace needs one challenge per round",
    );
    assert!(!proof.round_polys.is_empty(), "empty sumcheck trace");

    let num_rounds = proof.round_polys.len();
    let height = num_rounds.next_power_of_two();
    let width = 1 + num_rounds + D * (degree + 4);
    let mut values = Vec::with_capacity(width * height);
    let mut claim = initial_claim;
    for (round, (coeffs, &challenge)) in proof
        .round_polys
        .iter()
        .zip(round_challenges.iter())
        .enumerate()
    {
        assert_eq!(
            coeffs.len(),
            degree + 1,
            "sumcheck round {round} has wrong degree",
        );
        let h_0 = coeffs[0];
        let h_1: BinomialExtensionField<F, D> = coeffs.iter().copied().sum();
        assert_eq!(h_0 + h_1, claim, "sumcheck round {round} is inconsistent");

        let claim_out = coeffs
            .iter()
            .rev()
            .fold(BinomialExtensionField::<F, D>::ZERO, |acc, &coeff| {
                acc * challenge + coeff
            });
        values.push(F::ONE);
        for selector in 0..num_rounds {
            values.push(F::from_bool(selector == round));
        }
        push_ext(&mut values, claim);
        push_ext(&mut values, claim_out);
        push_ext(&mut values, challenge);
        for &coeff in coeffs {
            push_ext(&mut values, coeff);
        }
        claim = claim_out;
    }
    values.resize(width * height, F::ZERO);

    let mut public_values = Vec::with_capacity(D * (num_rounds + 2));
    push_ext(&mut public_values, initial_claim);
    push_ext(&mut public_values, claim);
    for &challenge in round_challenges {
        push_ext(&mut public_values, challenge);
    }
    (RowMajorMatrix::new(values, width), public_values, claim)
}

#[cfg(feature = "stark-backend")]
pub fn binomial_sumcheck_air<F, SC, const D: usize>(num_rounds: usize, degree: usize) -> AirRef<SC>
where
    F: Field + BinomiallyExtendable<D>,
    SC: StarkProtocolConfig<F = F>,
{
    alloc::sync::Arc::new(BinomialSumcheckAir::<F, D>::for_binomial_extension(
        num_rounds, degree,
    )) as AirRef<SC>
}

#[cfg(feature = "stark-backend")]
pub fn binomial_sumcheck_air_context<SC, const D: usize>(
    degree: usize,
    initial_claim: BinomialExtensionField<SC::F, D>,
    proof: &SumcheckProof<BinomialExtensionField<SC::F, D>>,
    round_challenges: &[BinomialExtensionField<SC::F, D>],
) -> (
    AirProvingContext<CpuColMajorBackend<SC>>,
    BinomialExtensionField<SC::F, D>,
)
where
    SC: StarkProtocolConfig,
    SC::F: Field + BinomiallyExtendable<D>,
{
    let (trace, public_values, final_claim) =
        binomial_sumcheck_air_trace::<SC::F, D>(degree, initial_claim, proof, round_challenges);
    (
        AirProvingContext::simple(ColMajorMatrix::from_row_major(&trace), public_values),
        final_claim,
    )
}

#[cfg(feature = "stark-backend")]
pub fn binomial_sumcheck_proving_context<SC, const D: usize>(
    air_id: usize,
    degree: usize,
    initial_claim: BinomialExtensionField<SC::F, D>,
    proof: &SumcheckProof<BinomialExtensionField<SC::F, D>>,
    round_challenges: &[BinomialExtensionField<SC::F, D>],
) -> (
    ProvingContext<CpuColMajorBackend<SC>>,
    BinomialExtensionField<SC::F, D>,
)
where
    SC: StarkProtocolConfig,
    SC::F: Field + BinomiallyExtendable<D>,
{
    let (ctx, final_claim) =
        binomial_sumcheck_air_context::<SC, D>(degree, initial_claim, proof, round_challenges);
    (ProvingContext::new(alloc::vec![(air_id, ctx)]), final_claim)
}

fn push_ext<F, const D: usize>(out: &mut Vec<F>, value: BinomialExtensionField<F, D>)
where
    F: Field + BinomiallyExtendable<D>,
{
    out.extend_from_slice(value.as_basis_coefficients_slice());
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_air::check_constraints;
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    fn honest_trace() -> (BinomialSumcheckAir<F, 4>, RowMajorMatrix<F>, Vec<F>) {
        let r0 = EF::new([F::from_u64(3), F::from_u64(1), F::ZERO, F::ZERO]);
        let r1 = EF::new([F::from_u64(5), F::ZERO, F::from_u64(1), F::ZERO]);
        let round0 = vec![EF::ONE, EF::from_u64(2), EF::from_u64(3)];
        let initial = round0[0] + round0.iter().copied().sum::<EF>();
        let claim1 = round0.iter().rev().fold(EF::ZERO, |acc, &c| acc * r0 + c);
        let round1 = vec![EF::ZERO, claim1, EF::ZERO];
        let proof = SumcheckProof {
            round_polys: vec![round0, round1],
        };
        let air = BinomialSumcheckAir::<F, 4>::for_binomial_extension(2, 2);
        let (trace, public_values, _final_claim) =
            binomial_sumcheck_air_trace::<F, 4>(2, initial, &proof, &[r0, r1]);
        (air, trace, public_values)
    }

    #[test]
    fn binomial_sumcheck_air_accepts_honest_trace() {
        let (air, trace, public_values) = honest_trace();
        check_constraints(&air, &trace, &public_values);
    }

    #[test]
    #[should_panic(expected = "constraints not satisfied")]
    fn binomial_sumcheck_air_rejects_bad_transition() {
        let (air, mut trace, public_values) = honest_trace();
        trace.values[air.width()] += F::ONE;
        check_constraints(&air, &trace, &public_values);
    }

    #[test]
    #[should_panic(expected = "constraints not satisfied")]
    fn binomial_sumcheck_air_rejects_bad_public_challenge() {
        let (air, trace, mut public_values) = honest_trace();
        public_values[8] += F::ONE;
        check_constraints(&air, &trace, &public_values);
    }
}
