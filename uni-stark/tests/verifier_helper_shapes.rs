//! The public verifier helpers reject mis-shaped input with an error instead of panicking.
//!
//! Recursive and external verifiers call `verify_constraints` and
//! `recompose_quotient_from_chunks` directly, without the shape checks `verify` runs first.

use core::marker::PhantomData;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::Pcs;
use p3_commit::testing::TrivialPcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_uni_stark::{
    Domain, InvalidProofShapeError, StarkConfig, VerificationError, recompose_quotient_from_chunks,
    verify_constraints,
};

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyChallenger = DuplexChallenger<Val, Perm, 16, 8>;
type Dft = Radix2DitParallel<Val>;
type MyPcs = TrivialPcs<Val, Dft>;
type MyConfig = StarkConfig<MyPcs, Challenge, MyChallenger>;

/// Two columns, three public values, reads the next row.
struct Fib;

impl<F> BaseAir<F> for Fib {
    fn width(&self) -> usize {
        2
    }

    fn num_public_values(&self) -> usize {
        3
    }
}

impl<AB: AirBuilder> Air<AB> for Fib {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let pis = builder.public_values();
        let (a, b, x) = (pis[0], pis[1], pis[2]);
        let local = main.current_slice();
        let next = main.next_slice();
        let (l0, l1, n0, n1) = (local[0], local[1], next[0], next[1]);
        let mut first = builder.when_first_row();
        first.assert_eq(l0, a);
        first.assert_eq(l1, b);
        let mut transition = builder.when_transition();
        transition.assert_eq(l1, n0);
        transition.assert_eq(l0 + l1, n1);
        builder.when_last_row().assert_eq(l1, x);
    }
}

fn trace_domain() -> Domain<MyConfig> {
    let pcs = MyPcs {
        dft: Dft::default(),
        log_n: 3,
        _phantom: PhantomData,
    };
    <MyPcs as Pcs<Challenge, MyChallenger>>::natural_domain_for_degree(&pcs, 8)
}

/// Calls `verify_constraints` on `Fib` with the given rows and public values.
fn check(
    trace_local: &[Challenge],
    trace_next: &[Challenge],
    periodic_values: &[Challenge],
    public_values: &[Val],
) -> Result<(), VerificationError<()>> {
    verify_constraints::<MyConfig, Fib, ()>(
        &Fib,
        trace_local,
        trace_next,
        None,
        None,
        periodic_values,
        public_values,
        trace_domain(),
        Challenge::from(Val::from_u64(11)),
        Challenge::from(Val::from_u64(7)),
        Challenge::ZERO,
    )
}

fn shape_error(result: Result<(), VerificationError<()>>) -> InvalidProofShapeError {
    match result {
        Err(VerificationError::InvalidProofShape(err)) => err,
        other => panic!("expected a proof-shape error, got {other:?}"),
    }
}

const PIS: [Val; 3] = [Val::ZERO, Val::ONE, Val::new(21)];

#[test]
fn well_shaped_rows_reach_the_constraint_check() {
    // A wrong quotient is the only thing that can reject well-shaped rows.
    let row = [Challenge::ONE; 2];
    assert!(matches!(
        check(&row, &row, &[], &PIS),
        Err(VerificationError::OodEvaluationMismatch { .. })
    ));
}

#[test]
fn a_short_next_row_is_rejected() {
    let local = [Challenge::ONE; 2];
    let next = [Challenge::ONE; 1];
    assert!(matches!(
        shape_error(check(&local, &next, &[], &PIS)),
        InvalidProofShapeError::TraceNextMismatch { air: None }
    ));
}

#[test]
fn rows_narrower_than_the_air_are_rejected() {
    let row = [Challenge::ONE; 1];
    assert!(matches!(
        shape_error(check(&row, &row, &[], &PIS)),
        InvalidProofShapeError::OpenedValuesDimensionMismatch
    ));
}

#[test]
fn a_missing_public_value_is_rejected() {
    let row = [Challenge::ONE; 2];
    assert!(matches!(
        shape_error(check(&row, &row, &[], &PIS[..2])),
        InvalidProofShapeError::PublicValuesLengthMismatch {
            expected: 3,
            got: 2
        }
    ));
}

#[test]
fn periodic_values_for_undeclared_columns_are_rejected() {
    let row = [Challenge::ONE; 2];
    assert!(matches!(
        shape_error(check(&row, &row, &[Challenge::ONE], &PIS)),
        InvalidProofShapeError::OpenedValuesDimensionMismatch
    ));
}

#[test]
fn recompose_rejects_more_chunks_than_domains() {
    let domain = trace_domain();
    let chunk = vec![Challenge::ONE; <Challenge as BasedVectorSpace<Val>>::DIMENSION];
    assert!(matches!(
        recompose_quotient_from_chunks::<MyConfig>(
            &[domain],
            &[chunk.clone(), chunk],
            Challenge::ONE
        ),
        Err(InvalidProofShapeError::OpenedValuesDimensionMismatch)
    ));
}

#[test]
fn recompose_rejects_a_chunk_of_the_wrong_length() {
    let domain = trace_domain();
    let chunk = vec![Challenge::ONE; <Challenge as BasedVectorSpace<Val>>::DIMENSION + 1];
    assert!(matches!(
        recompose_quotient_from_chunks::<MyConfig>(&[domain], &[chunk], Challenge::ONE),
        Err(InvalidProofShapeError::OpenedValuesDimensionMismatch)
    ));
}

#[test]
fn recompose_accepts_one_chunk_per_domain() {
    let domain = trace_domain();
    let mut coefficients = vec![Challenge::ZERO; <Challenge as BasedVectorSpace<Val>>::DIMENSION];
    coefficients[0] = Challenge::ONE;
    // With a single domain the Lagrange factor is the empty product, so the chunk comes back.
    assert!(matches!(
        recompose_quotient_from_chunks::<MyConfig>(&[domain], &[coefficients], Challenge::ONE),
        Ok(value) if value == Challenge::ONE
    ));
}
