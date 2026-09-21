use std::borrow::Cow;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{
    PcsError, PeriodicColumnShapeError, StarkConfig, VerificationError, prove, verify,
};
use rand::SeedableRng;
use rand::rngs::SmallRng;

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;
type Perm = Poseidon2BabyBear<16>;
type Hash = PaddingFreeSponge<Perm, 16, 8, 8>;
type Compress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, Hash, Compress, 2, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Dft = Radix2DitParallel<Val>;
type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Config = StarkConfig<Pcs, Challenge, Challenger>;

/// One-column AIR with a single periodic column of the given period.
///
/// The constraint pins each trace cell to the periodic value at that row.
/// Two instances with different periods share the same symbolic shape:
/// both report one periodic column of degree one and one width-one trace.
#[derive(Clone)]
struct SinglePeriodicAir {
    period: usize,
}

impl<F: Field> BaseAir<F> for SinglePeriodicAir {
    fn width(&self) -> usize {
        1
    }

    fn num_periodic_columns(&self) -> usize {
        1
    }

    fn periodic_columns(&self) -> Cow<'_, [Vec<F>]> {
        // A single column holding 0, 1, ..., period - 1.
        // A period of zero yields one empty column, which has no valid subdomain.
        Cow::Owned(vec![(0..self.period as u64).map(F::from_u64).collect()])
    }
}

impl<AB: AirBuilder> Air<AB> for SinglePeriodicAir
where
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        // Pin the single trace column to the periodic value at the current row.
        let main = builder.main();
        let local = main.current_slice();
        let p0 = builder.periodic_values()[0].into();
        builder.assert_eq(local[0], p0);
    }
}

/// Number of trace rows used by the fixtures, so the trace length is 2^6 = 64.
const LOG_TRACE_ROWS: usize = 6;
const TRACE_LENGTH: usize = 1 << LOG_TRACE_ROWS;

/// Build a trace whose only column repeats 0, 1, ..., period - 1.
fn periodic_trace(period: usize, rows: usize) -> RowMajorMatrix<Val> {
    let column: Vec<Val> = (0..period as u64).map(Val::from_u64).collect();
    let values: Vec<Val> = (0..rows).map(|i| column[i % period]).collect();
    RowMajorMatrix::new(values, 1)
}

fn config() -> Config {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = Hash::new(perm.clone());
    let compress = Compress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let fri_params = FriParameters {
        log_blowup: 2,
        log_final_poly_len: 3,
        max_log_arity: 2,
        num_queries: 40,
        batch_proof_of_work_bits: 0,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };
    let pcs = Pcs::new(Dft::default(), val_mmcs, fri_params);
    StarkConfig::new(pcs, Challenger::new(perm))
}

/// Verify a valid proof against an AIR whose single periodic column has the given length.
///
/// - The proof comes from a well-formed period-two AIR.
/// - The verifying AIR shares the same symbolic shape: one width-one trace and one degree-one periodic column.
/// - The quotient layout therefore matches.
/// - Verification then reaches the periodic-length check.
fn verify_with_period(bad_period: usize) -> Result<(), VerificationError<PcsError<Config>>> {
    let config = config();

    // Prover side: period 2 column [0, 1], trace column [0, 1, 0, 1, ...].
    let good_air = SinglePeriodicAir { period: 2 };
    let proof = prove(&config, &good_air, periodic_trace(2, TRACE_LENGTH), &[]).unwrap();

    // Verifier side: same shape, but a malformed period the verifier cannot evaluate.
    let bad_air = SinglePeriodicAir { period: bad_period };
    verify(&config, &bad_air, &proof, &[])
}

#[test]
fn empty_periodic_column_is_rejected() {
    // Period 0 yields one empty column [[]]; zero is not a power of two.
    // Without the check this panics taking log2 of zero.
    let result = verify_with_period(0);
    assert!(
        matches!(
            result,
            Err(VerificationError::PeriodicColumn(
                PeriodicColumnShapeError::LengthNotPowerOfTwo {
                    index: 0,
                    length: 0
                }
            ))
        ),
        "expected the empty column to be reported as a non-power-of-two length, got {result:?}"
    );
}

#[test]
fn non_power_of_two_periodic_column_is_rejected() {
    // Period 3 lies inside the 64-row trace but is not a power of two, so it has no subdomain.
    // The report names the power-of-two requirement, not the tiling one.
    let result = verify_with_period(3);
    assert!(
        matches!(
            result,
            Err(VerificationError::PeriodicColumn(
                PeriodicColumnShapeError::LengthNotPowerOfTwo {
                    index: 0,
                    length: 3
                }
            ))
        ),
        "expected period 3 to be reported as a non-power-of-two length, got {result:?}"
    );
}

#[test]
fn oversized_periodic_column_is_rejected() {
    // Period 128 is a power of two but twice the 64-row trace length.
    //
    //     [0,...,63|64,...,127]  <- only the first half would ever be read
    //
    // A column longer than the trace cannot tile it, so the report names the trace height.
    let oversized = 2 * TRACE_LENGTH;
    let result = verify_with_period(oversized);
    assert!(
        matches!(
            result,
            Err(VerificationError::PeriodicColumn(
                PeriodicColumnShapeError::LengthNotDividingHeight {
                    index: 0,
                    length,
                    height
                }
            )) if length == oversized && height == TRACE_LENGTH
        ),
        "expected period {oversized} to be reported against height {TRACE_LENGTH}, got {result:?}"
    );
}

#[test]
fn every_period_dividing_the_trace_length_verifies() {
    // The trace length is a power of two, so for a power-of-two period p the two candidate
    // rules coincide: p <= 2^b iff p divides 2^b.
    //
    //     TRACE_LENGTH = 64 = 2^6
    //     p = 2^a fits  <=>  a <= 6  <=>  p divides 64
    //
    // Every such period therefore has to verify.
    // A looser size comparison would accept nothing new on this height.
    // A stricter rule would break one of these rounds.
    let config = config();

    for log_period in 0..=TRACE_LENGTH.ilog2() {
        let period = 1usize << log_period;
        let air = SinglePeriodicAir { period };
        let trace = periodic_trace(period, TRACE_LENGTH);

        let proof = prove(&config, &air, trace, &[]).unwrap();
        verify(&config, &air, &proof, &[])
            .unwrap_or_else(|err| panic!("period {period} must verify, got {err:?}"));
    }
}
