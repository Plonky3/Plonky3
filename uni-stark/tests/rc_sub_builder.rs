//! Minimal range-check example that reuses a bit-decomposition gadget via [`SubAirBuilder`].
//!
//! Column layout:
//! - `c[0]`: running sum owned by the parent AIR.
//! - `c[1]`: value that must stay in `[0, 2^NUM_RANGE_BITS)`.
//! - `c[2..]`: boolean limbs proving the decomposition of `c[1]`.
//!
//! The sub-AIR enforces the decomposition + booleanity over columns `1..`, while the parent AIR
//! never touches the bit columns and only reasons about the accumulated sum.

use core::marker::PhantomData;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::testing::TrivialPcs;
use p3_dft::Radix2DitParallel;
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkConfig, SubAirBuilder, SymbolicAirBuilder, prove, verify};
use rand::SeedableRng;
use rand::rngs::SmallRng;

const NUM_RANGE_BITS: usize = 4;
const TRACE_WIDTH: usize = 2 + NUM_RANGE_BITS;

/// Range-check gadget: proves a value equals the sum of weighted boolean limbs.
#[derive(Copy, Clone)]
struct RangeDecompAir;

impl BaseAir<BabyBear> for RangeDecompAir {
    fn width(&self) -> usize {
        1 + NUM_RANGE_BITS
    }
}

impl<AB> Air<AB> for RangeDecompAir
where
    AB: AirBuilder<F = BabyBear>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("matrix should have a local row");

        let value = local[0].clone();
        let bits = &local[1..];

        let mut recomposed = AB::Expr::ZERO;
        for (i, bit) in bits.iter().enumerate() {
            let weight = BabyBear::from_u32(1 << i);
            recomposed += bit.clone() * weight;
            builder.assert_zero(bit.clone() * (bit.clone() - AB::Expr::ONE));
        }

        builder.assert_zero(value - recomposed);
    }
}

/// Parent AIR that reuses the range gadget but only reasons about the running sum.
#[derive(Copy, Clone)]
struct RangeCheckAir;

impl BaseAir<BabyBear> for RangeCheckAir {
    fn width(&self) -> usize {
        TRACE_WIDTH
    }
}

impl<AB> Air<AB> for RangeCheckAir
where
    AB: AirBuilder<F = BabyBear>,
{
    fn eval(&self, builder: &mut AB) {
        // Declare the sub-AIR and evaluate it via `SubAirBuilder`
        let sub_air = RangeDecompAir;
        {
            let mut sub_builder =
                SubAirBuilder::<AB, RangeDecompAir, AB::Var>::new(builder, 1..TRACE_WIDTH);
            sub_air.eval(&mut sub_builder);
        }

        // Evaluate the parent AIR
        let main = builder.main();
        let local = main.row_slice(0).expect("matrix should have a local row");
        let next = main.row_slice(1).expect("matrix only has 1 row?");

        let accumulator = local[0].clone();
        let range_value = local[1].clone();
        let next_accumulator = next[0].clone();

        builder.when_first_row().assert_zero(accumulator.clone());
        builder
            .when_transition()
            .assert_eq(next_accumulator, accumulator + range_value);
    }
}

impl RangeCheckAir {
    fn generate_trace(&self, rows: usize) -> RowMajorMatrix<BabyBear> {
        assert!(
            rows.is_power_of_two(),
            "trace height must be a power of two"
        );
        let mut values = BabyBear::zero_vec(rows * TRACE_WIDTH);
        let mut accumulator = BabyBear::ZERO;
        for row in 0..rows {
            let base = row * TRACE_WIDTH;
            let raw_value = (row * 7) % (1 << NUM_RANGE_BITS);
            values[base] = accumulator;
            values[base + 1] = BabyBear::from_u32(raw_value as u32);
            let mut tmp = raw_value;
            for bit in 0..NUM_RANGE_BITS {
                values[base + 2 + bit] = BabyBear::from_u32((tmp & 1) as u32);
                tmp >>= 1;
            }
            accumulator += BabyBear::from_u32(raw_value as u32);
        }
        RowMajorMatrix::new(values, TRACE_WIDTH)
    }
}

// Ensures the range-check gadget stays scoped to its columns and the whole AIR proves.
#[test]
fn range_checked_sub_builder() {
    let air = RangeCheckAir;
    let mut builder = SymbolicAirBuilder::<BabyBear>::new(0, TRACE_WIDTH, 0, 0, 0);
    air.eval(&mut builder);

    let constraints = builder.base_constraints();
    assert!(
        !constraints.is_empty(),
        "Range-check AIR should emit constraints"
    );

    prove_bb_trivial_deg4(&air, 3);
}

/// Tests the whole AIR on a trivial trace.
fn prove_bb_trivial_deg4(air: &RangeCheckAir, log_n: usize) {
    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Dft = Radix2DitParallel<Val>;
    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
    type Pcs = TrivialPcs<Val, Dft>;
    type Config = StarkConfig<Pcs, Challenge, Challenger>;

    let rows = 1 << log_n;
    let trace = air.generate_trace(rows);

    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let dft = Dft::default();

    let pcs = Pcs {
        dft,
        log_n,
        _phantom: PhantomData,
    };
    let challenger = Challenger::new(perm);
    let config = Config::new(pcs, challenger);

    let proof = prove(&config, air, trace, &[]);
    verify(&config, air, &proof, &[]).expect("verification failed");
}
