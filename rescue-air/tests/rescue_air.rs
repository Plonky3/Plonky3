use core::borrow::Borrow;

use p3_air::{AirLayout, check_constraints, get_max_constraint_degree};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_mersenne_31::{MdsMatrixMersenne31, Mersenne31};
use p3_rescue::Rescue;
use p3_rescue_air::{RescueAir, RescueCols, generate_trace_rows, num_cols};
use p3_symmetric::Permutation;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type F = Mersenne31;
const WIDTH: usize = 12;
const ALPHA: u64 = 5;
const NUM_ROUNDS: usize = 8;

type Air = RescueAir<F, WIDTH, ALPHA, NUM_ROUNDS>;
type Perm = Rescue<F, MdsMatrixMersenne31, WIDTH, ALPHA>;

fn air_and_perm(rng: &mut SmallRng) -> (Air, Perm) {
    let rc: Vec<F> = (0..2 * WIDTH * NUM_ROUNDS).map(|_| rng.random()).collect();
    let air = Air::from_mds(&MdsMatrixMersenne31, rc.clone());
    let perm = Perm::new(NUM_ROUNDS, rc, MdsMatrixMersenne31);
    (air, perm)
}

fn inputs(rng: &mut SmallRng, n: usize) -> Vec<[F; WIDTH]> {
    (0..n)
        .map(|_| core::array::from_fn(|_| rng.random()))
        .collect()
}

fn trace(air: &Air, ins: &[[F; WIDTH]]) -> RowMajorMatrix<F> {
    generate_trace_rows::<F, WIDTH, ALPHA, NUM_ROUNDS>(ins, &air.mds, &air.round_constants)
}

/// The extracted matrix must act exactly as the MDS permutation it came from,
/// or the AIR would constrain a different linear layer than the hash uses.
#[test]
fn the_extracted_matrix_matches_the_mds_permutation() {
    let mut rng = SmallRng::seed_from_u64(1);
    let (air, _) = air_and_perm(&mut rng);
    for _ in 0..64 {
        let x: [F; WIDTH] = core::array::from_fn(|_| rng.random());
        let want = MdsMatrixMersenne31.permute(x);
        let got: [F; WIDTH] = core::array::from_fn(|i| {
            (0..WIDTH)
                .map(|j| air.mds[i][j] * x[j])
                .fold(F::ZERO, |a, b| a + b)
        });
        assert_eq!(got, want);
    }
}

/// The oracle: the trace's last round output must be what `p3-rescue` computes,
/// which ties this AIR to the permutation it claims to prove.
#[test]
fn the_last_round_output_is_the_rescue_permutation() {
    let mut rng = SmallRng::seed_from_u64(2);
    let (air, perm) = air_and_perm(&mut rng);
    let ins = inputs(&mut rng, 16);
    let tr = trace(&air, &ins);

    for (row, input) in ins.into_iter().enumerate() {
        let slice: Vec<F> = (0..tr.width()).map(|c| tr.get(row, c).unwrap()).collect();
        let cols: &RescueCols<F, WIDTH, NUM_ROUNDS> = slice.as_slice().borrow();
        assert_eq!(cols.rounds[NUM_ROUNDS - 1].post, perm.permute(input));
        assert_eq!(cols.inputs, input);
    }
}

#[test]
fn a_generated_trace_satisfies_the_constraints() {
    let mut rng = SmallRng::seed_from_u64(3);
    let (air, _) = air_and_perm(&mut rng);
    let tr = trace(&air, &inputs(&mut rng, 16));
    check_constraints(&air, &tr, &[]);
}

#[test]
fn padding_rows_also_satisfy_the_constraints() {
    let mut rng = SmallRng::seed_from_u64(6);
    let (air, _) = air_and_perm(&mut rng);
    // 5 inputs pad to 8 rows, so three rows are all-zero inputs.
    let tr = trace(&air, &inputs(&mut rng, 5));
    assert_eq!(tr.height(), 8);
    check_constraints(&air, &tr, &[]);
}

#[test]
fn the_width_is_the_column_count() {
    assert_eq!(
        num_cols::<WIDTH, NUM_ROUNDS>(),
        WIDTH + NUM_ROUNDS * 4 * WIDTH
    );
}

#[test]
fn the_constraint_degree_is_the_sbox_degree() {
    let mut rng = SmallRng::seed_from_u64(4);
    let (air, _) = air_and_perm(&mut rng);
    let layout = AirLayout {
        main_width: num_cols::<WIDTH, NUM_ROUNDS>(),
        ..Default::default()
    };
    assert_eq!(
        get_max_constraint_degree::<F, _>(&air, layout, 1 << 4),
        ALPHA as usize
    );
}

/// Every committed column must be pinned by some constraint: flipping any one
/// of them has to be rejected, or the AIR would accept a trace that is not a
/// Rescue evaluation.
#[test]
fn tampering_with_any_column_is_rejected() {
    let mut rng = SmallRng::seed_from_u64(5);
    let (air, _) = air_and_perm(&mut rng);
    let base = trace(&air, &inputs(&mut rng, 4));

    let width = num_cols::<WIDTH, NUM_ROUNDS>();
    let mut accepted = Vec::new();
    for col in 0..width {
        let mut tr = base.clone();
        tr.values[col] += F::ONE;
        let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_constraints(&air, &tr, &[]);
        }))
        .is_err();
        if !rejected {
            accepted.push(col);
        }
    }
    assert!(
        accepted.is_empty(),
        "these columns are not constrained: {accepted:?}"
    );
}
