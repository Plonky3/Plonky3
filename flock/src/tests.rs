//! Circuit, witness and CCS-operator tests.

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::{BinaryChallenger, Poly64, Poly192};
use p3_ccs::LinearOperator;
use p3_challenger::HashChallenger;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_keccak::Keccak256Hash;
use p3_multilinear_util::poly::Poly;
use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};

use crate::r1cs::{self, CircuitMatrix, Side};
use crate::{Builder, Circuit, Wire, Witness};

/// Committed field.
type F = Poly64;
/// Challenge field.
type EF = Poly192;
/// Fiat-Shamir sponge.
type Challenger = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;

/// A random circuit over the builder's whole vocabulary, with a narrow port and undriven outputs.
fn random_circuit(rng: &mut SmallRng, max_gates: u32) -> Circuit {
    let narrow = rng.random_range(1..64);
    let mut c = Builder::new(&[64, narrow, 64], &[64, narrow]);
    let mut pool: Vec<Wire> = (0..3).flat_map(|port| c.input(port).to_vec()).collect();
    pool.extend([Wire::ZERO, c.one()]);

    // Random gates, each result an operand for the next.
    for _ in 0..rng.random_range(max_gates / 4..max_gates) {
        let pick = |rng: &mut SmallRng, pool: &[Wire]| pool[rng.random_range(0..pool.len())];
        let (s, x, y) = (pick(rng, &pool), pick(rng, &pool), pick(rng, &pool));
        let wire = match rng.random_range(0..5) {
            0 => c.xor(x, y),
            1 => c.and(x, y),
            2 => c.or(x, y),
            3 => c.mux(s, x, y),
            _ => c.not(x),
        };
        pool.push(wire);
    }

    // About half the output bits driven, by a copy or a product.
    for (port, bits) in [(0, 64), (1, narrow)] {
        for bit in 0..bits {
            let pick = |rng: &mut SmallRng| pool[rng.random_range(0..pool.len())];
            match rng.random_range(0..4) {
                0 | 1 => {}
                2 => {
                    let w = pick(rng);
                    c.output(port, bit, w);
                }
                _ => {
                    let (x, y) = (pick(rng), pick(rng));
                    c.and_output(port, bit, x, y);
                }
            }
        }
    }
    c.finish()
}

/// Random input rows plus a padding row.
fn random_rows(rng: &mut SmallRng, n: usize) -> (Vec<[u64; 3]>, [u64; 3]) {
    let mut row = || -> [u64; 3] { core::array::from_fn(|_| rng.next_u64()) };
    ((0..n).map(|_| row()).collect(), row())
}

/// The batch witness by one-instance walks.
fn witness_by_instance(
    c: &Circuit,
    rows: &[[u64; 3]],
    padding: &[u64; 3],
    log_n: usize,
) -> Witness {
    let words = (1 << c.k_log()) / 64;
    let (mut z, mut az, mut bz) = (
        vec![0; words << log_n],
        vec![0; words << log_n],
        vec![0; words << log_n],
    );
    for i in 0..1 << log_n {
        let range = i * words..(i + 1) * words;
        c.witness_instance(
            rows.get(i).unwrap_or(padding),
            &mut z[range.clone()],
            &mut az[range.clone()],
            &mut bz[range],
        );
    }
    Witness::from_tables(z, az, bz, log_n)
}

/// Packed bits as field elements, one per bit.
fn unpack<V: Field>(words: &[u64]) -> Vec<V> {
    words
        .iter()
        .flat_map(|&w| (0..64).map(move |b| V::from_bool((w >> b) & 1 == 1)))
        .collect()
}

#[test]
fn bitsliced_witness_is_the_one_instance_walk() {
    let mut rng = SmallRng::seed_from_u64(0xB175);
    for round in 0..30 {
        // Fixture state: batches of 8 to 128 instances, some taking the padding row.
        let c = random_circuit(&mut rng, 1000);
        let log_n = 3 + round % 5;
        let (rows, padding) = random_rows(&mut rng, (1 << log_n) - (round % 3) * 3);

        let walked = witness_by_instance(&c, &rows, &padding, log_n);
        let sliced = Witness::generate(&c, log_n, &rows, &padding, |row, words| {
            words.copy_from_slice(row);
        });
        assert_eq!(walked, sliced, "round {round}");
    }
}

#[test]
fn the_ccs_operators_reproduce_the_witness_tables() {
    let mut rng = SmallRng::seed_from_u64(0xCC5);
    for _ in 0..5 {
        // Fixture state: 8 instances; the CCS is satisfied and its outputs are A z, B z, z.
        let c = random_circuit(&mut rng, 600);
        let (rows, padding) = random_rows(&mut rng, 8);
        let w = Witness::generate(&c, 3, &rows, &padding, |row, words| {
            words.copy_from_slice(row);
        });
        let ccs = r1cs::ccs::<F>(Arc::new(c), 3);
        let z = unpack::<F>(w.z());
        assert_eq!(ccs.check(&z), Ok(()));
        let tables = ccs.operand_tables(&z);
        assert_eq!(tables[0], unpack::<F>(w.az()));
        assert_eq!(tables[1], unpack::<F>(w.bz()));
        assert_eq!(tables[2], z);
    }
}

#[test]
fn the_constant_cells_reject_the_all_zero_witness() {
    // Invariant: the homogeneous R1CS accepts z = 0; only the public constant tells it apart.
    let mut rng = SmallRng::seed_from_u64(0xC0);
    let c = random_circuit(&mut rng, 300);
    let (rows, padding) = random_rows(&mut rng, 8);
    let w = Witness::generate(&c, 3, &rows, &padding, |row, words| {
        words.copy_from_slice(row);
    });
    let one = r1cs::constant_cells::<F>(&c, 3);
    let ccs = r1cs::ccs::<F>(Arc::new(c), 3);

    // Honest batch: the CCS holds and every instance's constant is one.
    let z = unpack::<F>(w.z());
    assert_eq!(ccs.check(&z), Ok(()));
    assert!(one.matches(&z));

    // Mutation: the all-zero witness satisfies every row but not the constant.
    let zero = vec![F::ZERO; z.len()];
    assert_eq!(ccs.check(&zero), Ok(()));
    assert!(!one.matches(&zero));
}

#[test]
#[should_panic(expected = "already committed")]
fn an_output_bit_cannot_be_committed_twice() {
    // Mutation: the second copy would silently replace the first row.
    let mut b = Builder::new(&[1], &[1]);
    let x = b.input(0)[0];
    b.output(0, 0, x);
    let not_x = b.not(x);
    b.output(0, 0, not_x);
}

#[test]
#[should_panic(expected = "already committed")]
fn a_product_cannot_land_on_a_committed_output_bit() {
    // Mutation: a structural zero commits its slot too, so a later product is refused.
    let mut b = Builder::new(&[1], &[1]);
    let x = b.input(0)[0];
    b.output(0, 0, Wire::ZERO);
    b.and_output(0, 0, x, x);
}

#[test]
fn the_digest_separates_circuits_and_matches_equal_ones() {
    // Invariant: one circuit, one digest; one changed gate, another digest.
    let build = |flip: bool| {
        let mut b = Builder::new(&[2], &[1]);
        let (x, y) = (b.input(0)[0], b.input(0)[1]);
        let out = if flip { b.and(y, x) } else { b.and(x, y) };
        b.output(0, 0, out);
        b.finish()
    };
    assert_eq!(build(false).digest(), build(false).digest());
    assert_ne!(build(false).digest(), build(true).digest());
}

#[test]
fn transpose_and_evaluation_match_the_dense_matrix() {
    // Invariant: the backward walk is the transpose and the extension is the dense one.
    let mut rng = SmallRng::seed_from_u64(0xDE45E);
    let c = Arc::new(random_circuit(&mut rng, 120));
    let k = c.k_log();
    let n = 1 << k;
    for side in [Side::A, Side::B, Side::C] {
        let m = CircuitMatrix::new(c.clone(), side);

        // Dense columns from unit vectors.
        let mut dense = vec![vec![F::ZERO; n]; n];
        for j in 0..n {
            let mut unit = vec![F::ZERO; n];
            unit[j] = F::ONE;
            let mut col = vec![F::ZERO; n];
            m.apply(&unit, &mut col);
            for i in 0..n {
                dense[i][j] = col[i];
            }
        }

        // Transpose against random weights and a random starting accumulator.
        let u: Vec<EF> = (0..n).map(|_| rng.random()).collect();
        let scale: EF = rng.random();
        let start: Vec<EF> = (0..n).map(|_| rng.random()).collect();
        let mut acc = start.clone();
        LinearOperator::<F>::add_transpose(&m, &u, scale, &mut acc);
        for j in 0..n {
            let moved: EF = (0..n).map(|i| u[i] * dense[i][j]).sum();
            assert_eq!(acc[j], start[j] + scale * moved, "{side:?} column {j}");
        }

        // Extension at a random point.
        let r: Vec<EF> = (0..k).map(|_| rng.random()).collect();
        let y: Vec<EF> = (0..k).map(|_| rng.random()).collect();
        let (er, ey) = (
            Poly::new_from_point(&r, EF::ONE),
            Poly::new_from_point(&y, EF::ONE),
        );
        let expected: EF = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| er.as_slice()[i] * ey.as_slice()[j] * dense[i][j])
            .sum();
        assert_eq!(
            LinearOperator::<F>::evaluate(&m, &r, &y),
            expected,
            "{side:?}"
        );
    }
}

#[test]
fn the_generic_reduction_proves_a_circuit_batch() {
    // Fixture state: 16 instances of a random circuit, proved by the generic CCS reduction.
    let mut rng = SmallRng::seed_from_u64(0x9E9);
    let c = random_circuit(&mut rng, 400);
    let (rows, padding) = random_rows(&mut rng, 16);
    let w = Witness::generate(&c, 4, &rows, &padding, |row, words| {
        words.copy_from_slice(row);
    });
    let slot = c.useful_bits() - 1;
    let ccs = r1cs::ccs::<F>(Arc::new(c), 4);
    let fresh = || Challenger::from_hasher(Vec::new(), Keccak256Hash);

    // Honest batch: the claim opens to the witness extension.
    let z = unpack::<F>(w.z());
    let (proof, claim) = p3_ccs::prove::<F, EF, _, _>(&ccs, &z, 0, &mut fresh());
    let verified = p3_ccs::verify(&ccs, &proof, 0, &mut fresh()).unwrap();
    assert_eq!(claim, verified);
    assert_eq!(
        Poly::new(z.clone()).eval_base(&verified.point),
        verified.value
    );

    // Mutation: flip one committed product bit, so its row breaks.
    let mut bad = z;
    bad[slot] += F::ONE;
    assert!(ccs.check(&bad).is_err());
    let (proof, _) = p3_ccs::prove::<F, EF, _, _>(&ccs, &bad, 0, &mut fresh());
    assert!(p3_ccs::verify(&ccs, &proof, 0, &mut fresh()).is_err());
}
