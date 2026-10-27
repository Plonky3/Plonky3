//! Circuit and witness tests.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::{Field, PrimeCharacteristicRing};
use p3_keccak::Keccak256Hash;
use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};

use crate::{Builder, Circuit, FlockChallenger, Val, Wire, Witness};

/// A random circuit over the builder's whole vocabulary, with a narrow port and undriven outputs.
pub(crate) fn random_circuit(rng: &mut SmallRng, max_gates: u32) -> Circuit {
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
pub(crate) fn random_rows(rng: &mut SmallRng, n: usize) -> (Vec<[u64; 3]>, [u64; 3]) {
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
pub(crate) fn unpack<V: Field>(words: &[u64]) -> Vec<V> {
    words
        .iter()
        .flat_map(|&w| (0..64).map(move |b| V::from_bool((w >> b) & 1 == 1)))
        .collect()
}

/// Whether packed `z` satisfies every row `(A z)_s * (B z)_s = z_s` of every instance, by the forward walk.
pub(crate) fn satisfies(c: &Circuit, z: &[u64]) -> bool {
    z.chunks((1 << c.k_log()) / 64).all(|words| {
        let z = unpack::<Val>(words);
        let mut rows = vec![Val::ZERO; z.len()];
        c.walk_rows(&z, |s, a, b| rows[s] = a * b);
        rows == z
    })
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
fn every_row_of_the_witness_holds() {
    // Invariant: C is the identity, so row s reads (A z)_s * (B z)_s = z_s, an AND of the packed tables.
    let mut rng = SmallRng::seed_from_u64(0xCC5);
    for _ in 0..5 {
        // Fixture state: 8 instances of a random circuit.
        let c = random_circuit(&mut rng, 600);
        let (rows, padding) = random_rows(&mut rng, 8);
        let w = Witness::generate(&c, 3, &rows, &padding, |row, words| {
            words.copy_from_slice(row);
        });
        for ((&z, &a), &b) in w.z().iter().zip(w.az()).zip(w.bz()) {
            assert_eq!(a & b, z);
        }

        // Every instance holds the constant one and zero padding.
        let bit = |words: &[u64], s: usize| (words[s / 64] >> (s % 64)) & 1;
        for z in w.z().chunks((1 << c.k_log()) / 64) {
            assert_eq!(bit(z, c.const_pos()), 1);
            assert!((c.useful_bits()..1 << c.k_log()).all(|s| bit(z, s) == 0));
        }
    }
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
fn the_forward_walk_reproduces_the_witness_tables() {
    let mut rng = SmallRng::seed_from_u64(0xF0E);
    for _ in 0..5 {
        // Fixture state: 8 instances of a random circuit.
        let c = random_circuit(&mut rng, 600);
        let (rows, padding) = random_rows(&mut rng, 8);
        let w = Witness::generate(&c, 3, &rows, &padding, |row, words| {
            words.copy_from_slice(row);
        });

        // Invariant: the walk gives A z and B z, and rows no gate commits stay zero.
        let words = (1 << c.k_log()) / 64;
        let tables = w
            .z()
            .chunks(words)
            .zip(w.az().chunks(words))
            .zip(w.bz().chunks(words));
        for ((z, az), bz) in tables {
            let mut ab = [vec![Val::ZERO; 64 * words], vec![Val::ZERO; 64 * words]];
            c.walk_rows(&unpack::<Val>(z), |s, a, b| {
                ab[0][s] = a;
                ab[1][s] = b;
            });
            assert_eq!(ab, [unpack::<Val>(az), unpack::<Val>(bz)]);
        }
        assert!(satisfies(&c, w.z()));

        // Mutation: one committed product bit of instance 0 flips, so its row breaks.
        let mut bad = w.z().to_vec();
        let slot = c.useful_bits() - 1;
        bad[slot / 64] ^= 1 << (slot % 64);
        assert!(!satisfies(&c, &bad));

        // Invariant: the rows are homogeneous, so only the pinned constant rules out z = 0.
        assert!(satisfies(&c, &vec![0; bad.len()]));
    }
}

/// One batch of a random circuit large enough for the zerocheck, with its witness.
fn proof_fixture(seed: u64) -> (Circuit, Witness, usize) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let c = random_circuit(&mut rng, 1500);
    let log_n = crate::zerocheck::MIN_LOG_N.saturating_sub(c.k_log()).max(3);
    let (rows, padding) = random_rows(&mut rng, (1 << log_n) - 5);
    let w = Witness::generate(&c, log_n, &rows, &padding, |row, words| {
        words.copy_from_slice(row);
    });
    (c, w, log_n)
}

/// A fresh sponge.
fn sponge() -> FlockChallenger {
    FlockChallenger::from_hasher(Vec::new(), Keccak256Hash)
}

#[test]
fn a_batch_proof_verifies() {
    let (c, w, log_n) = proof_fixture(0xF10C);
    let batch = crate::Batch::new(&c, log_n).unwrap();
    let config = crate::FlockConfig::default();
    let bare = crate::PublicPorts::default();
    let proof = crate::prove(&config, &batch, &bare, w, &mut sponge()).unwrap();
    crate::verify(&config, &batch, &bare, &proof, &mut sponge()).unwrap();
}

#[test]
fn a_wrong_witness_or_a_tampered_proof_is_rejected() {
    let (c, w, log_n) = proof_fixture(0xBAD);
    let batch = crate::Batch::new(&c, log_n).unwrap();
    let config = crate::FlockConfig::default();
    let bare = crate::PublicPorts::default();

    // Mutation: one committed product bit of instance 0 flips, so its row breaks.
    let mut bad = w.clone();
    let slot = c.useful_bits() - 1;
    bad.tables_mut()[0][slot / 64] ^= 1 << (slot % 64);
    let proof = crate::prove(&config, &batch, &bare, bad, &mut sponge()).unwrap();
    assert!(crate::verify(&config, &batch, &bare, &proof, &mut sponge()).is_err());

    // Mutation: an honest proof checked against another batch's commitment.
    let honest = crate::prove(&config, &batch, &bare, w.clone(), &mut sponge()).unwrap();
    let mut swapped = honest.clone();
    swapped.commitment = proof.commitment;
    assert!(crate::verify(&config, &batch, &bare, &swapped, &mut sponge()).is_err());

    // Mutation: the proof of one batch size checked as another.
    let wider = crate::Batch::new(&c, log_n + 1).unwrap();
    assert!(crate::verify(&config, &wider, &bare, &honest, &mut sponge()).is_err());

    // Mutation: a witness of another batch size is refused before anything is sent.
    assert!(crate::prove(&config, &wider, &bare, w, &mut sponge()).is_err());
}

// A parallel grinding search may return any witness, so the pin needs the sequential one.
#[test]
#[cfg(not(feature = "parallel"))]
fn a_proof_on_fixed_seeds_has_a_pinned_digest() {
    use tiny_keccak::{Hasher, Keccak};

    // Invariant: proofs on fixed seeds serialize to the same bytes across refactors.
    let mut keccak = Keccak::v256();
    for (seed, extra) in [(0xF10C, 0), (0x5EED, 3)] {
        let (c, w, log_n) = proof_fixture(seed);
        let w = if extra == 0 {
            w
        } else {
            let mut rng = SmallRng::seed_from_u64(seed);
            let (rows, padding) = random_rows(&mut rng, (1 << (log_n + extra)) - 5);
            Witness::generate(&c, log_n + extra, &rows, &padding, |row, words| {
                words.copy_from_slice(row);
            })
        };
        let batch = crate::Batch::new(&c, log_n + extra).unwrap();
        let config = crate::FlockConfig::default();
        let bare = crate::PublicPorts::default();
        let proof = crate::prove(&config, &batch, &bare, w, &mut sponge()).unwrap();
        crate::verify(&config, &batch, &bare, &proof, &mut sponge()).unwrap();
        keccak.update(&postcard::to_allocvec(&proof).unwrap());
    }
    let mut digest = [0u8; 32];
    keccak.finalize(&mut digest);
    assert_eq!(
        digest,
        hex_literal::hex!("a7c678c7d48a8a692045d1229a493d2f208d5d7cd6bead20db0b53ec1d79505c")
    );
}

#[test]
fn a_batch_charges_its_two_reductions_by_their_bad_challenge_counts() {
    // Fixture state: a 2^8-bit circuit on 2^10 instances, so 18 witness variables.
    //
    //     zerocheck   n = 18 - 6 = 12 past the skip  ->  (12 - 7) + 127 + 2 * 12 = 156
    //     lincheck    3 + 2 * (8 - 6) + 63 + 10                                  =  80
    let mut c = Builder::new(&[64], &[64]);
    let x = c.input(0).to_vec();
    let y = c.and_word(&x, &x);
    for (bit, &wire) in y.iter().enumerate() {
        c.output(0, bit, wire);
    }
    let circuit = c.finish();
    assert_eq!(circuit.k_log(), 8);
    let batch = crate::Batch::new(&circuit, 10).unwrap();

    // Each term is the field's 192 bits less the logarithm of its count.
    let [zerocheck, lincheck] = batch.security_terms();
    assert_eq!(zerocheck.label, crate::ZEROCHECK_LABEL);
    assert_eq!(lincheck.label, crate::LINCHECK_LABEL);
    assert!((zerocheck.bits.bits() - (192.0 - 156f64.log2())).abs() < 1e-9);
    assert!((lincheck.bits.bits() - (192.0 - 80f64.log2())).abs() < 1e-9);
}
