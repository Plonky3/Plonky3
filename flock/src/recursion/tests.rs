//! Recursion tests: field gadgets and the BLAKE3 hash against native, then whole proofs verified in a circuit.

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::{Poly64, Poly192};
use p3_blake3::Blake3;
use p3_field::PrimeCharacteristicRing;
use p3_symmetric::CryptographicHasher;
use proptest::prelude::*;

use super::blake3::hash;
use super::gates::Byte;
use super::*;
use crate::{Batch, Circuit, FlockChallenger, FlockConfig, FlockProof, PublicPorts};

/// Whether the advice satisfies the circuit: every revealed bit takes the value the statement pins.
fn satisfied(rc: &RecursionCircuit, advice: &[u64]) -> bool {
    let witness = rc.witness(advice);
    let outputs = rc.circuit.output_words(witness.z());
    let mut expected = rc.revealed.clone();
    expected.resize(outputs.len(), 0);
    outputs == expected.as_slice()
}

fn poly192(limbs: [u64; 3]) -> Poly192 {
    Poly192::new(limbs.map(Poly64::new))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn field_gadgets_are_the_native_operations(
        a in any::<[u64; 3]>(),
        b in any::<[u64; 3]>(),
        c in any::<[u64; 3]>(),
    ) {
        let (x, y) = (Poly64::new(a[0]), Poly64::new(b[1]));
        let (a, b, c) = (poly192(a), poly192(b), poly192(c));
        let mut arith = WireArith::new();
        let (wa, wb, wx, wy) = (arith.advice(a), arith.advice(b), arith.advice(x), arith.advice(y));
        let outputs: Vec<Gf192> = vec![
            wa.clone() * wb.clone(),
            wa.clone() * wx.clone(),
            Gf192::from(wx.clone() * wy),
            wa.clone() * Gf192::from(c),
            wa.square(),
            Gf192::from(wx.square()),
            wa + wb + Gf192::from(c),
        ];
        let natives = [a * b, a * Poly192::from(x), Poly192::from(x * y), a * c, a.square(), Poly192::from(x.square()), a + b + c];
        for (out, native) in outputs.iter().zip(natives) {
            prop_assert_eq!(out.value(), native);
            arith.reveal(out);
        }
        let rc = arith.finish();
        prop_assert!(satisfied(&rc, &rc.advice));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    #[test]
    fn the_equality_element_is_the_native_one(len in 0usize..9, seed in any::<u64>()) {
        use p3_sumcheck::ring_switch::bits::BitTensorArith;
        use p3_verifier_arith::Native;
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(seed);
        let a: Vec<Poly192> = (0..len).map(|_| rng.random()).collect();
        let b: Vec<Poly192> = (0..len).map(|_| rng.random()).collect();
        let native = Native::<Poly64, Poly192>::new().equality_rows(&a, &b);
        let mut arith = WireArith::new();
        let wa: Vec<Gf192> = a.iter().map(|&v| arith.advice(v)).collect();
        let wb: Vec<Gf192> = b.iter().map(|&v| arith.advice(v)).collect();
        let rows = arith.equality_rows(&wa, &wb);
        for (row, &expected) in rows.iter().zip(&native) {
            prop_assert_eq!(row.value(), expected);
            arith.reveal(row);
        }
        let rc = arith.finish();
        prop_assert!(satisfied(&rc, &rc.advice));
        // Tamper: a flipped coordinate moves the revealed rows.
        if len > 0 {
            let mut advice = rc.advice.clone();
            advice[0] ^= 1;
            prop_assert!(!satisfied(&rc, &advice));
        }
    }
}

#[test]
fn products_cost_the_stated_ands() {
    let arith = WireArith::new();
    let (a, b) = (
        arith.advice(poly192([1, 2, 3])),
        arith.advice(poly192([4, 5, 6])),
    );
    let (x, y) = (arith.advice(Poly64::new(7)), arith.advice(Poly64::new(9)));
    let ands = |arith: &WireArith| arith.with_gates(|g| g.builder.num_ands());
    let start = ands(&arith);
    let _ = x.clone() * y;
    assert_eq!(ands(&arith) - start, 729);
    let start = ands(&arith);
    let _ = a.clone() * b;
    assert_eq!(ands(&arith) - start, 4374);
    let start = ands(&arith);
    let _ = a * x;
    assert_eq!(ands(&arith) - start, 3 * 729);
}

#[test]
fn the_hash_gadget_is_blake3() {
    for len in [0, 1, 31, 64, 65, 1000, 1024, 1025, 2048, 3100] {
        let bytes: Vec<u8> = (0..len).map(|i| (i * 7 + 3) as u8).collect();
        let arith = WireArith::new();
        let input: Vec<Byte> = bytes
            .iter()
            .map(|&b| {
                let wires = arith.with_gates(|g| g.advice((0..8).map(|i| (b >> i) & 1 == 1)));
                Byte {
                    wires: core::array::from_fn(|i| wires[i]),
                    value: b,
                }
            })
            .collect();
        let digest = arith.with_gates(|g| hash(&mut g.builder, &input));
        assert_eq!(
            digest.map(|b| b.value),
            Blake3.hash_slice(&bytes),
            "length {len}"
        );
        arith.with_gates(|g| {
            for byte in &digest {
                for (i, &w) in byte.wires.iter().enumerate() {
                    g.reveal(w, (byte.value >> i) & 1 == 1);
                }
            }
        });
        let rc = arith.finish();
        assert!(satisfied(&rc, &rc.advice), "length {len}");
        // A flipped input bit changes the digest the circuit computes.
        if len > 0 {
            let mut advice = rc.advice.clone();
            advice[0] ^= 1;
            assert!(!satisfied(&rc, &advice), "length {len}");
        }
    }
}

/// A BLAKE3-transcript proof of `2^log_instances` BLAKE3 compressions.
fn compression_proof(log_instances: usize, config: &FlockConfig) -> (Circuit, FlockProof<Blake3>) {
    use crate::Gadget;
    use crate::gadgets::blake3::Blake3Compress;
    let circuit = Blake3Compress::circuit();
    let rows: Vec<Blake3Compress> = (0..1u32 << log_instances)
        .map(|i| Blake3Compress {
            cv: core::array::from_fn(|j| i.wrapping_mul(0x9E37_79B9) ^ j as u32),
            block: core::array::from_fn(|j| i ^ (j as u32) << 8),
            counter: u64::from(i),
            block_len: 64,
            flags: 3,
        })
        .collect();
    let witness = Blake3Compress::witness(&circuit, log_instances, &rows, &rows[0]);
    let batch = Batch::new(&circuit, log_instances).unwrap();
    let mut challenger = FlockChallenger::<Blake3>::from_hasher(Vec::new(), Blake3);
    let proof = crate::prove(
        config,
        &batch,
        &PublicPorts::default(),
        witness,
        &mut challenger,
    )
    .unwrap();
    (circuit, proof)
}

#[test]
#[ignore = "builds and proves a circuit of 2^26 bits (peak 5 GB); run with --release --include-ignored"]
fn a_proof_verifies_in_a_circuit_and_its_claim_natively() {
    let mut config = FlockConfig::default();
    config.pcs.log_inv_rate = 4;
    let log = 3;
    let (circuit, proof) = compression_proof(log, &config);
    let batch = Batch::new(&circuit, log).unwrap();
    let mut arith = WireArith::new();
    let claim = arith
        .verify_proof(
            &config,
            &batch,
            &PublicPorts::default(),
            &proof,
            &mut Blake3Mirror::new(&arith, Vec::new()),
            None,
        )
        .unwrap();
    for c in claim.point.iter().chain([&claim.value]) {
        arith.reveal(c);
    }
    let point: Vec<Poly192> = claim.point.iter().map(Bin::value).collect();
    assert_eq!(evaluate_claim(&circuit, &point), claim.value.value());
    let rc = arith.finish();
    std::println!(
        "outer: 2^{} bits, {} ANDs, {} gates, {} ops, {} terms",
        rc.circuit.k_log(),
        rc.ands,
        rc.gates,
        rc.ops,
        rc.terms
    );
    for cost in &rc.costs {
        std::println!(
            "  {:<17} {:>11} ANDs {:>12} gates",
            cost.component,
            cost.ands,
            cost.gates
        );
    }
    assert!(satisfied(&rc, &rc.advice));

    // Any flipped proof bit leaves the statement unsatisfied: the transcript, a check or the revealed claim moves.
    let words = rc.advice.len();
    for at in [
        0,
        1,
        words / 7,
        words / 3,
        words / 2,
        2 * words / 3,
        words - 1,
    ] {
        let mut advice = rc.advice.clone();
        advice[at] ^= 1 << (at % 64);
        assert!(!satisfied(&rc, &advice), "advice word {at}");
    }
    // A wrong deferred claim fails the native final check.
    assert_ne!(
        evaluate_claim(&circuit, &point),
        claim.value.value() + Poly192::ONE
    );
    let start = std::time::Instant::now();
    let outer = rc.prove(&config, Vec::new()).unwrap();
    std::println!("outer prove {:?}", start.elapsed());
    let batch = rc.batch().unwrap();
    let public = rc.public(&batch, &rc.revealed);
    let start = std::time::Instant::now();
    let mut challenger = FlockChallenger::<Blake3>::from_hasher(Vec::new(), Blake3);
    crate::verify(&config, &batch, &public, &outer, &mut challenger).unwrap();
    std::println!(
        "outer verify {:?}, peak {:.1} GB",
        start.elapsed(),
        peak_gb()
    );
}

#[test]
#[ignore = "builds a verifier circuit of 2^26 bits (peak 9 GB); run with --release --include-ignored"]
fn a_statement_of_wires_binds_like_a_constant_one() {
    use crate::Gadget;
    use crate::gadgets::blake3::Blake3Compress;
    let config = FlockConfig::default();
    let circuit = Blake3Compress::circuit();
    let rows: Vec<Blake3Compress> = (0..4u32)
        .map(|i| Blake3Compress {
            cv: [i; 8],
            block: [i ^ 0x55; 16],
            counter: 7,
            block_len: 64,
            flags: 11,
        })
        .collect();
    let batch = Batch::new(&circuit, 2).unwrap();
    // Every output word of every instance is public.
    let public = PublicPorts::new(&batch, &[crate::Port::Output(0)], |i, words| {
        let out = crate::Port::Output(0);
        let range = circuit.port(out).unwrap();
        let mut packed = [0u64; 8];
        Blake3Compress::write_output(&compress_native(&rows[i]), &mut packed);
        words[range].copy_from_slice(&packed);
    });
    let witness = Blake3Compress::witness(&circuit, 2, &rows, &rows[0]);
    let mut challenger = FlockChallenger::<Blake3>::from_hasher(Vec::new(), Blake3);
    let proof = crate::prove(&config, &batch, &public, witness, &mut challenger).unwrap();

    let mut arith = WireArith::new();
    let statement = arith.advice_statement(&public);
    let first = statement.bits[0];
    let claim = arith
        .verify_proof(
            &config,
            &batch,
            &public,
            &proof,
            &mut Blake3Mirror::new(&arith, Vec::new()),
            Some(statement),
        )
        .unwrap();
    arith.reveal(&claim.value);
    let rc = arith.finish();
    assert!(satisfied(&rc, &rc.advice));
    // A flipped statement bit changes the digest, so the transcript and the fail bit.
    let mut advice = rc.advice.clone();
    let _ = first;
    advice[0] ^= 1;
    assert!(!satisfied(&rc, &advice));
}

/// The native BLAKE3 compression output of a gadget row.
fn compress_native(row: &crate::gadgets::blake3::Blake3Compress) -> [u32; 16] {
    use crate::Gadget;
    use crate::gadgets::blake3::Blake3Compress;
    let circuit = Blake3Compress::circuit();
    let witness = Blake3Compress::witness(&circuit, 0, core::slice::from_ref(row), row);
    Blake3Compress::decode_output(circuit.output_words(witness.z()))
}

/// `M(point)` by the dense matrices: every row's `A` and `B` entries against every column's equality weight.
fn dense_claim(circuit: &Circuit, point: &[Poly192]) -> Poly192 {
    let k = circuit.k_log();
    let n = 1 << k;
    let (rows, cols) = point[1..].split_at(k);
    let eq = |p: &[Poly192], i: usize| -> Poly192 {
        p.iter()
            .enumerate()
            .map(|(t, &r)| {
                if (i >> (k - 1 - t)) & 1 == 1 {
                    r
                } else {
                    Poly192::ONE + r
                }
            })
            .product()
    };
    let mut total = Poly192::ZERO;
    for j in 0..n {
        // Column `j` alone: the walk gives column `j` of `A` and `B`.
        let unit: Vec<Poly192> = (0..n).map(|s| Poly192::from_bool(s == j)).collect();
        let cj = eq(cols, j);
        circuit.walk_rows(&unit, |s, a, b| {
            total += eq(rows, s) * cj * (a + point[0] * (a + b));
        });
    }
    total
}

#[test]
fn the_streamed_matrix_is_the_dense_one() {
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};
    let mut rng = SmallRng::seed_from_u64(0xB2);
    for _ in 0..4 {
        let circuit = crate::tests::random_circuit(&mut rng, 200);
        let k = circuit.k_log();
        let matrix = Matrix::new(&circuit);
        let points: Vec<Vec<Poly192>> = (0..11)
            .map(|_| (0..1 + 2 * k).map(|_| rng.random()).collect())
            .collect();
        let streamed = matrix.evaluate(&points);
        for (point, value) in points.iter().zip(streamed) {
            assert_eq!(value, dense_claim(&circuit, point));
        }
        // Zero-extended coordinates leave the value unchanged at zero and scale it by eq(high, 0) elsewhere.
        let point = &points[0];
        let extended: Vec<Poly192> = core::iter::once(point[0])
            .chain([Poly192::ZERO])
            .chain(point[1..=k].iter().copied())
            .chain([Poly192::ZERO])
            .chain(point[k + 1..].iter().copied())
            .collect();
        assert_eq!(
            matrix.evaluate(&[extended])[0],
            dense_claim(&circuit, point)
        );
    }
}

#[test]
fn the_row_fold_of_a_form_is_the_dense_one() {
    // Invariant: the backward walk's `h(v)` is the form weighed column by column.
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};
    let mut rng = SmallRng::seed_from_u64(0xF0);
    for _ in 0..4 {
        let circuit = crate::tests::random_circuit(&mut rng, 300);
        let k = circuit.k_log();
        let n = 1 << k;
        let (skip, alpha): (Poly192, Poly192) = (rng.random(), rng.random());
        let inner: Vec<Poly192> = (0..k - 6).map(|_| rng.random()).collect();
        let point: Vec<Poly192> = (0..k - 6).map(|_| rng.random()).collect();
        let h = deferred::form_rows(&Matrix::new(&circuit), skip, &inner, &point, alpha);
        let lagrange = crate::skip::lagrange_weights(&skip, 64);
        let eq = |p: &[Poly192], i: usize| -> Poly192 {
            p.iter()
                .enumerate()
                .map(|(t, &r)| {
                    if (i >> (p.len() - 1 - t)) & 1 == 1 {
                        r
                    } else {
                        Poly192::ONE + r
                    }
                })
                .product()
        };
        for (v, &hv) in h.iter().enumerate() {
            let columns: Vec<Poly192> = (0..n)
                .map(|c| {
                    if c % 64 == v {
                        eq(&point, c / 64)
                    } else {
                        Poly192::ZERO
                    }
                })
                .collect();
            let mut expected = Poly192::ZERO;
            circuit.walk_rows(&columns, |r, a, b| {
                expected += lagrange[r % 64] * eq(&inner, r / 64) * (a + alpha * b);
            });
            assert_eq!(hv, expected, "v = {v}");
        }
    }
}

#[test]
fn two_claims_fold_into_one_the_matrix_satisfies() {
    use p3_verifier_arith::Native;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};
    let mut rng = SmallRng::seed_from_u64(0xF01D);
    let circuit = crate::tests::random_circuit(&mut rng, 300);
    let matrix = Matrix::new(&circuit);
    let n = 1 + 2 * circuit.k_log();
    let claim = |rng: &mut SmallRng| {
        let point: Vec<Poly192> = (0..n).map(|_| rng.random()).collect();
        let value = matrix.evaluate(core::slice::from_ref(&point))[0];
        MatrixClaim { point, value }
    };
    let (first, second) = (claim(&mut rng), claim(&mut rng));
    let q = line_polynomial(&matrix, &first, &second);
    let mut challenger = FlockChallenger::<Blake3>::from_hasher(Vec::new(), Blake3);
    let folded = fold_claims_in(&mut Native::new(), &first, &second, &q, &mut challenger).unwrap();
    assert_eq!(
        matrix.evaluate(core::slice::from_ref(&folded.point))[0],
        folded.value
    );

    // A wrong value is caught by q(1), whatever q the prover sends.
    let mut wrong = second.clone();
    wrong.value += Poly192::ONE;
    let mut challenger = FlockChallenger::<Blake3>::from_hasher(Vec::new(), Blake3);
    assert!(fold_claims_in(&mut Native::new(), &first, &wrong, &q, &mut challenger).is_err());
    // A q through the wrong value leaves a folded claim the matrix refutes.
    let mut forged = q.clone();
    forged[1] += Poly192::ONE;
    let mut challenger = FlockChallenger::<Blake3>::from_hasher(Vec::new(), Blake3);
    let folded =
        fold_claims_in(&mut Native::new(), &first, &wrong, &forged, &mut challenger).unwrap();
    assert_ne!(
        matrix.evaluate(core::slice::from_ref(&folded.point))[0],
        folded.value
    );
}

/// A BLAKE3-transcript proof of four BLAKE3 compressions whose outputs are public, seeded by `seed`.
fn public_compression_proof(
    seed: u32,
    config: &FlockConfig,
) -> (Circuit, PublicPorts, FlockProof<Blake3>) {
    use crate::Gadget;
    use crate::gadgets::blake3::Blake3Compress;
    let circuit = Blake3Compress::circuit();
    let rows: Vec<Blake3Compress> = (0..4u32)
        .map(|i| Blake3Compress {
            cv: [i ^ seed; 8],
            block: [i.wrapping_mul(seed) ^ 0x55; 16],
            counter: u64::from(seed),
            block_len: 64,
            flags: 11,
        })
        .collect();
    let batch = Batch::new(&circuit, 2).unwrap();
    let public = PublicPorts::new(&batch, &[crate::Port::Output(0)], |i, words| {
        let range = circuit.port(crate::Port::Output(0)).unwrap();
        let mut packed = [0u64; 8];
        Blake3Compress::write_output(&compress_native(&rows[i]), &mut packed);
        words[range].copy_from_slice(&packed);
    });
    let witness = Blake3Compress::witness(&circuit, 2, &rows, &rows[0]);
    let mut challenger = FlockChallenger::<Blake3>::from_hasher(Vec::new(), Blake3);
    let proof = crate::prove(config, &batch, &public, witness, &mut challenger).unwrap();
    (circuit, public, proof)
}

/// Print a node's figures.
fn report(level: usize, node: &NodeProof, elapsed: std::time::Duration) {
    std::println!(
        "level {level}: instance 2^{} bits, {} ANDs, {} gates, {} ops, {} terms, {} advice words, built and proved in {elapsed:.1?}",
        node.circuit.circuit.k_log(),
        node.circuit.ands,
        node.circuit.gates,
        node.circuit.ops,
        node.circuit.terms,
        node.circuit.advice.len(),
    );
    for cost in &node.circuit.costs {
        std::println!(
            "  {:<17} {:>11} ANDs {:>12} gates",
            cost.component,
            cost.ands,
            cost.gates
        );
    }
}

/// Peak resident memory so far, in GB.
fn peak_gb() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let kb: f64 = status
        .lines()
        .find(|l| l.starts_with("VmHWM"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    kb / (1024.0 * 1024.0)
}

#[test]
#[ignore = "builds and proves node circuits of 2^27 bits per instance (peak 16 GB); run with --release --include-ignored"]
fn a_tree_of_proofs_folds_its_claims_and_reaches_a_fixed_point() {
    use alloc::sync::Arc;
    use std::time::Instant;
    let mut config = FlockConfig::default();
    config.pcs.log_inv_rate = 4;
    // Every node instance padded to one shape, so a node verifying nodes is one circuit at every level.
    let layout = NodeLayout {
        slots: 3,
        k_max: 30,
        advice_words: Some(24_000),
        k_log: Some(27),
    };
    let leaves: Vec<_> = (1..=4)
        .map(|seed| public_compression_proof(seed, &config))
        .collect();
    let leaf_matrix = Arc::new(Matrix::new(&leaves[0].0));
    let children: Vec<Child<'_>> = leaves
        .iter()
        .map(|(circuit, public, proof)| Child::Leaf {
            config: &config,
            batch: Batch::new(circuit, 2).unwrap(),
            public,
            proof,
        })
        .collect();
    let mut level1: Vec<NodeProof> = Vec::new();
    for pair in children.chunks(2) {
        let start = Instant::now();
        // Both level-1 nodes have one circuit: the second reuses the first's, compiled once.
        let reuse = level1.first().map(|n| &n.circuit.circuit);
        let node = build_node(
            &layout,
            [&pair[0], &pair[1]],
            &[Arc::clone(&leaf_matrix)],
            reuse,
        )
        .unwrap()
        .prove(&config)
        .unwrap();
        report(1, &node, start.elapsed());
        level1.push(node);
    }
    std::println!("peak {:.1} GB", peak_gb());
    let digests: Vec<[u8; 32]> = leaves.iter().map(|l| *l.1.digest()).collect();
    let expected1 = [
        Blake3.hash_slice(&[digests[0], digests[1]].concat()),
        Blake3.hash_slice(&[digests[2], digests[3]].concat()),
    ];
    let start = Instant::now();
    verify_root(
        &layout,
        &config,
        &level1[0],
        &[Arc::clone(&leaf_matrix)],
        &expected1[0],
    )
    .unwrap();
    std::println!(
        "level 1 final check {:?}, peak {:.1} GB",
        start.elapsed(),
        peak_gb()
    );

    // Level 2: two different level-1 proofs, so the leaf slot and the level-1 slot fold real claims.
    let level1_matrix = Arc::new(Matrix::new(&level1[0].circuit.circuit));
    let matrices = [Arc::clone(&leaf_matrix), Arc::clone(&level1_matrix)];
    let start = Instant::now();
    let node = build_node(
        &layout,
        [
            &Child::Node {
                config: &config,
                node: &level1[0],
            },
            &Child::Node {
                config: &config,
                node: &level1[1],
            },
        ],
        &matrices,
        None,
    )
    .unwrap();
    // One level-1 matrix serves both: they are one circuit.
    level1.truncate(1);
    let level2 = node.prove(&config).unwrap();
    report(2, &level2, start.elapsed());
    std::println!("peak {:.1} GB", peak_gb());
    let expected2 = Blake3.hash_slice(&[expected1[0], expected1[1]].concat());
    assert_eq!(
        expected2,
        leaf_digest(&leaves.iter().map(|l| &l.1).collect::<Vec<_>>())
    );
    let start = Instant::now();
    verify_root(&layout, &config, &level2, &matrices, &expected2).unwrap();
    std::println!(
        "level 2 final check {:?}, peak {:.1} GB",
        start.elapsed(),
        peak_gb()
    );
    std::println!(
        "level 2 proof {} KiB",
        postcard::to_allocvec(&level2.proof).unwrap().len() / 1024
    );

    // Level 3 verifies level-2 proofs on the level-2 circuit, which the build checks by digest: the tree has a fixed point.
    drop(level1);
    let level2_matrix = Arc::new(Matrix::new(&level2.circuit.circuit));
    let start = Instant::now();
    let child = Child::Node {
        config: &config,
        node: &level2,
    };
    let level3 = build_node(
        &layout,
        [&child, &child],
        &[Arc::clone(&leaf_matrix), level1_matrix, level2_matrix],
        Some(&level2.circuit.circuit),
    )
    .unwrap();
    std::println!(
        "level 3 instances built on the level-2 circuit in {:?}, peak {:.1} GB",
        start.elapsed(),
        peak_gb()
    );
    assert_eq!(
        level3.circuit.circuit.digest(),
        level2.circuit.circuit.digest()
    );
}
