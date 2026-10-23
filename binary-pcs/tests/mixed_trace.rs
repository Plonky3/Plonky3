//! Mixed source columns packed in GF(2^64), opened at GF(2^192) points through WHIR.

use p3_binary_field::{BinaryChallenger, Poly64, Poly192};
use p3_binary_pcs::whir::{
    BinaryWhirDomain, BinaryWhirProfile, BooleanWhirPcs, BooleanWhirProver, recommended_cap_height,
};
use p3_binary_pcs::{
    BooleanTraceCommitment, BooleanTraceCommitmentError, MixedTraceCommitment, committed_shapes,
};
use p3_challenger::HashChallenger;
use p3_commit::MultilinearPcs;
use p3_field::PrimeCharacteristicRing;
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::layout::{Table, plan_stacked_layout};
use p3_sumcheck::{OpeningBatch, OpeningProtocol, PrescribedPointPcs, TableShape, TableSpec};
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type F = Poly64;
type EF = Poly192;
type Hash = SerializingHasher<Keccak256Hash>;
type Compress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
type Mmcs = MerkleTreeMmcs<F, u8, Hash, Compress, 2, 32>;
type Challenger = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;
type Domain = BinaryWhirDomain<F>;
type Backend = BooleanWhirPcs<F, EF, Domain, Mmcs, Challenger>;
type Scheme = MixedTraceCommitment<EF, Backend, F>;

const fn challenger() -> Challenger {
    Challenger::from_hasher(Vec::new(), Keccak256Hash)
}

fn scheme(shapes: &[TableShape], bits: &[usize]) -> Scheme {
    let (arity, _) = plan_stacked_layout(&committed_shapes::<F>(shapes, bits));
    let domain = Domain::default();
    let config = BinaryWhirProfile::proven_list_decoding(100, 2, 3)
        .config::<EF, F, Challenger, _>(arity - 6, &domain)
        .unwrap();
    let merkle = Mmcs::new(
        Hash::new(Keccak256Hash),
        Compress::new(Keccak256Hash),
        recommended_cap_height(&config),
    );
    let inner = Backend::new(BooleanWhirProver::new(config, domain, merkle), arity).unwrap();
    MixedTraceCommitment::new(
        BooleanTraceCommitment::from_commitment(inner),
        bits.to_vec(),
    )
}

fn tables(shapes: &[TableShape], bits: &[usize]) -> Vec<Table<F>> {
    let mut rng = SmallRng::seed_from_u64(0x6419_2000);
    shapes
        .iter()
        .zip(bits)
        .map(|(shape, &bits)| {
            let rows = 1 << shape.num_variables();
            let cells = (0..shape.width())
                .flat_map(|column| {
                    (0..rows)
                        .map(|_| {
                            if column < bits {
                                F::from_bool(rng.random())
                            } else {
                                rng.random()
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            Table::new(RowMajorMatrix::new(cells, rows))
        })
        .collect()
}

#[test]
fn source_values_and_successors_are_opened_in_the_extension() {
    // Changing the dense split to use the challenge dimension or omitting the source
    // basis embedding breaks the direct multilinear evaluations checked below.
    // Heights straddle the packing boundary, and a partial last word is exercised too.
    let shapes = [TableShape::new(8, 3), TableShape::new(4, 3)];
    let bits = [1, 2];
    let pcs = scheme(&shapes, &bits);
    let tables = tables(&shapes, &bits);
    let protocol = OpeningProtocol::new(vec![
        TableSpec::new(shapes[0], vec![OpeningBatch::new(vec![2, 0], vec![1, 0])]),
        TableSpec::new(shapes[1], vec![OpeningBatch::new(vec![1, 2], vec![2, 1])]),
    ]);
    let mut rng = SmallRng::seed_from_u64(0x192);
    let points = vec![
        Point::<EF>::rand(&mut rng, 8),
        Point::<EF>::rand(&mut rng, 4),
    ];
    let mut prover = challenger();
    let (commitment, data) = pcs.commit(tables.clone(), &mut prover).unwrap();
    assert_eq!(data.table(0).shape(), shapes[0]);
    assert_eq!(
        data.table(0).poly(2).as_slice(),
        tables[0].poly(2).as_slice()
    );
    let proof = pcs.open_at(data, &protocol, &points, &mut prover).unwrap();
    let mut verifier = challenger();
    pcs.observe_commitment(&commitment, &mut verifier);
    let evals = pcs
        .verify_at(&commitment, &proof, &protocol, &points, &mut verifier)
        .unwrap();
    for (((table_index, batch), point), actual) in protocol.iter_openings().zip(&points).zip(evals)
    {
        let table = &tables[table_index];
        for (&column, &actual) in batch.current().iter().zip(actual.current()) {
            let values = table.poly(column).as_slice().to_vec();
            assert_eq!(actual, Poly::new(values).eval_base(point));
        }
        for (&column, &actual) in batch.next().iter().zip(actual.next()) {
            let values = table.poly(column).as_slice().to_vec();
            let shifted: Vec<_> = (0..values.len())
                .map(|row| values[(row + 1).min(values.len() - 1)])
                .collect();
            assert_eq!(actual, Poly::new(shifted).eval_base(point));
        }
    }

    // Each table's widened batch contains current then successor coordinates.
    // Touch bits and the highest dense coordinate, in both views and both tables.
    for index in [0, 64, 129, 193, 258, 323, 324, 389] {
        let mut tampered = proof.clone();
        tampered.values[index] += EF::ONE;
        let mut verifier = challenger();
        pcs.observe_commitment(&commitment, &mut verifier);
        assert!(
            pcs.verify_at(&commitment, &tampered, &protocol, &points, &mut verifier)
                .is_err()
        );
    }
}

#[test]
fn sampled_mixed_openings_bind_the_commitment() {
    let shapes = [TableShape::new(6, 3)];
    let bits = [1];
    let pcs = scheme(&shapes, &bits);
    for next in [vec![], vec![0, 1, 2]] {
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            shapes[0],
            vec![OpeningBatch::new(vec![0, 1, 2], next)],
        )]);
        let mut prover = challenger();
        let (commitment, data) = pcs.commit(tables(&shapes, &bits), &mut prover).unwrap();
        let proof = pcs.open(data, protocol.clone(), &mut prover).unwrap();
        pcs.verify(&commitment, &proof, &mut challenger(), protocol)
            .unwrap();
    }
}

#[test]
fn security_prices_the_widened_source_coordinates_at_the_challenge_width() {
    // One bit + two Poly64 cells is 129 columns: a single padded 256-column block.
    // The block has 8 row and 8 column variables, with 6 absorbed by packing.
    let shapes = [TableShape::new(8, 3)];
    let pcs = scheme(&shapes, &[1]);
    for next in [vec![], vec![1]] {
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            shapes[0],
            vec![OpeningBatch::new(vec![0, 2], next.clone())],
        )]);
        let all: Vec<_> = (0..129).collect();
        let expanded = OpeningProtocol::new(vec![TableSpec::new(
            TableShape::new(8, 129),
            vec![OpeningBatch::new(
                all.clone(),
                if next.is_empty() { vec![] } else { all },
            )],
        )]);
        let security =
            PrescribedPointPcs::<EF, Challenger>::prescribed_security(&pcs, &protocol).unwrap();
        let inner =
            PrescribedPointPcs::<EF, Challenger>::prescribed_security(pcs.inner(), &expanded)
                .unwrap();
        assert_eq!(security.terms, inner.terms);
        assert_eq!(security.log2_max_candidates, inner.log2_max_candidates);
        let candidates = security.candidates().unwrap();
        let tensors = if next.is_empty() { 1 } else { 3 };
        let expected =
            p3_security::multilinear::bit_ring_switch_tensors_term(1, tensors, 8, 10, 192)
                .over_candidates(candidates);
        let actual = security
            .terms
            .iter()
            .find(|term| term.label == expected.label)
            .unwrap();
        assert_eq!(actual, &expected);
        assert!(actual.bits.bits() > 128.0);
        let expected = p3_security::multilinear::column_batch_term(
            if next.is_empty() { 1 } else { 2 },
            8,
            192,
        )
        .over_candidates(candidates);
        let actual = security
            .terms
            .iter()
            .find(|term| term.label == expected.label)
            .unwrap();
        assert_eq!(actual, &expected);
        assert_eq!(security.error(), inner.error());
        assert!(actual.bits.bits() > 128.0);
    }
}

#[test]
fn a_dense_width_whose_coordinates_wrap_is_refused() {
    // Invariant: a protocol names exactly the table the commitment holds.
    //
    // Fixture state: one table of 2^6 rows, 1 bit column and 2 dense Poly64 columns.
    //
    //     committed width = 1 + 2 * 64 = 129
    //
    // A forged table of 2^58 + 3 columns has 2^58 + 2 dense columns.
    //
    //     1 + (2^58 + 2) * 64 = 2^64 + 129, which wraps to 129
    //
    // Wrapped, it commits the same shape, and its last column aliases dense column 2.
    let shapes = [TableShape::new(6, 3)];
    let bits = [1];
    let pcs = scheme(&shapes, &bits);
    let protocol = OpeningProtocol::new(vec![TableSpec::new(
        shapes[0],
        vec![OpeningBatch::new(vec![0, 1, 2], vec![])],
    )]);
    let points = vec![Point::<EF>::rand(&mut SmallRng::seed_from_u64(1), 6)];
    let mut prover = challenger();
    let (commitment, data) = pcs.commit(tables(&shapes, &bits), &mut prover).unwrap();
    let proof = pcs.open_at(data, &protocol, &points, &mut prover).unwrap();

    // Mutation: the same proof, against the forged table.
    let wide = (1usize << 58) + 3;
    let forged = OpeningProtocol::new(vec![TableSpec::new(
        TableShape::new(6, wide),
        vec![OpeningBatch::new(vec![0, 1, wide - 1], vec![])],
    )]);
    let mut verifier = challenger();
    pcs.observe_commitment(&commitment, &mut verifier);
    assert!(matches!(
        pcs.verify_at(&commitment, &proof, &forged, &points, &mut verifier),
        Err(BooleanTraceCommitmentError::InvalidOpeningProtocol)
    ));
}
