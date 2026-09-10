use itertools::{Itertools, izip};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{CanObserve, DuplexChallenger, FieldChallenger};
use p3_commit::testing::assert_pcs_opening_contract;
use p3_commit::{ExtensionMmcs, Pcs, PolynomialSpace, UnivariateStarkPcs};
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};

fn seeded_rng() -> impl Rng {
    SmallRng::seed_from_u64(0)
}

fn do_test_fri_pcs<Val, Challenge, Challenger, P>(
    (pcs, challenger): &(P, Challenger),
    log_degrees_by_round: &[&[usize]],
) where
    P: Pcs<Challenge, Challenger>,
    P::Domain: PolynomialSpace<Val = Val>,
    Val: Field,
    StandardUniform: Distribution<Val>,
    Challenge: ExtensionField<Val>,
    Challenger: Clone + CanObserve<P::Commitment> + FieldChallenger<Val>,
{
    let num_rounds = log_degrees_by_round.len();
    let mut rng = seeded_rng();

    let mut p_challenger = challenger.clone();

    let domains_and_polys_by_round = log_degrees_by_round
        .iter()
        .map(|log_degrees| {
            log_degrees
                .iter()
                .map(|&log_degree| {
                    let d = 1 << log_degree;
                    // random width 5-15
                    let width = 5 + rng.random_range(0..=10);
                    (
                        pcs.natural_domain_for_degree(d),
                        RowMajorMatrix::<Val>::rand(&mut rng, d, width),
                    )
                })
                .collect_vec()
        })
        .collect_vec();

    let (commits_by_round, data_by_round): (Vec<_>, Vec<_>) = domains_and_polys_by_round
        .iter()
        .map(|domains_and_polys| pcs.commit(domains_and_polys.iter().cloned()))
        .unzip();
    assert_eq!(commits_by_round.len(), num_rounds);
    assert_eq!(data_by_round.len(), num_rounds);
    p_challenger.observe_slice(&commits_by_round);

    let zeta: Challenge = p_challenger.sample_algebra_element();

    let points_by_round = log_degrees_by_round
        .iter()
        .map(|log_degrees| vec![vec![zeta]; log_degrees.len()])
        .collect_vec();
    let data_and_points = data_by_round
        .iter()
        .zip(points_by_round)
        .map(Into::into)
        .collect();
    let (opening_by_round, proof) = pcs.open(data_and_points, &mut p_challenger);
    assert_eq!(opening_by_round.len(), num_rounds);

    // Verify the proof.
    let mut v_challenger = challenger.clone();
    v_challenger.observe_slice(&commits_by_round);
    let verifier_zeta: Challenge = v_challenger.sample_algebra_element();
    assert_eq!(verifier_zeta, zeta);

    let commits_and_claims_by_round = izip!(
        commits_by_round,
        domains_and_polys_by_round,
        opening_by_round
    )
    .map(|(commit, domains_and_polys, openings)| {
        let claims = domains_and_polys
            .iter()
            .zip(openings)
            .map(|((domain, _), mat_openings)| (*domain, vec![(zeta, mat_openings[0].clone())]))
            .collect_vec();
        (commit, claims)
    })
    .collect_vec();
    assert_eq!(commits_and_claims_by_round.len(), num_rounds);

    pcs.verify(
        commits_and_claims_by_round
            .into_iter()
            .map(Into::into)
            .collect(),
        &proof,
        &mut v_challenger,
    )
    .unwrap();
}

// Add distinct constants to point-dependent basis polynomials, so neither point nor
// matrix reordering can accidentally satisfy the expected openings.
fn shared_opening_contract<Val, Challenge, Challenger, P>(
    (pcs, challenger): &(P, Challenger),
    basis: impl Fn(Val) -> [Val; 3],
    extension_basis: impl Fn(Challenge) -> [Challenge; 3],
) where
    P: Pcs<Challenge, Challenger>,
    P::Domain: PolynomialSpace<Val = Val>,
    Val: Field,
    Challenge: ExtensionField<Val>,
    Challenger: Clone + CanObserve<P::Commitment> + FieldChallenger<Val>,
{
    let constants = [vec![vec![2, 7], vec![11]], vec![vec![19, 23, 29]]];
    let heights = [vec![8, 16], vec![8]];
    let rounds: Vec<_> = constants
        .iter()
        .zip(heights)
        .map(|(round, heights)| {
            round
                .iter()
                .zip(heights)
                .map(|(columns, height)| {
                    let domain = pcs.natural_domain_for_degree(height);
                    let mut x = domain.first_point();
                    let mut values = Vec::new();
                    for _ in 0..height {
                        values.extend(
                            columns
                                .iter()
                                .zip(basis(x))
                                .map(|(&v, b)| b + Val::from_u64(v)),
                        );
                        x = domain.next_point(x).unwrap();
                    }
                    (domain, RowMajorMatrix::new(values, columns.len()))
                })
                .collect()
        })
        .collect();
    assert_pcs_opening_contract(pcs, challenger, &rounds, |round, matrix, point| {
        constants[round][matrix]
            .iter()
            .zip(extension_basis(point))
            .map(|(&v, b)| b + Challenge::from_u64(v))
            .collect()
    });
}

// Set it up so we create tests inside a module for each pcs, so we get nice error reports
// specific to a failing PCS.
macro_rules! make_tests_for_pcs {
    ($p:expr) => {
        #[test]
        fn shared_opening_contract() {
            $crate::shared_opening_contract(&$p, super::contract_basis, super::contract_basis);
        }

        #[test]
        fn single() {
            let p = $p;
            for i in 3..6 {
                $crate::do_test_fri_pcs(&p, &[&[i]]);
            }
        }

        #[test]
        fn many_equal() {
            let p = $p;
            for i in 2..6 {
                $crate::do_test_fri_pcs(&p, &[&[i; 5]]);
                println!("{i} ok");
            }
        }

        #[test]
        fn many_different() {
            let p = $p;
            for i in 2..5 {
                let degrees = (3..3 + i).collect::<Vec<_>>();
                $crate::do_test_fri_pcs(&p, &[&degrees]);
            }
        }

        #[test]
        fn many_different_rev() {
            let p = $p;
            for i in 2..5 {
                let degrees = (3..3 + i).rev().collect::<Vec<_>>();
                $crate::do_test_fri_pcs(&p, &[&degrees]);
            }
        }

        #[test]
        fn multiple_rounds() {
            let p = $p;
            $crate::do_test_fri_pcs(&p, &[&[3]]);
            $crate::do_test_fri_pcs(&p, &[&[3], &[3]]);
            $crate::do_test_fri_pcs(&p, &[&[3], &[2]]);
            $crate::do_test_fri_pcs(&p, &[&[2], &[3]]);
            $crate::do_test_fri_pcs(&p, &[&[3, 4], &[3, 4]]);
            $crate::do_test_fri_pcs(&p, &[&[4, 2], &[4, 2]]);
            $crate::do_test_fri_pcs(&p, &[&[2, 2], &[3, 3]]);
            $crate::do_test_fri_pcs(&p, &[&[3, 3], &[2, 2]]);
            $crate::do_test_fri_pcs(&p, &[&[2], &[3, 3]]);
        }
    };
}

mod babybear_fri_pcs {
    use super::*;

    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;

    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;

    type ValMmcs =
        MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 2, 8>;
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;

    type Dft = Radix2DitParallel<Val>;
    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
    type MyPcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;

    fn contract_basis<F: Field>(point: F) -> [F; 3] {
        [point, point.square(), point.exp_const_u64::<3>()]
    }

    fn get_pcs(log_blowup: usize) -> (MyPcs, Challenger) {
        let perm = Perm::new_from_rng_128(&mut seeded_rng());
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());

        let val_mmcs = ValMmcs::new(hash, compress, 0);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

        let fri_params = FriParameters {
            log_blowup,
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 10,
            batch_proof_of_work_bits: 0,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 8,
            mmcs: challenge_mmcs,
        };

        let pcs = MyPcs::new(Dft::default(), val_mmcs, fri_params);
        (pcs, Challenger::new(perm))
    }

    fn get_pcs_high_arity(log_blowup: usize) -> (MyPcs, Challenger) {
        let perm = Perm::new_from_rng_128(&mut seeded_rng());
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());

        let val_mmcs = ValMmcs::new(hash, compress, 0);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

        let fri_params = FriParameters {
            log_blowup,
            log_final_poly_len: 0,
            max_log_arity: 2,
            num_queries: 10,
            batch_proof_of_work_bits: 0,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 8,
            mmcs: challenge_mmcs,
        };

        let pcs = MyPcs::new(Dft::default(), val_mmcs, fri_params);
        (pcs, Challenger::new(perm))
    }

    mod blowup_1 {
        make_tests_for_pcs!(super::get_pcs(1));
    }
    mod blowup_2 {
        make_tests_for_pcs!(super::get_pcs(2));
    }
    mod high_arity_blowup_1 {
        make_tests_for_pcs!(super::get_pcs_high_arity(1));
    }

    #[test]
    fn shared_contract_finds_point_dependent_matrix_in_later_commitment() {
        let (pcs, challenger) = get_pcs(1);
        let domain = <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 8);
        let mut x = domain.first_point();
        let mut values = Vec::new();
        for _ in 0..8 {
            values.extend([x + Val::from_u64(2), x.square() + Val::from_u64(7)]);
            x = domain.next_point(x).unwrap();
        }
        let matrix = RowMajorMatrix::new(values, 2);
        assert_pcs_opening_contract(
            &pcs,
            &challenger,
            &[
                vec![(
                    domain,
                    RowMajorMatrix::new([Val::from_u64(2), Val::from_u64(7)].repeat(8), 2),
                )],
                vec![(domain, matrix.clone()), (domain, matrix)],
            ],
            |round, _, point| {
                if round == 0 {
                    vec![Challenge::from_u64(2), Challenge::from_u64(7)]
                } else {
                    vec![
                        point + Challenge::from_u64(2),
                        point.square() + Challenge::from_u64(7),
                    ]
                }
            },
        );
    }

    #[test]
    #[should_panic(expected = "fixture needs a point-dependent column")]
    fn shared_contract_rejects_constant_fixture() {
        let (pcs, challenger) = get_pcs(1);
        let domain = <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 8);
        let matrix = RowMajorMatrix::new(vec![Val::from_u64(2); 8], 1);
        assert_pcs_opening_contract(
            &pcs,
            &challenger,
            &[vec![(domain, matrix.clone()), (domain, matrix)]],
            |_, _, _| vec![Challenge::from_u64(2)],
        );
    }

    #[test]
    fn extrapolation() {
        use p3_dft::TwoAdicSubgroupDft;
        use p3_matrix::Matrix;

        let (pcs, _) = get_pcs(1);
        let mut rng = seeded_rng();

        let log_degree = 4;
        let degree = 1 << log_degree;
        let width = 3;
        let trace = RowMajorMatrix::<Val>::rand(&mut rng, degree, width);

        let domain = <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, degree);
        let (_, data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, [(domain, trace.clone())]);

        let disjoint_domain = domain.create_disjoint_domain(degree);
        let evals = <MyPcs as UnivariateStarkPcs<Challenge, Challenger>>::get_evaluations_on_domain(
            &pcs,
            &data,
            0,
            disjoint_domain,
        );
        let evals = evals.to_row_major_matrix();

        let dft = Dft::default();
        let coeffs = dft.idft_batch(trace);
        let expected = dft
            .coset_dft_batch(coeffs, disjoint_domain.shift())
            .to_row_major_matrix();

        assert_eq!(evals, expected);
    }
}

mod m31_fri_pcs {
    use core::marker::PhantomData;

    use p3_challenger::{HashChallenger, SerializingChallenger32};
    use p3_circle::CirclePcs;
    use p3_keccak::Keccak256Hash;
    use p3_mersenne_31::Mersenne31;
    use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};

    use super::*;

    type Val = Mersenne31;
    type Challenge = BinomialExtensionField<Mersenne31, 3>;

    type ByteHash = Keccak256Hash;
    type FieldHash = SerializingHasher<ByteHash>;

    type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;

    type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, MyCompress, 2, 32>;

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;

    type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;

    type Pcs = CirclePcs<Val, ValMmcs, ChallengeMmcs>;

    fn contract_basis<F: Field>(point: F) -> [F; 3] {
        // Domain iteration and opening points use the projective-line parameter t.
        // The circle basis contains y, x and xy, not powers of t.
        let inv_denom = (F::ONE + point.square()).inverse();
        let x = (F::ONE - point.square()) * inv_denom;
        let y = point.double() * inv_denom;
        [y, x, x * y]
    }

    fn get_pcs(log_blowup: usize) -> (Pcs, Challenger) {
        let byte_hash = ByteHash {};
        let field_hash = FieldHash::new(byte_hash);
        let compress = MyCompress::new(byte_hash);
        let val_mmcs = ValMmcs::new(field_hash, compress, 0);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        let fri_params = FriParameters {
            log_blowup,
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 10,
            batch_proof_of_work_bits: 0,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 8,
            mmcs: challenge_mmcs,
        };
        let pcs = Pcs {
            mmcs: val_mmcs,
            fri_params,
            _phantom: PhantomData,
        };
        (pcs, Challenger::from_hasher(vec![], byte_hash))
    }

    mod blowup_1 {
        make_tests_for_pcs!(super::get_pcs(1));
    }
    mod blowup_2 {
        make_tests_for_pcs!(super::get_pcs(2));
    }
}
