use alloc::vec;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_blake3::Blake3;
use p3_challenger::{CanSample, DuplexChallenger};
use p3_commit::MultilinearPcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::{ExtensionField, Field, PackedValue, PrimeCharacteristicRing, dot_product};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::{Poly, PolyMaybePacked};
use p3_sumcheck::SumcheckData;
use p3_sumcheck::constraints::statement::SelectStatement;
use p3_sumcheck::product_polynomial::ProductPolynomial;
use p3_sumcheck::strategy::{SumcheckProver, VariableOrder};
use p3_symmetric::{CryptographicHasher, PaddingFreeSponge, TruncatedPermutation};
use p3_util::log2_strict_usize;
use rand::rngs::{SmallRng, StdRng};
use rand::{RngExt, SeedableRng};

use crate::parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption};
use crate::pcs::zk::{HidingWhirPcs, ZkParameters, ZkWhirConfig};
use crate::utils::par_add_scaled_powers_packed;

#[test]
fn packed_constraint_delta_preserves_round_messages_both_orders() {
    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type EP = <EF as ExtensionField<F>>::ExtensionPacking;
    type Perm = Poseidon2BabyBear<16>;
    type Challenger = DuplexChallenger<F, Perm, 16, 8>;

    let mut rng = SmallRng::seed_from_u64(913);
    let k_pack = log2_strict_usize(<F as Field>::Packing::WIDTH);
    for n in [k_pack, k_pack + 3] {
        let evals = Poly::<EF>::rand(&mut rng, n);
        let weights = Poly::<EF>::rand(&mut rng, n);
        let ood_points = [rng.random::<EF>(), rng.random::<EF>()];
        let query_points = [F::from_u64(3), F::from_u64(7)];
        let combination: EF = rng.random();
        let coeffs = combination.shifted_powers(combination).collect_n(4);
        let mut packed_delta = Poly::new(EP::zero_vec(1 << (n - k_pack)));
        let mut select = SelectStatement::<F, EF>::initialize(n);
        for point in query_points {
            select.add_constraint(point, EF::ZERO);
        }
        let mut unused_sum = EF::ZERO;
        select.combine_packed(&mut packed_delta, &mut unused_sum, combination, 3);
        for (&point, &coeff) in ood_points.iter().zip(&coeffs[..2]) {
            par_add_scaled_powers_packed::<F, EF>(packed_delta.as_mut_slice(), point, coeff, 3);
        }
        let mut scalar_delta = EF::zero_vec(1 << n);
        for (point, coeff) in ood_points
            .into_iter()
            .chain(query_points.map(EF::from))
            .zip(coeffs)
        {
            let mut term = coeff;
            for dst in &mut scalar_delta {
                *dst += term;
                term *= point;
            }
        }
        assert_eq!(packed_delta.unpack::<F, EF>().as_slice(), scalar_delta);
        let sum = dot_product::<EF, _, _>(evals.iter().copied(), weights.iter().copied());
        let sum_delta =
            dot_product::<EF, _, _>(evals.iter().copied(), scalar_delta.iter().copied());
        for order in [VariableOrder::Prefix, VariableOrder::Suffix] {
            let mut packed = SumcheckProver::new(
                ProductPolynomial::<F, EF>::new_packed(
                    order,
                    evals.pack::<F, EF>(),
                    weights.pack::<F, EF>(),
                ),
                sum,
            );
            let mut scalar = SumcheckProver::new(
                ProductPolynomial::<F, EF>::new_unpacked(order, evals.clone(), weights.clone()),
                sum,
            );
            packed
                .accumulate_claim_view(PolyMaybePacked::Packed(packed_delta.as_view()), sum_delta);
            scalar.accumulate_claim(&scalar_delta, sum_delta);
            let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(914));
            let mut packed_challenger = Challenger::new(perm.clone());
            let mut scalar_challenger = Challenger::new(perm);
            let mut packed_data = SumcheckData::default();
            let mut scalar_data = SumcheckData::default();
            let packed_point = packed.compute_sumcheck_polynomials(
                &mut packed_data,
                &mut packed_challenger,
                n,
                0,
                None,
            );
            let scalar_point = scalar.compute_sumcheck_polynomials(
                &mut scalar_data,
                &mut scalar_challenger,
                n,
                0,
                None,
            );
            assert_eq!(packed_point, scalar_point);
            assert_eq!(
                packed_data.polynomial_evaluations,
                scalar_data.polynomial_evaluations
            );
            assert_eq!(packed.claimed_sum(), scalar.claimed_sum());
            assert_eq!(packed.evals(), scalar.evals());
        }
    }
}

/// Pins the scalar-staging prover's complete wire output and transcript state.
#[test]
fn packed_staging_preserves_proof_bytes() {
    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Hash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type Compress = TruncatedPermutation<Perm, 2, 8, 16>;
    type Challenger = DuplexChallenger<F, Perm, 16, 8>;
    type PackedF = <F as Field>::Packing;
    type Mmcs = MerkleTreeMmcs<PackedF, PackedF, Hash, Compress, 2, 8>;

    let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1));
    let config = ZkWhirConfig::<EF, F, Challenger>::new(
        12,
        ProtocolParameters {
            security_level: 32,
            pow_bits: 0,
            round_log_inv_rates: vec![],
            folding_factor: FoldingFactor::Constant(4),
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 2,
        },
        ZkParameters {
            ell_zk: 4,
            mask_log_inv_rate: 1,
        },
    )
    .unwrap();
    let pcs = HidingWhirPcs::new(
        config,
        Radix2DFTSmallBatch::<F>::default(),
        Mmcs::new(Hash::new(perm.clone()), Compress::new(perm.clone()), 0),
        StdRng::seed_from_u64(741),
    );
    let mut rng = SmallRng::seed_from_u64(742);
    let witness = Poly::<F>::rand(&mut rng, 12);
    let points = vec![
        Point::<EF>::rand(&mut rng, 12),
        Point::<EF>::rand(&mut rng, 12),
    ];
    let mut challenger = Challenger::new(perm.clone());
    let (commitment, data) = pcs.commit(witness, &mut challenger).unwrap();
    let proof = pcs.open(data, points.clone(), &mut challenger).unwrap();
    assert!(!proof.rounds.is_empty());
    let bytes = postcard::to_allocvec(&(
        &commitment,
        &proof,
        CanSample::<EF>::sample(&mut challenger),
    ))
    .unwrap();
    assert_eq!(
        Blake3.hash_slice(&bytes),
        [
            108, 222, 73, 75, 48, 250, 234, 23, 70, 31, 116, 103, 200, 33, 224, 31, 8, 25, 154,
            157, 69, 147, 155, 6, 25, 5, 68, 68, 176, 207, 121, 141
        ]
    );
    pcs.verify(&commitment, &proof, &mut Challenger::new(perm), points)
        .unwrap();
}
