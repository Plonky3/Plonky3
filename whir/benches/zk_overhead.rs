//! Benchmark: ZK overhead in WHIR PCS.
//!
//! Measures prover time, verifier time, and proof size for `zk: true` vs
//! `zk: false`. The paper claims `1 + o(1)` overhead.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::MultilinearPcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::poly::Poly;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_whir::fiat_shamir::domain_separator::DomainSeparator;
use p3_whir::parameters::{ProtocolParameters, SecurityAssumption, WhirConfig};
use p3_whir::pcs::HidingWhirPcs;
use p3_whir::pcs::prover::WhirProver;
use p3_whir::sumcheck::layout::{Layout, SuffixProver, Table};
use p3_whir::sumcheck::{OpeningProtocol, TableShape, TableSpec};
use rand::SeedableRng;
use rand::rngs::SmallRng;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;
type PackedF = <F as Field>::Packing;
type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;
type MyDft = Radix2DFTSmallBatch<F>;
type L = SuffixProver<F, EF>;
type Whir = WhirProver<EF, F, MyDft, MyMmcs, MyChallenger, L>;

fn challenger() -> MyChallenger {
    let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1));
    MyChallenger::new(perm)
}

fn mmcs() -> MyMmcs {
    let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1));
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm);
    MyMmcs::new(hash, compress, 0)
}

fn bench_prover(c: &mut Criterion) {
    let mut group = c.benchmark_group("whir_zk_prover");

    for num_vars in [12, 16] {
        let folding_factor = p3_whir::parameters::FoldingFactor::Constant(4);
        let (num_rounds, _) = folding_factor.compute_number_of_rounds(num_vars);
        let round_log_inv_rates = {
            let mut rates = Vec::with_capacity(num_rounds);
            let mut rate = 1usize;
            for round in 0..num_rounds {
                rate += folding_factor.at_round(round) - 1;
                rates.push(rate);
            }
            rates
        };
        let params_plain = ProtocolParameters {
            security_level: 32,
            pow_bits: 0,
            round_log_inv_rates,
            folding_factor,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
            zk: false,
        };
        let params_zk = ProtocolParameters {
            zk: true,
            ..params_plain.clone()
        };

        let folding = params_plain.folding_factor.at_round(0);
        let mut rng = SmallRng::seed_from_u64(42);
        let table = Table::new(vec![Poly::<F>::rand(&mut rng, num_vars)]);
        let witness = L::new_witness(vec![table], folding);
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            TableShape::new(num_vars, 1),
            vec![vec![0]],
        )]);

        // Plain
        {
            let pcs = Whir::new(
                WhirConfig::new(num_vars, params_plain),
                MyDft::default(),
                mmcs(),
            );
            group.bench_function(BenchmarkId::new("plain", num_vars), |b| {
                b.iter(|| {
                    let mut ch = challenger();
                    let mut ds = DomainSeparator::new(vec![]);
                    pcs.add_domain_separator::<8>(&mut ds);
                    ds.observe_domain_separator(&mut ch);
                    let (_, pd) = <Whir as MultilinearPcs<EF, MyChallenger>>::commit(
                        &pcs,
                        witness.clone(),
                        &mut ch,
                    );
                    let proof = <Whir as MultilinearPcs<EF, MyChallenger>>::open(
                        &pcs,
                        pd,
                        protocol.clone(),
                        &mut ch,
                    );
                    black_box(proof);
                });
            });
        }

        // ZK
        {
            let inner = Whir::new(
                WhirConfig::new(num_vars, params_zk),
                MyDft::default(),
                mmcs(),
            );
            let pcs = HidingWhirPcs::new(inner, SmallRng::seed_from_u64(99));
            type HP = HidingWhirPcs<EF, F, MyDft, MyMmcs, MyChallenger, L, SmallRng>;
            group.bench_function(BenchmarkId::new("zk", num_vars), |b| {
                b.iter(|| {
                    let mut ch = challenger();
                    let mut ds = DomainSeparator::new(vec![]);
                    pcs.add_domain_separator::<8>(&mut ds);
                    ds.observe_domain_separator(&mut ch);
                    let (_, pd) = <HP as MultilinearPcs<EF, MyChallenger>>::commit(
                        &pcs,
                        witness.clone(),
                        &mut ch,
                    );
                    let proof = <HP as MultilinearPcs<EF, MyChallenger>>::open(
                        &pcs,
                        pd,
                        protocol.clone(),
                        &mut ch,
                    );
                    black_box(proof);
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_prover);
criterion_main!(benches);
