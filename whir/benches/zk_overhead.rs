//! Zero-knowledge overhead benchmarks: HVZK-WHIR against plain WHIR.
//!
//! - Measures prover time, verifier time, and serialized proof size.
//! - Both pipelines run the same polynomial size and round structure.
//! - Eprint 2026/391 predicts a `1 + o(1)` overhead.

use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main,
};
use p3_challenger::{CanObserve, DuplexChallenger};
use p3_commit::{Mmcs as MmcsTrait, MultilinearPcs};
use p3_dft::Radix2DFTSmallBatch;
use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::layout::{Layout, PrefixProver, Table};
use p3_sumcheck::{OpeningBatch, OpeningProtocol, PrescribedPointPcs, TableShape, TableSpec};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_whir::fiat_shamir::domain_separator::DomainSeparator;
use p3_whir::parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig};
use p3_whir::pcs::proof::{PcsProof, QueryOpenings, SharedProofOpening};
use p3_whir::pcs::prover::WhirProver;
use p3_whir::pcs::zk::{HidingWhirPcs, ZkParameters, ZkWhirConfig, ZkWhirProof};
use rand::SeedableRng;
use rand::rngs::SmallRng;

type F = KoalaBear;
type EF = BinomialExtensionField<F, 4>;
type OcticEF = BinomialExtensionField<F, 8>;

type Poseidon16 = Poseidon2KoalaBear<16>;
type Poseidon24 = Poseidon2KoalaBear<24>;
type MerkleHash = PaddingFreeSponge<Poseidon24, 24, 16, 8>;
type MerkleCompress = TruncatedPermutation<Poseidon16, 2, 8, 16>;
type Challenger = DuplexChallenger<F, Poseidon16, 16, 8>;
type PackedF = <F as Field>::Packing;
type Mmcs = MerkleTreeMmcs<PackedF, PackedF, MerkleHash, MerkleCompress, 2, 8>;
type Dft = Radix2DFTSmallBatch<F>;

type PlainPcs = WhirProver<EF, F, Dft, Mmcs, Challenger, PrefixProver<F, EF>>;
type ZkPcs = HidingWhirPcs<EF, F, Dft, Mmcs, Challenger, SmallRng>;
type OcticPlainPcs = WhirProver<OcticEF, F, Dft, Mmcs, Challenger, PrefixProver<F, OcticEF>>;
type OcticZkPcs = HidingWhirPcs<OcticEF, F, Dft, Mmcs, Challenger, SmallRng>;
type Commitment = <Mmcs as MmcsTrait<F>>::Commitment;
type MerkleMultiProof = <Mmcs as MmcsTrait<F>>::MultiProof;
type OcticPlainProof = PcsProof<F, OcticEF, Mmcs>;
type OcticZkProof = ZkWhirProof<F, OcticEF, Mmcs>;

// Polynomial sizes (log_2 of coefficient count) and shared knobs.
const SIZES: [usize; 2] = [16, 18];
const FOLDING: usize = 4;
const LOG_INV_RATE: usize = 2;
const SECURITY_LEVEL: usize = 100;
const OCTIC_OPEN_SIZES: [usize; 3] = [18, 19, 20];

const fn protocol_params() -> ProtocolParameters {
    ProtocolParameters {
        security_level: SECURITY_LEVEL,
        pow_bits: 16,
        round_log_inv_rates: Vec::new(),
        folding_factor: FoldingFactor::Constant(FOLDING),
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: LOG_INV_RATE,
    }
}

fn octic_no_pow_protocol_params() -> ProtocolParameters {
    ProtocolParameters {
        security_level: 128,
        pow_bits: 0,
        round_log_inv_rates: vec![4],
        folding_factor: FoldingFactor::ConstantFromSecondRound(8, 6),
        soundness_type: SecurityAssumption::JohnsonBound,
        starting_log_inv_rate: 1,
    }
}

fn octic_no_pow_plain_config(num_variables: usize) -> WhirConfig<OcticEF, F, Challenger> {
    WhirConfig::new(num_variables, octic_no_pow_protocol_params()).unwrap()
}

fn octic_no_pow_zk_config(num_variables: usize) -> ZkWhirConfig<OcticEF, F, Challenger> {
    ZkWhirConfig::new(
        num_variables,
        octic_no_pow_protocol_params(),
        ZkParameters {
            ell_zk: 3,
            mask_log_inv_rate: 3,
        },
    )
    .unwrap()
}

fn mmcs() -> Mmcs {
    let mut rng = SmallRng::seed_from_u64(1);
    let p16 = Poseidon16::new_from_rng_128(&mut rng);
    let p24 = Poseidon24::new_from_rng_128(&mut rng);
    Mmcs::new(MerkleHash::new(p24), MerkleCompress::new(p16), 0)
}

fn challenger() -> Challenger {
    let mut rng = SmallRng::seed_from_u64(2);
    Challenger::new(Poseidon16::new_from_rng_128(&mut rng))
}

/// Plain and hiding fixtures for the matched, grind-free comparison.
///
/// Both sides use the same field, security parameters, folding schedule,
/// polynomial size, and one-point opening workload. Their adapter-specific
/// witness/protocol types are kept separate so the benchmark exercises the
/// public PCS interfaces exactly as downstream users do.
struct OcticPlainFixture {
    pcs: OcticPlainPcs,
    witness: <OcticPlainPcs as MultilinearPcs<OcticEF, Challenger>>::Witness,
    protocol: <OcticPlainPcs as MultilinearPcs<OcticEF, Challenger>>::OpeningProtocol,
    points: Vec<Point<OcticEF>>,
    domain_separator: DomainSeparator<OcticEF, F>,
}

impl OcticPlainFixture {
    fn new(num_variables: usize) -> Self {
        let config = octic_no_pow_plain_config(num_variables);
        let pcs = OcticPlainPcs::new(config, Dft::default(), mmcs());

        let mut rng = SmallRng::seed_from_u64(3);
        let poly = Poly::<F>::rand(&mut rng, num_variables);
        let table = Table::new(RowMajorMatrix::new(
            poly.as_slice().to_vec(),
            1 << num_variables,
        ));
        let witness = PrefixProver::<F, OcticEF>::new_witness(vec![table], 8);
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            TableShape::new(num_variables, 1),
            vec![OpeningBatch::new(vec![0], Vec::new())],
        )]);
        let mut point_rng = SmallRng::seed_from_u64(5);
        let points = vec![Point::<OcticEF>::rand(&mut point_rng, num_variables)];

        let mut domain_separator = DomainSeparator::new(vec![]);
        pcs.add_domain_separator::<8>(&mut domain_separator);
        Self {
            pcs,
            witness,
            protocol,
            points,
            domain_separator,
        }
    }

    fn challenger(&self) -> Challenger {
        let mut challenger = challenger();
        self.domain_separator
            .observe_domain_separator(&mut challenger);
        challenger
    }

    fn build_proof(&self) -> (Commitment, OcticPlainProof) {
        let mut challenger = self.challenger();
        let (commitment, data) = self.pcs.commit(self.witness.clone(), &mut challenger);
        let proof = self
            .pcs
            .open_at(data, &self.protocol, &self.points, &mut challenger);
        (commitment, proof)
    }
}

struct OcticZkFixture {
    pcs: OcticZkPcs,
    witness: Poly<F>,
    points: Vec<Point<OcticEF>>,
    domain_separator: DomainSeparator<OcticEF, F>,
}

impl OcticZkFixture {
    fn new(num_variables: usize) -> Self {
        let pcs = OcticZkPcs::new(
            octic_no_pow_zk_config(num_variables),
            Dft::default(),
            mmcs(),
            SmallRng::seed_from_u64(4),
        );
        let mut rng = SmallRng::seed_from_u64(3);
        let witness = Poly::<F>::rand(&mut rng, num_variables);
        let mut point_rng = SmallRng::seed_from_u64(5);
        let points = vec![Point::<OcticEF>::rand(&mut point_rng, num_variables)];

        let mut domain_separator = DomainSeparator::new(vec![]);
        pcs.add_domain_separator::<8>(&mut domain_separator);
        Self {
            pcs,
            witness,
            points,
            domain_separator,
        }
    }

    fn challenger(&self) -> Challenger {
        let mut challenger = challenger();
        self.domain_separator
            .observe_domain_separator(&mut challenger);
        challenger
    }

    fn build_proof(&self) -> (Commitment, OcticZkProof) {
        let mut challenger = self.challenger();
        let (commitment, data) = self.pcs.commit(self.witness.clone(), &mut challenger);
        let proof = self.pcs.open(data, self.points.clone(), &mut challenger);
        (commitment, proof)
    }
}

/// Benchmarks the plain (non-hiding) prover for one size.
fn bench_plain(group: &mut BenchmarkGroup<'_, WallTime>, num_variables: usize) {
    let config = WhirConfig::new(num_variables, protocol_params()).unwrap();
    let pcs = PlainPcs::new(config, Dft::default(), mmcs());

    let mut rng = SmallRng::seed_from_u64(3);
    let table = Table::rand(&mut rng, 1, num_variables);
    let protocol = OpeningProtocol::new(vec![TableSpec::new(
        TableShape::new(num_variables, 1),
        vec![OpeningBatch::new(vec![0], Vec::new())],
    )]);

    group.bench_function(BenchmarkId::new("plain", num_variables), |b| {
        b.iter_batched(
            || {
                let mut ch = challenger();
                let mut ds = DomainSeparator::new(vec![]);
                pcs.add_domain_separator::<8>(&mut ds);
                ds.observe_domain_separator(&mut ch);
                let witness = PrefixProver::<F, EF>::new_witness(vec![table.clone()], FOLDING);
                (ch, witness)
            },
            |(mut ch, witness)| {
                let (_, data) = pcs.commit(witness, &mut ch);
                pcs.open(data, protocol.clone(), &mut ch)
            },
            BatchSize::SmallInput,
        );
    });
}

/// Benchmarks the hiding prover for one size.
fn bench_zk(group: &mut BenchmarkGroup<'_, WallTime>, num_variables: usize) {
    let config = ZkWhirConfig::new(
        num_variables,
        protocol_params(),
        // Section 2.7 of eprint 2026/391: O~(lambda)-sized masks at low
        // rate, so few spot checks reach the security level.
        ZkParameters {
            ell_zk: 16,
            mask_log_inv_rate: 5,
        },
    )
    .unwrap();
    let pcs = ZkPcs::new(config, Dft::default(), mmcs(), SmallRng::seed_from_u64(4));

    let mut rng = SmallRng::seed_from_u64(3);
    let witness = Poly::<F>::rand(&mut rng, num_variables);
    let points = vec![Point::<EF>::rand(&mut rng, num_variables)];

    group.bench_function(BenchmarkId::new("zk", num_variables), |b| {
        b.iter_batched(
            || {
                let mut ch = challenger();
                let mut ds = DomainSeparator::new(vec![]);
                pcs.add_domain_separator::<8>(&mut ds);
                ds.observe_domain_separator(&mut ch);
                (ch, witness.clone())
            },
            |(mut ch, witness)| {
                let (_, data) = pcs.commit(witness, &mut ch);
                pcs.open(data, points.clone(), &mut ch)
            },
            BatchSize::SmallInput,
        );
    });
}

/// Reports serialized proof sizes side by side (printed once, not measured).
fn report_proof_sizes(num_variables: usize) {
    // Plain proof size.
    let config = WhirConfig::new(num_variables, protocol_params()).unwrap();
    let pcs = PlainPcs::new(config, Dft::default(), mmcs());
    let mut rng = SmallRng::seed_from_u64(3);
    let table = Table::rand(&mut rng, 1, num_variables);
    let protocol = OpeningProtocol::new(vec![TableSpec::new(
        TableShape::new(num_variables, 1),
        vec![OpeningBatch::new(vec![0], Vec::new())],
    )]);
    let mut ch = challenger();
    let mut ds = DomainSeparator::new(vec![]);
    pcs.add_domain_separator::<8>(&mut ds);
    ds.observe_domain_separator(&mut ch);
    let witness = PrefixProver::<F, EF>::new_witness(vec![table], FOLDING);
    let (_, data) = pcs.commit(witness, &mut ch);
    let plain_proof = pcs.open(data, protocol, &mut ch);
    let plain_size = postcard::to_allocvec(&plain_proof).unwrap().len();

    // Hiding proof size.
    let config = ZkWhirConfig::new(
        num_variables,
        protocol_params(),
        // Section 2.7 of eprint 2026/391: O~(lambda)-sized masks at low
        // rate, so few spot checks reach the security level.
        ZkParameters {
            ell_zk: 16,
            mask_log_inv_rate: 5,
        },
    )
    .unwrap();
    let pcs = ZkPcs::new(config, Dft::default(), mmcs(), SmallRng::seed_from_u64(4));
    let mut rng = SmallRng::seed_from_u64(3);
    let witness = Poly::<F>::rand(&mut rng, num_variables);
    let points = vec![Point::<EF>::rand(&mut rng, num_variables)];
    let mut ch = challenger();
    let mut ds = DomainSeparator::new(vec![]);
    pcs.add_domain_separator::<8>(&mut ds);
    ds.observe_domain_separator(&mut ch);
    let (_, data) = pcs.commit(witness, &mut ch);
    let zk_proof = pcs.open(data, points, &mut ch);
    let zk_size = postcard::to_allocvec(&zk_proof).unwrap().len();

    println!(
        "proof size @ 2^{num_variables}: plain = {plain_size} B, zk = {zk_size} B, overhead = {:.2}x",
        zk_size as f64 / plain_size as f64,
    );
}

fn bench_zk_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("whir_zk_overhead");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(15));

    for num_variables in SIZES {
        report_proof_sizes(num_variables);
        bench_plain(&mut group, num_variables);
        bench_zk(&mut group, num_variables);
    }
    group.finish();
}

/// Matched grind-free comparison, split into commit, open, and verify phases.
///
/// The legacy group above keeps the production-like PoW configuration for
/// continuity. It must not be used for close plain-vs-ZK timing comparisons:
/// the two transcripts reach grinding in different states, so they search
/// different nonce sequences even when their challengers start from one seed.
fn bench_octic_no_pow(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("whir_zk_overhead_no_pow/commit");
        group
            .sample_size(10)
            .measurement_time(Duration::from_secs(20));
        for num_variables in OCTIC_OPEN_SIZES {
            let plain = OcticPlainFixture::new(num_variables);
            let zk = OcticZkFixture::new(num_variables);
            let label = format!("n{num_variables}_ff8_6_ell3_rate3");

            group.bench_function(BenchmarkId::new("plain", &label), |b| {
                b.iter_batched(
                    || (plain.witness.clone(), plain.challenger()),
                    |(witness, mut challenger)| plain.pcs.commit(witness, &mut challenger),
                    BatchSize::PerIteration,
                );
            });
            group.bench_function(BenchmarkId::new("zk", &label), |b| {
                b.iter_batched(
                    || (zk.witness.clone(), zk.challenger()),
                    |(witness, mut challenger)| zk.pcs.commit(witness, &mut challenger),
                    BatchSize::PerIteration,
                );
            });
        }
        group.finish();
    }

    {
        let mut group = c.benchmark_group("whir_zk_overhead_no_pow/open");
        group
            .sample_size(20)
            .measurement_time(Duration::from_secs(20));
        for num_variables in OCTIC_OPEN_SIZES {
            let plain = OcticPlainFixture::new(num_variables);
            let zk = OcticZkFixture::new(num_variables);
            let label = format!("n{num_variables}_ff8_6_ell3_rate3");

            group.bench_function(BenchmarkId::new("plain", &label), |b| {
                b.iter_batched(
                    || {
                        let mut challenger = plain.challenger();
                        let (_, data) = plain.pcs.commit(plain.witness.clone(), &mut challenger);
                        (
                            challenger,
                            data,
                            plain.protocol.clone(),
                            plain.points.clone(),
                        )
                    },
                    |(mut challenger, data, protocol, points)| {
                        plain.pcs.open_at(data, &protocol, &points, &mut challenger)
                    },
                    BatchSize::PerIteration,
                );
            });
            group.bench_function(BenchmarkId::new("zk", &label), |b| {
                b.iter_batched(
                    || {
                        let mut challenger = zk.challenger();
                        let (_, data) = zk.pcs.commit(zk.witness.clone(), &mut challenger);
                        (challenger, data, zk.points.clone())
                    },
                    |(mut challenger, data, points)| zk.pcs.open(data, points, &mut challenger),
                    BatchSize::PerIteration,
                );
            });
        }
        group.finish();
    }

    {
        let mut group = c.benchmark_group("whir_zk_overhead_no_pow/verify");
        group
            .sample_size(30)
            .measurement_time(Duration::from_secs(10));
        for num_variables in OCTIC_OPEN_SIZES {
            let plain = OcticPlainFixture::new(num_variables);
            let zk = OcticZkFixture::new(num_variables);
            let (plain_commitment, plain_proof) = plain.build_proof();
            let (zk_commitment, zk_proof) = zk.build_proof();
            let label = format!("n{num_variables}_ff8_6_ell3_rate3");

            group.bench_function(BenchmarkId::new("plain", &label), |b| {
                b.iter_batched(
                    || {
                        (
                            plain.challenger(),
                            plain.protocol.clone(),
                            plain.points.clone(),
                        )
                    },
                    |(mut challenger, protocol, points)| {
                        challenger.observe(plain_commitment.clone());
                        plain
                            .pcs
                            .verify_at(
                                &plain_commitment,
                                &plain_proof,
                                &protocol,
                                &points,
                                &mut challenger,
                            )
                            .unwrap();
                    },
                    BatchSize::PerIteration,
                );
            });
            group.bench_function(BenchmarkId::new("zk", &label), |b| {
                b.iter_batched(
                    || (zk.challenger(), zk.points.clone()),
                    |(mut challenger, points)| {
                        zk.pcs
                            .verify(&zk_commitment, &zk_proof, &mut challenger, points)
                            .unwrap();
                    },
                    BatchSize::PerIteration,
                );
            });
        }
        group.finish();
    }
}

/// Logical Merkle payload counts, independent of postcard's integer encoding.
#[derive(Clone, Copy, Default)]
struct OpeningShape {
    multiproofs: usize,
    authenticated_rows: usize,
    base_values: usize,
    extension_values: usize,
    sibling_digests: usize,
}

impl OpeningShape {
    const fn merge(&mut self, rhs: Self) {
        self.multiproofs += rhs.multiproofs;
        self.authenticated_rows += rhs.authenticated_rows;
        self.base_values += rhs.base_values;
        self.extension_values += rhs.extension_values;
        self.sibling_digests += rhs.sibling_digests;
    }
}

fn query_opening_shape<Ext>(opening: &QueryOpenings<F, Ext, MerkleMultiProof>) -> OpeningShape {
    match opening {
        QueryOpenings::Base(opening) => OpeningShape {
            multiproofs: 1,
            authenticated_rows: opening.rows.len(),
            base_values: opening.rows.iter().map(Vec::len).sum(),
            extension_values: 0,
            sibling_digests: opening.proof.sibling_hashes.len(),
        },
        QueryOpenings::Extension(opening) => OpeningShape {
            multiproofs: 1,
            authenticated_rows: opening.rows.len(),
            base_values: 0,
            extension_values: opening.rows.iter().map(Vec::len).sum(),
            sibling_digests: opening.proof.sibling_hashes.len(),
        },
    }
}

fn extension_opening_shape<Ext>(
    opening: &SharedProofOpening<Ext, MerkleMultiProof>,
) -> OpeningShape {
    OpeningShape {
        multiproofs: 1,
        authenticated_rows: opening.rows.len(),
        base_values: 0,
        extension_values: opening.rows.iter().map(Vec::len).sum(),
        sibling_digests: opening.proof.sibling_hashes.len(),
    }
}

fn postcard_len<T: serde::Serialize>(value: &T) -> usize {
    postcard::to_allocvec(value)
        .expect("proof component serializes")
        .len()
}

/// Print a reproducible proof-shape report for the matched no-PoW case.
///
/// Bytes and sibling counts are the fixed-seed realized cost: varint widths
/// and multiproof overlap can change with another transcript. Roots, rows,
/// and leaf-value counts expose the protocol shape independently.
fn report_octic_proof_shape(_c: &mut Criterion) {
    eprintln!();
    eprintln!("WHIR ZK overhead report (128-bit, no PoW, octic extension, ff=8/6)");
    eprintln!(
        "{:>5} {:>6} {:>12} {:>7} {:>7} {:>10} {:>11} {:>12} {:>15}",
        "vars",
        "kind",
        "proof_bytes",
        "roots",
        "proofs",
        "auth_rows",
        "base_values",
        "ext_values",
        "merkle_digests",
    );

    for num_variables in OCTIC_OPEN_SIZES {
        let plain = OcticPlainFixture::new(num_variables);
        let zk = OcticZkFixture::new(num_variables);
        let (_, plain_proof) = plain.build_proof();
        let (_, zk_proof) = zk.build_proof();
        assert_eq!(plain_proof.evals[0].current()[0], zk_proof.evals[0]);

        let mut plain_shape = OpeningShape::default();
        for round in &plain_proof.whir.rounds {
            plain_shape.merge(query_opening_shape(&round.openings));
        }
        plain_shape.merge(query_opening_shape(&plain_proof.whir.final_openings));
        let plain_roots = 1 + plain_proof
            .whir
            .rounds
            .iter()
            .filter(|round| round.commitment.is_some())
            .count();

        let mut zk_shape = OpeningShape::default();
        for round in &zk_proof.rounds {
            zk_shape.merge(query_opening_shape(&round.openings));
        }
        zk_shape.merge(query_opening_shape(&zk_proof.base_case.source_openings));
        zk_shape.merge(extension_opening_shape(
            &zk_proof.base_case.fresh_main_openings,
        ));
        for pair in &zk_proof.base_case.mask_openings {
            zk_shape.merge(extension_opening_shape(&pair.carried));
            zk_shape.merge(extension_opening_shape(&pair.fresh));
        }
        let zk_roots = 1
            + zk_proof.sumcheck_mask_commitments.len()
            + 2 * zk_proof.rounds.len()
            + 1
            + zk_proof.base_case.fresh_mask_commitments.len();

        eprintln!(
            "{:>5} {:>6} {:>12} {:>7} {:>7} {:>10} {:>11} {:>12} {:>15}",
            num_variables,
            "plain",
            postcard_len(&plain_proof),
            plain_roots,
            plain_shape.multiproofs,
            plain_shape.authenticated_rows,
            plain_shape.base_values,
            plain_shape.extension_values,
            plain_shape.sibling_digests,
        );
        eprintln!(
            "{:>5} {:>6} {:>12} {:>7} {:>7} {:>10} {:>11} {:>12} {:>15}",
            num_variables,
            "zk",
            postcard_len(&zk_proof),
            zk_roots,
            zk_shape.multiproofs,
            zk_shape.authenticated_rows,
            zk_shape.base_values,
            zk_shape.extension_values,
            zk_shape.sibling_digests,
        );
        let blinded_mask_elements: usize = zk_proof
            .base_case
            .blinded_masks
            .iter()
            .map(|mask| mask.message.len() + mask.randomness.len())
            .sum();
        eprintln!(
            "      zk components: rounds={} sumchecks={} base_case={} mask_openings={} blinded_masks={} mask_groups={} blinded_mask_elements={}",
            postcard_len(&zk_proof.rounds),
            postcard_len(&zk_proof.sumchecks),
            postcard_len(&zk_proof.base_case),
            postcard_len(&zk_proof.base_case.mask_openings),
            postcard_len(&zk_proof.base_case.blinded_masks),
            zk_proof.base_case.mask_openings.len(),
            blinded_mask_elements,
        );
    }
    eprintln!();
}

criterion_group!(benches, bench_zk_overhead, bench_octic_no_pow);
criterion_group!(reports, report_octic_proof_shape);
criterion_main!(benches, reports);
