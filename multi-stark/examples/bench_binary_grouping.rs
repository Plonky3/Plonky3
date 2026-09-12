//! Benchmark adjacent-symbol Merkle grouping on the binary recurrence AIR.
//! Run with `-- [repetitions=9] [log_height=18] [vary] [batched]`; CSV goes to stdout.
//! Group 0 is the unwrapped baseline; group 1 measures adapter overhead.
//! `vary` uses a different public transcript separator in each iteration to sample proof sizes.
//! `batched` compares the original PCS, fixed groups 4/8, and batches of 2/3/4 folds.

use std::time::Instant;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_binary_field::{BinaryChallenger, BinaryField128, TowerLevel};
use p3_binary_pcs::{
    BinaryPcs, BinaryPcsConfig, BinaryPcsParams, BinaryPcsProverData, GroupedCodewordMmcs,
};
use p3_challenger::HashChallenger;
use p3_commit::Mmcs as MmcsTrait;
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_multi_stark::config::MultiStarkConfig;
use p3_multi_stark::{
    MultiStarkProof, ProverInstance, ProverInstances, VerifierInstance, VerifierInstances, prove,
    setup, verify,
};
use p3_sumcheck::layout::{Layout, SuffixProver, Table, Witness};
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use tracing_forest::ForestLayer;
use tracing_forest::util::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

type F = BinaryField128;
type Hash = SerializingHasher<Keccak256Hash>;
type Compress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
type Mmcs = p3_merkle_tree::MerkleTreeMmcs<F, u8, Hash, Compress, 2, 32>;
type Challenger = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;

struct Config<M> {
    pcs: BinaryPcs<M>,
}

impl<M: MmcsTrait<F, Commitment = <Mmcs as MmcsTrait<F>>::Commitment>> MultiStarkConfig
    for Config<M>
{
    type Val = F;
    type Challenge = F;
    type Challenger = Challenger;
    type Pcs = BinaryPcs<M>;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }

    fn min_num_variables(&self) -> usize {
        // The binary PCS folds at least one variable and does not pad individual tables.
        1
    }

    fn build_witness(&self, tables: Vec<Table<F>>) -> Witness<F> {
        SuffixProver::<F, F>::new_witness(tables, 0)
    }

    fn committed_table<'a>(
        &self,
        prover_data: &'a BinaryPcsProverData<M>,
        table_index: usize,
    ) -> &'a Table<F> {
        prover_data.table(table_index)
    }
}

fn pcs_config(log_height: usize, log_folding_factor: usize) -> BinaryPcsConfig {
    // Two trace columns add one variable to the stacked polynomial.
    let params = BinaryPcsParams {
        log_inv_rate: 2,
        pow_bits: 0,
        security_level: 100,
    };
    BinaryPcsConfig::try_new(log_height + 1, params)
        .unwrap()
        .try_with_folding(log_folding_factor)
        .unwrap()
}

fn config<M>(log_height: usize, mmcs: M, log_folding_factor: usize) -> Config<M> {
    Config {
        pcs: BinaryPcs::new(pcs_config(log_height, log_folding_factor), mmcs),
    }
}

fn challenger(seed: u64) -> Challenger {
    let mut separator = b"p3-multi-stark-binary-recurrence-v1".to_vec();
    if seed != 0 {
        separator.extend_from_slice(&seed.to_le_bytes());
    }
    Challenger::from_hasher(separator, Keccak256Hash)
}

/// A nonlinear recurrence: (a, b) -> (b, a * b + a).
///
/// Addition is XOR and multiplication is tower-field multiplication. Nonlinear
/// constraints exercise interpolation beyond the two prime-subfield elements.
struct RecurrenceAir;

impl<F> BaseAir<F> for RecurrenceAir {
    fn width(&self) -> usize {
        2
    }

    fn num_public_values(&self) -> usize {
        3
    }
}

impl<AB: AirBuilder> Air<AB> for RecurrenceAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let next = main.next_slice();
        let public = builder.public_values();
        let (a, b, output) = (public[0], public[1], public[2]);

        builder.when_first_row().assert_eq(local[0], a);
        builder.when_first_row().assert_eq(local[1], b);
        builder.when_transition().assert_eq(next[0], local[1]);
        builder
            .when_transition()
            .assert_eq(next[1], local[0] * local[1] + local[0]);
        builder.when_last_row().assert_eq(local[1], output);
    }
}

fn trace(log_height: usize) -> (Table<F>, [F; 3]) {
    // Integer constructors map into GF(2), so use tower representations for seeds.
    let initial = [
        F::from_repr(0x0123_4567_89ab_cdef_fedc_ba98_7654_3210),
        F::from_repr(0xfedc_ba98_7654_3210_0123_4567_89ab_cdef),
    ];
    let [mut a, mut b] = initial;
    let mut values = Vec::with_capacity(2 << log_height);
    for _ in 0..1 << log_height {
        values.extend([a, b]);
        (a, b) = (b, a * b + a);
    }
    let output = values[values.len() - 1];
    let table = Table::new(RowMajorMatrix::new(values, 2).transpose());
    (table, [initial[0], initial[1], output])
}

fn run<M: MmcsTrait<F, Commitment = <Mmcs as MmcsTrait<F>>::Commitment>>(
    log_height: usize,
    mmcs: M,
    seed: u64,
    log_folding_factor: usize,
) -> (f64, f64, usize)
where
    M::ProverData<RowMajorMatrix<F>>: Clone,
{
    let config = config(log_height, mmcs, log_folding_factor);
    let (table, public) = trace(log_height);
    let (pk, vk) = setup(&config, &[&RecurrenceAir], &mut challenger(seed)).unwrap();
    let start = Instant::now();
    let proof = prove(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &RecurrenceAir,
            table,
            &pk,
            &public,
        )]),
        0,
        &mut challenger(seed),
    )
    .unwrap();
    let prove_ms = start.elapsed().as_secs_f64() * 1000.0;
    let bytes = postcard::to_allocvec(&proof).unwrap();
    let proof: MultiStarkProof<Config<M>> = postcard::from_bytes(&bytes).unwrap();
    let start = Instant::now();
    verify(
        &config,
        VerifierInstances::new(vec![VerifierInstance::new(
            &RecurrenceAir,
            &vk,
            log_height,
            &public,
        )]),
        &proof,
        0,
        &mut challenger(seed),
    )
    .expect("grouped binary AIR proof must verify");
    let verify_ms = start.elapsed().as_secs_f64() * 1000.0;
    (prove_ms, verify_ms, bytes.len())
}

fn main() {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::ERROR.into())
        .from_env_lossy();
    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();
    let args: Vec<_> = std::env::args().skip(1).collect();
    let repetitions: usize = args.first().map_or(9, |s| s.parse().unwrap());
    let log_height: usize = args.get(1).map_or(18, |s| s.parse().unwrap());
    let vary_transcript = match args.get(2).map(String::as_str) {
        None => false,
        Some("vary") => true,
        _ => panic!("usage: bench_binary_grouping [repetitions] [log_height] [vary]"),
    };
    let batched = match args.get(3).map(String::as_str) {
        None => false,
        Some("batched") => true,
        _ => panic!("expected batched as the fourth argument"),
    };
    println!("iteration,group_size,log_folding_factor,seed,prove_ms,verify_ms,proof_bytes");
    // Warm every configuration, then rotate and reverse order to spread drift across groups.
    // Iteration zero is warm-up and must be excluded from summary statistics.
    for iteration in 0..=repetitions {
        let mut groups = if batched {
            [(0, 1), (4, 1), (8, 1), (4, 2), (8, 3), (16, 4)]
        } else {
            [(0, 1), (1, 1), (2, 1), (4, 1), (8, 1), (16, 1)]
        };
        let order = iteration.saturating_sub(1);
        groups.rotate_left(order % 6);
        if (order / groups.len()) % 2 == 1 {
            groups.reverse();
        }
        let seed = if vary_transcript { iteration as u64 } else { 0 };
        for (group, arity) in groups {
            let mmcs = Mmcs::new(Hash::new(Keccak256Hash), Compress::new(Keccak256Hash), 0);
            let (prove_ms, verify_ms, bytes) = if group == 0 {
                run(log_height, mmcs, seed, arity)
            } else {
                let grouped = if arity > 1 {
                    GroupedCodewordMmcs::for_folding(mmcs, &pcs_config(log_height, arity))
                } else {
                    GroupedCodewordMmcs::new(mmcs, group)
                };
                run(log_height, grouped, seed, arity)
            };
            println!("{iteration},{group},{arity},{seed},{prove_ms:.6},{verify_ms:.6},{bytes}");
        }
    }
}
