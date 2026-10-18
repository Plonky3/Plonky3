//! Full WHIR lookup proofs, including dormant AIR rounds and compact bus periods.
//! Set P3_LOOKUP_REPORT=1 to print one allocation/proof-size measurement per fixture.
use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicUsize, Ordering};

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_air::{Air, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_bus::{BusActivation, BusDirection, BusInteractionBuilder, BusName};
use p3_challenger::DuplexChallenger;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_keccak::Keccak256Hash;
use p3_lookup::{Count, InteractionBuilder};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multi_stark::config::MultiStarkConfig;
use p3_multi_stark::{
    ProverInstance, ProverInstances, VerifierInstance, VerifierInstances, prove, setup, verify,
};
use p3_sumcheck::layout::{Layout, PrefixProver, Table, Witness};
use p3_symmetric::{CryptographicHasher, PaddingFreeSponge, TruncatedPermutation};
use p3_util::log2_ceil_usize;
use p3_whir::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig, WhirProver};
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
type L = PrefixProver<F, EF>;
type TestPcs = WhirProver<EF, F, MyDft, MyMmcs, MyChallenger, L>;

/// First-round folding factor; also the per-table padding floor.
const FOLDING: usize = 2;

/// A WHIR-backed multilinear AIR configuration over BabyBear.
struct WhirConfigForTest {
    /// The WHIR commitment scheme, fixed to one stacked-table arity.
    pcs: TestPcs,
    collision_bits: Option<usize>,
}

impl MultiStarkConfig for WhirConfigForTest {
    type Val = F;
    type Challenge = EF;
    type Challenger = MyChallenger;
    type Pcs = TestPcs;

    fn pcs(&self) -> &TestPcs {
        &self.pcs
    }

    fn collision_resistance_bits(&self) -> Option<usize> {
        self.collision_bits
    }

    fn min_num_variables(&self) -> usize {
        // The witness pads each table to the first-round folding factor.
        FOLDING
    }

    fn build_witness(&self, tables: Vec<Table<F>>) -> Witness<F> {
        L::new_witness(tables, FOLDING)
    }

    fn committed_table<'a>(
        &self,
        prover_data: &'a p3_whir::WhirProverData<F, EF, MyMmcs, L>,
        table_index: usize,
    ) -> &'a Table<F> {
        prover_data.table(table_index)
    }
}

/// Fixed permutation so prover and verifier transcripts match exactly.
fn perm() -> Perm {
    let mut rng = SmallRng::seed_from_u64(0xD15EA5E);
    Perm::new_from_rng_128(&mut rng)
}

/// Per-round log-inverse rates for a stacked polynomial.
fn default_round_log_inv_rates(num_variables: usize, folding_factor: &FoldingFactor) -> Vec<usize> {
    let folding_schedule = folding_factor
        .compute_folding_schedule(num_variables)
        .expect("valid folding schedule");
    let num_rounds = folding_schedule.len() - 1;
    let mut rates = Vec::with_capacity(num_rounds);
    let mut rate = 1;
    for &folding in folding_schedule.iter().take(num_rounds) {
        rate += folding - 1;
        rates.push(rate);
    }
    rates
}

/// Build a configuration sized for an already-stacked polynomial arity.
fn config_for_stacked(stacked_num_variables: usize) -> WhirConfigForTest {
    let folding_factor = FoldingFactor::Constant(FOLDING);

    let mmcs = MyMmcs::new(MyHash::new(perm()), MyCompress::new(perm()), 0);
    let params = ProtocolParameters {
        security_level: 32,
        pow_bits: 0,
        round_log_inv_rates: default_round_log_inv_rates(stacked_num_variables, &folding_factor),
        folding_factor,
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: 1,
    };
    let whir_config = WhirConfig::new(stacked_num_variables, params).unwrap();
    WhirConfigForTest {
        pcs: TestPcs::new(whir_config, MyDft::default(), mmcs),
        collision_bits: None,
    }
}

/// A fresh challenger.
///
/// The scheme seeds its own transcript when it opens.
fn challenger() -> MyChallenger {
    MyChallenger::new(perm())
}

/// System allocator that tracks live and high-water bytes while armed.
///
/// Disarmed, it adds one relaxed load per allocation and nothing else.
/// So criterion's timing runs are left essentially unperturbed.
struct TrackingAlloc;

/// Live bytes since the last arm, signed so a pre-arm free cannot underflow.
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicIsize = AtomicIsize::new(0);
/// High-water mark of `LIVE` while armed.
static PEAK: AtomicUsize = AtomicUsize::new(0);
/// Whether allocations are currently being counted.
static ARMED: AtomicBool = AtomicBool::new(false);

unsafe impl GlobalAlloc for TrackingAlloc {
    unsafe fn alloc(&self, layout: AllocLayout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() && ARMED.load(Ordering::Relaxed) {
            ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
            let live =
                LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed) + layout.size() as isize;
            if live > 0 {
                PEAK.fetch_max(live as usize, Ordering::Relaxed);
            }
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: AllocLayout) {
        if ARMED.load(Ordering::Relaxed) {
            LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        }
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static GLOBAL: TrackingAlloc = TrackingAlloc;

/// Begin counting allocations from a zero baseline.
fn arm_heap() {
    ALLOCATED.store(0, Ordering::Relaxed);
    LIVE.store(0, Ordering::Relaxed);
    PEAK.store(0, Ordering::Relaxed);
    ARMED.store(true, Ordering::Relaxed);
}

/// Stop counting and return the peak live bytes seen while armed.
fn disarm_heap() -> usize {
    ARMED.store(false, Ordering::Relaxed);
    PEAK.load(Ordering::Relaxed)
}

struct LookupAir {
    tuples: usize,
    period: Option<Vec<F>>,
}
impl BaseAir<F> for LookupAir {
    fn width(&self) -> usize {
        2
    }
    fn num_periodic_columns(&self) -> usize {
        usize::from(self.period.is_some())
    }
    fn periodic_columns(&self) -> Cow<'_, [Vec<F>]> {
        Cow::Borrowed(self.period.as_slice())
    }
    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }
}
impl<AB: InteractionBuilder<F = F> + BusInteractionBuilder> Air<AB> for LookupAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        builder.assert_eq(local[0], local[1]);
        for _ in 0..self.tuples / 2 {
            builder.push_local_interaction([
                (vec![local[0].into()], Count::bounded(AB::Expr::ONE, 1)),
                (vec![local[1].into()], Count::provided(-AB::Expr::ONE)),
            ]);
        }
        if self.period.is_some() {
            let periodic: AB::Expr = builder.periodic_values()[0].into();
            for direction in [BusDirection::Push, BusDirection::Pull] {
                builder.push_bus_interaction(
                    BusName::new("period"),
                    direction,
                    [local[0] + periodic.clone()],
                    BusActivation::Boolean(periodic.clone()),
                );
            }
        }
    }
}
fn table(log: usize) -> Table<F> {
    let n = 1 << log;
    Table::new(RowMajorMatrix::new(
        (0..2 * n)
            .map(|i| F::from_u64((i % n % 251) as u64))
            .collect(),
        n,
    ))
}
fn bench_lookup_proof(c: &mut Criterion) {
    let mut group = c.benchmark_group("lookup_full_proof");
    group.sample_size(10);
    let log = std::env::var("P3_LOOKUP_LOG")
        .ok()
        .map(|s| s.parse().unwrap())
        .unwrap_or(14);
    let height = 1 << log;
    let mut cases = Vec::new();
    for tuples in [2, 4, 16, 64] {
        cases.push((format!("tuples_{tuples}"), tuples, None, false));
    }
    for period in std::collections::BTreeSet::from([1, 4, 256, height]) {
        if period <= height {
            cases.push((format!("bus_period_{period}"), 0, Some(period), false));
        }
    }
    cases.push(("mixed_lookup_bus".to_string(), 16, Some(4), true));
    for (label, tuples, period, mixed) in cases {
        let air = LookupAir {
            tuples,
            period: period.map(|p| (0..p).map(|i| F::from_bool(i % 7 < 3)).collect()),
        };
        let logs = if mixed { vec![log, 2] } else { vec![log] };
        let tables = logs.iter().map(|&log| table(log)).collect::<Vec<_>>();
        let airs = vec![&air; tables.len()];
        let config = config_for_stacked(log2_ceil_usize(
            logs.iter().map(|&log| 2 * (1 << log)).sum(),
        ));
        let (pk, vk) = setup(&config, &airs, &mut challenger()).unwrap();
        let prove_tables = |tables: Vec<Table<F>>| {
            prove(
                &config,
                ProverInstances::new(
                    tables
                        .into_iter()
                        .map(|table| ProverInstance::new(&air, table, &pk, &[]))
                        .collect(),
                ),
                0,
                &mut challenger(),
            )
            .unwrap()
        };
        // Verify every fixture once; data and all prover settings are identical across revisions.
        let inputs = tables.clone();
        arm_heap();
        let proof = prove_tables(inputs);
        let peak = disarm_heap();
        let allocated = ALLOCATED.load(Ordering::Relaxed);
        verify(
            &config,
            VerifierInstances::new(
                logs.iter()
                    .map(|&log| VerifierInstance::new(&air, &vk, log, &[]))
                    .collect(),
            ),
            &proof,
            0,
            &mut challenger(),
        )
        .unwrap();
        if std::env::var_os("P3_LOOKUP_REPORT").is_some() {
            let bytes = postcard::to_allocvec(&proof).unwrap();
            let size = bytes.len();
            let digest = Keccak256Hash
                .hash_iter(bytes)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            println!(
                "{label},log={log},peak_bytes={peak},allocated_bytes={allocated},proof_bytes={size},proof_hash={digest}"
            );
        }
        group.bench_function(BenchmarkId::new(label, log), |b| {
            b.iter_batched(|| tables.clone(), &prove_tables, BatchSize::SmallInput);
        });
    }
    group.finish();
}
criterion_group!(benches, bench_lookup_proof);
criterion_main!(benches);
