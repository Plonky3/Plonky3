//! End-to-end lookup benchmark harness.
//!
//! Drives a parametric lookup workload through the full batch-STARK pipeline.
//!
//! Reports the numbers a lookup-layer change actually moves:
//!
//!     prove time       criterion-timed
//!     verify time      criterion-timed
//!     aux columns      committed permutation columns (the packing target)
//!     proof size       serialized bytes
//!     peak heap        high-water bytes during one prove
//!
//! The workload is one balanced lookup AIR with these knobs:
//!
//!     n_interactions   query/provide pairs, all on one shared bus
//!     tuple_width      field elements per message
//!     base_degree      degree of an optional base constraint, or 0 for none
//!     active_period    one active row every `active_period` rows (row-sparsity)
//!     n_airs           identical instances proven in one batch
//!
//! Every case is measured twice, with same-bus packing on and off:
//!
//!     base_degree 0  → no spare budget → packing folds nothing → both rows match
//!     base_degree d  → spare budget    → packing folds → fewer columns, less cost
//!
//! So the table proves the two claims side by side:
//!
//!     no-budget rows identical  → packing never regresses
//!     budget rows packed lower  → packing is a strict win
//!
//! Running it:
//!
//!     cargo bench -p p3-batch-stark --bench lookup
//!
//! and optionally with `--features parallel`.
//!
//! The metrics table prints first, then the criterion timings.
//! Comment out entries in `CASES` to shorten a local run.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicUsize, Ordering};
use std::time::Instant;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_air::{Air, BaseAir, PermutationAirBuilder, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_batch_stark::{ProverData, StarkInstance, prove_batch, verify_batch};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_lookup::{Count, InteractionBuilder, LookupBus, Lookups};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::StarkConfig;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 2, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Dft = Radix2DitParallel<Val>;
type MyPcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type MyConfig = StarkConfig<MyPcs, Challenge, Challenger>;

// ---------------------------------------------------------------------------
// Peak-heap accounting
// ---------------------------------------------------------------------------

/// System allocator that tracks live and high-water bytes while armed.
///
/// Disarmed, it adds one relaxed load per allocation and nothing else.
/// So criterion's timing runs are left essentially unperturbed.
struct TrackingAlloc;

/// Live bytes since the last arm, signed so a pre-arm free cannot underflow.
static LIVE: AtomicIsize = AtomicIsize::new(0);
/// High-water mark of `LIVE` while armed.
static PEAK: AtomicUsize = AtomicUsize::new(0);
/// Whether allocations are currently being counted.
static ARMED: AtomicBool = AtomicBool::new(false);

unsafe impl GlobalAlloc for TrackingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() && ARMED.load(Ordering::Relaxed) {
            // Bump live bytes, then raise the peak if this is a new high.
            let live =
                LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed) + layout.size() as isize;
            if live > 0 {
                PEAK.fetch_max(live as usize, Ordering::Relaxed);
            }
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
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
    LIVE.store(0, Ordering::Relaxed);
    PEAK.store(0, Ordering::Relaxed);
    ARMED.store(true, Ordering::Relaxed);
}

/// Stop counting and return the peak live bytes seen while armed.
fn disarm_heap() -> usize {
    ARMED.store(false, Ordering::Relaxed);
    PEAK.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Workload: a balanced same-bus lookup AIR
// ---------------------------------------------------------------------------

/// The single bus every interaction speaks on.
///
/// One shared bus lets the packer fold interactions into common columns.
const SHARED_BUS: LookupBus<'static> = LookupBus::new("shared");

/// A balanced lookup AIR with a tunable base-constraint degree.
///
/// - Every slot queries a key and provides the same key on the shared bus.
/// - Query and provide cancel, so any trace balances to a zero terminal.
/// - An optional degree-`base_degree` constraint sets the spare quotient budget.
/// - That budget is exactly what the same-bus packer folds into.
#[derive(Clone, Copy)]
struct LookupLoadAir {
    /// Query/provide pairs the AIR declares, all on the shared bus.
    n_interactions: usize,
    /// Field elements per message.
    tuple_width: usize,
    /// Degree of the base constraint, or 0 for none.
    base_degree: usize,
}

impl LookupLoadAir {
    /// Main-trace columns per interaction: the key plus one selector.
    const fn stride(&self) -> usize {
        self.tuple_width + 1
    }

    /// Whether a base constraint, hence a witness pair, is present.
    const fn has_base(&self) -> bool {
        self.base_degree >= 2
    }

    /// First column of the witness pair `(w, y = w^base_degree)`.
    const fn witness_offset(&self) -> usize {
        self.n_interactions * self.stride()
    }
}

impl<F: Field> BaseAir<F> for LookupLoadAir {
    fn width(&self) -> usize {
        // Per-interaction columns, then the optional two-column witness.
        self.n_interactions * self.stride() + if self.has_base() { 2 } else { 0 }
    }
}

impl<AB: PermutationAirBuilder + InteractionBuilder> Air<AB> for LookupLoadAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let stride = self.stride();

        // Optional base constraint `w^base_degree - y = 0`.
        // It alone fixes the AIR's degree, hence the packer's spare budget.
        if self.has_base() {
            let off = self.witness_offset();
            let w = local[off];
            let y = local[off + 1];

            // Raise w to base_degree by repeated multiplication.
            let mut power: AB::Expr = w.into();
            for _ in 1..self.base_degree {
                power *= w.into();
            }
            builder.assert_zero(power - y.into());
        }

        // Each slot sends a query and a matching provide on the shared bus.
        for i in 0..self.n_interactions {
            let off = i * stride;
            let key: Vec<AB::Expr> = (0..self.tuple_width)
                .map(|j| local[off + j].into())
                .collect();
            let selector = local[off + self.tuple_width];

            // Query (+selector, weight 1) cancels the provide (-selector) on the same key.
            SHARED_BUS.lookup_key(builder, key.clone(), Count::bounded(selector.into(), 1));
            SHARED_BUS.table_entry(builder, key, selector.into());
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration matrix
// ---------------------------------------------------------------------------

/// One benchmark point.
#[derive(Clone, Copy)]
struct Case {
    /// Label shown in the table and in criterion ids.
    name: &'static str,
    /// log2 of the trace height.
    log_height: usize,
    /// Query/provide pairs the AIR declares.
    n_interactions: usize,
    /// Field elements per message.
    tuple_width: usize,
    /// Degree of the base constraint, or 0 for none.
    base_degree: usize,
    /// One active row every `active_period` rows.
    active_period: usize,
    /// Number of identical AIR instances in the batch.
    n_airs: usize,
}

/// Terse case constructor so the sweep below stays readable.
const fn case(
    name: &'static str,
    log_height: usize,
    n_interactions: usize,
    tuple_width: usize,
    base_degree: usize,
    active_period: usize,
    n_airs: usize,
) -> Case {
    Case {
        name,
        log_height,
        n_interactions,
        tuple_width,
        base_degree,
        active_period,
        n_airs,
    }
}

/// The default sweep, grouped by the axis each block isolates.
///
/// Columns are: name, logH, n, tuple_width, base_degree, active_period, n_airs.
/// Comment out blocks to shorten a local run.
const CASES: &[Case] = &[
    // Smoke point: tiny, validates the pipeline fast.
    case("smoke", 8, 1, 1, 0, 1, 1),
    // Interaction count: cost as a function of #lookups, one column each.
    case("n=1", 14, 1, 1, 0, 1, 1),
    case("n=4", 14, 4, 1, 0, 1, 1),
    case("n=8", 14, 8, 1, 0, 1, 1),
    case("n=16", 14, 16, 1, 0, 1, 1),
    // Tuple width: wider messages, same per-lookup degree.
    case("tw=2", 14, 4, 2, 0, 1, 1),
    case("tw=4", 14, 4, 4, 0, 1, 1),
    // Row-sparsity: fraction of active rows. Baseline cost is sparsity-blind today.
    case("period=4", 14, 4, 1, 0, 4, 1),
    case("period=16", 14, 4, 1, 0, 16, 1),
    // Batch size: several identical instances proven together.
    case("airs=4", 14, 4, 1, 0, 1, 4),
    // Spare budget: a degree-3 base constraint lets the packer fold two per column.
    // The benchmark FRI uses log-blowup 1, capping the quotient at degree 3.
    // So degree 3 is the deepest packing this config can show.
    case("deg3_n=8", 14, 8, 1, 3, 1, 1),
    case("deg3_n=16", 14, 16, 1, 3, 1, 1),
    // Heavy points at log_height 18: comment these out for a faster bench.
    case("n=32_h18", 18, 32, 1, 0, 1, 1),
    case("deg3_n=32_h18", 18, 32, 1, 3, 1, 1),
];

// ---------------------------------------------------------------------------
// Setup helpers
// ---------------------------------------------------------------------------

fn make_config() -> MyConfig {
    let mut rng = SmallRng::seed_from_u64(1337);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    // Production-like FRI settings (100 queries) so the proof-size column reflects
    // realistic openings rather than the 2-query testing config.
    let fri_params = FriParameters::new_benchmark(challenge_mmcs);
    let pcs = MyPcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    StarkConfig::new(pcs, challenger)
}

/// Build the AIRs, traces, and (empty) public values for one case.
fn build_workload(case: &Case) -> (Vec<LookupLoadAir>, Vec<RowMajorMatrix<Val>>, Vec<Vec<Val>>) {
    let air = LookupLoadAir {
        n_interactions: case.n_interactions,
        tuple_width: case.tuple_width,
        base_degree: case.base_degree,
    };

    let height = 1 << case.log_height;
    let width = <LookupLoadAir as BaseAir<Val>>::width(&air);
    let witness = air.witness_offset();
    let mut rng = SmallRng::seed_from_u64(0xD1CE_F00D);

    let mut values = Val::zero_vec(height * width);
    for r in 0..height {
        let base = r * width;
        // A row is active once every `active_period` rows.
        let active = r % case.active_period == 0;

        // Keys are arbitrary: query and provide cancel whatever the key holds.
        for i in 0..case.n_interactions {
            let off = base + i * (case.tuple_width + 1);
            for j in 0..case.tuple_width {
                values[off + j] = Val::from_u32(rng.random());
            }
            // Selector gates both sides, so inactive rows contribute nothing.
            values[off + case.tuple_width] = if active { Val::ONE } else { Val::ZERO };
        }

        // Witness pair satisfies the base constraint: y = w^base_degree.
        if air.has_base() {
            let w = Val::from_u32(rng.random());
            values[base + witness] = w;
            values[base + witness + 1] = w.exp_u64(case.base_degree as u64);
        }
    }

    let trace = RowMajorMatrix::new(values, width);

    // Replicate the same instance to fill the batch.
    let airs = vec![air; case.n_airs];
    let traces = (0..case.n_airs).map(|_| trace.clone()).collect();
    let public_values = vec![Vec::new(); case.n_airs];
    (airs, traces, public_values)
}

/// Bundle parallel slices into prover instances.
fn make_instances<'a>(
    airs: &'a [LookupLoadAir],
    traces: &'a [RowMajorMatrix<Val>],
    public_values: &[Vec<Val>],
) -> Vec<StarkInstance<'a, MyConfig, LookupLoadAir>> {
    airs.iter()
        .zip(traces)
        .zip(public_values)
        .map(|((air, trace), pv)| StarkInstance {
            air,
            trace,
            public_values: pv.clone(),
        })
        .collect()
}

/// Committed auxiliary columns across the batch.
///
/// An instance with lookups commits one accumulator plus one column per lookup.
/// An instance with no lookups commits none.
fn aux_columns(prover_data: &ProverData<MyConfig>) -> usize {
    prover_data
        .common
        .lookups
        .iter()
        .map(|l| if l.is_empty() { 0 } else { l.len() + 1 })
        .sum()
}

/// Rebuild `prover_data` with the unpacked layout: one column per interaction.
///
/// The default path folds same-bus interactions; this overwrites that decision
/// with the straight-from-the-AIR lookups, so a caller can measure both layouts.
fn force_unpacked(prover_data: &mut ProverData<MyConfig>, airs: &[LookupLoadAir]) {
    prover_data.common.lookups = airs
        .iter()
        .map(Lookups::<Val>::from_air::<Challenge, _>)
        .collect();
}

// ---------------------------------------------------------------------------
// Metrics report
// ---------------------------------------------------------------------------

/// One row of the printed table.
struct Metrics {
    aux_columns: usize,
    proof_bytes: usize,
    peak_bytes: usize,
    prove_ms: f64,
    verify_ms: f64,
}

/// Run one case once, packed or unpacked, checking correctness and collecting metrics.
///
/// - `packed = true` keeps the default folded layout.
/// - `packed = false` rebuilds one column per interaction.
/// - The two differ only in the aux-trace layout, isolating the packer's effect.
fn measure(config: &MyConfig, case: &Case, packed: bool) -> Metrics {
    let (airs, traces, public_values) = build_workload(case);
    let instances = make_instances(&airs, &traces, &public_values);
    let mut prover_data = ProverData::from_instances(config, &instances);
    if !packed {
        force_unpacked(&mut prover_data, &airs);
    }

    // One-shot prove with a wall-clock estimate; the metrics table reads this.
    let t = Instant::now();
    let proof = prove_batch(config, &instances, &prover_data);
    let prove_ms = t.elapsed().as_secs_f64() * 1e3;

    // Verify is the correctness gate: a malformed workload panics here.
    let t = Instant::now();
    verify_batch(config, &airs, &proof, &public_values, &prover_data.common)
        .expect("batch proof must verify");
    let verify_ms = t.elapsed().as_secs_f64() * 1e3;

    let proof_bytes = postcard::to_allocvec(&proof)
        .expect("proof must serialize")
        .len();
    let aux_columns = aux_columns(&prover_data);

    // Peak heap of a fresh prove, measured in isolation.
    arm_heap();
    let proof = prove_batch(config, &instances, &prover_data);
    let peak_bytes = disarm_heap();
    drop(proof);

    Metrics {
        aux_columns,
        proof_bytes,
        peak_bytes,
        prove_ms,
        verify_ms,
    }
}

/// Print the full metrics table once, before criterion timing.
///
/// Each case prints two rows, packed and unpacked, so the no-regression and the
/// win read straight off the pairs.
fn print_report(config: &MyConfig) {
    eprintln!();
    eprintln!("lookup end-to-end metrics (each case packed and unpacked)");
    eprintln!(
        "{:<14} {:>5} {:>4} {:>4} {:>4} {:>7} {:>5} {:>7} {:>8} {:>10} {:>9} {:>9} {:>10}",
        "case",
        "logH",
        "n",
        "tw",
        "deg",
        "period",
        "airs",
        "packed",
        "aux_cols",
        "proof(KB)",
        "peak(MB)",
        "prove(ms)",
        "verify(ms)",
    );
    for case in CASES {
        for packed in [false, true] {
            let m = measure(config, case, packed);
            eprintln!(
                "{:<14} {:>5} {:>4} {:>4} {:>4} {:>7} {:>5} {:>7} {:>8} {:>10.1} {:>9.1} {:>9.1} {:>10.2}",
                case.name,
                case.log_height,
                case.n_interactions,
                case.tuple_width,
                case.base_degree,
                case.active_period,
                case.n_airs,
                packed,
                m.aux_columns,
                m.proof_bytes as f64 / 1024.0,
                m.peak_bytes as f64 / (1024.0 * 1024.0),
                m.prove_ms,
                m.verify_ms,
            );
        }
    }
    eprintln!();
}

// ---------------------------------------------------------------------------
// Criterion timing
// ---------------------------------------------------------------------------

fn bench_lookup(c: &mut Criterion) {
    let config = make_config();

    // The table carries aux columns, proof size, and peak heap, packed vs unpacked.
    // Criterion below carries the rigorous prove and verify timings.
    print_report(&config);

    let mut group = c.benchmark_group("lookup");
    group.sample_size(10);

    for case in CASES {
        let (airs, traces, public_values) = build_workload(case);
        let instances = make_instances(&airs, &traces, &public_values);

        // Packing changes the layout only when the AIR has spare budget.
        // Time both arms there; otherwise the default path alone.
        let arms: &[bool] = if case.base_degree >= 2 {
            &[true, false]
        } else {
            &[true]
        };

        for &packed in arms {
            let mut prover_data = ProverData::from_instances(&config, &instances);
            if !packed {
                force_unpacked(&mut prover_data, &airs);
            }

            // The default path keeps the bare name; the unpacked arm is suffixed.
            let id = if packed {
                case.name.to_string()
            } else {
                format!("{}/unpacked", case.name)
            };

            group.bench_function(BenchmarkId::new("prove", id.clone()), |b| {
                b.iter(|| prove_batch(&config, &instances, &prover_data));
            });

            // A committed proof to verify repeatedly.
            let proof = prove_batch(&config, &instances, &prover_data);
            group.bench_function(BenchmarkId::new("verify", id), |b| {
                b.iter(|| {
                    verify_batch(&config, &airs, &proof, &public_values, &prover_data.common)
                        .unwrap();
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_lookup);
criterion_main!(benches);
