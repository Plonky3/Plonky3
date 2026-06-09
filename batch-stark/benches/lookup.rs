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
//! The workload is a balanced range-check AIR with four knobs:
//!
//!     n_interactions   how many lookups the AIR declares
//!     tuple_width      field elements per message
//!     fold             lookups merged into one committed column
//!     active_period    one active row every `active_period` rows (row-sparsity)
//!
//! The `fold` knob stands in for grouping:
//!
//!     merge lookups  → one multi-tuple local interaction
//!     same aux-trace shape as a same-bus grouping
//!     so grouped-vs-ungrouped reads off the `fold` sweep
//!
//! Running it:
//!
//!     cargo bench -p p3-batch-stark --bench lookup
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
use p3_lookup::InteractionBuilder;
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
// Workload: a balanced range-check AIR
// ---------------------------------------------------------------------------

/// A lookup-only AIR with no base constraints.
///
/// Each interaction queries a key and provides the same key on every active row.
/// Query and table cancel per row, so any trace balances to a zero terminal.
/// Cost is set by the structure, not the values: column count, height, width.
#[derive(Clone, Copy)]
struct LookupLoadAir {
    /// Number of independent lookups the AIR declares.
    n_interactions: usize,
    /// Field elements per message.
    tuple_width: usize,
    /// Lookups merged into one committed column.
    fold: usize,
}

impl LookupLoadAir {
    /// Main-trace columns per interaction: the key plus one selector.
    const fn stride(&self) -> usize {
        self.tuple_width + 1
    }
}

impl<F: Field> BaseAir<F> for LookupLoadAir {
    fn width(&self) -> usize {
        self.n_interactions * self.stride()
    }
}

impl<AB: PermutationAirBuilder + InteractionBuilder> Air<AB> for LookupLoadAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let stride = self.stride();

        // Walk interactions in fixed-size folds.
        // Each fold becomes one local lookup, hence one committed fraction column.
        let mut start = 0;
        while start < self.n_interactions {
            let end = (start + self.fold).min(self.n_interactions);

            let mut tuples = Vec::with_capacity(2 * (end - start));
            for i in start..end {
                let off = i * stride;

                // Key occupies the first `tuple_width` columns of this interaction.
                let key: Vec<AB::Expr> = (0..self.tuple_width)
                    .map(|j| local[off + j].into())
                    .collect();

                // Selector gates both sides, so inactive rows contribute nothing.
                let selector = local[off + self.tuple_width];

                // Query side: +selector.
                tuples.push((key.clone(), selector.into()));
                // Table side: -selector on the same key, cancelling the query.
                tuples.push((key, -(selector.into())));
            }

            builder.push_local_interaction(tuples);
            start = end;
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
    /// Lookups declared by the AIR.
    n_interactions: usize,
    /// Field elements per message.
    tuple_width: usize,
    /// Lookups merged per committed column.
    fold: usize,
    /// One active row every `active_period` rows.
    active_period: usize,
    /// Number of identical AIR instances in the batch.
    n_airs: usize,
}

/// The default sweep.
///
/// Grouped by the axis each block isolates.
/// Comment out blocks to shorten a local run.
const CASES: &[Case] = &[
    // Smoke point: tiny, validates the pipeline fast.
    Case {
        name: "smoke",
        log_height: 8,
        n_interactions: 1,
        tuple_width: 1,
        fold: 1,
        active_period: 1,
        n_airs: 1,
    },
    // Interaction count: cost as a function of #lookups, one column each.
    Case {
        name: "n=1",
        log_height: 14,
        n_interactions: 1,
        tuple_width: 1,
        fold: 1,
        active_period: 1,
        n_airs: 1,
    },
    Case {
        name: "n=4",
        log_height: 14,
        n_interactions: 4,
        tuple_width: 1,
        fold: 1,
        active_period: 1,
        n_airs: 1,
    },
    Case {
        name: "n=8",
        log_height: 14,
        n_interactions: 8,
        tuple_width: 1,
        fold: 1,
        active_period: 1,
        n_airs: 1,
    },
    Case {
        name: "n=16",
        log_height: 14,
        n_interactions: 16,
        tuple_width: 1,
        fold: 1,
        active_period: 1,
        n_airs: 1,
    },
    // Fold: 8 lookups, more merged per column → fewer columns, higher degree.
    Case {
        name: "fold=1",
        log_height: 14,
        n_interactions: 8,
        tuple_width: 1,
        fold: 1,
        active_period: 1,
        n_airs: 1,
    },
    Case {
        name: "fold=2",
        log_height: 14,
        n_interactions: 8,
        tuple_width: 1,
        fold: 2,
        active_period: 1,
        n_airs: 1,
    },
    Case {
        name: "fold=4",
        log_height: 14,
        n_interactions: 8,
        tuple_width: 1,
        fold: 4,
        active_period: 1,
        n_airs: 1,
    },
    // Tuple width: wider messages raise the combine degree.
    Case {
        name: "tw=2",
        log_height: 14,
        n_interactions: 4,
        tuple_width: 2,
        fold: 1,
        active_period: 1,
        n_airs: 1,
    },
    Case {
        name: "tw=4",
        log_height: 14,
        n_interactions: 4,
        tuple_width: 4,
        fold: 1,
        active_period: 1,
        n_airs: 1,
    },
    // Row-sparsity: fraction of active rows. Baseline cost is sparsity-blind today.
    Case {
        name: "period=4",
        log_height: 14,
        n_interactions: 4,
        tuple_width: 1,
        fold: 1,
        active_period: 4,
        n_airs: 1,
    },
    Case {
        name: "period=16",
        log_height: 14,
        n_interactions: 4,
        tuple_width: 1,
        fold: 1,
        active_period: 16,
        n_airs: 1,
    },
    // Batch size: several identical instances proven together.
    Case {
        name: "airs=4",
        log_height: 14,
        n_interactions: 4,
        tuple_width: 1,
        fold: 1,
        active_period: 1,
        n_airs: 4,
    },
    // Heavy points at log_height 18: comment these out for a faster bench.
    Case {
        name: "n=32_h18",
        log_height: 18,
        n_interactions: 32,
        tuple_width: 1,
        fold: 1,
        active_period: 1,
        n_airs: 1,
    },
    Case {
        name: "fold=8_h18",
        log_height: 18,
        n_interactions: 8,
        tuple_width: 1,
        fold: 8,
        active_period: 1,
        n_airs: 1,
    },
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
    let fri_params = FriParameters::new_testing(challenge_mmcs, 2);
    let pcs = MyPcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    StarkConfig::new(pcs, challenger)
}

/// Build the AIRs, traces, and (empty) public values for one case.
fn build_workload(case: &Case) -> (Vec<LookupLoadAir>, Vec<RowMajorMatrix<Val>>, Vec<Vec<Val>>) {
    let air = LookupLoadAir {
        n_interactions: case.n_interactions,
        tuple_width: case.tuple_width,
        fold: case.fold,
    };

    let height = 1 << case.log_height;
    let width = case.n_interactions * (case.tuple_width + 1);
    let mut rng = SmallRng::seed_from_u64(0xD1CE_F00D);

    // Fill keys with arbitrary values; the lookup balances regardless.
    let mut values = Val::zero_vec(height * width);
    for r in 0..height {
        let active = r % case.active_period == 0;
        for i in 0..case.n_interactions {
            let off = i * (case.tuple_width + 1);
            for j in 0..case.tuple_width {
                values[r * width + off + j] = Val::from_u32(rng.random());
            }
            values[r * width + off + case.tuple_width] = if active { Val::ONE } else { Val::ZERO };
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

/// Run one case once, checking correctness and collecting every metric.
fn measure(config: &MyConfig, case: &Case) -> Metrics {
    let (airs, traces, public_values) = build_workload(case);
    let instances = make_instances(&airs, &traces, &public_values);
    let prover_data = ProverData::from_instances(config, &instances);

    // One-shot prove, with a wall-clock estimate.
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
fn print_report(config: &MyConfig) {
    eprintln!();
    eprintln!("lookup end-to-end metrics");
    eprintln!(
        "{:<11} {:>5} {:>5} {:>4} {:>5} {:>7} {:>5} {:>8} {:>10} {:>9} {:>9} {:>9}",
        "case",
        "logH",
        "n",
        "tw",
        "fold",
        "period",
        "airs",
        "aux_cols",
        "proof(KB)",
        "peak(MB)",
        "prove(ms)",
        "verify(ms)",
    );
    for case in CASES {
        let m = measure(config, case);
        eprintln!(
            "{:<11} {:>5} {:>5} {:>4} {:>5} {:>7} {:>5} {:>8} {:>10.1} {:>9.1} {:>9.1} {:>9.2}",
            case.name,
            case.log_height,
            case.n_interactions,
            case.tuple_width,
            case.fold,
            case.active_period,
            case.n_airs,
            m.aux_columns,
            m.proof_bytes as f64 / 1024.0,
            m.peak_bytes as f64 / (1024.0 * 1024.0),
            m.prove_ms,
            m.verify_ms,
        );
    }
    eprintln!();
}

// ---------------------------------------------------------------------------
// Criterion timing
// ---------------------------------------------------------------------------

fn bench_lookup(c: &mut Criterion) {
    let config = make_config();

    // The table carries aux columns, proof size, and peak heap.
    // Criterion below carries the rigorous prove and verify timings.
    print_report(&config);

    let mut group = c.benchmark_group("lookup");
    group.sample_size(10);

    for case in CASES {
        let (airs, traces, public_values) = build_workload(case);
        let instances = make_instances(&airs, &traces, &public_values);
        let prover_data = ProverData::from_instances(&config, &instances);

        group.bench_function(BenchmarkId::new("prove", case.name), |b| {
            b.iter(|| prove_batch(&config, &instances, &prover_data));
        });

        // A committed proof to verify repeatedly.
        let proof = prove_batch(&config, &instances, &prover_data);
        group.bench_function(BenchmarkId::new("verify", case.name), |b| {
            b.iter(|| {
                verify_batch(&config, &airs, &proof, &public_values, &prover_data.common).unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_lookup);
criterion_main!(benches);
