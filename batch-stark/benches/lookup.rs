//! End-to-end lookup benchmark harness.
//!
//! Drives parametric lookup workloads through the full batch-STARK pipeline.
//!
//! Reports the numbers a lookup-layer change actually moves:
//!
//!     prove time       criterion-timed
//!     verify time      criterion-timed
//!     aux columns      committed permutation columns (the layout target)
//!     proof size       serialized bytes
//!     peak heap        high-water bytes during one prove
//!
//! Two workload families exercise the two ways lookups fold into columns:
//!
//!     same-bus    co-firing interactions summed into a column; degree grows with the fold
//!     exclusive   one-of-N branches multiplexed into a column; degree stays flat
//!
//! Each case is measured as a pair of layout variants, so the win reads off the pair:
//!
//!     same-bus    packed    vs unpacked     (folding on vs one column per interaction)
//!     exclusive   exclusive vs additive     (the selector multiplex vs one column per branch)
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
// Workload AIRs
// ---------------------------------------------------------------------------

/// The single bus every interaction speaks on.
///
/// One shared bus lets the packer fold interactions into common columns.
const SHARED_BUS: LookupBus<'static> = LookupBus::new("shared");

/// A balanced same-bus lookup AIR with a tunable base-constraint degree.
///
/// - Every slot queries a key and provides the same key on the shared bus.
/// - Query and provide cancel, so any trace balances to a zero terminal.
/// - All slots fire together on an active row, so they are summed, not multiplexed.
/// - An optional degree-`base_degree` constraint sets the spare quotient budget.
/// - That budget is exactly what the same-bus packer folds into.
#[derive(Clone, Copy)]
struct SameBusAir {
    /// Query/provide pairs the AIR declares, all on the shared bus.
    n_interactions: usize,
    /// Field elements per message.
    tuple_width: usize,
    /// Degree of the base constraint, or 0 for none.
    base_degree: usize,
}

impl SameBusAir {
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

impl<F: Field> BaseAir<F> for SameBusAir {
    fn width(&self) -> usize {
        // Per-interaction columns, then the optional two-column witness.
        self.n_interactions * self.stride() + if self.has_base() { 2 } else { 0 }
    }
}

impl<AB: PermutationAirBuilder + InteractionBuilder> Air<AB> for SameBusAir {
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

/// A balanced one-of-N lookup AIR, emitted with or without the exclusive layout.
///
/// - One selector flag per branch, exactly one active per row.
/// - Each branch queries its key and provides the same key, so the bus balances.
/// - Exclusive layout: two columns, one for the query side, one for the provide.
/// - Additive layout: one column per branch per side, so `2 * n_branches` in total.
///
/// Both layouts compute the same per-row bus contribution.
/// They differ only in how many auxiliary columns carry it.
#[derive(Clone, Copy)]
struct ExclusiveAir {
    /// Mutually-exclusive branches the AIR declares.
    n_branches: usize,
    /// Whether to use the exclusive layout or the per-branch baseline.
    exclusive: bool,
}

impl<F: Field> BaseAir<F> for ExclusiveAir {
    fn width(&self) -> usize {
        // Layout per row: n_branches flags, then n_branches width-1 keys.
        2 * self.n_branches
    }
}

impl<AB: PermutationAirBuilder + InteractionBuilder> Air<AB> for ExclusiveAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let m = self.n_branches;

        // Selector flags are boolean in both layouts.
        for &flag in &local[..m] {
            let f: AB::Expr = flag.into();
            builder.assert_zero(f.clone() * (f - AB::Expr::ONE));
        }

        if self.exclusive {
            // Exclusivity obligation: at most one flag fires per row.
            // A boolean sum of booleans is `0` or `1`, so this pins "at most one".
            let mut sum: AB::Expr = AB::Expr::ZERO;
            for &flag in &local[..m] {
                sum += flag.into();
            }
            builder.assert_zero(sum.clone() * (sum - AB::Expr::ONE));

            // One exclusive column queries the active branch's key.
            let query: Vec<(AB::Expr, Vec<AB::Expr>)> = (0..m)
                .map(|k| (local[k].into(), vec![local[m + k].into()]))
                .collect();
            SHARED_BUS.lookup_key_exclusive(builder, query);

            // One exclusive column provides the same active key, multiplicity -1.
            builder.push_exclusive_interaction(
                SHARED_BUS.name(),
                (0..m).map(|k| {
                    (
                        local[k].into(),
                        Count::provided(-AB::Expr::ONE),
                        vec![local[m + k].into()],
                    )
                }),
            );
        } else {
            // Baseline: one gated query column and one provide column per branch.
            for k in 0..m {
                let key = vec![local[m + k].into()];
                let flag = local[k];
                SHARED_BUS.lookup_key(builder, key.clone(), Count::bounded(flag.into(), 1));
                SHARED_BUS.table_entry(builder, key, flag.into());
            }
        }
    }
}

/// The workload AIRs the harness can prove, behind one monomorphic type.
///
/// A single type lets every case flow through one measure, report, and timing path.
#[derive(Clone, Copy)]
enum BenchAir {
    /// Co-firing same-bus interactions, summed into folded columns.
    SameBus(SameBusAir),
    /// One-of-N branches, in either the exclusive or the per-branch layout.
    Exclusive(ExclusiveAir),
}

impl<F: Field> BaseAir<F> for BenchAir {
    fn width(&self) -> usize {
        match self {
            Self::SameBus(a) => BaseAir::<F>::width(a),
            Self::Exclusive(a) => BaseAir::<F>::width(a),
        }
    }
}

impl<AB: PermutationAirBuilder + InteractionBuilder> Air<AB> for BenchAir {
    fn eval(&self, builder: &mut AB) {
        match self {
            Self::SameBus(a) => a.eval(builder),
            Self::Exclusive(a) => a.eval(builder),
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration matrix
// ---------------------------------------------------------------------------

/// Which workload family a case exercises, with its shape knobs.
#[derive(Clone, Copy)]
enum Workload {
    /// Co-firing interactions on the shared bus; the packer may fold them.
    SameBus {
        /// Query/provide pairs the AIR declares.
        n_interactions: usize,
        /// Field elements per message.
        tuple_width: usize,
        /// Degree of the base constraint, or 0 for none.
        base_degree: usize,
    },
    /// Mutually-exclusive branches; one fires per row.
    Exclusive {
        /// Branches folded into a single column by the exclusive layout.
        n_branches: usize,
    },
}

/// One benchmark point.
#[derive(Clone, Copy)]
struct Case {
    /// Label shown in the table and in criterion ids.
    name: &'static str,
    /// log2 of the trace height.
    log_height: usize,
    /// Workload family and its shape.
    workload: Workload,
    /// One active row every `active_period` rows; applies to the same-bus family only.
    active_period: usize,
    /// Number of identical AIR instances in the batch.
    n_airs: usize,
}

/// Same-bus case constructor.
const fn same_bus(
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
        workload: Workload::SameBus {
            n_interactions,
            tuple_width,
            base_degree,
        },
        active_period,
        n_airs,
    }
}

/// Exclusive case constructor.
const fn exclusive(name: &'static str, log_height: usize, n_branches: usize) -> Case {
    Case {
        name,
        log_height,
        workload: Workload::Exclusive { n_branches },
        active_period: 1,
        n_airs: 1,
    }
}

/// The default sweep, grouped by the axis each block isolates.
///
/// Comment out blocks to shorten a local run.
const CASES: &[Case] = &[
    // Smoke point: tiny, validates the pipeline fast.
    same_bus("smoke", 8, 1, 1, 0, 1, 1),
    // Interaction count: cost as a function of #lookups, one column each.
    same_bus("n=1", 14, 1, 1, 0, 1, 1),
    same_bus("n=4", 14, 4, 1, 0, 1, 1),
    same_bus("n=8", 14, 8, 1, 0, 1, 1),
    same_bus("n=16", 14, 16, 1, 0, 1, 1),
    // Tuple width: wider messages, same per-lookup degree.
    same_bus("tw=2", 14, 4, 2, 0, 1, 1),
    same_bus("tw=4", 14, 4, 4, 0, 1, 1),
    // Row-sparsity: fraction of active rows. Baseline cost is sparsity-blind today.
    same_bus("period=4", 14, 4, 1, 0, 4, 1),
    same_bus("period=16", 14, 4, 1, 0, 16, 1),
    // Batch size: several identical instances proven together.
    same_bus("airs=4", 14, 4, 1, 0, 1, 4),
    // Spare budget: a degree-3 base constraint lets the packer fold two per column.
    // The benchmark FRI uses log-blowup 1, capping the quotient at degree 3.
    // So degree 3 is the deepest packing this config can show.
    same_bus("deg3_n=8", 14, 8, 1, 3, 1, 1),
    same_bus("deg3_n=16", 14, 16, 1, 3, 1, 1),
    // Exclusivity: one-of-N per row collapses to a single column, flat degree.
    exclusive("excl_n=8", 16, 8),
    exclusive("excl_n=16", 16, 16),
    exclusive("excl_n=32", 16, 32),
    // Heavy points at log_height 18: comment these out for a faster bench.
    same_bus("n=32_h18", 18, 32, 1, 0, 1, 1),
    same_bus("deg3_n=32_h18", 18, 32, 1, 3, 1, 1),
];

// ---------------------------------------------------------------------------
// Layout variants
// ---------------------------------------------------------------------------

/// One layout to prove for a case, plus how to measure it.
struct Variant {
    /// Label shown next to the case name.
    label: &'static str,
    /// The AIR description for this layout.
    air: BenchAir,
    /// Whether to overwrite the folded layout with one column per interaction.
    force_unpacked: bool,
    /// Whether to also run the criterion timing for this variant.
    criterion: bool,
}

/// The pair of layouts a case is compared across.
///
/// - Same-bus: the folded layout against one column per interaction.
/// - Exclusive: the selector multiplex against one column per branch.
fn variants(case: &Case) -> Vec<Variant> {
    match case.workload {
        Workload::SameBus {
            n_interactions,
            tuple_width,
            base_degree,
        } => {
            let air = BenchAir::SameBus(SameBusAir {
                n_interactions,
                tuple_width,
                base_degree,
            });
            // The unpacked layout differs from the packed one only with spare budget.
            // Without a base constraint the packer folds nothing, so the rows match.
            let time_unpacked = base_degree >= 2;
            vec![
                Variant {
                    label: "packed",
                    air,
                    force_unpacked: false,
                    criterion: true,
                },
                Variant {
                    label: "unpacked",
                    air,
                    force_unpacked: true,
                    criterion: time_unpacked,
                },
            ]
        }
        Workload::Exclusive { n_branches } => vec![
            Variant {
                label: "exclusive",
                air: BenchAir::Exclusive(ExclusiveAir {
                    n_branches,
                    exclusive: true,
                }),
                force_unpacked: false,
                criterion: true,
            },
            Variant {
                label: "additive",
                air: BenchAir::Exclusive(ExclusiveAir {
                    n_branches,
                    exclusive: false,
                }),
                force_unpacked: false,
                criterion: true,
            },
        ],
    }
}

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

/// Build the single-instance trace for a case.
fn build_case_trace(case: &Case) -> RowMajorMatrix<Val> {
    let height = 1 << case.log_height;
    let mut rng = SmallRng::seed_from_u64(0xD1CE_F00D);

    match case.workload {
        Workload::SameBus {
            n_interactions,
            tuple_width,
            base_degree,
        } => {
            let air = SameBusAir {
                n_interactions,
                tuple_width,
                base_degree,
            };
            let width = <SameBusAir as BaseAir<Val>>::width(&air);
            let witness = air.witness_offset();

            let mut values = Val::zero_vec(height * width);
            for r in 0..height {
                let base = r * width;
                // A row is active once every `active_period` rows.
                let active = r % case.active_period == 0;

                // Keys are arbitrary: query and provide cancel whatever the key holds.
                for i in 0..n_interactions {
                    let off = base + i * (tuple_width + 1);
                    for j in 0..tuple_width {
                        values[off + j] = Val::from_u32(rng.random());
                    }
                    // Selector gates both sides, so inactive rows contribute nothing.
                    values[off + tuple_width] = if active { Val::ONE } else { Val::ZERO };
                }

                // Witness pair satisfies the base constraint: y = w^base_degree.
                if air.has_base() {
                    let w = Val::from_u32(rng.random());
                    values[base + witness] = w;
                    values[base + witness + 1] = w.exp_u64(base_degree as u64);
                }
            }
            RowMajorMatrix::new(values, width)
        }
        Workload::Exclusive { n_branches } => {
            let width = 2 * n_branches;

            let mut values = Val::zero_vec(height * width);
            for r in 0..height {
                let base = r * width;
                // Random keys: a query and its provide cancel whatever the key holds.
                for k in 0..n_branches {
                    values[base + n_branches + k] = Val::from_u32(rng.random());
                }
                // Exactly one flag fires, satisfying the boolean and at-most-one rules.
                values[base + r % n_branches] = Val::ONE;
            }
            RowMajorMatrix::new(values, width)
        }
    }
}

/// Replicate one instance into a full batch of `n_airs` identical instances.
fn build_batch(
    case: &Case,
    air: BenchAir,
) -> (Vec<BenchAir>, Vec<RowMajorMatrix<Val>>, Vec<Vec<Val>>) {
    let trace = build_case_trace(case);
    let airs = vec![air; case.n_airs];
    let traces = vec![trace; case.n_airs];
    let public_values = vec![Vec::new(); case.n_airs];
    (airs, traces, public_values)
}

/// Bundle parallel slices into prover instances.
fn make_instances<'a>(
    airs: &'a [BenchAir],
    traces: &'a [RowMajorMatrix<Val>],
    public_values: &[Vec<Val>],
) -> Vec<StarkInstance<'a, MyConfig, BenchAir>> {
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
fn force_unpacked(prover_data: &mut ProverData<MyConfig>, airs: &[BenchAir]) {
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

/// Prove and verify one variant once, checking correctness and collecting metrics.
fn measure(config: &MyConfig, case: &Case, variant: &Variant) -> Metrics {
    let (airs, traces, public_values) = build_batch(case, variant.air);
    let instances = make_instances(&airs, &traces, &public_values);
    let mut prover_data = ProverData::from_instances(config, &instances);
    if variant.force_unpacked {
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
/// Each case prints its two variants, so the win reads straight off the pair.
fn print_report(config: &MyConfig) {
    eprintln!();
    eprintln!("lookup end-to-end metrics (each case as a pair of layout variants)");
    eprintln!(
        "{:<16} {:>5} {:>10} {:>8} {:>10} {:>9} {:>9} {:>10}",
        "case", "logH", "variant", "aux_cols", "proof(KB)", "peak(MB)", "prove(ms)", "verify(ms)",
    );
    for case in CASES {
        for variant in variants(case) {
            let m = measure(config, case, &variant);
            eprintln!(
                "{:<16} {:>5} {:>10} {:>8} {:>10.1} {:>9.1} {:>9.1} {:>10.2}",
                case.name,
                case.log_height,
                variant.label,
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

    // The table carries aux columns, proof size, and peak heap for every variant.
    // Criterion below carries the rigorous prove and verify timings.
    print_report(&config);

    let mut group = c.benchmark_group("lookup");
    group.sample_size(10);

    for case in CASES {
        for variant in variants(case) {
            // Skip variants whose timing would duplicate another (e.g. unpacked == packed).
            if !variant.criterion {
                continue;
            }

            let (airs, traces, public_values) = build_batch(case, variant.air);
            let instances = make_instances(&airs, &traces, &public_values);
            let mut prover_data = ProverData::from_instances(&config, &instances);
            if variant.force_unpacked {
                force_unpacked(&mut prover_data, &airs);
            }

            let id = format!("{}/{}", case.name, variant.label);

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
