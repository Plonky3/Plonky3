//! A complete AIR proof over a Boolean trace, committed as bits.
//!
//! Run with `cargo run --release -p p3-multi-stark --example prove_boolean_trace`.
//! The commitment is binding but not hiding; this example does not provide zero knowledge.
//!
//! # The two arms
//!
//! ```text
//!     embedded   one field element per trace bit, committed as elements
//!     packed     one field element per 128 trace bits, committed as bits
//! ```
//!
//! Both arms prove and verify the same statement over the same trace.
//!
//! They differ only in the commitment scheme the configuration selects.
//!
//! The run prints the peak heap and the wall time of each, so the saving is measured.
//!
//! # What the trace costs either way
//!
//! The batched prover lends its trace back as a borrowed table of base-field cells.
//!
//! The base field must be one the challenge field extends.
//!
//! The narrowest such field in the binary tower is `GF(2^8)`.
//!
//! A Boolean commitment also draws its challenges from the field its own elements live in.
//!
//! So a configuration pairing this commitment with this prover holds one cell per bit.
//!
//! That cell is the full challenge width, and the saving is the commitment's alone.

use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_binary_field::{BinaryChallenger, BinaryField128};
use p3_binary_pcs::{
    BinaryPcs, BinaryPcsConfig, BinaryPcsParams, BinaryPcsProverData, BooleanTraceData,
    BooleanTracePcs, GroupedCodewordMmcs,
};
use p3_challenger::{CanObserve, HashChallenger};
use p3_field::PrimeCharacteristicRing;
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_multi_stark::config::{Commitment, MultiStarkConfig, ProverData};
use p3_multi_stark::{
    ProverInstance, ProverInstances, VerifierInstance, VerifierInstances, prove_with_security,
    setup, verify_with_security,
};
use p3_sumcheck::layout::{Layout, SuffixProver, Table, Witness, plan_stacked_layout};
use p3_sumcheck::{PrescribedPointPcs, TableShape};
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type F = BinaryField128;
type Hash = SerializingHasher<Keccak256Hash>;
type Compress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
type MerkleMmcs = p3_merkle_tree::MerkleTreeMmcs<F, u8, Hash, Compress, 2, 32>;
type Mmcs = GroupedCodewordMmcs<MerkleMmcs>;
type Challenger = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;

/// Coordinates one committed element absorbs, so the packing keeps the rest.
const ABSORBED: usize = 7;

/// Columns the gate table holds: three inputs and two outputs.
const WIDTH: usize = 5;

/// Security target both arms are proved and verified against.
const SECURITY_BITS: usize = 100;

/// Trace heights measured, each one instance of the gate table.
const LOG_HEIGHTS: [usize; 3] = [14, 16, 18];

/// A tracking allocator counting live bytes while armed, keeping the high-water mark.
///
/// The lookup benchmark in the batch prover accounts for its memory the same way.
struct TrackingAlloc;

/// Live bytes since the last arm, signed so a pre-arm free cannot underflow.
static LIVE: AtomicIsize = AtomicIsize::new(0);
/// High-water mark of the live count while armed.
static PEAK: AtomicUsize = AtomicUsize::new(0);
/// Whether allocations are currently being counted.
static ARMED: AtomicBool = AtomicBool::new(false);

unsafe impl GlobalAlloc for TrackingAlloc {
    unsafe fn alloc(&self, layout: AllocLayout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() && ARMED.load(Ordering::Relaxed) {
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
fn arm() {
    LIVE.store(0, Ordering::Relaxed);
    PEAK.store(0, Ordering::Relaxed);
    ARMED.store(true, Ordering::Relaxed);
}

/// Stop counting and return the peak live bytes seen while armed.
fn disarm() -> usize {
    ARMED.store(false, Ordering::Relaxed);
    PEAK.load(Ordering::Relaxed)
}

/// A bit-sliced full-adder gate table.
///
/// Every row is one independent adder over `GF(2)`:
///
/// ```text
///     sum  = a + b + cin              exclusive or of the three inputs
///     cout = a*b + cin*a + cin*b      majority of the three inputs
/// ```
///
/// Addition is exclusive or and multiplication is conjunction, but only on Boolean cells.
///
/// Nothing here constrains a cell to be Boolean, and nothing needs to.
///
/// The commitment holds bits, so an accepted proof's trace is Boolean.
///
/// A trace that is not disagrees with the values the opening certifies.
///
/// Every constraint reads the current row only, so no batch asks for a successor view.
struct GateTableAir;

impl<T> BaseAir<T> for GateTableAir {
    fn width(&self) -> usize {
        WIDTH
    }

    fn num_public_values(&self) -> usize {
        // The three inputs of the first row, pinned as the public boundary.
        3
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        // No constraint reads a row ahead, so no column is opened through that view.
        // The default names every column, which would ask for an opening none of them needs.
        Vec::new()
    }
}

impl<AB: AirBuilder<F = F>> Air<AB> for GateTableAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let (a, b, cin) = (local[0], local[1], local[2]);
        let (sum, cout) = (local[3], local[4]);
        let public = builder.public_values();
        let (first_a, first_b, first_cin) = (public[0], public[1], public[2]);

        // Exclusive or of the three inputs, degree one.
        builder.assert_eq(sum, a + b + cin);
        // Majority of the three inputs, degree two.
        builder.assert_eq(cout, a * b + cin * a + cin * b);

        // The statement's public boundary: the first row's inputs.
        builder.when_first_row().assert_eq(a, first_a);
        builder.when_first_row().assert_eq(b, first_b);
        builder.when_first_row().assert_eq(cin, first_cin);
    }
}

/// A satisfying gate table of `2^log_height` rows, and its public boundary.
fn trace(seed: u64, log_height: usize) -> (Table<F>, [F; 3]) {
    let rows = 1usize << log_height;
    let mut rng = SmallRng::seed_from_u64(seed);

    // Draw the three input columns, then derive the two output columns from them.
    let inputs: Vec<[F; 3]> = (0..rows)
        .map(|_| [(); 3].map(|()| F::from_bool(rng.random::<bool>())))
        .collect();

    // Columns are laid out one after another, which is the orientation a table row is.
    let mut columns = Vec::with_capacity(WIDTH * rows);
    for index in 0..3 {
        columns.extend(inputs.iter().map(|row| row[index]));
    }
    columns.extend(inputs.iter().map(|[a, b, c]| *a + *b + *c));
    columns.extend(inputs.iter().map(|[a, b, c]| *a * *b + *c * *a + *c * *b));

    let public = inputs[0];
    (Table::new(RowMajorMatrix::new(columns, rows)), public)
}

/// The shapes one batch of gate tables commits.
fn shapes(log_heights: &[usize]) -> Vec<TableShape> {
    log_heights
        .iter()
        .map(|&log_height| TableShape::new(log_height, WIDTH))
        .collect()
}

/// The commitment schedule for a codeword of `num_variables` committed elements.
fn schedule(num_variables: usize) -> (BinaryPcsConfig, Mmcs) {
    let params = BinaryPcsParams {
        log_inv_rate: 2,
        pow_bits: 0,
        security_level: SECURITY_BITS,
    };
    // Commit after up to three variable folds, with one coset per leaf.
    let config = BinaryPcsConfig::try_new::<F, F>(num_variables, params)
        .unwrap()
        .try_with_folding(3.min(num_variables))
        .unwrap();
    let merkle = MerkleMmcs::new(Hash::new(Keccak256Hash), Compress::new(Keccak256Hash), 0);
    let mmcs = Mmcs::for_folding(merkle, &config);
    (config, mmcs)
}

fn challenger() -> Challenger {
    Challenger::from_hasher(b"p3-multi-stark-boolean-gate-v1".to_vec(), Keccak256Hash)
}

/// The Boolean arm: the trace is committed as bits, packed into the challenge field.
struct PackedConfig {
    /// The bit commitment every column claim is discharged against.
    pcs: BooleanTracePcs<F, Mmcs, Mmcs>,
}

impl PackedConfig {
    /// Wire a configuration for the batch these shapes describe.
    fn new(shapes: &[TableShape]) -> Self {
        // The stacked bit arity is the planner's, so the commitment covers every column.
        let (arity, _) = plan_stacked_layout(shapes);
        let (config, mmcs) = schedule(arity - ABSORBED);
        Self {
            pcs: BooleanTracePcs::new(config, mmcs.clone(), mmcs, arity).unwrap(),
        }
    }
}

impl MultiStarkConfig for PackedConfig {
    type Val = F;
    type Challenge = F;
    type Challenger = Challenger;
    type Pcs = BooleanTracePcs<F, Mmcs, Mmcs>;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }

    fn collision_resistance_bits(&self) -> Option<usize> {
        // Keccak-256 is shared by the transcript and the Merkle tree.
        Some(128)
    }

    fn min_num_variables(&self) -> usize {
        // Every column keeps its own exact run, so no table is zero-extended.
        1
    }

    fn build_witness(&self, tables: Vec<Table<F>>) -> Vec<Table<F>> {
        // The scheme packs the bits itself at commit time, so nothing is built here.
        tables
    }

    fn committed_table<'a>(
        &self,
        prover_data: &'a BooleanTraceData<F, Mmcs>,
        table_index: usize,
    ) -> &'a Table<F> {
        prover_data.table(table_index)
    }
}

/// The reference arm: one committed field element per trace bit.
struct EmbeddedConfig {
    /// The element commitment the stacked trace is discharged against.
    pcs: BinaryPcs<F, F, Mmcs, Mmcs>,
}

impl EmbeddedConfig {
    /// Wire a configuration for the batch these shapes describe.
    fn new(shapes: &[TableShape]) -> Self {
        // The same planner lays out the same slots, one element wide instead of one bit.
        let (arity, _) = plan_stacked_layout(shapes);
        let (config, mmcs) = schedule(arity);
        Self {
            pcs: BinaryPcs::new(config, mmcs.clone(), mmcs).unwrap(),
        }
    }
}

impl MultiStarkConfig for EmbeddedConfig {
    type Val = F;
    type Challenge = F;
    type Challenger = Challenger;
    type Pcs = BinaryPcs<F, F, Mmcs, Mmcs>;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }

    fn collision_resistance_bits(&self) -> Option<usize> {
        Some(128)
    }

    fn min_num_variables(&self) -> usize {
        1
    }

    fn build_witness(&self, tables: Vec<Table<F>>) -> Witness<F> {
        SuffixProver::<F, F>::new_witness(tables, 0)
    }

    fn committed_table<'a>(
        &self,
        prover_data: &'a BinaryPcsProverData<F, F, Mmcs>,
        table_index: usize,
    ) -> &'a Table<F> {
        prover_data.table(table_index)
    }
}

/// Prove and verify one batch under one configuration, returning the proof size.
///
/// The macro-free way to run both arms is one generic function over the configuration.
fn run<C>(config: &C, tables: Vec<Table<F>>, publics: &[[F; 3]]) -> usize
where
    C: MultiStarkConfig<Val = F, Challenge = F, Challenger = Challenger>,
    C::Pcs: PrescribedPointPcs<F, Challenger>,
    Challenger: CanObserve<Commitment<C>>,
    Commitment<C>: Clone,
    ProverData<C>: Clone,
{
    let airs = vec![&GateTableAir; tables.len()];
    let log_heights: Vec<usize> = tables.iter().map(Table::num_variables).collect();
    let (pk, vk) = setup(config, &airs, &mut challenger()).unwrap();

    // One prover instance per table, in the order the tables are committed.
    let prover = ProverInstances::new(
        tables
            .into_iter()
            .zip(publics)
            .map(|(table, public)| ProverInstance::new(&GateTableAir, table, &pk, public))
            .collect(),
    );
    let proof = prove_with_security(config, prover, 0, SECURITY_BITS, &mut challenger())
        .expect("a Boolean gate table must meet the composed target");
    let bytes = postcard::to_allocvec(&proof).unwrap();

    // The verifier reads the heights from the statement, never from the proof.
    let verifier = VerifierInstances::new(
        log_heights
            .iter()
            .zip(publics)
            .map(|(&log_height, public)| {
                VerifierInstance::new(&GateTableAir, &vk, log_height, public)
            })
            .collect(),
    );
    verify_with_security(
        config,
        verifier,
        &proof,
        0,
        SECURITY_BITS,
        &mut challenger(),
    )
    .expect("an honest Boolean gate table proof must verify");
    bytes.len()
}

/// The wall time of one run, measured with the allocator counter disarmed.
fn timed<C>(config: &C, tables: Vec<Table<F>>, publics: &[[F; 3]]) -> (Duration, usize)
where
    C: MultiStarkConfig<Val = F, Challenge = F, Challenger = Challenger>,
    C::Pcs: PrescribedPointPcs<F, Challenger>,
    Challenger: CanObserve<Commitment<C>>,
    Commitment<C>: Clone,
    ProverData<C>: Clone,
{
    let start = Instant::now();
    let bytes = run(config, tables, publics);
    (start.elapsed(), bytes)
}

/// The peak heap of one run, the trace it proves included.
///
/// The trace is built inside the armed region, so the figure covers it too.
///
/// Both arms hold one, so leaving it out would flatter the shorter commitment.
fn measured<C>(config: &C, log_heights: &[usize]) -> usize
where
    C: MultiStarkConfig<Val = F, Challenge = F, Challenger = Challenger>,
    C::Pcs: PrescribedPointPcs<F, Challenger>,
    Challenger: CanObserve<Commitment<C>>,
    Commitment<C>: Clone,
    ProverData<C>: Clone,
{
    arm();
    let (tables, publics) = batch(log_heights);
    let _ = run(config, tables, &publics);
    disarm()
}

/// One gate table per height, all from the same seed so both arms prove one statement.
fn batch(log_heights: &[usize]) -> (Vec<Table<F>>, Vec<[F; 3]>) {
    log_heights
        .iter()
        .enumerate()
        .map(|(index, &log_height)| trace(0xB001 + index as u64, log_height))
        .unzip()
}

fn main() {
    println!("one prove and one verify of a Boolean gate table, both arms\n");
    println!(
        "{:<10} {:>6} {:>12} {:>12} {:>8} {:>10} {:>10} {:>8}",
        "size", "cells", "heap MiB", "heap MiB", "gain", "time ms", "time ms", "gain"
    );
    println!(
        "{:<10} {:>6} {:>12} {:>12} {:>8} {:>10} {:>10} {:>8}\n",
        "", "", "embedded", "packed", "", "embedded", "packed", ""
    );

    for &log_height in &LOG_HEIGHTS {
        let heights = [log_height];
        let shapes = shapes(&heights);
        let packed = PackedConfig::new(&shapes);
        let embedded = EmbeddedConfig::new(&shapes);

        // Peak heap first, then wall time on a fresh trace with the counter disarmed.
        let heap_embedded = measured(&embedded, &heights);
        let heap_packed = measured(&packed, &heights);

        let (tables, publics) = batch(&heights);
        let (time_embedded, _) = timed(&embedded, tables, &publics);
        let (tables, publics) = batch(&heights);
        let (time_packed, size_packed) = timed(&packed, tables, &publics);

        let mib = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);
        println!(
            "2^{log_height:<8} {:>6} {:>12.2} {:>12.2} {:>7.1}x {:>10.0} {:>10.0} {:>7.1}x",
            WIDTH << log_height,
            mib(heap_embedded),
            mib(heap_packed),
            heap_embedded as f64 / heap_packed as f64,
            time_embedded.as_secs_f64() * 1e3,
            time_packed.as_secs_f64() * 1e3,
            time_embedded.as_secs_f64() / time_packed.as_secs_f64(),
        );
        if log_height == *LOG_HEIGHTS.last().unwrap() {
            let cells = (WIDTH << log_height) * size_of::<F>();
            println!(
                "\ntrace held by both arms at 2^{log_height}: {:.2} MiB",
                cells as f64 / (1024.0 * 1024.0)
            );
            println!("packed proof size: {size_packed} bytes");
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_binary_field::TowerLevel;
    use p3_binary_pcs::BooleanTraceError;
    use p3_commit::MultilinearPcs;
    use p3_multi_stark::config::PcsError;
    use p3_multi_stark::zerocheck::ZerocheckError;
    use p3_multi_stark::{VerificationError, security_report};
    use p3_multilinear_util::point::Point;
    use p3_multilinear_util::poly::Poly;
    use p3_sumcheck::{OpeningBatch, OpeningProtocol, PrescribedPointPcs, TableSpec};

    use super::*;

    /// Both arms accept an honest proof, at every height the fixture admits.
    #[test]
    fn a_boolean_gate_table_round_trips_on_both_arms() {
        // Fixture state: one table per height, five columns each.
        //
        //     2^9 rows  ->  5 * 512  = 2560 cells  ->  arity 12
        //     2^10 rows ->  5 * 1024 = 5120 cells  ->  arity 13
        for log_height in [9, 10] {
            let shapes = shapes(&[log_height]);
            let (table, public) = trace(0x9001, log_height);
            run(&PackedConfig::new(&shapes), vec![table], &[public]);
            let (table, public) = trace(0x9001, log_height);
            run(&EmbeddedConfig::new(&shapes), vec![table], &[public]);
        }
    }

    /// Two instances of different heights share one commitment and one proof.
    #[test]
    fn a_batch_of_two_heights_shares_one_bit_commitment() {
        // Fixture state: 5 * 512 + 5 * 128 = 3200 cells, so the stack pads to arity 12.
        let heights = [9, 7];
        let shapes = shapes(&heights);
        let mut tables = Vec::new();
        let mut publics = Vec::new();
        for (index, &log_height) in heights.iter().enumerate() {
            let (table, public) = trace(0x9100 + index as u64, log_height);
            tables.push(table);
            publics.push(public);
        }
        run(&PackedConfig::new(&shapes), tables, &publics);
    }

    /// A public boundary the trace does not meet is rejected.
    #[test]
    fn rejects_changed_public_values() {
        // Invariant: the three first-row constraints are what tie the trace to the statement.
        //
        // The public values are absorbed before the zerocheck draws anything, so:
        //
        //     prove with p, verify with p'   every later challenge moves, so any proof fails
        //     prove with p', verify with p'  the two sponges agree, so only the AIR can reject
        //
        // The second is the one that reaches the boundary check, so it is the one run here.
        // The first would pass with the boundary constraints deleted, which pins nothing.
        let log_height = 9;
        let shapes = shapes(&[log_height]);
        let config = PackedConfig::new(&shapes);
        let (table, public) = trace(0x9200, log_height);
        let (pk, vk) = setup(&config, &[&GateTableAir], &mut challenger()).unwrap();

        // Mutation: flip each pinned input in turn, on both sides at once.
        //
        //     trace row 0   a, b, cin, exactly as the table holds them
        //     statement     one of the three moved by one
        //
        // The trace never changes, so exactly one first-row equality is false.
        for index in 0..3 {
            let mut tampered = public;
            tampered[index] += F::ONE;
            let proof = prove_with_security(
                &config,
                ProverInstances::new(vec![ProverInstance::new(
                    &GateTableAir,
                    table.clone(),
                    &pk,
                    &tampered,
                )]),
                0,
                SECURITY_BITS,
                &mut challenger(),
            )
            .unwrap();
            let result: Result<(), VerificationError<PcsError<PackedConfig>>> =
                verify_with_security(
                    &config,
                    VerifierInstances::new(vec![VerifierInstance::new(
                        &GateTableAir,
                        &vk,
                        log_height,
                        &tampered,
                    )]),
                    &proof,
                    0,
                    SECURITY_BITS,
                    &mut challenger(),
                );

            // The batched constraint is nonzero at the first row, so the closing check fails.
            // An opening-side rejection would mean the transcripts split instead.
            assert!(
                matches!(
                    result,
                    Err(VerificationError::Zerocheck(
                        ZerocheckError::FinalSumMismatch
                    ))
                ),
                "public value {index}: {result:?}"
            );
        }

        // The untampered statement is accepted, so the rejections are the boundary alone.
        let proof = prove_with_security(
            &config,
            ProverInstances::new(vec![ProverInstance::new(
                &GateTableAir,
                table,
                &pk,
                &public,
            )]),
            0,
            SECURITY_BITS,
            &mut challenger(),
        )
        .unwrap();
        verify_with_security(
            &config,
            VerifierInstances::new(vec![VerifierInstance::new(
                &GateTableAir,
                &vk,
                log_height,
                &public,
            )]),
            &proof,
            0,
            SECURITY_BITS,
            &mut challenger(),
        )
        .unwrap();
    }

    /// A trace whose carry column is wrong is rejected.
    #[test]
    fn rejects_an_invalid_gate_table() {
        // Fixture state: row 3's majority output is flipped, the boundary row untouched.
        //
        //     column 4 (cout), cell 3  ->  cout + 1
        //
        // The zerocheck's degree-two constraint fails at that row.
        let log_height = 9;
        let rows = 1usize << log_height;
        let shapes = shapes(&[log_height]);
        let config = PackedConfig::new(&shapes);
        let (table, public) = trace(0x9300, log_height);

        let mut cells = table.iter_polys().flatten().copied().collect::<Vec<_>>();
        cells[4 * rows + 3] += F::ONE;
        let table = Table::new(RowMajorMatrix::new(cells, rows));

        let (pk, vk) = setup(&config, &[&GateTableAir], &mut challenger()).unwrap();
        let proof = prove_with_security(
            &config,
            ProverInstances::new(vec![ProverInstance::new(
                &GateTableAir,
                table,
                &pk,
                &public,
            )]),
            0,
            SECURITY_BITS,
            &mut challenger(),
        )
        .unwrap();
        let result: Result<(), VerificationError<PcsError<PackedConfig>>> = verify_with_security(
            &config,
            VerifierInstances::new(vec![VerifierInstance::new(
                &GateTableAir,
                &vk,
                log_height,
                &public,
            )]),
            &proof,
            0,
            SECURITY_BITS,
            &mut challenger(),
        );

        // The majority constraint is nonzero at row 3, so the closing check is what fails.
        // Accepting any rejection would let an opening-side failure pass this test.
        assert!(
            matches!(
                result,
                Err(VerificationError::Zerocheck(
                    ZerocheckError::FinalSumMismatch
                ))
            ),
            "{result:?}"
        );
    }

    /// The report charges the commitment and the reduction, each under its own label.
    #[test]
    fn the_report_charges_every_part_of_the_boolean_path() {
        let log_height = 9;
        let shapes = shapes(&[log_height]);
        let config = PackedConfig::new(&shapes);
        let (_, vk) = setup(&config, &[&GateTableAir], &mut challenger()).unwrap();
        let public = [F::ZERO; 3];
        let instances = VerifierInstances::new(vec![VerifierInstance::new(
            &GateTableAir,
            &vk,
            log_height,
            &public,
        )]);

        let report = security_report(&config, &instances).unwrap();
        assert!(report.unassessed_components().is_empty());

        // Every part of the Boolean path reaches the report under its own label.
        //
        //     binary-pcs-opening   the commitment, its own claim batching included
        //     bit-ring-switch      one reduction per batch
        //     column-batching      the column point folding a batch's claimed values
        //
        // All three are attributed to the commitment they were charged for.
        // A report holding two openings then still says which of the three is short.
        for label in ["binary-pcs-opening", "bit-ring-switch", "column-batching"] {
            assert!(
                report
                    .terms()
                    .iter()
                    .any(|term| term.label == label && term.component == Some("main-pcs")),
                "missing {label}"
            );
        }

        report.require_security(SECURITY_BITS).unwrap();
        assert!(report.require_security(128).is_err());
    }

    /// A table whose transition constraint reads the row after the current one.
    ///
    /// The declaration is a field, so the same constraints can be run both ways.
    ///
    /// ```text
    ///     complete     the column the constraint reads a row ahead is named
    ///     incomplete   nothing is named, which no AIR is allowed to do
    /// ```
    struct SuccessorAir {
        /// Whether the declaration names the column the constraint reads a row ahead.
        complete: bool,
    }

    impl<T> BaseAir<T> for SuccessorAir {
        fn width(&self) -> usize {
            WIDTH
        }

        fn num_public_values(&self) -> usize {
            // This table pins no boundary, so the statement carries nothing.
            0
        }

        fn main_next_row_columns(&self) -> Vec<usize> {
            // Column 3 is the one the transition constraint reads a row ahead.
            if self.complete { vec![3] } else { Vec::new() }
        }
    }

    impl<AB: AirBuilder<F = F>> Air<AB> for SuccessorAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let local = main.current_slice();
            let next = main.next_slice();

            // The next row's sum column is the exclusive or of this row's three inputs.
            builder
                .when_transition()
                .assert_eq(next[3], local[0] + local[1] + local[2]);
        }
    }

    /// The same statement, assessed against whichever declaration the flag selects.
    fn assess_successor_air(complete: bool) {
        let log_height = 9;
        let shapes = shapes(&[log_height]);
        let config = PackedConfig::new(&shapes);
        let air = SuccessorAir { complete };
        let (_, vk) = setup(&config, &[&air], &mut challenger()).unwrap();
        let instances =
            VerifierInstances::new(vec![VerifierInstance::new(&air, &vk, log_height, &[])]);
        security_report(&config, &instances).unwrap();
    }

    #[test]
    fn a_declared_successor_read_is_assessed() {
        // Naming the column the constraint reads a row ahead is all that is asked.
        //
        // This is the control for the refusal below.
        // Same constraints and same shapes, with only the declaration moving.
        assess_successor_air(true);
    }

    #[test]
    #[should_panic = "AIR reads an undeclared successor column"]
    fn an_undeclared_successor_read_is_refused() {
        // Invariant: the declaration must name every column a constraint reads ahead.
        //
        // Undeclared next-row values are folded as zero by both sides:
        //
        //     declares nothing, reads the next row of column 3  ->  that read is a zero
        //
        // Proving the weaker statement instead would be silent.
        // So the pass fixing the round degree compares the two and refuses a mismatch.
        //
        // That pass runs on the proving and the verifying path alike.
        // An incomplete declaration reaches neither a commitment nor a transcript.
        assess_successor_air(false);
    }

    /// A cell outside the two Boolean values is refused at commitment.
    #[test]
    fn a_non_boolean_cell_is_refused() {
        // Fixture state: one otherwise valid table with a single cell set to a tower element.
        let log_height = 9;
        let rows = 1usize << log_height;
        let shapes = shapes(&[log_height]);
        let config = PackedConfig::new(&shapes);
        let (table, _) = trace(0x9400, log_height);

        let mut cells = table.iter_polys().flatten().copied().collect::<Vec<_>>();
        cells[7] = F::from_repr(0x1234);
        let table = Table::new(RowMajorMatrix::new(cells, rows));

        let Err(error) = config.pcs.commit(vec![table], &mut challenger()) else {
            panic!("a non-Boolean cell addresses no bit")
        };
        assert!(matches!(
            error,
            BooleanTraceError::NonBooleanCell {
                table: 0,
                column: 0
            }
        ));
    }

    /// A batch reading one row ahead is answered alongside the current row.
    #[test]
    fn a_successor_view_round_trips() {
        // Fixture state: one table, one batch naming column 0 on both sides.
        //
        //     current [0]   an evaluation at the point
        //     next    [0]   the same column weighted by a shifted equality table
        //
        // One reduction answers both, and the verifier returns the successor reading.
        let log_height = 9;
        let shapes = shapes(&[log_height]);
        let config = PackedConfig::new(&shapes);
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            shapes[0],
            vec![OpeningBatch::new(vec![0], vec![0])],
        )]);

        // The protocol is assessed, so a security-checked caller can run it.
        assert!(
            PrescribedPointPcs::<F, Challenger>::prescribed_security(&config.pcs, &protocol)
                .is_some()
        );

        let (table, _) = trace(0x9500, log_height);
        let column = Poly::new(table.poly(0).as_slice().to_vec());
        let point = Point::<F>::rand(&mut SmallRng::seed_from_u64(0x9501), log_height);

        let mut prover_chal = challenger();
        let (commitment, data) = config.pcs.commit(vec![table], &mut prover_chal).unwrap();
        let proof = config
            .pcs
            .open_at(
                data,
                &protocol,
                core::slice::from_ref(&point),
                &mut prover_chal,
            )
            .unwrap();

        let mut verifier_chal = challenger();
        config
            .pcs
            .observe_commitment(&commitment, &mut verifier_chal);
        let evals = config
            .pcs
            .verify_at(
                &commitment,
                &proof,
                &protocol,
                core::slice::from_ref(&point),
                &mut verifier_chal,
            )
            .unwrap();

        // Row x reads row x + 1 and the last row reads itself, weighted at the point.
        assert_eq!(evals[0].next()[0], column.eval_next_base(&point));
    }

    /// A Boolean trace whose sum column is set by the row before, as `SuccessorAir` asks.
    ///
    /// ```text
    ///     columns 0, 1, 2, 4   random bits
    ///     column 3, row 0      a random bit
    ///     column 3, row x + 1  column 0 + column 1 + column 2 at row x
    /// ```
    fn successor_trace(seed: u64, log_height: usize) -> Table<F> {
        let rows = 1usize << log_height;
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut columns: Vec<Vec<F>> = (0..WIDTH)
            .map(|_| {
                (0..rows)
                    .map(|_| F::from_bool(rng.random::<bool>()))
                    .collect()
            })
            .collect();
        for row in 1..rows {
            columns[3][row] = columns[0][row - 1] + columns[1][row - 1] + columns[2][row - 1];
        }
        Table::new(RowMajorMatrix::new(columns.concat(), rows))
    }

    /// A register whose every column is read one row ahead.
    ///
    /// ```text
    ///     next[i]         = local[i + 1]        for i below the last column
    ///     next[WIDTH - 1] = local[0] + local[1]
    /// ```
    ///
    /// Every column is read a row ahead, so the default declaration names them all and the
    /// batch opens the whole width through both views.
    struct ShiftRegisterAir;

    impl<T> BaseAir<T> for ShiftRegisterAir {
        fn width(&self) -> usize {
            WIDTH
        }

        fn num_public_values(&self) -> usize {
            // Each row follows from the one before it, so the statement pins no boundary.
            0
        }
    }

    impl<AB: AirBuilder<F = F>> Air<AB> for ShiftRegisterAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let local = main.current_slice();
            let next = main.next_slice();

            // The last row reads itself under the repeat-last view, so the shift is stated
            // on the transition rows alone.
            for column in 0..WIDTH - 1 {
                builder
                    .when_transition()
                    .assert_eq(next[column], local[column + 1]);
            }
            builder
                .when_transition()
                .assert_eq(next[WIDTH - 1], local[0] + local[1]);
        }
    }

    /// A trace the shift register satisfies, grown row by row from a random Boolean row.
    fn shift_register_trace(seed: u64, log_height: usize) -> Table<F> {
        let rows = 1usize << log_height;
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut columns: Vec<Vec<F>> = (0..WIDTH)
            .map(|_| {
                let mut column = vec![F::ZERO; rows];
                column[0] = F::from_bool(rng.random::<bool>());
                column
            })
            .collect();
        for row in 1..rows {
            let previous: [F; WIDTH] = core::array::from_fn(|column| columns[column][row - 1]);
            for column in 0..WIDTH - 1 {
                columns[column][row] = previous[column + 1];
            }
            // Exclusive or of two bits is a bit, so every cell the register grows is one.
            columns[WIDTH - 1][row] = previous[0] + previous[1];
        }

        let table = Table::new(RowMajorMatrix::new(columns.concat(), rows));
        // An all-zero table satisfies the shift while saying nothing about it, so the
        // successor check would pass on a trace that exercises none of it.
        assert!(
            table.iter_polys().flatten().any(|&cell| cell != F::ZERO),
            "the shift register must hold a cell the constraint can move"
        );
        table
    }

    /// An AIR reading every column one row ahead proves through the batched successor route.
    #[test]
    fn a_whole_width_successor_air_proves_on_the_boolean_commitment() {
        // Fixture state: one table of 2^9 rows, all five columns read at both rows.
        //
        // That is the complete-batch shape, so one reduction over one shared column point
        // answers both views of the batch, and the report charges both parts of it.
        let log_height = 9;
        let shapes = shapes(&[log_height]);
        let config = PackedConfig::new(&shapes);
        let air = ShiftRegisterAir;
        let (pk, vk) = setup(&config, &[&air], &mut challenger()).unwrap();

        let report = security_report(
            &config,
            &VerifierInstances::new(vec![VerifierInstance::new(&air, &vk, log_height, &[])]),
        )
        .unwrap();
        assert!(report.unassessed_components().is_empty());
        for label in ["bit-ring-switch", "column-batching"] {
            assert!(
                report
                    .terms()
                    .iter()
                    .any(|term| term.label == label && term.component == Some("main-pcs")),
                "missing {label}"
            );
        }
        report.require_security(SECURITY_BITS).unwrap();

        // Both views of the batch are combined at the one column point they share, so the
        // batching term charges two claims over the three coordinates a width of five needs.
        //
        //     128 - log2(2 * 3) = 125.415...
        //
        // A batch reading the current row alone would carry one claim and charge more.
        let batching = report
            .terms()
            .iter()
            .find(|term| term.label == "column-batching")
            .expect("the optimized route charges the column point it draws");
        assert!((batching.bits.bits() - (128.0 - 6f64.log2())).abs() < 1e-9);

        let proof = prove_with_security(
            &config,
            ProverInstances::new(vec![ProverInstance::new(
                &air,
                shift_register_trace(0x9700, log_height),
                &pk,
                &[],
            )]),
            0,
            SECURITY_BITS,
            &mut challenger(),
        )
        .unwrap();
        verify_with_security(
            &config,
            VerifierInstances::new(vec![VerifierInstance::new(&air, &vk, log_height, &[])]),
            &proof,
            0,
            SECURITY_BITS,
            &mut challenger(),
        )
        .unwrap();
    }

    /// An AIR linking each row to the next proves on the bit commitment.
    #[test]
    fn a_transition_air_proves_on_the_boolean_commitment() {
        // Fixture state: one table of 2^9 rows, column 3 opened now and one row ahead.
        //
        // The transition constraint reads column 3 at the next row, so the batch asks for both.
        let log_height = 9;
        let shapes = shapes(&[log_height]);
        let config = PackedConfig::new(&shapes);
        let air = SuccessorAir { complete: true };
        let (pk, vk) = setup(&config, &[&air], &mut challenger()).unwrap();

        let proof = prove_with_security(
            &config,
            ProverInstances::new(vec![ProverInstance::new(
                &air,
                successor_trace(0x9600, log_height),
                &pk,
                &[],
            )]),
            0,
            SECURITY_BITS,
            &mut challenger(),
        )
        .unwrap();
        verify_with_security(
            &config,
            VerifierInstances::new(vec![VerifierInstance::new(&air, &vk, log_height, &[])]),
            &proof,
            0,
            SECURITY_BITS,
            &mut challenger(),
        )
        .unwrap();
    }
}
