//! Peak heap during one commit and one opening of a bit witness.
//!
//! Commitment volume is the reason to pack a narrow alphabet.
//!
//! So the space one prove takes is the figure that says whether it worked.
//!
//! A tracking allocator counts every live byte of the process, not a delta from some mark.
//!
//! ```text
//!     live   bytes allocated and not yet freed, counted from the first allocation
//!     peak   the highest that count has reached since the region opened
//! ```
//!
//! Counting a delta instead would leave out what a region merely holds.
//!
//! An opening holds the codeword and the tree the commitment built, for its whole run.
//!
//! A delta would also go negative when such a buffer is freed, hiding the next allocation.
//!
//! Each region therefore opens before its own inputs exist and closes before the next opens.
//!
//! The figures are taken with one worker thread, since a pool makes the peak depend on it.
//!
//! The lookup benchmark in the batch prover accounts for its memory the same way.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use p3_binary_field::{BinaryChallenger, BinaryField128, Gf2, PackedGf2x64};
use p3_binary_pcs::{
    BinaryPcs, BinaryPcsConfig, BinaryPcsParams, BooleanMultilinearPcs, BooleanPcs,
};
use p3_challenger::HashChallenger;
use p3_commit::MultilinearPcs;
use p3_field::PrimeCharacteristicRing;
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_sumcheck::layout::{Layout as _, SuffixProver, Table};
use p3_sumcheck::{OpeningBatch, OpeningProtocol, PrescribedPointPcs, TableShape, TableSpec};
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type EF = BinaryField128;
type MyHash = SerializingHasher<Keccak256Hash>;
type MyCompress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
type MyMmcs = MerkleTreeMmcs<EF, u8, MyHash, MyCompress, 2, 32>;
type MyChallenger = BinaryChallenger<EF, HashChallenger<u8, Keccak256Hash, 32>>;

/// Bit counts measured, matching the timing benchmark's shapes.
const LOG_BITS: [usize; 3] = [16, 18, 20];

/// Coordinates the packing absorbs, so the committed polynomial keeps the rest.
const ABSORBED: usize = 7;

/// System allocator that tracks the live byte count and its high-water mark.
struct TrackingAlloc;

/// Bytes allocated and not yet freed, counted from the program's first allocation.
static LIVE: AtomicUsize = AtomicUsize::new(0);
/// Highest the live count has reached since the current region opened.
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            // Every deallocation below pairs an allocation here, so the count never wraps.
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static GLOBAL: TrackingAlloc = TrackingAlloc;

/// Open a region: the high-water mark restarts from what is live right now.
///
/// Whatever the previous region left behind is therefore charged to this one too.
fn open_region() {
    PEAK.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
}

/// The highest live byte count reached since the region opened.
fn peak() -> usize {
    PEAK.load(Ordering::Relaxed)
}

const fn mmcs() -> MyMmcs {
    MyMmcs::new(
        MyHash::new(Keccak256Hash),
        MyCompress::new(Keccak256Hash),
        0,
    )
}

const fn challenger() -> MyChallenger {
    BinaryChallenger::new(HashChallenger::new(Vec::new(), Keccak256Hash))
}

const fn params() -> BinaryPcsParams {
    BinaryPcsParams {
        log_inv_rate: 2,
        pow_bits: 0,
        security_level: 100,
    }
}

/// A random bit-sliced witness of the given log bit count.
fn witness(log_bits: usize) -> Vec<PackedGf2x64> {
    let mut rng = SmallRng::seed_from_u64(0xB001);
    (0..1 << (log_bits - 6))
        .map(|_| PackedGf2x64::new(rng.random::<u64>()))
        .collect()
}

/// The same bits as one element per bit, which is what no packing commits.
fn embedded(bits: &[PackedGf2x64]) -> Vec<EF> {
    bits.iter()
        .flat_map(|block| {
            (0..PackedGf2x64::WIDTH).map(move |lane| {
                if block.get(lane) == Gf2::ONE {
                    EF::ONE
                } else {
                    EF::ZERO
                }
            })
        })
        .collect()
}

fn boolean_pcs(log_bits: usize) -> BooleanPcs<EF, MyMmcs, MyMmcs> {
    let config = BinaryPcsConfig::try_new::<EF, EF>(log_bits - ABSORBED, params()).unwrap();
    BooleanPcs::new(config, mmcs(), mmcs(), log_bits).unwrap()
}

fn embedded_pcs(log_bits: usize) -> BinaryPcs<EF, EF, MyMmcs, MyMmcs> {
    let config = BinaryPcsConfig::try_new::<EF, EF>(log_bits, params()).unwrap();
    BinaryPcs::new(config, mmcs(), mmcs()).unwrap()
}

fn embedded_protocol(log_bits: usize) -> OpeningProtocol {
    OpeningProtocol::new(vec![TableSpec::new(
        TableShape::new(log_bits, 1),
        vec![OpeningBatch::new(vec![0], Vec::new())],
    )])
}

/// One row of the table: the two arms of one stage at one size.
fn row(what: &str, log_bits: usize, before: usize, after: usize) {
    let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
    println!(
        "{what:<22} 2^{log_bits:<3} {:>12.2} {:>12.2} {:>8.1}x",
        mib(before),
        mib(after),
        before as f64 / after as f64
    );
}

/// Commit a bit witness and open it at one point, reporting the peak at two stages.
///
/// The region opens before the witness exists, so nothing the run holds is left out.
///
/// ```text
///     after commit   the codeword and the tree, plus the witness they came from
///     after open     the same, plus everything the ring switch and the opening add
/// ```
fn prove_packed(log_bits: usize, point: &Point<EF>) -> (usize, usize) {
    open_region();
    let bits = witness(log_bits);
    let pcs = boolean_pcs(log_bits);
    let (_, data) = pcs.commit_bits(&bits, &mut challenger()).unwrap();
    let after_commit = peak();

    let points = vec![point.clone()];
    let _ = pcs
        .open_at_points(data, &points, &mut challenger())
        .unwrap();
    (after_commit, peak())
}

/// The same two stages with one field element per bit, which is what no packing commits.
fn prove_embedded(log_bits: usize, point: &Point<EF>) -> (usize, usize) {
    open_region();
    let bits = witness(log_bits);
    let cells = embedded(&bits);
    drop(bits);

    let pcs = embedded_pcs(log_bits);
    let protocol = embedded_protocol(log_bits);
    // One column of every cell, which is the shape the stacked layout commits.
    let column_len = cells.len();
    let table = Table::new(RowMajorMatrix::new(cells, column_len));
    let plain_witness = SuffixProver::<EF, EF>::new_witness(vec![table], 0);
    let (_, data) = pcs.commit(plain_witness, &mut challenger()).unwrap();
    let after_commit = peak();

    let _ = pcs
        .open_at(
            data,
            &protocol,
            core::slice::from_ref(point),
            &mut challenger(),
        )
        .unwrap();
    (after_commit, peak())
}

fn main() {
    println!("peak heap while proving, MiB, one worker thread\n");
    println!(
        "{:<22} {:<5} {:>12} {:>12} {:>9}",
        "stage", "size", "embedded", "packed", "gain"
    );

    for &log_bits in &LOG_BITS {
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB002), log_bits);

        // Each arm runs in a call of its own, so the next region opens after this one frees.
        let (commit_packed, prove_packed_peak) = prove_packed(log_bits, &point);
        let (commit_embedded, prove_embedded_peak) = prove_embedded(log_bits, &point);

        row("commit", log_bits, commit_embedded, commit_packed);
        row(
            "commit and open",
            log_bits,
            prove_embedded_peak,
            prove_packed_peak,
        );

        // The witness itself, for context: this is the part the packing shrinks by construction.
        let held_packed = (1usize << log_bits) / 8;
        let held_embedded = (1usize << log_bits) * size_of::<EF>();
        row("witness held", log_bits, held_embedded, held_packed);
        println!();
    }
}
