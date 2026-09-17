//! Peak heap during one commit and one opening of a bit witness.
//!
//! Commitment volume is the reason to pack a narrow alphabet.
//!
//! So the space one prove takes is the figure that says whether it worked.
//!
//! A tracking allocator counts live bytes while armed and keeps their high-water mark.
//!
//! The lookup benchmark in the batch prover accounts for its memory the same way.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicUsize, Ordering};

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

/// System allocator that tracks live and high-water bytes while armed.
struct TrackingAlloc;

/// Live bytes since the last arm, signed so a pre-arm free cannot underflow.
static LIVE: AtomicIsize = AtomicIsize::new(0);
/// High-water mark of the live count while armed.
static PEAK: AtomicUsize = AtomicUsize::new(0);
/// Whether allocations are currently being counted.
static ARMED: AtomicBool = AtomicBool::new(false);

unsafe impl GlobalAlloc for TrackingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
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
    BinaryPcs::new(config, mmcs(), mmcs())
}

fn embedded_protocol(log_bits: usize) -> OpeningProtocol {
    OpeningProtocol::new(vec![TableSpec::new(
        TableShape::new(log_bits, 1),
        vec![OpeningBatch::new(vec![0], Vec::new())],
    )])
}

/// One row of the table: the two arms of one operation at one size.
fn row(what: &str, log_bits: usize, before: usize, after: usize) {
    let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
    println!(
        "{what:<22} 2^{log_bits:<3} {:>12.2} {:>12.2} {:>8.1}x",
        mib(before),
        mib(after),
        before as f64 / after as f64
    );
}

fn main() {
    println!("peak heap during one operation, MiB, single-threaded\n");
    println!(
        "{:<22} {:<5} {:>12} {:>12} {:>9}",
        "operation", "size", "embedded", "packed", "gain"
    );

    for &log_bits in &LOG_BITS {
        let bits = witness(log_bits);
        let cells = embedded(&bits);
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB002), log_bits);

        // Commit, packed: the witness stays one bit per cell.
        let packed = boolean_pcs(log_bits);
        arm();
        let (_, data) = packed.commit_bits(&bits, &mut challenger()).unwrap();
        let commit_packed = disarm();

        // Commit, embedded: the same bits as one field element each.
        let plain = embedded_pcs(log_bits);
        let protocol = embedded_protocol(log_bits);
        arm();
        let table = Table::new(RowMajorMatrix::new(cells.clone(), cells.len()));
        let plain_witness = SuffixProver::<EF, EF>::new_witness(vec![table], 0);
        let (_, plain_data) = plain.commit(plain_witness, &mut challenger()).unwrap();
        let commit_embedded = disarm();

        row("commit", log_bits, commit_embedded, commit_packed);

        // Open, packed: a ring switch, then one opening of the packed commitment.
        let points = vec![point.clone()];
        arm();
        let _ = packed
            .open_at_points(data, &points, &mut challenger())
            .unwrap();
        let open_packed = disarm();

        // Open, embedded: one opening of a commitment a hundred and twenty-eight times longer.
        arm();
        let _ = plain
            .open_at(
                plain_data,
                &protocol,
                core::slice::from_ref(&point),
                &mut challenger(),
            )
            .unwrap();
        let open_embedded = disarm();

        row("open", log_bits, open_embedded, open_packed);

        // The witness itself, for context: this is the part the packing shrinks by construction.
        let held_packed = core::mem::size_of_val(bits.as_slice());
        let held_embedded = core::mem::size_of_val(cells.as_slice());
        row("witness held", log_bits, held_embedded, held_packed);
        println!();
    }
}
