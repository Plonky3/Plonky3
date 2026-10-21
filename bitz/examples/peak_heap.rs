//! Peak heap of one commit and one proof of a Goldilocks claim on `2^n` committed bits.
//!
//! A tracking allocator counts every live byte, so the figure includes the words, the codeword and the tree.
//!
//! Run it with one worker thread, since a pool makes the peak depend on it.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use p3_bitz::{BitShape, BitzParams, BitzPcs, Challenger, ColumnWeights, RowWeights, TensorClaim};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_keccak::Keccak256Hash;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Committed variables measured.
const SIZES: [usize; 3] = [20, 24, 28];

/// System allocator that tracks the live byte count and its high-water mark.
struct TrackingAlloc;

/// Bytes allocated and not yet freed.
static LIVE: AtomicUsize = AtomicUsize::new(0);

/// Highest the live count has reached.
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's layout contract is forwarded unchanged.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        // SAFETY: `ptr` came from `alloc` with this layout.
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static GLOBAL: TrackingAlloc = TrackingAlloc;

/// A fresh sponge.
const fn sponge() -> Challenger {
    Challenger::from_hasher(Vec::new(), Keccak256Hash)
}

/// Random words and one dense-row, equality-column claim on them.
fn instance(shape: BitShape) -> (Vec<u64>, TensorClaim<Goldilocks>) {
    let mut rng = SmallRng::seed_from_u64(shape.num_variables() as u64);
    let words: Vec<u64> = (0..shape.words()).map(|_| rng.next_u64()).collect();
    let u1: Vec<Goldilocks> = (0..shape.rows())
        .map(|_| Goldilocks::from_u64(rng.next_u64()))
        .collect();
    let point: Vec<Goldilocks> = (0..shape.log_columns())
        .map(|_| Goldilocks::from_u64(rng.next_u64()))
        .collect();
    let mut eq = vec![Goldilocks::ONE];
    for &x in &point {
        eq = eq
            .iter()
            .flat_map(|&e| [e * (Goldilocks::ONE - x), e * x])
            .collect();
    }
    let value = words
        .chunks_exact(shape.column_words())
        .zip(&eq)
        .map(|(column, &e)| {
            let fold: u128 = (0..shape.rows())
                .filter(|b| column[b / 64] >> (b % 64) & 1 == 1)
                .map(|b| u128::from(u1[b].as_canonical_u64()))
                .sum();
            e * Goldilocks::from_u128(fold)
        })
        .sum();
    let claim = TensorClaim {
        rows: RowWeights::Dense(u1),
        columns: ColumnWeights::Equality(point),
        value,
    };
    (words, claim)
}

fn main() {
    println!("{:<6} {:>14}", "bits", "peak MiB");
    for n in SIZES {
        let shape = BitShape::balanced(n).unwrap();
        let pcs = BitzPcs::<Goldilocks>::new(shape, &BitzParams::default()).unwrap();
        let (words, claim) = instance(shape);
        // The region opens with the words and the claim live, as a caller holds them.
        PEAK.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
        let mut challenger = sponge();
        let (_, data) = pcs.commit(words, &mut challenger).unwrap();
        let proof = pcs.prove(data, &[claim], &mut challenger).unwrap();
        drop(proof);
        let peak = PEAK.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0);
        println!("2^{n:<4} {peak:>14.1}");
    }
}
