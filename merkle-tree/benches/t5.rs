//! Binary Poseidon2 trees against T5 trees of 2021/373, on the same leaves and sponge.
//!
//! The bench times commit, single and pruned multi openings, and their verification.
//!
//! Before timing, it prints each scheme's proof size and collision security.
//!
//! The aggressive T5 opening has no MMCS, so its path is replayed from `T5::bypass` over the same depth.

use core::array;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_commit::{BatchOpeningRef, Mmcs};
use p3_field::{Field, PackedValue, PrimeField32};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_security::{Evidence, MerkleScheme, TreeBuilder};
use p3_symmetric::{
    CryptographicHasher, FieldAdd, PaddingFreeSponge, PseudoCompressionFunction, T5,
    T5_AGGRESSIVE_OPENING_LEN, TruncatedPermutation,
};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type F = BabyBear;
type Packed = <F as Field>::Packing;
type Perm = Poseidon2BabyBear<16>;
type Sponge = PaddingFreeSponge<Perm, 16, 8, 8>;
type Pair = TruncatedPermutation<Perm, 2, 8, 16>;
type Node5 = T5<Pair, FieldAdd>;
type Digest = [F; 8];

type Binary = MerkleTreeMmcs<Packed, Packed, Sponge, Pair, 2, 8>;
type Quinary = MerkleTreeMmcs<Packed, Packed, Sponge, Node5, 5, 8>;

/// Rows of the committed matrix: a FRI or WHIR round at `2^20` leaves.
const LOG_HEIGHT: usize = 20;

/// Queries per multi-opening, the order of a 100-bit STARK.
const QUERIES: usize = 100;

/// Row widths: one sponge block per leaf, then eight.
const WIDTHS: [usize; 2] = [8, 64];

/// The sponge, the binary compression, and a T5 over three independent permutations.
fn primitives() -> (Sponge, Pair, Node5) {
    let mut rng = SmallRng::seed_from_u64(1);
    let sponge = Sponge::new(Perm::new_from_rng_128(&mut rng));
    let pair = Pair::new(Perm::new_from_rng_128(&mut rng));
    let [h1, h2, h3] = array::from_fn(|_| Pair::new(Perm::new_from_rng_128(&mut rng)));
    (sponge, pair, T5::new(h1, h2, h3))
}

fn queries(height: usize) -> Vec<usize> {
    let mut rng = SmallRng::seed_from_u64(3);
    (0..QUERIES).map(|_| rng.random_range(0..height)).collect()
}

/// Serialized size, which is what a proof carries.
fn bytes<T: serde::Serialize>(value: &T) -> usize {
    postcard::to_allocvec(value).expect("serializable").len()
}

/// Print proof sizes and security levels for every scheme, once.
fn report(width: usize) {
    let (sponge, pair, node) = primitives();
    let height = 1 << LOG_HEIGHT;
    let mat = RowMajorMatrix::<F>::rand(&mut SmallRng::seed_from_u64(2), height, width);
    let qs = queries(height);

    let binary = Binary::new(sponge.clone(), pair, 0);
    let quinary = Quinary::new(sponge, node, 0);
    let (_, bdata) = binary.commit(vec![mat.clone()]);
    let (_, qdata) = quinary.commit(vec![mat]);
    let bpath = binary.open_batch(0, &bdata).opening_proof;
    let qpath = quinary.open_batch(0, &qdata).opening_proof;
    let (_, bmulti) = binary.open_multi_batch(&qs, &bdata);
    let (_, qmulti) = quinary.open_multi_batch(&qs, &qdata);

    // An aggressive path holds three digests per T5 level, over the same levels as the conservative one.
    let t5_levels = qpath.len() / 4;
    let digest_bytes = bytes(&[F::default(); 8]);
    let aggressive_path = t5_levels * T5_AGGRESSIVE_OPENING_LEN * digest_bytes;

    let n = 8.0 * f64::from(F::ORDER_U32).log2();
    let sec = |scheme: MerkleScheme, builder| {
        let proven = scheme.collision_bits(n, builder, Evidence::Proven).bits();
        let conjectured = scheme
            .collision_bits(n, builder, Evidence::Conjectured)
            .bits();
        format!("{proven:>6.1} / {conjectured:>6.1}")
    };
    let row = |name: &str, path: String, multi: String, calls: usize, scheme| {
        eprintln!(
            "  {name:<16} {path:>10} {multi:>12} {calls:>13}   {:>16}   {:>16}",
            sec(scheme, TreeBuilder::Adversarial),
            sec(scheme, TreeBuilder::Honest),
        );
    };

    eprintln!(
        "\nwidth {width}, 2^{LOG_HEIGHT} leaves, {QUERIES} queries, digest = 8 BabyBear ({n:.1} bits)"
    );
    eprintln!(
        "  {:<16} {:>10} {:>12} {:>13}   {:>16}   {:>16}",
        "scheme", "path B", "multi B", "verify calls", "prover-built p/c", "honest root p/c"
    );
    row(
        "binary",
        bytes(&bpath).to_string(),
        bytes(&bmulti).to_string(),
        bpath.len(),
        MerkleScheme::Plain,
    );
    row(
        "T5 conservative",
        bytes(&qpath).to_string(),
        bytes(&qmulti).to_string(),
        3 * t5_levels,
        MerkleScheme::T5Conservative,
    );
    row(
        "T5 aggressive",
        format!("{aggressive_path}*"),
        "-".into(),
        2 * t5_levels,
        MerkleScheme::T5Aggressive,
    );
    eprintln!("  * analytic: three digests per level over the same {t5_levels} T5 levels\n");
}

fn bench_t5(c: &mut Criterion) {
    for width in WIDTHS {
        report(width);
    }

    let (sponge, pair, node) = primitives();
    let binary = Binary::new(sponge.clone(), pair, 0);
    let quinary = Quinary::new(sponge.clone(), node.clone(), 0);
    let height = 1 << LOG_HEIGHT;
    let qs = queries(height);

    for width in WIDTHS {
        let mat = RowMajorMatrix::<F>::rand(&mut SmallRng::seed_from_u64(2), height, width);
        let dims = [mat.dimensions()];

        {
            let mut group = c.benchmark_group(format!("commit/w{width}"));
            group.sample_size(10);
            group.bench_function("binary", |b| b.iter(|| binary.commit(vec![mat.clone()])));
            group.bench_function("t5", |b| b.iter(|| quinary.commit(vec![mat.clone()])));
            group.finish();
        }

        let (bcap, bdata) = binary.commit(vec![mat.clone()]);
        let (qcap, qdata) = quinary.commit(vec![mat]);
        let (bvals, bproof) = binary.open_multi_batch(&qs, &bdata);
        let (qvals, qproof) = quinary.open_multi_batch(&qs, &qdata);
        let bone = binary.open_batch(qs[0], &bdata);
        let qone = quinary.open_batch(qs[0], &qdata);

        {
            let mut group = c.benchmark_group(format!("verify/w{width}"));
            group.bench_function(BenchmarkId::new("single", "binary"), |b| {
                b.iter(|| {
                    binary
                        .verify_batch(&bcap, &dims, qs[0], BatchOpeningRef::from(&bone))
                        .unwrap();
                });
            });
            group.bench_function(BenchmarkId::new("single", "t5"), |b| {
                b.iter(|| {
                    quinary
                        .verify_batch(&qcap, &dims, qs[0], BatchOpeningRef::from(&qone))
                        .unwrap();
                });
            });

            // The aggressive path: one leaf hash, then two calls per T5 level.
            let levels = qone.opening_proof.len() / 4;
            let opening: [Digest; T5_AGGRESSIVE_OPENING_LEN] =
                array::from_fn(|k| qone.opening_proof[k]);
            let row = &qone.opened_values[0];
            group.bench_function(BenchmarkId::new("single", "t5-aggressive"), |b| {
                b.iter(|| {
                    let mut digest: Digest = sponge.hash_slice(row);
                    for level in 0..levels {
                        digest = node.bypass(digest, level % 5, opening);
                    }
                    digest
                });
            });

            group.bench_function(BenchmarkId::new("multi", "binary"), |b| {
                b.iter(|| {
                    binary
                        .verify_multi_batch(&bcap, &dims, &qs, &bvals, &bproof)
                        .unwrap();
                });
            });
            group.bench_function(BenchmarkId::new("multi", "t5"), |b| {
                b.iter(|| {
                    quinary
                        .verify_multi_batch(&qcap, &dims, &qs, &qvals, &qproof)
                        .unwrap();
                });
            });
            group.finish();
        }

        {
            let mut group = c.benchmark_group(format!("open/w{width}"));
            group.bench_function(BenchmarkId::new("multi", "binary"), |b| {
                b.iter(|| binary.open_multi_batch(&qs, &bdata));
            });
            group.bench_function(BenchmarkId::new("multi", "t5"), |b| {
                b.iter(|| quinary.open_multi_batch(&qs, &qdata));
            });
            group.finish();
        }
    }

    // The node on its own: three packed calls against the two-level binary subtree it replaces.
    let mut rng = SmallRng::seed_from_u64(4);
    let kids: [[Packed; 8]; 5] =
        array::from_fn(|_| array::from_fn(|_| Packed::from_fn(|_| rng.random())));
    {
        let mut group = c.benchmark_group("node");
        group.bench_function("binary-subtree-of-4", |b| {
            let (_, pair, _) = primitives();
            b.iter(|| {
                let l = pair.compress([kids[0], kids[1]]);
                let r = pair.compress([kids[2], kids[3]]);
                pair.compress([l, r])
            });
        });
        group.bench_function("t5-of-5", |b| b.iter(|| node.compress(kids)));
        group.finish();
    }
}

criterion_group!(benches, bench_t5);
criterion_main!(benches);
