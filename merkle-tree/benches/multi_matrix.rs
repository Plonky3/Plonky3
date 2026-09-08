//! Compare equal-width commitments split across matrices, including injected layers.
//! `current` uses the hasher's staging hint; `unstaged` retains the flat iterator.
//! Input matrices are generated once and borrowed during each timed commitment.
//!
//! Run with a fixed worker count, for example:
//! `RAYON_NUM_THREADS=8 cargo bench -p p3-merkle-tree --features parallel --bench multi_matrix`

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear as F, Poseidon2BabyBear};
use p3_commit::Mmcs;
use p3_field::{Field, PackedValue};
use p3_goldilocks::Goldilocks;
use p3_keccak::{KeccakF, VECTOR_LEN};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{
    CompressionFunctionFromHasher, CryptographicHasher, PaddingFreeSponge,
    PseudoCompressionFunction, SerializingHasher, TruncatedPermutation,
};
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use serde::Serialize;
use serde::de::DeserializeOwned;

#[derive(Clone)]
struct Unstaged<H>(H);

impl<T: Clone, Out, H: CryptographicHasher<T, Out>> CryptographicHasher<T, Out> for Unstaged<H> {
    fn hash_iter<I: IntoIterator<Item = T>>(&self, input: I) -> Out {
        self.0.hash_iter(input)
    }
}

fn split<F: Field>(matrix: &RowMajorMatrix<F>, widths: &[usize]) -> Vec<RowMajorMatrix<F>> {
    let mut start = 0;
    widths
        .iter()
        .map(|&width| {
            let mut values = Vec::with_capacity(matrix.height() * width);
            for row in matrix.values.chunks_exact(matrix.width()) {
                values.extend_from_slice(&row[start..start + width]);
            }
            start += width;
            RowMajorMatrix::new(values, width)
        })
        .collect()
}

fn bench_hash<F, P, PW, H, C, const D: usize>(criterion: &mut Criterion, name: &str, h: H, c: C)
where
    F: Field + Serialize + DeserializeOwned,
    StandardUniform: Distribution<F>,
    P: PackedValue<Value = F>,
    PW: PackedValue,
    PW::Value: Eq,
    [PW::Value; D]: Serialize + DeserializeOwned,
    H: CryptographicHasher<F, [PW::Value; D]> + CryptographicHasher<P, [PW; D]> + Sync,
    C: PseudoCompressionFunction<[PW::Value; D], 2> + PseudoCompressionFunction<[PW; D], 2> + Sync,
{
    let mmcs = MerkleTreeMmcs::<P, PW, _, _, 2, D>::new(h.clone(), c.clone(), 0);
    let reference = MerkleTreeMmcs::<P, PW, _, _, 2, D>::new(Unstaged(h), c, 0);
    let mut group = criterion.benchmark_group(format!("multi_matrix/{name}"));
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(1));

    for (rows, cols) in [(128, 2633), (4096, 32), (32768, 256)] {
        let mut rng = SmallRng::seed_from_u64(1);
        let matrix = RowMajorMatrix::<F>::rand(&mut rng, rows, cols);
        let mut layouts: Vec<Vec<usize>> = [1, 2, 4, 8]
            .into_iter()
            .map(|n| {
                (0..n)
                    .map(|i| cols / n + usize::from(i < cols % n))
                    .collect()
            })
            .collect();
        layouts.push(vec![1, cols / 2, cols - cols / 2 - 1]);
        for widths in layouts {
            let mut matrices = split(&matrix, &widths);
            let layout = widths
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join("+");
            for layer in ["leaf", "injection"] {
                if layer == "injection" {
                    matrices.insert(0, RowMajorMatrix::rand(&mut rng, rows * 2, 1));
                }
                let params = format!("{layer}/{rows}x{layout}");
                // Borrow the input data so cloning a potentially large trace is not timed.
                group.bench_function(BenchmarkId::new("current", &params), |b| {
                    b.iter(|| {
                        black_box(mmcs.commit(matrices.iter().map(|m| m.as_view()).collect()))
                    });
                });
                group.bench_function(BenchmarkId::new("unstaged", &params), |b| {
                    b.iter(|| {
                        black_box(reference.commit(matrices.iter().map(|m| m.as_view()).collect()))
                    });
                });
            }
        }
    }
    group.finish();
}

fn benches(criterion: &mut Criterion) {
    let sponge = PaddingFreeSponge::<KeccakF, 25, 17, 4>::new(KeccakF);
    bench_hash::<F, [F; VECTOR_LEN], [u64; VECTOR_LEN], _, _, 4>(
        criterion,
        "keccak_f",
        SerializingHasher::new(sponge),
        CompressionFunctionFromHasher::new(sponge),
    );

    bench_hash::<Goldilocks, [Goldilocks; VECTOR_LEN], [u64; VECTOR_LEN], _, _, 4>(
        criterion,
        "keccak_f_goldilocks",
        SerializingHasher::new(sponge),
        CompressionFunctionFromHasher::new(sponge),
    );

    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Poseidon2BabyBear::<16>::new_from_rng_128(&mut rng);
    bench_hash::<F, <F as Field>::Packing, <F as Field>::Packing, _, _, 8>(
        criterion,
        "poseidon2",
        PaddingFreeSponge::<_, 16, 8, 8>::new(perm.clone()),
        TruncatedPermutation::<_, 2, 8, 16>::new(perm),
    );
}

criterion_group!(multi_matrix, benches);
criterion_main!(multi_matrix);
