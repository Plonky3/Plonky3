use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::extension::BinomialExtensionField;
use p3_field::{PrimeCharacteristicRing, TwoAdicField};
use p3_koala_bear::KoalaBear;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_zk_codes::{ReedSolomonZkEncoding, ZkEncoding, ZkEncodingWithRandomness, stack_codewords};
use rand::distr::{Distribution, StandardUniform};
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

fn bench_rs_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("rs_encoding");
    let dft = Radix2Dit::<BabyBear>::default();

    for &log_n in &[10, 12, 14] {
        let n = 1 << log_n;
        let t = 8;
        let msg_len = n - t;
        let m = n; // codeword length
        let encoding = ReedSolomonZkEncoding::<BabyBear, _>::new(t, msg_len, m, dft.clone());

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        let msg: Vec<_> = (0..msg_len).map(|_| rng.random()).collect();

        group.bench_with_input(BenchmarkId::new("zk_rs", log_n), &log_n, |b, _| {
            b.iter(|| encoding.encode(&msg, &mut rng));
        });

        group.bench_with_input(BenchmarkId::new("plain_rs", log_n), &log_n, |b, _| {
            b.iter(|| {
                let mut coeffs = msg.clone();
                coeffs.resize(m, BabyBear::ZERO);
                let mat = RowMajorMatrix::new_col(coeffs);
                dft.dft_batch(mat).to_row_major_matrix()
            });
        });
    }
    group.finish();
}

fn bench_rs_batch_encoding(c: &mut Criterion) {
    bench_rs_batch_encoding_for_field::<BabyBear>(c, "baby_bear");
    bench_rs_batch_encoding_for_field::<BinomialExtensionField<BabyBear, 4>>(c, "baby_bear_ext4");
    bench_whir_octic_mask_encoding(c);
}

/// Exercises the mask-code geometry used by high-folding octic HVZK-WHIR.
///
/// Enable `p3-dft/parallel` to model the threaded prover path, where batching
/// also amortizes Rayon dispatch across columns.
fn bench_whir_octic_mask_encoding(c: &mut Criterion) {
    type EF = BinomialExtensionField<KoalaBear, 8>;

    let mut group = c.benchmark_group("rs_batch_encoding/koala_bear_ext8_whir_masks");
    let dft = Radix2Dit::<EF>::default();
    let log_m = 10;
    let msg_len = 3;
    let t = 92;
    let m = 1 << log_m;
    let encoding = ReedSolomonZkEncoding::<EF, _>::new(t, msg_len, m, dft);

    // Widths 8 and 6 model successive folding rounds. Width 1 guards the
    // code-switching case against regressions from taking the batch API.
    for &width in &[1, 6, 8] {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xc0de + width as u64);
        let messages: Vec<Vec<EF>> = (0..width)
            .map(|_| (0..msg_len).map(|_| rng.random()).collect())
            .collect();
        let randomness: Vec<Vec<EF>> = (0..width)
            .map(|_| (0..t).map(|_| rng.random()).collect())
            .collect();
        let id = format!("log_m={log_m}/width={width}");

        group.bench_with_input(BenchmarkId::new("separate_stack", &id), &id, |b, _| {
            b.iter(|| {
                let codewords: Vec<_> = messages
                    .iter()
                    .zip(&randomness)
                    .map(|(message, randomness)| {
                        encoding.encode_with_randomness(message, randomness)
                    })
                    .collect();
                stack_codewords(&codewords)
            });
        });

        group.bench_with_input(BenchmarkId::new("batch", &id), &id, |b, _| {
            b.iter(|| encoding.encode_batch_with_randomness(&messages, &randomness));
        });
    }
    group.finish();
}

fn bench_rs_batch_encoding_for_field<F>(c: &mut Criterion, field_name: &str)
where
    F: TwoAdicField,
    StandardUniform: Distribution<F>,
{
    let mut group = c.benchmark_group(format!("rs_batch_encoding/{field_name}"));
    let dft = Radix2Dit::<F>::default();

    for &log_m in &[10, 12, 14] {
        let msg_len = 8;
        let t = 8;
        let m = 1 << log_m;
        let encoding = ReedSolomonZkEncoding::<F, _>::new(t, msg_len, m, dft.clone());

        for &width in &[4, 8, 48] {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5eed + width as u64);
            let messages: Vec<Vec<F>> = (0..width)
                .map(|_| (0..msg_len).map(|_| rng.random()).collect())
                .collect();
            let randomness: Vec<Vec<F>> = (0..width)
                .map(|_| (0..t).map(|_| rng.random()).collect())
                .collect();
            let id = format!("log_m={log_m}/width={width}");

            group.bench_with_input(BenchmarkId::new("separate_stack", &id), &id, |b, _| {
                b.iter(|| {
                    let codewords: Vec<_> = messages
                        .iter()
                        .zip(&randomness)
                        .map(|(message, randomness)| {
                            encoding.encode_with_randomness(message, randomness)
                        })
                        .collect();
                    stack_codewords(&codewords)
                });
            });

            group.bench_with_input(BenchmarkId::new("batch", &id), &id, |b, _| {
                b.iter(|| encoding.encode_batch_with_randomness(&messages, &randomness));
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_rs_encoding, bench_rs_batch_encoding);
criterion_main!(benches);
