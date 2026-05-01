use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_zk_codes::{ReedSolomonZkEncoding, ZkEncoding};
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

criterion_group!(benches, bench_rs_encoding);
criterion_main!(benches);
