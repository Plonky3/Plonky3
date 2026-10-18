//! Packed binary Keccak witness generation, including padding-only block boundaries.
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_binary_field::Gf2;
use p3_keccak_air::generate_binary_trace_packed;

fn bench_packed_trace(c: &mut Criterion) {
    let mut group = c.benchmark_group("keccak_binary_packed_trace");
    for count in [5, 6, 7, 41, 62, 64, 65, 1024] {
        let inputs: Vec<[u64; 25]> = (0..count)
            .map(|i| core::array::from_fn(|j| (i * 25 + j) as u64))
            .collect();
        group.bench_with_input(BenchmarkId::from_parameter(count), &inputs, |b, inputs| {
            b.iter_batched(
                || inputs.clone(),
                |inputs| generate_binary_trace_packed::<Gf2>(std::hint::black_box(inputs)),
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}
criterion_group!(benches, bench_packed_trace);
criterion_main!(benches);
