use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_field::extension::Complex;
use p3_field::PrimeCharacteristicRing;
use p3_mersenne_31::Mersenne31;

/// Original implementation using expensive exponentiation
fn ext_two_adic_generator_original(bits: usize) -> [Mersenne31; 2] {
    assert!(bits <= 32);
    
    if bits == 0 {
        [Mersenne31::ONE, Mersenne31::ZERO]
    } else {
        // Generator of the whole 2^TWO_ADICITY group
        // sage: p = 2^31 - 1
        // sage: F = GF(p)
        // sage: R.<x> = F[]
        // sage: F2.<u> = F.extension(x^2 + 1)
        // sage: g = F2.multiplicative_generator()^((p^2 - 1) / 2^32); g
        // 1117296306*u + 1166849849
        // sage: assert(g.multiplicative_order() == 2^32)
        let base = Complex::<Mersenne31>::new_complex(
            Mersenne31::new_checked(1_166_849_849).unwrap(), 
            Mersenne31::new_checked(1_117_296_306).unwrap()
        );
        base.exp_power_of_2(32 - bits).to_array()
    }
}

/// Optimized implementation using precomputed table
fn ext_two_adic_generator_optimized(bits: usize) -> [Mersenne31; 2] {
    assert!(bits <= 32);
    
    if bits == 0 {
        [Mersenne31::ONE, Mersenne31::ZERO]
    } else {
        Mersenne31::EXT_TWO_ADIC_GENERATORS[bits - 1]
    }
}

fn bench_ext_two_adic_generator(c: &mut Criterion) {
    let mut group = c.benchmark_group("ext_two_adic_generator");
    
    // Test different bit values to show the performance difference
    let test_bits = vec![1, 4, 8, 16, 24, 32];
    
    for bits in test_bits {
        // Benchmark original implementation
        group.bench_with_input(
            BenchmarkId::new("original", bits),
            &bits,
            |b, &bits| {
                b.iter(|| ext_two_adic_generator_original(bits))
            },
        );
        
        // Benchmark optimized implementation
        group.bench_with_input(
            BenchmarkId::new("optimized", bits),
            &bits,
            |b, &bits| {
                b.iter(|| ext_two_adic_generator_optimized(bits))
            },
        );
    }
    
    group.finish();
}

fn bench_ext_two_adic_generator_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("ext_two_adic_generator_throughput");
    
    // Test with a range of bits to show consistent performance
    let test_bits = (1..=32).collect::<Vec<_>>();
    
    // Benchmark original implementation
    group.bench_function("original_all_bits", |b| {
        b.iter(|| {
            for &bits in &test_bits {
                std::hint::black_box(ext_two_adic_generator_original(bits));
            }
        })
    });
    
    // Benchmark optimized implementation
    group.bench_function("optimized_all_bits", |b| {
        b.iter(|| {
            for &bits in &test_bits {
                std::hint::black_box(ext_two_adic_generator_optimized(bits));
            }
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_ext_two_adic_generator,
    bench_ext_two_adic_generator_throughput
);
criterion_main!(benches);
