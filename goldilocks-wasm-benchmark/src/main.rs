use std::time::Instant;

// Import the actual Plonky3 field implementations
use p3_field::integers::QuotientMap;
use p3_goldilocks::Goldilocks;
use p3_goldilocks_monty::Goldilocks as GoldilocksMonty;

// Prevent compiler optimization
#[inline(never)]
fn black_box<T>(dummy: T) -> T {
    std::hint::black_box(dummy)
}

fn benchmark_goldilocks_multiplication() -> f64 {
    println!("Starting Goldilocks multiplication benchmark...");

    // Generate test data using proper Plonky3 API
    let a = Goldilocks::from_int(12345678u64);
    let b = Goldilocks::from_int(87654321u64);

    // Warm up
    let mut result = a;
    for _ in 0..100 {
        result = result * b;
    }

    // Benchmark 1,000,000 multiplications
    let start = Instant::now();
    for _ in 0..1_000_000 {
        result = result * b;
    }
    let duration = start.elapsed();

    // Prevent optimization by using the result
    let _ = black_box(result);

    let duration_ms = duration.as_secs_f64() * 1000.0;
    println!(
        "Goldilocks multiplication: 1,000,000 ops in {:.3}ms",
        duration_ms
    );
    duration_ms
}

fn benchmark_goldilocks_addition() -> f64 {
    println!("Starting Goldilocks addition benchmark...");

    let a = Goldilocks::from_int(12345678u64);
    let b = Goldilocks::from_int(87654321u64);

    // Warm up
    let mut result = a;
    for _ in 0..100 {
        result = result + b;
    }

    // Benchmark 1,000,000 additions
    let start = Instant::now();
    for _ in 0..1_000_000 {
        result = result + b;
    }
    let duration = start.elapsed();

    // Prevent optimization by using the result
    let _ = black_box(result);

    let duration_ms = duration.as_secs_f64() * 1000.0;
    println!("Goldilocks addition: 1,000,000 ops in {:.3}ms", duration_ms);
    duration_ms
}

fn benchmark_goldilocks_monty_multiplication() -> f64 {
    println!("Starting Goldilocks Monty multiplication benchmark...");

    // Generate test data using GoldilocksMonty
    let a = GoldilocksMonty::new(12345678u64);
    let b = GoldilocksMonty::new(87654321u64);

    // Warm up
    let mut result = a;
    for _ in 0..100 {
        result = result * b;
    }

    // Benchmark 1,000,000 multiplications
    let start = Instant::now();
    for _ in 0..1_000_000 {
        result = result * b;
    }
    let duration = start.elapsed();

    // Prevent optimization by using the result
    let _ = black_box(result);

    let duration_ms = duration.as_secs_f64() * 1000.0;
    println!(
        "Goldilocks Monty multiplication: 1,000,000 ops in {:.3}ms",
        duration_ms
    );
    duration_ms
}

fn benchmark_goldilocks_monty_addition() -> f64 {
    println!("Starting Goldilocks Monty addition benchmark...");

    let a = GoldilocksMonty::new(12345678u64);
    let b = GoldilocksMonty::new(87654321u64);

    // Warm up
    let mut result = a;
    for _ in 0..100 {
        result = result + b;
    }

    // Benchmark 1,000,000 additions
    let start = Instant::now();
    for _ in 0..1_000_000 {
        result = result + b;
    }
    let duration = start.elapsed();

    // Prevent optimization by using the result
    let _ = black_box(result);

    let duration_ms = duration.as_secs_f64() * 1000.0;
    println!(
        "Goldilocks Monty addition: 1,000,000 ops in {:.3}ms",
        duration_ms
    );
    duration_ms
}

fn main() {
    println!("=== Native Goldilocks Field Operations Benchmark ===");
    println!("Running single-threaded benchmarks for comparison with WASM...");
    println!();

    let mult_time = benchmark_goldilocks_multiplication();
    let add_time = benchmark_goldilocks_addition();
    let monty_mult_time = benchmark_goldilocks_monty_multiplication();
    let monty_add_time = benchmark_goldilocks_monty_addition();

    println!();
    println!("=== Summary (1,000,000 operations each) ===");
    println!("Goldilocks multiplication: {:.3}ms", mult_time);
    println!("Goldilocks addition: {:.3}ms", add_time);
    println!("Goldilocks-Monty multiplication: {:.3}ms", monty_mult_time);
    println!("Goldilocks-Monty addition: {:.3}ms", monty_add_time);

    println!();
    println!("=== Performance Analysis ===");

    if monty_mult_time > 0.0 {
        let mult_speedup = mult_time / monty_mult_time;
        if mult_speedup > 1.0 {
            println!(
                "✅ Monty multiplication is {:.2}x faster than standard",
                mult_speedup
            );
        } else {
            println!(
                "⚠️  Standard multiplication is {:.2}x faster than Monty",
                1.0 / mult_speedup
            );
        }
    }

    if monty_add_time > 0.0 {
        let add_speedup = add_time / monty_add_time;
        if add_speedup > 1.0 {
            println!(
                "✅ Monty addition is {:.2}x faster than standard",
                add_speedup
            );
        } else {
            println!(
                "⚠️  Standard addition is {:.2}x faster than Monty",
                1.0 / add_speedup
            );
        }
    }

    let total_ops = 4_000_000u64;
    let total_time_ms = mult_time + add_time + monty_mult_time + monty_add_time;
    let ops_per_second = (total_ops as f64) / (total_time_ms / 1000.0);

    println!();
    println!("=== Overall Performance ===");
    println!("Total operations: {}", total_ops);
    println!("Total time: {:.3}ms", total_time_ms);
    println!("Operations per second: {:.0}", ops_per_second);
}
