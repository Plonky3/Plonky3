// Import the actual Plonky3 field implementations
use p3_field::integers::QuotientMap;
use p3_goldilocks::Goldilocks;
use p3_goldilocks_monty::Goldilocks as GoldilocksMonty;
use wasm_bindgen::prelude::*;
use web_sys::console;

// High-precision timing
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ["performance"], js_name = now)]
    fn performance_now() -> f64;
}

// A macro for easier console.log calls from Rust
macro_rules! log {
    ( $( $t:tt )* ) => {
        console::log_1(&format!( $( $t )* ).into());
    }
}

// Prevent compiler optimization
#[inline(never)]
fn black_box<T>(dummy: T) -> T {
    unsafe { core::ptr::read_volatile(&dummy) }
}

#[wasm_bindgen(start)]
pub fn main() {
    // Set panic hook but don't fail on table errors
    std::panic::set_hook(Box::new(|info| {
        if let Some(msg) = info.payload().downcast_ref::<String>() {
            if !msg.contains("WebAssembly.Table.grow") {
                console::error_1(&format!("Panic: {}", msg).into());
            }
        }
    }));

    log!("🚀 Goldilocks WASM benchmark loaded with Plonky3 implementations!");
}

#[wasm_bindgen]
pub fn benchmark_goldilocks_multiplication() -> f64 {
    log!("Starting Goldilocks multiplication benchmark...");

    // Generate test data using proper Plonky3 API
    let a = Goldilocks::from_int(12345678u64);
    let b = Goldilocks::from_int(87654321u64);

    // Warm up
    let mut result = a;
    for _ in 0..100 {
        result = result * b;
    }

    // Benchmark 1,000,000 multiplications with proper timing
    let start = performance_now();
    for _ in 0..1_000_000 {
        result = result * b;
    }
    let end = performance_now();

    // Prevent optimization by using the result
    let _ = black_box(result);

    let duration_ms = end - start;
    log!(
        "Goldilocks multiplication: 1,000,000 ops in {:.3}ms",
        duration_ms
    );
    duration_ms
}

#[wasm_bindgen]
pub fn benchmark_goldilocks_addition() -> f64 {
    log!("Starting Goldilocks addition benchmark...");

    let a = Goldilocks::from_int(12345678u64);
    let b = Goldilocks::from_int(87654321u64);

    // Warm up
    let mut result = a;
    for _ in 0..100 {
        result = result + b;
    }

    // Benchmark 1,000,000 additions
    let start = performance_now();
    for _ in 0..1_000_000 {
        result = result + b;
    }
    let end = performance_now();

    // Prevent optimization by using the result
    let _ = black_box(result);

    let duration_ms = end - start;
    log!("Goldilocks addition: 1,000,000 ops in {:.3}ms", duration_ms);
    duration_ms
}

#[wasm_bindgen]
pub fn benchmark_goldilocks_monty_multiplication() -> f64 {
    log!("Starting Goldilocks Monty multiplication benchmark...");

    // Generate test data using GoldilocksMonty
    let a = GoldilocksMonty::new(12345678u64);
    let b = GoldilocksMonty::new(87654321u64);

    // Warm up
    let mut result = a;
    for _ in 0..100 {
        result = result * b;
    }

    // Benchmark 1,000,000 multiplications
    let start = performance_now();
    for _ in 0..1_000_000 {
        result = result * b;
    }
    let end = performance_now();

    // Prevent optimization by using the result
    let _ = black_box(result);

    let duration_ms = end - start;
    log!(
        "Goldilocks Monty multiplication: 1,000,000 ops in {:.3}ms",
        duration_ms
    );
    duration_ms
}

#[wasm_bindgen]
pub fn benchmark_goldilocks_monty_addition() -> f64 {
    log!("Starting Goldilocks Monty addition benchmark...");

    let a = GoldilocksMonty::new(12345678u64);
    let b = GoldilocksMonty::new(87654321u64);

    // Warm up
    let mut result = a;
    for _ in 0..100 {
        result = result + b;
    }

    // Benchmark 1,000,000 additions
    let start = performance_now();
    for _ in 0..1_000_000 {
        result = result + b;
    }
    let end = performance_now();

    // Prevent optimization by using the result
    let _ = black_box(result);

    let duration_ms = end - start;
    log!(
        "Goldilocks Monty addition: 1,000,000 ops in {:.3}ms",
        duration_ms
    );
    duration_ms
}

#[wasm_bindgen]
pub fn run_all_benchmarks() {
    log!("=== Goldilocks Field Operations Benchmark ===");

    let mult_time = benchmark_goldilocks_multiplication();
    let add_time = benchmark_goldilocks_addition();
    let monty_mult_time = benchmark_goldilocks_monty_multiplication();
    let monty_add_time = benchmark_goldilocks_monty_addition();

    log!("=== Summary (1,000,000 operations each) ===");
    log!("Goldilocks multiplication: {:.3}ms", mult_time);
    log!("Goldilocks addition: {:.3}ms", add_time);
    log!("Goldilocks-Monty multiplication: {:.3}ms", monty_mult_time);
    log!("Goldilocks-Monty addition: {:.3}ms", monty_add_time);

    if monty_mult_time > 0.0 {
        let mult_speedup = mult_time / monty_mult_time;
        log!("Monty multiplication speedup: {:.2}x", mult_speedup);
    }

    if monty_add_time > 0.0 {
        let add_speedup = add_time / monty_add_time;
        log!("Monty addition speedup: {:.2}x", add_speedup);
    }
}
