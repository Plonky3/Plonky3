//! Coarse-grained micro-benchmark for the `wasm32+simd128`
//! `PackedGoldilocksWasmSimd128` backend.
//!
//! Criterion does not build on wasm targets (its `rayon` dep won't compile on
//! wasi, and its harness assumes a host process model). This binary stands in
//! as a small, hand-rolled timer: each op is exercised in a tight dependent
//! chain so the optimizer cannot fold or vectorize the loop away, and we print
//! `Instant::now()` deltas as ns/op.
//!
//! Numbers under `wasmtime` are not representative of production wasm runtimes
//! (V8 / SpiderMonkey), do not treat them as absolute performance.
//!
//! Run with:
//! ```text
//! RUSTFLAGS="-C target-feature=+simd128" \
//!   cargo run --release --target wasm32-wasip1 \
//!     --bin wasm_bench -p p3-goldilocks
//! ```

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
fn main() {
    eprintln!(
        "wasm_bench is a no-op on this target. Re-run with \
         `cargo run --release --target wasm32-wasip1 --bin wasm_bench -p p3-goldilocks` \
         and RUSTFLAGS=\"-C target-feature=+simd128\"."
    );
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn main() {
    use core::hint::black_box;
    use std::time::Instant;

    use p3_field::PrimeCharacteristicRing;
    use p3_goldilocks::{Goldilocks, PackedGoldilocksWasmSimd128};

    // Number of dependent ops per timed run. Tuned so a single run takes ~ms
    // under wasmtime — large enough to drown out Instant resolution noise,
    // small enough that CI doesn't spend minutes here.
    const N: u64 = 1_000_000;

    fn report(name: &str, n: u64, elapsed_ns: u128) {
        let per_op = elapsed_ns as f64 / n as f64;
        println!("{name:>14}: {per_op:>8.2} ns/op   ({n} ops in {elapsed_ns} ns)");
    }

    let a = PackedGoldilocksWasmSimd128([
        Goldilocks::new(0x1234_5678_9abc_def0),
        Goldilocks::new(0xfedc_ba98_7654_3210),
    ]);
    let b = PackedGoldilocksWasmSimd128([
        Goldilocks::new(0x0fed_cba9_8765_4321),
        Goldilocks::new(0xabcd_ef01_2345_6789),
    ]);

    // Each loop chains the result back into the input so the optimizer cannot
    // hoist the op outside the loop. `black_box` further suppresses constant
    // folding across the iteration boundary.

    {
        let mut acc = a;
        let t0 = Instant::now();
        for _ in 0..N {
            acc = black_box(acc) + black_box(b);
        }
        let dt = t0.elapsed().as_nanos();
        let _ = black_box(acc);
        report("add", N, dt);
    }

    {
        let mut acc = a;
        let t0 = Instant::now();
        for _ in 0..N {
            acc = black_box(acc) - black_box(b);
        }
        let dt = t0.elapsed().as_nanos();
        let _ = black_box(acc);
        report("sub", N, dt);
    }

    {
        let mut acc = a;
        let t0 = Instant::now();
        for _ in 0..N {
            acc = -black_box(acc);
        }
        let dt = t0.elapsed().as_nanos();
        let _ = black_box(acc);
        report("neg", N, dt);
    }

    {
        let mut acc = a;
        let t0 = Instant::now();
        for _ in 0..N {
            acc = black_box(acc) * black_box(b);
        }
        let dt = t0.elapsed().as_nanos();
        let _ = black_box(acc);
        report("mul", N, dt);
    }

    {
        let mut acc = a;
        let t0 = Instant::now();
        for _ in 0..N {
            acc = black_box(acc).double();
        }
        let dt = t0.elapsed().as_nanos();
        let _ = black_box(acc);
        report("double", N, dt);
    }

    {
        let mut acc = a;
        let t0 = Instant::now();
        for _ in 0..N {
            acc = black_box(acc).square();
        }
        let dt = t0.elapsed().as_nanos();
        let _ = black_box(acc);
        report("square", N, dt);
    }

    {
        let mut acc = a;
        let t0 = Instant::now();
        for _ in 0..N {
            acc = black_box(acc).halve();
        }
        let dt = t0.elapsed().as_nanos();
        let _ = black_box(acc);
        report("halve", N, dt);
    }
}
