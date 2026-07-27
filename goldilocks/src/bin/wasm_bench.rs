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

    bench_poseidon2();
    bench_dot_product();
}

/// Compares the vectorized delayed-reduction `dot_product` (one `reduce128` for the whole
/// sum, computed 2 lanes at once) against the previous per-lane fallback (calling the
/// scalar `Goldilocks::dot_product` — itself already delayed-reduction, just not
/// vectorized — once per lane), for a range of `N`.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn bench_dot_product() {
    use core::hint::black_box;
    use std::time::Instant;

    use p3_field::PrimeCharacteristicRing;
    use p3_goldilocks::{Goldilocks, PackedGoldilocksWasmSimd128};

    const M: u64 = 200_000;

    fn report(name: &str, n: u64, elapsed_ns: u128) {
        let per_op = elapsed_ns as f64 / n as f64;
        println!("{name:>28}: {per_op:>8.2} ns/op   ({n} ops in {elapsed_ns} ns)");
    }

    /// Mirrors the removed per-lane fallback: unpack to 2 scalar arrays, call the
    /// (already delayed-reduction) scalar `Goldilocks::dot_product` once per lane, repack.
    fn dot_product_per_lane_fallback<const N: usize>(
        lhs: &[PackedGoldilocksWasmSimd128; N],
        rhs: &[PackedGoldilocksWasmSimd128; N],
    ) -> PackedGoldilocksWasmSimd128 {
        let lhs0: [Goldilocks; N] = core::array::from_fn(|i| lhs[i].0[0]);
        let rhs0: [Goldilocks; N] = core::array::from_fn(|i| rhs[i].0[0]);
        let lhs1: [Goldilocks; N] = core::array::from_fn(|i| lhs[i].0[1]);
        let rhs1: [Goldilocks; N] = core::array::from_fn(|i| rhs[i].0[1]);
        PackedGoldilocksWasmSimd128([
            Goldilocks::dot_product(&lhs0, &rhs0),
            Goldilocks::dot_product(&lhs1, &rhs1),
        ])
    }

    macro_rules! bench_n {
        ($n:literal) => {{
            let lhs: [PackedGoldilocksWasmSimd128; $n] = core::array::from_fn(|i| {
                PackedGoldilocksWasmSimd128([
                    Goldilocks::new((i as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ 0x1234),
                    Goldilocks::new((i as u64).wrapping_mul(0xBF58476D1CE4E5B9) ^ 0x5678),
                ])
            });
            let rhs: [PackedGoldilocksWasmSimd128; $n] = core::array::from_fn(|i| {
                PackedGoldilocksWasmSimd128([
                    Goldilocks::new((i as u64).wrapping_mul(0x94D049BB133111EB) ^ 0x9abc),
                    Goldilocks::new((i as u64).wrapping_mul(0x2545F4914F6CDD1D) ^ 0xdef0),
                ])
            });

            {
                let mut acc = PackedGoldilocksWasmSimd128::ZERO;
                let t0 = Instant::now();
                for _ in 0..M {
                    acc +=
                        PackedGoldilocksWasmSimd128::dot_product(black_box(&lhs), black_box(&rhs));
                }
                let dt = t0.elapsed().as_nanos();
                let _ = black_box(acc);
                report(concat!("dot_product_", $n, "_vectorized"), M, dt);
            }

            {
                let mut acc = PackedGoldilocksWasmSimd128::ZERO;
                let t0 = Instant::now();
                for _ in 0..M {
                    acc += dot_product_per_lane_fallback(black_box(&lhs), black_box(&rhs));
                }
                let dt = t0.elapsed().as_nanos();
                let _ = black_box(acc);
                report(concat!("dot_product_", $n, "_per_lane"), M, dt);
            }
        }};
    }

    bench_n!(2);
    bench_n!(3);
    bench_n!(4);
    bench_n!(5);
    bench_n!(8);
    bench_n!(16);
    bench_n!(32);
}

/// Compares the specialized `Poseidon2ExternalLayerGoldilocksWasmSimd128`/
/// `Poseidon2InternalLayerGoldilocksWasmSimd128` layers (pre-broadcast round constants,
/// batched S-box, split lane-sum) against the generic `Algebra<Goldilocks>` fallback path
/// (crate-root `Poseidon2ExternalLayerGoldilocks`/`Poseidon2InternalLayerGoldilocks`, which
/// re-broadcasts every round constant from scalar on every call), for the same round
/// constants, applied to a packed wasm32 state.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn bench_poseidon2() {
    use core::hint::black_box;
    use std::time::Instant;

    use p3_goldilocks::{
        Goldilocks, PackedGoldilocksWasmSimd128, Poseidon2ExternalLayerGoldilocks,
        Poseidon2InternalLayerGoldilocks, default_goldilocks_poseidon2_8,
        default_goldilocks_poseidon2_12, default_goldilocks_poseidon2_16,
    };
    use p3_poseidon2::Poseidon2;
    use p3_symmetric::Permutation;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    const M: u64 = 100_000;

    fn report(name: &str, n: u64, elapsed_ns: u128) {
        let per_op = elapsed_ns as f64 / n as f64;
        println!("{name:>24}: {per_op:>8.2} ns/op   ({n} ops in {elapsed_ns} ns)");
    }

    fn make_state<const WIDTH: usize>(rng: &mut SmallRng) -> [PackedGoldilocksWasmSimd128; WIDTH] {
        core::array::from_fn(|_| PackedGoldilocksWasmSimd128([rng.random(), rng.random()]))
    }

    macro_rules! bench_width {
        ($width:literal, $specialized:expr, $generic:expr) => {{
            let mut rng = SmallRng::seed_from_u64(3);
            let specialized = $specialized;
            let generic = $generic;

            {
                let mut state = make_state::<$width>(&mut rng);
                let t0 = Instant::now();
                for _ in 0..M {
                    specialized.permute_mut(black_box(&mut state));
                }
                let dt = t0.elapsed().as_nanos();
                let _ = black_box(&state);
                report(concat!("poseidon2_", $width, "_specialized"), M, dt);
            }

            {
                let mut state = make_state::<$width>(&mut rng);
                let t0 = Instant::now();
                for _ in 0..M {
                    generic.permute_mut(black_box(&mut state));
                }
                let dt = t0.elapsed().as_nanos();
                let _ = black_box(&state);
                report(concat!("poseidon2_", $width, "_generic"), M, dt);
            }
        }};
    }

    bench_width!(
        8,
        default_goldilocks_poseidon2_8(),
        Poseidon2::<
            Goldilocks,
            Poseidon2ExternalLayerGoldilocks<8>,
            Poseidon2InternalLayerGoldilocks,
            8,
            7,
        >::new_from_rng(8, 22, &mut SmallRng::seed_from_u64(99))
    );
    bench_width!(
        12,
        default_goldilocks_poseidon2_12(),
        Poseidon2::<
            Goldilocks,
            Poseidon2ExternalLayerGoldilocks<12>,
            Poseidon2InternalLayerGoldilocks,
            12,
            7,
        >::new_from_rng(8, 22, &mut SmallRng::seed_from_u64(99))
    );
    bench_width!(
        16,
        default_goldilocks_poseidon2_16(),
        Poseidon2::<
            Goldilocks,
            Poseidon2ExternalLayerGoldilocks<16>,
            Poseidon2InternalLayerGoldilocks,
            16,
            7,
        >::new_from_rng(8, 22, &mut SmallRng::seed_from_u64(99))
    );
}
