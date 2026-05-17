//! Perf comparison: single `Poseidon2::permute_mut` vs the cross-permute
//! batched `permute_batch_b2_p2_w{8,12}` on the same packed type.
//!
//! On aarch64 the production `Poseidon2Goldilocks<8>` resolves to the
//! fused NEON kernel; an extra fused-baseline row is printed for context.
//!
//! Build (AVX-2):
//!   RUSTFLAGS="-C target-feature=+avx2" cargo build --release \
//!     --example poseidon2_batched_perf -p p3-goldilocks
//! Build (NEON):
//!   cargo build --release --example poseidon2_batched_perf -p p3-goldilocks
//!
//! Run:
//!   target/release/examples/poseidon2_batched_perf [N] [--seed K]
//! Defaults: N=1_000_000, seed=1.

#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    target_arch = "aarch64",
)))]
fn main() {
    eprintln!("poseidon2_batched_perf: requires x86_64+avx2 or aarch64.");
    std::process::exit(1);
}

#[cfg(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    target_arch = "aarch64",
))]
fn main() {
    harness::run();
}

#[cfg(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    target_arch = "aarch64",
))]
mod harness {
    use std::env;
    use std::hint::black_box;
    use std::time::Instant;

    use p3_goldilocks::poseidon2_batched::{
        default_goldilocks_poseidon2_batched_8, default_goldilocks_poseidon2_batched_12,
        permute_batch_b2_p2_w8, permute_batch_b2_p2_w12,
    };
    use p3_goldilocks::{
        GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_FINAL, GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_INITIAL,
        GOLDILOCKS_POSEIDON2_RC_8_INTERNAL, GOLDILOCKS_POSEIDON2_RC_12_EXTERNAL_FINAL,
        GOLDILOCKS_POSEIDON2_RC_12_EXTERNAL_INITIAL, GOLDILOCKS_POSEIDON2_RC_12_INTERNAL,
        Goldilocks, Poseidon2ExternalLayerGoldilocks, Poseidon2InternalLayerGoldilocks,
    };
    use p3_poseidon2::{ExternalLayerConstants, Poseidon2};
    use p3_symmetric::Permutation;
    use rand::SeedableRng;
    use rand::distr::{Distribution, StandardUniform};
    use rand::rngs::SmallRng;

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    type Packed = p3_goldilocks::PackedGoldilocksAVX512;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    const PACKED_LABEL: &str = "avx512 (8L)";

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(target_feature = "avx512f")
    ))]
    type Packed = p3_goldilocks::PackedGoldilocksAVX2;
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(target_feature = "avx512f")
    ))]
    const PACKED_LABEL: &str = "avx2 (4L)";

    #[cfg(target_arch = "aarch64")]
    type Packed = p3_goldilocks::PackedGoldilocksNeon;
    #[cfg(target_arch = "aarch64")]
    const PACKED_LABEL: &str = "neon (2L)";

    type GenericP2W8 = Poseidon2<
        Goldilocks,
        Poseidon2ExternalLayerGoldilocks<8>,
        Poseidon2InternalLayerGoldilocks,
        8,
        7,
    >;

    fn build_generic() -> GenericP2W8 {
        Poseidon2::new(
            ExternalLayerConstants::new(
                GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_INITIAL.to_vec(),
                GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_FINAL.to_vec(),
            ),
            GOLDILOCKS_POSEIDON2_RC_8_INTERNAL.to_vec(),
        )
    }

    type GenericP2W12 = Poseidon2<
        Goldilocks,
        Poseidon2ExternalLayerGoldilocks<12>,
        Poseidon2InternalLayerGoldilocks,
        12,
        7,
    >;

    fn build_generic_w12() -> GenericP2W12 {
        Poseidon2::new(
            ExternalLayerConstants::new(
                GOLDILOCKS_POSEIDON2_RC_12_EXTERNAL_INITIAL.to_vec(),
                GOLDILOCKS_POSEIDON2_RC_12_EXTERNAL_FINAL.to_vec(),
            ),
            GOLDILOCKS_POSEIDON2_RC_12_INTERNAL.to_vec(),
        )
    }

    pub fn run() {
        let mut args: Vec<String> = env::args().skip(1).collect();
        let mut seed: u64 = 1;
        if let Some(idx) = args.iter().position(|a| a == "--seed") {
            seed = args.get(idx + 1).unwrap().parse().unwrap();
            args.drain(idx..=idx + 1);
        }
        let n: usize = args
            .first()
            .map(|s| s.parse().unwrap())
            .unwrap_or(1_000_000);

        println!("packed path:    {}", PACKED_LABEL);
        println!("iters per run:  {}", n);
        println!("seed:           {}", seed);
        println!();

        // Build perms.
        let generic = build_generic();
        let constants = default_goldilocks_poseidon2_batched_8();
        #[cfg(target_arch = "aarch64")]
        let fused = p3_goldilocks::default_goldilocks_poseidon2_8();

        // Seed two random states (broadcast to packed lanes).
        let mut rng = SmallRng::seed_from_u64(seed);
        let scalars_a: [Goldilocks; 8] = std::array::from_fn(|_| {
            <StandardUniform as Distribution<Goldilocks>>::sample(&StandardUniform, &mut rng)
        });
        let scalars_b: [Goldilocks; 8] = std::array::from_fn(|_| {
            <StandardUniform as Distribution<Goldilocks>>::sample(&StandardUniform, &mut rng)
        });

        // Bitwise gate: batched-generic == single-generic for both states.
        {
            let mut s_a: [Packed; 8] = scalars_a.map(Into::into);
            let mut s_b: [Packed; 8] = scalars_b.map(Into::into);
            generic.permute_mut(&mut s_a);
            generic.permute_mut(&mut s_b);

            let mut b_a: [Packed; 8] = scalars_a.map(Into::into);
            let mut b_b: [Packed; 8] = scalars_b.map(Into::into);
            permute_batch_b2_p2_w8(&mut b_a, &mut b_b, &constants);
            assert_eq!(b_a, s_a, "batched-generic state_a mismatch");
            assert_eq!(b_b, s_b, "batched-generic state_b mismatch");
        }
        println!("bitwise gate:   batched-generic == single-generic   OK\n");

        // single-generic: time N permute_mut calls (alternating states).
        let single_generic_ns = {
            let mut s_a: [Packed; 8] = scalars_a.map(Into::into);
            let mut s_b: [Packed; 8] = scalars_b.map(Into::into);
            let t0 = Instant::now();
            for i in 0..n {
                if i & 1 == 0 {
                    generic.permute_mut(black_box(&mut s_a));
                } else {
                    generic.permute_mut(black_box(&mut s_b));
                }
            }
            let _ = black_box(s_a);
            let _ = black_box(s_b);
            t0.elapsed().as_secs_f64() * 1e9 / n as f64
        };

        // batched-generic: time N/2 permute_batch_b2 calls (each = 2 permutes).
        let batched_generic_ns = {
            let mut s_a: [Packed; 8] = scalars_a.map(Into::into);
            let mut s_b: [Packed; 8] = scalars_b.map(Into::into);
            let n_batches = n / 2;
            let t0 = Instant::now();
            for _ in 0..n_batches {
                permute_batch_b2_p2_w8(black_box(&mut s_a), black_box(&mut s_b), &constants);
            }
            let _ = black_box(s_a);
            let _ = black_box(s_b);
            t0.elapsed().as_secs_f64() * 1e9 / (n_batches * 2) as f64
        };

        // single-fused (NEON only).
        #[cfg(target_arch = "aarch64")]
        let single_fused_ns = {
            let mut s_a: [Packed; 8] = scalars_a.map(Into::into);
            let mut s_b: [Packed; 8] = scalars_b.map(Into::into);
            let t0 = Instant::now();
            for i in 0..n {
                if i & 1 == 0 {
                    fused.permute_mut(black_box(&mut s_a));
                } else {
                    fused.permute_mut(black_box(&mut s_b));
                }
            }
            let _ = black_box(s_a);
            let _ = black_box(s_b);
            t0.elapsed().as_secs_f64() * 1e9 / n as f64
        };

        // Serial (non-alternating) variants: hash-chain workload, period-1
        // dependency per call. Tests whether the alternating A/B loop above
        // is unfair to the fused path (which already extracts cross-lane
        // ILP within a single packed state via *_dual_w8 asm and may have
        // less marginal benefit from cross-packed-state OoO than generic).
        let single_generic_serial_ns = {
            let mut s_a: [Packed; 8] = scalars_a.map(Into::into);
            let t0 = Instant::now();
            for _ in 0..n {
                generic.permute_mut(black_box(&mut s_a));
            }
            let _ = black_box(s_a);
            t0.elapsed().as_secs_f64() * 1e9 / n as f64
        };

        #[cfg(target_arch = "aarch64")]
        let single_fused_serial_ns = {
            let mut s_a: [Packed; 8] = scalars_a.map(Into::into);
            let t0 = Instant::now();
            for _ in 0..n {
                fused.permute_mut(black_box(&mut s_a));
            }
            let _ = black_box(s_a);
            t0.elapsed().as_secs_f64() * 1e9 / n as f64
        };

        println!("| path                      | ns/permute |  vs single-generic |");
        println!("|---------------------------|-----------:|-------------------:|");
        println!(
            "| single-generic (alt)      | {:>9.2}  |             1.000x |",
            single_generic_ns
        );
        println!(
            "| batched-generic (alt)     | {:>9.2}  |             {:>5.3}x |",
            batched_generic_ns,
            batched_generic_ns / single_generic_ns
        );
        #[cfg(target_arch = "aarch64")]
        println!(
            "| single-fused (alt)        | {:>9.2}  |             {:>5.3}x |",
            single_fused_ns,
            single_fused_ns / single_generic_ns
        );
        println!(
            "| single-generic (serial)   | {:>9.2}  |             {:>5.3}x |",
            single_generic_serial_ns,
            single_generic_serial_ns / single_generic_ns
        );
        #[cfg(target_arch = "aarch64")]
        println!(
            "| single-fused (serial)     | {:>9.2}  |             {:>5.3}x |",
            single_fused_serial_ns,
            single_fused_serial_ns / single_generic_ns
        );

        println!();
        println!(
            "mechanism check (batched-generic vs single-generic):    {:>5.2}% reduction",
            (1.0 - batched_generic_ns / single_generic_ns) * 100.0
        );
        #[cfg(target_arch = "aarch64")]
        {
            println!(
                "ship-relevance (batched-generic vs single-fused):       {:>5.2}% reduction",
                (1.0 - batched_generic_ns / single_fused_ns) * 100.0
            );
            println!();
            println!("H2 test (serial / hash-chain workload):");
            println!(
                "  single-fused-serial vs single-generic-serial:        {:>+6.2}% (negative = fused faster)",
                (single_fused_serial_ns / single_generic_serial_ns - 1.0) * 100.0
            );
            println!(
                "  single-fused-alt    vs single-generic-alt:           {:>+6.2}% (alternating loop, for comparison)",
                (single_fused_ns / single_generic_ns - 1.0) * 100.0
            );
        }

        // --- width 12 ---
        println!();
        println!("=== width 12 ===");

        let generic_w12 = build_generic_w12();
        let constants_w12 = default_goldilocks_poseidon2_batched_12();

        let scalars_a_w12: [Goldilocks; 12] = std::array::from_fn(|_| {
            <StandardUniform as Distribution<Goldilocks>>::sample(&StandardUniform, &mut rng)
        });
        let scalars_b_w12: [Goldilocks; 12] = std::array::from_fn(|_| {
            <StandardUniform as Distribution<Goldilocks>>::sample(&StandardUniform, &mut rng)
        });

        {
            let mut s_a: [Packed; 12] = scalars_a_w12.map(Into::into);
            let mut s_b: [Packed; 12] = scalars_b_w12.map(Into::into);
            generic_w12.permute_mut(&mut s_a);
            generic_w12.permute_mut(&mut s_b);

            let mut b_a: [Packed; 12] = scalars_a_w12.map(Into::into);
            let mut b_b: [Packed; 12] = scalars_b_w12.map(Into::into);
            permute_batch_b2_p2_w12(&mut b_a, &mut b_b, &constants_w12);
            assert_eq!(b_a, s_a, "w12 batched-generic state_a mismatch");
            assert_eq!(b_b, s_b, "w12 batched-generic state_b mismatch");
        }
        println!("bitwise gate:   batched-generic == single-generic   OK\n");

        let single_generic_ns_w12 = {
            let mut s_a: [Packed; 12] = scalars_a_w12.map(Into::into);
            let mut s_b: [Packed; 12] = scalars_b_w12.map(Into::into);
            let t0 = Instant::now();
            for i in 0..n {
                if i & 1 == 0 {
                    generic_w12.permute_mut(black_box(&mut s_a));
                } else {
                    generic_w12.permute_mut(black_box(&mut s_b));
                }
            }
            let _ = black_box(s_a);
            let _ = black_box(s_b);
            t0.elapsed().as_secs_f64() * 1e9 / n as f64
        };

        let batched_generic_ns_w12 = {
            let mut s_a: [Packed; 12] = scalars_a_w12.map(Into::into);
            let mut s_b: [Packed; 12] = scalars_b_w12.map(Into::into);
            let n_batches = n / 2;
            let t0 = Instant::now();
            for _ in 0..n_batches {
                permute_batch_b2_p2_w12(black_box(&mut s_a), black_box(&mut s_b), &constants_w12);
            }
            let _ = black_box(s_a);
            let _ = black_box(s_b);
            t0.elapsed().as_secs_f64() * 1e9 / (n_batches * 2) as f64
        };

        println!("| path                      | ns/permute |  vs single-generic |");
        println!("|---------------------------|-----------:|-------------------:|");
        println!(
            "| single-generic (alt)      | {:>9.2}  |             1.000x |",
            single_generic_ns_w12
        );
        println!(
            "| batched-generic (alt)     | {:>9.2}  |             {:>5.3}x |",
            batched_generic_ns_w12,
            batched_generic_ns_w12 / single_generic_ns_w12
        );
        println!();
        println!(
            "mechanism check (batched-generic vs single-generic):    {:>5.2}% reduction",
            (1.0 - batched_generic_ns_w12 / single_generic_ns_w12) * 100.0
        );
    }
}
