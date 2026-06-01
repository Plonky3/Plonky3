//! Smoke test for the `wasm32+simd128` `PackedGoldilocksWasmSimd128` backend.
//!
//! Run with:
//! ```text
//! RUSTFLAGS="-C target-feature=+simd128" \
//!   cargo run --release --target wasm32-wasip1 \
//!     --bin wasm_smoke -p p3-goldilocks
//! ```
//! (requires the `wasm32-wasip1` toolchain target and a `wasmtime` runner —
//! either via `CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime` env var or a
//! `[target.wasm32-wasip1] runner = "wasmtime"` entry in `.cargo/config.toml`).
//!
//! On non-wasm32 targets this binary is a no-op.

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
fn main() {
    eprintln!(
        "wasm_smoke is a no-op on this target. Re-run with \
         `cargo run --target wasm32-wasip1 --bin wasm_smoke -p p3-goldilocks` \
         and RUSTFLAGS=\"-C target-feature=+simd128\"."
    );
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn main() {
    use p3_field::{Field, PackedFieldPow2, PackedValue, PrimeCharacteristicRing, PrimeField64};
    use p3_goldilocks::{Goldilocks, PackedGoldilocksWasmSimd128};

    const P: u64 = Goldilocks::ORDER_U64;

    // Distinct lane values catch lane-swap bugs.
    let pack =
        |a: u64, b: u64| PackedGoldilocksWasmSimd128([Goldilocks::new(a), Goldilocks::new(b)]);

    // ---- add ---------------------------------------------------------------
    let cases: &[(u64, u64, u64, u64)] = &[
        (0, 0, 0, 0),
        (1, 2, 3, 4),
        (P - 1, 1, 1, P - 1), // both lanes wrap differently
        (P - 1, P - 1, P - 1, P - 1),
        (0xffff_ffff, 0xffff_ffff_ffff_ffff, 1, 1),
    ];
    for &(a0, a1, b0, b1) in cases {
        let got = pack(a0, a1) + pack(b0, b1);
        let want = [
            Goldilocks::new(a0) + Goldilocks::new(b0),
            Goldilocks::new(a1) + Goldilocks::new(b1),
        ];
        assert_eq!(
            got.0, want,
            "add mismatch: ({a0:#x},{a1:#x}) + ({b0:#x},{b1:#x})"
        );
    }
    println!("add: OK");

    // ---- sub ---------------------------------------------------------------
    for &(a0, a1, b0, b1) in cases {
        let got = pack(a0, a1) - pack(b0, b1);
        let want = [
            Goldilocks::new(a0) - Goldilocks::new(b0),
            Goldilocks::new(a1) - Goldilocks::new(b1),
        ];
        assert_eq!(
            got.0, want,
            "sub mismatch: ({a0:#x},{a1:#x}) - ({b0:#x},{b1:#x})"
        );
    }
    println!("sub: OK");

    // ---- neg ---------------------------------------------------------------
    for &a in &[0u64, 1, 2, P - 1, P - 2, 0xffff_ffff, 0xffff_ffff_ffff_ffff] {
        let got = -PackedGoldilocksWasmSimd128([Goldilocks::new(a), Goldilocks::new(a)]);
        let want = -Goldilocks::new(a);
        assert_eq!(got.0, [want, want], "neg mismatch: {a:#x}");
    }
    println!("neg: OK");

    // ---- mul ---------------------------------------------------------------
    // Includes the `(P-1)^2` boundary, which is the worst case for `reduce128`.
    let mul_cases: &[(u64, u64)] = &[
        (0, 0),
        (1, 1),
        (2, 3),
        (P - 1, 2),
        (P - 1, P - 1),
        (0xffff_ffff, 0xffff_ffff),
        (0x1234_5678_9abc_def0, 0xfedc_ba98_7654_3210),
    ];
    for &(a, b) in mul_cases {
        let got = PackedGoldilocksWasmSimd128([Goldilocks::new(a), Goldilocks::new(a)])
            * PackedGoldilocksWasmSimd128([Goldilocks::new(b), Goldilocks::new(b)]);
        let want = Goldilocks::new(a) * Goldilocks::new(b);
        assert_eq!(got.0, [want, want], "mul mismatch: {a:#x} * {b:#x}");
    }
    println!("mul: OK");

    // ---- halve -------------------------------------------------------------
    // Mixed-parity lanes exercise both branches of `halve`.
    for &(a, b) in &[
        (0u64, 0u64),
        (2, 4),
        (1, 3),
        (0xffff_fffe, 0xffff_ffff),
        (P - 2, P - 1),
    ] {
        let got = pack(a, b).halve();
        let want = [Goldilocks::new(a).halve(), Goldilocks::new(b).halve()];
        assert_eq!(got.0, want, "halve mismatch: ({a:#x},{b:#x})");
    }
    println!("halve: OK");

    // ---- double ------------------------------------------------------------
    // Hits both halves of the canonicalize-then-add path used by `double`.
    for &(a, b) in &[
        (0u64, 0u64),
        (1, P - 1),
        (P - 1, P - 1),
        (P - 2, 2),
        (0xffff_ffff, 0xffff_ffff_ffff_ffff),
    ] {
        let got = pack(a, b).double();
        let want = [Goldilocks::new(a).double(), Goldilocks::new(b).double()];
        assert_eq!(got.0, want, "double mismatch: ({a:#x},{b:#x})");
    }
    println!("double: OK");

    // ---- square ------------------------------------------------------------
    for &a in &[0u64, 1, 2, P - 1, 0xdead_beef_cafe_babe] {
        let p = PackedGoldilocksWasmSimd128([Goldilocks::new(a), Goldilocks::new(a)]);
        let got = p.square();
        let want = Goldilocks::new(a).square();
        assert_eq!(got.0, [want, want], "square mismatch: {a:#x}");
    }
    println!("square: OK");

    // ---- interleave --------------------------------------------------------
    let v0 = pack(1, 2);
    let v1 = pack(3, 4);
    let (r0, r1) = v0.interleave(v1, 1);
    assert_eq!(
        r0.0,
        [Goldilocks::new(1), Goldilocks::new(3)],
        "interleave block_len=1 lane 0"
    );
    assert_eq!(
        r1.0,
        [Goldilocks::new(2), Goldilocks::new(4)],
        "interleave block_len=1 lane 1"
    );
    let (r0, r1) = v0.interleave(v1, 2);
    assert_eq!(r0.0, v0.0, "interleave block_len=2 lane 0");
    assert_eq!(r1.0, v1.0, "interleave block_len=2 lane 1");
    println!("interleave: OK");

    // ---- inverse -----------------------------------------------------------
    // Round-trip via the scalar inverse to validate that mul+reduce composes
    // cleanly with the inverse algorithm — catches reduction errors that pure
    // `a * b` tests don't surface because they only exercise one direction.
    for &(a, b) in &[(1u64, 2u64), (3, 7), (P - 1, 2), (0xffff_ffff, 0xdead_beef)] {
        let av = Goldilocks::new(a);
        let bv = Goldilocks::new(b);
        let packed = pack(a, b)
            * pack(
                av.inverse().as_canonical_u64(),
                bv.inverse().as_canonical_u64(),
            );
        assert_eq!(
            packed.0,
            [Goldilocks::ONE, Goldilocks::ONE],
            "a * a^-1 != 1 for ({a:#x},{b:#x})"
        );
    }
    println!("inverse: OK");

    // ---- mixed scalar op ---------------------------------------------------
    // Exercises the `Algebra<Goldilocks>` / `impl_*_base_field!` wiring: a
    // packed + scalar add must match a per-lane add against `broadcast(s)`.
    for &(a, b, s) in &[
        (0u64, 0u64, 0u64),
        (1, P - 1, 5),
        (P - 1, P - 2, P - 1),
        (0xffff_ffff_ffff_ffff, 1, 0xdead_beef),
    ] {
        let sv = Goldilocks::new(s);
        let got = pack(a, b) + sv;
        let want = [Goldilocks::new(a) + sv, Goldilocks::new(b) + sv];
        assert_eq!(
            got.0, want,
            "packed + scalar mismatch: ({a:#x},{b:#x}) + {s:#x}"
        );
    }
    println!("mixed scalar add: OK");

    // ---- PackedValue from_fn round-trip ------------------------------------
    let lanes = [Goldilocks::new(0x42), Goldilocks::new(0xdead_beef)];
    let packed = PackedGoldilocksWasmSimd128::from_fn(|i| lanes[i]);
    assert_eq!(packed.0, lanes, "from_fn round-trip mismatch");
    println!("from_fn: OK");

    println!("all packed wasm32+simd128 ops match scalar Goldilocks");
}
