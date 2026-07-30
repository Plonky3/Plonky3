//! End-to-end-ish benchmark for wasm32+simd128: builds a Merkle tree over a matrix of
//! Goldilocks field elements using the specialized `PackedGoldilocksWasmSimd128` Poseidon2
//! permutation for both leaf hashing and internal-node compression — the dominant real cost
//! in Merkle-tree-heavy STARK proving. Serves as a realistic baseline for judging whether
//! further Poseidon2-level optimizations (e.g. software-pipelined dual-permutation hashing)
//! are worth pursuing, closer to a real workload than the isolated single-permutation
//! microbenchmark in `wasm_bench.rs`.
//!
//! Run with:
//! ```text
//! RUSTFLAGS="-C target-feature=+simd128" \
//!   cargo run --release --target wasm32-wasip1 \
//!     --example wasm_merkle_bench -p p3-goldilocks
//! ```
//! On non-wasm32 targets this binary is a no-op.

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
fn main() {
    eprintln!(
        "wasm_merkle_bench is a no-op on this target. Re-run with \
         `cargo run --release --target wasm32-wasip1 --example wasm_merkle_bench -p p3-goldilocks` \
         and RUSTFLAGS=\"-C target-feature=+simd128\"."
    );
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn main() {
    use std::time::Instant;

    use p3_commit::Mmcs;
    use p3_field::Field;
    use p3_goldilocks::{Goldilocks, default_goldilocks_poseidon2_8};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    type Perm = p3_goldilocks::Poseidon2Goldilocks<8>;
    type Hasher = PaddingFreeSponge<Perm, 8, 4, 4>;
    type Compressor = TruncatedPermutation<Perm, 2, 4, 8>;
    type Packing = <Goldilocks as Field>::Packing;
    type Mmcs8 = MerkleTreeMmcs<Packing, Packing, Hasher, Compressor, 2, 4>;

    fn report(name: &str, n: u64, elapsed_ns: u128) {
        let per_op = elapsed_ns as f64 / n as f64;
        println!("{name:>28}: {per_op:>10.1} ns/leaf   ({n} leaves in {elapsed_ns} ns total)");
    }

    let perm = default_goldilocks_poseidon2_8();
    let hasher = Hasher::new(perm.clone());
    let compressor = Compressor::new(perm);
    let mmcs = Mmcs8::new(hasher, compressor, 0);

    let mut rng = SmallRng::seed_from_u64(0xBA5E);

    // Sweep leaf counts spanning small (few tree levels) to large (many levels, more
    // compression work relative to leaf hashing) so the pipelining question — does
    // batching independent permutations help — has enough range to show a trend.
    for &log_leaves in &[8usize, 12, 16] {
        let rows = 1usize << log_leaves;
        const COLS: usize = 8;
        let values: Vec<Goldilocks> = (0..rows * COLS).map(|_| rng.random()).collect();
        let matrix = RowMajorMatrix::new(values, COLS);

        let t0 = Instant::now();
        let (_commitment, _prover_data) = mmcs.commit(vec![matrix]);
        let elapsed_ns = t0.elapsed().as_nanos();

        report(
            &format!("merkle_2^{log_leaves}_leaves"),
            rows as u64,
            elapsed_ns,
        );
    }
}
