//! Reproducible WASI benchmarks for Goldilocks and byte-hash primitives.
//!
//! Build with `--profile optimized --target wasm32-wasip1`; add
//! `RUSTFLAGS="-C target-feature=+simd128"` for SIMD and
//! `--features p3-blake3/wasm32-simd` to enable BLAKE3's SIMD backend.
//! Run the resulting module with Wasmtime or Node's WASI runner. An optional
//! first argument filters benchmark names by substring. Each case warms up,
//! then prints its name, operation count, elapsed nanoseconds, and checksum.
//! Repeat whole runs in alternating baseline/candidate order on an idle host.
//! Packed operation counts include every lane. Hash counts are messages and
//! Merkle counts are leaves. Permutations and initial inputs are constructed
//! before timing; resetting inputs inside each run and Merkle matrix clones are timed.

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!("Build this example for wasm32-wasip1, with or without +simd128.");
}

#[cfg(target_arch = "wasm32")]
fn main() {
    bench::run();
}

#[cfg(target_arch = "wasm32")]
mod bench {
    use core::hint::black_box;
    use std::time::Instant;

    use p3_blake3::Blake3;
    use p3_commit::Mmcs;
    use p3_field::extension::{binomial_mul, binomial_square, cubic_square, trinomial_cubic_mul};
    use p3_field::{Algebra, Field, PackedValue, PrimeCharacteristicRing, PrimeField64};
    use p3_goldilocks::poseidon1::{
        default_goldilocks_poseidon1_8, default_goldilocks_poseidon1_12,
    };
    use p3_goldilocks::{
        Goldilocks as F, MdsMatrixGoldilocks, default_goldilocks_poseidon2_8,
        default_goldilocks_poseidon2_12, default_goldilocks_poseidon2_16,
    };
    use p3_keccak::Keccak256Hash;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_sha256::{Sha256, Sha256Compress};
    use p3_symmetric::{
        CryptographicHasher, PaddingFreeSponge, Permutation, PseudoCompressionFunction,
        TruncatedPermutation,
    };

    type PF = <F as Field>::Packing;

    fn field(i: usize) -> F {
        F::new((i as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15))
    }

    fn packed(i: usize) -> PF {
        PF::from_fn(|lane| field(i * PF::WIDTH + lane))
    }

    fn field_checksum(values: &[PF]) -> u64 {
        values.iter().fold(0xcbf2_9ce4_8422_2325u64, |sum, value| {
            value.as_slice().iter().fold(sum, |sum, x| {
                sum.wrapping_mul(0x100_0000_01b3) ^ x.as_canonical_u64()
            })
        })
    }

    fn bytes_checksum(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |sum, &b| {
            sum.wrapping_mul(0x100_0000_01b3) ^ u64::from(b)
        })
    }

    fn input_byte(i: usize) -> u8 {
        ((i as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 56) as u8
    }

    fn measure(
        filter: &str,
        name: &str,
        iterations: usize,
        units: usize,
        mut run: impl FnMut(usize) -> u64,
    ) {
        if !name.contains(filter) {
            return;
        }
        // Warm the same function and inputs before timing. Every invocation resets
        // its state, so checksum comparisons do not depend on warmup or engine speed.
        black_box(run(iterations / 4 + 1));
        black_box(run(iterations / 4 + 1));
        let start = Instant::now();
        let checksum = black_box(run(iterations));
        let ns = start.elapsed().as_nanos();
        println!("{name}\t{}\t{ns}\t{checksum:016x}", iterations * units);
    }

    pub fn run() {
        let filter = std::env::args().nth(1).unwrap_or_default();
        arithmetic(&filter);
        permutations(&filter);
        hashes(&filter, "keccak", Keccak256Hash);
        hashes(&filter, "sha256", Sha256);
        hashes(&filter, "blake3", Blake3);
        compressions(&filter);
        merkle(&filter);
    }

    fn arithmetic(filter: &str) {
        const N: usize = 2_000_000;
        macro_rules! scalar {
            ($name:literal, $op:expr) => {
                measure(filter, $name, N, 1, |n| {
                    let mut a = field(7);
                    let b = black_box(field(13));
                    for _ in 0..n {
                        a = $op(black_box(a), b);
                    }
                    a.as_canonical_u64()
                });
            };
        }
        scalar!("scalar_mul", |a: F, b: F| a * b);
        scalar!("scalar_square", |a: F, _b: F| a.square());
        scalar!("scalar_pow2_16", |a: F, _b: F| a.mul_2exp_u64(16));
        macro_rules! vector {
            ($name:literal, $op:expr) => {
                measure(filter, $name, N, PF::WIDTH, |n| {
                    let mut a = packed(7);
                    let b = black_box(packed(13));
                    for _ in 0..n {
                        a = $op(black_box(a), b);
                    }
                    field_checksum(&[a])
                });
            };
        }
        vector!("packed_mul", |a: PF, b: PF| a * b);
        vector!("packed_square", |a: PF, _b: PF| a.square());
        vector!("packed_double", |a: PF, _b: PF| a.double());
        vector!("packed_pow2_4", |a: PF, _b: PF| a.mul_2exp_u64(4));
        vector!("packed_pow2_16", |a: PF, _b: PF| a.mul_2exp_u64(16));
        vector!("packed_pow2_32", |a: PF, _b: PF| a.mul_2exp_u64(32));
        let coefficient = black_box(field(13));
        vector!("packed_scalar_mul", |a: PF, _b: PF| a * coefficient);

        measure(filter, "packed_mul_bulk", 8_192, 256 * PF::WIDTH, |n| {
            let mut a: [PF; 256] = core::array::from_fn(packed);
            let b = black_box(packed(71));
            for _ in 0..n {
                for x in black_box(&mut a) {
                    *x *= b;
                }
            }
            field_checksum(&a)
        });

        macro_rules! dots {
            ($n:literal) => {{
                let initial: [PF; $n] = core::array::from_fn(packed);
                let b: [PF; $n] = core::array::from_fn(|i| packed(i + 17));
                let f: [F; $n] = core::array::from_fn(|i| field(i + 17));
                measure(filter, concat!("packed_dot", $n), 500_000, PF::WIDTH, |n| {
                    let mut a = initial;
                    for _ in 0..n {
                        a[0] = PF::dot_product(black_box(&a), black_box(&b));
                    }
                    field_checksum(&a)
                });
                measure(filter, concat!("mixed_dot", $n), 500_000, PF::WIDTH, |n| {
                    let mut a = initial;
                    for _ in 0..n {
                        a[0] = PF::mixed_dot_product(black_box(&a), black_box(&f));
                    }
                    field_checksum(&a)
                });
            }};
        }
        dots!(2);
        dots!(3);
        dots!(4);
        dots!(8);
        for len in [1, 2, 3, 4, 8, 16, 31, 63, 64, 65, 127, 255, 256] {
            let initial: Vec<PF> = (0..len).map(packed).collect();
            let f: Vec<F> = (17..len + 17).map(field).collect();
            measure(filter, &format!("mixed_lc{len}"), 50_000, PF::WIDTH, |n| {
                let mut a = initial.clone();
                for _ in 0..n {
                    a[0] = PF::batched_linear_combination(black_box(&a), black_box(&f));
                }
                field_checksum(&a)
            });
        }
        macro_rules! extension {
            ($name:literal, $degree:literal, $operation:expr) => {{
                let b: [PF; $degree] = core::array::from_fn(|i| packed(i + 17));
                measure(filter, $name, 500_000, PF::WIDTH, |n| {
                    let mut a: [PF; $degree] = core::array::from_fn(packed);
                    for _ in 0..n {
                        let mut result = [PF::ZERO; $degree];
                        $operation(black_box(&a), black_box(&b), &mut result);
                        a = result;
                    }
                    field_checksum(&a)
                });
            }};
        }
        extension!("ext2_mul", 2, |a, b, res| binomial_mul(
            a,
            b,
            res,
            F::new(7)
        ));
        extension!("ext2_square", 2, |a, _b, res| binomial_square(
            a,
            res,
            F::new(7)
        ));
        extension!("ext3_mul", 3, |a, b, res| trinomial_cubic_mul(a, b, res));
        extension!("ext3_square", 3, |a, _b, res| cubic_square(a, res));
    }

    fn permutations(filter: &str) {
        macro_rules! poseidon {
            ($name:literal, $width:literal, $constructor:expr) => {{
                let perm = $constructor;
                measure(filter, concat!($name, "_scalar"), 20_000, 1, |n| {
                    let mut state: [F; $width] = core::array::from_fn(field);
                    for _ in 0..n {
                        perm.permute_mut(black_box(&mut state));
                    }
                    state.iter().fold(0u64, |sum, x| {
                        sum.wrapping_mul(0x100_0000_01b3) ^ x.as_canonical_u64()
                    })
                });
                measure(filter, concat!($name, "_packed"), 20_000, PF::WIDTH, |n| {
                    let mut state: [PF; $width] = core::array::from_fn(packed);
                    for _ in 0..n {
                        perm.permute_mut(black_box(&mut state));
                    }
                    field_checksum(&state)
                });
            }};
        }
        poseidon!("poseidon2_8", 8, default_goldilocks_poseidon2_8());
        poseidon!("poseidon2_12", 12, default_goldilocks_poseidon2_12());
        poseidon!("poseidon2_16", 16, default_goldilocks_poseidon2_16());
        poseidon!("poseidon1_8", 8, default_goldilocks_poseidon1_8());
        poseidon!("poseidon1_12", 12, default_goldilocks_poseidon1_12());
        macro_rules! mds {
            ($width:literal) => {
                measure(filter, concat!("mds", $width), 100_000, PF::WIDTH, |n| {
                    let mut state: [PF; $width] = core::array::from_fn(packed);
                    for _ in 0..n {
                        MdsMatrixGoldilocks.permute_mut(black_box(&mut state));
                    }
                    field_checksum(&state)
                });
            };
        }
        mds!(8);
        mds!(12);
        mds!(16);
        mds!(24);
    }

    fn hashes(filter: &str, name: &str, hasher: impl CryptographicHasher<u8, [u8; 32]>) {
        const MESSAGES: usize = 256;
        for len in [64, 136, 1_024, 4_096] {
            let initial: Vec<u8> = (0..len * MESSAGES).map(input_byte).collect();
            let mut out = [[0u8; 32]; MESSAGES];
            hasher.hash_many(&initial, &mut out);
            for (message, digest) in initial.chunks_exact(len).zip(out) {
                assert_eq!(hasher.hash_slice(message), digest);
            }
            let rounds = 2_048 / (len / 64).max(1);
            measure(filter, &format!("{name}_{len}"), rounds, MESSAGES, |n| {
                let mut input = initial.clone();
                for i in 0..n {
                    input[0] = i as u8;
                    hasher.hash_many(black_box(&input), black_box(&mut out));
                }
                bytes_checksum(out.as_flattened())
            });
        }
    }

    fn compressions(filter: &str) {
        const MESSAGES: usize = 256;
        let input: [[[u8; 32]; 2]; MESSAGES] = core::array::from_fn(|i| {
            core::array::from_fn(|j| core::array::from_fn(|k| input_byte(i * 64 + j * 32 + k)))
        });
        let mut out = [[0u8; 32]; MESSAGES];
        Sha256Compress.compress_many(&input, &mut out);
        for (input, digest) in input.iter().zip(out) {
            assert_eq!(Sha256Compress.compress(*input), digest);
        }
        measure(filter, "sha256_compress", 4_096, MESSAGES, |n| {
            let mut input = input;
            for i in 0..n {
                input[0][0][0] = i as u8;
                Sha256Compress.compress_many(black_box(&input), black_box(&mut out));
            }
            bytes_checksum(out.as_flattened())
        });
    }

    fn merkle(filter: &str) {
        type Perm = p3_goldilocks::Poseidon2Goldilocks<8>;
        type Hasher = PaddingFreeSponge<Perm, 8, 4, 4>;
        type Compressor = TruncatedPermutation<Perm, 2, 4, 8>;
        type Tree = MerkleTreeMmcs<PF, PF, Hasher, Compressor, 2, 4>;
        let perm = default_goldilocks_poseidon2_8();
        let tree = Tree::new(Hasher::new(perm.clone()), Compressor::new(perm), 0);
        const ROWS: usize = 1 << 14;
        let matrix = RowMajorMatrix::new((0..ROWS * 8).map(field).collect(), 8);
        measure(filter, "merkle_poseidon2_8", 4, ROWS, |n| {
            let mut checksum = 0u64;
            for _ in 0..n {
                let (cap, data) = tree.commit(vec![black_box(matrix.clone())]);
                let digests: &[[F; 4]] = cap.as_ref();
                checksum =
                    checksum.wrapping_mul(0x100_0000_01b3) ^ digests[0][0].as_canonical_u64();
                black_box(data);
            }
            checksum
        });
        merkle_bytes(filter, "keccak", Keccak256Hash);
        merkle_bytes(filter, "sha256", Sha256);
        merkle_bytes(filter, "blake3", Blake3);
    }

    fn merkle_bytes(filter: &str, name: &str, hasher: impl CryptographicHasher<u8, [u8; 32]>) {
        const LEAVES: usize = 1 << 14;
        let input: Vec<u8> = (0..LEAVES * 64).map(input_byte).collect();
        let mut left = vec![[0u8; 32]; LEAVES];
        let mut right = vec![[0u8; 32]; LEAVES / 2];
        measure(filter, &format!("merkle_{name}"), 4, LEAVES, |n| {
            let mut checksum = 0u64;
            for _ in 0..n {
                hasher.hash_many(black_box(&input), &mut left);
                let mut width = LEAVES;
                while width > 1 {
                    hasher.hash_many(
                        black_box(left[..width].as_flattened()),
                        &mut right[..width / 2],
                    );
                    std::mem::swap(&mut left, &mut right);
                    width /= 2;
                }
                checksum = checksum.wrapping_mul(0x100_0000_01b3) ^ bytes_checksum(&left[0]);
                // Restore capacities for the next tree without allocating in the loop.
                if left.len() < LEAVES {
                    std::mem::swap(&mut left, &mut right);
                }
            }
            checksum
        });
    }
}
