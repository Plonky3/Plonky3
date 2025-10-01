# Baby Bear Field

Implementation of the Baby Bear prime field F_p, where p = 2^31 - 2^27 + 1.

## Properties

- **Prime**: p = 2^31 - 2^27 + 1 = 0x78000001
- **2-adicity**: 27 (highest possible for 31-bit prime)
- **Montgomery arithmetic**: Optimized 32-bit operations

## Features

- **Field extensions**: Quartic (4), quintic (5), and octic (8) extensions
- **Poseidon2 hash**: Optimized implementation with widths 16 and 24
- **MDS matrices**: Support for sizes 8, 12, 16, 24, 32, 64
- **SIMD optimizations**: ARM NEON, x86_64 AVX2/AVX512

## Usage

```rust
use p3_baby_bear::BabyBear;

// Basic operations
let a = BabyBear::from_u32(100);
let b = BabyBear::from_u32(200);
let sum = a + b;
let product = a * b;
```

```rust
use p3_baby_bear::{default_babybear_poseidon2_16, BabyBear};

// Poseidon2 hashing
let poseidon = default_babybear_poseidon2_16();
let mut input: [BabyBear; 16] = BabyBear::new_array([/* ... */]);
poseidon.permute_mut(&mut input);
```
