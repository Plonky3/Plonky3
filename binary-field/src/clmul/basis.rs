//! The `F_2`-linear change of basis between the Wiedemann tower and a polynomial basis.
//!
//! Carryless multiplication computes in `GF(2)[x]/(g)` for an irreducible `g`, a field whose
//! basis is `1, x, …, x^(n−1)`. The tower stores elements in the basis of products
//! `∏_{i ∈ S} X_i` instead, so the two representations have to be related before the hardware
//! instruction is of any use.
//!
//! Writing `T_n` for the tower level of width `n` and `P_n = GF(2)[x]/(g_n)`, the tower is
//! generated over `GF(2)` by `X_0, …, X_{log n − 1}` subject to `X_k² + X_{k−1}·X_k + 1 = 0`
//! with `X_{−1} = 1`. Any images `ξ_k ∈ P_n` satisfying those same relations therefore extend
//! uniquely to a `GF(2)`-algebra homomorphism `N: T_n → P_n`; because `T_n` is a field, `N` is
//! injective, and since both sides have dimension `n` over `GF(2)` it is an isomorphism. So it
//! suffices to pin down the `ξ_k` — the relations are asserted at compile time by
//! `relations_hold`, and `derivation_reproduces_the_generator_images` recomputes the images from
//! scratch to show they are the ones the derivation yields.
//!
//! `N` sends the tower basis element indexed by the bit pattern `j` to `∏_{i ∈ bits(j)} ξ_i`.
//! Those `n` images are the columns of `N`; the columns of `M = N⁻¹` come from Gaussian
//! elimination over `GF(2)`. Both matrices are applied one byte at a time through the lookup
//! tables built below.

/// `GF(2^32)` as `GF(2)[x] / (x^32 + x^7 + x^3 + x^2 + 1)`.
pub(super) const TAIL_32: u128 = 0x8d;

const XI_32: [u128; 5] = [
    0x54fd_1265,
    0xee55_98fa,
    0x7a9d_86c2,
    0xc890_6c73,
    0xe87a_3f19,
];

/// `GF(2^64)` as `GF(2)[x] / (x^64 + x^4 + x^3 + x + 1)`.
pub(super) const TAIL_64: u128 = 0b1_1011;

/// `GF(2^128)` as `GF(2)[x] / (x^128 + x^7 + x^2 + x + 1)`.
pub(crate) const TAIL_128: u128 = 0b1000_0111;

/// The images of the tower generators `X_0, …, X_5` in `GF(2)[x]/(x^64 + x^4 + x^3 + x + 1)`.
const XI_64: [u128; 6] = [
    0x19c9_369f_278a_dc03,
    0xfc39_a481_a127_aa9d,
    0xae4a_b740_40e7_9118,
    0xed58_cafe_f8f2_2bc9,
    0xea6b_20d9_15fd_bb77,
    0x3f2a_7c50_1e89_8bcd,
];

/// The images of the tower generators `X_0, …, X_6` in `GF(2)[x]/(x^128 + x^7 + x^2 + x + 1)`.
const XI_128: [u128; 7] = [
    0x295a_c0b1_f473_1af9_676a_ac9f_a4b2_0b08,
    0x7959_d70c_e1ee_6942_53b8_5b64_02b1_e849,
    0x6cfe_46e8_277b_2e9d_f167_7bc1_710c_5c54,
    0xd748_3fda_e776_3b6f_f501_c92b_7a41_cc34,
    0x681c_6a8c_3679_2c02_92d6_dad7_72f9_2df5,
    0x496c_b27f_a3eb_e728_9596_0776_ef42_f90b,
    0x9cfc_a256_33f9_993f_4e1f_5110_70b2_3e78,
];

/// Multiplication in `GF(2)[x]/(x^bits + tail)`, bit-serial from the top coefficient down.
///
/// Operands and result occupy the low `bits` bits of a `u128`. This is the always-correct
/// definition of the polynomial-basis product: it is what builds the tables at compile time
/// and what the hardware path is checked against in the tests.
pub(super) const fn poly_mul(a: u128, b: u128, bits: usize, tail: u128) -> u128 {
    let mask = u128::MAX >> (128 - bits);
    let top = 1u128 << (bits - 1);
    let mut acc = 0u128;
    let mut i = bits;
    while i > 0 {
        i -= 1;
        // `acc *= x`, folding `x^bits ≡ tail` back in.
        let overflowed = acc & top != 0;
        acc = (acc << 1) & mask;
        if overflowed {
            acc ^= tail;
        }
        if (b >> i) & 1 == 1 {
            acc ^= a;
        }
    }
    acc
}

/// Whether the images satisfy `ξ_k² + ξ_{k−1}·ξ_k + 1 = 0` for every `k`, with `ξ_{−1} = 1`.
/// This is what makes `N` a field isomorphism rather than merely an invertible `GF(2)`-linear
/// map: invertibility alone says nothing about multiplicativity, and almost every invertible
/// matrix over `GF(2)` fails to be multiplicative. Asserting it below puts the property in the
/// build rather than in the test suite, so no `ξ` can be wrong in a crate that compiles.
const fn relations_hold(bits: usize, tail: u128, xi: &[u128]) -> bool {
    let mut k = 0;
    while k < xi.len() {
        let previous = if k == 0 { 1 } else { xi[k - 1] };
        if poly_mul(xi[k], xi[k], bits, tail) ^ poly_mul(previous, xi[k], bits, tail) ^ 1 != 0 {
            return false;
        }
        k += 1;
    }
    true
}

const _: () = assert!(
    relations_hold(32, TAIL_32, &XI_32),
    "XI_32 violates the tower relations"
);

const _: () = assert!(
    relations_hold(64, TAIL_64, &XI_64),
    "XI_64 violates the tower relations"
);
const _: () = assert!(
    relations_hold(128, TAIL_128, &XI_128),
    "XI_128 violates the tower relations"
);

/// The columns of `N`: the image of each tower basis element `∏_{i ∈ bits(j)} ξ_i`.
///
/// Only the first `bits` entries are meaningful. Splitting off the lowest set bit of `j` reuses
/// the already-computed image of the rest, so each column costs a single multiplication.
const fn columns(bits: usize, tail: u128, xi: &[u128]) -> [u128; 128] {
    let mut cols = [0u128; 128];
    // The empty product is the multiplicative identity of both representations.
    cols[0] = 1;
    let mut j = 1;
    while j < bits {
        let low = j.trailing_zeros() as usize;
        cols[j] = poly_mul(cols[j ^ (1 << low)], xi[low], bits, tail);
        j += 1;
    }
    cols
}

/// The columns of `M = N⁻¹`, by Gauss-Jordan elimination over `GF(2)`.
///
/// Column operations on `N` that reduce it to the identity turn the identity into `N⁻¹`, so the
/// same operations are replayed on `inv`. Reaching the identity at all is exactly the statement
/// that `N` is invertible, hence that `N` is an isomorphism and `x^bits + tail` is irreducible.
///
/// # Panics
/// Panics at compile time if `N` is singular.
const fn invert(cols: &[u128; 128], bits: usize) -> [u128; 128] {
    let mut mat = *cols;
    let mut inv = [0u128; 128];
    let mut j = 0;
    while j < bits {
        inv[j] = 1 << j;
        j += 1;
    }

    let mut pivot = 0;
    while pivot < bits {
        let mut col = pivot;
        while col < bits && (mat[col] >> pivot) & 1 == 0 {
            col += 1;
        }
        assert!(col < bits, "the change-of-basis matrix is singular");

        let swapped = mat[pivot];
        mat[pivot] = mat[col];
        mat[col] = swapped;
        let swapped = inv[pivot];
        inv[pivot] = inv[col];
        inv[col] = swapped;

        let mut col = 0;
        while col < bits {
            if col != pivot && (mat[col] >> pivot) & 1 == 1 {
                mat[col] ^= mat[pivot];
                inv[col] ^= inv[pivot];
            }
            col += 1;
        }
        pivot += 1;
    }
    inv
}

/// One lookup table per byte of the input, holding that byte's contribution to the product.
///
/// Only the first `bits / 8` tables are meaningful. Each entry drops the lowest set bit of the
/// index and reuses the entry for the rest, so every entry costs a single `XOR`.
///
/// A byte is the widest chunk worth tabulating: the whole set of tables comes to 176 KiB, and
/// the conversions are limited by how many loads the core can retire, so halving the lookups
/// pays for the extra footprint. Narrower nibble tables fit a first-level cache several times
/// over but measure substantially slower at both widths.
const fn byte_tables(cols: &[u128; 128], bits: usize) -> [[u128; 256]; 16] {
    let mut tables = [[0u128; 256]; 16];
    let mut byte = 0;
    while byte < bits / 8 {
        let mut value = 1usize;
        while value < 256 {
            let low = value.trailing_zeros() as usize;
            tables[byte][value] = tables[byte][value ^ (1 << low)] ^ cols[byte * 8 + low];
            value += 1;
        }
        byte += 1;
    }
    tables
}

/// The columns of the tower-basis matrix of squaring.
///
/// Squaring is `GF(2)`-linear in characteristic 2 — the cross term of `(a + b)²` is `2ab` — so
/// in any basis it is a matrix, and the tower basis is no exception. Conjugating the
/// polynomial-basis square by the change of basis gives that matrix: column `j` is
/// `M(N(e_j)²)`, for `e_j` the `j`-th tower basis element and `M = N⁻¹`.
///
/// Applying `M` to a vector is the sum of the columns of `M` selected by its set bits, which is
/// what the inner loop does. Only the first `bits` entries are meaningful.
const fn square_columns(bits: usize, tail: u128, cols: &[u128; 128]) -> [u128; 128] {
    let inverse = invert(cols, bits);
    let mut squared = [0u128; 128];
    let mut j = 0;
    while j < bits {
        let image = poly_mul(cols[j], cols[j], bits, tail);
        let mut column = 0u128;
        let mut i = 0;
        while i < bits {
            if (image >> i) & 1 == 1 {
                column ^= inverse[i];
            }
            i += 1;
        }
        squared[j] = column;
        j += 1;
    }
    squared
}

/// The low 64 bits of each entry of the first eight tables.
const fn narrow(tables: &[[u128; 256]; 16]) -> [[u64; 256]; 8] {
    let mut narrowed = [[0u64; 256]; 8];
    let mut byte = 0;
    while byte < 8 {
        let mut value = 0;
        while value < 256 {
            narrowed[byte][value] = tables[byte][value] as u64;
            value += 1;
        }
        byte += 1;
    }
    narrowed
}

const fn narrow_32(tables: &[[u128; 256]; 16]) -> [[u32; 256]; 4] {
    let mut result = [[0; 256]; 4];
    let mut i = 0;
    while i < 4 {
        let mut j = 0;
        while j < 256 {
            result[i][j] = tables[i][j] as u32;
            j += 1;
        }
        i += 1;
    }
    result
}

const COLUMNS_32: [u128; 128] = columns(32, TAIL_32, &XI_32);
static TOWER_TO_POLY_32: [[u32; 256]; 4] = narrow_32(&byte_tables(&COLUMNS_32, 32));
static POLY_TO_TOWER_32: [[u32; 256]; 4] = narrow_32(&byte_tables(&invert(&COLUMNS_32, 32), 32));

#[inline]
fn apply_32(tables: &[[u32; 256]; 4], value: u32) -> u32 {
    tables.iter().enumerate().fold(0, |acc, (i, row)| {
        acc ^ row[(value >> (8 * i)) as u8 as usize]
    })
}

#[inline]
pub(super) fn tower_to_poly_32(value: u32) -> u32 {
    apply_32(&TOWER_TO_POLY_32, value)
}

#[inline]
pub(super) fn poly_to_tower_32(value: u32) -> u32 {
    apply_32(&POLY_TO_TOWER_32, value)
}

const COLUMNS_64: [u128; 128] = columns(64, TAIL_64, &XI_64);
const COLUMNS_128: [u128; 128] = columns(128, TAIL_128, &XI_128);

static TOWER_TO_POLY_64: [[u64; 256]; 8] = narrow(&byte_tables(&COLUMNS_64, 64));
static POLY_TO_TOWER_64: [[u64; 256]; 8] = narrow(&byte_tables(&invert(&COLUMNS_64, 64), 64));
static SQUARE_64: [[u64; 256]; 8] =
    narrow(&byte_tables(&square_columns(64, TAIL_64, &COLUMNS_64), 64));
static TOWER_TO_POLY_128: [[u128; 256]; 16] = byte_tables(&COLUMNS_128, 128);
static POLY_TO_TOWER_128: [[u128; 256]; 16] = byte_tables(&invert(&COLUMNS_128, 128), 128);

/// The polynomial-basis coordinates of a tower-basis bit pattern, at compile time.
///
/// The change of basis is `GF(2)`-linear.
/// An element's image is therefore the sum of the columns its set bits select.
///
/// The table-driven route sums a byte at a time, which constant evaluation cannot index into.
/// This walks the bits instead.
pub(crate) const fn tower_image_128(v: u128) -> u128 {
    image(v, 128, &COLUMNS_128)
}

/// The polynomial-basis coordinates of a 64-bit tower-basis bit pattern, at compile time.
pub(crate) const fn tower_image_64(v: u64) -> u64 {
    image(v as u128, 64, &COLUMNS_64) as u64
}

/// The image of a bit pattern under the change of basis with the given columns.
const fn image(v: u128, bits: usize, columns: &[u128; 128]) -> u128 {
    let mut acc = 0;
    let mut i = 0;
    while i < bits {
        if (v >> i) & 1 == 1 {
            acc ^= columns[i];
        }
        i += 1;
    }
    acc
}

/// Applies the eight-table form of a `64 × 64` matrix over `GF(2)`.
#[inline]
fn apply_64(tables: &[[u64; 256]; 8], v: u64) -> u64 {
    let mut acc = 0;
    for (byte, table) in tables.iter().enumerate() {
        acc ^= table[(v >> (8 * byte)) as u8 as usize];
    }
    acc
}

/// Applies the sixteen-table form of a `128 × 128` matrix over `GF(2)`.
#[inline]
fn apply_128(tables: &[[u128; 256]; 16], v: u128) -> u128 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        use core::arch::aarch64::{uint8x16_t, vdupq_n_u8, veorq_u8, vld1q_u8};
        // Keep each table entry and the XOR accumulator in one 128-bit register.
        // SAFETY: NEON is enabled, every selected entry contains 16 initialized bytes,
        // and unaligned vector loads are supported. XOR is independent of byte order.
        unsafe {
            let mut acc = vdupq_n_u8(0);
            for (byte, table) in tables.iter().enumerate() {
                let entry = &table[(v >> (8 * byte)) as u8 as usize];
                acc = veorq_u8(acc, vld1q_u8(core::ptr::from_ref(entry).cast()));
            }
            core::mem::transmute::<uint8x16_t, u128>(acc)
        }
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    {
        use core::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_setzero_si128, _mm_xor_si128};
        // Keep each table entry and the XOR accumulator in one 128-bit register.
        //
        // SAFETY: `sse2` is enabled. It is in the baseline of the ordinary `x86_64` targets but
        // not of the bare-metal ones, where the gate sends this to the scalar fold below.
        //
        // Every selected entry holds sixteen initialized bytes.
        //
        // The load is the unaligned form, and exclusive or ignores byte order.
        unsafe {
            let mut acc = _mm_setzero_si128();
            for (byte, table) in tables.iter().enumerate() {
                let entry = &table[(v >> (8 * byte)) as u8 as usize];
                acc = _mm_xor_si128(acc, _mm_loadu_si128(core::ptr::from_ref(entry).cast()));
            }
            core::mem::transmute::<__m128i, u128>(acc)
        }
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "x86_64", target_feature = "sse2")
    )))]
    {
        let mut acc = 0;
        for (byte, table) in tables.iter().enumerate() {
            acc ^= table[(v >> (8 * byte)) as u8 as usize];
        }
        acc
    }
}

/// The blocked kernel, over whichever instruction set converts many elements at once.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
mod blocked {
    use core::arch::x86_64::{
        __m512i, _mm512_gf2p8affine_epi64_epi8, _mm512_loadu_si512, _mm512_set1_epi64,
        _mm512_storeu_si512, _mm512_ternarylogic_epi64, _mm512_unpackhi_epi8,
        _mm512_unpackhi_epi16, _mm512_unpackhi_epi32, _mm512_unpackhi_epi64, _mm512_unpacklo_epi8,
        _mm512_unpacklo_epi16, _mm512_unpacklo_epi32, _mm512_unpacklo_epi64, _mm512_xor_si512,
    };

    use super::{COLUMNS_128, invert};

    /// Elements per block: one byte of each fills the 64 bytes of a `512`-bit register.
    ///
    /// Sixteen such registers hold the whole block, one per byte position of an element.
    pub(super) const BLOCK: usize = 64;

    /// The immediate of a three-input exclusive-or, as a truth table of its three operands.
    const XOR3: i32 = 0x96;

    /// Where the interleaving butterfly leaves each row of the transpose.
    ///
    /// Four rounds of `vpunpck` transpose a `16 × 16` byte block with the row index reversed.
    ///
    /// ```text
    ///     butterfly(a)[bitrev4(j)] = row j of the transpose
    /// ```
    ///
    /// Reindexing by this table recovers the true transpose and costs no instruction.
    const BITREV4: [usize; 16] = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15];

    /// A runtime 128 × 128 map prepared for the GFNI byte-affine kernel.
    ///
    /// The columns are retained for the scalar tail; the blocks hold the same map in GFNI's
    /// reversed row-byte convention. Construction is intentionally per call because transcript
    /// challenges make the map dynamic.
    pub(super) struct PreparedMap {
        pub(super) columns: [u128; 128],
        blocks: [[u64; 16]; 16],
    }

    /// The `8 × 8` blocks of a `128 × 128` map, at `blocks[k][j]` for input byte `j` to output `k`.
    ///
    /// # Algorithm
    ///
    /// `vgf2p8affineqb` reads the row for output bit `i` from byte `7 − i` of the quadword.
    /// It pairs bit `b` of that row with bit `b` of the input byte.
    ///
    /// ```text
    ///     out_byte_k[i] = XOR over b of block[k][j][8·(7−i) + b] · in_byte_j[b]
    /// ```
    ///
    /// Column `8j + b` of the map holds the image of input bit `8j + b`.
    /// Its bit `8k + i` is therefore exactly that matrix entry.
    const fn affine_blocks(cols: &[u128; 128]) -> [[u64; 16]; 16] {
        let mut blocks = [[0u64; 16]; 16];
        let mut k = 0;
        while k < 16 {
            let mut j = 0;
            while j < 16 {
                let mut quadword = 0u64;
                let mut i = 0;
                while i < 8 {
                    let mut row = 0u64;
                    let mut b = 0;
                    while b < 8 {
                        if (cols[8 * j + b] >> (8 * k + i)) & 1 == 1 {
                            row |= 1 << b;
                        }
                        b += 1;
                    }
                    quadword |= row << (8 * (7 - i));
                    i += 1;
                }
                blocks[k][j] = quadword;
                j += 1;
            }
            k += 1;
        }
        blocks
    }

    /// Transposes the bits of an 8 × 8 byte matrix packed row-major in a u64.
    ///
    /// The three exchanges swap the three-bit byte and bit indices. GFNI reads output rows in
    /// reverse order, so the caller reverses the resulting row bytes afterward.
    #[inline]
    const fn transpose8(mut value: u64) -> u64 {
        let mut exchange = (value ^ (value >> 7)) & 0x00aa_00aa_00aa_00aa;
        value ^= exchange ^ (exchange << 7);
        exchange = (value ^ (value >> 14)) & 0x0000_cccc_0000_cccc;
        value ^= exchange ^ (exchange << 14);
        exchange = (value ^ (value >> 28)) & 0x0000_0000_f0f0_f0f0;
        value ^ exchange ^ (exchange << 28)
    }

    /// Builds runtime GFNI blocks by gathering column bytes and transposing each 8 × 8 submatrix.
    ///
    /// `affine_blocks` above remains the static reference constructor. Keeping this candidate
    /// separate lets native tests compare every generated word against that independent oracle.
    #[inline]
    const fn affine_blocks_runtime(cols: &[u128; 128]) -> [[u64; 16]; 16] {
        let mut blocks = [[0u64; 16]; 16];
        let mut j = 0;
        while j < 16 {
            let c0 = cols[8 * j].to_le_bytes();
            let c1 = cols[8 * j + 1].to_le_bytes();
            let c2 = cols[8 * j + 2].to_le_bytes();
            let c3 = cols[8 * j + 3].to_le_bytes();
            let c4 = cols[8 * j + 4].to_le_bytes();
            let c5 = cols[8 * j + 5].to_le_bytes();
            let c6 = cols[8 * j + 6].to_le_bytes();
            let c7 = cols[8 * j + 7].to_le_bytes();
            let mut k = 0;
            while k < 16 {
                let packed =
                    u64::from_le_bytes([c0[k], c1[k], c2[k], c3[k], c4[k], c5[k], c6[k], c7[k]]);
                blocks[k][j] = transpose8(packed).swap_bytes();
                k += 1;
            }
            j += 1;
        }
        blocks
    }

    impl PreparedMap {
        /// Builds the GFNI blocks for one runtime coordinate map.
        #[inline]
        pub(super) const fn new(columns: [u128; 128]) -> Self {
            Self {
                blocks: affine_blocks_runtime(&columns),
                columns,
            }
        }
    }

    /// The blocks of `N`, the map out of the tower basis.
    static TOWER_TO_POLY: [[u64; 16]; 16] = affine_blocks(&COLUMNS_128);

    /// The blocks of `M = N⁻¹`, the map back into it.
    static POLY_TO_TOWER: [[u64; 16]; 16] = affine_blocks(&invert(&COLUMNS_128, 128));

    /// One butterfly round, pairing registers `STEP` apart at `8·STEP`-bit granularity.
    ///
    /// # Safety
    ///
    /// The caller must be compiled with `avx512f` and `avx512bw`.
    #[inline(always)]
    unsafe fn interleave<const STEP: usize>(a: &mut [__m512i; 16]) {
        let mut i = 0;
        while i < 16 {
            // Each pair is visited once, from its lower member.
            if i & STEP == 0 {
                let (x, y) = (a[i], a[i + STEP]);
                // SAFETY: guaranteed by the caller.
                let (lo, hi) = unsafe {
                    match STEP {
                        1 => (_mm512_unpacklo_epi8(x, y), _mm512_unpackhi_epi8(x, y)),
                        2 => (_mm512_unpacklo_epi16(x, y), _mm512_unpackhi_epi16(x, y)),
                        4 => (_mm512_unpacklo_epi32(x, y), _mm512_unpackhi_epi32(x, y)),
                        _ => (_mm512_unpacklo_epi64(x, y), _mm512_unpackhi_epi64(x, y)),
                    }
                };
                a[i] = lo;
                a[i + STEP] = hi;
            }
            i += 1;
        }
    }

    /// Transposes a `16 × 16` byte matrix inside each 128-bit lane of sixteen registers.
    ///
    /// Every `vpunpck` acts within a lane, so the four lanes transpose side by side.
    ///
    /// Being a transpose, this is its own inverse.
    ///
    /// # Safety
    ///
    /// The caller must be compiled with `avx512f` and `avx512bw`.
    #[inline(always)]
    unsafe fn transpose(a: [__m512i; 16]) -> [__m512i; 16] {
        let mut a = a;
        // SAFETY: guaranteed by the caller.
        unsafe {
            interleave::<1>(&mut a);
            interleave::<2>(&mut a);
            interleave::<4>(&mut a);
            interleave::<8>(&mut a);
        }
        core::array::from_fn(|j| a[BITREV4[j]])
    }

    /// One output byte plane: the sixteen affine images of the input planes, summed.
    ///
    /// # Safety
    ///
    /// The caller must be compiled with `gfni` and `avx512f`.
    #[inline(always)]
    unsafe fn plane(input: &[__m512i; 16], row: &[u64; 16]) -> __m512i {
        // SAFETY: guaranteed by the caller.
        unsafe {
            // The same block applies to every element in the plane, so it broadcasts.
            let t: [__m512i; 16] = core::array::from_fn(|j| {
                _mm512_gf2p8affine_epi64_epi8::<0>(input[j], _mm512_set1_epi64(row[j] as i64))
            });
            // A three-input exclusive-or halves the depth and the instruction count of the sum.
            let a = _mm512_ternarylogic_epi64::<XOR3>(t[0], t[1], t[2]);
            let b = _mm512_ternarylogic_epi64::<XOR3>(t[3], t[4], t[5]);
            let c = _mm512_ternarylogic_epi64::<XOR3>(t[6], t[7], t[8]);
            let d = _mm512_ternarylogic_epi64::<XOR3>(t[9], t[10], t[11]);
            let e = _mm512_ternarylogic_epi64::<XOR3>(t[12], t[13], t[14]);
            let left = _mm512_ternarylogic_epi64::<XOR3>(a, b, c);
            let right = _mm512_ternarylogic_epi64::<XOR3>(d, e, t[15]);
            _mm512_xor_si512(left, right)
        }
    }

    /// Applies one map to whole blocks of a slice, returning how many elements it covered.
    ///
    /// # Algorithm
    ///
    /// A block is transposed to byte planes, mapped, and transposed back.
    ///
    /// ```text
    ///     register r  holds elements 4r … 4r+3          the layout in memory
    ///     plane j     holds byte j of every element     what the affine map wants
    /// ```
    ///
    /// Within a plane every byte needs the same `8 × 8` block.
    /// That is the one thing `vgf2p8affineqb` does: its matrix operand is shared by a quadword.
    ///
    /// A whole block costs 256 affine maps and 128 exclusive-ors.
    ///
    /// The per-element route over the same 64 elements takes 1024 dependent table loads.
    ///
    /// Out of line so that the dispatch above stays small enough to inline into its callers.
    #[inline(never)]
    fn apply(blocks: &[[u64; 16]; 16], values: &mut [u128]) -> usize {
        // Splitting into fixed-size blocks hands the tail back and needs no bounds check.
        let (chunks, _) = values.as_chunks_mut::<BLOCK>();

        for chunk in chunks.iter_mut() {
            // SAFETY: a chunk holds one whole block, which is sixteen 512-bit registers.
            //
            // Every offset `4r` for `r < 16` therefore addresses 64 bytes inside it.
            //
            // Both accesses are the unaligned forms, so the alignment of the slice is free.
            //
            // The target features every intrinsic needs gate this module.
            unsafe {
                let raw: [__m512i; 16] =
                    core::array::from_fn(|r| _mm512_loadu_si512(chunk.as_ptr().add(4 * r).cast()));
                let input = transpose(raw);
                let output: [__m512i; 16] = core::array::from_fn(|k| plane(&input, &blocks[k]));
                let output = transpose(output);
                for (r, &value) in output.iter().enumerate() {
                    _mm512_storeu_si512(chunk.as_mut_ptr().add(4 * r).cast(), value);
                }
            }
        }
        chunks.len() * BLOCK
    }

    /// Applies a prepared map out of place to complete 64-element blocks.
    ///
    /// Each source block is loaded before that block's destination stores, and the caller handles
    /// the remainder.
    #[inline(never)]
    pub(super) fn apply_out_of_place(
        prepared: &PreparedMap,
        input: &[u128],
        output: &mut [u128],
    ) -> usize {
        assert_eq!(
            input.len(),
            output.len(),
            "input and output must have equal lengths"
        );
        let (input_chunks, _) = input.as_chunks::<BLOCK>();
        let (output_chunks, _) = output.as_chunks_mut::<BLOCK>();

        for (input_chunk, output_chunk) in input_chunks.iter().zip(output_chunks.iter_mut()) {
            // SAFETY: each chunk holds one whole block, which is sixteen 512-bit registers.
            // The unaligned forms accept every slice alignment, and this module's target gate
            // enables every intrinsic used by the transpose and affine operations.
            unsafe {
                let raw: [__m512i; 16] = core::array::from_fn(|r| {
                    _mm512_loadu_si512(input_chunk.as_ptr().add(4 * r).cast())
                });
                let input = transpose(raw);
                let mapped: [__m512i; 16] =
                    core::array::from_fn(|k| plane(&input, &prepared.blocks[k]));
                let mapped = transpose(mapped);
                for (r, &value) in mapped.iter().enumerate() {
                    _mm512_storeu_si512(output_chunk.as_mut_ptr().add(4 * r).cast(), value);
                }
            }
        }
        input_chunks.len() * BLOCK
    }

    /// Crosses whole blocks out of the tower basis, returning how many elements it covered.
    #[inline]
    pub(super) fn tower_to_poly(values: &mut [u128]) -> usize {
        apply(&TOWER_TO_POLY, values)
    }

    /// Crosses whole blocks back into the tower basis, returning how many elements it covered.
    #[inline]
    pub(super) fn poly_to_tower(values: &mut [u128]) -> usize {
        apply(&POLY_TO_TOWER, values)
    }

    #[cfg(test)]
    mod runtime_constructor_tests {
        use super::{affine_blocks, affine_blocks_runtime};

        fn next_word(state: &mut u128) -> u128 {
            *state ^= *state << 7;
            *state ^= *state >> 9;
            *state ^= *state << 8;
            *state
        }

        #[test]
        fn runtime_constructor_matches_reference_for_singletons_and_random_maps() {
            // Both constructors are GF(2)-linear in the column array: the static one is a bit
            // permutation of its input, and the runtime one is a byte gather, an
            // exclusive-or/shift/mask transpose and a byte reversal.
            // The 16,384 single-entry matrices below are a basis of the space of all 128 x 128
            // binary maps, so agreement on them proves agreement on every one of the 2^16384 maps.
            // The random maps that follow add nothing to that argument; they guard against an
            // edit that breaks the linearity it rests on.
            for column in 0..128 {
                for output_bit in 0..128 {
                    let mut columns = [0u128; 128];
                    columns[column] = 1u128 << output_bit;
                    assert_eq!(
                        affine_blocks_runtime(&columns),
                        affine_blocks(&columns),
                        "column {column}, output bit {output_bit}"
                    );
                }
            }

            let mut state = 0xC011_8A8E_51ED_5EED_1234_5678_9ABC_DEF0u128;
            for map in 0..8 {
                let columns = core::array::from_fn(|_| next_word(&mut state));
                assert_eq!(
                    affine_blocks_runtime(&columns),
                    affine_blocks(&columns),
                    "random map {map}"
                );
            }
        }
    }
}

/// No instruction set here converts more than one element at a time, so no prefix is blocked.
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
)))]
mod blocked {
    /// A block no slice can fill, so every dispatch below takes the per-element path outright.
    pub(super) const BLOCK: usize = usize::MAX;

    /// Reports that no element crossed out of the tower basis.
    #[inline]
    pub(super) const fn tower_to_poly(_values: &mut [u128]) -> usize {
        0
    }

    /// Reports that no element crossed back into the tower basis.
    #[inline]
    pub(super) const fn poly_to_tower(_values: &mut [u128]) -> usize {
        0
    }
}

/// A runtime map must cover enough entries to amortize preparing its 256 GFNI affine blocks.
pub(crate) const DYNAMIC_MAP_THRESHOLD: usize = 4096;

/// Tries to apply an arbitrary runtime 128 × 128 binary map with the prepared GFNI kernel.
///
/// The caller has already checked the concrete field types, while this layer owns the target
/// gate, matrix construction, blocked application, and scalar tail. A refusal leaves `output`
/// untouched so callers can use their existing portable map without a second allocation.
#[inline]
pub(crate) fn try_map_tower_coordinates_into(
    columns: &[u128; 128],
    input: &[u128],
    output: &mut [u128],
) -> bool {
    assert_eq!(
        input.len(),
        output.len(),
        "input and output must have equal lengths"
    );

    if input.len() < DYNAMIC_MAP_THRESHOLD {
        return false;
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    {
        let prepared = blocked::PreparedMap::new(*columns);
        let processed = blocked::apply_out_of_place(&prepared, input, output);
        for (source, destination) in input[processed..].iter().zip(&mut output[processed..]) {
            *destination = image(*source, 128, &prepared.columns);
        }
        true
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    )))]
    {
        let _ = (columns, input, output);
        false
    }
}

/// `GF(2^64)` from the tower basis to the polynomial basis.
#[inline]
pub(crate) fn tower_to_poly_64(v: u64) -> u64 {
    apply_64(&TOWER_TO_POLY_64, v)
}

/// `GF(2^64)` from the polynomial basis back to the tower basis.
#[inline]
pub(crate) fn poly_to_tower_64(v: u64) -> u64 {
    apply_64(&POLY_TO_TOWER_64, v)
}

/// Squaring in `GF(2^64)`, taking and returning the tower representation.
#[inline]
pub(super) fn tower_square_64(v: u64) -> u64 {
    apply_64(&SQUARE_64, v)
}

/// `GF(2^128)` from the tower basis to the polynomial basis.
#[inline]
pub(crate) fn tower_to_poly_128(v: u128) -> u128 {
    apply_128(&TOWER_TO_POLY_128, v)
}

/// `GF(2^128)` from the polynomial basis back to the tower basis.
#[inline]
pub(crate) fn poly_to_tower_128(v: u128) -> u128 {
    apply_128(&POLY_TO_TOWER_128, v)
}

/// `GF(2^128)` from the tower basis to the polynomial basis, over a whole slice.
///
/// Returns how many leading elements the blocked kernel took, which is zero wherever the target
/// has none. The whole slice is converted either way, so a caller can ignore the count; what it
/// buys is a test that can see which of the two paths ran.
#[inline]
pub(crate) fn tower_to_poly_128_slice(values: &mut [u128]) -> usize {
    // Below one block there is nothing to transpose, so the call boundary buys nothing.
    //
    // Otherwise the blocked kernel reports how many leading elements it covered.
    let blocked = if values.len() < blocked::BLOCK {
        0
    } else {
        blocked::tower_to_poly(values)
    };

    // Whatever is left over costs sixteen table lookups per element.
    for value in &mut values[blocked..] {
        *value = apply_128(&TOWER_TO_POLY_128, *value);
    }

    blocked
}

/// `GF(2^128)` from the polynomial basis back to the tower basis, over a whole slice.
///
/// Returns how many leading elements the blocked kernel took, which is zero wherever the target
/// has none. The whole slice is converted either way, so a caller can ignore the count; what it
/// buys is a test that can see which of the two paths ran.
#[inline]
pub(crate) fn poly_to_tower_128_slice(values: &mut [u128]) -> usize {
    // Below one block there is nothing to transpose, so the call boundary buys nothing.
    //
    // Otherwise the blocked kernel reports how many leading elements it covered.
    let blocked = if values.len() < blocked::BLOCK {
        0
    } else {
        blocked::poly_to_tower(values)
    };

    // Whatever is left over costs sixteen table lookups per element.
    for value in &mut values[blocked..] {
        *value = apply_128(&POLY_TO_TOWER_128, *value);
    }

    blocked
}

#[cfg(test)]
mod tests {
    extern crate std;

    use alloc::vec::Vec;
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    use std::hint::black_box;
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    use std::time::Instant;

    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;

    use super::*;
    use crate::tower::TowerLevel;
    use crate::{BinaryField64, BinaryField128};

    /// Whether this build is one the blocked kernel is compiled for.
    ///
    /// Stated a second time here, independently of the gate on the kernel itself.
    ///
    /// A gate that drifts then shows up as a failure rather than as a silent per-element pass.
    const BLOCKED_BUILD: bool = cfg!(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ));

    // The gate on the blocked kernel has to agree with the target predicate restated above.
    //
    // Drift in either direction leaves the sentinel width below behind.
    //
    // Checking the block here turns that into a build failure.
    //
    // Otherwise it waits for someone to run this leg.
    const _: () = assert!(if BLOCKED_BUILD {
        super::blocked::BLOCK == 64
    } else {
        super::blocked::BLOCK == usize::MAX
    });

    /// Guard elements on each side of the payload, two whole blocks of the widest kernel.
    ///
    /// One block wide would catch only a store that overran by exactly one block.
    ///
    /// Two make the next one out detectable as well, still inside the buffer's own allocation.
    const SENTINELS: usize = 2 * 64;

    /// What those elements hold, which no image of the inputs below reproduces.
    const SENTINEL: u128 = 0x5a5a_5a5a_5a5a_5a5a_5a5a_5a5a_5a5a_5a5a;

    /// The image of a bit pattern under the map whose columns these are.
    ///
    /// The map is `GF(2)`-linear, so the image is the sum of the columns the set bits select.
    /// This rests on no lookup table and on no vector instruction.
    ///
    /// That is what makes it a reference the kernels below can be wrong against.
    fn column_walk(cols: &[u128; 128], v: u128) -> u128 {
        (0..128)
            .filter(|i| (v >> i) & 1 == 1)
            .fold(0, |acc, i| acc ^ cols[i])
    }

    /// The columns of `M = N⁻¹`, which the reverse direction is checked against.
    fn inverse_columns() -> [u128; 128] {
        invert(&COLUMNS_128, 128)
    }

    #[test]
    fn byte_maps_match_scalar_evaluation() {
        // Invariant: the dispatched map is the matrix the columns describe, byte for byte.
        //
        // Every byte value in every position is swept.
        //
        // That is what pins the load byte order of each vector arm.
        let inverse = inverse_columns();
        for (tables, cols) in [
            (&TOWER_TO_POLY_128, &COLUMNS_128),
            (&POLY_TO_TOWER_128, &inverse),
        ] {
            for byte in 0..16 {
                for value in 0..256u128 {
                    // Every other byte is all ones, so no position can be silently dropped.
                    let input = !(255u128 << (8 * byte)) | (value << (8 * byte));

                    // The table fold, which is what the vector arms replace.
                    let folded = tables.iter().enumerate().fold(0, |acc, (i, table)| {
                        acc ^ table[(input >> (8 * i)) as u8 as usize]
                    });
                    assert_eq!(apply_128(tables, input), folded);

                    // And the same answer from the matrix alone.
                    assert_eq!(apply_128(tables, input), column_walk(cols, input));
                }
            }
        }
    }

    /// A slice whose elements share no structure with one another.
    fn sample(len: usize) -> Vec<u128> {
        (0..len)
            .map(|i| (i as u128 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15_f39c_c060_5ced_c835))
            .collect()
    }

    /// The same values, walled off from the rest of the allocation on both sides.
    ///
    /// ```text
    ///     [ guard | offset | values | guard ]
    ///                ^ slides the payload across every alignment one block can see
    /// ```
    fn padded(values: &[u128], offset: usize) -> Vec<u128> {
        let lead = SENTINELS + offset;

        let mut buffer = Vec::with_capacity(lead + values.len() + SENTINELS);
        buffer.extend(core::iter::repeat_n(SENTINEL, lead));
        buffer.extend_from_slice(values);
        buffer.extend(core::iter::repeat_n(SENTINEL, SENTINELS));
        buffer
    }

    /// Both directions over one length and one offset, against the column walk.
    fn slices_agree(values: &[u128], offset: usize) -> Result<(), TestCaseError> {
        let (len, start) = (values.len(), SENTINELS + offset);
        let inverse = inverse_columns();

        // Every whole block of the payload belongs to the blocked kernel, and the remainder to
        // the per-element tail. Where no kernel is compiled in, `BLOCK` is wider than any
        // slice, so the same expression asks for nothing.
        let covered = len - len % super::blocked::BLOCK;

        // Out of the tower basis, in place, inside its walls.
        let mut buffer = padded(values, offset);
        prop_assert_eq!(
            tower_to_poly_128_slice(&mut buffer[start..start + len]),
            covered
        );

        // Nothing may have run off either end of the payload.
        prop_assert!(buffer[..start].iter().all(|&v| v == SENTINEL));
        prop_assert!(buffer[start + len..].iter().all(|&v| v == SENTINEL));

        // Every element must be the image the matrix alone gives.
        let forward: Vec<u128> = values
            .iter()
            .map(|&v| column_walk(&COLUMNS_128, v))
            .collect();
        prop_assert_eq!(&buffer[start..start + len], &forward[..]);

        // And back again, which must restore the input.
        prop_assert_eq!(
            poly_to_tower_128_slice(&mut buffer[start..start + len]),
            covered
        );
        prop_assert!(buffer[..start].iter().all(|&v| v == SENTINEL));
        prop_assert!(buffer[start + len..].iter().all(|&v| v == SENTINEL));
        prop_assert_eq!(&buffer[start..start + len], values);

        // The reverse direction on its own, against its own column walk.
        let mut buffer = padded(values, offset);
        prop_assert_eq!(
            poly_to_tower_128_slice(&mut buffer[start..start + len]),
            covered
        );
        prop_assert!(buffer[..start].iter().all(|&v| v == SENTINEL));
        prop_assert!(buffer[start + len..].iter().all(|&v| v == SENTINEL));
        let backward: Vec<u128> = values.iter().map(|&v| column_walk(&inverse, v)).collect();
        prop_assert_eq!(&buffer[start..start + len], &backward[..]);

        Ok(())
    }

    #[test]
    fn the_blocked_prefix_and_the_per_element_tail_agree_at_every_length() {
        // An unbroken run of lengths covers every block count and every remainder.
        //
        //     len 0..63     below one block, so nothing is blocked
        //     len 64        exactly one block, with no tail
        //     len 65..127   one block plus a tail of every possible size
        //     len 128..200  several blocks, with and without a tail
        for len in 0..=200 {
            let values = sample(len);
            // Offsets 0 through 3 place the payload at every alignment one register can see.
            for offset in 0..4 {
                slices_agree(&values, offset)
                    .unwrap_or_else(|e| panic!("len {len}, offset {offset}: {e}"));
            }
        }
    }

    #[test]
    fn the_dispatchers_hand_every_whole_block_to_the_blocked_kernel() {
        // Invariant: a dispatcher reports how many leading elements the kernel took.
        //
        // It uses that count only as a tail offset, so a dispatch that stopped reaching the
        // kernel would still return right answers, silently and at the per-element rate.
        // Asking for the count back through the ordinary entry point is what makes that a
        // failure rather than a regression nobody sees.
        //
        // Whole blocks only: the remainder is the per-element tail's business. Where no kernel
        // is compiled in `BLOCK` is wider than any slice, so the same expression asks for
        // nothing, which is exactly that build's claim.
        let block = super::blocked::BLOCK;

        // Four blocks of the widest kernel, 64 elements each, and every remainder between.
        for len in 0..=256 {
            let want = len - len % block;

            let mut values = sample(len);
            assert_eq!(tower_to_poly_128_slice(&mut values), want, "len {len}");
            assert_eq!(poly_to_tower_128_slice(&mut values), want, "len {len}");
        }
    }

    /// The dynamic kernel reports exactly its complete-block prefix and leaves the caller's tail
    /// untouched for the independent scalar path.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    #[test]
    fn dynamic_prepared_map_reports_prefix_and_leaves_tail_untouched() {
        let columns = core::array::from_fn(|index| 1u128 << index);
        let input = sample(4097);
        let poison = SENTINEL;
        let mut output = alloc::vec![poison; input.len()];
        let prepared = blocked::PreparedMap::new(columns);

        let processed = blocked::apply_out_of_place(&prepared, &input, &mut output);

        assert_eq!(processed, 4096);
        assert_eq!(&output[..processed], &input[..processed]);
        assert!(output[processed..].iter().all(|&value| value == poison));
    }

    /// Diagnostic-only native benchmark for the runtime-map constructor and prepared kernel.
    ///
    /// This intentionally bypasses the production threshold so the constructor, apply-only,
    /// and combined costs can be compared at every candidate size. It is ignored because it is
    /// for the native GFNI host used to choose that threshold, not a correctness test.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    #[test]
    #[ignore = "native GFNI benchmark; run with --ignored --nocapture"]
    fn dynamic_prepared_map_forced_kernel_benchmark() {
        for len in [64, 256, 1024, 4096, 16384] {
            let input = sample(len);
            let mut output = alloc::vec![0u128; len];
            let mut construct = 0u128;
            let mut apply = 0u128;
            let mut combined = 0u128;

            for iteration in 0..7 {
                let started = Instant::now();
                let prepared = blocked::PreparedMap::new(black_box(COLUMNS_128));
                let construct_elapsed = started.elapsed().as_nanos();
                let started_apply = Instant::now();
                let processed = blocked::apply_out_of_place(
                    &prepared,
                    black_box(&input),
                    black_box(&mut output),
                );
                let apply_elapsed = started_apply.elapsed().as_nanos();

                let started_combined = Instant::now();
                let prepared = blocked::PreparedMap::new(black_box(COLUMNS_128));
                let combined_processed = blocked::apply_out_of_place(
                    &prepared,
                    black_box(&input),
                    black_box(&mut output),
                );
                let combined_elapsed = started_combined.elapsed().as_nanos();
                assert_eq!(processed, len - len % super::blocked::BLOCK);
                assert_eq!(combined_processed, processed);

                if iteration >= 2 {
                    construct += construct_elapsed;
                    apply += apply_elapsed;
                    combined += combined_elapsed;
                }
            }
            let samples = 5;
            std::println!(
                "dynamic_prepared_map len={len} construct_ns={} apply_ns={} construct_and_apply_ns={}",
                construct / samples,
                apply / samples,
                combined / samples
            );
        }
    }

    /// Squaring in `GF(2)[x]/(x^bits + tail)`.
    fn poly_square(a: u128, bits: usize, tail: u128) -> u128 {
        poly_mul(a, a, bits, tail)
    }

    /// `a^e` in `GF(2)[x]/(x^bits + tail)`.
    fn poly_pow(a: u128, mut e: u128, bits: usize, tail: u128) -> u128 {
        let mut acc = 1;
        let mut cur = a;
        while e != 0 {
            if e & 1 == 1 {
                acc = poly_mul(acc, cur, bits, tail);
            }
            cur = poly_square(cur, bits, tail);
            e >>= 1;
        }
        acc
    }

    /// `a⁻¹` in `GF(2)[x]/(x^bits + tail)`, by Fermat: `a^(2^bits − 2)`.
    fn poly_inverse(a: u128, bits: usize, tail: u128) -> u128 {
        assert_ne!(a, 0, "zero is not invertible");
        let order = if bits == 128 {
            u128::MAX
        } else {
            (1u128 << bits) - 1
        };
        poly_pow(a, order - 1, bits, tail)
    }

    /// The absolute trace `Tr(a) = a + a² + … + a^(2^(bits−1))`, an element of `GF(2)`.
    fn poly_trace(a: u128, bits: usize, tail: u128) -> u128 {
        let mut acc = 0;
        let mut cur = a;
        for _ in 0..bits {
            acc ^= cur;
            cur = poly_square(cur, bits, tail);
        }
        assert!(acc <= 1, "the trace must land in GF(2)");
        acc
    }

    /// The lowest power of `x` whose absolute trace is `1`.
    ///
    /// The trace is a surjective `GF(2)`-linear form, so it cannot vanish on the whole basis.
    fn trace_one_element(bits: usize, tail: u128) -> u128 {
        (0..bits)
            .map(|i| 1u128 << i)
            .find(|&t| poly_trace(t, bits, tail) == 1)
            .expect("the trace form cannot vanish on every basis element")
    }

    /// The `u` with `u² + u = c`, which exists exactly when `Tr(c) = 0`.
    ///
    /// With `s_i = Σ_{j ≤ i} t^(2^j)` for any `t` of trace `1`, the sum `u = Σ_i s_i·c^(2^i)`
    /// telescopes under squaring to `u² + u = s_0·c + s_bits·c = t·c + (1 + t)·c = c`.
    fn solve_quadratic(c: u128, bits: usize, tail: u128) -> u128 {
        assert_eq!(poly_trace(c, bits, tail), 0, "u² + u = c has no solution");
        let mut acc = 0;
        let mut partial_trace = 0;
        let mut t_power = trace_one_element(bits, tail);
        let mut c_power = c;
        for _ in 0..bits {
            partial_trace ^= t_power;
            acc ^= poly_mul(partial_trace, c_power, bits, tail);
            t_power = poly_square(t_power, bits, tail);
            c_power = poly_square(c_power, bits, tail);
        }
        assert_eq!(
            poly_square(acc, bits, tail) ^ acc,
            c,
            "quadratic solve failed"
        );
        acc
    }

    /// Rederives the images of the tower generators from the defining relations alone.
    ///
    /// Substituting `ξ_k = ξ_{k−1}·u` into `ξ_k² + ξ_{k−1}·ξ_k + 1 = 0` and dividing by
    /// `ξ_{k−1}²` turns it into `u² + u = ξ_{k−1}⁻²`, which the closed form above solves.
    fn derive_generator_images(bits: usize, tail: u128, levels: usize) -> Vec<u128> {
        let mut images = Vec::with_capacity(levels);
        // `X_{−1} = 1`.
        let mut previous = 1u128;
        for _ in 0..levels {
            let c = poly_inverse(poly_square(previous, bits, tail), bits, tail);
            let u = solve_quadratic(c, bits, tail);
            previous = poly_mul(previous, u, bits, tail);
            images.push(previous);
        }
        images
    }

    #[test]
    fn derivation_reproduces_the_generator_images() {
        assert_eq!(derive_generator_images(32, TAIL_32, 5), XI_32.to_vec());
        assert_eq!(derive_generator_images(64, TAIL_64, 6), XI_64.to_vec());
        assert_eq!(derive_generator_images(128, TAIL_128, 7), XI_128.to_vec());
    }

    /// The whole construction rests on these: any images satisfying them induce a `GF(2)`-algebra
    /// homomorphism out of the tower, and the invertibility of `N` upgrades it to an isomorphism.
    #[test]
    fn generator_images_satisfy_the_tower_relations() {
        for (bits, tail, xi) in [
            (64, TAIL_64, XI_64.as_slice()),
            (128, TAIL_128, XI_128.as_slice()),
        ] {
            // `X_{−1} = 1`.
            let mut previous = 1u128;
            for (k, &x) in xi.iter().enumerate() {
                let relation =
                    poly_square(x, bits, tail) ^ poly_mul(previous, x, bits, tail) ^ 1u128;
                assert_eq!(
                    relation, 0,
                    "X_{k}² + X_{{k−1}}·X_{k} + 1 ≠ 0 at {bits} bits"
                );
                previous = x;
            }
        }
    }

    /// The tables must send the tower's distinguished elements where the construction says.
    #[test]
    fn the_change_of_basis_maps_the_generators_to_their_images() {
        assert_eq!(tower_to_poly_64(1), 1);
        assert_eq!(tower_to_poly_128(1), 1);
        for (k, &x) in XI_64.iter().enumerate() {
            assert_eq!(u128::from(tower_to_poly_64(1u64 << (1 << k))), x);
        }
        for (k, &x) in XI_128.iter().enumerate() {
            assert_eq!(tower_to_poly_128(1u128 << (1 << k)), x);
        }
    }

    /// A hand-checkable case: `X_0` squares to `X_0 + 1` in the tower, so its image must satisfy
    /// `ξ_0² = ξ_0 + 1` in the polynomial basis — the two differ in the constant term alone.
    #[test]
    fn squaring_the_first_generator_flips_one_bit_of_its_image() {
        let image = tower_to_poly_128(0b10);
        assert_eq!(image, XI_128[0]);
        assert_eq!(poly_square(image, 128, TAIL_128), image ^ 1);
        assert_eq!(poly_to_tower_128(image ^ 1), 0b11);
        assert_eq!(
            BinaryField128::from_repr(0b10)
                .reference_mul(BinaryField128::from_repr(0b10))
                .to_repr(),
            0b11
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        /// `M` and `N` are mutually inverse, so the representations carry the same information.
        #[test]
        fn the_change_of_basis_round_trips(a: u64, b: u128) {
            prop_assert_eq!(poly_to_tower_64(tower_to_poly_64(a)), a);
            prop_assert_eq!(tower_to_poly_64(poly_to_tower_64(a)), a);
            prop_assert_eq!(poly_to_tower_128(tower_to_poly_128(b)), b);
            prop_assert_eq!(tower_to_poly_128(poly_to_tower_128(b)), b);
        }

        /// `N` carries the tower product to the polynomial-basis product. This is the property
        /// the fast path leans on, checked here against the bit-serial [`poly_mul`] alone so that
        /// it holds independently of any hardware instruction.
        #[test]
        fn the_change_of_basis_is_a_ring_isomorphism(a: u128, b: u128) {
            let product_64 = BinaryField64::from_repr(a as u64)
                .reference_mul(BinaryField64::from_repr(b as u64))
                .to_repr();
            prop_assert_eq!(
                poly_mul(
                    u128::from(tower_to_poly_64(a as u64)),
                    u128::from(tower_to_poly_64(b as u64)),
                    64,
                    TAIL_64,
                ),
                u128::from(tower_to_poly_64(product_64)),
            );

            let product_128 = BinaryField128::from_repr(a)
                .reference_mul(BinaryField128::from_repr(b))
                .to_repr();
            prop_assert_eq!(
                poly_mul(tower_to_poly_128(a), tower_to_poly_128(b), 128, TAIL_128),
                tower_to_poly_128(product_128),
            );
        }

        /// The slice kernels agree with the matrix, at lengths spanning several blocks.
        #[test]
        fn the_slice_kernels_agree_with_the_matrix(
            values in prop::collection::vec(any::<u128>(), 0..300),
            offset in 0usize..4,
        ) {
            slices_agree(&values, offset)?;
        }

        /// A slice crosses over exactly as its elements do one at a time.
        #[test]
        fn the_slice_kernels_agree_with_the_per_element_maps(
            values in prop::collection::vec(any::<u128>(), 0..300),
        ) {
            let mut forward = values.clone();
            tower_to_poly_128_slice(&mut forward);
            let want: Vec<u128> = values.iter().map(|&v| tower_to_poly_128(v)).collect();
            prop_assert_eq!(&forward, &want);

            let mut backward = values.clone();
            poly_to_tower_128_slice(&mut backward);
            let want: Vec<u128> = values.iter().map(|&v| poly_to_tower_128(v)).collect();
            prop_assert_eq!(&backward, &want);
        }

        /// The additive half of the isomorphism.
        #[test]
        fn the_change_of_basis_is_additive(a: u128, b: u128) {
            prop_assert_eq!(
                tower_to_poly_128(a ^ b),
                tower_to_poly_128(a) ^ tower_to_poly_128(b),
            );
            prop_assert_eq!(
                tower_to_poly_64(a as u64 ^ b as u64),
                tower_to_poly_64(a as u64) ^ tower_to_poly_64(b as u64),
            );
        }
    }

    /// Products computed by an independent implementation of the Wiedemann tower, as a check on
    /// the reference multiplication the whole verification ladder is anchored to.
    #[test]
    fn reference_multiplication_matches_independent_vectors() {
        const VECTORS_64: [(u64, u64, u64); 4] = [
            (
                0xf2ed_66ff_dcc9_9396,
                0x21ba_de02_6a6a_e768,
                0x4948_cf04_a001_e0dd,
            ),
            (
                0x9dd8_904f_0748_9671,
                0x6102_dd70_63e8_540e,
                0xd4da_4e01_a290_3ef6,
            ),
            (
                0x466d_e486_522c_4f8d,
                0x83fa_ac57_2f56_4652,
                0xd2e1_b9fb_8a75_b624,
            ),
            (
                0x0620_f087_7e5f_e381,
                0x781b_9a43_d04c_e50b,
                0x4dfa_5e46_61fb_eb68,
            ),
        ];
        const VECTORS_128: [(u128, u128, u128); 4] = [
            (
                0x21ba_de02_6a6a_e768_f2ed_66ff_dcc9_9396,
                0x6102_dd70_63e8_540e_9dd8_904f_0748_9671,
                0xe979_0238_73d0_74c5_03a0_3109_a53d_e616,
            ),
            (
                0x83fa_ac57_2f56_4652_466d_e486_522c_4f8d,
                0x781b_9a43_d04c_e50b_0620_f087_7e5f_e381,
                0xa67d_71dc_b60e_3b54_ce1c_34e0_e36d_c1c2,
            ),
            (
                0xc35d_7d3b_92e4_016e_27e4_7ffc_284a_2d4f,
                0x06e7_df8e_1eb1_c66e_79f7_4d60_ac03_031e,
                0x7f0d_3f85_5277_2bd0_ff4f_ffdd_83bf_f1c5,
            ),
            (
                0x2e09_e4b8_245e_debc_817a_f708_2074_73b7,
                0xb7c0_3984_2be3_8ecc_1f07_a223_563e_bc38,
                0x88b0_9d4e_93c3_1775_be08_2cb4_3a09_1df5,
            ),
        ];

        for (a, b, want) in VECTORS_64 {
            let got = BinaryField64::from_repr(a).reference_mul(BinaryField64::from_repr(b));
            assert_eq!(got.to_repr(), want, "GF(2^64): {a:#x} * {b:#x}");
        }
        for (a, b, want) in VECTORS_128 {
            let got = BinaryField128::from_repr(a).reference_mul(BinaryField128::from_repr(b));
            assert_eq!(got.to_repr(), want, "GF(2^128): {a:#x} * {b:#x}");
        }

        // The top generator satisfies `X² = αX + 1`, and `α·X` is `X²` again for `α = X_{k−1}`.
        assert_eq!(
            BinaryField128::from_repr(1 << 64).square().to_repr(),
            0x0000_0001_0000_0000_0000_0000_0000_0001,
        );
        assert_eq!(
            BinaryField128::from_repr(1 << 64).mul_alpha().to_repr(),
            0x0000_0001_0000_0000_0000_0000_0000_0001,
        );
    }
}
