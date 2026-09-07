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
///
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
    let mut acc = 0;
    let mut i = 0;
    while i < 128 {
        if (v >> i) & 1 == 1 {
            acc ^= COLUMNS_128[i];
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
    let mut acc = 0;
    for (byte, table) in tables.iter().enumerate() {
        acc ^= table[(v >> (8 * byte)) as u8 as usize];
    }
    acc
}

/// `GF(2^64)` from the tower basis to the polynomial basis.
#[inline]
pub(super) fn tower_to_poly_64(v: u64) -> u64 {
    apply_64(&TOWER_TO_POLY_64, v)
}

/// `GF(2^64)` from the polynomial basis back to the tower basis.
#[inline]
pub(super) fn poly_to_tower_64(v: u64) -> u64 {
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

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;

    use super::*;
    use crate::tower::TowerLevel;
    use crate::{BinaryField64, BinaryField128};

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
