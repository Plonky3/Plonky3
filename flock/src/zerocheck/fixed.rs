//! The seven eq coordinates the protocol fixes past the skip, chosen so their eq weights are cheap.
//!
//! - Three small ones, `phi8(x^(2^i) / (1 + x^(2^i)))`, make `eq_small[k] = C_s * phi8(x^k)`.
//! - Four medium ones, `g^(2^i) / (1 + g^(2^i))`, make `eq_med[b] = g^b / D`: geometric pinned weights.
//!
//! See ePrint 2026/1329, Section 4.3.
//!
//! Soundness needs their `2^7` eq weights to be linearly independent over `GF(2)`.

use alloc::vec::Vec;

use p3_binary_field::{Poly64, Poly192};
use p3_field::{Algebra, Dup, Field, PrimeCharacteristicRing};

use crate::skip::phi8;

/// Small fixed coordinates.
pub(crate) const N_SMALL: usize = 3;

/// Medium fixed coordinates.
pub(crate) const N_MEDIUM: usize = 4;

/// Fixed coordinates past the skip.
pub(crate) const N_FIXED: usize = N_SMALL + N_MEDIUM;

/// Medium positions of a window.
pub(crate) const MEDIUM_VALUES: usize = 1 << N_MEDIUM;

/// The small coordinates as AES bytes.
const SMALL_BYTES: [u8; N_SMALL] = [0xF7, 0x53, 0xB5];

/// `C_s` as an AES byte: the small eq weights are `C_s` times the powers of `x`.
const C_S_BYTE: u8 = 0x1C;

/// The medium generator `g`, a fixed element of the challenge field.
const GENERATOR: [u64; 3] = [
    0x243f_6a88_85a3_08d3,
    0x1319_8a2e_0370_7344,
    0xa409_3822_299f_31d0,
];

/// `C_s`, the factor the round-1 sweep leaves out.
pub(crate) fn c_s() -> Poly192 {
    Poly192::from(phi8(C_S_BYTE))
}

/// `g, g^2, g^4, g^8`.
fn medium_squares() -> [Poly192; N_MEDIUM] {
    let mut g = [Poly192::new(GENERATOR.map(Poly64::new)); N_MEDIUM];
    for i in 1..N_MEDIUM {
        g[i] = g[i - 1].square();
    }
    g
}

/// The seven fixed coordinates, small first.
pub(crate) fn fixed_challenges() -> [Poly192; N_FIXED] {
    let small = SMALL_BYTES.map(|b| Poly192::from(phi8(b)));
    let medium = medium_squares().map(|g| g * (Poly192::ONE + g).inverse());
    core::array::from_fn(|i| {
        if i < N_SMALL {
            small[i]
        } else {
            medium[i - N_SMALL]
        }
    })
}

/// `1 / D`, `D = prod_i (1 + g^(2^i))`, which the medium eq weights are divided by.
pub(crate) fn d_inv() -> Poly192 {
    medium_squares()
        .into_iter()
        .map(|g| Poly192::ONE + g)
        .product::<Poly192>()
        .inverse()
}

/// `g^b` for every medium position `b`.
pub(crate) fn gamma_powers() -> [Poly192; MEDIUM_VALUES] {
    let g = Poly192::new(GENERATOR.map(Poly64::new));
    let mut pow = [Poly192::ONE; MEDIUM_VALUES];
    for b in 1..MEDIUM_VALUES {
        pow[b] = pow[b - 1] * g;
    }
    pow
}

/// The eq coordinates past the skip, low variable first: the fixed ones, then the drawn ones.
pub(crate) fn eq_coordinates<E: Algebra<Poly192>>(outer: &[E]) -> Vec<E> {
    fixed_challenges()
        .into_iter()
        .map(E::from)
        .chain(outer.iter().map(Dup::dup))
        .collect()
}

#[cfg(test)]
mod tests {
    use p3_binary_field::Rijndael8b;
    use p3_field::BasedVectorSpace;
    use p3_multilinear_util::point::Point;

    use super::*;

    #[test]
    fn the_fixed_eq_weights_factor_as_designed() {
        // Invariant: eq over the fixed coordinates is C_s x^k times g^b / D.
        let eq = Point::new(fixed_challenges().to_vec()).equality_weights_lsb();
        let x = Rijndael8b::from_byte(2);
        for (i, &e) in eq.iter().enumerate() {
            let (k, b) = (i & 7, i >> 3);
            let xk = (0..k).fold(Rijndael8b::ONE, |acc, _| acc * x);
            let want = c_s() * Poly192::from(phi8(xk.to_byte())) * gamma_powers()[b] * d_inv();
            assert_eq!(e, want, "corner {i}");
        }
    }

    #[test]
    fn the_fixed_eq_weights_are_independent() {
        // Invariant: the 2^7 eq weights have rank 128 over GF(2), which soundness needs.
        //
        // Rank 7 of the coordinates alone would pass while a relation among their products broke it.
        let mut rows: Vec<[u64; 3]> = Point::new(fixed_challenges().to_vec())
            .equality_weights_lsb()
            .iter()
            .map(|w| {
                let c: &[Poly64] = w.as_basis_coefficients_slice();
                [c[0].to_bits(), c[1].to_bits(), c[2].to_bits()]
            })
            .collect();

        // Row-reduce over GF(2), one pivot per column.
        let mut rank = 0;
        for col in 0..192 {
            let (limb, mask) = (col / 64, 1u64 << (col % 64));
            let Some(p) = (rank..rows.len()).find(|&i| rows[i][limb] & mask != 0) else {
                continue;
            };
            rows.swap(rank, p);
            let pivot = rows[rank];
            for (i, row) in rows.iter_mut().enumerate() {
                if i != rank && row[limb] & mask != 0 {
                    for (x, y) in row.iter_mut().zip(pivot) {
                        *x ^= y;
                    }
                }
            }
            rank += 1;
        }
        assert_eq!(
            rank,
            1 << N_FIXED,
            "the fixed eq weights must be independent"
        );
    }
}
