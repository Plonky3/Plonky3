//! The univariate skip's domain: the AES byte field inside the committed field (ePrint 2026/1329, Section 4.2).
//!
//! The skip domain `S` is `phi8(0..64)` and the round-1 coset `Lambda` is `phi8(64..128)`.
//!
//! Both lie in a seven-dimensional subspace, so every node is a byte and every Lagrange denominator is a constant.

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_dft::{BasisNtt, RijndaelLde};
use p3_binary_field::{Poly64, Poly192, Rijndael8b};
use p3_field::{Algebra, Field, PrimeCharacteristicRing};

/// Variables the univariate skip binds at once: the bit inside a packed word.
pub(crate) const K_SKIP: usize = 6;

/// Points of the skip domain, and values of the round-1 message.
pub(crate) const SKIP_SIZE: usize = 1 << K_SKIP;

/// The first point of `Lambda`, as a byte.
const LAMBDA_START: u8 = 1 << K_SKIP;

/// `phi8` of the AES polynomial basis `x^k`, in the committed field's polynomial basis.
const PHI8_BASIS: [u64; 8] = [
    0x0000_0000_0000_0001,
    0x033c_e8be_ddc8_a656,
    0x5126_2037_5ed2_a108,
    0x0c9e_6360_90aa_fc01,
    0xba4f_3cd8_2801_769c,
    0xba26_e790_4adb_4a47,
    0x4676_9859_8926_dc01,
    0x4418_ae80_8b28_bdd0,
];

/// `phi8` of every byte, as the XOR of its set bits' images.
static PHI8: [Poly64; 256] = {
    let mut table = [Poly64::new(0); 256];
    let mut v = 1;
    while v < 256 {
        let mut acc = 0;
        let mut bit = 0;
        while bit < 8 {
            if v >> bit & 1 == 1 {
                acc ^= PHI8_BASIS[bit];
            }
            bit += 1;
        }
        table[v] = Poly64::new(acc);
        v += 1;
    }
    table
};

/// The embedding of the AES field into the committed field.
#[inline]
pub(crate) fn phi8(byte: u8) -> Poly64 {
    PHI8[usize::from(byte)]
}

/// `1 / prod_{w != 0} w` over the subspace `phi8(0..n)`.
///
/// It is every node's Lagrange denominator, since `u + v` runs over the nonzero points as `v` does.
fn subspace_denominator(n: usize) -> Poly64 {
    PHI8[1..n]
        .iter()
        .fold(Poly64::ONE, |acc, &w| acc * w)
        .inverse()
}

/// The Lagrange basis of the subspace `phi8(0..n)` at `z`, by prefix and suffix products.
///
/// No inverse of `z + node` is taken, so `z` may be a node.
pub(crate) fn lagrange_weights<E: Algebra<Poly192>>(z: &E, n: usize) -> Vec<E> {
    assert!(n.is_power_of_two() && n <= 128, "a subspace of the window");
    let d = Poly192::from(subspace_denominator(n));
    // Prefix pass: weight `i` holds the product over the nodes before it.
    let mut weights = vec![E::from(d); n];
    for i in 1..n {
        weights[i] = weights[i - 1].dup() * (z.dup() + Poly192::from(PHI8[i - 1]));
    }
    // Suffix pass: multiply in the product over the nodes after it.
    let mut suffix = E::ONE;
    for i in (0..n).rev() {
        weights[i] *= suffix.dup();
        suffix *= z.dup() + Poly192::from(PHI8[i]);
    }
    weights
}

/// `sum_j L_j(z) values_j`: the interpolant of values on the subspace `phi8(0..n)` at `z`.
///
/// # Panics
///
/// - There is not a power of two of at most 128 values.
pub(crate) fn interpolate<E: Algebra<Poly192>>(z: &E, values: &[E]) -> E {
    lagrange_weights(z, values.len())
        .into_iter()
        .zip(values)
        .map(|(w, v)| w * v.dup())
        .sum()
}

/// The round-1 message at `z`: known on `Lambda`, zero on `S`, so one polynomial on the 128-point window.
pub(crate) fn round1_at<E: Algebra<Poly192>>(z: &E, message: &[E]) -> E {
    assert_eq!(message.len(), SKIP_SIZE, "one value per coset point");
    let window = lagrange_weights(z, 2 * SKIP_SIZE);
    window[SKIP_SIZE..]
        .iter()
        .zip(message)
        .map(|(w, v)| w.dup() * v.dup())
        .sum()
}

/// The extension from `S` to `Lambda`, as one byte table and as two lifted transforms.
#[derive(Clone, Debug)]
pub(crate) struct SkipLde {
    /// Packed Boolean rows to their AES-field values on `Lambda`.
    pub(crate) bytes: RijndaelLde,
    /// Interpolation on `S`, lifted through `phi8`.
    source: BasisNtt<Poly64>,
    /// Evaluation on `Lambda`, lifted through `phi8`.
    target: BasisNtt<Poly64>,
}

impl SkipLde {
    /// The extension of the protocol's skip.
    pub(crate) fn new() -> Self {
        let bytes = RijndaelLde::new(
            K_SKIP,
            Rijndael8b::ZERO,
            Rijndael8b::from_byte(LAMBDA_START),
        );
        let embed = |x: Rijndael8b| phi8(x.to_byte());
        Self {
            source: bytes.source().map(embed),
            target: bytes.target().map(embed),
            bytes,
        }
    }

    /// Extend challenge-field values from `S` to `Lambda`, in place.
    pub(crate) fn extend(&self, values: &mut [Poly192]) {
        assert_eq!(values.len(), SKIP_SIZE, "one value per skip point");
        self.source.transform_algebra::<Poly192, true>(values, 1);
        self.target.transform_algebra::<Poly192, false>(values, 1);
    }
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    #[test]
    fn phi8_is_a_field_embedding() {
        // Invariant: phi8 is additive and multiplicative on every pair of bytes.
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                let (x, y) = (Rijndael8b::from_byte(a), Rijndael8b::from_byte(b));
                assert_eq!(phi8((x * y).to_byte()), phi8(a) * phi8(b), "{a} * {b}");
                assert_eq!(phi8(a ^ b), phi8(a) + phi8(b), "{a} + {b}");
            }
        }
    }

    #[test]
    fn lagrange_weights_interpolate_every_node() {
        // Invariant: at a node the weights are its indicator; elsewhere they sum to one.
        let mut rng = SmallRng::seed_from_u64(0x1A6);
        for n in [1, 2, 8, 64, 128] {
            for (i, &node) in PHI8[..n].iter().enumerate() {
                let w = lagrange_weights(&Poly192::from(node), n);
                for (j, &wj) in w.iter().enumerate() {
                    assert_eq!(
                        wj,
                        Poly192::from_bool(i == j),
                        "n={n}, node {i}, weight {j}"
                    );
                }
            }
            let z: Poly192 = rng.random();
            let total: Poly192 = lagrange_weights(&z, n).into_iter().sum();
            assert_eq!(total, Poly192::ONE, "n={n}");
        }
    }

    #[test]
    fn round1_at_is_the_window_interpolant() {
        // Invariant: a degree-127 polynomial vanishing on S is recovered from its values on Lambda.
        let mut rng = SmallRng::seed_from_u64(0x1A7);
        // P(X) = V_S(X) Q(X), Q of degree 63.
        let q: Vec<Poly192> = (0..SKIP_SIZE).map(|_| rng.random()).collect();
        let p = |x: Poly192| {
            let v: Poly192 = PHI8[..SKIP_SIZE].iter().map(|&s| x + s).product();
            v * q.iter().rev().fold(Poly192::ZERO, |acc, &c| acc * x + c)
        };
        let message: Vec<Poly192> = (0..SKIP_SIZE)
            .map(|i| p(Poly192::from(PHI8[SKIP_SIZE + i])))
            .collect();
        let z: Poly192 = rng.random();
        assert_eq!(round1_at(&z, &message), p(z));
    }

    #[test]
    fn the_lifted_extension_is_lagrange_interpolation() {
        // Invariant: the lifted transforms extend any challenge-field values from S to Lambda.
        let mut rng = SmallRng::seed_from_u64(0x1A8);
        let lde = SkipLde::new();
        let on_s: Vec<Poly192> = (0..SKIP_SIZE).map(|_| rng.random()).collect();
        let mut extended = on_s.clone();
        lde.extend(&mut extended);
        for (i, &e) in extended.iter().enumerate() {
            let w = lagrange_weights(&Poly192::from(PHI8[SKIP_SIZE + i]), SKIP_SIZE);
            let want: Poly192 = w.iter().zip(&on_s).map(|(&w, &v)| w * v).sum();
            assert_eq!(e, want, "point {i}");
        }
    }

    #[test]
    fn the_byte_table_extends_boolean_rows() {
        // Invariant: the byte table is the lifted extension on Boolean rows.
        let mut rng = SmallRng::seed_from_u64(0x1A9);
        let lde = SkipLde::new();
        for _ in 0..16 {
            let row: u64 = rng.random();
            let mut out = [Rijndael8b::ZERO; SKIP_SIZE];
            lde.bytes.apply(&row.to_le_bytes(), &mut out);
            let mut lifted: Vec<Poly192> = (0..SKIP_SIZE)
                .map(|s| Poly192::from_bool(row >> s & 1 == 1))
                .collect();
            lde.extend(&mut lifted);
            for (i, (&o, &l)) in out.iter().zip(&lifted).enumerate() {
                assert_eq!(Poly192::from(phi8(o.to_byte())), l, "point {i}");
            }
        }
    }
}
