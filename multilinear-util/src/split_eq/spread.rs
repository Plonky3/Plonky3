//! Inner products of extension weights against extension values, with no transpose.
//!
//! An extension element is `D` base coordinates over a basis `beta`:
//!
//! ```text
//!     w = sum_j w_j * beta_j        f = sum_k f_k * beta_k
//!     w * f = sum_{j, k} (w_j * f_k) * beta_j * beta_k
//! ```
//!
//! So a whole inner product needs only the `D * D` base sums
//!
//! ```text
//!     S_jk = sum_i w_j(i) * f_k(i),
//! ```
//!
//! and one small recombination at the end:
//!
//! ```text
//!     sum_i w(i) * f(i) = sum_j beta_j * (sum_k S_jk * beta_k)
//! ```
//!
//! The values stay where they are, `W / D` extension elements per packed base vector.
//! The weights are spread once to match that layout: coordinate `j` of an element repeats
//! across the `D` lanes the element fills.
//!
//! ```text
//!     W = 8, D = 4:
//!
//!     values     [ f0_0 f0_1 f0_2 f0_3 | f1_0 f1_1 f1_2 f1_3 ]
//!     spread_j   [ w0_j w0_j w0_j w0_j | w1_j w1_j w1_j w1_j ]
//!     lane t of spread_j * values  =  w_j * f_{t mod D}
//! ```
//!
//! Each `S_jk` is then a plain packed base dot product, reduced once per block.
//! Packing the values instead would transpose every one of them before it is multiplied.

use alloc::vec::Vec;

use p3_field::{ExtensionField, Field, PackedValue, PrimeCharacteristicRing};

/// Packed vectors each unreduced block of a spread dot product accumulates.
///
/// Eight terms keep the block inside the widest unrolled base dot product a packing ships.
const BLOCK: usize = 8;

/// Largest extension degree the spread kernel keeps accumulators for on the stack.
const MAX_DIMENSION: usize = 8;

/// Extension weights spread across base lanes, ready to dot against extension values.
///
/// The spread vectors are stored block by block, so one block's `D` weight rows are adjacent:
///
/// ```text
///     spread[(b * D + j) * BLOCK + u]   =   spread_j of packed group b * BLOCK + u
/// ```
#[derive(Debug, Clone)]
pub(super) struct SpreadWeights<F: Field, EF> {
    /// Spread coordinate vectors, `D` rows of `BLOCK` per block.
    spread: Vec<F::Packing>,
    /// The extension basis `beta_0, ..., beta_{D-1}` the recombination multiplies by.
    basis: Vec<EF>,
}

impl<F: Field, EF: ExtensionField<F>> SpreadWeights<F, EF> {
    /// Spreads a weight table, or returns `None` when its shape does not fit the kernel.
    ///
    /// The kernel needs:
    /// - whole extension elements per vector: `W % D == 0`,
    /// - whole blocks of vectors: `(len * D) % (W * BLOCK) == 0`,
    /// - stack accumulators: `D <= MAX_DIMENSION`.
    pub(super) fn new(weights: &[EF]) -> Option<Self> {
        let width = F::Packing::WIDTH;
        let dim = EF::DIMENSION;
        let per_vector = width / dim;
        let fits = width.is_multiple_of(dim)
            && dim <= MAX_DIMENSION
            && (weights.len() * dim).is_multiple_of(width * BLOCK);
        if !fits {
            return None;
        }

        // One block covers BLOCK vectors, each holding `per_vector` weights.
        let block_weights = BLOCK * per_vector;
        let mut spread = Vec::with_capacity(weights.len() * dim * dim / width);
        for block in weights.chunks_exact(block_weights) {
            for j in 0..dim {
                for group in block.chunks_exact(per_vector) {
                    // Lane t reads coordinate j of the element that fills lane t.
                    spread.push(F::Packing::from_fn(|lane| {
                        group[lane / dim].as_basis_coefficients_slice()[j]
                    }));
                }
            }
        }

        let basis = (0..dim)
            .map(|j| EF::ith_basis_element(j).expect("j < DIMENSION"))
            .collect();
        Some(Self { spread, basis })
    }

    /// Inner product of the spread weights against extension values.
    ///
    /// ```text
    ///     sum_i weights[i] * values[i]
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `values` is not as long as the weight table this was built from.
    pub(super) fn dot(&self, values: &[EF]) -> EF {
        let width = F::Packing::WIDTH;
        let dim = EF::DIMENSION;
        let per_vector = width / dim;
        assert_eq!(values.len() * dim * dim, self.spread.len() * width);

        // Invariant: lane t of acc[j] holds a partial S_{j, t mod D}.
        let mut acc = [F::Packing::ZERO; MAX_DIMENSION];
        let block_values = BLOCK * per_vector;
        for (block, rows) in values
            .chunks_exact(block_values)
            .zip(self.spread.chunks_exact(dim * BLOCK))
        {
            // The values are read in place, D coordinates at a time.
            let packed: [F::Packing; BLOCK] = core::array::from_fn(|u| {
                F::Packing::from_fn(|lane| {
                    block[u * per_vector + lane / dim].as_basis_coefficients_slice()[lane % dim]
                })
            });
            for (acc, row) in acc.iter_mut().zip(rows.as_chunks::<BLOCK>().0) {
                *acc += F::Packing::dot_product::<BLOCK>(row, &packed);
            }
        }

        // Fold the lanes into S_jk, then recombine over the basis.
        (0..dim)
            .map(|j| {
                let lanes = acc[j].as_slice();
                let row = EF::from_basis_coefficients_fn(|k| {
                    lanes.iter().skip(k).step_by(dim).copied().sum()
                });
                row * self.basis[j]
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::dot_product;
    use p3_field::extension::BinomialExtensionField;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    proptest! {
        #[test]
        fn spread_dot_matches_the_extension_dot(log_len in 0usize..=10, seed in any::<u64>()) {
            // Invariant: whenever the shape fits, the spread kernel equals the plain dot product.
            let mut rng = SmallRng::seed_from_u64(seed);
            let len = 1 << log_len;
            let weights: Vec<EF> = (0..len).map(|_| rng.random()).collect();
            let values: Vec<EF> = (0..len).map(|_| rng.random()).collect();
            let expected: EF = dot_product(weights.iter().copied(), values.iter().copied());

            // A shape the kernel rejects must say so, never compute a wrong answer.
            if let Some(spread) = SpreadWeights::<F, EF>::new(&weights) {
                prop_assert_eq!(spread.dot(&values), expected);
            }
        }
    }

    #[test]
    fn boundary_values_survive_the_delayed_reduction() {
        // Fixture state: every coordinate at p - 1, the largest product the block sums.
        let len = 1024;
        let top = EF::new([F::NEG_ONE; 4]);
        let weights = alloc::vec![top; len];
        let values = alloc::vec![top; len];
        let expected: EF = dot_product(weights.iter().copied(), values.iter().copied());

        // A packing narrower than one extension element has no spread layout.
        let Some(spread) = SpreadWeights::<F, EF>::new(&weights) else {
            return;
        };
        assert_eq!(spread.dot(&values), expected);
    }

    #[test]
    fn rejects_shapes_that_split_an_element_or_a_block() {
        // A table shorter than one block of vectors never fits.
        let width = <F as Field>::Packing::WIDTH;
        let short = alloc::vec![EF::ONE; (width * BLOCK / 4).max(4) / 2];
        assert!(SpreadWeights::<F, EF>::new(&short).is_none());

        // A degree that does not divide the width never fits, whatever the length.
        if !width.is_multiple_of(4) {
            let long = alloc::vec![EF::ONE; 1 << 12];
            assert!(SpreadWeights::<F, EF>::new(&long).is_none());
        }
    }
}
