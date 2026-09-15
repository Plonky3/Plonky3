//! A bit witness held as the packed multilinear a commitment holds (Construction 3.1).

use alloc::vec::Vec;

use p3_binary_field::TowerLevel;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::poly::Poly;

use super::basis::Coefficients;

/// A bit witness read as the packed multilinear over a tower level.
///
/// # Overview
///
/// The reduction's soundness rests on the committed polynomial being a packing of bits.
///
/// An arbitrary multilinear will not do.
///
/// Carrying that as a type stops the wrong polynomial being handed to the reduction.
///
/// # The packing
///
/// Coordinate `j` of element `w` is cell `d*w + j` of the witness:
///
/// ```text
///     cells   0..128    ->  element 0
///     cells 128..256    ->  element 1
/// ```
///
/// Because the basis is the byte representation's own, packing moves no bits.
///
/// It reads the same bytes back as a wider integer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitPacking<EF> {
    /// The packed multilinear, one element per `d` cells of the witness.
    packed: Poly<EF>,
}

impl<EF: TowerLevel> BitPacking<EF> {
    /// Read a packed bit witness, one element per element's worth of bytes.
    ///
    /// # Arguments
    ///
    /// The witness as it is held, eight cells to the byte, least significant bit first.
    ///
    /// # Errors
    ///
    /// - The cell count is no power of two, so the witness covers no hypercube.
    /// - The witness is too short to fill one element, so there is nothing to pack.
    pub fn new(bits: &[u8]) -> Result<Self, BitPackingError> {
        let stride = EF::NUM_BYTES;
        if !(bits.len() * 8).is_power_of_two() {
            return Err(BitPackingError::NotAHypercube {
                cells: bits.len() * 8,
            });
        }
        if bits.len() < stride {
            return Err(BitPackingError::TooShort {
                needed: stride,
                actual: bits.len(),
            });
        }

        Ok(Self {
            packed: Poly::new(
                bits.par_chunks_exact(stride)
                    .map(|chunk| EF::from_le_byte_iter(chunk.iter().copied()))
                    .collect::<Vec<_>>(),
            ),
        })
    }

    /// The multilinear a commitment holds.
    pub const fn poly(&self) -> &Poly<EF> {
        &self.packed
    }

    /// Variables the packed multilinear has, which is the witness's less the absorbed ones.
    pub fn num_variables(&self) -> usize {
        self.packed.num_variables()
    }

    /// Elements the packing holds.
    pub fn len(&self) -> usize {
        self.packed.num_evals()
    }

    /// Whether the packing holds no element.
    ///
    /// Never true of a value this type builds, since one element is the minimum it accepts.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The coordinates of one packed element.
    pub fn coefficients(&self, index: usize) -> Coefficients<EF> {
        Coefficients::of(self.packed.as_slice()[index])
    }

    /// Give up the packing, for a caller that commits to it.
    pub fn into_poly(self) -> Poly<EF> {
        self.packed
    }
}

/// Reasons a bit witness cannot be packed.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum BitPackingError {
    /// The witness does not cover a power-of-two number of cells.
    #[error("a bit witness covers {cells} cells, which is no hypercube")]
    NotAHypercube {
        /// Cells the witness holds.
        cells: usize,
    },
    /// The witness is shorter than one packed element.
    #[error("packing needs at least {needed} bytes, got {actual}")]
    TooShort {
        /// Bytes one element absorbs.
        needed: usize,
        /// Bytes the witness holds.
        actual: usize,
    },
}

#[cfg(test)]
mod tests {
    use p3_binary_field::{BinaryField16, BinaryField128};
    use p3_field::PrimeCharacteristicRing;
    use p3_multilinear_util::point::Point;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::univariate_skip::embed_bits;

    type EF = BinaryField16;

    /// A random bit witness of the given byte length.
    fn bits(seed: u64, bytes: usize) -> Vec<u8> {
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..bytes).map(|_| rng.random::<u8>()).collect()
    }

    #[test]
    fn the_coordinates_are_the_cells_in_order() {
        // Invariant: coordinate `j` of element `w` is cell `d*w + j` of the witness.
        //
        // Both sides of the reduction index the same cells only if this holds.
        let witness = bits(0x9AC, 8);
        let packing = BitPacking::<EF>::new(&witness).unwrap();

        assert_eq!(packing.len(), 4);
        for element in 0..packing.len() {
            for (j, bit) in packing.coefficients(element).iter().enumerate() {
                let cell = element * Coefficients::<EF>::DIMENSION + j;
                let expected = (witness[cell / 8] >> (cell % 8)) & 1 == 1;
                assert_eq!(bit, expected, "element={element} coordinate={j}");
            }
        }
    }

    #[test]
    fn packing_agrees_with_the_one_element_per_bit_embedding() {
        // The embedding spends one element per bit.
        //
        // The packing spends one per `d` bits.
        //
        // Both have to describe the same witness, cell for cell.
        let witness = bits(0xE3B, 32);
        let embedded = embed_bits::<BinaryField128>(&witness);
        let packing = BitPacking::<BinaryField128>::new(&witness).unwrap();

        for element in 0..packing.len() {
            for (j, bit) in packing.coefficients(element).iter().enumerate() {
                let cell = element * Coefficients::<BinaryField128>::DIMENSION + j;
                let expected = if bit {
                    BinaryField128::ONE
                } else {
                    BinaryField128::ZERO
                };
                assert_eq!(embedded.as_slice()[cell], expected, "cell={cell}");
            }
        }
    }

    #[test]
    fn the_packing_loses_exactly_the_absorbed_variables() {
        // Fixture state: 256 cells is 8 variables, of which a 16-bit level absorbs 4.
        let packing = BitPacking::<EF>::new(&[0u8; 32]).unwrap();

        assert_eq!(
            packing.num_variables(),
            8 - Coefficients::<EF>::LOG_DIMENSION
        );
    }

    #[test]
    fn the_packing_is_the_bit_planes_read_together() {
        // Invariant: the packing at a point is the basis-weighted sum of the bit planes.
        //
        //     t'(r) = sum_j g_j(r) * beta_j,   g_j the multilinear of cells d*w + j
        //
        // This is the identity the whole reduction is built on, so it is pinned directly.
        let mut rng = SmallRng::seed_from_u64(0x91A2);
        let witness = bits(0x91A3, 16);
        let packing = BitPacking::<EF>::new(&witness).unwrap();
        let point = Point::new(
            (0..packing.num_variables())
                .map(|_| rng.random::<EF>())
                .collect(),
        );

        let d = Coefficients::<EF>::DIMENSION;
        let mut expected = EF::ZERO;
        for j in 0..d {
            // Bit plane `j`, read as its own multilinear over the packed variables.
            let plane = Poly::new(
                (0..packing.len())
                    .map(|w| {
                        if packing.coefficients(w).get(j) {
                            EF::ONE
                        } else {
                            EF::ZERO
                        }
                    })
                    .collect::<Vec<_>>(),
            );
            let mut basis = Coefficients::<EF>::zero();
            basis.set(j);
            expected += plane.eval_base(&point) * basis.element();
        }

        assert_eq!(packing.poly().eval_base(&point), expected);
    }

    #[test]
    fn a_witness_that_is_no_hypercube_is_refused() {
        // Three bytes is 24 cells, which is no hypercube.
        assert_eq!(
            BitPacking::<EF>::new(&[0u8; 3]).unwrap_err(),
            BitPackingError::NotAHypercube { cells: 24 }
        );
    }

    #[test]
    fn a_witness_too_short_for_one_element_is_refused() {
        // One byte does not fill a 16-bit element, so there is nothing to pack.
        assert_eq!(
            BitPacking::<EF>::new(&[0u8; 1]).unwrap_err(),
            BitPackingError::TooShort {
                needed: 2,
                actual: 1
            }
        );
    }
}
