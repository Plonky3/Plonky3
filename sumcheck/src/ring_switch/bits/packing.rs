//! A bit witness held as the multilinear a commitment holds (Construction 3.1).

use alloc::vec::Vec;
use core::borrow::Borrow;

use p3_binary_field::TowerLevel;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::poly::Poly;

use super::basis::Coefficients;

/// A bit witness read as the packed multilinear over a tower level.
///
/// # What the type does and does not carry
///
/// At a byte-aligned level, reading bytes and reading them back are inverse.
/// So every packed multilinear unpacks to exactly one bit witness.
///
/// No multilinear is excluded, and booleanity comes from the packing itself.
/// This type carries no part of that argument.
///
/// What it does carry is the width and the level.
/// The reduction it feeds cannot be handed a polynomial of another shape.
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
///
/// # Where the elements live
///
/// `S` is the backing store, owned by default and a slice in [`BitPackingView`].
///
/// A commitment already holds the elements, so reading them through a view spares the
/// reduction a copy of the whole witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitPacking<EF, S = Vec<EF>> {
    /// The packed multilinear, one element per `d` cells of the witness.
    packed: Poly<EF, S>,
}

/// Borrowed view of a packed bit witness.
pub type BitPackingView<'a, EF> = BitPacking<EF, &'a [EF]>;

impl<EF: TowerLevel> BitPacking<EF> {
    /// Read a packed bit witness, one element per element's worth of bytes.
    ///
    /// # Arguments
    ///
    /// The witness as held, eight cells to the byte, lowest bit first.
    ///
    /// # Errors
    ///
    /// - The level's elements are narrower than the byte its stride reads.
    /// - The cell count is no power of two, so the witness covers no hypercube.
    /// - The witness is too short to fill one element, so nothing to pack.
    pub fn new(bits: &[u8]) -> Result<Self, BitPackingError> {
        let stride = EF::NUM_BYTES;

        // Below a byte the stride still reads a whole one.
        // The level keeps only its own low bits.
        //
        //     Gf2           1 bit  of 8 kept, 7 dropped
        //     BinaryField4  4 bits of 8 kept, 4 dropped
        //
        // The dropped cells leave a polynomial that is not the witness.
        // So the level is refused rather than packed.
        if 8 * stride != Coefficients::<EF>::DIMENSION {
            return Err(BitPackingError::SubByteLevel {
                bits: Coefficients::<EF>::DIMENSION,
            });
        }
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

    /// Give up the packing, for a caller that commits to it.
    pub fn into_poly(self) -> Poly<EF> {
        self.packed
    }
}

impl<EF: TowerLevel, S: Borrow<[EF]>> BitPacking<EF, S> {
    /// Read an already-packed multilinear as a bit witness, sweeping no byte twice.
    ///
    /// The packing is a bijection, so nothing needs checking beyond the shape.
    ///
    /// # Errors
    ///
    /// - The level's elements are narrower than the byte its stride reads.
    /// - The element count is no power of two, so the packing covers no hypercube.
    pub fn from_packed(packed: Poly<EF, S>) -> Result<Self, BitPackingError> {
        if 8 * EF::NUM_BYTES != Coefficients::<EF>::DIMENSION {
            return Err(BitPackingError::SubByteLevel {
                bits: Coefficients::<EF>::DIMENSION,
            });
        }
        if !packed.num_evals().is_power_of_two() {
            return Err(BitPackingError::NotAHypercube {
                cells: packed.num_evals() * Coefficients::<EF>::DIMENSION,
            });
        }
        Ok(Self { packed })
    }

    /// The multilinear a commitment holds.
    pub const fn poly(&self) -> &Poly<EF, S> {
        &self.packed
    }

    /// Variables the packing has: the witness's, less the absorbed ones.
    pub fn num_variables(&self) -> usize {
        self.packed.num_variables()
    }

    /// Elements the packing holds.
    pub fn len(&self) -> usize {
        self.packed.num_evals()
    }

    /// Whether the packing holds no element.
    ///
    /// Never true of a value this type builds: one element is its minimum.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The coordinates of one packed element.
    pub fn coefficients(&self, index: usize) -> Coefficients<EF> {
        Coefficients::of(self.packed.as_slice()[index])
    }
}

/// Reasons a bit witness cannot be packed.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum BitPackingError {
    /// The level's elements are narrower than one byte.
    ///
    /// Its stride still reads a whole byte, so cells would be dropped.
    #[error("a {bits}-bit level cannot pack a byte's worth of cells")]
    SubByteLevel {
        /// Bits one element of the level holds.
        bits: usize,
    },
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
    use p3_binary_field::{BinaryField4, BinaryField16, BinaryField128, Gf2};
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
        // Invariant: coordinate `j` of element `w` is cell `d*w + j`.
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
        // Fixture state: 256 cells is 8 variables, of which 16 bits absorb 4.
        let packing = BitPacking::<EF>::new(&[0u8; 32]).unwrap();

        assert_eq!(
            packing.num_variables(),
            8 - Coefficients::<EF>::LOG_DIMENSION
        );
    }

    #[test]
    fn the_packing_is_the_bit_planes_read_together() {
        // Invariant: the packing at a point is the weighted sum of bit planes.
        //
        //     t'(r) = sum_j g_j(r) * beta_j,   g_j = cells d*w + j
        //
        // The whole reduction is built on this, so it is pinned directly.
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
            // Bit plane `j`, as its own multilinear over the packed variables.
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
    fn a_level_narrower_than_a_byte_is_refused() {
        // The stride reads a whole byte whatever the level holds.
        //
        //     BinaryField4  keeps 4 of the 8 cells, drops the rest
        //     Gf2           keeps 1 of the 8
        //
        // Packing anyway would make two different witnesses equal.
        // It hands back a polynomial that is neither of them.
        assert_eq!(
            BitPacking::<BinaryField4>::new(&[0x0F, 0x0F]).unwrap_err(),
            BitPackingError::SubByteLevel { bits: 4 }
        );
        assert_eq!(
            BitPacking::<Gf2>::new(&[0xFE]).unwrap_err(),
            BitPackingError::SubByteLevel { bits: 1 }
        );

        // A byte-aligned level is unaffected.
        assert!(BitPacking::<EF>::new(&[0u8; 2]).is_ok());
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
