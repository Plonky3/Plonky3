//! Reading a packed bit witness as a polynomial a commitment can hold.

use alloc::vec::Vec;

use p3_field::Field;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::poly::Poly;

use super::lde::CHUNK_BITS;

/// Read a packed bit witness as its multilinear over the hypercube.
///
/// # Overview
///
/// A binary zerocheck proves its constraint over a bit-valued witness.
///
/// It ends on a claim about that witness at one point.
///
/// Discharging the claim means committing to the very polynomial the claim is about:
///
/// ```text
///     packed bits              one bit per cell, eight to the byte
///     embedded multilinear     one field element per cell
/// ```
///
/// # Cost
///
/// This is the unoptimised path, and the cost is not small.
///
/// One 128-bit element per bit is 128 times the commitment the witness needs.
///
/// Not paying that is the whole point of a binary field.
///
/// The optimisation is to commit the bits packed, `2^k` of them to an element.
///
/// Ring switching then relates a claim about the bit polynomial to one about the packed one.
///
/// That reduction is already written generically in this crate.
///
/// It cannot be instantiated at a bit alphabet.
///
/// Its basis-coefficient accessor hands out a borrowed slice.
///
/// For single bits that would mean borrowing 128 bytes out of a 16-byte value.
///
/// Closing that gap needs a packing built on the tower's own bit representation.
///
/// That is a piece of work in its own right.
///
/// # Panics
///
/// Panics if the packed length is not a whole number of bytes' worth of cells.
pub fn embed_bits<F: Field>(packed: &[u8]) -> Poly<F> {
    assert!(
        (packed.len() * CHUNK_BITS).is_power_of_two(),
        "the hypercube covers a power of two cells"
    );

    // One element per bit, least significant bit first within each byte.
    //
    // The order matches the one the skip round reads rows in.
    //
    // A claim about this polynomial is therefore about the cells the constraint checked.
    let values = packed
        .par_iter()
        .flat_map_iter(|&byte| {
            (0..CHUNK_BITS).map(move |bit| {
                if (byte >> bit) & 1 == 1 {
                    F::ONE
                } else {
                    F::ZERO
                }
            })
        })
        .collect::<Vec<_>>();

    Poly::new(values)
}

#[cfg(test)]
mod tests {
    use p3_binary_field::BinaryField128;
    use p3_field::PrimeCharacteristicRing;
    use p3_multilinear_util::point::Point;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type EF = BinaryField128;

    #[test]
    fn the_embedding_reads_the_documented_bit_order() {
        // The skip round reads bit `j` of byte `b` as the cell at index `8b + j`.
        //
        // The embedding has to agree, or the claim would be about different cells.
        //
        //     [0b0000_0010, 0]  ->  cell 1 is one, every other cell zero
        let packed = [0b0000_0010u8, 0];
        let embedded = embed_bits::<EF>(&packed);

        assert_eq!(embedded.num_evals(), 16);
        for cell in 0..16 {
            let expected = if cell == 1 { EF::ONE } else { EF::ZERO };
            assert_eq!(embedded.as_slice()[cell], expected, "cell={cell}");
        }
    }

    #[test]
    fn the_second_byte_holds_the_next_eight_cells() {
        // Byte one carries cells eight through fifteen, which is what the chunk width fixes.
        let packed = [0u8, 0b1000_0000];
        let embedded = embed_bits::<EF>(&packed);

        assert_eq!(embedded.as_slice()[15], EF::ONE);
        assert!(
            embedded.as_slice()[..15]
                .iter()
                .all(|&value| value == EF::ZERO)
        );
    }

    #[test]
    fn every_cell_is_a_bit() {
        // The embedded polynomial's cells are exactly zero and one.
        //
        // That is the premise every reduction over a bit witness rests on.
        //
        // It is pinned here rather than assumed from the construction.
        let mut rng = SmallRng::seed_from_u64(0xB175);
        let packed = (0..32).map(|_| rng.random::<u8>()).collect::<Vec<_>>();
        let embedded = embed_bits::<EF>(&packed);

        assert!(
            embedded
                .as_slice()
                .iter()
                .all(|&value| value == EF::ZERO || value == EF::ONE)
        );
    }

    #[test]
    fn the_embedding_is_the_multilinear_the_rows_describe() {
        // Reading the embedding at a random point must agree with weighing the set bits.
        //
        //     f~(r) = sum over cells with the bit set of eq(r, cell)
        let mut rng = SmallRng::seed_from_u64(0x3B1);
        let packed = (0..16).map(|_| rng.random::<u8>()).collect::<Vec<_>>();
        let embedded = embed_bits::<EF>(&packed);

        let point = Point::new((0..7).map(|_| rng.random::<EF>()).collect());
        let eq = Poly::new_from_point(point.as_slice(), EF::ONE);

        let expected = (0..packed.len() * CHUNK_BITS)
            .filter(|&cell| (packed[cell / CHUNK_BITS] >> (cell % CHUNK_BITS)) & 1 == 1)
            .map(|cell| eq.as_slice()[cell])
            .sum::<EF>();

        assert_eq!(embedded.eval_base(&point), expected);
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn rejects_a_witness_that_is_not_a_hypercube() {
        // Three bytes is 24 cells, which is no hypercube.
        let _ = embed_bits::<EF>(&[0u8; 3]);
    }
}
