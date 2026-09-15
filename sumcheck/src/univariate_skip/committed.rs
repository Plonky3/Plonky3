//! Reading a packed bit witness as a polynomial a commitment can hold.
//!
//! One element per bit.
//!
//! That is the unoptimised cost.
//!
//! It is also short of a sound bit statement, as the function below records.

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
/// # What this proves, and what it does not
///
/// A cell of the commitment is a whole field element.
///
/// Nothing in the proof constrains one to a bit.
///
/// The constraint cannot tell the difference either:
///
/// ```text
///     a * b - c = 0        holds for any field a, b with c = a*b
/// ```
///
/// So what a verifier learns is a statement about field cells, not about bits.
///
/// Booleanity is supplied here only by the prover's own interface.
///
/// It takes packed bytes, and can therefore represent nothing else.
///
/// A prover that forms the round message itself, over field values, is not held to it.
///
/// This path is therefore a scaffold for the chain's mechanics.
///
/// The transcript, the point, the opening, the recombination.
///
/// It is not a sound proof of a bit statement.
///
/// # Why the packed commitment is the answer
///
/// Committing the bits packed, `2^k` of them to an element, is usually called a size win.
///
/// And it is: one element per bit is 128 times the commitment the witness needs.
///
/// Not paying that is the whole point of a binary field.
///
/// It is also what makes the statement a bit statement.
///
/// There a witness cell is an `F_2`-coordinate of a committed element.
///
/// It is a bit by construction, not by a constraint anyone has to add.
///
/// Booleanity stops being something to prove.
///
/// # Why the packed path is not here yet
///
/// Relating a claim about the bit polynomial to one about the packed one is ring switching.
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
/// Panics unless the cell count is a power of two.
pub fn embed_bits<F: Field>(packed: &[u8]) -> Poly<F> {
    assert!(
        (packed.len() * CHUNK_BITS).is_power_of_two(),
        "the cell count must be a power of two"
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
    use crate::univariate_skip::{Composition, Conjunction};

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
    fn this_embedding_puts_only_bits_in_the_cells() {
        // Every cell of what this function returns is zero or one.
        //
        // That is a property of this function and of nothing else.
        //
        // A verifier never runs it, so no proof carries the fact.
        //
        // The test beside this one is the one that matters for soundness.
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
    fn the_constraint_cannot_tell_a_bit_from_a_field_element() {
        // Invariant: the conjunction vanishes on field triples that are not bits.
        //
        //     a, b uniform in GF(2^128),  c = a*b   ->   a*b - c = 0
        //
        // This is why one element per cell proves a field statement and not a bit statement.
        //
        // The commitment holds field cells.
        //
        // The constraint is satisfied by field cells that are nowhere near Boolean.
        //
        // Only the prover's interface keeps the witness Boolean, and an interface is no proof.
        //
        // Under the packed commitment this test would have nothing to say.
        //
        // A cell there is an `F_2`-coordinate, so a non-bit cell cannot be expressed at all.
        let mut rng = SmallRng::seed_from_u64(0xF1E1D);
        for _ in 0..32 {
            let (a, b) = (rng.random::<EF>(), rng.random::<EF>());
            let c = a * b;

            // Not bits, by overwhelming probability over a 128-bit field.
            assert!(a != EF::ZERO && a != EF::ONE);
            assert!(b != EF::ZERO && b != EF::ONE);

            // And yet the constraint the zerocheck proves vanishing is satisfied.
            assert_eq!(Composition::<EF>::eval(&Conjunction, &[a, b, c]), EF::ZERO);
        }
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
    #[should_panic(expected = "cell count must be a power of two")]
    fn rejects_a_witness_that_is_not_a_hypercube() {
        // Three bytes is 24 cells, which is no hypercube.
        let _ = embed_bits::<EF>(&[0u8; 3]);
    }
}
