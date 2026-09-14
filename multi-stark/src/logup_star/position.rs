//! Embedding of table positions into the field.

use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;

/// Embed one table position.
///
/// Over a prime field this is the integer itself.
///
/// Over a binary tower it is the element whose bit pattern is that integer.
///
/// Both are injective as far as the field reaches.
///
/// That is what lets the fraction identity tell two table entries apart.
#[inline]
pub fn embed<F: Field>(entry: usize) -> F {
    F::interpolation_node(entry)
}

/// Evaluate the multilinear extension of the embedding.
///
/// The embedding is additive over the bits of a position, so its extension is affine:
///
/// ```text
///     J(y) = sum_{s < m} iota(2^(m - 1 - s)) * y_s
/// ```
///
/// A verifier evaluates that itself, in time linear in the variable count.
///
/// The exponent runs backwards because a point carries its most significant coordinate first.
pub fn eval<F: Field, EF: ExtensionField<F>>(point: &Point<EF>) -> EF {
    let num_variables = point.num_variables();

    // Each coordinate carries one bit of the entry, so it weighs what that bit switches on.
    point
        .iter()
        .enumerate()
        .map(|(coordinate, &value)| value * embed::<F>(1 << (num_variables - 1 - coordinate)))
        .sum()
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_binary_field::BinaryField128;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_multilinear_util::poly::Poly;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type Small = BabyBear;
    type SmallExt = BinomialExtensionField<Small, 4>;
    type Binary = BinaryField128;

    #[test]
    fn distinct_entries_embed_to_distinct_elements() {
        // Two entries sharing an embedding leave the identity unable to tell them apart.
        //
        // Telling them apart is the one thing this map has to guarantee.
        //
        // Over a binary tower the first 256 entries are the 256 low bit patterns.
        let embedded = (0..256).map(embed::<Binary>).collect::<Vec<_>>();
        for (entry, value) in embedded.iter().enumerate() {
            assert!(
                embedded[..entry].iter().all(|earlier| earlier != value),
                "entry {entry} collides with an earlier one"
            );
        }

        // Over a prime field the embedding is the integer, so it runs 0, 1, 2, ...
        assert_eq!(embed::<Small>(0), Small::ZERO);
        assert_eq!(embed::<Small>(1), Small::ONE);
        assert_eq!(embed::<Small>(7), Small::from_u8(7));
    }

    #[test]
    fn the_first_entry_embeds_to_zero() {
        // The transcript rejects a zero challenge for exactly this reason.
        //
        // It would land on the first entry and put a zero in a denominator.
        assert_eq!(embed::<Binary>(0), Binary::ZERO);
        assert_eq!(embed::<Small>(0), Small::ZERO);
    }

    #[test]
    fn a_power_of_two_embeds_to_one_basis_element() {
        // The closed form below is a sum over set bits.
        //
        // It holds only if each bit maps to one basis element, additively.
        //
        //     iota(3) = iota(1) + iota(2)
        assert_eq!(embed::<Binary>(3), embed::<Binary>(1) + embed::<Binary>(2));
        assert_eq!(embed::<Binary>(5), embed::<Binary>(1) + embed::<Binary>(4));
    }

    #[test]
    fn the_extension_matches_the_explicit_table() {
        // The verifier evaluates this closed form instead of building the table.
        //
        // The two have to agree everywhere.
        let mut rng = SmallRng::seed_from_u64(0x1071_A000);

        for num_variables in 0..=6 {
            let explicit = (0..1 << num_variables)
                .map(embed::<Binary>)
                .collect::<Vec<_>>();
            let point = Point::<Binary>::rand(&mut rng, num_variables);

            assert_eq!(
                eval::<Binary, Binary>(&point),
                Poly::new(explicit).eval_ext::<Binary>(&point)
            );
        }

        // The same form must hold over a prime field, where entries embed to integers.
        for num_variables in 0..=6 {
            let explicit = (0..1 << num_variables)
                .map(|entry| SmallExt::from(embed::<Small>(entry)))
                .collect::<Vec<_>>();
            let point = Point::<SmallExt>::rand(&mut rng, num_variables);

            assert_eq!(
                eval::<Small, SmallExt>(&point),
                Poly::new(explicit).eval_ext::<Small>(&point)
            );
        }
    }

    #[test]
    fn the_extension_agrees_with_the_embedding_on_the_cube() {
        // A multilinear extension has to reproduce the table it extends at every vertex.
        //
        // Points are most significant coordinate first, so entry 6 is (1, 1, 0).
        for entry in 0..8usize {
            let vertex = Point::<Binary>::hypercube(entry, 3);
            assert_eq!(eval::<Binary, Binary>(&vertex), embed::<Binary>(entry));
        }
    }
}
