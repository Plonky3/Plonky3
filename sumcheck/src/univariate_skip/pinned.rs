//! Zerocheck challenge coordinates fixed ahead of time, with structured equality weights.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_binary_field::TowerLevel;
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::poly::Poly;
use thiserror::Error;

use super::lde::{CHUNK_BITS, TABLE_ROWS};

/// Reasons a set of pinned coordinates cannot be used.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PinnedEqError {
    /// The pinned coordinates would need more equality weights than the field can keep independent.
    ///
    /// The weights have to be linearly independent over the prime field.
    ///
    /// A `b`-bit binary field holds at most `b` independent elements.
    #[error("{count} pinned coordinates need 2^{count} weights, above the {field_bits} available")]
    TooManyPinned {
        /// Number of coordinates the caller asked to pin.
        count: usize,
        /// Bit width of the field the weights live in.
        field_bits: usize,
    },
    /// The generator raised to a power of two reached one, so one coordinate has no finite value.
    ///
    /// The coordinate is a ratio whose denominator is that power plus one, which vanishes there.
    #[error("the generator's 2^{index} power is one, leaving coordinate {index} undefined")]
    DegenerateGenerator {
        /// Index of the coordinate whose denominator vanished.
        index: usize,
    },
    /// The equality weights are linearly dependent over the prime field.
    ///
    /// A dependent set leaves a nonzero constraint pattern the zerocheck cannot see.
    ///
    /// Pinning these coordinates would therefore not be sound.
    #[error("the equality weights are linearly dependent over the prime field")]
    DependentWeights,
}

/// Zerocheck challenge coordinates fixed ahead of time, together with their equality weights.
///
/// # Overview
///
/// A zerocheck weighs its constraint by an equality polynomial at a random challenge point.
///
/// Every coordinate of that point becomes a multiplier in the prover's inner loop.
///
/// Multiplying by a generic large-field element is the most expensive operation there.
///
/// Fixing some coordinates to constants replaces those multipliers with cheaper ones.
///
/// Nothing is paid for it in transcript randomness.
///
/// See Dao, Thaler, *More Optimizations to Sum-Check Proving*, and Bünz, Rothblum, Wang,
/// *Flock*, Section 4.3.
///
/// This is a reduction of its own, with no consumer in the crate yet.
///
/// A skip round weighs its rows with whatever equality table the caller supplies.
///
/// Laying those rows out so the pinned coordinates index blocks is the arithmetization's choice.
///
/// The pinned coordinates sit at the tail of the point, so they weigh contiguous blocks:
///
/// ```text
///     sum_b eq(r, b) f(b)  =  sum_{b_out} eq(r_out, b_out) * g(b_out)
///
///     g(b_out) = sum_{b_in} eq(r_in, b_in) f(b_out, b_in)
///                 ^ the pinned block, one contiguous run of 2^d values
/// ```
///
/// # Soundness
///
/// Pinning restricts the verifier's challenge from the whole space to an affine subspace.
///
/// That has to be argued rather than assumed.
///
/// Write the constraint polynomial as `p`.
///
/// Fixing the pinned coordinates to each `b` splits its extension into `2^d` pieces `p_b`.
///
/// The pinned part of the check reads one fixed combination of those pieces:
///
/// ```text
///     sum_b weight_b * p_b
/// ```
///
/// Suppose the pieces have coefficients in the prime field.
///
/// The combination is then nonzero whenever some piece is.
///
/// That happens exactly when the weights are independent over the prime field.
///
/// Every constructor here rejects a dependent weight set for that reason.
///
/// The caller owes the other half, and it is a strong obligation:
///
/// **Every witness the prover can commit to must keep the constraint prime-field-valued.**
///
/// Not the honest witness, every witness.
///
/// Over a wider alphabet the fold `y -> sum_b weight_b * y_b` has a large kernel.
///
/// Any cell pattern inside that kernel is invisible to this check.
///
/// - The premise must therefore come from the commitment alphabet.
/// - A prime-field-packed commitment gives it, and so does ring switching.
/// - **A booleanity constraint in the same zerocheck does not give it.**
/// - The pinned fold is what would have to catch the non-bit witness in the first place.
///
/// Batching several prime-field-valued constraints stays sound.
///
/// The coefficients must be drawn after the commitment.
///
/// The batched sum is then prime-field-valued for every committed witness, whatever they are.
///
/// A constraint valued in a larger subfield needs independence over *that* subfield.
///
/// That is a stronger condition, and it is not checked here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedEqWeights<F> {
    /// The pinned coordinates, ordered as they sit at the tail of the challenge point.
    challenges: Vec<F>,
    /// The equality weight of each assignment of the pinned coordinates.
    ///
    /// Entry `k` weighs the assignment whose bits spell `k`, first coordinate most significant.
    ///
    /// That is how an evaluation table is indexed.
    weights: Vec<F>,
}

impl<F: TowerLevel> PinnedEqWeights<F>
where
    F::Repr: Into<u128>,
{
    /// Pin coordinates so that their equality weights run through the powers of a generator.
    ///
    /// # Overview
    ///
    /// Choosing the coordinates as
    ///
    /// ```text
    ///     r_i = u_i / (1 + u_i)        with u_i = generator^(2^(i-1))
    /// ```
    ///
    /// makes `r_i / (1 + r_i) = u_i`, and the equality weight of an assignment telescopes:
    ///
    /// ```text
    ///     eq(r, b) = prod_i (1 + r_i) * prod_{i : b_i = 1} u_i
    ///              = C * generator^int(b)
    /// ```
    ///
    /// so the whole weight set is one geometric progression scaled by a constant.
    ///
    /// In a polynomial basis the generator is the class of the indeterminate.
    ///
    /// Multiplying by its powers is then a bit shift followed by one reduction.
    ///
    /// # Arguments
    ///
    /// - `generator`: the element whose powers the weights run through.
    /// - `count`: how many coordinates to pin, giving `2^count` weights.
    ///
    /// # Errors
    ///
    /// - When the weights would outnumber the independent elements the field holds.
    /// - When a power of the generator reaches one, leaving a coordinate undefined.
    /// - When the resulting weights are linearly dependent over the prime field.
    pub fn geometric(generator: F, count: usize) -> Result<Self, PinnedEqError> {
        // The weights must stay independent, and a b-bit field holds at most b of those.
        let field_bits = 1usize << F::LOG_BITS;
        if count >= usize::BITS as usize || (1usize << count) > field_bits {
            return Err(PinnedEqError::TooManyPinned { count, field_bits });
        }

        // Build the coordinates from the repeated squares of the generator.
        //
        //     u_1 = generator, u_{i+1} = u_i^2
        //
        // Coordinate i is collected in the paper's order, lowest weight bit first.
        let mut challenges = Vec::with_capacity(count);
        let mut power = generator;
        for index in 0..count {
            // The coordinate divides by one plus the power, which must therefore be nonzero.
            let denominator = F::ONE + power;
            if denominator.is_zero() {
                return Err(PinnedEqError::DegenerateGenerator { index });
            }
            challenges.push(power * denominator.inverse());
            power = power.square();
        }

        // An evaluation table indexes its first variable most significantly.
        //
        // The tail of the challenge point therefore lists the coordinates the other way round.
        challenges.reverse();

        Self::from_challenges(challenges)
    }

    /// Pin an explicit list of coordinates, checking that the weights they induce stay usable.
    ///
    /// The coordinates are ordered as they sit at the tail of the challenge point.
    ///
    /// The first one is therefore the most significant bit of a block index.
    ///
    /// # Errors
    ///
    /// - When the weights would outnumber the independent elements the field holds.
    /// - When the resulting weights are linearly dependent over the prime field.
    pub fn from_challenges(challenges: Vec<F>) -> Result<Self, PinnedEqError> {
        let count = challenges.len();
        let field_bits = 1usize << F::LOG_BITS;
        if count >= usize::BITS as usize || (1usize << count) > field_bits {
            return Err(PinnedEqError::TooManyPinned { count, field_bits });
        }

        // Expand the equality polynomial over the pinned coordinates into its evaluation table.
        //
        // The shared builder fixes which coordinate is the most significant bit of an index.
        //
        // Reusing it is what keeps weight `k` aligned with the block entry the sumcheck weighs.
        let weights = Poly::new_from_point(&challenges, F::ONE).into_evals();

        // A dependent weight set leaves a constraint pattern invisible to the zerocheck.
        if !prime_field_independent(&weights) {
            return Err(PinnedEqError::DependentWeights);
        }

        Ok(Self {
            challenges,
            weights,
        })
    }
}

impl<F: Field> PinnedEqWeights<F> {
    /// Number of pinned coordinates.
    #[must_use]
    pub const fn num_pinned(&self) -> usize {
        self.challenges.len()
    }

    /// Number of values one pinned block covers.
    #[must_use]
    pub const fn block_len(&self) -> usize {
        self.weights.len()
    }

    /// The pinned coordinates, ordered as they sit at the tail of the challenge point.
    #[must_use]
    pub fn challenges(&self) -> &[F] {
        &self.challenges
    }

    /// The equality weight of each assignment of the pinned coordinates.
    #[must_use]
    pub fn weights(&self) -> &[F] {
        &self.weights
    }

    /// Combine one block of values under the pinned weights.
    ///
    /// # Panics
    ///
    /// Panics if the block length differs from the number of weights.
    #[must_use]
    pub fn fold<A>(&self, block: &[A]) -> A
    where
        A: p3_field::Algebra<F> + Copy,
    {
        assert_eq!(block.len(), self.weights.len(), "one value per weight");

        // The weights are fixed, so this is a plain inner product over the block.
        block
            .iter()
            .zip(&self.weights)
            .map(|(&value, &weight)| value * weight)
            .sum()
    }

    /// Combine every block of a slice under the pinned weights.
    ///
    /// # Panics
    ///
    /// Panics if the input is not a whole number of blocks.
    ///
    /// Panics if the output length differs from the block count.
    pub fn fold_blocks<A>(&self, values: &[A], out: &mut [A])
    where
        A: p3_field::Algebra<F> + Copy,
    {
        assert_eq!(values.len(), out.len() * self.weights.len(), "whole blocks");

        // Each block collapses independently to one value.
        for (block, entry) in values.chunks_exact(self.weights.len()).zip(out) {
            *entry = self.fold(block);
        }
    }
}

/// A byte table that folds subfield-valued blocks under a fixed weight set.
///
/// # Overview
///
/// Extending bit-valued rows onto a subfield domain leaves the results in that subfield.
///
/// A zerocheck weighing them straight afterwards multiplies subfield values by field weights.
///
/// Both the weight and the subfield embedding are fixed, so their composition tabulates.
///
/// ```text
///     T[k][v] = weight_k * embed(v)        k < 2^d,  v < 256
/// ```
///
/// Folding a block is then one lookup and one exclusive-or per entry, with no multiplications.
///
/// # Soundness
///
/// This is an arithmetic shortcut, not a weaker premise.
///
/// The weights it tabulates were only certified independent over the **prime** field.
///
/// - Safe where the values are off-hypercube, as the extended round values are.
/// - **Not** safe as the pinned fold of a zerocheck whose hypercube cells are byte-valued.
/// - That case needs the weights independent over the byte field, which nothing here checks.
///
/// # Performance
///
/// The table is `2^d * 256` large-field elements.
/// Four pinned coordinates over a 128-bit field make that 64 KiB, built once at setup.
#[derive(Debug, Clone)]
pub struct SubfieldFoldTable<F, Sub> {
    /// One row of products per weight, each row indexed by the subfield element's byte.
    table: Vec<F>,
    /// Number of values one block covers, which is the number of rows.
    block_len: usize,
    /// Marker for the subfield the rows were tabulated against.
    ///
    /// Carrying it on the type stops a fold reading the table through another subfield.
    ///
    /// That would index the right row with the wrong column.
    _sub: PhantomData<Sub>,
}

impl<F, Sub> SubfieldFoldTable<F, Sub>
where
    F: Field + ExtensionField<Sub>,
    Sub: TowerLevel,
{
    /// Tabulate the weights against every value of a byte-wide subfield.
    ///
    /// # Panics
    ///
    /// Panics if the subfield is not byte-wide, since the table is indexed by a single byte.
    #[must_use]
    pub fn new(weights: &PinnedEqWeights<F>) -> Self
    where
        Sub::Repr: From<u8>,
    {
        assert_eq!(
            1usize << Sub::LOG_BITS,
            CHUNK_BITS,
            "the fold table is indexed by one byte, so the subfield must be byte-wide"
        );

        // One row per weight, one column per subfield element.
        let block_len = weights.block_len();
        let mut table = F::zero_vec(block_len * TABLE_ROWS);
        for (row, &weight) in table
            .as_chunks_mut::<TABLE_ROWS>()
            .0
            .iter_mut()
            .zip(weights.weights())
        {
            for (value, entry) in row.iter_mut().enumerate() {
                // Embed the subfield element, then scale it by this row's weight.
                let embedded = F::from(Sub::from_repr(Sub::Repr::from(value as u8)));
                *entry = weight * embedded;
            }
        }

        Self {
            table,
            block_len,
            _sub: PhantomData,
        }
    }

    /// Number of values one block covers.
    #[must_use]
    pub const fn block_len(&self) -> usize {
        self.block_len
    }

    /// Combine one block of subfield values under the tabulated weights.
    ///
    /// # Panics
    ///
    /// Panics if the block length differs from the tabulated one.
    #[must_use]
    pub fn fold(&self, block: &[Sub]) -> F
    where
        Sub::Repr: Into<u128>,
    {
        assert_eq!(block.len(), self.block_len, "one value per weight");

        // Read each value's row at that value's byte and accumulate.
        block
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                let byte = value.to_repr().into() as usize & 0xff;
                self.table[index * TABLE_ROWS + byte]
            })
            .sum()
    }
}

/// Whether the given elements are linearly independent over the prime field.
///
/// # Algorithm
///
/// Gaussian elimination on the elements' coordinate patterns, kept as a list of pivots:
///
/// ```text
///     for each element:
///         reduce it by every pivot whose leading bit it carries
///         a nonzero remainder becomes a new pivot
///         a zero remainder means the element was already spanned  ->  dependent
/// ```
///
/// The coordinate pattern comes from the field's own representation.
///
/// Independence over the prime field does not depend on which basis expresses it.
fn prime_field_independent<F: TowerLevel>(elements: &[F]) -> bool
where
    F::Repr: Into<u128>,
{
    // Each pivot is a pattern already reduced against every pivot before it.
    let mut pivots = Vec::<u128>::with_capacity(elements.len());

    for &element in elements {
        let mut pattern: u128 = element.to_repr().into();

        // Cancel every pivot whose leading bit this pattern still carries.
        //
        // Invariant: a stored pivot is zero at the leading bit of every earlier pivot.
        //
        // Cancelling by one pivot can therefore only disturb later ones.
        //
        // This loop has yet to reach those, so a single forward pass reduces fully.
        for &pivot in &pivots {
            let leading = 1u128 << (127 - pivot.leading_zeros());
            if pattern & leading != 0 {
                pattern ^= pivot;
            }
        }

        // Nothing left means this element was already in the span of the earlier ones.
        if pattern == 0 {
            return false;
        }
        pivots.push(pattern);
    }

    true
}

#[cfg(test)]
mod tests {
    use p3_binary_field::{BinaryField8, BinaryField16, BinaryField128, Ghash128};
    use p3_field::PrimeCharacteristicRing;
    use p3_multilinear_util::poly::Poly;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    /// The class of the indeterminate in a polynomial basis, whose powers form the standard basis.
    fn indeterminate() -> Ghash128 {
        Ghash128::from_repr(2)
    }

    #[test]
    fn geometric_weights_are_the_equality_table_of_the_pinned_point() {
        // Invariant: the structured weights are what the equality polynomial produces.
        //
        // Anything else would make the fold compute a different sum.
        //
        // Fixture state: 4 pinned coordinates, so a block of 16 values.
        let pinned = PinnedEqWeights::geometric(indeterminate(), 4).unwrap();
        let expected = Poly::new_from_point(pinned.challenges(), Ghash128::ONE);
        assert_eq!(pinned.weights(), expected.as_slice());
    }

    #[test]
    fn geometric_weights_are_a_progression_in_the_generator() {
        // The point of this choice is that consecutive weights differ by one generator factor.
        //
        // In a polynomial basis that factor is a shift.
        //
        //     weight_k = C * x^k
        let pinned = PinnedEqWeights::geometric(indeterminate(), 4).unwrap();
        for index in 1..pinned.block_len() {
            assert_eq!(
                pinned.weights()[index],
                pinned.weights()[index - 1] * indeterminate(),
                "index={index}"
            );
        }
    }

    #[test]
    fn folding_a_block_matches_weighing_it_by_hand() {
        // Fixture state: 3 pinned coordinates over a 128-bit field, so blocks of 8.
        let mut rng = SmallRng::seed_from_u64(19);
        let pinned = PinnedEqWeights::geometric(indeterminate(), 3).unwrap();

        let block = (0..pinned.block_len())
            .map(|_| rng.random::<Ghash128>())
            .collect::<Vec<_>>();

        // The fold is the inner product of the block with the weights.
        let expected = block
            .iter()
            .zip(pinned.weights())
            .map(|(&value, &weight)| value * weight)
            .sum::<Ghash128>();
        assert_eq!(pinned.fold(&block), expected);
    }

    #[test]
    fn folding_blocks_collapses_each_run_independently() {
        // Fixture state: 5 blocks of 4 values each.
        //
        //     values: [ b0 | b1 | b2 | b3 | b4 ]   ->  out: [ f0, f1, f2, f3, f4 ]
        let mut rng = SmallRng::seed_from_u64(23);
        let pinned = PinnedEqWeights::geometric(indeterminate(), 2).unwrap();
        let num_blocks = 5;

        let values = (0..num_blocks * pinned.block_len())
            .map(|_| rng.random::<Ghash128>())
            .collect::<Vec<_>>();
        let mut out = Ghash128::zero_vec(num_blocks);
        pinned.fold_blocks(&values, &mut out);

        for (index, block) in values.chunks_exact(pinned.block_len()).enumerate() {
            assert_eq!(out[index], pinned.fold(block), "block={index}");
        }
    }

    #[test]
    fn pinning_reproduces_the_sum_the_random_point_would_have_given() {
        // Invariant: splitting the point into a free part and a pinned tail changes nothing.
        //
        // The equality-weighted sum must come out the same either way.
        //
        // Fixture state: 6 variables, the last 3 pinned, the first 3 free.
        //
        //     sum_b eq(r, b) f(b)  ==  sum_{b_out} eq(r_out, b_out) * fold(block b_out)
        let mut rng = SmallRng::seed_from_u64(29);
        let pinned = PinnedEqWeights::geometric(indeterminate(), 3).unwrap();
        let free = (0..3).map(|_| rng.random::<Ghash128>()).collect::<Vec<_>>();

        let values = (0..1 << 6)
            .map(|_| rng.random::<Ghash128>())
            .collect::<Vec<_>>();

        // The direct reading: one equality table over the whole point.
        let whole = free
            .iter()
            .chain(pinned.challenges())
            .copied()
            .collect::<Vec<_>>();
        let direct = Poly::new_from_point(&whole, Ghash128::ONE)
            .as_slice()
            .iter()
            .zip(&values)
            .map(|(&weight, &value)| weight * value)
            .sum::<Ghash128>();

        // The split reading: fold the pinned blocks first, then weigh by the free part.
        let mut folded = Ghash128::zero_vec(1 << 3);
        pinned.fold_blocks(&values, &mut folded);
        let split = Poly::new_from_point(&free, Ghash128::ONE)
            .as_slice()
            .iter()
            .zip(&folded)
            .map(|(&weight, &value)| weight * value)
            .sum::<Ghash128>();

        assert_eq!(direct, split);
    }

    #[test]
    fn a_polynomial_basis_supports_seven_pinned_coordinates() {
        // 128 weights is exactly the dimension of a 128-bit field over the prime field.
        //
        // Seven pinned coordinates is therefore the most that can stay independent.
        assert!(PinnedEqWeights::geometric(indeterminate(), 7).is_ok());
        assert_eq!(
            PinnedEqWeights::geometric(indeterminate(), 8).unwrap_err(),
            PinnedEqError::TooManyPinned {
                count: 8,
                field_bits: 128,
            }
        );
    }

    #[test]
    fn the_independence_check_rejects_a_generator_whose_powers_repeat() {
        // Mutation: a generator of small multiplicative order makes its powers wrap round.
        //
        // The weight set then repeats and collapses.
        //
        // The element one is the extreme case: its powers never move at all.
        assert_eq!(
            PinnedEqWeights::geometric(Ghash128::ONE, 3).unwrap_err(),
            PinnedEqError::DegenerateGenerator { index: 0 }
        );

        // A byte field element cannot have 16 independent powers, since the field holds only 8.
        assert_eq!(
            PinnedEqWeights::<BinaryField8>::geometric(BinaryField8::from_repr(2), 4).unwrap_err(),
            PinnedEqError::TooManyPinned {
                count: 4,
                field_bits: 8,
            }
        );
    }

    #[test]
    fn the_independence_check_rejects_a_dependent_explicit_point() {
        // Mutation: pin every coordinate to zero, so every weight but the first is zero.
        //
        //     eq((0,0), b) = 1 for b = 00, and 0 otherwise
        //
        // Three zero weights are trivially dependent, so this must be refused.
        assert_eq!(
            PinnedEqWeights::from_challenges(alloc::vec![Ghash128::ZERO; 2]).unwrap_err(),
            PinnedEqError::DependentWeights
        );
    }

    #[test]
    fn the_independence_check_accepts_a_basis_and_rejects_a_spanned_element() {
        // Fixture state: the first three powers of the indeterminate, which are basis vectors.
        let independent = [1u128, 2, 4].map(Ghash128::from_repr);
        assert!(prime_field_independent(&independent));

        // Mutation: append the sum of two of them, which the earlier ones already span.
        //
        //     1 + 2 = 3  ->  already in the span of {1, 2, 4}
        let dependent = [1u128, 2, 4, 3].map(Ghash128::from_repr);
        assert!(!prime_field_independent(&dependent));
    }

    #[test]
    fn a_generator_from_a_small_subfield_is_refused() {
        // Independence of the powers is a property of the generator, not of the representation.
        //
        //     {g^k : k < 2^d} independent  <=>  g has degree at least 2^d over the prime field
        //
        // A generator in a small subfield has too few independent powers, in any basis.
        //
        //     from_repr(2) in the tower basis generates the four-element subfield
        //     its square is already spanned, so four weights cannot be independent
        assert_eq!(
            PinnedEqWeights::geometric(BinaryField128::from_repr(2), 2).unwrap_err(),
            PinnedEqError::DependentWeights
        );
        assert_eq!(
            PinnedEqWeights::geometric(BinaryField16::from_repr(2), 2).unwrap_err(),
            PinnedEqError::DependentWeights
        );

        // A full-degree generator works in either representation, so the basis is not the gate.
        //
        // What the basis decides is only whether multiplying by the generator is a shift.
        assert!(PinnedEqWeights::geometric(indeterminate(), 7).is_ok());
        assert!(
            PinnedEqWeights::geometric(BinaryField128::from(indeterminate()), 7).is_ok(),
            "the same element in the tower basis is still full degree"
        );
    }

    /// Pinned coordinates over the tower basis, drawn so their weights stay independent.
    ///
    /// The subfield fold table works with any usable weight set.
    ///
    /// It is exercised over the tower levels, which carry the extension relation it needs.
    fn tower_pinned(count: usize) -> PinnedEqWeights<BinaryField128> {
        let mut rng = SmallRng::seed_from_u64(0xF01D);
        loop {
            let challenges = (0..count)
                .map(|_| rng.random::<BinaryField128>())
                .collect::<Vec<_>>();
            if let Ok(pinned) = PinnedEqWeights::from_challenges(challenges) {
                return pinned;
            }
        }
    }

    #[test]
    fn the_subfield_table_folds_the_same_way_the_weights_do() {
        // Invariant: the table only fuses the subfield embedding into the weight multiplication.
        //
        // Fixture state: 4 pinned coordinates, blocks of 16 byte-field values.
        let mut rng = SmallRng::seed_from_u64(31);
        let pinned = tower_pinned(4);
        let table = SubfieldFoldTable::<_, BinaryField8>::new(&pinned);

        let block = (0..pinned.block_len())
            .map(|_| rng.random::<BinaryField8>())
            .collect::<Vec<_>>();

        // Embedding first and folding with the plain weights must give the same answer.
        let embedded = block
            .iter()
            .map(|&value| BinaryField128::from(value))
            .collect::<Vec<_>>();
        assert_eq!(table.fold(&block), pinned.fold(&embedded));
    }

    #[test]
    fn the_subfield_table_is_sixty_four_kibibytes_at_four_pinned_coordinates() {
        // Fixture state: 16 weights, 256 byte values, 16 bytes per large-field element.
        //
        //     16 * 256 * 16 = 64 KiB, built once at setup
        let table = SubfieldFoldTable::<_, BinaryField8>::new(&tower_pinned(4));
        assert_eq!(table.block_len(), 16);
        assert_eq!(table.table.len() * size_of::<BinaryField128>(), 64 * 1024);
    }

    proptest! {
        #[test]
        fn pinned_coordinates_agree_with_the_equality_polynomial_at_every_width(
            count in 1usize..=7,
        ) {
            // Invariant: whatever the width, the weights are the equality table of the coordinates.
            let pinned = PinnedEqWeights::geometric(indeterminate(), count).unwrap();
            let expected = Poly::new_from_point(pinned.challenges(), Ghash128::ONE);
            prop_assert_eq!(pinned.weights(), expected.as_slice());
        }

        #[test]
        fn the_subfield_table_matches_the_plain_fold_at_every_width(
            count in 1usize..=4,
            seed: u64,
        ) {
            // Invariant: the tabulated fold and the multiplying fold agree at every block width.
            let mut rng = SmallRng::seed_from_u64(seed);
            let pinned = tower_pinned(count);
            let table = SubfieldFoldTable::<_, BinaryField8>::new(&pinned);

            let block = (0..pinned.block_len())
                .map(|_| rng.random::<BinaryField8>())
                .collect::<Vec<_>>();
            let embedded = block.iter().map(|&v| BinaryField128::from(v)).collect::<Vec<_>>();

            prop_assert_eq!(table.fold(&block), pinned.fold(&embedded));
        }
    }
}
