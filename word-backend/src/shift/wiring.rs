//! Sparse evaluation of the public shift wiring.

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::Gf2;
use p3_field::{Algebra, Field};
use p3_multilinear_util::poly::Poly;
#[cfg(test)]
use p3_word::ConstraintSystem;
use p3_word::{ConstraintKind, OperandRole, Shift, Word};

use super::ShiftClaim;
use super::polynomial::SumOfProducts;
use super::transcript::{BatchWeights, equality_weights};
use crate::{CompiledKey, CompiledKeyLayout, CompiledSegment, PackedWitness, PackedWord};

/// Equality and batching tables shared by both proving phases.
pub(super) struct PreparedWeights<F> {
    /// One equality weight per padded constraint row.
    constraint: Vec<F>,
    /// One equality weight per relation-family slot.
    operation: Vec<F>,
    /// One equality weight per operand-position slot.
    operand: Vec<F>,
}

impl<F: Field> PreparedWeights<F> {
    /// Expands the statement point and retains the transcript batching tables.
    pub(super) fn new(claim: &ShiftClaim<F>, batch: BatchWeights<F>) -> Self {
        // Every compiled reference reads one entry from each table.
        Self {
            constraint: equality_weights(claim.constraint_point()),
            operation: batch.operation,
            operand: batch.operand,
        }
    }

    /// Returns the coefficient carried by one compiled relation occurrence.
    fn reference<W: Word>(&self, key: &CompiledKey<'_, W>, reference: usize) -> F {
        // The compiled role is translated into its family-local operand position.
        let reference = key.references()[reference];
        let operation = key.operation().code() as usize;
        let operand = operand_index(key.operation(), reference.operand());
        self.operation[operation]
            * self.operand[operand]
            * self.constraint[reference.constraint() as usize]
    }

    /// Sums all relation occurrences grouped under one compiled key.
    fn key<W: Word>(&self, key: &CompiledKey<'_, W>) -> F {
        // Repeated terms add in the challenge field and cancel in characteristic two.
        (0..key.references().len())
            .map(|reference| self.reference(key, reference))
            .sum()
    }

    /// Batches the supplied operand claims with these transcript weights.
    pub(super) fn claim(&self, claim: &ShiftClaim<F>) -> F {
        // Centralizing the order prevents claim batching and sparse wiring from diverging.
        claim.batched(&self.operation, &self.operand)
    }
}

/// Returns a role's position in the operand batching cube.
fn operand_index(kind: ConstraintKind, role: OperandRole) -> usize {
    // Each family uses the semantic order exposed by its relation type.
    match (kind, role) {
        (ConstraintKind::Zero, OperandRole::Value)
        | (ConstraintKind::And, OperandRole::Left)
        | (ConstraintKind::IntegerMul, OperandRole::Left) => 0,
        (ConstraintKind::And, OperandRole::Right)
        | (ConstraintKind::IntegerMul, OperandRole::Right) => 1,
        (ConstraintKind::And, OperandRole::Output)
        | (ConstraintKind::IntegerMul, OperandRole::Low) => 2,
        (ConstraintKind::IntegerMul, OperandRole::High) => 3,
        _ => unreachable!("a checked relation never pairs with a foreign operand role"),
    }
}

/// Evaluates one word's bit multilinear at a point.
pub(super) fn evaluate_word<W: Word, F: Field>(word: W, weights: &[F]) -> F {
    // Set bits select the equality weights that survive the Boolean word.
    let bits = word.to_u64();
    (0..W::BITS as usize)
        .filter(|&bit| (bits >> bit) & 1 == 1)
        .map(|bit| weights[bit])
        .sum()
}

/// Applies the transpose of one fixed shift to a bit-evaluation table.
fn shift_transpose<W: Word, F: Field>(shift: Shift<W>, output: &[F]) -> Vec<F> {
    // An identity preserves the table and needs no basis walk.
    if shift.is_identity() {
        return output.to_vec();
    }

    // Column `j` records where input basis bit `j` reaches after this shift.
    (0..W::BITS as usize)
        .map(|input| {
            let basis = W::from_low_bits(1_u64 << input);
            let shifted = shift.apply(basis).to_u64();
            (0..W::BITS as usize)
                .filter(|&bit| (shifted >> bit) & 1 == 1)
                .map(|bit| output[bit])
                .sum()
        })
        .collect()
}

/// Builds the factorized transpose action of one two-slot shift sequence.
fn shift_weights<W: Word, F: Field>(shifts: [Shift<W>; 2], output: &[F]) -> Vec<F> {
    // If the forward map is `outer(inner(word))`, its transpose is
    // `inner^T(outer^T(output))`. Keeping the two slots separate avoids materializing a dense
    // composed operator and makes their protocol order explicit.
    let [inner, outer] = shifts;
    let intermediate = shift_transpose(outer, output);
    shift_transpose(inner, &intermediate)
}

/// Returns the public-segment share of the batched operand claim.
pub(super) fn public_contribution<W, F>(
    segment: &CompiledSegment<W>,
    words: &[W],
    prepared: &PreparedWeights<F>,
    bit_point: &[F],
) -> F
where
    W: Word,
    F: Field + Algebra<Gf2>,
{
    // One equality table serves every shifted public word.
    let output = equality_weights(bit_point);
    let mut contribution = F::ZERO;

    for (word_index, &word) in words.iter().enumerate() {
        // Every key fixes one shift sequence and one relation family.
        let keys = segment
            .keys(word_index)
            .expect("the checked public segment contains this word");
        for key in keys {
            let coefficient = prepared.key(&key);
            if coefficient == F::ZERO {
                continue;
            }

            // Applying the fixed shifts before evaluation is the public reference path.
            let [inner, outer] = key.shifts();
            let shifted = outer.apply(inner.apply(word));
            contribution += coefficient * evaluate_word(shifted, &output);
        }
    }
    contribution
}

/// Constructs the sparse bit-index sumcheck and its shift rows.
pub(super) fn bit_prover<W, F>(
    layout: &CompiledKeyLayout<W>,
    values: &PackedWitness<W>,
    prepared: &PreparedWeights<F>,
    bit_point: &[F],
) -> SumOfProducts<F>
where
    W: PackedWord,
    F: Field + Algebra<Gf2>,
{
    // One row is retained per shift spelling used by the committed segment.
    let segment = layout.witness();
    let sequences = segment.shift_sequences();
    let mut witness_rows = (0..sequences.len())
        .map(|_| F::zero_vec(W::BITS as usize))
        .collect::<Vec<_>>();

    for (word_index, &packed) in values.witness().iter().enumerate() {
        // Source words remain bit-packed while coefficients are scattered into sparse rows.
        let bits = W::unpack(packed).to_u64();
        let keys = segment
            .keys(word_index)
            .expect("the checked witness segment contains this word");
        for key in keys {
            let coefficient = prepared.key(&key);
            if coefficient == F::ZERO {
                continue;
            }
            let row = &mut witness_rows[key.shift_index() as usize];
            let mut remaining = bits;
            while remaining != 0 {
                // Visit only set lanes; random 64-bit words halve the scatter work.
                let bit = remaining.trailing_zeros() as usize;
                row[bit] += coefficient;
                remaining &= remaining - 1;
            }
        }
    }

    // The other factor is the transpose action of each fixed shift sequence.
    let output = equality_weights(bit_point);
    let pairs = witness_rows
        .into_iter()
        .zip(sequences)
        .map(|(witness, &shifts)| {
            [
                Poly::new(witness),
                Poly::new(shift_weights(shifts, &output)),
            ]
        })
        .collect();
    SumOfProducts::new(pairs, W::BITS.ilog2() as usize)
}

/// Builds the word-index sumcheck after the bit axis is bound.
pub(super) fn word_prover<W, F>(
    layout: &CompiledKeyLayout<W>,
    values: &PackedWitness<W>,
    prepared: &PreparedWeights<F>,
    bit_point: &[F],
    shift_evaluations: &[F],
) -> SumOfProducts<F>
where
    W: PackedWord,
    F: Field + Algebra<Gf2>,
{
    // The bit point folds each packed source word directly to one challenge-field element.
    let bit_weights = equality_weights(bit_point);
    let mut witness = values
        .witness()
        .iter()
        .map(|&packed| evaluate_word(W::unpack(packed), &bit_weights))
        .collect::<Vec<_>>();

    // The public wiring contributes one scalar per committed word.
    let mut wiring = (0..values.witness().len())
        .map(|word| word_weight(layout.witness(), word, prepared, shift_evaluations))
        .collect::<Vec<_>>();

    // Both tables share the smallest power-of-two cube covering the committed segment.
    let capacity = witness.len().max(1).next_power_of_two();
    witness.resize(capacity, F::ZERO);
    wiring.resize(capacity, F::ZERO);
    SumOfProducts::new(
        vec![[Poly::new(witness), Poly::new(wiring)]],
        capacity.ilog2() as usize,
    )
}

/// Evaluates the committed wiring polynomial at a word point.
pub(super) fn wiring_evaluation<W, F>(
    layout: &CompiledKeyLayout<W>,
    prepared: &PreparedWeights<F>,
    shift_evaluations: &[F],
    word_point: &[F],
) -> F
where
    W: Word,
    F: Field + Algebra<Gf2>,
{
    // Missing words in the final power-of-two domain carry zero wiring weight.
    let word_weights = equality_weights(word_point);
    (0..layout.witness().len())
        .map(|word| {
            word_weights[word] * word_weight(layout.witness(), word, prepared, shift_evaluations)
        })
        .sum()
}

/// Accumulates every compiled key attached to one source word.
fn word_weight<W: Word, F: Field>(
    segment: &CompiledSegment<W>,
    word: usize,
    prepared: &PreparedWeights<F>,
    shift_evaluations: &[F],
) -> F {
    // A key coefficient factorizes into wiring, batching, and shift evaluations.
    segment
        .keys(word)
        .expect("the checked segment contains this word")
        .map(|key| prepared.key(&key) * shift_evaluations[key.shift_index() as usize])
        .sum()
}

/// Evaluates every fixed shift operator at a sampled input-bit point.
pub(super) fn shift_evaluations<W, F>(
    layout: &CompiledKeyLayout<W>,
    output_point: &[F],
    input_point: &[F],
) -> Vec<F>
where
    W: Word,
    F: Field,
{
    // Each small operator row is evaluated independently at the same input point.
    let output = equality_weights(output_point);
    let input = equality_weights(input_point);
    layout
        .witness()
        .shift_sequences()
        .iter()
        .map(|&shifts| {
            shift_weights(shifts, &output)
                .into_iter()
                .zip(&input)
                .map(|(value, &weight)| value * weight)
                .sum()
        })
        .collect()
}

/// Computes claims directly from a checked witness for reduction tests.
#[cfg(test)]
pub(super) fn evaluate_claim<W, F>(
    system: &ConstraintSystem<W>,
    values: &PackedWitness<W>,
    constraint_point: Vec<F>,
    bit_point: Vec<F>,
) -> ShiftClaim<F>
where
    W: PackedWord,
    F: Field + Algebra<Gf2>,
{
    // Constraint rows are zero-padded to the widest relation family.
    let constraint_weights = equality_weights(&constraint_point);
    let bit_weights = equality_weights(&bit_point);
    let public = values
        .public()
        .iter()
        .copied()
        .map(W::unpack)
        .collect::<Vec<_>>();
    let witness = values
        .witness()
        .iter()
        .copied()
        .map(W::unpack)
        .collect::<Vec<_>>();
    let evaluate = |operand: &p3_word::Operand<W>| {
        let word = operand
            .evaluate(&public, &witness)
            .expect("a checked system contains only in-bounds terms");
        evaluate_word(word, &bit_weights)
    };
    // Each semantic column is evaluated independently at the shared points.
    let zero = [system
        .zero_constraints()
        .iter()
        .enumerate()
        .map(|(row, relation)| constraint_weights[row] * evaluate(relation.value()))
        .sum()];
    let mut bitwise_and = [F::ZERO; 3];
    for (row, relation) in system.and_constraints().iter().enumerate() {
        let weight = constraint_weights[row];
        bitwise_and[0] += weight * evaluate(relation.left());
        bitwise_and[1] += weight * evaluate(relation.right());
        bitwise_and[2] += weight * evaluate(relation.output());
    }

    let mut integer_mul = [F::ZERO; 4];
    for (row, relation) in system.integer_mul_constraints().iter().enumerate() {
        let weight = constraint_weights[row];
        integer_mul[0] += weight * evaluate(relation.left());
        integer_mul[1] += weight * evaluate(relation.right());
        integer_mul[2] += weight * evaluate(relation.low());
        integer_mul[3] += weight * evaluate(relation.high());
    }
    ShiftClaim::new(constraint_point, bit_point, zero, bitwise_and, integer_mul)
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_binary_field::{BinaryField128, TowerLevel};
    use p3_word::{ShiftKind, Word64};

    use super::*;

    type F = BinaryField128;

    /// Enumerates every distinct shift accepted by a 64-bit word.
    fn shifts() -> Vec<Shift<Word64>> {
        // Distance zero normalizes to one identity, so the other seven spellings are dropped.
        (0..8)
            .flat_map(|code| {
                let kind = ShiftKind::from_code(code).expect("codes below eight are assigned");
                let width = if kind.is_lane32() { 32 } else { 64 };
                let first = usize::from(code != 0);
                (first..width).map(move |amount| Shift::new(kind, amount).unwrap())
            })
            .collect()
    }

    /// Computes the transpose table by applying both slots directly to every basis bit.
    fn direct_weights(shifts: [Shift<Word64>; 2], output: &[F]) -> Vec<F> {
        (0..64)
            .map(|input| {
                let basis = Word64::from_low_bits(1_u64 << input);
                let shifted = shifts[1].apply(shifts[0].apply(basis)).to_u64();
                (0..64)
                    .filter(|&bit| (shifted >> bit) & 1 == 1)
                    .map(|bit| output[bit])
                    .sum()
            })
            .collect()
    }

    /// Compares both transpose paths over the supplied slot pairs.
    fn check_pairs(pairs: impl Iterator<Item = [Shift<Word64>; 2]>) {
        // Distinct coefficients expose bit order, lane boundaries, and sign-extension fan-out.
        let output = (0..64)
            .map(|bit| F::from_repr(1_u128 << bit))
            .collect::<Vec<_>>();
        for sequence in pairs {
            assert_eq!(
                shift_weights(sequence, &output),
                direct_weights(sequence, &output)
            );
        }
    }

    #[test]
    fn factorized_transpose_matches_a_sampled_shift_pair_grid() {
        // Strides coprime to the count visit every shift in both slots, three pairings each.
        let shifts = shifts();
        let count = shifts.len();
        let mut pairs = Vec::with_capacity(3 * count);
        for stride in [1_usize, 31, 137] {
            for inner in 0..count {
                pairs.push([shifts[inner], shifts[(inner * stride + 1) % count]]);
            }
        }
        check_pairs(pairs.into_iter());
    }

    #[test]
    #[ignore = "the full square is 142k pairs and dominates the default test profile"]
    fn factorized_transpose_matches_every_shift_pair() {
        // Exhausting both slots catches reversed composition and every legal kind and amount.
        let shifts = shifts();
        check_pairs(
            shifts
                .iter()
                .flat_map(|&inner| shifts.iter().map(move |&outer| [inner, outer])),
        );
    }
}
