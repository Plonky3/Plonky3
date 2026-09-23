//! Full-width unsigned multiplication, reduced to evaluation claims on its four operand columns.
//!
//! Each multiplication row states one integer identity over words of width `w`.
//!
//! ```text
//!     a * b = lo + 2^w * hi        (as integers)
//! ```
//!
//! A binary field has no integer arithmetic, so both sides are lifted into an exponent.
//!
//! ```text
//!     g         a generator of the challenge field's multiplicative group, of order 2^n - 1
//!
//!     factor(row) = prod_{i, j < w} (1 + a_i * b_j * (g^(2^(i + j)) - 1))  = g^(a * b)
//!     result(row) = prod_{t < 2w}   (1 + c_t * (g^(2^t) - 1))              = g^(lo + 2^w * hi)
//!
//!     c         the 2w bits of lo followed by those of hi
//! ```
//!
//! The reduction proves `factor(row) = result(row)` on every row in four steps.
//!
//! 1. A random row point compares the two lifts at one value, the root.
//! 2. A product check walks the factor tree from that root down to its leaves.
//! 3. A second product check walks the result tree from the same root.
//! 4. Each leaf check ends on evaluations of the committed operand columns.
//!
//! The four evaluations leave as claims at two points the caller must still discharge.
//!
//! This module is the whole relation, so a faster construction replaces it as one unit.
//!
//! # Soundness
//!
//! Equal lifts mean equal exponents modulo `2^n - 1`.
//!
//! ```text
//!     n >= 2w                       checked by both the prover and the verifier
//!     a * b      <= (2^w - 1)^2     <  2^n - 1
//!     lo + 2^w hi <= 2^(2w) - 1    <= 2^n - 1
//! ```
//!
//! Congruent exponents in that range are equal, with one exception.
//!
//! When `n = 2w`, the all-ones limbs `2^n - 1` lift to one, exactly as a zero product does.
//!
//! The caller closes that gap with the low-bit relation `lo_0 = a_0 * b_0`.
//!
//! An odd claimed product then needs two odd factors, whose product is never zero.
//!
//! The statistical error, over a field of `2^n` elements, is a sum of Schwartz-Zippel terms.
//!
//! ```text
//!     row point       m / 2^n
//!     tree of depth D sum_{d < D} (3 * (m + d) + 1) / 2^n  +  3 * (m + D) / 2^n
//!
//!     factor tree     D = 2k
//!     result tree     D = k + 1
//! ```
//!
//! The field check runs at different moments on the two sides.
//!
//! - The prover refuses the field before it commits to anything.
//! - The verifier refuses it at the start of its multiplication step.
//! - A word proof reaches that step after absorbing the commitment and the statement.
//!
//! Either way a small field never yields an accepted proof.
//!
//! The prover performs `O(2^m * w^2)` field operations, dominated by the factor tree leaves.
//!
//! It also holds `O(2^m * w^2)` field elements, about 600 KB per 64-bit product.
//!
//! ```text
//!     factor leaves     w^2 per row, 4096 at w = 64
//!     leaf tables       a, b, and the weights, each repeated over the w^2 leaf cells
//!     layers            one product tree above the leaves
//! ```

mod error;
mod exponent;
mod record;
mod round;
mod transcript;

use alloc::vec::Vec;

pub use error::IntegerMulError;
use exponent::{evaluate, factor_weights, generator_squarings, result_weights};
use p3_challenger::fs::TranscriptField;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;
use p3_security::word::WordIntegerMulSecurityModel;
use p3_sumcheck::generic_degree::RoundProver;
pub use record::IntegerMulProof;
use record::{LayerProof, TreeProof};
use round::{Composite, DEGREE};
use transcript::{ProverTranscript, TranscriptShape, VerifierTranscript};

use crate::{Packed, PackedWord};

/// One of the two product trees the reduction walks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Tree {
    /// The tree whose leaves select `g^(2^(i + j))` by the bit pair `a_i * b_j`.
    Factor,
    /// The tree whose leaves select `g^(2^t)` by bit `t` of the two limbs.
    Result,
}

impl Tree {
    /// Returns the number of layers between the root and the leaves.
    const fn depth(self, bit_variables: usize) -> usize {
        match self {
            // Two bit indices address one leaf.
            Self::Factor => 2 * bit_variables,
            // One index over twice the word width addresses one leaf.
            Self::Result => bit_variables + 1,
        }
    }

    /// Returns the name reported by a depth mismatch.
    const fn name(self) -> &'static str {
        match self {
            Self::Factor => "factor",
            Self::Result => "result",
        }
    }

    /// Returns the transcript label around one layer sumcheck.
    const fn layer_label(self) -> &'static str {
        match self {
            Self::Factor => "factor_layer",
            Self::Result => "result_layer",
        }
    }

    /// Returns the transcript label around the leaf sumcheck.
    const fn leaf_label(self) -> &'static str {
        match self {
            Self::Factor => "factor_leaf",
            Self::Result => "result_leaf",
        }
    }
}

/// One evaluation claim on an operand column over the multiplication rows.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OperandClaim<F> {
    /// Row coordinates followed by within-word coordinates.
    pub(crate) point: Vec<F>,
    /// Claimed evaluation of the column's bits.
    pub(crate) value: F,
}

/// The multiplication reduction of one statement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct IntegerMulReduction {
    /// Variables selecting one padded multiplication row.
    row_variables: usize,
    /// Variables selecting one bit within a word.
    bit_variables: usize,
}

impl IntegerMulReduction {
    /// Creates the reduction for a statement, or nothing when it declares no product.
    pub(crate) const fn new(relations: usize, bit_variables: usize) -> Option<Self> {
        if relations == 0 {
            return None;
        }
        Some(Self {
            row_variables: relations.next_power_of_two().ilog2() as usize,
            bit_variables,
        })
    }

    /// Returns the word width.
    const fn width(&self) -> usize {
        1 << self.bit_variables
    }

    /// Rejects a challenge field whose multiplicative group is too small for the lift.
    ///
    /// # Errors
    ///
    /// Returns an error when the field has fewer than `2^(2w)` elements.
    pub(crate) fn check_field<EF: Field>(&self) -> Result<(), IntegerMulError> {
        // A field of order at least 2^(2w) has a group of order at least 2^(2w) - 1.
        let required = 2 * self.width();
        if EF::order().bits() <= required as u64 {
            return Err(IntegerMulError::FieldTooSmall { required });
        }
        Ok(())
    }

    /// Returns the soundness model of the executed schedule.
    pub(crate) const fn security_model(&self, field_bits: usize) -> WordIntegerMulSecurityModel {
        match WordIntegerMulSecurityModel::new(field_bits, self.row_variables, self.bit_variables) {
            Some(model) => model,
            None => panic!("a nonzero field width gives a valid multiplication model"),
        }
    }

    /// Dimensions bound into the reduction's own transcript.
    const fn transcript_shape(&self) -> TranscriptShape {
        TranscriptShape {
            row_variables: self.row_variables,
            bit_variables: self.bit_variables,
        }
    }

    /// Proves every product and returns claims on the left, right, low, and high columns.
    ///
    /// The field must already have passed [`Self::check_field`].
    pub(crate) fn prove<F, EF, W, C>(
        &self,
        columns: &[Vec<Packed<W>>; 4],
        challenger: &mut C,
    ) -> (IntegerMulProof<F, EF>, [OperandClaim<EF>; 4])
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
        W: PackedWord,
        C: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let (m, k, width) = (self.row_variables, self.bit_variables, self.width());
        let rows = 1 << m;

        // Padding rows are zero products, which lift to one on both sides.
        let words = columns.each_ref().map(|column| {
            (0..rows)
                .map(|row| {
                    column
                        .get(row)
                        .map_or(0, |&packed| W::unpack(packed).to_u64())
                })
                .collect::<Vec<_>>()
        });
        let [left, right, low, high] = &words;
        let powers = generator_squarings::<EF>(2 * width);

        let factor_layers = product_layers(
            factor_leaves(left, right, width, &powers),
            Tree::Factor.depth(k),
        );
        let result_layers = product_layers(
            result_leaves(low, high, width, &powers),
            Tree::Result.depth(k),
        );

        // One row point compares the two lifts, and the factor root is the shared value.
        let mut transcript = ProverTranscript::<C, F, EF>::new(challenger, self.transcript_shape());
        let (row_point, root) = transcript.row_point(m, |point| evaluate(&factor_layers[0], point));

        // Factor tree: the leaf check ends on a(x, i) and b(x, j).
        let (layers, point, claim) = prove_layers(
            &mut transcript,
            Tree::Factor,
            &factor_layers,
            row_point.clone(),
            root,
        );
        let [a, b, p] = factor_leaf_tables::<EF>(left, right, width);
        let mut prover = Composite::new(
            [Point::new(point.as_slice()).equality_weights_msb(), a, b, p],
            |&[eq, a, b, p]| eq * (EF::ONE + a * b * p),
        );
        let rounds = point.len();
        let ((leaf, end), values) = transcript.leaf(Tree::Factor, |challenger| {
            let (proof, end) = prover.prove::<F, _>(challenger, rounds, DEGREE, 0, claim);
            let [_, a, b, _] = prover.terminal();
            ((proof, end), [a, b])
        });
        let (x, bits) = end.as_slice().split_at(m);
        let factor = TreeProof {
            layers,
            leaf,
            values,
        };
        let left_claim = OperandClaim {
            point: [x, &bits[..k]].concat(),
            value: values[0],
        };
        let right_claim = OperandClaim {
            point: [x, &bits[k..]].concat(),
            value: values[1],
        };

        // Result tree: the leaf check ends on c(x, s, t), split into its two limbs.
        let (layers, point, claim) = prove_layers(
            &mut transcript,
            Tree::Result,
            &result_layers,
            row_point,
            root,
        );
        let [c, q] = result_leaf_tables::<EF>(low, high, width);
        let mut prover = Composite::new(
            [Point::new(point.as_slice()).equality_weights_msb(), c, q],
            |&[eq, c, q]| eq * (EF::ONE + c * q),
        );
        let rounds = point.len();
        let low_bits = bit_rows::<EF>(low, width);
        let high_bits = bit_rows::<EF>(high, width);
        let ((leaf, end), values) = transcript.leaf(Tree::Result, |challenger| {
            let (proof, end) = prover.prove::<F, _>(challenger, rounds, DEGREE, 0, claim);

            // The limb selector is the coordinate after the rows, so each limb is read without it.
            let (x, rest) = end.as_slice().split_at(m);
            let limb_point = [x, &rest[1..]].concat();
            let values = [
                evaluate(&low_bits, &limb_point),
                evaluate(&high_bits, &limb_point),
            ];
            ((proof, end), values)
        });
        transcript.finish();
        let (x, rest) = end.as_slice().split_at(m);
        let limb_point = [x, &rest[1..]].concat();
        let result = TreeProof {
            layers,
            leaf,
            values,
        };

        let claims = [
            left_claim,
            right_claim,
            OperandClaim {
                point: limb_point.clone(),
                value: values[0],
            },
            OperandClaim {
                point: limb_point,
                value: values[1],
            },
        ];
        (
            IntegerMulProof {
                root,
                factor,
                result,
            },
            claims,
        )
    }

    /// Verifies every product and returns claims on the left, right, low, and high columns.
    ///
    /// # Errors
    ///
    /// Returns an error when the field is too small, the record is malformed, or a check fails.
    pub(crate) fn verify<F, EF, C>(
        &self,
        proof: &IntegerMulProof<F, EF>,
        challenger: &mut C,
    ) -> Result<[OperandClaim<EF>; 4], IntegerMulError>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
        C: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // The field and both depths are checked before the reduction's own transcript absorbs anything.
        self.check_field::<EF>()?;
        for (tree, record) in [(Tree::Factor, &proof.factor), (Tree::Result, &proof.result)] {
            let expected = tree.depth(self.bit_variables);
            if record.layers.len() != expected {
                return Err(IntegerMulError::TreeDepth {
                    tree: tree.name(),
                    expected,
                    actual: record.layers.len(),
                });
            }
        }

        let mut transcript =
            VerifierTranscript::<C, F, EF>::new(challenger, self.transcript_shape());
        match self.replay(&mut transcript, proof) {
            Ok(claims) => {
                transcript.finish();
                Ok(claims)
            }
            Err(error) => {
                transcript.abort();
                Err(error)
            }
        }
    }

    /// Replays both trees, stopping at the first failed check.
    fn replay<C, F, EF>(
        &self,
        transcript: &mut VerifierTranscript<'_, C, F, EF>,
        proof: &IntegerMulProof<F, EF>,
    ) -> Result<[OperandClaim<EF>; 4], IntegerMulError>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
        C: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let (m, k, width) = (self.row_variables, self.bit_variables, self.width());
        let row_point = transcript.row_point(m, proof.root);

        // Factor tree: the leaf closes on eq * (1 + a * b * (g^(2^(i + j)) - 1)).
        let (point, claim) = verify_layers(
            transcript,
            Tree::Factor,
            &proof.factor.layers,
            row_point.clone(),
            proof.root,
        )?;
        let (end, value) = verify_leaf(transcript, Tree::Factor, &proof.factor, &point, claim)?;
        let [a, b] = proof.factor.values;
        let weight = evaluate(&factor_weights::<EF>(width), &end[m..]);
        if value != Point::eval_eq(&point, &end) * (EF::ONE + a * b * weight) {
            return Err(IntegerMulError::LeafClaim);
        }
        let (x, bits) = end.split_at(m);
        let left_claim = OperandClaim {
            point: [x, &bits[..k]].concat(),
            value: a,
        };
        let right_claim = OperandClaim {
            point: [x, &bits[k..]].concat(),
            value: b,
        };

        // Result tree: the limb selector s mixes the two limbs, c = (1 - s) * lo + s * hi.
        let (point, claim) = verify_layers(
            transcript,
            Tree::Result,
            &proof.result.layers,
            row_point,
            proof.root,
        )?;
        let (end, value) = verify_leaf(transcript, Tree::Result, &proof.result, &point, claim)?;
        let [low, high] = proof.result.values;
        let selector = end[m];
        let limbs = low + selector * (high - low);
        let weight = evaluate(&result_weights::<EF>(width), &end[m..]);
        if value != Point::eval_eq(&point, &end) * (EF::ONE + limbs * weight) {
            return Err(IntegerMulError::LeafClaim);
        }
        let limb_point = [&end[..m], &end[m + 1..]].concat();

        Ok([
            left_claim,
            right_claim,
            OperandClaim {
                point: limb_point.clone(),
                value: low,
            },
            OperandClaim {
                point: limb_point,
                value: high,
            },
        ])
    }
}

/// Builds every layer of a product tree from its leaves, root first.
///
/// ```text
///     V_d(z) = V_{d+1}(z, 0) * V_{d+1}(z, 1)
/// ```
fn product_layers<F: Field>(leaves: Vec<F>, depth: usize) -> Vec<Vec<F>> {
    let mut layers = Vec::with_capacity(depth + 1);
    layers.push(leaves);
    for _ in 0..depth {
        let below = layers.last().expect("the leaves are always present");
        let above = below
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&[zero, one]| zero * one)
            .collect();
        layers.push(above);
    }
    layers.reverse();
    layers
}

/// Runs every layer reduction of one tree, returning the leaf point and claim.
///
/// ```text
///     V_d(r) = sum_z eq(r, z) * V_{d+1}(z, 0) * V_{d+1}(z, 1)
///
///     sumcheck ends at r'     ->  halves h_0 = V_{d+1}(r', 0), h_1 = V_{d+1}(r', 1)
///     line challenge tau      ->  V_{d+1}(r', tau) = h_0 + tau * (h_1 - h_0)
/// ```
fn prove_layers<C, F, EF>(
    transcript: &mut ProverTranscript<'_, C, F, EF>,
    tree: Tree,
    layers: &[Vec<EF>],
    mut point: Vec<EF>,
    mut claim: EF,
) -> (Vec<LayerProof<F, EF>>, Vec<EF>, EF)
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let mut proofs = Vec::with_capacity(layers.len() - 1);
    for below in &layers[1..] {
        // The new variable is the least significant, so halves are even and odd entries.
        let (even, odd): (Vec<EF>, Vec<EF>) = below
            .as_chunks::<2>()
            .0
            .iter()
            .copied()
            .map(|[zero, one]| (zero, one))
            .unzip();
        let mut prover = Composite::new(
            [
                Point::new(point.as_slice()).equality_weights_msb(),
                even,
                odd,
            ],
            |&[eq, zero, one]| eq * zero * one,
        );
        let rounds = point.len();
        let ((sumcheck, end), halves, line) = transcript.layer(tree, |challenger| {
            let (proof, end) = prover.prove::<F, _>(challenger, rounds, DEGREE, 0, claim);
            let [_, zero, one] = prover.terminal();
            ((proof, end), [zero, one])
        });

        claim = halves[0] + line * (halves[1] - halves[0]);
        point = end.as_slice().to_vec();
        point.push(line);
        proofs.push(LayerProof { sumcheck, halves });
    }
    (proofs, point, claim)
}

/// Replays every layer reduction of one tree, returning the leaf point and claim.
fn verify_layers<C, F, EF>(
    transcript: &mut VerifierTranscript<'_, C, F, EF>,
    tree: Tree,
    layers: &[LayerProof<F, EF>],
    mut point: Vec<EF>,
    mut claim: EF,
) -> Result<(Vec<EF>, EF), IntegerMulError>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    for layer in layers {
        // The sum is the claim the previous step left, not one the prover picks.
        if layer.sumcheck.claimed_sum != claim {
            return Err(IntegerMulError::EnteringClaim);
        }
        let rounds = point.len();
        let (replay, line) = transcript.layer(tree, &layer.halves, |challenger| {
            layer.sumcheck.verify(challenger, rounds, DEGREE, 0)
        });
        let (end, value) = replay?;

        // The equality factor is public, so only the two halves are the prover's.
        let [zero, one] = layer.halves;
        if value != Point::eval_eq(&point, end.as_slice()) * zero * one {
            return Err(IntegerMulError::LayerClaim);
        }
        claim = zero + line * (one - zero);
        point = end.as_slice().to_vec();
        point.push(line);
    }
    Ok((point, claim))
}

/// Replays one leaf sumcheck, returning its end point and final value.
fn verify_leaf<C, F, EF>(
    transcript: &mut VerifierTranscript<'_, C, F, EF>,
    tree: Tree,
    record: &TreeProof<F, EF>,
    point: &[EF],
    claim: EF,
) -> Result<(Vec<EF>, EF), IntegerMulError>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    if record.leaf.claimed_sum != claim {
        return Err(IntegerMulError::EnteringClaim);
    }
    let (end, value) = transcript.leaf(tree, &record.values, |challenger| {
        record.leaf.verify(challenger, point.len(), DEGREE, 0)
    })?;
    Ok((end.as_slice().to_vec(), value))
}

/// Returns the factor-tree leaves of every row, addressed `(row, i, j)`.
///
/// Leaf `(i, j)` is `g^(2^(i + j))` when `a_i * b_j = 1`, and one otherwise.
fn factor_leaves<F: Field>(left: &[u64], right: &[u64], width: usize, powers: &[F]) -> Vec<F> {
    let mut leaves = Vec::with_capacity(left.len() * width * width);
    for (&a, &b) in left.iter().zip(right) {
        for i in 0..width {
            leaves.extend((0..width).map(|j| {
                if (a >> i) & (b >> j) & 1 == 1 {
                    powers[i + j]
                } else {
                    F::ONE
                }
            }));
        }
    }
    leaves
}

/// Returns the result-tree leaves of every row, addressed `(row, t)`.
///
/// Leaf `t` is `g^(2^t)` when bit `t` of `lo + 2^w * hi` is set, and one otherwise.
fn result_leaves<F: Field>(low: &[u64], high: &[u64], width: usize, powers: &[F]) -> Vec<F> {
    let mut leaves = Vec::with_capacity(low.len() * 2 * width);
    for (&lo, &hi) in low.iter().zip(high) {
        let limbs = u128::from(lo) | (u128::from(hi) << width);
        leaves.extend((0..2 * width).map(|t| {
            if (limbs >> t) & 1 == 1 {
                powers[t]
            } else {
                F::ONE
            }
        }));
    }
    leaves
}

/// Returns the factor leaf tables `a(x, i)`, `b(x, j)`, and `g^(2^(i + j)) - 1`.
///
/// Each spans the leaf cube `(row, i, j)`, so `a` repeats each bit `w` times.
fn factor_leaf_tables<F: Field>(left: &[u64], right: &[u64], width: usize) -> [Vec<F>; 3] {
    let weights = factor_weights::<F>(width);
    let cells = left.len() * width * width;
    let (mut a, mut b, mut p) = (
        Vec::with_capacity(cells),
        Vec::with_capacity(cells),
        Vec::with_capacity(cells),
    );
    for (&x, &y) in left.iter().zip(right) {
        for i in 0..width {
            for j in 0..width {
                a.push(bit::<F>(x, i));
                b.push(bit::<F>(y, j));
                p.push(weights[i * width + j]);
            }
        }
    }
    [a, b, p]
}

/// Returns the result leaf tables `c(x, s, t)` and `g^(2^t) - 1`.
///
/// The limb selector `s` picks the low limb at zero and the high limb at one.
fn result_leaf_tables<F: Field>(low: &[u64], high: &[u64], width: usize) -> [Vec<F>; 2] {
    let weights = result_weights::<F>(width);
    let cells = low.len() * 2 * width;
    let (mut c, mut q) = (Vec::with_capacity(cells), Vec::with_capacity(cells));
    for (&lo, &hi) in low.iter().zip(high) {
        let limbs = [lo, hi];
        for t in 0..2 * width {
            c.push(bit::<F>(limbs[t / width], t % width));
            q.push(weights[t]);
        }
    }
    [c, q]
}

/// Returns bit `index` of a word as a field element.
fn bit<F: Field>(word: u64, index: usize) -> F {
    F::from_bool((word >> index) & 1 == 1)
}

/// Expands one word per row into its bit table, row major and bit minor.
fn bit_rows<F: Field>(words: &[u64], width: usize) -> Vec<F> {
    words
        .iter()
        .flat_map(|&word| (0..width).map(move |index| bit(word, index)))
        .collect()
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_binary_field::{BinaryChallenger, BinaryField64, BinaryField128};
    use p3_challenger::{CanSample, HashChallenger};
    use p3_field::PrimeCharacteristicRing;
    use p3_keccak::Keccak256Hash;
    use p3_word::{Word32, Word64};
    use proptest::prelude::*;

    use super::*;

    type EF = BinaryField128;
    type Challenger = BinaryChallenger<EF, HashChallenger<u8, Keccak256Hash, 32>>;

    fn challenger() -> Challenger {
        Challenger::from_hasher(Vec::new(), Keccak256Hash)
    }

    // Packs one row per product into the four operand columns.
    fn columns<W: PackedWord>(rows: &[[W; 4]]) -> [Vec<Packed<W>>; 4] {
        core::array::from_fn(|column| rows.iter().map(|row| row[column].pack()).collect())
    }

    // Proves, verifies, and checks every claim against the column it names.
    fn round_trip<W: PackedWord>(rows: &[[W; 4]]) -> Result<(), IntegerMulError> {
        let reduction = IntegerMulReduction::new(rows.len(), W::BITS.ilog2() as usize).unwrap();
        let columns = columns(rows);
        let mut prover = challenger();
        let (proof, claims) = reduction.prove::<EF, EF, W, _>(&columns, &mut prover);
        let mut verifier = challenger();
        let replayed = reduction.verify::<EF, EF, _>(&proof, &mut verifier)?;

        // Both sides end on the same claims and the same sponge state.
        assert_eq!(replayed, claims);
        assert_eq!(
            CanSample::<EF>::sample(&mut prover),
            CanSample::<EF>::sample(&mut verifier)
        );

        // The scalar reference: each claim is its column's bit extension at its point.
        let padded = 1 << reduction.row_variables;
        for (claim, column) in claims.iter().zip(&columns) {
            let words = (0..padded)
                .map(|row| {
                    column
                        .get(row)
                        .map_or(0, |&packed| W::unpack(packed).to_u64())
                })
                .collect::<Vec<_>>();
            let bits = bit_rows::<EF>(&words, W::BITS as usize);
            assert_eq!(evaluate(&bits, &claim.point), claim.value);
        }
        Ok(())
    }

    fn word64_row(a: u64, b: u64) -> [Word64; 4] {
        let product = u128::from(a) * u128::from(b);
        [a, b, product as u64, (product >> 64) as u64].map(Word64::new)
    }

    fn word32_row(a: u32, b: u32) -> [Word32; 4] {
        let product = u64::from(a) * u64::from(b);
        [a, b, product as u32, (product >> 32) as u32].map(Word32::new)
    }

    // One dishonest prover step, each aimed at a single verifier check.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Forgery {
        // The result tree starts from its own lift instead of the shared root.
        ResultTreeFromItsOwnRoot,
        // The false root runs down the result layers, and the leaf restarts from the truth.
        ResultLeafFromItsTrueClaim,
        // The factor tree reads the claimed low limb times one.
        FactorTreeOverTheClaimedProduct,
    }

    // Proves a 64-bit record like the honest prover, except for the one forged step.
    //
    // Every operand value it reports is the true evaluation of its column.
    fn forge(rows: &[[Word64; 4]], forgery: Forgery) -> IntegerMulProof<EF, EF> {
        let reduction = IntegerMulReduction::new(rows.len(), 6).unwrap();
        let (m, k, width) = (
            reduction.row_variables,
            reduction.bit_variables,
            reduction.width(),
        );
        let words: [Vec<u64>; 4] = core::array::from_fn(|column| {
            (0..1 << m)
                .map(|row| rows.get(row).map_or(0, |row| row[column].get()))
                .collect()
        });
        let [left, right, low, high] = &words;
        let ones = vec![1; 1 << m];
        let (tree_left, tree_right) = if forgery == Forgery::FactorTreeOverTheClaimedProduct {
            (low, &ones)
        } else {
            (left, right)
        };
        let powers = generator_squarings::<EF>(2 * width);
        let factor_layers = product_layers(
            factor_leaves(tree_left, tree_right, width, &powers),
            Tree::Factor.depth(k),
        );
        let result_layers = product_layers(
            result_leaves(low, high, width, &powers),
            Tree::Result.depth(k),
        );
        let mut sponge = challenger();
        let mut transcript =
            ProverTranscript::<Challenger, EF, EF>::new(&mut sponge, reduction.transcript_shape());
        let (row_point, root) = transcript.row_point(m, |point| evaluate(&factor_layers[0], point));

        // Factor tree over the forged or honest factors.
        let (layers, point, claim) = prove_layers(
            &mut transcript,
            Tree::Factor,
            &factor_layers,
            row_point.clone(),
            root,
        );
        let [a, b, p] = factor_leaf_tables::<EF>(tree_left, tree_right, width);
        let mut prover = Composite::new(
            [Point::new(point.as_slice()).equality_weights_msb(), a, b, p],
            |&[eq, a, b, p]| eq * (EF::ONE + a * b * p),
        );
        let (leaf, values) = transcript.leaf(Tree::Factor, |challenger| {
            let (proof, end) = prover.prove::<EF, _>(challenger, point.len(), DEGREE, 0, claim);
            let (x, bits) = end.as_slice().split_at(m);
            let values = [
                evaluate(&bit_rows(left, width), &[x, &bits[..k]].concat()),
                evaluate(&bit_rows(right, width), &[x, &bits[k..]].concat()),
            ];
            (proof, values)
        });
        let factor = TreeProof {
            layers,
            leaf,
            values,
        };

        // Result tree, entered from the root, from its own lift, or through forged halves.
        let (layers, point, claim) = match forgery {
            Forgery::ResultTreeFromItsOwnRoot => {
                let own = evaluate(&result_layers[0], &row_point);
                prove_layers(
                    &mut transcript,
                    Tree::Result,
                    &result_layers,
                    row_point,
                    own,
                )
            }
            Forgery::ResultLeafFromItsTrueClaim => {
                let (layers, point) =
                    carry_false_claim(&mut transcript, &result_layers, row_point, root);
                let truth = evaluate(result_layers.last().unwrap(), &point);
                (layers, point, truth)
            }
            Forgery::FactorTreeOverTheClaimedProduct => prove_layers(
                &mut transcript,
                Tree::Result,
                &result_layers,
                row_point,
                root,
            ),
        };
        let [c, q] = result_leaf_tables::<EF>(low, high, width);
        let mut prover = Composite::new(
            [Point::new(point.as_slice()).equality_weights_msb(), c, q],
            |&[eq, c, q]| eq * (EF::ONE + c * q),
        );
        let (leaf, values) = transcript.leaf(Tree::Result, |challenger| {
            let (proof, end) = prover.prove::<EF, _>(challenger, point.len(), DEGREE, 0, claim);
            let (x, rest) = end.as_slice().split_at(m);
            let limb_point = [x, &rest[1..]].concat();
            let values = [
                evaluate(&bit_rows(low, width), &limb_point),
                evaluate(&bit_rows(high, width), &limb_point),
            ];
            (proof, values)
        });
        transcript.finish();
        IntegerMulProof {
            root,
            factor,
            result: TreeProof {
                layers,
                leaf,
                values,
            },
        }
    }

    // Runs every result layer from a false claim, closing each one through its upper half.
    //
    //     v    the value the verifier's replay of the round polynomials reaches
    //     h_0  the true lower half
    //     h_1  v / (eq * h_0), so eq * h_0 * h_1 = v
    fn carry_false_claim(
        transcript: &mut ProverTranscript<'_, Challenger, EF, EF>,
        layers: &[Vec<EF>],
        mut point: Vec<EF>,
        mut claim: EF,
    ) -> (Vec<LayerProof<EF, EF>>, Vec<EF>) {
        let mut proofs = Vec::new();
        for below in &layers[1..] {
            let (even, odd) = below
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&[zero, one]| (zero, one))
                .unzip();
            let mut prover = Composite::new(
                [
                    Point::new(point.as_slice()).equality_weights_msb(),
                    even,
                    odd,
                ],
                |&[eq, zero, one]| eq * zero * one,
            );
            let rounds = point.len();
            let ((sumcheck, end), halves, line) = transcript.layer(Tree::Result, |challenger| {
                // A copy of the sponge replays the rounds exactly as the verifier will.
                let mut replay = (**challenger).clone();
                let (proof, end) = prover.prove::<EF, _>(challenger, rounds, DEGREE, 0, claim);
                let (_, reached) = proof.verify(&mut replay, rounds, DEGREE, 0).unwrap();
                let [_, zero, _] = prover.terminal();
                let one = reached * (Point::eval_eq(&point, end.as_slice()) * zero).inverse();
                ((proof, end), [zero, one])
            });
            claim = halves[0] + line * (halves[1] - halves[0]);
            point = end.as_slice().to_vec();
            point.push(line);
            proofs.push(LayerProof { sumcheck, halves });
        }
        (proofs, point)
    }

    #[test]
    fn the_row_cube_is_padded_to_a_power_of_two() {
        // No product means no reduction at all.
        assert_eq!(IntegerMulReduction::new(0, 6), None);

        // One product needs no row variable, three need two, and four still need two.
        assert_eq!(IntegerMulReduction::new(1, 6).unwrap().row_variables, 0);
        assert_eq!(IntegerMulReduction::new(3, 6).unwrap().row_variables, 2);
        assert_eq!(IntegerMulReduction::new(4, 5).unwrap().row_variables, 2);
    }

    #[test]
    fn a_field_too_small_for_the_product_width_is_refused() {
        // 64-bit words need a field of order at least 2^128.
        let word64 = IntegerMulReduction::new(1, 6).unwrap();
        assert_eq!(word64.check_field::<EF>(), Ok(()));
        assert_eq!(
            word64.check_field::<BinaryField64>(),
            Err(IntegerMulError::FieldTooSmall { required: 128 })
        );

        // 32-bit words fit a field of order 2^64 exactly.
        let word32 = IntegerMulReduction::new(1, 5).unwrap();
        assert_eq!(word32.check_field::<BinaryField64>(), Ok(()));
    }

    #[test]
    fn the_verifier_refuses_a_field_too_small_for_the_product_width() {
        // Fixture state: one 64-bit product, 2^33 * 2^32 = 2^65, claimed as lo = 2, hi = 0.
        //
        //     2^65 = 2 * 2^64 = 2  (mod 2^64 - 1)
        //
        // Over GF(2^64) both lifts are g^2, and the low bit agrees, so only the field check refuses.
        type Small = BinaryField64;
        let small = || BinaryChallenger::<Small, _>::from_hasher(Vec::new(), Keccak256Hash);
        let row = [1 << 33, 1 << 32, 2, 0].map(Word64::new);
        let reduction = IntegerMulReduction::new(1, 6).unwrap();

        // The reduction's prover does not check the field, so it produces a record.
        let (proof, _) = reduction.prove::<Small, Small, Word64, _>(&columns(&[row]), &mut small());

        // The verifier refuses it on its own.
        assert_eq!(
            reduction.verify::<Small, Small, _>(&proof, &mut small()),
            Err(IntegerMulError::FieldTooSmall { required: 128 })
        );
    }

    #[test]
    fn a_result_tree_that_leaves_the_shared_root_is_refused() {
        // Fixture state: the middle high limb is one too large, so the two lifts differ there.
        let mut rows = [
            word64_row(3, 5),
            word64_row(0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210),
            word64_row(7, 9),
        ];
        rows[1][3] = Word64::new(rows[1][3].get() + 1);

        // The forger starts the result tree from its own lift, and is honest everywhere else.
        let proof = forge(&rows, Forgery::ResultTreeFromItsOwnRoot);

        // Only the first result layer's entering claim ties that tree to the root.
        assert_eq!(
            IntegerMulReduction::new(3, 6)
                .unwrap()
                .verify::<EF, EF, _>(&proof, &mut challenger()),
            Err(IntegerMulError::EnteringClaim)
        );
    }

    #[test]
    fn a_result_leaf_that_restarts_from_its_true_claim_is_refused() {
        // Fixture state: the same false product.
        let mut rows = [
            word64_row(3, 5),
            word64_row(0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210),
            word64_row(7, 9),
        ];
        rows[1][3] = Word64::new(rows[1][3].get() + 1);

        // The forger carries the false root through every result layer, closing each by its halves.
        //
        // The leaf sumcheck then restarts from the true leaf evaluation.
        let proof = forge(&rows, Forgery::ResultLeafFromItsTrueClaim);

        // Only the leaf's entering claim ties it to the claim the last layer left.
        assert_eq!(
            IntegerMulReduction::new(3, 6)
                .unwrap()
                .verify::<EF, EF, _>(&proof, &mut challenger()),
            Err(IntegerMulError::EnteringClaim)
        );
    }

    #[test]
    fn a_factor_tree_over_other_factors_is_refused() {
        // Fixture state: 3 * 5 claimed as 17, and 7 * 9 claimed as 63.
        //
        // Both claimed products are odd, so the low bit agrees.
        let rows = [[3, 5, 17, 0], [7, 9, 63, 0]].map(|row| row.map(Word64::new));

        // The forger walks the factor tree over (17, 1) and (63, 1), so both lifts agree.
        //
        // It then reports the true evaluations of a and b at the leaf point.
        let proof = forge(&rows, Forgery::FactorTreeOverTheClaimedProduct);

        // Only the factor leaf check ties the tree to the reported factors.
        assert_eq!(
            IntegerMulReduction::new(2, 6)
                .unwrap()
                .verify::<EF, EF, _>(&proof, &mut challenger()),
            Err(IntegerMulError::LeafClaim)
        );
    }

    #[test]
    fn a_single_product_needs_no_row_point() {
        // Fixture state: the largest squares, whose limbs are both nearly all ones.
        round_trip(&[word64_row(u64::MAX, u64::MAX)]).unwrap();
        round_trip(&[word32_row(u32::MAX, u32::MAX)]).unwrap();
    }

    #[test]
    fn a_zero_product_lifts_to_one_on_both_sides() {
        // Every leaf of a zero factor is one, as is every leaf of a zero result.
        round_trip(&[word64_row(0, u64::MAX), word64_row(u64::MAX, 0)]).unwrap();
    }

    #[test]
    fn a_false_product_is_rejected() {
        // Mutation: the high limb of the second row is off by one.
        let mut rows = [word32_row(7, 9), word32_row(u32::MAX, 3), word32_row(5, 5)];
        rows[1][3] = Word32::new(rows[1][3].get() + 1);
        assert!(round_trip(&rows).is_err());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(8))]

        #[test]
        fn random_word32_products_round_trip(
            rows in prop::collection::vec(any::<(u32, u32)>(), 1..6),
        ) {
            let rows = rows.into_iter().map(|(a, b)| word32_row(a, b)).collect::<Vec<_>>();
            prop_assert_eq!(round_trip(&rows), Ok(()));
        }

        #[test]
        fn random_word64_products_round_trip(
            rows in prop::collection::vec(any::<(u64, u64)>(), 1..4),
        ) {
            let rows = rows.into_iter().map(|(a, b)| word64_row(a, b)).collect::<Vec<_>>();
            prop_assert_eq!(round_trip(&rows), Ok(()));
        }
    }
}
