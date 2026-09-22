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
//!     n >= 2w                       checked before anything is absorbed
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
//! The prover performs `O(2^m * w^2)` field operations, dominated by the factor tree leaves.

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

        // Factor leaves are addressed (row, i, j), result leaves (row, t).
        let mut factor_leaves = Vec::with_capacity(rows * width * width);
        let mut result_leaves = Vec::with_capacity(rows * 2 * width);
        for row in 0..rows {
            for i in 0..width {
                let a = (left[row] >> i) & 1;
                factor_leaves.extend((0..width).map(|j| {
                    if a & (right[row] >> j) & 1 == 1 {
                        powers[i + j]
                    } else {
                        EF::ONE
                    }
                }));
            }
            let limbs = u128::from(low[row]) | (u128::from(high[row]) << width);
            result_leaves.extend((0..2 * width).map(|t| {
                if (limbs >> t) & 1 == 1 {
                    powers[t]
                } else {
                    EF::ONE
                }
            }));
        }
        let factor_layers = product_layers(factor_leaves, Tree::Factor.depth(k));
        let result_layers = product_layers(result_leaves, Tree::Result.depth(k));

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
        let weights = factor_weights::<EF>(width);
        let cells = rows * width * width;
        let (mut a, mut b, mut p) = (
            Vec::with_capacity(cells),
            Vec::with_capacity(cells),
            Vec::with_capacity(cells),
        );
        for row in 0..rows {
            for i in 0..width {
                for j in 0..width {
                    a.push(bit::<EF>(left[row], i));
                    b.push(bit::<EF>(right[row], j));
                    p.push(weights[i * width + j]);
                }
            }
        }
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
        let weights = result_weights::<EF>(width);
        let cells = rows * 2 * width;
        let (mut c, mut q) = (Vec::with_capacity(cells), Vec::with_capacity(cells));
        for row in 0..rows {
            let limbs = [low[row], high[row]];
            for t in 0..2 * width {
                c.push(bit::<EF>(limbs[t / width], t % width));
                q.push(weights[t]);
            }
        }
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
        // Every shape is checked before the transcript absorbs a single value.
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
    use p3_binary_field::{BinaryChallenger, BinaryField64, BinaryField128};
    use p3_challenger::{CanSample, HashChallenger};
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
