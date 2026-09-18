//! Proof construction and verification for batched product-tree GKR.

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::FieldChallenger;
use p3_challenger::fs::TranscriptField;
use p3_field::ExtensionField;
use p3_sumcheck::generic_degree::RoundPolyInterpolator;
use serde::{Deserialize, Serialize};

use crate::transcript::{ProductGkrProverTranscript, ProductGkrVerifierTranscript};

use super::math::{
    combine, equality_evaluation, equality_weights, fold_dense, has_distinct_round_nodes,
    interpolate_pair, interpolate_quad,
};
use super::prover::{ProductLayers, RadixFourBatch};
use super::{ProductGkrError, ProductGkrShape};

/// Number of transmitted evaluations for a degree-five round polynomial.
pub(crate) const ROUND_POLY_LEN: usize = 5;

/// Prover messages for one product-tree layer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProductGkrLayerProof<EF> {
    /// One root-most binary layer for an odd-height tree.
    Binary {
        /// Two child evaluations for every batched tree.
        children: Vec<[EF; 2]>,
    },
    /// Two multiplication levels contracted into one layer.
    RadixFour {
        /// Degree-five sumcheck messages in binding order.
        round_polys: Vec<[EF; ROUND_POLY_LEN]>,
        /// Four child evaluations for every batched tree.
        children: Vec<[EF; 4]>,
    },
}

/// Proof of several identity-padded product trees reduced at one point.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProductGkrProof<EF> {
    /// Root messages encoded according to the statement shape.
    pub roots: Vec<EF>,
    /// Layer messages in root-to-leaf order.
    pub layers: Vec<ProductGkrLayerProof<EF>>,
}

/// Unauthenticated leaf evaluations produced by a product reduction.
#[must_use = "leaf evaluations are unauthenticated until bound to committed polynomials"]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProductGkrOutput<EF> {
    /// One expanded root value per product tree.
    pub roots: Vec<EF>,
    /// Shared multilinear point in most-significant-variable-first order.
    pub point: Vec<EF>,
    /// Evaluations of the logical tables after their explicit prefixes are padded with ones.
    pub values: Vec<EF>,
}

impl<EF> ProductGkrProof<EF> {
    /// Proves several product trees in one transcript.
    ///
    /// Each slice is an arbitrary leaf prefix.
    /// Every omitted suffix value is the multiplicative identity.
    ///
    /// An output value for a prefix of length `n` is
    /// `sum_(i < n) eq(point, i) * input[i] + sum_(i >= n) eq(point, i)`.
    /// A caller must authenticate them against committed leaf polynomials.
    ///
    /// # Panics
    ///
    /// Panics when the input count or a prefix length disagrees with the supplied statement shape.
    /// Panics when a shared-root statement is false.
    pub fn prove<F, Challenger>(
        inputs: &[&[EF]],
        shape: ProductGkrShape,
        challenger: &mut Challenger,
    ) -> (Self, ProductGkrOutput<EF>)
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F>,
    {
        // Prover input errors are programming errors.
        // The verifier handles the corresponding proof errors without panicking.
        assert!(
            has_distinct_round_nodes::<EF>(),
            "product GKR requires six distinct challenge-field interpolation nodes"
        );
        assert_eq!(
            inputs.len(),
            shape.num_trees,
            "product GKR tree count mismatch"
        );
        let capacity = 1usize << shape.log_height;
        for input in inputs {
            assert!(
                input.len() <= capacity,
                "product GKR input exceeds its logical tree"
            );
        }

        // Keep only explicit prefixes at every product level.
        let all_layers = inputs
            .iter()
            .map(|input| ProductLayers::new(input, shape.log_height))
            .collect::<Vec<_>>();
        let roots = all_layers
            .iter()
            .map(ProductLayers::root)
            .collect::<Vec<_>>();
        let root_messages = shape
            .root_shape
            .encode(&roots)
            .expect("a shared-root statement requires equal roots");

        // The typed transcript binds both the dimensions and every prover message.
        let mut transcript =
            ProductGkrProverTranscript::<Challenger, F, EF>::new(challenger, shape, &root_messages);
        let mut point = Vec::with_capacity(shape.log_height);
        let mut values = roots.clone();
        let mut layer_proofs = Vec::with_capacity(shape.layers().len());
        let mut product_depth = shape.log_height;

        for (arity, round_count) in shape.layers() {
            let batching = transcript.begin_layer();
            debug_assert_eq!(round_count, point.len());
            let child_depth = product_depth - arity.trailing_zeros() as usize;

            if arity == 2 {
                // The root-most binary layer has no parent variables to sum over.
                let children = all_layers
                    .iter()
                    .map(|layers| layers.binary_children(child_depth))
                    .collect::<Vec<_>>();
                let branches = transcript.end_binary_layer(&children);
                values = children
                    .iter()
                    .map(|children| interpolate_pair(*children, branches[0]))
                    .collect();
                point = vec![branches[0]];
                layer_proofs.push(ProductGkrLayerProof::Binary { children });
            } else {
                // Four child tables share one eq-weighted degree-five sumcheck.
                let mut batch =
                    RadixFourBatch::new(&all_layers, child_depth, 1usize << round_count);
                let mut equality = equality_weights(&point);
                let mut logical_len = 1usize << round_count;
                let mut round_point = Vec::with_capacity(round_count);
                let mut round_polys = Vec::with_capacity(round_count);

                for _ in 0..round_count {
                    let round_poly = batch.round(&equality, logical_len, batching);
                    let challenge = transcript.round(&round_poly);
                    batch.fold(challenge, logical_len);
                    fold_dense(&mut equality, challenge);
                    logical_len /= 2;
                    round_point.push(challenge);
                    round_polys.push(round_poly);
                }

                let children = batch.children();
                let branches = transcript.end_radix_four_layer(&children);
                values = children
                    .iter()
                    .map(|children| interpolate_quad(*children, branches))
                    .collect();
                point = vec![branches[0], branches[1]];
                point.extend(round_point);
                layer_proofs.push(ProductGkrLayerProof::RadixFour {
                    round_polys,
                    children,
                });
            }

            product_depth = child_depth;
        }

        transcript.finish();

        // Internal folds bind low-order address bits first.
        // The public convention addresses subcubes with leading coordinates.
        point.reverse();

        (
            Self {
                roots: root_messages,
                layers: layer_proofs,
            },
            ProductGkrOutput {
                roots,
                point,
                values,
            },
        )
    }

    /// Verifies the internal consistency of several product trees.
    ///
    /// All counts come from the supplied statement shape.
    /// Attacker-controlled proof lengths are checked before transcript replay.
    ///
    /// The returned leaf evaluations remain unauthenticated.
    /// The surrounding protocol must tie them to committed polynomials.
    /// A distinct-root statement checks no relation between different roots.
    ///
    /// Product soundness is statistical rather than a fixed property of this primitive.
    /// The caller must choose enough challenge-field bits for every documented error term.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed proofs or a failed product relation.
    pub fn verify<F, Challenger>(
        &self,
        shape: ProductGkrShape,
        challenger: &mut Challenger,
    ) -> Result<ProductGkrOutput<EF>, ProductGkrError>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F>,
    {
        // Reject every attacker-controlled length before building a strict transcript driver.
        if !has_distinct_round_nodes::<EF>() {
            return Err(ProductGkrError::ChallengeFieldTooSmall);
        }
        self.validate_shape(shape)?;
        let roots = shape.root_shape.decode(&self.roots);
        let mut transcript =
            ProductGkrVerifierTranscript::<Challenger, F, EF>::new(challenger, shape, &self.roots);
        let mut values = roots.clone();
        let mut point = Vec::with_capacity(shape.log_height);
        let mut inconsistent_layer = None;
        let interpolator = RoundPolyInterpolator::new(5);

        for (layer_index, ((arity, _), layer)) in
            shape.layers().into_iter().zip(&self.layers).enumerate()
        {
            let batching = transcript.begin_layer();
            let mut running_sum = combine(&values, batching);

            match layer {
                ProductGkrLayerProof::Binary { children } => {
                    debug_assert_eq!(arity, 2);
                    let branches = transcript.end_binary_layer(children);
                    let expected = combine(
                        &children
                            .iter()
                            .map(|children| children[0] * children[1])
                            .collect::<Vec<_>>(),
                        batching,
                    );
                    if running_sum != expected {
                        inconsistent_layer.get_or_insert(layer_index);
                    }
                    values = children
                        .iter()
                        .map(|children| interpolate_pair(*children, branches[0]))
                        .collect();
                    point = vec![branches[0]];
                }
                ProductGkrLayerProof::RadixFour {
                    round_polys,
                    children,
                } => {
                    debug_assert_eq!(arity, 4);
                    let mut round_point = Vec::with_capacity(round_polys.len());
                    for round_poly in round_polys {
                        let challenge = transcript.round(round_poly);
                        running_sum = interpolator.eval(round_poly, running_sum, challenge);
                        round_point.push(challenge);
                    }
                    let expected = equality_evaluation(&point, &round_point)
                        * combine(
                            &children
                                .iter()
                                .map(|children| children.iter().copied().product())
                                .collect::<Vec<_>>(),
                            batching,
                        );
                    if running_sum != expected {
                        inconsistent_layer.get_or_insert(layer_index);
                    }
                    let branches = transcript.end_radix_four_layer(children);
                    values = children
                        .iter()
                        .map(|children| interpolate_quad(*children, branches))
                        .collect();
                    point = vec![branches[0], branches[1]];
                    point.extend(round_point);
                }
            }
        }

        transcript.finish();

        if let Some(layer) = inconsistent_layer {
            return Err(ProductGkrError::LayerConsistency { layer });
        }

        // Match the repository-wide most-significant-variable-first point convention.
        point.reverse();

        Ok(ProductGkrOutput {
            roots,
            point,
            values,
        })
    }

    /// Check the canonical proof layout before transcript replay.
    fn validate_shape(&self, shape: ProductGkrShape) -> Result<(), ProductGkrError> {
        if self.roots.len() != shape.root_message_len() {
            return Err(ProductGkrError::RootCountMismatch {
                expected: shape.root_message_len(),
                actual: self.roots.len(),
            });
        }

        let layers = shape.layers();
        if self.layers.len() != layers.len() {
            return Err(ProductGkrError::LayerCountMismatch {
                expected: layers.len(),
                actual: self.layers.len(),
            });
        }

        for (layer_index, ((arity, rounds), layer)) in layers.iter().zip(&self.layers).enumerate() {
            let valid = match (arity, layer) {
                (2, ProductGkrLayerProof::Binary { children }) => children.len() == shape.num_trees,
                (
                    4,
                    ProductGkrLayerProof::RadixFour {
                        round_polys,
                        children,
                    },
                ) => round_polys.len() == *rounds && children.len() == shape.num_trees,
                _ => false,
            };
            if !valid {
                return Err(ProductGkrError::MalformedLayer { layer: layer_index });
            }
        }

        Ok(())
    }
}
