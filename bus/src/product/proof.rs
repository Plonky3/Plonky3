//! Proof construction and verification for batched product-tree GKR.

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::FieldChallenger;
use p3_challenger::fs::TranscriptField;
use p3_field::{ExtensionField, Field};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_sumcheck::generic_degree::RoundPolyInterpolator;
use serde::{Deserialize, Serialize};

use super::math::{combine, has_distinct_round_nodes, interpolate_pair, interpolate_quad};
use super::prover::{ProductLayers, RadixFourBatch};
use super::transcript::{ProductGkrProverTranscript, ProductGkrVerifierTranscript};
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

/// Product layers and roots bound to one validated statement shape.
///
/// Borrowed leaves stay borrowed until the first fold of the leaf layer.
pub(crate) struct PreparedProductGkr<'a, EF: Clone> {
    shape: ProductGkrShape,
    all_layers: Vec<ProductLayers<'a, EF>>,
    roots: Vec<EF>,
}

impl<'a, EF: Field> PreparedProductGkr<'a, EF> {
    /// Prepare layers without asserting the shared-root relation or touching a transcript.
    pub(crate) fn new(inputs: &[&'a [EF]], shape: ProductGkrShape) -> Self {
        Self::validate_lengths(inputs.iter().map(|input| input.len()), shape);

        // Keep only explicit prefixes at every product level.
        // The leaves stay borrowed, and a leaf moves fewer than two elements to be reduced.
        // Each tree is charged the average, so the gate sees every tree's bytes together.
        let tree_bytes = inputs
            .iter()
            .map(|input| input.len() * 2 * size_of::<EF>())
            .sum::<usize>()
            .checked_div(inputs.len())
            .unwrap_or(0);
        let all_layers = inputs
            .par_iter()
            .map_collect_min_task_bytes(tree_bytes, |input| {
                ProductLayers::new(input, shape.log_height)
            });
        Self::from_layers(shape, all_layers)
    }

    /// Consume the bus's owned leaves without copying their explicit prefixes.
    pub(crate) fn new_owned(mut inputs: [Vec<EF>; 2], shape: ProductGkrShape) -> Self {
        Self::validate_lengths(inputs.iter().map(Vec::len), shape);
        // No leaf copy is needed; only the retained levels move field values.
        let tree_bytes = inputs
            .iter()
            .map(|input| input.len() * 2 * size_of::<EF>())
            .sum::<usize>()
            .checked_div(inputs.len())
            .unwrap_or(0);
        let all_layers = inputs
            .par_iter_mut()
            .map_collect_min_task_bytes(tree_bytes, |input| {
                ProductLayers::new_owned(core::mem::take(input), shape.log_height)
            });
        Self::from_layers(shape, all_layers)
    }

    fn validate_lengths(lengths: impl ExactSizeIterator<Item = usize>, shape: ProductGkrShape) {
        assert_eq!(
            lengths.len(),
            shape.num_trees,
            "product GKR tree count mismatch"
        );
        let capacity = 1usize << shape.log_height;
        for length in lengths {
            assert!(
                length <= capacity,
                "product GKR input exceeds its logical tree"
            );
        }
    }

    fn from_layers(shape: ProductGkrShape, all_layers: Vec<ProductLayers<'a, EF>>) -> Self {
        let roots = all_layers
            .iter()
            .map(ProductLayers::root)
            .collect::<Vec<_>>();
        Self {
            shape,
            all_layers,
            roots,
        }
    }

    pub(crate) fn roots(&self) -> &[EF] {
        &self.roots
    }

    /// Prove a statement whose roots have already been checked by the bus reduction.
    pub(crate) fn prove<F, Challenger>(
        self,
        challenger: &mut Challenger,
    ) -> (ProductGkrProof<EF>, ProductGkrOutput<EF>)
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
        ProductGkrProof::prove_inner::<F, Challenger>(self, challenger)
    }
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
        Self::prove_inner::<F, Challenger>(PreparedProductGkr::new(inputs, shape), challenger)
    }

    /// Run the transcript after the statement and product layers have been checked.
    fn prove_inner<F, Challenger>(
        prepared: PreparedProductGkr<'_, EF>,
        challenger: &mut Challenger,
    ) -> (Self, ProductGkrOutput<EF>)
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F>,
    {
        let PreparedProductGkr {
            shape,
            mut all_layers,
            roots,
        } = prepared;
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
                    .iter_mut()
                    .map(|layers| layers.take_binary_children(child_depth))
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
                let mut batch = RadixFourBatch::new_consuming(&mut all_layers, child_depth);
                let mut round_polys = Vec::with_capacity(round_count);
                let round_point = batch.prove_rounds::<F>(&point, batching, |round_poly| {
                    round_polys.push(round_poly);
                    transcript.round(&round_poly)
                });

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
                    let expected = Point::eval_eq(&point, &round_point)
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
