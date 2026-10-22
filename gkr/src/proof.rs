//! Proof construction and verification for batched product-tree GKR.

use alloc::borrow::Cow;
use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::FieldChallenger;
use p3_challenger::fs::TranscriptField;
use p3_field::{ExtensionField, Field};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_sumcheck::generic_degree::RoundPolyInterpolator;
use serde::{Deserialize, Serialize};

use super::math::{
    combine, has_distinct_round_nodes, interpolate_pair, interpolate_quad, sum_adjacent,
};
use super::prover::{LayerRounds, ProductLayers, RadixFourBatch};
use super::transcript::{ProductGkrProverTranscript, ProductGkrVerifierTranscript};
use super::{ProductGkrError, ProductGkrShape};

/// Number of transmitted evaluations for a degree-five round polynomial.
pub const ROUND_POLY_LEN: usize = 5;

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
    /// A root-most binary layer of a batched table, summed over the batch coordinates.
    BinarySumcheck {
        /// Sumcheck messages in binding order, of degree three within the degree-five format.
        round_polys: Vec<[EF; ROUND_POLY_LEN]>,
        /// Two child evaluations.
        children: Vec<[EF; 2]>,
    },
}

/// Leaves of a batched table whose bottom radix-four layers the caller proves itself.
///
/// Those layers must return what the dense layers over the same leaves return, so proofs and transcripts are unchanged.
pub trait ProductLeaves<EF: Clone> {
    /// The rounds of one bottom layer.
    type Layer<'a>: LayerRounds<EF, 4>
    where
        Self: 'a;

    /// Bottom radix-four layers proved through `layer`.
    fn own_layers(&self) -> usize;

    /// The flat level `2 * own_layers()` above the leaves, batch index most significant, identity suffix optional.
    fn materialize(&self) -> Cow<'_, [EF]>;

    /// The rounds of bottom layer `j`, whose children are `2 * j` levels above the leaves, over `logical_len` parent rows.
    fn layer(&self, j: usize, logical_len: usize) -> Self::Layer<'_>;
}

/// Leaves given by their values, proving no layer of their own.
#[derive(Clone, Debug)]
pub struct DenseLeaves<EF>(pub Vec<EF>);

impl<EF: Field> ProductLeaves<EF> for DenseLeaves<EF> {
    type Layer<'a>
        = RadixFourBatch<'a, EF, EF, 4>
    where
        Self: 'a;

    fn own_layers(&self) -> usize {
        0
    }

    fn materialize(&self) -> Cow<'_, [EF]> {
        Cow::Borrowed(&self.0)
    }

    fn layer(&self, _: usize, _: usize) -> Self::Layer<'_> {
        unreachable!("dense leaves prove no layer of their own")
    }
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

/// Product layers and roots bound to one validated statement shape, ready to prove.
///
/// Borrowed leaves stay borrowed until the first fold of the leaf layer.
pub struct PreparedProductGkr<'a, EF: Clone> {
    shape: ProductGkrShape,
    all_layers: Vec<ProductLayers<'a, EF>>,
    roots: Vec<EF>,
    /// Product levels below the retained ones, proved by the caller's leaves.
    own_levels: usize,
}

impl<'a, EF: Field> PreparedProductGkr<'a, EF> {
    /// Prepare layers without asserting the shared-root relation or touching a transcript.
    pub fn new<F: Field>(inputs: &[&'a [EF]], shape: ProductGkrShape) -> Self
    where
        EF: ExtensionField<F>,
    {
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
                ProductLayers::new::<F>(*input, shape.log_height)
            });
        Self::from_layers(shape, all_layers)
    }

    /// Consume two owned leaf tables without copying their explicit prefixes.
    pub fn new_owned<F: Field>(mut inputs: [Vec<EF>; 2], shape: ProductGkrShape) -> Self
    where
        EF: ExtensionField<F>,
    {
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
                ProductLayers::new_owned::<F>(core::mem::take(input), shape.log_height)
            });
        Self::from_layers(shape, all_layers)
    }

    fn validate_lengths(lengths: impl ExactSizeIterator<Item = usize>, shape: ProductGkrShape) {
        assert_eq!(
            shape.batch_variables, 0,
            "a batched table is proved through prove_batched"
        );
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
            own_levels: 0,
        }
    }

    /// Prepare a batched table from the level its leaves materialize above their own layers.
    fn batched<F: Field, L: ProductLeaves<EF>>(leaves: &'a L, shape: ProductGkrShape) -> Self
    where
        EF: ExtensionField<F>,
    {
        let own_levels = 2 * leaves.own_layers();
        assert!(
            own_levels <= shape.log_height,
            "the leaves prove more levels than the tree has"
        );
        // A borrowed level is read in place, never copied.
        let level = leaves.materialize();
        let height = shape.log_height - own_levels;
        assert!(
            level.len() <= 1usize << (shape.batch_variables + height),
            "the materialized level exceeds its logical table"
        );
        let layers = ProductLayers::new::<F>(level, height);
        Self {
            shape,
            roots: layers.roots(1 << shape.batch_variables),
            all_layers: vec![layers],
            own_levels,
        }
    }

    /// The product root of every tree, before root encoding.
    pub fn roots(&self) -> &[EF] {
        &self.roots
    }

    /// Prove a statement whose roots the caller has already checked.
    ///
    /// # Panics
    ///
    /// Panics when the challenge field has fewer than six interpolation nodes, or a shared-root statement is false.
    pub fn prove<F, Challenger>(
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
        ProductGkrProof::prove_inner::<F, Challenger, DenseLeaves<EF>>(self, None, challenger)
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
        Self::prove_inner::<F, Challenger, DenseLeaves<EF>>(
            PreparedProductGkr::new::<F>(inputs, shape),
            None,
            challenger,
        )
    }

    /// Proves a batched table of trees, the bottom layers through the caller's leaves.
    ///
    /// The output carries every root, the point over the batch and tree coordinates, and the one leaf evaluation there.
    ///
    /// # Panics
    ///
    /// Panics when the shape is not batched, the challenge field has fewer than six interpolation nodes, or the leaves disagree with the shape.
    pub fn prove_batched<F, Challenger, L>(
        leaves: &L,
        shape: ProductGkrShape,
        challenger: &mut Challenger,
    ) -> (Self, ProductGkrOutput<EF>)
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F>,
        L: ProductLeaves<EF>,
    {
        assert!(
            shape.batch_variables > 0,
            "separate trees are proved through prove"
        );
        assert!(
            has_distinct_round_nodes::<EF>(),
            "product GKR requires six distinct challenge-field interpolation nodes"
        );
        Self::prove_inner::<F, Challenger, L>(
            PreparedProductGkr::batched::<F, L>(leaves, shape),
            Some(leaves),
            challenger,
        )
    }

    /// Run the transcript after the statement and product layers have been checked.
    fn prove_inner<F, Challenger, L>(
        prepared: PreparedProductGkr<'_, EF>,
        leaves: Option<&L>,
        challenger: &mut Challenger,
    ) -> (Self, ProductGkrOutput<EF>)
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F>,
        L: ProductLeaves<EF>,
    {
        let PreparedProductGkr {
            shape,
            mut all_layers,
            roots,
            own_levels,
        } = prepared;
        let root_messages = shape
            .root_shape
            .encode(&roots)
            .expect("a shared-root statement requires equal roots");

        // The typed transcript binds both the dimensions and every prover message.
        let mut transcript =
            ProductGkrProverTranscript::<Challenger, F, EF>::new(challenger, shape, &root_messages);
        let mut point = transcript.batch_point(shape.batch_variables);
        let mut values = if point.is_empty() {
            roots.clone()
        } else {
            // A batched table compares its roots at the batch point, the first coordinate least significant.
            vec![
                Point::new(point.as_slice())
                    .equality_weights_lsb()
                    .iter()
                    .zip(&roots)
                    .map(|(&weight, &root)| weight * root)
                    .sum(),
            ]
        };
        let mut layer_proofs = Vec::with_capacity(shape.layers().len());
        let mut product_depth = shape.log_height;

        for (arity, round_count) in shape.layers() {
            let batching = transcript.begin_layer();
            debug_assert_eq!(round_count, point.len());
            let child_depth = product_depth - arity.trailing_zeros() as usize;
            let logical_len = 1usize << round_count;

            if arity == 2 && round_count == 0 {
                // The root-most binary layer has no parent variables to sum over.
                let children = all_layers
                    .iter_mut()
                    .map(|layers| layers.take_binary_children(child_depth - own_levels))
                    .collect::<Vec<_>>();
                let branches = transcript.end_binary_layer(&children);
                values = children
                    .iter()
                    .map(|children| interpolate_pair(*children, branches[0]))
                    .collect();
                point = vec![branches[0]];
                layer_proofs.push(ProductGkrLayerProof::Binary { children });
            } else if arity == 2 {
                // A batched table sums its root-most binary layer over the batch coordinates.
                let mut batch = RadixFourBatch::<F, EF, 2>::new_consuming(
                    &mut all_layers,
                    child_depth - own_levels,
                );
                let mut round_polys = Vec::with_capacity(round_count);
                let round_point = batch.prove_rounds(&point, batching, |round_poly| {
                    round_polys.push(round_poly);
                    transcript.round(&round_poly)
                });
                let children = batch.children();
                let branches = transcript.end_binary_layer(&children);
                values = children
                    .iter()
                    .map(|children| interpolate_pair(*children, branches[0]))
                    .collect();
                point = vec![branches[0]];
                point.extend(round_point);
                layer_proofs.push(ProductGkrLayerProof::BinarySumcheck {
                    round_polys,
                    children,
                });
            } else {
                // Four child tables share one eq-weighted degree-five sumcheck.
                let (round_polys, round_point, children) = match leaves {
                    Some(leaves) if child_depth < own_levels => {
                        let mut layer = leaves.layer(child_depth / 2, logical_len);
                        let (polys, round_point) =
                            Self::layer_rounds(&mut layer, &point, batching, &mut transcript);
                        (polys, round_point, layer.children())
                    }
                    _ => {
                        let mut batch = RadixFourBatch::<F, EF, 4>::new_consuming(
                            &mut all_layers,
                            child_depth - own_levels,
                        );
                        let mut round_polys = Vec::with_capacity(round_count);
                        let round_point = batch.prove_rounds(&point, batching, |round_poly| {
                            round_polys.push(round_poly);
                            transcript.round(&round_poly)
                        });
                        (round_polys, round_point, batch.children())
                    }
                };
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

    /// Play one caller-proved layer's sumcheck at parent point `point`, the first coordinate bound first.
    fn layer_rounds<F, Challenger, R, const N: usize>(
        layer: &mut R,
        point: &[EF],
        batching: EF,
        transcript: &mut ProductGkrProverTranscript<'_, Challenger, F, EF>,
    ) -> (Vec<[EF; ROUND_POLY_LEN]>, Vec<EF>)
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F>,
        R: LayerRounds<EF, N>,
    {
        // The current coordinate factors out of the row sum. Keep only its
        // normalized suffix table; the root-most layer can have no coordinates.
        let mut equality = Point::new(point.get(1..).unwrap_or(&[])).equality_weights_lsb();
        let mut spare = Vec::new();
        let mut equality_prefix = EF::ONE;
        let mut logical_len = 1usize << point.len();
        let mut round_point = Vec::with_capacity(point.len());
        let mut round_polys = Vec::with_capacity(point.len());

        for &anchor in point {
            let one = equality_prefix * anchor;
            let equality_pair = [equality_prefix - one, one];
            let round_poly = layer.round(&equality, equality_pair, logical_len, batching);
            let challenge = transcript.round(&round_poly);
            layer.fold(challenge, logical_len);
            equality_prefix = interpolate_pair(equality_pair, challenge);
            logical_len /= 2;
            if logical_len > 1 {
                // Product GKR binds adjacent pairs (LSB-first), unlike Poly's
                // leading-variable convention. No division by a prefix is needed.
                sum_adjacent(&mut equality, &mut spare);
            }
            round_point.push(challenge);
            round_polys.push(round_poly);
        }
        (round_polys, round_point)
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
        let mut point = transcript.batch_point(shape.batch_variables);
        let mut values = if point.is_empty() {
            roots.clone()
        } else {
            // A batched table compares its roots at the batch point, the first coordinate least significant.
            vec![
                Point::new(point.as_slice())
                    .equality_weights_lsb()
                    .iter()
                    .zip(&roots)
                    .map(|(&weight, &root)| weight * root)
                    .sum(),
            ]
        };
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
                ProductGkrLayerProof::BinarySumcheck {
                    round_polys,
                    children,
                } => {
                    debug_assert_eq!(arity, 2);
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
                                .map(|children| children[0] * children[1])
                                .collect::<Vec<_>>(),
                            batching,
                        );
                    if running_sum != expected {
                        inconsistent_layer.get_or_insert(layer_index);
                    }
                    let branches = transcript.end_binary_layer(children);
                    values = children
                        .iter()
                        .map(|children| interpolate_pair(*children, branches[0]))
                        .collect();
                    point = vec![branches[0]];
                    point.extend(round_point);
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
                (2, ProductGkrLayerProof::Binary { children }) => {
                    *rounds == 0 && children.len() == shape.num_trees
                }
                (
                    2,
                    ProductGkrLayerProof::BinarySumcheck {
                        round_polys,
                        children,
                    },
                ) => {
                    *rounds > 0 && round_polys.len() == *rounds && children.len() == shape.num_trees
                }
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
