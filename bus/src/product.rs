//! Batched product-tree GKR with identity-padded input prefixes.
//!
//! Two multiplication levels are contracted into one radix-four layer:
//!
//! ```text
//!     parent(x) = child(0, 0, x) * child(1, 0, x)
//!               * child(0, 1, x) * child(1, 1, x)
//! ```
//!
//! Multiplying by the equality polynomial gives degree five in each sumcheck variable.
//! One random challenge batches the same relation across every tree.
//!
//! A sumcheck round contributes at most `5 / |F|` soundness error.
//! Batching `T` trees contributes at most `(T - 1) / |F|` per layer.
//! Collapsing four child claims contributes at most `2 / |F|` per layer.
//!
//! These bounds cover only the product reduction.
//! The caller must also authenticate the returned leaf evaluations.

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::FieldChallenger;
use p3_challenger::fs::TranscriptField;
use p3_field::{ExtensionField, Field};
use p3_sumcheck::generic_degree::RoundPolyInterpolator;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::transcript::{ProductGkrProverTranscript, ProductGkrVerifierTranscript};

/// Number of transmitted evaluations for a degree-five round polynomial.
pub(crate) const ROUND_POLY_LEN: usize = 5;

/// How product roots are represented in a proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProductGkrRootShape {
    /// Every tree carries an independent root.
    Distinct,
    /// The first two trees use one shared root value.
    FirstTwoShared,
}

/// Verifier-derived dimensions of one batched product reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProductGkrShape {
    /// Logical base-two height of every identity-padded tree.
    log_height: usize,
    /// Number of product trees reduced in lockstep.
    num_trees: usize,
    /// Encoding of the root claims.
    root_shape: ProductGkrRootShape,
}

/// Invalid dimensions rejected before a transcript is created.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum ProductGkrShapeError {
    /// A batch without trees has no statement.
    #[error("product GKR requires at least one tree")]
    NoTrees,
    /// A shared-root shape needs two roots to share.
    #[error("a shared-root product GKR requires at least two trees")]
    SharedRootNeedsTwoTrees,
    /// Per-layer child messages would overflow their machine-word length.
    #[error("product GKR tree count overflows a radix-four child message")]
    TreeCountOverflow,
    /// The logical tree size cannot be represented by the target architecture.
    #[error("product GKR logical height overflows usize")]
    HeightOverflow,
}

impl ProductGkrShape {
    /// Construct dimensions shared by the prover and verifier.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty batch, an impossible shared root, or an oversized tree.
    pub fn new(
        log_height: usize,
        num_trees: usize,
        root_shape: ProductGkrRootShape,
    ) -> Result<Self, ProductGkrShapeError> {
        // Product inputs use one machine word as their address space.
        if log_height >= usize::BITS as usize {
            return Err(ProductGkrShapeError::HeightOverflow);
        }
        if num_trees == 0 {
            return Err(ProductGkrShapeError::NoTrees);
        }
        if num_trees > usize::MAX / 4 {
            return Err(ProductGkrShapeError::TreeCountOverflow);
        }
        if root_shape == ProductGkrRootShape::FirstTwoShared && num_trees < 2 {
            return Err(ProductGkrShapeError::SharedRootNeedsTwoTrees);
        }

        Ok(Self {
            log_height,
            num_trees,
            root_shape,
        })
    }

    /// Logical base-two height of each product tree.
    #[must_use]
    pub const fn log_height(&self) -> usize {
        self.log_height
    }

    /// Number of product trees reduced together.
    #[must_use]
    pub const fn num_trees(&self) -> usize {
        self.num_trees
    }

    /// Root encoding fixed by the statement.
    #[must_use]
    pub const fn root_shape(&self) -> ProductGkrRootShape {
        self.root_shape
    }

    /// Number of root values carried by the proof.
    pub(crate) const fn root_message_len(&self) -> usize {
        match self.root_shape {
            ProductGkrRootShape::Distinct => self.num_trees,
            ProductGkrRootShape::FirstTwoShared => self.num_trees - 1,
        }
    }

    /// Root-to-leaf arity and sumcheck-round count of each layer.
    pub(crate) fn layers(&self) -> Vec<(usize, usize)> {
        // The point starts empty at the root.
        let mut point_len = 0;
        let mut remaining = self.log_height;
        let mut layers = Vec::with_capacity(self.log_height.div_ceil(2));

        while remaining > 0 {
            // An odd tree begins with one binary level.
            // Every later layer contracts two levels at once.
            let arity: usize = if remaining == self.log_height && remaining % 2 == 1 {
                2
            } else {
                4
            };
            layers.push((arity, point_len));
            let branch_count = arity.trailing_zeros() as usize;
            point_len += branch_count;
            remaining -= branch_count;
        }

        layers
    }
}

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

/// Invalid inputs or proofs rejected by the product reduction.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ProductGkrError {
    /// Degree-five interpolation needs six distinct challenge-field elements.
    #[error("product GKR requires six distinct challenge-field interpolation nodes")]
    ChallengeFieldTooSmall,
    /// The input batch disagrees with the verifier-derived tree count.
    #[error("product GKR received {actual} trees, expected {expected}")]
    TreeCountMismatch {
        /// Tree count fixed by the statement.
        expected: usize,
        /// Tree count supplied by the caller.
        actual: usize,
    },
    /// An explicit prefix is longer than its logical tree.
    #[error("product GKR tree {tree} has {actual} leaves, logical capacity is {capacity}")]
    InputTooLong {
        /// Position of the malformed tree.
        tree: usize,
        /// Logical leaf capacity.
        capacity: usize,
        /// Supplied prefix length.
        actual: usize,
    },
    /// Two roots required to be structurally shared are unequal.
    #[error("the first two product roots are not equal")]
    SharedRootMismatch,
    /// A proof carries the wrong number of encoded roots.
    #[error("product GKR proof has {actual} root messages, expected {expected}")]
    RootCountMismatch {
        /// Root count fixed by the statement.
        expected: usize,
        /// Root count carried by the proof.
        actual: usize,
    },
    /// A proof carries the wrong number of reduction layers.
    #[error("product GKR proof has {actual} layers, expected {expected}")]
    LayerCountMismatch {
        /// Layer count fixed by the statement.
        expected: usize,
        /// Layer count carried by the proof.
        actual: usize,
    },
    /// One layer has the wrong arity or number of rounds.
    #[error("product GKR layer {layer} has a malformed shape")]
    MalformedLayer {
        /// Position of the malformed layer.
        layer: usize,
    },
    /// One layer's sumcheck does not close on its child claims.
    #[error("product GKR layer {layer} failed its consistency check")]
    LayerConsistency {
        /// Position of the inconsistent layer.
        layer: usize,
    },
}

/// Prove several product trees in one transcript.
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
pub fn prove_product_gkr<F, EF, Challenger>(
    inputs: &[&[EF]],
    shape: ProductGkrShape,
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
    let mut all_layers = inputs
        .iter()
        .map(|input| build_product_layers(input, shape.log_height))
        .collect::<Vec<_>>();
    let roots = all_layers
        .iter()
        .map(|layers| layers[shape.log_height].first().copied().unwrap_or(EF::ONE))
        .collect::<Vec<_>>();
    let root_messages = encode_roots(&roots, shape.root_shape)
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
                .map(|layers| binary_children(&layers[child_depth]))
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
            let mut states = all_layers
                .iter_mut()
                .map(|layers| RadixFourState::new(&layers[child_depth], 1usize << round_count))
                .collect::<Vec<_>>();
            let mut equality = equality_weights(&point);
            let mut logical_len = 1usize << round_count;
            let mut round_point = Vec::with_capacity(round_count);
            let mut round_polys = Vec::with_capacity(round_count);

            for _ in 0..round_count {
                let round_poly = radix_four_round(&states, &equality, logical_len, batching);
                let challenge = transcript.round(&round_poly);
                for state in &mut states {
                    state.fold(challenge, logical_len);
                }
                fold_dense(&mut equality, challenge);
                logical_len /= 2;
                round_point.push(challenge);
                round_polys.push(round_poly);
            }

            let children = states
                .iter()
                .map(RadixFourState::children)
                .collect::<Vec<_>>();
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
        ProductGkrProof {
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

/// Verify the internal consistency of several product trees.
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
pub fn verify_product_gkr<F, EF, Challenger>(
    proof: &ProductGkrProof<EF>,
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
    validate_proof_shape(proof, shape)?;
    let roots = decode_roots(&proof.roots, shape.root_shape);
    let mut transcript =
        ProductGkrVerifierTranscript::<Challenger, F, EF>::new(challenger, shape, &proof.roots);
    let mut values = roots.clone();
    let mut point = Vec::with_capacity(shape.log_height);
    let mut inconsistent_layer = None;
    let interpolator = RoundPolyInterpolator::new(5);

    for (layer_index, ((arity, _), layer)) in
        shape.layers().into_iter().zip(&proof.layers).enumerate()
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
fn validate_proof_shape<EF>(
    proof: &ProductGkrProof<EF>,
    shape: ProductGkrShape,
) -> Result<(), ProductGkrError> {
    if proof.roots.len() != shape.root_message_len() {
        return Err(ProductGkrError::RootCountMismatch {
            expected: shape.root_message_len(),
            actual: proof.roots.len(),
        });
    }

    let layers = shape.layers();
    if proof.layers.len() != layers.len() {
        return Err(ProductGkrError::LayerCountMismatch {
            expected: layers.len(),
            actual: proof.layers.len(),
        });
    }

    for (layer_index, ((arity, rounds), layer)) in layers.iter().zip(&proof.layers).enumerate() {
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

/// Build every product level needed by the radix-four descent.
fn build_product_layers<F: Field>(leaves: &[F], log_height: usize) -> Vec<Vec<F>> {
    // Trailing identities remain implicit from the first level onward.
    let mut layers = vec![Vec::new(); log_height + 1];
    layers[0] = leaves.to_vec();
    trim_identities(&mut layers[0]);

    let mut depth = 0;
    while depth + 2 <= log_height {
        layers[depth + 2] = reduce_prefix(&layers[depth], 4);
        depth += 2;
    }
    if depth < log_height {
        layers[log_height] = reduce_prefix(&layers[depth], 2);
    }

    layers
}

/// Multiply fixed-size groups while leaving the all-one suffix absent.
fn reduce_prefix<F: Field>(values: &[F], arity: usize) -> Vec<F> {
    let mut reduced = values
        .chunks(arity)
        .map(|chunk| chunk.iter().copied().product())
        .collect::<Vec<_>>();
    trim_identities(&mut reduced);
    reduced
}

/// Remove values represented by implicit identity padding.
fn trim_identities<F: Field>(values: &mut Vec<F>) {
    while values.last() == Some(&F::ONE) {
        values.pop();
    }
}

/// Encode roots according to the statement rather than prover-controlled metadata.
fn encode_roots<F: Field>(
    roots: &[F],
    shape: ProductGkrRootShape,
) -> Result<Vec<F>, ProductGkrError> {
    match shape {
        ProductGkrRootShape::Distinct => Ok(roots.to_vec()),
        ProductGkrRootShape::FirstTwoShared => {
            if roots[0] != roots[1] {
                return Err(ProductGkrError::SharedRootMismatch);
            }
            let mut encoded = Vec::with_capacity(roots.len() - 1);
            encoded.push(roots[0]);
            encoded.extend_from_slice(&roots[2..]);
            Ok(encoded)
        }
    }
}

/// Expand the structural shared-root encoding.
fn decode_roots<F: Copy>(encoded: &[F], shape: ProductGkrRootShape) -> Vec<F> {
    match shape {
        ProductGkrRootShape::Distinct => encoded.to_vec(),
        ProductGkrRootShape::FirstTwoShared => {
            let mut roots = Vec::with_capacity(encoded.len() + 1);
            roots.extend([encoded[0], encoded[0]]);
            roots.extend_from_slice(&encoded[1..]);
            roots
        }
    }
}

/// Read the root-most binary children from an explicit prefix.
fn binary_children<F: Field>(values: &[F]) -> [F; 2] {
    [
        values.first().copied().unwrap_or(F::ONE),
        values.get(1).copied().unwrap_or(F::ONE),
    ]
}

/// Four child multilinears represented as arbitrary prefixes of constant-one tables.
struct RadixFourState<F> {
    /// One prefix per low-bit child slot.
    children: [Vec<F>; 4],
}

impl<F: Field> RadixFourState<F> {
    /// Split an interleaved product level into four child tables.
    fn new(values: &[F], logical_len: usize) -> Self {
        let mut children: [Vec<F>; 4] = core::array::from_fn(|_| Vec::new());
        for (index, &value) in values.iter().enumerate() {
            let slot = index % 4;
            let row = index / 4;
            debug_assert!(row < logical_len);
            children[slot].push(value);
        }
        for child in &mut children {
            trim_identities(child);
        }
        Self { children }
    }

    /// Bind one parent variable in every child multilinear.
    fn fold(&mut self, challenge: F, logical_len: usize) {
        for child in &mut self.children {
            fold_prefix(child, logical_len, challenge);
        }
    }

    /// Read the four terminal child claims after every parent variable is bound.
    fn children(&self) -> [F; 4] {
        core::array::from_fn(|slot| self.children[slot].first().copied().unwrap_or(F::ONE))
    }
}

/// Compute one degree-five batched sumcheck message.
fn radix_four_round<F: Field>(
    states: &[RadixFourState<F>],
    equality: &[F],
    logical_len: usize,
    batching: F,
) -> [F; ROUND_POLY_LEN] {
    debug_assert!(logical_len >= 2);
    debug_assert_eq!(equality.len(), logical_len);

    // Node one is omitted because the running sum reconstructs it.
    let nodes = [0, 2, 3, 4, 5].map(F::interpolation_node);
    let mut evaluations = [F::ZERO; ROUND_POLY_LEN];

    for row in 0..logical_len / 2 {
        let eq_zero = equality[2 * row];
        let eq_one = equality[2 * row + 1];

        for (node_index, node) in nodes.into_iter().enumerate() {
            let eq_value = interpolate_pair([eq_zero, eq_one], node);
            let mut power = F::ONE;
            let mut batched_product = F::ZERO;

            for state in states {
                let product = state
                    .children
                    .iter()
                    .map(|child| {
                        let zero = child.get(2 * row).copied().unwrap_or(F::ONE);
                        let one = child.get(2 * row + 1).copied().unwrap_or(F::ONE);
                        interpolate_pair([zero, one], node)
                    })
                    .product::<F>();
                batched_product += power * product;
                power *= batching;
            }

            evaluations[node_index] += eq_value * batched_product;
        }
    }

    evaluations
}

/// Fold a constant-one prefix along its lowest remaining variable.
fn fold_prefix<F: Field>(values: &mut Vec<F>, logical_len: usize, challenge: F) {
    debug_assert!(values.len() <= logical_len);
    debug_assert!(logical_len >= 2);
    let output_len = values.len().div_ceil(2);

    for row in 0..output_len {
        let zero = values.get(2 * row).copied().unwrap_or(F::ONE);
        let one = values.get(2 * row + 1).copied().unwrap_or(F::ONE);
        values[row] = interpolate_pair([zero, one], challenge);
    }
    values.truncate(output_len);
    trim_identities(values);
}

/// Fold a fully materialized multilinear table in place.
fn fold_dense<F: Field>(values: &mut Vec<F>, challenge: F) {
    let output_len = values.len() / 2;
    for row in 0..output_len {
        values[row] = interpolate_pair([values[2 * row], values[2 * row + 1]], challenge);
    }
    values.truncate(output_len);
}

/// Equality weights over a low-variable-first Boolean cube.
fn equality_weights<F: Field>(point: &[F]) -> Vec<F> {
    let mut weights = vec![F::ONE];
    for &coordinate in point {
        let old_len = weights.len();
        weights.resize(old_len * 2, F::ZERO);
        for index in 0..old_len {
            let weight = weights[index];
            weights[index] = weight * (F::ONE - coordinate);
            weights[old_len + index] = weight * coordinate;
        }
    }
    weights
}

/// Evaluate the multilinear equality polynomial at two points.
fn equality_evaluation<F: Field>(left: &[F], right: &[F]) -> F {
    debug_assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(&left, &right)| (F::ONE - left) * (F::ONE - right) + left * right)
        .product()
}

/// Combine one claim per tree with consecutive powers of one challenge.
fn combine<F: Field>(values: &[F], challenge: F) -> F {
    values
        .iter()
        .zip(challenge.powers())
        .map(|(&value, power)| value * power)
        .sum()
}

/// Interpolate a line whose endpoints are indexed by one Boolean variable.
fn interpolate_pair<F: Field>(values: [F; 2], point: F) -> F {
    values[0] + point * (values[1] - values[0])
}

/// Interpolate a four-entry table at two low-order coordinates.
fn interpolate_quad<F: Field>(values: [F; 4], point: [F; 2]) -> F {
    let low_zero = interpolate_pair([values[0], values[1]], point[0]);
    let low_one = interpolate_pair([values[2], values[3]], point[0]);
    interpolate_pair([low_zero, low_one], point[1])
}

/// Check that degree-five interpolation has a valid six-point domain.
fn has_distinct_round_nodes<F: Field>() -> bool {
    // Six distinct nodes cannot exist in a field with fewer than eight elements.
    // This guard also avoids calling interpolation-node constructors outside their domain.
    if F::bits() < 3 {
        return false;
    }
    // Pairwise comparison avoids allocating at the proof boundary.
    let nodes = core::array::from_fn::<_, 6, _>(F::interpolation_node);
    nodes
        .iter()
        .enumerate()
        .all(|(index, node)| !nodes[index + 1..].contains(node))
}

/// Evaluate the constant-one suffix after an explicit prefix.
///
/// Coordinates are ordered from the most significant address bit to the least significant bit.
/// The result is `sum_(i >= prefix_len) eq(point, i)` over the logical Boolean cube.
///
/// # Panics
///
/// Panics when the prefix is longer than the logical table.
#[must_use]
pub fn identity_padding_evaluation<F: Field>(prefix_len: usize, point: &[F]) -> F {
    let capacity = 1usize
        .checked_shl(point.len() as u32)
        .expect("multilinear point must fit in usize");
    assert!(prefix_len <= capacity, "prefix exceeds the logical table");
    if prefix_len == capacity {
        return F::ZERO;
    }

    // Sum the address weights strictly below the binary threshold.
    let mut below = F::ZERO;
    let mut equal_prefix = F::ONE;
    for (bit_index, &coordinate) in point.iter().enumerate() {
        let shift = point.len() - 1 - bit_index;
        if (prefix_len >> shift) & 1 == 1 {
            below += equal_prefix * (F::ONE - coordinate);
            equal_prefix *= coordinate;
        } else {
            equal_prefix *= F::ONE - coordinate;
        }
    }

    F::ONE - below
}
