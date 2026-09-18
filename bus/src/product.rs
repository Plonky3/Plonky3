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

mod math;

use math::{
    combine, equality_evaluation, equality_weights, fold_dense, has_distinct_round_nodes,
    interpolate_pair, interpolate_quad,
};

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

impl ProductGkrRootShape {
    /// Encodes roots according to the verifier-derived statement shape.
    fn encode<F: Field>(self, roots: &[F]) -> Result<Vec<F>, ProductGkrError> {
        match self {
            Self::Distinct => Ok(roots.to_vec()),
            Self::FirstTwoShared => {
                // Structural sharing is valid only when both represented roots agree.
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

    /// Expands roots encoded according to the verifier-derived statement shape.
    fn decode<F: Copy>(self, encoded: &[F]) -> Vec<F> {
        match self {
            Self::Distinct => encoded.to_vec(),
            Self::FirstTwoShared => {
                // The first message represents both structurally equal roots.
                let mut roots = Vec::with_capacity(encoded.len() + 1);
                roots.extend([encoded[0], encoded[0]]);
                roots.extend_from_slice(&encoded[1..]);
                roots
            }
        }
    }
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
    #[error("product GKR received {num_trees} trees, minimum is {minimum}")]
    NoTrees {
        /// The rejected tree count.
        num_trees: usize,
        /// The smallest nonempty batch.
        minimum: usize,
    },
    /// A shared-root shape needs two roots to share.
    #[error("shared-root product GKR received {num_trees} trees, minimum is {minimum}")]
    SharedRootNeedsTwoTrees {
        /// The rejected tree count.
        num_trees: usize,
        /// The smallest batch containing two roots.
        minimum: usize,
    },
    /// Per-layer child messages would overflow their machine-word length.
    #[error("product GKR received {num_trees} trees, maximum is {maximum}")]
    TreeCountOverflow {
        /// The rejected tree count.
        num_trees: usize,
        /// The largest count whose four-child message fits in memory.
        maximum: usize,
    },
    /// The logical tree size cannot be represented by the target architecture.
    #[error("product GKR log height {log_height} exceeds maximum {maximum}")]
    HeightOverflow {
        /// The rejected base-two tree height.
        log_height: usize,
        /// The largest height whose capacity fits in one machine word.
        maximum: usize,
    },
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
            return Err(ProductGkrShapeError::HeightOverflow {
                log_height,
                maximum: usize::BITS as usize - 1,
            });
        }
        if num_trees == 0 {
            return Err(ProductGkrShapeError::NoTrees {
                num_trees,
                minimum: 1,
            });
        }
        if num_trees > usize::MAX / 4 {
            return Err(ProductGkrShapeError::TreeCountOverflow {
                num_trees,
                maximum: usize::MAX / 4,
            });
        }
        if root_shape == ProductGkrRootShape::FirstTwoShared && num_trees < 2 {
            return Err(ProductGkrShapeError::SharedRootNeedsTwoTrees {
                num_trees,
                minimum: 2,
            });
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

    /// Evaluates the constant-one suffix after an explicit prefix.
    ///
    /// Coordinates run from the most significant address bit to the least significant bit.
    /// The result is `sum_(i >= prefix_len) eq(point, i)` over the logical Boolean cube.
    ///
    /// # Panics
    ///
    /// Panics when the point length differs from the statement height.
    /// Panics when the prefix is longer than the logical table.
    #[must_use]
    pub fn identity_padding_evaluation<F: Field>(&self, prefix_len: usize, point: &[F]) -> F {
        // The statement fixes both the logical capacity and the point dimension.
        assert_eq!(
            point.len(),
            self.log_height,
            "point dimension must match the product GKR height"
        );
        let capacity = 1usize << self.log_height;
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

/// An arbitrary table prefix whose omitted suffix is the multiplicative identity.
struct IdentityPrefix<F> {
    /// Explicit values after removing trailing identities.
    values: Vec<F>,
}

impl<F: Field> IdentityPrefix<F> {
    /// Creates a canonical explicit prefix from a complete or partial table.
    fn new(mut values: Vec<F>) -> Self {
        // Trailing identities have the same semantics when left implicit.
        while values.last() == Some(&F::ONE) {
            values.pop();
        }
        Self { values }
    }

    /// Reads one logical value through implicit identity padding.
    #[inline]
    fn get(&self, index: usize) -> F {
        self.values.get(index).copied().unwrap_or(F::ONE)
    }

    /// Multiplies fixed-size groups into the next retained product layer.
    fn reduce(&self, arity: usize) -> Self {
        // A partial final group is completed by implicit identity factors.
        let values = self
            .values
            .chunks(arity)
            .map(|chunk| chunk.iter().copied().product())
            .collect();
        Self::new(values)
    }

    /// Binds the lowest remaining variable in place.
    fn fold(&mut self, logical_len: usize, challenge: F) {
        debug_assert!(self.values.len() <= logical_len);
        debug_assert!(logical_len >= 2);
        let output_len = self.values.len().div_ceil(2);

        // Missing entries retain the constant-one suffix during interpolation.
        for row in 0..output_len {
            let zero = self.get(2 * row);
            let one = self.get(2 * row + 1);
            self.values[row] = interpolate_pair([zero, one], challenge);
        }
        self.values.truncate(output_len);

        // Restore the canonical shortest representation after folding.
        while self.values.last() == Some(&F::ONE) {
            self.values.pop();
        }
    }
}

/// Product levels retained for one identity-padded input prefix.
struct ProductLayers<F> {
    /// Explicit prefixes indexed by their base-two depth above the leaves.
    layers: Vec<IdentityPrefix<F>>,
}

impl<F: Field> ProductLayers<F> {
    /// Builds every product level needed by the radix-four descent.
    fn new(leaves: &[F], log_height: usize) -> Self {
        // Unvisited depths remain empty constant-one prefixes.
        let mut layers = (0..=log_height)
            .map(|_| IdentityPrefix::new(Vec::new()))
            .collect::<Vec<_>>();
        layers[0] = IdentityPrefix::new(leaves.to_vec());

        // Two multiplication levels are retained per radix-four layer.
        let mut depth = 0;
        while depth + 2 <= log_height {
            layers[depth + 2] = layers[depth].reduce(4);
            depth += 2;
        }
        if depth < log_height {
            layers[log_height] = layers[depth].reduce(2);
        }

        Self { layers }
    }

    /// Returns the fully reduced product root.
    #[inline]
    fn root(&self) -> F {
        self.layers
            .last()
            .expect("every product tree retains its root layer")
            .get(0)
    }

    /// Reads the two children of a root-most binary layer.
    #[inline]
    fn binary_children(&self, depth: usize) -> [F; 2] {
        [self.layers[depth].get(0), self.layers[depth].get(1)]
    }

    /// Splits one retained level into four low-bit child prefixes.
    fn radix_four_state(&self, depth: usize, logical_len: usize) -> RadixFourState<F> {
        RadixFourState::new(&self.layers[depth], logical_len)
    }
}

/// Four child multilinears represented as arbitrary prefixes of constant-one tables.
struct RadixFourState<F> {
    /// One prefix per low-bit child slot.
    children: [IdentityPrefix<F>; 4],
}

impl<F: Field> RadixFourState<F> {
    /// Split an interleaved product level into four child tables.
    fn new(values: &IdentityPrefix<F>, logical_len: usize) -> Self {
        let mut children: [Vec<F>; 4] = core::array::from_fn(|_| Vec::new());
        for (index, &value) in values.values.iter().enumerate() {
            let slot = index % 4;
            let row = index / 4;
            debug_assert!(row < logical_len);
            children[slot].push(value);
        }
        let children = children.map(IdentityPrefix::new);
        Self { children }
    }

    /// Bind one parent variable in every child multilinear.
    fn fold(&mut self, challenge: F, logical_len: usize) {
        for child in &mut self.children {
            child.fold(logical_len, challenge);
        }
    }

    /// Read the four terminal child claims after every parent variable is bound.
    fn children(&self) -> [F; 4] {
        core::array::from_fn(|slot| self.children[slot].get(0))
    }
}

/// Per-tree states reduced by one shared radix-four sumcheck.
struct RadixFourBatch<F> {
    /// One folding state for each product tree.
    states: Vec<RadixFourState<F>>,
}

impl<F: Field> RadixFourBatch<F> {
    /// Creates the batched states from one retained level per tree.
    fn new(layers: &[ProductLayers<F>], depth: usize, logical_len: usize) -> Self {
        let states = layers
            .iter()
            .map(|layers| layers.radix_four_state(depth, logical_len))
            .collect();
        Self { states }
    }

    /// Computes one degree-five batched sumcheck message.
    fn round(&self, equality: &[F], logical_len: usize, batching: F) -> [F; ROUND_POLY_LEN] {
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

                for state in &self.states {
                    let product = state
                        .children
                        .iter()
                        .map(|child| {
                            let zero = child.get(2 * row);
                            let one = child.get(2 * row + 1);
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

    /// Binds one parent variable across every tree in the batch.
    fn fold(&mut self, challenge: F, logical_len: usize) {
        for state in &mut self.states {
            state.fold(challenge, logical_len);
        }
    }

    /// Collects the terminal child claims in tree order.
    fn children(&self) -> Vec<[F; 4]> {
        self.states.iter().map(RadixFourState::children).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_errors_report_rejected_values_and_bounds() {
        // An empty batch reports both the supplied count and its minimum.
        assert_eq!(
            ProductGkrShape::new(0, 0, ProductGkrRootShape::Distinct),
            Err(ProductGkrShapeError::NoTrees {
                num_trees: 0,
                minimum: 1,
            })
        );

        // Structural root sharing requires two concrete trees.
        assert_eq!(
            ProductGkrShape::new(0, 1, ProductGkrRootShape::FirstTwoShared),
            Err(ProductGkrShapeError::SharedRootNeedsTwoTrees {
                num_trees: 1,
                minimum: 2,
            })
        );

        // Four children per tree determine the largest addressable batch.
        let num_trees = usize::MAX / 4 + 1;
        assert_eq!(
            ProductGkrShape::new(0, num_trees, ProductGkrRootShape::Distinct),
            Err(ProductGkrShapeError::TreeCountOverflow {
                num_trees,
                maximum: usize::MAX / 4,
            })
        );

        // A machine word cannot address a table with its own bit width as log height.
        let log_height = usize::BITS as usize;
        assert_eq!(
            ProductGkrShape::new(log_height, 1, ProductGkrRootShape::Distinct),
            Err(ProductGkrShapeError::HeightOverflow {
                log_height,
                maximum: log_height - 1,
            })
        );
    }
}
