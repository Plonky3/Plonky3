//! Identity-padded storage and folding state used by the product prover.

use alloc::vec::Vec;

use p3_field::Field;

use super::ROUND_POLY_LEN;
use super::math::interpolate_pair;

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
pub(super) struct ProductLayers<F> {
    /// Explicit prefixes indexed by their base-two depth above the leaves.
    layers: Vec<IdentityPrefix<F>>,
}

impl<F: Field> ProductLayers<F> {
    /// Builds every product level needed by the radix-four descent.
    pub(super) fn new(leaves: &[F], log_height: usize) -> Self {
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
    pub(super) fn root(&self) -> F {
        self.layers
            .last()
            .expect("every product tree retains its root layer")
            .get(0)
    }

    /// Reads the two children of a root-most binary layer.
    #[inline]
    pub(super) fn binary_children(&self, depth: usize) -> [F; 2] {
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
    /// Splits an interleaved product level into four child tables.
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

    /// Binds one parent variable in every child multilinear.
    fn fold(&mut self, challenge: F, logical_len: usize) {
        for child in &mut self.children {
            child.fold(logical_len, challenge);
        }
    }

    /// Reads the four terminal child claims after every parent variable is bound.
    fn children(&self) -> [F; 4] {
        core::array::from_fn(|slot| self.children[slot].get(0))
    }
}

/// Per-tree states reduced by one shared radix-four sumcheck.
pub(super) struct RadixFourBatch<F> {
    /// One folding state for each product tree.
    states: Vec<RadixFourState<F>>,
}

impl<F: Field> RadixFourBatch<F> {
    /// Creates the batched states from one retained level per tree.
    pub(super) fn new(layers: &[ProductLayers<F>], depth: usize, logical_len: usize) -> Self {
        let states = layers
            .iter()
            .map(|layers| layers.radix_four_state(depth, logical_len))
            .collect();
        Self { states }
    }

    /// Computes one degree-five batched sumcheck message.
    pub(super) fn round(
        &self,
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
    pub(super) fn fold(&mut self, challenge: F, logical_len: usize) {
        for state in &mut self.states {
            state.fold(challenge, logical_len);
        }
    }

    /// Collects the terminal child claims in tree order.
    pub(super) fn children(&self) -> Vec<[F; 4]> {
        self.states.iter().map(RadixFourState::children).collect()
    }
}
