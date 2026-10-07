//! Identity-padded storage and folding state used by the product prover.

use alloc::vec::Vec;

use p3_field::Field;
use p3_maybe_rayon::prelude::*;

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
            .par_chunks(arity)
            .map_collect_min_task_bytes((arity + 1) * size_of::<F>(), |chunk| {
                chunk.iter().copied().product()
            });
        Self::new(values)
    }

    /// Binds the lowest remaining variable.
    fn fold(&mut self, logical_len: usize, challenge: F) {
        debug_assert!(self.values.len() <= logical_len);
        debug_assert!(logical_len >= 2);
        let output_len = self.values.len().div_ceil(2);

        // Missing entries retain the constant-one suffix during interpolation.
        self.values = (0..output_len)
            .into_par_iter()
            .map_collect_min_task_bytes(3 * size_of::<F>(), |row| {
                interpolate_pair([self.get(2 * row), self.get(2 * row + 1)], challenge)
            });

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

        let nodes = RoundNodes::<F>::new();
        // Tree `i` is weighted by the `i`-th power of the batching challenge.
        let powers = batching.powers().collect_n(self.states.len());
        let accumulate = |evaluations: &mut [F; ROUND_POLY_LEN], row: usize| {
            let eq_values = nodes.line(equality[2 * row], equality[2 * row + 1]);
            let mut batched = [F::ZERO; ROUND_POLY_LEN];

            for (state, &power) in self.states.iter().zip(&powers) {
                let [first, rest @ ..] = &state.children;
                let mut product = nodes.line(first.get(2 * row), first.get(2 * row + 1));
                for child in rest {
                    let values = nodes.line(child.get(2 * row), child.get(2 * row + 1));
                    for (product, value) in product.iter_mut().zip(values) {
                        *product *= value;
                    }
                }
                for (batched, product) in batched.iter_mut().zip(product) {
                    *batched += power * product;
                }
            }

            for ((evaluation, eq_value), batched) in
                evaluations.iter_mut().zip(eq_values).zip(batched)
            {
                *evaluation += eq_value * batched;
            }
        };

        // A row reads two equality and eight child values per tree, and uses each at every node.
        let rows = logical_len / 2;
        let row_bytes = ROUND_POLY_LEN * (2 + 8 * self.states.len()) * size_of::<F>();

        // Each task sums one run of rows, so only its partial message crosses threads.
        let task_rows = min_task_len(rows, row_bytes);
        (0..rows.div_ceil(task_rows))
            .into_par_iter()
            .map_collect_min_task_bytes(task_rows * row_bytes, |task| {
                let mut evaluations = [F::ZERO; ROUND_POLY_LEN];
                for row in task * task_rows..rows.min((task + 1) * task_rows) {
                    accumulate(&mut evaluations, row);
                }
                evaluations
            })
            .into_iter()
            .fold([F::ZERO; ROUND_POLY_LEN], |mut evaluations, partial| {
                for (evaluation, partial) in evaluations.iter_mut().zip(partial) {
                    *evaluation += partial;
                }
                evaluations
            })
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

/// The transmitted round-message nodes, with each line evaluated by as few products as possible.
struct RoundNodes<F> {
    /// Node one is omitted because the running sum reconstructs it.
    nodes: [F; ROUND_POLY_LEN],
    /// Whether a node equals its predecessor plus one, so its value is one slope further.
    unit_steps: [bool; ROUND_POLY_LEN],
}

impl<F: Field> RoundNodes<F> {
    /// Enumerates the nodes and marks every unit step between neighbours.
    fn new() -> Self {
        let nodes = [0, 2, 3, 4, 5].map(F::interpolation_node);
        let unit_steps =
            core::array::from_fn(|index| index > 0 && nodes[index] == nodes[index - 1] + F::ONE);
        Self { nodes, unit_steps }
    }

    /// Evaluates the line through `zero` and `one` at every node.
    #[inline]
    fn line(&self, zero: F, one: F) -> [F; ROUND_POLY_LEN] {
        let slope = one - zero;
        let mut values = [zero; ROUND_POLY_LEN];
        for index in 0..ROUND_POLY_LEN {
            values[index] = if self.unit_steps[index] {
                values[index - 1] + slope
            } else if self.nodes[index] == F::ZERO {
                zero
            } else {
                zero + self.nodes[index] * slope
            };
        }
        values
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_binary_field::{BinaryField128, Ghash128};
    use p3_field::extension::BinomialExtensionField;
    use rand::distr::{Distribution, StandardUniform};
    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoroshiro128Plus;

    use super::*;
    use crate::product::math::fold_dense;

    type BabyBearQuartic = BinomialExtensionField<BabyBear, 4>;

    /// Interpolates every line separately at every node, as an obvious reference.
    fn per_node_round<F: Field>(
        batch: &RadixFourBatch<F>,
        equality: &[F],
        logical_len: usize,
        batching: F,
    ) -> [F; ROUND_POLY_LEN] {
        let nodes = [0, 2, 3, 4, 5].map(F::interpolation_node);
        let mut evaluations = [F::ZERO; ROUND_POLY_LEN];
        for row in 0..logical_len / 2 {
            for (node_index, node) in nodes.into_iter().enumerate() {
                let eq_value = interpolate_pair([equality[2 * row], equality[2 * row + 1]], node);
                let mut power = F::ONE;
                let mut batched_product = F::ZERO;
                for state in &batch.states {
                    let product = state
                        .children
                        .iter()
                        .map(|child| {
                            interpolate_pair([child.get(2 * row), child.get(2 * row + 1)], node)
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

    /// Checks every node of random lines, after pinning which nodes take the slope step.
    fn check_lines<F: Field>(rng: &mut Xoroshiro128Plus, unit_steps: [bool; ROUND_POLY_LEN])
    where
        StandardUniform: Distribution<F>,
    {
        let nodes = RoundNodes::<F>::new();
        assert_eq!(nodes.unit_steps, unit_steps);
        for _ in 0..32 {
            let zero = rng.random::<F>();
            let one = rng.random::<F>();
            let expected = [0, 2, 3, 4, 5]
                .map(|index| interpolate_pair([zero, one], F::interpolation_node(index)));
            assert_eq!(nodes.line(zero, one), expected);
        }
    }

    /// Checks every round message of every radix-four layer against the per-node reference.
    fn check_rounds<F: Field>(rng: &mut Xoroshiro128Plus, log_height: usize, prefix_lens: &[usize])
    where
        StandardUniform: Distribution<F>,
    {
        let inputs = prefix_lens
            .iter()
            .map(|&len| (0..len).map(|_| rng.random::<F>()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let layers = inputs
            .iter()
            .map(|input| ProductLayers::new(input, log_height))
            .collect::<Vec<_>>();

        // Every retained even depth with at least one parent variable is a child level.
        for depth in (0..=log_height - 3).step_by(2) {
            let mut logical_len = 1usize << (log_height - depth - 2);
            let mut batch = RadixFourBatch::new(&layers, depth, logical_len);
            let mut equality = (0..logical_len)
                .map(|_| rng.random::<F>())
                .collect::<Vec<_>>();
            let batching = rng.random::<F>();

            // Folding between rounds walks every child prefix down to a single row.
            while logical_len >= 2 {
                assert_eq!(
                    batch.round(&equality, logical_len, batching),
                    per_node_round(&batch, &equality, logical_len, batching),
                );
                let challenge = rng.random::<F>();
                batch.fold(challenge, logical_len);
                fold_dense(&mut equality, challenge);
                logical_len /= 2;
            }
        }
    }

    #[test]
    fn slope_steps_match_interpolation_at_every_node() {
        // In characteristic two only nodes 3 and 5 are one past their predecessors.
        // In a prime field nodes 3, 4 and 5 all are, and node 2 never is in either.
        // Each field therefore checks both the slope step and the direct product.
        let mut rng = Xoroshiro128Plus::seed_from_u64(0x5107_0001);
        check_lines::<BinaryField128>(&mut rng, [false, false, true, false, true]);
        check_lines::<Ghash128>(&mut rng, [false, false, true, false, true]);
        check_lines::<BabyBearQuartic>(&mut rng, [false, false, true, true, true]);
    }

    #[test]
    fn round_messages_match_the_per_node_reference() {
        // Unequal, empty and non-multiple-of-four prefixes leave children of different lengths.
        // Several trees fix the batching power each tree must receive.
        let mut rng = Xoroshiro128Plus::seed_from_u64(0x5107_0002);
        let shapes: [&[usize]; 3] = [&[77], &[128, 0], &[13, 128, 91]];
        for prefix_lens in shapes {
            check_rounds::<BinaryField128>(&mut rng, 7, prefix_lens);
            check_rounds::<BabyBearQuartic>(&mut rng, 7, prefix_lens);
        }
    }
}
