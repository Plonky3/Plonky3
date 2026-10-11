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

    /// Bytes one fold moves: two reads and one write per folded row.
    #[inline]
    const fn fold_bytes(&self) -> usize {
        3 * self.values.len().div_ceil(2) * size_of::<F>()
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
        Self::new_owned(leaves.to_vec(), log_height)
    }

    /// Builds the retained levels while taking ownership of the leaf allocation.
    pub(super) fn new_owned(leaves: Vec<F>, log_height: usize) -> Self {
        // Unvisited depths remain empty constant-one prefixes.
        let mut layers = (0..=log_height)
            .map(|_| IdentityPrefix::new(Vec::new()))
            .collect::<Vec<_>>();
        layers[0] = IdentityPrefix::new(leaves);

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
        let values = &values.values;
        debug_assert!(values.len().div_ceil(4) <= logical_len);

        // Slot `s` gathers explicit entries `s, s + 4, s + 8, ...`, reading and writing each once.
        let children = core::array::from_fn(|slot| {
            let rows = (values.len() + 3 - slot) / 4;
            IdentityPrefix::new(
                (0..rows)
                    .into_par_iter()
                    .map_collect_min_task_bytes(2 * size_of::<F>(), |row| values[4 * row + slot]),
            )
        });
        Self { children }
    }

    /// Bytes one fold of every child moves.
    fn fold_bytes(&self) -> usize {
        self.children.iter().map(IdentityPrefix::fold_bytes).sum()
    }

    /// Binds one parent variable in every child multilinear.
    fn fold(&mut self, challenge: F, logical_len: usize) {
        // Each child is charged the average, so the gate sees the whole fold, and still splits its
        // own rows when they pay.
        let child_bytes = self.fold_bytes() / 4;
        self.children
            .par_iter_mut()
            .for_each_min_task_bytes(child_bytes, |child| child.fold(logical_len, challenge));
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
        // A tree reads its retained level once and writes it back as four children.
        let tree_bytes = layers
            .iter()
            .map(|layers| 2 * layers.layers[depth].values.len() * size_of::<F>())
            .sum::<usize>()
            .checked_div(layers.len())
            .unwrap_or(0);
        let states = layers
            .par_iter()
            .map_collect_min_task_bytes(tree_bytes, |layers| {
                layers.radix_four_state(depth, logical_len)
            });
        Self { states }
    }

    /// Computes one degree-five batched sumcheck message. The equality polynomial
    /// is split into current-coordinate endpoints and normalized suffix weights.
    pub(super) fn round(
        &self,
        equality: &[F],
        equality_pair: [F; 2],
        logical_len: usize,
        batching: F,
    ) -> [F; ROUND_POLY_LEN] {
        debug_assert!(logical_len >= 2);
        debug_assert_eq!(equality.len(), logical_len / 2);

        let nodes = RoundNodes::<F>::new();
        let eq_values = nodes.line(equality_pair[0], equality_pair[1]);
        let Some((first_tree, other_trees)) = self.states.split_first() else {
            return [F::ZERO; ROUND_POLY_LEN];
        };
        // The first tree has coefficient one; only later trees need batching products.
        let powers = batching
            .shifted_powers(batching)
            .collect_n(other_trees.len());
        let accumulate = |evaluations: &mut [F; ROUND_POLY_LEN], row: usize| {
            let tree_product = |state: &RadixFourState<F>| {
                let [first, rest @ ..] = &state.children;
                let mut product = nodes.line(first.get(2 * row), first.get(2 * row + 1));
                for child in rest {
                    let values = nodes.line(child.get(2 * row), child.get(2 * row + 1));
                    for (product, value) in product.iter_mut().zip(values) {
                        *product *= value;
                    }
                }
                product
            };
            let mut batched = tree_product(first_tree);
            for (state, &power) in other_trees.iter().zip(&powers) {
                for (batched, product) in batched.iter_mut().zip(tree_product(state)) {
                    *batched += power * product;
                }
            }

            for (evaluation, batched) in evaluations.iter_mut().zip(batched) {
                *evaluation += equality[row] * batched;
            }
        };

        // Past every explicit child prefix, all tree products are constant one.
        // Keep a partial final pair in the active rows.
        let rows = self
            .states
            .iter()
            .flat_map(|state| &state.children)
            .map(|child| child.values.len())
            .max()
            .unwrap_or(0)
            .div_ceil(2);
        debug_assert!(rows <= equality.len());
        // An active row reads one equality weight and eight child values per tree.
        let row_bytes = ROUND_POLY_LEN * (1 + 8 * self.states.len()) * size_of::<F>();

        // Each task sums one run of rows, so only its partial message crosses threads.
        let task_rows = min_task_len(rows, row_bytes);
        let mut evaluations = (0..rows.div_ceil(task_rows))
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
            });
        if rows < equality.len() {
            // Normalized suffix weights sum to one. Read the shorter side.
            let tail_weight: F = if rows < equality.len() - rows {
                F::ONE
                    - equality[..rows]
                        .par_iter()
                        .with_min_task::<F>()
                        .copied()
                        .sum::<F>()
            } else {
                equality[rows..]
                    .par_iter()
                    .with_min_task::<F>()
                    .copied()
                    .sum()
            };
            let tail = tail_weight * (F::ONE + powers.iter().copied().sum::<F>());
            for evaluation in &mut evaluations {
                *evaluation += tail;
            }
        }
        for (evaluation, eq_value) in evaluations.iter_mut().zip(eq_values) {
            *evaluation *= eq_value;
        }
        evaluations
    }

    /// Binds one parent variable across every tree in the batch.
    pub(super) fn fold(&mut self, challenge: F, logical_len: usize) {
        // Each tree is charged the average, so the gate sees the whole fold, and still splits its
        // children when they pay.
        let tree_bytes = self
            .states
            .iter()
            .map(RadixFourState::fold_bytes)
            .sum::<usize>()
            .checked_div(self.states.len())
            .unwrap_or(0);
        self.states
            .par_iter_mut()
            .for_each_min_task_bytes(tree_bytes, |state| state.fold(challenge, logical_len));
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
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_binary_field::{BinaryField128, Ghash128};
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_multilinear_util::point::Point;
    use rand::distr::{Distribution, StandardUniform};
    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoroshiro128Plus;

    use super::*;
    use crate::product::math::{fold_dense, sum_adjacent};

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
        check_input_rounds(rng, log_height, &inputs);
    }

    fn check_input_rounds<F: Field>(
        rng: &mut Xoroshiro128Plus,
        log_height: usize,
        inputs: &[Vec<F>],
    ) where
        StandardUniform: Distribution<F>,
    {
        let layers = inputs
            .iter()
            .map(|input| ProductLayers::new(input, log_height))
            .collect::<Vec<_>>();

        // Every retained even depth with at least one parent variable is a child level.
        for depth in (0..=log_height - 3).step_by(2) {
            let variables = log_height - depth - 2;
            for anchor_kind in 0..4 {
                let anchor = (0..variables)
                    .map(|index| match anchor_kind {
                        0 => F::ZERO,
                        1 => F::ONE,
                        2 => F::from_bool(index % 2 != 0),
                        _ => rng.random::<F>(),
                    })
                    .collect::<Vec<_>>();
                for challenge_kind in 0..3 {
                    let mut logical_len = 1usize << variables;
                    let mut batch = RadixFourBatch::new(&layers, depth, logical_len);
                    let mut equality = Point::new(anchor.as_slice()).equality_weights_lsb();
                    let mut suffix = Point::new(&anchor[1..]).equality_weights_lsb();
                    let mut prefix = F::ONE;
                    let batching = rng.random::<F>();

                    // Keep the fully materialized equality table as the independent reference.
                    for (round, &coordinate) in anchor.iter().enumerate() {
                        let one = prefix * coordinate;
                        let pair = [prefix - one, one];
                        for (weights, &weight) in equality.as_chunks::<2>().0.iter().zip(&suffix) {
                            assert_eq!(weights, &[weight * pair[0], weight * pair[1]]);
                        }
                        for batching in [F::ZERO, F::ONE, batching] {
                            assert_eq!(
                                batch.round(&suffix, pair, logical_len, batching),
                                per_node_round(&batch, &equality, logical_len, batching),
                                "depth={depth} len={logical_len} anchor={anchor_kind} challenge={challenge_kind}",
                            );
                        }
                        let challenge = match challenge_kind {
                            0 => F::ZERO,
                            1 => F::ONE,
                            _ => rng.random::<F>(),
                        };
                        batch.fold(challenge, logical_len);
                        fold_dense(&mut equality, challenge);
                        prefix = interpolate_pair(pair, challenge);
                        logical_len /= 2;
                        if logical_len > 1 {
                            sum_adjacent(&mut suffix);
                            // Check the normalized table even when the accumulated prefix is zero.
                            assert_eq!(
                                suffix,
                                Point::new(&anchor[round + 2..]).equality_weights_lsb(),
                            );
                        } else {
                            assert_eq!(equality, [prefix]);
                        }
                    }
                }
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
    fn child_split_matches_an_interleaved_walk() {
        // Lengths through seventeen end each slot on a full, partial or missing final row.
        // Every third entry is one, so trailing identities are trimmed from some children.
        let mut rng = Xoroshiro128Plus::seed_from_u64(0x5107_0003);
        for len in 0..=17usize {
            let values = (0..len)
                .map(|index| {
                    if index % 3 == 2 {
                        BabyBear::ONE
                    } else {
                        rng.random::<BabyBear>()
                    }
                })
                .collect::<Vec<_>>();
            let state = RadixFourState::new(&IdentityPrefix::new(values.clone()), len.div_ceil(4));

            let mut expected: [Vec<BabyBear>; 4] = Default::default();
            for (index, &value) in values.iter().enumerate() {
                expected[index % 4].push(value);
            }
            for (child, expected) in state.children.iter().zip(expected) {
                assert_eq!(child.values, IdentityPrefix::new(expected).values);
            }
        }
    }

    #[test]
    fn round_messages_match_the_per_node_reference() {
        // Unequal, empty and non-multiple-of-four prefixes leave children of different lengths.
        // Several trees fix the batching power each tree must receive.
        let mut rng = Xoroshiro128Plus::seed_from_u64(0x5107_0002);
        let shapes: [&[usize]; 3] = [&[77], &[128, 0], &[13, 128, 91]];
        for prefix_lens in shapes {
            check_rounds::<BinaryField128>(&mut rng, 7, prefix_lens);
            check_rounds::<Ghash128>(&mut rng, 7, prefix_lens);
            check_rounds::<BabyBearQuartic>(&mut rng, 7, prefix_lens);
        }
        for log_height in [3, 4, 6] {
            let capacity = 1usize << log_height;
            let prefix_lens = [capacity, 0, capacity - 1];
            check_rounds::<BinaryField128>(&mut rng, log_height, &prefix_lens);
            check_rounds::<Ghash128>(&mut rng, log_height, &prefix_lens);
            check_rounds::<BabyBearQuartic>(&mut rng, log_height, &prefix_lens);
        }
    }

    fn check_identity_tails<F: Field>(rng: &mut Xoroshiro128Plus)
    where
        StandardUniform: Distribution<F>,
    {
        // Put the longest prefix in each child slot, including in a later tree.
        // Odd lengths keep one partial pair; zero and full lengths cover both extremes.
        for child_len in [0, 1, 2, 3, 15, 16, 17, 31, 32] {
            for slot in 0..4 {
                for trees in 1..=3 {
                    let mut inputs = vec![vec![F::ONE; 128]; trees];
                    if child_len > 0 {
                        let last = 4 * (child_len - 1) + slot;
                        inputs[trees - 1][last] = F::interpolation_node(2);
                        if last > 4 {
                            // A zero product and interior identities must not move the tail cut.
                            inputs[trees - 1][slot] = F::ZERO;
                        }
                    }
                    check_input_rounds(rng, 7, &inputs);
                }
            }
        }
    }

    #[test]
    fn identity_tails_match_the_full_domain_reference() {
        let mut rng = Xoroshiro128Plus::seed_from_u64(0x5107_0005);
        check_identity_tails::<BinaryField128>(&mut rng);
        check_identity_tails::<Ghash128>(&mut rng);
        check_identity_tails::<BabyBearQuartic>(&mut rng);
    }
}
