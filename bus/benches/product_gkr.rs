//! Product-tree arity benchmarks.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_binary_field::BinaryField128;
use p3_field::{Field, PrimeCharacteristicRing};

type F = BinaryField128;

/// Multiply explicit prefixes into every level consumed by one arity schedule.
fn build_layers(leaves: &[F], log_height: usize, radix_four: bool) -> Vec<Vec<F>> {
    // Benchmark code materializes each input but leaves its identity suffix absent.
    let mut layers = vec![Vec::new(); log_height + 1];
    layers[0] = leaves.to_vec();
    let mut depth = 0;

    while depth < log_height {
        let step = if radix_four && log_height - depth >= 2 {
            2
        } else {
            1
        };
        let arity = 1usize << step;
        layers[depth + step] = layers[depth]
            .chunks(arity)
            .map(|chunk| chunk.iter().copied().product())
            .collect();
        depth += step;
    }

    layers
}

/// Build equality weights in low-variable-first address order.
fn equality_weights(point: &[F]) -> Vec<F> {
    // Each coordinate selects between two complete halves of the current table.
    let mut weights = vec![F::ONE];
    for &coordinate in point {
        let old_len = weights.len();
        weights.resize(2 * old_len, F::ZERO);
        for index in 0..old_len {
            let weight = weights[index];
            weights[index] = weight * (F::ONE - coordinate);
            weights[old_len + index] = weight * coordinate;
        }
    }
    weights
}

/// Bind one low-order variable in an identity-padded prefix.
fn fold_prefix(values: &mut Vec<F>, logical_len: usize, challenge: F) {
    // Missing values represent one and stay implicit after interpolation.
    let output_len = values.len().div_ceil(2);
    for row in 0..output_len {
        let zero = values.get(2 * row).copied().unwrap_or(F::ONE);
        let one = values.get(2 * row + 1).copied().unwrap_or(F::ONE);
        values[row] = zero + challenge * (one - zero);
    }
    values.truncate(output_len.min(logical_len / 2));
}

/// Exercise all product-tree and sumcheck arithmetic for one schedule.
fn arity_workload(inputs: &[Vec<F>; 3], log_height: usize, radix_four: bool) -> F {
    // All layers are retained because GKR descends from root to leaves.
    let layers = inputs
        .each_ref()
        .map(|input| build_layers(input, log_height, radix_four));
    let mut point = Vec::new();
    let mut depth = log_height;
    let mut digest = F::ZERO;

    while depth > 0 {
        let step = if radix_four && depth >= 2 { 2 } else { 1 };
        let arity = 1usize << step;
        let child_depth = depth - step;
        let logical_len = 1usize << point.len();

        // Deinterleave child slots into multilinear tables over the parent point.
        let mut states = layers.each_ref().map(|tree| {
            let mut slots = (0..arity).map(|_| Vec::new()).collect::<Vec<_>>();
            for (index, &value) in tree[child_depth].iter().enumerate() {
                slots[index % arity].push(value);
            }
            slots
        });
        let mut equality = equality_weights(&point);
        let mut remaining = logical_len;
        let mut round_point = Vec::with_capacity(point.len());

        for round in 0..point.len() {
            // Equality raises the product gate's per-variable degree by one.
            let degree = arity + 1;
            for node_index in 0..degree {
                let node = F::interpolation_node(if node_index == 0 { 0 } else { node_index + 1 });
                let mut message = F::ZERO;
                for row in 0..remaining / 2 {
                    let eq = equality[2 * row] + node * (equality[2 * row + 1] - equality[2 * row]);
                    let mut power = F::ONE;
                    for tree in &states {
                        let product = tree
                            .iter()
                            .map(|slot| {
                                let zero = slot.get(2 * row).copied().unwrap_or(F::ONE);
                                let one = slot.get(2 * row + 1).copied().unwrap_or(F::ONE);
                                zero + node * (one - zero)
                            })
                            .product::<F>();
                        message += power * product;
                        power *= F::interpolation_node(11);
                    }
                    digest += eq * message;
                }
            }

            // Fixed coordinates remove transcript hashing from the arity comparison.
            let challenge = F::interpolation_node(round + 17);
            for tree in &mut states {
                for slot in tree {
                    fold_prefix(slot, remaining, challenge);
                }
            }
            fold_prefix(&mut equality, remaining, challenge);
            remaining /= 2;
            round_point.push(challenge);
        }

        // Child claims and branch interpolation feed the next layer's point.
        for tree in &states {
            for slot in tree {
                digest += slot.first().copied().unwrap_or(F::ONE);
            }
        }
        let mut next_point = (0..step)
            .map(|branch| F::interpolation_node(branch + 29))
            .collect::<Vec<_>>();
        next_point.extend(round_point);
        point = next_point;
        depth = child_depth;
    }

    digest
}

/// Compare complete arithmetic schedules before choosing the production arity.
fn product_gkr(criterion: &mut Criterion) {
    // Three unequal prefixes model push, pull, and an auxiliary product tree.
    for log_height in [14, 18] {
        let capacity = 1usize << log_height;
        let inputs = [capacity, 3 * capacity / 4, capacity / 2].map(|length| {
            (0..length)
                .map(|index| F::from_u64((index as u64).wrapping_mul(0x9E37_79B9) + 1))
                .collect::<Vec<_>>()
        });
        let mut group = criterion.benchmark_group(format!("product_gkr_arity/{log_height}"));

        for (name, radix_four) in [("radix_2", false), ("radix_4", true)] {
            group.bench_with_input(
                BenchmarkId::new(name, log_height),
                &inputs,
                |bencher, inputs| {
                    bencher.iter(|| {
                        black_box(arity_workload(black_box(inputs), log_height, radix_four))
                    });
                },
            );
        }
        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(3));
    targets = product_gkr
}
criterion_main!(benches);
