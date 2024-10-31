use std::collections::HashMap;

use p3_baby_bear::BabyBear;
use p3_matrix::dense::RowMajorMatrix;
use rand::{thread_rng, Rng};

pub fn get_random_leaves(
    num_matrices: usize,
    max_rows: usize,
    max_cols: usize,
) -> Vec<RowMajorMatrix<BabyBear>> {
    let mut pow2_to_size = HashMap::new();
    (0..num_matrices)
        .map(|_| {
            let mut n_rows = rand::thread_rng().gen_range(1..max_rows);
            let n_cols = rand::thread_rng().gen_range(1..max_cols);

            // Same-power-of-two row numbers must match
            n_rows = pow2_to_size
                .entry(n_rows.next_power_of_two())
                .or_insert(n_rows)
                .to_owned();

            RowMajorMatrix::<BabyBear>::rand(&mut thread_rng(), n_rows, n_cols)
        })
        .collect()
}
