use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::{AbstractField, PrimeField};
use p3_symmetric::Permutation;

extern crate alloc;

// The t x t matrix M_E := circ(2M_4, M_4, ..., M_4), where M_4 is the 4 x 4 matrix
// [ 5 7 1 3 ]
// [ 4 6 1 1 ]
// [ 1 3 5 7 ]
// [ 1 1 4 6 ].
// The permutation calculation is based on Appendix B from the Poseidon2 paper.
#[derive(Clone)]
pub struct Poseidon2MEMatrix<AF, const WIDTH: usize, const D: u64>
where
    AF: AbstractField,
    AF::F: PrimeField,
{
    _phantom: PhantomData<AF>,
}

impl<AF, const WIDTH: usize, const D: u64> Poseidon2MEMatrix<AF, WIDTH, D>
where
    AF: AbstractField,
    AF::F: PrimeField,
{
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

// Multiply a 4-element vector x by M_4, in place.
// This uses the formula from the start of Appendix B, with multiplications unrolled into additions.
fn apply_m_4<AF>(x: &mut [AF])
where
    AF: AbstractField,
    AF::F: PrimeField,
{
    let t0 = x[0].clone() + x[1].clone();
    let t1 = x[2].clone() + x[3].clone();
    let t2 = x[1].clone() + x[1].clone() + t1.clone();
    let t3 = x[3].clone() + x[3].clone() + t0.clone();
    let t4 = t1.clone() + t1.clone() + t1.clone() + t1 + t3.clone();
    let t5 = t0.clone() + t0.clone() + t0.clone() + t0 + t2.clone();
    let t6 = t3 + t5.clone();
    let t7 = t2 + t4.clone();
    x[0] = t6;
    x[1] = t5;
    x[2] = t7;
    x[3] = t4;
}

impl<AF, const WIDTH: usize, const D: u64> Permutation<[AF; WIDTH]>
    for Poseidon2MEMatrix<AF, WIDTH, D>
where
    AF: AbstractField,
    AF::F: PrimeField,
{
    fn permute_mut(&self, state: &mut [AF; WIDTH]) {
        // First, we apply M_4 to each consecutive four elements of the state.
        // In Appendix B's terminology, this replaces each x_i with x_i'.
        for i in 0..WIDTH / 4 {
            apply_m_4(&mut state[4 * i..4 * i + 4]);
        }

        // Now, we apply the outer circulant matrix (to compute the y_i values).

        // We first precompute the four sums of every four elements.
        let sums: Vec<AF> = (0..4)
            .map(|k| {
                (0..state.len() / 4)
                    .map(|j| state[4 * j + k].clone())
                    .sum::<AF>()
            })
            .collect();

        // The formula for each y_i involves 2x_i' term and x_j' terms for each j that equals i mod 4.
        // In other words, we can add a single copy of x_i' to the appropriate one of our precomputed sums
        for i in 0..state.len() {
            state[i] += sums[i % 4].clone();
        }
    }
}
