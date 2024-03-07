use p3_field::{AbstractField, PrimeField};
use p3_symmetric::Permutation;

extern crate alloc;

// The t x t matrix M_E := circ(2M_4, M_4, ..., M_4), where M_4 is the 4 x 4 matrix
// [ 5 7 1 3 ]
// [ 4 6 1 1 ]
// [ 1 3 5 7 ]
// [ 1 1 4 6 ].
// The permutation calculation is based on Appendix B from the Poseidon2 paper.
#[derive(Copy, Clone, Default)]
pub struct Poseidon2MEMatrix<const WIDTH: usize, const D: u64>;

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

// At some point we should switch this matrix to:
// [ 2 3 1 1 ]
// [ 1 2 3 1 ]
// [ 1 1 2 3 ]
// [ 3 1 1 2 ].
// This is more efficient than the one above (11 additions vs 16 additions) and leads to a ~5% speed up.
// Unfortunately it breaks all the tests as we are testing against the implementation from zkhash.
// Hence will leave this as a comment for now and implement later.
// fn apply_m_4<AF>(x: &mut [AF])
// where
//     AF: AbstractField,
//     AF::F: PrimeField,
// {
//     let t01 = x[0].clone() + x[1].clone();
//     let t23 = x[2].clone() + x[3].clone();
//     let t0123 = t01.clone() + t23.clone();
//     let t01123 = t0123.clone() + x[1].clone();
//     let t01233 = t0123.clone() + x[3].clone();
//     x[3] = t01233.clone() + x[0].clone() + x[0].clone(); // 3*x[0] + x[1] + x[2] + 2*x[3]
//     x[1] = t01123.clone() + x[2].clone() + x[2].clone(); // x[0] + 2*x[1] + 3*x[2] + x[3]
//     x[0] = t01123 + t01; // 2*x[0] + 3*x[1] + x[2] + x[3]
//     x[2] = t01233 + t23; // x[0] + x[1] + 2*x[2] + 3*x[3]
// }

impl<AF, const WIDTH: usize, const D: u64> Permutation<[AF; WIDTH]> for Poseidon2MEMatrix<WIDTH, D>
where
    AF: AbstractField,
    AF::F: PrimeField,
{
    fn permute_mut(&self, state: &mut [AF; WIDTH]) {

        match WIDTH {
            2 => {
                let sum = state[0].clone() + state[1].clone();
                state[0] += sum.clone();
                state[1] += sum;
            }

            3 => {
                let sum = state[0].clone() + state[1].clone() + state[2].clone();
                state[0] += sum.clone();
                state[1] += sum.clone();
                state[2] += sum;
            }

            4 | 8 | 12 | 16 | 20 | 24  => {
                // First, we apply M_4 to each consecutive four elements of the state.
                // In Appendix B's terminology, this replaces each x_i with x_i'.
                for i in (0..WIDTH).step_by(4) {
                    apply_m_4(&mut state[i..i + 4]);
                }

                // Now, we apply the outer circulant matrix (to compute the y_i values).

                // We first precompute the four sums of every four elements.
                let sums: [AF; 4] = core::array::from_fn(|k| {
                    (0..WIDTH)
                        .step_by(4)
                        .map(|j| state[j + k].clone())
                        .sum::<AF>()
                });

                // The formula for each y_i involves 2x_i' term and x_j' terms for each j that equals i mod 4.
                // In other words, we can add a single copy of x_i' to the appropriate one of our precomputed sums
                for i in 0..WIDTH {
                    state[i] += sums[i % 4].clone();
                }
            }

            _ => {
                panic!("Unsupported width");
            }
        }
    }
}
