//! The MDS permutation for 4 elements corresponding to the matrix M8,4 from
//! https://eprint.iacr.org/2018/260.pdf.
//!
//! See also: [Poseidon2](https://eprint.iacr.org/2023/323.pdf).

use p3_field::AbstractField;
use p3_symmetric::Permutation;

use crate::MdsPermutation;

/// Implements the permutation given by the matrix:
///  ```ignore
///     M4 = [[5, 7, 1, 3],
///           [4, 6, 1, 1],
///           [1, 3, 5, 7],
///           [1, 1, 4, 6]];
///   ```
#[derive(Debug, Clone, Copy, Default)]
pub struct M4Mds;

impl<AF: AbstractField> Permutation<[AF; 4]> for M4Mds {
    fn permute_mut(&self, input: &mut [AF; 4]) {
        // Implements the permutation given by the matrix M4 with multiplications unrolled as
        // additions and doublings.
        let mut t_0 = input[0].clone();
        t_0 += input[1].clone();
        let mut t_1 = input[2].clone();
        t_1 += input[3].clone();
        let mut t_2 = input[1].clone();
        t_2 += t_2.clone();
        t_2 += t_1.clone();
        let mut t_3 = input[3].clone();
        t_3 += t_3.clone();
        t_3 += t_0.clone();
        let mut t_4 = t_1.clone();
        t_4 += t_4.clone();
        t_4 += t_4.clone();
        t_4 += t_3.clone();
        let mut t_5 = t_0.clone();
        t_5 += t_5.clone();
        t_5 += t_5.clone();
        t_5 += t_2.clone();
        let mut t_6 = t_3.clone();
        t_6 += t_5.clone();
        let mut t_7 = t_2.clone();
        t_7 += t_4.clone();
        input[0] = t_6;
        input[1] = t_5;
        input[2] = t_7;
        input[3] = t_4;
    }
}

impl<AF: AbstractField> MdsPermutation<AF, 4> for M4Mds {}
