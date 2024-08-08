use p3_field::AbstractField;
use p3_mds::MdsPermutation;
use p3_symmetric::Permutation;

use crate::Poseidon2ExternalPackedConstants;

/// Multiply a 4-element vector x by
/// [ 5 7 1 3 ]
/// [ 4 6 1 1 ]
/// [ 1 3 5 7 ]
/// [ 1 1 4 6 ].
/// This uses the formula from the start of Appendix B in the Poseidon2 paper, with multiplications unrolled into additions.
/// It is also the matrix used by the Horizon Labs implementation.
fn apply_hl_mat4<AF>(x: &mut [AF; 4])
where
    AF: AbstractField,
{
    let t0 = x[0].clone() + x[1].clone();
    let t1 = x[2].clone() + x[3].clone();
    let t2 = x[1].clone() + x[1].clone() + t1.clone();
    let t3 = x[3].clone() + x[3].clone() + t0.clone();
    let t4 = t1.double().double() + t3.clone();
    let t5 = t0.double().double() + t2.clone();
    let t6 = t3 + t5.clone();
    let t7 = t2 + t4.clone();
    x[0] = t6;
    x[1] = t5;
    x[2] = t7;
    x[3] = t4;
}

// It turns out we can find a 4x4 matrix which is more efficient than the above.

/// Multiply a 4-element vector x by:
/// [ 2 3 1 1 ]
/// [ 1 2 3 1 ]
/// [ 1 1 2 3 ]
/// [ 3 1 1 2 ].
fn apply_mat4<AF>(x: &mut [AF; 4])
where
    AF: AbstractField,
{
    let t01 = x[0].clone() + x[1].clone();
    let t23 = x[2].clone() + x[3].clone();
    let t0123 = t01.clone() + t23.clone();
    let t01123 = t0123.clone() + x[1].clone();
    let t01233 = t0123.clone() + x[3].clone();
    // The order here is important. Need to overwrite x[0] and x[2] after x[1] and x[3].
    x[3] = t01233.clone() + x[0].double(); // 3*x[0] + x[1] + x[2] + 2*x[3]
    x[1] = t01123.clone() + x[2].double(); // x[0] + 2*x[1] + 3*x[2] + x[3]
    x[0] = t01123 + t01; // 2*x[0] + 3*x[1] + x[2] + x[3]
    x[2] = t01233 + t23; // x[0] + x[1] + 2*x[2] + 3*x[3]
}

// The 4x4 MDS matrix used by the Horizon Labs implementation of Poseidon2.
#[derive(Clone, Default)]
pub struct HLMDSMat4;

impl<AF: AbstractField> Permutation<[AF; 4]> for HLMDSMat4 {
    fn permute(&self, input: [AF; 4]) -> [AF; 4] {
        let mut output = input;
        self.permute_mut(&mut output);
        output
    }

    fn permute_mut(&self, input: &mut [AF; 4]) {
        apply_hl_mat4(input)
    }
}
impl<AF: AbstractField> MdsPermutation<AF, 4> for HLMDSMat4 {}

#[derive(Clone, Default)]
pub struct MDSMat4;

impl<AF: AbstractField> Permutation<[AF; 4]> for MDSMat4 {
    fn permute(&self, input: [AF; 4]) -> [AF; 4] {
        let mut output = input;
        self.permute_mut(&mut output);
        output
    }

    fn permute_mut(&self, input: &mut [AF; 4]) {
        apply_mat4(input)
    }
}
impl<AF: AbstractField> MdsPermutation<AF, 4> for MDSMat4 {}

fn mds_light_permutation<AF: AbstractField, MdsPerm4: MdsPermutation<AF, 4>, const WIDTH: usize>(
    state: &mut [AF; WIDTH],
    mdsmat: MdsPerm4,
) {
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

        4 | 8 | 12 | 16 | 20 | 24 => {
            // First, we apply M_4 to each consecutive four elements of the state.
            // In Appendix B's terminology, this replaces each x_i with x_i'.
            for i in (0..WIDTH).step_by(4) {
                let mut state_4 = [
                    state[i].clone(),
                    state[i + 1].clone(),
                    state[i + 2].clone(),
                    state[i + 3].clone(),
                ];
                mdsmat.permute_mut(&mut state_4);
                state[i..i + 4].clone_from_slice(&state_4);
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

/// A trait containing all data needed to implement the external layers of Poseidon2.
pub trait ExternalLayer<AF, PackedConstants, const WIDTH: usize, const D: u64>:
    Sync + Clone
where
    AF: AbstractField,
    PackedConstants: Poseidon2ExternalPackedConstants<AF::F, WIDTH>,
{
    /// The type used internally by the Poseidon2 implementation.
    /// In the scalar case, InternalState = [AF; WIDTH] but for PackedFields it's faster to use packed vectors.
    type InternalState;

    /// ArrayState should be [InternalState; N] for some small value of N depending on the precise packed field.
    /// The Poseidon2 permutation will be mapped across this array.
    type ArrayState: AsMut<[Self::InternalState]>;

    /// Convert data from input type which will always be [AF; WIDTH] into the internal representation.
    fn to_internal_rep(&self, state: [AF; WIDTH]) -> Self::ArrayState;

    // permute_state_initial, permute_state_final are split as the Poseidon2 specifications are slightly different
    // with the initial rounds involving an extra matrix multiplication.

    /// Compute the initial external permutation.
    fn permute_state_initial(
        &self,
        state: &mut Self::InternalState,
        initial_external_constants: &[[AF::F; WIDTH]],
        initial_external_packed_constants: &[PackedConstants::ExternalConstantsType],
    );

    /// Compute the final external permutation.
    fn permute_state_final(
        &self,
        state: &mut Self::InternalState,
        final_external_constants: &[[AF::F; WIDTH]],
        final_external_packed_constants: &[PackedConstants::ExternalConstantsType],
    );

    /// Convert data back from the internal representation to the expected output type [AF; WIDTH].
    fn to_output_rep(&self, state: Self::ArrayState) -> [AF; WIDTH];
}

// A pair of helper methods which allow any field to easily implement External Layer.
// These should be used in places where performance is not critical.
#[inline]
pub fn external_final_permute_state<AF: AbstractField, const WIDTH: usize, const D: u64>(
    state: &mut [AF; WIDTH],
    final_external_constants: &[[AF::F; WIDTH]],
) {
    for elem in final_external_constants.iter() {
        state
            .iter_mut()
            .zip(elem.iter())
            .for_each(|(s, rc)| *s += AF::from_f(*rc));
        state.iter_mut().for_each(|s| *s = s.exp_const_u64::<D>());
        mds_light_permutation(state, MDSMat4);
    }
}

#[inline]
pub fn external_initial_permute_state<AF: AbstractField, const WIDTH: usize, const D: u64>(
    state: &mut [AF; WIDTH],
    initial_external_constants: &[[AF::F; WIDTH]],
) {
    mds_light_permutation(state, MDSMat4);
    // After the initial mds_light_permutation, the remaining layers are identical
    // to the final permutation simply with different constants.
    external_final_permute_state::<AF, WIDTH, D>(state, initial_external_constants)
}
