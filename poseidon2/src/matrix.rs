use alloc::vec::Vec;

use p3_field::AbstractField;
use p3_mds::MdsPermutation;
use p3_symmetric::Permutation;
use rand::distributions::{Distribution, Standard};
use rand::Rng;

/// Multiply a 4-element vector x by
/// [ 5 7 1 3 ]
/// [ 4 6 1 1 ]
/// [ 1 3 5 7 ]
/// [ 1 1 4 6 ].
/// This uses the formula from the start of Appendix B in the Poseidon2 paper, with multiplications unrolled into additions.
/// It is also the matrix used by the Horizon Labs implementation.
#[inline(always)]
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
#[inline(always)]
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
    #[inline(always)]
    fn permute(&self, input: [AF; 4]) -> [AF; 4] {
        let mut output = input;
        self.permute_mut(&mut output);
        output
    }

    #[inline(always)]
    fn permute_mut(&self, input: &mut [AF; 4]) {
        apply_hl_mat4(input)
    }
}
impl<AF: AbstractField> MdsPermutation<AF, 4> for HLMDSMat4 {}

#[derive(Clone, Default)]
pub struct MDSMat4;

impl<AF: AbstractField> Permutation<[AF; 4]> for MDSMat4 {
    #[inline(always)]
    fn permute(&self, input: [AF; 4]) -> [AF; 4] {
        let mut output = input;
        self.permute_mut(&mut output);
        output
    }

    #[inline(always)]
    fn permute_mut(&self, input: &mut [AF; 4]) {
        apply_mat4(input)
    }
}
impl<AF: AbstractField> MdsPermutation<AF, 4> for MDSMat4 {}

#[inline(always)]
pub fn mds_light_permutation<
    AF: AbstractField,
    MdsPerm4: MdsPermutation<AF, 4>,
    const WIDTH: usize,
>(
    state: &mut [AF; WIDTH],
    mdsmat: &MdsPerm4,
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
            for chunk in state.chunks_exact_mut(4) {
                mdsmat.permute_mut(chunk.try_into().unwrap());
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
            state
                .iter_mut()
                .enumerate()
                .for_each(|(i, elem)| *elem += sums[i % 4].clone());
        }

        _ => {
            panic!("Unsupported width");
        }
    }
}

/// A simple struct which holds the constants for the external layer.
#[derive(Clone)]
pub struct ExternalLayerConstants<T, const WIDTH: usize> {
    // Once initialised, these constants should be immutable.
    initial: Vec<[T; WIDTH]>,
    terminal: Vec<[T; WIDTH]>, // We use terminal instead of final as final is a reserved keyword.
}

impl<T, const WIDTH: usize> ExternalLayerConstants<T, WIDTH> {
    pub fn new(initial: Vec<[T; WIDTH]>, terminal: Vec<[T; WIDTH]>) -> Self {
        assert_eq!(
            initial.len(),
            terminal.len(),
            "The number of initial and terminal external rounds should be equal."
        );
        Self { initial, terminal }
    }

    pub fn new_from_rng<R: Rng>(external_round_number: usize, rng: &mut R) -> Self
    where
        Standard: Distribution<[T; WIDTH]>,
    {
        let half_f = external_round_number / 2;
        assert_eq!(
            2 * half_f,
            external_round_number,
            "The total number of external rounds should be even"
        );
        let initial_constants = rng.sample_iter(Standard).take(half_f).collect();
        let terminal_constants = rng.sample_iter(Standard).take(half_f).collect();

        Self::new(initial_constants, terminal_constants)
    }

    pub fn new_from_saved_array<U, const N: usize>(
        [initial, terminal]: [[[U; WIDTH]; N]; 2],
        conversion_fn: fn([U; WIDTH]) -> [T; WIDTH],
    ) -> Self
    where
        T: Clone,
    {
        let initial_consts = initial.map(conversion_fn).to_vec();
        let terminal_consts = terminal.map(conversion_fn).to_vec();
        Self::new(initial_consts, terminal_consts)
    }

    pub fn get_initial_constants(&self) -> &Vec<[T; WIDTH]> {
        &self.initial
    }

    pub fn get_terminal_constants(&self) -> &Vec<[T; WIDTH]> {
        &self.terminal
    }
}

pub trait ExternalLayerConstructor<AF, const WIDTH: usize>
where
    AF: AbstractField,
{
    /// A constructor which internally will convert the supplied
    /// constants into the appropriate form for the implementation.
    fn new_from_constants(external_constants: ExternalLayerConstants<AF::F, WIDTH>) -> Self;
}

/// A trait containing all data needed to implement the external layers of Poseidon2.
pub trait ExternalLayer<AF, const WIDTH: usize, const D: u64>: Sync + Clone
where
    AF: AbstractField,
{
    /// The type used internally by the Poseidon2 implementation.
    /// In the scalar case, InternalState = [AF; WIDTH] but for PackedFields it's faster to use packed vectors.
    type InternalState;

    // permute_state_initial, permute_state_terminal are split as the Poseidon2 specifications are slightly different
    // with the initial rounds involving an extra matrix multiplication.

    /// Compute the initial external permutation.
    /// Implementations will usually not use both constants fields.
    /// Input state will be [AF; WIDTH], output state will be
    /// in appropriate form to feed into the Internal Layer.
    fn permute_state_initial(&self, state: [AF; WIDTH]) -> Self::InternalState;

    /// Compute the terminal external permutation.
    /// Implementations will usually not use both constants fields.
    /// Input state will be in appropriate form from Internal Layer.
    /// Output state will be [AF; WIDTH].
    fn permute_state_terminal(&self, state: Self::InternalState) -> [AF; WIDTH];
}

/// A helper method which allow any field to easily implement the terminal External Layer.
/// This should only be used in places where performance is not critical.
#[inline]
pub fn external_terminal_permute_state<
    AF: AbstractField,
    MdsPerm4: MdsPermutation<AF, 4>,
    const WIDTH: usize,
    const D: u64,
>(
    state: &mut [AF; WIDTH],
    terminal_external_constants: &[[AF::F; WIDTH]],
    mat4: &MdsPerm4,
) {
    for elem in terminal_external_constants.iter() {
        state
            .iter_mut()
            .zip(elem.iter())
            .for_each(|(s, rc)| *s += AF::from_f(*rc));
        state.iter_mut().for_each(|s| *s = s.exp_const_u64::<D>());
        mds_light_permutation(state, mat4);
    }
}

/// A helper method which allow any field to easily implement the initial External Layer.
/// This should only be used in places where performance is not critical.
#[inline]
pub fn external_initial_permute_state<
    AF: AbstractField,
    MdsPerm4: MdsPermutation<AF, 4>,
    const WIDTH: usize,
    const D: u64,
>(
    state: &mut [AF; WIDTH],
    initial_external_constants: &[[AF::F; WIDTH]],
    mat4: &MdsPerm4,
) {
    mds_light_permutation(state, mat4);
    // After the initial mds_light_permutation, the remaining layers are identical
    // to the terminal permutation simply with different constants.
    external_terminal_permute_state::<AF, MdsPerm4, WIDTH, D>(
        state,
        initial_external_constants,
        mat4,
    )
}
