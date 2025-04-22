//! External layers for the Poseidon2 permutation.
//!
//! Poseidon2 applies *external layers* at both the beginning and end of the permutation.
//! These layers are critical for ensuring proper diffusion and enhancing security,
//! particularly against structural and algebraic attacks.
//!
//! An external round consists of:
//! 1. Addition of round constants,
//! 2. Application of a nonlinear S-box,
//! 3. A lightweight matrix multiplication (external linear layer).
//!
//! The constants and linear transformations used in these rounds are designed
//! to complement the internal structure of Poseidon2.
//!
//! Main purpose of these constants:
//! - Inject randomness between rounds.

use alloc::vec::Vec;

use p3_field::{Field, PrimeCharacteristicRing};
use p3_mds::MdsPermutation;
use p3_symmetric::Permutation;
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};

/// Multiply a 4-element vector x by
/// [ 5 7 1 3 ]
/// [ 4 6 1 1 ]
/// [ 1 3 5 7 ]
/// [ 1 1 4 6 ].
/// This uses the formula from the start of Appendix B in the Poseidon2 paper, with multiplications unrolled into additions.
/// It is also the matrix used by the Horizon Labs implementation.
#[inline(always)]
fn apply_hl_mat4<R>(x: &mut [R; 4])
where
    R: PrimeCharacteristicRing,
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
fn apply_mat4<R>(x: &mut [R; 4])
where
    R: PrimeCharacteristicRing,
{
    let t01 = x[0].clone() + x[1].clone();
    let t23 = x[2].clone() + x[3].clone();
    let t0123 = t01.clone() + t23.clone();
    let t01123 = t0123.clone() + x[1].clone();
    let t01233 = t0123 + x[3].clone();
    // The order here is important. Need to overwrite x[0] and x[2] after x[1] and x[3].
    x[3] = t01233.clone() + x[0].double(); // 3*x[0] + x[1] + x[2] + 2*x[3]
    x[1] = t01123.clone() + x[2].double(); // x[0] + 2*x[1] + 3*x[2] + x[3]
    x[0] = t01123 + t01; // 2*x[0] + 3*x[1] + x[2] + x[3]
    x[2] = t01233 + t23; // x[0] + x[1] + 2*x[2] + 3*x[3]
}

/// The 4x4 MDS matrix used by the Horizon Labs implementation of Poseidon2.
///
/// This requires 10 additions and 4 doubles to compute.
#[derive(Clone, Default)]
pub struct HLMDSMat4;

impl<R: PrimeCharacteristicRing> Permutation<[R; 4]> for HLMDSMat4 {
    #[inline(always)]
    fn permute_mut(&self, input: &mut [R; 4]) {
        apply_hl_mat4(input)
    }
}
impl<R: PrimeCharacteristicRing> MdsPermutation<R, 4> for HLMDSMat4 {}

/// The fastest 4x4 MDS matrix.
///
/// This requires 7 additions and 2 doubles to compute.
#[derive(Clone, Default)]
pub struct MDSMat4;

impl<R: PrimeCharacteristicRing> Permutation<[R; 4]> for MDSMat4 {
    #[inline(always)]
    fn permute_mut(&self, input: &mut [R; 4]) {
        apply_mat4(input)
    }
}
impl<R: PrimeCharacteristicRing> MdsPermutation<R, 4> for MDSMat4 {}

/// Implement the matrix multiplication used by the external layer.
///
/// Given a 4x4 MDS matrix M, we multiply by the `4N x 4N` matrix
/// `[[2M M  ... M], [M  2M ... M], ..., [M  M ... 2M]]`.
///
/// # Panics
/// This will panic if `WIDTH` is not supported. Currently, the
/// supported `WIDTH` values are 2, 3, 4, 8, 12, 16, 20, 24.`
#[inline(always)]
pub fn mds_light_permutation<
    R: PrimeCharacteristicRing,
    MdsPerm4: MdsPermutation<R, 4>,
    const WIDTH: usize,
>(
    state: &mut [R; WIDTH],
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
            let sums: [R; 4] = core::array::from_fn(|k| {
                (0..WIDTH)
                    .step_by(4)
                    .map(|j| state[j + k].clone())
                    .sum::<R>()
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

/// A struct which stores round-specific constants for both initial and terminal external layers.
#[derive(Debug, Clone)]
pub struct ExternalLayerConstants<T, const WIDTH: usize> {
    /// Constants applied during each initial external round.
    ///
    /// Used in `permute_state_initial`. Each `[T; WIDTH]` is a full-width vector of constants.
    initial: Vec<[T; WIDTH]>,

    /// Constants applied during each terminal external round.
    ///
    /// Used in `permute_state_terminal`. The term "terminal" avoids using Rust’s reserved word `final`.
    terminal: Vec<[T; WIDTH]>,
}

impl<T, const WIDTH: usize> ExternalLayerConstants<T, WIDTH> {
    /// Create a new instance of external layer constants.
    ///
    /// # Panics
    /// Panics if `initial.len() != terminal.len()` since the Poseidon2 spec requires
    /// the same number of initial and terminal rounds to maintain symmetry.
    pub fn new(initial: Vec<[T; WIDTH]>, terminal: Vec<[T; WIDTH]>) -> Self {
        assert_eq!(
            initial.len(),
            terminal.len(),
            "The number of initial and terminal external rounds should be equal."
        );
        Self { initial, terminal }
    }

    /// Randomly generate a new set of external constants using a provided RNG.
    ///
    /// # Arguments
    /// - `external_round_number`: Total number of external rounds (must be even).
    /// - `rng`: A random number generator that supports uniform sampling.
    ///
    /// The constants are split equally between the initial and terminal rounds.
    ///
    /// # Panics
    /// Panics if `external_round_number` is not even.
    pub fn new_from_rng<R: Rng>(external_round_number: usize, rng: &mut R) -> Self
    where
        StandardUniform: Distribution<[T; WIDTH]>,
    {
        let half_f = external_round_number / 2;
        assert_eq!(
            2 * half_f,
            external_round_number,
            "The total number of external rounds should be even"
        );
        let initial_constants = rng.sample_iter(StandardUniform).take(half_f).collect();
        let terminal_constants = rng.sample_iter(StandardUniform).take(half_f).collect();

        Self::new(initial_constants, terminal_constants)
    }

    /// Construct constants from statically stored arrays, using a conversion function.
    ///
    /// This is useful when deserializing precomputed constants or embedding
    /// them directly in the codebase (e.g., from `[[[u32; WIDTH]; N]; 2]` arrays).
    ///
    /// # Arguments
    /// - `initial`, `terminal`: Two fixed-size arrays of size `N` containing round constants.
    /// - `conversion_fn`: A function to convert from the source type `U` to `T`.
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

    /// Get a reference to the list of initial round constants.
    ///
    /// These are used in the first half of the external rounds.
    pub const fn get_initial_constants(&self) -> &Vec<[T; WIDTH]> {
        &self.initial
    }

    /// Get a reference to the list of terminal round constants.
    ///
    /// These are used in the second half (terminal rounds) of the external layer.
    pub const fn get_terminal_constants(&self) -> &Vec<[T; WIDTH]> {
        &self.terminal
    }
}

/// Initialize an external layer from a set of constants.
pub trait ExternalLayerConstructor<F, const WIDTH: usize>
where
    F: Field,
{
    /// A constructor which internally will convert the supplied
    /// constants into the appropriate form for the implementation.
    fn new_from_constants(external_constants: ExternalLayerConstants<F, WIDTH>) -> Self;
}

/// A trait containing all data needed to implement the external layers of Poseidon2.
pub trait ExternalLayer<R, const WIDTH: usize, const D: u64>: Sync + Clone
where
    R: PrimeCharacteristicRing,
{
    // permute_state_initial, permute_state_terminal are split as the Poseidon2 specifications are slightly different
    // with the initial rounds involving an extra matrix multiplication.

    /// Perform the initial external layers of the Poseidon2 permutation on the given state.
    fn permute_state_initial(&self, state: &mut [R; WIDTH]);

    /// Perform the terminal external layers of the Poseidon2 permutation on the given state.
    fn permute_state_terminal(&self, state: &mut [R; WIDTH]);
}

/// Applies the terminal external rounds of the Poseidon2 permutation.
///
/// Each external round consists of three steps:
/// 1. Adding round constants to each element of the state.
/// 2. Apply the S-box to each element of the state.
/// 3. Applying an external linear layer (based on a `4x4` MDS matrix).
///
/// # Parameters
/// - `state`: The current state of the permutation (size `WIDTH`).
/// - `terminal_external_constants`: Per-round constants which are added to each state element.
/// - `add_rc_and_sbox`: A function that adds the round constant and applies the S-box to a given element.
/// - `mat4`: The 4x4 MDS matrix used in the external linear layer.
#[inline]
pub fn external_terminal_permute_state<
    R: PrimeCharacteristicRing,
    CT: Copy, // Whatever type the constants are stored as.
    MdsPerm4: MdsPermutation<R, 4>,
    const WIDTH: usize,
>(
    state: &mut [R; WIDTH],
    terminal_external_constants: &[[CT; WIDTH]],
    add_rc_and_sbox: fn(&mut R, CT),
    mat4: &MdsPerm4,
) {
    for elem in terminal_external_constants {
        state
            .iter_mut()
            .zip(elem.iter())
            .for_each(|(s, &rc)| add_rc_and_sbox(s, rc));
        mds_light_permutation(state, mat4);
    }
}

/// Applies the initial external rounds of the Poseidon2 permutation.
///
/// Apply the external linear layer and run a sequence of standard external rounds consisting of
/// 1. Adding round constants to each element of the state.
/// 2. Apply the S-box to each element of the state.
/// 3. Applying an external linear layer (based on a `4x4` MDS matrix).
///
/// # Parameters
/// - `state`: The state array at the start of the permutation.
/// - `initial_external_constants`: Per-round constants which are added to each state element.
/// - `add_rc_and_sbox`: A function that adds the round constant and applies the S-box to a given element.
/// - `mat4`: The 4x4 MDS matrix used in the external linear layer.
#[inline]
pub fn external_initial_permute_state<
    R: PrimeCharacteristicRing,
    CT: Copy, // Whatever type the constants are stored as.
    MdsPerm4: MdsPermutation<R, 4>,
    const WIDTH: usize,
>(
    state: &mut [R; WIDTH],
    initial_external_constants: &[[CT; WIDTH]],
    add_rc_and_sbox: fn(&mut R, CT),
    mat4: &MdsPerm4,
) {
    mds_light_permutation(state, mat4);
    // After the initial mds_light_permutation, the remaining layers are identical
    // to the terminal permutation simply with different constants.
    external_terminal_permute_state(state, initial_external_constants, add_rc_and_sbox, mat4)
}
