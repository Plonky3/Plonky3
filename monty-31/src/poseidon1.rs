use core::marker::PhantomData;

use p3_field::{Field, InjectiveMonomial};
use p3_poseidon1::external::{
    FullRoundLayer, full_round_initial_permute_state, full_round_terminal_permute_state,
};
use p3_poseidon1::generic::GenericPoseidon1LinearLayers;
use p3_poseidon1::internal::{PartialRoundLayer, partial_permute_state};
use p3_symmetric::Permutation;

use crate::{
    FieldParameters, MDSUtils, MdsMatrixMontyField31, MontyField31, MontyParameters,
    Poseidon1ExternalLayerMonty31, Poseidon1InternalLayerMonty31, RelativelyPrimePower,
};

/// Trait for Poseidon1 partial round scalar operations.
///
/// Provides compile-time dispatch between two partial round strategies:
///
/// - **Sparse decomposition** (`USE_TEXTBOOK = false`, default): Uses the sparse matrix
///   factorization from Appendix B of the Poseidon1 paper. Best for most field/width combos.
///
/// - **Textbook with scalar constants** (`USE_TEXTBOOK = true`): Keeps the fast MDS
///   permutation (e.g., Karatsuba convolution) per round, but folds `state[1..WIDTH]`
///   constants forward so only a scalar is added to `state[0]` each round. Best when
///   the MDS is very fast (e.g., BabyBear width-16 with power-of-2 Karatsuba).
pub trait PartialRoundBaseParameters<MP: MontyParameters, const WIDTH: usize>:
    Clone + Sync
{
    /// Whether to use the textbook (MDS-per-round) path for partial rounds.
    ///
    /// - When `true`, the Karatsuba MDS is applied per round with scalar constants.
    /// - When `false` (default), the sparse matrix decomposition is used.
    const USE_TEXTBOOK: bool = false;

    /// Apply the MDS permutation. Only called when `USE_TEXTBOOK` is `true`.
    fn mds_permute(_state: &mut [MontyField31<MP>; WIDTH]) {}
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub trait PartialRoundParameters<FP: FieldParameters, const WIDTH: usize>:
    PartialRoundBaseParameters<FP, WIDTH> + crate::PartialRoundParametersNeon<FP, WIDTH>
{
}
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
pub trait PartialRoundParameters<FP: FieldParameters, const WIDTH: usize>:
    PartialRoundBaseParameters<FP, WIDTH> + crate::PartialRoundParametersAVX2<FP, WIDTH>
{
}
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub trait PartialRoundParameters<FP: FieldParameters, const WIDTH: usize>:
    PartialRoundBaseParameters<FP, WIDTH> + crate::PartialRoundParametersAVX512<FP, WIDTH>
{
}
#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(target_feature = "avx512f")
    ),
    all(target_arch = "x86_64", target_feature = "avx512f"),
)))]
pub trait PartialRoundParameters<FP: FieldParameters, const WIDTH: usize>:
    PartialRoundBaseParameters<FP, WIDTH>
{
}

impl<FP, const WIDTH: usize, P1P, const D: u64> PartialRoundLayer<MontyField31<FP>, WIDTH, D>
    for Poseidon1InternalLayerMonty31<FP, WIDTH, P1P>
where
    FP: FieldParameters + RelativelyPrimePower<D>,
    P1P: PartialRoundParameters<FP, WIDTH>,
{
    fn permute_state(&self, state: &mut [MontyField31<FP>; WIDTH]) {
        if P1P::USE_TEXTBOOK {
            // Textbook: scalar constant + S-box + Karatsuba MDS per round.
            for &c in &self.internal_constants.textbook_scalar_constants {
                state[0] += c;
                state[0] = InjectiveMonomial::<D>::injective_exp_n(&state[0]);
                P1P::mds_permute(state);
            }
            // Add residual after all partial rounds.
            for (s, &r) in state
                .iter_mut()
                .zip(self.internal_constants.textbook_residual.iter())
            {
                *s += r;
            }
        } else {
            // Sparse decomposition (default).
            partial_permute_state::<MontyField31<FP>, MontyField31<FP>, WIDTH, D>(
                state,
                &self.internal_constants,
            );
        }
    }
}

impl<FP, MU, const WIDTH: usize, const D: u64> FullRoundLayer<MontyField31<FP>, WIDTH, D>
    for Poseidon1ExternalLayerMonty31<FP, MU, WIDTH>
where
    FP: FieldParameters + RelativelyPrimePower<D>,
    MU: MDSUtils + Default,
    MdsMatrixMontyField31<MU>: Permutation<[MontyField31<FP>; WIDTH]>,
{
    fn permute_state_initial(&self, state: &mut [MontyField31<FP>; WIDTH]) {
        let mds = MdsMatrixMontyField31::<MU>::default();
        full_round_initial_permute_state::<MontyField31<FP>, MontyField31<FP>, _, WIDTH, D>(
            state,
            &self.external_constants,
            &mds,
        );
    }

    fn permute_state_terminal(&self, state: &mut [MontyField31<FP>; WIDTH]) {
        let mds = MdsMatrixMontyField31::<MU>::default();
        full_round_terminal_permute_state::<MontyField31<FP>, MontyField31<FP>, _, WIDTH, D>(
            state,
            &self.external_constants,
            &mds,
        );
    }
}

/// Generic Poseidon1 linear layers for MontyField31.
pub struct GenericPoseidon1LinearLayersMonty31<FP, PRBP> {
    _phantom1: PhantomData<FP>,
    _phantom2: PhantomData<PRBP>,
}

impl<FP, PRBP, F, const WIDTH: usize> GenericPoseidon1LinearLayers<F, WIDTH>
    for GenericPoseidon1LinearLayersMonty31<FP, PRBP>
where
    FP: FieldParameters,
    PRBP: PartialRoundBaseParameters<FP, WIDTH>,
    F: Field,
{
}
