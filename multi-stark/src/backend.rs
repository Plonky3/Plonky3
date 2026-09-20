//! Zerocheck round kernels, selected by type.
//!
//! The zerocheck prover owns the transcript, the stages, and the claims.
//! For each stage it asks a backend for the arithmetic of one round:
//!
//! ```text
//!     round0   : first round polynomial of a newly activated stage
//!     fold0    : bind that stage's first variable
//!     round    : round polynomial of an already folded stage
//!     fold     : bind the next variable of an already folded stage
//!     openings : column values once every variable is bound
//! ```
//!
//! A backend chooses how these values are computed, not what they are.
//! The transcript and the proof therefore do not depend on the backend.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_air::Air;
use p3_field::{Algebra, ExtensionField, Field, HasSubfield};
use p3_multilinear_util::poly::Poly;

use crate::folder::{InteractionMultilinearFolder, MultilinearFolder, ProverAir};
use crate::packed_ext::PackedRepr;
use crate::rounds::{AirOpenings, RoundStateBase, RoundStateExt};
use crate::sliced::SlicedFolder;
use crate::subfield::{SubfieldAcc, SubfieldVar};

// The trait lives in a private module, so no caller outside this crate can name or call it.
mod private {
    use alloc::vec::Vec;

    use p3_field::{ExtensionField, Field};
    use p3_multilinear_util::poly::Poly;

    use crate::rounds::{AirOpenings, RoundStateBase, RoundStateExt};

    /// The per-stage kernels the zerocheck prover calls.
    ///
    /// A round polynomial comes back as the stage's internal evaluations over the challenge field.
    /// The prover recovers the node-one value from the stage claim, then extends and accumulates.
    // The kernels take the crate-private round states.
    #[expect(private_interfaces)]
    pub trait Dispatch<F: Field, EF: ExtensionField<F>, A> {
        /// The field a folded stage computes its rounds in, isomorphic to the challenge field.
        type Repr;

        /// Evaluate the first round polynomial of a newly activated stage.
        fn round0(state: &mut RoundStateBase<'_, '_, A, F, EF>, eq_suffix: &Poly<EF>) -> Vec<EF>;

        /// Bind the first variable of a stage at `r`.
        fn fold0<'air, 'data>(
            state: RoundStateBase<'air, 'data, A, F, EF>,
            r: EF,
        ) -> RoundStateExt<'air, 'data, A, F, EF, Self::Repr>;

        /// Evaluate the round polynomial of a stage whose first variable is bound.
        fn round(
            state: &mut RoundStateExt<'_, '_, A, F, EF, Self::Repr>,
            eq_suffix: &Poly<EF>,
        ) -> Vec<EF>;

        /// Bind the next variable of a folded stage at `r`.
        fn fold(state: &mut RoundStateExt<'_, '_, A, F, EF, Self::Repr>, r: EF);

        /// Split a fully bound stage into per-AIR openings, each tagged with its caller index.
        fn openings(
            state: RoundStateExt<'_, '_, A, F, EF, Self::Repr>,
        ) -> Vec<(usize, AirOpenings<EF>)>;
    }
}

/// A backend for the zerocheck rounds of [`crate::prove_with_backend`].
///
/// The trait is sealed: a type is a backend exactly when it implements the crate's private
/// kernels. [`GenericBackend`] is one for every AIR [`crate::prove`] accepts.
/// A backend decides how round polynomials, folds, and openings are computed.
/// It never changes the transcript, so every backend produces the same proof.
pub trait ZerocheckBackend<F: Field, EF: ExtensionField<F>, A>:
    private::Dispatch<F, EF, A>
{
}

/// The backend for every field pair and every AIR the prover supports.
///
/// ```text
///     round 0      : base-field rows, extension-field accumulators
///     later rounds : extension-field columns
/// ```
///
/// Both run the SIMD-packed kernel while half the residual rows fill a lane, and the scalar
/// kernel after that.
#[derive(Debug)]
pub struct GenericBackend;

// The sealed kernels take the crate-private round states.
#[expect(private_interfaces)]
impl<F, EF, A> private::Dispatch<F, EF, A> for GenericBackend
where
    F: Field,
    EF: ExtensionField<F>,
    A: ProverAir<F, EF>,
    EF::ExtensionPacking: From<EF> + From<F::Packing>,
{
    type Repr = EF;

    fn round0(state: &mut RoundStateBase<'_, '_, A, F, EF>, eq_suffix: &Poly<EF>) -> Vec<EF> {
        state.round_poly(eq_suffix)
    }

    fn fold0<'air, 'data>(
        state: RoundStateBase<'air, 'data, A, F, EF>,
        r: EF,
    ) -> RoundStateExt<'air, 'data, A, F, EF> {
        state.fold(r)
    }

    fn round(state: &mut RoundStateExt<'_, '_, A, F, EF>, eq_suffix: &Poly<EF>) -> Vec<EF> {
        state.round_poly(eq_suffix)
    }

    fn fold(state: &mut RoundStateExt<'_, '_, A, F, EF>, r: EF) {
        state.fold(r);
    }

    fn openings(state: RoundStateExt<'_, '_, A, F, EF>) -> Vec<(usize, AirOpenings<EF>)> {
        state.into_openings()
    }
}

impl<T, F, EF, A> ZerocheckBackend<F, EF, A> for T
where
    F: Field,
    EF: ExtensionField<F>,
    T: private::Dispatch<F, EF, A>,
{
}

/// The backend that evaluates a stage's first round inside the small subfield `S` when it fits.
///
/// ```text
///     stage sliced   : its first rounds sixty-four rows at a time on bit planes of GF(4),
///                      and every column folds from the planes once those rounds are done
///     round 0, fits S: expressions in S, alpha-batched over the challenge field
///     fold 0,  fits S: lo + r * (hi - lo) with hi - lo applied as an element of S
///     otherwise      : as GenericBackend
///     later rounds   : as GenericBackend
/// ```
///
/// A stage is sliced when `S` is `GF(4)`, the stage fits it, and each half of the stage holds at
/// least sixty-four rows; see [`crate::sliced`].
///
/// A stage fits `S` when all of these hold:
///
/// - no AIR in it declares a lookup;
/// - the challenge field embeds `S` the way the trace field does;
/// - the steps between its first-round interpolation nodes lie in `S`;
/// - its public values lie in `S`;
/// - its main, preprocessed, and periodic cells lie in `S`.
///
/// An AIR constant outside `S` shows up while the round runs, and the round is then recomputed
/// by the generic kernel. Every fallback emits a `debug` tracing event naming its reason.
///
/// Every round polynomial is the one [`GenericBackend`] computes, so the proof is the same.
#[derive(Debug)]
pub struct SubfieldBackend<S>(PhantomData<fn() -> S>);

// The sealed kernels take the crate-private round states.
#[expect(private_interfaces)]
impl<F, EF, A, S> private::Dispatch<F, EF, A> for SubfieldBackend<S>
where
    S: Field,
    F: HasSubfield<S>,
    EF: ExtensionField<F> + HasSubfield<S>,
    A: ProverAir<F, EF>
        + for<'a> Air<MultilinearFolder<'a, F, SubfieldVar<F, S>, SubfieldAcc<EF, S>>>
        + for<'a> Air<SlicedFolder<'a, F, S, EF>>,
    EF::ExtensionPacking: From<EF> + From<F::Packing>,
{
    type Repr = EF;

    fn round0(state: &mut RoundStateBase<'_, '_, A, F, EF>, eq_suffix: &Poly<EF>) -> Vec<EF> {
        state
            .round_poly_sliced::<S, EF>(eq_suffix)
            .or_else(|| state.round_poly_subfield::<S>(eq_suffix))
            .unwrap_or_else(|| state.round_poly(eq_suffix))
    }

    fn fold0<'air, 'data>(
        state: RoundStateBase<'air, 'data, A, F, EF>,
        r: EF,
    ) -> RoundStateExt<'air, 'data, A, F, EF> {
        if state.is_sliced() {
            state.fold_sliced(r)
        } else if state.fits_subfield() {
            state.fold_subfield::<S>(r)
        } else {
            state.fold(r)
        }
    }

    fn round(state: &mut RoundStateExt<'_, '_, A, F, EF>, eq_suffix: &Poly<EF>) -> Vec<EF> {
        state.round_poly_sliced::<S>(eq_suffix).unwrap_or_else(|| {
            state.unslice_packed::<S>();
            <GenericBackend as private::Dispatch<F, EF, A>>::round(state, eq_suffix)
        })
    }

    fn fold(state: &mut RoundStateExt<'_, '_, A, F, EF>, r: EF) {
        if !state.fold_sliced(r) {
            <GenericBackend as private::Dispatch<F, EF, A>>::fold(state, r);
        }
    }

    fn openings(state: RoundStateExt<'_, '_, A, F, EF>) -> Vec<(usize, AirOpenings<EF>)> {
        <GenericBackend as private::Dispatch<F, EF, A>>::openings(state)
    }
}

/// The backend that runs a stage's rounds after the first in a field `R` isomorphic to the
/// challenge field.
///
/// ```text
///     stage sliced : as SubfieldBackend<S>, its sums accumulated in R, and every column
///                    folding from the planes into R once its sliced rounds are done
///     round 0      : as SubfieldBackend<S>
///     fold 0       : every column folds straight into R
///     later rounds : columns, selectors, and AIR expressions in R, one lane group of rows
///                    at a time, one row at a time once the rows no longer fill a group
/// ```
///
/// Two representations of one field can multiply at very different costs. In the tower basis
/// of `GF(2^128)` a product changes basis three times around one carryless multiply. In its
/// polynomial basis the product is that multiply alone.
///
/// What each round reads crosses into `R` at most once per round:
///
/// ```text
///     once per stage : alpha powers, lookup coefficients, repeat-last tails, selector prefix
///     once per round : eq weights, interpolation steps, challenge
///     never          : zerocheck point, beta powers, lookup scale, claims, interpolators
/// ```
///
/// Each round's per-node sums and the final openings cross back into the challenge field.
///
/// In a stage that fits `S`, every pair of cells folds into `R` through two table lookups.
/// Any other stage converts each cell pair and folds it with a product in `R`.
/// A stage declaring a lookup runs its later rounds in `R` like any other.
///
/// `R::from` and `EF::from` must be mutually inverse field isomorphisms, and `R`'s embedding of
/// the trace field must be the challenge field's followed by `R::from`. The interpolation steps
/// are the challenge field's own, carried into `R`, never `R`'s interpolation nodes.
/// Every round polynomial is then the one [`GenericBackend`] computes, so the proof is the same.
#[derive(Debug)]
pub struct ReprBackend<S, R>(PhantomData<fn() -> (S, R)>);

// The sealed kernels take the crate-private round states.
#[expect(private_interfaces)]
impl<F, EF, A, S, R> private::Dispatch<F, EF, A> for ReprBackend<S, R>
where
    S: Field,
    F: HasSubfield<S>,
    EF: ExtensionField<F> + HasSubfield<S> + From<R>,
    R: Field + Algebra<F> + From<EF>,
    R::Packing: Algebra<F::Packing>,
    A: ProverAir<F, EF>
        + for<'a> Air<MultilinearFolder<'a, F, SubfieldVar<F, S>, SubfieldAcc<EF, S>>>
        + for<'a> Air<SlicedFolder<'a, F, S, R>>
        + for<'a> Air<MultilinearFolder<'a, F, R, R>>
        + for<'a> Air<InteractionMultilinearFolder<'a, F, R, R>>
        + for<'a> Air<MultilinearFolder<'a, F, PackedRepr<F, R>, PackedRepr<F, R>>>
        + for<'a> Air<InteractionMultilinearFolder<'a, F, PackedRepr<F, R>, PackedRepr<F, R>>>,
    EF::ExtensionPacking: From<EF> + From<F::Packing>,
{
    type Repr = R;

    fn round0(state: &mut RoundStateBase<'_, '_, A, F, EF>, eq_suffix: &Poly<EF>) -> Vec<EF> {
        state
            .round_poly_sliced::<S, R>(eq_suffix)
            .or_else(|| state.round_poly_subfield::<S>(eq_suffix))
            .unwrap_or_else(|| state.round_poly(eq_suffix))
    }

    fn fold0<'air, 'data>(
        state: RoundStateBase<'air, 'data, A, F, EF>,
        r: EF,
    ) -> RoundStateExt<'air, 'data, A, F, EF, R> {
        debug_assert!(
            R::from(EF::from(F::GENERATOR)) == R::from(F::GENERATOR)
                && EF::from(R::from(EF::GENERATOR)) == EF::GENERATOR,
            "the representation field must embed the trace field through the challenge field"
        );
        if state.is_sliced() {
            state.fold_sliced::<R>(r)
        } else if state.fits_subfield() {
            state.fold_subfield_into::<S, R>(r)
        } else {
            state.fold_into::<R>(r)
        }
    }

    fn round(state: &mut RoundStateExt<'_, '_, A, F, EF, R>, eq_suffix: &Poly<EF>) -> Vec<EF> {
        state
            .round_poly_sliced::<S>(eq_suffix)
            .or_else(|| state.round_poly_boundary::<S>(eq_suffix))
            .unwrap_or_else(|| {
                state.unslice::<S>();
                state.round_poly_repr(eq_suffix)
            })
    }

    fn fold(state: &mut RoundStateExt<'_, '_, A, F, EF, R>, r: EF) {
        if !state.fold_boundary::<S>(r) && !state.fold_sliced(r) {
            state.fold_repr(r);
        }
    }

    fn openings(state: RoundStateExt<'_, '_, A, F, EF, R>) -> Vec<(usize, AirOpenings<EF>)> {
        state.into_openings()
    }
}
