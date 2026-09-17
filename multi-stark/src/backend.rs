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

use p3_field::{ExtensionField, Field};
use p3_multilinear_util::poly::Poly;

use crate::folder::ProverAir;
use crate::rounds::{AirOpenings, RoundStateBase, RoundStateExt};

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
        /// Evaluate the first round polynomial of a newly activated stage.
        fn round0(state: &mut RoundStateBase<'_, '_, A, F, EF>, eq_suffix: &Poly<EF>) -> Vec<EF>;

        /// Bind the first variable of a stage at `r`.
        fn fold0<'air, 'data>(
            state: RoundStateBase<'air, 'data, A, F, EF>,
            r: EF,
        ) -> RoundStateExt<'air, 'data, A, F, EF>;

        /// Evaluate the round polynomial of a stage whose first variable is bound.
        fn round(state: &mut RoundStateExt<'_, '_, A, F, EF>, eq_suffix: &Poly<EF>) -> Vec<EF>;

        /// Bind the next variable of a folded stage at `r`.
        fn fold(state: &mut RoundStateExt<'_, '_, A, F, EF>, r: EF);

        /// Split a fully bound stage into per-AIR openings, each tagged with its caller index.
        fn openings(state: RoundStateExt<'_, '_, A, F, EF>) -> Vec<(usize, AirOpenings<EF>)>;
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
