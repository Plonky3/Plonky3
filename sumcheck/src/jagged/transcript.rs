//! Fiat-Shamir transcript for the basic jagged reduction.

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptField, VerifierState,
};
use p3_challenger::{CanObserve, CanSample};
use p3_field::ExtensionField;

use super::{JaggedLayout, JaggedPoint};

/// Version bound into every jagged transcript seed.
const VERSION: u8 = 1;

/// Protocol name separating this reduction from a bare quadratic sumcheck.
const NAME: &[u8] = b"p3-sumcheck-jagged";

/// Marker around the delegated product sumcheck.
const PRODUCT_SUMCHECK: &str = "product_sumcheck";

/// Value of the surviving dense evaluation claim.
const DENSE_EVALUATION: &str = "dense_evaluation";

/// Sponge alphabet of a challenger that speaks the base field natively.
type Alphabet<F> = FieldUnit<F>;

/// Type-level name of the delegated quadratic sumcheck.
struct ProductSumcheck;

/// Describes the outer protocol steps.
fn pattern<F, EF>() -> InteractionPattern
where
    F: TranscriptField,
    EF: ExtensionField<F>,
{
    // The inner rounds own their transcript description.
    // The outer description records their position and binds the surviving value.
    let steps = vec![
        Interaction::marker::<ProductSumcheck>(Hierarchy::Begin, Kind::Protocol, PRODUCT_SUMCHECK),
        Interaction::marker::<ProductSumcheck>(Hierarchy::End, Kind::Protocol, PRODUCT_SUMCHECK),
        Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Message,
            DENSE_EVALUATION,
            Length::Scalar,
        ),
    ];

    InteractionPattern::new(steps).expect("one matched bracket is structurally valid")
}

/// Binds the public sparse statement into the protocol seed.
fn domain_separator<F, EF>(
    layout: &JaggedLayout,
    point: &JaggedPoint<EF>,
    claimed_value: EF,
) -> DomainSeparator<Alphabet<F>>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
{
    // The step description distinguishes field identities and protocol shape.
    let mut separator = DomainSeparator::new(VERSION, NAME, pattern::<F, EF>());

    // The instance binds geometry, every cumulative height, the point and its value.
    // A prover cannot choose another sparse statement after seeing a round challenge.
    let statement = encode_statement::<F, EF>(layout, point, claimed_value);
    separator.instance(&statement);
    separator
}

/// Prover-side driver for one jagged reduction.
pub(super) struct JaggedProverTranscript<'a, C, F: TranscriptField, EF> {
    /// Typed transcript state borrowing the surrounding challenger.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// Marker for the extension field carried by the final claim.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> JaggedProverTranscript<'a, C, F, EF>
where
    C: CanObserve<F> + CanSample<F>,
    F: TranscriptField,
    EF: ExtensionField<F>,
{
    /// Seeds a transcript from the full public sparse statement.
    pub(super) fn new(
        challenger: &'a mut C,
        layout: &JaggedLayout,
        point: &JaggedPoint<EF>,
        claimed_value: EF,
    ) -> Self {
        // Both sides derive the same statement bytes locally.
        let separator = domain_separator::<F, EF>(layout, point, claimed_value);
        Self {
            state: ProverState::new(challenger, &separator),
            _ef: PhantomData,
        }
    }

    /// Lends the sponge to the quadratic sumcheck.
    pub(super) fn product_sumcheck<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        // Bracketing fixes the sub-protocol's position in the outer transcript.
        self.state
            .begin_protocol::<ProductSumcheck>(PRODUCT_SUMCHECK);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<ProductSumcheck>(PRODUCT_SUMCHECK);
        output
    }

    /// Binds the dense evaluation that the underlying PCS must authenticate.
    pub(super) fn dense_evaluation(&mut self, value: EF) {
        // The claim is observed before control returns to the surrounding protocol.
        self.state
            .observe_extension::<F, EF, FieldToFieldCodec<F>>(DENSE_EVALUATION, &value);
    }

    /// Closes the transcript after every described step has been played.
    pub(super) fn finish(self) {
        // Every value lives in the proof's typed fields rather than an auxiliary byte wire.
        // No step writes to it, so a nonempty wire would mean the description above had drifted.
        assert!(
            self.state.finalize().is_empty(),
            "the jagged reduction carries every value in its proof"
        );
    }
}

/// Verifier-side driver mirroring the prover call for call.
pub(super) struct JaggedVerifierTranscript<'a, C, F: TranscriptField, EF> {
    /// Typed replay state borrowing the surrounding challenger.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// Marker for the extension field carried by the final claim.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> JaggedVerifierTranscript<'a, C, F, EF>
where
    C: CanObserve<F> + CanSample<F>,
    F: TranscriptField,
    EF: ExtensionField<F>,
{
    /// Seeds a replay from the verifier's public sparse statement.
    pub(super) fn new(
        challenger: &'a mut C,
        layout: &JaggedLayout,
        point: &JaggedPoint<EF>,
        claimed_value: EF,
    ) -> Self {
        // No statement component is accepted from the proof.
        let separator = domain_separator::<F, EF>(layout, point, claimed_value);
        Self {
            state: VerifierState::new(challenger, &separator, &[]),
            _ef: PhantomData,
        }
    }

    /// Lends the sponge to the quadratic sumcheck replay.
    pub(super) fn product_sumcheck<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        // The bracket closes even when the nested verifier returns an error.
        self.state
            .begin_protocol::<ProductSumcheck>(PRODUCT_SUMCHECK);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<ProductSumcheck>(PRODUCT_SUMCHECK);
        output
    }

    /// Binds the dense evaluation supplied by the proof.
    pub(super) fn dense_evaluation(&mut self, value: EF) {
        // The verifier absorbs exactly the value it later returns for PCS authentication.
        self.state
            .observe_extension::<F, EF, FieldToFieldCodec<F>>(DENSE_EVALUATION, &value);
    }

    /// Releases the structural completeness check after an inner rejection.
    pub(super) fn abort(&mut self) {
        // The original verification error remains the only reported failure.
        self.state.abort();
    }

    /// Closes the transcript after every described step has been replayed.
    pub(super) fn finish(self) {
        // The replay reads values from the typed proof rather than a byte wire.
        // No step can advance the cursor over that empty wire, so this failure is unreachable.
        self.state
            .finalize()
            .expect("the jagged reduction reads an empty wire");
    }
}

/// Encodes the sparse statement injectively for domain separation.
fn encode_statement<F, EF>(
    layout: &JaggedLayout,
    point: &JaggedPoint<EF>,
    claimed_value: EF,
) -> Vec<u8>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
{
    // Fixed-width integer encoding makes proofs portable across 32-bit and 64-bit hosts.
    let mut bytes = Vec::new();
    encode_usize(&mut bytes, layout.row_variables());
    encode_usize(&mut bytes, layout.dense_variables());
    encode_usize(&mut bytes, layout.cumulative_heights().len());
    for &height in layout.cumulative_heights() {
        encode_usize(&mut bytes, height);
    }

    // Field elements use the transcript field's canonical coefficient encoding.
    for value in point
        .row()
        .iter()
        .chain(point.column().iter())
        .chain(core::iter::once(&claimed_value))
    {
        for coefficient in value.as_basis_coefficients_slice() {
            F::encode(coefficient, &mut bytes);
        }
    }

    bytes
}

/// Appends a public index as an unsigned 64-bit big-endian integer.
fn encode_usize(bytes: &mut Vec<u8>, value: usize) {
    // Every supported Rust target has a pointer width of at most 64 bits.
    bytes.extend_from_slice(&(value as u64).to_be_bytes());
}
