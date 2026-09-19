//! Typed Fiat-Shamir transcript for the bus multiset argument.

use alloc::vec;

use p3_challenger::FieldChallenger;
use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptField, VerifierState,
};
use p3_field::ExtensionField;

use super::BusChallenges;
use crate::{BusDirection, BusPlan};

/// Version byte bound into the protocol seed.
const VERSION: u8 = 1;

/// Protocol name bound into the protocol seed.
const NAME: &[u8] = b"p3-bus-argument";

/// Label of the tuple-fingerprint challenge point.
const FINGERPRINT: &str = "fingerprint";

/// Label of the random shift applied to every tuple fingerprint.
const OFFSET: &str = "offset";

/// Label of the delegated product-tree reduction.
const PRODUCT: &str = "product_gkr";

/// Sponge alphabet of a challenger native to the base field.
type Alphabet<F> = FieldUnit<F>;

/// Type-level marker for the delegated product-tree reduction.
struct ProductReduction;

impl BusPlan {
    /// Describe the statement-derived challenge and delegation schedule.
    fn interaction_pattern<F, EF>(&self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        // The public layout fixes both challenge length and the delegated reduction shape.
        InteractionPattern::new(vec![
            Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                FINGERPRINT,
                Length::Fixed(self.security_geometry().tuple_variables()),
            ),
            Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                OFFSET,
                Length::Scalar,
            ),
            Interaction::marker::<ProductReduction>(Hierarchy::Begin, Kind::Protocol, PRODUCT),
            Interaction::marker::<ProductReduction>(Hierarchy::End, Kind::Protocol, PRODUCT),
        ])
        .expect("one matched product-reduction bracket is well formed")
    }

    /// Bind every public dimension that changes a tuple or tree position.
    fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        // The seed commits to the full verifier-derived statement before any challenge.
        let mut separator =
            DomainSeparator::new(VERSION, NAME, self.interaction_pattern::<F, EF>());
        separator
            .instance(&(self.domains().len() as u64).to_be_bytes())
            .instance(&(self.payload_slots() as u64).to_be_bytes())
            .instance(&(self.domain_slots() as u64).to_be_bytes())
            .instance(&(self.fingerprint_width() as u64).to_be_bytes());

        // Domain names and identities prevent two named buses from sharing one tuple space.
        for domain in self.domains() {
            separator
                .instance(&(domain.name.len() as u64).to_be_bytes())
                .instance(domain.name.as_bytes())
                .instance(&(domain.payload_width as u64).to_be_bytes())
                .instance(&(domain.identity as u64).to_be_bytes());
        }

        // Physical block order determines the product-tree leaf address of every declaration.
        for direction in BusDirection::ALL {
            let blocks = self.blocks(direction);
            separator.instance(&(blocks.len() as u64).to_be_bytes());
            for block in blocks {
                separator
                    .instance(&(block.bus as u64).to_be_bytes())
                    .instance(&(block.owner.air as u64).to_be_bytes())
                    .instance(&(block.owner.declaration as u64).to_be_bytes())
                    .instance(&(block.log_height as u64).to_be_bytes())
                    .instance(&(block.offset as u64).to_be_bytes());
            }
        }
        separator
    }
}

/// Prover-side driver for the bus argument transcript.
pub(super) struct BusProverTranscript<'a, C, F: TranscriptField, EF> {
    /// Pattern player borrowing the surrounding challenger.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// Number of coordinates in the tuple-fingerprint point.
    fingerprint_variables: usize,
    /// Challenge-field marker used by the typed codec.
    _ef: core::marker::PhantomData<EF>,
}

impl<'a, C, F, EF> BusProverTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    /// Seed the transcript from the complete public bus layout.
    pub(super) fn new(challenger: &'a mut C, plan: &BusPlan) -> Self {
        // Every later draw is scoped to the exact tuple and tree geometry.
        Self {
            state: ProverState::new(challenger, &plan.domain_separator::<F, EF>()),
            fingerprint_variables: plan.security_geometry().tuple_variables(),
            _ef: core::marker::PhantomData,
        }
    }

    /// Sample the challenges defining every bus leaf factor.
    pub(super) fn challenges(&mut self) -> BusChallenges<EF> {
        // The tuple point precedes the independent product shift.
        let fingerprint = self
            .state
            .challenge_extensions::<F, EF, FieldToFieldCodec<F>>(
                FINGERPRINT,
                self.fingerprint_variables,
            )
            .into_iter()
            .map(|value| value.into_inner())
            .collect();
        let offset = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(OFFSET)
            .into_inner();
        BusChallenges {
            fingerprint,
            offset,
        }
    }

    /// Execute the delegated product proof inside its transcript bracket.
    pub(super) fn product<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        // The nested proof owns every product-reduction message and challenge.
        self.state.begin_protocol::<ProductReduction>(PRODUCT);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<ProductReduction>(PRODUCT);
        output
    }

    /// Finish after every shape-derived transcript step has been played.
    pub(super) fn finish(self) {
        // The proof object carries every message, so the typed wire is empty.
        assert!(self.state.finalize().is_empty());
    }

    /// Disable completeness checking after a checked honest-prover input error.
    pub(super) fn abort(&mut self) {
        // No transcript output is consumed after an honest-prover shape failure.
        self.state.abort();
    }
}

/// Verifier-side replay of the bus argument transcript.
pub(super) struct BusVerifierTranscript<'a, C, F: TranscriptField, EF> {
    /// Pattern player borrowing the surrounding challenger over an empty wire.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// Number of coordinates in the tuple-fingerprint point.
    fingerprint_variables: usize,
    /// Challenge-field marker used by the typed codec.
    _ef: core::marker::PhantomData<EF>,
}

impl<'a, C, F, EF> BusVerifierTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    /// Seed the replay from the complete public bus layout.
    pub(super) fn new(challenger: &'a mut C, plan: &BusPlan) -> Self {
        // This protocol reads no wire values outside the delegated proof object.
        Self {
            state: VerifierState::new(challenger, &plan.domain_separator::<F, EF>(), &[]),
            fingerprint_variables: plan.security_geometry().tuple_variables(),
            _ef: core::marker::PhantomData,
        }
    }

    /// Replay the challenges defining every bus leaf factor.
    pub(super) fn challenges(&mut self) -> BusChallenges<EF> {
        // The verifier redraws the exact statement-derived challenge count.
        let fingerprint = self
            .state
            .challenge_extensions::<F, EF, FieldToFieldCodec<F>>(
                FINGERPRINT,
                self.fingerprint_variables,
            )
            .into_iter()
            .map(|value| value.into_inner())
            .collect();
        let offset = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(OFFSET)
            .into_inner();
        BusChallenges {
            fingerprint,
            offset,
        }
    }

    /// Execute delegated product verification inside its transcript bracket.
    pub(super) fn product<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        // Nested verification replays every product message before leaving the bracket.
        self.state.begin_protocol::<ProductReduction>(PRODUCT);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<ProductReduction>(PRODUCT);
        output
    }

    /// Finish after every shape-derived transcript step has been replayed.
    pub(super) fn finish(self) {
        // Complete replay consumes the empty typed wire exactly.
        self.state
            .finalize()
            .expect("the bus argument reads an empty wire");
    }

    /// Disable outer completeness checking after delegated verification rejects.
    pub(super) fn abort(&mut self) {
        // The caller returns the nested verification error immediately.
        self.state.abort();
    }
}
