//! Typed Fiat-Shamir transcript for batched product-tree GKR.
//!
//! The verifier supplies the tree height, tree count, and root encoding.
//! Those values determine every message length and enter the transcript seed.
//!
//! Roots precede the first batching challenge.
//! Every round polynomial precedes its evaluation challenge.
//! Every child claim precedes the coordinates that collapse it.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptField, VerifierState,
};
use p3_challenger::{CanObserve, CanSample};
use p3_field::ExtensionField;

use crate::product::{ProductGkrShape, ROUND_POLY_LEN};

/// Version byte bound into the protocol seed.
const VERSION: u8 = 1;

/// Protocol name bound into the protocol seed.
const NAME: &[u8] = b"p3-bus-product-gkr";

/// Label of the encoded root claims.
const ROOTS: &str = "roots";

/// Label of one reduction-layer container.
const LAYER: &str = "layer";

/// Label of the challenge batching product trees.
const BATCHING: &str = "batching";

/// Label of one sumcheck message.
const ROUND_POLY: &str = "round_poly";

/// Label of one sumcheck challenge.
const ROUND_CHALLENGE: &str = "round_challenge";

/// Label of the child evaluations closing a layer.
const CHILDREN: &str = "children";

/// Label of the coordinates collapsing child evaluations.
const BRANCHES: &str = "branches";

/// Sponge alphabet of a challenger native to the base field.
type Alphabet<F> = FieldUnit<F>;

/// Type-level marker for one product-tree layer.
struct ReductionLayer;

impl ProductGkrShape {
    /// Describe every message and challenge fixed by this statement shape.
    pub(crate) fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        // The statement fixes all lengths before either side sees a proof.
        let mut steps = Vec::new();
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Message,
            ROOTS,
            Length::Fixed(self.root_message_len()),
        ));

        for (arity, rounds) in self.layers() {
            steps.push(Interaction::marker::<ReductionLayer>(
                Hierarchy::Begin,
                Kind::Protocol,
                LAYER,
            ));
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                BATCHING,
                Length::Scalar,
            ));
            for _ in 0..rounds {
                steps.push(Interaction::algebra::<F, EF>(
                    Hierarchy::Atomic,
                    Kind::Message,
                    ROUND_POLY,
                    Length::Fixed(ROUND_POLY_LEN),
                ));
                steps.push(Interaction::algebra::<F, EF>(
                    Hierarchy::Atomic,
                    Kind::Challenge,
                    ROUND_CHALLENGE,
                    Length::Scalar,
                ));
            }
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                CHILDREN,
                Length::Fixed(arity * self.num_trees()),
            ));
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                BRANCHES,
                Length::Fixed(arity.trailing_zeros() as usize),
            ));
            steps.push(Interaction::marker::<ReductionLayer>(
                Hierarchy::End,
                Kind::Protocol,
                LAYER,
            ));
        }

        InteractionPattern::new(steps).expect("one matched bracket per layer is well formed")
    }

    /// Bind the protocol identity and the complete verifier-derived shape.
    pub(crate) fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        DomainSeparator::new(VERSION, NAME, self.pattern::<F, EF>())
    }
}

/// Prover-side driver for the product reduction transcript.
pub(crate) struct ProductGkrProverTranscript<'a, C, F: TranscriptField, EF> {
    /// Pattern player and borrowed challenger.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// Extension-field marker for the typed codec.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> ProductGkrProverTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F>,
{
    /// Seed the transcript and bind the encoded roots before any challenge.
    pub(crate) fn new(challenger: &'a mut C, shape: ProductGkrShape, roots: &[EF]) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        let mut state = ProverState::new(challenger, &separator);
        state.observe_extensions::<F, EF, FieldToFieldCodec<F>>(ROOTS, roots);
        Self {
            state,
            _ef: PhantomData,
        }
    }

    /// Begin one layer and sample its tree-batching challenge.
    pub(crate) fn begin_layer(&mut self) -> EF {
        self.state.begin_protocol::<ReductionLayer>(LAYER);
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(BATCHING)
            .into_inner()
    }

    /// Bind one degree-five sumcheck message before sampling its coordinate.
    pub(crate) fn round(&mut self, round_poly: &[EF; ROUND_POLY_LEN]) -> EF {
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(ROUND_POLY, round_poly);
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ROUND_CHALLENGE)
            .into_inner()
    }

    /// Bind binary child claims before sampling their collapse coordinate.
    pub(crate) fn end_binary_layer(&mut self, children: &[[EF; 2]]) -> [EF; 1] {
        let flattened = children.iter().flatten().copied().collect::<Vec<_>>();
        self.end_layer(&flattened, 1)
            .try_into()
            .expect("one branch coordinate")
    }

    /// Bind radix-four child claims before sampling their collapse coordinates.
    pub(crate) fn end_radix_four_layer(&mut self, children: &[[EF; 4]]) -> [EF; 2] {
        let flattened = children.iter().flatten().copied().collect::<Vec<_>>();
        self.end_layer(&flattened, 2)
            .try_into()
            .expect("two branch coordinates")
    }

    /// Close a layer after its child claims have been fixed.
    fn end_layer(&mut self, children: &[EF], branch_count: usize) -> Vec<EF> {
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(CHILDREN, children);
        let branches = self
            .state
            .challenge_extensions::<F, EF, FieldToFieldCodec<F>>(BRANCHES, branch_count)
            .into_iter()
            .map(|value| value.into_inner())
            .collect();
        self.state.end_protocol::<ReductionLayer>(LAYER);
        branches
    }

    /// Finish after every shape-derived transcript step has been played.
    pub(crate) fn finish(self) {
        assert!(
            self.state.finalize().is_empty(),
            "the product reduction carries every value in its proof",
        );
    }
}

/// Verifier-side replay of the product reduction transcript.
pub(crate) struct ProductGkrVerifierTranscript<'a, C, F: TranscriptField, EF> {
    /// Pattern player and borrowed challenger over an empty wire.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// Extension-field marker for the typed codec.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> ProductGkrVerifierTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F>,
{
    /// Seed the replay and bind the checked root message.
    pub(crate) fn new(challenger: &'a mut C, shape: ProductGkrShape, roots: &[EF]) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        let mut state = VerifierState::new(challenger, &separator, &[]);
        state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(ROOTS, roots)
            .expect("checked root width matches the transcript shape");
        Self {
            state,
            _ef: PhantomData,
        }
    }

    /// Begin one layer and redraw its tree-batching challenge.
    pub(crate) fn begin_layer(&mut self) -> EF {
        self.state.begin_protocol::<ReductionLayer>(LAYER);
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(BATCHING)
            .into_inner()
    }

    /// Replay one checked-width sumcheck message and redraw its coordinate.
    pub(crate) fn round(&mut self, round_poly: &[EF; ROUND_POLY_LEN]) -> EF {
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(ROUND_POLY, round_poly)
            .expect("a fixed-size round message matches the transcript shape");
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ROUND_CHALLENGE)
            .into_inner()
    }

    /// Replay binary child claims and redraw their collapse coordinate.
    pub(crate) fn end_binary_layer(&mut self, children: &[[EF; 2]]) -> [EF; 1] {
        let flattened = children.iter().flatten().copied().collect::<Vec<_>>();
        self.end_layer(&flattened, 1)
            .try_into()
            .expect("one branch coordinate")
    }

    /// Replay radix-four child claims and redraw their collapse coordinates.
    pub(crate) fn end_radix_four_layer(&mut self, children: &[[EF; 4]]) -> [EF; 2] {
        let flattened = children.iter().flatten().copied().collect::<Vec<_>>();
        self.end_layer(&flattened, 2)
            .try_into()
            .expect("two branch coordinates")
    }

    /// Close a replayed layer after binding its checked child claims.
    fn end_layer(&mut self, children: &[EF], branch_count: usize) -> Vec<EF> {
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(CHILDREN, children)
            .expect("checked child width matches the transcript shape");
        let branches = self
            .state
            .challenge_extensions::<F, EF, FieldToFieldCodec<F>>(BRANCHES, branch_count)
            .into_iter()
            .map(|value| value.into_inner())
            .collect();
        self.state.end_protocol::<ReductionLayer>(LAYER);
        branches
    }

    /// Finish after every shape-derived transcript step has been replayed.
    pub(crate) fn finish(self) {
        self.state
            .finalize()
            .expect("the product reduction reads an empty wire");
    }
}
