//! Fiat-Shamir transcripts of the reduction, shaped by the statement alone.
//!
//! The folds open the reduction; the grand product and the inner-product rounds run under their own separators; the bit evaluation closes it.

use alloc::vec::Vec;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, VerifierState,
};

use crate::{Challenge, Challenger, Val};

/// Protocol version bound into both seeds.
const VERSION: u8 = 1;

/// Name of the phase sending the folds.
const REDUCTION: &[u8] = b"p3-bitz-reduction";

/// Name of the phase closing the inner product.
const INNER: &[u8] = b"p3-bitz-inner";

/// Folds label.
const FOLDS: &str = "folds";

/// Bit evaluation label.
const EVAL: &str = "inner_eval";

/// Base-field sponge alphabet.
type Alphabet = FieldUnit<Val>;

/// Everything that fixes the reduction's transcript.
#[derive(Clone, Debug)]
pub(crate) struct ReductionShape {
    /// `r1`.
    pub(crate) log_rows: usize,
    /// `r2`.
    pub(crate) log_columns: usize,
    /// Claims reduced together.
    pub(crate) claims: usize,
    /// Grinding before every inner-product challenge.
    pub(crate) inner_pow_bits: usize,
    /// The order of the target field.
    pub(crate) order: u64,
    /// The claim binding tag.
    pub(crate) binding: u8,
    /// The claims' digest, when they are public.
    pub(crate) digest: Option<[u8; 32]>,
    /// Variables of the whole commitment.
    pub(crate) num_variables: usize,
    /// First bit of the region.
    pub(crate) offset_bits: usize,
    /// Words of the fold message.
    pub(crate) fold_words: usize,
}

impl ReductionShape {
    /// The phase sending the folds, with every statement component bound.
    fn reduction_separator(&self) -> DomainSeparator<Alphabet> {
        let pattern = InteractionPattern::new(alloc::vec![Interaction::algebra::<Val, Val>(
            Hierarchy::Atomic,
            Kind::Message,
            FOLDS,
            Length::Fixed(self.fold_words),
        )])
        .expect("a flat pattern is always well formed");
        let mut separator = DomainSeparator::new(VERSION, REDUCTION, pattern);
        for value in [
            self.log_rows,
            self.log_columns,
            self.claims,
            self.inner_pow_bits,
            self.num_variables,
            self.offset_bits,
        ] {
            separator.instance(&(value as u64).to_le_bytes());
        }
        separator.instance(&self.order.to_le_bytes());
        separator.instance(&[self.binding]);
        if let Some(digest) = &self.digest {
            separator.instance(digest);
        }
        separator
    }

    /// The phase closing the inner product.
    fn inner_separator(&self) -> DomainSeparator<Alphabet> {
        let pattern = InteractionPattern::new(alloc::vec![Interaction::algebra::<Val, Challenge>(
            Hierarchy::Atomic,
            Kind::Message,
            EVAL,
            Length::Scalar,
        )])
        .expect("a flat pattern is always well formed");
        let mut separator = DomainSeparator::new(VERSION, INNER, pattern);
        separator.instance(&(self.log_rows as u64).to_le_bytes());
        separator
    }

    /// Bind the statement and the folds, on either side.
    pub(crate) fn observe_folds(&self, folds: &[u64], challenger: &mut Challenger, prover: bool) {
        let folds: Vec<Val> = folds.iter().map(|&w| Val::new(w)).collect();
        let separator = self.reduction_separator();
        if prover {
            let mut state = ProverState::new(&mut *challenger, &separator);
            let _ = state.observe_extensions::<Val, Val, FieldToFieldCodec<Val>>(FOLDS, &folds);
            assert!(
                state.finalize().is_empty(),
                "the reduction carries every value in its proof"
            );
        } else {
            let mut state = VerifierState::new(&mut *challenger, &separator, &[]);
            let _ = state.observe_extensions::<Val, Val, FieldToFieldCodec<Val>>(FOLDS, &folds);
            state.finalize().expect("the reduction reads an empty wire");
        }
    }

    /// Bind the bit evaluation closing the inner product, on either side.
    pub(crate) fn observe_eval(&self, eval: Challenge, challenger: &mut Challenger, prover: bool) {
        let separator = self.inner_separator();
        if prover {
            let mut state = ProverState::new(&mut *challenger, &separator);
            let _ = state.observe_extension::<Val, Challenge, FieldToFieldCodec<Val>>(EVAL, &eval);
            assert!(
                state.finalize().is_empty(),
                "the reduction carries every value in its proof"
            );
        } else {
            let mut state = VerifierState::new(&mut *challenger, &separator, &[]);
            let _ = state.observe_extension::<Val, Challenge, FieldToFieldCodec<Val>>(EVAL, &eval);
            state.finalize().expect("the reduction reads an empty wire");
        }
    }
}
