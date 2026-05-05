use core::fmt::Debug;
use core::ops::Deref;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_multilinear_util::point::Point;

use crate::constraints::statement::EqStatement;
use crate::parameters::WhirConfig;
use crate::pcs::proof::WhirProof;

/// Parsed commitment extracted from the verifier's transcript.
///
/// Contains the Merkle root and the OOD equality constraints
/// needed for verification.
#[derive(Debug, Clone)]
pub struct ParsedCommitment<F, D> {
    /// Merkle root of the committed evaluation table.
    pub root: D,
    /// OOD challenge points and their claimed evaluations.
    pub ood_statement: EqStatement<F>,
}

impl<F, D> ParsedCommitment<F, D>
where
    F: Field,
{
    /// Parse a commitment for a specific round (or initial if `None`).
    pub fn parse_with_round<EF, MT: Mmcs<F>, Challenger>(
        proof: &WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
        num_variables: usize,
        ood_samples: usize,
        round_index: usize,
    ) -> ParsedCommitment<EF, MT::Commitment>
    where
        F: TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField,
        Challenger:
            FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment>,
    {
        // Extract root and OOD answers from either the initial commitment or a round.
        let round_proof = &proof.rounds[round_index];
        let root = round_proof.commitment.clone().unwrap();
        let ood_answers = round_proof.ood_answers.clone();

        // Observe the Merkle root in the transcript.
        challenger.observe(root.clone());

        // Reconstruct equality constraints from OOD challenge points and answers.
        let mut ood_statement = EqStatement::initialize(num_variables);
        (0..ood_samples).for_each(|i| {
            let point = challenger.sample_algebra_element();
            let point = Point::expand_from_univariate(point, num_variables);
            let eval = ood_answers[i];
            challenger.observe_algebra_element(eval);
            ood_statement.add_evaluated_constraint(point, eval);
        });

        ParsedCommitment {
            root,
            ood_statement,
        }
    }
}

/// Lightweight wrapper for parsing commitment data during verification.
#[derive(Debug)]
pub struct CommitmentReader<'a, EF, F, Challenger>(&'a WhirConfig<EF, F, Challenger>)
where
    F: Field,
    EF: ExtensionField<F>;

impl<'a, EF, F, Challenger> CommitmentReader<'a, EF, F, Challenger>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    pub const fn new(params: &'a WhirConfig<EF, F, Challenger>) -> Self {
        Self(params)
    }
}

impl<EF, F, Challenger> Deref for CommitmentReader<'_, EF, F, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type Target = WhirConfig<EF, F, Challenger>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}
