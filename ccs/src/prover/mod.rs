//! Prover of the two-phase reduction from CCS satisfiability to one witness evaluation.

mod linear;
mod zerocheck;

use alloc::vec::Vec;

use p3_challenger::fs::TranscriptField;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, dot_product};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::generic_degree::RoundProver;
use tracing::instrument;

use self::linear::LinearProver;
use self::zerocheck::ZerocheckProver;
use crate::operator::LinearOperator;
use crate::proof::{CcsProof, WitnessClaim};
use crate::relation::Ccs;
use crate::transcript::{ProverTranscript, ReductionShape};

/// Reduce satisfiability of an already committed witness to one witness evaluation claim.
///
/// SuperSpartan polynomial IOP (Figure 2 of ePrint 2023/552), over the homogeneous relation alone.
///
/// # Panics
///
/// - The witness length is not the column count.
pub fn prove<F, EF, O, C>(
    ccs: &Ccs<F, O>,
    z: &[F],
    pow_bits: usize,
    challenger: &mut C,
) -> (CcsProof<F, EF>, WitnessClaim<EF>)
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    O: LinearOperator<F>,
    C: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let (proof, claim, _) = reduce(ccs, z, ReductionShape::new(ccs, pow_bits), challenger);
    (proof, claim)
}

/// Run the reduction under a shape, returning the proof, the claim and the public points.
#[instrument(skip_all, name = "ccs prove", fields(log_rows = ccs.log_rows(), log_cols = ccs.log_cols()))]
pub(crate) fn reduce<F, EF, O, C>(
    ccs: &Ccs<F, O>,
    z: &[F],
    shape: ReductionShape,
    challenger: &mut C,
) -> (CcsProof<F, EF>, WitnessClaim<EF>, Vec<Vec<EF>>)
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    O: LinearOperator<F>,
    C: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    assert_eq!(z.len(), 1 << ccs.log_cols(), "witness length");

    let (log_rows, log_cols) = (ccs.log_rows(), ccs.log_cols());
    let pow_bits = shape.pow_bits();
    let mut transcript = ProverTranscript::<C, F, EF>::new(challenger, shape);

    // Phase 1: zerocheck, 0 = sum_x eq(tau, x) * sum_k alpha_k * g_k(x).
    let alphas = transcript.alphas();
    let tau = transcript.tau();
    let mut zerocheck = ZerocheckProver::new(
        ccs.constraints(),
        alphas,
        ccs.degree(),
        tau,
        ccs.operand_tables(z),
    );
    let (zerocheck_proof, row_point) = transcript.zerocheck(|challenger| {
        zerocheck.prove::<F, C>(challenger, log_rows, ccs.degree() + 1, pow_bits, EF::ZERO)
    });

    // Tables are now the matrix outputs at the row point.
    let operand_evals = zerocheck.operand_evals();
    transcript.operand_evals(&operand_evals);

    // Phase 2: sum_j beta_j * v_j = < sum_j beta_j * M_j^T eq(r, .), z >.
    let batching = transcript.batching();
    let row_eq = Poly::new_from_point(row_point.as_slice(), EF::ONE);
    // Zero in parallel so page faults spread across threads.
    let mut weights: Vec<EF> = (0..1usize << log_cols)
        .into_par_iter()
        .map(|_| EF::ZERO)
        .collect();
    for (op, &beta) in ccs.operators().iter().zip(&batching) {
        op.add_transpose(row_eq.as_slice(), beta, &mut weights);
    }
    let linear_claim = dot_product(batching.iter().copied(), operand_evals.iter().copied());

    // One product sumcheck closes every matrix claim.
    let mut linear = LinearProver::new(weights, z);
    let (linear_proof, col_point) = transcript
        .linear(|challenger| linear.prove::<F, C>(challenger, log_cols, 2, pow_bits, linear_claim));

    // Witness is now its value at the final point.
    let witness_eval = linear.witness_eval();
    let public_points = transcript.finish(witness_eval);

    let proof = CcsProof {
        zerocheck: zerocheck_proof,
        operand_evals,
        linear: linear_proof,
        witness_eval,
    };
    (
        proof,
        WitnessClaim {
            point: col_point,
            value: witness_eval,
        },
        public_points,
    )
}
