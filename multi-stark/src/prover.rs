use p3_air::Air;
use p3_challenger::FieldChallenger;
use p3_commit::Pcs;
use p3_matrix::dense::RowMajorMatrix;

use crate::{ConstraintFolder, StarkConfig};

pub fn prove<SC, A, Chal>(
    config: &SC,
    _air: &A,
    _challenger: &mut Chal,
    trace: RowMajorMatrix<SC::Val>,
) where
    SC: StarkConfig,
    A: for<'a> Air<ConstraintFolder<'a, SC::Val, SC::Challenge, SC::PackedChallenge>>,
    Chal: FieldChallenger<SC::Val>,
{
    let (_trace_commit, _trace_data) = config.pcs().commit_batch(trace.as_view());

    // challenger.observe_ext_element(trace_commit); // TODO
}
