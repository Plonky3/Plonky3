use alloc::vec::Vec;
use core::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Dup, Field, PrimeCharacteristicRing};
use p3_mds::MdsPermutation;

use crate::columns::{RescueCols, num_cols};

/// An AIR for the Rescue permutation, one permutation per row.
///
/// `ALPHA` is the S-box exponent and must match the `p3_rescue::Rescue`
/// instance whose traces this AIR is meant to accept.
#[derive(Debug, Clone)]
pub struct RescueAir<F, const WIDTH: usize, const ALPHA: u64, const NUM_ROUNDS: usize> {
    /// The MDS layer as an explicit matrix: `out[i] = sum_j mds[i][j] * in[j]`.
    ///
    /// Held explicitly because the MDS implementations are concrete over their
    /// field and so cannot be applied to an [`AirBuilder`]'s expressions.
    pub mds: [[F; WIDTH]; WIDTH],

    /// `2 * WIDTH * NUM_ROUNDS` constants, laid out exactly as
    /// `p3_rescue::Rescue` consumes them: the first `WIDTH` of each round's
    /// pair follow the forward S-box, the second `WIDTH` the inverse S-box.
    pub round_constants: Vec<F>,
}

impl<F: Field, const WIDTH: usize, const ALPHA: u64, const NUM_ROUNDS: usize>
    RescueAir<F, WIDTH, ALPHA, NUM_ROUNDS>
{
    /// # Panics
    /// If `round_constants` is not `2 * WIDTH * NUM_ROUNDS` long.
    pub fn new(mds: [[F; WIDTH]; WIDTH], round_constants: Vec<F>) -> Self {
        assert_eq!(
            round_constants.len(),
            2 * WIDTH * NUM_ROUNDS,
            "round_constants must have length 2 * WIDTH * NUM_ROUNDS"
        );
        Self {
            mds,
            round_constants,
        }
    }

    /// Recover the MDS matrix from any [`MdsPermutation`] over `F` by applying
    /// it to the basis vectors, so the AIR and the permutation cannot disagree
    /// about the linear layer.
    ///
    /// `mds.permute(e_j)` is column `j` of the matrix.
    pub fn from_mds<Mds: MdsPermutation<F, WIDTH>>(mds: &Mds, round_constants: Vec<F>) -> Self {
        let mut matrix = [[F::ZERO; WIDTH]; WIDTH];
        for j in 0..WIDTH {
            let mut basis = [F::ZERO; WIDTH];
            basis[j] = F::ONE;
            let column = mds.permute(basis);
            for i in 0..WIDTH {
                matrix[i][j] = column[i];
            }
        }
        Self::new(matrix, round_constants)
    }
}

impl<F: Sync, const WIDTH: usize, const ALPHA: u64, const NUM_ROUNDS: usize> BaseAir<F>
    for RescueAir<F, WIDTH, ALPHA, NUM_ROUNDS>
{
    fn width(&self) -> usize {
        num_cols::<WIDTH, NUM_ROUNDS>()
    }
}

impl<AB: AirBuilder, const WIDTH: usize, const ALPHA: u64, const NUM_ROUNDS: usize> Air<AB>
    for RescueAir<AB::F, WIDTH, ALPHA, NUM_ROUNDS>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &RescueCols<AB::Var, WIDTH, NUM_ROUNDS> = main.current_slice().borrow();

        for r in 0..NUM_ROUNDS {
            let round = &local.rounds[r];

            // The state entering this round: the inputs, or the previous round's output.
            let state: [AB::Expr; WIDTH] = if r == 0 {
                core::array::from_fn(|i| local.inputs[i].into())
            } else {
                core::array::from_fn(|i| local.rounds[r - 1].post[i].into())
            };

            // Forward S-box: sbox_pow = state^ALPHA.
            for (pow, x) in round.sbox_pow.iter().zip(&state) {
                builder.assert_eq((*pow).into(), x.dup().exp_const_u64::<ALPHA>());
            }

            // First MDS layer and its constants.
            for i in 0..WIDTH {
                let mut acc = AB::Expr::ZERO;
                for j in 0..WIDTH {
                    acc += round.sbox_pow[j].into() * self.mds[i][j].clone();
                }
                acc += self.round_constants[r * 2 * WIDTH + i].clone();
                builder.assert_eq(round.mid[i].into(), acc);
            }

            // Inverse S-box, stated forwards: inv_sbox^ALPHA = mid.
            for i in 0..WIDTH {
                builder.assert_eq(
                    round.inv_sbox[i].into().exp_const_u64::<ALPHA>(),
                    round.mid[i].into(),
                );
            }

            // Second MDS layer and its constants.
            for i in 0..WIDTH {
                let mut acc = AB::Expr::ZERO;
                for j in 0..WIDTH {
                    acc += round.inv_sbox[j].into() * self.mds[i][j].clone();
                }
                acc += self.round_constants[r * 2 * WIDTH + WIDTH + i].clone();
                builder.assert_eq(round.post[i].into(), acc);
            }
        }
    }
}
