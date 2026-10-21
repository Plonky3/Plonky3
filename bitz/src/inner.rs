//! `Pi_ip`: the leaf claim of the grand product reduced to one evaluation of the committed bits (Construction 6.2).
//!
//! The leaf claim is `v - 1 = sum_b W0(b) B(b)` with `B(b) = bits~(rho_c, b)`, a quadratic sumcheck over the row variables.

use alloc::vec::Vec;

use p3_binary_field::fold_rows;
use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::SumcheckData;
use p3_sumcheck::product_polynomial::ProductPolynomial;
use p3_sumcheck::strategy::{SumcheckProver, VariableOrder};

use crate::params::BitShape;
use crate::proof::InnerProof;
use crate::{Challenge, Challenger, Val};

/// `W0(b) = eq(b; rho_b) sum_t eq(t; rho_t) (y^t_b - 1)`, the claims beyond `images` weighing nothing.
pub(crate) fn row_weights(
    images: &[Vec<Challenge>],
    tree_point: &[Challenge],
    row_point: &[Challenge],
) -> Vec<Challenge> {
    let trees = Poly::new_from_point(tree_point, Challenge::ONE);
    let mut weights = Poly::new_from_point(row_point, Challenge::ONE).into_evals();
    weights.par_iter_mut().enumerate().for_each(|(b, weight)| {
        let shift: Challenge = images
            .iter()
            .zip(trees.as_slice())
            .map(|(images, &tree)| tree * (images[b] - Challenge::ONE))
            .sum();
        *weight *= shift;
    });
    weights
}

/// Prove `sum_b W0(b) B(b) = sum` and return the proof, the point `rho'`, and the fold `B`.
pub(crate) fn prove(
    shape: BitShape,
    words: &[u64],
    column_point: &[Challenge],
    weights: Vec<Challenge>,
    sum: Challenge,
    pow_bits: usize,
    challenger: &mut Challenger,
) -> (InnerProof, Point<Challenge>, Vec<Challenge>) {
    let columns = Poly::new_from_point(column_point, Challenge::ONE);
    let folded = fold_rows(words, shape.log_rows(), shape.rows(), columns.as_slice());
    let width = <Val as Field>::Packing::WIDTH;
    let pack = |table: &[Challenge]| {
        Poly::new(
            table
                .chunks_exact(width)
                .map(|chunk| {
                    <<Challenge as ExtensionField<Val>>::ExtensionPacking as PackedFieldExtension<
                        Val,
                        Challenge,
                    >>::from_ext_slice(chunk)
                })
                .collect(),
        )
    };
    let poly = if shape.rows() >= 2 * width {
        ProductPolynomial::<Val, Challenge>::new_packed(
            VariableOrder::Prefix,
            pack(&folded),
            pack(&weights),
        )
    } else {
        ProductPolynomial::<Val, Challenge>::new_unpacked(
            VariableOrder::Prefix,
            Poly::new(folded.clone()),
            Poly::new(weights),
        )
    };
    let mut prover = SumcheckProver::new(poly, sum);
    let mut sumcheck = SumcheckData::default();
    let point = prover.compute_sumcheck_polynomials(
        &mut sumcheck,
        challenger,
        shape.log_rows(),
        pow_bits,
        None,
    );
    let eval = prover.evals().as_slice()[0];
    (InnerProof { sumcheck, eval }, point, folded)
}
