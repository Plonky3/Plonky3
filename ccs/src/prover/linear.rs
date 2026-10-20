//! Linear sumcheck prover of the merged weights against the witness.

use alloc::vec::Vec;

use p3_field::{ExtensionField, Field};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::generic_degree::RoundProver;

/// Witness, kept in the base field until the first fold.
enum Witness<F, EF> {
    /// Unbound witness.
    Base(Poly<F>),
    /// Witness after the first fold.
    Ext(Poly<EF>),
}

/// Degree-two product sumcheck prover, sending each round at nodes 0 and 2.
pub(crate) struct LinearProver<F, EF> {
    /// Merged weights of every matrix claim.
    weights: Poly<EF>,
    /// Committed witness.
    witness: Witness<F, EF>,
    /// Node 2 in the extension.
    node_two: EF,
}

impl<F: Field, EF: ExtensionField<F>> LinearProver<F, EF> {
    /// Start the product sumcheck of weights against a witness.
    ///
    /// # Panics
    ///
    /// - The two tables differ in length.
    pub(crate) fn new(weights: Vec<EF>, witness: &[F]) -> Self {
        assert_eq!(weights.len(), witness.len(), "one weight per witness entry");

        // Stay in the base field when node 2 lives there; otherwise lift up front.
        let node_two = EF::interpolation_node(2);
        let witness = if <EF as ExtensionField<F>>::as_base(&node_two).is_some() {
            Witness::Base(Poly::new(witness.to_vec()))
        } else {
            Witness::Ext(Poly::new(witness.iter().copied().map(EF::from).collect()))
        };

        Self {
            weights: Poly::new(weights),
            witness,
            node_two,
        }
    }

    /// Witness value at the bound point.
    ///
    /// # Panics
    ///
    /// - Some witness variable is still free.
    pub(crate) fn witness_eval(&self) -> EF {
        match &self.witness {
            Witness::Base(z) => z.as_constant().map(EF::from),
            Witness::Ext(z) => z.as_constant(),
        }
        .expect("the witness still has free variables")
    }
}

impl<F: Field, EF: ExtensionField<F>> RoundProver<EF> for LinearProver<F, EF> {
    fn fold(&mut self, r: EF) {
        // Bind the leading variable of both factors.
        self.weights.fix_prefix_var_mut(r);
        match &mut self.witness {
            // First fold lifts into the extension.
            Witness::Base(z) => self.witness = Witness::Ext(z.fix_prefix_var(r)),
            // Later folds are in place.
            Witness::Ext(z) => z.fix_prefix_var_mut(r),
        }
    }

    fn round_poly(&self) -> Vec<EF> {
        // Node 1 follows from the running sum.
        let weights = self.weights.as_slice();
        let [h0, h2] = match &self.witness {
            Witness::Base(z) => {
                let two = <EF as ExtensionField<F>>::as_base(&self.node_two)
                    .expect("a base witness is kept only when node 2 is a base element");
                product_round::<F, EF>(weights, z.as_slice(), two, self.node_two)
            }
            Witness::Ext(z) => {
                product_round::<EF, EF>(weights, z.as_slice(), self.node_two, self.node_two)
            }
        };
        alloc::vec![h0, h2]
    }
}

/// Product round polynomial at nodes 0 and 2, with the witness line evaluated in its own field.
fn product_round<A, EF>(weights: &[EF], witness: &[A], two: A, ext_two: EF) -> [EF; 2]
where
    A: Field,
    EF: ExtensionField<A>,
{
    let half = witness.len() / 2;
    let (w_lo, w_hi) = weights.split_at(half);
    let (z_lo, z_hi) = witness.split_at(half);

    // Integer node 2 needs no multiplication: u(2) = 2 * u(1) - u(0).
    let integer_two = ext_two == EF::TWO;

    (0..half)
        .into_par_iter()
        .with_min_task_bytes(2 * size_of::<EF>() + 2 * size_of::<A>())
        .par_fold_reduce(
            || [EF::ZERO; 2],
            |[h0, h2], x| {
                // Both factors at node 2.
                let (w2, z2) = if integer_two {
                    (w_hi[x].double() - w_lo[x], z_hi[x].double() - z_lo[x])
                } else {
                    (
                        w_lo[x] + (w_hi[x] - w_lo[x]) * ext_two,
                        z_lo[x] + (z_hi[x] - z_lo[x]) * two,
                    )
                };
                [h0 + w_lo[x] * z_lo[x], h2 + w2 * z2]
            },
            |[a0, a2], [b0, b2]| [a0 + b0, a2 + b2],
        )
}
