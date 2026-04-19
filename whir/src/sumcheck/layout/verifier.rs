use alloc::vec::Vec;

use p3_field::{ExtensionField, Field};
use p3_matrix::Dimensions;
use p3_multilinear_util::point::Point;
use p3_util::log2_ceil_usize;

use crate::constraints::Constraint;
use crate::constraints::statement::EqStatement;
use crate::sumcheck::layout::{
    MultiClaimVerifier as MultiClaim, OpeningVerifier, Selector, TableLayout,
    VirtualClaimVerifier as VirtualClaim,
};

/// Shape-only description of one verifier table.
#[derive(Debug, Clone)]
pub struct TableShape(Dimensions);

impl TableShape {
    /// Creates a table shape with `2^k` rows and `width` polynomial columns.
    pub fn new(k: usize, width: usize) -> Self {
        assert!(width > 0);
        Self(Dimensions {
            width,
            height: 1 << k,
        })
    }

    /// Returns the total number of stacked evaluations contributed by this table.
    const fn area(&self) -> usize {
        self.0.width * self.0.height
    }

    /// Returns the number of variables of each table polynomial.
    const fn k(&self) -> usize {
        log2_ceil_usize(self.0.height)
    }

    /// Returns the number of polynomial columns in the table.
    const fn width(&self) -> usize {
        self.0.width
    }
}

/// Verifier-side reconstruction of the stacked layout and recorded claims.
#[derive(Debug, Clone)]
pub struct VerifierLayout<F: Field, EF: ExtensionField<F>> {
    /// Placement metadata describing how each table column is stacked.
    layout: Vec<TableLayout>,
    /// Number of variables of the stacked committed polynomial.
    k: usize,
    /// Concrete opening claims recorded per table.
    claim_map: Vec<Vec<MultiClaim<F, EF>>>,
    /// Additional claims sampled directly on the stacked polynomial.
    virtual_claims: Vec<VirtualClaim<F, EF>>,
}

impl<F: Field, EF: ExtensionField<F>> VerifierLayout<F, EF> {
    /// Returns layout metadata for the table with source index `table_idx`.
    fn get_table(&self, table_idx: usize) -> &TableLayout {
        self.layout
            .iter()
            .find(|layout| layout.idx() == table_idx)
            .unwrap()
    }

    /// Reconstructs the verifier-side stacking layout from table shapes alone.
    pub fn new(tables: &[TableShape]) -> Self {
        let mut table_order = (0..tables.len()).collect::<Vec<usize>>();
        table_order.sort_by_key(|&i| tables[i].k());
        let k = log2_ceil_usize(tables.iter().map(TableShape::area).sum::<usize>());

        let mut offset = 0usize;
        let mut layout = Vec::new();
        table_order.iter().rev().for_each(|&table_idx| {
            let table = &tables[table_idx];
            let size = 1usize << table.k();
            let selectors = (0..table.width())
                .map(|_| {
                    let selector = Selector::new(k - table.k(), offset >> table.k());
                    offset += size;
                    selector
                })
                .collect();
            layout.push(TableLayout::new(table_idx, selectors));
        });

        Self {
            layout,
            k,
            claim_map: (0..tables.len()).map(|_| Vec::new()).collect(),
            virtual_claims: Default::default(),
        }
    }

    /// Returns the number of concrete openings recorded so far.
    fn num_claims(&self) -> usize {
        self.claim_map
            .iter()
            .flat_map(|claims| claims.iter().map(MultiClaim::len))
            .sum()
    }

    /// Returns the number of variables of table `table_idx`.
    pub fn num_vars_table(&self, table_idx: usize) -> usize {
        let table = self.get_table(table_idx);
        let selector_vars = table.selectors.first().map(Selector::num_vars).unwrap_or(0);
        self.k - selector_vars
    }

    /// Records concrete opening claims for one table at the given local point.
    pub fn add_claim(&mut self, table_idx: usize, point: Point<EF>, polys: &[usize], evals: &[EF]) {
        let table_layout = self.get_table(table_idx);
        assert_eq!(point.num_vars(), self.num_vars_table(table_idx));
        assert_eq!(polys.len(), evals.len());
        assert!(polys.iter().all(|&i| i < table_layout.num_polys()));

        let openings = polys
            .iter()
            .zip(evals.iter())
            .map(|(&poly_idx, &eval)| OpeningVerifier::new(poly_idx, eval))
            .collect::<Vec<_>>();

        self.claim_map[table_idx].push(MultiClaim::new(point, openings));
    }

    /// Records a virtual evaluation claim on the stacked polynomial.
    pub fn add_virtual_eval(&mut self, point: Point<EF>, evals: EF) {
        assert_eq!(point.num_vars(), self.k);
        self.virtual_claims.push(VirtualClaim::new(point, evals));
    }

    /// Computes the batched claimed sum
    pub fn sum(&self, alpha: EF) -> EF {
        let mut sum = EF::ZERO;
        let mut alpha_i = EF::ONE;
        self.layout.iter().for_each(|table_layout| {
            let claims = &self.claim_map[table_layout.idx()];
            (0..table_layout.num_polys()).for_each(|poly_idx| {
                claims.iter().for_each(|claim| {
                    claim.openings.iter().for_each(|opening| {
                        if opening.poly_idx == poly_idx {
                            sum += opening.eval * alpha_i;
                            alpha_i *= alpha;
                        }
                    });
                });
            });
        });

        self.virtual_claims
            .iter()
            .map(VirtualClaim::eval)
            .zip(alpha.powers().skip(self.num_claims()))
            .for_each(|(eval, alpha_i)| sum += eval * alpha_i);

        sum
    }

    /// Builds the verifier-side equality constraint over all recorded claims.
    pub fn constraint(&self, alpha: EF) -> Constraint<F, EF> {
        let mut eq_statement = EqStatement::initialize(self.k);

        self.layout.iter().for_each(|table_layout| {
            let claims = &self.claim_map[table_layout.idx()];

            table_layout
                .selectors
                .iter()
                .enumerate()
                .for_each(|(poly_idx, selector)| {
                    claims.iter().for_each(|claim| {
                        claim.openings.iter().for_each(|opening| {
                            if opening.poly_idx == poly_idx {
                                let lifted = selector.lift(&claim.point);
                                eq_statement.add_evaluated_constraint(lifted, opening.eval);
                            }
                        });
                    });
                });
        });

        self.virtual_claims.iter().for_each(|claim| {
            eq_statement.add_evaluated_constraint(claim.point.clone(), claim.eval);
        });

        Constraint::new_eq_only(alpha, eq_statement)
    }
}
