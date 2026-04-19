use alloc::vec::Vec;
use core::marker::PhantomData;

use itertools::Itertools;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_multilinear_util::split_eq::SplitEq;
use p3_util::log2_ceil_usize;

use crate::constraints::Constraint;
use crate::sumcheck::strategy::{SumcheckProver, SumcheckStrategy, VariableOrder};
use crate::sumcheck::svo::{SvoAccumulators, SvoPoint};
use crate::sumcheck::{Claim, SumcheckData};

pub mod prefix;
pub mod suffix;
pub mod verifier;
pub use prefix::*;
pub use suffix::*;
pub use verifier::*;

#[cfg(test)]
pub mod test;

/// Identifies one stacked polynomial slot inside a layouted witness.
#[derive(Debug, Clone, Copy)]
pub struct Selector {
    /// Number of selector bits needed to address the stacked slot.
    num_vars: usize,
    /// Hypercube index of the selected slot.
    index: usize,
}

impl Selector {
    /// Creates a selector over `num_vars` selector bits with the given slot index.
    pub fn new(num_vars: usize, index: usize) -> Self {
        assert!(index < (1 << num_vars));
        Self { num_vars, index }
    }

    /// Returns the selector as a Boolean hypercube point.
    pub fn point<F: Field>(&self) -> Point<F> {
        Point::hypercube(self.index, self.num_vars)
    }

    /// Returns the number of selector variables.
    pub const fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Returns the selected hypercube index.
    pub const fn index(&self) -> usize {
        self.index
    }

    /// Lifts a local point into the stacked witness by prefixing the selector bits.
    pub fn lift<Ext: Field>(&self, other: &Point<Ext>) -> Point<Ext> {
        Point::new(
            self.point()
                .iter()
                .chain(other.iter())
                .copied()
                .collect::<Vec<_>>(),
        )
    }
}

/// Virtual constraint claim used by the prefix layout prover.
pub type VirtualClaimPrefixLayout<F, EF> = Claim<F, EF, Point<EF>, ()>;
/// Virtual constraint claim used by the suffix layout prover.
pub type VirtualClaimSuffixLayout<F, EF> = Claim<F, EF, Point<EF>, SvoAccumulators<EF>>;
/// Virtual constraint claim carried by the verifier-side layout.
pub type VirtualClaimVerifier<F, EF> = Claim<F, EF, Point<EF>, ()>;

impl<F: Field, EF: ExtensionField<F>> VirtualClaimPrefixLayout<F, EF> {
    /// Creates a virtual prefix claim at `point` with value `eval`.
    pub const fn new(point: Point<EF>, eval: EF) -> Self {
        Self {
            point,
            eval,
            data: (),
            _marker: PhantomData,
        }
    }

    /// Returns the claim arity.
    pub const fn num_vars(&self) -> usize {
        self.point.num_vars()
    }
}

impl<F: Field, EF: ExtensionField<F>> VirtualClaimSuffixLayout<F, EF> {
    /// Creates a virtual suffix claim together with its precomputed SVO accumulators.
    pub const fn new(point: Point<EF>, eval: EF, accumulators: SvoAccumulators<EF>) -> Self {
        Self {
            point,
            eval,
            data: accumulators,
            _marker: PhantomData,
        }
    }

    /// Returns the claim arity.
    pub const fn num_vars(&self) -> usize {
        self.point.num_vars()
    }
}

/// An opening claim for one polynomial inside a table.
#[derive(Debug, Clone)]
pub struct Opening<EF: Field, Data> {
    /// Polynomial index inside the source table.
    poly_idx: usize,
    /// Opened value of that polynomial.
    eval: EF,
    /// Layout-specific auxiliary opening data.
    data: Data,
}

impl<EF: Field, Data> Opening<EF, Data> {
    /// Returns the opened value.
    pub const fn eval(&self) -> EF {
        self.eval
    }

    /// Returns layout-specific auxiliary data attached to this opening.
    pub const fn data(&self) -> &Data {
        &self.data
    }
}

/// Prefix-layout opening with no extra prover-side data.
pub type OpeningPrefixLayout<EF> = Opening<EF, ()>;
/// Suffix-layout opening together with per-round partial evaluations.
pub type OpeningSuffixLayout<EF> = Opening<EF, Vec<Poly<EF>>>;
/// Verifier-side opening representation.
pub type OpeningVerifier<EF> = Opening<EF, ()>;

impl<EF: Field> OpeningPrefixLayout<EF> {
    /// Creates a prefix opening for `poly_idx` with value `eval`.
    pub const fn new(poly_idx: usize, eval: EF) -> Self {
        Self {
            poly_idx,
            eval,
            data: (),
        }
    }

    /// Evaluates one table polynomial at a prefix-layout point.
    pub fn eval_poly<F: Field>(
        poly_idx: Option<usize>,
        point: &SplitEq<F, EF>,
        poly: &Poly<F>,
    ) -> Self
    where
        EF: ExtensionField<F>,
    {
        let eval = point.eval_base(poly);
        Self {
            poly_idx: poly_idx.unwrap_or(usize::MAX),
            eval,
            data: (),
        }
    }
}

impl<EF: Field> OpeningSuffixLayout<EF> {
    /// Evaluates one table polynomial at a suffix-layout point and stores SVO partials.
    pub fn eval_poly<F: Field>(
        poly_idx: Option<usize>,
        point: &SvoPoint<F, EF>,
        poly: &Poly<F>,
    ) -> Self
    where
        EF: ExtensionField<F>,
    {
        let (eval, partial_evals) = point.eval(poly);
        Self {
            poly_idx: poly_idx.unwrap_or(usize::MAX),
            eval,
            data: partial_evals,
        }
    }
}

/// A group of openings sharing the same lifted evaluation point.
#[derive(Debug, Clone)]
pub struct MultiClaim<F: Field, EF: ExtensionField<F>, Point, Data> {
    /// Common lifted point used by every opening in the batch.
    point: Point,
    /// Openings attached to that common point.
    openings: Vec<Opening<EF, Data>>,
    /// Keeps the base field in the type without storing a runtime value.
    _marker: PhantomData<F>,
}

impl<F: Field, EF: ExtensionField<F>, Point, Data> MultiClaim<F, EF, Point, Data> {
    /// Creates a multi-opening claim at `point`.
    pub const fn new(point: Point, openings: Vec<Opening<EF, Data>>) -> Self {
        Self {
            point,
            openings,
            _marker: PhantomData,
        }
    }

    /// Returns the common point shared by all openings.
    pub const fn point(&self) -> &Point {
        &self.point
    }

    /// Returns the number of openings in the claim.
    pub const fn len(&self) -> usize {
        self.openings.len()
    }

    /// Returns whether the claim contains no openings.
    pub const fn is_empty(&self) -> bool {
        self.openings.is_empty()
    }

    /// Returns the constituent openings.
    pub fn openings(&self) -> &[Opening<EF, Data>] {
        &self.openings
    }
}

/// Prefix-layout claim over a split equality polynomial.
pub type MultiClaimPrefixLayout<F, EF> = MultiClaim<F, EF, SplitEq<F, EF>, ()>;
/// Suffix-layout claim over an SVO point and per-round partials.
pub type MultiClaimSuffixLayout<F, EF> = MultiClaim<F, EF, SvoPoint<F, EF>, Vec<Poly<EF>>>;
/// Verifier-side multi-opening claim.
pub type MultiClaimVerifier<F, EF> = MultiClaim<F, EF, Point<EF>, ()>;

impl<F: Field, EF: ExtensionField<F>> MultiClaimSuffixLayout<F, EF> {
    /// Materializes the residual equality weights after the SVO rounds.
    pub fn accumulate_into(&self, accumulators: &mut [EF], rs: &Point<EF>, scale: EF) {
        self.point().accumulate_into(accumulators, rs, scale);
    }

    /// Returns the round-`round` partial evaluation from each opening.
    pub fn partial_evals(&self, round: usize) -> Vec<&Poly<EF>> {
        assert!(round < self.num_vars_svo());
        self.openings
            .iter()
            .map(|opening| &opening.data[round])
            .collect()
    }

    /// Returns the number of SVO variables carried by the shared point.
    pub const fn num_vars_svo(&self) -> usize {
        self.point.num_vars_svo()
    }
}

impl<F: Field, EF: ExtensionField<F>> MultiClaimPrefixLayout<F, EF> {
    /// Materializes packed residual equality weights for prefix-style proving.
    pub fn accumulate_into_packed(
        &self,
        accumulators: &mut [EF::ExtensionPacking],
        scale: Option<EF>,
    ) {
        self.point().accumulate_into_packed(accumulators, scale);
    }
}

/// A table of multilinear polynomials with common arity.
#[derive(Debug, Clone)]
pub struct Table<F: Field>(
    /// Polynomials stored in this table.
    Vec<Poly<F>>,
);

impl<F: Field> Table<F> {
    /// Creates a table and checks that all polynomials have the same arity.
    pub fn new(polys: Vec<Poly<F>>) -> Self {
        assert!(!polys.is_empty());
        polys.iter().map(Poly::num_vars).all_equal_value().unwrap();
        Self(polys)
    }

    /// Returns all polynomials in the table.
    pub fn polys(&self) -> &[Poly<F>] {
        &self.0
    }

    /// Returns the `id`-th polynomial.
    pub fn poly(&self, id: usize) -> &Poly<F> {
        &self.0[id]
    }

    /// Evaluates one table polynomial in the prefix layout.
    pub fn eval<EF: ExtensionField<F>>(
        &self,
        idx: usize,
        point: &SplitEq<F, EF>,
    ) -> OpeningPrefixLayout<EF> {
        OpeningPrefixLayout::eval_poly(Some(idx), point, &self.0[idx])
    }

    /// Evaluates one table polynomial in the suffix layout.
    pub fn eval_svo<EF: ExtensionField<F>>(
        &self,
        idx: usize,
        point: &SvoPoint<F, EF>,
    ) -> OpeningSuffixLayout<EF> {
        OpeningSuffixLayout::eval_poly(Some(idx), point, &self.0[idx])
    }

    /// Returns the number of polynomials in the table.
    pub const fn num_polys(&self) -> usize {
        self.0.len()
    }

    /// Returns the common number of variables of the table polynomials.
    pub fn num_vars(&self) -> usize {
        self.0.iter().map(Poly::num_vars).all_equal_value().unwrap()
    }

    /// Returns the total number of stacked evaluations contributed by this table.
    pub fn size(&self) -> usize {
        (1 << self.num_vars()) * self.num_polys()
    }
}

/// Placement metadata for one table inside the stacked witness polynomial.
#[derive(Debug, Clone)]
pub struct TableLayout {
    /// Index of the source table in the witness.
    idx: usize,
    /// Selector assigned to each polynomial of the table.
    selectors: Vec<Selector>,
}

impl TableLayout {
    /// Creates layout metadata for one table and its polynomial selectors.
    pub const fn new(idx: usize, selectors: Vec<Selector>) -> Self {
        Self { idx, selectors }
    }

    /// Returns the number of polynomials placed for this table.
    pub const fn num_polys(&self) -> usize {
        self.selectors.len()
    }

    /// Returns the source table index in the witness.
    pub const fn idx(&self) -> usize {
        self.idx
    }
}

/// Owns the original tables together with their stacked committed polynomial.
#[derive(Debug, Clone)]
pub struct Witness<F: Field> {
    /// Source tables before stacking.
    tables: Vec<Table<F>>,
    /// Placement metadata describing where each table polynomial was stacked.
    layout: Vec<TableLayout>,
    /// Number of variables of the stacked committed polynomial.
    num_vars: usize,
    /// Number of preprocessing rounds handled by the chosen layout strategy.
    folding: usize,
    /// Stacked polynomial committed to the sumcheck prover.
    poly: Poly<F>,
}

impl<F: Field> Witness<F> {
    /// Returns the number of variables of table `id`.
    pub fn num_vars_table(&self, id: usize) -> usize {
        self.tables[id].num_vars()
    }

    /// Builds the stacked witness polynomial and selector layout for the input tables.
    pub fn new(tables: Vec<Table<F>>, folding: usize) -> Self {
        assert!(tables.iter().all(|table| table.num_vars() > folding));

        let mut table_order = (0..tables.len()).collect::<Vec<_>>();
        table_order.sort_by_key(|&i| tables[i].num_vars());
        let num_vars = log2_ceil_usize(tables.iter().map(Table::size).sum::<usize>());

        let mut offset = 0usize;
        let mut layout = Vec::new();
        let mut stacked_poly = Poly::<F>::zero(num_vars);
        table_order.iter().rev().for_each(|&table_idx| {
            let table = &tables[table_idx];
            let num_vars_table = table.num_vars();
            let size = 1usize << num_vars_table;
            let selectors = table
                .0
                .iter()
                .map(|poly| {
                    assert_eq!(num_vars_table, poly.num_vars());
                    let selector =
                        Selector::new(num_vars - num_vars_table, offset >> num_vars_table);
                    let dst_offset = selector.index << poly.num_vars();
                    stacked_poly.as_mut_slice()[dst_offset..dst_offset + poly.num_evals()]
                        .copy_from_slice(poly.as_slice());
                    offset += size;
                    selector
                })
                .collect();
            layout.push(TableLayout::new(table_idx, selectors));
        });

        Self {
            tables,
            layout,
            num_vars,
            folding,
            poly: stacked_poly,
        }
    }

    /// Returns the number of variables of the stacked witness polynomial.
    pub const fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Converts the raw witness into a strategy-specific prover layout.
    pub fn as_committed<EF, L>(self) -> ProverLayout<F, EF, L>
    where
        EF: ExtensionField<F>,
        L: LayoutStrategy,
    {
        let num_tables = self.tables.len();
        ProverLayout {
            tables: self.tables,
            layout: self.layout,
            num_vars: self.num_vars,
            folding: self.folding,
            claim_map: (0..num_tables).map(|_| Vec::new()).collect(),
            poly: self.poly,
            virtual_claims: Vec::new(),
            _marker: PhantomData,
        }
    }
}

/// Strategy hooks for proving and verifying over a shared `ProverLayout` shape.
pub trait LayoutStrategy: Sized {
    type SumcheckStrategy: SumcheckStrategy;
    type Point<F: Field, EF: ExtensionField<F>>;
    type DataOpening<EF: Field>;
    type DataVirtual<EF: Field>;

    fn eval<F: Field, EF: ExtensionField<F>>(
        l: &mut ProverLayout<F, EF, Self>,
        point: &Point<EF>,
        table_idx: usize,
        polys: Vec<usize>,
    ) -> Vec<EF>;

    fn add_virtual_eval<Challenger, F: Field, EF: ExtensionField<F>>(
        l: &mut ProverLayout<F, EF, Self>,
        challenger: &mut Challenger,
    ) -> EF
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>;

    fn new_prover<Challenger, F: Field, EF: ExtensionField<F>>(
        l: ProverLayout<F, EF, Self>,
        sumcheck_data: &mut SumcheckData<F, EF>,
        pow_bits: usize,
        challenger: &mut Challenger,
    ) -> (SumcheckProver<F, EF, Self::SumcheckStrategy>, Point<EF>)
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>;

    fn eval_constraints_poly<F: Field, EF: ExtensionField<F>>(
        constraints: &[Constraint<F, EF>],
        point: &Point<EF>,
    ) -> EF {
        Self::SumcheckStrategy::eval_constraints_poly(constraints, point)
    }

    fn var_order() -> VariableOrder {
        Self::SumcheckStrategy::var_order()
    }
}

/// Multi-opening claim type specialized to one concrete layout strategy.
type LayoutMultiClaim<F, EF, L> =
    MultiClaim<F, EF, <L as LayoutStrategy>::Point<F, EF>, <L as LayoutStrategy>::DataOpening<EF>>;
/// Opening type specialized to one concrete layout strategy.
type LayoutOpening<EF, L> = Opening<EF, <L as LayoutStrategy>::DataOpening<EF>>;
/// Virtual-claim type specialized to one concrete layout strategy.
type LayoutVirtualClaim<F, EF, L> = Claim<F, EF, Point<EF>, <L as LayoutStrategy>::DataVirtual<EF>>;

/// Strategy-parametric prover state shared by prefix and suffix layouts.
pub struct ProverLayout<F: Field, EF: ExtensionField<F>, L: LayoutStrategy> {
    /// Source tables available for opening requests.
    tables: Vec<Table<F>>,
    /// Selector layout describing how tables were embedded into `poly`.
    layout: Vec<TableLayout>,
    /// Number of variables of the stacked committed polynomial.
    num_vars: usize,
    /// Number of preprocessing rounds handled before residual sumcheck.
    folding: usize,
    /// Concrete opening claims recorded per table.
    claim_map: Vec<Vec<LayoutMultiClaim<F, EF, L>>>,
    /// Stacked committed polynomial used by prover and virtual claims.
    poly: Poly<F>,
    /// Additional sampled claims against the stacked polynomial itself.
    virtual_claims: Vec<LayoutVirtualClaim<F, EF, L>>,
    /// Keeps the layout strategy in the type without storing a runtime value.
    _marker: PhantomData<L>,
}

impl<F: Field, EF: ExtensionField<F>, L: LayoutStrategy> ProverLayout<F, EF, L> {
    /// Returns the number of variables of table `id`.
    pub fn num_vars_table(&self, id: usize) -> usize {
        self.tables[id].num_vars()
    }

    /// Returns the number of variables of the stacked witness polynomial.
    pub const fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Returns the total number of concrete openings recorded so far.
    pub fn num_claims(&self) -> usize {
        self.claim_map
            .iter()
            .flat_map(|claims| claims.iter().map(MultiClaim::len))
            .sum()
    }

    /// Returns the stacked witness polynomial committed by this layout.
    pub const fn poly(&self) -> &Poly<F> {
        &self.poly
    }

    /// Returns the configured number of rounds handled before generic sumcheck starts.
    const fn folding(&self) -> usize {
        self.folding
    }

    /// Iterates over recorded openings in stacked-polynomial order.
    fn for_each_opening(
        &self,
        alpha: EF,
        mut f: impl FnMut(
            usize,
            usize,
            usize,
            &LayoutMultiClaim<F, EF, L>,
            &LayoutOpening<EF, L>,
            EF,
            core::ops::Range<usize>,
        ),
    ) {
        let mut alpha_i = EF::ONE;
        let mut off = 0usize;
        for table_layout in self.layout.iter() {
            let combined_size = 1usize << self.num_vars_table(table_layout.idx);
            for poly_idx in 0..table_layout.num_polys() {
                for (claim_idx, claim) in self.claim_map[table_layout.idx].iter().enumerate() {
                    for (opening_idx, opening) in claim.openings.iter().enumerate() {
                        if opening.poly_idx == poly_idx {
                            f(
                                table_layout.idx,
                                claim_idx,
                                opening_idx,
                                claim,
                                opening,
                                alpha_i,
                                off..off + combined_size,
                            );
                            alpha_i *= alpha;
                        }
                    }
                }
                off += combined_size;
            }
        }
    }

    /// Computes the batched claimed sum contributed by concrete and virtual openings.
    fn sum(&self, alpha: EF) -> EF {
        let mut sum = EF::ZERO;
        self.for_each_opening(alpha, |_, _, _, _, opening, alpha_i, _| {
            sum += opening.eval * alpha_i;
        });

        self.virtual_claims
            .iter()
            .map(Claim::eval)
            .zip(alpha.powers().skip(self.num_claims()))
            .for_each(|(eval, alpha_i)| sum += eval * alpha_i);

        sum
    }

    #[tracing::instrument(skip_all)]
    /// Records openings for the selected table polynomials at `point`.
    pub fn eval(&mut self, point: &Point<EF>, table_idx: usize, polys: Vec<usize>) -> Vec<EF> {
        L::eval(self, point, table_idx, polys)
    }

    #[tracing::instrument(skip_all)]
    /// Adds one virtual constraint evaluation sampled through the challenger.
    pub fn add_virtual_eval<Challenger>(&mut self, challenger: &mut Challenger) -> EF
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        L::add_virtual_eval(self, challenger)
    }

    #[tracing::instrument(skip_all)]
    /// Constructs a sumcheck prover over the current layout state.
    pub fn new_prover<Challenger>(
        self,
        sumcheck_data: &mut SumcheckData<F, EF>,
        pow_bits: usize,
        challenger: &mut Challenger,
    ) -> (SumcheckProver<F, EF, L::SumcheckStrategy>, Point<EF>)
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        L::new_prover(self, sumcheck_data, pow_bits, challenger)
    }
}
