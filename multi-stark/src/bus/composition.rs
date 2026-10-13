//! Bus family of the shared AIR sumcheck, binding product-GKR leaf claims to committed traces.
//!
//! For each direction, ProductGKR leaves a claim `L(q)`. This family proves
//! `L(q) - 1` equals the weighted sum of the rowwise bus factors minus one.
//! Short tables are lifted with `eq(prefix, 1^k)`, whose Boolean-cube sum is one.
//!
//! The family runs inside the zerocheck sumcheck, over the same cube and challenges.
//! Its terminal expression is checked against the same openings the AIR constraints read.
//!
//! Wide affine declarations sharing an activation combine their weighted factors before
//! projecting into separate factor and activation tables.
//! Projection is used only when those full-height tables fit within the generic path's first
//! half-height fold. Other declarations read their committed columns or period vectors in place.
//!
//! The first challenge that binds one of its table's row variables folds it in half.
//! A shorter table therefore stays in place while it is dormant.
//! The fold is a half-height polynomial over the representation field.
//!
//! The generic path keeps row selectors in closed form; affine projections absorb them.
//!
//! Every value is computed in the zerocheck backend's representation field `R`, which is
//! isomorphic to the challenge field. Challenges cross into `R` once and round polynomials
//! cross back once, so the composition is the one computed in the challenge field.

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use hashbrown::HashMap;
use hashbrown::hash_map::Entry;
use p3_bus::{BusActivation, BusAffineExpression, BusDirection, BusEvaluation, BusReductionOutput};
use p3_field::{Algebra, ExtensionField, Field};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::generic_degree::{RoundPolyInterpolator, RoundProver};
use p3_sumcheck::layout::{ColumnView, Table};

use crate::bus::BusContext;
use crate::selectors::BoundaryEvals;

/// Prover state for the mixed-height bus composition polynomial.
pub(crate) struct BusCompositionProver<'a, F: Field, EF: ExtensionField<F>, R> {
    /// Checked declarations and physical layout used by every evaluation.
    context: &'a BusContext<F, EF>,
    /// Tuple equality coefficients sampled after commitment.
    fingerprint_weights: Vec<R>,
    /// Random tuple-fingerprint shift.
    offset: R,
    /// Product-GKR anchor, shared by the suffix equality polynomials of every AIR.
    equality_point: Vec<R>,
    /// Source columns and selectors grouped once per AIR.
    airs: Vec<AirState<'a, F, R>>,
    /// Maximum round degree, derived once from the public plan.
    degree: usize,
    /// Number of global variables already bound.
    round: usize,
    /// The challenge field's isomorphism into the representation field.
    to_repr: fn(EF) -> R,
    /// The representation field's isomorphism back into the challenge field.
    from_repr: fn(R) -> EF,
}

/// One bus term evaluated from an AIR's shared folded columns.
struct CompositionTerm<R> {
    /// Coordinates of the symbolic declaration in the bus context.
    owner: p3_bus::BusBlockOwner,
    /// Named bus selecting the tuple-domain slots.
    bus: usize,
    /// Fixed coefficient from direction batching and ProductGKR block selection.
    coefficient: R,
    /// Cube sum of the unweighted row composition before this AIR activates.
    row_claim: R,
}

/// Separate multilinears for a weighted affine factor difference and its activation.
struct ProjectedTerm<R> {
    difference: Poly<R>,
    activation: Option<Poly<R>>,
}

/// Groups belonging to one AIR, with the fixed term coefficients already absorbed.
struct ProjectedAir<R> {
    groups: Vec<ProjectedTerm<R>>,
    /// Equality factors for row variables already bound by the shared sumcheck.
    equality_prefix: R,
    /// Weighted cube sum used while this AIR is dormant.
    row_claim: R,
}

/// Sparse weighted sum without repeatedly copying a growing coefficient vector.
struct AffineSum<R> {
    expression: BusAffineExpression<R>,
    columns: [BTreeMap<usize, R>; 3],
}

impl<R: Field> AffineSum<R> {
    fn new(mut expression: BusAffineExpression<R>, weight: R) -> Self {
        let columns = [
            core::mem::take(&mut expression.main),
            core::mem::take(&mut expression.preprocessed),
            core::mem::take(&mut expression.periodic),
        ]
        .map(|entries| {
            entries
                .into_iter()
                .map(|(column, coefficient)| (column, weight * coefficient))
                .collect()
        });
        expression.constant *= weight;
        expression.first_row *= weight;
        expression.last_row *= weight;
        expression.transition *= weight;
        Self {
            expression,
            columns,
        }
    }

    fn add(&mut self, expression: BusAffineExpression<R>, weight: R) {
        self.expression.constant += weight * expression.constant;
        self.expression.first_row += weight * expression.first_row;
        self.expression.last_row += weight * expression.last_row;
        self.expression.transition += weight * expression.transition;
        for (sum, entries) in self.columns.iter_mut().zip([
            expression.main,
            expression.preprocessed,
            expression.periodic,
        ]) {
            for (column, coefficient) in entries {
                *sum.entry(column).or_insert(R::ZERO) += weight * coefficient;
            }
        }
    }

    fn finish(mut self) -> BusAffineExpression<R> {
        for (entries, sum) in [
            &mut self.expression.main,
            &mut self.expression.preprocessed,
            &mut self.expression.periodic,
        ]
        .into_iter()
        .zip(self.columns)
        {
            *entries = sum
                .into_iter()
                .filter(|&(_, coefficient)| coefficient != R::ZERO)
                .collect();
        }
        self.expression
    }
}

impl<R: Field> ProjectedTerm<R> {
    fn at(&self, row: usize) -> R {
        let difference = self.difference.as_slice()[row];
        self.activation
            .as_ref()
            .map_or(difference, |poly| difference * poly.as_slice()[row])
    }

    fn fold(&mut self, challenge: R) {
        self.difference.fix_prefix_var_mut(challenge);
        if let Some(activation) = &mut self.activation {
            activation.fix_prefix_var_mut(challenge);
        }
    }
}

impl<R: Field> ProjectedAir<R> {
    /// Weighted body coefficients, before applying the current equality factor.
    fn round_coefficients(&self, equality: &Poly<R>) -> [R; 3] {
        let values = equality.as_slice();
        let half = values.len() / 2;
        let quadratic = self.groups.iter().any(|group| group.activation.is_some());
        let add = |mut sums: [R; 3], row: [R; 3]| {
            for (sum, value) in sums.iter_mut().zip(row) {
                *sum += value;
            }
            sums
        };
        let sums = (0..half)
            .into_par_iter()
            .map(|row| {
                let mut sums = [R::ZERO; 3];
                for group in &self.groups {
                    let difference = group.difference.as_slice();
                    let (low, high) = (difference[row], difference[row + half]);
                    if let Some(activation) = &group.activation {
                        let activation = activation.as_slice();
                        let (a_low, a_high) = (activation[row], activation[row + half]);
                        sums[0] += low * a_low;
                        sums[1] += high * a_high;
                        sums[2] += (high - low) * (a_high - a_low);
                    } else {
                        sums[0] += low;
                        sums[1] += high;
                    }
                }
                // The leading Boolean equality factor sums to one.
                let weight = values[row] + values[row + half];
                sums[0] *= weight;
                sums[1] *= weight;
                if quadratic {
                    sums[2] *= weight;
                }
                sums
            })
            .par_fold_reduce(|| [R::ZERO; 3], add, add);
        // The endpoint total at one contains all three monomial coefficients.
        [sums[0], sums[1] - sums[0] - sums[2], sums[2]]
    }
}

/// One source multilinear, read in place until one of its row variables is bound, then folded.
enum Source<'a, F: Field, EF> {
    /// A committed column, decoding Boolean words where the table packs them.
    Committed(ColumnView<'a, F>),
    /// A periodic column, which repeats its period vector down the whole table.
    ///
    /// The period is a power of two dividing the height, so row `r` reads entry `r mod period`.
    Periodic {
        /// Values of one period.
        period: Vec<F>,
        /// Rows of the table the column belongs to.
        height: usize,
    },
    /// The column after one or more of its row variables are bound.
    Folded(Poly<EF>),
}

impl<F: Field, EF: Field + Algebra<F>> Source<'_, F, EF> {
    /// The value at `row` of the current hypercube.
    #[inline]
    fn at(&self, row: usize) -> EF {
        match self {
            Self::Committed(column) => column.value(row).into(),
            Self::Periodic { period, .. } => period[row & (period.len() - 1)].into(),
            Self::Folded(poly) => poly.as_slice()[row],
        }
    }

    /// The line through rows `row` and `row + half` of the current hypercube, at `node`.
    #[inline]
    fn interpolate(&self, row: usize, half: usize, node: EF) -> EF {
        // Every round transmits node zero. Read its endpoint without loading the
        // other row or multiplying the column difference by zero.
        if node == EF::ZERO {
            return self.at(row);
        }
        match self {
            Self::Committed(column @ ColumnView::Boolean { .. }) => {
                let low = usize::from(column.boolean_bit(row).unwrap());
                let upper = row + half;
                let high = usize::from(column.boolean_bit(upper).unwrap());
                // The line through two bits has only these four possible values.
                [EF::ZERO, node, EF::ONE - node, EF::ONE][2 * low + high]
            }
            Self::Committed(column) => {
                let low = column.value(row);
                node * (column.value(row + half) - low) + low
            }
            Self::Periodic { period, .. } => {
                let mask = period.len() - 1;
                let low = period[row & mask];
                node * (period[(row + half) & mask] - low) + low
            }
            Self::Folded(poly) => {
                let values = poly.as_slice();
                values[row] + (values[row + half] - values[row]) * node
            }
        }
    }

    /// Binds the leading row variable, moving a column read in place into `EF`.
    fn fold(&mut self, challenge: EF) {
        let folded = match self {
            Self::Committed(column @ ColumnView::Boolean { .. }) => {
                let column = *column;
                let half = column.len() / 2;
                let lookup = [EF::ZERO, challenge, EF::ONE - challenge, EF::ONE];
                Poly::new((0..half).into_par_iter().map_collect_min_task_bytes(
                    2 * size_of::<F>() + size_of::<EF>(),
                    |row| {
                        let low = usize::from(column.boolean_bit(row).unwrap());
                        let upper = row + half;
                        let high = usize::from(column.boolean_bit(upper).unwrap());
                        lookup[2 * low + high]
                    },
                ))
            }
            Self::Committed(column) => {
                let column = *column;
                fold_in_half(column.len() / 2, challenge, |row| column.value(row))
            }
            Self::Periodic { period, height } => {
                let mask = period.len() - 1;
                fold_in_half(*height / 2, challenge, |row| period[row & mask])
            }
            Self::Folded(poly) => {
                poly.fix_prefix_var_mut(challenge);
                return;
            }
        };
        *self = Self::Folded(folded);
    }
}

/// Binds the leading variable of a base-field column of `2 * half` rows, read cell by cell.
fn fold_in_half<F, EF>(half: usize, challenge: EF, cell: impl Fn(usize) -> F + Sync) -> Poly<EF>
where
    F: Field,
    EF: Field + Algebra<F>,
{
    // One item reads a pair of column cells and writes one folded entry.
    Poly::new((0..half).into_par_iter().map_collect_min_task_bytes(
        2 * size_of::<F>() + size_of::<EF>(),
        |row| {
            let low = cell(row);
            challenge * (cell(row + half) - low) + low
        },
    ))
}

/// Source multilinears shared by every bus term owned by one AIR.
struct AirState<'a, F: Field, R> {
    /// Sources of the main columns this AIR's declarations read.
    main: Vec<Source<'a, F, R>>,
    /// Column index of each of those, and the declared width they are placed into.
    main_layout: (Vec<usize>, usize),
    /// Sources of the preprocessed columns this AIR's declarations read.
    preprocessed: Vec<Source<'a, F, R>>,
    /// Column index of each of those, and the declared width they are placed into.
    preprocessed_layout: (Vec<usize>, usize),
    /// Sources of the periodic columns this AIR's declarations read.
    periodic: Vec<Source<'a, F, R>>,
    /// Column index of each of those, and the declared width they are placed into.
    periodic_layout: (Vec<usize>, usize),
    /// Folded boundary-selector values at the bound row variables.
    boundary: BoundaryEvals<R>,
    /// Equality polynomial anchored at the ProductGKR row point. Projected AIRs keep
    /// its unbound factors normalized; generic AIRs include the bound prefix here.
    equality: Poly<R>,
    /// Bus terms emitted by this AIR.
    terms: Vec<CompositionTerm<R>>,
    /// Optional compact affine representation, grouped by exact activation expression.
    projected: Option<ProjectedAir<R>>,
    /// Number of global prefix variables absent from this AIR table.
    unused_prefix: usize,
    /// Evaluation of the fixed all-one-vertex selector on bound prefix coordinates.
    prefix_evaluation: R,
    /// Public inputs read by this block's symbolic expressions.
    public_values: Vec<F>,
}

impl<F, R> AirState<'_, F, R>
where
    F: Field,
    R: Field + Algebra<F>,
{
    /// Project only when every declaration is affine and the retained tables are no larger
    /// than the generic path's first fold. Count actual source columns, not declared widths.
    fn initialize_projection<EF: ExtensionField<F>>(
        &mut self,
        context: &BusContext<F, EF>,
        fingerprint_weights: &[R],
        offset: R,
    ) {
        let sources = self.main.len() + self.preprocessed.len() + self.periodic.len();
        let mut always = false;
        let mut conditional = false;
        for term in &self.terms {
            if matches!(
                context.interaction(term.owner).activation,
                BusActivation::Always
            ) {
                always = true;
            } else {
                conditional = true;
            }
        }
        let minimum_projections = usize::from(always) + 2 * usize::from(conditional);
        // Poly folding retains its allocation, so compare full projected capacity to
        // half-height source capacity. A one-row table never needs a source fold.
        if self.equality.as_slice().len() == 1 || minimum_projections > sources / 2 {
            return;
        }
        let affine = |term: &CompositionTerm<R>| {
            context
                .plan()
                .compile_factor(
                    term.bus,
                    context.interaction(term.owner),
                    fingerprint_weights,
                    offset,
                )
                .expect("a checked bus plan compiles against its own declarations")
                .affine(&self.public_values)
                .expect("a planned expression resolves its public inputs")
        };
        if let [term] = self.terms.as_slice() {
            let Some(mut factor) = affine(term) else {
                return;
            };
            // With no terms to combine, skip sparse aggregation and weight
            // whichever is shorter: the affine coefficients or the row table.
            let expression = &mut factor.factor_minus_one;
            let scale_rows = self.equality.as_slice().len()
                < expression.main.len()
                    + expression.preprocessed.len()
                    + expression.periodic.len()
                    + 4;
            if !scale_rows && term.coefficient != R::ONE {
                for coefficient in [
                    &mut expression.constant,
                    &mut expression.first_row,
                    &mut expression.last_row,
                    &mut expression.transition,
                ] {
                    *coefficient *= term.coefficient;
                }
                for (_, coefficient) in expression
                    .main
                    .iter_mut()
                    .chain(&mut expression.preprocessed)
                    .chain(&mut expression.periodic)
                {
                    *coefficient *= term.coefficient;
                }
            }
            let mut difference = self.materialize_affine(&factor.factor_minus_one);
            if scale_rows && term.coefficient != R::ONE {
                for value in difference.as_mut_slice() {
                    *value *= term.coefficient;
                }
            }
            self.projected = Some(ProjectedAir {
                equality_prefix: R::ONE,
                groups: vec![ProjectedTerm {
                    difference,
                    activation: factor
                        .activation
                        .as_ref()
                        .map(|expression| self.materialize_affine(expression)),
                }],
                row_claim: R::ZERO,
            });
            self.periodic.clear();
            return;
        }
        // All terms share this AIR's row point. For each identical activation S,
        // sum_i c_i * eq * D_i * S = eq * (sum_i c_i * D_i) * S.
        // Keep None distinct from Some(1), and retain even zero aggregate groups:
        // the table count can only increase, so exceeding the bound can abort early.
        let mut groups = HashMap::new();
        let mut projections = 0;
        for term in &self.terms {
            let Some(factor) = affine(term) else {
                return;
            };
            let index = groups.len();
            match groups.entry(factor.activation) {
                Entry::Occupied(mut entry) => {
                    let (_, difference): &mut (usize, AffineSum<R>) = entry.get_mut();
                    difference.add(factor.factor_minus_one, term.coefficient);
                }
                Entry::Vacant(entry) => {
                    projections += 1 + usize::from(entry.key().is_some());
                    if projections > sources / 2 {
                        return;
                    }
                    entry.insert((
                        index,
                        AffineSum::new(factor.factor_minus_one, term.coefficient),
                    ));
                }
            }
        }
        // Never let hash iteration order determine evaluation order. No source is
        // changed until every declaration passes extraction and the storage bound.
        let mut groups = groups.into_iter().collect::<Vec<_>>();
        groups.sort_unstable_by_key(|(_, (index, _))| *index);
        self.projected = Some(ProjectedAir {
            equality_prefix: R::ONE,
            groups: groups
                .into_iter()
                .map(|(activation, (_, difference))| ProjectedTerm {
                    difference: self.materialize_affine(&difference.finish()),
                    activation: activation
                        .as_ref()
                        .map(|expression| self.materialize_affine(expression)),
                })
                .collect(),
            row_claim: R::ZERO,
        });
        // Borrowed committed views cost no column buffers. Period vectors are owned.
        self.periodic.clear();
    }

    /// Evaluate one affine expression on the AIR's own Boolean cube, column by column.
    fn materialize_affine(&self, expression: &BusAffineExpression<R>) -> Poly<R> {
        let height = self.equality.as_slice().len();
        let mut values = vec![expression.constant + expression.transition; height];
        values[0] += expression.first_row;
        values[height - 1] += expression.last_row - expression.transition;
        for (coefficients, sources, indices) in [
            (&expression.main, &self.main, &self.main_layout.0),
            (
                &expression.preprocessed,
                &self.preprocessed,
                &self.preprocessed_layout.0,
            ),
            (
                &expression.periodic,
                &self.periodic,
                &self.periodic_layout.0,
            ),
        ] {
            for &(index, coefficient) in coefficients {
                let source = &sources[indices
                    .binary_search(&index)
                    .expect("an affine leaf is a planned source")];
                match source {
                    Source::Committed(column @ ColumnView::Boolean { .. }) => {
                        // Packed bits select a coefficient without a representation-field product.
                        let lookup = [R::ZERO, coefficient];
                        for (word_index, chunk) in values.chunks_mut(64).enumerate() {
                            let mut bits = column.boolean_word(word_index).unwrap();
                            for value in chunk {
                                *value += lookup[(bits & 1) as usize];
                                bits >>= 1;
                            }
                        }
                    }
                    Source::Committed(column) => {
                        for (row, value) in values.iter_mut().enumerate() {
                            *value += coefficient * column.value(row);
                        }
                    }
                    Source::Periodic { period, .. } => {
                        let mask = period.len() - 1;
                        for (row, value) in values.iter_mut().enumerate() {
                            *value += coefficient * period[row & mask];
                        }
                    }
                    Source::Folded(_) => unreachable!("projection precedes every fold"),
                }
            }
        }
        Poly::new(values)
    }

    /// Compute initial cube sums in one pass over this AIR's active representation.
    ///
    /// The result is read only while the global prefix is still ahead of this table.
    fn initialize_claims<EF: ExtensionField<F>>(
        &mut self,
        context: &BusContext<F, EF>,
        fingerprint_weights: &[R],
        offset: R,
    ) {
        let height = self.equality.as_slice().len();

        if let Some(projected) = &mut self.projected {
            projected.row_claim = (0..height)
                .into_par_iter()
                .map(|row| {
                    self.equality.as_slice()[row]
                        * projected
                            .groups
                            .iter()
                            .map(|group| group.at(row))
                            .sum::<R>()
                })
                .sum();
            return;
        }

        // Slot placement is settled once per term, outside the row loop below.
        let factors = self
            .terms
            .iter()
            .map(|term| {
                context
                    .plan()
                    .compile_factor(
                        term.bus,
                        context.interaction(term.owner),
                        fingerprint_weights,
                        offset,
                    )
                    .expect("a checked bus plan compiles against its own declarations")
            })
            .collect::<Vec<_>>();

        let main_polys = &self.main;
        let (main_indices, main_width) = (&self.main_layout.0, self.main_layout.1);
        let fixed_polys = &self.preprocessed;
        let (fixed_indices, fixed_width) =
            (&self.preprocessed_layout.0, self.preprocessed_layout.1);
        let periodic_polys = &self.periodic;
        let (periodic_indices, periodic_width) = (&self.periodic_layout.0, self.periodic_layout.1);
        let equality = &self.equality;
        let public_values = &self.public_values;
        let claims = (0..height)
            .into_par_iter()
            .par_fold_reduce(
                || {
                    (
                        R::zero_vec(factors.len()),
                        R::zero_vec(main_width),
                        R::zero_vec(fixed_width),
                        R::zero_vec(periodic_width),
                        Vec::new(),
                    )
                },
                |(mut claims, mut main, mut preprocessed, mut periodic, mut scratch), row| {
                    // Unread columns keep their zero, which no planned expression names.
                    for (&index, column) in main_indices.iter().zip(main_polys) {
                        main[index] = column.at(row);
                    }
                    for (&index, column) in fixed_indices.iter().zip(fixed_polys) {
                        preprocessed[index] = column.at(row);
                    }
                    for (&index, column) in periodic_indices.iter().zip(periodic_polys) {
                        periodic[index] = column.at(row);
                    }
                    let boundary = BoundaryEvals::from_row(row, height);
                    let evaluation = BusEvaluation {
                        main: &main,
                        preprocessed: &preprocessed,
                        public: public_values,
                        periodic: &periodic,
                        is_first_row: boundary.first,
                        is_last_row: boundary.last,
                        is_transition: boundary.transition,
                    };
                    let weight = equality.as_slice()[row];
                    for (claim, factor) in claims.iter_mut().zip(&factors) {
                        let value = factor
                            .evaluate::<R>(&mut scratch, evaluation)
                            .expect("a planned expression resolves against its owning table");
                        *claim += weight * (value - R::ONE);
                    }
                    (claims, main, preprocessed, periodic, scratch)
                },
                |(mut left, main, preprocessed, periodic, scratch), (right, ..)| {
                    for (claim, partial) in left.iter_mut().zip(right) {
                        *claim += partial;
                    }
                    (left, main, preprocessed, periodic, scratch)
                },
            )
            .0;

        for (term, claim) in self.terms.iter_mut().zip(claims) {
            term.row_claim = claim;
        }
    }
}

impl<'a, F, EF, R> BusCompositionProver<'a, F, EF, R>
where
    F: Field,
    EF: ExtensionField<F>,
    R: Field + Algebra<F>,
{
    /// Build the formal polynomial whose cube sum must equal the ProductGKR claims.
    ///
    /// # Arguments
    ///
    /// - `num_variables`: width of the shared cube, at least the tallest bus table.
    /// - `periodic`: the period vectors [`BusContext::period_vectors`] returns.
    /// - `to_repr`, `from_repr`: mutually inverse field isomorphisms between the challenge
    ///   field and the representation field, agreeing on the trace field.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        context: &'a BusContext<F, EF>,
        output: &BusReductionOutput<EF>,
        tables: &[&'a Table<F>],
        preprocessed: &[Option<&'a Table<F>>],
        periodic: &[Option<Vec<Vec<F>>>],
        public_values: &[&[F]],
        direction_challenge: EF,
        num_variables: usize,
        to_repr: fn(EF) -> R,
        from_repr: fn(R) -> EF,
    ) -> Self {
        debug_assert!(num_variables >= context.max_num_variables());
        // Sources, public values and plan constants enter `R` through its own embedding of `F`.
        debug_assert!(
            R::from(F::GENERATOR) == to_repr(EF::from(F::GENERATOR))
                && from_repr(to_repr(EF::GENERATOR)) == EF::GENERATOR,
            "the representation field must embed the trace field through the challenge field"
        );
        let weights: Vec<R> = output
            .challenges
            .fingerprint_weights()
            .into_iter()
            .map(to_repr)
            .collect();
        let offset = to_repr(output.challenges.offset);
        let mut airs = (0..tables.len()).map(|_| None).collect::<Vec<_>>();

        for direction in BusDirection::ALL {
            let direction_weight = match direction {
                BusDirection::Push => EF::ONE,
                BusDirection::Pull => direction_challenge,
            };
            for share in context.plan().terminal_shares(direction) {
                let air = share.owner.air;
                let row_point = &output.product.point[share.prefix_variables..];
                let block_weight = share
                    .prefix_weight(&output.product.point)
                    .expect("a planned share addresses its own product-tree point");
                let coefficient = to_repr(direction_weight * block_weight);
                let state = airs[air].get_or_insert_with(|| {
                    // Column views read a packed Boolean table without expanding it first.
                    // Only the columns a declaration reads are kept, and they are read in place.
                    let main_columns = context.main_columns(air);
                    let table = tables[air];
                    let main = main_columns
                        .iter()
                        .map(|&column| Source::Committed(table.column(column)))
                        .collect::<Vec<_>>();
                    let fixed_columns = context.preprocessed_columns(air);
                    let fixed_width = preprocessed[air].map_or(0, Table::num_polys);
                    let preprocessed = preprocessed[air]
                        .iter()
                        .flat_map(|&table| {
                            fixed_columns
                                .iter()
                                .map(move |&column| Source::Committed(table.column(column)))
                        })
                        .collect::<Vec<_>>();
                    // Periodic columns keep one period each and fold exactly like committed ones.
                    let periodic_columns = context.periodic_columns(air);
                    let periodic_width = periodic[air].as_ref().map_or(0, Vec::len);
                    let height = 1usize << table.num_variables();
                    let periodic = periodic[air]
                        .iter()
                        .flat_map(|periods| {
                            periodic_columns
                                .iter()
                                .map(move |&column| Source::Periodic {
                                    period: periods[column].clone(),
                                    height,
                                })
                        })
                        .collect::<Vec<_>>();
                    AirState {
                        main,
                        main_layout: (main_columns.to_vec(), tables[air].num_polys()),
                        preprocessed,
                        preprocessed_layout: (fixed_columns.to_vec(), fixed_width),
                        periodic,
                        periodic_layout: (periodic_columns.to_vec(), periodic_width),
                        // No row variable is bound yet, so both prefix products are empty.
                        boundary: BoundaryEvals::at(&[]),
                        equality: Poly::new(
                            Point::new(row_point.iter().copied().map(to_repr).collect::<Vec<_>>())
                                .equality_weights_msb(),
                        ),
                        terms: Vec::new(),
                        projected: None,
                        unused_prefix: num_variables - share.row_variables,
                        prefix_evaluation: R::ONE,
                        public_values: public_values[air].to_vec(),
                    }
                });
                // Row geometry is captured from the first share and reused by every later one.
                debug_assert_eq!(
                    state.unused_prefix,
                    num_variables - share.row_variables,
                    "every block of one AIR shares its trace height"
                );
                state.terms.push(CompositionTerm {
                    owner: share.owner,
                    bus: share.bus,
                    coefficient,
                    row_claim: R::ZERO,
                });
            }
        }

        for air in airs.iter_mut().flatten() {
            // A dormant wide table benefits before its first active round: its initial
            // row claim reads the compact projection instead of revisiting every source.
            air.initialize_projection(context, &weights, offset);
            // A full-height table never reads its own initial row claim.
            if air.unused_prefix > 0 {
                air.initialize_claims(context, &weights, offset);
            }
        }

        Self {
            context,
            fingerprint_weights: weights,
            offset,
            equality_point: if airs.iter().flatten().any(|air| air.projected.is_some()) {
                output.product.point.iter().copied().map(to_repr).collect()
            } else {
                Vec::new()
            },
            airs: airs.into_iter().flatten().collect(),
            degree: context.composition_degree(),
            round: 0,
            to_repr,
            from_repr,
        }
    }

    fn evaluate_air(&self, air: &AirState<'_, F, R>, node: R) -> R {
        debug_assert!(air.projected.is_none());
        let half = air.equality.as_slice().len() / 2;
        // Slot placement is settled once per term, outside the row loop below.
        let factors = air
            .terms
            .iter()
            .map(|term| {
                self.context
                    .plan()
                    .compile_factor(
                        term.bus,
                        self.context.interaction(term.owner),
                        &self.fingerprint_weights,
                        self.offset,
                    )
                    .expect("a checked bus plan compiles against its own declarations")
            })
            .collect::<Vec<_>>();

        // Interpolate shared columns once, then evaluate every declaration owned by this AIR.
        // The selectors' line at `node` is the selectors with `node` bound as one more variable.
        let mut prefix = air.boundary;
        prefix.apply(node);
        (0..half)
            .into_par_iter()
            .map_init(
                || {
                    (
                        R::zero_vec(air.main_layout.1),
                        R::zero_vec(air.preprocessed_layout.1),
                        R::zero_vec(air.periodic_layout.1),
                        Vec::new(),
                    )
                },
                |(main, prep, periodic, scratch), row| {
                    // Unread columns keep their zero, which no planned expression names.
                    for (&index, source) in air.main_layout.0.iter().zip(&air.main) {
                        main[index] = source.interpolate(row, half, node);
                    }
                    for (&index, source) in air.preprocessed_layout.0.iter().zip(&air.preprocessed)
                    {
                        prep[index] = source.interpolate(row, half, node);
                    }
                    for (&index, source) in air.periodic_layout.0.iter().zip(&air.periodic) {
                        periodic[index] = source.interpolate(row, half, node);
                    }
                    let boundary = BoundaryEvals::from_row_with_prefix(row, half, prefix);
                    let evaluation = BusEvaluation {
                        main,
                        preprocessed: prep,
                        public: &air.public_values,
                        periodic,
                        is_first_row: boundary.first,
                        is_last_row: boundary.last,
                        is_transition: boundary.transition,
                    };
                    let values = air.equality.as_slice();
                    let equality = values[row] + (values[row + half] - values[row]) * node;
                    air.terms
                        .iter()
                        .zip(&factors)
                        .map(|(term, factor)| {
                            let value = factor
                                .evaluate::<R>(scratch, evaluation)
                                .expect("a planned expression resolves against its folded table");
                            term.coefficient * equality * (value - R::ONE)
                        })
                        .sum::<R>()
                },
            )
            .sum()
    }
}

impl<F, EF, R> RoundProver<EF> for BusCompositionProver<'_, F, EF, R>
where
    F: Field,
    EF: ExtensionField<F>,
    R: Field + Algebra<F>,
{
    fn fold(&mut self, challenge: EF) {
        let challenge = (self.to_repr)(challenge);
        // Dormant blocks evaluate one more coordinate of χ_k at the all-one vertex.
        for air in &mut self.airs {
            if self.round < air.unused_prefix {
                air.prefix_evaluation *= challenge;
                continue;
            }
            if let Some(projected) = &mut air.projected {
                for projection in &mut projected.groups {
                    projection.fold(challenge);
                }
                let anchor =
                    self.equality_point[self.equality_point.len() - air.equality.num_variables()];
                projected.equality_prefix *= Point::eval_eq(&[anchor], &[challenge]);
                // Keep the suffix normalized even when the accumulated prefix becomes zero.
                air.equality.sum_prefix_var_mut();
            } else {
                for source in air
                    .main
                    .iter_mut()
                    .chain(&mut air.preprocessed)
                    .chain(&mut air.periodic)
                {
                    source.fold(challenge);
                }
                air.boundary.apply(challenge);
                air.equality.fix_prefix_var_mut(challenge);
            }
        }
        self.round += 1;
    }

    fn round_poly(&self) -> Vec<EF> {
        // Every projected body has degree at most two. Read its rows once for all
        // transmitted nodes, including when another AIR raises the shared degree.
        let projected = self
            .airs
            .iter()
            .map(|air| {
                if self.round < air.unused_prefix {
                    None
                } else {
                    air.projected
                        .as_ref()
                        .map(|projection| projection.round_coefficients(&air.equality))
                }
            })
            .collect::<Vec<_>>();
        // Generic-degree encoding omits node one, which the verifier derives from the claim.
        (0..self.degree)
            .map(RoundPolyInterpolator::<EF>::transmitted_node)
            .map(|node| {
                let node = (self.to_repr)(node);
                let sum = self
                    .airs
                    .iter()
                    .zip(&projected)
                    .map(|(air, coefficients)| {
                        let body = if self.round < air.unused_prefix {
                            // χ_k contributes the current node; its remaining cube sum is one.
                            node * air.projected.as_ref().map_or_else(
                                || {
                                    air.terms
                                        .iter()
                                        .map(|term| term.coefficient * term.row_claim)
                                        .sum::<R>()
                                },
                                |projected| projected.row_claim,
                            )
                        } else if let Some([constant, linear, quadratic]) = *coefficients {
                            let projection = air.projected.as_ref().unwrap();
                            let anchor = self.equality_point
                                [self.equality_point.len() - air.equality.num_variables()];
                            let equality =
                                projection.equality_prefix * Point::eval_eq(&[anchor], &[node]);
                            equality * (constant + node * (linear + node * quadratic))
                        } else {
                            // Active terms share one interpolation of their AIR columns.
                            self.evaluate_air(air, node)
                        };
                        air.prefix_evaluation * body
                    })
                    .sum();
                (self.from_repr)(sum)
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "composition_projection_tests.rs"]
mod projection_tests;

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_air::symbolic::AirLayout;
    use p3_air::{Air, BaseAir, WindowAccess};
    use p3_baby_bear::BabyBear;
    use p3_binary_field::{BinaryField128, Ghash128};
    use p3_bus::{BusActivation, BusDirection, BusInteractionBuilder, BusName, BusSymbolicBuilder};
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{Algebra, Field, PrimeCharacteristicRing};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_multilinear_util::poly::Poly;
    use p3_sumcheck::generic_degree::RoundPolyInterpolator;
    use p3_sumcheck::layout::Table;
    use p3_util::log2_strict_usize;
    use rand::distr::{Distribution, StandardUniform};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::Source;
    use crate::selectors::BoundaryEvals;

    struct TransitionBusAir;

    impl BaseAir<BabyBear> for TransitionBusAir {
        fn width(&self) -> usize {
            // One column supplies the payload used by the transition-weighted tuple.
            1
        }
    }

    impl<AB: BusInteractionBuilder<F = BabyBear>> Air<AB> for TransitionBusAir {
        fn eval(&self, builder: &mut AB) {
            // Multiplying by the transition selector exposes its round-degree contribution.
            let value: AB::Expr = builder.main().current_slice()[0].into();
            builder.push_bus_interaction(
                BusName::new("transition"),
                BusDirection::Push,
                [builder.is_transition() * value],
                BusActivation::Always,
            );
        }
    }

    #[test]
    fn transition_selectors_contribute_one_to_each_round_degree() {
        let profile: BusSymbolicBuilder<BabyBear> =
            BusSymbolicBuilder::from_air(&TransitionBusAir, AirLayout::from_air(&TransitionBusAir));

        assert_eq!(
            profile.interactions()[0].factor_degree_multiple_with_transition(1),
            2
        );
    }

    #[test]
    fn round_nodes_are_distinct_in_characteristic_two() {
        // Integer embedding would map node two back to zero in characteristic two.
        let nodes = (0..5)
            .map(RoundPolyInterpolator::<BinaryField128>::transmitted_node)
            .collect::<alloc::vec::Vec<_>>();

        assert_eq!(nodes[0], BinaryField128::ZERO);
        assert_ne!(nodes[1], BinaryField128::ZERO);
        for (index, node) in nodes.iter().enumerate() {
            assert!(!nodes[index + 1..].contains(node));
        }
    }

    /// Runs one source through every round beside the dense polynomial it stands for.
    ///
    /// Each round checks the line the prover evaluates at a node, then the fold it keeps.
    /// The closed-form selectors are checked the same way against materialized indicators.
    fn assert_folds_like_dense<F, EF>(
        mut source: Source<'_, F, EF>,
        cells: &[F],
        rng: &mut SmallRng,
    ) where
        F: Field,
        EF: Field + Algebra<F>,
        StandardUniform: Distribution<EF>,
    {
        let height = cells.len();
        // The line through rows `row` and `row + half` of a dense polynomial, at `node`.
        let line = |poly: &Poly<EF>, row: usize, node: EF| {
            let values = poly.as_slice();
            let half = values.len() / 2;
            values[row] + (values[row + half] - values[row]) * node
        };
        let selectors =
            |boundary: BoundaryEvals<EF>| [boundary.first, boundary.last, boundary.transition];
        let mut dense = Poly::new(cells.iter().map(|&cell| EF::from(cell)).collect());
        let mut prefix = BoundaryEvals::<EF>::at(&[]);
        let mut indicators = [
            Poly::new((0..height).map(|row| EF::from_bool(row == 0)).collect()),
            Poly::new(
                (0..height)
                    .map(|row| EF::from_bool(row + 1 == height))
                    .collect(),
            ),
            Poly::new(
                (0..height)
                    .map(|row| EF::from_bool(row + 1 < height))
                    .collect(),
            ),
        ];

        // A dormant table's claim reads every row before any fold.
        for row in 0..height {
            assert_eq!(source.at(row), dense.as_slice()[row]);
            assert_eq!(
                selectors(BoundaryEvals::from_row(row, height)),
                indicators.each_ref().map(|poly| poly.as_slice()[row])
            );
        }

        for round in 0..log2_strict_usize(height) {
            let half = height >> (round + 1);
            for node in [EF::ZERO, EF::ONE, EF::interpolation_node(2), rng.random()] {
                let mut at_node = prefix;
                at_node.apply(node);
                for row in 0..half {
                    assert_eq!(source.interpolate(row, half, node), line(&dense, row, node));
                    assert_eq!(
                        selectors(BoundaryEvals::from_row_with_prefix(row, half, at_node)),
                        indicators.each_ref().map(|poly| line(poly, row, node))
                    );
                }
            }

            let challenge: EF = rng.random();
            source.fold(challenge);
            dense.fix_prefix_var_mut(challenge);
            prefix.apply(challenge);
            for poly in &mut indicators {
                poly.fix_prefix_var_mut(challenge);
            }
            // Binding a variable leaves a challenge-field copy of only the surviving half.
            let Source::Folded(folded) = &source else {
                panic!("a bound column lives in the challenge field");
            };
            assert_eq!(folded.as_slice(), dense.as_slice());
        }
    }

    /// Dense, Boolean-packed and periodic columns of every height up to two packed words.
    fn assert_sources_fold_like_dense<F, EF>(rng: &mut SmallRng)
    where
        F: Field,
        EF: Field + Algebra<F>,
        StandardUniform: Distribution<F> + Distribution<EF>,
    {
        for log_height in 0..=7 {
            let height = 1usize << log_height;

            // Two columns, so the one under test starts at a nonzero offset.
            let cells: Vec<F> = (0..2 * height).map(|_| rng.random()).collect();
            let dense = Table::new(RowMajorMatrix::new(cells.clone(), height));
            let source = Source::Committed(dense.column(1));
            assert_folds_like_dense::<F, EF>(source, &cells[height..], rng);

            // Three Boolean columns, 64 rows to a word, with bit zero holding the first row.
            let bits: Vec<[bool; 3]> = (0..height)
                .map(|_| [rng.random(), rng.random(), rng.random()])
                .collect();
            let mut words = vec![0u64; 3 * height.div_ceil(64)];
            for (row, row_bits) in bits.iter().enumerate() {
                for (column, &bit) in row_bits.iter().enumerate() {
                    words[(row / 64) * 3 + column] |= u64::from(bit) << (row % 64);
                }
            }
            let packed = Table::from_packed_bits(RowMajorMatrix::new(words, 3), log_height);
            let cells: Vec<F> = bits.iter().map(|row| F::from_bool(row[1])).collect();
            assert_folds_like_dense::<F, EF>(Source::Committed(packed.column(1)), &cells, rng);

            // Every period that divides the height, from a constant up to the whole column.
            for log_period in 0..=log_height {
                let period: Vec<F> = (0..1 << log_period).map(|_| rng.random()).collect();
                let cells: Vec<F> = (0..height).map(|row| period[row % period.len()]).collect();
                let source = Source::Periodic { period, height };
                assert_folds_like_dense::<F, EF>(source, &cells, rng);
            }
        }
    }

    #[test]
    fn sources_fold_like_dense_polynomials() {
        // The prover reads table columns and period vectors in place.
        // It keeps the selectors in closed form.
        // Any value differing from the dense challenge-field polynomials would change the proof.
        let mut rng = SmallRng::seed_from_u64(0xB05);
        assert_sources_fold_like_dense::<BinaryField128, BinaryField128>(&mut rng);
        assert_sources_fold_like_dense::<BabyBear, BinomialExtensionField<BabyBear, 4>>(&mut rng);
        // A representation backend keeps the same sources in another basis of the challenge field.
        assert_sources_fold_like_dense::<BinaryField128, Ghash128>(&mut rng);
    }
}
