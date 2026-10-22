//! What the multi-STARK verifier delegates to the arithmetic it runs on.
//!
//! [`Native`] answers with the native folders and commitment scheme.
//!
//! Any other arithmetic evaluates the AIR from one symbolic capture.

use alloc::vec::Vec;

use hashbrown::HashMap;
use p3_air::symbolic::{AirLayout, BaseEntry, BaseLeaf, SymbolicExprVisitor, SymbolicExpression};
use p3_air::{Air, BaseAir, BoundaryEnd, fold_symbolic_expression_cached};
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::MultilinearPcs;
use p3_field::{Dup, ExtensionField, Field, PrimeCharacteristicRing};
use p3_lookup::InteractionSymbolicBuilder;
use p3_multilinear_util::point::Point;
use p3_sumcheck::{OpeningEvals, OpeningProtocol, PrescribedPointPcs};
use p3_verifier_arith::{Arith, Lift, Native};

use crate::config::{Commitment, MultiStarkConfig, PcsError, PcsProof};
use crate::folder::{
    FolderEvaluations, InteractionMultilinearFolder, MultilinearFolder, VerifierAir, link_term,
};
use crate::lookup::AirLinkInstance;
use crate::selectors::{
    BoundaryEvals, PeriodicError, air_periodic_evals_at, periodic_num_variables,
};

/// The commitment scheme and transcript a multi-STARK verifier runs on, within an arithmetic.
pub trait MultiStarkArith<C: MultiStarkConfig>:
    Lift<Val = C::Val, Challenge = C::Challenge>
{
    /// Whether the values are native, which the binary bus and the indexed reduction require.
    const NATIVE: bool = false;

    /// A commitment within the arithmetic.
    type Commitment: Clone;

    /// The challenger the native transcript runs on.
    type Challenger: FieldChallenger<C::Val>
        + GrindingChallenger<Witness = C::Val>
        + CanSampleUniformBits<C::Val>
        + CanObserve<Commitment<C>>;

    /// Lift a commitment the prover sent.
    fn load_commitment(&mut self, pcs: &C::Pcs, commitment: &Commitment<C>) -> Self::Commitment;

    /// Lift a commitment the verifier holds, such as its preprocessed columns.
    fn load_key_commitment(&mut self, pcs: &C::Pcs, commitment: &Commitment<C>)
    -> Self::Commitment;

    /// Announce a commitment the native transcript absorbs next.
    fn will_observe_commitment(&mut self, pcs: &C::Pcs, commitment: &Self::Commitment);

    /// Replay the absorb the scheme performs when it commits.
    fn observe_commitment(
        &mut self,
        pcs: &C::Pcs,
        native: &Commitment<C>,
        commitment: &Self::Commitment,
        challenger: &mut Self::Challenger,
    );

    /// Verify an opening at prescribed points, returning the opened values.
    ///
    /// # Errors
    ///
    /// When the opening does not verify.
    fn verify_at(
        &mut self,
        pcs: &C::Pcs,
        commitment: &Self::Commitment,
        proof: &PcsProof<C>,
        protocol: &OpeningProtocol,
        points: &[Point<Self::EF>],
        challenger: &mut Self::Challenger,
    ) -> Result<Vec<OpeningEvals<Self::EF>>, PcsError<C>>;
}

impl<C> MultiStarkArith<C> for Native<C::Val, C::Challenge>
where
    C: MultiStarkConfig,
    C::Pcs: PrescribedPointPcs<C::Challenge, C::Challenger>,
    C::Challenger: FieldChallenger<C::Val>
        + GrindingChallenger<Witness = C::Val>
        + CanSampleUniformBits<C::Val>
        + CanObserve<Commitment<C>>,
    Commitment<C>: Clone,
{
    const NATIVE: bool = true;
    type Commitment = Commitment<C>;
    type Challenger = C::Challenger;

    #[inline]
    fn load_commitment(&mut self, _pcs: &C::Pcs, commitment: &Commitment<C>) -> Commitment<C> {
        commitment.clone()
    }

    #[inline]
    fn load_key_commitment(&mut self, _pcs: &C::Pcs, commitment: &Commitment<C>) -> Commitment<C> {
        commitment.clone()
    }

    #[inline]
    fn will_observe_commitment(&mut self, _pcs: &C::Pcs, _commitment: &Commitment<C>) {}

    #[inline]
    fn observe_commitment(
        &mut self,
        pcs: &C::Pcs,
        native: &Commitment<C>,
        _commitment: &Commitment<C>,
        challenger: &mut C::Challenger,
    ) {
        pcs.observe_commitment(native, challenger);
    }

    #[inline]
    fn verify_at(
        &mut self,
        pcs: &C::Pcs,
        commitment: &Commitment<C>,
        proof: &PcsProof<C>,
        protocol: &OpeningProtocol,
        points: &[Point<C::Challenge>],
        challenger: &mut C::Challenger,
    ) -> Result<Vec<OpeningEvals<C::Challenge>>, PcsError<C>> {
        pcs.verify_at(commitment, proof, protocol, points, challenger)
    }
}

/// One AIR's opened values at the bound point, as its constraints read them.
pub struct AirInputs<'a, AR: Arith + ?Sized> {
    /// Main columns at the point.
    pub(crate) main_local: &'a [AR::EF],
    /// Main columns at the successor, zero where the AIR does not read ahead.
    pub(crate) main_next: &'a [AR::EF],
    /// Preprocessed columns at the point.
    pub(crate) preprocessed_local: &'a [AR::EF],
    /// Preprocessed columns at the successor.
    pub(crate) preprocessed_next: &'a [AR::EF],
    /// Periodic columns at the point.
    pub(crate) periodic: &'a [AR::EF],
    /// Public values of the AIR.
    pub(crate) public_values: &'a [AR::F],
    /// Boundary selectors at the point.
    pub(crate) boundary: &'a BoundaryEvals<AR::EF>,
    /// The constraint batching challenge.
    pub(crate) alpha: &'a AR::EF,
    /// The AIR's lookup link, when it declares lookups.
    pub(crate) link: Option<&'a AirLinkInstance<AR::EF>>,
    /// The payload coefficients `theta * beta^k` the link reads.
    pub(crate) theta_beta_powers: &'a [AR::EF],
}

/// How an arithmetic evaluates one AIR at the bound point.
pub trait AirArith<A>: Arith {
    /// The AIR's batched constraints and lookup link at the point.
    fn eval_air(&mut self, air: &A, inputs: &AirInputs<'_, Self>) -> FolderEvaluations<Self::EF>;

    /// The AIR's periodic columns at the point.
    ///
    /// # Errors
    ///
    /// When the declaration does not fit a trace of `2^point.len()` rows.
    fn periodic_values(
        &mut self,
        air: &A,
        point: &[Self::EF],
    ) -> Result<Vec<Self::EF>, PeriodicError>;
}

impl<F, EF, A> AirArith<A> for Native<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
    A: VerifierAir<F, EF>,
{
    #[inline]
    fn eval_air(&mut self, air: &A, inputs: &AirInputs<'_, Self>) -> FolderEvaluations<EF> {
        let folder = MultilinearFolder::new(
            inputs.main_local,
            inputs.main_next,
            *inputs.boundary,
            inputs.public_values,
            *inputs.alpha,
        )
        .with_preprocessed(inputs.preprocessed_local, inputs.preprocessed_next)
        .with_periodic(inputs.periodic);
        match inputs.link {
            Some(link) => {
                InteractionMultilinearFolder::new(folder, link, inputs.theta_beta_powers, true)
                    .eval_air(air)
            }
            None => FolderEvaluations {
                constraints: folder.eval_air(air),
                interactions: EF::ZERO,
            },
        }
    }

    #[inline]
    fn periodic_values(&mut self, air: &A, point: &[EF]) -> Result<Vec<EF>, PeriodicError> {
        air_periodic_evals_at::<F, EF, _>(air, point)
    }
}

/// The circuit-agnostic [`AirArith::eval_air`]: one symbolic capture, evaluated on the inputs.
///
/// Constraints fold by Horner in `alpha`, public boundary pins last, as the native folder does.
///
/// # Panics
///
/// When the AIR asserts a challenge-field constraint, which no multi-STARK folder evaluates.
pub fn eval_air_symbolic<AR, A>(air: &A, inputs: &AirInputs<'_, AR>) -> FolderEvaluations<AR::EF>
where
    AR: Arith,
    A: BaseAir<AR::Val> + Air<InteractionSymbolicBuilder<AR::Val, AR::Challenge>>,
{
    let builder = InteractionSymbolicBuilder::<AR::Val, AR::Challenge>::from_air(
        air,
        AirLayout::from_air(air),
    );
    assert!(
        builder.extension_constraints().is_empty(),
        "multi-STARK AIRs assert base-field constraints only"
    );
    let mut eval = Evaluator {
        visitor: Leaves(inputs),
        cache: HashMap::new(),
        stack: Vec::new(),
    };

    let base = builder.base_constraints();
    let mut constraints = base.iter().fold(AR::EF::ZERO, |acc, constraint| {
        acc * inputs.alpha.dup() + eval.at(constraint)
    });
    for cell in air.public_boundary_io() {
        let selector = match cell.end {
            BoundaryEnd::First => &inputs.boundary.first,
            BoundaryEnd::Last => &inputs.boundary.last,
        };
        let public = AR::EF::from(inputs.public_values[cell.public_value].dup());
        constraints = constraints * inputs.alpha.dup()
            + selector.dup() * (inputs.main_local[cell.column].dup() - public);
    }

    let mut interactions = AR::EF::ZERO;
    if let Some(link) = inputs.link {
        // Every tuple, with its gate when it is an exclusive branch, its block weight and its bus offset.
        let local = builder
            .local_interactions()
            .iter()
            .zip(link.local_lookups())
            .flat_map(|(interaction, lookup)| {
                interaction.tuples.iter().zip(&lookup.block_weights).map(
                    |((fields, count), weight)| {
                        (fields, None, count.expr(), weight, &lookup.theta_bus_offset)
                    },
                )
            });
        let global = builder
            .global_interactions()
            .iter()
            .zip(link.global_lookups())
            .map(|(interaction, lookup)| {
                (
                    &interaction.fields,
                    None,
                    &interaction.count,
                    &lookup.block_weights[0],
                    &lookup.theta_bus_offset,
                )
            });
        let exclusive =
            builder
                .exclusive_interactions()
                .iter()
                .zip(link.exclusive_lookups())
                .flat_map(|(interaction, lookup)| {
                    interaction.branches.iter().zip(&lookup.block_weights).map(
                        |(branch, weight)| {
                            (
                                &branch.fields,
                                Some(&branch.flag),
                                &branch.count,
                                weight,
                                &lookup.theta_bus_offset,
                            )
                        },
                    )
                });
        for (fields, flag, count, weight, offset) in local.chain(global).chain(exclusive) {
            let mut multiplicity = eval.at(count);
            if let Some(flag) = flag {
                multiplicity *= eval.at(flag);
            }
            let fields = fields
                .iter()
                .map(|field| eval.at(field))
                .collect::<Vec<_>>();
            interactions += link_term(
                weight.dup(),
                multiplicity,
                offset.dup(),
                inputs.theta_beta_powers,
                fields,
            );
        }
    }

    FolderEvaluations {
        constraints,
        interactions,
    }
}

/// The circuit-agnostic [`AirArith::periodic_values`]: each column's multilinear extension at the point.
///
/// # Errors
///
/// When the declaration does not fit a trace of `2^point.len()` rows.
pub fn periodic_values_symbolic<AR, A>(
    arith: &mut AR,
    air: &A,
    point: &[AR::EF],
) -> Result<Vec<AR::EF>, PeriodicError>
where
    AR: Arith,
    A: BaseAir<AR::Val>,
{
    let columns = air.periodic_columns();
    let periods = columns.iter().map(Vec::len).collect::<Vec<_>>();
    let num_variables = periodic_num_variables(air.num_periodic_columns(), &periods, point.len())?;
    Ok(columns
        .iter()
        .zip(num_variables)
        .map(|(values, j)| {
            let values = values.iter().map(|&v| AR::F::from(v)).collect::<Vec<_>>();
            let suffix = Point::new(point[point.len() - j..].to_vec());
            arith.eval_multilinear(&values, &suffix)
        })
        .collect())
}

/// Evaluates symbolic expressions on the inputs, sharing one cache across every root.
struct Evaluator<'v, 'a, 'e, AR: Arith> {
    /// The leaf interpretation.
    visitor: Leaves<'v, 'a, AR>,
    /// Values of the nodes seen so far, by address.
    cache: HashMap<*const SymbolicExpression<AR::Val>, AR::EF>,
    /// Traversal stack, kept across roots.
    stack: Vec<(&'e SymbolicExpression<AR::Val>, bool)>,
}

impl<'e, AR: Arith> Evaluator<'_, '_, 'e, AR> {
    /// The value of one expression.
    fn at(&mut self, root: &'e SymbolicExpression<AR::Val>) -> AR::EF {
        fold_symbolic_expression_cached(root, &mut self.visitor, &mut self.cache, &mut self.stack)
    }
}

/// Maps each symbolic leaf to its input value.
struct Leaves<'v, 'a, AR: Arith>(&'v AirInputs<'a, AR>);

impl<'e, AR: Arith> SymbolicExprVisitor<'e, BaseLeaf<AR::Val>> for Leaves<'_, '_, AR> {
    type Output = AR::EF;

    fn leaf(&mut self, leaf: &'e BaseLeaf<AR::Val>) -> AR::EF {
        let inputs = self.0;
        match leaf {
            BaseLeaf::Variable(v) => match v.entry {
                BaseEntry::Main { offset: 0 } => inputs.main_local[v.index].dup(),
                BaseEntry::Main { .. } => inputs.main_next[v.index].dup(),
                BaseEntry::Preprocessed { offset: 0 } => inputs.preprocessed_local[v.index].dup(),
                BaseEntry::Preprocessed { .. } => inputs.preprocessed_next[v.index].dup(),
                BaseEntry::Periodic => inputs.periodic[v.index].dup(),
                BaseEntry::Public => AR::EF::from(inputs.public_values[v.index].dup()),
            },
            BaseLeaf::IsFirstRow => inputs.boundary.first.dup(),
            BaseLeaf::IsLastRow => inputs.boundary.last.dup(),
            BaseLeaf::IsTransition => inputs.boundary.transition.dup(),
            BaseLeaf::Constant(c) => AR::EF::from(AR::F::from(*c)),
        }
    }

    fn add(&mut self, x: &AR::EF, y: &AR::EF) -> AR::EF {
        x.dup() + y.dup()
    }

    fn sub(&mut self, x: &AR::EF, y: &AR::EF) -> AR::EF {
        x.dup() - y.dup()
    }

    fn neg(&mut self, x: &AR::EF) -> AR::EF {
        -x.dup()
    }

    fn mul(&mut self, x: &AR::EF, y: &AR::EF) -> AR::EF {
        x.dup() * y.dup()
    }
}
