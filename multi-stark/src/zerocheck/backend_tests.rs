//! Every zerocheck backend must emit the generic backend's transcript, byte for byte.
//!
//! The fixtures run over `BinaryField128`, mostly with bit-valued traces, where the subfield
//! backend evaluates the first round of a stage inside `GF(4)`. Each way a stage can fail to fit
//! gets its own fixture. The representation backend runs every later round of every fixture in
//! the polynomial basis.

use alloc::borrow::Cow;
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use p3_air::{Air, AirBuilder, BaseAir, BoundaryEnd, BoundaryPublic, WindowAccess};
use p3_binary_field::{BinaryChallenger, BinaryField2, BinaryField128, Ghash128, TowerLevel};
use p3_challenger::{
    CanObserve, CanSample, CanSampleBits, FieldChallenger, GrindingChallenger, HashChallenger,
};
use p3_field::{HasSubfield, PrimeCharacteristicRing};
use p3_keccak::Keccak256Hash;
use p3_lookup::{Count, InteractionBuilder};
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::point::Point;
use p3_sumcheck::layout::{ColumnView, Table};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::AirZerocheck;
use crate::backend::{GenericBackend, ReprBackend, SubfieldBackend, ZerocheckBackend};
use crate::config::DEFAULT_SLICED_ROUNDS;
use crate::lookup::{
    ActiveLookupRuntime, AirLinkClaim, AirLinkInstance, AirLinkLookup, LookupRuntime,
};
use crate::rounds::sliced::MAX_SLICED_ROUNDS;
use crate::sliced::SLICED_LANES;

/// The trace and challenge field of every fixture.
pub(crate) type Tower = BinaryField128;

/// The subfield the subfield backend evaluates in.
pub(crate) type Gf4 = BinaryField2;

/// The representation the representation backend runs later rounds in.
type PolyBasis = Ghash128;

type Binary = BinaryChallenger<Tower, HashChallenger<u8, Keccak256Hash, 32>>;

/// A transcript whose grinding returns the first valid witness in a fixed order.
///
/// A parallel search returns whichever valid witness a worker finds first.
/// Two runs could then bind different witnesses, and their transcripts differ for that reason
/// alone.
#[derive(Clone)]
struct Challenger(Binary);

fn challenger() -> Challenger {
    Challenger(Binary::from_hasher(
        b"p3-multi-stark-backend-agreement".to_vec(),
        Keccak256Hash,
    ))
}

impl CanObserve<Tower> for Challenger {
    fn observe(&mut self, value: Tower) {
        self.0.observe(value);
    }
}

impl CanSample<Tower> for Challenger {
    fn sample(&mut self) -> Tower {
        self.0.sample()
    }
}

impl CanSampleBits<usize> for Challenger {
    fn sample_bits(&mut self, bits: usize) -> usize {
        self.0.sample_bits(bits)
    }
}

impl FieldChallenger<Tower> for Challenger {}

impl GrindingChallenger for Challenger {
    type Witness = Tower;

    fn grind(&mut self, bits: usize) -> Tower {
        let witness = (0..)
            .map(Tower::from_repr)
            .find(|&witness| self.0.clone().check_witness(bits, witness))
            .expect("some witness passes");
        assert!(self.check_witness(bits, witness));
        witness
    }
}

/// The `GF(4)` element inside the tower whose bit pattern is the low two bits of `bits`.
pub(crate) fn gf4(bits: usize) -> Tower {
    Tower::from_repr((bits & 3) as u128)
}

/// The first bit pattern above `GF(4)`, the smallest element the subfield cannot hold.
pub(crate) fn outside() -> Tower {
    Tower::from_repr(4)
}

/// The one cell the gate AIR binds to a public value.
const GATE_CELLS: [BoundaryPublic; 1] = [BoundaryPublic::new(0, BoundaryEnd::First, 0)];

/// Public boundary pin used by the eligible quadratic-input fixture.
const QUADRATIC_PUBLIC: [BoundaryPublic; 1] = [BoundaryPublic::new(0, BoundaryEnd::First, 0)];

/// Period of the gate AIR's periodic column.
const GATE_PERIOD: usize = 4;

/// Small AIRs over the tower, one per shape the subfield backend treats differently.
pub(crate) enum FixtureAir {
    /// Bit-valued degree-three AIR reading every column group, a public value, and a constant.
    ///
    /// Main columns `a, b, c, e`, preprocessed column `q`, periodic column
    /// `p = [0, 1, X_0, X_0 + 1]`:
    ///
    /// ```text
    ///     always     : scale * (a^2 - a) = 0
    ///     transition : next.c = a * q
    ///     transition : next.q = q + 1
    ///     always     : e = p * a * b
    ///     last row   : c = public[1]
    ///     pin        : first row, a = public[0]
    /// ```
    ///
    /// Booleanity holds whatever `scale` is, so an honest trace satisfies every scale.
    Gate {
        /// Constant multiplying the booleanity constraint.
        scale: Tower,
    },
    /// Bit-valued degree-two AIR reading no successor column.
    ///
    /// ```text
    ///     always : s = a * b
    /// ```
    Pair,
    /// Bit-valued AIR whose transition has degree four.
    ///
    /// ```text
    ///     always     : a, b, c are bits
    ///     transition : next.d = a * b * c
    /// ```
    ///
    /// Its first round reaches node four, one step past the nodes inside `GF(4)`.
    Quartic,
    /// Bit-valued degree-three AIR scaling booleanity by a periodic column of period two.
    ///
    /// ```text
    ///     always : p * (a^2 - a) = 0
    /// ```
    ///
    /// Booleanity holds whatever `p` is, so the trace stays bit-valued for every period vector.
    Periodic {
        /// The period vector of `p`.
        period: [Tower; 2],
    },
    /// Bit-valued AIR declaring one local lookup beside one ordinary constraint.
    ///
    /// ```text
    ///     always : a is a bit
    ///     lookup : a requested once, b provided once
    /// ```
    Link,
    /// Degree-three nonlinear recurrence over full-width cells.
    ///
    /// ```text
    ///     transition : next.a = b
    ///     transition : next.b = a * b + a
    /// ```
    ///
    /// Its cells are arbitrary tower elements, so a stage holding it never fits `GF(4)`.
    Recurrence,
    /// Bit-valued degree-one AIR over two equal columns.
    ///
    /// ```text
    ///     always : scale * (a - b) = 0
    /// ```
    ///
    /// Its first round evaluates no node, so a constant outside `GF(4)` shows up only later.
    Linear {
        /// Constant multiplying the equality.
        scale: Tower,
    },
    /// Eligible quadratic AIR using fixed, periodic, selector, and public inputs.
    QuadraticInputs,
    /// The same eligible shape with one periodic value outside GF(4).
    QuadraticInputsOutsidePeriodic,
    /// Degree-two successor-reading AIR used to isolate successor rejection.
    QuadraticSuccessor,
    /// Degree-two AIR whose only successor is a fixed Boolean column.
    QuadraticPreprocessedSuccessor,
    /// AIR with a main column but no constraints, whose native degree is zero.
    Empty,
}

impl BaseAir<Tower> for FixtureAir {
    fn width(&self) -> usize {
        match self {
            Self::Gate { .. } | Self::Quartic => 4,
            Self::Pair => 3,
            Self::Link | Self::Recurrence | Self::Linear { .. } => 2,
            Self::QuadraticInputs
            | Self::QuadraticInputsOutsidePeriodic
            | Self::QuadraticSuccessor
            | Self::QuadraticPreprocessedSuccessor
            | Self::Empty => 1,
            Self::Periodic { .. } => 1,
        }
    }

    fn preprocessed_width(&self) -> usize {
        match self {
            Self::Gate { .. }
            | Self::QuadraticInputs
            | Self::QuadraticInputsOutsidePeriodic
            | Self::QuadraticPreprocessedSuccessor => 1,
            _ => 0,
        }
    }

    fn num_public_values(&self) -> usize {
        match self {
            Self::Gate { .. } => 2,
            Self::QuadraticInputs | Self::QuadraticInputsOutsidePeriodic => 1,
            _ => 0,
        }
    }

    fn num_periodic_columns(&self) -> usize {
        match self {
            Self::Gate { .. }
            | Self::Periodic { .. }
            | Self::QuadraticInputs
            | Self::QuadraticInputsOutsidePeriodic => 1,
            _ => 0,
        }
    }

    fn periodic_columns(&self) -> Cow<'_, [Vec<Tower>]> {
        match self {
            Self::Gate { .. } => Cow::Owned(vec![(0..GATE_PERIOD).map(gf4).collect()]),
            Self::Periodic { period } => Cow::Owned(vec![period.to_vec()]),
            Self::QuadraticInputs => Cow::Owned(vec![vec![gf4(0), gf4(1), gf4(2), gf4(3)]]),
            Self::QuadraticInputsOutsidePeriodic => {
                Cow::Owned(vec![vec![outside(), gf4(1), gf4(2), gf4(3)]])
            }
            _ => Cow::Owned(vec![]),
        }
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        match self {
            Self::Gate { .. } => vec![2],
            Self::Quartic => vec![3],
            Self::Recurrence => vec![0, 1],
            Self::Pair
            | Self::Link
            | Self::Periodic { .. }
            | Self::Linear { .. }
            | Self::QuadraticInputs
            | Self::QuadraticInputsOutsidePeriodic
            | Self::Empty => vec![],
            Self::QuadraticSuccessor => vec![0],
            Self::QuadraticPreprocessedSuccessor => vec![],
        }
    }

    fn preprocessed_next_row_columns(&self) -> Vec<usize> {
        match self {
            Self::Gate { .. } => vec![0],
            Self::QuadraticPreprocessedSuccessor => vec![0],
            _ => vec![],
        }
    }

    fn public_boundary_io(&self) -> &[BoundaryPublic] {
        match self {
            Self::Gate { .. } => &GATE_CELLS,
            Self::QuadraticInputs | Self::QuadraticInputsOutsidePeriodic => &QUADRATIC_PUBLIC,
            _ => &[],
        }
    }
}

impl<AB: AirBuilder<F = Tower> + InteractionBuilder> Air<AB> for FixtureAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.current_slice(), main.next_slice());
        match self {
            Self::Gate { scale } => {
                let (a, b, c, e) = (local[0], local[1], local[2], local[3]);
                let preprocessed = builder.preprocessed();
                let (q, next_q) = (
                    preprocessed.current_slice()[0],
                    preprocessed.next_slice()[0],
                );
                let p: AB::Expr = builder.periodic_values()[0].into();
                let last_c = builder.public_values()[1];

                let a_expr: AB::Expr = a.into();
                builder.assert_zero(a_expr.bool_check() * *scale);
                builder.when_transition().assert_eq(next[2], a * q);
                builder
                    .when_transition()
                    .assert_eq(next_q, q + AB::Expr::ONE);
                builder.assert_eq(e, p * a * b);
                builder.when_last_row().assert_eq(c, last_c);
            }
            Self::Pair => {
                builder.assert_eq(local[2], local[0] * local[1]);
            }
            Self::Quartic => {
                let (a, b, c) = (local[0], local[1], local[2]);
                builder.assert_bool(a);
                builder.assert_bool(b);
                builder.assert_bool(c);
                builder.when_transition().assert_eq(next[3], a * b * c);
            }
            Self::Periodic { .. } => {
                let p: AB::Expr = builder.periodic_values()[0].into();
                let a: AB::Expr = local[0].into();
                builder.assert_zero(p * a.bool_check());
            }
            Self::Link => {
                let (a, b) = (local[0], local[1]);
                builder.assert_bool(a);
                builder.push_local_interaction([
                    (vec![a.into()], Count::bounded(AB::Expr::ONE, 1)),
                    (vec![b.into()], Count::provided(AB::Expr::ONE)),
                ]);
            }
            Self::Recurrence => {
                let (a, b) = (local[0], local[1]);
                builder.when_transition().assert_eq(next[0], b);
                builder.when_transition().assert_eq(next[1], a * b + a);
            }
            Self::Linear { scale } => {
                builder.assert_zero((local[0] - local[1]) * *scale);
            }
            Self::QuadraticInputs | Self::QuadraticInputsOutsidePeriodic => {
                let fixed = builder.preprocessed().current_slice()[0];
                let periodic: AB::Expr = builder.periodic_values()[0].into();
                let value: AB::Expr = local[0].into();
                let public = builder.public_values()[0];
                builder.assert_zero(value * periodic - fixed);
                builder.when_first_row().assert_eq(local[0], public);
                builder.when_last_row().assert_eq(local[0], public);
            }
            Self::QuadraticSuccessor => {
                let value: AB::Expr = local[0].into();
                builder.assert_zero(value.bool_check());
                builder.when_transition().assert_eq(next[0], value);
            }
            Self::QuadraticPreprocessedSuccessor => {
                let value: AB::Expr = local[0].into();
                let (fixed_local, fixed_next) = {
                    let fixed = builder.preprocessed();
                    (fixed.current_slice()[0], fixed.next_slice()[0])
                };
                builder.assert_zero(value.bool_check());
                builder.when_transition().assert_eq(fixed_next, fixed_local);
            }
            Self::Empty => {}
        }
    }
}

/// One AIR of a batch with its trace, in row-major form so a test can overwrite a cell.
pub(crate) struct Instance {
    pub(crate) air: FixtureAir,
    /// Main trace, one row per trace row.
    pub(crate) main: RowMajorMatrix<Tower>,
    /// Preprocessed trace, present exactly when the AIR declares preprocessed columns.
    pub(crate) preprocessed: Option<RowMajorMatrix<Tower>>,
    pub(crate) public_values: Vec<Tower>,
}

impl Instance {
    /// An instance of `air` over `height` rows that satisfies every constraint.
    pub(crate) fn honest(air: FixtureAir, height: usize, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut bit = || Tower::from_bool(rng.random());
        let (main, preprocessed, public_values) = match air {
            FixtureAir::Gate { .. } => {
                let q = (0..height)
                    .map(|row| Tower::from_bool(row % 2 == 1))
                    .collect::<Vec<_>>();
                let mut values = Vec::with_capacity(4 * height);
                let mut c = bit();
                for (row, &q) in q.iter().enumerate() {
                    let (a, b) = (bit(), bit());
                    values.extend([a, b, c, gf4(row % GATE_PERIOD) * a * b]);
                    c = a * q;
                }
                let public_values = vec![values[0], values[4 * height - 2]];
                (
                    RowMajorMatrix::new(values, 4),
                    Some(RowMajorMatrix::new(q, 1)),
                    public_values,
                )
            }
            FixtureAir::Pair => {
                let values = (0..height)
                    .flat_map(|_| {
                        let (a, b) = (bit(), bit());
                        [a, b, a * b]
                    })
                    .collect();
                (RowMajorMatrix::new(values, 3), None, vec![])
            }
            FixtureAir::Quartic => {
                let mut values = Vec::with_capacity(4 * height);
                let mut d = bit();
                for _ in 0..height {
                    let (a, b, c) = (bit(), bit(), bit());
                    values.extend([a, b, c, d]);
                    d = a * b * c;
                }
                (RowMajorMatrix::new(values, 4), None, vec![])
            }
            FixtureAir::Link => {
                let values = (0..2 * height).map(|_| bit()).collect();
                (RowMajorMatrix::new(values, 2), None, vec![])
            }
            FixtureAir::Periodic { .. } => {
                let values = (0..height).map(|_| bit()).collect();
                (RowMajorMatrix::new(values, 1), None, vec![])
            }
            FixtureAir::Linear { .. } => {
                let values = (0..height)
                    .flat_map(|_| {
                        let a = bit();
                        [a, a]
                    })
                    .collect();
                (RowMajorMatrix::new(values, 2), None, vec![])
            }
            ref
            air @ (FixtureAir::QuadraticInputs | FixtureAir::QuadraticInputsOutsidePeriodic) => {
                let mut values = (0..height)
                    .map(|row| gf4((row % 3) + 1))
                    .collect::<Vec<_>>();
                values[height - 1] = values[0];
                let periodic = match air {
                    FixtureAir::QuadraticInputs => [gf4(0), gf4(1), gf4(2), gf4(3)],
                    FixtureAir::QuadraticInputsOutsidePeriodic => {
                        [outside(), gf4(1), gf4(2), gf4(3)]
                    }
                    _ => unreachable!(),
                };
                let fixed = values
                    .iter()
                    .enumerate()
                    .map(|(row, &value)| value * periodic[row % 4])
                    .collect::<Vec<_>>();
                (
                    RowMajorMatrix::new(values, 1),
                    Some(RowMajorMatrix::new(fixed, 1)),
                    vec![gf4(1)],
                )
            }
            FixtureAir::QuadraticSuccessor => {
                let value = Tower::ONE;
                (RowMajorMatrix::new(vec![value; height], 1), None, vec![])
            }
            FixtureAir::QuadraticPreprocessedSuccessor => (
                RowMajorMatrix::new(
                    (0..height)
                        .map(|row| Tower::from_bool(row % 2 == 0))
                        .collect(),
                    1,
                ),
                Some(RowMajorMatrix::new(vec![Tower::ONE; height], 1)),
                vec![],
            ),
            FixtureAir::Empty => (
                RowMajorMatrix::new(vec![Tower::ZERO; height], 1),
                None,
                vec![],
            ),
            FixtureAir::Recurrence => {
                let (mut a, mut b): (Tower, Tower) = (rng.random(), rng.random());
                let mut values = Vec::with_capacity(2 * height);
                for _ in 0..height {
                    values.extend([a, b]);
                    (a, b) = (b, a * b + a);
                }
                (RowMajorMatrix::new(values, 2), None, vec![])
            }
        };
        Self {
            air,
            main,
            preprocessed,
            public_values,
        }
    }

    /// The main trace laid out for the sumcheck, one polynomial per column.
    pub(crate) fn main_table(&self) -> Table<Tower> {
        Table::new(self.main.clone().transpose())
    }

    /// The preprocessed trace laid out for the sumcheck, if the AIR declares one.
    pub(crate) fn preprocessed_table(&self) -> Option<Table<Tower>> {
        self.preprocessed
            .as_ref()
            .map(|trace| Table::new(trace.clone().transpose()))
    }
}

/// Convert a Boolean dense table to the storage representation used by the packed prover path.
///
/// The fixture traces are deliberately Boolean, so this test helper exercises the packed table
/// contract without introducing a second trace generator.
fn packed_table(table: &Table<Tower>) -> Table<Tower> {
    let height = 1usize << table.num_variables();
    let words = (0..height.div_ceil(64))
        .flat_map(|block| {
            (0..table.num_polys()).map(move |column| {
                (0..64).fold(0_u64, |word, lane| {
                    let row = block * 64 + lane;
                    if row < height && table.column(column).value(row) == Tower::ONE {
                        word | (1_u64 << lane)
                    } else {
                        word
                    }
                })
            })
        })
        .collect();
    Table::from_packed_bits(
        RowMajorMatrix::new(words, table.num_polys()),
        table.num_variables(),
    )
}

/// Lookup-reduction output for one lookup AIR, with random coefficients and claim.
///
/// The zerocheck prover never checks it against the trace.
/// Backends can only be compared on it, not verified.
pub(crate) fn link_runtime(
    air_index: usize,
    num_variables: usize,
    seed: u64,
) -> LookupRuntime<Tower> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let claim = rng.random();
    let link = AirLinkInstance {
        num_local_lookups: 1,
        lookups: vec![AirLinkLookup {
            theta_bus_offset: rng.random(),
            block_weights: vec![rng.random(), rng.random()],
        }],
    };
    LookupRuntime::Active(ActiveLookupRuntime {
        claims_by_air: BTreeMap::from([(air_index, claim)]),
        air_link: AirLinkClaim {
            point: Point::rand(&mut rng, num_variables),
            claimed_sum: claim,
            theta_beta_powers: vec![rng.random()],
            links_by_air: BTreeMap::from([(air_index, link)]),
        },
    })
}

/// Prove the batch through backend `B` and return its transcript.
///
/// The transcript is the serialized sumcheck proof, openings, and point, followed by the next
/// challenge the challenger draws.
fn transcript<B>(
    instances: &[Instance],
    lookup: LookupRuntime<Tower>,
    pow_bits: usize,
    packed: bool,
) -> (Vec<u8>, Tower)
where
    B: ZerocheckBackend<Tower, Tower, FixtureAir>,
{
    transcript_with_storage::<B>(
        DEFAULT_SLICED_ROUNDS,
        instances,
        lookup,
        pow_bits,
        |_, _| packed,
    )
}

fn transcript_with_storage<B>(
    sliced_rounds: usize,
    instances: &[Instance],
    lookup: LookupRuntime<Tower>,
    pow_bits: usize,
    is_packed: impl Fn(usize, &Instance) -> bool,
) -> (Vec<u8>, Tower)
where
    B: ZerocheckBackend<Tower, Tower, FixtureAir>,
{
    let airs = instances
        .iter()
        .map(|instance| &instance.air)
        .collect::<Vec<_>>();
    let dense_main = instances
        .iter()
        .map(Instance::main_table)
        .collect::<Vec<_>>();
    let dense_preprocessed = instances
        .iter()
        .map(Instance::preprocessed_table)
        .collect::<Vec<_>>();
    let needs_packed = instances
        .iter()
        .enumerate()
        .any(|(index, instance)| is_packed(index, instance));
    let packed_main = needs_packed.then(|| dense_main.iter().map(packed_table).collect::<Vec<_>>());
    let packed_preprocessed = needs_packed.then(|| {
        dense_preprocessed
            .iter()
            .map(|table| table.as_ref().map(packed_table))
            .collect::<Vec<_>>()
    });
    let main = instances
        .iter()
        .enumerate()
        .map(|(index, instance)| {
            if is_packed(index, instance) {
                &packed_main.as_ref().expect("packed fixture was requested")[index]
            } else {
                &dense_main[index]
            }
        })
        .collect::<Vec<_>>();
    let preprocessed = instances
        .iter()
        .enumerate()
        .map(|(index, instance)| {
            if is_packed(index, instance) {
                packed_preprocessed
                    .as_ref()
                    .expect("packed fixture was requested")[index]
                    .as_ref()
            } else {
                dense_preprocessed[index].as_ref()
            }
        })
        .collect::<Vec<_>>();
    let public_values = instances
        .iter()
        .map(|instance| instance.public_values.as_slice())
        .collect::<Vec<_>>();

    let mut challenger = challenger();
    let (proof, point) = AirZerocheck::new(&airs, pow_bits)
        .prove_with_lookup::<Tower, Tower, B, _>(
            &preprocessed,
            &main,
            &public_values,
            lookup,
            sliced_rounds,
            &mut challenger,
        );
    let bytes = postcard::to_allocvec(&(
        &proof.sumcheck,
        &proof.local,
        &proof.next,
        &proof.preprocessed_local,
        &proof.preprocessed_next,
        point.as_slice(),
    ))
    .expect("postcard serialization must not fail");
    (bytes, CanSample::<Tower>::sample(&mut challenger))
}

/// Require every other backend to emit the generic backend's transcript on this batch.
fn assert_backends_agree(
    instances: &[Instance],
    lookup: impl Fn() -> LookupRuntime<Tower>,
    pow_bits: usize,
) {
    let generic = transcript::<GenericBackend>(instances, lookup(), pow_bits, false);
    let subfield = transcript::<SubfieldBackend<Gf4>>(instances, lookup(), pow_bits, false);
    assert_eq!(subfield, generic, "subfield backend");
    let repr = transcript::<ReprBackend<Gf4, PolyBasis>>(instances, lookup(), pow_bits, false);
    assert_eq!(repr, generic, "representation backend");
    let late =
        transcript::<ReprBackend<Gf4, PolyBasis, true>>(instances, lookup(), pow_bits, false);
    assert_eq!(late, generic, "late representation backend");
}

fn assert_packed_matches_dense(
    instances: &[Instance],
    lookup: impl Fn() -> LookupRuntime<Tower>,
    pow_bits: usize,
) {
    for instance in instances {
        let dense = instance.main_table();
        let packed = packed_table(&dense).into_dense();
        assert_eq!(
            dense
                .columns()
                .flat_map(ColumnView::values)
                .collect::<Vec<_>>(),
            packed
                .columns()
                .flat_map(ColumnView::values)
                .collect::<Vec<_>>(),
            "packed source cells"
        );
        if let Some(dense) = instance.preprocessed_table() {
            let packed = packed_table(&dense).into_dense();
            assert_eq!(
                dense
                    .columns()
                    .flat_map(ColumnView::values)
                    .collect::<Vec<_>>(),
                packed
                    .columns()
                    .flat_map(ColumnView::values)
                    .collect::<Vec<_>>(),
                "packed preprocessed cells"
            );
        }
    }
    let dense = transcript::<GenericBackend>(instances, lookup(), pow_bits, false);
    for (name, packed) in [
        (
            "generic",
            transcript::<GenericBackend>(instances, lookup(), pow_bits, true),
        ),
        (
            "subfield",
            transcript::<SubfieldBackend<Gf4>>(instances, lookup(), pow_bits, true),
        ),
        (
            "representation",
            transcript::<ReprBackend<Gf4, PolyBasis>>(instances, lookup(), pow_bits, true),
        ),
        (
            "late representation",
            transcript::<ReprBackend<Gf4, PolyBasis, true>>(instances, lookup(), pow_bits, true),
        ),
    ] {
        assert_eq!(packed, dense, "packed {name} backend");
    }

    if instances.len() > 1 {
        for (name, mixed) in [
            (
                "generic",
                transcript_with_storage::<GenericBackend>(
                    DEFAULT_SLICED_ROUNDS,
                    instances,
                    lookup(),
                    pow_bits,
                    |index, _| index % 2 == 0,
                ),
            ),
            (
                "subfield",
                transcript_with_storage::<SubfieldBackend<Gf4>>(
                    DEFAULT_SLICED_ROUNDS,
                    instances,
                    lookup(),
                    pow_bits,
                    |index, _| index % 2 == 0,
                ),
            ),
            (
                "representation",
                transcript_with_storage::<ReprBackend<Gf4, PolyBasis>>(
                    DEFAULT_SLICED_ROUNDS,
                    instances,
                    lookup(),
                    pow_bits,
                    |index, _| index % 2 == 0,
                ),
            ),
            (
                "late representation",
                transcript_with_storage::<ReprBackend<Gf4, PolyBasis, true>>(
                    DEFAULT_SLICED_ROUNDS,
                    instances,
                    lookup(),
                    pow_bits,
                    |index, _| index % 2 == 0,
                ),
            ),
        ] {
            assert_eq!(mixed, dense, "mixed dense/packed {name} backend");
        }
    }
}

#[test]
fn honest_gate_fixture_verifies() {
    // The transcripts below are compared on real round polynomials, not on a rejected statement.
    let instance = Instance::honest(FixtureAir::Gate { scale: Tower::ONE }, 32, 1);
    let airs = [&instance.air];
    let zerocheck = AirZerocheck::new(&airs, 0);
    let (main, preprocessed) = (instance.main_table(), instance.preprocessed_table());
    let (proof, point) = zerocheck.prove::<Tower, Tower, _>(
        &[preprocessed.as_ref()],
        &[&main],
        &[&instance.public_values],
        &mut challenger(),
    );
    let verified = zerocheck
        .verify::<Tower, Tower, _>(&proof, &[5], &[&instance.public_values], &mut challenger())
        .expect("honest gate proof must verify");
    assert_eq!(verified, point);
}

#[test]
fn backends_agree_on_a_bit_valued_degree_three_air() {
    for height in [4, 32] {
        let instances = [Instance::honest(
            FixtureAir::Gate { scale: Tower::ONE },
            height,
            1,
        )];
        assert_backends_agree(&instances, || LookupRuntime::Inactive, 0);
    }
}

#[test]
fn packed_backends_match_dense_across_round_boundaries_and_fallback() {
    for height in [32, 64, 128, 256, 512] {
        // Quartic exercises current/next reads while keeping every source cell Boolean.
        let successor = [
            Instance::honest(FixtureAir::Quartic, height, height as u64),
            Instance::honest(FixtureAir::Pair, height, height as u64 + 2),
        ];
        assert_packed_matches_dense(&successor, || LookupRuntime::Inactive, 0);

        // An outside constant keeps the source Boolean while forcing the generic fallback path.
        let fallback = [Instance::honest(
            FixtureAir::Linear { scale: outside() },
            height,
            height as u64 + 1,
        )];
        assert_packed_matches_dense(&fallback, || LookupRuntime::Inactive, 0);
    }

    let mixed_heights = [
        Instance::honest(FixtureAir::Quartic, 32, 0xB601),
        Instance::honest(FixtureAir::Pair, 64, 0xB602),
        Instance::honest(FixtureAir::Linear { scale: outside() }, 128, 0xB603),
    ];
    assert_packed_matches_dense(&mixed_heights, || LookupRuntime::Inactive, 0);
}

#[test]
fn packed_successor_stages_match_dense_on_the_sliced_kernel() {
    // A degree-two stage reading a successor column reaches the sliced kernel at these heights,
    // so its packed source must yield the same successor planes as its dense one.
    // Random bits make the successor column vary from row to row, and a transcript needs no
    // valid witness.
    for height in [128, 256, 1 << 10] {
        let mut instance = Instance::honest(
            FixtureAir::QuadraticSuccessor,
            height,
            0xB604 + height as u64,
        );
        let mut rng = SmallRng::seed_from_u64(0xB605 + height as u64);
        for value in &mut instance.main.values {
            *value = Tower::from_bool(rng.random());
        }
        assert_packed_matches_dense(&[instance], || LookupRuntime::Inactive, 0);
    }
}

#[test]
fn backends_agree_with_grinding() {
    let instances = [Instance::honest(FixtureAir::Gate { scale: gf4(2) }, 16, 2)];
    assert_backends_agree(&instances, || LookupRuntime::Inactive, 2);
}

#[test]
fn backends_agree_across_two_stages() {
    // Fixture state:
    //
    //     stage 64 rows : gate (degree 3) and pair (degree 2), activating in round 0
    //     stage  8 rows : gate, activating three rounds later
    let instances = [
        Instance::honest(FixtureAir::Gate { scale: Tower::ONE }, 64, 3),
        Instance::honest(FixtureAir::Pair, 64, 4),
        Instance::honest(FixtureAir::Gate { scale: gf4(3) }, 8, 5),
    ];
    assert_backends_agree(&instances, || LookupRuntime::Inactive, 0);
}

#[test]
fn backends_agree_when_a_lookup_stage_falls_back() {
    // Fixture state:
    //
    //     stage 32 rows : gate, which fits GF(4)
    //     stage 16 rows : pair and a lookup AIR, which falls back as a whole
    let instances = [
        Instance::honest(FixtureAir::Gate { scale: Tower::ONE }, 32, 6),
        Instance::honest(FixtureAir::Pair, 16, 7),
        Instance::honest(FixtureAir::Link, 16, 8),
    ];
    assert_backends_agree(&instances, || link_runtime(2, 5, 9), 0);
}

#[test]
fn backends_agree_when_a_cell_lies_outside_the_subfield() {
    let mut instance = Instance::honest(FixtureAir::Gate { scale: Tower::ONE }, 16, 10);
    // Column e of row 5.
    instance.main.values[4 * 5 + 3] = outside();
    assert_backends_agree(&[instance], || LookupRuntime::Inactive, 0);
}

#[test]
fn backends_agree_when_a_preprocessed_cell_lies_outside_the_subfield() {
    let mut instance = Instance::honest(FixtureAir::Gate { scale: Tower::ONE }, 16, 14);
    instance
        .preprocessed
        .as_mut()
        .expect("the gate AIR declares a preprocessed column")
        .values[5] = outside();
    assert_backends_agree(&[instance], || LookupRuntime::Inactive, 0);
}

#[test]
fn backends_agree_when_a_periodic_value_lies_outside_the_subfield() {
    let instances = [Instance::honest(
        FixtureAir::Periodic {
            period: [gf4(2), outside()],
        },
        16,
        15,
    )];
    assert_backends_agree(&instances, || LookupRuntime::Inactive, 0);
}

#[test]
fn backends_agree_when_an_air_constant_lies_outside_the_subfield() {
    let instances = [Instance::honest(
        FixtureAir::Gate {
            scale: Tower::from_repr(5),
        },
        16,
        11,
    )];
    assert_backends_agree(&instances, || LookupRuntime::Inactive, 0);
}

#[test]
fn backends_agree_when_a_public_value_lies_outside_the_subfield() {
    let mut instance = Instance::honest(FixtureAir::Gate { scale: Tower::ONE }, 16, 12);
    instance.public_values[1] = outside();
    assert_backends_agree(&[instance], || LookupRuntime::Inactive, 0);
}

#[test]
fn backends_agree_when_an_interpolation_step_lies_outside_the_subfield() {
    let instances = [Instance::honest(FixtureAir::Quartic, 16, 13)];
    assert_backends_agree(&instances, || LookupRuntime::Inactive, 0);
}

#[test]
fn backends_agree_on_full_width_cells() {
    // Fixture state:
    //
    //     stage 32 rows : recurrence, whose first round cannot run in GF(4)
    //     stage  8 rows : gate, which fits GF(4)
    let instances = [
        Instance::honest(FixtureAir::Recurrence, 32, 16),
        Instance::honest(FixtureAir::Gate { scale: Tower::ONE }, 8, 17),
    ];
    assert!(!<Tower as HasSubfield<Gf4>>::all_in_subfield(
        &instances[0].main.values
    ));
    assert_backends_agree(&instances, || LookupRuntime::Inactive, 0);
    assert_backends_agree(&instances, || LookupRuntime::Inactive, 2);
}

#[test]
fn backends_agree_on_stages_tall_enough_to_fold_in_parallel() {
    // Fixture state:
    //
    //     stage 2^14 rows : pair, which fits GF(4); its first later round lifts 2^12 eq weights
    //     stage 2^12 rows : recurrence, whose columns fold across threads outside GF(4)
    let instances = [
        Instance::honest(FixtureAir::Pair, 1 << 14, 18),
        Instance::honest(FixtureAir::Recurrence, 1 << 12, 19),
    ];
    assert_backends_agree(&instances, || LookupRuntime::Inactive, 0);
}

#[test]
fn backends_agree_on_stages_tall_enough_to_slice() {
    // Fixture state:
    //
    //     stage 2^9 rows : gate (degree 3) and pair (degree 2), sliced from round 0
    //     stage 2^7 rows : gate, sliced once it activates two rounds later
    let instances = [
        Instance::honest(FixtureAir::Gate { scale: Tower::ONE }, 1 << 9, 20),
        Instance::honest(FixtureAir::Pair, 1 << 9, 21),
        Instance::honest(FixtureAir::Gate { scale: gf4(3) }, 1 << 7, 22),
    ];
    assert_backends_agree(&instances, || LookupRuntime::Inactive, 0);
    assert_backends_agree(&instances, || LookupRuntime::Inactive, 2);
}

#[test]
fn representation_tensor4_matches_generic_on_invalid_boolean_and_gf4_traces() {
    let height = 1 << 10;
    let mut invalid_boolean = Instance::honest(FixtureAir::Pair, height, 0x007E_5010);
    for row in 0..height {
        invalid_boolean.main.values[3 * row..3 * row + 3].copy_from_slice(&[
            Tower::ZERO,
            Tower::ZERO,
            Tower::ONE,
        ]);
    }
    let mut non_boolean = Instance::honest(FixtureAir::Pair, height, 0x007E_5011);
    for row in 0..height {
        non_boolean.main.values[3 * row..3 * row + 3].copy_from_slice(&[
            gf4(2),
            Tower::ONE,
            Tower::ZERO,
        ]);
    }

    for instance in [invalid_boolean, non_boolean] {
        let instances = [instance];
        let generic = transcript::<GenericBackend>(&instances, LookupRuntime::Inactive, 0, false);
        let repr = transcript::<ReprBackend<Gf4, PolyBasis>>(
            &instances,
            LookupRuntime::Inactive,
            0,
            false,
        );
        let late = transcript::<ReprBackend<Gf4, PolyBasis, true>>(
            &instances,
            LookupRuntime::Inactive,
            0,
            false,
        );
        assert_eq!(
            repr, generic,
            "tensor4 must preserve invalid witness transcript"
        );
        assert_eq!(
            late, generic,
            "late boundary must preserve invalid witness transcript"
        );
    }
}

#[test]
fn representation_invalid_proofs_are_rejected_at_n11() {
    let height = 1 << 11;
    let mut invalid_boolean = Instance::honest(FixtureAir::Pair, height, 0x007E_5016);
    for row in 0..height {
        invalid_boolean.main.values[3 * row..3 * row + 3].copy_from_slice(&[
            Tower::ZERO,
            Tower::ZERO,
            Tower::ONE,
        ]);
    }
    let mut non_boolean = Instance::honest(FixtureAir::Pair, height, 0x007E_5017);
    for row in 0..height {
        non_boolean.main.values[3 * row..3 * row + 3].copy_from_slice(&[
            gf4(2),
            Tower::ONE,
            Tower::ZERO,
        ]);
    }

    for instance in [invalid_boolean, non_boolean] {
        for (name, backend) in [("incumbent", false), ("late", true)] {
            let airs = [&instance.air];
            let zerocheck = AirZerocheck::new(&airs, 0);
            let main = instance.main_table();
            let (proof, _) = if backend {
                zerocheck.prove_with_lookup::<Tower, Tower, ReprBackend<Gf4, PolyBasis, true>, _>(
                    &[None],
                    &[&main],
                    &[&instance.public_values],
                    LookupRuntime::Inactive,
                    DEFAULT_SLICED_ROUNDS,
                    &mut challenger(),
                )
            } else {
                zerocheck.prove_with_lookup::<Tower, Tower, ReprBackend<Gf4, PolyBasis>, _>(
                    &[None],
                    &[&main],
                    &[&instance.public_values],
                    LookupRuntime::Inactive,
                    DEFAULT_SLICED_ROUNDS,
                    &mut challenger(),
                )
            };
            assert!(
                zerocheck
                    .verify::<Tower, Tower, _>(
                        &proof,
                        &[11],
                        &[&instance.public_values],
                        &mut challenger(),
                    )
                    .is_err(),
                "invalid {name} proof must be rejected"
            );
        }
    }
}

#[test]
fn representation_tensor4_invalid_proof_at_n10_is_rejected() {
    let height = 1 << 10;
    let mut instance = Instance::honest(FixtureAir::Pair, height, 0x007E_5019);
    for row in 0..height {
        instance.main.values[3 * row..3 * row + 3].copy_from_slice(&[
            Tower::ZERO,
            Tower::ZERO,
            Tower::ONE,
        ]);
    }
    let airs = [&instance.air];
    let zerocheck = AirZerocheck::new(&airs, 0);
    let main = instance.main_table();
    let (proof, _) = zerocheck.prove_with_lookup::<Tower, Tower, ReprBackend<Gf4, PolyBasis>, _>(
        &[None],
        &[&main],
        &[&instance.public_values],
        LookupRuntime::Inactive,
        DEFAULT_SLICED_ROUNDS,
        &mut challenger(),
    );
    assert!(
        zerocheck
            .verify::<Tower, Tower, _>(
                &proof,
                &[10],
                &[&instance.public_values],
                &mut challenger(),
            )
            .is_err()
    );
}

#[test]
fn representation_tensor4_honest_proof_verifies() {
    let instance = Instance::honest(FixtureAir::Pair, 1 << 10, 0x007E_5018);
    let airs = [&instance.air];
    let zerocheck = AirZerocheck::new(&airs, 0);
    let main = instance.main_table();
    let (proof, point) = zerocheck
        .prove_with_lookup::<Tower, Tower, ReprBackend<Gf4, PolyBasis>, _>(
            &[None],
            &[&main],
            &[&instance.public_values],
            LookupRuntime::Inactive,
            DEFAULT_SLICED_ROUNDS,
            &mut challenger(),
        );
    let verified = zerocheck
        .verify::<Tower, Tower, _>(&proof, &[10], &[&instance.public_values], &mut challenger())
        .expect("honest tensor4 proof must verify");
    assert_eq!(verified, point);
}

#[test]
fn representation_late_boundary_honest_proof_verifies() {
    let instance = Instance::honest(FixtureAir::Pair, 1 << 11, 0x007E_501A);
    let airs = [&instance.air];
    let zerocheck = AirZerocheck::new(&airs, 0);
    let main = instance.main_table();
    let (proof, point) = zerocheck
        .prove_with_lookup::<Tower, Tower, ReprBackend<Gf4, PolyBasis, true>, _>(
            &[None],
            &[&main],
            &[&instance.public_values],
            LookupRuntime::Inactive,
            DEFAULT_SLICED_ROUNDS,
            &mut challenger(),
        );
    let verified = zerocheck
        .verify::<Tower, Tower, _>(&proof, &[11], &[&instance.public_values], &mut challenger())
        .expect("honest late-boundary proof must verify");
    assert_eq!(verified, point);
}

#[test]
fn representation_tensor4_matches_generic_for_mixed_native_degrees() {
    let height = 1 << 10;
    let mut linear = Instance::honest(
        FixtureAir::Linear { scale: Tower::ONE },
        height,
        0x007E_5012,
    );
    linear.main.values[0] = gf4(2);
    let instances = [
        linear,
        Instance::honest(FixtureAir::Pair, height, 0x007E_5013),
    ];
    let generic = transcript::<GenericBackend>(&instances, LookupRuntime::Inactive, 0, false);
    let repr =
        transcript::<ReprBackend<Gf4, PolyBasis>>(&instances, LookupRuntime::Inactive, 0, false);
    assert_eq!(repr, generic, "mixed degree tensor4 transcript");
}

#[test]
fn representation_tensor4_matches_generic_at_two_eligible_heights() {
    let instances = [
        Instance::honest(FixtureAir::Pair, 1 << 12, 0x007E_5022),
        Instance::honest(FixtureAir::Pair, 1 << 10, 0x007E_5023),
    ];
    let generic = transcript::<GenericBackend>(&instances, LookupRuntime::Inactive, 0, false);
    let repr =
        transcript::<ReprBackend<Gf4, PolyBasis>>(&instances, LookupRuntime::Inactive, 0, false);
    assert_eq!(repr, generic, "two eligible tensor4 activation heights");
}

#[test]
fn representation_late_boundary_matches_generic_at_two_activation_heights() {
    for height in [1 << 11, 1 << 12] {
        let instances = [Instance::honest(
            FixtureAir::Pair,
            height,
            0x007E_5030 + height as u64,
        )];
        let generic = transcript::<GenericBackend>(&instances, LookupRuntime::Inactive, 0, false);
        let incumbent = transcript::<ReprBackend<Gf4, PolyBasis>>(
            &instances,
            LookupRuntime::Inactive,
            0,
            false,
        );
        let late = transcript::<ReprBackend<Gf4, PolyBasis, true>>(
            &instances,
            LookupRuntime::Inactive,
            0,
            false,
        );
        assert_eq!(incumbent, generic, "incumbent representation at {height}");
        assert_eq!(late, generic, "late representation at {height}");
    }
}

#[test]
fn representation_late_boundary_matches_generic_for_n13_and_n11_stages() {
    let instances = [
        Instance::honest(FixtureAir::Pair, 1 << 13, 0x007E_5032),
        Instance::honest(FixtureAir::Pair, 1 << 11, 0x007E_5033),
    ];
    let generic = transcript::<GenericBackend>(&instances, LookupRuntime::Inactive, 0, false);
    let late = transcript::<ReprBackend<Gf4, PolyBasis, true>>(
        &instances,
        LookupRuntime::Inactive,
        0,
        false,
    );
    assert_eq!(late, generic, "n13+n11 late stages");
}

#[test]
fn representation_late_boundary_rejects_preprocessed_successor_optimization() {
    let instances = [Instance::honest(
        FixtureAir::QuadraticPreprocessedSuccessor,
        1 << 11,
        0x007E_5034,
    )];
    assert_backends_agree(&instances, || LookupRuntime::Inactive, 0);
}

#[test]
fn representation_tensor4_is_restricted_to_default_three_slices() {
    let height = 1 << 10;
    let instances = [Instance::honest(FixtureAir::Pair, height, 0x007E_5014)];
    let generic = transcript_with_storage::<GenericBackend>(
        DEFAULT_SLICED_ROUNDS,
        &instances,
        LookupRuntime::Inactive,
        0,
        |_, _| false,
    );
    for sliced_rounds in [0, 1, 2, 4] {
        let repr = transcript_with_storage::<ReprBackend<Gf4, PolyBasis>>(
            sliced_rounds,
            &instances,
            LookupRuntime::Inactive,
            0,
            |_, _| false,
        );
        assert_eq!(repr, generic, "configured sliced rounds {sliced_rounds}");
    }
}

#[test]
fn backends_agree_when_a_tall_stage_does_not_fit() {
    // Each stage is tall enough to slice, so every misfit reaches the sliced kernel first.
    let height = 1 << 8;
    let gate = |seed| Instance::honest(FixtureAir::Gate { scale: Tower::ONE }, height, seed);
    let mut cell = gate(23);
    cell.main.values[4 * (height - 1) + 3] = outside();
    let mut public = gate(24);
    public.public_values[1] = outside();
    let constant = Instance::honest(
        FixtureAir::Gate {
            scale: Tower::from_repr(5),
        },
        height,
        25,
    );
    let periodic = Instance::honest(
        FixtureAir::Periodic {
            period: [gf4(2), outside()],
        },
        height,
        26,
    );
    let quartic = Instance::honest(FixtureAir::Quartic, height, 27);
    for instance in [cell, public, constant, periodic, quartic] {
        assert_backends_agree(&[instance], || LookupRuntime::Inactive, 0);
    }
}

#[test]
fn backends_agree_when_a_constant_first_poisons_a_later_sliced_round() {
    // The first round of a degree-one stage evaluates no node, so the sliced kernel only meets
    // the out-of-subfield constant in the next round, and leaves its planes there.
    let scale = Tower::from_repr(5);
    for height in [1 << 7, 1 << 8, 1 << 11] {
        let instances = [Instance::honest(FixtureAir::Linear { scale }, height, 28)];
        assert_backends_agree(&instances, || LookupRuntime::Inactive, 0);
    }
}

/// Degrees the symbolic pass sees, so a fixture cannot drift from the shape its test names.
#[test]
fn fixture_degrees_are_the_named_ones() {
    let degree = |air: &FixtureAir| {
        super::get_air_profile::<Tower, Tower, _>(air)
            .degrees
            .constraints
    };
    assert_eq!(degree(&FixtureAir::Gate { scale: Tower::ONE }), 3);
    assert_eq!(degree(&FixtureAir::Pair), 2);
    assert_eq!(degree(&FixtureAir::Quartic), 4);
    let periodic = FixtureAir::Periodic {
        period: [gf4(2), gf4(3)],
    };
    assert_eq!(degree(&periodic), 3);
    assert_eq!(degree(&FixtureAir::Recurrence), 3);
    assert_eq!(degree(&FixtureAir::Linear { scale: gf4(2) }), 1);
    assert_eq!(degree(&FixtureAir::QuadraticInputs), 2);
    assert_eq!(degree(&FixtureAir::QuadraticInputsOutsidePeriodic), 2);
    let link = super::get_air_profile::<Tower, Tower, _>(&FixtureAir::Link).degrees;
    assert!(link.interactions > 0);
}

#[test]
fn representation_tensor4_handles_fixed_periodic_selectors_and_public_pins() {
    let height = 1 << 10;
    let instances = [Instance::honest(
        FixtureAir::QuadraticInputs,
        height,
        0x007E_5015,
    )];
    assert_backends_agree(&instances, || LookupRuntime::Inactive, 0);
}

/// Every sliced-round count must leave the proof the generic backend's.
///
/// The count only decides how many rounds run on the planes. Each of those rounds computes
/// the polynomial the generic kernel computes, so no count may move the transcript.
#[test]
fn every_sliced_round_count_agrees_with_the_generic_backend() {
    // The kernel also caps the count at the row variables a stage keeps once a word's lanes
    // are spent, so a shorter fixture would run fewer rounds than the loop below asks for and
    // leave the top of the range unreached. Deriving the height from the ceiling pins that.
    const LANE_VARIABLES: usize = SLICED_LANES.trailing_zeros() as usize;
    const HEIGHT: usize = 1 << (MAX_SLICED_ROUNDS + LANE_VARIABLES);

    let instances = [
        Instance::honest(FixtureAir::Gate { scale: Tower::ONE }, HEIGHT, 20),
        Instance::honest(FixtureAir::Pair, HEIGHT, 21),
    ];
    let generic = transcript::<GenericBackend>(&instances, LookupRuntime::Inactive, 0, false);

    for sliced_rounds in 0..=MAX_SLICED_ROUNDS + 1 {
        let sliced = transcript_with_storage::<SubfieldBackend<Gf4>>(
            sliced_rounds,
            &instances,
            LookupRuntime::Inactive,
            0,
            |_, _| false,
        );
        assert_eq!(sliced, generic, "{sliced_rounds} sliced rounds");
    }
}
