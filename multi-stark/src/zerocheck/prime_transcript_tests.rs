//! The generic backend's zerocheck transcript over a prime field, pinned by digest.
//!
//! Where a prime field packs several rows into one lane, a stage runs the packed kernels while
//! its residual rows fill a lane and the scalar kernels after that; the batch below then reaches
//! both. It carries every column group, a public pin, a lookup, and two stages. Any change to a
//! round polynomial, a fold, or an opening changes the digest.

use alloc::borrow::Cow;
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use p3_air::{Air, AirBuilder, BaseAir, BoundaryEnd, BoundaryPublic, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{DuplexChallenger, FieldChallenger};
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_keccak::Keccak256Hash;
use p3_lookup::{Count, InteractionBuilder};
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::point::Point;
use p3_sumcheck::layout::Table;
use p3_symmetric::CryptographicHasher;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::AirZerocheck;
use crate::backend::GenericBackend;
use crate::config::DEFAULT_SLICED_ROUNDS;
use crate::lookup::{
    ActiveLookupRuntime, AirLinkClaim, AirLinkInstance, AirLinkLookup, LookupRuntime,
};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Challenger = DuplexChallenger<F, Poseidon2BabyBear<16>, 16, 8>;

/// The one cell the gate AIR binds to a public value.
const GATE_CELLS: [BoundaryPublic; 1] = [BoundaryPublic::new(0, BoundaryEnd::First, 0)];

/// Small AIRs over a prime field, together touching every input a round kernel reads.
enum PrimeAir {
    /// Degree-three AIR reading every column group, a public value, and a constant.
    ///
    /// Main columns `a, b, c`, preprocessed column `q`, periodic column `p = [1, 2, 3, 4]`:
    ///
    /// ```text
    ///     transition : next.c = a * b * q
    ///     transition : next.q = q + a
    ///     always     : p * a * c = b + 7
    ///     pin        : first row, a = public[0]
    /// ```
    Gate,
    /// Degree-two AIR declaring one local lookup beside one ordinary constraint.
    ///
    /// ```text
    ///     always : a * b = c
    ///     lookup : a * b requested once, c provided once
    /// ```
    Link,
}

impl BaseAir<F> for PrimeAir {
    fn width(&self) -> usize {
        3
    }

    fn preprocessed_width(&self) -> usize {
        match self {
            Self::Gate => 1,
            Self::Link => 0,
        }
    }

    fn num_public_values(&self) -> usize {
        match self {
            Self::Gate => 1,
            Self::Link => 0,
        }
    }

    fn num_periodic_columns(&self) -> usize {
        match self {
            Self::Gate => 1,
            Self::Link => 0,
        }
    }

    fn periodic_columns(&self) -> Cow<'_, [Vec<F>]> {
        match self {
            Self::Gate => Cow::Owned(vec![(1..=4).map(F::from_u8).collect()]),
            Self::Link => Cow::Owned(vec![]),
        }
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        match self {
            Self::Gate => vec![2],
            Self::Link => vec![],
        }
    }

    fn preprocessed_next_row_columns(&self) -> Vec<usize> {
        match self {
            Self::Gate => vec![0],
            Self::Link => vec![],
        }
    }

    fn public_boundary_io(&self) -> &[BoundaryPublic] {
        match self {
            Self::Gate => &GATE_CELLS,
            Self::Link => &[],
        }
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for PrimeAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.current_slice(), main.next_slice());
        let (a, b, c) = (local[0], local[1], local[2]);
        match self {
            Self::Gate => {
                let preprocessed = builder.preprocessed();
                let (q, next_q) = (
                    preprocessed.current_slice()[0],
                    preprocessed.next_slice()[0],
                );
                let p: AB::Expr = builder.periodic_values()[0].into();
                builder.when_transition().assert_eq(next[2], a * b * q);
                builder.when_transition().assert_eq(next_q, q + a);
                builder.assert_eq(p * a * c, b + F::from_u8(7));
            }
            Self::Link => {
                builder.assert_eq(a * b, c);
                builder.push_local_interaction([
                    (vec![a * b], Count::bounded(AB::Expr::ONE, 1)),
                    (vec![c.into()], Count::provided(AB::Expr::ONE)),
                ]);
            }
        }
    }
}

/// One AIR of the batch with random traces, which the zerocheck prover never checks.
struct Instance {
    air: PrimeAir,
    main: Table<F>,
    preprocessed: Option<Table<F>>,
    public_values: Vec<F>,
}

impl Instance {
    fn random(air: PrimeAir, log_height: usize, rng: &mut SmallRng) -> Self {
        let height = 1 << log_height;
        let mut table = |width: usize| {
            let values = (0..width * height).map(|_| rng.random()).collect();
            Table::new(RowMajorMatrix::new(values, width).transpose())
        };
        let main = table(3);
        let preprocessed = (air.preprocessed_width() > 0).then(|| table(1));
        let public_values = (0..air.num_public_values()).map(|_| rng.random()).collect();
        Self {
            air,
            main,
            preprocessed,
            public_values,
        }
    }
}

/// Lookup-reduction output for the lookup AIR at `air_index`, with random coefficients.
fn link_runtime(air_index: usize, num_variables: usize, rng: &mut SmallRng) -> LookupRuntime<EF> {
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
            point: Point::rand(rng, num_variables),
            claimed_sum: claim,
            theta_beta_powers: vec![rng.random()],
            links_by_air: BTreeMap::from([(air_index, link)]),
        },
    })
}

#[test]
fn generic_backend_prime_field_transcript_is_pinned() {
    // Fixture state:
    //
    //     stage 128 rows : gate, packed rounds while a lane fills, then scalar rounds
    //     stage   8 rows : gate and a lookup AIR, activating four rounds later
    let mut rng = SmallRng::seed_from_u64(0x9A1E);
    let instances = [
        Instance::random(PrimeAir::Gate, 7, &mut rng),
        Instance::random(PrimeAir::Gate, 3, &mut rng),
        Instance::random(PrimeAir::Link, 3, &mut rng),
    ];
    let lookup = link_runtime(2, 7, &mut rng);

    let airs = instances
        .iter()
        .map(|instance| &instance.air)
        .collect::<Vec<_>>();
    let mut challenger = Challenger::new(Poseidon2BabyBear::new_from_rng_128(&mut rng));
    let (proof, point) = AirZerocheck::new(&airs, 0).prove_with_lookup::<F, EF, GenericBackend, _>(
        &instances
            .iter()
            .map(|instance| instance.preprocessed.as_ref())
            .collect::<Vec<_>>(),
        &instances
            .iter()
            .map(|instance| &instance.main)
            .collect::<Vec<_>>(),
        &instances
            .iter()
            .map(|instance| instance.public_values.as_slice())
            .collect::<Vec<_>>(),
        lookup,
        DEFAULT_SLICED_ROUNDS,
        &mut challenger,
    );
    let next_challenge: EF = challenger.sample_algebra_element();
    let bytes = postcard::to_allocvec(&(
        &proof.sumcheck,
        &proof.local,
        &proof.next,
        &proof.preprocessed_local,
        &proof.preprocessed_next,
        point.as_slice(),
        next_challenge,
    ))
    .expect("postcard serialization must not fail");

    let digest = Keccak256Hash.hash_iter(bytes);
    assert_eq!(
        digest,
        [
            0xd4, 0xed, 0x69, 0x6d, 0xf2, 0x19, 0x81, 0x3b, 0xc9, 0x8a, 0xde, 0xae, 0x4e, 0x9b,
            0x6a, 0x7d, 0xed, 0xf9, 0x1e, 0xc4, 0x7d, 0xe5, 0xb3, 0x35, 0x1b, 0x81, 0x13, 0xce,
            0xbc, 0x8c, 0x38, 0x4d
        ]
    );
}
