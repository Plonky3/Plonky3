use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::TowerLevel;
use p3_lookup::InteractionSymbolicBuilder;
use p3_multilinear_util::point::Point;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::*;
use crate::lookup::{AirLinkInstance, AirLinkLookup};
use crate::rounds::{Stage, StageCoupling};
use crate::zerocheck::backend_tests::{FixtureAir, Gf4, Instance, Tower, gf4, outside};
use crate::zerocheck::get_air_profile;

/// The fixture round state every test here drives.
pub(in crate::rounds) type BaseState<'air, 'data> =
    RoundStateBase<'air, 'data, FixtureAir, Tower, Tower>;

/// The challenge every test binds the first variable at.
pub(in crate::rounds) fn first_challenge() -> Tower {
    Tower::from_repr(0xF01D_0000_0000_0000_0000_0000_0000_0007)
}

/// Activate one stage from AIRs of equal height and hand its first-round state to `body`.
///
/// The tables and public values are given per AIR, in the order of `airs`.
/// The challenges are fixed random elements of the tower.
pub(in crate::rounds) fn with_stage_state<A, R>(
    airs: &[&A],
    public_values: &[&[Tower]],
    preprocessed: &[Option<&Table<Tower>>],
    main: &[&Table<Tower>],
    coupling: StageCoupling<Tower>,
    body: impl FnOnce(RoundStateBase<'_, '_, A, Tower, Tower>, &Poly<Tower>) -> R,
) -> R
where
    A: BaseAir<Tower> + Air<InteractionSymbolicBuilder<Tower, Tower>>,
{
    let stage = Stage::new(
        airs.to_vec(),
        public_values.to_vec(),
        (0..airs.len()).collect(),
        preprocessed.to_vec(),
        main.to_vec(),
        airs.iter()
            .map(|air| get_air_profile::<Tower, Tower, _>(*air))
            .collect(),
        coupling,
    );

    let mut rng = SmallRng::seed_from_u64(0x5B);
    let tau = Point::rand(&mut rng, stage.num_vars);
    let eq_suffix = Poly::new_from_point(&tau.as_slice()[1..], Tower::ONE);
    let betas = (0..airs.len()).map(|_| rng.random()).collect();
    let state = RoundStateBase::new(stage, rng.random(), rng.random(), betas, tau);
    body(state, &eq_suffix)
}

/// Activate one stage from fixture instances of equal height, as [`with_stage_state`] does.
pub(in crate::rounds) fn with_state<R>(
    instances: &[Instance],
    coupling: StageCoupling<Tower>,
    body: impl FnOnce(BaseState<'_, '_>, &Poly<Tower>) -> R,
) -> R {
    let main = instances
        .iter()
        .map(Instance::main_table)
        .collect::<Vec<_>>();
    let preprocessed = instances
        .iter()
        .map(Instance::preprocessed_table)
        .collect::<Vec<_>>();
    with_stage_state(
        &instances
            .iter()
            .map(|instance| &instance.air)
            .collect::<Vec<_>>(),
        &instances
            .iter()
            .map(|instance| instance.public_values.as_slice())
            .collect::<Vec<_>>(),
        &preprocessed.iter().map(Option::as_ref).collect::<Vec<_>>(),
        &main.iter().collect::<Vec<_>>(),
        coupling,
        body,
    )
}

pub(in crate::rounds) fn no_lookups() -> StageCoupling<Tower> {
    StageCoupling::new(BTreeMap::new(), BTreeMap::new(), vec![])
}

/// Lookup coefficients for a stage whose only AIR is the lookup fixture.
pub(in crate::rounds) fn link_coupling() -> StageCoupling<Tower> {
    let link = AirLinkInstance {
        num_local_lookups: 1,
        lookups: vec![AirLinkLookup {
            theta_bus_offset: gf4(1),
            block_weights: vec![gf4(2), gf4(3)],
        }],
    };
    StageCoupling::new(
        BTreeMap::from([(0, gf4(1))]),
        BTreeMap::from([(0, link)]),
        vec![gf4(2)],
    )
}

/// The subfield kernel's first round polynomial, beside the generic kernel's.
fn first_rounds(
    instances: &[Instance],
    coupling: StageCoupling<Tower>,
) -> (Option<Vec<Tower>>, Vec<Tower>) {
    with_state(instances, coupling, |mut state, eq_suffix| {
        let subfield = state.round_poly_subfield::<Gf4>(eq_suffix);
        (subfield, state.round_poly(eq_suffix))
    })
}

/// Every later round polynomial of a folded stage, then its openings.
pub(in crate::rounds) fn later_rounds(
    mut state: RoundStateExt<'_, '_, FixtureAir, Tower, Tower>,
) -> (Vec<Vec<Tower>>, Vec<[Vec<Tower>; 4]>) {
    let tau = state.tau.as_slice().to_vec();
    let round_polys = (1..tau.len())
        .map(|round| {
            let eq_suffix = Poly::new_from_point(&tau[round + 1..], Tower::ONE);
            let round_poly = state.round_poly(&eq_suffix);
            state.fold(first_challenge() + Tower::from_repr(round as u128));
            round_poly
        })
        .collect();
    let openings = state
        .into_openings()
        .into_iter()
        .map(|(_, opening)| {
            [
                opening.local,
                opening.next,
                opening.preprocessed_local,
                opening.preprocessed_next,
            ]
        })
        .collect();
    (round_polys, openings)
}

/// A gate trace with column e of row 5 outside `GF(4)`.
fn gate_with_an_outside_cell(seed: u64) -> Instance {
    let mut instance = Instance::honest(FixtureAir::Gate { scale: Tower::ONE }, 16, seed);
    instance.main.values[4 * 5 + 3] = outside();
    instance
}

/// A gate trace with its preprocessed column outside `GF(4)` at row 5.
fn gate_with_an_outside_preprocessed_cell(seed: u64) -> Instance {
    let mut instance = Instance::honest(FixtureAir::Gate { scale: Tower::ONE }, 16, seed);
    instance
        .preprocessed
        .as_mut()
        .expect("the gate AIR declares a preprocessed column")
        .values[5] = outside();
    instance
}

/// A periodic trace whose period vector holds an element outside `GF(4)`.
fn periodic_with_an_outside_value(seed: u64) -> Instance {
    let period = [gf4(2), outside()];
    Instance::honest(FixtureAir::Periodic { period }, 16, seed)
}

#[test]
fn a_fitting_stage_runs_in_the_subfield_and_matches_the_generic_kernel() {
    // Residual rows per height: one row, part of a task, one full task, several tasks.
    //
    // An honest two-row pair can have a zero round polynomial, so its product cell breaks it.
    let mut pair = Instance::honest(FixtureAir::Pair, 2, 1);
    pair.main.values[2] = gf4(3);
    let stages = [
        vec![pair],
        vec![
            Instance::honest(FixtureAir::Gate { scale: Tower::ONE }, 4, 2),
            Instance::honest(FixtureAir::Pair, 4, 3),
        ],
        vec![Instance::honest(FixtureAir::Gate { scale: gf4(2) }, 32, 4)],
        vec![
            Instance::honest(FixtureAir::Pair, 128, 5),
            Instance::honest(FixtureAir::Gate { scale: gf4(3) }, 128, 6),
        ],
    ];
    for instances in stages {
        let (subfield, generic) = first_rounds(&instances, no_lookups());
        assert!(generic.iter().any(|value| *value != Tower::ZERO));
        assert_eq!(subfield, Some(generic));
    }
}

#[test]
fn each_misfit_falls_back() {
    let mut public = Instance::honest(FixtureAir::Gate { scale: Tower::ONE }, 16, 7);
    public.public_values[1] = outside();
    let constant = Instance::honest(
        FixtureAir::Gate {
            scale: Tower::from_repr(5),
        },
        16,
        8,
    );
    let quartic = Instance::honest(FixtureAir::Quartic, 16, 9);

    for (name, instance) in [
        ("cell", gate_with_an_outside_cell(7)),
        (
            "preprocessed cell",
            gate_with_an_outside_preprocessed_cell(7),
        ),
        ("periodic value", periodic_with_an_outside_value(7)),
        ("public value", public),
        ("constant", constant),
        ("interpolation step", quartic),
    ] {
        let (subfield, _) = first_rounds(&[instance], no_lookups());
        assert_eq!(subfield, None, "{name}");
    }
}

#[test]
fn a_lookup_stage_falls_back() {
    let (subfield, _) = first_rounds(
        &[Instance::honest(FixtureAir::Link, 16, 10)],
        link_coupling(),
    );
    assert_eq!(subfield, None);
}

#[test]
fn an_out_of_subfield_constant_passes_the_check_and_poisons_the_probe() {
    let instance = Instance::honest(
        FixtureAir::Gate {
            scale: Tower::from_repr(5),
        },
        16,
        11,
    );
    with_state(&[instance], no_lookups(), |state, eq_suffix| {
        let schedule = state
            .subfield_schedule::<Gf4>()
            .expect("every cell, public value, and step fits");
        assert!(
            state
                .subfield_pass(&eq_suffix.as_slice()[..1], &schedule)
                .poisoned
        );
    });
}

#[test]
fn the_full_pass_catches_a_misfit_the_probe_does_not_read() {
    let gate = Instance::honest(FixtureAir::Gate { scale: Tower::ONE }, 16, 12);
    let schedule = with_state(&[gate], no_lookups(), |state, _| {
        state
            .subfield_schedule::<Gf4>()
            .expect("an honest gate fits")
    });

    // Only row 5 reads column e of row 5, and the probe reads row 0 alone.
    with_state(
        &[gate_with_an_outside_cell(12)],
        no_lookups(),
        |state, eq_suffix| {
            assert!(!state.cells_fit_subfield::<Gf4>());
            let eq_suffix = eq_suffix.as_slice();
            assert!(!state.subfield_pass(&eq_suffix[..1], &schedule).poisoned);
            assert!(state.subfield_pass(eq_suffix, &schedule).poisoned);
        },
    );
}

#[test]
fn the_fit_is_recorded_for_the_fold() {
    let mut public = Instance::honest(FixtureAir::Gate { scale: Tower::ONE }, 16, 13);
    public.public_values[1] = outside();
    let cases = [
        (
            "honest gate",
            Instance::honest(FixtureAir::Gate { scale: Tower::ONE }, 16, 13),
            no_lookups(),
            true,
        ),
        // Poison stops the first round, but every cell still fits the fold.
        (
            "constant",
            Instance::honest(
                FixtureAir::Gate {
                    scale: Tower::from_repr(5),
                },
                16,
                13,
            ),
            no_lookups(),
            true,
        ),
        ("cell", gate_with_an_outside_cell(13), no_lookups(), false),
        (
            "preprocessed cell",
            gate_with_an_outside_preprocessed_cell(13),
            no_lookups(),
            false,
        ),
        (
            "periodic value",
            periodic_with_an_outside_value(13),
            no_lookups(),
            false,
        ),
        ("public value", public, no_lookups(), false),
        (
            "interpolation step",
            Instance::honest(FixtureAir::Quartic, 16, 13),
            no_lookups(),
            false,
        ),
        (
            "lookup",
            Instance::honest(FixtureAir::Link, 16, 13),
            link_coupling(),
            false,
        ),
    ];
    for (name, instance, coupling, fits) in cases {
        with_state(&[instance], coupling, |mut state, eq_suffix| {
            assert!(!state.fits_subfield(), "{name}: a fresh stage");
            let _round_poly = state.round_poly_subfield::<Gf4>(eq_suffix);
            assert_eq!(state.fits_subfield(), fits, "{name}");
        });
    }
}

#[test]
fn the_subfield_fold_matches_the_generic_fold() {
    // Two rows fold into scalar columns, more rows into packed ones.
    // A column of `PARALLEL_FOLD_CELLS` cells splits its fold across threads.
    let mut pair = Instance::honest(FixtureAir::Pair, 2, 14);
    pair.main.values[2] = gf4(3);
    let stages = [
        vec![pair],
        vec![
            Instance::honest(FixtureAir::Gate { scale: Tower::ONE }, 4, 15),
            Instance::honest(FixtureAir::Pair, 4, 16),
        ],
        vec![Instance::honest(FixtureAir::Gate { scale: gf4(2) }, 32, 17)],
        vec![Instance::honest(FixtureAir::Pair, PARALLEL_FOLD_CELLS, 19)],
    ];
    for instances in stages {
        let subfield = with_state(&instances, no_lookups(), |mut state, eq_suffix| {
            assert!(state.round_poly_subfield::<Gf4>(eq_suffix).is_some());
            later_rounds(state.fold_subfield::<Gf4>(first_challenge()))
        });
        let generic = with_state(&instances, no_lookups(), |mut state, eq_suffix| {
            state.round_poly(eq_suffix);
            later_rounds(state.fold(first_challenge()))
        });
        assert_eq!(subfield, generic);
    }
}

#[test]
fn a_pair_outside_the_subfield_folds_with_the_general_product() {
    let fold = |subfield: bool| {
        with_state(
            &[gate_with_an_outside_cell(18)],
            no_lookups(),
            |mut state, eq_suffix| {
                state.round_poly(eq_suffix);
                let folded = if subfield {
                    state.fold_subfield::<Gf4>(first_challenge())
                } else {
                    state.fold(first_challenge())
                };
                later_rounds(folded)
            },
        )
    };
    assert_eq!(fold(true), fold(false));
}
