use alloc::vec::Vec;

use p3_air::{Air, AirBuilder, BaseAir, BoundaryEnd, BoundaryPublic, WindowAccess};
use p3_binary_field::{BinaryField2, BinaryField128, TowerLevel};
use proptest::prelude::*;

use super::*;
use crate::folder::MultilinearFolder;
use crate::selectors::BoundaryEvals;

type F = BinaryField128;
type S = BinaryField2;
type Var = SubfieldVar<F, S>;
type Acc = SubfieldAcc<F, S>;

/// The `GF(4)` element with the given low two bits.
fn gf4(bits: u8) -> S {
    S::from_repr(bits & 3)
}

/// The first bit pattern above `GF(4)`, so the smallest element outside it.
fn outside() -> F {
    F::from_repr(4)
}

/// Lift an unpoisoned subfield value into the trace field.
fn lift(x: Var) -> F {
    assert!(!x.is_poisoned(), "a clean computation must stay unpoisoned");
    F::from(x.value())
}

#[test]
fn narrowing_poisons_exactly_the_values_outside_the_subfield() {
    for bits in 0..4 {
        let x = F::from_repr(bits);
        assert_eq!(lift(Var::narrow(x)), x);
    }
    for bits in [4, 5, 7, 1 << 127, u128::MAX] {
        assert!(Var::narrow(F::from_repr(bits)).is_poisoned(), "{bits:#x}");
    }
}

#[test]
fn constants_are_the_trace_field_constants() {
    assert_eq!(lift(Var::ZERO), F::ZERO);
    assert_eq!(lift(Var::ONE), F::ONE);
    assert_eq!(lift(Var::TWO), F::TWO);
    assert_eq!(lift(Var::NEG_ONE), F::NEG_ONE);
    assert_eq!(lift(Var::default()), F::ZERO);
    for n in 0..8 {
        assert_eq!(lift(Var::from_u64(n)), F::from_u64(n));
        assert_eq!(lift(Var::from_bool(n % 2 == 1)), F::from_bool(n % 2 == 1));
    }

    assert_eq!(Acc::TWO.value(), F::TWO);
    assert_eq!(Acc::from_u64(3).value(), F::from_u64(3));
    assert!(!Acc::default().is_poisoned());
}

/// One path an `F` value takes into an expression, beside the same computation over `F`.
type Entry = (&'static str, fn(Var, F) -> Var, fn(F, F) -> F);

/// Every path an `F` value takes into an expression through the algebra interface.
const ENTRIES: [Entry; 10] = [
    ("from", |_, f| Var::from(f), |_, f| f),
    ("add", |x, f| x + f, |x, f| x + f),
    (
        "add_assign",
        |mut x, f| {
            x += f;
            x
        },
        |x, f| x + f,
    ),
    ("sub", |x, f| x - f, |x, f| x - f),
    (
        "sub_assign",
        |mut x, f| {
            x -= f;
            x
        },
        |x, f| x - f,
    ),
    ("mul", |x, f| x * f, |x, f| x * f),
    (
        "mul_assign",
        |mut x, f| {
            x *= f;
            x
        },
        |x, f| x * f,
    ),
    (
        "mixed_dot_product",
        |x, f| Var::mixed_dot_product::<2>(&[x, x], &[f, F::ONE]),
        |x, f| x * f + x,
    ),
    (
        "batched_linear_combination",
        |x, f| Var::batched_linear_combination(&[x, x], &[F::ONE, f]),
        |x, f| x + x * f,
    ),
    (
        "quadratic_extension_square",
        |x, f| Var::quadratic_extension_square(&[x, x], f)[0],
        |x, f| F::quadratic_extension_square(&[x, x], f)[0],
    ),
];

#[test]
fn every_trace_field_operand_narrows() {
    for (name, entry, reference) in ENTRIES {
        for a in 0..4 {
            for b in 0..4 {
                let (x, f) = (Var::new(gf4(a)), F::from(gf4(b)));
                assert_eq!(lift(entry(x, f)), reference(F::from(gf4(a)), f), "{name}");
            }
        }
        assert!(entry(Var::ONE, outside()).is_poisoned(), "{name}");
    }
}

#[test]
fn poison_survives_every_operation() {
    let poisoned = Var::narrow(outside());
    let clean = Var::ONE;
    let results = [
        poisoned + clean,
        clean + poisoned,
        poisoned - clean,
        clean - poisoned,
        poisoned * clean,
        clean * poisoned,
        // Annihilating the poisoned operand still leaves the result poisoned.
        poisoned * Var::ZERO,
        -poisoned,
        poisoned.double(),
        poisoned.square(),
        poisoned.cube(),
        poisoned.bool_check(),
        poisoned.exp_u64(5),
        [clean, poisoned, clean].into_iter().sum(),
        [clean, poisoned].into_iter().product(),
    ];
    for (index, result) in results.into_iter().enumerate() {
        assert!(result.is_poisoned(), "variable operation {index}");
    }
    let mut assigned = [clean; 3];
    assigned[0] += poisoned;
    assigned[1] -= poisoned;
    assigned[2] *= poisoned;
    assert!(assigned.iter().all(|x| x.is_poisoned()));

    let dirty = Acc::from(poisoned);
    let weight = Acc::new(F::from_repr(0x1234_5678));
    let results = [
        dirty,
        weight * poisoned,
        weight + poisoned,
        weight - poisoned,
        weight * dirty,
        dirty * weight,
        weight + dirty,
        weight - dirty,
        -dirty,
        dirty.square(),
        [weight, dirty].into_iter().sum(),
        [dirty, weight].into_iter().product(),
    ];
    for (index, result) in results.into_iter().enumerate() {
        assert!(result.is_poisoned(), "accumulator operation {index}");
    }
    let mut assigned = [weight; 5];
    assigned[0] += poisoned;
    assigned[1] -= poisoned;
    assigned[2] *= poisoned;
    assigned[3] += dirty;
    assigned[4] *= dirty;
    assert!(assigned.iter().all(|x| x.is_poisoned()));
}

proptest! {
    #[test]
    fn variable_arithmetic_is_the_trace_field_arithmetic(a in 0u8..4, b in 0u8..4, k: u64) {
        let (x, y) = (Var::new(gf4(a)), Var::new(gf4(b)));
        let (fx, fy) = (F::from(gf4(a)), F::from(gf4(b)));

        prop_assert_eq!(lift(x + y), fx + fy);
        prop_assert_eq!(lift(x - y), fx - fy);
        prop_assert_eq!(lift(x * y), fx * fy);
        prop_assert_eq!(lift(-x), -fx);
        prop_assert_eq!(lift(x.double()), fx.double());
        prop_assert_eq!(lift(x.square()), fx.square());
        prop_assert_eq!(lift(x.cube()), fx.cube());
        prop_assert_eq!(lift(x.bool_check()), fx.bool_check());
        prop_assert_eq!(lift(x.exp_u64(k)), fx.exp_u64(k));
        prop_assert_eq!(lift([x, y, x].into_iter().sum()), fx + fy + fx);
        prop_assert_eq!(lift([x, y, y].into_iter().product()), fx * fy * fy);
    }

    #[test]
    fn accumulator_arithmetic_is_the_challenge_field_arithmetic(
        a: u128,
        b: u128,
        s in 0u8..4,
    ) {
        let (x, y) = (F::from_repr(a), F::from_repr(b));
        let (acc_x, acc_y) = (Acc::new(x), Acc::new(y));
        let (var, lifted) = (Var::new(gf4(s)), F::from(gf4(s)));

        prop_assert_eq!(Acc::from(var).value(), lifted);
        prop_assert_eq!((acc_x * var).value(), x * lifted);
        prop_assert_eq!((acc_x + var).value(), x + lifted);
        prop_assert_eq!((acc_x - var).value(), x - lifted);
        prop_assert_eq!((acc_x * acc_y).value(), x * y);
        prop_assert_eq!((acc_x + acc_y).value(), x + y);
        prop_assert_eq!((acc_x - acc_y).value(), x - y);
        prop_assert_eq!((-acc_x).value(), -x);

        let mut assigned = acc_x;
        assigned *= var;
        assigned += var;
        assigned -= acc_y;
        prop_assert_eq!(assigned.value(), x * lifted + lifted - y);
        prop_assert!(!assigned.is_poisoned());
    }
}

/// The one cell the entry AIR binds to a public value.
const ENTRY_CELLS: [BoundaryPublic; 1] = [BoundaryPublic::new(1, BoundaryEnd::Last, 0)];

/// Degree-three AIR reading one `F` constant, one public value, and one periodic column.
///
/// ```text
///     transition : a * next_b = next_b + constant
///     always     : a * constant - public = 0
///     first row  : constant * periodic = a
///     always     : a is a bit
///     pin        : last row, b = public
/// ```
struct EntryAir {
    constant: F,
}

impl BaseAir<F> for EntryAir {
    fn width(&self) -> usize {
        2
    }

    fn num_public_values(&self) -> usize {
        1
    }

    fn public_boundary_io(&self) -> &[BoundaryPublic] {
        &ENTRY_CELLS
    }
}

impl<AB: AirBuilder<F = F>> Air<AB> for EntryAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (a, next_b) = (main.current_slice()[0], main.next_slice()[1]);
        let public: AB::Expr = builder.public_values()[0].into();
        let periodic: AB::Expr = builder.periodic_values()[0].into();

        builder
            .when_transition()
            .assert_eq(a * next_b, next_b + self.constant);
        builder.assert_zero(a * self.constant - public);
        builder
            .when_first_row()
            .assert_eq(AB::Expr::from(self.constant) * periodic, a);
        builder.assert_bool(a);
    }
}

/// Evaluate the entry AIR through the subfield folder and through the trace-field folder.
///
/// Every cell, selector, and periodic value lies in `GF(4)`.
/// Only the constant and the public value vary.
fn eval_entry_air(constant: F, public: F) -> (Acc, F) {
    let air = EntryAir { constant };
    let cells = [3, 1, 2, 3].map(|bits| F::from(gf4(bits)));
    let periodic = [F::from(gf4(2))];
    let boundary = BoundaryEvals::new(F::ONE, F::from(gf4(3)), F::from(gf4(2)));
    let alpha = F::from_repr(0xA1FA_0000_0000_0000_0000_0000_0000_0001);
    let mut powers = alpha.powers().collect_n(5);
    powers.reverse();
    let publics = [public];

    let expected = MultilinearFolder::new(&cells[..2], &cells[2..], boundary, &publics, alpha)
        .with_alpha_powers(&powers)
        .with_periodic(&periodic)
        .eval_air(&air);

    let narrow = |values: &[F]| values.iter().copied().map(Var::narrow).collect::<Vec<_>>();
    let (var_cells, var_periodic) = (narrow(&cells), narrow(&periodic));
    let var_boundary = BoundaryEvals::new(
        Var::narrow(boundary.first),
        Var::narrow(boundary.last),
        Var::narrow(boundary.transition),
    );
    let acc_powers = powers.iter().copied().map(Acc::new).collect::<Vec<_>>();
    let actual = MultilinearFolder::new(
        &var_cells[..2],
        &var_cells[2..],
        var_boundary,
        &publics,
        Acc::new(alpha),
    )
    .with_alpha_powers(&acc_powers)
    .with_periodic(&var_periodic)
    .eval_air(&air);
    (actual, expected)
}

#[test]
fn folder_matches_the_trace_field_while_every_input_fits() {
    for constant in 0..4 {
        for public in 0..4 {
            let (actual, expected) = eval_entry_air(F::from_repr(constant), F::from_repr(public));
            assert!(
                !actual.is_poisoned(),
                "constant {constant}, public {public}"
            );
            assert_eq!(
                actual.value(),
                expected,
                "constant {constant}, public {public}"
            );
        }
    }
}

#[test]
fn folder_poisons_an_out_of_subfield_constant() {
    let (actual, _) = eval_entry_air(F::from_repr(5), F::ONE);
    assert!(actual.is_poisoned());
}

#[test]
fn folder_poisons_an_out_of_subfield_public_value() {
    let (actual, _) = eval_entry_air(F::ONE, outside());
    assert!(actual.is_poisoned());
}
