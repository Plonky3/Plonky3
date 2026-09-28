//! Agreement between the debug screen for periodic column shapes and the commitment rule.

use std::borrow::Cow;
use std::panic::{self, AssertUnwindSafe};

use p3_air::{Air, AirBuilder, BaseAir, check_constraints};
use p3_baby_bear::BabyBear;
use p3_commit::PeriodicColumns;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

type Val = BabyBear;

/// An AIR that declares one periodic column of a chosen length and asserts nothing.
///
/// Dropping every constraint leaves the shape screen as the only thing that can reject it.
struct DeclaresOnePeriodicColumn {
    length: usize,
}

impl BaseAir<Val> for DeclaresOnePeriodicColumn {
    fn width(&self) -> usize {
        1
    }

    fn num_periodic_columns(&self) -> usize {
        1
    }

    fn periodic_columns(&self) -> Cow<'_, [Vec<Val>]> {
        Cow::Owned(vec![vec![Val::ZERO; self.length]])
    }
}

impl<AB: AirBuilder<F = Val>> Air<AB> for DeclaresOnePeriodicColumn {
    fn eval(&self, _builder: &mut AB) {}
}

// The debug screen and the commitment rule are two separate pieces of code in two crates.
// They sit on opposite sides of the proving stack and neither can call the other.
//
//     p3-air     debug screen        panics, so an AIR author hears about it first
//     p3-commit  commitment rule     gates evaluation, so a proof cannot skip it
//
// An AIR the screen waves through and the rule then rejects would fail late and confusingly.
// One the screen rejects and the rule accepts would be a false alarm.
// So the two verdicts are compared over a shared table of heights and lengths.
#[test]
fn the_debug_screen_and_the_commitment_rule_agree() {
    // Heights: powers of two, an odd one, and one with an odd factor.
    let heights = [1, 2, 3, 4, 8, 12];

    // Lengths: powers of two, non-powers, zero, and lengths reaching past the height.
    let lengths = [0, 1, 2, 3, 4, 5, 6, 8, 12, 16];

    // The screen reports by panicking, so reading its verdict means catching the unwind.
    // Muting the hook keeps the expected panics out of the test output.
    let previous = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));

    // Verdicts are collected first and compared after the hook is back.
    // A failing comparison then prints the way every other test failure does.
    let mut verdicts = Vec::new();
    for height in heights {
        for length in lengths {
            // Both sides see the same declaration: one column listing `length` zeros.
            let columns = vec![vec![Val::ZERO; length]];
            let rule_accepts = PeriodicColumns::new(&columns, height).is_ok();

            // The trace carries no constraint, so only the screen can reject it.
            let air = DeclaresOnePeriodicColumn { length };
            let trace = RowMajorMatrix::new(Val::zero_vec(height), 1);
            let screen_accepts =
                panic::catch_unwind(AssertUnwindSafe(|| check_constraints(&air, &trace, &[])))
                    .is_ok();

            verdicts.push((height, length, rule_accepts, screen_accepts));
        }
    }

    panic::set_hook(previous);

    // A table that was all rejections, or all acceptances, would compare equal for free.
    // Requiring both outcomes keeps the comparison from passing vacuously.
    assert!(verdicts.iter().any(|&(.., accepts, _)| accepts));
    assert!(verdicts.iter().any(|&(.., accepts, _)| !accepts));

    for (height, length, rule_accepts, screen_accepts) in verdicts {
        assert_eq!(
            rule_accepts, screen_accepts,
            "height {height}, length {length}"
        );
    }
}
