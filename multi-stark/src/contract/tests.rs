use alloc::vec;

use super::declaration::{
    ColumnCounts, DeclarationError, FlushDeclaration, FlushDirection, HeightRange,
    LocalConstraints, MAX_LOG_HEIGHT, MAX_PROOF_BYTES, MachineDeclaration, TableDeclaration,
};
use super::secrecy::{BindingOnly, Hiding, Secrecy, SecrecyLevel};

fn table() -> TableDeclaration {
    TableDeclaration::new(
        ColumnCounts {
            committed: 4,
            preprocessed: 0,
            public: 1,
        },
        LocalConstraints {
            count: 3,
            degree: 2,
        },
        HeightRange::new(2, 16),
    )
}

#[test]
fn a_statement_needs_at_least_one_table() {
    assert_eq!(
        MachineDeclaration::<BindingOnly>::new(vec![], 1024).unwrap_err(),
        DeclarationError::NoTables
    );
}

#[test]
fn a_budget_outside_the_ceiling_is_refused() {
    for budget in [0, MAX_PROOF_BYTES + 1] {
        assert_eq!(
            MachineDeclaration::<BindingOnly>::new(vec![table()], budget).unwrap_err(),
            DeclarationError::BudgetOutOfRange {
                found: budget,
                limit: MAX_PROOF_BYTES,
            }
        );
    }
}

#[test]
fn an_unreachable_height_range_is_refused() {
    let inverted = TableDeclaration::new(
        ColumnCounts::default(),
        LocalConstraints::default(),
        HeightRange::new(9, 8),
    );
    assert_eq!(
        MachineDeclaration::<BindingOnly>::new(vec![inverted], 1024).unwrap_err(),
        DeclarationError::EmptyHeightRange {
            table: 0,
            min: 9,
            max: 8,
        }
    );
}

#[test]
fn a_height_above_the_ceiling_is_refused() {
    let tall = TableDeclaration::new(
        ColumnCounts::default(),
        LocalConstraints::default(),
        HeightRange::new(0, MAX_LOG_HEIGHT + 1),
    );
    assert!(matches!(
        MachineDeclaration::<BindingOnly>::new(vec![tall], 1024).unwrap_err(),
        DeclarationError::AboveLimit { .. }
    ));
}

#[test]
fn grinding_above_the_ceiling_is_refused() {
    let declaration = MachineDeclaration::<BindingOnly>::new(vec![table()], 1024).unwrap();
    assert!(matches!(
        declaration.run(&[8], 1000).unwrap_err(),
        DeclarationError::PowBitsAboveLimit { .. }
    ));
}

#[test]
fn a_run_of_one_statement_is_refused_by_another() {
    // Two statements differing only in the channel a table flushes on.
    let plain = MachineDeclaration::<BindingOnly>::new(vec![table()], 1024).unwrap();
    let flushing = MachineDeclaration::<BindingOnly>::new(
        vec![table().with_flushes(vec![FlushDeclaration {
            channel: "shared".into(),
            direction: FlushDirection::Push,
            tuple_width: 3,
            max_multiplicity: 1,
        }])],
        1024,
    )
    .unwrap();

    let run = plain.run(&[8], 0).unwrap();
    assert_eq!(
        flushing.run_digest(&run).unwrap_err(),
        DeclarationError::ForeignRun
    );
    assert!(plain.run_digest(&run).is_ok());
}

#[test]
fn the_promise_changes_the_fingerprint() {
    // The same tables under a different promise must not share a run.
    let binding = MachineDeclaration::<BindingOnly>::new(vec![table()], 1024).unwrap();
    let hiding = MachineDeclaration::<Hiding>::new(vec![table()], 1024).unwrap();

    let run = binding.run(&[8], 0).unwrap();
    assert_eq!(
        hiding.run_digest(&run).unwrap_err(),
        DeclarationError::ForeignRun
    );
    assert_ne!(BindingOnly::SECRECY.tag(), Hiding::SECRECY.tag());
    assert_eq!(BindingOnly::SECRECY, Secrecy::BindingOnly);
}

#[test]
fn the_height_a_run_picks_changes_the_fingerprint() {
    let declaration = MachineDeclaration::<BindingOnly>::new(vec![table()], 1024).unwrap();
    let low = declaration.run(&[8], 0).unwrap();
    let high = declaration.run(&[9], 0).unwrap();
    let ground = declaration.run(&[8], 1).unwrap();

    let digest = |run| declaration.run_digest(run).unwrap();
    assert_ne!(digest(&low), digest(&high));
    assert_ne!(digest(&low), digest(&ground));
}
