//! Typed bus abstractions for cross-AIR communication.
//!
//! # Overview
//!
//! A bus is a named channel shared between multiple AIRs.
//!
//! The proving system guarantees that all messages on a bus balance globally.

use crate::builder::InteractionBuilder;

/// Subset (table-lookup) bus.
///
/// One AIR holds a table. Other AIRs query it.
///
/// The proof guarantees every query hits a real entry.
///
/// ```text
///                         ┌───────────┐
/// CPU AIR ──lookup_key──▶ │ MEMORY bus│ ◀──table_entry── Memory AIR
///                         └───────────┘
/// ```
#[derive(Clone, Debug)]
pub struct LookupBus<'a> {
    name: &'a str,
}

impl<'a> LookupBus<'a> {
    /// Create a bus with the given name.
    pub const fn new(name: &'a str) -> Self {
        Self { name }
    }

    /// Bus name.
    pub const fn name(&self) -> &str {
        self.name
    }

    /// Query the table.
    ///
    /// Asserts that `key` exists in the table on each active row.
    ///
    /// # Arguments
    ///
    /// - `key` — elements identifying the entry.
    /// - `multiplicity` — number of lookups this row performs.
    pub fn lookup_key<AB, E>(
        &self,
        builder: &mut AB,
        key: impl IntoIterator<Item = E>,
        multiplicity: impl Into<AB::Expr>,
    ) where
        AB: InteractionBuilder,
        E: Into<AB::Expr>,
    {
        builder.push_interaction(self.name, key, multiplicity, 1);
    }

    /// Provide a table entry.
    ///
    /// Contributes `key` to the table on each active row.
    ///
    /// # Arguments
    ///
    /// - `key` — elements defining the entry.
    /// - `num_lookups` — times this entry is consumed.
    pub fn table_entry<AB, E>(
        &self,
        builder: &mut AB,
        key: impl IntoIterator<Item = E>,
        num_lookups: impl Into<AB::Expr>,
    ) where
        AB: InteractionBuilder,
        E: Into<AB::Expr>,
    {
        builder.push_interaction(self.name, key, -num_lookups.into(), 0);
    }
}

/// Multiset equality (permutation check) bus.
///
/// AIRs send and receive messages.
///
/// The proof guarantees sends exactly equal receives.
///
/// ```text
///                       ┌──────────────┐
/// Decoder AIR ──send──▶ │ DISPATCH bus │ ◀──receive── Executor AIR
///                       └──────────────┘
/// ```
#[derive(Clone, Debug)]
pub struct PermutationCheckBus<'a> {
    name: &'a str,
}

impl<'a> PermutationCheckBus<'a> {
    /// Create a bus with the given name.
    pub const fn new(name: &'a str) -> Self {
        Self { name }
    }

    /// Bus name.
    pub const fn name(&self) -> &str {
        self.name
    }

    /// Send a message.
    ///
    /// # Arguments
    ///
    /// - `fields` — the message elements.
    /// - `multiplicity` — number of sends this row performs.
    pub fn send<AB, E>(
        &self,
        builder: &mut AB,
        fields: impl IntoIterator<Item = E>,
        multiplicity: impl Into<AB::Expr>,
    ) where
        AB: InteractionBuilder,
        E: Into<AB::Expr>,
    {
        builder.push_interaction(self.name, fields, multiplicity, 1);
    }

    /// Receive a message.
    ///
    /// # Arguments
    ///
    /// - `fields` — the message elements.
    /// - `multiplicity` — number of receives this row performs.
    pub fn receive<AB, E>(
        &self,
        builder: &mut AB,
        fields: impl IntoIterator<Item = E>,
        multiplicity: impl Into<AB::Expr>,
    ) where
        AB: InteractionBuilder,
        E: Into<AB::Expr>,
    {
        builder.push_interaction(self.name, fields, -multiplicity.into(), 1);
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::String;
    use alloc::vec::Vec;

    use p3_air::{AirBuilder, RowWindow};
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    type F = BabyBear;

    struct MockInteraction {
        bus_name: String,
        num_fields: usize,
        count_weight: u32,
    }

    /// Records interactions without evaluating constraints.
    struct MockBuilder {
        interactions: Vec<MockInteraction>,
    }

    impl MockBuilder {
        fn new() -> Self {
            Self {
                interactions: Vec::new(),
            }
        }
    }

    impl AirBuilder for MockBuilder {
        type F = F;
        type Expr = F;
        type Var = F;
        type PreprocessedWindow = RowWindow<'static, F>;
        type MainWindow = RowWindow<'static, F>;
        type PublicVar = F;
        type PeriodicVar = F;

        fn main(&self) -> Self::MainWindow {
            unimplemented!()
        }
        fn preprocessed(&self) -> &Self::PreprocessedWindow {
            unimplemented!()
        }
        fn is_first_row(&self) -> Self::Expr {
            unimplemented!()
        }
        fn is_last_row(&self) -> Self::Expr {
            unimplemented!()
        }
        fn is_transition_window(&self, _: usize) -> Self::Expr {
            unimplemented!()
        }
        fn assert_zero<I: Into<Self::Expr>>(&mut self, _: I) {}
    }

    impl InteractionBuilder for MockBuilder {
        fn push_interaction<E: Into<Self::Expr>>(
            &mut self,
            bus_name: &str,
            fields: impl IntoIterator<Item = E>,
            _count: impl Into<Self::Expr>,
            count_weight: u32,
        ) {
            let num_fields = fields.into_iter().count();
            self.interactions.push(MockInteraction {
                bus_name: String::from(bus_name),
                num_fields,
                count_weight,
            });
        }

        fn push_local_interaction(
            &mut self,
            tuples: impl IntoIterator<Item = (Vec<Self::Expr>, Self::Expr)>,
        ) {
            tuples.into_iter().for_each(drop);
        }

        fn num_global_interactions(&self) -> usize {
            self.interactions.len()
        }
    }

    #[test]
    fn lookup_key_uses_weight_1() {
        let bus = LookupBus::new("mem");
        let mut b = MockBuilder::new();
        bus.lookup_key(&mut b, [F::ONE, F::TWO], F::ONE);

        assert_eq!(b.interactions.len(), 1);
        assert_eq!(b.interactions[0].bus_name, "mem");
        assert_eq!(b.interactions[0].num_fields, 2);
        assert_eq!(b.interactions[0].count_weight, 1);
    }

    #[test]
    fn table_entry_uses_weight_0() {
        let bus = LookupBus::new("mem");
        let mut b = MockBuilder::new();
        bus.table_entry(&mut b, [F::ONE], F::ONE);

        assert_eq!(b.interactions[0].count_weight, 0);
    }

    #[test]
    fn permutation_check_send_receive() {
        let bus = PermutationCheckBus::new("dispatch");
        let mut b = MockBuilder::new();

        bus.send(&mut b, [F::ONE], F::ONE);
        bus.receive(&mut b, [F::TWO], F::ONE);

        assert_eq!(b.interactions.len(), 2);
        assert_eq!(b.interactions[0].count_weight, 1);
        assert_eq!(b.interactions[1].count_weight, 1);
    }

    #[test]
    fn multiple_buses_independent() {
        let mem = LookupBus::new("memory");
        let rc = LookupBus::new("range_check");
        let mut b = MockBuilder::new();

        mem.lookup_key(&mut b, [F::ONE], F::ONE);
        rc.lookup_key(&mut b, [F::TWO], F::ONE);

        assert_eq!(b.num_global_interactions(), 2);
        assert_eq!(b.interactions[0].bus_name, "memory");
        assert_eq!(b.interactions[1].bus_name, "range_check");
    }
}
